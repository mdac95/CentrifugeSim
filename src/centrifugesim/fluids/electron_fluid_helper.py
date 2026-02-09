import numpy as np
import numba
from numba import njit

from scipy import sparse
from scipy.sparse.linalg import splu, lgmres, LinearOperator, bicgstab, spilu

from centrifugesim import constants

# Small floors to avoid division-by-zero in edge cells
NU_FLOOR = 1e-30
B_FLOOR  = 1e-5

#################################################################################
################################### Helper kernels ##############################
#################################################################################

@njit(cache=True)
def _kBT(Te, Te_is_eV):
    """Return k_B T in Joules (same shape as Te)."""
    if Te_is_eV:
        # k_B*T = Te[eV]*q_e
        return Te * constants.q_e
    else:
        return Te * constants.kb

@njit(nogil=True)
def get_anomalous_collision_frequency(ne, Te, Ti, J_mag, mi):
    """
    Compute anomalous electron collision frequency (1/s) due to ion-acoustic
    instability, using Bychenkov-Silin approximate scaling.
    Inputs are arrays Te [K], Ti [K], ne [m^-3], J_mag [A/m^2] with identical shapes.
    Ion mass mi [kg] is a scalar.
    """
    qe = constants.q_e
    me = constants.m_e

    # Calculate Drift Velocity
    v_drift = J_mag / (ne * qe)

    # Calculate Thermal/Sound Speeds
    # v_th_e = sqrt(2*Te/me)
    # c_s = sqrt(Te/mi)
    v_th_e = np.sqrt(2.0 * constants.kb * Te / me)
    c_s    = np.sqrt(constants.kb * Te / mi)

    # Calculate Plasma Frequency
    w_pe = np.sqrt(ne * qe**2 / (me * constants.ep0))

    # Temperature Ratio (The "Gain" knob)
    # Real physics: scales with Te/Ti.
    # Safety: Clamp Ti_eff to at least 0.1 eV to prevent Te/Ti -> Infinity
    Ti_eff = np.where(Ti < 0.1*11604, 0.1*11604, Ti)
    Tratio = Te / Ti_eff
    
    # Bychenkov-Silin Scaling (Approximate)
    # This roughly matches the kinetic theory limit without free parameters
    # Factor ~ 0.01 comes from Sagdeev saturation theory
    alpha_eff = 1.0e-2 * (Tratio) * (v_drift / v_th_e)
    
    nu_anom = alpha_eff * w_pe
    
    # Buneman Limit (Safety Cap)
    # If v_drift is HUGE (> v_th_e), the instability changes to Buneman.
    # The growth rate saturates at roughly (me/mi)^(1/3) * w_pe.
    # This prevents the term from going to infinity if density drops to zero.
    nu_max = 0.1 * w_pe

    # Threshold Check: Ion Acoustic Instability
    # Only active if electrons move faster than the acoustic wave
    #nu_anom = np.where(v_drift > c_s, nu_anom, 0.0)
    
    return np.minimum(nu_anom, nu_max)


@njit(cache=True)
def electron_collision_frequencies(
    Te, ne, nn,
    lnLambda=12.0,
    sigma_en_m2=2.0e-19, # momentum transfer cross section, should have integral of cross section and distribution function and save it (interpolate) to then use here.
    Te_is_eV=False
):
    """
    Compute electron collision frequencies (1/s):
      - nu_en: electron-neutral momentum-transfer
      - nu_ei: electron-ion (Spitzer)
      - nu_e : total = nu_en + nu_ei
    Inputs are arrays Te [K or eV], ne [m^-3], nn [m^-3] with identical shapes.
    """
    kBT = _kBT(Te, Te_is_eV)                                   # J
    vth_e = np.sqrt(8.0 * kBT / (np.pi * constants.m_e))       # m/s

    # Electron-neutral (hard-sphere-like, momentum-transfer)
    nu_en = nn * sigma_en_m2 * vth_e                           # 1/s

    # Electron-ion (Spitzer), Z=1, ni=ne
    c_num = 4.0 * np.sqrt(2.0 * np.pi) * (constants.q_e**4) * lnLambda
    c_den = 3.0 * (4.0 * np.pi * constants.ep0)**2 * np.sqrt(constants.m_e)
    nu_ei = c_num * ne / (c_den * (kBT**1.5))            # 1/s

    # Total + floors
    nu_e  = nu_en + nu_ei
    nu_en = np.maximum(nu_en, NU_FLOOR)
    nu_ei = np.maximum(nu_ei, NU_FLOOR)
    nu_e  = np.maximum(nu_e,  NU_FLOOR)
    return nu_en, nu_ei, nu_e


@njit(cache=True)
def electron_conductivities(
    Te, ne, Bmag, nu_e, nu_e_anom
):
    """
    Electron conductivity tensor components (SI, S/m):
      - sigma_par_e : parallel to B
      - sigma_P_e   : Pedersen (perpendicular, in-plane with E_perp)
      - sigma_H_e   : Hall (perpendicular, out-of-phase; negative for electrons)
    Inputs:
      Te [K or eV], ne [m^-3], nn [m^-3], Br [T], Bz [T] (same shape)
    Assumes: Z=1, ni = ne (quasineutral).
    """
    # |B|
    Bmag = np.where(Bmag < B_FLOOR, B_FLOOR, Bmag)

    # Electron gyrofrequency magnitude
    Omega_e = constants.q_e * Bmag / constants.m_e  # rad/s

    # Prefactor ne e^2 / m_e
    pref = ne * (constants.q_e * constants.q_e) / constants.m_e  # S·s/m

    # Components (electrons only)
    sigma_par_e = pref / (nu_e-nu_e_anom)  # subtract anomalous collision freq
    denom = (nu_e*nu_e + Omega_e*Omega_e)
    sigma_P_e = pref * (nu_e / denom)
    sigma_H_e = - pref * (Omega_e / denom)

    return sigma_par_e, sigma_P_e, sigma_H_e, Omega_e/nu_e  # β_e


@njit(cache=True)
def time_advance_ne_analytic_kernel_anisotropic(ne_out, ne_old, 
                                                nu_iz, nu_loss, nu_RR, beta_rec, 
                                                dt, mask, ne_floor):
    """
    Solves the logistic growth/decay equation analytically for density:
        dn/dt = (nu_iz - nu_loss - nu_RR) * n - beta_rec * n^2
    
    This kernel is 'unconditionally stable' (it will not oscillate or explode 
    even if dt is very large). It automatically detects if dt is large enough 
    to jump straight to equilibrium.

    Parameters
    ----------
    ne_out   : Output array (Nr, Nz)
    ne_old   : Input array (Nr, Nz) - Previous density
    nu_iz    : Input array (Nr, Nz) - Ionization frequency [1/s]
    nu_loss  : Input array (Nr, Nz) - Diffusive loss rate [1/s] (calculated externally)
    nu_RR    : Input array (Nr, Nz) - Radiative recombination frequency [1/s]
    beta_rec : Input array (Nr, Nz) - Recombination coeff [m^3/s]
    dt       : float - Time step [s]
    mask     : Input array (Nr, Nz) - 1=Plasma, 0=Solid
    """
    Nr, Nz = ne_out.shape
    
    # Exponent cap to prevent float64 overflow (exp(709) is max)
    # We use 100 as a safe "infinity" threshold. exp(100) is ~2e43.
    EXP_LIMIT = 100.0 
    
    # Minimum density floor to prevent div-by-zero or negative densities
    NE_FLOOR = ne_floor
    
    for i in range(Nr):
        for j in range(Nz):
            if mask[i, j] == 1:
                n0 = max(ne_old[i, j], NE_FLOOR)
                
                # 1. Net Linear Growth Rate (alpha)
                # alpha = Production - Loss
                alpha = nu_iz[i, j] - nu_loss[i, j] - nu_RR[i, j]
                
                beta = beta_rec[i, j]
                
                # --- CASE A: DECAY (alpha < 0) ---
                # The plasma is cooling/dying. 
                # Equation behaves like: n(t) -> 0
                if alpha < -1e-20:
                    # Check for underflow (if decay is too fast)
                    # arg is negative
                    arg = alpha * dt
                    
                    if arg < -EXP_LIMIT:
                        # Decayed effectively to zero (floor)
                        ne_out[i, j] = NE_FLOOR
                    else:
                        # Full Logistic Decay Solution
                        # (Formula is same as growth, but alpha is negative)
                        exp_factor = np.exp(arg) # < 1.0
                        
                        if beta > 1e-30:
                            numerator = n0 * exp_factor
                            # Denom can't be zero because alpha is negative and beta > 0
                            denominator = 1.0 + (n0 * beta / alpha) * (exp_factor - 1.0)
                            
                            # Safety check for denominator
                            if abs(denominator) > 1e-15:
                                ne_out[i, j] = numerator / denominator
                            else:
                                ne_out[i, j] = NE_FLOOR
                        else:
                            # Pure exponential decay
                            ne_out[i, j] = n0 * exp_factor

                # --- CASE B: GROWTH (alpha > 0) ---
                # The plasma is ionizing.
                # Equation behaves like: n(t) -> alpha / beta
                else:
                    arg = alpha * dt
                    
                    # 1. Check for Equilibrium (Large dt or fast rate)
                    if arg > EXP_LIMIT:
                        # We assume t -> infinity.
                        # Equilibrium = alpha / beta
                        if beta > 1e-30:
                            ne_out[i, j] = alpha / beta
                        else:
                            # Beta is 0 (No recombination) -> Runaway growth!
                            # Cap it to avoid infinity (e.g., 100x current value or some physics limit)
                            ne_out[i, j] = n0 * 100.0 
                    
                    # 2. Standard Time Advance
                    else:
                        exp_factor = np.exp(arg) # > 1.0
                        
                        if beta > 1e-30:
                             numerator = n0 * exp_factor
                             denominator = 1.0 + (n0 * beta / alpha) * (exp_factor - 1.0)
                             ne_out[i, j] = numerator / denominator
                        else:
                             # Pure exponential growth (no recombination)
                             ne_out[i, j] = n0 * exp_factor
                             
                # Final safety clamp
                if ne_out[i, j] < NE_FLOOR: 
                    ne_out[i, j] = NE_FLOOR
                
            else:
                # Masked Region (Solid)
                ne_out[i, j] = NE_FLOOR


@njit(cache=True)
def compute_ambipolar_loss_rate_anisotropic(Te, Ti, nu_in, beta_e, beta_i, 
                                            mi, kb, 
                                            inv_Lambda_z_sq, inv_Lambda_r_sq):
    """
    Computes the effective diffusive loss rate [1/s] accounting for magnetization.
    
    nu_diff = (Da_par * inv_Lambda_z^2) + (Da_perp * inv_Lambda_r^2)
    
    where:
      Da_par  = kb*(Te+Ti) / (mi*nu_in)
      Da_perp = Da_par / (1 + beta_e * beta_i)
    """
    Nr, Nz = Te.shape
    nu_loss = np.zeros_like(Te)
    
    for i in range(Nr):
        for j in range(Nz):
            # 1. Parallel Ambipolar Diffusion (Classical)
            # nu_in floor to avoid infinity
            nu_safe = max(nu_in[i, j], 1e-5)
            
            Da_par = (kb * (Te[i, j] + Ti[i, j])) / (mi * nu_safe)
            
            # 2. Perpendicular Ambipolar Diffusion (Magnetized)
            # Simon Short-Circuit / Classical Ambipolar scaling
            magnetization_factor = 1.0 + beta_e[i, j] * beta_i[i, j]
            Da_perp = Da_par / magnetization_factor
            
            # 3. Total Geometric Loss Rate
            # Loss = D_par/Lz^2 + D_perp/Lr^2
            nu_loss[i, j] = (Da_par * inv_Lambda_z_sq) + (Da_perp * inv_Lambda_r_sq)
            
    return nu_loss

@njit(cache=True)
def update_Te_local_physics(Te, ne, nn, T_n, T_i, 
                            nu_en, nu_ei, 
                            Q_Joule, epsilon_iz_J, dt, mi, mn, mask, Te_floor,
                            chem_T_arr, chem_k_arr):
    """
    Updates Te using Fully Implicit local physics.
    Uses Internal Interpolation to evaluate Ionization Rate(T) inside the loop,
    preventing time-lag instabilities.
    """
    Nr, Nz = Te.shape
    kb = constants.kb
    me = constants.m_e

    for i in range(Nr):
        for j in range(Nz):
            if mask[i, j] == 1:
                
                n_e = max(ne[i, j], 1e10)
                n_n = max(nn[i, j], 1e10)
                T_old = Te[i, j]
                Cv = 1.5 * n_e * kb
                
                # 1. Source Terms
                S_heating = Q_Joule[i, j]
                
                # 2. Relaxation Params
                freq_en = 2.0 * (me / mn) * nu_en[i, j] 
                freq_ei = 2.0 * (me / mi) * nu_ei[i, j]
                nu_relax = freq_en + freq_ei
                
                if nu_relax > 1e-20:
                    T_target = (freq_en * T_n[i, j] + freq_ei * T_i[i, j]) / nu_relax
                else:
                    T_target = T_n[i, j]

                # 3. NEWTON-RAPHSON SOLVER
                T_guess = T_old
                
                for k in range(10): 
                    T_safe = max(T_guess, 300.0)
                    
                    # --- A. Internal Interpolation ---
                    # We evaluate the rate at the CURRENT guess T_safe.
                    # This replaces calling 'k_interp(T_safe)' which Numba cannot do.
                    
                    k_rate = np.interp(T_safe, chem_T_arr, chem_k_arr)
                    
                    # --- B. Numerical Derivative for Jacobian ---
                    # Perturb T slightly to find the "Stiffness" (slope)
                    delta_T = 0.01 * T_safe
                    k_rate_plus = np.interp(T_safe + delta_T, chem_T_arr, chem_k_arr)
                    dk_dT = (k_rate_plus - k_rate) / delta_T
                    
                    # --- C. Calculate Cooling ---
                    # Cooling = n_e * nu_iz * Cost
                    # nu_iz = n_n * k_rate
                    S_cooling = n_e * (n_n * k_rate) * epsilon_iz_J
                    
                    # Derivative dS/dT 
                    dS_cool_dT = n_e * n_n * dk_dT * epsilon_iz_J
                    
                    # --- D. Residual & Jacobian ---
                    # F = Inertia + Relax + Cooling - Heating
                    term_inertia = (Cv / dt) * (T_guess - T_old)
                    term_relax   = Cv * nu_relax * (T_guess - T_target)
                    
                    F_val = term_inertia + term_relax + S_cooling - S_heating
                    J_val = (Cv / dt) + (Cv * nu_relax) + dS_cool_dT
                    
                    # --- E. Newton Step ---
                    if abs(J_val) < 1e-20: J_val = 1e-20
                    delta = F_val / J_val
                    
                    # Damping
                    max_step = 0.5 * T_guess
                    if delta > max_step: delta = max_step
                    if delta < -max_step: delta = -max_step
                    
                    T_guess = T_guess - delta
                    
                    if abs(delta) < 1e-3 * T_guess:
                        break
                
                if T_guess < Te_floor: T_guess = Te_floor
                Te[i, j] = T_guess


@numba.jit(nopython=True, parallel=True, cache=True)
def assemble_Te_diffusion_FD(Te, ne, kappa_par, kappa_perp, 
                             br, bz, mask, dr, dz, r_coords, dt, 
                             mi, closed_top, 
                             i_cathode_r, j_cathode_r,
                             i_cathode_z, j_cathode_z,
                             delta_sheath=1.0):
    """
    Finite Difference Assembly with closed top and internal cathode sheath BCs (R and Z).
    """
    Nr, Nz = Te.shape
    kb = constants.kb
    
    # Initialize Matrices
    aP = np.zeros((Nr, Nz))
    aE = np.zeros((Nr, Nz))
    aW = np.zeros((Nr, Nz))
    aN = np.zeros((Nr, Nz))
    aS = np.zeros((Nr, Nz))
    b  = np.zeros((Nr, Nz))
    
    inv_dr2 = 1.0 / (dr * dr)
    inv_dz2 = 1.0 / (dz * dz)

    # --- Main Domain Assembly ---
    for i in numba.prange(Nr):
        r_loc = r_coords[i]
        inv_r = 1.0 / (r_loc + 1e-12) 
        
        for j in range(Nz):
            if mask[i, j] == 0:
                aP[i, j] = 1.0
                b[i, j]  = Te[i, j]
                continue
            
            # 1. Identify Neighbors
            has_N = (j < Nz-1) and (mask[i, j+1] == 1)
            has_S = (j > 0)    and (mask[i, j-1] == 1)
            has_E = (i < Nr-1) and (mask[i+1, j] == 1)
            has_W = (i > 0)    and (mask[i-1, j] == 1)
            
            # 2. Tensors
            k_p = kappa_par[i, j]
            k_v = kappa_perp[i, j]
            Br = br[i, j]
            Bz = bz[i, j]
            
            k_rr = k_v + (k_p - k_v) * Br*Br
            k_zz = k_v + (k_p - k_v) * Bz*Bz
            
            # 3. Coefficients
            geom_fac = 0.5 * dr * inv_r
            val_E = k_rr * inv_dr2 * (1.0 + geom_fac)
            val_W = k_rr * inv_dr2 * (1.0 - geom_fac)
            val_N = k_zz * inv_dz2
            val_S = k_zz * inv_dz2
            
            # Inertia
            Cv_term = 1.5 * ne[i, j] * kb / dt
            
            # 4. Standard Boundaries
            # NORTH (Top)
            if not has_N:
                val_N = 0.0
                if j == Nz - 1 and closed_top:
                    cs = np.sqrt(kb * Te[i, j] / mi)
                    G_sheath = delta_sheath * ne[i, j] * cs * kb
                    Cv_term += G_sheath / dz
            
            # SOUTH (Bottom)
            if not has_S:
                val_S = 0.0
                if j == 0: 
                    cs = np.sqrt(kb * Te[i, j] / mi)
                    G_sheath = delta_sheath * ne[i, j] * cs * kb
                    Cv_term += G_sheath / dz

            # EAST (Outer Wall)
            if not has_E:
                val_E = 0.0
                if i == Nr - 1:
                    cs = np.sqrt(kb * Te[i, j] / mi)
                    G_sheath = delta_sheath * ne[i, j] * cs * kb
                    Cv_term += G_sheath / dr
            
            # WEST (Axis/Internal)
            if not has_W:
                val_W = 0.0 
            
            # 5. Assemble
            aE[i, j] = val_E
            aW[i, j] = val_W
            aN[i, j] = val_N
            aS[i, j] = val_S
            
            aP[i, j] = Cv_term + val_E + val_W + val_N + val_S
            b[i, j] = (1.5 * ne[i, j] * kb / dt) * Te[i, j]

    # --- SPECIAL CATHODE BOUNDARY LOOPS ---

    # 1. Radial Face (Vertical Wall)
    # Assumes plasma is at i_c + 1 (Right of cathode)
    num_pts_r = len(i_cathode_r)
    for k in range(num_pts_r):
        i_c = i_cathode_r[k]
        j_c = j_cathode_r[k]
        
        i_p = i_c + 1 # Plasma Node
        
        if i_p < Nr and mask[i_p, j_c] == 1:
            cs = np.sqrt(kb * Te[i_p, j_c] / mi)
            G_sheath = delta_sheath * ne[i_p, j_c] * cs * kb
            aP[i_p, j_c] += G_sheath / dr # <--- Divided by dr

    # 2. Axial Face (Horizontal Cap) <--- NEW SECTION
    # Assumes plasma is at j_c + 1 (Above cathode)
    num_pts_z = len(i_cathode_z)
    for k in range(num_pts_z):
        i_c = i_cathode_z[k]
        j_c = j_cathode_z[k]
        
        j_p = j_c + 1 # Plasma Node (North)
        
        # Check bounds and mask
        if j_p < Nz and mask[i_c, j_p] == 1:
            cs = np.sqrt(kb * Te[i_c, j_p] / mi)
            G_sheath = delta_sheath * ne[i_c, j_p] * cs * kb
            
            # Add to Diagonal (Sink term)
            # Divided by dz because this is an axial flux
            aP[i_c, j_p] += G_sheath / dz  # <--- Divided by dz

    return aP, aE, aW, aN, aS, b


# Broken, need to fix!
@numba.jit(nopython=True, parallel=True, cache=True)
def assemble_Te_advection_diffusion_FD(Te, ne, ur, uz,
                                     kappa_par, kappa_perp, 
                                     br, bz, mask, dr, dz, r_coords, dt, 
                                     mi, closed_top, 
                                     i_cathode_r, j_cathode_r,
                                     i_cathode_z, j_cathode_z,
                                     delta_sheath=1.0):
    """
    FD Assembly with Advection-Diffusion.
    """
    Nr, Nz = Te.shape
    kb = 1.380649e-23 # Explicit constant or pass it in
    
    # Enthalpy flux factor (5/2). Use 1.5 for Energy flux only.
    gamma_adv = 2.5 
    
    # Initialize Matrices
    aP = np.zeros((Nr, Nz))
    aE = np.zeros((Nr, Nz))
    aW = np.zeros((Nr, Nz))
    aN = np.zeros((Nr, Nz))
    aS = np.zeros((Nr, Nz))
    b  = np.zeros((Nr, Nz))
    
    inv_dr = 1.0 / dr
    inv_dz = 1.0 / dz
    inv_dr2 = 1.0 / (dr * dr)
    inv_dz2 = 1.0 / (dz * dz)

    # --- Main Domain Assembly ---
    for i in numba.prange(Nr):
        r_loc = r_coords[i]
        inv_r = 1.0 / (r_loc + 1e-12) 
        
        for j in range(Nz):
            # --- 0. VALIDITY CHECK ---
            # If current node is Solid/Invalid, do nothing (or Identity)
            if mask[i, j] == 0:
                aP[i, j] = 1.0
                b[i, j]  = Te[i, j]
                continue
            
            # --- 1. SAFE NEIGHBOR LOOKUP ---
            # We explicitly check bounds first, THEN check the mask.
            # This prevents IndexError on array edges.
            
            # West (i-1)
            valid_W = False
            if i > 0:
                if mask[i-1, j] == 1:
                    valid_W = True

            # East (i+1)
            valid_E = False
            if i < Nr - 1:
                if mask[i+1, j] == 1:
                    valid_E = True
            
            # South (j-1)
            valid_S = False
            if j > 0:
                if mask[i, j-1] == 1:
                    valid_S = True

            # North (j+1)
            valid_N = False
            if j < Nz - 1:
                if mask[i, j+1] == 1:
                    valid_N = True
            
            # --- 2. DIFFUSION TERMS (Conductivity) ---
            k_p = kappa_par[i, j]
            k_v = kappa_perp[i, j]
            Br_val = br[i, j]
            Bz_val = bz[i, j]
            
            k_rr = k_v + (k_p - k_v) * Br_val*Br_val
            k_zz = k_v + (k_p - k_v) * Bz_val*Bz_val
            
            geom_fac = 0.5 * dr * inv_r
            
            # Diffusion Coefficients (Set to 0 if neighbor invalid)
            val_E_diff = (k_rr * inv_dr2 * (1.0 + geom_fac)) if valid_E else 0.0
            val_W_diff = (k_rr * inv_dr2 * (1.0 - geom_fac)) if valid_W else 0.0
            val_N_diff = (k_zz * inv_dz2) if valid_N else 0.0
            val_S_diff = (k_zz * inv_dz2) if valid_S else 0.0
            
            # --- 3. ADVECTION TERMS (Strict Upwinding) ---
            # Reset advection accumulators
            val_E_adv = 0.0
            val_W_adv = 0.0
            val_N_adv = 0.0
            val_S_adv = 0.0
            div_adv_P = 0.0 

            # EAST FACE (+r)
            # Only compute if East neighbor is VALID
            if valid_E:
                # Safe to access i+1 here
                u_face = 0.5 * (ur[i, j] + ur[i+1, j])
                n_face = 0.5 * (ne[i, j] + ne[i+1, j])
                flux = gamma_adv * n_face * kb * u_face * inv_dr * (1.0 + geom_fac)
                
                if flux > 0: div_adv_P += flux        # Flow OUT -> Loss (Diagonal)
                else:        val_E_adv += -flux       # Flow IN  -> Gain (Neighbor)

            # WEST FACE (-r)
            if valid_W:
                u_face = 0.5 * (ur[i, j] + ur[i-1, j])
                n_face = 0.5 * (ne[i, j] + ne[i-1, j])
                # Note: normal is -1
                flux = -1.0 * gamma_adv * n_face * kb * u_face * inv_dr * (1.0 - geom_fac)
                
                if flux > 0: div_adv_P += flux
                else:        val_W_adv += -flux

            # NORTH FACE (+z)
            if valid_N:
                u_face = 0.5 * (uz[i, j] + uz[i, j+1])
                n_face = 0.5 * (ne[i, j] + ne[i, j+1])
                flux = gamma_adv * n_face * kb * u_face * inv_dz
                
                if flux > 0: div_adv_P += flux
                else:        val_N_adv += -flux

            # SOUTH FACE (-z)
            if valid_S:
                u_face = 0.5 * (uz[i, j] + uz[i, j-1])
                n_face = 0.5 * (ne[i, j] + ne[i, j-1])
                flux = -1.0 * gamma_adv * n_face * kb * u_face * inv_dz
                
                if flux > 0: div_adv_P += flux
                else:        val_S_adv += -flux

            # --- 4. TIME & BOUNDARIES ---
            Cv_term = 1.5 * ne[i, j] * kb / dt
            
            # Standard Sheath Boundaries (Only if neighbor is NOT valid)
            # NORTH (Top Wall)
            if (not valid_N) and (j == Nz - 1 and closed_top):
                cs = np.sqrt(kb * Te[i, j] / mi)
                Cv_term += (delta_sheath * ne[i, j] * cs * kb) / dz
            
            # SOUTH (Bottom Wall)
            if (not valid_S) and (j == 0): 
                cs = np.sqrt(kb * Te[i, j] / mi)
                Cv_term += (delta_sheath * ne[i, j] * cs * kb) / dz

            # EAST (Outer Wall)
            if (not valid_E) and (i == Nr - 1):
                cs = np.sqrt(kb * Te[i, j] / mi)
                Cv_term += (delta_sheath * ne[i, j] * cs * kb) / dr
            
            # --- 5. FINAL ASSEMBLY ---
            aE[i, j] = val_E_diff + val_E_adv
            aW[i, j] = val_W_diff + val_W_adv
            aN[i, j] = val_N_diff + val_N_adv
            aS[i, j] = val_S_diff + val_S_adv
            
            aP[i, j] = Cv_term + div_adv_P + \
                       val_E_diff + val_W_diff + val_N_diff + val_S_diff
                       
            b[i, j] = (1.5 * ne[i, j] * kb / dt) * Te[i, j]

    # --- SPECIAL CATHODE BOUNDARY LOOPS ---
    # (Kept identical but with extra boundary checks just in case)
    
    num_pts_r = len(i_cathode_r)
    for k in range(num_pts_r):
        i_c = i_cathode_r[k]
        j_c = j_cathode_r[k]
        i_p = i_c + 1 
        # Strict check: Plasma node must be inside domain AND masked 1
        if i_p < Nr and mask[i_p, j_c] == 1:
            cs = np.sqrt(kb * Te[i_p, j_c] / mi)
            G_sheath = delta_sheath * ne[i_p, j_c] * cs * kb
            aP[i_p, j_c] += G_sheath / dr 

    num_pts_z = len(i_cathode_z)
    for k in range(num_pts_z):
        i_c = i_cathode_z[k]
        j_c = j_cathode_z[k]
        j_p = j_c + 1 
        if j_p < Nz and mask[i_c, j_p] == 1:
            cs = np.sqrt(kb * Te[i_c, j_p] / mi)
            G_sheath = delta_sheath * ne[i_c, j_p] * cs * kb
            aP[i_c, j_p] += G_sheath / dz 

    return aP, aE, aW, aN, aS, b


def solve_direct_fast(Nr, Nz, aP, aE, aW, aN, aS, b):
    """
    Vectorized construction of the sparse matrix for A * phi = b.
    Much faster than looping in Python.
    """
    n_dofs = Nr * Nz
    
    # 1. Create a grid of indices: ID[i, j] = i * Nz + j
    # 'C' order implies the last index (j) changes fastest. 
    idx_grid = np.arange(n_dofs, dtype=np.int32).reshape((Nr, Nz))

    # Lists to hold the COO matrix data
    rows = []
    cols = []
    data = []

    # --- DIAGONAL (P) ---
    # Equation: +aP * phi_P ... = b
    rows.append(idx_grid.flatten())
    cols.append(idx_grid.flatten())
    data.append(aP.flatten())

    # --- EAST (-aE * phi_E) ---
    # Valid where i < Nr-1
    # Current node: i,   Neighbor: i+1
    current_ids = idx_grid[:-1, :].flatten()
    neighbor_ids = idx_grid[1:, :].flatten()
    values = -aE[:-1, :].flatten() # Move to LHS => becomes negative
    
    rows.append(current_ids)
    cols.append(neighbor_ids)
    data.append(values)

    # --- WEST (-aW * phi_W) ---
    # Valid where i > 0
    # Current node: i,   Neighbor: i-1
    current_ids = idx_grid[1:, :].flatten()
    neighbor_ids = idx_grid[:-1, :].flatten()
    values = -aW[1:, :].flatten()
    
    rows.append(current_ids)
    cols.append(neighbor_ids)
    data.append(values)

    # --- NORTH (-aN * phi_N) ---
    # Valid where j < Nz-1
    # Current node: j,   Neighbor: j+1
    current_ids = idx_grid[:, :-1].flatten()
    neighbor_ids = idx_grid[:, 1:].flatten()
    values = -aN[:, :-1].flatten()
    
    rows.append(current_ids)
    cols.append(neighbor_ids)
    data.append(values)

    # --- SOUTH (-aS * phi_S) ---
    # Valid where j > 0
    # Current node: j,   Neighbor: j-1
    current_ids = idx_grid[:, 1:].flatten()
    neighbor_ids = idx_grid[:, :-1].flatten()
    values = -aS[:, 1:].flatten()
    
    rows.append(current_ids)
    cols.append(neighbor_ids)
    data.append(values)

    # 3. Concatenate and Build
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)
    
    # Construct CSR Matrix (efficient for arithmetic/solving)
    A = sparse.coo_matrix((data, (rows, cols)), shape=(n_dofs, n_dofs)).tocsc()
    
    # 4. Solve
    # splu is the fastest direct solver for general unsymmetric matrices
    solve = splu(A)
    phi_vec = solve.solve(b.flatten())
    
    return phi_vec.reshape((Nr, Nz))


def solve_Te_diffusion_direct(Te, ne, kappa_par, kappa_perp, 
                              br, bz, mask, dr, dz, r_coords, dt, 
                              mi, Te_floor,
                              i_cathode_r, j_cathode_r,
                              i_cathode_z, j_cathode_z,
                              ur=None, uz=None,
                              verbose=False, closed_top=False, delta_sheath=1.0, include_advection=False):
    """
    Direct Solver for Anisotropic Electron Heat Diffusion.
    Uses Splu (SuperLU) for robust, instant solution.
    """
    Nr, Nz = Te.shape
    
    # 1. Assemble Coefficients (Numba)
    # We pass 'Te' as the "old" temperature for linearization of non-linear terms (sheath)
    if(include_advection):
        if(ur is None or uz is None):
            print("Error, uer or uez not passed, None values")

        aP, aE, aW, aN, aS, b = assemble_Te_advection_diffusion_FD(Te, ne, ur, uz,
                                     kappa_par, kappa_perp, 
                                     br, bz, mask, dr, dz, r_coords, dt, 
                                     mi, closed_top, 
                                     i_cathode_r, j_cathode_r,
                                     i_cathode_z, j_cathode_z,
                                     delta_sheath=1.0)

    else:
        aP, aE, aW, aN, aS, b = assemble_Te_diffusion_FD(
            Te, ne, kappa_par, kappa_perp, 
            br, bz, mask, dr, dz, r_coords, dt, 
            mi, closed_top,
            i_cathode_r, j_cathode_r,
            i_cathode_z, j_cathode_z,
            delta_sheath=delta_sheath
        )
    
    # 2. Direct Solve (Scipy/SuperLU)
    # Uses the same helper you wrote for the Poisson solver
    Te_new = solve_direct_fast(Nr, Nz, aP, aE, aW, aN, aS, b)
    
    # 3. Apply Physics Floor
    Te_new = np.maximum(Te_new, Te_floor)
    
    # 4. Mask Cleanup (Force solids to 0 or keeping T_old)
    # The assembly fixes them to T_old, but the solve might drift slightly due to precision.
    # Strictly speaking, masked values don't matter for the plasma physics.
    
    if verbose:
        diff = np.max(np.abs(Te_new - Te))
        print(f"[Te-DIRECT] Solved. Max Delta T = {diff:.3f} K")
        
    return Te_new


@njit(cache=True)
def compute_ambipolar_coefficients(Te, Ti, nu_in, beta_e, beta_i, mi, kb):
    """
    Returns the Parallel and Perpendicular Ambipolar Diffusion Coefficients [m^2/s]
    """
    Nr, Nz = Te.shape
    Da_par = np.zeros((Nr, Nz))
    Da_perp = np.zeros((Nr, Nz))
    
    for i in range(Nr):
        for j in range(Nz):
            nu_safe = max(nu_in[i, j], 1e-5)
            # Classical Ambipolar Diffusion D = k(Te+Ti)/(m*nu)
            Da_p = (kb * (Te[i, j] + Ti[i, j])) / (mi * nu_safe)
            
            # Magnetized reduction
            magnetization_factor = 1.0 + beta_e[i, j] * beta_i[i, j]
            
            Da_par[i, j] = Da_p
            Da_perp[i, j] = Da_p / magnetization_factor
            
    return Da_par, Da_perp



@njit(cache=True)
def assemble_ne_diffusion_FV(ne_star, Da_par, Da_perp, 
                             mask, dr, dz, r_coords, dt, 
                             Te, mi, 
                             i_cathode, j_cathode, Jiz_grid, # Jiz_grid unused now
                             delta_sheath=1.0, 
                             closed_top=True):
    """
    Assembles Matrix for ne diffusion with explicit Cathode Sink (Bohm Flux).
    """
    Nr, Nz = ne_star.shape
    kb = constants.kb
    qe = constants.q_e
    
    aP = np.zeros((Nr, Nz))
    aE = np.zeros((Nr, Nz))
    aW = np.zeros((Nr, Nz))
    aN = np.zeros((Nr, Nz))
    aS = np.zeros((Nr, Nz))
    b  = np.zeros((Nr, Nz))
    
    # Map not strictly needed if we iterate the indices directly, 
    # but good for safety if we need to check neighbors later.
    cathode_flag = np.zeros((Nr, Nz), dtype=np.int32)
    for k in range(len(i_cathode)):
        ic = i_cathode[k]
        jc = j_cathode[k]
        if ic < Nr and jc < Nz:
            cathode_flag[ic, jc] = 1

    for i in range(Nr):
        r_i = r_coords[i]
        
        # --- Cell Geometry ---
        if i == 0: 
            r_edge = 0.5 * dr
            Vol = np.pi * r_edge**2 * dz
            Area_E = 2.0 * np.pi * r_edge * dz
            Area_W = 0.0 
            Area_N = np.pi * r_edge**2
            Area_S = np.pi * r_edge**2
            dist_E = dr
            dist_W = 1.0 
        else:
            Vol = 2.0 * np.pi * r_i * dr * dz
            r_east = r_i + 0.5*dr
            r_west = r_i - 0.5*dr
            Area_E = 2.0 * np.pi * r_east * dz
            Area_W = 2.0 * np.pi * r_west * dz
            Area_N = 2.0 * np.pi * r_i * dr
            Area_S = 2.0 * np.pi * r_i * dr
            dist_E = dr
            dist_W = dr
            
        dist_N = dz
        dist_S = dz
        
        for j in range(Nz):
            if mask[i, j] == 0:
                aP[i, j] = 1.0
                b[i, j] = ne_star[i, j]
                continue
            
            cs = np.sqrt(kb * Te[i, j] / mi)
            D_par_loc = Da_par[i, j]
            D_perp_loc = Da_perp[i, j]
            
            has_N = (j < Nz-1) and (mask[i, j+1] == 1)
            has_S = (j > 0)    and (mask[i, j-1] == 1)
            has_E = (i < Nr-1) and (mask[i+1, j] == 1)
            has_W = (i > 0)    and (mask[i-1, j] == 1)
            
            # --- Standard Faces ---
            if has_E:
                D_face = 0.5 * (D_perp_loc + Da_perp[i+1, j])
                aE[i, j] = D_face * Area_E / dist_E
            elif i == Nr-1: # R_MAX Bohm Sink
                aP[i, j] += delta_sheath * cs * Area_E

            if has_W:
                D_face = 0.5 * (D_perp_loc + Da_perp[i-1, j])
                aW[i, j] = D_face * Area_W / dist_W

            if has_N:
                D_face = 0.5 * (D_par_loc + Da_par[i, j+1])
                aN[i, j] = D_face * Area_N / dist_N
            elif j == Nz-1 and closed_top: # Z_MAX Bohm Sink
                aP[i, j] += delta_sheath * cs * Area_N
                
            if has_S:
                D_face = 0.5 * (D_par_loc + Da_par[i, j-1])
                aS[i, j] = D_face * Area_S / dist_S
            elif j == 0: # Z_MIN Bohm Sink
                aP[i, j] += delta_sheath * cs * Area_S

            # --- Assembly ---
            inertia = Vol / dt
            sum_diff = aE[i, j] + aW[i, j] + aN[i, j] + aS[i, j]
            aP[i, j] += inertia + sum_diff
            b[i, j] = inertia * ne_star[i, j]

    """
    # --- 4. Explicit Cathode Sink (Internal Boundary) ---
    # Since the cathode is masked, the main loop sees it as insulating (Neumann=0).
    # We must explicitly add the Bohm sink term.
    for k in range(len(i_cathode)):
        i = i_cathode[k]
        j = j_cathode[k]
        
        # Ensure we are on a valid plasma node
        if i < Nr and j < Nz and mask[i, j] == 1:
            # We assume cathode is the "South" neighbor (at j-1)
            # Area_S was calculated implicitly above, we reconstruct it:
            if i == 0:
                r_edge = 0.5 * dr
                Area_S = np.pi * r_edge**2
            else:
                r_i = r_coords[i]
                Area_S = 2.0 * np.pi * r_i * dr
            
            cs = np.sqrt(kb * Te[i, j] / mi)
            
            # Add Sink to Diagonal
            # Flux = delta * n * cs
            # Term = Flux * Area = (delta * cs * Area) * n
            aP[i, j] += delta_sheath * cs * Area_S
    """

    return aP, aE, aW, aN, aS, b


@njit(cache=True)
def compute_boundary_recycling_source(ne_new, Te, mi, mask, 
                                      dr, dz, r_coords, dt, 
                                      i_cathode, j_cathode, Jiz_grid,
                                      delta_sheath=1.0, closed_top=True):
    """
    Calculates neutral source from wall recycling.
    """
    Nr, Nz = ne_new.shape
    dne_src = np.zeros((Nr, Nz))
    kb = constants.kb
    qe = constants.q_e
    
    inv_dr = 1.0 / dr
    inv_dz = 1.0 / dz
    
    # 1. Standard Bohm Recycling (Outer Walls)
    for i in range(Nr):
        for j in range(Nz):
            if mask[i, j] == 0: continue
                
            cs = np.sqrt(kb * Te[i, j] / mi)
            flux_mag = delta_sheath * ne_new[i, j] * cs
            
            if j == 0:
                dne_src[i, j] += flux_mag * dt * inv_dz
            if j == Nz-1 and closed_top:
                dne_src[i, j] += flux_mag * dt * inv_dz
            if i == Nr-1:
                dne_src[i, j] += flux_mag * dt * inv_dr

    """
    # 2. Cathode Recycling (Bohm Flux)
    for k in range(len(i_cathode)):
        i = i_cathode[k]
        j = j_cathode[k]
        
        if i < Nr and j < Nz and mask[i, j] == 1:
            cs = np.sqrt(kb * Te[i, j] / mi)
            
            # Flux = delta * n * cs
            flux_mag = delta_sheath * ne_new[i, j] * cs
            
            # Add neutrals to this cell
            dne_src[i, j] += flux_mag * dt * inv_dz
    """

    return dne_src