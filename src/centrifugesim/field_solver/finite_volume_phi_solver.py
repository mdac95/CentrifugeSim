import numpy as np
from numba import njit, prange

from scipy import sparse
from scipy.sparse.linalg import splu, lgmres, LinearOperator, bicgstab, spilu

from centrifugesim import constants

@njit(parallel=True, fastmath=True, cache=True)
def _compute_EJ_flux_based(phi, mask_u8, r_c, r_e, z_c, dz_j, dr_i,
                           sigma_P, sigma_par,
                           ne, pe, Bz, un_theta, Ji_r, Ji_z,
                           ne_floor, echarge,
                           use_pe, use_rot, use_ji,
                           fill_solid_with_nan):
    """
    Reconstructs J and E consistent with the Finite Volume fluxes.
    Instead of differentiating phi, we calculate the flux across faces 
    and average them to the cell center.
    """
    Nr, Nz = phi.shape
    Jr = np.zeros((Nr, Nz), dtype=phi.dtype)
    Jz = np.zeros((Nr, Nz), dtype=phi.dtype)
    Er = np.zeros((Nr, Nz), dtype=phi.dtype)
    Ez = np.zeros((Nr, Nz), dtype=phi.dtype)
    
    # Gradient of pressure diagnostic terms (optional)
    Er_gradpe = np.zeros((Nr, Nz), dtype=phi.dtype)
    Ez_gradpe = np.zeros((Nr, Nz), dtype=phi.dtype)

    EPS = 1e-30

    # Parallel loop over cells
    for j in prange(Nz):
        for i in range(Nr):
            # 1. Skip solids
            if mask_u8[i, j] == 0:
                if fill_solid_with_nan:
                    Jr[i, j] = np.nan; Jz[i, j] = np.nan
                    Er[i, j] = np.nan; Ez[i, j] = np.nan
                continue

            # ---------------------------------------------------------
            # RADIAL FLUX (Jr) RECONSTRUCTION
            # ---------------------------------------------------------
            # We need Flux_East (i+1/2) and Flux_West (i-1/2)
            
            # --- EAST FACE (i to i+1) ---
            flux_east = 0.0
            if i < Nr - 1 and mask_u8[i+1, j] == 1:
                # Standard internal face
                dr_face = r_c[i+1] - r_c[i]
                r_face  = r_e[i]
                
                # Harmonic mean conductivity (consistent with solver)
                sP_L = sigma_P[i, j]
                sP_R = sigma_P[i+1, j]
                sig_face = (2.0 * sP_L * sP_R) / (sP_L + sP_R + EPS)
                
                # Terms: -sigma * dphi/dr
                dphi = phi[i+1, j] - phi[i, j]
                flux_east = -sig_face * (dphi / dr_face)
                
                # Add Pressure & Rotation terms to flux
                if use_pe:
                    dpe = pe[i+1, j] - pe[i, j]
                    n_avg = 0.5 * (ne[i, j] + ne[i+1, j])
                    inv_ne = 1.0 / (echarge * max(n_avg, ne_floor))
                    term = -sig_face * (inv_ne * (dpe / dr_face))
                    flux_east += term
                    # (Diagnostic only: store contribution)
                    Er_gradpe[i, j] += 0.5 * term / (sig_face + EPS) # Approximate center contrib

                if use_rot:
                    Bz_face = 0.5 * (Bz[i, j] + Bz[i+1, j])
                    ut_face = 0.5 * (un_theta[i, j] + un_theta[i+1, j])
                    flux_east += sig_face * (Bz_face * ut_face)

            elif i < Nr - 1:
                # Boundary: Plasma (i) -> Solid (i+1) (Anode or Wall)
                # If Anode (Dirichlet), we calculate flux. If Insulator, flux is 0.
                flux_east = 0.0 

            # --- WEST FACE (i-1 to i) ---
            flux_west = 0.0
            if i > 0 and mask_u8[i-1, j] == 1:
                # Symmetric to East logic, but looking back
                dr_face = r_c[i] - r_c[i-1]
                # Harmonic mean
                sP_L = sigma_P[i-1, j]
                sP_R = sigma_P[i, j]
                sig_face = (2.0 * sP_L * sP_R) / (sP_L + sP_R + EPS)
                
                dphi = phi[i, j] - phi[i-1, j]
                flux_west = -sig_face * (dphi / dr_face)
                
                if use_pe:
                    dpe = pe[i, j] - pe[i-1, j]
                    n_avg = 0.5 * (ne[i-1, j] + ne[i, j])
                    inv_ne = 1.0 / (echarge * max(n_avg, ne_floor))
                    flux_west += -sig_face * (inv_ne * (dpe / dr_face))
                
                if use_rot:
                    Bz_face = 0.5 * (Bz[i-1, j] + Bz[i, j])
                    ut_face = 0.5 * (un_theta[i-1, j] + un_theta[i, j])
                    flux_west += sig_face * (Bz_face * ut_face)
            
            elif i > 0:
                # Boundary: Solid (i-1) -> Plasma (i)
                # This catches the Cathode Side Wall (Insulator).
                # Since mask[i-1] is 0, flux_west is strictly 0.0.
                flux_west = 0.0

            # Average fluxes to center
            Jr[i, j] = 0.5 * (flux_east + flux_west)


            # ---------------------------------------------------------
            # AXIAL FLUX (Jz) RECONSTRUCTION
            # ---------------------------------------------------------
            flux_north = 0.0
            if j < Nz - 1 and mask_u8[i, j+1] == 1:
                dz_face = z_c[j+1] - z_c[j]
                sPar_S = sigma_par[i, j]
                sPar_N = sigma_par[i, j+1]
                sig_face = (2.0 * sPar_S * sPar_N) / (sPar_S + sPar_N + EPS)
                
                dphi = phi[i, j+1] - phi[i, j]
                flux_north = -sig_face * (dphi / dz_face)
                
                if use_pe:
                    dpe = pe[i, j+1] - pe[i, j]
                    n_avg = 0.5 * (ne[i, j] + ne[i, j+1])
                    inv_ne = 1.0 / (echarge * max(n_avg, ne_floor))
                    term = -sig_face * (inv_ne * (dpe / dz_face))
                    flux_north += term
                    Ez_gradpe[i, j] += 0.5 * term / (sig_face + EPS)

            flux_south = 0.0
            if j > 0 and mask_u8[i, j-1] == 1:
                dz_face = z_c[j] - z_c[j-1]
                sPar_S = sigma_par[i, j-1]
                sPar_N = sigma_par[i, j]
                sig_face = (2.0 * sPar_S * sPar_N) / (sPar_S + sPar_N + EPS)
                
                dphi = phi[i, j] - phi[i, j-1]
                flux_south = -sig_face * (dphi / dz_face)

                if use_pe:
                    dpe = pe[i, j] - pe[i, j-1]
                    n_avg = 0.5 * (ne[i, j-1] + ne[i, j])
                    inv_ne = 1.0 / (echarge * max(n_avg, ne_floor))
                    flux_south += -sig_face * (inv_ne * (dpe / dz_face))
            
            # Average fluxes to center
            Jz[i, j] = 0.5 * (flux_north + flux_south)

            # Add Ion Current (Collocated)
            if use_ji:
                Jr[i, j] += Ji_r[i, j]
                Jz[i, j] += Ji_z[i, j]

            # ---------------------------------------------------------
            # BACK-CALCULATE E
            # ---------------------------------------------------------
            # Invert Ohm's law: E = (J_total - J_driver) / sigma
            # This is safer than dphi/dx because it respects the flux limits.
            
            # Subtract drivers from total J to get Ohmic J
            Jr_ohmic = Jr[i, j]
            Jz_ohmic = Jz[i, j]
            
            if use_ji:
                Jr_ohmic -= Ji_r[i, j]
                Jz_ohmic -= Ji_z[i, j]
                        
            # Better approach for E: Use the dphi terms we just calculated
            # Re-calculate simple dphi/dr average
            # (Reuse the logic or just trust the flux division)
            Er[i, j] = Jr_ohmic / (sigma_P[i, j] + EPS)
            Ez[i, j] = Jz_ohmic / (sigma_par[i, j] + EPS)

    return Er, Ez, Jr, Jz, Er_gradpe, Ez_gradpe


def compute_E_and_J(phi, geom,
                    sigma_P, sigma_parallel,
                    ne=None, pe=None,
                    Bz=None, un_theta=None,
                    Ji_r=None, Ji_z=None,
                    ne_floor=1e10,
                    fill_solid_with_nan=False):
    """
    Wrapper for flux-consistent E and J computation.
    """
    # Build geometry faces locally
    (r_c, r_w, r_e, dr_w, dr_e, dr_i,
     z_c, z_s, z_n, dz_s, dz_n, dz_j) = build_geometry_faces(geom.r, geom.z)
    
    # Prepare arrays
    mask_u8 = geom.mask.astype(np.int8)
    phi     = np.asarray(phi, dtype=np.float64)
    sigma_P = np.asarray(sigma_P, dtype=np.float64)
    sigma_par = np.asarray(sigma_parallel, dtype=np.float64)
    
    zero = np.zeros_like(phi)
    use_pe  = (ne is not None) and (pe is not None)
    use_rot = (Bz is not None) and (un_theta is not None)
    use_ji  = (Ji_r is not None) or (Ji_z is not None)

    ne_arr     = np.asarray(ne, dtype=np.float64) if use_pe else zero
    pe_arr     = np.asarray(pe, dtype=np.float64) if use_pe else zero
    Bz_arr     = np.asarray(Bz, dtype=np.float64) if use_rot else zero
    utheta_arr = np.asarray(un_theta, dtype=np.float64) if use_rot else zero
    Ji_r_arr   = np.asarray(Ji_r, dtype=np.float64) if use_ji else zero
    Ji_z_arr   = np.asarray(Ji_z, dtype=np.float64) if use_ji else zero

    echarge = constants.q_e

    return _compute_EJ_flux_based(phi, mask_u8, r_c, r_e, z_c, dz_j, dr_i,
                                  sigma_P, sigma_par,
                                  ne_arr, pe_arr, Bz_arr, utheta_arr, Ji_r_arr, Ji_z_arr,
                                  float(ne_floor), float(echarge),
                                  use_pe, use_rot, use_ji,
                                  fill_solid_with_nan)


@njit(parallel=True, fastmath=True, cache=True)
def _assemble_coefficients_core(
    Nr, Nz,
    r_c, r_w, r_e, dr_w, dr_e, dr_i,
    dz_s, dz_n, dz_j,
    sigP_e, sigPar_n,                 # face conductivities
    sigmaP_cell, sigmaPar_cell,       # cell-centered conductivities
    mask_u8, cathode_u8, anode_u8,    # uint8 masks (1=True, 0=False)
    rmax_dirichlet_u8,                # uint8 of length Nz (1=Dirichlet at rmax for this j)
    g_top_cathode,                    # length Nr: dphi/dz at top-of-cathode, Neumann array
    phi_dirichlet_val,                # typically 0.0 (anode potential)
    S,                                # source term on cell centers
    cathode_dirichlet_val_arr,        # array of length Nr for Dirichlet on top of cathode
    use_cathode_dirichlet             # Boolean flag
    ):
    aE = np.zeros((Nr, Nz))
    aW = np.zeros((Nr, Nz))
    aN = np.zeros((Nr, Nz))
    aS = np.zeros((Nr, Nz))
    aP = np.zeros((Nr, Nz))
    b  = np.zeros((Nr, Nz))

    for j in prange(Nz):
        for i in range(Nr):
            # -----------------------------------------------------------------
            # 1. Solid Mask Check
            # -----------------------------------------------------------------
            if mask_u8[i, j] == 0:
                aP[i, j] = 1.0
                aE[i, j] = 0.0; aW[i, j] = 0.0; aN[i, j] = 0.0; aS[i, j] = 0.0
                b[i, j]  = phi_dirichlet_val
                continue

            # -----------------------------------------------------------------
            # 2. Clamped Boundary Node Check (FIX #1)
            # If this node is at r=rmax AND is an anode region, force it to 0V.
            # -----------------------------------------------------------------
            if (i == Nr - 1) and (rmax_dirichlet_u8[j] == 1):
                aP[i, j] = 1.0
                aE[i, j] = 0.0; aW[i, j] = 0.0; aN[i, j] = 0.0; aS[i, j] = 0.0
                b[i, j]  = phi_dirichlet_val
                continue

            # -----------------------------------------------------------------
            # 3. EAST Face (Radial +)
            # -----------------------------------------------------------------
            if i < Nr - 1:
                # Is the *next* node the clamped boundary?
                neighbor_is_clamped = ((i + 1) == Nr - 1) and (rmax_dirichlet_u8[j] == 1)

                if mask_u8[i+1, j] == 1:
                    # Plasma-Plasma Interface
                    if neighbor_is_clamped:
                        # (FIX #2) UPWIND CONDUCTIVITY
                        # Ignore the low conductivity of the wall node. 
                        # Use the current cell's conductivity to drive current into the 0V boundary.
                        ke = sigmaP_cell[i, j] 
                        ae = (r_e[i] * ke) / (dr_e[i] * dr_i[i])
                        aP[i, j] += ae
                        b[i, j]  += ae * phi_dirichlet_val # Neighbor phi is fixed
                        aE[i, j]  = 0.0
                    else:
                        # Standard Harmonic Mean
                        ke = sigP_e[i, j]
                        ae = (r_e[i] * ke) / (dr_e[i] * dr_i[i])
                        aE[i, j] = ae
                else:
                    # Plasma-Solid Interface (Internal Anode)
                    if anode_u8[i+1, j] == 1:
                        # Dirichlet Anode
                        ke = sigmaP_cell[i, j]
                        ae_face = (r_e[i] * ke) / (dr_e[i] * dr_i[i])
                        aP[i, j] += ae_face
                        b[i, j]  += ae_face * phi_dirichlet_val
                        aE[i, j]  = 0.0
                    else:
                        # Neumann (Cathode/Insulator)
                        aE[i, j] = 0.0
            else:
                # r = rmax boundary (Neumann case only, Dirichlet caught by Fix #1)
                aE[i, j] = 0.0

            # -----------------------------------------------------------------
            # 4. WEST Face (Radial -)
            # -----------------------------------------------------------------
            if i > 0:
                if mask_u8[i-1, j] == 1:
                    kw = sigP_e[i-1, j]
                    aw = (r_w[i] * kw) / (dr_w[i] * dr_i[i])
                    aW[i, j] = aw
                else:
                    if anode_u8[i-1, j] == 1:
                        kw = sigmaP_cell[i, j]
                        aw_face = (r_w[i] * kw) / (dr_w[i] * dr_i[i])
                        aP[i, j] += aw_face
                        b[i, j]  += aw_face * phi_dirichlet_val
                        aW[i, j]  = 0.0
                    else:
                        aW[i, j] = 0.0
            else:
                aW[i, j] = 0.0

            # -----------------------------------------------------------------
            # 5. NORTH Face (Axial +)
            # -----------------------------------------------------------------
            if j < Nz - 1:
                if mask_u8[i, j+1] == 1:
                    kn = sigPar_n[i, j]
                    an = (r_c[i] * kn) / (dz_n[j] * dz_j[j])
                    aN[i, j] = an
                else:
                    if anode_u8[i, j+1] == 1:
                        kn = sigmaPar_cell[i, j]
                        an_face = (r_c[i] * kn) / (dz_n[j] * dz_j[j])
                        aP[i, j] += an_face
                        b[i, j]  += an_face * phi_dirichlet_val
                        aN[i, j]  = 0.0
                    else:
                        aN[i, j] = 0.0
            else:
                aN[i, j] = 0.0

            # -----------------------------------------------------------------
            # 6. SOUTH Face (Axial -)
            # -----------------------------------------------------------------
            if j > 0:
                if mask_u8[i, j-1] == 1:
                    ks = sigPar_n[i, j-1]
                    as_ = (r_c[i] * ks) / (dz_s[j] * dz_j[j])
                    aS[i, j] = as_
                else:
                    if anode_u8[i, j-1] == 1:
                        ks = sigmaPar_cell[i, j]
                        as_face = (r_c[i] * ks) / (dz_s[j] * dz_j[j])
                        aP[i, j] += as_face
                        b[i, j]  += as_face * phi_dirichlet_val
                        aS[i, j]  = 0.0
                    elif cathode_u8[i, j-1] == 1:
                        if use_cathode_dirichlet:
                            val = cathode_dirichlet_val_arr[i]
                            if(not np.isnan(val)):
                                ks = sigmaPar_cell[i, j]
                                as_face = (r_c[i] * ks) / (dz_s[j] * dz_j[j])
                                aP[i, j] += as_face
                                b[i, j]  += as_face * val 
                                aS[i, j]  = 0.0
                            else:
                                aS[i, j]  = 0.0
                        else:
                            ks = sigmaPar_cell[i, j]
                            qzS = ks * g_top_cathode[i]
                            b[i, j] += r_c[i] * qzS / dz_j[j]
                            aS[i, j]  = 0.0
                    else:
                        aS[i, j] = 0.0
            else:
                aS[i, j] = 0.0

            # -----------------------------------------------------------------
            # 7. Final Assembly
            # -----------------------------------------------------------------
            aP[i, j] += (aE[i, j] + aW[i, j] + aN[i, j] + aS[i, j])
            b[i, j]  += r_c[i] * S[i, j]

    return aP, aE, aW, aN, aS, b

def build_geometry_faces(r, z):
    r = np.asarray(r)
    z = np.asarray(z)
    Nr = r.size
    Nz = z.size

    # radial faces and widths
    r_c = r.copy()
    r_e = np.empty_like(r_c)
    r_w = np.empty_like(r_c)
    dr_e = np.empty_like(r_c)
    dr_w = np.empty_like(r_c)

    r_e[:-1] = 0.5*(r[:-1] + r[1:])
    r_e[-1]  = r[-1] + 0.5*(r[-1] - r[-2]) if Nr > 1 else r[-1]
    dr_e[:-1] = r[1:] - r[:-1]
    dr_e[-1]  = (r[-1] - r[-2]) if Nr > 1 else 1.0

    r_w[0]    = 0.0
    r_w[1:]   = 0.5*(r[:-1] + r[1:])
    dr_w[0]   = (r[1] - r[0]) if Nr > 1 else 1.0
    dr_w[1:]  = r[1:] - r[:-1]

    dr_i = r_e - r_w  # control-volume radial width

    # axial faces and widths
    z_c = z.copy()
    z_n = np.empty_like(z_c)
    z_s = np.empty_like(z_c)
    dz_n = np.empty_like(z_c)
    dz_s = np.empty_like(z_c)

    z_n[:-1] = 0.5*(z[:-1] + z[1:])
    z_n[-1]  = z[-1] + 0.5*(z[-1] - z[-2]) if Nz > 1 else z[-1]
    dz_n[:-1] = z[1:] - z[:-1]
    dz_n[-1]  = (z[-1] - z[-2]) if Nz > 1 else 1.0

    z_s[0]   = z[0]
    z_s[1:]  = 0.5*(z[:-1] + z[1:])
    dz_s[0]  = 0.5*(z[1] - z[0]) if Nz > 1 else 1.0
    dz_s[1:] = z[1:] - z[:-1]

    dz_j = z_n - z_s  # control-volume axial width

    return (r_c, r_w, r_e, dr_w, dr_e, dr_i,
            z_c, z_s, z_n, dz_s, dz_n, dz_j)


def build_face_conductivities(sigmaP, sigmaPar, eps=1e-30):
    # radial faces (east): shape (Nr-1, Nz)
    sigP_e = (2.0 * sigmaP[:-1, :] * sigmaP[1:, :]) / (sigmaP[:-1, :] + sigmaP[1:, :] + eps)

    # axial faces (north): shape (Nr, Nz-1)
    sigPar_n = (2.0 * sigmaPar[:, :-1] * sigmaPar[:, 1:]) / (sigmaPar[:, :-1] + sigmaPar[:, 1:] + eps)

    return sigP_e, sigPar_n


def assemble_coefficients(
    r, z, sigmaP, sigmaPar,
    geom,
    dphi_dz_cathode_top=None,         # shape (Nr,), gradient at z=zmax_cathode (on plasma side)
    cathode_voltage_profile=None,     # If not None, overrides dphi_dz
    phi_anode_value=0.0,
    S=None,
    float_outer_wall_top=False
):
    """
    Assemble FV coefficients for:
        (1/r) ∂_r [ r σ_P ∂_r φ ] + ∂_z [ σ_|| ∂_z φ ] = -S

    Boundary conditions (from geom):
      - Axis r=0:                         Neumann (∂φ/∂r = 0)
      - z=0 outside cathode footprint:    Neumann (∂φ/∂z = 0)
      - z=zmax:                           Neumann (∂φ/∂z = 0)
      - Cathode vertical face:            Neumann (∂φ/∂r = 0)
      - Cathode top (z=zmax_cathode):     Neumann array ∂φ/∂z = dphi_dz_cathode_top[i]
      - Any plasma↔anode face:            Dirichlet φ = phi_anode_value
      - r=rmax:  Neumann for z < zmin_anode, Dirichlet (φ=phi_anode_value) for z ≥ zmin_anode
    """
    r = np.asarray(r); z = np.asarray(z)
    Nr = r.size; Nz = z.size

    (r_c, r_w, r_e, dr_w, dr_e, dr_i,
     z_c, z_s, z_n, dz_s, dz_n, dz_j) = build_geometry_faces(r, z)

    sigP_e, sigPar_n = build_face_conductivities(sigmaP, sigmaPar)

    # Masks: convert to uint8 for numba
    mask_u8    = geom.mask.astype(np.uint8)
    cathode_u8 = geom.cathode_mask.astype(np.uint8)
    anode_u8   = (geom.anode1_mask | geom.anode2_mask).astype(np.uint8)

    # Outer wall r=rmax: Dirichlet φ=0 for z >= zmin_anode, else Neumann zero flux
    rmax_dirichlet_by_j = (z >= geom.zmin_anode).astype(np.uint8)

    if float_outer_wall_top:
        rmax_dirichlet_by_j = rmax_dirichlet_by_j & (z <= geom.zmax_anode)
    
    # Convert to uint8 for Numba
    rmax_dirichlet_by_j = rmax_dirichlet_by_j.astype(np.uint8)

    # Cathode top Neumann array
    if dphi_dz_cathode_top is None:
        g_top = np.zeros(Nr, dtype=sigmaP.dtype)
    else:
        g_top = np.asarray(dphi_dz_cathode_top, dtype=sigmaP.dtype)
        if g_top.shape != (Nr,):
            raise ValueError("dphi_dz_cathode_top must be shape (Nr,)")

    # Dirichlet Array (New)
    if cathode_voltage_profile is not None:
        use_cathode_dirichlet = True
        c_val_arr = np.asarray(cathode_voltage_profile, dtype=sigmaP.dtype)
        if c_val_arr.shape != (Nr,):
             # Broadcast scalar if user was lazy and sent a float
             if c_val_arr.size == 1:
                 c_val_arr = np.full(Nr, float(c_val_arr), dtype=sigmaP.dtype)
             else:
                 raise ValueError("cathode_voltage_profile must be shape (Nr,) or scalar")
    else:
        use_cathode_dirichlet = False
        c_val_arr = np.zeros(Nr, dtype=sigmaP.dtype) # Dummy array

    # Source term
    if S is None:
        S = np.zeros((Nr, Nz), dtype=sigmaP.dtype)
    # Do not source inside solids
    S = np.where(mask_u8 == 1, S, 0.0)

    aP, aE, aW, aN, aS, b = _assemble_coefficients_core(
        Nr, Nz,
        r_c, r_w, r_e, dr_w, dr_e, dr_i,
        dz_s, dz_n, dz_j,
        sigP_e, sigPar_n,
        sigmaP, sigmaPar,
        mask_u8, cathode_u8, anode_u8,
        rmax_dirichlet_by_j,
        g_top, phi_anode_value, S,
        c_val_arr,
        use_cathode_dirichlet
    )
    return aP, aE, aW, aN, aS, b


def compute_source_S(r, z,
                     sigma_P, sigma_parallel,
                     ne, pe,
                     ne_floor,
                     Bz=None, un_theta=None,
                     Ji_r=None, Ji_z=None,
                     mask=None,
                     eps=1e-30):
    """
    Build S(r,z) for:
        -(1/r) ∂_r [ r σ_P ∂_r φ ] - ∂_z [ σ_|| ∂_z φ ] = S

    where
        -S = (1/r) ∂_r { r σ_P [ (1/e ne) ∂_r p_e + Bz u_{nθ} ] }
            + ∂_z { σ_|| [ (1/e ne) ∂_z p_e ] }
            + (1/r) ∂_r ( r J_{i,r} ) + ∂_z J_{i,z}

    Notes
    -----
    * Uses the *plus* sign in front of the ion-current divergence terms.
    * ne, pe, sigma_P, sigma_parallel are cell-centered (shape Nr x Nz).
    * Ji_r and Ji_z are cell-centered; they are averaged to faces internally.
    * Bz and un_theta are optional; if omitted, that term is treated as zero.
    * Returns S on cell centers (Nr x Nz), ready to pass into assemble_coefficients.
    """
    Nr, Nz = sigma_P.shape
    echarge = constants.q_e

    # --- geometry and face conductivities (consistent with your assembly) ---
    (r_c, r_w, r_e, dr_w, dr_e, dr_i,
     z_c, z_s, z_n, dz_s, dz_n, dz_j) = build_geometry_faces(r, z)

    sigP_e, sigPar_n = build_face_conductivities(sigma_P, sigma_parallel)

    if mask is not None:
        mu = (mask.astype(np.uint8) != 0)
        face_r = (mu[:-1, :] & mu[1:, :]).astype(float)     # (Nr-1, Nz)
        face_z = (mu[:, :-1] & mu[:,  1:]).astype(float)    # (Nr, Nz-1)
    else:
        face_r = 1.0
        face_z = 1.0

    sigP_e   = sigP_e   * face_r
    sigPar_n = sigPar_n * face_z

    # --- helpers for padding face quantities to cell centers ---
    def pad_east_faces(F_e):       # (Nr-1, Nz) -> (Nr, Nz), east-face flux for each cell
        G = np.zeros((Nr, Nz), dtype=F_e.dtype)
        G[:-1, :] = F_e
        return G

    def pad_west_faces(F_e):       # west-face flux for each cell (east flux of i-1 cell)
        G = np.zeros((Nr, Nz), dtype=F_e.dtype)
        G[1:, :] = F_e
        return G

    def pad_north_faces(F_n):      # (Nr, Nz-1) -> (Nr, Nz), north-face flux for each cell
        G = np.zeros((Nr, Nz), dtype=F_n.dtype)
        G[:, :-1] = F_n
        return G

    def pad_south_faces(F_n):      # south-face flux for each cell (north flux of j-1 cell)
        G = np.zeros((Nr, Nz), dtype=F_n.dtype)
        G[:, 1:] = F_n
        return G

    # --- face-centered 1/(e ne) and pe gradients ---
    # radial faces (between i and i+1)
    ne_e = 0.5 * (ne[:-1, :] + ne[1:, :])
    inv_e_ne_e = 1.0 / (echarge * np.maximum(ne_e, ne_floor))
    dpdr_e = (pe[1:, :] - pe[:-1, :]) / (dr_e[:-1][:, None]) * face_r  # (Nr-1, Nz)

    # axial faces (between j and j+1)
    ne_n = 0.5 * (ne[:, :-1] + ne[:, 1:])
    inv_e_ne_n = 1.0 / (echarge * np.maximum(ne_n, ne_floor))
    dpdz_n = (pe[:, 1:] - pe[:, :-1]) / (dz_n[:-1][None, :]) * face_z  # (Nr, Nz-1)

    # --- "pressure battery" + neutral-rotation (Bz uθ) fluxes ---
    # radial fluxes: F_r = -σP*(1/e ne)*∂r pe - σP*(Bz uθ)
    Fr = -sigP_e * inv_e_ne_e * dpdr_e
    if (Bz is not None) and (un_theta is not None):
        Bz_e = 0.5 * (Bz[:-1, :] + Bz[1:, :])
        u_e  = 0.5 * (un_theta[:-1, :] + un_theta[1:, :])
        Fr -= sigP_e * (Bz_e * u_e)

    # axial fluxes: F_z = -σ||*(1/e ne)*∂z pe
    Fz = -sigPar_n * inv_e_ne_n * dpdz_n

    if mask is not None:
        mu8 = (mask.astype(np.uint8) != 0)
        # radial faces exist only if both adjacent cells are plasma
        face_r = (mu8[:-1,:] & mu8[1:,:]).astype(Fr.dtype)     # (Nr-1,Nz)
        # axial faces exist only if both adjacent cells are plasma
        face_z = (mu8[:, :-1] & mu8[:,  1:]).astype(Fz.dtype)  # (Nr,Nz-1)
        Fr *= face_r
        Fz *= face_z

    # --- divergence of those fluxes (axisymmetric in r) ---
    # radial: (1/r) ∂_r( r Fr ) -> ( r_e*Fr_e - r_w*Fr_w ) / (r_c*dr_i)
    Fr_e_pad = pad_east_faces(Fr)
    Fr_w_pad = pad_west_faces(Fr)
    S_r = (r_e[:, None] * Fr_e_pad - r_w[:, None] * Fr_w_pad) / (np.maximum(r_c, eps)[:, None] * (dr_i[:, None] + eps))

    # axial: ∂_z(Fz) -> ( Fz_north - Fz_south ) / dz_j
    Fz_n_pad = pad_north_faces(Fz)
    Fz_s_pad = pad_south_faces(Fz)
    S_z = (Fz_n_pad - Fz_s_pad) / (dz_j[None, :] + eps)

    # --- ion current divergence ---
    # face-average the supplied cell-centered currents
    if Ji_r is not None:
        Jir_e = 0.5 * (Ji_r[1:, :] + Ji_r[:-1, :]) * face_r # (Nr-1, Nz)
    else:
        Jir_e = np.zeros((Nr-1, Nz), dtype=sigma_P.dtype)

    if Ji_z is not None:
        Jiz_n = 0.5 * (Ji_z[:, 1:] + Ji_z[:, :-1]) * face_z   # (Nr, Nz-1)
    else:
        Jiz_n = np.zeros((Nr, Nz-1), dtype=sigma_P.dtype)

    if mask is not None:
        Jir_e *= face_r
        Jiz_n *= face_z

    Jir_e_pad = pad_east_faces(Jir_e)
    Jir_w_pad = pad_west_faces(Jir_e)  # west = east of left cell
    S_Jr = -(r_e[:, None] * Jir_e_pad - r_w[:, None] * Jir_w_pad) / (np.maximum(r_c, eps)[:, None] * (dr_i[:, None] + eps))

    Jiz_n_pad = pad_north_faces(Jiz_n)
    Jiz_s_pad = pad_south_faces(Jiz_n)
    S_Jz = -(Jiz_n_pad - Jiz_s_pad) / (dz_j[None, :] + eps)

    # total source
    S = S_r + S_z + S_Jr + S_Jz
    if mask is not None:
        S = np.where(mu8, S, 0.0)

    return S

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


def solve_anisotropic_poisson_FV_direct(geom,
                                 sigma_P, sigma_parallel,
                                 ne=None, pe=None, Bz=None, un_theta=None,
                                 Ji_r=None, Ji_z=None,
                                 ne_floor=1.0,
                                 dphi_dz_cathode_top=None,  # array (Nr,) at z=zmax_cathode
                                 cathode_voltage_profile=None, # array (Nr,) or None
                                 phi_anode_value=0.0,
                                 float_outer_wall_top=False,
                                 phi0=None, omega=1.8, tol=1e-10, max_iter=50_000,
                                 verbose=True):
    """
    Direct Solver (SuperLU) wrapper for the Finite Volume scheme.
    Replaces SOR iteration with a single sparse matrix solve.
    """
    Nr, Nz = geom.Nr, geom.Nz

    # --- 1. Compute RHS Source ---
    S = compute_source_S(geom.r, geom.z,
                         sigma_P, sigma_parallel,
                         ne, pe,
                         ne_floor,
                         Bz=Bz, un_theta=un_theta,
                         Ji_r=Ji_r, Ji_z=Ji_z,
                         mask=geom.mask)

    # Zero out source at boundaries/axis to avoid noise
    S[0,:] = 0
    S[geom.i_bc_list, geom.j_bc_list] *= 0

    # --- 2. Assemble Coefficients (Numba) ---
    aP, aE, aW, aN, aS, b = assemble_coefficients(
        geom.r, geom.z,
        sigmaP=sigma_P, sigmaPar=sigma_parallel,
        geom=geom,
        dphi_dz_cathode_top=dphi_dz_cathode_top,
        cathode_voltage_profile=cathode_voltage_profile,
        phi_anode_value=phi_anode_value,
        S=S,
        float_outer_wall_top=float_outer_wall_top
    )

    # --- 3. Direct Sparse Solve ---
    # Note: solve_direct_fast builds the matrix A and solves Ax=b
    phi = solve_direct_fast(Nr, Nz, aP, aE, aW, aN, aS, b)

    # --- 4. Compute Residual (Verification) ---
    # Since this is not iterative, we manually calculate ||Ax-b|| to confirm precision.
    # We use numpy array slicing to mimic the neighbor interactions.
    
    # Create shifted views for neighbors (padding with 0 where no neighbor exists)
    phi_E = np.zeros_like(phi); phi_E[:-1, :] = phi[1:, :]   # i+1
    phi_W = np.zeros_like(phi); phi_W[1:, :]  = phi[:-1, :]  # i-1
    phi_N = np.zeros_like(phi); phi_N[:, :-1] = phi[:, 1:]   # j+1
    phi_S = np.zeros_like(phi); phi_S[:, 1:]  = phi[:, :-1]  # j-1

    # LHS = aP*phi - (Neighbors)
    LHS = (aP * phi) - (aE * phi_E + aW * phi_W + aN * phi_N + aS * phi_S)
    
    # Residual = b - LHS
    res_grid = np.abs(b - LHS)
    
    # Mask out solid nodes (where residual is trivially 0)
    res_grid[geom.mask == 0] = 0.0
    
    max_res = np.max(res_grid)

    if verbose:
        print(f"[FV-DIRECT] iterations = 1, residual = {max_res:.3e}")

    return phi, {"iterations": 1, "residual": float(max_res)}