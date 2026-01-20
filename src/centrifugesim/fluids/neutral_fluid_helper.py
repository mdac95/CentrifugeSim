import numpy as np
from numba import njit, prange
from time import time

from centrifugesim import constants
from centrifugesim.geometry.geometry import Geometry

# Lennard–Jones parameters (sigma in meters, eps_over_k in Kelvin)
_LJ_DB = {
    # name: (sigma [m], eps_over_k [K], kind)
    "H2":  (2.92e-10,  59.7,  "diatomic"),
    "He":  (2.576e-10, 10.22, "monatomic"),
    "N2":  (3.798e-10, 91.5,  "diatomic"),
    "O2":  (3.467e-10, 106.7, "diatomic"),
    "Ar":  (3.405e-10, 119.8, "monatomic"),
}

# STABLE (old), problematic if solid nodes inside but works for left z and up r boundaries
"""
@njit
def dpdr_masked(p, dr, i, k, face_r):
    Nr = p.shape[0] # Get grid size
    
    # Force a one-sided Forward Difference: (p[i] - p[i-1]) / dr
    # This prevents noise at i=Nr-1 from affecting the domain.
    if i == Nr - 2:
        # Check if left face is open (standard case)
        if face_r[i, k] == 1:
            return (p[i, k] - p[i-1, k]) / dr
        else:
            return 0.0
    # ------------------------------

    # Standard Central Difference for everywhere else
    right_open = face_r[i+1, k] == 1
    left_open  = face_r[i,   k] == 1
    if right_open and left_open:
        return (p[i+1,k] - p[i-1,k]) / (2*dr)
    elif right_open and not left_open:
        return (p[i+1,k] - p[i,k]) / dr
    elif left_open and not right_open:
        return (p[i,k] - p[i-1,k]) / dr
    else:
        return 0.0

@njit
def dpdz_masked(p, dz, i, k, face_z):
    Nz = p.shape[1]
    
    # Force a one-sided Forward Difference: (p[k+1] - p[k]) / dz
    # This prevents noise at k=0 from affecting the domain.
    if k == 1:
        if face_z[i, k+1] == 1: # Check if "up" face is open
            return (p[i, k+1] - p[i, k]) / dz
        else:
            return 0.0
    # -------------------------------------

    up_open   = face_z[i, k+1] == 1
    down_open = face_z[i, k  ] == 1
    
    if up_open and down_open:
        return (p[i,k+1] - p[i,k-1]) / (2*dz)
    elif up_open and not down_open:
        return (p[i,k+1] - p[i,k]) / dz
    elif down_open and not up_open:
        return (p[i,k] - p[i,k-1]) / dz
    else:
        return 0.0
"""

@njit
def dpdr_masked(p, dr, i, k, face_r, mask_vel):
    """
    Computes dp/dr. 
    1. Checks Internal Walls using mask_vel (for Anodes).
    2. Checks Domain Boundaries using index i (for Outer Wall).
    """
    Nr = p.shape[0]

    # --- Check Right Face (i -> i+1) ---
    right_open = (face_r[i+1, k] == 1)
    if right_open:
        # A. Domain Boundary Protection (The "Nr-2" case)
        # If we are at Nr-2, the right neighbor is Nr-1 (Ghost Wall). Ignore it.
        if i >= Nr - 2:
            right_open = False
        
        # B. Internal Obstacle Protection (The "Anode" case)
        # If neighbor is a Wall Node (locked velocity), treat face as closed.
        elif mask_vel is not None and mask_vel[i+1, k] == 0:
            right_open = False

    # --- Check Left Face (i -> i-1) ---
    left_open = (face_r[i, k] == 1)
    if left_open:
        # A. Domain Boundary Protection
        # If i=0, left is axis (open). If i=1, left is i=0. Usually safe.
        # If you had an inner wall at i=0, you would add `if i <= 1: left_open = False`
        pass 

        # B. Internal Obstacle Protection
        if mask_vel is not None and mask_vel[i-1, k] == 0:
            left_open = False

    # --- Standard Stencil Selection ---
    if right_open and left_open:
        return (p[i+1,k] - p[i-1,k]) / (2*dr)
    elif right_open and not left_open:
        return (p[i+1,k] - p[i,k]) / dr # Forward Difference
    elif left_open and not right_open:
        return (p[i,k] - p[i-1,k]) / dr # Backward Difference (Triggers at Nr-2)
    else:
        return 0.0

@njit
def dpdz_masked(p, dz, i, k, face_z, mask_vel):
    """
    Computes dp/dz.
    1. Checks Internal Walls using mask_vel.
    2. Checks Domain Boundaries using index k.
    """
    Nz = p.shape[1]

    # --- Check Up Face (k -> k+1) ---
    up_open = (face_z[i, k+1] == 1)
    if up_open:
        # A. Domain Boundary (Top Wall / Symmetry)
        if k >= Nz - 2: 
             up_open = False
        
        # B. Internal Obstacle
        elif mask_vel is not None and mask_vel[i, k+1] == 0:
            up_open = False

    # --- Check Down Face (k -> k-1) ---
    down_open = (face_z[i, k] == 1)
    if down_open:
        # A. Domain Boundary (The "k=1" bottom wall case)
        # If k=1, neighbor is k=0 (Ghost Wall). Ignore it.
        if k <= 1:
            down_open = False
        
        # B. Internal Obstacle
        elif mask_vel is not None and mask_vel[i, k-1] == 0:
            down_open = False

    # --- Standard Stencil Selection ---
    if up_open and down_open:
        return (p[i,k+1] - p[i,k-1]) / (2*dz)
    elif up_open and not down_open:
        return (p[i,k+1] - p[i,k]) / dz # Forward Difference (Triggers at k=1)
    elif down_open and not up_open:
        return (p[i,k] - p[i,k-1]) / dz
    else:
        return 0.0

def build_face_masks(fluid):
    """
    Given cell-centered fluid mask (Nr,Nz) with 1=fluid,0=solid,
    build face masks for r- and z-faces:
      face_r[i,k] is the face between cells (i-1,k) and (i,k) for i=1..Nr-1
      face_z[i,k] is the face between cells (i,k-1) and (i,k) for k=1..Nz-1
    A face is open only if BOTH adjacent cells are fluid.
    """
    Nr, Nz = fluid.shape
    face_r = np.zeros((Nr+1, Nz), dtype=np.uint8)
    face_z = np.zeros((Nr, Nz+1), dtype=np.uint8)

    # interior faces
    face_r[1:Nr, :] = (fluid[0:Nr-1, :] & fluid[1:Nr, :]).astype(np.uint8)
    face_z[:, 1:Nz] = (fluid[:, 0:Nz-1] & fluid[:, 1:Nz]).astype(np.uint8)

    # physical boundaries: keep as-is (0/1) based on your BCs; default closed
    return face_r, face_z

@njit(parallel=True)
def apply_solid_mask_inplace(fluid, rho, ur, ut, uz):
    Nr, Nz = fluid.shape
    for i in prange(Nr):
        for k in range(Nz):
            if fluid[i,k] == 0:
                # no fluid in solid: no-slip, keep density frozen (or set to a fixed rho_solid)
                ur[i,k] = 0.0
                ut[i,k] = 0.0
                uz[i,k] = 0.0

@njit(parallel=True)
def div_stress_tensor_masked(r, dr, dz,
                             tau_rr, tau_tt, tau_zz, tau_rz, tau_rt, tau_tz,
                             div_tau_r, div_tau_t, div_tau_z,
                             fluid=None, face_r=None, face_z=None):
    Nr, Nz = tau_rr.shape
    for i in prange(1, Nr - 1):
        ri = r[i]
        if ri < 1e-12:
            continue
        rip = 0.5 * (r[i] + r[i+1])
        rim = 0.5 * (r[i] + r[i-1])

        for k in range(1, Nz - 1):
            if fluid is not None and fluid[i, k] == 0:
                div_tau_r[i,k] = div_tau_t[i,k] = div_tau_z[i,k] = 0.0
                continue

            # Helper: masks for faces around (i,k)
            mr_ip = 1 if face_r is None else face_r[i+1, k]  # face (i+1/2,k)
            mr_im = 1 if face_r is None else face_r[i,   k]  # face (i-1/2,k)
            mz_kp = 1 if face_z is None else face_z[i, k+1]  # face (i,k+1/2)
            mz_km = 1 if face_z is None else face_z[i, k]    # face (i,k-1/2)

            # --- (∇·τ)_r ---
            flux_rr_ip = rip * 0.5 * (tau_rr[i+1,k] + tau_rr[i,k]) * mr_ip
            flux_rr_im = rim * 0.5 * (tau_rr[i,  k] + tau_rr[i-1,k]) * mr_im
            term_r_dr  = (flux_rr_ip - flux_rr_im) / (dr * ri)

            flux_rz_kp = 0.5 * (tau_rz[i,k+1] + tau_rz[i,k]) * mz_kp
            flux_rz_km = 0.5 * (tau_rz[i,k  ] + tau_rz[i,k-1]) * mz_km
            term_r_dz  = (flux_rz_kp - flux_rz_km) / dz

            div_tau_r[i,k] = term_r_dr + term_r_dz - tau_tt[i,k] / ri

            # --- (∇·τ)_θ ---
            flux_rt_ip = rip**2 * 0.5 * (tau_rt[i+1,k] + tau_rt[i,k]) * mr_ip
            flux_rt_im = rim**2 * 0.5 * (tau_rt[i,  k] + tau_rt[i-1,k]) * mr_im
            term_t_dr  = (flux_rt_ip - flux_rt_im) / (dr * ri**2)

            flux_tz_kp = 0.5 * (tau_tz[i,k+1] + tau_tz[i,k]) * mz_kp
            flux_tz_km = 0.5 * (tau_tz[i,k  ] + tau_tz[i,k-1]) * mz_km
            term_t_dz  = (flux_tz_kp - flux_tz_km) / dz

            div_tau_t[i,k] = term_t_dr + term_t_dz

            # --- (∇·τ)_z ---
            flux_rz_ip = rip * 0.5 * (tau_rz[i+1,k] + tau_rz[i,k]) * mr_ip
            flux_rz_im = rim * 0.5 * (tau_rz[i,  k] + tau_rz[i-1,k]) * mr_im
            term_z_dr  = (flux_rz_ip - flux_rz_im) / (dr * ri)

            flux_zz_kp = 0.5 * (tau_zz[i,k+1] + tau_zz[i,k]) * mz_kp
            flux_zz_km = 0.5 * (tau_zz[i,k  ] + tau_zz[i,k-1]) * mz_km
            term_z_dz  = (flux_zz_kp - flux_zz_km) / dz

            div_tau_z[i,k] = term_z_dr + term_z_dz

@njit(parallel=True)
def grad_r_masked(f, dr, out, face_r):
    Nr, Nz = f.shape
    out[:] = 0.0
    for i in prange(1, Nr-1):
        for k in range(Nz):
            right_open = face_r[i+1,k] == 1
            left_open  = face_r[i,  k] == 1
            if right_open and left_open:
                out[i,k] = (f[i+1,k] - f[i-1,k]) / (2*dr)
            elif right_open and not left_open:
                out[i,k] = (f[i+1,k] - f[i,k]) / dr
            elif left_open and not right_open:
                out[i,k] = (f[i,k] - f[i-1,k]) / dr
            else:
                out[i,k] = 0.0

@njit(parallel=True)
def grad_z_masked(f, dz, out, face_z):
    Nr, Nz = f.shape
    out[:] = 0.0
    for i in prange(Nr):
        for k in range(1, Nz-1):
            up_open   = face_z[i, k+1] == 1
            down_open = face_z[i, k  ] == 1
            if up_open and down_open:
                out[i,k] = (f[i,k+1] - f[i,k-1]) / (2*dz)
            elif up_open and not down_open:
                out[i,k] = (f[i,k+1] - f[i,k]) / dz
            elif down_open and not up_open:
                out[i,k] = (f[i,k] - f[i,k-1]) / dz
            else:
                out[i,k] = 0.0

def stable_dt(fluid, r, dr, dz,
              ur, uz, c_field,
              mu, rho,
              kappa=None, c_v=None,
              safety=0.5):
    # advection/acoustics
    a_r = np.max(np.abs(ur) + c_field)
    a_z = np.max(np.abs(uz) + c_field)
    dt_adv = np.inf
    if a_r > 0: dt_adv = dr / a_r
    if a_z > 0: dt_adv = min(dt_adv, dz / a_z)

    # explicit viscous diffusion (1/4 factor in 2D)
    nu_max = np.max(mu[fluid==1] / np.maximum(rho[fluid==1], 1e-10))
    dt_visc = np.inf if nu_max == 0 else 0.25 * min(dr*dr, dz*dz) / nu_max

    # thermal diffusion
    dt_cond = np.inf
    if (kappa is not None) and (c_v is not None):
        alpha = kappa / np.maximum(rho*c_v, 1e-30)
        alpha_max = np.max(alpha[fluid==1])
        if alpha_max > 0:
            dt_cond = 0.25 * min(dr*dr, dz*dz) / alpha_max

    dt = safety * min(dt_adv, dt_visc, dt_cond)
    return dt, dt_adv, dt_visc, dt_cond

def stable_adv_dt(fluid, r, dr, dz,
                ur, uz, c_field,
                safety=0.5):
    # advection/acoustics        
    a_r = np.max((np.abs(ur) + c_field)[fluid==1])
    a_z = np.max((np.abs(uz) + c_field)[fluid==1])
    dt_adv = dr / a_r
    dt_adv = min(dt_adv, dz / a_z)
    dt_adv*=safety
    return dt_adv

# ---------- Rusanov (LLF) flux for scalar advection in RZ ----------
@njit(parallel=True, fastmath=True, cache=True)
def rusanov_div_scalar_masked(q, ur, uz, r, dr, dz, a_r, a_z, out,
                              face_r=None, face_z=None, fluid=None):
    """
    Rusanov ∇·(q u) with optional solid masks.
    Updated to compute divergence on the FULL domain [0, Nr-1] and [0, Nz-1]
    by assuming zero flux at the physical boundaries (r=0, r=R, z=0, z=L).
    """
    Nr, Nz = q.shape
    Fr = np.zeros((Nr+1, Nz))
    Fz = np.zeros((Nr,   Nz+1))

    # radial faces (1..Nr-1)
    # Fr[i] is face between cell i-1 and i
    # Fr[0] and Fr[Nr] remain 0.0 (impermeable domain walls/axis)
    for i in prange(1, Nr):
        for k in range(0, Nz): # Full Z range to support boundary updates
            qL = q[i-1,k]; qR = q[i,k]
            uL = ur[i-1,k]; uR = ur[i,k]
            a  = max(a_r[i-1,k], a_r[i,k])
            fL = qL * uL
            fR = qR * uR
            flux = 0.5*(fL + fR) - 0.5*a*(qR - qL)
            if face_r is not None:
                flux *= face_r[i,k]    # close face if 0
            Fr[i,k] = flux

    # axial faces (1..Nz-1)
    # Fz[k] is face between cell k-1 and k
    # Fz[0] and Fz[Nz] remain 0.0 (impermeable domain walls)
    for i in prange(0, Nr): # Full R range
        for k in range(1, Nz):
            qL = q[i,k-1]; qR = q[i,k]
            uL = uz[i,k-1]; uR = uz[i,k]
            a  = max(a_z[i,k-1], a_z[i,k])
            fL = qL * uL
            fR = qR * uR
            flux = 0.5*(fL + fR) - 0.5*a*(qR - qL)
            if face_z is not None:
                flux *= face_z[i,k]
            Fz[i,k] = flux

    # Axisymmetric divergence on FULL domain (0..Nr-1, 0..Nz-1)
    for i in prange(0, Nr):
        # Prepare geometric factors
        if i == 0:
            # Axis case: Limit r->0 of (1/r) d(rF)/dr
            # Using FV limit for wedge volume: Div = 4 * Fr[1] / dr
            # (Flux_left is 0 at axis)
            rip = 0.0 # Not used for axis special logic below, but strictly r[0]+dr/2
            rim = 0.0
            inv_vol = 0.0 # Handled logic below
        elif i == Nr-1:
            # Wall case: Flux_right (Fr[Nr]) is 0.
            # Volume centered at r[Nr-1].
            #rip = r[i] + 0.5*dr # Right face radius
            #rim = r[i] - 0.5*dr # Left face radius
            #inv_vol = 1.0 / (r[i] * dr)
            rip = r[i] # Right face radius
            rim = r[i] - 0.5*dr # Left face radius
            inv_vol = 1.0 / (r[i] * 0.5 * dr)
        else:
            # Standard interior
            rip = 0.5*(r[i] + r[i+1])
            rim = 0.5*(r[i] + r[i-1])
            inv_vol = 1.0 / (r[i] * dr)

        for k in range(0, Nz):
            # --- Radial contribution ---
            if i == 0:
                # Singularity handling: 4 * Flux_Right / dr
                dFr = 4.0 * Fr[1, k] / dr
            else:
                # Standard FV divergence: (r+ F+ - r- F-) / (r dr)
                # Note: Fr[i+1] is right face, Fr[i] is left face
                dFr = (rip * Fr[i+1, k] - rim * Fr[i, k]) * inv_vol

            # --- Axial contribution ---
            # Fz[k+1] is Top face, Fz[k] is Bottom face
            dFz = (Fz[i, k+1] - Fz[i, k]) / dz

            val = dFr + dFz
            if fluid is not None and fluid[i,k] == 0:
                val = 0.0
            out[i,k] = val

# ---------- Stresses (axisymmetric) ----------
@njit(parallel=True, fastmath=True, cache=True)
def stresses(r, ur, ut, uz, mu, mub, dr, dz,
             tau_rr, tau_tt, tau_zz, tau_rz, tau_rt, tau_tz, divu,
             fluid=None, face_r=None, face_z=None):
    Nr, Nz = ur.shape

    dur_dr = np.zeros_like(ur); grad_r_masked(ur, dr, dur_dr, face_r)
    duz_dz = np.zeros_like(uz); grad_z_masked(uz, dz, duz_dz, face_z)
    dut_dr = np.zeros_like(ut); grad_r_masked(ut, dr, dut_dr, face_r)
    dut_dz = np.zeros_like(ut); grad_z_masked(ut, dz, dut_dz, face_z)

    # cross derivative sum: ∂r uz + ∂z ur
    uz_r = np.zeros_like(ur); grad_r_masked(uz, dr, uz_r, face_r)
    ur_z = np.zeros_like(ur); grad_z_masked(ur, dz, ur_z, face_z)

    du_rz = uz_r + ur_z

    for i in prange(1, Nr-1):
        # safer "axis" radius: use half-cell radius at i=0 if needed
        ri = r[i] if r[i] > 0 else (0.5*r[1] if len(r) > 1 else 1e-14)
        inv_ri = 1.0/ri
        for k in range(1, Nz-1):
            # div u = (1/r) ∂r(r ur) + ∂z uz
            divu[i,k] = ((r[i+1]*ur[i+1,k]-r[i-1]*ur[i-1,k])/(2*dr*ri)) + duz_dz[i,k]
            lam = mub[i,k] - 2.0*mu[i,k]/3.0
            tau_rr[i,k] = 2*mu[i,k]*dur_dr[i,k] + lam*divu[i,k]
            tau_tt[i,k] = 2*mu[i,k]*(ur[i,k]*inv_ri) + lam*divu[i,k]
            tau_zz[i,k] = 2*mu[i,k]*duz_dz[i,k] + lam*divu[i,k]
            tau_rz[i,k] = mu[i,k]*du_rz[i,k]
            tau_rt[i,k] = mu[i,k]*(dut_dr[i,k] - ut[i,k]*inv_ri)
            tau_tz[i,k] = mu[i,k]*dut_dz[i,k]

    if fluid is not None:
        Nr, Nz = ur.shape
        for i in prange(Nr):
            for k in range(Nz):
                if fluid[i,k] == 0:
                    tau_rr[i,k] = 0.0
                    tau_tt[i,k] = 0.0
                    tau_zz[i,k] = 0.0
                    tau_rz[i,k] = 0.0
                    tau_rt[i,k] = 0.0
                    tau_tz[i,k] = 0.0
                    divu[i,k]   = 0.0

# ---------- Momentum RHS (viscous + curvature + drag) ----------
@njit(parallel=True)
def mom_rhs(r, rho, ur, ut, uz, p,
            tau_rr, tau_tt, tau_zz, tau_rz, tau_rt, tau_tz,
            dr, dz,
            rhs_r, rhs_t, rhs_z,
            fluid=None, face_r=None, face_z=None):
    """
    Compute the right-hand-side of the momentum equation:
      RHS = -∇p + ∇·τ + F_curvature + F_drag
    with optional solid masks.
    """
    Nr, Nz = rho.shape

    # --- Correctly calculate divergence of the full stress tensor ---
    div_tau_r = np.zeros_like(rho)
    div_tau_t = np.zeros_like(rho)
    div_tau_z = np.zeros_like(rho)

    div_stress_tensor_masked(r, dr, dz,
            tau_rr, tau_tt, tau_zz, tau_rz, tau_rt, tau_tz,
            div_tau_r, div_tau_t, div_tau_z,
            fluid=fluid, face_r=face_r, face_z=face_z)

    # --- Assemble the final right-hand-side for the momentum equation ---
    for i in prange(1, Nr - 1):
        ri = r[i] if r[i] > 1e-12 else 1e-12
        inv_ri = 1.0 / ri

        for k in range(1, Nz - 1):
            if fluid is not None and fluid[i, k] == 0:
                rhs_r[i, k] = 0.0
                rhs_t[i, k] = 0.0
                rhs_z[i, k] = 0.0
                continue

            # Pressure gradients
            dp_dr = dpdr_masked(p, dr, i, k, face_r, fluid)
            dp_dz = dpdz_masked(p, dz, i, k, face_z, fluid)
                
            # Centrifugal and Coriolis forces
            curv_r = +rho[i, k] * ut[i, k]**2 * inv_ri
            curv_t = -rho[i, k] * ur[i, k] * ut[i, k] * inv_ri

            # Assemble RHS: -∇p + ∇·τ + F_curvature + F_drag
            rhs_r[i, k] = -dp_dr + div_tau_r[i, k] + curv_r
            rhs_t[i, k] =          div_tau_t[i, k] + curv_t
            rhs_z[i, k] = -dp_dz + div_tau_z[i, k]


@njit(parallel=True, fastmath=True, cache=True)
def energy_rhs_masked(r, ur, ut, uz, p, T, kappa,
                      tau_rr, tau_tt, tau_zz, tau_rz, tau_rt, tau_tz,
                      dr, dz, out, divu_out,
                      fluid=None, face_r=None, face_z=None):
    Nr, Nz = T.shape

    # ---------------------------
    # 1) Face-centered conduction
    #    G_r = k * dT/dr on r-faces (Nr+1,Nz)
    #    G_z = k * dT/dz on z-faces (Nr,Nz+1)
    # ---------------------------
    G_r = np.zeros((Nr+1, Nz))
    G_z = np.zeros((Nr,   Nz+1))

    # radial faces i = 1..Nr-1, k = 1..Nz-2 valid
    for i in prange(1, Nr):
        for k in range(1, Nz-1):
            # Check if this face allows conduction.
            # Conduct if face is open in face_r OR if it is a solid-fluid boundary
            conduct = False
            if face_r is None:
                conduct = True
            elif face_r[i, k] == 1:
                conduct = True
            elif fluid is not None:
                # Allow flux if at least one neighbor is fluid (Solid-Fluid interface)
                # Face i is between cell i-1 and cell i
                if (fluid[i, k] == 1) or (fluid[i-1, k] == 1):
                    conduct = True

            if conduct:
                # arithmetic average of kappa to the face
                kf = 0.5 * (kappa[i, k] + kappa[i-1, k])
                G_r[i, k] = kf * (T[i, k] - T[i-1, k]) / dr
            else:
                G_r[i, k] = 0.0

    # axial faces i = 1..Nr-2, k = 1..Nz-1 valid
    for i in prange(1, Nr-1):
        for k in range(1, Nz):
            conduct = False
            if face_z is None:
                conduct = True
            elif face_z[i, k] == 1:
                conduct = True
            elif fluid is not None:
                # Allow flux if at least one neighbor is fluid
                # Face k is between cell k-1 and cell k
                if (fluid[i, k] == 1) or (fluid[i, k-1] == 1):
                    conduct = True

            if conduct:
                kf = 0.5 * (kappa[i, k] + kappa[i, k-1])
                G_z[i, k] = kf * (T[i, k] - T[i, k-1]) / dz
            else:
                G_z[i, k] = 0.0

    # divergence of k∇T at cell centers (axisymmetric)
    div_kgradT = np.zeros_like(T)
    for i in prange(1, Nr-1):
        rip = 0.5 * (r[i] + r[i+1])  # r at i+1/2
        rim = 0.5 * (r[i] + r[i-1])  # r at i-1/2
        inv_ridr = 1.0 / (r[i] * dr)
        for k in range(1, Nz-1):
            dFr = (rip * G_r[i+1, k] - rim * G_r[i, k]) * inv_ridr
            dFz = (G_z[i, k+1] - G_z[i, k]) / dz
            div_kgradT[i, k] = dFr + dFz

    # ---------------------------
    # 2) div(u): masked everywhere
    #    div u = (1/r) ∂r (r ur) + ∂z uz
    # ---------------------------
    wur = np.zeros_like(T)          # r * ur
    for i in prange(Nr):
        ri = r[i]
        for k in range(Nz):
            wur[i, k] = ri * ur[i, k]

    d_wur_dr = np.zeros_like(T)
    duz_dz   = np.zeros_like(T)
    grad_r_masked(wur, dr, d_wur_dr, face_r)
    grad_z_masked(uz,  dz, duz_dz,   face_z)

    divu = np.zeros_like(T)
    for i in prange(1, Nr-1):
        ri = r[i] if r[i] > 0.0 else (0.5*r[1] if Nr > 1 else 1e-14)
        inv_ri = 1.0 / ri
        for k in range(1, Nz-1):
            divu[i, k] = inv_ri * d_wur_dr[i, k] + duz_dz[i, k]

    # ---------------------------
    # 3) Viscous dissipation Φ = τ:∇u (masked derivatives)
    # ---------------------------
    dur_dr = np.zeros_like(ur); grad_r_masked(ur, dr, dur_dr, face_r)
    dut_dr = np.zeros_like(ut); grad_r_masked(ut, dr, dut_dr, face_r)
    dut_dz = np.zeros_like(ut); grad_z_masked(ut, dz, dut_dz, face_z)
    uz_r   = np.zeros_like(ur); grad_r_masked(uz, dr, uz_r, face_r)
    ur_z   = np.zeros_like(ur); grad_z_masked(ur, dz, ur_z, face_z)

    Phi = np.zeros_like(T)
    for i in prange(1, Nr-1):
        ri = r[i] if r[i] > 0.0 else (0.5*r[1] if Nr > 1 else 1e-14)
        inv_ri = 1.0 / ri
        for k in range(1, Nz-1):
            Phi[i, k] = (
                tau_rr[i, k] * dur_dr[i, k] +
                tau_tt[i, k] * (ur[i, k] * inv_ri) +
                tau_zz[i, k] * duz_dz[i, k] +
                tau_rz[i, k] * (uz_r[i, k] + ur_z[i, k]) +
                tau_rt[i, k] * (dut_dr[i, k] - ut[i, k] * inv_ri) +
                tau_tz[i, k] * dut_dz[i, k]
            )

    # ---------------------------
    # 4) Assemble RHS and mask solids
    # ---------------------------
    for i in prange(1, Nr-1):
        for k in range(1, Nz-1):
            S = -p[i, k]*divu[i, k] + div_kgradT[i, k] + Phi[i, k]
            if (fluid is not None) and (fluid[i, k] == 0):
                S = 0.0
                divu[i, k] = 0.0
            out[i, k]      = S
            divu_out[i, k] = divu[i, k]

@njit(parallel=True, fastmath=True, cache=True)
def step_isothermal(r, dr, dz, dt,
                    rho, ur, ut, uz, p,
                    mu, mub,
                    c_iso, rho_floor=1e-12,
                    fluid=None, face_r=None, face_z=None):
    """
    Isothermal momentum+continuity step with optional solid masks.

    Masks:
      fluid  : (Nr,Nz)  {0,1} 1=fluid, 0=solid
      face_r : (Nr+1,Nz){0,1} open/closed radial faces
      face_z : (Nr,Nz+1){0,1} open/closed axial  faces
    If masks are None, falls back to fully open domain.
    """
    Nr, Nz = rho.shape

    # ---- small artificial bulk viscosity
    h = dr if dr < dz else dz
    mub_eff = mub + 0.02 * rho * c_iso * h

    # ---- stresses (same as before)
    tau_rr = np.zeros_like(rho); tau_tt = np.zeros_like(rho); tau_zz = np.zeros_like(rho)
    tau_rz = np.zeros_like(rho); tau_rt = np.zeros_like(rho); tau_tz = np.zeros_like(rho)
    divu   = np.zeros_like(rho)
    stresses(r, ur, ut, uz, mu, mub_eff, dr, dz,
         tau_rr, tau_tt, tau_zz, tau_rz, tau_rt, tau_tz, divu,
         fluid=fluid, face_r=face_r, face_z=face_z)

    # ---- viscous + pressure + curvature + drag RHS
    rhs_r = np.zeros_like(rho); rhs_t = np.zeros_like(rho); rhs_z = np.zeros_like(rho)
    mom_rhs(r, rho, ur, ut, uz, p,
        tau_rr, tau_tt, tau_zz, tau_rz, tau_rt, tau_tz,
        dr, dz, rhs_r, rhs_t, rhs_z,
        fluid=fluid, face_r=face_r, face_z=face_z)

    # Zero RHS inside solid cells so they don't accumulate sources
    if fluid is not None:
        for i in prange(1, Nr-1):
            for k in range(1, Nz-1):
                if fluid[i,k] == 0:
                    rhs_r[i,k] = 0.0
                    rhs_t[i,k] = 0.0
                    rhs_z[i,k] = 0.0

    # ---- add momentum convection via masked Rusanov
    a_r = np.abs(ur) + c_iso
    a_z = np.abs(uz) + c_iso

    div_m_r = np.zeros_like(rho)
    div_m_t = np.zeros_like(rho)
    div_m_z = np.zeros_like(rho)

    # rusanov div scalar masked
    rusanov_div_scalar_masked(rho*ur, ur, uz, r, dr, dz, a_r, a_z, div_m_r,
                              face_r=face_r, face_z=face_z, fluid=fluid)
    rusanov_div_scalar_masked(rho*ut, ur, uz, r, dr, dz, a_r, a_z, div_m_t,
                              face_r=face_r, face_z=face_z, fluid=fluid)
    rusanov_div_scalar_masked(rho*uz, ur, uz, r, dr, dz, a_r, a_z, div_m_z,
                              face_r=face_r, face_z=face_z, fluid=fluid)

    rhs_r[1:-1,1:-1] -= div_m_r[1:-1,1:-1]
    rhs_t[1:-1,1:-1] -= div_m_t[1:-1,1:-1]
    rhs_z[1:-1,1:-1] -= div_m_z[1:-1,1:-1]

    # ---- explicit update of u in fluid cells only
    rho_safe = np.maximum(rho, rho_floor)
    for i in prange(1, Nr-1):
        for k in range(1, Nz-1):
            if (fluid is None) or (fluid[i,k] == 1):
                ur[i,k] += dt * rhs_r[i,k] / rho_safe[i,k]
                ut[i,k] += dt * rhs_t[i,k] / rho_safe[i,k]
                uz[i,k] += dt * rhs_z[i,k] / rho_safe[i,k]

    # ---- continuity with masked Rusanov (only evolve in fluid)
    divF = np.zeros_like(rho)
    rusanov_div_scalar_masked(rho, ur, uz, r, dr, dz, a_r, a_z, divF,
                              face_r=face_r, face_z=face_z, fluid=fluid)

    for i in prange(0, Nr):
        for k in range(0, Nz):
            if (fluid is None) or (fluid[i,k] == 1):
                rho[i,k] -= dt * divF[i,k]
            # else: keep rho as-is in solid

    # ---- positivity clamp in fluid cells
    for i in prange(0, Nr):
        for k in range(0, Nz):
            if ((fluid is None) or (fluid[i,k] == 1)) and (rho[i,k] < rho_floor):
                rho[i,k] = rho_floor


@njit(parallel=True, fastmath=True, cache=True)
def step_temperature_masked(r, dr, dz, dt,
                            T, rho, ur, ut, uz, p, kappa,
                            tau_rr, tau_tt, tau_zz, tau_rz, tau_rt, tau_tz,
                            c_v,
                            fluid=None, face_r=None, face_z=None,
                            T_floor=300.0):
    """
    ∂t T = -∇·(T u) + T ∇·u + [ -p ∇·u + ∇·(k∇T) + Φ ] / (ρ c_v)
    with masked advection and zero sources inside solids.
    """
    Nr, Nz = T.shape

    # conservative advection of T with closed faces
    a_r = np.abs(ur)
    a_z = np.abs(uz)
    div_Tu = np.zeros_like(T)
    rusanov_div_scalar_masked(T, ur, uz, r, dr, dz, a_r, a_z, div_Tu,
                              face_r=face_r, face_z=face_z, fluid=fluid)

    # source term and div(u)
    S = np.zeros_like(T)
    divu = np.zeros_like(T)
    energy_rhs_masked(r, ur, ut, uz, p=p, T=T, kappa=kappa,
                      tau_rr=tau_rr, tau_tt=tau_tt, tau_zz=tau_zz,
                      tau_rz=tau_rz, tau_rt=tau_rt, tau_tz=tau_tz,
                      dr=dr, dz=dz, out=S, divu_out=divu,
                      fluid=fluid, face_r=face_r, face_z=face_z)

    rho_safe = np.maximum(rho, 1e-30)
    inv_rho_cv = 1.0/(rho_safe*c_v)

    # update interior
    for i in prange(1, Nr-1):
        for k in range(1, Nz-1):
            if fluid is not None and fluid[i,k] == 0:
                continue  # do not evolve inside solid
            adv = -div_Tu[i,k] + T[i,k]*divu[i,k]
            src = S[i,k] * inv_rho_cv[i,k]
            T[i,k] += dt * (adv + src)

    # clamp and impose wall temperature (if provided)
    for i in prange(1, Nr-1):
        for k in range(1, Nz-1):
            if T[i,k] < T_floor:
                T[i,k] = T_floor
            #if fluid is not None and fluid[i,k] == 0 and T_wall == T_wall:
            #    T[i,k] = T_wall

###########################################################################################################
############################ Parameters for viscosity and conductivity calculation ########################
###########################################################################################################

def _omega_22(T_star):
    """
    Reduced collision integral Ω^(2,2)(T*) for LJ(12-6), Neufeld-Janzen-Aziz correlation.
    T_star = k_B T / eps  (dimensionless). Valid over a wide range (T* ~ 0.3-100).
    """
    # Clip to positive to be safe numerically
    Ts = np.clip(T_star, 1e-6, None)
    return (1.16145 * Ts**(-0.14874)
            + 0.52487 * np.exp(-0.77320 * Ts)
            + 2.16178 * np.exp(-2.43787 * Ts))

def _mu_LJ(T, m, sigma, eps_over_k):
    """
    Dynamic viscosity via Chapman-Enskog at first approximation for an LJ gas.
    μ = (5/16) * sqrt(m k_B T / π) / (σ^2 Ω^(2,2)(T*))
    Returns μ [Pa·s]; T can be ndarray; broadcasting is supported.
    """
    eps = eps_over_k * constants.kb
    T_star = (constants.kb * T) / eps
    Omega22 = _omega_22(T_star)
    pref = (5.0 / 16.0) * np.sqrt(m * constants.kb / np.pi)   # everything but sqrt(T)
    return pref * np.sqrt(T) / (sigma**2 * Omega22)

def _cp_Rspec(kind, f=None):
    """
    c_p in units of R_specific = k_B/m, based on degrees of freedom.
    monotonic 'kind' sets the default: monatomic (f=3), diatomic (f=5).
    You can override by providing f explicitly (e.g., to include vibrations).
    """
    if f is None:
        if kind == "monatomic":
            f = 3
        elif kind == "diatomic":
            f = 5
        else:
            raise ValueError("kind must be 'monatomic' or 'diatomic' unless you provide f.")
    return (f/2.0 + 1.0)  # c_p / R_spec

def viscosity_and_conductivity(
    geom,
    T,
    mass,
    *,
    species=None,
    kind=None,
    lj_params=None,         # tuple (sigma [m], eps_over_k [K]) if you don't want to use 'species'
    model="LJ",             # "LJ" or "Sutherland"
    sutherland=None,        # dict with keys {'mu_ref','T_ref','S'} if model="Sutherland"
    f_dof=None              # integer degrees of freedom to override kind (e.g., include vibrations)
):
    """
    Compute dynamic viscosity μ [Pa·s] and thermal conductivity k [W/m/K]
    for a *neutral* ideal gas on the same grid as T (shape (Nr, Nz)).

    Parameters
    ----------
    T : ndarray
        Temperature field [K], shape (Nr, Nz) or broadcastable array.
    mass : float
        Mass per gas particle [kg] (e.g., for Ar: 39.948 u -> 39.948*1.66054e-27 kg).
    species : str, optional
        One of {'H2','He','N2','O2','Ar'} to pull common Lennard-Jones params and default 'kind'.
    kind : {'monatomic','diatomic'}, optional
        Needed if you don't use 'species'. Controls how k is computed (Eucken vs monatomic).
    lj_params : tuple, optional
        (sigma [m], eps_over_k [K]) for the LJ model; overrides 'species'.
    model : {'LJ','Sutherland'}, default 'LJ'
        Transport model for viscosity μ.
    sutherland : dict, optional
        If model == 'Sutherland': provide {'mu_ref':..., 'T_ref':..., 'S':...}.
    f_dof : int, optional
        Degrees of freedom (trans + rot + possibly vib). Overrides 'kind' in cp.

    Returns
    -------
    mu : ndarray
        Dynamic viscosity [Pa·s], same shape as T.
    k  : ndarray
        Thermal conductivity [W/m/K], same shape as T.
    meta : dict
        Metadata echoing model choices actually used.
    """
    T = np.asarray(T) + 1
    if np.any(T <= 0):
        raise ValueError("Temperature contains non-positive values.")
    # Decide transport model for μ
    used = {"model": model}

    if model == "LJ":
        if lj_params is not None:
            sigma, eps_over_k = lj_params
            used["source"] = "user_lj_params"
        else:
            if species is None and kind is None:
                raise ValueError("For LJ model, provide either 'species' or both 'kind' and 'lj_params'.")
            if species is not None:
                if species not in _LJ_DB:
                    raise ValueError(f"Unknown species '{species}'. Available: {sorted(_LJ_DB.keys())}")
                sigma, eps_over_k, default_kind = _LJ_DB[species]
                used["species"] = species
                if kind is None:
                    kind = default_kind
                used["kind"] = kind
            else:
                if kind is None:
                    raise ValueError("Provide 'kind' when using custom 'lj_params'.")
                used["kind"] = kind

        mu = _mu_LJ(T, mass, sigma, eps_over_k)
        used["sigma_m"] = sigma
        used["eps_over_k_K"] = eps_over_k

    elif model == "Sutherland":
        if sutherland is None:
            raise ValueError("For Sutherland model, provide sutherland={'mu_ref','T_ref','S'}.")
        mu_ref = sutherland["mu_ref"]
        T_ref = sutherland["T_ref"]
        S = sutherland["S"]
        # μ(T) = μ_ref * (T/T_ref)^(3/2) * (T_ref + S)/(T + S)
        mu = mu_ref * (T / T_ref)**1.5 * (T_ref + S) / (T + S)
        used.update({"suth_mu_ref": mu_ref, "suth_T_ref": T_ref, "suth_S": S})
        if kind is None and species is not None and species in _LJ_DB:
            # use default kind from DB for conductivity
            used["kind"] = _LJ_DB[species][2]
            kind = used["kind"]
        elif kind is None:
            raise ValueError("For Sutherland model, please specify 'kind' (monatomic/diatomic) or a 'species' I know.")
    else:
        raise ValueError("model must be 'LJ' or 'Sutherland'.")

    # Thermal conductivity via Eucken / monatomic relation
    R_spec = constants.kb / mass
    if f_dof is not None:
        cp_over_R = _cp_Rspec(kind="monatomic", f=f_dof)  # kind ignored if f provided
        used["f_dof"] = int(f_dof)
    else:
        cp_over_R = _cp_Rspec(kind)
    cp = cp_over_R * R_spec

    if kind == "monatomic" and f_dof is None:
        # exact Chapman–Enskog relation for monatomic ideal gas
        k = (15.0 / 4.0) * (constants.kb / mass) * mu
    else:
        # Extended Eucken relation for polyatomic gases
        k = mu * (cp + 1.25 * R_spec)

    mu[geom.mask==0]*=0; k[geom.mask==0]*=0
    return mu, k, used

@njit(parallel=True, cache=True)
def update_u_in_collisions(
    mask, rho_i, rho_n,
    ui_r, ui_t, ui_z,
    un_r, un_t, un_z,
    nu, rho_floor,
    Tn, Ti, c_v, dt
):
    """
    This assumes that dt*nu<0.1
    and also performs explicit update of neutral temperature Tn
    due to collisional energy exchange and viscous heating.
    Returns updated (un_r, un_t, un_z, Tn)
    """
    NR, NZ = mask.shape

    un_r_new = np.copy(un_r)
    un_t_new = np.copy(un_t)
    un_z_new = np.copy(un_z)
    Tn_new = np.copy(Tn)

    # Loop over the interior of the grid
    for i in prange(1, NR - 1):
        for j in range(1, NZ - 1):
            if(mask[i, j] == 1 and rho_n[i, j]>rho_floor):
                factor =  dt*rho_i[i, j]/rho_n[i, j] * nu[i, j]
                un_r_new[i, j] = un_r[i, j] + factor * (ui_r[i, j] - un_r[i, j])
                un_t_new[i, j] = un_t[i, j] + factor * (ui_t[i, j] - un_t[i, j])
                un_z_new[i, j] = un_z[i, j] + factor * (ui_z[i, j] - un_z[i, j])

                du2 = (ui_r[i, j] - un_r[i, j])**2 + (ui_t[i, j] - un_t[i, j])**2 + (ui_z[i, j] - un_z[i, j])**2
                Tn_new[i, j] = Tn[i, j] + factor/c_v*du2 + factor*(Ti[i, j] - Tn[i, j])

    return un_r_new, un_t_new, un_z_new, Tn_new

@njit(parallel=True)
def compute_knudsen_field(mask, T, p, sigma, L_char, kb, out):
    """
    Computes local Knudsen number Kn = lambda / L_char
    Mean free path lambda = k_B * T / (sqrt(2) * pi * sigma^2 * p)
    """
    Nr, Nz = T.shape
    # Precompute constant factor: k_B / (sqrt(2) * pi * sigma^2)

    prefactor = kb / (1.41421356 * 3.14159265 * sigma**2)
    inv_L = 1.0 / L_char

    for i in prange(Nr):
        for k in range(Nz):
            if(mask[i, k] == 0):
                out[i, k] = 0.0
                continue
            # p can be very close to zero in centrifuge core (vacuum).
            # If p -> 0, lambda -> infinity, Kn -> infinity.
            # We clamp to a large number or 0 depending on preference. 
            # Here we just compute valid values and set 0.0 for pure vacuum.
            if p[i, k] > 1e-20:
                lam = prefactor * T[i, k] / p[i, k]
                out[i, k] = lam * inv_L
            else:
                out[i, k] = 0.0 # Vacuum

#############################################################################################
#############################################################################################
########## The kernels below are for semi-implicit Navier stokes solver #####################
#############################################################################################
#############################################################################################

"""
@njit(cache=True)
def solve_implicit_viscosity_r_sor(ur, nn, mn, mask, mu_grid,
                                   dt, dr, dz, max_iter=20, omega=1.4):
    Nr, Nz = ur.shape
    inv_dr2 = 1.0 / (dr * dr)
    inv_dz2 = 1.0 / (dz * dz)

    # 1. Create a copy for the Source Term (Inertia)
    ur_old = ur.copy()

    for k in range(max_iter):
        max_diff = 0.0
        
        # Start at i=1 because ur[0,:] is always 0 (axis)
        for i in range(1, Nr-1):
            r = i * dr
            
            # Geometric coeffs
            c_east = ((r + 0.5*dr) / r) * inv_dr2
            c_west = ((r - 0.5*dr) / r) * inv_dr2
            c_self_geo = 1.0 / (r * r) # Hoop stress term for vr

            for j in range(1, Nz):
                if mask[i, j] == 1:
                    
                    # --- Neighbors ---
                    val_E = ur[i+1, j] 
                    val_W = ur[i-1, j]

                    # Axial
                    if j == Nz - 1: # Top Symmetry/Slip
                        val_N = ur[i, j]; c_north = 0.0
                    else:
                        val_N = ur[i, j+1]; c_north = inv_dz2

                    if j == 0: # Bottom Wall
                        val_S = 0.0; c_south = inv_dz2
                    else:
                        val_S = ur[i, j-1]; c_south = inv_dz2
                        
                    # --- Matrix & RHS ---
                    mu_val = mu_grid[i, j]
                    rho = nn[i, j] * mn
                    A_time = rho / dt
                    
                    A_P = A_time + mu_val * (c_east + c_west + c_north + c_south + c_self_geo)
                    
                    # Use ur_old[i,j] for the inertia source
                    RHS = (A_time * ur_old[i, j]) + mu_val * (
                        c_east * val_E + c_west * val_W + 
                        c_north * val_N + c_south * val_S
                    )
                    
                    u_new = (1.0 - omega)*ur[i, j] + omega*(RHS / A_P)
                    
                    diff = abs(u_new - ur[i, j])
                    if diff > max_diff: max_diff = diff
                    ur[i, j] = u_new
                    
        if max_diff < 1e-5: break
"""

"""
@njit(cache=True)
def solve_implicit_viscosity_z_sor(uz, nn, mn, mask, mu_grid,
                                   dt, dr, dz, max_iter=20, omega=1.4):
    #Implicit Viscosity for AXIAL velocity v_z.
    Nr, Nz = uz.shape
    inv_dr2 = 1.0 / (dr * dr)
    inv_dz2 = 1.0 / (dz * dz)

    # 1. Create a copy for the Source Term (Inertia)
    uz_old = uz.copy()

    for k in range(max_iter):
        
        # Enforce Axis Symmetry: uz[0] = uz[1]
        for j_sym in range(Nz):
            if mask[0, j_sym] == 1:
                uz[0, j_sym] = uz[1, j_sym]

        max_diff = 0.0
        
        for i in range(1, Nr-1):
            r = i * dr
            c_east = ((r + 0.5*dr) / r) * inv_dr2
            c_west = ((r - 0.5*dr) / r) * inv_dr2
            
            for j in range(1, Nz-1):
                if mask[i, j] == 1:
                    
                    # --- Neighbors ---
                    val_E = uz[i+1, j]; val_W = uz[i-1, j] 
                    val_N = uz[i, j+1]; val_S = uz[i, j-1]
                    
                    c_north = inv_dz2
                    c_south = inv_dz2
                        
                    # --- Matrix & RHS ---
                    mu_val = mu_grid[i, j]
                    rho = max(nn[i, j], 1e12) * mn
                    A_time = rho / dt
                    
                    A_P = A_time + mu_val * (c_east + c_west + c_north + c_south)
                    
                    # Use uz_old[i,j] for the inertia source
                    RHS = (A_time * uz_old[i, j]) + mu_val * (
                        c_east * val_E + c_west * val_W + 
                        c_north * val_N + c_south * val_S
                    )
                    
                    u_new = (1.0 - omega)*uz[i, j] + omega*(RHS / A_P)
                    
                    diff = abs(u_new - uz[i, j])
                    if diff > max_diff: max_diff = diff
                    uz[i, j] = u_new
                    
        if max_diff < 1e-5: break
"""

@njit(cache=True)
def solve_implicit_viscosity_r_sor(ur, nn, mn, mask, mu_grid,
                                   dt, dr, dz, max_iter=20, omega=1.4):
    """
    Implicit Viscosity for RADIAL velocity v_r with HALF-CELL WALL CORRECTION.
    """
    Nr, Nz = ur.shape
    inv_dr2 = 1.0 / (dr * dr)
    inv_dz2 = 1.0 / (dz * dz)

    # 1. Create a copy for the Source Term (Inertia)
    ur_old = ur.copy()

    for k in range(max_iter):
        max_diff = 0.0
        
        # Start at i=1 (axis is 0)
        for i in range(1, Nr-1):
            r = i * dr
            c_east_base = ((r + 0.5*dr) / r) * inv_dr2
            c_west_base = ((r - 0.5*dr) / r) * inv_dr2
            c_self_geo = 1.0 / (r * r) 
            c_north_base = inv_dz2
            c_south_base = inv_dz2

            for j in range(1, Nz-1): # Careful with Z boundaries
                if mask[i, j] == 1:
                    
                    # --- CHECK NEIGHBORS ---
                    
                    # East
                    if mask[i+1, j] == 0:
                        val_E = 0.0
                        c_east = 2.0 * c_east_base
                    else:
                        val_E = ur[i+1, j]
                        c_east = c_east_base

                    # West
                    if mask[i-1, j] == 0:
                        val_W = 0.0
                        c_west = 2.0 * c_west_base
                    else:
                        val_W = ur[i-1, j]
                        c_west = c_west_base
                    
                    # North (Top is symmetry usually, but check mask first)
                    if j == Nz-1: 
                        # Top Domain Boundary (Symmetry/Wall)
                        # If symmetry, val_N=u_i, coeff=0. If wall, val=0, coeff=2x
                        # Assuming Symmetry for Top Domain boundary (kept simpler here)
                        val_N = ur[i, j]; c_north = 0.0 
                    elif mask[i, j+1] == 0:
                        val_N = 0.0; c_north = 2.0 * c_north_base
                    else:
                        val_N = ur[i, j+1]; c_north = c_north_base

                    # South
                    if j == 0: 
                         # Bottom Domain Boundary (Wall)
                         val_S = 0.0; c_south = 2.0 * c_south_base # Face is at z=0
                    elif mask[i, j-1] == 0:
                        val_S = 0.0; c_south = 2.0 * c_south_base
                    else:
                        val_S = ur[i, j-1]; c_south = c_south_base
                        
                    # --- Matrix & RHS ---
                    mu_val = mu_grid[i, j]
                    rho = nn[i, j] * mn
                    A_time = rho / dt
                    
                    A_P = A_time + mu_val * (c_east + c_west + c_north + c_south + c_self_geo)
                    
                    RHS = (A_time * ur_old[i, j]) + mu_val * (
                        c_east * val_E + c_west * val_W + 
                        c_north * val_N + c_south * val_S
                    )
                    
                    u_new = (1.0 - omega)*ur[i, j] + omega*(RHS / A_P)
                    
                    diff = abs(u_new - ur[i, j])
                    if diff > max_diff: max_diff = diff
                    ur[i, j] = u_new
                    
        if max_diff < 1e-5: break

@njit(cache=True)
def solve_implicit_viscosity_z_sor(uz, nn, mn, mask, mu_grid,
                                   dt, dr, dz, max_iter=20, omega=1.4):
    """
    Implicit Viscosity for AXIAL velocity v_z with HALF-CELL WALL CORRECTION.
    """
    Nr, Nz = uz.shape
    inv_dr2 = 1.0 / (dr * dr)
    inv_dz2 = 1.0 / (dz * dz)

    # 1. Create a copy for the Source Term (Inertia)
    uz_old = uz.copy()

    for k in range(max_iter):
        
        # Enforce Axis Symmetry: uz[0] = uz[1]
        for j_sym in range(Nz):
            if mask[0, j_sym] == 1:
                uz[0, j_sym] = uz[1, j_sym]

        max_diff = 0.0
        
        for i in range(1, Nr-1):
            r = i * dr
            # Standard geometric coefficients
            c_east_base = ((r + 0.5*dr) / r) * inv_dr2
            c_west_base = ((r - 0.5*dr) / r) * inv_dr2
            c_north_base = inv_dz2
            c_south_base = inv_dz2
            
            for j in range(1, Nz-1):
                if mask[i, j] == 1:
                    
                    # --- CHECK NEIGHBORS FOR WALLS ---
                    
                    # East
                    if mask[i+1, j] == 0:
                        val_E = 0.0
                        c_east = 2.0 * c_east_base # Wall at face -> Double Gradient
                    else:
                        val_E = uz[i+1, j]
                        c_east = c_east_base

                    # West
                    if mask[i-1, j] == 0:
                        val_W = 0.0
                        c_west = 2.0 * c_west_base
                    else:
                        val_W = uz[i-1, j]
                        c_west = c_west_base

                    # North
                    if mask[i, j+1] == 0:
                        val_N = 0.0
                        c_north = 2.0 * c_north_base
                    else:
                        val_N = uz[i, j+1]
                        c_north = c_north_base

                    # South
                    if mask[i, j-1] == 0:
                        val_S = 0.0
                        c_south = 2.0 * c_south_base
                    else:
                        val_S = uz[i, j-1]
                        c_south = c_south_base
                        
                    # --- Matrix & RHS ---
                    mu_val = mu_grid[i, j]
                    rho = max(nn[i, j], 1e12) * mn
                    A_time = rho / dt
                    
                    A_P = A_time + mu_val * (c_east + c_west + c_north + c_south)
                    
                    RHS = (A_time * uz_old[i, j]) + mu_val * (
                        c_east * val_E + c_west * val_W + 
                        c_north * val_N + c_south * val_S
                    )
                    
                    u_new = (1.0 - omega)*uz[i, j] + omega*(RHS / A_P)
                    
                    diff = abs(u_new - uz[i, j])
                    if diff > max_diff: max_diff = diff
                    uz[i, j] = u_new
                    
        if max_diff < 1e-5: break


@njit(parallel=True)
def mom_rhs_inviscid_old(r, rho, ur, ut, uz, p,
                     dr, dz,
                     rhs_r, rhs_t, rhs_z,
                     fluid=None, face_r=None, face_z=None):
    """
    RHS for momentum considering ONLY:
    1. Pressure Gradient (-grad P)
    2. Centrifugal Force (+rho u_theta^2 / r)
    3. Coriolis Force    (-rho u_r u_theta / r)
    NO Viscosity.
    """
    Nr, Nz = rho.shape

    for i in prange(1, Nr - 1):
        ri = r[i] if r[i] > 1e-12 else 1e-12
        inv_ri = 1.0 / ri

        for k in range(1, Nz - 1):
            if fluid is not None and fluid[i, k] == 0:
                rhs_r[i, k] = 0.0
                rhs_t[i, k] = 0.0
                rhs_z[i, k] = 0.0
                continue

            # Pressure gradients
            dp_dr = dpdr_masked(p, dr, i, k, face_r, fluid)
            dp_dz = dpdz_masked(p, dz, i, k, face_z, fluid)
                
            # Centrifugal and Coriolis forces
            curv_r = +rho[i, k] * ut[i, k]**2 * inv_ri
            curv_t = -rho[i, k] * ur[i, k] * ut[i, k] * inv_ri

            rhs_r[i, k] = -dp_dr + curv_r
            rhs_t[i, k] =          curv_t
            rhs_z[i, k] = -dp_dz


@njit(parallel=True)
def mom_rhs_inviscid(r, rho, ur, ut, uz, p,
                     dr, dz,
                     rhs_r, rhs_t, rhs_z,
                     fluid=None, mask_vel=None, face_r=None, face_z=None):
    """
    RHS for momentum considering ONLY Pressure, Centrifugal, Coriolis.
    """
    Nr, Nz = rho.shape

    for i in prange(1, Nr - 1):
        ri = r[i] if r[i] > 1e-12 else 1e-12
        inv_ri = 1.0 / ri

        for k in range(1, Nz - 1):
            # 2. Skip ONLY true solids (keep calculating forces on Wall Nodes)
            # Use 'fluid' (or mask_rho) here to determine if physics exists.
            if fluid is not None and fluid[i, k] == 0:
                rhs_r[i, k] = 0.0
                rhs_t[i, k] = 0.0
                rhs_z[i, k] = 0.0
                continue

            # 3. Pass mask_vel to gradients! 
            # This protects the NEIGHBOR nodes from using this node's pressure 
            # if this node happens to be a Wall Node (mask_vel=0).
            dp_dr = dpdr_masked(p, dr, i, k, face_r, mask_vel)
            dp_dz = dpdz_masked(p, dz, i, k, face_z, mask_vel)
                
            curv_r = +rho[i, k] * ut[i, k]**2 * inv_ri
            curv_t = -rho[i, k] * ur[i, k] * ut[i, k] * inv_ri

            rhs_r[i, k] = -dp_dr + curv_r
            rhs_t[i, k] =          curv_t
            rhs_z[i, k] = -dp_dz

            
@njit(parallel=True, fastmath=True, cache=True)
def step_advection_hydro_old(r, dr, dz, dt,
                         rho, ur, ut, uz, p,
                         c_iso, rho_floor=1e-12,
                         fluid=None, face_r=None, face_z=None):
    
    #Explicit RK substep for Mass + Momentum Advection + Pressure.
    #Does NOT apply viscosity.
    
    Nr, Nz = rho.shape
    
    # 1. Compute Inviscid RHS (Pressure + Curvature)
    rhs_r = np.zeros_like(rho)
    rhs_t = np.zeros_like(rho)
    rhs_z = np.zeros_like(rho)
    
    mom_rhs_inviscid(r, rho, ur, ut, uz, p, dr, dz, 
                     rhs_r, rhs_t, rhs_z, 
                     fluid, face_r, face_z)

    # 2. Add Momentum Convection (Div(rho u u)) via Rusanov
    a_r = np.abs(ur) + c_iso
    a_z = np.abs(uz) + c_iso

    div_m_r = np.zeros_like(rho); div_m_t = np.zeros_like(rho); div_m_z = np.zeros_like(rho)

    rusanov_div_scalar_masked(rho*ur, ur, uz, r, dr, dz, a_r, a_z, div_m_r, face_r, face_z, fluid)
    rusanov_div_scalar_masked(rho*ut, ur, uz, r, dr, dz, a_r, a_z, div_m_t, face_r, face_z, fluid)
    rusanov_div_scalar_masked(rho*uz, ur, uz, r, dr, dz, a_r, a_z, div_m_z, face_r, face_z, fluid)

    rhs_r[1:-1,1:-1] -= div_m_r[1:-1,1:-1]
    rhs_t[1:-1,1:-1] -= div_m_t[1:-1,1:-1]
    rhs_z[1:-1,1:-1] -= div_m_z[1:-1,1:-1]

    # 3. Update Momenta (Explicit)
    rho_safe = np.maximum(rho, rho_floor)
    for i in prange(1, Nr-1):
        for k in range(1, Nz-1):
            if (fluid is None) or (fluid[i,k] == 1):
                ur[i,k] += dt * rhs_r[i,k] / rho_safe[i,k]
                ut[i,k] += dt * rhs_t[i,k] / rho_safe[i,k]
                uz[i,k] += dt * rhs_z[i,k] / rho_safe[i,k]

    # 4. Update Density (Continuity)
    divF = np.zeros_like(rho)
    rusanov_div_scalar_masked(rho, ur, uz, r, dr, dz, a_r, a_z, divF, face_r, face_z, fluid)

    for i in prange(0, Nr):
        for k in range(0, Nz):
            if (fluid is None) or (fluid[i,k] == 1):
                rho[i,k] -= dt * divF[i,k]
                if rho[i,k] < rho_floor: rho[i,k] = rho_floor


@njit(parallel=True, fastmath=True, cache=True)
def step_advection_hydro(r, dr, dz, dt,
                         rho, ur, ut, uz, p,
                         c_iso, rho_floor=1e-12,
                         mask_rho=None, mask_vel=None,  # <--- The two masks
                         face_r=None, face_z=None):
    
    Nr, Nz = rho.shape
    rhs_r = np.zeros_like(rho); rhs_t = np.zeros_like(rho); rhs_z = np.zeros_like(rho)
    
    # 1. Calculate Forces (Physics everywhere)
    # We pass mask_rho because we need gradients everywhere
    mom_rhs_inviscid(r, rho, ur, ut, uz, p, dr, dz, 
                     rhs_r, rhs_t, rhs_z, 
                     mask_rho, mask_vel, face_r, face_z)

    # 2. Calculate Advection
    a_r = np.abs(ur) + c_iso; a_z = np.abs(uz) + c_iso
    div_m_r = np.zeros_like(rho); div_m_t = np.zeros_like(rho); div_m_z = np.zeros_like(rho)

    rusanov_div_scalar_masked(rho*ur, ur, uz, r, dr, dz, a_r, a_z, div_m_r, face_r, face_z, mask_rho)
    rusanov_div_scalar_masked(rho*ut, ur, uz, r, dr, dz, a_r, a_z, div_m_t, face_r, face_z, mask_rho)
    rusanov_div_scalar_masked(rho*uz, ur, uz, r, dr, dz, a_r, a_z, div_m_z, face_r, face_z, mask_rho)

    rhs_r[1:-1,1:-1] -= div_m_r[1:-1,1:-1]
    rhs_t[1:-1,1:-1] -= div_m_t[1:-1,1:-1]
    rhs_z[1:-1,1:-1] -= div_m_z[1:-1,1:-1]

    # 3. Update Velocity: USE mask_vel
    # This automatically skips the wall nodes!
    rho_safe = np.maximum(rho, rho_floor)
    for i in prange(1, Nr-1):
        for k in range(1, Nz-1):
            if (mask_vel is None) or (mask_vel[i,k] == 1):
                ur[i,k] += dt * rhs_r[i,k] / rho_safe[i,k]
                ut[i,k] += dt * rhs_t[i,k] / rho_safe[i,k]
                uz[i,k] += dt * rhs_z[i,k] / rho_safe[i,k]

            else:
                # Force Wall Nodes to 0.0
                ur[i,k] = 0.0
                ut[i,k] = 0.0
                uz[i,k] = 0.0

    # 4. Update Density: USE mask_rho
    # Density still evolves at the wall nodes to create stagnation pressure
    divF = np.zeros_like(rho)
    rusanov_div_scalar_masked(rho, ur, uz, r, dr, dz, a_r, a_z, divF, face_r, face_z, mask_rho)

    for i in prange(0, Nr):
        for k in range(0, Nz):
            if (mask_rho is None) or (mask_rho[i,k] == 1):
                rho[i,k] -= dt * divF[i,k]
                if rho[i,k] < rho_floor: rho[i,k] = rho_floor
                

@njit(parallel=True, fastmath=True, cache=True)
def step_advection_energy(r, dr, dz, dt,
                          T, rho, ur, ut, uz, p,
                          c_v,
                          fluid=None, face_r=None, face_z=None,
                          T_floor=300.0):
    """
    Explicit RK substep for Energy Advection + PV Work.
    Equation: rho Cv (dT/dt + u.grad T) = -p div(u)
    Does NOT apply Thermal Conduction or Viscous Heating.
    """
    Nr, Nz = T.shape

    # 1. T Advection: - div(T u) + T div(u)
    #    (Using conservative form for stability)
    a_r = np.abs(ur)
    a_z = np.abs(uz)
    div_Tu = np.zeros_like(T)
    rusanov_div_scalar_masked(T, ur, uz, r, dr, dz, a_r, a_z, div_Tu, face_r, face_z, fluid)

    # 2. Velocity Divergence
    wur = np.zeros_like(T)
    for i in prange(Nr):
        for k in range(Nz):
            wur[i, k] = r[i] * ur[i, k] # Pre-calc r*ur

    d_wur_dr = np.zeros_like(T); duz_dz = np.zeros_like(T)
    grad_r_masked(wur, dr, d_wur_dr, face_r)
    grad_z_masked(uz,  dz, duz_dz,   face_z)

    divu = np.zeros_like(T)
    for i in prange(1, Nr-1):
        ri = r[i] if r[i] > 0.0 else 1e-14
        for k in range(Nz):
             divu[i, k] = (1.0/ri) * d_wur_dr[i, k] + duz_dz[i, k]

    # 3. Update
    rho_safe = np.maximum(rho, 1e-30)
    inv_rho_cv = 1.0/(rho_safe*c_v)

    for i in prange(1, Nr-1):
        for k in range(1, Nz-1):
            if fluid is not None and fluid[i,k] == 0:
                continue
            
            # Advection term: -div(Tu) + T*div(u)
            adv = -div_Tu[i,k] + T[i,k]*divu[i,k]
            
            # PV Work term: -p div(u)
            pv_work = -p[i,k] * divu[i,k]
            
            T[i,k] += dt * (adv + pv_work * inv_rho_cv[i,k])
            
            if T[i,k] < T_floor: T[i,k] = T_floor


@njit(parallel=True, fastmath=True, cache=True)
def add_viscous_heating(T, rho, ur, ut, uz, mu, c_v, dr, dz, dt, 
                        fluid=None, face_r=None, face_z=None):
    """
    Computes viscous dissipation Phi = tau : grad(u) and adds 
    dT = dt * Phi / (rho * Cv) to the temperature field.
    """
    Nr, Nz = T.shape
    
    # We need derivatives of velocity
    dur_dr = np.zeros_like(T); grad_r_masked(ur, dr, dur_dr, face_r)
    dut_dr = np.zeros_like(T); grad_r_masked(ut, dr, dut_dr, face_r)
    duz_dz = np.zeros_like(T); grad_z_masked(uz, dz, duz_dz, face_z)
    dut_dz = np.zeros_like(T); grad_z_masked(ut, dz, dut_dz, face_z)
    
    # Cross terms
    uz_r = np.zeros_like(T); grad_r_masked(uz, dr, uz_r, face_r)
    ur_z = np.zeros_like(T); grad_z_masked(ur, dz, ur_z, face_z)
    
    for i in prange(1, Nr-1):
        ri = (i * dr)
        inv_ri = 1.0 / ri
        
        for k in range(1, Nz-1):
            if fluid is not None and fluid[i, k] == 0:
                continue
                
            # 1. Divergence of u (for bulk viscosity/compressible terms)
            divu = dur_dr[i, k] + (ur[i, k] * inv_ri) + duz_dz[i, k]
            
            # 2. Strain rates & Stresses (Standard Newtonian)
            lam = -2.0/3.0 * mu[i, k]
            
            tau_rr = 2*mu[i, k]*dur_dr[i, k] + lam*divu
            tau_tt = 2*mu[i, k]*(ur[i, k]*inv_ri) + lam*divu
            tau_zz = 2*mu[i, k]*duz_dz[i, k] + lam*divu
            
            tau_rz = mu[i, k]*(uz_r[i, k] + ur_z[i, k])
            tau_rt = mu[i, k]*(dut_dr[i, k] - ut[i, k]*inv_ri)
            tau_tz = mu[i, k]*dut_dz[i, k]
            
            # 3. Dissipation Function Phi = Tau : Grad(u)
            Phi = (
                tau_rr * dur_dr[i, k] +
                tau_tt * (ur[i, k] * inv_ri) +
                tau_zz * duz_dz[i, k] +
                tau_rz * (uz_r[i, k] + ur_z[i, k]) +
                tau_rt * (dut_dr[i, k] - ut[i, k] * inv_ri) +
                tau_tz * dut_dz[i, k]
            )
            
            # Update Temperature with Safety Clamp
            T[i, k] += dt * Phi / (rho[i, k] * c_v)


#############################################################################################
######################### THE KERNELS BELOW ARE TO TEST IMPLICIT UPDATES ####################
######################### FULL NAVIER STOKES IMPLICIT WILL BE IMPLEMENTED LATER #############
#############################################################################################

@njit(cache=True)
def update_neutral_vtheta_implicit_source(un_theta, vi_theta, 
                                          ni, nu_in, mi, 
                                          nn, mn, 
                                          dt, mask):
    """
    Updates Neutral v_theta using an Implicit Source term for Drag.
    Allows large timesteps (dt) without instability from stiff collisions.
    
    Formula:
      u_new = ( u_old + dt * alpha * v_i ) / ( 1 + dt * alpha )
      where alpha = (ni * mi * nu_in) / (nn * mn)
    """
    Nr, Nz = un_theta.shape
    
    for i in range(Nr):
        for j in range(Nz):
            if mask[i, j] == 1:
                # 1Local Densities
                rho_n = nn[i, j] * mn
                rho_i = ni[i, j] * mi
                
                # Collision Rate (Coupling Strength)
                # alpha has units [1/s]
                if rho_n > 1e-20:
                    nu_coupling = (rho_i * nu_in[i, j]) / rho_n
                else:
                    nu_coupling = 0.0
                
                
                # Implicit Update
                # The denominator handles the stiffness
                denom = 1.0 + dt * nu_coupling
                
                u_old = un_theta[i, j]
                v_ion = vi_theta[i, j]
                
                # Numerator: Old Velocity + 'Target' velocity weighted by coupling
                numerator = u_old + dt * (nu_coupling * v_ion )
                
                un_theta[i, j] = numerator / denom
                
            else:
                un_theta[i, j] = 0.0

@njit(cache=True)
def update_neutral_temperature_implicit(Tn, Te, Ti, ne, nn, 
                                        nu_en, nu_in, 
                                        me, mi, mn, dt, mask, Cv):
    """
    Updates Neutral Temperature (Tn) using a Semi-Implicit collisional operator.
    """
    Nr, Nz = Tn.shape
    kb = constants.kb
    
    # 3.0 factor for Maxwellian thermal relaxation
    coeff_en = 3.0 * (me / mn) * kb
    coeff_in = 3.0 * (mi / mn) * kb
    
    for i in range(Nr):
        for j in range(Nz):
            if mask[i, j] == 1:
                
                # Local variables
                n_n = nn[i, j]
                n_e = ne[i, j]
                
                # 1. Calculate Heat Transfer Coefficients [W / (m^3 K)]
                K_en = coeff_en * n_e * nu_en[i, j]
                K_in = coeff_in * n_e * nu_in[i, j] 
                
                K_total = K_en + K_in
                
                if K_total > 0.0:
                    # Target Weighted Temperature
                    T_target = (K_en * Te[i, j] + K_in * Ti[i, j]) / K_total
                    
                    # 2. Calculate Relaxation Frequency [1/s]
                    # Divide by Volumetric Heat Capacity (rho * Cv)
                    rho_n = n_n * mn
                    vol_Cv = rho_n * Cv
                    
                    nu_relax = K_total / vol_Cv
                    
                    # 3. Implicit Update
                    # Tn_new = (Tn_old + dt * nu * T_target) / (1 + dt * nu)
                    numerator = Tn[i, j] + dt * nu_relax * T_target
                    denominator = 1.0 + dt * nu_relax
                    
                    Tn[i, j] = numerator / denominator

@njit(parallel=True, cache=True)
def add_ion_neutral_frictional_heating(Tn, un_t, vi_t,
                                       ni, nn, nu_in,
                                       mi, mn, Cv, dt, mask):
    """
    Explicitly adds Frictional Heating (Slip Heating) to Neutrals.
    Q = rho_i * nu_in * |v_i - v_n|^2
    dT = dt * Q / (rho_n * Cv)
    """
    Nr, Nz = Tn.shape
    
    for i in prange(Nr):
        for j in range(Nz):
            if mask[i, j] == 1:
                rho_n = nn[i, j] * mn
                rho_i = ni[i, j] * mi
                
                if rho_n > 1e-20:
                    # Squared Slip Velocity
                    dv2 = (vi_t[i, j] - un_t[i, j])**2 
                          
                    # Heating Source [W/m^3]
                    # For Charge Exchange, the energy transfer is efficient.
                    # Q ~ rho_i * nu_in * dv^2
                    Q_fric = rho_i * nu_in[i, j] * dv2
                    
                    # Temperature Increment
                    dT = dt * Q_fric / (rho_n * Cv)                        
                    Tn[i, j] += dT
                    
@njit(cache=True)
def solve_implicit_viscosity_sor_prev(un_theta, nn, mn, mask, mu_grid,
                                 dt, dr, dz, max_iter=20, omega=1.4):
    Nr, Nz = un_theta.shape
    inv_dr2 = 1.0 / (dr * dr)
    inv_dz2 = 1.0 / (dz * dz)

    # Make a copy of the "Old" velocity for the Inertia Source Term
    # This ensures u_old stays fixed while un_theta (u_new) converges.
    u_old = un_theta.copy() 

    for k in range(max_iter):
        max_diff = 0.0
        
        for i in range(1, Nr-1):
            # ... [Geometric coefficients setup is fine] ...
            r = i * dr
            c_east = ((r + 0.5*dr) / r) * inv_dr2
            c_west = ((r - 0.5*dr) / r) * inv_dr2
            c_self_geo = 1.0 / (r * r)

            for j in range(1, Nz):
                if mask[i, j] == 1:
                    # ... [Neighbor identification is fine] ...
                    val_E = un_theta[i+1, j]; val_W = un_theta[i-1, j]
                    
                    if j == Nz - 1: val_N = un_theta[i, j]; c_north = 0.0
                    else:           val_N = un_theta[i, j+1]; c_north = inv_dz2

                    if j == 0: val_S = 0.0; c_south = inv_dz2
                    else:      val_S = un_theta[i, j-1]; c_south = inv_dz2

                    # --- Matrix & RHS ---
                    mu_val = mu_grid[i, j]
                    rho = max(nn[i, j], 1e12) * mn
                    A_time = rho / dt
                    
                    A_P = A_time + mu_val * (c_east + c_west + c_north + c_south + c_self_geo)
                    
                    RHS = (A_time * u_old[i, j]) + mu_val * (
                        c_east * val_E + c_west * val_W + 
                        c_north * val_N + c_south * val_S
                    )
                    
                    # SOR Update
                    v_star = RHS / A_P
                    v_new = (1.0 - omega)*un_theta[i, j] + omega*v_star
                    
                    diff = abs(v_new - un_theta[i, j])
                    if diff > max_diff: max_diff = diff
                    
                    # Update the guess for the next iteration
                    un_theta[i, j] = v_new
                    
        if max_diff < 1e-5: break

@njit(cache=True)
def solve_implicit_viscosity_sor(un_theta, nn, mn, mask, mu_grid,
                                 dt, dr, dz, max_iter=20, omega=1.4):
    """
    Implicit Viscosity for AZIMUTHAL velocity v_theta.
    """
    Nr, Nz = un_theta.shape
    inv_dr2 = 1.0 / (dr * dr)
    inv_dz2 = 1.0 / (dz * dz)

    u_old = un_theta.copy() 

    for k in range(max_iter):
        max_diff = 0.0
        
        for i in range(1, Nr-1):
            r = i * dr
            # Base geometric coefficients
            c_east_base = ((r + 0.5*dr) / r) * inv_dr2
            c_west_base = ((r - 0.5*dr) / r) * inv_dr2
            c_self_geo = 1.0 / (r * r)
            c_north_base = inv_dz2
            c_south_base = inv_dz2

            for j in range(1, Nz-1): # Iterate interior
                if mask[i, j] == 1:
                    
                    # --- EAST ---
                    if mask[i+1, j] == 0:
                        val_E = 0.0
                        c_east = 2.0 * c_east_base # Wall at face
                    else:
                        val_E = un_theta[i+1, j]
                        c_east = c_east_base

                    # --- WEST ---
                    if mask[i-1, j] == 0:
                        val_W = 0.0
                        c_west = 2.0 * c_west_base
                    else:
                        val_W = un_theta[i-1, j]
                        c_west = c_west_base
                    
                    # --- NORTH ---
                    if mask[i, j+1] == 0:
                        val_N = 0.0
                        c_north = 2.0 * c_north_base
                    else:
                        val_N = un_theta[i, j+1]
                        c_north = c_north_base

                    # --- SOUTH ---
                    if mask[i, j-1] == 0:
                        val_S = 0.0
                        c_south = 2.0 * c_south_base
                    else:
                        val_S = un_theta[i, j-1]
                        c_south = c_south_base

                    # --- Matrix & RHS ---
                    mu_val = mu_grid[i, j]
                    rho = max(nn[i, j], 1e12) * mn
                    A_time = rho / dt
                    
                    A_P = A_time + mu_val * (c_east + c_west + c_north + c_south + c_self_geo)
                    
                    RHS = (A_time * u_old[i, j]) + mu_val * (
                        c_east * val_E + c_west * val_W + 
                        c_north * val_N + c_south * val_S
                    )
                    
                    v_new = (1.0 - omega)*un_theta[i, j] + omega*(RHS / A_P)
                    
                    diff = abs(v_new - un_theta[i, j])
                    if diff > max_diff: max_diff = diff
                    
                    un_theta[i, j] = v_new
                    
        if max_diff < 1e-5: break
        
"""
@njit(cache=True)
def solve_implicit_heat_sor(Tn, nn, mn, mask, kappa_grid, c_v,
                            dt, dr, dz, max_iter=20, omega=1.4):
    #Solves Neutral Temperature Diffusion.
    Nr, Nz = Tn.shape
    
    inv_dr2 = 1.0 / (dr * dr)
    inv_dz2 = 1.0 / (dz * dz)

    # Create a copy for the Inertia Source Term
    # We need the temperature at time 'n' to calculate dT/dt properly.
    Tn_old = Tn.copy()

    for k in range(max_iter):
        
        # Enforce Axis Symmetry
        for j_sym in range(Nz):
            if mask[0, j_sym] == 1:
                Tn[0, j_sym] = Tn[1, j_sym]

        max_diff = 0.0

        # Iterate only up to Nr-1 (Excluding the outer wall)
        for i in range(1, Nr - 1):
            r = i * dr
            
            c_east = ((r + 0.5*dr) / r) * inv_dr2
            c_west = ((r - 0.5*dr) / r) * inv_dr2
            
            # Iterate only up to Nz-1 (Excluding top/bottom walls)
            for j in range(1, Nz - 1):
                if mask[i, j] == 1:
                    
                    # --- Neighbors ---
                    val_E = Tn[i+1, j] 
                    val_W = Tn[i-1, j]
                    val_N = Tn[i, j+1]
                    val_S = Tn[i, j-1]
                    
                    c_north = inv_dz2
                    c_south = inv_dz2

                    # --- Matrix Setup ---
                    kappa_val = kappa_grid[i, j]
                    
                    rho = nn[i, j] * mn
                    Cv_vol = rho * c_v
                    A_time = Cv_vol / dt
                    
                    A_P = A_time + kappa_val * (c_east + c_west + c_north + c_south)
                    
                    # Use Tn_old[i, j] 
                    RHS = (A_time * Tn_old[i, j]) + kappa_val * (
                        c_east * val_E + c_west * val_W + 
                        c_north * val_N + c_south * val_S
                    )
                    
                    # --- SOR Update ---
                    T_star = RHS / A_P
                    T_new = (1.0 - omega) * Tn[i, j] + omega * T_star
                    
                    diff = abs(T_new - Tn[i, j])
                    if diff > max_diff: max_diff = diff
                    
                    Tn[i, j] = T_new
                    
        if max_diff < 1e-5: break
"""

@njit(cache=True)
def solve_implicit_heat_sor(Tn, nn, mn, mask, kappa_grid, c_v,
                            dt, dr, dz, max_iter=20, omega=1.4,
                            mu_grid=None, rho_grid=None, 
                            mass=1.0, gamma=1.4, cp=1.0, 
                            T_wall_fixed=300.0, alpha=0.5):
    """
    Solves Neutral Temperature Diffusion with COUPLED Robin Boundary Condition.
    Updates the wall temperature inside the SOR loop to ensure consistency.
    """
    Nr, Nz = Tn.shape
    
    inv_dr2 = 1.0 / (dr * dr)
    inv_dz2 = 1.0 / (dz * dz)

    Tn_old = Tn.copy()
    
    # Physics Constants
    kb = constants.kb
    pi = np.pi
    
    # Pre-calculate Jump Coefficients
    # L_jump ~ (2-alpha)/alpha * (2*gamma)/(gamma+1) * (lambda / Pr)
    coeff_alpha = (2.0 - alpha) / alpha
    coeff_gamma = (2.0 * gamma) / (gamma + 1.0)

    for k in range(max_iter):
        
        # 1. Enforce Symmetry BCs first
        for j_sym in range(Nz):
            if mask[0, j_sym] == 1:
                Tn[0, j_sym] = Tn[1, j_sym]       # r=0
            if mask[Nr-1, j_sym] == 1:
                Tn[Nr-1, j_sym] = Tn[Nr-2, j_sym] # z=L (top)

        max_diff = 0.0

        # 2. Update Interior Nodes (1 to Nr-2)
        for i in range(1, Nr - 1):
            r = i * dr
            c_east = ((r + 0.5*dr) / r) * inv_dr2
            c_west = ((r - 0.5*dr) / r) * inv_dr2
            
            for j in range(1, Nz - 1):
                if mask[i, j] == 1:
                    
                    val_E = Tn[i+1, j]; val_W = Tn[i-1, j]
                    val_N = Tn[i, j+1]; val_S = Tn[i, j-1]
                    
                    c_north = inv_dz2; c_south = inv_dz2

                    # --- Matrix Setup ---
                    kappa_val = kappa_grid[i, j]
                    rho = nn[i, j] * mn
                    Cv_vol = rho * c_v
                    A_time = Cv_vol / dt
                    
                    A_P = A_time + kappa_val * (c_east + c_west + c_north + c_south)
                    
                    RHS = (A_time * Tn_old[i, j]) + kappa_val * (
                        c_east * val_E + c_west * val_W + 
                        c_north * val_N + c_south * val_S
                    )
                    
                    # --- SOR Update ---
                    T_star = RHS / A_P
                    T_new = (1.0 - omega) * Tn[i, j] + omega * T_star
                    
                    diff = abs(T_new - Tn[i, j])
                    if diff > max_diff: max_diff = diff
                    
                    Tn[i, j] = T_new
        
        # 3. Update Wall Node (Nr-1)
        # We only do this for the radial wall (i = Nr-1)
        if mu_grid is not None and rho_grid is not None:
            i_wall = Nr - 1
            i_neigh = Nr - 2
            
            for j in range(1, Nz - 1):
                if mask[i_wall, j] == 1:
                    # Get Local Properties from neighbor
                    T_gas = Tn[i_neigh, j]
                    rho_gas = max(rho_grid[i_neigh, j], 1e-25)
                    mu_gas = mu_grid[i_neigh, j]
                    kap_gas = kappa_grid[i_neigh, j]
                    
                    # --- Calculate h_eff (Knudsen Layer) ---
                    # v_th = sqrt(8 k T / pi m)
                    v_th = np.sqrt(8.0 * kb * T_gas / (pi * mn))
                    
                    # lambda = 2 * mu / (rho * v_th)
                    lam = 2.0 * mu_gas / (rho_gas * v_th)
                    
                    # Pr = Cp * mu / k
                    Pr = cp * mu_gas / max(kap_gas, 1e-15)
                    
                    # Jump Length
                    L_jump = coeff_alpha * coeff_gamma * (lam / Pr)
                    L_jump = max(L_jump, 1e-12) # Avoid /0
                    
                    h_eff = kap_gas / L_jump
                    
                    # --- Apply Robin Update ---
                    # T_wall = (k*T_inner + h*dr*T_fixed) / (k + h*dr)
                    numerator = (kap_gas * T_gas) + (h_eff * dr * T_wall_fixed)
                    denominator = kap_gas + (h_eff * dr)
                    
                    Tn[i_wall, j] = numerator / denominator

        if max_diff < 1e-5: break



# Not being used right now
@njit(parallel=True, cache=True)
def update_neutral_vtheta_explicit_force(un_theta, Jir, Bz, nn, mn, dt, mask):
    """
    Directly applies Lorentz Force to neutrals.
    Fixes the '1/7th Force' issue seen in the core of your plot.
    """
    Nr, Nz = un_theta.shape
    
    for i in prange(Nr):
        for j in range(Nz):
            if mask[i, j] == 1:
                # 1. Total Drive Force
                # Note: If you have electron current, add it here: J_tot = Jir + Jer
                F_drive = -1.0 * Jir[i, j] * Bz[i, j]
                
                # 2. Mass Density
                rho = max(nn[i, j] * mn, 1e-12)
                
                # 3. Acceleration (100% efficiency)
                accel = F_drive / rho
                
                # 4. Explicit Update
                un_theta[i, j] += dt * accel