import numpy as np
from centrifugesim import constants

kB = constants.kb

def separation_factor_map_integral(
    T_n_grid: np.ndarray,          # (Nr, Nz) [K]
    un_theta_grid: np.ndarray,     # (Nr, Nz) [m/s]
    mask: np.ndarray,              # (Nr, Nz) 1=fluid, 0=solid
    r: np.ndarray,                 # (Nr,) [m], includes r[0]=0
    m_light: float,                # [kg]
    m_heavy: float,                # [kg]
) -> np.ndarray:
    """
    ln alpha(r,z) = ∫ (Δm/kB) * v_theta^2/(r T) dr.

    - Piecewise integration over contiguous valid-r segments per z.
    - Integration never uses i=0 (axis); it starts at i=1 (or the first valid i>=1).
    - alpha is set to 0 on the axis where mask==1 (user request).
    - NaN only where mask==0.
    """
    T = T_n_grid.astype(float, copy=False)
    v = un_theta_grid.astype(float, copy=False)
    m = mask.astype(float, copy=False)
    r = r.astype(float, copy=False)

    dm = float(m_heavy) - float(m_light)

    Nr, Nz = T.shape
    alpha = np.full((Nr, Nz), np.nan, dtype=float)

    # Only defined for i>=1 (avoid r=0). We'll never read integrand[0,:].
    integrand = (dm / kB) * (v * v) / (r[:, None] * T)  # (Nr, Nz)

    for iz in range(Nz):
        valid = m[:, iz] > 0.5
        if not np.any(valid):
            continue

        idx = np.where(valid)[0]
        splits = np.where(np.diff(idx) != 1)[0] + 1
        blocks = np.split(idx, splits)

        lnalpha_col = np.full(Nr, np.nan, dtype=float)

        for b in blocks:
            i0 = b[0]
            i1 = b[-1]

            # Start index for this block: never use axis node (i=0)
            i_start = i0 if i0 >= 1 else 1

            # If the block is entirely at the axis only (i1 == 0), nothing to integrate
            if i1 < i_start:
                continue

            # Reference at i_start
            lnalpha_col[i_start] = 0.0
            if i1 == i_start:
                continue

            rr = r[i_start:i1+1]
            ff = integrand[i_start:i1+1, iz]

            dr = np.diff(rr)
            trap = 0.5 * (ff[1:] + ff[:-1]) * dr
            lnalpha_col[i_start+1:i1+1] = np.cumsum(trap)

        # User request: alpha = 1 on axis where it's fluid
        if valid[0]:
            alpha[0, iz] = 1.0

        # Fill alpha on valid nodes i>=1 from exp(lnalpha_col)
        alpha[1:, iz] = np.where(valid[1:], np.exp(lnalpha_col[1:]), np.nan)

        # Ensure solids are NaN (including axis if solid)
        alpha[:, iz] = np.where(valid, alpha[:, iz], np.nan)

    return alpha