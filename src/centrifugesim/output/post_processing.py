import numpy as np
from centrifugesim import constants

kB = constants.kb

def separation_factor_map_integral(
        T_n_grid: np.ndarray,       # (Nr, Nz) [K]
        un_theta_grid: np.ndarray,  # (Nr, Nz) [m/s]
        mask: np.ndarray,           # (Nr, Nz) 1=fluid, 0=solid
        r: np.ndarray,              # (Nr,) [m], includes r[0]=0
        m_light: float,             # [kg]
        m_heavy: float,             # [kg]
        ni_grid: np.ndarray = None,    # (Nr, Nz) [m^-3]
        nn_grid: np.ndarray = None,    # (Nr, Nz) [m^-3]
        nu_in_grid: np.ndarray = None, # (Nr, Nz) [s^-1]
        vir_grid: np.ndarray = None,   # (Nr, Nz) [m/s]
        vnr_grid: np.ndarray = None,   # (Nr, Nz) [m/s]
        include_drag: bool = False
    ):
        """
        Calculates the separation factor alpha and a drag correction factor.

        Original alpha:
        ln alpha(r,z) = ∫ (Δm/kB) * v_theta^2/(r T) dr

        Drag correction:
        ln alpha_corr(r,z) = ∫ (Δm/kB) * (ni/nn) * nu_in * (vir-vnr) / T dr

        Returns:
            (alpha, alpha_corr)
            - alpha: The original separation factor map.
            - alpha_corr: The correction map. If include_drag is False, this is 1.0 where valid.
        """
        T = T_n_grid.astype(float, copy=False)
        v = un_theta_grid.astype(float, copy=False)
        m = mask.astype(float, copy=False)
        r = r.astype(float, copy=False)

        dm = float(m_heavy) - float(m_light)

        Nr, Nz = T.shape
        alpha = np.full((Nr, Nz), np.nan, dtype=float)
        alpha_corr = np.full((Nr, Nz), np.nan, dtype=float)

        # --- 1. Prepare Integrands ---
        
        # Standard Integrand: (Nr, Nz)
        # Only defined for i>=1 (avoid r=0). We'll never read integrand[0,:].
        integrand = (dm / kB) * (v * v) / (r[:, None] * T)

        # Drag Integrand: (Nr, Nz)
        integrand_drag = None
        if include_drag:
            # Verify inputs
            required_fields = [ni_grid, nn_grid, nu_in_grid, vir_grid, vnr_grid]
            if any(f is None for f in required_fields):
                raise ValueError("All drag-related fields (ni, nn, nu_in, vir, vnr) must be provided when include_drag=True.")
            
            ni = ni_grid.astype(float, copy=False)
            nn = nn_grid.astype(float, copy=False)
            nu = nu_in_grid.astype(float, copy=False)
            vir = vir_grid.astype(float, copy=False)
            vnr = vnr_grid.astype(float, copy=False)

            # Formula: (Δm/kB) * (ni/nn) * nu_in * (vir-vnr) / T
            # We assume nn > 0 in valid fluid regions.
            integrand_drag = (dm / kB) * (ni / nn) * nu * (vir - vnr) / T

        # --- 2. Integration Loop ---

        for iz in range(Nz):
            valid = m[:, iz] > 0.5
            if not np.any(valid):
                continue

            idx = np.where(valid)[0]
            splits = np.where(np.diff(idx) != 1)[0] + 1
            blocks = np.split(idx, splits)

            lnalpha_col = np.full(Nr, np.nan, dtype=float)
            lnalpha_corr_col = np.full(Nr, np.nan, dtype=float)

            for b in blocks:
                i0 = b[0]
                i1 = b[-1]

                # Start index for this block: never use axis node (i=0)
                i_start = i0 if i0 >= 1 else 1

                # If the block is entirely at the axis only (i1 == 0), nothing to integrate
                if i1 < i_start:
                    continue

                # Reference at i_start (set integration constant to 0 => alpha=1)
                lnalpha_col[i_start] = 0.0
                
                # Initialize drag correction at reference
                if include_drag:
                    lnalpha_corr_col[i_start] = 0.0

                if i1 == i_start:
                    continue

                # Common geometry for this segment
                rr = r[i_start:i1+1]
                dr = np.diff(rr)

                # Integrate Standard Alpha
                ff = integrand[i_start:i1+1, iz]
                trap = 0.5 * (ff[1:] + ff[:-1]) * dr
                lnalpha_col[i_start+1:i1+1] = np.cumsum(trap)

                # Integrate Drag Correction
                if include_drag:
                    ff_d = integrand_drag[i_start:i1+1, iz]
                    trap_d = 0.5 * (ff_d[1:] + ff_d[:-1]) * dr
                    lnalpha_corr_col[i_start+1:i1+1] = np.cumsum(trap_d)

            # User request: alpha = 1 on axis where it's fluid
            if valid[0]:
                alpha[0, iz] = 1.0
                alpha_corr[0, iz] = 1.0

            # Fill alpha on valid nodes i>=1 from exp(lnalpha_col)
            alpha[1:, iz] = np.where(valid[1:], np.exp(lnalpha_col[1:]), np.nan)

            # Fill alpha_corr
            if include_drag:
                alpha_corr[1:, iz] = np.where(valid[1:], np.exp(lnalpha_corr_col[1:]), np.nan)
            else:
                # If drag is disabled, correction is 1.0 everywhere fluid exists
                alpha_corr[1:, iz] = np.where(valid[1:], 1.0, np.nan)

            # Ensure solids are NaN (including axis if solid)
            alpha[:, iz] = np.where(valid, alpha[:, iz], np.nan)
            alpha_corr[:, iz] = np.where(valid, alpha_corr[:, iz], np.nan)

        return alpha, alpha_corr