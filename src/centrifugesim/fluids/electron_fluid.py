import numpy as np

from centrifugesim.fluids.neutral_fluid import NeutralFluidContainer
from centrifugesim.fluids import electron_fluid_helper
from centrifugesim.geometry.geometry import Geometry
from centrifugesim import constants

class ElectronFluidContainer:
    """
    update this
    """
    def __init__(self, geom:Geometry, ne0, ne_floor, Te_floor):

        self.Nr = geom.Nr
        self.Nz = geom.Nz

        # floor values
        self.ne0 = ne0
        self.ne_floor = ne_floor
        self.Te_floor = Te_floor

        self.ne_grid = np.zeros((geom.Nr,geom.Nz)).astype(np.float64) # [m^-3]
        self.Te_grid = np.zeros((geom.Nr,geom.Nz)).astype(np.float64) # [K] !

        self.ne_grid_prev = np.zeros((geom.Nr,geom.Nz)).astype(np.float64) # [m^-3]
        self.Te_grid_prev = np.zeros((geom.Nr,geom.Nz)).astype(np.float64) # [K] !

        self.pe_grid = np.zeros((geom.Nr,geom.Nz)).astype(np.float64) # [Pa] !
        self.grad_pe_grid_r = np.zeros((geom.Nr,geom.Nz)).astype(np.float64) # [Pa/m] !
        self.grad_pe_grid_z = np.zeros((geom.Nr,geom.Nz)).astype(np.float64) # [Pa/m] !

        self.nu_ei_grid = np.zeros((geom.Nr,geom.Nz)).astype(np.float64)
        self.nu_en_grid = np.zeros((geom.Nr,geom.Nz)).astype(np.float64)
        self.nu_anom_grid = np.zeros((geom.Nr,geom.Nz)).astype(np.float64)
        self.nu_e_grid = np.zeros((geom.Nr,geom.Nz)).astype(np.float64)

        self.sigma_P_grid = np.zeros((geom.Nr,geom.Nz)).astype(np.float64)
        self.sigma_parallel_grid = np.zeros((geom.Nr,geom.Nz)).astype(np.float64)
        self.sigma_H_grid = np.zeros((geom.Nr,geom.Nz)).astype(np.float64)

        self.kappa_parallel_grid = np.zeros((geom.Nr,geom.Nz)).astype(np.float64)
        self.kappa_perp_grid = np.zeros((geom.Nr,geom.Nz)).astype(np.float64)

        self.beta_e_grid = np.zeros((geom.Nr,geom.Nz)).astype(np.float64)

        # electron drift velocity components (for particle pusher)
        self.uer_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.uet_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.uez_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)

    def update_drift_velocities(self, hybrid_pic):
        """
        Update electron drift velocities from current densities and ne:
            u_e = J_e / ( -e * n_e )
        """
        qe = constants.q_e  # electron charge (C)
        ne_eff = np.maximum(self.ne_grid, self.ne_floor)  # m^-3 (avoid div-by-zero)
        self.uer_grid[:, :] = hybrid_pic.Jer_grid / (-qe * ne_eff)
        self.uez_grid[:, :] = hybrid_pic.Jez_grid / (-qe * ne_eff)

    def update_pressure(self):
        self.pe_grid = constants.kb*self.Te_grid*self.ne_grid

    def set_kappa(self, hybrid_pic):
        """
        Update electron thermal conductivities (W / m / K),
        including e-i and e-n collisions:
            kappa_parallel = C * n_e * kB^2 * T_e * tau_e / m_e
            kappa_perp     = kappa_parallel / (1 + (Omega_e * tau_e)^2)

        where tau_e = 1 / (nu_ei + nu_en), Omega_e = e * |B| / m_e.
        """
        # Short-hands
        ne = self.ne_grid
        Te = self.Te_grid

        # Physical constants
        kb = constants.kb
        me = constants.m_e
        qe = constants.q_e

        # Numerically safe floors
        ne_eff = np.maximum(ne, self.ne_floor)        # m^-3 (very small floor just to avoid div-by-zero)
        Te_eff = np.maximum(Te, self.Te_floor)       # K
        nu_e   = np.maximum(self.nu_e_grid, 1e-30)  # total electron momentum-transfer frequency
        tau_e  = 1.0 / nu_e

        # Electron gyrofrequency
        Omega_e = np.abs(qe) * hybrid_pic.Bmag_grid / me

        # Parallel electron thermal conductivity (Spitzer–Härm with collisions folded into tau_e)
        Ck = 3.16  # Braginskii/Spitzer-Härm coefficient for electrons
        kappa_par = Ck * ne_eff * (kb*kb) * Te_eff * tau_e / me  # W / m / K

        # Perpendicular electron thermal conductivity (classical suppression by magnetization)
        chi2 = (Omega_e * tau_e)**2
        kappa_perp = kappa_par / (1.0 + chi2)

        # Write back
        self.kappa_parallel_grid[:, :] = kappa_par
        self.kappa_perp_grid[:, :]     = kappa_perp

    def set_electron_collision_frequencies(
        self, nn_grid, lnLambda=12.0, sigma_en_m2=2.0e-19, geom=None, J_mag=None, Ti=None, mi=None, include_anomalous=False, Te_is_eV=False, k_interp_elastic=None
    ):
        """
        Compute and set electron collision frequencies:
          - self.nu_en : electron-neutral momentum-transfer
          - self.nu_ei : electron-ion (Spitzer)
          - self.nu_e  : total = nu_en + nu_ei
        """
        nu_en_grid, nu_ei_grid, nu_e_grid = electron_fluid_helper.electron_collision_frequencies(
            self.Te_grid,
            self.ne_grid,
            nn_grid,
            lnLambda=lnLambda, sigma_en_m2=sigma_en_m2,
            Te_is_eV=Te_is_eV
        )
        self.nu_en_grid[:] = nu_en_grid
        self.nu_ei_grid[:] = nu_ei_grid

        if(k_interp_elastic is not None):
            self.nu_en_grid[:]*=0
            self.nu_en_grid[geom.mask==1] = nn_grid[geom.mask==1]*k_interp_elastic(self.Te_grid[geom.mask==1])
            self.nu_en_grid[geom.mask==0] = 1e9 # for safety

            self.nu_e_grid = self.nu_en_grid + self.nu_ei_grid
        
        if include_anomalous:
            nu_anom_grid = electron_fluid_helper.get_anomalous_collision_frequency(
                self.ne_grid, self.Te_grid, Ti, J_mag, mi
            )
            nu_e_grid[geom.mask==1] += nu_anom_grid[geom.mask==1]
            self.nu_anom_grid[geom.mask==1] = nu_anom_grid[geom.mask==1]

        self.nu_e_grid[:]  = nu_e_grid

    def set_electron_conductivities(
        self, hybrid_pic, lnLambda=12.0, Te_is_eV=False
    ):
        """
        Compute and set electron conductivity tensor components:
          - self.sigma_parallel, self.sigma_P, self.sigma_H
        """

        Bmag_grid = hybrid_pic.Bmag_grid

        sigma_par_e, sigma_P_e, sigma_H_e, _beta_e = electron_fluid_helper.electron_conductivities(
            self.Te_grid, self.ne_grid, Bmag_grid, self.nu_e_grid, self.nu_anom_grid
        )
        self.sigma_parallel_grid[:] = sigma_par_e
        self.sigma_P_grid[:]        = sigma_P_e
        self.sigma_H_grid[:]        = sigma_H_e
        self.beta_e_grid[:]         = _beta_e

    # stable version being used. However no diffusion across cells, leading to unphysical jumps in ne
    def update_density_implicit_local(self, geom, ion_fluid, nu_iz_grid, nu_RR_recomb_grid, beta_rec_grid, dt):
        """
        Updates ne using the Analytic Logistic Solution with Anisotropic Diffusion.
        Includes Log-Space Clamping to prevent ionization-heating instabilities.
        """
        
        # Fetch Rates (Inputs from Chemistry)
        nu_iz = nu_iz_grid      # [1/s]
        nu_RR = nu_RR_recomb_grid  # [1/s] (Radiative Recomb)
        beta_rec = beta_rec_grid # [m^3/s]
        
        # Geometric Loss Factors
        R_max = geom.r.max()
        L_z = geom.z.max() - geom.z.min()
        
        inv_Lambda_r_sq = (2.405 / R_max)**2
        inv_Lambda_z_sq = (np.pi / (2.0 * L_z))**2
        
        # Compute Effective Loss Rate (Anisotropic)
        nu_diff_grid = electron_fluid_helper.compute_ambipolar_loss_rate_anisotropic(
            self.Te_grid,
            ion_fluid.Ti_grid,
            ion_fluid.nu_i_grid,
            self.beta_e_grid,
            ion_fluid.beta_i_grid,
            ion_fluid.m_i,
            constants.kb,
            inv_Lambda_z_sq,
            inv_Lambda_r_sq
        )
        
        # Call the Anisotropic Analytic Kernel
        ne_new = np.zeros_like(self.ne_grid)
        
        electron_fluid_helper.time_advance_ne_analytic_kernel_anisotropic(
            ne_new,             # Output
            self.ne_grid,       # Input (Old)
            nu_iz,              # Ionization Source
            nu_diff_grid,       # Calculated Anisotropic Loss
            nu_RR,              # Radiative Recombination Loss
            beta_rec,           # Recombination Sink
            dt,                 # Timestep
            geom.mask,
            self.ne_floor
        )
        
        # =========================================================
        # Stability: Log-Space Clamping & Under-Relaxation
        # =========================================================
        
        # Safety floor for logs
        ne_floor_log = 1e10
        
        # Compute Logarithms
        log_old = np.log10(np.maximum(self.ne_grid, ne_floor_log))
        log_target = np.log10(np.maximum(ne_new, ne_floor_log))
        
        # Calculate desired change
        diff = log_target - log_old
        
        # MAX_LOG_STEP = 0.5 means max change is 10^0.5 ~= 3.16x per timestep.
        # This prevents the 10^8 jumps.
        MAX_LOG_STEP = 0.5 
        diff_clipped = np.clip(diff, -MAX_LOG_STEP, MAX_LOG_STEP)
        
        # Apply Under-Relaxation to the *clipped* change
        # RELAX = 0.2 means we take 20% of that clipped step.
        # Increase this towards 1.0 once the simulation is stable.
        RELAX = 0.2 
        
        log_final = log_old + RELAX * diff_clipped
        
        # Finalize
        self.ne_grid = 10**(log_final)

    def update_density_implicit(self, geom, ion_fluid, hybrid_pic,
                                nu_iz_grid, nu_RR_recomb_grid, beta_rec_grid, dt, 
                                delta_sheath=1.0, closed_top=True, include_Bohm=False):
        """
        Updates electron density using Operator Splitting.
        
        Returns:
            dne_chem (Array): Change in density due to Ionization/Recombination only. 
                              (Use this to update nn everywhere).
            dne_boundary (Array): Density of neutrals created at walls due to recycling.
                                  (Add this to nn at boundaries).
        """
        
        # --- PHASE 1: CHEMISTRY (Local Source/Sink) ---
        ne_star = np.zeros_like(self.ne_grid)
        
        # Disable 0D diffusion loss for this step
        nu_loss_dummy = np.zeros_like(self.ne_grid) 
        
        electron_fluid_helper.time_advance_ne_analytic_kernel_anisotropic(
            ne_star,            # Output: n_star
            self.ne_grid,       # Input: n_old
            nu_iz_grid,         
            nu_loss_dummy,      # <--- DISABLED
            nu_RR_recomb_grid,
            beta_rec_grid,
            dt,
            geom.mask,
            self.ne_floor
        )

        # Stability: Clamp Log-Space Overshoot
        ne_floor_log = 1e10
        log_old = np.log10(np.maximum(self.ne_grid, ne_floor_log))
        log_star = np.log10(np.maximum(ne_star, ne_floor_log))
        
        diff = log_star - log_old
        MAX_LOG_STEP = 0.5 
        diff_clipped = np.clip(diff, -MAX_LOG_STEP, MAX_LOG_STEP)
        
        # Recalculate n_star
        ne_star = 10**(log_old + diff_clipped)
        
        # 1. Calculate Chemistry Delta (for Neutral update)
        dne_chem = ne_star - self.ne_grid

        # --- PHASE 2: TRANSPORT (Global 2D Diffusion) ---
        
        # Diffusion Coefficients
        Da_par, Da_perp = electron_fluid_helper.compute_ambipolar_coefficients(
            self.Te_grid,
            ion_fluid.Ti_grid,
            ion_fluid.nu_i_grid,
            self.beta_e_grid,
            ion_fluid.beta_i_grid,
            ion_fluid.m_i,
            constants.kb
        )
        if(include_Bohm):
            D_Bohm = 1/16.0*constants.kb*self.Te_grid/(constants.q_e*hybrid_pic.Bmag_grid)
            Da_perp += D_Bohm

        #Get Cathode info
        #i_cathode = geom.i_cathode_z_sheath # Or specific attribute from your geom
        #j_cathode = geom.j_cathode_z_sheath
        # Note: Ensure these match the user snippet's definition or pass them in
        # If geom doesn't store them exactly as user snippet, reconstruct them here:
        rmax_injection = geom.rmax_cathode
        i_cathode = (np.arange(geom.Nr)[geom.r <= rmax_injection]).astype(np.int32)
        j_cathode = ((int(geom.zmax_cathode/geom.dz)+1)*np.ones_like(i_cathode)).astype(np.int32)
        
        # Assemble 2D Matrix (Insulating Internal, Bohm External)
        aP, aE, aW, aN, aS, b = electron_fluid_helper.assemble_ne_diffusion_FV(
            ne_star,
            Da_par, Da_perp,
            geom.mask, geom.dr, geom.dz, geom.r, dt,
            self.Te_grid, ion_fluid.m_i,
            i_cathode, j_cathode, hybrid_pic.Jiz_grid, 
            delta_sheath=delta_sheath, 
            closed_top=closed_top
        )
        
        # Solve
        ne_final = electron_fluid_helper.solve_direct_fast(
            self.Nr, self.Nz, aP, aE, aW, aN, aS, b
        )
        
        self.ne_grid = np.maximum(ne_final, self.ne_floor)
        
        # 2. Calculate Boundary Recycling Delta
        dne_boundary = electron_fluid_helper.compute_boundary_recycling_source(
            self.ne_grid,
            self.Te_grid,
            ion_fluid.m_i,
            geom.mask,
            geom.dr, geom.dz, geom.r, dt,
            i_cathode, j_cathode, hybrid_pic.Jiz_grid, 
            delta_sheath=delta_sheath,
            closed_top=closed_top
        )
        
        return dne_chem, dne_boundary
    
    def update_Te_implicit(self, geom, hybrid_pic, neutral_fluid, ion_fluid, 
                           Q_Joule_e_grid, 
                           delta_E_eV_ionization, dt, 
                           chem_T_array, chem_k_array,
                           q_RF_grid=None,
                           closed_top=False,
                           delta_sheath=1.0,
                           include_advection=False,
                           include_Bohm=False,
                           speed_limit=5e5):
        """
        Implicit update for Electron Temperature.
        Allows large timesteps by splitting Local Physics and Global Transport.
        """
        
        mi = ion_fluid.m_i
        mn = neutral_fluid.mass
        
        # A: Local Physics (Heating + Collisions + Ionization Cost)
        # ----------------------------------------------------------------
        # Ionization Energy Cost
        E_cost_J = delta_E_eV_ionization * constants.q_e 
        
        if q_RF_grid is not None:
            Q_Joule_e_grid += q_RF_grid

        #Q_Joule_e_grid[geom.i_cathode_z_sheath, geom.j_cathode_z_sheath+1] = 0.0
        electron_fluid_helper.update_Te_local_physics(
            self.Te_grid,
            self.ne_grid,
            neutral_fluid.nn_grid,
            neutral_fluid.T_n_grid,
            ion_fluid.Ti_grid,
            self.nu_en_grid,
            self.nu_ei_grid,
            Q_Joule_e_grid,
            E_cost_J,
            dt,
            mi,
            mn,
            geom.mask,
            self.Te_floor,
            chem_T_array,
            chem_k_array
        )
        
        # B: Global Transport (Implicit Diffusion)
        # ----------------------------------------------------------------
        r_coords = geom.r
        
        # Update thermal conductivities
        self.set_kappa(hybrid_pic)
        if(include_Bohm):
            kappa_perp_Bohm = (1/16.0)*self.ne_grid*self.Te_grid*(constants.kb)**2/(constants.q_e*hybrid_pic.Bmag_grid)
            self.kappa_perp_grid+=kappa_perp_Bohm

        ur_clipped = None
        uz_clipped = None

        if(include_advection):
            ur = np.copy(self.uer_grid)
            uz = np.copy(self.uez_grid)

            u_mag = np.sqrt(ur**2 + uz**2) + 1e-12 # avoid div/0
            scale_factor = np.minimum(1.0, speed_limit / u_mag)

            ur_clipped = ur * scale_factor
            uz_clipped = uz * scale_factor
        
        self.Te_grid = electron_fluid_helper.solve_Te_diffusion_direct(
            self.Te_grid,
            self.ne_grid,
            self.kappa_parallel_grid,
            self.kappa_perp_grid,
            hybrid_pic.br_grid,
            hybrid_pic.bz_grid,
            geom.mask,
            geom.dr,
            geom.dz,
            r_coords,
            dt,
            ion_fluid.m_i, # Use ion mass for Bohm speed
            self.Te_floor,
            geom.i_cathode_r,
            geom.j_cathode_r,
            geom.i_cathode_z_sheath,
            geom.j_cathode_z_sheath,
            ur_clipped, uz_clipped,
            closed_top=closed_top,
            delta_sheath=delta_sheath,
            include_advection=include_advection,
        )

        
        # 5. Boundary Conditions (Enforce Dirichlet if any)
        self.Te_grid[geom.cathode_mask] = geom.temperature_cathode
        self.Te_grid[geom.anode1_mask] = geom.temperature_anode
        self.Te_grid[geom.anode2_mask] = geom.temperature_anode
        self.apply_boundary_conditions(closed_top)


    def apply_boundary_conditions(self, closed_top=False):
        
        # Axis of symmetry (r=0): dTe/dr = 0
        self.Te_grid[0, :] = self.Te_grid[1, :]