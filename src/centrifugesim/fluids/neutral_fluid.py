import numpy as np

from centrifugesim.fluids import neutral_fluid_helper
from centrifugesim.geometry.geometry import Geometry

from centrifugesim import constants

class NeutralFluidContainer:
    """
    """
    def __init__(self, geom:Geometry, species_list, nn_floor, mass, name, kind, Tn0=0.0, alpha=1.0, min_T_mu_calc=2000.0, T_wall=300.0):

        self.name = name

        self.geom = geom

        self.Nr = geom.Nr
        self.Nz = geom.Nz

        self.nn_floor = nn_floor
        self.mass = mass

        self.alpha = alpha # Accommodation coefficient for wall BCs (Temperature)
        self.min_T_mu_calc = min_T_mu_calc # minimum T used to calculate viscosity

        self.kind = kind
        if(self.kind=='monatomic'):
            gamma = 5/3.0
        elif(self.kind=='diatomic'):
            gamma = 7/5.0

        self.gamma = gamma
        self.Rgas_over_m = constants.kb/self.mass # J/(kg·K) 
        self.c_v = self.Rgas_over_m/(gamma - 1.0) # J/(kg·K)
        self.cp  = self.c_v + self.Rgas_over_m # J/(kg·K)

        self.T_wall = T_wall  # Default wall temperature in K

        # For ground/excited states
        self.str_states_list = species_list.copy()
        self.list_nn_grid = [np.zeros((self.Nr, self.Nz)).astype(np.float64) for _ in species_list]
        self.list_N_states = np.zeros(len(species_list)).astype(np.float64)
        self.list_max_nn_states = np.zeros(len(species_list)).astype(np.float64)

        self.nn_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.rho_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.p_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)

        self.un_r_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.un_theta_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.un_z_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        
        self.T_n_grid = (Tn0*np.ones((self.Nr, self.Nz))).astype(np.float64)
        self.mu_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.kappa_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)

        self.fluid = (geom.mask.astype(np.uint8)).copy()
        self.mask_rho = self.fluid.copy()
        self.mask_vel = self.fluid.copy()
        self.build_domain_masks()
        
        self.face_r, self.face_z = neutral_fluid_helper.build_face_masks(self.fluid)

        self.i_bc_list, self.j_bc_list = geom.i_bc_list.copy(), geom.j_bc_list.copy()

        print("initialized")
        print(self.str_states_list)
        print()

    def build_domain_masks(self):
        """
        Builds two masks:
        1. mask_rho: Standard fluid mask (1=Fluid, 0=Solid). Used for P, T, Rho.
        2. mask_vel: Shrunk mask (1=Interior Fluid, 0=Wall/Solid). Used for u_r, u_z, u_t.
        """       
        Nr, Nz = self.mask_rho.shape
        
        # Loop over domain (excluding boundaries r=0, r=R, z=0, z=L for safety)
        # We check neighbors to identify "Wall Nodes"
        for i in range(1, Nr-1):
            for k in range(1, Nz-1):
                if self.mask_rho[i, k] == 1:
                    is_wall_node = False
                    # Check Von Neumann neighbors (Up, Down, Left, Right)
                    if (self.mask_rho[i+1, k] == 0) or (self.mask_rho[i-1, k] == 0):
                        is_wall_node = True
                    elif (self.mask_rho[i, k+1] == 0) or (self.mask_rho[i, k-1] == 0):
                        is_wall_node = True
                    
                    if is_wall_node:
                        self.mask_vel[i, k] = 0 # Lock velocity here

    def initialize_state(self, state_name, value):
        idx = self.str_states_list.index(state_name)
        self.list_nn_grid[idx][:,:] = value

    def compute_nn_grid_from_states(self):
        self.nn_grid[:,:] = 0.0
        for i, species in enumerate(self.str_states_list):
            self.nn_grid[:,:] += self.list_nn_grid[i][:,:]

    def compute_total_N_states(self, geom):
        for i in range(len(self.str_states_list)):
            self.list_N_states[i] = np.sum(self.list_nn_grid[i][geom.mask==1] * geom.volume_field[geom.mask==1])

    def compute_max_nn_states(self, geom):
        for i in range(len(self.str_states_list)):
            self.list_max_nn_states[i] = np.max(self.list_nn_grid[i][geom.mask==1])

    def update_rho(self):
        self.rho_grid[self.fluid==1] = self.mass*self.nn_grid[self.fluid==1]

    def update_nn(self):
        self.nn_grid[self.fluid==1] = self.rho_grid[self.fluid==1]/self.mass

    def update_p(self):
        self.p_grid[self.fluid==1] = self.rho_grid[self.fluid==1] * self.Rgas_over_m * self.T_n_grid[self.fluid==1]
        self.p_grid[:,-1] = self.p_grid[:,-3]
        self.p_grid[:, 0] = self.p_grid[:, 2]
        self.p_grid[-1, :] = self.p_grid[-3, :]

    def compute_sound_speed(self, Tfield):
        return np.sqrt(self.gamma * self.Rgas_over_m * Tfield)

    def get_knudsen_number_field(self, geom, L_char=None):
        """
        Calculates the local Knudsen number field grid.
        Kn = lambda / L_char
        
        Parameters
        ----------
        L_char : float, optional
            Characteristic length (e.g. R_max). If None, uses geom.rmax.
        
        Returns
        -------
        Kn_grid : np.ndarray
        """
        if L_char is None:
            L_char = geom.Nr * geom.dr # Approx Rmax or use geom.rmax if available
            if hasattr(geom, 'rmax'):
                L_char = geom.rmax

        # --- 1. Determine Sigma (Collision Diameter) ---
        # Default fallback
        sigma = 3.4e-10 
        
        # Try to find species parameters in the helper's DB
        found = False
        if self.name in neutral_fluid_helper._LJ_DB:
            # _LJ_DB structure is {name: (sigma, eps, kind)}
            sigma = neutral_fluid_helper._LJ_DB[self.name][0]
            found = True
        
        if not found:
            print(f"Warning: Could not find Lennard-Jones parameters for Kn calc. Using default sigma={sigma:.2e} m")

        # --- 2. Compute Grid ---
        Kn_grid = np.zeros((self.Nr, self.Nz), dtype=np.float64)
        
        neutral_fluid_helper.compute_knudsen_field(
            geom.mask,
            self.T_n_grid, 
            self.p_grid, 
            sigma, 
            L_char, 
            constants.kb, 
            Kn_grid
        )
        
        return Kn_grid

    # --------------------------- Boundary conditions -------------------------

    def apply_bc_isothermal_prev(self, closed_top=False):
        Nr, Nz = self.rho_grid.shape

        # --- Axis r = 0 (i = 0): regularity ---
        self.un_r_grid[0,:]  = 0.0                 # odd
        self.un_theta_grid[0,:]  = 0.0                 # odd
        self.un_z_grid[0,:]  = self.un_z_grid[1,:]             # ∂r uz = 0

        # --- Radial wall r = R (i = Nr-1): no-slip, impermeable ---
        self.un_r_grid[-1,:] = 0.0
        self.un_theta_grid[-1,:] = 0.0
        self.un_z_grid[-1,:] = 0.0

        # --- Bottom plate z = 0 (k = 0): no-slip, impermeable ---
        self.un_r_grid[:,0] = 0.0
        self.un_theta_grid[:,0] = 0.0
        self.un_z_grid[:,0] = 0.0

        # --- Top plate z = L (k = Nz-1) ---
        if closed_top:
            self.un_r_grid[:,-1] = 0.0         # No-slip
            self.un_theta_grid[:,-1] = 0.0     # No-slip
            self.un_z_grid[:,-1] = 0.0         # Impermeable
        else:
            self.un_r_grid[:,-1] = self.un_r_grid[:,-2]         # Slip (Neumann)
            self.un_theta_grid[:,-1] = self.un_theta_grid[:,-2] # Slip (Neumann)
            self.un_z_grid[:,-1] = 0.0                          # Impermeable

        # no slip solid surfaces inside domain
        self.un_r_grid[self.i_bc_list, self.j_bc_list] = 0.0
        self.un_theta_grid[self.i_bc_list, self.j_bc_list] = 0.0
        self.un_z_grid[self.i_bc_list, self.j_bc_list] = 0.0

    def apply_bc_isothermal(self, closed_top=False):

        # --- Axis r = 0 (i = 0): regularity ---
        self.un_r_grid[0,:]      = 0.0                 
        self.un_theta_grid[0,:]  = 0.0                 
        self.un_z_grid[0,:]      = self.un_z_grid[1,:] # Neumann (d/dr = 0)

        # --- Radial wall r = R (i = Nr-1): No-Slip + Impermeable ---
        self.un_r_grid[-1,:]     = -self.un_r_grid[-2,:]      # Impermeable
        self.un_theta_grid[-1,:] = -self.un_theta_grid[-2,:]  # No-Slip
        self.un_z_grid[-1,:]     = -self.un_z_grid[-2,:]      # No-Slip

        # --- Bottom plate z = 0 (k = 0): No-Slip + Impermeable ---
        self.un_r_grid[:,0]      = -self.un_r_grid[:,1]       # No-Slip
        self.un_theta_grid[:,0]  = -self.un_theta_grid[:,1]   # No-Slip
        self.un_z_grid[:,0]      = -self.un_z_grid[:,1]       # Impermeable

        # --- Top plate z = L (k = Nz-1) ---
        if closed_top:
            self.un_r_grid[:,-1]     = -self.un_r_grid[:,-2]      # No-Slip
            self.un_theta_grid[:,-1] = -self.un_theta_grid[:,-2]  # No-Slip
            self.un_z_grid[:,-1]     = -self.un_z_grid[:,-2]      # Impermeable
        else:
            self.un_r_grid[:,-1]     = self.un_r_grid[:,-2]       # Slip (Neumann)
            self.un_theta_grid[:,-1] = self.un_theta_grid[:,-2]   # Slip (Neumann)
            self.un_z_grid[:,-1] = 0.0 

        # --- Internal Solids ---
        self.un_r_grid[self.i_bc_list, self.j_bc_list] = 0.0
        self.un_theta_grid[self.i_bc_list, self.j_bc_list] = 0.0
        self.un_z_grid[self.i_bc_list, self.j_bc_list] = 0.0

    def apply_bc_T(self, closed_top=False):
        Nr, Nz = self.T_n_grid.shape

        # Axis
        self.T_n_grid[0,:] = self.T_n_grid[1,:]       # r=0

        # --- Top Boundary ---
        if closed_top:
            self.T_n_grid[:,-1] = self.T_wall         # Fixed Wall Temp
        else:
            self.T_n_grid[:,-1] = self.T_n_grid[:,-2] # Adiabatic / Symmetry
        
        # Bottom Wall (z=0): Fixed Dirichlet
        self.T_n_grid[:,0] = self.T_wall 

        # Radial Wall (r=R): Fixed Dirichlet
        self.T_n_grid[-1,:] = self.T_wall 


    ###########################################################################################
    ### methods for Implicit navier stokes (no test)
    ###########################################################################################
    def advance_semi_implicit(self, geom, dt, c_iso, apply_bc_vel, apply_bc_temp, 
                              ion_fluid=None, electron_fluid=None, closed_top=False, max_iter_sor=750):
        """
        Advances the neutral fluid using Operator Splitting with HYDRO SUB-CYCLING.
        1. Sub-cycled Explicit Advection (Hydro + Energy) to handle acoustic CFL.
        2. Implicit Collisions (Source).
        3. Implicit Diffusion (Viscosity + Conduction).
        """
        r, dr, dz = geom.r, geom.dr, geom.dz

        # ==========================================================
        # STEP A: EXPLICIT ADVECTION (Hydro + Energy Transport)
        # ==========================================================
        # 1. Calculate stable timestep for hydro
        dt_stability_limit = neutral_fluid_helper.stable_adv_dt(
                                self.fluid,
                                geom.r, geom.dr, geom.dz,
                                self.un_r_grid,
                                self.un_z_grid,
                                c_iso,
                                safety=0.5)
        
        # 2. Determine number of sub-steps needed
        n_substeps = int(np.ceil(dt / dt_stability_limit))
        if n_substeps < 1: n_substeps = 1
                
        dt_sub = dt / n_substeps
        
        # 3. Run Sub-cycling Loop
        for s in range(n_substeps):
            
            # --- Store state U^n for RK2 ---
            rho0 = self.rho_grid.copy()
            ur0  = self.un_r_grid.copy()
            ut0  = self.un_theta_grid.copy()
            uz0  = self.un_z_grid.copy()
            T0   = self.T_n_grid.copy()

            # --- RK Stage 1 ---
            self.update_p()

            neutral_fluid_helper.step_advection_hydro(r, dr, dz, dt_sub,
                        self.rho_grid, self.un_r_grid, self.un_theta_grid, self.un_z_grid,
                        self.p_grid, c_iso, self.nn_floor*self.mass,
                        self.mask_rho, self.mask_vel, self.face_r, self.face_z)
            
            self.update_p()

            neutral_fluid_helper.step_advection_energy(r, dr, dz, dt_sub,
                        self.T_n_grid, self.rho_grid, self.un_r_grid, self.un_theta_grid, self.un_z_grid,
                        self.p_grid, self.c_v,
                        self.fluid, self.face_r, self.face_z)
            
            apply_bc_vel(closed_top=closed_top); apply_bc_temp(closed_top=closed_top)
            self.update_p() 

            # --- RK Stage 2 ---
            neutral_fluid_helper.step_advection_hydro(r, dr, dz, dt_sub,
                        self.rho_grid, self.un_r_grid, self.un_theta_grid, self.un_z_grid,
                        self.p_grid, c_iso, self.nn_floor*self.mass,
                        self.mask_rho, self.mask_vel, self.face_r, self.face_z)
            
            self.update_p()

            neutral_fluid_helper.step_advection_energy(r, dr, dz, dt_sub,
                        self.T_n_grid, self.rho_grid, self.un_r_grid, self.un_theta_grid, self.un_z_grid,
                        self.p_grid, self.c_v,
                        self.fluid, self.face_r, self.face_z)
            
            apply_bc_vel(closed_top=closed_top); apply_bc_temp(closed_top=closed_top)
            self.update_p()

            # --- Average (SSP-RK2) ---
            self.rho_grid[:]       = 0.5 * (rho0 + self.rho_grid)
            self.un_r_grid[:]      = 0.5 * (ur0  + self.un_r_grid)
            self.un_theta_grid[:]  = 0.5 * (ut0  + self.un_theta_grid)
            self.un_z_grid[:]      = 0.5 * (uz0  + self.un_z_grid)
            self.T_n_grid[:]       = 0.5 * (T0   + self.T_n_grid)
            
            apply_bc_vel(closed_top=closed_top); apply_bc_temp(closed_top=closed_top)
            self.update_p()

        # Update nn before collisions
        self.update_nn()
        self.nn_grid[:,0] = self.nn_grid[:,1]
        self.nn_grid[:,-1] = self.nn_grid[:,-2]
        self.nn_grid[-1,:] = self.nn_grid[-2,:]
        self.rho_grid[:,0] = self.rho_grid[:,1]
        self.rho_grid[:,-1] = self.rho_grid[:,-2]
        self.rho_grid[-1,:] = self.rho_grid[-2,:]

        # ==========================================================
        # STEP B: IMPLICIT COLLISIONS (Source) - ONCE PER GLOBAL STEP
        # ==========================================================
        if ion_fluid is not None:
             self.update_u_in_collisions_implicit(geom, ion_fluid, dt)
        
        if (ion_fluid is not None) and (electron_fluid is not None):
             self.update_temperature_collisions_implicit(ion_fluid, electron_fluid, geom, dt)

        apply_bc_vel(closed_top=closed_top); apply_bc_temp(closed_top=closed_top)
        self.update_p()
        self.update_nn()

        # ==========================================================
        # STEP C: IMPLICIT DIFFUSION (Sink) - ONCE PER GLOBAL STEP
        # ==========================================================
        mu_val, kappa_val, _ = neutral_fluid_helper.viscosity_and_conductivity(
             geom, self.T_n_grid, self.mass, species=self.name, kind=self.kind, min_T_mu_calc=self.min_T_mu_calc
        )
        self.mu_grid[:,:] = mu_val
        self.kappa_grid[:,:] = kappa_val

        # Viscous Heating (EXPLICIT)
        visc_safety_factor = 0.5
        dz_sq = geom.dz**2
        max_nu = np.max(self.mu_grid[geom.mask==1] / (self.rho_grid[geom.mask==1]))
        dt_heat_limit = visc_safety_factor * dz_sq / (max_nu)
        n_heat_steps = int(np.ceil(dt / dt_heat_limit))
        dt_heat_sub = dt / n_heat_steps

        # Sub-cycle
        for _ in range(n_heat_steps):
            neutral_fluid_helper.add_viscous_heating(
                self.T_n_grid, self.rho_grid,
                self.un_r_grid, self.un_theta_grid, self.un_z_grid,
                self.mu_grid, self.c_v,
                dr, dz, dt_heat_sub,
                self.fluid, self.face_r, self.face_z
            )
            apply_bc_temp(closed_top=closed_top)

        self.update_p()
        self.update_nn()

        # Viscosity
        mu_val, kappa_val, _ = neutral_fluid_helper.viscosity_and_conductivity(
             geom, self.T_n_grid, self.mass, species=self.name, kind=self.kind, min_T_mu_calc=self.min_T_mu_calc
        )
        self.mu_grid[:,:] = mu_val
        self.kappa_grid[:,:] = kappa_val

        # Thermal Conduction
        neutral_fluid_helper.solve_implicit_heat_sor_robin_anode(
            self.T_n_grid, self.nn_grid, self.mass, self.fluid, self.geom.anode_mask, self.geom.anode_T_field, self.kappa_grid, self.c_v,
            dt, dr, dz, self.T_wall, self.geom.temperature_cathode, self.geom.i_cath_max, self.geom.j_cath_max,
            max_iter=max_iter_sor, omega=1.8,
            closed_top=closed_top
        )

        apply_bc_temp(closed_top=closed_top)
        self.update_p()
        self.update_nn()

        # Update Transport Coeffs
        mu_val, kappa_val, _ = neutral_fluid_helper.viscosity_and_conductivity(
             geom, self.T_n_grid, self.mass, species=self.name, kind=self.kind, min_T_mu_calc=self.min_T_mu_calc
        )
        self.mu_grid[:,:] = mu_val
        self.kappa_grid[:,:] = kappa_val

        # Calculate Bulk Viscosity (Physical + Artificial)
        # mub_eff = mub + 0.02 * rho * c_iso * h
        
        # Grid spacing h
        h = min(geom.dr, geom.dz) 
        
        # Physical Bulk Viscosity (if any, typically 0 for monoatomic, but let's allow a placeholder)
        mub_physical = 0.0 
        
        # Artificial Bulk Viscosity
        mub_grid = mub_physical + 0.02 * self.rho_grid * c_iso * h
        
        # Mask solids
        mub_grid[self.fluid == 0] = 0.0

        # Compute Explicit Cross-Term Sources
        Sr, St, Sz = neutral_fluid_helper.compute_viscous_cross_terms(
            r, dr, dz, 
            self.un_r_grid, self.un_theta_grid, self.un_z_grid,
            self.mu_grid,
            mub_grid,
            self.face_r, self.face_z
        )

        # Bottom and Top Walls (z boundaries)
        Sr[:, 1] = 0.0;  Sr[:, -2] = 0.0
        St[:, 1] = 0.0;  St[:, -2] = 0.0
        Sz[:, 1] = 0.0;  Sz[:, -2] = 0.0
        
        # Radial Wall (r boundary)
        Sr[-2, :] = 0.0
        St[-2, :] = 0.0
        Sz[-2, :] = 0.0

        # Solve Azimuthal 
        neutral_fluid_helper.solve_implicit_viscosity_sor(
            self.un_theta_grid, self.nn_grid, self.mass, self.mask_vel, self.mu_grid, St,
            dt, dr, dz, max_iter=max_iter_sor, omega=1.8, closed_top=closed_top
        )

        # Solve Radial
        neutral_fluid_helper.solve_implicit_viscosity_r_sor(
            self.un_r_grid, self.nn_grid, self.mass, self.mask_vel, self.mu_grid, Sr,
            dt, dr, dz, max_iter=max_iter_sor, omega=1.8, closed_top=closed_top
        )
        
        # Solve Axial
        neutral_fluid_helper.solve_implicit_viscosity_z_sor(
            self.un_z_grid, self.nn_grid, self.mass, self.mask_vel, self.mu_grid, Sz,
            dt, dr, dz, max_iter=max_iter_sor, omega=1.8
        )

        apply_bc_vel(closed_top=closed_top)
        
    ###########################################################################################
    ### Methods used when fluid ions are on (not kinetic ions)
    ###########################################################################################

    # To test temperature update implicit due to collisions
    def update_temperature_collisions_implicit(self, ion_fluid, electron_fluid, geom, dt):
        """
        Updates T_gas implicitly based on elastic collisions with Electrons and Ions.
        """
        # Apply Thermal Relaxation (Implicit)
        neutral_fluid_helper.update_neutral_temperature_implicit(
            self.T_n_grid,          # In/Out: Updated in place
            electron_fluid.Te_grid,
            ion_fluid.Ti_grid,
            electron_fluid.ne_grid,
            self.nn_grid,
            electron_fluid.nu_en_grid,
            ion_fluid.nu_i_grid,    # Ion-neutral collision freq
            constants.m_e,
            ion_fluid.m_i,
            self.mass,              # Neutral mass
            dt,
            geom.mask,
            self.c_v
        )

    def update_u_in_collisions_implicit(self, geom, ion_fluid, dt):
        """
        Updates Neutral velocity due to collisions with Ions using an implicit formula.
        Used when fluid ions are on.
        """

        # --- 2. Update ut ---
        neutral_fluid_helper.update_neutral_vtheta_implicit_source(
            self.un_theta_grid,
            ion_fluid.vi_theta_grid,
            ion_fluid.ni_grid,
            ion_fluid.nu_i_grid,
            ion_fluid.m_i,
            self.nn_grid,
            self.mass,
            dt,
            self.mask_vel
        )

        # --- 2. Update ur ---
        neutral_fluid_helper.update_neutral_vr_implicit_source(
            self.un_r_grid,
            ion_fluid.vi_r_grid,
            ion_fluid.ni_grid,
            ion_fluid.nu_i_grid,
            ion_fluid.m_i,
            self.nn_grid,
            self.mass,
            dt,
            self.mask_vel
        )

        # --- 3. Update uz ---
        neutral_fluid_helper.update_neutral_vz_implicit_source(
            self.un_z_grid,
            ion_fluid.vi_z_grid,
            ion_fluid.ni_grid,
            ion_fluid.nu_i_grid,
            ion_fluid.m_i,
            self.nn_grid,
            self.mass,
            dt,
            self.mask_vel
        )