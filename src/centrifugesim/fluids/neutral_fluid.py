import numpy as np

from centrifugesim.fluids import neutral_fluid_helper
from centrifugesim.geometry.geometry import Geometry

from centrifugesim import constants

class NeutralFluidContainer:
    """
    """
    def __init__(self, geom:Geometry, species_list, nn_floor, mass, name, kind, Tn0=0.0, alpha=1.0):

        self.name = name

        self.geom = geom

        self.Nr = geom.Nr
        self.Nz = geom.Nz

        self.nn_floor = nn_floor
        self.mass = mass

        self.alpha = alpha # Accommodation coefficient for wall BCs (Temperature)

        self.kind = kind
        if(self.kind=='monatomic'):
            gamma = 5/3.0
        elif(self.kind=='diatomic'):
            gamma = 7/5.0

        self.gamma = gamma
        self.Rgas_over_m = constants.kb/self.mass # J/(kg·K) 
        self.c_v = self.Rgas_over_m/(gamma - 1.0) # J/(kg·K)
        self.cp  = self.c_v + self.Rgas_over_m # J/(kg·K)

        self.T_wall = 300.0  # Default wall temperature in K

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
        self.p_grid[-1,:] = self.p_grid[-2,:]  # Does not change anything, just for visualization purposes

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

    def update_u_in_collisions(self, geom, ni_grid, mi,
                                    ui_r, ui_t, ui_z,
                                    nu_in, Ti, dt):
        """
        Updates Neutral velocity due to collisions with Ions. Explicit equation and it also updates gas temperature
        Used when kinetic ions where on (as fluid counterpart of the MCC module used on kinetic ions).
        Do not use this when fluid ions are on.
        """

        dtnu_max = nu_in[geom.mask==1].max()*dt
        if(round(dtnu_max,2)>0.1):
            print("dt*nu_in.max() > 0.1 !", dtnu_max)

        un_r_new, un_t_new, un_z_new, Tn_new = neutral_fluid_helper.update_u_in_collisions(
                                        geom.mask, ni_grid*mi, self.rho_grid,
                                        ui_r, ui_t, ui_z,
                                        self.un_r_grid, self.un_theta_grid, self.un_z_grid,
                                        nu_in, self.nn_floor*self.mass,
                                        self.T_n_grid, Ti, self.c_v, dt)
                                        
        self.un_r_grid[geom.mask==1]        = np.copy(un_r_new[geom.mask==1])
        self.un_theta_grid[geom.mask==1]    = np.copy(un_t_new[geom.mask==1])
        self.un_z_grid[geom.mask==1]        = np.copy(un_z_new[geom.mask==1])
        self.T_n_grid[geom.mask==1]         = np.copy(Tn_new[geom.mask==1])

    def advance_with_T_ssp_rk2(self,
                            geom, dt,
                            c_iso,
                            apply_bc_vel, apply_bc_temp):

        r, dr, dz = geom.r, geom.dr, geom.dz

        # stage-0
        rho0 = self.rho_grid.copy()
        ur0 = self.un_r_grid.copy()
        ut0 = self.un_theta_grid.copy()
        uz0 = self.un_z_grid.copy()
        T0 = self.T_n_grid.copy()

        mub = np.zeros_like(T0)

        # ---------- stage 1: momentum + continuity
        neutral_fluid_helper.step_isothermal(r, dr, dz, dt,
                    self.rho_grid, self.un_r_grid, self.un_theta_grid, self.un_z_grid,
                    self.p_grid, self.mu_grid, mub,
                    c_iso,
                    fluid=self.fluid, face_r=self.face_r, face_z=self.face_z)

        # stresses for energy (use your masked version if you added one)
        tau_rr = np.zeros_like(rho0); tau_tt = np.zeros_like(rho0); tau_zz = np.zeros_like(rho0)
        tau_rz = np.zeros_like(rho0); tau_rt = np.zeros_like(rho0); tau_tz = np.zeros_like(rho0)
        divu_d = np.zeros_like(rho0)
        neutral_fluid_helper.stresses(r, self.un_r_grid, self.un_theta_grid, self.un_z_grid, self.mu_grid, mub, dr, dz,
            tau_rr, tau_tt, tau_zz, tau_rz, tau_rt, tau_tz, divu_d,
            fluid=self.fluid, face_r=self.face_r, face_z=self.face_z)

        neutral_fluid_helper.step_temperature_masked(r, dr, dz, dt,
                                self.T_n_grid, self.rho_grid, self.un_r_grid, self.un_theta_grid, self.un_z_grid,
                                self.p_grid, self.kappa_grid,
                                tau_rr, tau_tt, tau_zz, tau_rz, tau_rt, tau_tz,
                                self.c_v,
                                fluid=self.fluid, face_r=self.face_r, face_z=self.face_z)

        # EOS + BCs + projection
        self.p_grid[:,:] = self.rho_grid * self.Rgas_over_m * self.T_n_grid

        apply_bc_vel(); apply_bc_temp()
        #if self.fluid is not None:
        #    neutral_fluid_helper.apply_solid_mask_inplace_T(self.fluid, self.T_n_grid)

        # ---------- stage 2: repeat
        neutral_fluid_helper.step_isothermal(r, dr, dz, dt,
                    self.rho_grid, self.un_r_grid, self.un_theta_grid, self.un_z_grid,
                    self.p_grid, self.mu_grid, mub,
                    c_iso,
                    fluid=self.fluid, face_r=self.face_r, face_z=self.face_z)

        tau_rr.fill(0.0); tau_tt.fill(0.0); tau_zz.fill(0.0)
        tau_rz.fill(0.0); tau_rt.fill(0.0); tau_tz.fill(0.0); divu_d.fill(0.0)
        neutral_fluid_helper.stresses(r, self.un_r_grid, self.un_theta_grid, self.un_z_grid, 
            self.mu_grid, mub, dr, dz,
            tau_rr, tau_tt, tau_zz, tau_rz, tau_rt, tau_tz, divu_d,
            fluid=self.fluid, face_r=self.face_r, face_z=self.face_z)

        neutral_fluid_helper.step_temperature_masked(r, dr, dz, dt,
                                self.T_n_grid, self.rho_grid,
                                self.un_r_grid, self.un_theta_grid, self.un_z_grid,
                                self.p_grid, self.kappa_grid,
                                tau_rr, tau_tt, tau_zz, tau_rz, tau_rt, tau_tz,
                                self.c_v,
                                fluid=self.fluid, face_r=self.face_r, face_z=self.face_z)

        self.p_grid[:,:] = self.rho_grid * self.Rgas_over_m * self.T_n_grid

        apply_bc_vel(); apply_bc_temp()

        # ---------- combine
        self.rho_grid[:,:]          = 0.5*(rho0 + self.rho_grid)
        self.un_r_grid [:,:]        = 0.5*(ur0  + self.un_r_grid)
        self.un_theta_grid [:,:]    = 0.5*(ut0  + self.un_theta_grid)
        self.un_z_grid [:,:]        = 0.5*(uz0  + self.un_z_grid)
        self.T_n_grid[:,:]          = 0.5*(T0   + self.T_n_grid)
        self.p_grid[:,:] = self.rho_grid * self.Rgas_over_m * self.T_n_grid

        apply_bc_vel(); apply_bc_temp()

    # --------------------------- Boundary conditions -------------------------

    def apply_bc_isothermal(self):
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

        # --- Top plate z = L (k = Nz-1): no-slip, impermeable ---
        self.un_r_grid[:,-1] = self.un_r_grid[:,-2]
        self.un_theta_grid[:,-1] = self.un_theta_grid[:,-2]
        self.un_z_grid[:,-1] = 0.0

        # no slip solid surfaces inside domain
        self.un_r_grid[self.i_bc_list, self.j_bc_list] = 0.0
        self.un_theta_grid[self.i_bc_list, self.j_bc_list] = 0.0
        self.un_z_grid[self.i_bc_list, self.j_bc_list] = 0.0


    def apply_bc_T(self):
        Nr, Nz = self.T_n_grid.shape

        # 1. Symmetry Boundaries
        self.T_n_grid[0,:] = self.T_n_grid[1,:]       # r=0
        self.T_n_grid[:,-1] = self.T_n_grid[:,-2]     # z=L
        
        # 2. Bottom Wall (z=0): Fixed Dirichlet
        self.T_n_grid[:,0] = self.T_wall 

        # ---------------------------------------------------------
        # 2. Radial Wall (r=R): Unified Kinetic Theory BC
        # ---------------------------------------------------------
        # We MUST update Nr-1 here to prevent T from crashing to 300K
        
        mask_fluid = (self.geom.mask[-1,:] == 1)
        
        # A. Gather Neighbor Properties (approaching the wall)
        T_gas   = self.T_n_grid[-2, mask_fluid]
        rho_gas = self.rho_grid[-2, mask_fluid]
        mu_gas  = self.mu_grid[-2, mask_fluid]
        kap_gas = self.kappa_grid[-2, mask_fluid]
        
        # B. Physics Constants
        kb = constants.kb
        m  = self.mass
        gamma = self.gamma
        cp = self.cp
        
        # C. Calculate Mean Free Path (lambda)
        # v_th = sqrt(8 k T / pi m)
        v_th = np.sqrt(8.0 * kb * T_gas / (np.pi * m))
        
        # lambda = 2 * mu / (rho * v_th)
        rho_safe = rho_gas
        lam = 2.0 * mu_gas / (rho_safe * v_th)
        
        # D. Calculate Jump Length (L_jump)
        # Pr = Cp * mu / k
        Pr = cp * mu_gas / np.maximum(kap_gas, 1e-15)
        
        # Coefficients
        # Use self.alpha (Accommodation Coefficient)
        # alpha ~ 0.5 allows significant slip (T_wall > 300K)
        alpha = self.alpha
        coeff_alpha = (2.0 - alpha) / alpha
        coeff_gamma = (2.0 * gamma) / (gamma + 1.0)
        
        L_jump = coeff_alpha * coeff_gamma * (lam / Pr)
        L_jump = np.maximum(L_jump, 1e-12)
        
        # E. Calculate Effective h
        h_eff = kap_gas / L_jump

        # F. Apply Robin BC
        dr = self.geom.dr
        
        # Robin Formula: Balance Fluxes
        # T_wall = (k*T_inner + h*dr*T_fixed) / (k + h*dr)
        numerator   = (kap_gas * T_gas) + (h_eff * dr * self.T_wall)
        denominator = kap_gas + (h_eff * dr)
        
        self.T_n_grid[-1, mask_fluid] = numerator / denominator


    def apply_bc_density_compatible(self):
        """
        1. Enforces Radial Equilibrium (dP/dr = 0) at the wall.
        2. CONSERVES MASS by redistributing the correction to the neighbor node.
        """
        # --- A. Geometry & Volumes ---
        # Volume of Wall Cell (Nr-1) and Neighbor Cell (Nr-2)
        vol_wall  = self.geom.volume_field[-1, :]
        vol_neigh = self.geom.volume_field[-2, :]
        
        # --- B. Current State ---
        rho_wall  = self.rho_grid[-1, :]
        rho_neigh = self.rho_grid[-2, :]
        
        T_wall  = np.maximum(self.T_n_grid[-1, :], 1.0)
        T_neigh = self.T_n_grid[-2, :]

        # --- C. Calculate Constraint (Mass Preservation) ---
        # We must preserve the total mass currently in the last two cells.
        mass_skin_old = (rho_wall * vol_wall) + (rho_neigh * vol_neigh)
        
        # --- D. Calculate Target (Pressure Equilibrium) ---
        # We want P_wall = P_neigh  =>  rho_wall_new * T_wall = rho_neigh_new * T_neigh
        # Implies: rho_wall_new = rho_neigh_new * (T_neigh / T_wall)
        ratio = T_neigh / T_wall
        
        # --- E. Solve System of Equations ---
        # Eq 1 (Mass): rho_neigh_new * V_n + rho_wall_new * V_w = Mass_old
        # Eq 2 (Phys): rho_wall_new = rho_neigh_new * ratio
        #
        # Substitute: rho_neigh_new * V_n + (rho_neigh_new * ratio) * V_w = Mass_old
        #             rho_neigh_new * (V_n + ratio * V_w) = Mass_old
        
        rho_neigh_new = mass_skin_old / (vol_neigh + ratio * vol_wall)
        rho_wall_new  = rho_neigh_new * ratio
        
        # --- F. Apply ---
        mask_fluid = (self.geom.mask[-1,:] == 1)
        self.rho_grid[-1, mask_fluid] = rho_wall_new[mask_fluid]
        self.rho_grid[-2, mask_fluid] = rho_neigh_new[mask_fluid]
        
        # Update auxiliary number density
        self.nn_grid[-1, :] = self.rho_grid[-1, :] / self.mass
        self.nn_grid[-2, :] = self.rho_grid[-2, :] / self.mass


    ###########################################################################################
    ### methods for Implicit navier stokes (no test)
    ###########################################################################################
    def advance_semi_implicit(self, geom, dt, c_iso, apply_bc_vel, apply_bc_temp, 
                              ion_fluid=None, electron_fluid=None):
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
            neutral_fluid_helper.step_advection_hydro(r, dr, dz, dt_sub,
                        self.rho_grid, self.un_r_grid, self.un_theta_grid, self.un_z_grid,
                        self.p_grid, c_iso, self.nn_floor*self.mass,
                        self.mask_rho, self.mask_vel, self.face_r, self.face_z)
            
            neutral_fluid_helper.step_advection_energy(r, dr, dz, dt_sub,
                        self.T_n_grid, self.rho_grid, self.un_r_grid, self.un_theta_grid, self.un_z_grid,
                        self.p_grid, self.c_v,
                        self.fluid, self.face_r, self.face_z)
            
            apply_bc_vel(); apply_bc_temp()
            self.update_p() 

            # --- RK Stage 2 ---
            neutral_fluid_helper.step_advection_hydro(r, dr, dz, dt_sub,
                        self.rho_grid, self.un_r_grid, self.un_theta_grid, self.un_z_grid,
                        self.p_grid, c_iso, self.nn_floor*self.mass,
                        self.mask_rho, self.mask_vel, self.face_r, self.face_z)
            
            neutral_fluid_helper.step_advection_energy(r, dr, dz, dt_sub,
                        self.T_n_grid, self.rho_grid, self.un_r_grid, self.un_theta_grid, self.un_z_grid,
                        self.p_grid, self.c_v,
                        self.fluid, self.face_r, self.face_z)

            # --- Average (SSP-RK2) ---
            self.rho_grid[:]       = 0.5 * (rho0 + self.rho_grid)
            self.un_r_grid[:]      = 0.5 * (ur0  + self.un_r_grid)
            self.un_theta_grid[:]  = 0.5 * (ut0  + self.un_theta_grid)
            self.un_z_grid[:]      = 0.5 * (uz0  + self.un_z_grid)
            self.T_n_grid[:]       = 0.5 * (T0   + self.T_n_grid)
            
            apply_bc_vel(); apply_bc_temp()
            self.update_p()

        # Update nn before collisions
        self.update_nn()

        # ==========================================================
        # STEP B: IMPLICIT COLLISIONS (Source) - ONCE PER GLOBAL STEP
        # ==========================================================
        if ion_fluid is not None:
             self.update_u_in_collisions_implicit(geom, ion_fluid, dt)
        
        if (ion_fluid is not None) and (electron_fluid is not None):
             self.update_temperature_collisions_implicit(ion_fluid, electron_fluid, geom, dt)

        apply_bc_vel(); apply_bc_temp()
        self.update_p()

        # ==========================================================
        # STEP C: IMPLICIT DIFFUSION (Sink) - ONCE PER GLOBAL STEP
        # ==========================================================
        mu_val, kappa_val, _ = neutral_fluid_helper.viscosity_and_conductivity(
             geom, self.T_n_grid, self.mass, species=self.name, kind=self.kind
        )
        self.mu_grid[:,:] = mu_val
        self.kappa_grid[:,:] = kappa_val

        # Viscous Heating (EXPLICIT)
        neutral_fluid_helper.add_viscous_heating(
            self.T_n_grid, self.rho_grid,
            self.un_r_grid, self.un_theta_grid, self.un_z_grid,
            self.mu_grid, self.c_v,
            dr, dz, dt, 
            self.fluid, self.face_r, self.face_z
        )

        # Thermal Conduction
        neutral_fluid_helper.solve_implicit_heat_sor(
            self.T_n_grid, self.nn_grid, self.mass, self.fluid, self.kappa_grid, self.c_v,
            dt, dr, dz, max_iter=400, omega=1.8,
            mu_grid=self.mu_grid,
            rho_grid=self.rho_grid,
            mass=self.mass,
            gamma=self.gamma,
            cp=self.cp,
            T_wall_fixed=self.T_wall,
            alpha=self.alpha
        )

        # Viscosity
        mu_val, kappa_val, _ = neutral_fluid_helper.viscosity_and_conductivity(
             geom, self.T_n_grid, self.mass, species=self.name, kind=self.kind
        )
        self.mu_grid[:,:] = mu_val
        self.kappa_grid[:,:] = kappa_val

        # Viscosity
        # Note: should I combine all these kernel calls into one to save some time?
        neutral_fluid_helper.solve_implicit_viscosity_sor(
            self.un_theta_grid, self.nn_grid, self.mass, self.mask_vel, self.mu_grid,
            dt, dr, dz, max_iter=400, omega=1.8
        )
        neutral_fluid_helper.solve_implicit_viscosity_r_sor(
            self.un_r_grid, self.nn_grid, self.mass, self.mask_vel, self.mu_grid,
            dt, dr, dz, max_iter=400, omega=1.8
        )
        neutral_fluid_helper.solve_implicit_viscosity_z_sor(
            self.un_z_grid, self.nn_grid, self.mass, self.mask_vel, self.mu_grid,
            dt, dr, dz, max_iter=400, omega=1.8
        )

        apply_bc_vel()
        self.update_p()

        
    ###########################################################################################
    ### Methods used when fluid ions are on - test functions for implicit, not final (not kinetic ions)
    ###########################################################################################

    # To test temperature update implicit due to collisions
    def update_temperature_collisions_implicit(self, ion_fluid, electron_fluid, geom, dt):
        """
        Updates T_gas implicitly based on elastic collisions with Electrons and Ions.
        """
        # Add Frictional Heating due to ion-neutral slip (Explicit)
        neutral_fluid_helper.add_ion_neutral_frictional_heating(
            self.T_n_grid,
            self.un_theta_grid, ion_fluid.vi_theta_grid,
            ion_fluid.ni_grid, self.nn_grid, ion_fluid.nu_i_grid,
            ion_fluid.m_i, self.mass, self.c_v, dt, geom.mask
        )

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
        Do not use this when kinetic ions are on.
        """
        neutral_fluid_helper.update_neutral_vtheta_implicit_source(
            self.un_theta_grid,
            ion_fluid.vi_theta_grid,
            ion_fluid.ni_grid,
            ion_fluid.nu_i_grid,
            ion_fluid.m_i,
            self.nn_grid,
            self.mass,
            dt,
            geom.mask
        )

    def update_diffusion_implicit(self, geom, dt):
        """
        Applies implicit Viscosity and Thermal Conduction using 
        pre-calculated mu_grid and kappa_grid fields.
        """
        # 1. Update Viscosity (Momentum Sink)
        neutral_fluid_helper.solve_implicit_viscosity_sor(
            self.un_theta_grid,
            self.nn_grid,
            self.mass,
            self.fluid,      # The mask
            self.mu_grid,
            dt,
            geom.dr,
            geom.dz,
            max_iter=20,
            omega=1.4
        )
        
        # 2. Update Thermal Conduction (Energy Sink)
        neutral_fluid_helper.solve_implicit_heat_sor(
            self.T_n_grid,
            self.nn_grid,
            self.mass,
            self.fluid,
            self.kappa_grid,
            self.c_v,
            dt,
            geom.dr,
            geom.dz,
            max_iter=20,
            omega=1.4
        )

        self.apply_bc_T()
        self.apply_bc_isothermal()

    def update_vtheta_explicit_force(self, hybrid_pic, geom, dt):
        """
        Updates Neutral azimuthal velocity due to collisions with Ions using an explicit formula.
        Used when fluid ions are on.
        Do not use this when kinetic ions are on.
        """
        neutral_fluid_helper.update_neutral_vtheta_explicit_force(
            self.un_theta_grid,
            hybrid_pic.Jir_grid,
            hybrid_pic.Bz_grid,
            self.nn_grid,
            self.mass,
            dt,
            geom.mask
        )

    #######################################################################################################
    ####################################### TO CALCULATE PARAMS ###########################################
    #######################################################################################################

    def get_viscous_heating_timescale_field(self, geom):
        """
        Computes the local viscous heating timescale field:
            tau_heat(r,z) = (rho * Cv * T) / Phi(r,z)

        TO DO:
            UPDATE TO USE GRAD r and z using mask and face areas!!!!
        """
        # 1. Get Fields
        rho = self.rho_grid
        T   = self.T_n_grid
        mu  = self.mu_grid
        vt  = self.un_theta_grid
        
        # 2. Compute Gradients
        r_safe = geom.r.copy()
        r_safe[0] = 1.0 # Avoid divide by zero
        
        # FIX 1: Reshape r_safe to (Nr, 1) for broadcasting against (Nr, Nz)
        r_col = r_safe[:, None] 
        
        # Calculate Angular Velocity Omega = v / r
        omega = vt / r_col
        
        # Gradient along axis 0 (Radial)
        d_omega_dr = np.gradient(omega, geom.dr, axis=0)
        
        # FIX 2: Use the reshaped column r for the shear calculation too
        shear_r = r_col * d_omega_dr
        
        # Axial Shear: d(v)/dz
        shear_z = np.gradient(vt, geom.dz, axis=1)
        
        # 3. Compute Local Dissipation Phi [W/m^3]
        Phi_grid = mu * (shear_r**2 + shear_z**2)
        
        # 4. Compute Timescale
        E_thermal = rho * self.c_v * T
        
        tau_grid = np.zeros_like(T)
        
        # Avoid divide by zero / valid mask
        mask_valid = (geom.mask == 1)
        tau_grid[mask_valid] = E_thermal[mask_valid] / Phi_grid[mask_valid]
        
        # Set regions with no heating to a large number
        tau_grid[~mask_valid] = 1e9 

        return tau_grid, Phi_grid

    def get_wall_heat_flux(self):
        """
        Calculates the heat flux density [W/m^2] dissipated into the radial wall (r=R).
        q = h_eff * (T_slip - T_wall_fixed)
        """
        # 1. Gather Neighbor Properties (same as in apply_bc_T)
        mask_fluid = (self.geom.mask[-1,:] == 1)
        
        # We need full arrays to match shapes, but we'll mask later
        # Using neighbor (-2) properties for consistency with Kinetic Theory BC
        T_gas   = self.T_n_grid[-2, :]
        rho_gas = self.rho_grid[-2, :]
        mu_gas  = self.mu_grid[-2, :]
        kap_gas = self.kappa_grid[-2, :]
        
        # 2. Physics Constants
        kb = constants.kb
        m  = self.mass
        gamma = self.gamma
        cp = self.cp
        
        # 3. Calculate h_eff (Heat Transfer Coeff)
        # v_th = sqrt(8 k T / pi m)
        v_th = np.sqrt(8.0 * kb * T_gas / (np.pi * m))
        
        # lambda = 2 * mu / (rho * v_th)
        rho_safe = np.maximum(rho_gas, self.nn_floor * m)
        lam = 2.0 * mu_gas / (rho_safe * v_th)
        
        # Pr = Cp * mu / k
        Pr = cp * mu_gas / np.maximum(kap_gas, 1e-15)
        
        # Jump Length
        alpha = getattr(self, 'alpha', 0.5) 
        coeff_alpha = (2.0 - alpha) / alpha
        coeff_gamma = (2.0 * gamma) / (gamma + 1.0)
        
        L_jump = coeff_alpha * coeff_gamma * (lam / Pr)
        L_jump = np.maximum(L_jump, 1e-12)
        
        h_eff = kap_gas / L_jump
        
        # 4. Calculate Flux
        # Flux into wall = h_eff * (T_slip_node - T_solid_fixed)
        # T_n_grid[-1] is the slip temperature we computed in the BC
        T_slip = self.T_n_grid[-1, :]
        q_wall = h_eff * (T_slip - self.T_wall)
        
        # Mask out non-fluid regions
        q_wall[~mask_fluid] = 0.0
        
        return q_wall

    def get_total_wall_power(self):
        """
        Integrates the wall heat flux over the radial wall area to get Total Power [W].
        """
        q_wall = self.get_wall_heat_flux() # [W/m^2]
        
        # Area of wall faces: 2 * pi * R * dz
        # Note: We use geom.rmax. 
        # If your wall is stair-stepped, this assumes a straight cylinder at Rmax.
        dA = 2.0 * np.pi * self.geom.rmax * self.geom.dz
        
        # Integrate (Sum)
        total_power = np.sum(q_wall) * dA
        
        return total_power
