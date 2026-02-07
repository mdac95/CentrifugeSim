import numpy as np

from centrifugesim import constants
from centrifugesim.fluids import ion_fluid_helper

class IonFluidContainer:
    def __init__(self, geom, m_i, name, Z=1.0, sigma_cx=5e-19, Ti0=300.0):
        """
        Container for Ion Fluid moments and transport coefficients.
        
        Args:
            geom: Geometry object containing .Nr, .Nz, .mask
            m_i: Ion mass in kg
            name: Ion species name
            Z: Ion charge state
            sigma_cx: Charge Exchange cross-section (m^2)
            Ti0: Initial ion temperature (K)
        """
        self.name = name
        self.geom = geom
        self.m_i = float(m_i)
        self.Z = float(Z)
        self.sigma_cx = sigma_cx
        
        # Grid Dimensions
        self.Nr = geom.Nr
        self.Nz = geom.Nz
        shape = (self.Nr, self.Nz)
        
        # --- Primary Fields ---
        self.ni_grid = np.zeros(shape, dtype=np.float64)
        self.Ti_grid = Ti0 + np.zeros(shape, dtype=np.float64)

        # --- Velocity Fields ---
        self.vi_r_grid = np.zeros(shape, dtype=np.float64)      # Ion radial velocity (m/s)
        self.vi_theta_grid = np.zeros(shape, dtype=np.float64)  # Ion azimuthal velocity (m/s)
        self.vi_z_grid = np.zeros(shape, dtype=np.float64)      # Ion axial velocity (m/s)

        # --- Thermal Properties ---
        self.kappa_perp_grid = np.zeros(shape, dtype=np.float64)    # Perpendicular Thermal Conductivity (W/m/K)
        self.kappa_parallel_grid = np.zeros(shape, dtype=np.float64) # Parallel Thermal Conductivity (W/m/K)

        # --- Collision & Magnetization ---
        self.nu_i_grid = np.zeros(shape, dtype=np.float64)   # Total collision freq (s^-1)
        self.beta_i_grid = np.zeros(shape, dtype=np.float64) # Hall parameter (wci/nu_i)
        
        # --- Viscosity ---
        self.eta_0 = np.zeros(shape, dtype=np.float64)               # Parallel Viscosity
        
        # -- Conductivities ---
        self.sigma_P_grid = np.zeros(shape, dtype=np.float64)        # Pedersen Conductivity
        self.sigma_parallel_grid = np.zeros(shape, dtype=np.float64) # Parallel Conductivity
        self.sigma_H_grid = np.zeros(shape, dtype=np.float64) # Hall Conductivity

    def update_drift_velocities(self, geom, hybrid_pic):
        """
        Updates self.vi_r_grid and self.vi_z_grid from Jir and Jiz.
        self.vi_theta_grid is updated separately.
        """
        self.vi_r_grid[geom.mask==1] = hybrid_pic.Jir_grid[geom.mask==1] / (self.Z * constants.q_e * self.ni_grid[geom.mask==1])
        self.vi_z_grid[geom.mask==1] = hybrid_pic.Jiz_grid[geom.mask==1] / (self.Z * constants.q_e * self.ni_grid[geom.mask==1])

    def update_vtheta(self, geom, hybrid_pic, neutral_fluid):
        """
        Updates self.vtheta using the algebraic approximation (Drag = JxB).
        TO DO:
            - Extend to include Jz x Br term for coil geometry.
        """
        ion_fluid_helper.update_vtheta_kernel_algebraic(
            self.vi_theta_grid,           # Output
            hybrid_pic.Jir_grid,           # Input, total current
            hybrid_pic.Bz_grid,           # Input
            self.ni_grid,                 # Input
            self.nu_i_grid,               # Input
            neutral_fluid.un_theta_grid,  # Input
            geom.mask,                    # Geometry
            self.m_i                      # Constant
        )

    def update_vtheta_viscous(self, hybrid_pic, neutral_fluid, max_iter=10000, tol=1e-6, omega=1.4):
        """
        Updates self.vtheta by solving the full Viscous-Resistive-inertial balance.
        Uses the existing self.vtheta as the initial guess for the iterative solver.
        Note:
            - self.eta_0 is small so ion viscosity seems to have little effect.
            recommend instantanous (algebraic) update for now. Double check once
            fully integrated
        """
        
        # Call the SOR solver
        ion_fluid_helper.solve_vtheta_viscous_SOR(
            self.vi_theta_grid,                  # In/Out (Uses previous step as guess)
            hybrid_pic.Jir_grid,           # Input
            hybrid_pic.Bz_grid,           # Input
            self.ni_grid,                 # Input
            self.nu_i_grid,               # Input
            neutral_fluid.un_theta_grid,  # Input
            self.eta_0,                   # Input (Viscosity Grid)
            self.geom.mask,               # Geometry
            self.geom.dr, 
            self.geom.dz, 
            self.geom.r, 
            self.m_i,
            max_iter=max_iter,           # Tunable
            tol=tol,                     # Tunable
            omega=omega                  # SOR parameter (1.0 = Gauss Seidel)
        )

    def update_vtheta_viscous_implicit(self, hybrid_pic, neutral_fluid, dt, max_iter=10000, tol=1e-4, omega=1.4, closed_top=False):
        """
        Updates self.vtheta by solving the full Viscous-Resistive-inertial balance implicitly.
        Uses the existing self.vtheta as the initial guess for the iterative solver.
        Note:
            - self.eta_0 is small so ion viscosity seems to have little effect.
        """
        vi_theta_new = np.zeros_like(self.vi_theta_grid)
        vi_theta_old = self.vi_theta_grid.copy()

        # Call the implicit SOR solver
        ion_fluid_helper.solve_vtheta_viscous_implicit_SOR(
            vi_theta_new,                  # In/Out (Uses previous step as guess)
            vi_theta_old,                  # Input (Old vtheta)
            hybrid_pic.Jir_grid,           # Input
            hybrid_pic.Bz_grid,           # Input
            self.ni_grid,                 # Input
            self.nu_i_grid,               # Input
            neutral_fluid.un_theta_grid,  # Input
            self.eta_0,                   # Input (Viscosity Grid)
            self.geom.mask,               # Geometry
            self.geom.dr, 
            self.geom.dz, 
            self.geom.r, 
            self.m_i,
            dt,
            max_iter=max_iter,           # Tunable
            tol=tol,                     # Tunable
            omega=omega,                  # SOR parameter (1.0 = Gauss Seidel)
            closed_top=closed_top
        )

        self.vi_theta_grid[...] = vi_theta_new.copy()
        del vi_theta_new, vi_theta_old


    def update_collision_frequencies(self, geom, neutral_fluid):
        """
        Calculates nu_i = nu_in (Charge Exchange).
        """            
        ion_fluid_helper.compute_nu_i_kernel(
            self.nu_i_grid,    # Output
            self.ni_grid,      # Input
            self.Ti_grid,      # Input
            neutral_fluid.nn_grid,     # Input
            neutral_fluid.T_n_grid,    # Input
            self.Z,
            self.m_i,
            self.sigma_cx,
            geom.mask,
            self.vi_theta_grid,
            neutral_fluid.un_theta_grid
        )

    def update_beta_i(self, geom, hybrid_pic):
        """
        Calculates beta_i = wci / nu_i
        """
        ion_fluid_helper.compute_beta_i_kernel(
            self.beta_i_grid,   # Output
            self.nu_i_grid,     # Input (must be updated first)
            hybrid_pic.Bz_grid, # Input
            self.Z,
            constants.q_e,
            self.m_i,
            geom.mask
        )

    def update_conductivities(self, geom):
        """
        Calculates sigma_P and sigma_parallel.
        """
        ion_fluid_helper.compute_conductivities_kernel(
            self.sigma_P_grid,        # Output
            self.sigma_parallel_grid, # Output
            self.ni_grid,
            self.nu_i_grid,
            self.beta_i_grid,
            self.Z,
            constants.q_e,
            self.m_i,
            geom.mask
        )

    def update_thermal_conductivity(self, geom):
        """
        Calculates Parallel and Perpendicular Thermal Conductivity.
        kappa_par  = (ni * kb^2 * Ti) / (mi * nu_i)
        kappa_perp = kappa_par / (1 + beta_i^2)
        """
        ion_fluid_helper.compute_thermal_conductivity_kernel(
            self.kappa_parallel_grid,   # Output
            self.kappa_perp_grid,       # Output
            self.ni_grid,               # Input
            self.Ti_grid,               # Input
            self.nu_i_grid,             # Input
            self.beta_i_grid,           # Input
            self.m_i,
            geom.mask
        )

    def compute_viscosity_eta0(self, geom):
        """
        Computes parallel viscosity eta_0.
        """
        self.eta_0[geom.mask==1] = 0.96 * self.ni_grid[geom.mask==1] * constants.kb * self.Ti_grid[geom.mask==1] / self.nu_i_grid[geom.mask==1]

    def update_temperature(self, geom, neutral_fluid, electron_fluid, hybrid_pic, dt):
        Ti_new = np.zeros_like(self.Ti_grid)
        Ti_old = self.Ti_grid.copy()
        ion_fluid_helper.update_Ti_joule_heating_implicit_kernel(
            Ti_new,                       # Output
            Ti_old,                       # Input: Old Temperature
            neutral_fluid.T_n_grid,       # Input: Neutral Temp
            electron_fluid.Te_grid,       # Input: Electron Temp
            hybrid_pic.q_ohm_ions_grid,   # Input: Joule Heating Power (W/m^3)
            self.ni_grid,                 # Input: Ion Density
            self.nu_i_grid,               # Input: Ion-Neutral Freq
            electron_fluid.nu_ei_grid,    # Input: Electron-Ion Freq
            self.m_i,                     # Mass Ion
            neutral_fluid.mass,           # Mass Neutral
            geom.mask,                    # Mask
            constants.m_e,                # Constant: Electron Mass
            constants.kb,                 # Constant: Boltzmann
            dt                            # Time Step
        )
        self.Ti_grid[...] = Ti_new.copy()
        del Ti_new, Ti_old

        Ti_new = np.zeros_like(self.Ti_grid)
        Ti_old = self.Ti_grid.copy()
        ion_fluid_helper.solve_heat_conduction_sor_kernel(
            Ti_new,
            Ti_old,
            self.ni_grid,                 # Input: Ion Density
            self.kappa_parallel_grid,     # Input: Parallel Thermal Conductivity
            self.kappa_perp_grid,         # Input: Perpendicular Thermal Conductivity
            hybrid_pic.br_grid,           # Input: Radial Magnetic Field
            hybrid_pic.bz_grid,           # Input: Axial Magnetic Field
            geom.mask,                    # Mask
            dt,
            geom.dr,
            geom.dz,
            geom.r,
        )
        self.Ti_grid[...] = Ti_new.copy()
        del Ti_new, Ti_old

        # Add friction heating with neutrals in theta
        ion_fluid_helper.add_friction_heating_theta_kernel(
            self.Ti_grid,
            self.vi_theta_grid,
            neutral_fluid.un_theta_grid,
            self.nu_i_grid,
            self.m_i,
            dt,
            geom.mask
        )

        # Ensure Neumann BC at cathode
        rmax_injection = geom.rmax_cathode
        i_cathode = (np.arange(geom.Nr)[geom.r <= rmax_injection]).astype(np.int32)
        j_cathode = ((int(geom.zmax_cathode/geom.dz)+1)*np.ones_like(i_cathode)).astype(np.int32)
        self.Ti_grid[i_cathode, j_cathode] = self.Ti_grid[i_cathode, j_cathode+1]

        # Add Neumann or Dirichlet at anode?
        # Make solve_heat_conduction_sor_kernel compatible with geometry boundary conditions

        
