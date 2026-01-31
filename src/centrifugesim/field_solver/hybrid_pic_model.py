import numpy as np
import cupy as cp

from centrifugesim.geometry.geometry import Geometry
from centrifugesim.field_solver.finite_volume_phi_solver import solve_anisotropic_poisson_FV_direct, compute_E_and_J

class HybridPICModel:
    def __init__(self, geom:Geometry):

        self.t = 0 # current simulation time (s)

        # geometry info
        self.zmin = geom.zmin
        self.Nr = geom.Nr
        self.Nz = geom.Nz
        self.dr = geom.dr
        self.dz = geom.dz
        self.r  = geom.r   # 1D array of length Nr (cell centers)
        self.z  = geom.z

        # for cathode boundary condition
        self.phi_cathode_vec = np.zeros(self.Nr).astype(np.float64)
        self.dphi_dz_cathode_top_vec = np.zeros(self.Nr).astype(np.float64)
        self.Jz_cathode_top_vec = np.zeros(self.Nr).astype(np.float64)

        # fields
        self.phi_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)

        self.Er_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.Et_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)  # unused in solver; kept for pusher
        self.Ez_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)

        self.Er_grid_grad_pe = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.Ez_grid_grad_pe = np.zeros((self.Nr, self.Nz)).astype(np.float64)

        # Total conductivity components
        self.sigma_P_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.sigma_parallel_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)

        # electron current density components
        self.Jer_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.Jez_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)

        # ion current density components
        self.Jir_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.Jiz_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)

        # total current density components
        self.Jr_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.Jz_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)

        # q_ohm for electron energy equation
        self.q_ohm_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.q_ohm_ions_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.q_ohm_electrons_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.q_RF = np.zeros((self.Nr, self.Nz)).astype(np.float64) # electrons only

        self.Br_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)
        self.Bt_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)  # unused in solver; kept for pusher
        self.Bz_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)

        self.br_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64) # Br/Bmag
        self.bz_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64) # Bz/Bmag
        self.Bmag_grid = np.zeros((self.Nr, self.Nz)).astype(np.float64)

        # ------- Device fields ---------
        self.Er_grid_d = cp.zeros((self.Nr, self.Nz)).astype(cp.float64)
        self.Et_grid_d = cp.zeros((self.Nr, self.Nz)).astype(cp.float64)
        self.Ez_grid_d = cp.zeros((self.Nr, self.Nz)).astype(cp.float64)

        self.Br_grid_d = cp.zeros((self.Nr, self.Nz)).astype(cp.float64)
        self.Bt_grid_d = cp.zeros((self.Nr, self.Nz)).astype(cp.float64)
        self.Bz_grid_d = cp.zeros((self.Nr, self.Nz)).astype(cp.float64)

    def compute_B_aux(self):
        self.Bmag_grid[:] = np.sqrt(self.Br_grid**2 + self.Bz_grid**2)
        self.br_grid[:] = np.where(self.Bmag_grid==0, 0, self.Br_grid/self.Bmag_grid)
        self.bz_grid[:] = np.where(self.Bmag_grid==0, 0, self.Bz_grid/self.Bmag_grid)

        self.Br_grid_d = cp.asarray(self.Br_grid)
        self.Bz_grid_d = cp.asarray(self.Bz_grid)

    def update_conductivities(self, electron_fluid, ion_fluid):
        self.sigma_P_grid[:] = electron_fluid.sigma_P_grid + ion_fluid.sigma_P_grid
        self.sigma_parallel_grid[:] = electron_fluid.sigma_parallel_grid + ion_fluid.sigma_parallel_grid

    def update_Je_and_Ji_from_Jtotal(self, geom, electron_fluid, ion_fluid):
        """
        Updates self.Jer_grid, self.Jez_grid, self.Jir_grid, self.Jiz_grid
        from total current density Jr_grid, Jz_grid and components of conductivity.
        """
        self.Jer_grid*=0.0
        self.Jez_grid*=0.0
        self.Jir_grid*=0.0
        self.Jiz_grid*=0.0

        self.Jer_grid[geom.mask==1] = (electron_fluid.sigma_P_grid[geom.mask==1] / (electron_fluid.sigma_P_grid[geom.mask==1] + ion_fluid.sigma_P_grid[geom.mask==1])) * self.Jr_grid[geom.mask==1]
        self.Jez_grid[geom.mask==1] = (electron_fluid.sigma_parallel_grid[geom.mask==1] / (electron_fluid.sigma_parallel_grid[geom.mask==1] + ion_fluid.sigma_parallel_grid[geom.mask==1])) * self.Jz_grid[geom.mask==1]

        self.Jir_grid[geom.mask==1] = self.Jr_grid[geom.mask==1] - self.Jer_grid[geom.mask==1]
        self.Jiz_grid[geom.mask==1] = self.Jz_grid[geom.mask==1] - self.Jez_grid[geom.mask==1]
        
    def compute_dphi_dz_cathode(self, geom:Geometry, I, Jiz_grid=None, rmax_injection=None, sigma_r=None):
        # I is negative (enters cathode)

        # Should be input instead
        # add here check that sigma_r is smaller than rmax_injection/2, ideally /3
        rmax_injection = geom.rmax_cathode
        sigma_r = rmax_injection/3.0

        dphi_dz_vec = np.zeros(self.Nr).astype(np.float64)

        i_cathode = (np.arange(self.Nr)[geom.r <= rmax_injection]).astype(np.int32)
        j_cathode = ((int(geom.zmax_cathode/geom.dz)+1)*np.ones_like(i_cathode)).astype(np.int32)

        # calculate input current density (I is negative)
        Jz0 = I / (2*np.pi*sigma_r**2)
        Jz_cathode = Jz0*np.exp(-0.5*geom.r[i_cathode]**2 / sigma_r**2)

        sigma_parallel_cathode = self.sigma_parallel_grid[i_cathode, j_cathode]
        
        # dphi_dz = (Jiz-Jz)/sigma_parallel + dpe/dz /(e*ne) at cathode
        dphi_dz_vec_aux = -Jz_cathode/sigma_parallel_cathode
    
        if(Jiz_grid is not None):
            Jiz_cathode = Jiz_grid[i_cathode, j_cathode]
            dphi_dz_vec_aux += Jiz_cathode/sigma_parallel_cathode

        # commenting out for now to avoid unstable behavior at cathode face
        #dpe_dz_cathode = (electron_fluid.pe_grid[i_cathode, j_cathode+1] - electron_fluid.pe_grid[i_cathode, j_cathode])/geom.dz
        #ne_cathode = electron_fluid.ne_grid[i_cathode, j_cathode]
        #dphi_dz_vec_aux += dpe_dz_cathode/(constants.q_e*ne_cathode)
        
        # flipping sign here due to how the solver was written
        dphi_dz_vec[i_cathode] = - dphi_dz_vec_aux 

        self.dphi_dz_cathode_top_vec = np.copy(dphi_dz_vec)
        self.Jz_cathode_top_vec[i_cathode] = np.copy(Jz_cathode)

    #-----------------------------------------------------------------------------
    #------------------------------ Calls to FV solver ---------------------------
    #-----------------------------------------------------------------------------
    def solve_phi_and_update_fields_grid_FV(self,
        geom:Geometry,
        electron_fluid,
        neutral_fluid,
        tol=1e-9, max_iter=100_000,
        phi_anode_value=0,
        Ji_r=None, Ji_z=None,
        phi0=None,
        cathode_dirichlet = False,
        verbose=True,
        float_outer_wall_top=False):

        if(cathode_dirichlet):
            # write error message and exit
            raise ValueError("Cathode Dirichlet BC cannot be used. Not compatible with hollow cathode setup.")

        else:
            phi, info = solve_anisotropic_poisson_FV_direct(
                    geom,
                    self.sigma_P_grid,
                    self.sigma_parallel_grid,
                    ne=electron_fluid.ne_grid,
                    pe=electron_fluid.pe_grid,
                    Bz=self.Bz_grid,
                    un_theta=neutral_fluid.un_theta_grid,
                    ne_floor=electron_fluid.ne_floor,
                    Ji_r=Ji_r, Ji_z=Ji_z,
                    dphi_dz_cathode_top=self.dphi_dz_cathode_top_vec,
                    phi_anode_value=phi_anode_value,
                    float_outer_wall_top=float_outer_wall_top,
                    phi0=phi0,
                    omega=1.8, tol=tol, max_iter=max_iter,
                    verbose=verbose
            )

        Er, Ez, Jr, Jz, Er_gradpe, Ez_gradpe = compute_E_and_J(
                            phi, geom,
                            self.sigma_P_grid,
                            self.sigma_parallel_grid,
                            ne=electron_fluid.ne_grid,
                            pe=electron_fluid.pe_grid,
                            Bz=self.Bz_grid,
                            un_theta=neutral_fluid.un_theta_grid,
                            ne_floor=electron_fluid.ne_floor,
                            fill_solid_with_nan=False)
        
        Jr[0,:] = 0.0
        Er[0,:] = 0.0

        self.phi_grid = np.copy(phi)
        self.Er_grid = np.copy(Er)
        self.Ez_grid = np.copy(Ez)
        self.Er_grid_grad_pe = np.copy(Er_gradpe)
        self.Ez_grid_grad_pe = np.copy(Ez_gradpe)
        self.Jr_grid = np.copy(Jr)
        self.Jz_grid = np.copy(Jz)

        # apply cathode BC to Jz and Ez (missing from FV solver post calculation of J and E)
        rmax_injection = geom.rmax_cathode
        i_cathode = (np.arange(geom.Nr)[geom.r <= rmax_injection]).astype(np.int32)
        j_cathode = ((int(geom.zmax_cathode/geom.dz)+1)*np.ones_like(i_cathode)).astype(np.int32)
        self.Jz_grid[i_cathode, j_cathode] = self.Jz_cathode_top_vec[i_cathode]
        self.Ez_grid[i_cathode, j_cathode] = self.dphi_dz_cathode_top_vec[i_cathode] # remember signs are flipped!

        q_ohm = self.Jr_grid*self.Er_grid + self.Jz_grid*self.Ez_grid
        self.q_ohm_grid = np.copy(q_ohm)

        self.Er_grid_d = cp.asarray(self.Er_grid)
        self.Ez_grid_d = cp.asarray(self.Ez_grid)

        del phi, Er, Ez, Jr, Jz, q_ohm

    def update_Je_Ji_and_compute_DC_power_e_and_i(self, geom, electron_fluid, ion_fluid):
        self.update_Je_and_Ji_from_Jtotal(geom, electron_fluid, ion_fluid)
        # compute q_ohm for electrons and ions separately
        self.q_ohm_electrons_grid = self.Jer_grid*self.Er_grid + self.Jez_grid*self.Ez_grid
        self.q_ohm_ions_grid = self.Jir_grid*self.Er_grid + self.Jiz_grid*self.Ez_grid

        #self.q_ohm_electrons_grid[geom.i_cathode_z_sheath, geom.j_cathode_z_sheath+1] = 0.0
        #self.q_ohm_ions_grid[geom.i_cathode_z_sheath, geom.j_cathode_z_sheath+1] = 0.0
        
    # -------- Calculate electrodes currents
    def compute_electrode_currents(self, geom, return_parts=False):

        I_cathode = 0.0
        # cathode current rmax_cathode (Jr)
        I_cathode += np.trapz(self.Jr_grid[geom.i_cathode_r+1, geom.j_cathode_r]*geom.rmax_cathode*2*np.pi,geom.z[geom.j_cathode_r])
        # cathode current zmax_cathode (Jz)
        I_cathode += np.trapz(2*np.pi*geom.r[:geom.i_cath_max]*self.Jz_grid[:geom.i_cath_max, geom.j_cath_max+1], geom.r[:geom.i_cath_max])

        I_anode = 0.0
       
        return I_anode, I_cathode