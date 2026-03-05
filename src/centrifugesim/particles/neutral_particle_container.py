import cupy as cp
import numpy as np
import math

neutral_kernels_src = r'''

extern "C" __global__
void gather_emission_rate_kernel(
    const int N,
    const float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ z,
    const float* __restrict__ emission_grid,
    float* __restrict__ emission_rate,
    const int Nr, const int Nz, const float dr, const float dz, const float zmin,
    const bool* __restrict__ active)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N || !active[i]) return;

    // 1. Coordinate Transform for Interpolation
    float r_val = sqrtf(x[i]*x[i] + y[i]*y[i]);
    float z_val = z[i];
    
    float i_float = r_val / dr;
    float j_float = (z_val - zmin) / dz;
    
    int i0 = max(0, min((int)floorf(i_float), Nr - 2));
    int j0 = max(0, min((int)floorf(j_float), Nz - 2));
    
    float alpha = fminf(fmaxf(i_float - i0, 0.0f), 1.0f);
    float beta  = fminf(fmaxf(j_float - j0, 0.0f), 1.0f);
    
    float w00 = (1.0f - alpha) * (1.0f - beta);
    float w10 = alpha * (1.0f - beta);
    float w01 = (1.0f - alpha) * beta;
    float w11 = alpha * beta;
    
    int idx00 = i0 * Nz + j0;
    int idx10 = (i0 + 1) * Nz + j0;
    int idx01 = i0 * Nz + (j0 + 1);
    int idx11 = (i0 + 1) * Nz + (j0 + 1);
    
    // 2. Interpolate Emission Property
    emission_rate[i] = w00 * emission_grid[idx00] + w10 * emission_grid[idx10] + 
                       w01 * emission_grid[idx01] + w11 * emission_grid[idx11];
}

extern "C" __global__
void push_and_track_kernel(
    const int N, const float dt, const float current_time, const float r_max,
    float* __restrict__ x, float* __restrict__ y, float* __restrict__ z,
    const float* __restrict__ vx, const float* __restrict__ vy, const float* __restrict__ vz,
    float* __restrict__ accum_theta, float* __restrict__ time_of_flight, 
    float* __restrict__ final_r, bool* __restrict__ finished, bool* __restrict__ hit_wall,
    const float* __restrict__ injection_time, bool* __restrict__ active)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N || !active[i] || finished[i]) return;

    // 1. Calculate infinitesimal angle change d(theta) before updating positions
    float r2 = x[i]*x[i] + y[i]*y[i];
    float r2_safe = fmaxf(r2, 1e-12f); 
    float dtheta = (x[i] * vy[i] - y[i] * vx[i]) / r2_safe * dt;
    accum_theta[i] += fabsf(dtheta);

    // 2. Cartesian Push
    x[i] += vx[i] * dt;
    y[i] += vy[i] * dt;
    z[i] += vz[i] * dt;

    // 3. Check Boundaries and Completion
    float r_new = sqrtf(x[i]*x[i] + y[i]*y[i]);
    
    if (r_new >= r_max) {
        // Particle hit the outer wall
        finished[i] = true;
        hit_wall[i] = true;
        final_r[i] = r_new;
        time_of_flight[i] = current_time - injection_time[i]; // Record time of crash
    } 
    else if (accum_theta[i] >= 6.28318530718f) {
        // Particle successfully completed 1 cycle (2*Pi)
        finished[i] = true;
        time_of_flight[i] = current_time - injection_time[i];
        final_r[i] = r_new;
    }
}

extern "C" __global__
void mcc_he_h_kernel(
    const int N, const float dt,
    float* __restrict__ vx, float* __restrict__ vy, float* __restrict__ vz,
    const float* __restrict__ x, const float* __restrict__ y, const float* __restrict__ z,
    const float* __restrict__ nn_grid, const float* __restrict__ Tn_grid, 
    const float* __restrict__ un_theta_grid, const float* __restrict__ un_r_grid, const float* __restrict__ un_z_grid,
    const int Nr, const int Nz, const float dr, const float dz, const float zmin,
    const float* __restrict__ rcoll, const float* __restrict__ r_chi, const float* __restrict__ r_phi,
    const float* __restrict__ g1, const float* __restrict__ g2, const float* __restrict__ g3,
    const float sigma_mt, const float m_He, const float m_H, const float kB,
    bool* __restrict__ active, const bool* __restrict__ finished)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N || !active[i] || finished[i]) return;

    // 1. Coordinate Transform for Interpolation
    float r_val = sqrtf(x[i]*x[i] + y[i]*y[i]);
    float z_val = z[i];
    
    float i_float = r_val / dr;
    float j_float = (z_val - zmin) / dz;
    
    int i0 = max(0, min((int)floorf(i_float), Nr - 2));
    int j0 = max(0, min((int)floorf(j_float), Nz - 2));
    
    float alpha = fminf(fmaxf(i_float - i0, 0.0f), 1.0f);
    float beta  = fminf(fmaxf(j_float - j0, 0.0f), 1.0f);
    
    float w00 = (1.0f - alpha) * (1.0f - beta);
    float w10 = alpha * (1.0f - beta);
    float w01 = (1.0f - alpha) * beta;
    float w11 = alpha * beta;
    
    int idx00 = i0 * Nz + j0;
    int idx10 = (i0 + 1) * Nz + j0;
    int idx01 = i0 * Nz + (j0 + 1);
    int idx11 = (i0 + 1) * Nz + (j0 + 1);
    
    // 2. Interpolate Fluid Properties
    float n_H = w00 * nn_grid[idx00] + w10 * nn_grid[idx10] + w01 * nn_grid[idx01] + w11 * nn_grid[idx11];
    float T_H = w00 * Tn_grid[idx00] + w10 * Tn_grid[idx10] + w01 * Tn_grid[idx01] + w11 * Tn_grid[idx11];
    float u_theta = w00 * un_theta_grid[idx00] + w10 * un_theta_grid[idx10] + w01 * un_theta_grid[idx01] + w11 * un_theta_grid[idx11];
    float u_r = w00 * un_r_grid[idx00] + w10 * un_r_grid[idx10] + w01 * un_r_grid[idx01] + w11 * un_r_grid[idx11];
    float u_z = w00 * un_z_grid[idx00] + w10 * un_z_grid[idx10] + w01 * un_z_grid[idx01] + w11 * un_z_grid[idx11];

    // 3. Convert Fluid Velocity to Cartesian Components
    float r_safe = fmaxf(r_val, 1e-12f);
    float cos_th = x[i] / r_safe;
    float sin_th = y[i] / r_safe;
    
    float ux_H = u_r * cos_th - u_theta * sin_th;
    float uy_H = u_r * sin_th + u_theta * cos_th;
    float uz_H = u_z;

    // 4. Sample H Atom from Drifting Maxwellian
    float vth_H = sqrtf(kB * T_H / m_H);
    float v_Hx = ux_H + vth_H * g1[i];
    float v_Hy = uy_H + vth_H * g2[i];
    float v_Hz = uz_H + vth_H * g3[i];

    // 5. Relative Velocity & Collision Probability
    float u_rel_x = vx[i] - v_Hx;
    float u_rel_y = vy[i] - v_Hy;
    float u_rel_z = vz[i] - v_Hz;
    float u_rel_mag = sqrtf(u_rel_x*u_rel_x + u_rel_y*u_rel_y + u_rel_z*u_rel_z) + 1e-12f;

    float P = 1.0f - expf(-n_H * sigma_mt * u_rel_mag * dt);
    
    if (rcoll[i] < P) {
        // 6. Center of Mass Velocity
        float m_tot = m_He + m_H;
        float v_cm_x = (m_He * vx[i] + m_H * v_Hx) / m_tot;
        float v_cm_y = (m_He * vy[i] + m_H * v_Hy) / m_tot;
        float v_cm_z = (m_He * vz[i] + m_H * v_Hz) / m_tot;

        // 7. Isotropic Scattering
        float cos_chi = 1.0f - 2.0f * r_chi[i];
        float sin_chi = sqrtf(fmaxf(0.0f, 1.0f - cos_chi*cos_chi));
        float phi = 6.28318530718f * r_phi[i];
        
        float n_x = sin_chi * cosf(phi);
        float n_y = sin_chi * sinf(phi);
        float n_z = cos_chi;

        // 8. Update He Velocity
        vx[i] = v_cm_x + (m_H / m_tot) * u_rel_mag * n_x;
        vy[i] = v_cm_y + (m_H / m_tot) * u_rel_mag * n_y;
        vz[i] = v_cm_z + (m_H / m_tot) * u_rel_mag * n_z;
    }
}
'''

push_and_track_module = cp.RawModule(code=neutral_kernels_src)
push_and_track = push_and_track_module.get_function('push_and_track_kernel')
mcc_he_h = push_and_track_module.get_function('mcc_he_h_kernel')
gather_emission = push_and_track_module.get_function('gather_emission_rate_kernel')

# =============================================================================
# Python Container Class
# =============================================================================

class HeTrackerContainer:
    def __init__(self, max_particles=20000000):
        self.max_particles = max_particles
        self.active_count = 0
        
        # Physical Constants
        self.m_H = 1.67e-27  # kg
        self.m_He = 6.64e-27 # kg
        self.kB = 1.38e-23   # J/K
        self.sigma_mt = 1.0e-19 # m^2

        # Particle Arrays
        self.x = cp.zeros(max_particles, dtype=cp.float32)
        self.y = cp.zeros(max_particles, dtype=cp.float32)
        self.z = cp.zeros(max_particles, dtype=cp.float32)
        self.vx = cp.zeros(max_particles, dtype=cp.float32)
        self.vy = cp.zeros(max_particles, dtype=cp.float32)
        self.vz = cp.zeros(max_particles, dtype=cp.float32)
        
        # Tracking Arrays
        self.active = cp.zeros(max_particles, dtype=cp.bool_)
        self.finished = cp.zeros(max_particles, dtype=cp.bool_)
        self.hit_wall = cp.zeros(max_particles, dtype=cp.bool_)
        self.accum_theta = cp.zeros(max_particles, dtype=cp.float32)
        self.emission_rate = cp.zeros(max_particles, dtype=cp.float32)
        
        # Diagnostic Outputs
        self.injection_time = cp.zeros(max_particles, dtype=cp.float32)
        self.time_of_flight = cp.zeros(max_particles, dtype=cp.float32)
        self.final_r = cp.zeros(max_particles, dtype=cp.float32)

        # Cache for zero-grid to prevent slow memory allocation during the loop
        self._zero_grid = None

        self.threads_per_block = 256
        self.blocks_per_grid = (max_particles + self.threads_per_block - 1) // self.threads_per_block

    def inject_particles(self, num_to_inject, r_inj, z_inj, current_time, theta_inj=0, v_jet=1019.0, T_jet=20.0):
        """
        Injects a batch of He atoms representing a supersonic free jet.
        v_jet: Directed velocity of the expanded gas in m/s.
        T_jet: Cold thermal spread of the gas after expansion in Kelvin.
        """
        if self.active_count + num_to_inject > self.max_particles:
            print("Warning: Particle capacity reached. Injecting remaining capacity.")
            num_to_inject = self.max_particles - self.active_count
            if num_to_inject <= 0: return

        start_idx = self.active_count
        end_idx = start_idx + num_to_inject

        # at snorkel location
        self.x[start_idx:end_idx] = r_inj * cp.cos(theta_inj)
        self.y[start_idx:end_idx] = r_inj * cp.sin(theta_inj)
        self.z[start_idx:end_idx] = z_inj
        
        # 2. Calculate the cold thermal spread of the jet
        vth = math.sqrt(self.kB * T_jet / self.m_He)
        
        # 3. Generate initial velocities in Cylindrical coordinates (r, theta, z)
        # Random thermal fluctuations around the directed drift
        vth_r = cp.random.normal(0, vth, num_to_inject, dtype=cp.float32)
        vth_theta = cp.random.normal(0, vth, num_to_inject, dtype=cp.float32)
        vth_z = cp.random.normal(0, vth, num_to_inject, dtype=cp.float32)
        
        # Apply the directed jet velocity ONLY in the -r direction
        v_r = -v_jet + vth_r
        v_theta = vth_theta       # Drift is 0, only thermal
        v_z = vth_z               # Drift is 0, only thermal

        # 4. Transform Cylindrical velocities to Cartesian (vx, vy, vz)
        cos_th = cp.cos(theta_inj)
        sin_th = cp.sin(theta_inj)
        
        self.vx[start_idx:end_idx] = v_r * cos_th - v_theta * sin_th
        self.vy[start_idx:end_idx] = v_r * sin_th + v_theta * cos_th
        self.vz[start_idx:end_idx] = v_z

        # 5. Track state
        self.injection_time[start_idx:end_idx] = current_time
        self.active[start_idx:end_idx] = True
        
        self.active_count += num_to_inject

    def step(self, dt, current_time, neutral_fluid, geom, include_rz_vel=False):
        """Executes one sub-cycled Monte Carlo timestep."""
        if self.active_count == 0:
            return

        # =====================================================================
        # CPU to GPU MEMORY TRANSFER & TYPE CASTING (float64 -> float32)
        # =====================================================================
        # CuPy handles the NumPy conversion gracefully over the PCIe bus
        d_nn_grid = cp.asarray(neutral_fluid.nn_grid, dtype=cp.float32)
        d_Tn_grid = cp.asarray(neutral_fluid.T_n_grid, dtype=cp.float32)
        d_un_theta_grid = cp.asarray(neutral_fluid.un_theta_grid, dtype=cp.float32)

        # Handle optional r and z fluid velocities
        if include_rz_vel:
            un_r = cp.asarray(neutral_fluid.un_r_grid, dtype=cp.float32)
            un_z = cp.asarray(neutral_fluid.un_z_grid, dtype=cp.float32)
        else:
            if self._zero_grid is None or self._zero_grid.shape != neutral_fluid.nn_grid.shape:
                self._zero_grid = cp.zeros_like(neutral_fluid.nn_grid, dtype=cp.float32)
            un_r = self._zero_grid
            un_z = self._zero_grid

        # 1. Generate Random Numbers for Monte Carlo
        rcoll = cp.random.rand(self.max_particles, dtype=cp.float32)
        r_chi = cp.random.rand(self.max_particles, dtype=cp.float32)
        r_phi = cp.random.rand(self.max_particles, dtype=cp.float32)
        g1 = cp.random.randn(self.max_particles, dtype=cp.float32)
        g2 = cp.random.randn(self.max_particles, dtype=cp.float32)
        g3 = cp.random.randn(self.max_particles, dtype=cp.float32)

        # 2. Run Collisions (Passing the correctly casted GPU arrays)
        mcc_he_h((self.blocks_per_grid,), (self.threads_per_block,), (
            self.max_particles, np.float32(dt),
            self.vx, self.vy, self.vz,
            self.x, self.y, self.z,
            d_nn_grid, d_Tn_grid, 
            d_un_theta_grid, un_r, un_z,
            neutral_fluid.Nr, neutral_fluid.Nz, np.float32(geom.dr), np.float32(geom.dz), np.float32(0),
            rcoll, r_chi, r_phi, g1, g2, g3,
            np.float32(self.sigma_mt), np.float32(self.m_He), np.float32(self.m_H), np.float32(self.kB),
            self.active, self.finished
        ))

        # 3. Push and Track Angles
        push_and_track((self.blocks_per_grid,), (self.threads_per_block,), (
            self.max_particles, np.float32(dt), np.float32(current_time), np.float32(geom.rmax),
            self.x, self.y, self.z,
            self.vx, self.vy, self.vz,
            self.accum_theta, self.time_of_flight, self.final_r, 
            self.finished, self.hit_wall, self.injection_time, self.active
        ))


    def get_cylindrical_velocities(self):
        """
        Calculates the radial (vr) and azimuthal (vtheta) velocities 
        from Cartesian coordinates and returns them as CPU NumPy arrays.
        """
        # Calculate r, using a minimum threshold to prevent division by zero at the origin
        r = cp.sqrt(self.x**2 + self.y**2)
        r_safe = cp.maximum(r, 1e-12)
        
        # Calculate cylindrical velocity components (m/s)
        vr = (self.x * self.vx + self.y * self.vy) / r_safe
        vtheta = (self.x * self.vy - self.y * self.vx) / r_safe
        
        # Copy to CPU memory and return as NumPy arrays
        return cp.asnumpy(vr), cp.asnumpy(vtheta)
    

    def calculate_emission(self, emission_field_cpu, geom, Nr, Nz):
        """
        Interpolates the provided emission field (1/s) to all active particle positions.
        """
        if self.active_count == 0:
            return
            
        # Transfer the pre-calculated field to the GPU
        d_emission_grid = cp.asarray(emission_field_cpu, dtype=cp.float32)

        # Launch Kernel
        gather_emission((self.blocks_per_grid,), (self.threads_per_block,), (
            self.max_particles,
            self.x, self.y, self.z,
            d_emission_grid,
            self.emission_rate,
            Nr, Nz, np.float32(geom.dr), np.float32(geom.dz), np.float32(0),
            self.active
        ))

    def get_emission_rates(self, active_only=True):
        """
        Returns the interpolated emission rates as a CPU NumPy array.
        """
        end_idx = self.active_count if active_only else self.max_particles
        return cp.asnumpy(self.emission_rate[:end_idx])