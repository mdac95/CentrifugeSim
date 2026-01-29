import numpy as np
from typing import Dict, Tuple, List

class Geometry:
    """
    2D RZ grid geometry with anode/cathode regions and coil metadata.
    Coils MUST be outside the domain; any overlap with the domain raises.
    """

    def __init__(
        self,
        rmax: float,
        zmin: float,
        zmax: float,
        Nr: int,
        Nz: int,
        rmax_cathode: float,
        zmax_cathode: float,
        rmin_anode: float,
        zmin_anode: float,
        zmax_anode: float,
        zmin_anode2: float,
        temperature_cathode: float,
        temperature_anode: float
    ):
        # Domain limits
        self.rmin = 0.0
        self.rmax = float(rmax)
        self.zmin = float(zmin)
        self.zmax = float(zmax)
        self.Nr = int(Nr)
        self.Nz = int(Nz)

        # Grid spacings
        self.dr = (self.rmax - self.rmin) / (self.Nr - 1)
        self.dz = (self.zmax - self.zmin) / (self.Nz - 1)

        # Coordinates and mesh
        self.r = np.linspace(self.rmin, self.rmax, self.Nr, dtype=np.float64)
        self.z = np.linspace(self.zmin, self.zmax, self.Nz, dtype=np.float64)
        self.R, self.Z = np.meshgrid(self.r, self.z, indexing='ij')

        # Anode (bool mask)
        self.rmin_anode = float(rmin_anode)
        self.rmax_anode = self.rmax
        self.zmin_anode = float(zmin_anode)
        self.zmax_anode = float(zmax_anode)
        self.zmin_anode2 = float(zmin_anode2)
        self.zmax_anode2 = float(zmin_anode2+zmax_anode-zmin_anode)
        
        self.anode1_mask = (
            (self.r[:, None] >= self.rmin_anode) & (self.r[:, None] <= self.rmax_anode) &
            (self.z[None, :] >= self.zmin_anode) & (self.z[None, :] <= self.zmax_anode)
        )

        self.anode2_mask = (
            (self.r[:, None] >= self.rmin_anode) & (self.r[:, None] <= self.rmax_anode) &
            (self.z[None, :] >= self.zmin_anode2) & (self.z[None, :] <= self.zmin_anode2+(self.zmax_anode-self.zmin_anode))
        )

        # Cathode (bool mask)
        self.rmin_cathode = 0.0
        self.rmax_cathode = float(rmax_cathode)
        self.zmin_cathode = 0.0
        self.zmax_cathode = float(zmax_cathode)
        self.cathode_mask = (
            (self.r[:, None] >= self.rmin_cathode) & (self.r[:, None] <= self.rmax_cathode) &
            (self.z[None, :] >= self.zmin_cathode) & (self.z[None, :] <= self.zmax_cathode)
        )
        self.i_cath_max = int(self.rmax_cathode/self.dr)
        self.j_cath_max = int(self.zmax_cathode/self.dz)

        self.i_cathode_r = (self.i_cath_max + np.zeros(self.j_cath_max)).astype(np.int32)
        self.j_cathode_r = np.arange(self.j_cath_max).astype(np.int32)

        self.i_cathode_z_sheath = np.arange(2/3*self.i_cath_max, self.i_cath_max+1).astype(np.int32)
        self.j_cathode_z_sheath = (self.j_cath_max + np.zeros(len(self.i_cathode_z_sheath))).astype(np.int32)

        # Solve mask: int8 (1 = solve, 0 = masked)
        self.mask = np.ones((self.Nr, self.Nz), dtype=np.int8)
        self.mask[self.cathode_mask] = 0
        self.mask[self.anode1_mask] = 0
        self.mask[self.anode2_mask] = 0

        # find arrays with valid nodes right next to masked ones.
        self.i_bc_list, self.j_bc_list = self.find_boundary_nodes()

        # Separate lists for cathode and anode BC nodes
        self.anode_mask = np.ones((self.Nr, self.Nz), dtype=np.int8)
        self.anode_mask[self.anode1_mask] = 0
        self.anode_mask[self.anode2_mask] = 0
        self.i_anode_bc_list, self.j_anode_bc_list = self.find_boundary_nodes(mask=self.anode_mask)

        # Cathode and anode temperatures
        self.temperature_cathode = temperature_cathode
        self.temperature_anode = temperature_anode

        # Volume weights
        self.volume_field = self.compute_volume_field()
        self.volume_field[-1, :] = 0.5 * self.r[-1] * 2 * np.pi * self.dr * self.dz # corrected last cell volume

        # Volume weights for particle deposition
        self.volume_field_dep = self.compute_volume_field_dep()

        # For now, check cuda c kernel at z BCs
        self.volume_field_dep[:,0] = self.volume_field_dep[:,1]
        self.volume_field_dep[:,-1] = self.volume_field_dep[:,-2]

        # Coils store (outside-domain only)
        self.coils: Dict[str, Dict[str, float]] = {}

    @property
    def n_coils(self) -> int:
        return len(self.coils)

    # ---------- Coil helpers (outside-domain enforcement) ----------

    def _intervals_overlap_closed(self, a0: float, a1: float, b0: float, b1: float) -> bool:
        """Return True if [a0,a1] overlaps (even at a point) with [b0,b1]."""
        # Ensure ordering
        if a0 > a1: a0, a1 = a1, a0
        if b0 > b1: b0, b1 = b1, b0
        return max(a0, b0) <= min(a1, b1)

    def _assert_coil_outside_domain(self, rc: float, drc: float, zc: float, dzc: float) -> None:
        """
        Raise if the coil window intersects or touches the simulation domain.
        """
        if drc <= 0 or dzc <= 0:
            raise ValueError("drc and dzc must be positive.")
        if rc < 0:
            raise ValueError("rc must be >= 0 for cylindrical geometry.")

        r0, r1 = rc - 0.5 * drc, rc + 0.5 * drc
        z0, z1 = zc - 0.5 * dzc, zc + 0.5 * dzc

        r_overlap = self._intervals_overlap_closed(r0, r1, self.rmin, self.rmax)
        z_overlap = self._intervals_overlap_closed(z0, z1, self.zmin, self.zmax)

        if r_overlap and z_overlap:
            # Overlap in both r and z => coil rectangle intersects/touches domain
            raise ValueError(
                "Coil window intersects the simulation domain. "
                "Biot-Savart solver requires coils to be entirely outside the domain. "
                f"Coil r∈[{r0:.6g},{r1:.6g}], z∈[{z0:.6g},{z1:.6g}] vs "
                f"domain r∈[{self.rmin:.6g},{self.rmax:.6g}], z∈[{self.zmin:.6g},{self.zmax:.6g}]."
            )

    def coil_overlaps_domain(self, name: str) -> bool:
        """Check if a named coil overlaps/touches the domain."""
        c = self.get_coil(name)
        r0, r1 = c["rc"] - 0.5 * c["drc"], c["rc"] + 0.5 * c["drc"]
        z0, z1 = c["zc"] - 0.5 * c["dzc"], c["zc"] + 0.5 * c["dzc"]
        r_overlap = self._intervals_overlap_closed(r0, r1, self.rmin, self.rmax)
        z_overlap = self._intervals_overlap_closed(z0, z1, self.zmin, self.zmax)
        return r_overlap and z_overlap

    def validate_coils_outside_domain(self) -> None:
        """Raise if any stored coil overlaps/touches the domain."""
        offenders = [n for n in self.coils if self.coil_overlaps_domain(n)]
        if offenders:
            raise ValueError(f"These coils overlap the domain: {offenders}")

    # ----------------------- Coil API -----------------------

    def add_coil(
        self,
        name: str,
        rc: float,
        drc: float,
        zc: float,
        dzc: float,
        current: float = 0.0,
        overwrite: bool = False,
    ) -> None:
        """
        Add a rectangular coil centered at (rc, zc) with extents (drc, dzc).
        Coils MUST be strictly outside the simulation domain (no touching).
        """
        if not overwrite and name in self.coils:
            raise KeyError(f"Coil '{name}' already exists. Use overwrite=True to replace.")

        # Enforce 'outside-domain' constraint
        self._assert_coil_outside_domain(rc, drc, zc, dzc)

        self.coils[name] = {
            "rc": float(rc),
            "drc": float(drc),
            "zc": float(zc),
            "dzc": float(dzc),
            "current": float(current),
        }

    def set_coil_current(self, name: str, current: float) -> None:
        if name not in self.coils:
            raise KeyError(f"Coil '{name}' not found.")
        self.coils[name]["current"] = float(current)

    def get_coil(self, name: str) -> Dict[str, float]:
        if name not in self.coils:
            raise KeyError(f"Coil '{name}' not found.")
        return dict(self.coils[name])

    def remove_coil(self, name: str) -> None:
        if name not in self.coils:
            raise KeyError(f"Coil '{name}' not found.")
        del self.coils[name]

    # ---------------- Volume field ----------------

    def compute_volume_field(self) -> np.ndarray:
        """
        Axisymmetric control-volume weights (Verboncoeur-style corrections).
        Returns shape (Nr, Nz).
        """
        Nr, Nz = self.Nr, self.Nz
        dr, dz = self.dr, self.dz

        volume_r = np.empty((Nr,), dtype=np.float64)

        # Axis node
        volume_r[0] = (np.pi / 3.0) * dr * dr * dz

        if Nr > 1:
            i_vals = np.arange(1, Nr - 1, dtype=np.float64)  # 1..Nr-2
            volume_r[1:Nr - 1] = 2.0 * np.pi * i_vals * dr * dr * dz

        # Outer ring node
        volume_r[Nr - 1] = np.pi * ((Nr - 1) - 1.0 / 3.0) * dr * dr * dz

        return np.repeat(volume_r[:, None], Nz, axis=1)

    def compute_volume_field_dep(self) -> np.ndarray:
        """
        Computes the 'Effective Weighted Volume' for particle deposition.
        
        This version performs EXACT sub-grid integration. It clips the 
        control volume of each node against the geometric obstacles 
        (Cathode, Anode) to ensure that the normalization volume exactly 
        matches the accessible volume for particles.
        
        This fixes the density drop at boundaries where the mesh does not 
        perfectly align with the object geometry.
        
        Returns
        -------
        vol_dep : np.ndarray (Nr, Nz)
        """
        Nr, Nz = self.Nr, self.Nz
        dr, dz = self.dr, self.dz
        r_nodes = self.r
        z_nodes = self.z
        
        vol_dep = np.zeros((Nr, Nz), dtype=np.float64)

        # --- 1. Define Obstacles as Rectangles [r1, r2, z1, z2] ---
        # We will subtract these from the control volumes.
        obstacles = []
        
        # Cathode (Bottom-Left)
        obstacles.append([0.0, self.rmax_cathode, 0.0, self.zmax_cathode])
        
        # Anode 1 (Ring)
        obstacles.append([self.rmin_anode, 1e9, self.zmin_anode, self.zmax_anode])
        
        # Anode 2 (Ring)
        obstacles.append([self.rmin_anode, 1e9, self.zmin_anode2, self.zmax_anode2])

        # --- 2. Integration Helpers ---
        
        def integrate_weight_r(r_a, r_b, r_node, is_inner):
            """
            Integrates W(r) * 2*pi*r dr from r_a to r_b.
            W(r) is the linear shape function for 'r_node'.
            is_inner=True  => interval [r_node-dr, r_node]
            is_inner=False => interval [r_node, r_node+dr]
            """
            if r_a >= r_b: return 0.0
            
            # Constants for polynomial integration
            # Inner: W(r) = (r - (rn-dr))/dr = (1 - rn/dr) + r/dr
            # Outer: W(r) = ((rn+dr) - r)/dr = (1 + rn/dr) - r/dr
            
            # Int (A + B*r) * r dr = A*r^2/2 + B*r^3/3
            
            if is_inner:
                A = -(r_node - dr)/dr
                B = 1.0/dr
            else:
                A = (r_node + dr)/dr
                B = -1.0/dr
            
            val_b = 2.0 * np.pi * (A * r_b**2 / 2.0 + B * r_b**3 / 3.0)
            val_a = 2.0 * np.pi * (A * r_a**2 / 2.0 + B * r_a**3 / 3.0)
            
            return val_b - val_a

        def integrate_weight_z(z_a, z_b, z_node, is_lower):
            """
            Integrates W(z) dz from z_a to z_b.
            Linear weighting in Z (flat geometry).
            """
            if z_a >= z_b: return 0.0
            
            # Inner (Lower): W(z) = (z - (zn-dz))/dz
            # Outer (Upper): W(z) = ((zn+dz) - z)/dz
            
            if is_lower:
                # Integral of (z - (zn-dz))/dz
                # = [ z^2/2 - z*(zn-dz) ] / dz
                def indef(z): return (0.5*z**2 - z*(z_node - dz)) / dz
                return indef(z_b) - indef(z_a)
            else:
                # Integral of ((zn+dz) - z)/dz
                # = [ z*(zn+dz) - z^2/2 ] / dz
                def indef(z): return (z*(z_node + dz) - 0.5*z**2) / dz
                return indef(z_b) - indef(z_a)

        def subtract_rects(rects_list, obstacle):
            """
            Subtracts obstacle [or1, or2, oz1, oz2] from a list of valid rects.
            Returns new list of valid rects.
            """
            or1, or2, oz1, oz2 = obstacle
            new_list = []
            for r1, r2, z1, z2 in rects_list:
                # Intersection
                ir1, ir2 = max(r1, or1), min(r2, or2)
                iz1, iz2 = max(z1, oz1), min(z2, oz2)
                
                if ir1 >= ir2 or iz1 >= iz2:
                    # No intersection, keep original
                    new_list.append([r1, r2, z1, z2])
                else:
                    # Overlap exists. Breakdown the original rect into remaining pieces.
                    # 1. Left of intersection
                    if r1 < ir1: new_list.append([r1, ir1, z1, z2])
                    # 2. Right of intersection
                    if ir2 < r2: new_list.append([ir2, r2, z1, z2])
                    # 3. Below intersection (bounded by intersection R width)
                    if z1 < iz1: new_list.append([ir1, ir2, z1, iz1])
                    # 4. Above intersection
                    if iz2 < z2: new_list.append([ir1, ir2, iz2, z2])
            return new_list

        # --- 3. Main Loop over Nodes ---
        for i in range(Nr):
            rn = r_nodes[i]
            for j in range(Nz):
                zn = z_nodes[j]
                
                # Skip if node itself is inside solid (no particles here)
                if self.mask[i, j] == 0:
                    vol_dep[i, j] = 1.0 
                    continue
                
                total_vol = 0.0
                
                # Loop over 4 quadrants around the node
                # Quad 0: SW (Inner R, Lower Z)
                # Quad 1: SE (Outer R, Lower Z)
                # Quad 2: NW (Inner R, Upper Z)
                # Quad 3: NE (Outer R, Upper Z)
                
                quad_configs = [
                    # (dr_sign, dz_sign, is_inner_r, is_lower_z)
                    (-0.5, -0.5, True,  True),  # SW
                    ( 0.5, -0.5, False, True),  # SE
                    (-0.5,  0.5, True,  False), # NW
                    ( 0.5,  0.5, False, False)  # NE
                ]
                
                for dr_s, dz_s, is_inner_r, is_lower_z in quad_configs:
                    # Define Quadrant Bounds
                    # Be careful with boundaries at edges of domain
                    if (is_inner_r and i == 0) or (not is_inner_r and i == Nr-1): continue
                    if (is_lower_z and j == 0) or (not is_lower_z and j == Nz-1): continue
                    
                    q_r1 = min(rn, rn + dr_s * dr * 2.0) # *2 because dr_s is 0.5
                    q_r2 = max(rn, rn + dr_s * dr * 2.0)
                    q_z1 = min(zn, zn + dz_s * dz * 2.0)
                    q_z2 = max(zn, zn + dz_s * dz * 2.0)
                    
                    # Valid Rectangles List (starts with full quadrant)
                    valid_rects = [[q_r1, q_r2, q_z1, q_z2]]
                    
                    # Subtract all obstacles
                    for obs in obstacles:
                        valid_rects = subtract_rects(valid_rects, obs)
                    
                    # Integrate volume over remaining valid rects
                    for vr1, vr2, vz1, vz2 in valid_rects:
                        vol_r = integrate_weight_r(vr1, vr2, rn, is_inner_r)
                        vol_z = integrate_weight_z(vz1, vz2, zn, is_lower_z)
                        total_vol += vol_r * vol_z
                
                # Fallback for safety
                if total_vol < 1e-20:
                    # Should only happen if mask logic failed or floating point exact match
                    # Use standard volume to avoid NaN
                    total_vol = self.volume_field[i, j]
                    
                vol_dep[i, j] = total_vol

        return vol_dep

    def find_boundary_nodes(self, mask=None):
        """
        Finds fluid nodes (mask == 1) that are adjacent to at least one
        solid node (mask == 0) using explicit Python loops.
        """
        if mask is None:
            mask = self.mask
        Nr, Nz = self.Nr, self.Nz
        
        i_bc_list: List[int] = []
        j_bc_list: List[int] = []
        
        # Define the 4-way neighbor relative coordinates (up, down, left, right)
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, 1), (1, -1)]
        
        for i in range(self.Nr):
            for j in range(self.Nz):
                # Condition 1: Must be a fluid node
                if mask[i, j] == 1:
                    is_boundary = False
                    
                    # Condition 2: Check all 4 neighbors
                    for di, dj in neighbors:
                        ni, nj = i + di, j + dj
                        
                        # Check if the neighbor is *within the grid bounds*
                        if 0 <= ni < self.Nr and 0 <= nj < self.Nz:
                            # If neighbor is within bounds, check if it's solid
                            if mask[ni, nj] == 0:
                                is_boundary = True
                                # Found one, no need to check other neighbors
                                break 
                                
                    if is_boundary:
                        i_bc_list.append(i)
                        j_bc_list.append(j)
                            
        return np.array(i_bc_list), np.array(j_bc_list)