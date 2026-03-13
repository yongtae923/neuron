# simulate_xz_potential_map.py
# X-Z grid simulation to map maximum membrane potential

import numpy as np
import os
import io
from contextlib import redirect_stdout
from neuron import h
from neuron.units import um, ms, mV
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

h.load_file("stdrun.hoc")

# --- 0. Simulation Configuration ---
ALLEN_CELL_ID = '529898751'  # Neuron model ID

# E-field scale list (will iterate through all scales)
E_FIELD_SCALES = [1, 10, 20, 30, 40, 50]  # 1x, 10x, 20x, 30x, 40x, 50x

# X-Z grid configuration (tip reference, um)
X_POSITIONS = np.arange(-300, 301, 5)  # -300, -280, ..., 300 (31 points)
Z_POSITIONS = np.arange(310, 811, 5)   # 310, 330, ..., 810 (26 points)
Y_POSITION = 42.0 * um  # Fixed Y position

# Simulation time
SIM_TIME_MS = 1.0  # 0 ~ 1 ms

# E-field method
E_FIELD_METHOD = 'integrated'  # 'simple' or 'integrated'

# --- 1. File Paths and Constants ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
E_FIELD_VALUES_FILE = os.path.join(SCRIPT_DIR, 'E_field_40cycles.npy')
E_GRID_COORDS_FILE = os.path.join(SCRIPT_DIR, 'E_field_grid_coords.npy')

# Output directory
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'simulate_xz_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TIME_STEP_US = 50.0  # Ansys data time step (us)
TIME_STEP_MS = TIME_STEP_US / 1000.0  # 0.05 ms

# E-field unit conversion
E_UNIT_SCALE = 1e6  # V/um -> V/m
E_FACTOR = 1e-3  # (V/m * um) -> mV

# --- 2. Load E-field Data ---
print("--- 1. Loading E-field data ---")

try:
    E_field_values = np.load(E_FIELD_VALUES_FILE)
    print(f"E-field values loaded: {E_field_values.shape}")
    
    if E_UNIT_SCALE != 1.0:
        E_field_values = E_field_values * E_UNIT_SCALE
        print(f"  Unit conversion: E_UNIT_SCALE = {E_UNIT_SCALE:.0e} (FEM -> V/m)")
except Exception as e:
    print(f"Error: Cannot load E-field values file: {e}")
    exit()

try:
    E_grid_coords_M = np.load(E_GRID_COORDS_FILE)
    E_grid_coords_UM = E_grid_coords_M * 1e6  # M -> um
    N_SPATIAL_POINTS = E_grid_coords_UM.shape[0]
    print(f"E-field coords loaded and converted to um: {E_grid_coords_UM.shape}")
except Exception as e:
    print(f"Error: Cannot load E-field coords file: {e}")
    exit()

# --- 3. Spatial Mapping Functions ---

def find_nearest_spatial_index(x_um, y_um, z_um, grid_coords_um):
    """Find the nearest grid point index for given (x, y, z)."""
    target_coord = np.array([x_um, y_um, z_um])
    distances_sq = np.sum((grid_coords_um - target_coord)**2, axis=1)
    nearest_index = np.argmin(distances_sq)
    return nearest_index

def xyz_at_seg(sec, segx):
    """Get 3D coordinates at segment position (segx, 0.0~1.0)."""
    n = int(h.n3d(sec=sec))
    if n < 2:
        return 0.0, 0.0, 0.0
    
    x0 = h.x3d(0, sec=sec)
    y0 = h.y3d(0, sec=sec)
    z0 = h.z3d(0, sec=sec)
    x1 = h.x3d(n-1, sec=sec)
    y1 = h.y3d(n-1, sec=sec)
    z1 = h.z3d(n-1, sec=sec)
    
    x = x0 + (x1 - x0) * segx
    y = y0 + (y1 - y0) * segx
    z = z0 + (z1 - z0) * segx
    
    return x, y, z

def translate_morphology(all_secs, dx, dy, dz):
    """Translate all section pt3d coordinates."""
    for sec in all_secs:
        n = int(h.n3d(sec=sec))
        for i in range(n):
            x = h.x3d(i, sec=sec) + dx
            y = h.y3d(i, sec=sec) + dy
            z = h.z3d(i, sec=sec) + dz
            d = h.diam3d(i, sec=sec)
            h.pt3dchange(i, x, y, z, d, sec=sec)
    h.define_shape()

# --- 4. E-field Functions ---

# Global variable for current E-field scale (updated in loop)
current_e_field_scale = 1.0

def get_E_at(spatial_idx, current_time_ms):
    """Get E-field value at specific spatial index and time."""
    global current_e_field_scale
    
    time_index_float = current_time_ms / TIME_STEP_MS
    Tmax = E_field_values.shape[2] - 1
    
    t_idx_prev = int(math.floor(time_index_float))
    if t_idx_prev < 0:
        t_idx_prev = 0
    if t_idx_prev > Tmax:
        t_idx_prev = Tmax
    t_idx_next = min(t_idx_prev + 1, Tmax)
    
    ratio = time_index_float - t_idx_prev
    if ratio < 0.0:
        ratio = 0.0
    if ratio > 1.0:
        ratio = 1.0

    Ex_prev = E_field_values[0, spatial_idx, t_idx_prev]
    Ex_next = E_field_values[0, spatial_idx, t_idx_next]
    Ez_prev = E_field_values[1, spatial_idx, t_idx_prev]
    Ez_next = E_field_values[1, spatial_idx, t_idx_next]

    Ex = Ex_prev + ratio * (Ex_next - Ex_prev)
    Ez = Ez_prev + ratio * (Ez_next - Ez_prev)

    Ex *= current_e_field_scale
    Ez *= current_e_field_scale
    
    return Ex, 0.0, Ez

def interp_phi(arc_list, phi_list, target_arc):
    """Interpolate phi value at target_arc position."""
    if len(arc_list) == 0 or len(phi_list) == 0:
        return 0.0
    if target_arc <= arc_list[0]:
        return phi_list[0]
    if target_arc >= arc_list[-1]:
        return phi_list[-1]

    lo, hi = 0, len(arc_list) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if arc_list[mid] <= target_arc:
            lo = mid
        else:
            hi = mid

    a0, a1 = arc_list[lo], arc_list[hi]
    p0, p1 = phi_list[lo], phi_list[hi]
    if a1 == a0:
        return p0
    w = (target_arc - a0) / (a1 - a0)
    return p0 + w * (p1 - p0)

def build_morph_cache(neuron, grid_coords_um):
    """Build morphology cache for neuron (pt3d-based)."""
    cache = {}
    topo = {}

    for sec in neuron.all:
        n = int(h.n3d(sec=sec))
        if n < 2:
            cache[sec] = {"n": n, "arc": [0.0], "dl": [], "mid_spidx": []}
            continue

        xs = [h.x3d(i, sec=sec) for i in range(n)]
        ys = [h.y3d(i, sec=sec) for i in range(n)]
        zs = [h.z3d(i, sec=sec) for i in range(n)]
        arc = [h.arc3d(i, sec=sec) for i in range(n)]

        dl = []
        mid_spidx = []
        for i in range(n - 1):
            dx = xs[i+1] - xs[i]
            dy = ys[i+1] - ys[i]
            dz = zs[i+1] - zs[i]
            dl.append((dx, dy, dz))

            mx = 0.5 * (xs[i] + xs[i+1])
            my = 0.5 * (ys[i] + ys[i+1])
            mz = 0.5 * (zs[i] + zs[i+1])

            spidx = find_nearest_spatial_index(mx, my, mz, grid_coords_um)
            mid_spidx.append(spidx)

        cache[sec] = {"n": n, "arc": arc, "dl": dl, "mid_spidx": mid_spidx}

        sref = h.SectionRef(sec=sec)
        if sref.has_parent():
            try:
                pseg = sref.parentseg()
                topo[sec] = (pseg.sec, float(pseg.x))
            except:
                parent_sec = sref.parent
                if n > 0:
                    child_x0 = xs[0]
                    child_y0 = ys[0]
                    child_z0 = zs[0]
                    
                    pn = int(h.n3d(sec=parent_sec))
                    min_dist = float('inf')
                    parent_x = 0.0
                    
                    for pi in range(pn):
                        px = h.x3d(pi, sec=parent_sec)
                        py = h.y3d(pi, sec=parent_sec)
                        pz = h.z3d(pi, sec=parent_sec)
                        dist = ((px - child_x0)**2 + (py - child_y0)**2 + (pz - child_z0)**2)**0.5
                        if dist < min_dist:
                            min_dist = dist
                            parc = h.arc3d(pi, sec=parent_sec)
                            total_parc = h.arc3d(pn-1, sec=parent_sec) if pn > 0 else 1.0
                            parent_x = parc / total_parc if total_parc > 0 else 0.0
                    
                    topo[sec] = (parent_sec, parent_x)
                else:
                    topo[sec] = (parent_sec, 1.0)
        else:
            topo[sec] = None

    return cache, topo

def compute_phi_sections(neuron, morph_cache, topo, current_time_ms):
    """Compute phi by cumulative integration along section tree."""
    phi_sec = {}

    def ensure_section_phi(sec):
        if sec in phi_sec:
            return

        parent = topo.get(sec, None)
        if parent is None:
            phi0 = 0.0
        else:
            psec, px = parent
            ensure_section_phi(psec)
            parc, pphi = phi_sec[psec]
            total_arc = parc[-1] if len(parc) > 0 else 0.0
            phi0 = interp_phi(parc, pphi, px * total_arc)

        data = morph_cache[sec]
        n = data["n"]
        if n < 2:
            phi_sec[sec] = ([0.0], [phi0])
            return

        arc = data["arc"]
        dl = data["dl"]
        mid_spidx = data["mid_spidx"]

        phis = [phi0]
        for i in range(n - 1):
            spidx = mid_spidx[i]
            Ex, Ey, Ez = get_E_at(spidx, current_time_ms)
            dx, dy, dz = dl[i]
            dphi = -(Ex * dx + Ey * dy + Ez * dz) * E_FACTOR  # mV
            phis.append(phis[-1] + dphi)

        phi_sec[sec] = (arc, phis)

    for sec in neuron.all:
        ensure_section_phi(sec)

    return phi_sec

def apply_phi_to_segments(neuron, phi_sec):
    """Apply computed phi to each segment's e_extracellular."""
    for sec in neuron.all:
        if sec not in phi_sec:
            continue
        arc, phis = phi_sec[sec]
        total_arc = arc[-1] if len(arc) > 0 else 0.0
        for seg in sec:
            target_arc = seg.x * total_arc
            phi = interp_phi(arc, phis, target_arc)
            seg.e_extracellular = phi

# --- 5. Import Neuron Model ---
from allen_neuron_model import AllenNeuronModel, set_allen_cell_id

set_allen_cell_id(ALLEN_CELL_ID)

# --- 6. Main Simulation Loop (optimized: neuron created once per position) ---
print("\n--- 2. X-Z Grid Simulation ---")
print(f"X positions: {X_POSITIONS} um")
print(f"Z positions: {Z_POSITIONS} um")
print(f"Total grid points: {len(X_POSITIONS)} x {len(Z_POSITIONS)} = {len(X_POSITIONS) * len(Z_POSITIONS)}")
print(f"Simulation time: 0 ~ {SIM_TIME_MS} ms")
print(f"E-field scales: {E_FIELD_SCALES}")

# Warmup time for stabilization
WARMUP_TIME_MS = 50.0  # Reduced warmup for faster simulation

# Storage for all scale results: {scale: 2D array of max Vm}
all_results = {scale: np.zeros((len(Z_POSITIONS), len(X_POSITIONS))) for scale in E_FIELD_SCALES}

# Total simulations = positions x scales
total_sims = len(X_POSITIONS) * len(Z_POSITIONS) * len(E_FIELD_SCALES)
print(f"Total simulations: {len(X_POSITIONS) * len(Z_POSITIONS)} positions x {len(E_FIELD_SCALES)} scales = {total_sims}")

# Progress bar for entire simulation
pbar = tqdm(total=total_sims, desc="X-Z Grid Simulation", unit="sim")

# Iterate over positions first, then scales (neuron created once per position)
for zi, z_pos in enumerate(Z_POSITIONS):
    for xi, x_pos in enumerate(X_POSITIONS):
        # Target position
        target_x = float(x_pos)
        target_y = float(Y_POSITION)
        target_z = float(z_pos)
        
        # Clear all sections (reset NEURON)
        for sec in h.allsec():
            h.delete_section(sec=sec)
        
        # Create neuron at origin first (ONCE per position)
        # Suppress verbose output during neuron creation
        with redirect_stdout(io.StringIO()):
            neuron = AllenNeuronModel(x=0, y=0, z=0, cell_id=ALLEN_CELL_ID)
        
        # Get current soma position and translate to target
        sx, sy, sz = xyz_at_seg(neuron.soma, 0.5)
        translate_morphology(neuron.all, target_x - sx, target_y - sy, target_z - sz)
        
        # Build morphology cache for integrated method (ONCE per position)
        morph_cache = None
        topo = None
        if E_FIELD_METHOD == 'integrated':
            morph_cache, topo = build_morph_cache(neuron, E_grid_coords_UM)
        
        # Run simulation for each E-field scale at this position
        for E_FIELD_SCALE in E_FIELD_SCALES:
            # Update global scale variable
            current_e_field_scale = float(E_FIELD_SCALE)
            
            # Simulation setup (re-initialize for each scale)
            h.tstop = WARMUP_TIME_MS + SIM_TIME_MS
            h.dt = 0.025 * ms
            h.celsius = 34.0
            h.finitialize(-65.0 * mV)
            
            # Warmup phase (no E-field)
            while h.t < WARMUP_TIME_MS:
                for sec in neuron.all:
                    for seg in sec:
                        seg.e_extracellular = 0.0
                h.fadvance()
            
            # Track max Vm during E-field phase
            max_vm = float('-inf')
            
            # E-field application phase (0 ~ 1 ms)
            while h.t < h.tstop:
                current_time_ms = h.t - WARMUP_TIME_MS
                
                if current_time_ms >= 0:
                    if E_FIELD_METHOD == 'integrated' and morph_cache is not None and topo is not None:
                        phi_sec = compute_phi_sections(neuron, morph_cache, topo, current_time_ms)
                        apply_phi_to_segments(neuron, phi_sec)
                    else:
                        # Simple method (phi = -(E·r))
                        for sec in neuron.all:
                            for seg in sec:
                                seg_x, seg_y, seg_z = xyz_at_seg(sec, seg.x)
                                spatial_idx = find_nearest_spatial_index(seg_x, seg_y, seg_z, E_grid_coords_UM)
                                Ex, Ey, Ez = get_E_at(spatial_idx, current_time_ms)
                                phi_mV = -(Ex * seg_x + Ey * seg_y + Ez * seg_z) * E_FACTOR
                                seg.e_extracellular = phi_mV
                
                # Record Vm at each step (manual recording like simulate_tES.py)
                current_vm = neuron.soma(0.5).v
                if current_vm > max_vm:
                    max_vm = current_vm
                
                h.fadvance()
            
            # Handle case where no valid Vm was recorded
            if max_vm == float('-inf'):
                max_vm = np.nan
            
            # Store result for this scale
            all_results[E_FIELD_SCALE][zi, xi] = max_vm
            
            # Update progress
            pbar.update(1)
            pbar.set_postfix({'x': target_x, 'z': target_z, 'scale': f'{E_FIELD_SCALE}x', 'max_Vm': f'{max_vm:.3f} mV'})

pbar.close()

# Print summary for each scale
for E_FIELD_SCALE in E_FIELD_SCALES:
    max_vm_grid = all_results[E_FIELD_SCALE]
    print(f"\n--- E-field {E_FIELD_SCALE}x Complete ---")
    print(f"Max Vm grid shape: {max_vm_grid.shape}")
    print(f"Max Vm range: {np.nanmin(max_vm_grid):.3f} ~ {np.nanmax(max_vm_grid):.3f} mV")
    
    # --- 7. Plot Results for this scale ---
    print(f"\n--- Creating X-Z Potential Map ({E_FIELD_SCALE}x) ---")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create meshgrid for pcolormesh
    X, Z = np.meshgrid(X_POSITIONS, Z_POSITIONS)
    
    # Plot heatmap
    im = ax.pcolormesh(X, Z, max_vm_grid, cmap='hot', shading='auto')
    cbar = fig.colorbar(im, ax=ax, label='Max Membrane Potential (mV)')
    
    # Add contour lines
    contour = ax.contour(X, Z, max_vm_grid, colors='white', alpha=0.5, linewidths=0.5)
    ax.clabel(contour, inline=True, fontsize=8, fmt='%.2f')
    
    # Labels and title
    ax.set_xlabel('X Position (um)', fontsize=12)
    ax.set_ylabel('Z Position (um)', fontsize=12)
    ax.set_title(f'Maximum Membrane Potential Map (0-{SIM_TIME_MS} ms)\n'
                 f'Neuron: {ALLEN_CELL_ID}, E-field Scale: {E_FIELD_SCALE}x', fontsize=14)
    
    # Grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Mark soma positions on grid
    for zi_plot, z_pos in enumerate(Z_POSITIONS):
        for xi_plot, x_pos in enumerate(X_POSITIONS):
            ax.plot(x_pos, z_pos, 'ko', markersize=3, alpha=0.5)
    
    plt.tight_layout()
    
    # Save figure
    output_filename = f'xz_potential_map_{ALLEN_CELL_ID}_scale{E_FIELD_SCALE:.0f}x.png'
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to save memory
    print(f"Saved: {output_path}")
    
    # Also save data as numpy array
    data_filename = f'xz_potential_data_{ALLEN_CELL_ID}_scale{E_FIELD_SCALE:.0f}x.npz'
    data_path = os.path.join(OUTPUT_DIR, data_filename)
    np.savez(data_path, 
             x_positions=X_POSITIONS, 
             z_positions=Z_POSITIONS, 
             max_vm_grid=max_vm_grid,
             e_field_scale=E_FIELD_SCALE,
             cell_id=ALLEN_CELL_ID)
    print(f"Saved data: {data_path}")

# --- 8. Create comparison plot (all scales in one figure) ---
print("\n--- Creating Comparison Plot (All Scales) ---")

n_scales = len(E_FIELD_SCALES)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # 2x3 for 6 scales
axes = np.array(axes).flatten()

# Find global min/max for consistent colorbar
global_vmin = min(np.nanmin(all_results[s]) for s in E_FIELD_SCALES)
global_vmax = max(np.nanmax(all_results[s]) for s in E_FIELD_SCALES)

X, Z = np.meshgrid(X_POSITIONS, Z_POSITIONS)

for idx, scale in enumerate(E_FIELD_SCALES):
    ax = axes[idx]
    im = ax.pcolormesh(X, Z, all_results[scale], cmap='hot', shading='auto',
                       vmin=global_vmin, vmax=global_vmax)
    ax.set_xlabel('X Position (um)', fontsize=10)
    ax.set_ylabel('Z Position (um)', fontsize=10)
    ax.set_title(f'E-field Scale: {scale}x', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Mark grid points
    for zi_plot, z_pos in enumerate(Z_POSITIONS):
        for xi_plot, x_pos in enumerate(X_POSITIONS):
            ax.plot(x_pos, z_pos, 'ko', markersize=2, alpha=0.3)

# Add shared colorbar
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes((0.88, 0.15, 0.03, 0.7))
cbar = fig.colorbar(im, cax=cbar_ax, label='Max Membrane Potential (mV)')

fig.suptitle(f'Maximum Membrane Potential Map Comparison (0-{SIM_TIME_MS} ms)\n'
             f'Neuron: {ALLEN_CELL_ID}', fontsize=14, fontweight='bold')

plt.tight_layout(rect=(0, 0, 0.85, 0.95))

# Save comparison figure
comparison_filename = f'xz_potential_map_{ALLEN_CELL_ID}_comparison.png'
comparison_path = os.path.join(OUTPUT_DIR, comparison_filename)
plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved comparison: {comparison_path}")

# Save all results in one file
all_data_filename = f'xz_potential_data_{ALLEN_CELL_ID}_all_scales.npz'
all_data_path = os.path.join(OUTPUT_DIR, all_data_filename)
np.savez(all_data_path, 
         x_positions=X_POSITIONS, 
         z_positions=Z_POSITIONS, 
         e_field_scales=np.array(E_FIELD_SCALES),
         cell_id=ALLEN_CELL_ID,
         **{f'max_vm_grid_{s}x': all_results[s] for s in E_FIELD_SCALES})
print(f"Saved all data: {all_data_path}")

print("\n" + "="*70)
print("All simulations complete!")
print(f"Results saved to: {OUTPUT_DIR}")
print("="*70)
