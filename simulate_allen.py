# simulate_allen.py
"""
Allen model + E-field simulation at multiple spatial positions.

- For each downsampled E-field grid point (outside coil, within ROI),
  run a full neuron simulation and record Vm/vext at soma.
- Results: 3D Vm/vext arrays (position x time) saved as npy
- Time-lapse 3D plots with common colorbar
"""

import warnings

# Suppress numpy.longdouble warning
warnings.filterwarnings("ignore", message=".*Signature.*numpy.longdouble.*", category=UserWarning)

import numpy as np
import os
import sys
import math
import shutil
from tqdm import tqdm
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# --- Configuration ---
CELL_ID = "529898751"  # Allen model cell ID

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
E_FIELD_VALUES_FILE = os.path.join(SCRIPT_DIR, "efield", "E_field_4cycle.npy")
E_GRID_COORDS_FILE = os.path.join(SCRIPT_DIR, "efield", "E_field_grid_coords.npy")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Delete outputs option ---
if "-d" in sys.argv[1:]:
    deleted = 0
    for fname in os.listdir(OUTPUT_DIR):
        fpath = os.path.join(OUTPUT_DIR, fname)
        try:
            os.remove(fpath)
            deleted += 1
        except OSError:
            pass
    print(f"Deleted {deleted} files from '{OUTPUT_DIR}'.")
    sys.exit(0)

# --- Time settings ---
TIME_STEP_US = 50.0   # E-field data time step (us)
TIME_STEP_MS = TIME_STEP_US / 1000.0  # 0.05 ms
DT_MS = 0.05   # Simulation dt (ms)
WARMUP_TIME_MS = 100.0   # Warmup without E-field (ms)

# Time ROI (ms): simulate this range
TIME_ROI_MS = (0.0, 0.5)  # (start, end) or None for full

# E-field: npy is already V/m
E_FIELD_SCALE = 1.0
E_FACTOR = 1e-3  # (V/m * um) -> mV

# --- Space settings ---
X_ROI_UM = None  # e.g. (-200, 200) or None for full
Y_ROI_UM = None
Z_ROI_UM = None

# Coil region to exclude (um)
COIL_X_MIN, COIL_X_MAX = -79.5, 79.5
COIL_Y_MIN, COIL_Y_MAX = -32.0, 32.0
COIL_Z_MIN, COIL_Z_MAX = 498.0, 1502.0

# Adaptive grid spacing (um)
# |E|가 0에서 멀수록 더 촘촘하게 샘플링
GRID_SPACING_MAX_UM = 100.0   # 기본 (|E| ≈ 0인 지역)
GRID_SPACING_MIN_UM = 5.0     # 최소 (|E| 최대인 지역)

# 3D plot settings
COLORMAP_VM = "RdBu_r"
COLORMAP_VEXT = "viridis"
POINT_SIZE = 8.0
POINT_ALPHA = 0.9
VIEW_ELEV = 25
VIEW_AZIM = 48  # 기존보다 오른쪽으로 3도 회전


def get_valid_efield_indices(coords_um):
    """Return indices within ROI and outside coil."""
    x_um = coords_um[:, 0]
    y_um = coords_um[:, 1]
    z_um = coords_um[:, 2]
    
    roi_mask = np.ones(coords_um.shape[0], dtype=bool)
    if X_ROI_UM is not None:
        roi_mask &= (x_um >= X_ROI_UM[0]) & (x_um <= X_ROI_UM[1])
    if Y_ROI_UM is not None:
        roi_mask &= (y_um >= Y_ROI_UM[0]) & (y_um <= Y_ROI_UM[1])
    if Z_ROI_UM is not None:
        roi_mask &= (z_um >= Z_ROI_UM[0]) & (z_um <= Z_ROI_UM[1])
    
    inside_coil = (
        (x_um >= COIL_X_MIN) & (x_um <= COIL_X_MAX)
        & (y_um >= COIL_Y_MIN) & (y_um <= COIL_Y_MAX)
        & (z_um >= COIL_Z_MIN) & (z_um <= COIL_Z_MAX)
    )
    
    return np.where(roi_mask & (~inside_coil))[0]


def compute_voxel_representatives(coords_um, valid_idx, spacing_um):
    """Pick one representative index per voxel (uniform spacing)."""
    if spacing_um is None or spacing_um <= 0:
        return valid_idx
    pts = coords_um[valid_idx]
    mins = pts.min(axis=0)
    v = np.floor((pts - mins) / spacing_um).astype(np.int32)
    nx = int(v[:, 0].max()) + 1
    ny = int(v[:, 1].max()) + 1
    key = (v[:, 0].astype(np.int64) + v[:, 1].astype(np.int64) * nx + v[:, 2].astype(np.int64) * nx * ny)
    _, first_pos = np.unique(key, return_index=True)
    return valid_idx[first_pos]


def compute_adaptive_representatives(coords_um, valid_idx, E_magnitude, spacing_max, spacing_min):
    """
    Adaptive spatial sampling: denser where |E| is larger.
    
    - spacing_max: 기본 간격 (|E| ≈ 0인 지역)
    - spacing_min: 최소 간격 (|E| 최대인 지역)
    - E_magnitude[i]: valid_idx[i]에서의 |E| (시간 최대값 등)
    """
    if spacing_max <= 0:
        return valid_idx
    
    pts = coords_um[valid_idx]
    emag = E_magnitude[valid_idx] if E_magnitude is not None else np.zeros(len(valid_idx))
    
    # |E|를 0~1로 정규화
    e_min, e_max = emag.min(), emag.max()
    if e_max > e_min:
        e_norm = (emag - e_min) / (e_max - e_min)
    else:
        e_norm = np.zeros_like(emag)
    
    # 각 포인트의 "실효 간격": |E|가 클수록 작은 간격
    # spacing = spacing_max - (spacing_max - spacing_min) * e_norm
    # 즉, e_norm=0 → spacing_max, e_norm=1 → spacing_min
    
    # 전략: 여러 단계의 간격으로 누적 샘플링
    # 1단계: spacing_max로 전체 샘플링 (기본)
    # 2~N단계: |E|가 높은 지역만 더 작은 간격으로 추가 샘플링
    
    selected = set()
    mins = pts.min(axis=0)
    
    # 간격 단계 (큰 것부터)
    spacing_levels = []
    s = spacing_max
    while s >= spacing_min:
        spacing_levels.append(s)
        s = s / 2
    if spacing_levels[-1] > spacing_min:
        spacing_levels.append(spacing_min)
    
    print(f"    Adaptive grid levels: {[f'{s:.0f}um' for s in spacing_levels]}")
    
    for level_idx, spacing in enumerate(spacing_levels):
        # 이 간격 이하가 필요한 포인트들의 임계값
        # e_norm이 높을수록 작은 간격 필요
        # spacing = spacing_max - (spacing_max - spacing_min) * e_norm
        # → e_norm = (spacing_max - spacing) / (spacing_max - spacing_min)
        if spacing_max > spacing_min:
            threshold = (spacing_max - spacing) / (spacing_max - spacing_min)
        else:
            threshold = 0.0
        
        # 이 레벨에서 샘플링할 포인트: e_norm >= threshold
        mask = e_norm >= threshold
        if not np.any(mask):
            continue
        
        sub_pts = pts[mask]
        sub_idx = valid_idx[mask]
        
        # 해당 간격으로 복셀화
        v = np.floor((sub_pts - mins) / spacing).astype(np.int32)
        nx = int(v[:, 0].max()) + 1
        ny = int(v[:, 1].max()) + 1
        key = (v[:, 0].astype(np.int64) + v[:, 1].astype(np.int64) * nx + v[:, 2].astype(np.int64) * nx * ny)
        _, first_pos = np.unique(key, return_index=True)
        
        for p in first_pos:
            selected.add(int(sub_idx[p]))
    
    result = np.array(sorted(selected), dtype=int)
    print(f"    Adaptive sampling: {len(result)} points selected")
    return result


# ===========================================================================
# Load E-field data FIRST (before NEURON import to avoid conflicts)
# ===========================================================================
print("=" * 60)
print("Allen Neuron Multi-Position Simulation")
print("=" * 60)

print("\n--- Load E-field data ---")
E_field_values = np.load(E_FIELD_VALUES_FILE)  # (3, N_spatial, N_time) V/m
coords_m = np.load(E_GRID_COORDS_FILE)
E_grid_coords_UM = coords_m * 1e6  # m -> um
E_grid_tree = cKDTree(E_grid_coords_UM)

print(f"  E-field shape: {E_field_values.shape}")
print(f"  Grid coords: {E_grid_coords_UM.shape}")
print(f"  E-field unit: V/m")

# Compute time indices from TIME_ROI_MS
n_time_total = E_field_values.shape[2]
total_data_ms = (n_time_total - 1) * TIME_STEP_MS

if TIME_ROI_MS is not None:
    t_start_ms = TIME_ROI_MS[0]
    t_end_ms = min(TIME_ROI_MS[1], total_data_ms)
else:
    t_start_ms = 0.0
    t_end_ms = total_data_ms

start_tidx = max(0, int(round(t_start_ms / TIME_STEP_MS)))
end_tidx = max(start_tidx, int(round(t_end_ms / TIME_STEP_MS)))
time_indices = np.arange(start_tidx, end_tidx + 1, dtype=int)
time_ms_arr = time_indices * TIME_STEP_MS

TOTAL_TIME_MS = t_end_ms - t_start_ms
TIME_START_MS = t_start_ms

print(f"  Time ROI: {TIME_ROI_MS} -> {len(time_indices)} frames ({TOTAL_TIME_MS:.2f} ms)")

# Compute |E| magnitude for adaptive sampling
# Use time-maximum of |E| at each spatial point (within TIME_ROI)
print("\n--- Compute |E| for adaptive sampling ---")
E_mag_time_max = np.zeros(E_grid_coords_UM.shape[0], dtype=np.float32)
for t_idx in tqdm(time_indices, desc="Compute |E|", unit="frame", ncols=80):
    Ex = E_field_values[0, :, t_idx]
    Ey = E_field_values[1, :, t_idx] if E_field_values.shape[0] >= 3 else np.zeros_like(Ex)
    Ez = E_field_values[2, :, t_idx] if E_field_values.shape[0] >= 3 else E_field_values[1, :, t_idx]
    emag = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    E_mag_time_max = np.maximum(E_mag_time_max, emag)

print(f"  |E| range: {E_mag_time_max.min():.6f} ~ {E_mag_time_max.max():.6f} V/m")

# Compute spatial positions with adaptive sampling
print("\n--- Adaptive spatial sampling ---")
print(f"  Grid spacing: {GRID_SPACING_MAX_UM}um (sparse) ~ {GRID_SPACING_MIN_UM}um (dense)")
valid_idx = get_valid_efield_indices(E_grid_coords_UM)
ds_idx = compute_adaptive_representatives(
    E_grid_coords_UM, valid_idx, E_mag_time_max,
    GRID_SPACING_MAX_UM, GRID_SPACING_MIN_UM
)
positions_um = E_grid_coords_UM[ds_idx]  # (N_pos, 3)

n_positions = len(ds_idx)
n_times = len(time_indices)

print(f"  Total simulations: {n_positions}")

# ===========================================================================
# Now import NEURON
# ===========================================================================
from neuron import h
from neuron.units import mV
h.load_file("stdrun.hoc")

from model_allen_neuron import AllenNeuronModel


def find_nearest_spatial_index(x_um, y_um, z_um):
    """Return index of nearest E-field grid point."""
    return int(E_grid_tree.query(np.array([[x_um, y_um, z_um]]), k=1)[1][0])


def xyz_at_seg(sec, segx):
    """Return 3D coordinates at segment position segx (0~1)."""
    n = int(h.n3d(sec=sec))
    if n < 2:
        return 0.0, 0.0, 0.0
    x0, y0, z0 = h.x3d(0, sec=sec), h.y3d(0, sec=sec), h.z3d(0, sec=sec)
    x1, y1, z1 = h.x3d(n - 1, sec=sec), h.y3d(n - 1, sec=sec), h.z3d(n - 1, sec=sec)
    return x0 + (x1 - x0) * segx, y0 + (y1 - y0) * segx, z0 + (z1 - z0) * segx


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


def get_E_at(spatial_idx, current_time_ms):
    """Return (Ex, Ey, Ez) at given spatial index and time."""
    actual_time_ms = current_time_ms + TIME_START_MS
    time_index_float = actual_time_ms / TIME_STEP_MS
    Tmax = E_field_values.shape[2] - 1
    t_idx_prev = int(math.floor(time_index_float))
    t_idx_prev = max(0, min(t_idx_prev, Tmax))
    t_idx_next = min(t_idx_prev + 1, Tmax)
    ratio = time_index_float - t_idx_prev
    ratio = max(0.0, min(1.0, ratio))

    n_comp = E_field_values.shape[0]
    if n_comp >= 3:
        Ex = E_field_values[0, spatial_idx, t_idx_prev] + ratio * (
            E_field_values[0, spatial_idx, t_idx_next] - E_field_values[0, spatial_idx, t_idx_prev]
        )
        Ey = E_field_values[1, spatial_idx, t_idx_prev] + ratio * (
            E_field_values[1, spatial_idx, t_idx_next] - E_field_values[1, spatial_idx, t_idx_prev]
        )
        Ez = E_field_values[2, spatial_idx, t_idx_prev] + ratio * (
            E_field_values[2, spatial_idx, t_idx_next] - E_field_values[2, spatial_idx, t_idx_prev]
        )
    else:
        Ex = E_field_values[0, spatial_idx, t_idx_prev] + ratio * (
            E_field_values[0, spatial_idx, t_idx_next] - E_field_values[0, spatial_idx, t_idx_prev]
        )
        Ey = 0.0
        Ez = E_field_values[1, spatial_idx, t_idx_prev] + ratio * (
            E_field_values[1, spatial_idx, t_idx_next] - E_field_values[1, spatial_idx, t_idx_prev]
        )
    return Ex * E_FIELD_SCALE, Ey * E_FIELD_SCALE, Ez * E_FIELD_SCALE


def interp_phi(arc_list, phi_list, target_arc):
    """Interpolate phi at target_arc."""
    if not arc_list or not phi_list:
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


def build_morph_cache(neuron_model):
    """Build morphology cache for integrated E-field."""
    cache = {}
    topo = {}
    for sec in neuron_model.all:
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
            dl.append((xs[i + 1] - xs[i], ys[i + 1] - ys[i], zs[i + 1] - zs[i]))
            mx = 0.5 * (xs[i] + xs[i + 1])
            my = 0.5 * (ys[i] + ys[i + 1])
            mz = 0.5 * (zs[i] + zs[i + 1])
            mid_spidx.append(find_nearest_spatial_index(mx, my, mz))
        cache[sec] = {"n": n, "arc": arc, "dl": dl, "mid_spidx": mid_spidx}
        sref = h.SectionRef(sec=sec)
        if sref.has_parent():
            try:
                pseg = sref.parentseg()
                topo[sec] = (pseg.sec, float(pseg.x))
            except Exception:
                parent_sec = sref.parent
                if n > 0:
                    child_x0, child_y0, child_z0 = xs[0], ys[0], zs[0]
                    pn = int(h.n3d(sec=parent_sec))
                    min_dist = float("inf")
                    parent_x = 0.0
                    for pi in range(pn):
                        px = h.x3d(pi, sec=parent_sec)
                        py = h.y3d(pi, sec=parent_sec)
                        pz = h.z3d(pi, sec=parent_sec)
                        dist = ((px - child_x0) ** 2 + (py - child_y0) ** 2 + (pz - child_z0) ** 2) ** 0.5
                        if dist < min_dist:
                            min_dist = dist
                            parc = h.arc3d(pi, sec=parent_sec)
                            total_parc = h.arc3d(pn - 1, sec=parent_sec) if pn > 0 else 1.0
                            parent_x = parc / total_parc if total_parc > 0 else 0.0
                    topo[sec] = (parent_sec, parent_x)
                else:
                    topo[sec] = (parent_sec, 1.0)
        else:
            topo[sec] = None
    return cache, topo


def compute_phi_sections(neuron_model, morph_cache, topo, current_time_ms):
    """Compute extracellular potential phi along section tree."""
    phi_sec = {}

    def ensure_section_phi(sec):
        if sec in phi_sec:
            return
        parent = topo.get(sec)
        if parent is None:
            phi0 = 0.0
        else:
            psec, px = parent
            ensure_section_phi(psec)
            parc, pphi = phi_sec[psec]
            total_arc = parc[-1] if parc else 0.0
            phi0 = interp_phi(parc, pphi, px * total_arc)
        data = morph_cache[sec]
        n = data["n"]
        if n < 2:
            phi_sec[sec] = ([0.0], [phi0])
            return
        arc, dl, mid_spidx = data["arc"], data["dl"], data["mid_spidx"]
        phis = [phi0]
        for i in range(n - 1):
            Ex, Ey, Ez = get_E_at(mid_spidx[i], current_time_ms)
            dx, dy, dz = dl[i]
            dphi = -(Ex * dx + Ey * dy + Ez * dz) * E_FACTOR
            phis.append(phis[-1] + dphi)
        phi_sec[sec] = (arc, phis)

    for sec in neuron_model.all:
        ensure_section_phi(sec)
    return phi_sec


def apply_phi_to_segments(neuron_model, phi_sec):
    """Set e_extracellular on each segment."""
    for sec in neuron_model.all:
        if sec not in phi_sec:
            continue
        arc, phis = phi_sec[sec]
        total_arc = arc[-1] if arc else 0.0
        for seg in sec:
            target_arc = seg.x * total_arc
            seg.e_extracellular = interp_phi(arc, phis, target_arc)


def run_simulation_at_position(pos_um, record_times_ms, verbose=False):
    """
    Run a single neuron simulation at the given position.
    Returns (Vm_arr, vext_arr) for each time in record_times_ms.
    """
    # Create neuron (suppress repeated output)
    if verbose:
        neuron_model = AllenNeuronModel(x=0, y=0, z=0, cell_id=CELL_ID)
    else:
        import io
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            neuron_model = AllenNeuronModel(x=0, y=0, z=0, cell_id=CELL_ID)
    
    # Move to target position
    sx, sy, sz = xyz_at_seg(neuron_model.soma, 0.5)
    tx, ty, tz = pos_um
    translate_morphology(neuron_model.all, tx - sx, ty - sy, tz - sz)
    
    # Build morph cache
    morph_cache, topo = build_morph_cache(neuron_model)
    
    # Simulation setup
    h.tstop = WARMUP_TIME_MS + TOTAL_TIME_MS + DT_MS
    h.dt = DT_MS
    h.celsius = 34.0
    h.finitialize(-65.0 * mV)
    
    # Warmup (no E-field)
    while h.t < WARMUP_TIME_MS - 1e-9:
        for sec in neuron_model.all:
            for seg in sec:
                seg.e_extracellular = 0.0
        h.fadvance()
    
    # Record Vm/vext at specified times
    Vm_list = []
    vext_list = []
    
    rec_idx = 0
    while h.t < h.tstop - 1e-9 and rec_idx < len(record_times_ms):
        t_rel = h.t - WARMUP_TIME_MS
        
        # Apply E-field
        if t_rel >= 0:
            phi_sec = compute_phi_sections(neuron_model, morph_cache, topo, t_rel)
            apply_phi_to_segments(neuron_model, phi_sec)
        else:
            for sec in neuron_model.all:
                for seg in sec:
                    seg.e_extracellular = 0.0
        
        # Record if this is a target time
        if rec_idx < len(record_times_ms) and abs(t_rel - record_times_ms[rec_idx]) < DT_MS / 2:
            Vm_list.append(neuron_model.soma(0.5).v)
            try:
                vext_list.append(neuron_model.soma(0.5).e_extracellular)
            except Exception:
                vext_list.append(0.0)
            rec_idx += 1
        
        h.fadvance()
    
    # Pad if needed
    while len(Vm_list) < len(record_times_ms):
        Vm_list.append(Vm_list[-1] if Vm_list else -65.0)
        vext_list.append(vext_list[-1] if vext_list else 0.0)
    
    return np.array(Vm_list), np.array(vext_list)


# ===========================================================================
# Estimate time with a test simulation
# ===========================================================================
import time as time_module

print(f"\n--- Estimate simulation time ---")
print(f"  Spatial points: {n_positions}")
print(f"  Time frames: {n_times}")
print(f"  Each simulation: {WARMUP_TIME_MS}ms warmup + {TOTAL_TIME_MS:.2f}ms with E-field")

# Record times relative to warmup
record_times_ms = time_ms_arr - TIME_START_MS

# Format time nicely
def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f} sec"
    elif seconds < 3600:
        return f"{seconds / 60:.1f} min"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"

# Run a test simulation to estimate time per position
print("  Running test simulation (1st position)...")
test_pos = positions_um[0]
t_start = time_module.time()
test_Vm, test_vext = run_simulation_at_position(test_pos, record_times_ms, verbose=True)
t_elapsed = time_module.time() - t_start

time_per_sim_sec = t_elapsed
total_estimated_sec = time_per_sim_sec * n_positions

print(f"  Time per simulation: {format_time(time_per_sim_sec)}")
print(f"  Estimated total time: {format_time(total_estimated_sec)} ({n_positions} simulations)")

# ===========================================================================
# Main: Run simulations at all positions
# ===========================================================================
print(f"\n--- Running {n_positions} simulations ---")

# Results arrays
Vm_data = np.zeros((n_positions, n_times), dtype=np.float32)
vext_data = np.zeros((n_positions, n_times), dtype=np.float32)

# First position already done in test (reuse result)
Vm_data[0, :] = test_Vm
vext_data[0, :] = test_vext

# Main loop (start from 1, first already done)
if n_positions > 1:
    with tqdm(total=n_positions - 1, desc="Simulations", unit="pos", ncols=100) as pbar:
        for i in range(1, n_positions):
            pos = positions_um[i]
            pbar.set_postfix_str(f"[{i+1}/{n_positions}] pos=({pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f})")
            
            Vm_arr, vext_arr = run_simulation_at_position(pos, record_times_ms)
            Vm_data[i, :] = Vm_arr
            vext_data[i, :] = vext_arr
            
            pbar.update(1)

# ===========================================================================
# Save results
# ===========================================================================
print("\n--- Save results ---")

result_npy = os.path.join(OUTPUT_DIR, f"allen_{CELL_ID}_multipos_results.npy")
np.save(result_npy, {
    "positions_um": positions_um,
    "time_ms": time_ms_arr,
    "Vm_mV": Vm_data,
    "vext_mV": vext_data,
    "cell_id": CELL_ID,
    "time_roi_ms": TIME_ROI_MS,
    "grid_spacing_um": {"min": GRID_SPACING_MIN_UM, "max": GRID_SPACING_MAX_UM},
    "coil_box_um": {
        "x": (COIL_X_MIN, COIL_X_MAX),
        "y": (COIL_Y_MIN, COIL_Y_MAX),
        "z": (COIL_Z_MIN, COIL_Z_MAX),
    },
}, allow_pickle=True)
print(f"  Saved: {result_npy}")

# ===========================================================================
# Generate 3D plots (common colorbar)
# ===========================================================================
print("\n--- Generate 3D plots ---")

# Compute global min/max for consistent colorbar
Vm_min, Vm_max = float(Vm_data.min()), float(Vm_data.max())
vext_min, vext_max = float(vext_data.min()), float(vext_data.max())

print(f"  Vm range: {Vm_min:.2f} ~ {Vm_max:.2f} mV")
print(f"  vext range: {vext_min:.2f} ~ {vext_max:.2f} mV")


def save_3d_plot(coords, values, t_ms, output_name, vmin, vmax, cmap, label, title_prefix):
    """Save 3D scatter plot."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    sc = ax.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c=values, cmap=cmap, norm=norm,
        s=POINT_SIZE, alpha=POINT_ALPHA, linewidths=0
    )
    
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_zlabel("z (um)")
    ax.set_title(f"{title_prefix} at t={t_ms:.2f} ms | grid={GRID_SPACING_MIN_UM}-{GRID_SPACING_MAX_UM}um")
    fig.colorbar(sc, ax=ax, shrink=0.6, label=label)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, f"{output_name}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


with tqdm(total=n_times * 2, desc="Save 3D plots", unit="plot", ncols=100) as pbar:
    for t_idx, t_ms in enumerate(time_ms_arr):
        # Vm plot
        Vm_vals = Vm_data[:, t_idx]
        vm_name = f"allen_{CELL_ID}_Vm_3d_{t_ms:.2f}ms"
        save_3d_plot(
            positions_um, Vm_vals, t_ms, vm_name,
            Vm_min, Vm_max, COLORMAP_VM, "Vm (mV)", "Membrane potential"
        )
        pbar.update(1)
        
        # vext plot
        vext_vals = vext_data[:, t_idx]
        vext_name = f"allen_{CELL_ID}_vext_3d_{t_ms:.2f}ms"
        save_3d_plot(
            positions_um, vext_vals, t_ms, vext_name,
            vext_min, vext_max, COLORMAP_VEXT, "vext (mV)", "Extracellular potential"
        )
        pbar.update(1)

print(f"\nDone. Results in: {OUTPUT_DIR}")
print(f"  - {n_positions} positions x {n_times} timeframes")
print(f"  - Vm plots: allen_{CELL_ID}_Vm_3d_*.png")
print(f"  - vext plots: allen_{CELL_ID}_vext_3d_*.png")
print(f"  - Data: allen_{CELL_ID}_multipos_results.npy")
