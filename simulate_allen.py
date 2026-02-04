# simulate_allen.py
"""
Allen model + E-field simulation (integrated method).

- Allen neuron model with configurable cell_id
- E-field applied via integrated method (pt3d path integral)
- Plot type: 'all' (full time) or 'single' (first 1 ms)
- Warmup phase before E-field application
- Plots saved to plot/ folder (no display)

Speed: KD-tree for nearest grid lookup; DT_MS (larger = faster);
  WARMUP_TIME_MS / TOTAL_TIME_MS can be reduced for quick runs.
"""

import numpy as np
import os
import math
from neuron import h
from neuron.units import um, ms, mV
from tqdm import tqdm
from scipy.spatial import cKDTree

h.load_file("stdrun.hoc")

# --- Configuration ---
CELL_ID = "529898751"  # Allen model cell ID (change as needed)
PLOT_TYPE = "single"   # 'all' = full time, 'single' = first 1 ms only, 'both' = both

# E-field data paths (e.g. efield/ from extract_xyz, or old_xz/ for E_field_40cycles.npy)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
E_FIELD_VALUES_FILE = os.path.join(SCRIPT_DIR, "efield", "E_field_4cycle.npy")
E_GRID_COORDS_FILE = os.path.join(SCRIPT_DIR, "efield", "E_field_grid_coords.npy")

# Output: save plots to plot/ (no display)
PLOT_DIR = os.path.join(SCRIPT_DIR, "plot")
os.makedirs(PLOT_DIR, exist_ok=True)

# Simulation time (larger dt = fewer steps = faster; 0.05 ms matches E-field 50 us)
TIME_STEP_US = 50.0   # E-field data time step (us)
TIME_STEP_MS = TIME_STEP_US / 1000.0
TOTAL_TIME_MS = 402.0   # E-field duration (ms)
WARMUP_TIME_MS = 100.0   # Warmup without E-field (ms); reduce (e.g. 100) for faster run
E_FIELD_SCALE = 1.0      # E-field magnitude scale factor
DT_MS = 0.05   # Simulation dt (ms); 0.05 = fewer steps, 0.025 = finer

# E-field unit: current npy is already in V/m; only (V/m * um) -> mV below
E_UNIT_SCALE = 1.0
E_FACTOR = 1e-3      # (V/m * um) -> mV

# Neuron position (um)
NEURON_POSITION = (0.0, 42.0, 561.0)  # (x, y, z)

# Single-cycle plot window (ms)
SINGLE_CYCLE_TIME_MS = 1.0

N_SPATIAL_POINTS = None
E_field_values = None
E_grid_coords_UM = None
E_grid_tree = None  # cKDTree for fast nearest-neighbor (built after load)


def find_nearest_spatial_index(x_um, y_um, z_um, spatial_ref):
    """Return index of nearest E-field grid point to (x, y, z).
    spatial_ref: cKDTree (fast) or (N,3) array (brute force)."""
    if hasattr(spatial_ref, "query"):
        return int(spatial_ref.query(np.array([[x_um, y_um, z_um]]), k=1)[1][0])
    target = np.array([x_um, y_um, z_um])
    distances_sq = np.sum((spatial_ref - target) ** 2, axis=1)
    return int(np.argmin(distances_sq))


def xyz_at_seg(sec, segx):
    """Return 3D coordinates at segment position segx (0~1) by linear interpolation of pt3d."""
    n = int(h.n3d(sec=sec))
    if n < 2:
        return 0.0, 0.0, 0.0
    x0, y0, z0 = h.x3d(0, sec=sec), h.y3d(0, sec=sec), h.z3d(0, sec=sec)
    x1, y1, z1 = h.x3d(n - 1, sec=sec), h.y3d(n - 1, sec=sec), h.z3d(n - 1, sec=sec)
    x = x0 + (x1 - x0) * segx
    y = y0 + (y1 - y0) * segx
    z = z0 + (z1 - z0) * segx
    return x, y, z


def translate_morphology(all_secs, dx, dy, dz):
    """Translate all section pt3d coordinates by (dx, dy, dz)."""
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
    """Return (Ex, Ey, Ez) at given spatial index and time (linear interpolation in time)."""
    time_index_float = current_time_ms / TIME_STEP_MS
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

    if hasattr(set_extracellular_field, "_e_scale"):
        Ex *= set_extracellular_field._e_scale
        Ey *= set_extracellular_field._e_scale
        Ez *= set_extracellular_field._e_scale
    return Ex, Ey, Ez


def interp_phi(arc_list, phi_list, target_arc):
    """Interpolate phi at target_arc from (arc_list, phi_list)."""
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


def build_morph_cache(neuron, grid_coords_um, pbar=None):
    """Build morphology cache (arc, dl, mid_spidx) and topology for integrated E-field."""
    cache = {}
    topo = {}
    for sec in neuron.all:
        n = int(h.n3d(sec=sec))
        if n < 2:
            cache[sec] = {"n": n, "arc": [0.0], "dl": [], "mid_spidx": []}
            if pbar is not None:
                pbar.update(1)
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
            mid_spidx.append(find_nearest_spatial_index(mx, my, mz, grid_coords_um))
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
        if pbar is not None:
            pbar.update(1)
    return cache, topo


def compute_phi_sections(neuron, morph_cache, topo, current_time_ms):
    """Compute extracellular potential phi along section tree (integrated -E·dl)."""
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

    for sec in neuron.all:
        ensure_section_phi(sec)
    return phi_sec


def apply_phi_to_segments(neuron, phi_sec):
    """Set e_extracellular on each segment from computed phi."""
    for sec in neuron.all:
        if sec not in phi_sec:
            continue
        arc, phis = phi_sec[sec]
        total_arc = arc[-1] if arc else 0.0
        for seg in sec:
            target_arc = seg.x * total_arc
            seg.e_extracellular = interp_phi(arc, phis, target_arc)


def set_extracellular_field():
    """Set e_extracellular for all segments using integrated method (called each step)."""
    if h.t < WARMUP_TIME_MS:
        for sec in neuron.all:
            for seg in sec:
                seg.e_extracellular = 0.0
        return 0
    current_time_ms = h.t - WARMUP_TIME_MS
    if not hasattr(set_extracellular_field, "_e_scale_calculated"):
        set_extracellular_field._e_scale = E_FIELD_SCALE
        set_extracellular_field._e_scale_calculated = True
        peak = np.max(np.abs(E_field_values))
        print(f"  E-field scale: {E_FIELD_SCALE}, peak (scaled): {peak * E_FIELD_SCALE:.6f} V/m")
    phi_sec = compute_phi_sections(neuron, morph_cache, topo, current_time_ms)
    apply_phi_to_segments(neuron, phi_sec)
    return 0


# --- Load E-field data ---
print("--- Load E-field data ---")
with tqdm(total=2, desc="Loading", unit="file", ncols=80) as pbar_load:
    try:
        pbar_load.set_postfix_str("E-field")
        E_field_values = np.load(E_FIELD_VALUES_FILE)  # already V/m
        pbar_load.update(1)
        pbar_load.set_postfix_str("Grid coords")
        coords_m = np.load(E_GRID_COORDS_FILE)
        E_grid_coords_UM = coords_m * 1e6
        N_SPATIAL_POINTS = E_grid_coords_UM.shape[0]
        global E_grid_tree
        E_grid_tree = cKDTree(E_grid_coords_UM)
        pbar_load.update(1)
    except Exception as e:
        pbar_load.set_postfix_str("error")
        raise
print(f"  E-field shape: {E_field_values.shape}, Grid coords (um): {E_grid_coords_UM.shape}")

# --- Create Allen neuron ---
from model_allen_neuron import AllenNeuronModel

print("\n--- Create Allen neuron ---")
with tqdm(total=1, desc="Create neuron", unit="step", ncols=80) as pbar_neuron:
    pbar_neuron.set_postfix_str(f"cell_id={CELL_ID}")
    neuron = AllenNeuronModel(x=0, y=0, z=0, cell_id=CELL_ID)
    sx, sy, sz = xyz_at_seg(neuron.soma, 0.5)
    tx, ty, tz = NEURON_POSITION
    translate_morphology(neuron.all, tx - sx, ty - sy, tz - sz)
    pbar_neuron.update(1)
print(f"  Cell ID: {CELL_ID}, Position: {NEURON_POSITION} um")

# --- Morphology cache for integrated method ---
with tqdm(total=len(neuron.all), desc="Build morph cache", unit="sec", ncols=80) as pbar_cache:
    morph_cache, topo = build_morph_cache(neuron, E_grid_tree if E_grid_tree is not None else E_grid_coords_UM, pbar=pbar_cache)

# --- Simulation setup ---
h.tstop = WARMUP_TIME_MS + TOTAL_TIME_MS
h.dt = DT_MS
h.celsius = 34.0
h.finitialize(-65.0 * mV)

t_vec = h.Vector()
vm_vec = h.Vector()
vext_vec = h.Vector()

# --- Run: warmup then E-field ---
print("\n--- Warmup ---")
warmup_steps = int(WARMUP_TIME_MS / h.dt)
with tqdm(total=warmup_steps, desc="Warmup", unit="step", ncols=80) as pbar_warmup:
    while h.t < WARMUP_TIME_MS:
        h.fadvance()
        pbar_warmup.update(1)

resting = neuron.soma(0.5).v
print(f"  Resting Vm: {resting:.2f} mV")

print("\n--- Run with E-field ---")
total_steps = int((h.tstop - WARMUP_TIME_MS) / h.dt)
pbar = tqdm(total=total_steps, desc="Simulate", unit="step", ncols=80)

while h.t < h.tstop:
    t_rel = h.t - WARMUP_TIME_MS
    if t_rel >= 0:
        set_extracellular_field()
    else:
        for sec in neuron.all:
            for seg in sec:
                seg.e_extracellular = 0.0
    t_vec.append(t_rel)
    vm_vec.append(neuron.soma(0.5).v)
    try:
        vext_vec.append(neuron.soma(0.5).vext[0] if hasattr(neuron.soma(0.5), "vext") else 0.0)
    except Exception:
        vext_vec.append(0.0)
    h.fadvance()
    if t_rel >= 0 and int(t_rel / h.dt) < total_steps:
        pbar.update(1)
pbar.close()

t_arr = np.array(t_vec.as_numpy() if hasattr(t_vec, "as_numpy") else list(t_vec))
vm_arr = np.array(vm_vec.as_numpy() if hasattr(vm_vec, "as_numpy") else list(vm_vec))
vext_arr = np.array(vext_vec.as_numpy() if hasattr(vext_vec, "as_numpy") else list(vext_vec))

# --- Save plots to plot/ (no display) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def save_plot(t_data, vm_data, vext_data, suffix, title_suffix):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_data, vm_data, color="blue", linewidth=1.5, label="Vm (soma)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Vm (mV)", color="blue")
    ax.tick_params(axis="y", labelcolor="blue")
    # Fix y-axis so resting line is visible: center on resting with ±2 mV range
    vm_range = 2.0  # mV above/below resting to show modulation
    ax.set_ylim(resting - vm_range, resting + vm_range)
    ax.axhline(resting, color="red", linestyle="--", linewidth=0.8, label=f"Resting ({resting:.1f} mV)")
    if vext_data is not None and len(vext_data) == len(t_data):
        ax2 = ax.twinx()
        ax2.plot(t_data, vext_data, color="green", linewidth=1.0, alpha=0.7, linestyle="--", label="vext")
        ax2.set_ylabel("vext (mV)", color="green")
        ax2.tick_params(axis="y", labelcolor="green")
    ax.legend(loc="upper left")
    ax.grid(True)
    ax.set_title(f"Allen cell {CELL_ID} - E-field (integrated) - {title_suffix}")
    fname = os.path.join(PLOT_DIR, f"allen_{CELL_ID}_efield_{suffix}.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")

plots_to_save = []
if PLOT_TYPE in ("all", "both"):
    plots_to_save.append(("all", "full time", t_arr, vm_arr, vext_arr))
if PLOT_TYPE in ("single", "both"):
    mask = (t_arr >= 0) & (t_arr <= SINGLE_CYCLE_TIME_MS)
    if np.any(mask):
        plots_to_save.append(
            ("single_1ms", "first 1 ms", t_arr[mask], vm_arr[mask],
             vext_arr[mask] if vext_arr is not None else None)
        )
    else:
        print("  No data in first 1 ms for single plot.")

if plots_to_save:
    for suffix, title_suffix, t_data, vm_data, vext_data in tqdm(
            plots_to_save, desc="Save plots", unit="plot", ncols=80):
        save_plot(t_data, vm_data, vext_data, suffix, title_suffix)

print("\nDone. Plots in:", PLOT_DIR)
