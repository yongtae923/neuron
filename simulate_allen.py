# simulate_allen.py
"""
Allen model + E-field simulation at multiple spatial positions (PARALLEL).

- One-time: get equilibrium start point (0V -> equilibrium, no E-field, like simulate_one.py)
- For each E-field grid point (outside coil, within ROI),
  run neuron simulation from equilibrium, with E-field from t=0, record Vm at soma.
- Uses multiprocessing for parallel execution (cpu_count - 1 workers)
- Results: 3D Vm array (position x time) saved as npy
- Plotting is handled separately (see plot_allen.py)
"""

import warnings
warnings.filterwarnings("ignore", message=".*Signature.*numpy.longdouble.*", category=UserWarning)

import numpy as np
import os
import sys
import math
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
from scipy.spatial import cKDTree

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

# --- Multiprocessing settings ---
N_WORKERS = max(1, mp.cpu_count() - 1)  # 코어 수 - 1

# --- Time settings ---
TIME_STEP_US = 50.0   # E-field data time step (us)
TIME_STEP_MS = TIME_STEP_US / 1000.0  # 0.05 ms
DT_MS = 0.05   # Simulation dt (ms)

# Equilibrium (0V start, no E-field) for initial condition
EQ_MAX_TIME_MS = 500.0
EQ_THRESHOLD_MV = 0.001
EQ_STABLE_STEPS = 10

# Time ROI (ms): simulate this range
TIME_ROI_MS = (0.0, 0.5)  # (start, end) or None for full

# E-field: npy is already V/m
E_FIELD_SCALE = 1.0
E_FACTOR = 1e-3  # (V/m * um) -> mV

# --- Space settings ---
X_ROI_UM = None
Y_ROI_UM = None
Z_ROI_UM = None

# Coil region to exclude (um)
COIL_X_MIN, COIL_X_MAX = -79.5, 79.5
COIL_Y_MIN, COIL_Y_MAX = -32.0, 32.0
COIL_Z_MIN, COIL_Z_MAX = 498.0, 1502.0

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


# ===========================================================================
# Worker function for multiprocessing (NEURON imported inside)
# ===========================================================================
def _worker_init(efield_values, grid_coords_um, grid_tree_data, config):
    """Initialize worker process with shared data."""
    global _E_field_values, _E_grid_coords_UM, _E_grid_tree, _config
    _E_field_values = efield_values
    _E_grid_coords_UM = grid_coords_um
    _E_grid_tree = cKDTree(grid_coords_um)
    _config = config
    
    # Import NEURON in worker
    global h, mV, AllenNeuronModel
    from neuron import h as _h
    from neuron.units import mV as _mV
    h = _h
    mV = _mV
    h.load_file("stdrun.hoc")
    
    from model_allen_neuron import AllenNeuronModel as _AllenNeuronModel
    AllenNeuronModel = _AllenNeuronModel


def _find_nearest_spatial_index(x_um, y_um, z_um):
    """Return index of nearest E-field grid point."""
    return int(_E_grid_tree.query(np.array([[x_um, y_um, z_um]]), k=1)[1][0])


def _xyz_at_seg(sec, segx):
    """Return 3D coordinates at segment position."""
    n = int(h.n3d(sec=sec))
    if n < 2:
        return 0.0, 0.0, 0.0
    x0, y0, z0 = h.x3d(0, sec=sec), h.y3d(0, sec=sec), h.z3d(0, sec=sec)
    x1, y1, z1 = h.x3d(n - 1, sec=sec), h.y3d(n - 1, sec=sec), h.z3d(n - 1, sec=sec)
    return x0 + (x1 - x0) * segx, y0 + (y1 - y0) * segx, z0 + (z1 - z0) * segx


def _translate_morphology(all_secs, dx, dy, dz):
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


def _get_E_at(spatial_idx, current_time_ms):
    """Return (Ex, Ey, Ez) at given spatial index and time."""
    actual_time_ms = current_time_ms + _config['TIME_START_MS']
    time_index_float = actual_time_ms / _config['TIME_STEP_MS']
    Tmax = _E_field_values.shape[2] - 1
    t_idx_prev = int(math.floor(time_index_float))
    t_idx_prev = max(0, min(t_idx_prev, Tmax))
    t_idx_next = min(t_idx_prev + 1, Tmax)
    ratio = time_index_float - t_idx_prev
    ratio = max(0.0, min(1.0, ratio))

    n_comp = _E_field_values.shape[0]
    if n_comp >= 3:
        Ex = _E_field_values[0, spatial_idx, t_idx_prev] + ratio * (
            _E_field_values[0, spatial_idx, t_idx_next] - _E_field_values[0, spatial_idx, t_idx_prev])
        Ey = _E_field_values[1, spatial_idx, t_idx_prev] + ratio * (
            _E_field_values[1, spatial_idx, t_idx_next] - _E_field_values[1, spatial_idx, t_idx_prev])
        Ez = _E_field_values[2, spatial_idx, t_idx_prev] + ratio * (
            _E_field_values[2, spatial_idx, t_idx_next] - _E_field_values[2, spatial_idx, t_idx_prev])
    else:
        Ex = _E_field_values[0, spatial_idx, t_idx_prev] + ratio * (
            _E_field_values[0, spatial_idx, t_idx_next] - _E_field_values[0, spatial_idx, t_idx_prev])
        Ey = 0.0
        Ez = _E_field_values[1, spatial_idx, t_idx_prev] + ratio * (
            _E_field_values[1, spatial_idx, t_idx_next] - _E_field_values[1, spatial_idx, t_idx_prev])
    return Ex * _config['E_FIELD_SCALE'], Ey * _config['E_FIELD_SCALE'], Ez * _config['E_FIELD_SCALE']


def _interp_phi(arc_list, phi_list, target_arc):
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


def _build_morph_cache(neuron_model):
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
            mid_spidx.append(_find_nearest_spatial_index(mx, my, mz))
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


def _compute_phi_sections(neuron_model, morph_cache, topo, current_time_ms):
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
            phi0 = _interp_phi(parc, pphi, px * total_arc)
        data = morph_cache[sec]
        n = data["n"]
        if n < 2:
            phi_sec[sec] = ([0.0], [phi0])
            return
        arc, dl, mid_spidx = data["arc"], data["dl"], data["mid_spidx"]
        phis = [phi0]
        for i in range(n - 1):
            Ex, Ey, Ez = _get_E_at(mid_spidx[i], current_time_ms)
            dx, dy, dz = dl[i]
            dphi = -(Ex * dx + Ey * dy + Ez * dz) * _config['E_FACTOR']
            phis.append(phis[-1] + dphi)
        phi_sec[sec] = (arc, phis)

    for sec in neuron_model.all:
        ensure_section_phi(sec)
    return phi_sec


def _apply_phi_to_segments(neuron_model, phi_sec):
    """Set e_extracellular on each segment."""
    for sec in neuron_model.all:
        if sec not in phi_sec:
            continue
        arc, phis = phi_sec[sec]
        total_arc = arc[-1] if arc else 0.0
        for seg in sec:
            target_arc = seg.x * total_arc
            seg.e_extracellular = _interp_phi(arc, phis, target_arc)


def get_equilibrium_vm(neuron_model, dt_ms=0.05):
    """
    Run 0V start, no E-field until equilibrium (like simulate_one.py).
    Returns soma Vm at equilibrium for use as v_init.
    """
    from neuron import h
    from neuron.units import mV
    for sec in neuron_model.all:
        for seg in sec:
            seg.e_extracellular = 0.0
    h.finitialize(0.0 * mV)
    h.dt = dt_ms
    h.tstop = EQ_MAX_TIME_MS
    soma_seg = neuron_model.soma(0.5)
    stable_count = 0
    prev_v = float(soma_seg.v)
    while h.t < h.tstop - 1e-9:
        h.fadvance()
        curr_v = float(soma_seg.v)
        if abs(curr_v - prev_v) < EQ_THRESHOLD_MV:
            stable_count += 1
            if stable_count >= EQ_STABLE_STEPS:
                break
        else:
            stable_count = 0
        prev_v = curr_v
    return float(soma_seg.v)


def worker_simulate(args):
    """Worker function: run simulation at one position."""
    idx, pos_um, record_times_ms = args

    # Create neuron (suppress output)
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        neuron_model = AllenNeuronModel(x=0, y=0, z=0, cell_id=_config['CELL_ID'])

    # Move to target position
    sx, sy, sz = _xyz_at_seg(neuron_model.soma, 0.5)
    tx, ty, tz = pos_um
    _translate_morphology(neuron_model.all, tx - sx, ty - sy, tz - sz)

    # Build morph cache
    morph_cache, topo = _build_morph_cache(neuron_model)

    # One-time: get equilibrium start point (0V -> equilibrium, no E-field)
    v_init = _config['V_INIT_EQUILIBRIUM']
    h.finitialize(v_init * mV)
    h.dt = _config['DT_MS']
    h.celsius = 34.0
    h.tstop = _config['TOTAL_TIME_MS'] + _config['DT_MS']

    # Record transmembrane Vm (= v - e_extracellular) at specified times
    # No warmup: start at t=0 with E-field, from equilibrium
    Vm_list = []
    rec_idx = 0

    while h.t < h.tstop - 1e-9 and rec_idx < len(record_times_ms):
        t_rel = h.t  # t=0 is start

        # Apply E-field from t=0
        phi_sec = _compute_phi_sections(neuron_model, morph_cache, topo, t_rel)
        _apply_phi_to_segments(neuron_model, phi_sec)

        # Record if this is a target time
        if rec_idx < len(record_times_ms) and abs(t_rel - record_times_ms[rec_idx]) < _config['DT_MS'] / 2:
            soma = neuron_model.soma(0.5)
            Vm_list.append(float(soma.v - getattr(soma, "e_extracellular", 0.0)))
            rec_idx += 1

        h.fadvance()
    
    # Pad if needed
    pad_val = _config['V_INIT_EQUILIBRIUM']
    while len(Vm_list) < len(record_times_ms):
        Vm_list.append(Vm_list[-1] if Vm_list else pad_val)
    
    return idx, np.array(Vm_list, dtype=np.float32)


# ===========================================================================
# Main (only runs in main process)
# ===========================================================================
if __name__ == "__main__":
    import time as time_module
    
    print("=" * 60)
    print("Allen Neuron Multi-Position Simulation (PARALLEL)")
    print("=" * 60)
    print(f"  Workers: {N_WORKERS} (of {mp.cpu_count()} cores)")
    
    # --- Load E-field data ---
    print("\n--- Load E-field data ---")
    E_field_values = np.load(E_FIELD_VALUES_FILE)
    coords_m = np.load(E_GRID_COORDS_FILE)
    E_grid_coords_UM = coords_m * 1e6
    
    print(f"  E-field shape: {E_field_values.shape}")
    print(f"  Grid coords: {E_grid_coords_UM.shape}")
    print(f"  E-field unit: V/m")
    
    # Compute time indices
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
    
    # Compute spatial positions (all points outside coil, optionally within ROI)
    print("\n--- Spatial positions (no downsampling) ---")
    valid_idx = get_valid_efield_indices(E_grid_coords_UM)
    positions_um = E_grid_coords_UM[valid_idx]
    
    n_positions = len(valid_idx)
    n_times = len(time_indices)
    
    print(f"  Total simulations: {n_positions}")
    
    # Record times relative to warmup
    record_times_ms = time_ms_arr - TIME_START_MS
    
    # Config dict for workers
    # One-time: get equilibrium start point (0V -> equilibrium, no E-field)
    print("\n--- Equilibrium start point (0V, no E-field) ---")
    import io
    import contextlib
    from neuron import h as _h_main
    from neuron.units import mV as _mV_main
    _h_main.load_file("stdrun.hoc")
    from model_allen_neuron import AllenNeuronModel as _AllenModel
    with contextlib.redirect_stdout(io.StringIO()):
        _neuron_temp = _AllenModel(x=0, y=0, z=0, cell_id=CELL_ID)
    v_init_equilibrium = get_equilibrium_vm(_neuron_temp, dt_ms=DT_MS)
    print(f"  Equilibrium Vm: {v_init_equilibrium:.2f} mV")

    config = {
        'CELL_ID': CELL_ID,
        'V_INIT_EQUILIBRIUM': v_init_equilibrium,
        'TOTAL_TIME_MS': TOTAL_TIME_MS,
        'TIME_START_MS': TIME_START_MS,
        'TIME_STEP_MS': TIME_STEP_MS,
        'DT_MS': DT_MS,
        'E_FIELD_SCALE': E_FIELD_SCALE,
        'E_FACTOR': E_FACTOR,
    }
    
    # Estimate time (single test)
    print(f"\n--- Estimate simulation time ---")
    print(f"  Each simulation: {TOTAL_TIME_MS:.2f}ms with E-field (start from equilibrium)")
    print(f"  Running 1 test simulation...")
    
    def format_time(seconds):
        if seconds < 60:
            return f"{seconds:.1f} sec"
        elif seconds < 3600:
            return f"{seconds / 60:.1f} min"
        else:
            hours = int(seconds // 3600)
            mins = int((seconds % 3600) // 60)
            return f"{hours}h {mins}m"
    
    # Create pool and run test
    t_start = time_module.time()
    
    with mp.Pool(
        processes=1,
        initializer=_worker_init,
        initargs=(E_field_values, E_grid_coords_UM, None, config)
    ) as pool:
        test_result = pool.map(worker_simulate, [(0, positions_um[0], record_times_ms)])[0]
    
    t_elapsed = time_module.time() - t_start
    time_per_sim_sec = t_elapsed
    
    # With parallelization, effective time is reduced by N_WORKERS
    total_estimated_sec = time_per_sim_sec * n_positions / N_WORKERS
    
    print(f"  Time per simulation: {format_time(time_per_sim_sec)}")
    print(f"  Estimated total time (parallel): {format_time(total_estimated_sec)} ({n_positions} sims / {N_WORKERS} workers)")
    
    # --- Run all simulations in parallel ---
    print(f"\n--- Running {n_positions} simulations ({N_WORKERS} workers) ---")
    
    Vm_data = np.zeros((n_positions, n_times), dtype=np.float32)
    
    # Store test result
    Vm_data[0, :] = test_result[1]
    
    # Prepare arguments for remaining positions
    args_list = [(i, positions_um[i], record_times_ms) for i in range(1, n_positions)]
    
    if len(args_list) > 0:
        with mp.Pool(
            processes=N_WORKERS,
            initializer=_worker_init,
            initargs=(E_field_values, E_grid_coords_UM, None, config)
        ) as pool:
            with tqdm(total=len(args_list), desc="Simulations", unit="pos", ncols=100) as pbar:
                for result in pool.imap_unordered(worker_simulate, args_list):
                    idx, Vm_arr = result
                    Vm_data[idx, :] = Vm_arr
                    pbar.update(1)
    
    # --- Save results ---
    print("\n--- Save results ---")
    
    result_npy = os.path.join(OUTPUT_DIR, f"allen_{CELL_ID}_multipos_results.npy")
    np.save(result_npy, {
        "positions_um": positions_um,
        "time_ms": time_ms_arr,
        "Vm_mV": Vm_data,
        "cell_id": CELL_ID,
        "time_roi_ms": TIME_ROI_MS,
        "coil_box_um": {
            "x": (COIL_X_MIN, COIL_X_MAX),
            "y": (COIL_Y_MIN, COIL_Y_MAX),
            "z": (COIL_Z_MIN, COIL_Z_MAX),
        },
        "n_workers": N_WORKERS,
    }, allow_pickle=True)
    print(f"  Saved: {result_npy}")

    Vm_min, Vm_max = float(Vm_data.min()), float(Vm_data.max())
    print(f"\nDone. Results in: {OUTPUT_DIR}")
    print(f"  - {n_positions} positions x {n_times} timeframes")
    print(f"  - Vm range: {Vm_min:.2f} ~ {Vm_max:.2f} mV")
    print(f"  - Data: allen_{CELL_ID}_multipos_results.npy")
    print(f"\nTo plot: python plot_allen.py --input \"{result_npy}\"")
