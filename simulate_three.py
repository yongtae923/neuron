# simulate_three.py
"""
Simulate Allen neuron model at fixed positions with E-field.

- Neuron ID: 529898751
- E-field file: E_field_1cycle.npy
- E-field applied: 0 ~ 2 ms
- Pre-simulation (no E-field): -1 ~ 0 ms
- Positions: [80, 0.0, 550.0], [80, 35, 550.0], [0.0, 35, 550.0], [0, 0, 0]
- Coil region excluded
- Results saved as npy file

================================================================================
INPUT: E-FIELD DATA
================================================================================

  Location (address):
    - E_field values:  SCRIPT_DIR/efield/E_field_1cycle.npy  (config: E_FIELD_VALUES_FILE)
    - Grid coordinates: SCRIPT_DIR/efield/E_field_grid_coords.npy  (config: E_GRID_COORDS_FILE)

  Format:
    - Both are NumPy .npy files (binary).
    - E_field_1cycle.npy:
        shape: (n_components, N_spatial, N_time)
        - n_components: 3 for (Ex, Ey, Ez) or 2 for (Ex, Ez) with Ey assumed 0.
        - N_spatial: number of 3D grid points (e.g. 1771561).
        - N_time: number of time steps (e.g. 201 for 1 cycle at 50 us, 10 ms total).
        - Index 0 = Ex, 1 = Ey, 2 = Ez (V/m).
    - E_field_grid_coords.npy:
        shape: (N_spatial, 3)
        - Each row is (X, Y, Z) in METERS. Script converts to µm (× 1e6) for NEURON.

  Time:
    - Time step of E-field data: 50 µs (TIME_STEP_MS = 0.05 ms).
    - Time index k corresponds to t = k * 0.05 ms.

  How it is used:
    - For each segment midpoint of the neuron morphology, the script finds the
      nearest grid point (by 3D distance) and gets (Ex, Ey, Ez) at that point.
    - Time is linearly interpolated between two consecutive time indices.
    - Phi (extracellular potential) is computed by integrating -E·dl along the
      morphology (pt3d path); E_FACTOR = 1e-3 converts (V/m × µm) to mV.
    - E_field scale (1x, 10x, 100x, 1000x) multiplies the loaded E values (V/m).

  Creating E-field .npy files:
    - Use extract_xyz.py to build E_field_*cycle.npy and E_field_grid_coords.npy
      from Ansys text outputs in efield/twin_cons_*_Ex, _Ey, _Ez folders.

================================================================================
INPUT: ALLEN NEURON MODEL
================================================================================

  Location (address):
    - Model data root: SCRIPT_DIR/allen_model/  (see model_allen_neuron.ALLEN_MODEL_DIR)
    - Per-cell folder: allen_model/<CELL_ID>_*_ephys/  (e.g. 529898751_layer5_ephys)
    - simulate_three.py sets CELL_ID = "529898751"; the script resolves the path
      by glob: allen_model/529898751_*_ephys → first matching directory.

  Format (inside <CELL_ID>_*_ephys/):
    - reconstruction.swc (or any .swc): morphology (soma, dendrites, axon).
      SWC columns: id, type(1=soma,2=dend,3=axon,4=apical), x, y, z, radius, parent_id.
      Units: µm (x,y,z, radius).
    - fit_parameters.json: electrophysiology (passive and active parameters).
      Used: passive (ra, cm, e_pas), conditions (celsius, v_init, erev), genome, etc.

  Internal structure (after loading in model_allen_neuron.py):
    - AllenNeuronModel(x, y, z, cell_id=CELL_ID) builds NEURON sections from SWC:
      self.soma, self.dendrites, self.axon, self.all (list of all sections).
    - Each section has pt3d geometry and biophysical mechanisms from fit_parameters.
    - (x, y, z) are added to all SWC coordinates so the neuron can be placed at
      a given position; simulate_three then translates so soma center is at
      NEURON_POSITIONS_UM[i].

  How to use:
    - Put your cell folder under neuron/allen_model/ named <cell_id>_*_ephys
      (e.g. 529898751_layer5_ephys) with reconstruction.swc and fit_parameters.json.
    - Set CELL_ID = "529898751" (or your ID) at the top of simulate_three.py.
    - Optionally pass data_dir=... to AllenNeuronModel() to point to a specific
      folder instead of resolving by cell_id.
"""

from __future__ import annotations

import os
import sys
import math
from typing import Tuple

import numpy as np
from neuron import h
from neuron.units import mV

# Ensure we can import from project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

h.load_file("stdrun.hoc")

from model_allen_neuron import AllenNeuronModel  # noqa: E402


# --- Configuration ---
CELL_ID = "529898751"

E_FIELD_VALUES_FILE = os.path.join(SCRIPT_DIR, "efield", "E_field_1cycle.npy")
E_GRID_COORDS_FILE = os.path.join(SCRIPT_DIR, "efield", "E_field_grid_coords.npy")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Time settings
TIME_STEP_US = 50.0   # E-field data time step (us)
TIME_STEP_MS = TIME_STEP_US / 1000.0  # 0.05 ms
DT_MS = 0.05          # Simulation dt (ms)

# E-field를 적용할 시간 구간 (ms): 0 ~ 2 ms
TIME_ROI_MS: Tuple[float, float] = (0.0, 2.0)

# E-field 0인 상태로 선행 시뮬레이션할 구간 (ms): -1 ~ 0 ms
PRE_EFIELD_MS: float = 1.0

# E-field scaling
E_FIELD_SCALES = [1.0, 10.0, 100.0, 1000.0]  # Multiple scales to simulate
E_FACTOR = 1e-3            # (V/m * um) -> mV

# Positions for this script (µm)
NEURON_POSITIONS_UM = np.array(
    [
        [80,   0.0, 550.0],
        [80, 35, 550.0],
        [0.0,  35, 550.0],
        [0.0,   0.0, 0.0],
    ],
    dtype=float,
)

# Coil region to exclude (um) - from simulate_allen.py
COIL_X_MIN, COIL_X_MAX = -79.5, 79.5
COIL_Y_MIN, COIL_Y_MAX = -32.0, 32.0
COIL_Z_MIN, COIL_Z_MAX = 498.0, 1502.0


# --- Core simulation logic from old_xz/simulate_tES.py ---

def _find_nearest_spatial_index(x_um, y_um, z_um, grid_coords_um):
    """주어진 (x, y, z)에 가장 가까운 그리드 지점의 인덱스를 반환합니다. (from simulate_tES.py)"""
    target_coord = np.array([x_um, y_um, z_um])
    distances_sq = np.sum((grid_coords_um - target_coord)**2, axis=1)
    nearest_index = np.argmin(distances_sq)
    return nearest_index


def _xyz_at_seg(sec, segx: float) -> Tuple[float, float, float]:
    """
    세그먼트 위치(segx, 0.0~1.0)에서의 3D 좌표를 반환합니다. (from simulate_tES.py)
    h.x3d()는 정수 인덱스를 받으므로, pt3d 점들을 선형 보간하여 계산합니다.
    """
    n = int(h.n3d(sec=sec))
    if n < 2:
        return 0.0, 0.0, 0.0
    
    # Section의 양 끝점 좌표
    x0 = h.x3d(0, sec=sec)
    y0 = h.y3d(0, sec=sec)
    z0 = h.z3d(0, sec=sec)
    x1 = h.x3d(n-1, sec=sec)
    y1 = h.y3d(n-1, sec=sec)
    z1 = h.z3d(n-1, sec=sec)
    
    # 선형 보간
    x = x0 + (x1 - x0) * segx
    y = y0 + (y1 - y0) * segx
    z = z0 + (z1 - z0) * segx
    
    return x, y, z


def _translate_morphology(all_secs, dx: float, dy: float, dz: float) -> None:
    """
    모든 section의 pt3d 좌표를 이동시킵니다. (from simulate_tES.py)
    Allen 모델을 목표 위치로 이동시키기 위해 사용됩니다.
    """
    for sec in all_secs:
        n = int(h.n3d(sec=sec))
        for i in range(n):
            x = h.x3d(i, sec=sec) + dx
            y = h.y3d(i, sec=sec) + dy
            z = h.z3d(i, sec=sec) + dz
            d = h.diam3d(i, sec=sec)
            h.pt3dchange(i, x, y, z, d, sec=sec)
    h.define_shape()  # shape 재정의


def _interp_phi(arc_list, phi_list, target_arc: float) -> float:
    """arc_list에서 target_arc 위치의 phi 값을 보간합니다. (from simulate_tES.py)"""
    if len(arc_list) == 0 or len(phi_list) == 0:
        return 0.0
    if target_arc <= arc_list[0]:
        return phi_list[0]
    if target_arc >= arc_list[-1]:
        return phi_list[-1]

    # 이진 탐색
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


def _build_morph_cache(neuron_model, grid_coords_um):
    """뉴런의 morphology 캐시를 생성합니다 (pt3d 기반). (from simulate_tES.py)"""
    cache = {}
    topo = {}

    for sec in neuron_model.all:
        n = int(h.n3d(sec=sec))
        if n < 2:
            cache[sec] = {"n": n, "arc": [0.0], "dl": [], "mid_spidx": []}
            topo[sec] = None
            continue

        xs = [h.x3d(i, sec=sec) for i in range(n)]
        ys = [h.y3d(i, sec=sec) for i in range(n)]
        zs = [h.z3d(i, sec=sec) for i in range(n)]
        arc = [h.arc3d(i, sec=sec) for i in range(n)]  # um

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

            spidx = _find_nearest_spatial_index(mx, my, mz, grid_coords_um)
            mid_spidx.append(spidx)

        cache[sec] = {"n": n, "arc": arc, "dl": dl, "mid_spidx": mid_spidx}

        # 부모 section 정보 (from simulate_tES.py - 더 상세한 처리)
        sref = h.SectionRef(sec=sec)
        if sref.has_parent():
            # 우선 parentseg() 정보를 사용 (실제 연결 위치가 가장 정확)
            try:
                pseg = sref.parentseg()
                topo[sec] = (pseg.sec, float(pseg.x))
            except:
                # parentseg가 없으면 기존 방식으로 추정
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


def _get_E_at(
    E_field_values: np.ndarray,
    spatial_idx: int,
    current_time_ms: float,
    time_start_ms: float,
    time_step_ms: float,
    E_field_scale: float,
) -> Tuple[float, float, float]:
    """특정 공간 인덱스와 시간에서 E-field 값을 반환합니다. (from simulate_tES.py)"""
    actual_time_ms = current_time_ms + time_start_ms
    time_index_float = actual_time_ms / time_step_ms
    Tmax = E_field_values.shape[2] - 1
    
    # 시간 인덱스 경계 방어 (오프바이원 방지)
    t_idx_prev = int(math.floor(time_index_float))
    if t_idx_prev < 0:
        t_idx_prev = 0
    if t_idx_prev > Tmax:
        t_idx_prev = Tmax
    t_idx_next = min(t_idx_prev + 1, Tmax)
    
    ratio = time_index_float - t_idx_prev
    # ratio도 경계에서 클리핑
    if ratio < 0.0:
        ratio = 0.0
    if ratio > 1.0:
        ratio = 1.0

    n_comp = E_field_values.shape[0]
    if n_comp >= 3:
        Ex_prev = E_field_values[0, spatial_idx, t_idx_prev]
        Ex_next = E_field_values[0, spatial_idx, t_idx_next]
        Ey_prev = E_field_values[1, spatial_idx, t_idx_prev]
        Ey_next = E_field_values[1, spatial_idx, t_idx_next]
        Ez_prev = E_field_values[2, spatial_idx, t_idx_prev]
        Ez_next = E_field_values[2, spatial_idx, t_idx_next]
        
        Ex = Ex_prev + ratio * (Ex_next - Ex_prev)
        Ey = Ey_prev + ratio * (Ey_next - Ey_prev)
        Ez = Ez_prev + ratio * (Ez_next - Ez_prev)
    else:
        Ex_prev = E_field_values[0, spatial_idx, t_idx_prev]
        Ex_next = E_field_values[0, spatial_idx, t_idx_next]
        Ez_prev = E_field_values[1, spatial_idx, t_idx_prev] if n_comp >= 2 else 0.0
        Ez_next = E_field_values[1, spatial_idx, t_idx_next] if n_comp >= 2 else 0.0
        
        Ex = Ex_prev + ratio * (Ex_next - Ex_prev)
        Ey = 0.0
        Ez = Ez_prev + ratio * (Ez_next - Ez_prev) if n_comp >= 2 else 0.0
    
    return Ex * E_field_scale, Ey * E_field_scale, Ez * E_field_scale


def _compute_phi_sections(
    neuron_model,
    morph_cache,
    topo,
    current_time_ms: float,
    E_field_values: np.ndarray,
    time_start_ms: float,
    time_step_ms: float,
    E_field_scale: float,
    E_factor: float,
):
    """섹션 트리를 따라 phi를 누적 적분하여 계산합니다. (from simulate_tES.py)"""
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
            phi0 = _interp_phi(parc, pphi, px * total_arc)

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
            Ex, Ey, Ez = _get_E_at(
                E_field_values,
                spidx,
                current_time_ms,
                time_start_ms,
                time_step_ms,
                E_field_scale,
            )
            dx, dy, dz = dl[i]
            dphi = -(Ex * dx + Ey * dy + Ez * dz) * E_factor  # mV
            phis.append(phis[-1] + dphi)

        phi_sec[sec] = (arc, phis)

    # 모든 section에 대해 트리 방향으로 계산
    for sec in neuron_model.all:
        ensure_section_phi(sec)

    return phi_sec


def _apply_phi_to_segments(neuron_model, phi_sec) -> None:
    """계산된 phi를 각 세그먼트의 e_extracellular에 적용합니다. (from simulate_tES.py)"""
    for sec in neuron_model.all:
        if sec not in phi_sec:
            continue
        arc, phis = phi_sec[sec]
        total_arc = arc[-1] if len(arc) > 0 else 0.0
        # ghost node(0,1)를 제외하고 실제 세그먼트만 순회하기 위해 sec를 직접 순회
        for seg in sec:
            target_arc = seg.x * total_arc
            phi = _interp_phi(arc, phis, target_arc)
            seg.e_extracellular = phi


def get_equilibrium_vm(neuron_model, dt_ms: float = 0.05) -> float:
    """
    Run 0V start, no E-field until equilibrium.
    Returns soma Vm at equilibrium for use as v_init.
    """
    from neuron import h as _h
    from neuron.units import mV as _mV

    for sec in neuron_model.all:
        for seg in sec:
            seg.e_extracellular = 0.0
    _h.finitialize(0.0 * _mV)
    _h.dt = dt_ms
    _h.tstop = 500.0
    soma_seg = neuron_model.soma(0.5)
    stable_count = 0
    prev_v = float(soma_seg.v)
    threshold = 0.001
    stable_steps = 10
    while _h.t < _h.tstop - 1e-9:
        _h.fadvance()
        curr_v = float(soma_seg.v)
        if abs(curr_v - prev_v) < threshold:
            stable_count += 1
            if stable_count >= stable_steps:
                break
        else:
            stable_count = 0
        prev_v = curr_v
    return float(soma_seg.v)


def run_single_simulation(E_field_scale: float, E_field_values: np.ndarray, E_grid_coords_UM: np.ndarray, v_init_equilibrium: float) -> dict:
    """Run simulation for a single E-field scale."""
    # Time axis: -PRE_EFIELD_MS ~ TIME_ROI_MS[1]
    total_time_ms = PRE_EFIELD_MS + TIME_ROI_MS[1]
    record_times_ms = np.arange(-PRE_EFIELD_MS, TIME_ROI_MS[1] + 1e-9, DT_MS)
    n_times = len(record_times_ms)

    n_positions = NEURON_POSITIONS_UM.shape[0]
    Vm_data = np.zeros((n_positions, n_times), dtype=np.float32)
    vext_data = np.zeros((n_positions, n_times), dtype=np.float32)
    vin_data = np.zeros((n_positions, n_times), dtype=np.float32)

    for i, pos_um in enumerate(NEURON_POSITIONS_UM):
        print(f"\n--- Simulating position {i}: {tuple(pos_um)} (um), E-field scale: {E_field_scale:.1f}x ---")

        # Create neuron model
        print(f"  Creating neuron model...")
        neuron_model = AllenNeuronModel(x=0, y=0, z=0, cell_id=CELL_ID)
        print(f"  Neuron model created.")

        # Move morphology so that soma is at target position
        print(f"  Translating morphology...")
        sx, sy, sz = _xyz_at_seg(neuron_model.soma, 0.5)
        tx, ty, tz = pos_um
        _translate_morphology(neuron_model.all, tx - sx, ty - sy, tz - sz)
        print(f"  Morphology translated.")

        # Build morph cache
        print(f"  Building morphology cache...")
        morph_cache, topo = _build_morph_cache(neuron_model, E_grid_coords_UM)
        print(f"  Morphology cache built.")

        # Initialize from equilibrium Vm
        print(f"  Initializing simulation (t=0 to {total_time_ms:.2f} ms)...")
        h.finitialize(v_init_equilibrium * mV)
        h.dt = DT_MS
        h.celsius = 34.0
        h.tstop = total_time_ms + DT_MS

        soma = neuron_model.soma(0.5)

        Vm_list = []
        vext_list = []
        vin_list = []
        rec_idx = 0
        
        # Progress tracking
        total_steps = int(total_time_ms / DT_MS)
        progress_interval = max(1, total_steps // 20)  # 20개 구간으로 나눔
        step_count = 0

        print(f"  Running simulation ({total_steps} steps)...")
        while h.t < h.tstop - 1e-9 and rec_idx < len(record_times_ms):
            t_rel = h.t  # Relative time from start (-PRE_EFIELD_MS ~ TIME_ROI_MS[1])

            # Apply E-field only during TIME_ROI_MS (0 ~ 2 ms)
            if TIME_ROI_MS[0] <= t_rel <= TIME_ROI_MS[1]:
                # Compute phi and apply E-field
                phi_sec = _compute_phi_sections(
                    neuron_model,
                    morph_cache,
                    topo,
                    t_rel,
                    E_field_values,
                    TIME_ROI_MS[0],  # time_start_ms
                    TIME_STEP_MS,
                    E_field_scale,
                    E_FACTOR,
                )
                _apply_phi_to_segments(neuron_model, phi_sec)
            else:
                # No E-field before TIME_ROI_MS[0]
                for sec in neuron_model.all:
                    for seg in sec:
                        seg.e_extracellular = 0.0

            # Record if this is a target time
            if rec_idx < len(record_times_ms) and abs(t_rel - record_times_ms[rec_idx]) < DT_MS / 2:
                vext = float(getattr(soma, "e_extracellular", 0.0))
                vin = float(soma.v)
                vm = float(vin - vext)
                Vm_list.append(vm)
                vext_list.append(vext)
                vin_list.append(vin)
                rec_idx += 1

            h.fadvance()
            step_count += 1
            
            # Progress output
            if step_count % progress_interval == 0:
                progress_pct = 100.0 * step_count / total_steps
                print(f"    Progress: {progress_pct:.1f}% (t={h.t:.2f} ms, {step_count}/{total_steps} steps)")

        print(f"  Simulation completed for position {i}.")
        
        # Pad if needed
        pad_vm = v_init_equilibrium
        pad_vext = 0.0
        pad_vin = v_init_equilibrium
        while len(Vm_list) < len(record_times_ms):
            Vm_list.append(Vm_list[-1] if Vm_list else pad_vm)
            vext_list.append(vext_list[-1] if vext_list else pad_vext)
            vin_list.append(vin_list[-1] if vin_list else pad_vin)

        Vm_data[i, :] = np.asarray(Vm_list, dtype=np.float32)
        vext_data[i, :] = np.asarray(vext_list, dtype=np.float32)
        vin_data[i, :] = np.asarray(vin_list, dtype=np.float32)

    return {
        "cell_id": CELL_ID,
        "positions_um": NEURON_POSITIONS_UM,
        "time_ms": record_times_ms,
        "Vm_mV": Vm_data,
        "vext_mV": vext_data,
        "v_in_mV": vin_data,
        "v_init_equilibrium_mV": v_init_equilibrium,
        "time_roi_ms": TIME_ROI_MS,
        "pre_efield_ms": PRE_EFIELD_MS,
        "e_field_scale": E_field_scale,
    }


def plot_results(data: dict, output_path: str) -> None:
    """Plot results and save to file."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠️ Matplotlib not available, skipping plot")
        return

    positions_um = np.asarray(data["positions_um"])
    time_ms = np.asarray(data["time_ms"])
    Vm_mV = np.asarray(data["Vm_mV"])
    cell_id = str(data.get("cell_id", "allen"))
    e_field_scale = data.get("e_field_scale", 1.0)
    time_roi_ms = data.get("time_roi_ms", None)

    n_pos, n_times = Vm_mV.shape

    # Figure with n_pos rows, shared x only (y축은 독립)
    fig, axes = plt.subplots(
        n_pos, 1,
        figsize=(10, 6),
        sharex=True,
        sharey=False,
    )
    if n_pos == 1:
        axes = [axes]

    for i in range(n_pos):
        ax = axes[i]
        vm = Vm_mV[i, :]
        pos = positions_um[i]

        # Vm만 표시
        ax.plot(
            time_ms,
            vm,
            label="Vm = v - vext",
            color="C0",
            linewidth=1.5,
            linestyle="-",
            zorder=2,
        )

        # E-field 적용 구간 표시 (수직선) - 더 진하게
        if time_roi_ms is not None:
            ax.axvline(time_roi_ms[0], color="black", linestyle="--", linewidth=2.0, alpha=0.8, label="E-field start")
            ax.axvline(time_roi_ms[1], color="black", linestyle="--", linewidth=2.0, alpha=0.8, label="E-field end")

        # 각 subplot마다 개별 y축 범위 설정
        vm_min = float(vm.min())
        vm_max = float(vm.max())
        if vm_max > vm_min:
            pad = 0.05 * (vm_max - vm_min)
        else:
            pad = 1.0
        y_min = vm_min - pad
        y_max = vm_max + pad
        ax.set_ylim(y_min, y_max)

        ax.set_ylabel("mV")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"pos {i}: (x, y, z) = ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) µm")

        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (ms)")

    # 전체 제목 추가
    title_parts = [f"Cell {cell_id}", f"E-field scale: {e_field_scale:.1f}x"]
    if time_roi_ms is not None:
        title_parts.append(f"E-field: {time_roi_ms[0]:.1f}~{time_roi_ms[1]:.1f} ms")
    fig.suptitle(", ".join(title_parts), fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {output_path}")


def run_simulation() -> None:
    print("=" * 60)
    print("Allen three-position simulation with E-field (multiple scales)")
    print("=" * 60)

    # Load E-field data
    print("\n--- Load E-field data ---")
    try:
        E_field_values = np.load(E_FIELD_VALUES_FILE)
        print(f"  E-field values loaded: {E_field_values.shape}")
    except Exception as e:
        print(f"❌ 오류: E-field 값 파일을 로드할 수 없습니다: {e}")
        return

    try:
        E_grid_coords_M = np.load(E_GRID_COORDS_FILE)
        E_grid_coords_UM = E_grid_coords_M * 1e6  # Convert m to um
        print(f"  E-field coords loaded: {E_grid_coords_UM.shape}")
    except Exception as e:
        print(f"❌ 오류: E-field 좌표 파일을 로드할 수 없습니다: {e}")
        return

    # Check if positions are outside coil region
    print("\n--- Check positions ---")
    for i, pos in enumerate(NEURON_POSITIONS_UM):
        x, y, z = pos
        inside_coil = (
            (COIL_X_MIN <= x <= COIL_X_MAX)
            & (COIL_Y_MIN <= y <= COIL_Y_MAX)
            & (COIL_Z_MIN <= z <= COIL_Z_MAX)
        )
        if inside_coil:
            print(f"  ⚠️  Position {i} {tuple(pos)} is inside coil region!")
        else:
            print(f"  ✓ Position {i} {tuple(pos)} is outside coil region")

    # Equilibrium Vm (0 V start, no E-field) - 한 번만 계산
    print("\n--- Equilibrium start point (0V, no E-field) ---")
    tmp_model = AllenNeuronModel(x=0, y=0, z=0, cell_id=CELL_ID)
    v_init_equilibrium = get_equilibrium_vm(tmp_model, dt_ms=DT_MS)
    print(f"  Equilibrium Vm: {v_init_equilibrium:.2f} mV")

    # 여러 E-field 스케일로 시뮬레이션 반복
    print(f"\n--- Running simulations for {len(E_FIELD_SCALES)} E-field scales ---")
    print(f"  Scales: {E_FIELD_SCALES}")

    plot_dir = os.path.join(SCRIPT_DIR, "plot")
    os.makedirs(plot_dir, exist_ok=True)

    for scale_idx, E_field_scale in enumerate(E_FIELD_SCALES):
        print("\n" + "=" * 60)
        print(f"Simulation {scale_idx + 1}/{len(E_FIELD_SCALES)}: E-field scale = {E_field_scale:.1f}x")
        print("=" * 60)

        # Run simulation for this scale
        data = run_single_simulation(E_field_scale, E_field_values, E_grid_coords_UM, v_init_equilibrium)

        # Save results
        scale_suffix = f"_{int(E_field_scale)}x" if E_field_scale >= 1.0 else f"_{E_field_scale:.1f}x"
        out_path = os.path.join(OUTPUT_DIR, f"allen_{CELL_ID}_threepos_results{scale_suffix}.npy")
        np.save(out_path, data, allow_pickle=True)
        print(f"\n--- Save results ---")
        print(f"  Saved: {out_path}")
        print(f"  Vm range: {data['Vm_mV'].min():.2f} ~ {data['Vm_mV'].max():.2f} mV")
        print(f"  vext range: {data['vext_mV'].min():.2f} ~ {data['vext_mV'].max():.2f} mV")

        # Plot results
        plot_path = os.path.join(plot_dir, f"allen_{CELL_ID}_threepos_vm_vext{scale_suffix}.png")
        plot_results(data, plot_path)

    print("\n" + "=" * 60)
    print("All simulations completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_simulation()
