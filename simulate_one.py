# simulate_one.py
"""
Single-position Allen model simulation with E-field at three fixed spatial points.

- Positions (µm): (35, 0, 550), (35, -80, 550), (0, -80, 550)
- For each position, assume a neuron is located there and simulate Vm and vext over time.
- Initial Vm is obtained from a separate equilibrium run (no E-field), similar to get_equilibrium_vm in simulate_allen.py.
- Output: npy file with Vm and vext for 3 positions, and no plotting here.

Use plot_one.py to visualize results (3 subplots, shared y-axis).
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree
from neuron import h
from neuron.units import mV

# Ensure we can import from project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
os.chdir(SCRIPT_DIR)

h.load_file("stdrun.hoc")

from model_allen_neuron import AllenNeuronModel  # noqa: E402


# --- Configuration ---
CELL_ID = "529898751"  # use same cell as simulate_allen.py

E_FIELD_VALUES_FILE = os.path.join(SCRIPT_DIR, "efield", "E_field_4cycle.npy")
E_GRID_COORDS_FILE = os.path.join(SCRIPT_DIR, "efield", "E_field_grid_coords.npy")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Time settings (match simulate_allen.py)
TIME_STEP_US = 50.0   # E-field data time step (us)
TIME_STEP_MS = TIME_STEP_US / 1000.0  # 0.05 ms
DT_MS = 0.05          # Simulation dt (ms)

# E-field를 적용할 시간 구간 (ms): 0 ~ 2 ms
TIME_ROI_MS: Tuple[float, float] = (0.0, 2.0)

# E-field 0인 상태로 선행 시뮬레이션할 구간 (ms): -1 ~ 0 ms
PRE_EFIELD_MS: float = 1.0

# E-field scaling (match simulate_allen.py)
E_FIELD_SCALE = 1.0        # E-field npy is V/m
E_FACTOR = 1e-3            # (V/m * um) -> mV

# Positions for this script (µm)
NEURON_POSITIONS_UM = np.array(
    [
        [80,   0.0, 550.0],
        [80, 35, 550.0],
        [0.0,  35, 550.0],
    ],
    dtype=float,
)


def _xyz_at_seg(sec, segx: float) -> Tuple[float, float, float]:
    """Return 3D coordinates at segment position (copied from simulate_allen.py)."""
    n = int(h.n3d(sec=sec))
    if n < 2:
        return 0.0, 0.0, 0.0
    x0, y0, z0 = h.x3d(0, sec=sec), h.y3d(0, sec=sec), h.z3d(0, sec=sec)
    x1, y1, z1 = h.x3d(n - 1, sec=sec), h.y3d(n - 1, sec=sec), h.z3d(n - 1, sec=sec)
    return x0 + (x1 - x0) * segx, y0 + (y1 - y0) * segx, z0 + (z1 - z0) * segx


def _translate_morphology(all_secs, dx: float, dy: float, dz: float) -> None:
    """Translate all section pt3d coordinates (copied from simulate_allen.py)."""
    for sec in all_secs:
        n = int(h.n3d(sec=sec))
        for i in range(n):
            x = h.x3d(i, sec=sec) + dx
            y = h.y3d(i, sec=sec) + dy
            z = h.z3d(i, sec=sec) + dz
            d = h.diam3d(i, sec=sec)
            h.pt3dchange(i, x, y, z, d, sec=sec)
    h.define_shape()


def _interp_phi(arc_list, phi_list, target_arc: float) -> float:
    """Interpolate phi at target_arc (copied from simulate_allen.py)."""
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


def _build_morph_cache(neuron_model, grid_tree: cKDTree):
    """Build morphology cache for integrated E-field (simplified from simulate_allen.py)."""
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
        arc = [h.arc3d(i, sec=sec) for i in range(n)]
        dl = []
        mid_spidx = []
        for i in range(n - 1):
            dl.append((xs[i + 1] - xs[i], ys[i + 1] - ys[i], zs[i + 1] - zs[i]))
            mx = 0.5 * (xs[i] + xs[i + 1])
            my = 0.5 * (ys[i] + ys[i + 1])
            mz = 0.5 * (zs[i] + zs[i + 1])
            spidx = int(grid_tree.query([[mx, my, mz]], k=1)[1][0])
            mid_spidx.append(spidx)
        cache[sec] = {"n": n, "arc": arc, "dl": dl, "mid_spidx": mid_spidx}

        sref = h.SectionRef(sec=sec)
        topo[sec] = None
        if sref.has_parent():
            try:
                pseg = sref.parentseg()
                topo[sec] = (pseg.sec, float(pseg.x))
            except Exception:
                pass
    return cache, topo


def _get_E_at(
    E_field_values: np.ndarray,
    spatial_idx: int,
    current_time_ms: float,
    time_start_ms: float,
    time_step_ms: float,
    E_field_scale: float,
) -> Tuple[float, float, float]:
    """Return (Ex, Ey, Ez) at given spatial index and time (same logic as simulate_allen._get_E_at)."""
    import math

    actual_time_ms = current_time_ms + time_start_ms
    time_index_float = actual_time_ms / time_step_ms
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
    return Ex * E_field_scale, Ey * E_field_scale, Ez * E_field_scale


def _compute_phi_sections(
    neuron_model,
    morph_cache,
    topo,
    current_time_ms: float,
    E_field_values: np.ndarray,
    E_grid_coords_um: np.ndarray,
    grid_tree: cKDTree,
    time_start_ms: float,
    time_step_ms: float,
    E_field_scale: float,
    E_factor: float,
):
    """Compute extracellular potential phi along section tree (adapted from simulate_allen.py)."""
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
            Ex, Ey, Ez = _get_E_at(
                E_field_values,
                mid_spidx[i],
                current_time_ms,
                time_start_ms,
                time_step_ms,
                E_field_scale,
            )
            dx, dy, dz = dl[i]
            dphi = -(Ex * dx + Ey * dy + Ez * dz) * E_factor
            phis.append(phis[-1] + dphi)
        phi_sec[sec] = (arc, phis)

    for sec in neuron_model.all:
        ensure_section_phi(sec)
    return phi_sec


def _apply_phi_to_segments(neuron_model, phi_sec) -> None:
    """Set e_extracellular on each segment (copied from simulate_allen.py)."""
    for sec in neuron_model.all:
        if sec not in phi_sec:
            continue
        arc, phis = phi_sec[sec]
        total_arc = arc[-1] if arc else 0.0
        for seg in sec:
            target_arc = seg.x * total_arc
            seg.e_extracellular = _interp_phi(arc, phis, target_arc)


def get_equilibrium_vm(neuron_model, dt_ms: float = 0.05) -> float:
    """
    Run 0V start, no E-field until equilibrium.
    Returns soma Vm at equilibrium for use as v_init.
    (same as simulate_allen.get_equilibrium_vm)
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


def run_simulation() -> None:
    print("=" * 60)
    print("Allen single-position simulation at 3 fixed points (NO E-field, 0~1 ms)")
    print("=" * 60)

    # --- Time axis: 0 ~ 1 ms (no E-field) ---
    record_times_ms = np.arange(0.0, 1.0 + 1e-9, DT_MS)
    n_times = len(record_times_ms)

    # Equilibrium Vm (0 V start, no E-field)
    print("\n--- Equilibrium start point (0V, no E-field) ---")
    tmp_model = AllenNeuronModel(x=0, y=0, z=0, cell_id=CELL_ID)
    v_init_equilibrium = get_equilibrium_vm(tmp_model, dt_ms=DT_MS)
    print(f"  Equilibrium Vm: {v_init_equilibrium:.2f} mV")

    n_positions = NEURON_POSITIONS_UM.shape[0]
    Vm_data = np.zeros((n_positions, n_times), dtype=np.float32)
    vext_data = np.zeros((n_positions, n_times), dtype=np.float32)
    vin_data = np.zeros((n_positions, n_times), dtype=np.float32)  # soma v (inside)

    for i, pos_um in enumerate(NEURON_POSITIONS_UM):
        print(f"\n--- Simulating position {i}: {tuple(pos_um)} (um), E-field OFF ---")

        # Create neuron model
        neuron_model = AllenNeuronModel(x=0, y=0, z=0, cell_id=CELL_ID)

        # Move morphology so that soma is at target position
        sx, sy, sz = _xyz_at_seg(neuron_model.soma, 0.5)
        tx, ty, tz = pos_um
        _translate_morphology(neuron_model.all, tx - sx, ty - sy, tz - sz)

        # Ensure no E-field: e_extracellular = 0 everywhere
        for sec in neuron_model.all:
            for seg in sec:
                seg.e_extracellular = 0.0

        # Initialize from equilibrium Vm
        h.finitialize(v_init_equilibrium * mV)
        h.dt = DT_MS
        h.celsius = 34.0
        h.tstop = record_times_ms[-1] + DT_MS

        soma = neuron_model.soma(0.5)

        Vm_list = []
        vext_list = []
        vin_list = []
        rec_idx = 0

        while h.t < h.tstop - 1e-9 and rec_idx < len(record_times_ms):
            t_rel = h.t  # 0 ~ 1 ms

            if abs(t_rel - record_times_ms[rec_idx]) < DT_MS / 2:
                vext = 0.0  # E-field OFF
                vin = float(soma.v)
                vm = float(vin - vext)
                Vm_list.append(vm)
                vext_list.append(vext)
                vin_list.append(vin)
                rec_idx += 1

            h.fadvance()

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

    # Save results
    print("\n--- Save results ---")
    out_path = os.path.join(OUTPUT_DIR, f"allen_{CELL_ID}_threepos_results.npy")
    np.save(
        out_path,
        {
            "cell_id": CELL_ID,
            "positions_um": NEURON_POSITIONS_UM,
            "time_ms": record_times_ms,
            "Vm_mV": Vm_data,
            "vext_mV": vext_data,
            "v_in_mV": vin_data,
            "v_init_equilibrium_mV": v_init_equilibrium,
            "time_roi_ms": TIME_ROI_MS,
        },
        allow_pickle=True,
    )
    print(f"  Saved: {out_path}")
    print(f"  Vm range: {Vm_data.min():.2f} ~ {Vm_data.max():.2f} mV")
    print(f"  vext range: {vext_data.min():.2f} ~ {vext_data.max():.2f} mV")


if __name__ == "__main__":
    run_simulation()
