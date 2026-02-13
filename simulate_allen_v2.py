# simulate_allen_v2.py
# ------------------------------------------------------------
# Run Allen cell simulation for MANY spatial points using the
# same integrated E-field method as simulate_four_A_coil_excluded.py,
# while excluding the coil region at the GRID level.
#
# Key definitions (consistent with your final code):
#   V_in  = soma.v           (intracellular)
#   V_ext = soma.vext[0]     (extracellular)
#   vm    = V_in - V_ext     (membrane potential, computed in analysis/output)
#
# Default behavior:
#   - Iterate over ALL OUTSIDE-COIL E-field grid points (coords_out_um)
#   - For each point: restore base morphology, translate soma to point,
#     build cache/topology, restore equilibrium state, simulate,
#     store **full time traces** (V_in, V_ext, vm) + summary metrics.
#
# Output:
#   ./output/allpoints_cell<CELL_ID>_outsideOnly_integrated_<timestamp>.npy
#
# Examples:
#   python simulate_allen_v2.py
#   python simulate_allen_v2.py --stride 10
#   python simulate_allen_v2.py --max_points 1000
#   python simulate_allen_v2.py --start 0 --end 5000
#   python simulate_allen_v2.py --no_traces --max_points 50   # summary only
# ------------------------------------------------------------

from __future__ import annotations

import os
import math
import time
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

from neuron import h
from neuron.units import ms, mV

from model_allen_neuron import AllenNeuronModel

h.load_file("stdrun.hoc")


# =========================
# Config (default)
# =========================

CELL_ID = "529898751"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

E_FIELD_VALUES_FILE = os.path.join(SCRIPT_DIR, "efield", "E_field_1cycle.npy")
E_GRID_COORDS_FILE = os.path.join(SCRIPT_DIR, "efield", "E_field_grid_coords.npy")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# E-field timing
EFIELD_DT_MS = 0.05          # 50 us
SIM_DT_MS = 0.025            # 25 us
EFIELD_ON_T0_MS = 0.0
EFIELD_ON_T1_MS = 4.0        # apply E-field from 0 ~ 4 ms

# Sim window
TSTOP_REL_MS = 4.0

# E-field scale
E_FIELD_SCALE = 1.0

# Conversion: (V/m * um) -> mV
E_FACTOR = 1e-3

# Coil box in um (excluded region)
COIL_BOX_UM = {
    "x_min": -79.5, "x_max": 79.5,
    "y_min": -32.0, "y_max": 32.0,
    "z_min": 498.0, "z_max": 1502.0,
}

# Equilibrium
EQ_TSTOP_MS = 200.0
EQ_DT_MS = SIM_DT_MS
EQ_VINIT_MV = -65.0

# Checks
VM_EXTREME_WARN_MV = 200.0
E_EXTREME_WARN_VPM = 1e6   # V/m
MAPCHECK_PRINT_FIRST_NSECS = 5


def format_time(seconds: float) -> str:
    """Human-friendly time formatting (for ETA prints)."""
    if seconds < 60:
        return f"{seconds:.1f} s"
    if seconds < 3600:
        return f"{seconds/60:.1f} min"
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    return f"{hours} h {mins} min"


# =========================
# Helpers
# =========================

def coil_inside_mask(coords_um: np.ndarray) -> np.ndarray:
    x = coords_um[:, 0]
    y = coords_um[:, 1]
    z = coords_um[:, 2]
    inside = (
        (x >= COIL_BOX_UM["x_min"]) & (x <= COIL_BOX_UM["x_max"]) &
        (y >= COIL_BOX_UM["y_min"]) & (y <= COIL_BOX_UM["y_max"]) &
        (z >= COIL_BOX_UM["z_min"]) & (z <= COIL_BOX_UM["z_max"])
    )
    return inside


def xyz_at_seg_linear(sec, segx: float) -> Tuple[float, float, float]:
    n = int(h.n3d(sec=sec))
    if n < 2:
        return 0.0, 0.0, 0.0

    x0 = h.x3d(0, sec=sec)
    y0 = h.y3d(0, sec=sec)
    z0 = h.z3d(0, sec=sec)
    x1 = h.x3d(n - 1, sec=sec)
    y1 = h.y3d(n - 1, sec=sec)
    z1 = h.z3d(n - 1, sec=sec)

    x = x0 + (x1 - x0) * segx
    y = y0 + (y1 - y0) * segx
    z = z0 + (z1 - z0) * segx
    return x, y, z


def translate_morphology(all_secs, dx: float, dy: float, dz: float) -> None:
    for sec in all_secs:
        n = int(h.n3d(sec=sec))
        for i in range(n):
            x = h.x3d(i, sec=sec) + dx
            y = h.y3d(i, sec=sec) + dy
            z = h.z3d(i, sec=sec) + dz
            d = h.diam3d(i, sec=sec)
            h.pt3dchange(i, x, y, z, d, sec=sec)
    h.define_shape()


def interp_phi(arc_list: List[float], phi_list: List[float], target_arc: float) -> float:
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


@dataclass
class MorphCache:
    n: int
    arc: List[float]                          # pt3d arc positions (um)
    dl: List[Tuple[float, float, float]]      # vectors between pt3d points (um)
    mid_spidx_all: List[int]                  # mapped to ALL-grid index


def build_morph_cache_outside_only(
    neuron: AllenNeuronModel,
    tree_out: cKDTree,
    out_to_all: np.ndarray,
) -> Tuple[Dict, Dict]:
    cache: Dict = {}
    topo: Dict = {}

    for sec in neuron.all:
        n = int(h.n3d(sec=sec))
        if n < 2:
            cache[sec] = MorphCache(n=n, arc=[0.0], dl=[], mid_spidx_all=[])
        else:
            xs = [h.x3d(i, sec=sec) for i in range(n)]
            ys = [h.y3d(i, sec=sec) for i in range(n)]
            zs = [h.z3d(i, sec=sec) for i in range(n)]
            arc = [h.arc3d(i, sec=sec) for i in range(n)]

            dl: List[Tuple[float, float, float]] = []
            mids = np.zeros((n - 1, 3), dtype=np.float64)

            for i in range(n - 1):
                dx = xs[i + 1] - xs[i]
                dy = ys[i + 1] - ys[i]
                dz = zs[i + 1] - zs[i]
                dl.append((dx, dy, dz))

                mids[i, 0] = 0.5 * (xs[i] + xs[i + 1])
                mids[i, 1] = 0.5 * (ys[i] + ys[i + 1])
                mids[i, 2] = 0.5 * (zs[i] + zs[i + 1])

            _, idx_out = tree_out.query(mids, k=1)
            idx_all = out_to_all[idx_out].astype(np.int64).tolist()

            cache[sec] = MorphCache(n=n, arc=arc, dl=dl, mid_spidx_all=idx_all)

        sref = h.SectionRef(sec=sec)
        if sref.has_parent():
            try:
                pseg = sref.parentseg()
                topo[sec] = (pseg.sec, float(pseg.x))
            except Exception:
                topo[sec] = (sref.parent, 1.0)
        else:
            topo[sec] = None

    return cache, topo


def report_mapping_sanity_for_neuron(
    cache: Dict,
    inside_mask_all: np.ndarray,
    label: str,
    max_secs_print: int = MAPCHECK_PRINT_FIRST_NSECS,
) -> float:
    all_mid: List[int] = []
    nsec = 0
    for sec, data in cache.items():
        mids = data.mid_spidx_all if isinstance(data, MorphCache) else []
        if mids:
            all_mid.extend(mids)
        if nsec < max_secs_print and mids:
            frac = float(np.mean(inside_mask_all[np.array(mids, dtype=np.int64)]))
            print(f"[MAP CHECK] {label} sec={sec.name()} inside_fraction={frac*100:.6f}%  (n={len(mids)})")
            nsec += 1

    if len(all_mid) == 0:
        print(f"[MAP CHECK] {label}: no midpoints found (unexpected).")
        return 1.0

    all_mid_arr = np.array(all_mid, dtype=np.int64)
    frac_inside = float(np.mean(inside_mask_all[all_mid_arr]))
    print(f"[MAP CHECK] {label}: mapped-to-coil-inside fraction = {frac_inside*100:.8f}% (expected 0.0%)")
    return frac_inside


def get_E_at_time_interp(E: np.ndarray, spatial_idx_all: int, t_ms: float) -> Tuple[float, float, float]:
    Tmax = E.shape[2] - 1
    f = t_ms / EFIELD_DT_MS
    i0 = int(math.floor(f))
    if i0 < 0:
        i0 = 0
    if i0 > Tmax:
        i0 = Tmax
    i1 = min(i0 + 1, Tmax)
    w = f - i0
    if w < 0.0:
        w = 0.0
    if w > 1.0:
        w = 1.0

    if E.shape[0] == 3:
        Ex0, Ey0, Ez0 = E[0, spatial_idx_all, i0], E[1, spatial_idx_all, i0], E[2, spatial_idx_all, i0]
        Ex1, Ey1, Ez1 = E[0, spatial_idx_all, i1], E[1, spatial_idx_all, i1], E[2, spatial_idx_all, i1]
        Ex = Ex0 + w * (Ex1 - Ex0)
        Ey = Ey0 + w * (Ey1 - Ey0)
        Ez = Ez0 + w * (Ez1 - Ez0)
    elif E.shape[0] == 2:
        Ex0, Ez0 = E[0, spatial_idx_all, i0], E[1, spatial_idx_all, i0]
        Ex1, Ez1 = E[0, spatial_idx_all, i1], E[1, spatial_idx_all, i1]
        Ex = Ex0 + w * (Ex1 - Ex0)
        Ey = 0.0
        Ez = Ez0 + w * (Ez1 - Ez0)
    else:
        raise ValueError(f"Unexpected E components: {E.shape[0]}")

    return (float(Ex) * E_FIELD_SCALE, float(Ey) * E_FIELD_SCALE, float(Ez) * E_FIELD_SCALE)


def compute_phi_sections_integrated(
    neuron: AllenNeuronModel,
    cache: Dict,
    topo: Dict,
    E: np.ndarray,
    t_rel_ms: float,
) -> Dict:
    phi_sec: Dict = {}

    def ensure_section_phi(sec):
        if sec in phi_sec:
            return

        parent = topo.get(sec, None)
        if parent is None:
            phi0 = 0.0
        else:
            psec, px = parent
            ensure_section_phi(psec)
            parc, pphis = phi_sec[psec]
            total_parc = parc[-1] if len(parc) else 0.0
            phi0 = interp_phi(parc, pphis, px * total_parc)

        data: MorphCache = cache[sec]
        n = data.n
        if n < 2:
            phi_sec[sec] = ([0.0], [phi0])
            return

        arc = data.arc
        dl = data.dl
        mid_spidx_all = data.mid_spidx_all

        phis = [phi0]
        for i in range(n - 1):
            sp_all = mid_spidx_all[i]
            Ex, Ey, Ez = get_E_at_time_interp(E, sp_all, t_rel_ms)
            dx, dy, dz = dl[i]
            dphi = -(Ex * dx + Ey * dy + Ez * dz) * E_FACTOR
            phis.append(phis[-1] + dphi)

        phi_sec[sec] = (arc, phis)

    for sec in neuron.all:
        ensure_section_phi(sec)

    return phi_sec


def apply_phi_to_segments(neuron: AllenNeuronModel, phi_sec: Dict) -> None:
    for sec in neuron.all:
        if sec not in phi_sec:
            continue
        arc, phis = phi_sec[sec]
        total_arc = arc[-1] if len(arc) else 0.0
        for seg in sec:
            target_arc = float(seg.x) * total_arc
            seg.e_extracellular = float(interp_phi(arc, phis, target_arc))


def set_all_e_extracellular_zero(neuron: AllenNeuronModel) -> None:
    for sec in neuron.all:
        for seg in sec:
            seg.e_extracellular = 0.0


def efield_scale_check_outside(E: np.ndarray, outside_mask: np.ndarray) -> None:
    # exact max on outside, and quick p99 sample
    E_out = E[:, outside_mask, :]
    # max|E| across components/time/spatial
    max_abs = float(np.max(np.abs(E_out)))
    # sample percentiles on |E| at a sparse subset to keep it cheap
    nsp = E_out.shape[1]
    nt = E_out.shape[2]
    if nsp == 0:
        print("[WARN] No outside points. Check coil mask.")
        return
    sample_sp = np.linspace(0, nsp - 1, num=min(5000, nsp), dtype=np.int64)
    sample_t = np.linspace(0, nt - 1, num=min(200, nt), dtype=np.int64)
    # |E| magnitude using components
    Es = E_out[:, sample_sp][:, :, sample_t]  # (C, S, T)
    if Es.shape[0] == 3:
        mag = np.sqrt(Es[0] ** 2 + Es[1] ** 2 + Es[2] ** 2)
    else:
        mag = np.sqrt(Es[0] ** 2 + Es[1] ** 2)
    p99 = float(np.percentile(mag, 99.0))
    p999 = float(np.percentile(mag, 99.9))
    print(f"[E CHECK outside] max|E|={max_abs:.6g} V/m (exact), p99|E|~{p99:.6g} V/m, p99.9|E|~{p999:.6g} V/m")
    if max_abs > E_EXTREME_WARN_VPM:
        print("[WARN] max|E| outside가 1e6 V/m를 초과합니다. 단위/스케일 문제가 있을 수 있습니다.")


def snapshot_pt3d(neuron: AllenNeuronModel) -> Dict[Any, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    snap: Dict[Any, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for sec in neuron.all:
        n = int(h.n3d(sec=sec))
        xs = np.array([h.x3d(i, sec=sec) for i in range(n)], dtype=np.float64)
        ys = np.array([h.y3d(i, sec=sec) for i in range(n)], dtype=np.float64)
        zs = np.array([h.z3d(i, sec=sec) for i in range(n)], dtype=np.float64)
        ds = np.array([h.diam3d(i, sec=sec) for i in range(n)], dtype=np.float64)
        snap[sec] = (xs, ys, zs, ds)
    return snap


def restore_pt3d(neuron: AllenNeuronModel, snap: Dict[Any, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> None:
    for sec in neuron.all:
        xs, ys, zs, ds = snap[sec]
        n = int(h.n3d(sec=sec))
        if n != xs.size:
            raise RuntimeError(f"pt3d count changed for {sec.name()}: {n} vs {xs.size}")
        for i in range(n):
            h.pt3dchange(i, float(xs[i]), float(ys[i]), float(zs[i]), float(ds[i]), sec=sec)
    h.define_shape()


def build_time_vector(tstop_ms: float, dt_ms: float) -> np.ndarray:
    n = int(round(tstop_ms / dt_ms)) + 1
    return (np.arange(n, dtype=np.float64) * dt_ms)


# =========================
# Main
# =========================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0, help="Start index in outside grid points list.")
    ap.add_argument("--end", type=int, default=-1, help="End index (exclusive) in outside grid points list. -1 means to the end.")
    ap.add_argument("--stride", type=int, default=1, help="Stride over outside grid points list.")
    ap.add_argument("--max_points", type=int, default=-1, help="Cap number of simulated points after slicing/stride. -1 means no cap.")
    ap.add_argument("--no_traces", action="store_true", help="Do NOT save full time traces (V_in, V_ext, vm). Default: save full traces.")
    ap.add_argument("--mapcheck", action="store_true", help="Print mapping sanity check for every point (slow). Default prints only first few points.")
    ap.add_argument("--warmup_points", type=int, default=3, help="How many initial points to print extra debug/sanity info.")
    args = ap.parse_args()
    save_traces = not args.no_traces

    print("\n=== simulate_allen_v2.py ===")
    print(f"Cell ID: {CELL_ID}")
    print(f"E-field values: {E_FIELD_VALUES_FILE}")
    print(f"E-field grid:   {E_GRID_COORDS_FILE}")
    print(f"E-field window: {EFIELD_ON_T0_MS:.3f} ~ {EFIELD_ON_T1_MS:.3f} ms")
    print(f"Sim window:     0 ~ {TSTOP_REL_MS:.3f} ms, dt={SIM_DT_MS:.3f} ms")
    print(f"E-field dt:     {EFIELD_DT_MS:.3f} ms")

    # Load E-field and coords
    E = np.load(E_FIELD_VALUES_FILE)  # (C, Nspatial, Nt)
    coords_m = np.load(E_GRID_COORDS_FILE)
    coords_um = coords_m * 1e6

    if coords_um.ndim != 2 or coords_um.shape[1] != 3:
        raise ValueError(f"coords_um must be (N,3). Got {coords_um.shape}")

    nspatial = coords_um.shape[0]
    print(f"Loaded E-field shape: {E.shape} (V/m)")
    print(f"Loaded grid shape:   {coords_um.shape} (um)")

    # Coil mask
    inside = coil_inside_mask(coords_um)
    outside = ~inside
    n_inside = int(np.sum(inside))
    n_outside = int(np.sum(outside))
    print(f"Coil mask: inside={n_inside} ({100*n_inside/nspatial:.2f}%), outside={n_outside} ({100*n_outside/nspatial:.2f}%)")

    # E-field scale sanity on outside
    efield_scale_check_outside(E, outside)

    # Build KDTree on OUTSIDE points only
    coords_out_um = coords_um[outside]
    out_to_all = np.flatnonzero(outside)
    if coords_out_um.shape[0] == 0:
        raise RuntimeError("No outside points. Coil box may be wrong or grid is fully inside.")
    tree_out = cKDTree(coords_out_um)
    print("KDTree built on OUTSIDE points only.")

    # Build list of target positions = all outside grid points
    positions_all = coords_out_um  # (N_out, 3) in um

    # Slice/stride
    start = max(0, int(args.start))
    end = int(args.end)
    if end < 0 or end > positions_all.shape[0]:
        end = positions_all.shape[0]
    stride = max(1, int(args.stride))

    idx = np.arange(start, end, stride, dtype=np.int64)
    if args.max_points is not None and int(args.max_points) > 0:
        idx = idx[: int(args.max_points)]

    positions = positions_all[idx]  # (N,3)
    npos = positions.shape[0]
    print(f"Targets: outside grid points subset size = {npos} (start={start}, end={end}, stride={stride})")

    # Instantiate neuron once
    neuron = AllenNeuronModel(x=0.0, y=0.0, z=0.0, cell_id=CELL_ID)

    # Move soma to origin as base alignment
    sx, sy, sz = xyz_at_seg_linear(neuron.soma, 0.5)
    translate_morphology(neuron.all, -sx, -sy, -sz)

    # Snapshot base morphology at soma-centered origin
    base_snap = snapshot_pt3d(neuron)

    # Equilibrium
    print("\n=== Equilibrium run (E-field OFF), SaveState.save() ===")
    h.dt = EQ_DT_MS * ms
    h.tstop = EQ_TSTOP_MS * ms
    h.finitialize(EQ_VINIT_MV * mV)
    set_all_e_extracellular_zero(neuron)

    steps_eq = int(round(EQ_TSTOP_MS / EQ_DT_MS))
    for _ in tqdm(range(steps_eq), desc="Equilibrating", ncols=90):
        h.fadvance()

    eq_v = float(neuron.soma(0.5).v)
    print(f"Equilibrium soma V_in (soma.v): {eq_v:.6f} mV")

    ss = h.SaveState()
    ss.save()
    print("SaveState saved.")

    # Time axis for main sim
    t_vec = build_time_vector(TSTOP_REL_MS, SIM_DT_MS)
    nt = t_vec.size

    # Output arrays (summary)
    max_abs_V_in = np.zeros(npos, dtype=np.float64)
    max_abs_V_ext = np.zeros(npos, dtype=np.float64)
    max_abs_vm = np.zeros(npos, dtype=np.float64)

    # window-specific metrics
    in_mask = (t_vec >= EFIELD_ON_T0_MS) & (t_vec <= EFIELD_ON_T1_MS)
    out_mask = ~in_mask

    # peak-to-peak on demeaned vm in window
    p2p_vm_in = np.zeros(npos, dtype=np.float64)
    rms_vm_in = np.zeros(npos, dtype=np.float64)

    # For debugging, store baseline at t=0
    vin0 = np.zeros(npos, dtype=np.float64)
    vext0 = np.zeros(npos, dtype=np.float64)
    vm0 = np.zeros(npos, dtype=np.float64)

    # Optional traces (default: save full traces)
    traces_V_in = None
    traces_V_ext = None
    traces_vm = None
    if save_traces:
        traces_V_in = np.zeros((npos, nt), dtype=np.float32)
        traces_V_ext = np.zeros((npos, nt), dtype=np.float32)
        traces_vm = np.zeros((npos, nt), dtype=np.float32)

    # Main loop over points
    print("\n=== Main: iterate points, restore morphology + equilibrium, simulate ===")
    point_time_estimated = False
    for pi in tqdm(range(npos), desc="Points", ncols=90):
        t_point_start = time.time()
        tx, ty, tz = float(positions[pi, 0]), float(positions[pi, 1]), float(positions[pi, 2])

        # Restore base morphology (soma at origin)
        restore_pt3d(neuron, base_snap)

        # Translate soma to target
        translate_morphology(neuron.all, tx, ty, tz)

        # Build cache/topo for this position (outside-only mapping)
        cache, topo = build_morph_cache_outside_only(neuron, tree_out, out_to_all)

        # Mapping sanity check
        if args.mapcheck or (pi < int(args.warmup_points)):
            frac_inside = report_mapping_sanity_for_neuron(cache, inside, label=f"P{pi}")
            if frac_inside > 0.0:
                print("[WARN] mapped-to-coil-inside fraction > 0. This should be 0 for solution A.")

        # Restore equilibrium state for dynamics
        ss.restore()
        h.t = 0.0

        # Ensure E off at start
        set_all_e_extracellular_zero(neuron)

        # Run sim loop (store minimal metrics unless save_traces=False)
        # We compute vm = V_in - V_ext at soma
        V_in_soma = np.zeros(nt, dtype=np.float64) if save_traces else None
        V_ext_soma = np.zeros(nt, dtype=np.float64) if save_traces else None
        vm_soma = np.zeros(nt, dtype=np.float64) if save_traces else None

        # Incremental tracking
        local_max_abs_vin = 0.0
        local_max_abs_vext = 0.0
        local_max_abs_vm = 0.0

        for k in range(nt):
            t_rel = float(t_vec[k])

            # Apply E-field during window
            if (t_rel >= EFIELD_ON_T0_MS) and (t_rel <= EFIELD_ON_T1_MS):
                phi_sec = compute_phi_sections_integrated(
                    neuron=neuron,
                    cache=cache,
                    topo=topo,
                    E=E,
                    t_rel_ms=t_rel,
                )
                apply_phi_to_segments(neuron, phi_sec)
            else:
                set_all_e_extracellular_zero(neuron)

            vin = float(neuron.soma(0.5).v)
            try:
                vext = float(neuron.soma(0.5).vext[0])
            except Exception:
                vext = 0.0
            vm = vin - vext

            # Baseline at t=0
            if k == 0:
                vin0[pi] = vin
                vext0[pi] = vext
                vm0[pi] = vm

            # Update maxima
            avin = abs(vin)
            avext = abs(vext)
            avm = abs(vm)
            if avin > local_max_abs_vin:
                local_max_abs_vin = avin
            if avext > local_max_abs_vext:
                local_max_abs_vext = avext
            if avm > local_max_abs_vm:
                local_max_abs_vm = avm

            if save_traces:
                V_in_soma[k] = vin
                V_ext_soma[k] = vext
                vm_soma[k] = vm

            # Extreme warning
            if local_max_abs_vin > VM_EXTREME_WARN_MV:
                print(f"\n[WARN] |V_in| > {VM_EXTREME_WARN_MV:.1f} mV at point {pi} (t={t_rel:.6f} ms).")
                print("       E-field outside max|E|와 코일 제외 매핑(0%)을 다시 확인하십시오.\n")

            if k < nt - 1:
                h.fadvance()

        # Store summary
        max_abs_V_in[pi] = local_max_abs_vin
        max_abs_V_ext[pi] = local_max_abs_vext
        max_abs_vm[pi] = local_max_abs_vm

        # Window metrics based on vm
        if save_traces:
            dv = vm_soma - float(np.mean(vm_soma))
            dv_in = dv[in_mask]
            if dv_in.size > 0:
                p2p_vm_in[pi] = float(np.max(dv_in) - np.min(dv_in))
                rms_vm_in[pi] = float(np.sqrt(np.mean(dv_in ** 2)))
            else:
                p2p_vm_in[pi] = float("nan")
                rms_vm_in[pi] = float("nan")

            # Save traces
            traces_V_in[pi, :] = V_in_soma.astype(np.float32)
            traces_V_ext[pi, :] = V_ext_soma.astype(np.float32)
            traces_vm[pi, :] = vm_soma.astype(np.float32)
        else:
            # If not saving traces, we keep p2p/rms as NaN (summary only)
            p2p_vm_in[pi] = float("nan")
            rms_vm_in[pi] = float("nan")

        # After first point, estimate total point-loop wall time
        if (not point_time_estimated) and pi == 0:
            elapsed_point = time.time() - t_point_start
            total_est = elapsed_point * npos
            print(
                f"\n[ESTIMATE] ~{format_time(total_est)} for {npos} points "
                f"(~{elapsed_point:.2f} s/point, excluding equilibrium)."
            )
            point_time_estimated = True

    # Save payload
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_name = f"allpoints_cell{CELL_ID}_outsideOnly_integrated_{ts}.npy"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    payload: Dict[str, Any] = {
        "cell_id": CELL_ID,
        "coil_box_um": COIL_BOX_UM,
        "efield_values_file": E_FIELD_VALUES_FILE,
        "efield_grid_file": E_GRID_COORDS_FILE,
        "efield_dt_ms": EFIELD_DT_MS,
        "sim_dt_ms": SIM_DT_MS,
        "efield_scale": E_FIELD_SCALE,
        "efield_on_window_ms": (EFIELD_ON_T0_MS, EFIELD_ON_T1_MS),
        "t_ms": t_vec,
        # targets
        "positions_um": positions.astype(np.float64),     # (N,3) outside subset
        "positions_outside_indices": idx.astype(np.int64),# indices into coords_out_um
        # definitions reminder
        "definition": {
            "V_in": "soma.v (intracellular)",
            "V_ext": "soma.vext[0] (extracellular)",
            "vm": "V_in - V_ext (membrane potential, computed)",
        },
        # baselines
        "vin0_mV": vin0,
        "vext0_mV": vext0,
        "vm0_mV": vm0,
        # maxima
        "max_abs_V_in_mV": max_abs_V_in,
        "max_abs_V_ext_mV": max_abs_V_ext,
        "max_abs_vm_mV": max_abs_vm,
        # optional window metrics (filled only if save_traces=True)
        "p2p_vm_in_mV": p2p_vm_in,
        "rms_vm_in_mV": rms_vm_in,
        "save_traces": bool(save_traces),
    }

    if save_traces:
        payload["V_in_soma_mV"] = traces_V_in
        payload["V_ext_soma_mV"] = traces_V_ext
        payload["vm_soma_mV"] = traces_vm

    np.save(out_path, payload)
    print(f"\nSaved: {out_path}")

    # Quick final summary
    print("\n=== Summary ===")
    print(f"N points: {npos}")
    print(f"max|max_abs_vm|: {float(np.max(max_abs_vm)):.6f} mV")
    print(f"max|max_abs_V_ext|: {float(np.max(max_abs_V_ext)):.6f} mV")
    print("Done.")


if __name__ == "__main__":
    main()
