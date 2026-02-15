# simulate_allen_v3_mp.py
# ------------------------------------------------------------
# Multiprocessing (spawn) summary-only simulation for WSL/Linux.
#
# Changes requested:
#   1) SIM_DT_MS = 0.05 (match EFIELD_DT_MS)
#   2) Update ONLY a subset of sections (near soma by topology depth)
#   3) Clean np.full(..., np.inf) style (remove "+npinf ... if False else ...")
#   4) Run repeated E-field scales (default: 5x, 10x, 20x)
#   5) Show per-worker progress (each worker has its own tqdm bar)
#
# Output (summary only):
#   - vm0_mV      : initial Vm at t=0 ms
#   - vm_max_mV   : max Vm over the whole sim window
#   - vm_min_mV   : min Vm over the whole sim window
#   - spike_count : number of upward crossings of Vm through 0 mV
#
# Vm definition:
#   V_in  = soma.v
#   V_ext = soma.vext[0]
#   Vm    = V_in - V_ext
#
# Examples:
#   python simulate_allen_v3_mp.py
#   python simulate_allen_v3_mp.py --stride 10 --max_points 2000
#   python simulate_allen_v3_mp.py --workers 60
#   python simulate_allen_v3_mp.py --subset_depth 2
# ------------------------------------------------------------

from __future__ import annotations

import os
import math
import time
import argparse
import shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Set

import warnings

warnings.filterwarnings(
    "ignore",
    message=".*Signature.*numpy.longdouble.*",
    category=UserWarning,
)

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm


# =========================
# Config (default)
# =========================

CELL_ID = "529898751"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

E_FIELD_VALUES_FILE = os.path.join(SCRIPT_DIR, "efield", "E_field_1cycle.npy")
E_GRID_COORDS_FILE = os.path.join(SCRIPT_DIR, "efield", "E_field_grid_coords.npy")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TMP_DIR = os.path.join(OUTPUT_DIR, "_tmp_sim_allen_v3_mp")
os.makedirs(TMP_DIR, exist_ok=True)

# E-field timing
EFIELD_DT_MS = 0.05
SIM_DT_MS = 0.05  # requested: match EFIELD_DT_MS
EFIELD_ON_T0_MS = 0.0
EFIELD_ON_T1_MS = 4.0

# Sim window
TSTOP_REL_MS = 4.0

# Default E-field scales for repeated runs
# requested: 1x, 10e7x, 10e8x, 10e9x
DEFAULT_EFIELD_SCALES = (1.0, 10e7, 10e8, 10e9)

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

# Spike definition: upward crossing through 0 mV
SPIKE_THR_MV = 0.0


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    if seconds < 3600:
        return f"{seconds/60:.1f} min"
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    return f"{hours} h {mins} min"


def build_time_vector(tstop_ms: float, dt_ms: float) -> np.ndarray:
    n = int(round(tstop_ms / dt_ms)) + 1
    return (np.arange(n, dtype=np.float64) * dt_ms)


def chunk_list(arr: np.ndarray, n_chunks: int) -> List[np.ndarray]:
    if n_chunks <= 1:
        return [arr]
    n = arr.size
    chunks: List[np.ndarray] = []
    base = n // n_chunks
    rem = n % n_chunks
    s = 0
    for i in range(n_chunks):
        extra = 1 if i < rem else 0
        e = s + base + extra
        if s < e:
            chunks.append(arr[s:e])
        s = e
    return chunks


# =========================
# Worker code
# =========================

@dataclass
class MorphCache:
    n: int
    arc: List[float]
    dl: List[Tuple[float, float, float]]
    mid_spidx_all: List[int]


def _init_worker_tqdm(lock) -> None:
    # Ensure all tqdm bars from different processes use one lock
    tqdm.set_lock(lock)


def worker_run(
    worker_id: int,
    idx_chunk: np.ndarray,           # indices into coords_out_um (outside-only list)
    positions_chunk: np.ndarray,     # (Nw,3) um
    subset_depth: int,
    efield_scale: float,
) -> str:
    """
    Run one worker chunk and save a temp .npy result.
    Returns temp file path.
    """
    import warnings as _warnings
    _warnings.filterwarnings(
        "ignore",
        message=".*Signature.*numpy.longdouble.*",
        category=UserWarning,
    )

    # NEURON inside worker (spawn-safe)
    from neuron import h
    from neuron.units import ms, mV
    from model_allen_neuron import AllenNeuronModel

    def coil_inside_mask(coords_um: np.ndarray) -> np.ndarray:
        x = coords_um[:, 0]
        y = coords_um[:, 1]
        z = coords_um[:, 2]
        return (
            (x >= COIL_BOX_UM["x_min"]) & (x <= COIL_BOX_UM["x_max"]) &
            (y >= COIL_BOX_UM["y_min"]) & (y <= COIL_BOX_UM["y_max"]) &
            (z >= COIL_BOX_UM["z_min"]) & (z <= COIL_BOX_UM["z_max"])
        )

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
        return (
            x0 + (x1 - x0) * segx,
            y0 + (y1 - y0) * segx,
            z0 + (z1 - z0) * segx,
        )

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

    def set_e_extracellular_zero_subset(secs_subset: Set[Any]) -> None:
        for sec in secs_subset:
            for seg in sec:
                seg.e_extracellular = 0.0

    def build_morph_cache_outside_only(
        secs_subset: Set[Any],
        tree_out: cKDTree,
        out_to_all: np.ndarray,
    ) -> Dict[Any, MorphCache]:
        cache: Dict[Any, MorphCache] = {}
        for sec in secs_subset:
            n = int(h.n3d(sec=sec))
            if n < 2:
                cache[sec] = MorphCache(n=n, arc=[0.0], dl=[], mid_spidx_all=[])
                continue

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

            _, idx_out_local = tree_out.query(mids, k=1)
            idx_all = out_to_all[idx_out_local].astype(np.int64).tolist()
            cache[sec] = MorphCache(n=n, arc=arc, dl=dl, mid_spidx_all=idx_all)

        return cache

    def build_topology_map(neuron: AllenNeuronModel) -> Dict[Any, Optional[Tuple[Any, float]]]:
        topo: Dict[Any, Optional[Tuple[Any, float]]] = {}
        for sec in neuron.all:
            sref = h.SectionRef(sec=sec)
            if sref.has_parent():
                try:
                    pseg = sref.parentseg()
                    topo[sec] = (pseg.sec, float(pseg.x))
                except Exception:
                    topo[sec] = (sref.parent, 1.0)
            else:
                topo[sec] = None
        return topo

    def select_sections_by_depth(
        neuron: AllenNeuronModel,
        topo: Dict[Any, Optional[Tuple[Any, float]]],
        max_depth: int,
    ) -> Set[Any]:
        soma_sec = neuron.soma
        selected: Set[Any] = set()
        for sec in neuron.all:
            if sec is soma_sec:
                selected.add(sec)
                continue

            depth = 0
            cur = sec
            ok = False
            while True:
                parent = topo.get(cur, None)
                if parent is None:
                    break
                psec, _ = parent
                depth += 1
                if psec is soma_sec:
                    ok = True
                    break
                if depth >= max_depth:
                    # If we haven't reached soma by now, stop
                    break
                cur = psec

            if ok and depth <= max_depth:
                selected.add(sec)

        selected.add(soma_sec)
        return selected

    def get_E_at_index(E: np.ndarray, spatial_idx_all: int, ti: int) -> Tuple[float, float, float]:
        ti = int(ti)
        if ti < 0:
            ti = 0
        if ti > E.shape[2] - 1:
            ti = E.shape[2] - 1

        if E.shape[0] == 3:
            Ex = float(E[0, spatial_idx_all, ti]) * efield_scale
            Ey = float(E[1, spatial_idx_all, ti]) * efield_scale
            Ez = float(E[2, spatial_idx_all, ti]) * efield_scale
        elif E.shape[0] == 2:
            Ex = float(E[0, spatial_idx_all, ti]) * efield_scale
            Ey = 0.0
            Ez = float(E[1, spatial_idx_all, ti]) * efield_scale
        else:
            raise ValueError(f"Unexpected E components: {E.shape[0]}")
        return Ex, Ey, Ez

    def t_to_e_index(t_ms: float, nt_e: int) -> int:
        # piecewise constant: floor index
        i = int(math.floor((t_ms / EFIELD_DT_MS) + 1e-12))
        if i < 0:
            i = 0
        if i > nt_e - 1:
            i = nt_e - 1
        return i

    def compute_phi_sections_integrated_indexed_subset(
        secs_subset: Set[Any],
        cache: Dict[Any, MorphCache],
        topo: Dict[Any, Optional[Tuple[Any, float]]],
        E: np.ndarray,
        ti: int,
    ) -> Dict[Any, Tuple[List[float], List[float]]]:
        phi_sec: Dict[Any, Tuple[List[float], List[float]]] = {}

        def ensure_section_phi(sec) -> None:
            if sec in phi_sec:
                return

            parent = topo.get(sec, None)
            if parent is None:
                phi0 = 0.0
            else:
                psec, px = parent
                if psec in secs_subset:
                    ensure_section_phi(psec)
                    parc, pphis = phi_sec[psec]
                    total_parc = parc[-1] if parc else 0.0
                    phi0 = interp_phi(parc, pphis, px * total_parc)
                else:
                    # parent not in subset: treat as grounded
                    phi0 = 0.0

            data = cache.get(sec, None)
            if data is None:
                phi_sec[sec] = ([0.0], [phi0])
                return

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
                Ex, Ey, Ez = get_E_at_index(E, sp_all, ti)
                dx, dy, dz = dl[i]
                dphi = -(Ex * dx + Ey * dy + Ez * dz) * E_FACTOR
                phis.append(phis[-1] + dphi)

            phi_sec[sec] = (arc, phis)

        for sec in secs_subset:
            ensure_section_phi(sec)

        return phi_sec

    def apply_phi_to_segments_subset(
        secs_subset: Set[Any],
        phi_sec: Dict[Any, Tuple[List[float], List[float]]],
    ) -> None:
        for sec in secs_subset:
            if sec not in phi_sec:
                continue
            arc, phis = phi_sec[sec]
            total_arc = arc[-1] if arc else 0.0
            for seg in sec:
                target_arc = float(seg.x) * total_arc
                seg.e_extracellular = float(interp_phi(arc, phis, target_arc))

    # NEURON base init
    h.load_file("stdrun.hoc")

    # Load E/grid
    E = np.load(E_FIELD_VALUES_FILE)
    nt_e = int(E.shape[2])

    coords_m = np.load(E_GRID_COORDS_FILE)
    coords_um_all = coords_m * 1e6

    inside = coil_inside_mask(coords_um_all)
    outside = ~inside
    coords_out_um = coords_um_all[outside]
    out_to_all = np.flatnonzero(outside)
    tree_out = cKDTree(coords_out_um)

    # Build neuron, soma to origin, snapshot base
    neuron = AllenNeuronModel(x=0.0, y=0.0, z=0.0, cell_id=CELL_ID, verbose=False)
    sx, sy, sz = xyz_at_seg_linear(neuron.soma, 0.5)
    translate_morphology(neuron.all, -sx, -sy, -sz)
    base_snap = snapshot_pt3d(neuron)

    # Build topology once (morphology relative structure is constant)
    topo_all = build_topology_map(neuron)
    secs_subset_base = select_sections_by_depth(neuron, topo_all, max_depth=subset_depth)

    # Equilibrium once per worker
    h.dt = EQ_DT_MS * ms
    h.tstop = EQ_TSTOP_MS * ms
    h.finitialize(EQ_VINIT_MV * mV)
    set_e_extracellular_zero_subset(secs_subset_base)
    steps_eq = int(round(EQ_TSTOP_MS / EQ_DT_MS))
    for _ in range(steps_eq):
        h.fadvance()
    ss = h.SaveState()
    ss.save()

    # Time axis
    t_vec = build_time_vector(TSTOP_REL_MS, SIM_DT_MS)
    nt = t_vec.size

    nlocal = positions_chunk.shape[0]
    vm0 = np.zeros(nlocal, dtype=np.float64)
    vm_max = np.full(nlocal, -np.inf, dtype=np.float64)
    vm_min = np.full(nlocal, np.inf, dtype=np.float64)
    spike_count = np.zeros(nlocal, dtype=np.int32)

    # Per-worker progress bar
    pbar = tqdm(
        total=nlocal,
        desc=f"W{worker_id:02d}",
        position=int(worker_id),
        leave=False,
        ncols=90,
    )

    for li in range(nlocal):
        tx, ty, tz = float(positions_chunk[li, 0]), float(positions_chunk[li, 1]), float(positions_chunk[li, 2])

        restore_pt3d(neuron, base_snap)
        translate_morphology(neuron.all, tx, ty, tz)

        # subset sections after translation are the same objects; reuse the same set
        secs_subset = secs_subset_base

        # Cache only for subset sections (major speed win)
        cache = build_morph_cache_outside_only(secs_subset, tree_out, out_to_all)

        # Restore equilibrium state
        ss.restore()
        h.t = 0.0

        # Reset extracellular only on subset
        set_e_extracellular_zero_subset(secs_subset)
        e_active = False
        last_ei = -1

        prev_vm: Optional[float] = None
        sc = 0

        for k in range(nt):
            t_rel = float(t_vec[k])
            in_window = (t_rel >= EFIELD_ON_T0_MS) and (t_rel <= EFIELD_ON_T1_MS)

            if in_window:
                if not e_active:
                    e_active = True
                    last_ei = -1

                ei = t_to_e_index(t_rel, nt_e)
                if ei != last_ei:
                    phi_sec = compute_phi_sections_integrated_indexed_subset(secs_subset, cache, topo_all, E, ei)
                    apply_phi_to_segments_subset(secs_subset, phi_sec)
                    last_ei = ei
            else:
                if e_active:
                    set_e_extracellular_zero_subset(secs_subset)
                    e_active = False
                    last_ei = -1

            vin = float(neuron.soma(0.5).v)
            try:
                vext = float(neuron.soma(0.5).vext[0])
            except Exception:
                vext = 0.0
            vm = vin - vext

            if k == 0:
                vm0[li] = vm
            if vm > vm_max[li]:
                vm_max[li] = vm
            if vm < vm_min[li]:
                vm_min[li] = vm

            if prev_vm is not None:
                if (prev_vm < SPIKE_THR_MV) and (vm >= SPIKE_THR_MV):
                    sc += 1
            prev_vm = vm

            if k < nt - 1:
                h.fadvance()

        spike_count[li] = np.int32(sc)

        pbar.update(1)

    pbar.close()

    tmp_path = os.path.join(TMP_DIR, f"worker{worker_id:03d}_chunk_{time.strftime('%Y%m%d_%H%M%S')}.npy")
    tmp_payload: Dict[str, Any] = {
        "worker_id": int(worker_id),
        "positions_outside_indices": idx_chunk.astype(np.int64),
        "positions_um": positions_chunk.astype(np.float64),
        "t_ms": t_vec,
        "vm0_mV": vm0,
        "vm_max_mV": vm_max,
        "vm_min_mV": vm_min,
        "spike_count": spike_count,
        "spike_thr_mV": float(SPIKE_THR_MV),
        "subset_depth": int(subset_depth),
        "subset_sections_count": int(len(secs_subset_base)),
    }
    np.save(tmp_path, tmp_payload)
    return tmp_path


# =========================
# Main
# =========================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max_points", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=-1, help="Number of worker processes. Default: cpu_count()-4 (min 1).")
    ap.add_argument("--subset_depth", type=int, default=2, help="Topology depth from soma to include in the subset (default 2).")
    ap.add_argument(
        "--scales",
        type=str,
        default="1,10e7,10e8,10e9",
        help="Comma-separated E-field scale multipliers (default: 1,10e7,10e8,10e9).",
    )
    ap.add_argument("--keep_tmp", action="store_true", help="Keep temp worker files.")
    args = ap.parse_args()

    import multiprocessing as mp
    cpu_total = mp.cpu_count()
    if args.workers is not None and int(args.workers) > 0:
        n_workers = int(args.workers)
    else:
        n_workers = max(1, cpu_total - 4)

    subset_depth = max(0, int(args.subset_depth))

    print("\n=== simulate_allen_v3_mp.py (summary only, fast mode) ===", flush=True)
    print(f"multiprocessing spawn, workers={n_workers} (cpu_total={cpu_total})", flush=True)
    print(f"SIM_DT_MS={SIM_DT_MS:.3f} ms (matched to EFIELD_DT_MS={EFIELD_DT_MS:.3f} ms)", flush=True)
    print(f"Subset update by topology depth from soma: depth={subset_depth}", flush=True)
    print(f"Spike definition: Vm upward crossings through {SPIKE_THR_MV:.1f} mV", flush=True)
    print(f"E-field window: {EFIELD_ON_T0_MS:.3f} to {EFIELD_ON_T1_MS:.3f} ms", flush=True)
    print(f"Sim window: 0 to {TSTOP_REL_MS:.3f} ms", flush=True)

    if args.scales is None or str(args.scales).strip() == "":
        scales = [float(v) for v in DEFAULT_EFIELD_SCALES]
    else:
        scales = []
        for tok in str(args.scales).split(","):
            tok = tok.strip()
            if not tok:
                continue
            scales.append(float(tok))
        if len(scales) == 0:
            scales = [float(v) for v in DEFAULT_EFIELD_SCALES]

    print(f"E-field scales: {', '.join(f'{s:g}x' for s in scales)}", flush=True)

    os.environ.setdefault("ALLEN_NEURON_QUIET", "1")

    coords_m = np.load(E_GRID_COORDS_FILE)
    coords_um = coords_m * 1e6
    if coords_um.ndim != 2 or coords_um.shape[1] != 3:
        raise ValueError(f"coords_um must be (N,3). Got {coords_um.shape}")

    x = coords_um[:, 0]
    y = coords_um[:, 1]
    z = coords_um[:, 2]
    inside = (
        (x >= COIL_BOX_UM["x_min"]) & (x <= COIL_BOX_UM["x_max"]) &
        (y >= COIL_BOX_UM["y_min"]) & (y <= COIL_BOX_UM["y_max"]) &
        (z >= COIL_BOX_UM["z_min"]) & (z <= COIL_BOX_UM["z_max"])
    )
    outside = ~inside
    coords_out_um = coords_um[outside]
    if coords_out_um.shape[0] == 0:
        raise RuntimeError("No outside points. Coil box or grid is wrong.")

    positions_all = coords_out_um

    start = max(0, int(args.start))
    end = int(args.end)
    if end < 0 or end > positions_all.shape[0]:
        end = positions_all.shape[0]
    stride = max(1, int(args.stride))

    idx = np.arange(start, end, stride, dtype=np.int64)
    if args.max_points is not None and int(args.max_points) > 0:
        idx = idx[: int(args.max_points)]

    positions = positions_all[idx]
    npos = positions.shape[0]
    print(f"Targets: outside subset size={npos} (start={start}, end={end}, stride={stride})", flush=True)

    chunks_idx = chunk_list(idx, n_workers)
    chunks_pos: List[np.ndarray] = []
    idx_to_pos = {int(idx[i]): i for i in range(idx.size)}
    for c in chunks_idx:
        local_rows = [idx_to_pos[int(v)] for v in c.tolist()]
        chunks_pos.append(positions[np.array(local_rows, dtype=np.int64)])

    ctx = mp.get_context("spawn")

    # tqdm multiprocessing lock
    lock = ctx.RLock()
    tqdm.set_lock(lock)

    for efield_scale in scales:
        print(f"\n--- Running scale {efield_scale:g}x ---", flush=True)

        # Clean tmp dir per scale
        if os.path.isdir(TMP_DIR):
            for fn in os.listdir(TMP_DIR):
                if fn.endswith(".npy"):
                    try:
                        os.remove(os.path.join(TMP_DIR, fn))
                    except Exception:
                        pass

        t0 = time.time()
        tmp_paths: List[str] = []

        # Main overall bar (optional): count finished workers
        overall = tqdm(total=len(chunks_idx), desc=f"Workers done ({efield_scale:g}x)", position=0, leave=True, ncols=90)

        with ctx.Pool(
            processes=n_workers,
            initializer=_init_worker_tqdm,
            initargs=(lock,),
        ) as pool:
            jobs = []
            for wid in range(len(chunks_idx)):
                jobs.append(
                    pool.apply_async(
                        worker_run,
                        kwds=dict(
                            worker_id=wid + 1,  # +1 to keep position=0 for the overall bar
                            idx_chunk=chunks_idx[wid],
                            positions_chunk=chunks_pos[wid],
                            subset_depth=subset_depth,
                            efield_scale=float(efield_scale),
                        ),
                    )
                )

            for j in jobs:
                tmp_paths.append(j.get())
                overall.update(1)

        overall.close()

        elapsed = time.time() - t0
        print(f"\nWorkers done. Wall time: {format_time(elapsed)}", flush=True)
        print(f"Temp results: {len(tmp_paths)} files", flush=True)

        # Merge results
        t_vec = build_time_vector(TSTOP_REL_MS, SIM_DT_MS)
        vm0 = np.zeros(npos, dtype=np.float64)
        vm_max = np.full(npos, -np.inf, dtype=np.float64)
        vm_min = np.full(npos, np.inf, dtype=np.float64)
        spike_count = np.zeros(npos, dtype=np.int32)

        outside_to_row = {int(idx[i]): i for i in range(idx.size)}

        subset_sections_count_any = None

        for p in tmp_paths:
            payload = np.load(p, allow_pickle=True).item()

            idx_chunk = np.asarray(payload["positions_outside_indices"], dtype=np.int64)
            vm0_c = np.asarray(payload["vm0_mV"], dtype=np.float64)
            vmax_c = np.asarray(payload["vm_max_mV"], dtype=np.float64)
            vmin_c = np.asarray(payload["vm_min_mV"], dtype=np.float64)
            sc_c = np.asarray(payload["spike_count"], dtype=np.int32)

            if subset_sections_count_any is None:
                subset_sections_count_any = int(payload.get("subset_sections_count", -1))

            for j in range(idx_chunk.size):
                key = int(idx_chunk[j])
                row = outside_to_row[key]
                vm0[row] = vm0_c[j]
                vm_max[row] = vmax_c[j]
                vm_min[row] = vmin_c[j]
                spike_count[row] = sc_c[j]

        if abs(float(efield_scale) - round(float(efield_scale))) < 1e-12:
            scale_tag = str(int(round(float(efield_scale))))
        else:
            scale_tag = str(float(efield_scale)).rstrip("0").rstrip(".").replace(".", "p")
        out_name = f"allpoints_{scale_tag}x_cell{CELL_ID}.npy"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        final_payload: Dict[str, Any] = {
            "cell_id": CELL_ID,
            "coil_box_um": COIL_BOX_UM,
            "efield_values_file": E_FIELD_VALUES_FILE,
            "efield_grid_file": E_GRID_COORDS_FILE,
            "efield_dt_ms": float(EFIELD_DT_MS),
            "sim_dt_ms": float(SIM_DT_MS),
            "efield_scale": float(efield_scale),
            "efield_on_window_ms": (float(EFIELD_ON_T0_MS), float(EFIELD_ON_T1_MS)),
            "t_ms": t_vec,
            "positions_um": positions.astype(np.float64),
            "positions_outside_indices": idx.astype(np.int64),
            "subset_depth": int(subset_depth),
            "subset_sections_count": int(subset_sections_count_any if subset_sections_count_any is not None else -1),
            "definition": {
                "V_in": "soma.v (intracellular)",
                "V_ext": "soma.vext[0] (extracellular)",
                "Vm": "V_in - V_ext (membrane potential)",
                "spike_count": f"count of upward crossings of Vm through {SPIKE_THR_MV} mV",
            },
            "spike_thr_mV": float(SPIKE_THR_MV),
            "vm0_mV": vm0,
            "vm_max_mV": vm_max,
            "vm_min_mV": vm_min,
            "spike_count": spike_count,
            "workers": int(n_workers),
            "tmp_files_count": int(len(tmp_paths)),
            "elapsed_seconds": float(elapsed),
        }

        np.save(out_path, final_payload)
        print(f"\nSaved: {out_path}", flush=True)
        print("\n=== Summary ===", flush=True)
        print(f"N points: {npos}", flush=True)
        print(f"Scale: {efield_scale:g}x", flush=True)
        print(f"Subset depth: {subset_depth}, subset sections: {final_payload['subset_sections_count']}", flush=True)
        print(f"Vm max over points: {float(np.max(vm_max)):.6f} mV", flush=True)
        print(f"Vm min over points: {float(np.min(vm_min)):.6f} mV", flush=True)
        print(f"Total spikes (sum): {int(np.sum(spike_count))}", flush=True)
        print(f"Points with spikes: {int(np.sum(spike_count > 0))}", flush=True)
        print(f"Elapsed: {format_time(elapsed)}", flush=True)

    if not args.keep_tmp:
        try:
            shutil.rmtree(TMP_DIR, ignore_errors=True)
            os.makedirs(TMP_DIR, exist_ok=True)
        except Exception:
            pass
    else:
        print(f"[INFO] keep_tmp=True, temp dir kept at: {TMP_DIR}", flush=True)


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
