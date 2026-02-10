# simulate_four_A_coil_excluded.py
# ------------------------------------------------------------
# 4-position sanity simulation with Allen cell model + integrated E-field
# Coil region is excluded at the GRID level by building KDTree only on OUTSIDE points.
# Also includes mapping sanity checks and basic gating/magnitude report (no plotting).
#
# Usage:
#   python simulate_four_A_coil_excluded.py
#
# Output:
#   ./output/sanity_cell<CELL_ID>_4pos_integrated_coilexcl_efield0to2ms_dt0.025ms_<timestamp>.npy
# ------------------------------------------------------------

from __future__ import annotations

import os
import re
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

from neuron import h
from neuron.units import ms, mV

# IMPORTANT: user requested this import path
from model_allen_neuron import AllenNeuronModel

h.load_file("stdrun.hoc")


# =========================
# Config
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
EFIELD_ON_T1_MS = 2.0

# Sim window (relative to 0)
TSTOP_REL_MS = 3.0           # record a little after E-field window

# E-field scale (keep 1.0 for sanity)
E_FIELD_SCALE = 1.0

# Conversion: (V/m * um) -> mV
E_FACTOR = 1e-3

# Positions (um)
POSITIONS_UM = [
    (80.0, 0.0, 550.0),
    (80.0, 35.0, 550.0),
    (0.0, 35.0, 550.0),
    (0.0, 0.0, 0.0),
]

# Coil box in um (excluded region)
COIL_BOX_UM = {
    "x_min": -79.5, "x_max": 79.5,
    "y_min": -32.0, "y_max": 32.0,
    "z_min": 498.0, "z_max": 1502.0,
}

# Equilibrium by SaveState
EQ_TSTOP_MS = 200.0          # enough to settle passive/active gates for this cell
EQ_DT_MS = SIM_DT_MS
EQ_VINIT_MV = -65.0          # NEURON init; Allen model will settle to its own rest

# Debug checks
VM_EXTREME_WARN_MV = 200.0   # if |Vm| exceeds this, print warn
MAPCHECK_PRINT_FIRST_NSECS = 5


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
    """
    Approximate segment coordinate by linear interpolation between pt3d endpoints.
    This is fast and robust, enough for nearest-neighbor mapping.
    """
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
    arc: List[float]              # pt3d arc positions (um)
    dl: List[Tuple[float, float, float]]    # pt3d segment vectors between i and i+1 (um)
    mid_spidx_all: List[int]      # mapped to global grid index (ALL grid index)


def build_morph_cache_outside_only(
    neuron: AllenNeuronModel,
    tree_out: cKDTree,
    out_to_all: np.ndarray,
) -> Tuple[Dict, Dict]:
    """
    Build cache for integrated method:
      - for each pt3d segment midpoint, find nearest OUTSIDE-COIL grid point
      - store the corresponding ALL-grid index
    Also build topology mapping using SectionRef.parentseg.
    """
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
            arc = [h.arc3d(i, sec=sec) for i in range(n)]  # um

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
) -> None:
    """
    Verify that cached mid_spidx_all NEVER points into coil-inside indices.
    """
    all_mid = []
    nsec = 0
    for sec, data in cache.items():
        if isinstance(data, MorphCache):
            mids = data.mid_spidx_all
        else:
            mids = data.get("mid_spidx_all", [])
        if mids:
            all_mid.extend(mids)
        if nsec < max_secs_print and mids:
            frac = float(np.mean(inside_mask_all[np.array(mids, dtype=np.int64)]))
            print(f"[MAP CHECK] {label} sec={sec.name()} inside_fraction={frac*100:.6f}%  (n={len(mids)})")
            nsec += 1

    if len(all_mid) == 0:
        print(f"[MAP CHECK] {label}: no midpoints found (unexpected).")
        return

    all_mid = np.array(all_mid, dtype=np.int64)
    frac_inside = float(np.mean(inside_mask_all[all_mid]))
    print(f"[MAP CHECK] {label}: mapped-to-coil-inside fraction = {frac_inside*100:.8f}% (expected 0.0%)")


def get_E_at_time_interp(E: np.ndarray, spatial_idx_all: int, t_ms: float) -> Tuple[float, float, float]:
    """
    E is (C, Nspatial, Nt). Components: 3 (Ex,Ey,Ez) or 2 (Ex,Ez) with Ey=0.
    t_ms is relative time within E-field file time axis.
    """
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
    """
    Integrate phi along the morphology tree:
      - uses cached dl and cached mid_spidx_all (already coil-excluded)
      - root phi(soma) = 0 mV (gauge)
    """
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
            dphi = -(Ex * dx + Ey * dy + Ez * dz) * E_FACTOR  # mV
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


def check_arrays_finite(name: str, arr: np.ndarray) -> bool:
    ok = True
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        print(f"[FAIL] {name}: NaN/Inf detected")
        ok = False
    else:
        print(f"[OK]   {name}: NaN/Inf not found")
    return ok


def gating_magnitude_report(
    t_ms: np.ndarray,
    Vm: np.ndarray,          # (4, T)
    Vext: np.ndarray,        # (4, T)
    positions_um: List[Tuple[float, float, float]],
    on0: float,
    on1: float,
) -> None:
    print("\n" + "=" * 30)
    print("Gating / magnitude report")
    print("=" * 30)
    print(f"E-field window: {on0:.3f} ~ {on1:.3f} ms")
    print("Units: Vm in mV, Vext in mV, ΔVm in mV (demeaned).")

    in_mask = (t_ms >= on0) & (t_ms <= on1)
    out_mask = ~in_mask

    nin = int(np.sum(in_mask))
    nout = int(np.sum(out_mask))
    print(f"\nWindow samples: in={nin}, out={nout} (tstop_rel_ms={t_ms[-1]:.3f})")

    for i in range(Vm.shape[0]):
        v = Vm[i].astype(np.float64)
        vext = Vext[i].astype(np.float64)

        dv = v - np.mean(v)
        dv_in = dv[in_mask]
        dv_out = dv[out_mask]

        p2p_in = float(np.max(dv_in) - np.min(dv_in)) if dv_in.size else float("nan")
        rms_in = float(np.sqrt(np.mean(dv_in ** 2))) if dv_in.size else float("nan")

        p2p_out = float(np.max(dv_out) - np.min(dv_out)) if dv_out.size else float("nan")
        rms_out = float(np.sqrt(np.mean(dv_out ** 2))) if dv_out.size else float("nan")

        vmax_abs = float(np.max(np.abs(v))) if v.size else float("nan")

        vext_in = vext[in_mask]
        vext_out = vext[out_mask]
        max_vext_in = float(np.max(np.abs(vext_in))) if vext_in.size else float("nan")
        max_vext_out = float(np.max(np.abs(vext_out))) if vext_out.size else float("nan")

        pos = positions_um[i]
        print(f"\n[Neuron {i+1}] position_um=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
        print(f"  Vm:  p2p_in={p2p_in:.6f} mV ({p2p_in*1000:.3f} uV), rms_in={rms_in:.6f} mV ({rms_in*1000:.3f} uV)")
        print(f"       p2p_out={p2p_out:.6f} mV ({p2p_out*1000:.3f} uV), rms_out={rms_out:.6f} mV ({rms_out*1000:.3f} uV)")

        if np.isfinite(p2p_in) and np.isfinite(p2p_out) and (p2p_in > p2p_out * 1.2):
            print("  [OK]   Vm modulation is larger inside the on-window.")
        else:
            print("  [WARN] Vm modulation is not clearly larger inside the on-window.")

        print(f"  Vext: max|in|={max_vext_in:.6f} mV, max|out|={max_vext_out:.6f} mV")

        if vmax_abs > VM_EXTREME_WARN_MV:
            print(f"  [WARN] |Vm| exceeds {VM_EXTREME_WARN_MV:.1f} mV (max|Vm|={vmax_abs:.3f}). 수치 폭주 가능성이 큽니다.")


# =========================
# Main
# =========================

def main() -> None:
    print("\n=== simulate_four_A_coil_excluded.py ===")
    print("Cell ID:", CELL_ID)
    print("E-field values:", E_FIELD_VALUES_FILE)
    print("E-field grid:  ", E_GRID_COORDS_FILE)

    # --- Load E-field and coords ---
    E = np.load(E_FIELD_VALUES_FILE)
    coords_m = np.load(E_GRID_COORDS_FILE)
    coords_um = coords_m * 1e6

    print(f"Loaded E-field shape: {E.shape} (V/m)")
    print(f"Loaded grid shape:   {coords_um.shape} (um)")
    print(f"E-field dt: {EFIELD_DT_MS:.3f} ms, sim dt: {SIM_DT_MS:.3f} ms")

    inside = coil_inside_mask(coords_um)
    outside = ~inside
    n_inside = int(np.sum(inside))
    n_outside = int(np.sum(outside))
    print(f"Coil mask: inside={n_inside} ({100*n_inside/coords_um.shape[0]:.2f}%), outside={n_outside} ({100*n_outside/coords_um.shape[0]:.2f}%)")

    # Build KDTree on OUTSIDE points only (Solution A)
    coords_out_um = coords_um[outside]
    out_to_all = np.flatnonzero(outside)  # map out-index -> all-index
    tree_out = cKDTree(coords_out_um)
    print("KDTree built on OUTSIDE points only.")

    # Allen model folder resolution (informational)
    allen_root = os.path.join(SCRIPT_DIR, "allen_model")
    cell_glob = os.path.join(allen_root, f"{CELL_ID}_*_ephys")
    matches = [p for p in sorted(glob_glob(cell_glob))]
    if matches:
        print("Allen cell dir:", matches[0])
    else:
        print("Allen cell dir: (not resolved by glob here; AllenNeuronModel will resolve internally if it does)")

    # --- Build 4 neurons at target positions ---
    print("\n=== Build 4 neurons and place to targets ===")
    neurons: List[AllenNeuronModel] = []
    caches: List[Dict] = []
    topos: List[Dict] = []

    for i, (tx, ty, tz) in enumerate(POSITIONS_UM):
        neuron = AllenNeuronModel(x=0.0, y=0.0, z=0.0, cell_id=CELL_ID)

        # translate so soma center is at target
        sx, sy, sz = xyz_at_seg_linear(neuron.soma, 0.5)
        translate_morphology(neuron.all, tx - sx, ty - sy, tz - sz)

        sx2, sy2, sz2 = xyz_at_seg_linear(neuron.soma, 0.5)
        print(f"Neuron {i+1}: target=({tx:.1f},{ty:.1f},{tz:.1f}) um, soma~({sx2:.1f},{sy2:.1f},{sz2:.1f}) um")

        neurons.append(neuron)

    # --- Build morphology caches with OUTSIDE-only mapping ---
    print("\n=== Build morphology caches (integrated method, OUTSIDE-only mapping) ===")
    for i, neuron in enumerate(neurons):
        cache, topo = build_morph_cache_outside_only(neuron, tree_out, out_to_all)
        caches.append(cache)
        topos.append(topo)
        print(f"Neuron {i+1}: cache ready")

    # --- Mapping sanity checks (must be 0%) ---
    print("\n=== Mapping sanity checks (must be 0% coil-inside) ===")
    for i in range(4):
        report_mapping_sanity_for_neuron(caches[i], inside, label=f"N{i+1}")

    # --- Equilibrium: run ONLY neuron1 to equilibrium then SaveState.save() ---
    print("\n=== Equilibrium run (E-field OFF), SaveState.save() from Neuron 1 ===")
    h.dt = EQ_DT_MS * ms
    h.tstop = EQ_TSTOP_MS * ms
    h.finitialize(EQ_VINIT_MV * mV)

    # Ensure e_extracellular is 0 during equilibrium
    for neuron in neurons:
        for sec in neuron.all:
            for seg in sec:
                seg.e_extracellular = 0.0

    steps_eq = int(round(EQ_TSTOP_MS / EQ_DT_MS))
    for _ in tqdm(range(steps_eq), desc="Equilibrating", ncols=80):
        h.fadvance()

    eq_vm = [float(neuron.soma(0.5).v) for neuron in neurons]
    print(f"Equilibrium Vm (soma): N1={eq_vm[0]:.3f} mV, N2={eq_vm[1]:.3f} mV, N3={eq_vm[2]:.3f} mV, N4={eq_vm[3]:.3f} mV")

    ss = h.SaveState()
    ss.save()
    print("SaveState saved.\n")

    # --- Main run ---
    print("=== Main run: restore equilibrium, apply E-field in 0~2ms ===")
    h.dt = SIM_DT_MS * ms
    h.tstop = TSTOP_REL_MS * ms

    # restore equilibrium state for whole model
    ss.restore()
    h.t = 0.0

    # recorders
    npos = len(POSITIONS_UM)
    t_vec = np.zeros(int(round(TSTOP_REL_MS / SIM_DT_MS)) + 1, dtype=np.float64)
    Vm_soma = np.zeros((npos, t_vec.size), dtype=np.float64)
    Vext_soma = np.zeros((npos, t_vec.size), dtype=np.float64)

    # simulation loop
    for k in tqdm(range(t_vec.size), desc="Simulating", ncols=80):
        t_rel = k * SIM_DT_MS
        t_vec[k] = t_rel

        # Apply E-field only within window; otherwise set e_extracellular = 0
        if (t_rel >= EFIELD_ON_T0_MS) and (t_rel <= EFIELD_ON_T1_MS):
            for i, neuron in enumerate(neurons):
                phi_sec = compute_phi_sections_integrated(
                    neuron=neuron,
                    cache=caches[i],
                    topo=topos[i],
                    E=E,
                    t_rel_ms=t_rel,
                )
                apply_phi_to_segments(neuron, phi_sec)
        else:
            for neuron in neurons:
                for sec in neuron.all:
                    for seg in sec:
                        seg.e_extracellular = 0.0

        # record after applying phi, before fadvance
        for i, neuron in enumerate(neurons):
            Vm_soma[i, k] = float(neuron.soma(0.5).v)
            try:
                Vext_soma[i, k] = float(neuron.soma(0.5).vext[0])
            except Exception:
                Vext_soma[i, k] = 0.0

        # quick debug: check e_extracellular range for neuron 1 early
        if k in (0, 1, 2):
            e_vals = []
            for sec in neurons[0].all:
                for seg in sec:
                    e_vals.append(float(seg.e_extracellular))
            if e_vals:
                mn, mx = float(np.min(e_vals)), float(np.max(e_vals))
                print(f"[DEBUG k={k} t={t_rel:.6f} ms] N1 e_extracellular: min={mn:.6f} mV, max={mx:.6f} mV, range={(mx-mn):.6f} mV")

            print(f"[DEBUG k={k} t={t_rel:.6f} ms] soma.v={Vm_soma[0,k]:.6f} mV, soma.vext0={Vext_soma[0,k]:.6e} mV")

        # warn if Vm is extreme
        vmax_abs = float(np.max(np.abs(Vm_soma[:, k])))
        if vmax_abs > VM_EXTREME_WARN_MV:
            print(f"\n[WARN] Extreme |Vm| detected at k={k}, t={t_rel:.6f} ms (max|Vm|={vmax_abs:.3f} mV).")
            print("       코일 제외 매핑이 0%인지, 그리고 E-field max|E|를 확인하십시오.\n")

        # advance
        if k < t_vec.size - 1:
            h.fadvance()

    # --- Save output ---
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_name = f"sanity_cell{CELL_ID}_4pos_integrated_coilexcl_efield0to2ms_dt{SIM_DT_MS:.3f}ms_{ts}.npy"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    payload = {
        "cell_id": CELL_ID,
        "positions_um": np.array(POSITIONS_UM, dtype=np.float64),
        "coil_box_um": COIL_BOX_UM,
        "efield_values_file": E_FIELD_VALUES_FILE,
        "efield_grid_file": E_GRID_COORDS_FILE,
        "efield_dt_ms": EFIELD_DT_MS,
        "sim_dt_ms": SIM_DT_MS,
        "efield_scale": E_FIELD_SCALE,
        "efield_on_window_ms": (EFIELD_ON_T0_MS, EFIELD_ON_T1_MS),
        "t_ms": t_vec,
        "Vm_soma_mV": Vm_soma,
        "Vext_soma_mV": Vext_soma,
    }
    np.save(out_path, payload)
    print(f"\nSaved: {out_path}")

    # --- Basic integrity checks + report (no plotting) ---
    print("\n=== Basic integrity checks ===")
    ok = True
    ok &= check_arrays_finite("t_ms", t_vec)
    ok &= check_arrays_finite("Vm_soma_mV", Vm_soma)
    ok &= check_arrays_finite("Vext_soma_mV", Vext_soma)
    if Vm_soma.shape[1] != t_vec.size:
        print("[FAIL] Vm time dimension mismatch.")
        ok = False
    else:
        print("[OK]   Vm time dimension matches t length.")

    gating_magnitude_report(
        t_ms=t_vec,
        Vm=Vm_soma,
        Vext=Vext_soma,
        positions_um=POSITIONS_UM,
        on0=EFIELD_ON_T0_MS,
        on1=EFIELD_ON_T1_MS,
    )

    print("\n=== Quick absolute range summary ===")
    for i, pos in enumerate(POSITIONS_UM):
        v = Vm_soma[i]
        ve = Vext_soma[i]
        print(f"Neuron {i+1} @ ({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}): "
              f"min(Vm)={np.min(v):.6f}, max(Vm)={np.max(v):.6f}, max|Vm|={np.max(np.abs(v)):.6f} mV")
        print(f"          min(Vext)={np.min(ve):.6f}, max(Vext)={np.max(ve):.6f}, max|Vext|={np.max(np.abs(ve)):.6f} mV")

    if ok:
        print("\n[OK] Basic integrity checks passed.")
    else:
        print("\n[WARN] Integrity checks failed. 위 로그를 확인하십시오.")

    print("\nDone.")


def glob_glob(pattern: str) -> List[str]:
    # local glob helper without importing glob at top-level
    import glob
    return glob.glob(pattern)


if __name__ == "__main__":
    main()
