"""
gradient_four.py
------------------------------------------------------------
4-position sanity simulation with Allen cell model + gradient-driven extracellular forcing.

This script follows the overall structure of simulate_four_v2.py, but replaces
E-field input with directional gradient input:
  Gxx = dEx/dx, Gyy = dEy/dy, Gzz = dEz/dz

Input:
  - data/gradient/grad_Exdx_Eydy_Ezdz_1cycle.npy, shape (3, Nx, Ny, Nz, Nt)
  - efield/E_field_grid_coords.npy, shape (N_spatial, 3) in meters

Output:
  - data/gradient_output/*.npy
------------------------------------------------------------
"""

from __future__ import annotations

import os
import re
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

from neuron import h
from neuron.units import ms, mV

from model_allen_neuron import AllenNeuronModel

h.load_file("stdrun.hoc")


# =========================
# Config
# =========================
CELL_ID = "529898751"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GRAD_VALUES_FILE = os.path.join(
    SCRIPT_DIR, "data", "gradient", "grad_Exdx_Eydy_Ezdz_1cycle.npy"
)
E_GRID_COORDS_FILE = os.path.join(SCRIPT_DIR, "efield", "E_field_grid_coords.npy")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data", "gradient_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Gradient timing
GRAD_DT_MS = 0.05
SIM_DT_MS = 0.025
GRAD_ON_T0_MS = 0.0
GRAD_ON_T1_MS = 4.0
TSTOP_REL_MS = 4.0

# Gain multiplies gradient-driven pseudo dphi.
# dphi (mV) = -gain * g_parallel(V/m^2) * ds_um^2 * 1e-9
DEFAULT_GAINS = tuple([10.0 ** k for k in range(0, 11)])  # 1 ~ 1e10
DEFAULT_GAINS_STR = ",".join(["1"] + [f"1e{k}" for k in range(1, 11)])
GRAD_GAIN = 1.0

# 4 target positions (um)
POSITIONS_UM = [
    (80.0, 0.0, 550.0),
    (80.0, 35.0, 550.0),
    (0.0, 35.0, 550.0),
    (0.0, 0.0, 0.0),
]

# Coil region in um (excluded from KDTree mapping)
# Pentagonal prism:
# Face (y=-32): (-79.5,561), (0,498), (79.5,561), (79.5,1502), (-79.5,1502) in (x,z)
# Other face at y=+32 with same (x,z)
COIL_REGION_UM = {
    "y_min": -32.0,
    "y_max": 32.0,
    "z_tip_min": 498.0,
    "z_shoulder": 561.0,
    "z_max": 1502.0,
    "x_half_max": 79.5,
    "polygon_xz_vertices": [
        (-79.5, 561.0),
        (0.0, 498.0),
        (79.5, 561.0),
        (79.5, 1502.0),
        (-79.5, 1502.0),
    ],
}

# SaveState equilibrium
EQ_TSTOP_MS = 200.0
EQ_DT_MS = SIM_DT_MS
EQ_VINIT_MV = -65.0

VM_EXTREME_WARN_MV = 200.0
MAPCHECK_PRINT_FIRST_NSECS = 5


# =========================
# Helpers
# =========================
def coil_inside_mask(coords_um: np.ndarray) -> np.ndarray:
    x = coords_um[:, 0]
    y = coords_um[:, 1]
    z = coords_um[:, 2]
    y_in = (y >= COIL_REGION_UM["y_min"]) & (y <= COIL_REGION_UM["y_max"])
    z_in = (z >= COIL_REGION_UM["z_tip_min"]) & (z <= COIL_REGION_UM["z_max"])
    x_lim = np.where(
        z <= COIL_REGION_UM["z_shoulder"],
        COIL_REGION_UM["x_half_max"] * (z - COIL_REGION_UM["z_tip_min"])
        / (COIL_REGION_UM["z_shoulder"] - COIL_REGION_UM["z_tip_min"]),
        COIL_REGION_UM["x_half_max"],
    )
    xz_in = z_in & (np.abs(x) <= x_lim)
    inside = y_in & xz_in
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
    arc: List[float]
    dl: List[Tuple[float, float, float]]
    mid_spidx_all: List[int]


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
) -> None:
    all_mid = []
    nsec = 0
    for sec, data in cache.items():
        mids = data.mid_spidx_all if isinstance(data, MorphCache) else data.get("mid_spidx_all", [])
        if mids:
            all_mid.extend(mids)
        if nsec < max_secs_print and mids:
            frac = float(np.mean(inside_mask_all[np.array(mids, dtype=np.int64)]))
            print(f"[MAP CHECK] {label} sec={sec.name()} inside_fraction={frac*100:.6f}% (n={len(mids)})")
            nsec += 1

    if len(all_mid) == 0:
        print(f"[MAP CHECK] {label}: no midpoints found (unexpected).")
        return
    all_mid = np.array(all_mid, dtype=np.int64)
    frac_inside = float(np.mean(inside_mask_all[all_mid]))
    print(f"[MAP CHECK] {label}: mapped-to-coil-inside fraction = {frac_inside*100:.8f}% (expected 0.0%)")


def build_all_to_grid_indices(coords_m: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_axis = np.unique(coords_m[:, 0])
    y_axis = np.unique(coords_m[:, 1])
    z_axis = np.unique(coords_m[:, 2])

    ix_all = np.searchsorted(x_axis, coords_m[:, 0]).astype(np.int64)
    iy_all = np.searchsorted(y_axis, coords_m[:, 1]).astype(np.int64)
    iz_all = np.searchsorted(z_axis, coords_m[:, 2]).astype(np.int64)

    expected = x_axis.size * y_axis.size * z_axis.size
    if expected != coords_m.shape[0]:
        raise ValueError(
            "Grid is not full Cartesian: "
            f"nx*ny*nz={expected}, N_spatial={coords_m.shape[0]}"
        )
    return ix_all, iy_all, iz_all


def get_grad_diag_at_time_interp(
    G: np.ndarray,
    ix_all: np.ndarray,
    iy_all: np.ndarray,
    iz_all: np.ndarray,
    spatial_idx_all: int,
    t_ms: float,
) -> Tuple[float, float, float]:
    tmax = G.shape[4] - 1
    f = t_ms / GRAD_DT_MS
    i0 = int(math.floor(f))
    i0 = max(0, min(i0, tmax))
    i1 = min(i0 + 1, tmax)
    w = max(0.0, min(1.0, f - i0))

    ix = int(ix_all[spatial_idx_all])
    iy = int(iy_all[spatial_idx_all])
    iz = int(iz_all[spatial_idx_all])

    gxx0 = float(G[0, ix, iy, iz, i0])
    gyy0 = float(G[1, ix, iy, iz, i0])
    gzz0 = float(G[2, ix, iy, iz, i0])
    gxx1 = float(G[0, ix, iy, iz, i1])
    gyy1 = float(G[1, ix, iy, iz, i1])
    gzz1 = float(G[2, ix, iy, iz, i1])

    gxx = gxx0 + w * (gxx1 - gxx0)
    gyy = gyy0 + w * (gyy1 - gyy0)
    gzz = gzz0 + w * (gzz1 - gzz0)
    return gxx * GRAD_GAIN, gyy * GRAD_GAIN, gzz * GRAD_GAIN


def compute_phi_sections_integrated_grad(
    neuron: AllenNeuronModel,
    cache: Dict,
    topo: Dict,
    G: np.ndarray,
    ix_all: np.ndarray,
    iy_all: np.ndarray,
    iz_all: np.ndarray,
    t_rel_ms: float,
) -> Dict:
    """
    Gradient-driven surrogate:
      g_parallel = ux^2 * Gxx + uy^2 * Gyy + uz^2 * Gzz
      dphi_mV = - g_parallel * ds_um^2 * 1e-9

    Note:
      This is a practical surrogate using only diagonal gradient components.
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
        if data.n < 2:
            phi_sec[sec] = ([0.0], [phi0])
            return

        phis = [phi0]
        for i in range(data.n - 1):
            sp_all = data.mid_spidx_all[i]
            gxx, gyy, gzz = get_grad_diag_at_time_interp(
                G, ix_all, iy_all, iz_all, sp_all, t_rel_ms
            )

            dx, dy, dz = data.dl[i]  # um
            ds = math.sqrt(dx * dx + dy * dy + dz * dz)
            if ds <= 0.0:
                phis.append(phis[-1])
                continue

            ux = dx / ds
            uy = dy / ds
            uz = dz / ds
            g_parallel = (ux * ux) * gxx + (uy * uy) * gyy + (uz * uz) * gzz
            dphi = -g_parallel * (ds * ds) * 1e-9  # mV
            phis.append(phis[-1] + dphi)

        phi_sec[sec] = (data.arc, phis)

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
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        print(f"[FAIL] {name}: NaN/Inf detected")
        return False
    print(f"[OK]   {name}: NaN/Inf not found")
    return True


def parse_gains(scales_text: str) -> List[float]:
    if scales_text is None or str(scales_text).strip() == "":
        return [float(v) for v in DEFAULT_GAINS]
    vals: List[float] = []
    for tok in str(scales_text).split(","):
        tok = tok.strip()
        if tok:
            vals.append(float(tok))
    if not vals:
        vals = [float(v) for v in DEFAULT_GAINS]
    out: List[float] = []
    seen = set()
    for v in vals:
        fv = float(v)
        if fv in seen:
            continue
        seen.add(fv)
        out.append(fv)
    return out


def gain_tag(v: float) -> str:
    fv = float(v)
    if fv <= 0.0:
        s = f"{fv:.6g}"
        return re.sub(r"[^0-9A-Za-z]+", "_", s)
    if abs(fv - 1.0) < 1e-12:
        return "1"
    exp = math.log10(fv)
    exp_i = int(round(exp))
    if abs(exp - exp_i) < 1e-12 and exp_i >= 1:
        return f"10e{exp_i - 1}"
    s = f"{fv:.6g}"
    return re.sub(r"[^0-9A-Za-z]+", "_", s)


# =========================
# Main
# =========================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gains",
        type=str,
        default=DEFAULT_GAINS_STR,
        help=f"Comma-separated gradient gains (default: {DEFAULT_GAINS_STR})",
    )
    args = ap.parse_args()
    gains = parse_gains(args.gains)

    print("\n=== gradient_four.py ===")
    print("Cell ID:", CELL_ID)
    print("Gradient file:", GRAD_VALUES_FILE)
    print("Grid file:    ", E_GRID_COORDS_FILE)
    print("Gradient gains:", ", ".join(f"{g:g}x" for g in gains))

    if not os.path.exists(GRAD_VALUES_FILE):
        raise FileNotFoundError(f"Missing gradient file: {GRAD_VALUES_FILE}")
    if not os.path.exists(E_GRID_COORDS_FILE):
        raise FileNotFoundError(f"Missing coords file: {E_GRID_COORDS_FILE}")

    G = np.load(GRAD_VALUES_FILE, mmap_mode="r")  # (3, nx, ny, nz, nt)
    coords_m = np.load(E_GRID_COORDS_FILE)        # (N_spatial, 3)
    coords_um = coords_m * 1e6

    if G.ndim != 5 or G.shape[0] != 3:
        raise ValueError(f"Unexpected gradient shape: {G.shape}, expected (3, Nx, Ny, Nz, Nt)")
    if coords_m.ndim != 2 or coords_m.shape[1] != 3:
        raise ValueError(f"Unexpected coords shape: {coords_m.shape}, expected (N_spatial, 3)")

    print(f"Loaded gradient shape: {G.shape} (component, nx, ny, nz, t)")
    print(f"Loaded grid shape:     {coords_um.shape} (um)")
    print(f"Gradient dt: {GRAD_DT_MS:.3f} ms, sim dt: {SIM_DT_MS:.3f} ms")

    ix_all, iy_all, iz_all = build_all_to_grid_indices(coords_m)

    inside = coil_inside_mask(coords_um)
    outside = ~inside
    n_inside = int(np.sum(inside))
    n_outside = int(np.sum(outside))
    print(
        f"Coil mask: inside={n_inside} ({100*n_inside/coords_um.shape[0]:.2f}%), "
        f"outside={n_outside} ({100*n_outside/coords_um.shape[0]:.2f}%)"
    )

    coords_out_um = coords_um[outside]
    out_to_all = np.flatnonzero(outside)
    tree_out = cKDTree(coords_out_um)
    print("KDTree built on OUTSIDE points only.")

    print("\n=== Build 4 neurons and place to targets ===")
    neurons: List[AllenNeuronModel] = []
    caches: List[Dict] = []
    topos: List[Dict] = []

    for i, (tx, ty, tz) in enumerate(POSITIONS_UM):
        neuron = AllenNeuronModel(x=0.0, y=0.0, z=0.0, cell_id=CELL_ID)
        sx, sy, sz = xyz_at_seg_linear(neuron.soma, 0.5)
        translate_morphology(neuron.all, tx - sx, ty - sy, tz - sz)
        sx2, sy2, sz2 = xyz_at_seg_linear(neuron.soma, 0.5)
        print(f"Neuron {i+1}: target=({tx:.1f},{ty:.1f},{tz:.1f}) um, soma~({sx2:.1f},{sy2:.1f},{sz2:.1f}) um")
        neurons.append(neuron)

    print("\n=== Build morphology caches (OUTSIDE-only mapping) ===")
    for i, neuron in enumerate(neurons):
        cache, topo = build_morph_cache_outside_only(neuron, tree_out, out_to_all)
        caches.append(cache)
        topos.append(topo)
        print(f"Neuron {i+1}: cache ready")

    print("\n=== Mapping sanity checks (must be 0% coil-inside) ===")
    for i in range(4):
        report_mapping_sanity_for_neuron(caches[i], inside, label=f"N{i+1}")

    print("\n=== Equilibrium run (forcing OFF), SaveState.save() from Neuron 1 ===")
    h.dt = EQ_DT_MS * ms
    h.tstop = EQ_TSTOP_MS * ms
    h.finitialize(EQ_VINIT_MV * mV)

    for neuron in neurons:
        for sec in neuron.all:
            for seg in sec:
                seg.e_extracellular = 0.0

    steps_eq = int(round(EQ_TSTOP_MS / EQ_DT_MS))
    for _ in tqdm(range(steps_eq), desc="Equilibrating", ncols=80):
        h.fadvance()

    eq_vm = [float(neuron.soma(0.5).v) for neuron in neurons]
    print(
        "Equilibrium Vm (soma): "
        f"N1={eq_vm[0]:.3f} mV, N2={eq_vm[1]:.3f} mV, N3={eq_vm[2]:.3f} mV, N4={eq_vm[3]:.3f} mV"
    )

    ss = h.SaveState()
    ss.save()
    print("SaveState saved.\n")

    global GRAD_GAIN

    for gain in gains:
        GRAD_GAIN = float(gain)
        tag = gain_tag(GRAD_GAIN)
        out_path = os.path.join(OUTPUT_DIR, f"gradient_sanity_{tag}x_cell{CELL_ID}.npy")
        if os.path.exists(out_path):
            print(f"[SKIP] Existing output for gain={GRAD_GAIN:g}x: {out_path}")
            continue

        print(f"\n=== Main run: gradient forcing in 0~4ms (gain={GRAD_GAIN:g}x) ===")
        h.dt = SIM_DT_MS * ms
        h.tstop = TSTOP_REL_MS * ms
        ss.restore()
        h.t = 0.0

        npos = len(POSITIONS_UM)
        t_vec = np.zeros(int(round(TSTOP_REL_MS / SIM_DT_MS)) + 1, dtype=np.float64)
        V_in_soma = np.zeros((npos, t_vec.size), dtype=np.float64)
        V_ext_soma = np.zeros((npos, t_vec.size), dtype=np.float64)

        for k in tqdm(range(t_vec.size), desc=f"Simulating ({GRAD_GAIN:g}x)", ncols=80):
            t_rel = k * SIM_DT_MS
            t_vec[k] = t_rel

            if (t_rel >= GRAD_ON_T0_MS) and (t_rel <= GRAD_ON_T1_MS):
                for i, neuron in enumerate(neurons):
                    phi_sec = compute_phi_sections_integrated_grad(
                        neuron=neuron,
                        cache=caches[i],
                        topo=topos[i],
                        G=G,
                        ix_all=ix_all,
                        iy_all=iy_all,
                        iz_all=iz_all,
                        t_rel_ms=t_rel,
                    )
                    apply_phi_to_segments(neuron, phi_sec)
            else:
                for neuron in neurons:
                    for sec in neuron.all:
                        for seg in sec:
                            seg.e_extracellular = 0.0

            for i, neuron in enumerate(neurons):
                V_in_soma[i, k] = float(neuron.soma(0.5).v)
                try:
                    V_ext_soma[i, k] = float(neuron.soma(0.5).vext[0])
                except Exception:
                    V_ext_soma[i, k] = 0.0

            vmax_abs = float(np.max(np.abs(V_in_soma[:, k])))
            if vmax_abs > VM_EXTREME_WARN_MV:
                print(
                    f"\n[WARN] Extreme |V_in| at k={k}, t={t_rel:.6f} ms "
                    f"(max|V_in|={vmax_abs:.3f} mV)."
                )

            if k < t_vec.size - 1:
                h.fadvance()

        payload = {
            "cell_id": CELL_ID,
            "positions_um": np.array(POSITIONS_UM, dtype=np.float64),
            "coil_region_um": COIL_REGION_UM,
            "gradient_values_file": GRAD_VALUES_FILE,
            "efield_grid_file": E_GRID_COORDS_FILE,
            "gradient_dt_ms": GRAD_DT_MS,
            "sim_dt_ms": SIM_DT_MS,
            "gradient_gain": GRAD_GAIN,
            "gradient_on_window_ms": (GRAD_ON_T0_MS, GRAD_ON_T1_MS),
            "gradient_formula": "dphi=-gain*(ux^2*Gxx+uy^2*Gyy+uz^2*Gzz)*ds_um^2*1e-9",
            "t_ms": t_vec,
            "V_in_soma_mV": V_in_soma,
            "V_ext_soma_mV": V_ext_soma,
        }
        np.save(out_path, payload)
        print(f"Saved: {out_path}")

        print("=== Basic integrity checks ===")
        ok = True
        ok &= check_arrays_finite("t_ms", t_vec)
        ok &= check_arrays_finite("V_in_soma_mV", V_in_soma)
        ok &= check_arrays_finite("V_ext_soma_mV", V_ext_soma)
        if V_in_soma.shape[1] != t_vec.size:
            print("[FAIL] V_in time dimension mismatch.")
            ok = False
        else:
            print("[OK]   V_in time dimension matches t length.")
        print("[OK] Integrity checks passed." if ok else "[WARN] Integrity checks failed.")

    print("\nDone.")


if __name__ == "__main__":
    main()
