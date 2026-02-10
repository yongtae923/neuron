# simulate_four.py
"""
Sanity-check simulation for 4 positions using Allen cell + integrated (-E·dl) extracellular potential.

Requirements
- E-field: efield/E_field_1cycle.npy
- Grid:    efield/E_field_grid_coords.npy  (meters -> um)
- Allen model folder under: allen_model/<CELL_ID>_*_ephys/  (with .swc and fit_parameters.json)
- Applies E-field only for 0 ~ 2 ms (relative time)
- Uses "integrated" method: phi along morphology via pt3d path integral
- Computes equilibrium Vm first (E-field OFF), then starts main run from that Vm
- Saves results to: output/*.npy  (np.save dict, allow_pickle)

Run
  python simulate_four.py
"""

import os
import re
import math
import time
import glob
import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree

from neuron import h
from neuron.units import ms, mV

h.load_file("stdrun.hoc")


# =============================================================================
# Config
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CELL_ID = "529898751"

E_FIELD_VALUES_FILE = os.path.join(SCRIPT_DIR, "efield", "E_field_1cycle.npy")
E_GRID_COORDS_FILE = os.path.join(SCRIPT_DIR, "efield", "E_field_grid_coords.npy")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# E-field data sampling
TIME_STEP_US = 50.0
TIME_STEP_MS = TIME_STEP_US / 1000.0  # 0.05 ms

# NEURON sim dt
DT_MS = 0.025  # ms

# Apply window (relative time)
E_ON_START_MS = 0.0
E_ON_END_MS = 2.0

# Main sim stop (relative time)
TSTOP_REL_MS = 2.0

# Equilibrium run (E-field OFF) to get v_init per neuron
EQ_RUN_MS = 200.0

# E-field unit: confirmed V/m
# Convert (V/m * um) -> mV : um->m (1e-6), V->mV (1e3) => 1e-3
E_FACTOR = 1e-3

# Ey handling: if E_field has 2 comps, treat Ey=0
ASSUME_EY_ZERO_IF_MISSING = True

# 4 positions (um)
POSITIONS_UM = [
    (80.0, 0.0, 550.0),
    (80.0, 35.0, 550.0),
    (0.0, 35.0, 550.0),
    (0.0, 0.0, 0.0),
]

# Coil box (um) provided, but not used here because all 4 points are outside
COIL_BOX_UM = {
    "x_min": -79.5, "x_max": 79.5,
    "y_min": -32.0, "y_max": 32.0,
    "z_min": 498.0, "z_max": 1502.0,
}


# =============================================================================
# Utilities: pt3d / topology / interpolation
# =============================================================================
def translate_morphology(all_secs, dx, dy, dz):
    """Translate pt3d coordinates for all sections by (dx,dy,dz) in um."""
    for sec in all_secs:
        n = int(h.n3d(sec=sec))
        for i in range(n):
            x = h.x3d(i, sec=sec) + dx
            y = h.y3d(i, sec=sec) + dy
            z = h.z3d(i, sec=sec) + dz
            d = h.diam3d(i, sec=sec)
            h.pt3dchange(i, x, y, z, d, sec=sec)
    h.define_shape()


def interp_phi(arc_list, phi_list, target_arc):
    """Linear interpolation of phi over arc-length (um)."""
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


def infer_allen_data_dir(cell_id: str) -> str:
    """
    Resolve allen_model/<CELL_ID>_*_ephys folder.
    """
    base = os.path.join(SCRIPT_DIR, "allen_model")
    pattern = os.path.join(base, f"{cell_id}_*_ephys")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"Allen cell folder not found: {pattern}")
    return matches[0]


# =============================================================================
# E-field loading + KDTree
# =============================================================================
def load_efield():
    E = np.load(E_FIELD_VALUES_FILE)  # (C, Nsp, Nt)
    grid_m = np.load(E_GRID_COORDS_FILE)  # (Nsp, 3) meters
    grid_um = grid_m * 1e6

    if E.ndim != 3:
        raise ValueError(f"E_field array must be 3D (C,Nsp,Nt). Got: {E.shape}")

    C = E.shape[0]
    if C not in (2, 3):
        raise ValueError(f"n_components must be 2 or 3. Got: {C}")

    if C == 2 and not ASSUME_EY_ZERO_IF_MISSING:
        raise ValueError("E-field has 2 components but Ey is not allowed to be assumed 0.")

    tree = cKDTree(grid_um)
    return E, grid_um, tree


def get_E_at(E_field_values, spatial_idx: int, t_ms: float):
    """
    Return (Ex,Ey,Ez) at grid index with linear time interpolation.
    E_field_values: (C, Nsp, Nt), V/m
    """
    time_index_float = t_ms / TIME_STEP_MS
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

    C = E_field_values.shape[0]

    Ex_prev = E_field_values[0, spatial_idx, t_idx_prev]
    Ex_next = E_field_values[0, spatial_idx, t_idx_next]
    Ex = Ex_prev + ratio * (Ex_next - Ex_prev)

    if C == 3:
        Ey_prev = E_field_values[1, spatial_idx, t_idx_prev]
        Ey_next = E_field_values[1, spatial_idx, t_idx_next]
        Ey = Ey_prev + ratio * (Ey_next - Ey_prev)

        Ez_prev = E_field_values[2, spatial_idx, t_idx_prev]
        Ez_next = E_field_values[2, spatial_idx, t_idx_next]
        Ez = Ez_prev + ratio * (Ez_next - Ez_prev)
    else:
        Ey = 0.0
        Ez_prev = E_field_values[1, spatial_idx, t_idx_prev]
        Ez_next = E_field_values[1, spatial_idx, t_idx_next]
        Ez = Ez_prev + ratio * (Ez_next - Ez_prev)

    return Ex, Ey, Ez


# =============================================================================
# Integrated method: cache + phi compute + apply
# =============================================================================
def build_morph_cache(neuron, tree: cKDTree):
    """
    Build per-section cache:
      - arc: pt3d arc positions (um)
      - dl:  pt3d segment vectors (dx,dy,dz) between pt3d points (um)
      - mid_spidx: nearest grid index for each pt3d segment midpoint
    Also build topology mapping for soma-rooted phi.
    """
    cache = {}
    topo = {}

    # Section list for stable traversal
    all_secs = list(neuron.all)

    # Build cache for each section
    for sec in all_secs:
        n = int(h.n3d(sec=sec))
        if n < 2:
            cache[sec] = {"n": n, "arc": [0.0], "dl": [], "mid_spidx": []}
            continue

        xs = np.array([h.x3d(i, sec=sec) for i in range(n)], dtype=float)
        ys = np.array([h.y3d(i, sec=sec) for i in range(n)], dtype=float)
        zs = np.array([h.z3d(i, sec=sec) for i in range(n)], dtype=float)
        arc = [float(h.arc3d(i, sec=sec)) for i in range(n)]  # um

        # dl per pt3d segment
        dx = xs[1:] - xs[:-1]
        dy = ys[1:] - ys[:-1]
        dz = zs[1:] - zs[:-1]
        dl = list(zip(dx.tolist(), dy.tolist(), dz.tolist()))

        # midpoint coords for KDTree query (vectorized)
        mx = 0.5 * (xs[1:] + xs[:-1])
        my = 0.5 * (ys[1:] + ys[:-1])
        mz = 0.5 * (zs[1:] + zs[:-1])
        mids = np.column_stack([mx, my, mz])

        # nearest grid indices
        _, idx = tree.query(mids, k=1)
        mid_spidx = idx.tolist()

        cache[sec] = {"n": n, "arc": arc, "dl": dl, "mid_spidx": mid_spidx}

    # Topology: soma-rooted
    soma_sec = neuron.soma

    for sec in all_secs:
        if sec == soma_sec:
            topo[sec] = None
            continue

        sref = h.SectionRef(sec=sec)
        if sref.has_parent():
            try:
                pseg = sref.parentseg()
                topo[sec] = (pseg.sec, float(pseg.x))
            except Exception:
                # Fallback: connect to soma(0.5) if parentseg fails
                topo[sec] = (soma_sec, 0.5)
        else:
            # Force orphan sections to be under soma
            topo[sec] = (soma_sec, 0.5)

    return cache, topo


def compute_phi_sections(neuron, morph_cache, topo, E_field_values, t_rel_ms: float):
    """
    Compute phi along each section by integrating -E·dl from soma-rooted tree.
    Returns dict: sec -> (arc_list, phi_list)
    """
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
            parc, pphis = phi_sec[psec]
            total_arc = parc[-1] if parc else 0.0
            phi0 = interp_phi(parc, pphis, px * total_arc)

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
            Ex, Ey, Ez = get_E_at(E_field_values, spidx, t_rel_ms)
            dx, dy, dz = dl[i]
            dphi = -(Ex * dx + Ey * dy + Ez * dz) * E_FACTOR  # mV
            phis.append(phis[-1] + dphi)

        phi_sec[sec] = (arc, phis)

    for sec in neuron.all:
        ensure_section_phi(sec)

    return phi_sec


def apply_phi_to_segments(neuron, phi_sec):
    """Apply computed phi to each segment's e_extracellular (mV)."""
    for sec in neuron.all:
        if sec not in phi_sec:
            continue
        arc, phis = phi_sec[sec]
        total_arc = arc[-1] if arc else 0.0
        for seg in sec:
            target_arc = float(seg.x) * total_arc
            seg.e_extracellular = float(interp_phi(arc, phis, target_arc))


def set_extracellular_field_integrated(
    neurons, caches, topos, E_field_values, t_rel_ms: float
):
    """
    Apply E-field via integrated phi, only within [E_ON_START_MS, E_ON_END_MS].
    Outside window: set e_extracellular = 0.
    """
    if not (E_ON_START_MS <= t_rel_ms <= E_ON_END_MS):
        for neuron in neurons:
            for sec in neuron.all:
                for seg in sec:
                    seg.e_extracellular = 0.0
        return

    for i, neuron in enumerate(neurons):
        phi_sec = compute_phi_sections(neuron, caches[i], topos[i], E_field_values, t_rel_ms)
        apply_phi_to_segments(neuron, phi_sec)


# =============================================================================
# Equilibrium helper
# =============================================================================
def run_to_equilibrium(neuron, eq_ms: float) -> float:
    """
    Run E-field OFF and return soma equilibrium Vm (mV).
    """
    h.finitialize(-65.0 * mV)
    h.t = 0.0
    steps = int(eq_ms / DT_MS)
    for _ in range(steps):
        h.fadvance()
    return float(neuron.soma(0.5).v)


def init_state_from_v(neurons, v_inits_mV):
    """
    Initialize NEURON state and force each neuron's segments to its v_init.
    This makes the starting Vm identical to previously measured equilibrium per neuron.
    """
    # Use average just to initialize gating variables consistently
    v_mean = float(np.mean(v_inits_mV)) if len(v_inits_mV) else -65.0
    h.finitialize(v_mean * mV)
    h.t = 0.0

    for neuron, v0 in zip(neurons, v_inits_mV):
        for sec in neuron.all:
            for seg in sec:
                seg.v = float(v0)


# =============================================================================
# Main
# =============================================================================
def main():
    print("\n=== Sanity check: 4 positions, integrated phi, 0~2 ms ===")
    print(f"Cell ID: {CELL_ID}")
    print(f"E-field: {E_FIELD_VALUES_FILE}")
    print(f"Grid:    {E_GRID_COORDS_FILE}")
    print(f"dt: {DT_MS} ms, E-field dt: {TIME_STEP_MS} ms")
    print(f"E-field window: {E_ON_START_MS} ~ {E_ON_END_MS} ms")
    print(f"tstop (relative): {TSTOP_REL_MS} ms")
    print(f"Equilibrium run (OFF): {EQ_RUN_MS} ms")

    # Simulation constants
    h.dt = DT_MS * ms
    h.celsius = 34.0

    # Load E-field and KDTree
    E_field_values, grid_um, tree = load_efield()
    C, Nsp, Nt = E_field_values.shape
    print(f"\nLoaded E-field: shape={E_field_values.shape} (components={C}, spatial={Nsp}, time={Nt})")
    print(f"Loaded grid:    shape={grid_um.shape} (um)")
    print("KDTree built.")

    # Load Allen model
    allen_dir = infer_allen_data_dir(CELL_ID)
    print(f"\nAllen data dir resolved: {allen_dir}")

    # Import AllenNeuronModel (model_allen_neuron.py in this project)
    from model_allen_neuron import AllenNeuronModel

    # Create 4 neurons, place by translating morphology so soma(0.5) matches target
    neurons = []
    for idx, (tx, ty, tz) in enumerate(POSITIONS_UM):
        print(f"\n[Neuron {idx+1}] building at target soma = ({tx}, {ty}, {tz}) um")

        # Create at origin
        neuron = AllenNeuronModel(x=0.0, y=0.0, z=0.0, cell_id=CELL_ID, data_dir=allen_dir)

        # Ensure extracellular mechanism exists everywhere
        for sec in neuron.all:
            try:
                sec.insert("extracellular")
            except Exception:
                # If already inserted or mechanism missing, proceed
                pass

        # Current soma center (approx using seg location)
        sx = float(neuron.soma(0.5).x3d(0)) if hasattr(neuron.soma(0.5), "x3d") else None

        # Robust soma pt3d-based position: use first and last pt3d of soma section mid
        # Use pt3d midpoint along arc (fallback: seg x=0.5 arc fraction)
        n = int(h.n3d(sec=neuron.soma))
        if n >= 2:
            soma_xs = np.array([h.x3d(i, sec=neuron.soma) for i in range(n)], dtype=float)
            soma_ys = np.array([h.y3d(i, sec=neuron.soma) for i in range(n)], dtype=float)
            soma_zs = np.array([h.z3d(i, sec=neuron.soma) for i in range(n)], dtype=float)
            sx0, sy0, sz0 = float(np.mean(soma_xs)), float(np.mean(soma_ys)), float(np.mean(soma_zs))
        else:
            # fallback: assume origin
            sx0, sy0, sz0 = 0.0, 0.0, 0.0

        dx, dy, dz = tx - sx0, ty - sy0, tz - sz0
        translate_morphology(neuron.all, dx, dy, dz)

        # Verify soma location after translate
        n2 = int(h.n3d(sec=neuron.soma))
        if n2 >= 2:
            soma_xs2 = np.array([h.x3d(i, sec=neuron.soma) for i in range(n2)], dtype=float)
            soma_ys2 = np.array([h.y3d(i, sec=neuron.soma) for i in range(n2)], dtype=float)
            soma_zs2 = np.array([h.z3d(i, sec=neuron.soma) for i in range(n2)], dtype=float)
            sx1, sy1, sz1 = float(np.mean(soma_xs2)), float(np.mean(soma_ys2)), float(np.mean(soma_zs2))
            print(f"  translated soma mean xyz = ({sx1:.2f}, {sy1:.2f}, {sz1:.2f}) um")
        else:
            print("  soma has <2 pt3d points; translation verification skipped.")

        neurons.append(neuron)

    # Compute equilibrium Vm per neuron (E-field OFF)
    print("\n=== Equilibrium Vm (E-field OFF) ===")
    v_inits = []
    for i, neuron in enumerate(neurons):
        v_eq = run_to_equilibrium(neuron, EQ_RUN_MS)
        v_inits.append(v_eq)
        print(f"  Neuron {i+1}: v_eq = {v_eq:.3f} mV")

    # Build integrated caches once (KDTree vectorized)
    print("\n=== Build integrated caches (KDTree vectorized) ===")
    caches = []
    topos = []
    t0 = time.time()
    for i, neuron in enumerate(neurons):
        cache, topo = build_morph_cache(neuron, tree)
        caches.append(cache)
        topos.append(topo)
        print(f"  Neuron {i+1}: cache built.")
    print(f"Cache build done in {time.time() - t0:.2f} s")

    # Initialize main simulation from equilibrium Vm
    init_state_from_v(neurons, v_inits)

    # Main run: 0 ~ TSTOP_REL_MS
    steps = int(TSTOP_REL_MS / DT_MS) + 1
    t_rel = np.zeros(steps, dtype=float)
    Vm_soma = np.zeros((len(neurons), steps), dtype=float)
    Vext_soma = np.zeros((len(neurons), steps), dtype=float)

    print("\n=== Main run (integrated phi) ===")
    pbar = tqdm(total=steps, desc="Sim", unit="step", ncols=90)

    for k in range(steps):
        tms = k * DT_MS
        t_rel[k] = tms

        # Apply field first (only within window)
        set_extracellular_field_integrated(neurons, caches, topos, E_field_values, tms)

        # Record after apply
        for i, neuron in enumerate(neurons):
            Vm_soma[i, k] = float(neuron.soma(0.5).v)
            try:
                if hasattr(neuron.soma(0.5), "vext"):
                    Vext_soma[i, k] = float(neuron.soma(0.5).vext[0])
                else:
                    Vext_soma[i, k] = 0.0
            except Exception:
                Vext_soma[i, k] = 0.0

        # Advance (skip after last sample)
        if k < steps - 1:
            h.fadvance()

        pbar.update(1)

    pbar.close()

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    fname = f"sanity_cell{CELL_ID}_4pos_integrated_efield0to2ms_dt{DT_MS:.3f}ms_{timestamp}.npy"
    out_path = os.path.join(OUTPUT_DIR, fname)

    results = {
        "meta": {
            "cell_id": CELL_ID,
            "positions_um": POSITIONS_UM,
            "dt_ms": DT_MS,
            "efield_dt_ms": TIME_STEP_MS,
            "tstop_rel_ms": TSTOP_REL_MS,
            "efield_window_ms": [E_ON_START_MS, E_ON_END_MS],
            "eq_run_ms": EQ_RUN_MS,
            "E_FACTOR_mV_per_(V/m*um)": E_FACTOR,
            "efield_components": int(E_field_values.shape[0]),
            "efield_shape": tuple(E_field_values.shape),
            "grid_shape": tuple(grid_um.shape),
            "coil_box_um": COIL_BOX_UM,
            "note": "Integrated phi via -E·dl along pt3d; soma-rooted topology. Coil exclusion not applied because all positions are outside.",
        },
        "t_ms": t_rel,
        "Vm_soma_mV": Vm_soma,
        "Vext_soma_mV": Vext_soma,
        "v_init_eq_mV": np.array(v_inits, dtype=float),
    }

    np.save(out_path, results, allow_pickle=True)
    print(f"\nSaved: {out_path}")

    # Quick sanity print
    print("\n=== Quick check (peak-to-peak ΔVm, soma) ===")
    for i in range(len(neurons)):
        x = Vm_soma[i]
        d = x - np.mean(x)
        p2p = float(np.max(d) - np.min(d))
        print(f"  Neuron {i+1} @ {POSITIONS_UM[i]}: p2p ΔVm = {p2p:.6f} mV ({p2p*1000:.3f} uV)")

    print("\nDone.")


if __name__ == "__main__":
    main()
