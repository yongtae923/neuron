"""
simulate_one.py

Run a small set of single-position Allen neuron simulations (with E-field),
and plot Vm(t) and |E|(t) where t=0 is the E-field onset.

Target position(s) (um): x in {-90, 0, 90}, y=42, z=561 (hardcoded)
Plot window: -0.5 ~ 5 ms (t=0 is E-field onset; pre-window has no E-field)
"""

from __future__ import annotations

import warnings

# Suppress noisy numpy longdouble warning on some WSL/Windows builds.
# Must run before importing numpy.
warnings.filterwarnings(
    "ignore",
    message=r".*Signature.*numpy\.longdouble.*",
    category=UserWarning,
)
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import simulate_allen as sim


@dataclass(frozen=True)
class OneSimConfig:
    xs_um: tuple[float, ...] = (-90.0, 0.0, 90.0)
    y_um: float = 42.0
    z_um: float = 561.0
    pre_ms: float = 0.5
    stim_ms: float = 5.0
    out_plot_dirname: str = "plot"


def _simulate_vm_with_pre_window(pos_um: np.ndarray, record_times_ms: np.ndarray) -> np.ndarray:
    """
    Simulate and record transmembrane Vm (= v - e_extracellular) at times relative to E-field onset.
    record_times_ms includes negative values for the pre-window (no E-field).
    
    Includes a stabilization period to ensure the neuron is at resting potential
    before the pre-window recording starts.
    """
    # Create neuron (suppress output)
    import io
    import contextlib

    with contextlib.redirect_stdout(io.StringIO()):
        neuron_model = sim.AllenNeuronModel(x=0, y=0, z=0, cell_id=sim._config["CELL_ID"])

    # Move to target position
    sx, sy, sz = sim._xyz_at_seg(neuron_model.soma, 0.5)
    tx, ty, tz = float(pos_um[0]), float(pos_um[1]), float(pos_um[2])
    sim._translate_morphology(neuron_model.all, tx - sx, ty - sy, tz - sz)

    # Build morph cache (for E-field application during stimulation)
    morph_cache, topo = sim._build_morph_cache(neuron_model)

    # --- Simulation Setup & Warmup (same as simulate_allen / old_xz v4) ---
    
    WARMUP_TIME_MS = float(sim._config["WARMUP_TIME_MS"])  # e.g. 500 ms, no E-field
    V_INIT = -75.0  # Initial voltage guess (closer to resting than -65)

    sim.h.dt = sim._config["DT_MS"]
    sim.h.celsius = 34.0
    
    sim.h.finitialize(V_INIT * sim.mV)
    
    # Time alignment (like simulate_tES_v4):
    # 0 ~ WARMUP_TIME_MS : Warmup (no E-field, no recording)
    # WARMUP_TIME_MS ~ WARMUP_TIME_MS + pre_ms : Pre-window (no E-field, record)
    # WARMUP_TIME_MS + pre_ms : E-FIELD ONSET (t_rel = 0)
    
    pre_ms_duration = abs(record_times_ms[0]) if len(record_times_ms) > 0 and record_times_ms[0] < 0 else 0.0
    onset_abs_time = WARMUP_TIME_MS + pre_ms_duration
    
    max_record_time = record_times_ms[-1]
    sim.h.tstop = onset_abs_time + max_record_time + sim.h.dt * 2

    sim.h.t = 0.0
    
    # 1) Warmup phase: run without E-field until WARMUP_TIME_MS (no recording)
    for sec in neuron_model.all:
        for seg in sec:
            seg.e_extracellular = 0.0
    while sim.h.t < WARMUP_TIME_MS - 1e-9:
        sim.h.fadvance()

    # 2) Main loop: pre-window + E-field, with recording
    Vm_list: list[float] = []
    rec_idx = 0
    while sim.h.t < sim.h.tstop:
        t_rel = sim.h.t - onset_abs_time

        if t_rel >= 0:
            phi_sec = sim._compute_phi_sections(neuron_model, morph_cache, topo, float(t_rel))
            sim._apply_phi_to_segments(neuron_model, phi_sec)
        else:
            for sec in neuron_model.all:
                for seg in sec:
                    seg.e_extracellular = 0.0

        if rec_idx < len(record_times_ms):
            target_t = float(record_times_ms[rec_idx])
            if abs(t_rel - target_t) < sim._config["DT_MS"] / 2:
                soma = neuron_model.soma(0.5)
                vm_val = float(soma.v - getattr(soma, "e_extracellular", 0.0))
                Vm_list.append(vm_val)
                rec_idx += 1

        sim.h.fadvance()

    # Pad if needed (rare case if floating point skip happens)
    while len(Vm_list) < len(record_times_ms):
        Vm_list.append(Vm_list[-1] if Vm_list else V_INIT)

    return np.array(Vm_list, dtype=np.float32)


def main() -> None:
    cfg = OneSimConfig()
    script_dir = Path(__file__).resolve().parent

    xs = list(cfg.xs_um)
    y_um = float(cfg.y_um)
    z_um = float(cfg.z_um)
    positions = [np.array([x, y_um, z_um], dtype=float) for x in xs]

    # --- Load E-field data (same files as simulate_allen.py) ---
    E_field_values = np.load(sim.E_FIELD_VALUES_FILE)
    coords_m = np.load(sim.E_GRID_COORDS_FILE)
    E_grid_coords_UM = coords_m * 1e6

    # --- Compute time indices for requested ROI (clip to data length) ---
    n_time_total = E_field_values.shape[2]
    total_data_ms = (n_time_total - 1) * sim.TIME_STEP_MS

    # We'll plot from -pre_ms to +stim_ms
    stim_ms = float(min(cfg.stim_ms, total_data_ms))
    if stim_ms <= 0:
        raise SystemExit(
            f"E-field data is too short: available 0 ~ {total_data_ms:.3f} ms."
        )

    # Record at simulation dt for both pre and stim window.
    dt = float(sim.DT_MS)
    record_times_ms = np.arange(-float(cfg.pre_ms), stim_ms + dt / 2, dt, dtype=float)
    time_ms_arr = record_times_ms.copy()  # x-axis: time relative to E-field onset

    # Simulation globals initialization
    # Note: TOTAL_TIME_MS here is just for E-field config checks, 
    # the actual runtime is controlled inside _simulate_vm_with_pre_window
    worker_config = {
        "CELL_ID": sim.CELL_ID,
        "WARMUP_TIME_MS": sim.WARMUP_TIME_MS,
        "TOTAL_TIME_MS": stim_ms, 
        "TIME_START_MS": 0.0,
        "TIME_STEP_MS": sim.TIME_STEP_MS,
        "DT_MS": sim.DT_MS,
        "E_FIELD_SCALE": sim.E_FIELD_SCALE,
        "E_FACTOR": sim.E_FACTOR,
    }

    sim._worker_init(E_field_values, E_grid_coords_UM, None, worker_config)

    # --- Run simulation(s) ---
    Vm_traces: list[np.ndarray] = []
    E_mags: list[np.ndarray] = []
    dists_um: list[float] = []

    print(f"Running simulations for positions: {xs} ...")
    
    for i, pos_um in enumerate(positions):
        print(f"  Simulating x={pos_um[0]}...")
        Vm_arr = _simulate_vm_with_pre_window(pos_um, record_times_ms)
        Vm_traces.append(Vm_arr)

        dist_um, nearest_idx = sim._E_grid_tree.query(pos_um)
        nearest_idx = int(nearest_idx)
        
        # Construct E-field trace for plotting
        # Pre-window (<0) has 0 field.
        E_trace = []
        for t in record_times_ms:
            if float(t) < 0:
                E_trace.append((0.0, 0.0, 0.0))
            else:
                E_trace.append(sim._get_E_at(nearest_idx, float(t)))
        
        E_xyz = np.array(E_trace, dtype=float)
        E_mags.append(np.linalg.norm(E_xyz, axis=1))
        dists_um.append(float(dist_um))

    # --- Plot ---
    plot_dir = script_dir / cfg.out_plot_dirname
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax, ax_zoom) = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    
    # Vm: NEURON uses mV; plot in V
    vm_lines = []
    for x, Vm_arr in zip(xs, Vm_traces):
        Vm_V = np.asarray(Vm_arr, dtype=float) * 1e-3
        (ln,) = ax.plot(time_ms_arr, Vm_V, linewidth=1.5, label=f"Vm x={x:g} um (V)")
        ax_zoom.plot(time_ms_arr, Vm_V, linewidth=1.2)
        vm_lines.append(ln)
        
    ax.set_ylabel("Membrane potential Vm (V)", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax_zoom.set_xlabel("Time (ms)")
    ax_zoom.set_ylabel("Vm (V) [zoom]")

    lines = list(vm_lines)

    ax2 = ax.twinx()
    e_lines = []
    for x, E_mag, dist in zip(xs, E_mags, dists_um):
        (ln,) = ax2.plot(
            time_ms_arr,
            np.asarray(E_mag, dtype=float),
            linewidth=1.1,
            linestyle="--",
            alpha=0.75,
            label=f"|E| x={x:g} um (dist {dist:.1f}um)",
        )
        e_lines.append(ln)
    ax2.set_ylabel("E-field magnitude |E| (V/m)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    lines = lines + e_lines

    # Mark E-field onset
    ax.axvline(0.0, color="k", linewidth=1.0, alpha=0.6)
    ax_zoom.axvline(0.0, color="k", linewidth=1.0, alpha=0.6)

    ax.set_title(
        "Vm and |E| at target position\n"
        f"y={y_um:g} um, z={z_um:g} um"
    )
    ax_zoom.set_title("Zoom around onset (Stabilized Baseline)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax_zoom.grid(True, linestyle="--", alpha=0.5)

    # Zoom limits: focus on pre-window and resting-scale voltages
    pre_mask = time_ms_arr < 0
    if np.any(pre_mask):
        pre_vals = np.concatenate([np.asarray(v, dtype=float)[pre_mask] for v in Vm_traces]) * 1e-3
        pre_min = float(np.nanmin(pre_vals))
        pre_max = float(np.nanmax(pre_vals))
        # Add a small padding, but ensure we see the baseline clearly
        pad = max(0.005, 0.5 * (pre_max - pre_min)) 
        ax_zoom.set_ylim(pre_min - pad, pre_max + pad)
        
    ax_zoom.set_xlim(-float(cfg.pre_ms), stim_ms)

    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="best", framealpha=0.9)

    xs_tag = "_".join([f"{x:g}" for x in xs])
    out_name = (
        f"Vm_E_trace_xs{xs_tag}_y{y_um:g}_z{z_um:g}_"
        f"-{cfg.pre_ms:.2f}-{stim_ms:.2f}ms.png"
    )
    out_path = plot_dir / out_name
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()