# plot_and_check_four.py
# ------------------------------------------------------------
# Plot + sanity checks for simulate_four_A_coil_excluded.py outputs.
#
# Shows (naming clarified):
#   - V_in  (soma.v, intracellular)          : V_in_soma_mV
#   - V_ext (soma.vext[0], extracellular)    : V_ext_soma_mV
#   - V_m   = V_in - V_ext (membrane)        : V_m_soma_mV
#   - ΔV_m  (demeaned V_m)                   : dV_m_demeaned_mV
#
# Options:
#   [scale]    : optional positional scale selector (1, 5, 10, 20)
#   --latest   : keep behavior of loading newest matched file
#   --file     : load a specific .npy
#   --no-plot  : checks only
#
# Usage:
#   python plot_and_check_four.py
#   python plot_and_check_four.py 5
#   python plot_and_check_four.py 10
#   python plot_and_check_four.py 20
#   python plot_and_check_four.py --latest
#   python plot_and_check_four.py --file output/<file>.npy
#   python plot_and_check_four.py --latest --no-plot
# ------------------------------------------------------------

from __future__ import annotations

import os
import argparse
import re
from typing import Tuple, Optional, Dict, Any, List

import numpy as np

DEFAULT_EFIELD_ON_WINDOW_MS = (0.0, 2.0)
VM_EXTREME_WARN_MV = 200.0


def find_latest_npy(output_dir: str) -> Optional[str]:
    if not os.path.isdir(output_dir):
        return None
    cand = []
    for fn in os.listdir(output_dir):
        if fn.lower().endswith(".npy"):
            fp = os.path.join(output_dir, fn)
            if os.path.isfile(fp):
                cand.append(fp)
    if not cand:
        return None
    cand.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cand[0]


def _scale_to_tag(scale: float) -> str:
    return str(float(scale)).rstrip("0").rstrip(".").replace(".", "p")


def _scale_alias_tags(scale: float) -> List[str]:
    """Return compatible filename tags for a numeric scale."""
    s = float(scale)
    tags = {_scale_to_tag(s), str(s).replace(".", "p")}
    # Also accept integer-like form if scale is whole number (e.g. 5 -> "5")
    if abs(s - round(s)) < 1e-12:
        tags.add(str(int(round(s))))
    return sorted(tags)


def find_latest_sanity_by_scale(output_dir: str, scale: float) -> Optional[str]:
    """
    Find the newest sanity file for the requested scale.
    - scale == 1.0: prefer legacy files without '_scale' in name.
    - scale != 1.0: match files containing '_scale{tag}x_'.
    """
    if not os.path.isdir(output_dir):
        return None

    files: List[str] = []
    for fn in os.listdir(output_dir):
        if not fn.lower().endswith(".npy"):
            continue
        # Support both legacy and new naming:
        # - legacy: sanity_cell<id>_..._scale5p0x_....npy or without scale token
        # - new:    sanity_100x_cell<id>.npy
        if not (fn.startswith("sanity_cell") or fn.startswith("sanity_")):
            continue
        files.append(fn)

    if not files:
        return None

    scale = float(scale)
    if abs(scale - 1.0) < 1e-12:
        matched = [
            fn for fn in files
            if (
                ("_scale" not in fn and not fn.startswith("sanity_"))
                or fn.startswith("sanity_1x_cell")
                or "_scale1x_" in fn
                or "_scale1p0x_" in fn
            )
        ]
        if not matched:
            # fallback for explicitly saved scale1x files
            matched = [fn for fn in files if "_scale1x_" in fn or "_scale1p0x_" in fn]
    else:
        tags = _scale_alias_tags(scale)
        matched = []
        for fn in files:
            for tag in tags:
                # legacy format token
                if f"_scale{tag}x_" in fn:
                    matched.append(fn)
                    break
                # new format: sanity_{tag}x_cell...
                if f"sanity_{tag}x_cell" in fn:
                    matched.append(fn)
                    break

    if not matched:
        return None

    paths = [os.path.join(output_dir, fn) for fn in matched]
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths[0]


def find_all_simulate_four_outputs(output_dir: str) -> List[str]:
    """
    Collect all sanity outputs from simulate_four_v2.py and sort by scale.
    Supports:
    - new:    sanity_<scale>x_cell<id>.npy
    - legacy: sanity_cell<id>_....npy
    """
    if not os.path.isdir(output_dir):
        return []

    def _scale_from_name(fn: str) -> float:
        # New style: sanity_100x_cell...
        m = re.match(r"^sanity_([0-9eE\+\-\.p]+)x_cell", fn)
        if m:
            tag = m.group(1).replace("p", ".")
            try:
                return float(tag)
            except Exception:
                return float("inf")
        # Legacy 1x or unknown; keep near front
        if fn.startswith("sanity_cell"):
            return 1.0
        return float("inf")

    paths: List[str] = []
    for fn in os.listdir(output_dir):
        if not fn.lower().endswith(".npy"):
            continue
        if fn.startswith("sanity_") or fn.startswith("sanity_cell"):
            paths.append(os.path.join(output_dir, fn))

    paths.sort(key=lambda p: (_scale_from_name(os.path.basename(p)), os.path.getmtime(p)))
    return paths


def load_payload(path: str) -> Dict[str, Any]:
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.shape == ():
        payload = obj.item()
    elif isinstance(obj, dict):
        payload = obj
    else:
        raise ValueError(f"Unsupported .npy structure: type={type(obj)}, shape={getattr(obj,'shape',None)}")
    if not isinstance(payload, dict):
        raise ValueError("Loaded payload is not a dict. Did you save a dict with np.save?")
    return payload


def check_arrays_finite(name: str, arr: np.ndarray) -> bool:
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        print(f"[FAIL] {name}: NaN/Inf detected")
        return False
    print(f"[OK]   {name}: NaN/Inf not found")
    return True


def compute_derived(
    t_ms: np.ndarray,
    V_in: np.ndarray,
    V_ext: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Derived signals from V_in (soma.v) and V_ext (soma.vext[0]):
      - V_m        : V_in - V_ext
      - dV_m       : V_m - mean(V_m)
    """
    V_in = V_in.astype(np.float64, copy=False)
    V_ext = V_ext.astype(np.float64, copy=False)

    V_m = V_in - V_ext
    dV_m = V_m - np.mean(V_m, axis=1, keepdims=True)

    return {
        "V_in_soma_mV": V_in,
        "V_ext_soma_mV": V_ext,
        "V_m_soma_mV": V_m,
        "dV_m_demeaned_mV": dV_m,
    }


def gating_magnitude_report(
    t_ms: np.ndarray,
    Vm: np.ndarray,
    Vext: np.ndarray,
    positions_um: np.ndarray,
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

        pos = positions_um[i] if (positions_um.ndim == 2 and positions_um.shape[1] == 3) else (np.nan, np.nan, np.nan)
        print(f"\n[Neuron {i+1}] position_um=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
        print(f"  ΔVm(demeaned): p2p_in={p2p_in:.6f} mV ({p2p_in*1000:.3f} uV), rms_in={rms_in:.6f} mV ({rms_in*1000:.3f} uV)")
        print(f"               p2p_out={p2p_out:.6f} mV ({p2p_out*1000:.3f} uV), rms_out={rms_out:.6f} mV ({rms_out*1000:.3f} uV)")

        if np.isfinite(p2p_in) and np.isfinite(p2p_out) and (p2p_in > p2p_out * 1.2):
            print("  [OK]   ΔVm modulation is larger inside the on-window.")
        else:
            print("  [WARN] ΔVm modulation is not clearly larger inside the on-window.")

        print(f"  Vext: max|in|={max_vext_in:.6f} mV, max|out|={max_vext_out:.6f} mV")

        if vmax_abs > VM_EXTREME_WARN_MV:
            print(f"  [WARN] |Vm| exceeds {VM_EXTREME_WARN_MV:.1f} mV (max|Vm|={vmax_abs:.3f}). 수치 폭주 가능성이 큽니다.")


def quick_absolute_summary(
    derived: Dict[str, np.ndarray],
    positions_um: np.ndarray,
) -> None:
    V_in = derived["V_in_soma_mV"]
    V_ext = derived["V_ext_soma_mV"]
    V_m = derived["V_m_soma_mV"]
    dV_m = derived["dV_m_demeaned_mV"]

    print("\n=== Quick absolute range summary ===")
    for i, pos in enumerate(positions_um):
        vin = V_in[i]
        vext = V_ext[i]
        vm = V_m[i]
        dv = dV_m[i]

        print(f"Neuron {i+1} @ ({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})")
        print(f"  V_in    : min={np.min(vin):.6f}, max={np.max(vin):.6f}, max|.|={np.max(np.abs(vin)):.6f} mV")
        print(f"  V_ext   : min={np.min(vext):.6f}, max={np.max(vext):.6f}, max|.|={np.max(np.abs(vext)):.6f} mV")
        print(f"  V_m     : min={np.min(vm):.6f}, max={np.max(vm):.6f}, max|.|={np.max(np.abs(vm)):.6f} mV")
        print(f"  ΔV_m    : min={np.min(dv):.6f}, max={np.max(dv):.6f}, p2p={np.max(dv)-np.min(dv):.6f} mV")


def print_start_values(
    t_ms: np.ndarray,
    derived: Dict[str, np.ndarray],
    positions_um: np.ndarray,
) -> None:
    print("\n=== Start values (k=0) ===")
    k0 = 0
    print(f"t0 = {t_ms[k0]:.6f} ms")
    for i, pos in enumerate(positions_um):
        vin0 = derived["V_in_soma_mV"][i, k0]
        vext0 = derived["V_ext_soma_mV"][i, k0]
        vm0 = derived["V_m_soma_mV"][i, k0]
        dv0 = derived["dV_m_demeaned_mV"][i, k0]
        print(
            f"Neuron {i+1} @ ({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}) : "
            f"V_in0={vin0:.6f} mV, V_ext0={vext0:.6f} mV, V_m0={vm0:.6f} mV, ΔV_m0={dv0:.6f} mV"
        )


def run_checks(payload: Dict[str, Any]) -> Tuple[bool, np.ndarray, Dict[str, np.ndarray], np.ndarray, Tuple[float, float]]:
    t_ms = np.asarray(payload["t_ms"], dtype=np.float64)

    # Prefer new names; fall back to old ones for backward compatibility.
    if "V_in_soma_mV" in payload and "V_ext_soma_mV" in payload:
        V_in = np.asarray(payload["V_in_soma_mV"], dtype=np.float64)
        V_ext = np.asarray(payload["V_ext_soma_mV"], dtype=np.float64)
    else:
        # Old schema (Vm_soma_mV = soma.v, Vext_soma_mV = soma.vext[0])
        print("[WARN] Using legacy keys 'Vm_soma_mV'/'Vext_soma_mV' as V_in/V_ext.")
        V_in = np.asarray(payload["Vm_soma_mV"], dtype=np.float64)
        V_ext = np.asarray(payload["Vext_soma_mV"], dtype=np.float64)
    positions_um = np.asarray(payload.get("positions_um", np.zeros((V_in.shape[0], 3))), dtype=np.float64)

    if "efield_on_window_ms" in payload:
        on0, on1 = payload["efield_on_window_ms"]
        on0, on1 = float(on0), float(on1)
    else:
        print(f"[WARN] 'efield_on_window_ms' not found. Using hard-coded DEFAULT_EFIELD_ON_WINDOW_MS={DEFAULT_EFIELD_ON_WINDOW_MS}.")
        on0, on1 = DEFAULT_EFIELD_ON_WINDOW_MS

    cell_id = str(payload.get("cell_id", "UNKNOWN"))
    print("Cell ID:", cell_id)
    print(f"V_in shape: {V_in.shape}, t shape: {t_ms.shape}")
    print(f"V_ext shape: {V_ext.shape}")

    ok = True
    ok &= check_arrays_finite("t_ms", t_ms)
    ok &= check_arrays_finite("V_in_soma_mV", V_in)
    ok &= check_arrays_finite("V_ext_soma_mV", V_ext)

    if V_in.ndim != 2 or V_ext.ndim != 2:
        print("[FAIL] V_in/V_ext must be 2D arrays shaped (Npos, T).")
        ok = False
    else:
        if V_in.shape != V_ext.shape:
            print("[FAIL] V_in and V_ext shapes do not match.")
            ok = False
        if V_in.shape[1] != t_ms.size:
            print("[FAIL] V_in time dimension does not match t length.")
            ok = False
        else:
            print("[OK]   V_in time dimension matches t length.")

    derived = compute_derived(t_ms, V_in, V_ext)

    print_start_values(t_ms, derived, positions_um)

    gating_magnitude_report(
        t_ms=t_ms,
        Vm=derived["V_m_soma_mV"],       # use membrane potential for gating report
        Vext=derived["V_ext_soma_mV"],
        positions_um=positions_um,
        on0=on0,
        on1=on1,
    )
    quick_absolute_summary(derived, positions_um)

    vmax = float(np.max(np.abs(derived["V_m_soma_mV"])))
    if vmax > 1e4:
        print(f"\n[WARN] max|Vm| is extremely large ({vmax:.3f} mV). 폭주 케이스 파일일 가능성이 큽니다.")
    elif vmax > VM_EXTREME_WARN_MV:
        print(f"\n[WARN] max|Vm| exceeds {VM_EXTREME_WARN_MV:.1f} mV ({vmax:.3f} mV).")
    else:
        print("\n[OK] Vm absolute range looks physiologically plausible for this sanity run.")

    return ok, t_ms, derived, positions_um, (on0, on1)


def plot_results(
    npy_path: str,
    t_ms: np.ndarray,
    derived: Dict[str, np.ndarray],
    positions_um: np.ndarray,
    on_window: Tuple[float, float],
    efield_scale: float,
) -> None:
    import matplotlib.pyplot as plt

    on0, on1 = on_window
    npos = derived["V_in_soma_mV"].shape[0]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(script_dir, "plot")
    os.makedirs(plot_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(npy_path))[0]
    scale_tag = str(float(efield_scale)).rstrip("0").rstrip(".").replace(".", "p")

    # Helper to draw one figure and always save PNGs into ./plot/
    def draw_fig(y: np.ndarray, ylabel: str, title: str, suffix: str) -> None:
        plt.figure()
        for i in range(npos):
            pos = positions_um[i] if (positions_um.ndim == 2 and positions_um.shape[1] == 3) else (np.nan, np.nan, np.nan)
            plt.plot(t_ms, y[i], label=f"N{i+1} ({pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f})")
        plt.axvspan(on0, on1, alpha=0.15)
        plt.xlabel("Time (ms)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()

        # Save into ./plot/ (default behavior)
        # Filename order requested: scale first, variable second.
        out1 = os.path.join(plot_dir, f"{scale_tag}x_{suffix}_{base_name}.png")
        plt.savefig(out1, dpi=200, bbox_inches="tight")
        print(f"[OK] Saved figure: {out1}")

    # TEMP: per request, disable V_in / V_ext plot+save.
    # draw_fig(
    #     y=derived["V_in_soma_mV"],
    #     ylabel="V_in (soma.v, mV)",
    #     title="V_in (soma.v, intracellular) for 4 positions",
    #     suffix="V_in",
    # )
    #
    # draw_fig(
    #     y=derived["V_ext_soma_mV"],
    #     ylabel="V_ext (soma.vext[0], mV)",
    #     title="V_ext (soma.vext[0], extracellular) for 4 positions",
    #     suffix="V_ext",
    # )

    # 3) V_m = V_in - V_ext (membrane)
    draw_fig(
        y=derived["V_m_soma_mV"],
        ylabel="V_m = V_in - V_ext (mV)",
        title="V_m (membrane, V_in - V_ext) for 4 positions",
        suffix="V_m",
    )

    plt.show()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "scale",
        nargs="?",
        default=None,
        help="Optional scale to load. If omitted, all simulate_four_v2 outputs are processed.",
    )
    ap.add_argument("--latest", action="store_true", help="Load the newest .npy in ./output by modified time.")
    ap.add_argument("--file", type=str, default=None, help="Path to a specific .npy file.")
    ap.add_argument("--no-plot", action="store_true", help="Run checks only, do not plot.")
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")

    if args.file is not None:
        paths = [os.path.abspath(args.file)]
    elif args.scale is not None:
        target_scale = float(args.scale)
        path = find_latest_sanity_by_scale(output_dir, target_scale)
        if path is None and args.latest:
            path = find_latest_npy(output_dir)
        if path is None:
            raise FileNotFoundError(
                f"No matched sanity .npy found in {output_dir} "
                f"(requested scale={target_scale:g}x)."
            )
        paths = [os.path.abspath(path)]
    else:
        # Default behavior requested: iterate all simulate_four_v2 outputs.
        paths = [os.path.abspath(p) for p in find_all_simulate_four_outputs(output_dir)]
        if not paths:
            raise FileNotFoundError(f"No sanity .npy files found in {output_dir}")

    for i, path in enumerate(paths, start=1):
        print(f"\n[{i}/{len(paths)}] Loaded: {path}")

        payload = load_payload(path)
        efield_scale = float(payload.get("efield_scale", 1.0))
        ok, t_ms, derived, positions_um, on_window = run_checks(payload)

        if ok:
            print("\n[OK] Basic integrity checks passed.")
        else:
            print("\n[WARN] Integrity checks failed. 위 로그를 확인하십시오.")

        if not args.no_plot:
            plot_results(
                npy_path=path,
                t_ms=t_ms,
                derived=derived,
                positions_um=positions_um,
                on_window=on_window,
                efield_scale=efield_scale,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
