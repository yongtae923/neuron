# plot_and_check_four.py
# ------------------------------------------------------------
# Plot + sanity checks for simulate_four_A_coil_excluded.py outputs.
#
# Supports:
#   - --latest : load newest .npy in ./output by mtime (robust)
#   - --file   : load a specific .npy file
#   - --no-plot: run checks only (print), skip plotting
#   - --savefig: save plot PNG next to the npy file
#
# Usage examples:
#   python plot_and_check_four.py --latest
#   python plot_and_check_four.py --file output/<file>.npy
#   python plot_and_check_four.py --latest --no-plot
#   python plot_and_check_four.py --latest --savefig
# ------------------------------------------------------------

from __future__ import annotations

import os
import argparse
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


def load_payload(path: str) -> Dict[str, Any]:
    obj = np.load(path, allow_pickle=True)
    # simulate_four saves dict payload via np.save, so loading returns ndarray scalar
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
    ok = True
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        print(f"[FAIL] {name}: NaN/Inf detected")
        ok = False
    else:
        print(f"[OK]   {name}: NaN/Inf not found")
    return ok


def gating_magnitude_report(
    t_ms: np.ndarray,
    Vm: np.ndarray,          # (N, T)
    Vext: np.ndarray,        # (N, T)
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


def quick_absolute_summary(
    Vm: np.ndarray,
    Vext: np.ndarray,
    positions_um: np.ndarray,
) -> None:
    print("\n=== Quick absolute range summary ===")
    for i, pos in enumerate(positions_um):
        v = Vm[i]
        ve = Vext[i]
        print(
            f"Neuron {i+1} @ ({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}): "
            f"min(Vm)={np.min(v):.6f}, max(Vm)={np.max(v):.6f}, max|Vm|={np.max(np.abs(v)):.6f} mV"
        )
        print(
            f"          min(Vext)={np.min(ve):.6f}, max(Vext)={np.max(ve):.6f}, max|Vext|={np.max(np.abs(ve)):.6f} mV"
        )


def run_checks(payload: Dict[str, Any]) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray, Tuple[float, float]]:
    # Required keys (based on your simulate script)
    t_ms = np.asarray(payload["t_ms"], dtype=np.float64)
    Vm = np.asarray(payload["Vm_soma_mV"], dtype=np.float64)
    Vext = np.asarray(payload["Vext_soma_mV"], dtype=np.float64)
    positions_um = np.asarray(payload["positions_um"], dtype=np.float64)

    if "efield_on_window_ms" in payload:
        on0, on1 = payload["efield_on_window_ms"]
        on0 = float(on0)
        on1 = float(on1)
    else:
        print(f"[WARN] 'efield_on_window_ms' not found. Using hard-coded DEFAULT_EFIELD_ON_WINDOW_MS={DEFAULT_EFIELD_ON_WINDOW_MS}.")
        on0, on1 = DEFAULT_EFIELD_ON_WINDOW_MS

    cell_id = str(payload.get("cell_id", "UNKNOWN"))
    print("Cell ID:", cell_id)
    print(f"Vm shape: {Vm.shape}, t shape: {t_ms.shape}")
    print(f"Vext shape: {Vext.shape}")

    ok = True
    ok &= check_arrays_finite("t_ms", t_ms)
    ok &= check_arrays_finite("Vm_soma_mV", Vm)
    ok &= check_arrays_finite("Vext_soma_mV", Vext)

    if Vm.ndim != 2 or Vext.ndim != 2:
        print("[FAIL] Vm/Vext must be 2D arrays shaped (Npos, T).")
        ok = False
    else:
        if Vm.shape != Vext.shape:
            print("[FAIL] Vm and Vext shapes do not match.")
            ok = False
        if Vm.shape[1] != t_ms.size:
            print("[FAIL] Vm time dimension does not match t length.")
            ok = False
        else:
            print("[OK]   Vm time dimension matches t length.")

    # Optional: positions sanity
    if positions_um.ndim != 2 or positions_um.shape[1] != 3 or positions_um.shape[0] != Vm.shape[0]:
        print("[WARN] positions_um shape is unexpected. Plot legends may be off.")
    else:
        for i in range(positions_um.shape[0]):
            pass

    gating_magnitude_report(
        t_ms=t_ms,
        Vm=Vm,
        Vext=Vext,
        positions_um=positions_um,
        on0=on0,
        on1=on1,
    )
    quick_absolute_summary(Vm, Vext, positions_um)

    # Strong sanity: Vm should not explode for coil-excluded small E
    vmax = float(np.max(np.abs(Vm)))
    if vmax > 1e4:
        print(f"\n[WARN] max|Vm| is extremely large ({vmax:.3f} mV). 저장된 파일이 폭주 케이스일 가능성이 큽니다.")
    elif vmax > VM_EXTREME_WARN_MV:
        print(f"\n[WARN] max|Vm| exceeds {VM_EXTREME_WARN_MV:.1f} mV ({vmax:.3f} mV).")
    else:
        print("\n[OK] Vm absolute range looks physiologically plausible for this sanity run.")

    return ok, t_ms, Vm, Vext, (on0, on1)


def plot_results(
    path: str,
    t_ms: np.ndarray,
    Vm: np.ndarray,
    Vext: np.ndarray,
    positions_um: np.ndarray,
    on_window: Tuple[float, float],
) -> None:
    """Plot Vm/Vext and always save PNGs into ./plot/ (default behavior)."""
    import matplotlib.pyplot as plt

    on0, on1 = on_window
    npos = Vm.shape[0]

    # Figure 1: Vm for all positions
    plt.figure()
    for i in range(npos):
        pos = positions_um[i] if (positions_um.ndim == 2 and positions_um.shape[1] == 3) else (np.nan, np.nan, np.nan)
        plt.plot(t_ms, Vm[i], label=f"N{i+1} ({pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f})")
    plt.axvspan(on0, on1, alpha=0.15)
    plt.xlabel("Time (ms)")
    plt.ylabel("Vm soma (mV)")
    plt.title("Soma Vm (4 positions)")
    plt.legend()

    # Figure 2: Vext for all positions
    plt.figure()
    for i in range(npos):
        pos = positions_um[i] if (positions_um.ndim == 2 and positions_um.shape[1] == 3) else (np.nan, np.nan, np.nan)
        plt.plot(t_ms, Vext[i], label=f"N{i+1} ({pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f})")
    plt.axvspan(on0, on1, alpha=0.15)
    plt.xlabel("Time (ms)")
    plt.ylabel("Vext soma (mV)")
    plt.title("Soma Vext (4 positions)")
    plt.legend()

    # Always save into ./plot/ folder by default
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(script_dir, "plot")
    os.makedirs(plot_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(path))[0]
    out_vm = os.path.join(plot_dir, base_name + "_Vm.png")
    out_vext = os.path.join(plot_dir, base_name + "_Vext.png")

    plt.figure(1)
    plt.savefig(out_vm, dpi=200, bbox_inches="tight")
    plt.figure(2)
    plt.savefig(out_vext, dpi=200, bbox_inches="tight")
    print(f"[OK] Saved figures:\n  {out_vm}\n  {out_vext}")

    plt.show()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latest", action="store_true", help="Load the newest .npy in ./output by modified time.")
    ap.add_argument("--file", type=str, default=None, help="Path to a specific .npy file.")
    ap.add_argument("--no-plot", action="store_true", help="Run checks only, do not plot.")
    # Saving PNG into ./plot is now the default; no flag needed.
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")

    if args.file is not None:
        path = args.file
    elif args.latest:
        path = find_latest_npy(output_dir)
        if path is None:
            raise FileNotFoundError(f"No .npy files found in {output_dir}")
    else:
        # default behavior: load latest by mtime as well (more user-friendly)
        path = find_latest_npy(output_dir)
        if path is None:
            raise FileNotFoundError(f"No .npy files found in {output_dir}")

    path = os.path.abspath(path)
    print("\nLoaded:", path)

    payload = load_payload(path)
    ok, t_ms, Vm, Vext, on_window = run_checks(payload)

    if ok:
        print("\n[OK] Basic integrity checks passed.")
    else:
        print("\n[WARN] Integrity checks failed. 위 로그를 확인하십시오.")

    if not args.no_plot:
        positions_um = np.asarray(payload.get("positions_um", np.zeros((Vm.shape[0], 3))), dtype=np.float64)
        plot_results(
            path=path,
            t_ms=t_ms,
            Vm=Vm,
            Vext=Vext,
            positions_um=positions_um,
            on_window=on_window,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
