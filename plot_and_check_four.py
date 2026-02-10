# plot_and_check_four.py
"""
Plot + sanity-check for simulate_four.py outputs.

What it does
1) Loads a results .npy saved by simulate_four.py (dict with allow_pickle=True)
2) Plots:
   - Vm_soma vs time for all 4 positions
   - Vext_soma vs time for all 4 positions
   - ΔVm (demeaned) zoom to see tiny changes
3) Checks:
   - NaN/Inf
   - E-field window gating: response mostly inside [0,2] ms
   - Peak-to-peak and RMS in-window vs out-of-window
   - Optional: compare vext magnitude across positions

Run examples
  python plot_and_check_four.py --file output/sanity_cell529898751_...npy
  python plot_and_check_four.py --latest
"""

import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_results(path: str) -> dict:
    obj = np.load(path, allow_pickle=True)
    # saved as dict directly => np.ndarray scalar with dtype=object
    if isinstance(obj, np.lib.npyio.NpzFile):
        raise ValueError("This loader expects a single .npy dict file, not .npz.")
    data = obj.item()
    if not isinstance(data, dict):
        raise ValueError("Loaded object is not a dict. Check the saved format.")
    return data


def pick_latest_output(output_dir: str) -> str:
    files = sorted(glob.glob(os.path.join(output_dir, "*.npy")))
    if not files:
        raise FileNotFoundError(f"No .npy files found in: {output_dir}")
    return files[-1]


def basic_clean_check(arr: np.ndarray, name: str) -> bool:
    ok = True
    if np.any(np.isnan(arr)):
        print(f"[FAIL] {name}: contains NaN")
        ok = False
    if np.any(np.isinf(arr)):
        print(f"[FAIL] {name}: contains Inf")
        ok = False
    if ok:
        print(f"[OK]   {name}: NaN/Inf not found")
    return ok


def window_stats(x: np.ndarray, mask: np.ndarray) -> dict:
    xin = x[mask]
    xout = x[~mask]
    if xin.size == 0:
        return {"p2p_in": np.nan, "rms_in": np.nan, "p2p_out": np.nan, "rms_out": np.nan}
    # Demeaned for small modulation visibility
    din = xin - np.mean(xin)
    dout = xout - np.mean(xout) if xout.size > 0 else np.array([0.0])

    return {
        "p2p_in": float(np.max(din) - np.min(din)),
        "rms_in": float(np.sqrt(np.mean(din**2))),
        "p2p_out": float(np.max(dout) - np.min(dout)) if dout.size > 0 else 0.0,
        "rms_out": float(np.sqrt(np.mean(dout**2))) if dout.size > 0 else 0.0,
    }


def print_gating_report(t, Vm, Vext, on0, on1, positions):
    mask_in = (t >= on0) & (t <= on1)
    print("\n==============================")
    print("Gating / magnitude report")
    print("==============================")
    print(f"E-field window: {on0:.3f} ~ {on1:.3f} ms")
    print("Units: Vm in mV, Vext in mV, ΔVm in mV (demeaned).")

    for i in range(Vm.shape[0]):
        vm = Vm[i]
        vx = Vext[i] if Vext is not None else None

        s_vm = window_stats(vm, mask_in)
        # For Vext, we care about absolute magnitude often
        if vx is not None:
            s_vx = {
                "maxabs_in": float(np.max(np.abs(vx[mask_in]))) if np.any(mask_in) else np.nan,
                "maxabs_out": float(np.max(np.abs(vx[~mask_in]))) if np.any(~mask_in) else np.nan,
                "p2p_in": float(np.max(vx[mask_in]) - np.min(vx[mask_in])) if np.any(mask_in) else np.nan,
                "p2p_out": float(np.max(vx[~mask_in]) - np.min(vx[~mask_in])) if np.any(~mask_in) else np.nan,
            }
        else:
            s_vx = None

        pos = positions[i] if positions and i < len(positions) else None
        print(f"\n[Neuron {i+1}] position_um={pos}")
        print(f"  Vm:  p2p_in={s_vm['p2p_in']:.6f} mV ({s_vm['p2p_in']*1000:.3f} uV), "
              f"rms_in={s_vm['rms_in']:.6f} mV ({s_vm['rms_in']*1000:.3f} uV)")
        print(f"       p2p_out={s_vm['p2p_out']:.6f} mV ({s_vm['p2p_out']*1000:.3f} uV), "
              f"rms_out={s_vm['rms_out']:.6f} mV ({s_vm['rms_out']*1000:.3f} uV)")
        # Simple gate expectation: in-window modulation should be >= out-of-window
        if np.isfinite(s_vm["p2p_in"]) and np.isfinite(s_vm["p2p_out"]):
            if s_vm["p2p_in"] < (s_vm["p2p_out"] * 1.2 + 1e-6):
                print("  [WARN] Vm modulation is not clearly larger inside the on-window.")
            else:
                print("  [OK]   Vm modulation is larger inside the on-window.")

        if s_vx is not None:
            print(f"  Vext: max|in|={s_vx['maxabs_in']:.6f} mV, max|out|={s_vx['maxabs_out']:.6f} mV")
            if np.isfinite(s_vx["maxabs_in"]) and np.isfinite(s_vx["maxabs_out"]):
                if s_vx["maxabs_in"] < (s_vx["maxabs_out"] * 1.2 + 1e-6):
                    print("  [WARN] Vext is not clearly larger inside the on-window.")
                else:
                    print("  [OK]   Vext is larger inside the on-window.")


def plot_results(t, Vm, Vext, on0, on1, positions, title_prefix=""):
    n = Vm.shape[0]
    labels = []
    for i in range(n):
        if positions and i < len(positions):
            x, y, z = positions[i]
            labels.append(f"N{i+1} ({x:.0f},{y:.0f},{z:.0f})um")
        else:
            labels.append(f"N{i+1}")

    # 1) Vm plot
    plt.figure(figsize=(12, 6))
    for i in range(n):
        plt.plot(t, Vm[i], label=labels[i], linewidth=1.2)
    plt.axvspan(on0, on1, alpha=0.15, label="E-field ON window")
    plt.xlabel("Time (ms)")
    plt.ylabel("Soma Vm (mV)")
    plt.title(f"{title_prefix}Soma Vm (raw)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # 2) Vext plot (if present)
    if Vext is not None:
        plt.figure(figsize=(12, 6))
        for i in range(n):
            plt.plot(t, Vext[i], label=labels[i], linewidth=1.2)
        plt.axvspan(on0, on1, alpha=0.15, label="E-field ON window")
        plt.xlabel("Time (ms)")
        plt.ylabel("Soma Vext (mV)")
        plt.title(f"{title_prefix}Soma Vext")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

    # 3) ΔVm plot (demeaned) with zoom
    plt.figure(figsize=(12, 6))
    for i in range(n):
        d = Vm[i] - np.mean(Vm[i])
        plt.plot(t, d, label=labels[i], linewidth=1.2)
    plt.axvspan(on0, on1, alpha=0.15, label="E-field ON window")
    plt.xlabel("Time (ms)")
    plt.ylabel("ΔVm (mV, demeaned)")
    plt.title(f"{title_prefix}ΔVm (demeaned) for small modulation visibility")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", type=str, default=None, help="Path to output .npy")
    ap.add_argument("--latest", action="store_true", help="Pick latest .npy from ./output")
    ap.add_argument("--savefig", action="store_true", help="Save figures as png next to .npy")
    args = ap.parse_args()

    if args.latest:
        path = pick_latest_output(os.path.join(SCRIPT_DIR, "output"))
    elif args.file is not None:
        path = args.file
    else:
        raise SystemExit("Use --file <path.npy> or --latest")

    data = load_results(path)
    meta = data.get("meta", {})
    t = np.asarray(data["t_ms"], dtype=float)
    Vm = np.asarray(data["Vm_soma_mV"], dtype=float)
    Vext = np.asarray(data.get("Vext_soma_mV", None), dtype=float) if "Vext_soma_mV" in data else None

    positions = meta.get("positions_um", None)
    on_window = meta.get("efield_window_ms", [0.0, 2.0])
    on0, on1 = float(on_window[0]), float(on_window[1])

    print(f"\nLoaded: {path}")
    print(f"Cell ID: {meta.get('cell_id', 'unknown')}")
    print(f"Vm shape: {Vm.shape}, t shape: {t.shape}")
    if Vext is not None:
        print(f"Vext shape: {Vext.shape}")

    # Clean checks
    basic_clean_check(t, "t_ms")
    basic_clean_check(Vm, "Vm_soma_mV")
    if Vext is not None:
        basic_clean_check(Vext, "Vext_soma_mV")

    # Consistency checks
    if Vm.shape[1] != t.shape[0]:
        print("[FAIL] Vm time dimension does not match t length.")
    else:
        print("[OK]   Vm time dimension matches t length.")
    if Vext is not None and Vext.shape != Vm.shape:
        print("[WARN] Vext shape differs from Vm shape. Plotting may be off.")

    # Gating report
    print_gating_report(t, Vm, Vext, on0, on1, positions)

    # Plot
    title_prefix = f"cell {meta.get('cell_id', '')} | "
    plot_results(t, Vm, Vext, on0, on1, positions, title_prefix=title_prefix)

    # Optional save figures
    if args.savefig:
        base = os.path.splitext(path)[0]
        # Re-plot but save to file without blocking
        n = Vm.shape[0]
        labels = []
        for i in range(n):
            if positions and i < len(positions):
                x, y, z = positions[i]
                labels.append(f"N{i+1} ({x:.0f},{y:.0f},{z:.0f})um")
            else:
                labels.append(f"N{i+1}")

        plt.figure(figsize=(12, 6))
        for i in range(n):
            plt.plot(t, Vm[i], label=labels[i], linewidth=1.2)
        plt.axvspan(on0, on1, alpha=0.15, label="E-field ON window")
        plt.xlabel("Time (ms)")
        plt.ylabel("Soma Vm (mV)")
        plt.title(f"{title_prefix}Soma Vm (raw)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(base + "_Vm.png", dpi=200)

        if Vext is not None:
            plt.figure(figsize=(12, 6))
            for i in range(n):
                plt.plot(t, Vext[i], label=labels[i], linewidth=1.2)
            plt.axvspan(on0, on1, alpha=0.15, label="E-field ON window")
            plt.xlabel("Time (ms)")
            plt.ylabel("Soma Vext (mV)")
            plt.title(f"{title_prefix}Soma Vext")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(base + "_Vext.png", dpi=200)

        plt.figure(figsize=(12, 6))
        for i in range(n):
            d = Vm[i] - np.mean(Vm[i])
            plt.plot(t, d, label=labels[i], linewidth=1.2)
        plt.axvspan(on0, on1, alpha=0.15, label="E-field ON window")
        plt.xlabel("Time (ms)")
        plt.ylabel("ΔVm (mV, demeaned)")
        plt.title(f"{title_prefix}ΔVm (demeaned)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(base + "_dVm.png", dpi=200)

        print(f"\nSaved figures: {base}_Vm.png, {base}_Vext.png, {base}_dVm.png")


if __name__ == "__main__":
    # Resolve SCRIPT_DIR for --latest convenience
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    main()
