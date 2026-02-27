"""
gradient_four_plot.py
------------------------------------------------------------
Plot gradient_four.py results for gains 1x ~ 1e10x.
Saves figures to data/gradient_plot/
------------------------------------------------------------
"""

from __future__ import annotations

import os
import math
from typing import List

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CELL_ID = "529898751"
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data", "gradient_output")
PLOT_DIR = os.path.join(SCRIPT_DIR, "data", "gradient_plot")
GRAD_ON_WINDOW_MS = (0.0, 4.0)
DEFAULT_GAINS = [10.0 ** k for k in range(0, 11)]  # 1 ~ 1e10


def gain_tag(v: float) -> str:
    fv = float(v)
    if fv <= 0.0:
        return f"{fv:.6g}".replace(".", "_")
    if abs(fv - 1.0) < 1e-12:
        return "1"
    exp = math.log10(fv)
    exp_i = int(round(exp))
    if abs(exp - exp_i) < 1e-12 and exp_i >= 1:
        return f"10e{exp_i - 1}"
    return f"{fv:.6g}".replace(".", "_")


def main() -> None:
    os.makedirs(PLOT_DIR, exist_ok=True)
    import matplotlib.pyplot as plt

    processed: List[str] = []
    skipped: List[str] = []

    for gain in DEFAULT_GAINS:
        gtag = gain_tag(gain)
        npy_name = f"gradient_sanity_{gtag}x_cell{CELL_ID}.npy"
        npy_path = os.path.join(OUTPUT_DIR, npy_name)
        if not os.path.exists(npy_path):
            skipped.append(npy_name)
            continue

        obj = np.load(npy_path, allow_pickle=True)
        payload = obj.item() if (isinstance(obj, np.ndarray) and obj.shape == ()) else obj

        t_ms = np.asarray(payload["t_ms"], dtype=np.float64)
        V_in = np.asarray(payload["V_in_soma_mV"], dtype=np.float64)
        V_ext = np.asarray(payload["V_ext_soma_mV"], dtype=np.float64)
        positions_um = np.asarray(payload["positions_um"], dtype=np.float64)

        V_m = V_in - V_ext
        on0, on1 = payload.get("gradient_on_window_ms", GRAD_ON_WINDOW_MS)

        plt.figure()
        for i in range(V_m.shape[0]):
            pos = positions_um[i]
            plt.plot(t_ms, V_m[i], label=f"N{i+1} ({pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f})")
        plt.axvspan(on0, on1, alpha=0.15)
        plt.xlabel("Time (ms)")
        plt.ylabel("V_m = V_in - V_ext (mV)")
        plt.title(f"Gradient run V_m (gain={gain:g}x)")
        plt.legend()

        out_path = os.path.join(PLOT_DIR, f"gradient_{gtag}x_V_m_{CELL_ID}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        processed.append(out_path)
        print(f"Saved: {out_path}")

    print(f"\nDone. Saved {len(processed)} plot(s).")
    if skipped:
        print(f"Skipped missing {len(skipped)} file(s).")


if __name__ == "__main__":
    main()
