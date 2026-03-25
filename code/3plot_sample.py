"""
3_plot_sample.py
------------------------------------------------------------
Plot 3_plot_sample.py results for selected gains.
Saves figures to plot/600us_50Hz_10umspaing_100mA

사용 예:
    cd D:\yongtae\neuron\
    conda activate neuronconda
    python .\code\3_plot_sample.py
------------------------------------------------------------
"""

from __future__ import annotations

import os
import math
from pathlib import Path
from typing import List

import numpy as np

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
CELL_ID = "529898751"

OUTPUT_DIR = BASE_DIR / "output" / "600us_50Hz_10umspaing_100mA"
PLOT_DIR = BASE_DIR / "plot" / "600us_50Hz_10umspaing_100mA"
GRAD_ON_WINDOW_MS = (0.0, 4.0)
DEFAULT_GAINS = [1.0, 5.0, 10.0, 20.0]
GAIN_TO_MA = {
    1.0: "100",
    5.0: "500",
    10.0: "1000",
    20.0: "2000",
}


def gain_tag(v: float) -> str:
    fv = float(v)
    if fv <= 0.0:
        return f"{fv:.6g}".replace(".", "_")
    if abs(fv - 1.0) < 1e-12:
        return "1"
    if abs(fv - 5.0) < 1e-12:
        return "5"
    if abs(fv - 10.0) < 1e-12:
        return "10"
    if abs(fv - 20.0) < 1e-12:
        return "20"
    return f"{fv:.6g}".replace(".", "_")


def main() -> None:
    os.makedirs(PLOT_DIR, exist_ok=True)
    import matplotlib.pyplot as plt

    processed: List[str] = []
    skipped: List[str] = []

    for gain in DEFAULT_GAINS:
        gtag = gain_tag(gain)
        ma_label = GAIN_TO_MA.get(gain, str(int(gain * 100)))
        npy_name = f"{gtag}x_cell{CELL_ID}_{ma_label}mA_gradient_sanity.npy"
        npy_path = OUTPUT_DIR / npy_name
        if not npy_path.exists():
            skipped.append(npy_name)
            continue

        obj = np.load(str(npy_path), allow_pickle=True)
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
        plt.title(f"Gradient run V_m (gain={gain:g}x, {ma_label} mA)")
        plt.legend()

        out_path = PLOT_DIR / f"gradient_{gtag}x_{ma_label}mA_V_m_{CELL_ID}.png"
        plt.savefig(str(out_path), dpi=200, bbox_inches="tight")
        plt.close()
        processed.append(str(out_path))
        print(f"Saved: {out_path}")

    print(f"\nDone. Saved {len(processed)} plot(s).")
    if skipped:
        print(f"Skipped missing {len(skipped)} file(s):")
        for s in skipped:
            print("  ", s)


if __name__ == "__main__":
    main()
