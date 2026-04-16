"""
3_plot_sample.py
------------------------------------------------------------
Plot 3_simulate_sample.py results for selected gains.
Saves figures to efield/30V_OUT10_IN20_CI as 3_*.png

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

OUTPUT_DIR = BASE_DIR / "efield" / "30V_OUT10_IN20_CI"
PLOT_DIR = OUTPUT_DIR
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


def choose_delta_unit(max_abs_delta_mV: float) -> tuple[float, str]:
    if max_abs_delta_mV >= 1e-3:
        return 1.0, "mV"
    if max_abs_delta_mV >= 1e-6:
        return 1e3, "uV"
    if max_abs_delta_mV >= 1e-9:
        return 1e6, "nV"
    if max_abs_delta_mV >= 1e-12:
        return 1e9, "pV"
    return 1e12, "fV"


def main() -> None:
    os.makedirs(PLOT_DIR, exist_ok=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    processed: List[str] = []
    skipped: List[str] = []

    for gain in DEFAULT_GAINS:
        gtag = gain_tag(gain)
        ma_label = GAIN_TO_MA.get(gain, str(int(gain * 100)))
        npy_name = f"3_{gtag}x_cell{CELL_ID}_{ma_label}mA_gradient_sanity.npy"
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
        # Visualize tiny responses clearly using baseline-relative delta.
        V_m_delta = V_m - V_m[:, [0]]
        on0, on1 = payload.get("gradient_on_window_ms", GRAD_ON_WINDOW_MS)

        delta_flat = V_m_delta[np.isfinite(V_m_delta)]
        max_abs_delta_mV = float(np.max(np.abs(delta_flat))) if delta_flat.size > 0 else 0.0
        scale, unit = choose_delta_unit(max_abs_delta_mV)
        V_plot = V_m_delta * scale

        fig, ax = plt.subplots()
        for i in range(V_plot.shape[0]):
            pos = positions_um[i]
            ax.plot(t_ms, V_plot[i], label=f"N{i+1} ({pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f})")

        if delta_flat.size > 0:
            y_abs = float(np.percentile(np.abs(delta_flat * scale), 99.5))
            y_abs = max(y_abs, 1e-6)
            ax.set_ylim(-1.15 * y_abs, 1.15 * y_abs)

        ax.axvspan(on0, on1, alpha=0.15)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel(f"Delta V_m = V_m(t) - V_m(t0) ({unit})")
        ax.set_title(f"Gradient run Delta V_m (gain={gain:g}x, {ma_label} mA)")
        ax.legend()
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

        out_path = PLOT_DIR / f"3_gradient_{gtag}x_{ma_label}mA_V_m_{CELL_ID}.png"
        fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
        plt.close(fig)
        processed.append(str(out_path))
        print(f"Saved: {out_path}")

    print(f"\nDone. Saved {len(processed)} plot(s).")
    if skipped:
        print(f"Skipped missing {len(skipped)} file(s):")
        for s in skipped:
            print("  ", s)


if __name__ == "__main__":
    main()
