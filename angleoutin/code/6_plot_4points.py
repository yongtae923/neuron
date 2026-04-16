# D:\yongtae\neuron\angleoutin\code\6_plot_4points.py

"""
기능:
- 5_allen_4points.py 결과를 불러와 soma 막전위(V_m) 그래프를 저장합니다.

입출력:
- 입력: data/30V_OUT10_IN20_CI/5_gradient_output/gradient_sanity_1x_cell529898751.npy
- 출력: data/30V_OUT10_IN20_CI/6_gradient_plot/*.png

실행 방법:
- python 6_plot_4points.py
"""

from __future__ import annotations

import os
from typing import List

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data", "30V_OUT10_IN20_CI")
CELL_ID = "529898751"
OUTPUT_DIR = os.path.join(DATA_DIR, "5_gradient_output")
PLOT_DIR = os.path.join(DATA_DIR, "6_gradient_plot")
GRAD_ON_WINDOW_MS = (0.0, 4.0)
TARGET_GAIN = 1.0


def main() -> None:
    os.makedirs(PLOT_DIR, exist_ok=True)
    import matplotlib.pyplot as plt

    npy_name = f"gradient_sanity_1x_cell{CELL_ID}.npy"
    npy_path = os.path.join(OUTPUT_DIR, npy_name)
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Missing input file: {npy_path}")

    obj = np.load(npy_path, allow_pickle=True)
    payload = obj.item() if (isinstance(obj, np.ndarray) and obj.shape == ()) else obj

    t_ms = np.asarray(payload["t_ms"], dtype=np.float64)
    V_in = np.asarray(payload["V_in_soma_mV"], dtype=np.float64)
    V_ext = np.asarray(payload["V_ext_soma_mV"], dtype=np.float64)
    positions_um = np.asarray(payload["positions_um"], dtype=np.float64)
    labels = payload.get("position_labels", None)

    if labels is None:
        labels_list: List[str] = [f"N{i+1}" for i in range(V_in.shape[0])]
    else:
        labels_list = [str(v) for v in np.asarray(labels).tolist()]

    V_m = V_in - V_ext
    on0, on1 = payload.get("gradient_on_window_ms", GRAD_ON_WINDOW_MS)

    plt.figure()
    for i in range(V_m.shape[0]):
        pos = positions_um[i]
        label = labels_list[i] if i < len(labels_list) else f"N{i+1}"
        plt.plot(t_ms, V_m[i], label=f"{label} ({pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f})")
    plt.axvspan(on0, on1, alpha=0.15)
    plt.xlabel("Time (ms)")
    plt.ylabel("V_m = V_in - V_ext (mV)")
    plt.title(f"Gradient run V_m (gain={TARGET_GAIN:g}x)")
    plt.legend()

    out_path = os.path.join(PLOT_DIR, f"gradient_1x_V_m_{CELL_ID}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")
    print("\nDone. Saved 1 plot.")


if __name__ == "__main__":
    main()
