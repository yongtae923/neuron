# plot_one.py
"""
plot_one.py

Plot soma v, outside v (vext), and Vm over time for three fixed neuron positions simulated by simulate_one.py.

- Input: allen_*_threepos_results.npy
  (dict with keys: positions_um, time_ms, Vm_mV, vext_mV, v_in_mV)
- Output: 하나의 figure에 3개 subplot(행), 공통 y축. 화면에만 표시하고 저장하지 않음.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


def _find_default_input(script_dir: Path) -> Optional[Path]:
    """Pick the most recently modified allen_*_threepos_results.npy in ./output."""
    outdir = script_dir / "output"
    if not outdir.is_dir():
        return None
    candidates = list(outdir.glob("allen_*_threepos_results.npy"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_results(path: Path) -> dict:
    data = np.load(str(path), allow_pickle=True)
    if isinstance(data, np.ndarray) and data.ndim == 0:
        data = data.item()
    if not isinstance(data, dict):
        raise TypeError(f"Unexpected root type in npy: {type(data)}")
    required = ["positions_um", "time_ms", "Vm_mV", "vext_mV", "v_in_mV"]
    for k in required:
        if k not in data:
            raise KeyError(f"Key '{k}' not found in npy. Keys: {list(data.keys())}")
    return data


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser(
        description="Three-position Vm/vext time traces (3 subplots, shared y-axis).",
    )
    ap.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to allen_*_threepos_results.npy "
             "(default: ./output에서 가장 최근 allen_*_threepos_results.npy 사용)",
    )
    ap.add_argument(
        "--debug-print",
        action="store_true",
        help="Vm/vext 전체 범위(min/max)를 콘솔에 출력 (디버그용).",
    )
    args = ap.parse_args()

    input_path = Path(args.input) if args.input else _find_default_input(script_dir)
    if input_path is None:
        raise SystemExit("입력 파일을 찾을 수 없습니다. ./output/allen_*_threepos_results.npy 를 확인하세요.")
    if not input_path.exists():
        raise SystemExit(f"입력 파일이 존재하지 않습니다: {input_path}")

    data = _load_results(input_path)

    positions_um = np.asarray(data["positions_um"])
    time_ms = np.asarray(data["time_ms"])
    Vm_mV = np.asarray(data["Vm_mV"])
    vext_mV = np.asarray(data["vext_mV"])
    v_in_mV = np.asarray(data["v_in_mV"])
    cell_id = str(data.get("cell_id", "allen"))

    if (
        positions_um.shape[0] != Vm_mV.shape[0]
        or positions_um.shape[0] != vext_mV.shape[0]
        or positions_um.shape[0] != v_in_mV.shape[0]
    ):
        raise SystemExit(
            f"positions_um, Vm_mV, vext_mV 첫 번째 차원이 다릅니다: "
            f"{positions_um.shape}, {Vm_mV.shape}, {vext_mV.shape}"
        )

    n_pos, n_times = Vm_mV.shape
    if n_pos != 3:
        print(f"경고: 예상한 3 포인트가 아니라 {n_pos} 포인트가 있습니다. 그대로 플롯합니다.")

    # Figure with 3 rows, shared x/y
    fig, axes = plt.subplots(
        n_pos, 1,
        figsize=(10, 6),
        sharex=True,
        sharey=True,
    )
    if n_pos == 1:
        axes = [axes]

    # 공통 y-limits를 위해 전체 soma v / vext / Vm 범위 계산
    all_min = float(min(Vm_mV.min(), vext_mV.min(), v_in_mV.min()))
    all_max = float(max(Vm_mV.max(), vext_mV.max(), v_in_mV.max()))

    if args.debug_print:
        print(f"Global Vm/vext range: {all_min:.3f} ~ {all_max:.3f} mV")

    # 살짝 여유를 두고 패딩을 추가해서 그래프가 축 밖으로 잘리지 않게 함
    if all_max > all_min:
        pad = 0.05 * (all_max - all_min)
    else:
        pad = 1.0
    y_min = all_min - pad
    y_max = all_max + pad

    for i in range(n_pos):
        ax = axes[i]
        vm = Vm_mV[i, :]
        ve = vext_mV[i, :]
        vin = v_in_mV[i, :]
        pos = positions_um[i]

        # 먼저 vext, Vm을 얇게 그린 뒤, soma v를 가장 두껍게/위에 오도록 그림
        ax.plot(
            time_ms,
            ve,
            label="vext (outside)",
            color="C1",
            linewidth=1.0,
            linestyle="--",
            zorder=1,
        )
        ax.plot(
            time_ms,
            vm,
            label="Vm = v - vext",
            color="C0",
            linewidth=1.5,
            linestyle="-",
            zorder=2,
        )
        ax.plot(
            time_ms,
            vin,
            label="soma v (inside)",
            color="C2",
            linewidth=2.0,
            linestyle="-",
            zorder=3,
        )

        ax.set_ylabel("mV")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"pos {i}: (x, y, z) = ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) µm")

        ax.set_ylim(y_min, y_max)

        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (ms)")

    plt.tight_layout()

    # 플롯 저장 경로: ./plot/allen_<cell_id>_threepos_vm_vext.png
    plot_dir = script_dir / "plot"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / f"allen_{cell_id}_threepos_vm_vext.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # 화면에도 표시
    plt.show()


if __name__ == "__main__":
    main()

