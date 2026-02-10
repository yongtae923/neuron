# plot_three.py
"""
plot_three.py

Plot soma v, outside v (vext), and Vm over time for three fixed neuron positions simulated by simulate_three.py.

- Input: allen_*_threepos_results.npy
  (dict with keys: positions_um, time_ms, Vm_mV, vext_mV, v_in_mV)
- Output: 하나의 figure에 3개 subplot(행), 공통 y축. 화면에 표시하고 plot/ 폴더에 저장.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


# Valid E-field scale suffixes (from simulate_three.py)
SCALE_SUFFIXES = ("1x", "10x", "100x", "1000x")


def _find_input_by_scale(script_dir: Path, scale: str) -> Optional[Path]:
    """Find allen_*_threepos_results_{scale}.npy in ./output."""
    if scale not in SCALE_SUFFIXES:
        return None
    outdir = script_dir / "output"
    if not outdir.is_dir():
        return None
    pattern = f"allen_*_threepos_results_{scale}.npy"
    candidates = list(outdir.glob(pattern))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _find_default_input(script_dir: Path) -> Optional[Path]:
    """Pick the most recently modified allen_*_threepos_results*.npy in ./output."""
    outdir = script_dir / "output"
    if not outdir.is_dir():
        return None
    candidates = list(outdir.glob("allen_*_threepos_results*.npy"))
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
        help="Path to allen_*_threepos_results.npy (or *_1x.npy, *_10x.npy, etc.).",
    )
    ap.add_argument(
        "--scale",
        type=str,
        default=None,
        choices=SCALE_SUFFIXES,
        help="E-field scale to plot: 1x, 10x, 100x, or 1000x (uses output/allen_*_threepos_results_{scale}.npy).",
    )
    ap.add_argument(
        "--debug-print",
        action="store_true",
        help="Vm/vext 전체 범위(min/max)를 콘솔에 출력 (디버그용).",
    )
    args = ap.parse_args()

    if args.input:
        input_path = Path(args.input)
    elif args.scale:
        input_path = _find_input_by_scale(script_dir, args.scale)
        if input_path is None:
            raise SystemExit(
                f"입력 파일을 찾을 수 없습니다. ./output/allen_*_threepos_results_{args.scale}.npy 를 확인하세요."
            )
    else:
        input_path = _find_default_input(script_dir)

    if input_path is None:
        raise SystemExit(
            "입력 파일을 찾을 수 없습니다. --input 경로 지정 또는 --scale 1x|10x|100x|1000x 를 사용하세요."
        )
    if not input_path.exists():
        raise SystemExit(f"입력 파일이 존재하지 않습니다: {input_path}")

    data = _load_results(input_path)

    positions_um = np.asarray(data["positions_um"])
    time_ms = np.asarray(data["time_ms"])
    Vm_mV = np.asarray(data["Vm_mV"])
    vext_mV = np.asarray(data["vext_mV"])
    v_in_mV = np.asarray(data["v_in_mV"])
    cell_id = str(data.get("cell_id", "allen"))
    time_roi_ms = data.get("time_roi_ms", None)
    pre_efield_ms = data.get("pre_efield_ms", None)
    e_field_scale = data.get("e_field_scale", None)

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
    if n_pos != 4:
        print(f"경고: 예상한 4 포인트가 아니라 {n_pos} 포인트가 있습니다. 그대로 플롯합니다.")

    # Figure with n_pos rows, shared x only (y축은 독립)
    fig, axes = plt.subplots(
        n_pos, 1,
        figsize=(10, 6),
        sharex=True,
        sharey=False,
    )
    if n_pos == 1:
        axes = [axes]

    if args.debug_print:
        print(f"Vm range: {Vm_mV.min():.3f} ~ {Vm_mV.max():.3f} mV")

    for i in range(n_pos):
        ax = axes[i]
        vm = Vm_mV[i, :]
        pos = positions_um[i]

        # Vm만 표시
        ax.plot(
            time_ms,
            vm,
            label="Vm = v - vext",
            color="C0",
            linewidth=1.5,
            linestyle="-",
            zorder=2,
        )

        # E-field 적용 구간 표시 (수직선) - 더 진하게
        if time_roi_ms is not None:
            ax.axvline(time_roi_ms[0], color="black", linestyle="--", linewidth=2.0, alpha=0.8, label="E-field start")
            ax.axvline(time_roi_ms[1], color="black", linestyle="--", linewidth=2.0, alpha=0.8, label="E-field end")

        # 각 subplot마다 개별 y축 범위 설정
        vm_min = float(vm.min())
        vm_max = float(vm.max())
        if vm_max > vm_min:
            pad = 0.05 * (vm_max - vm_min)
        else:
            pad = 1.0
        y_min = vm_min - pad
        y_max = vm_max + pad
        ax.set_ylim(y_min, y_max)

        ax.set_ylabel("mV")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"pos {i}: (x, y, z) = ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) µm")

        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (ms)")

    # 전체 제목 추가
    title_parts = [f"Cell {cell_id}"]
    if e_field_scale is not None:
        title_parts.append(f"E-field scale: {e_field_scale:.1f}x")
    if time_roi_ms is not None:
        title_parts.append(f"E-field: {time_roi_ms[0]:.1f}~{time_roi_ms[1]:.1f} ms")
    if pre_efield_ms is not None:
        title_parts.append(f"Pre-sim: -{pre_efield_ms:.1f}~0 ms")
    fig.suptitle(", ".join(title_parts), fontsize=10)

    plt.tight_layout()

    # 플롯 저장 경로: ./plot/allen_<cell_id>_threepos_vm_vext[_{scale}].png
    plot_dir = script_dir / "plot"
    plot_dir.mkdir(parents=True, exist_ok=True)
    scale_suffix = f"_{int(e_field_scale)}x" if e_field_scale is not None and e_field_scale >= 1.0 else (
        f"_{e_field_scale:.1f}x" if e_field_scale is not None else ""
    )
    out_path = plot_dir / f"allen_{cell_id}_threepos_vm_vext{scale_suffix}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # 메모리 절약을 위해 figure 닫기 (화면 표시 안 함)
    plt.close()


if __name__ == "__main__":
    main()
