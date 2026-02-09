"""
Utility functions for inspecting and visualizing allen_*_multipos_results.npy.

- inspect_multipos_npy: 구조/shape/요약 정보 출력
- plot_vm_hist_at_time: 특정 시간(ms)에서 Vm 분포 히스토그램 출력
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _load_multipos_dict(filepath: str) -> dict:
    """allen_*_multipos_results.npy (dict 저장) 로드 후 dict로 반환."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    data = np.load(filepath, allow_pickle=True)

    # np.save(dict, allow_pickle=True) → 0-dim ndarray(object) 인 경우 처리
    if isinstance(data, np.ndarray) and data.ndim == 0:
        data = data.item()

    if not isinstance(data, dict):
        raise TypeError(f"Unexpected root type in npy: {type(data)}")
    return data


def inspect_multipos_npy(filepath: str) -> None:
    """파일 구조와 주요 배열 정보 출력."""
    try:
        data = _load_multipos_dict(filepath)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    print("=" * 60)
    print("Structure of multipos_results.npy")
    print("=" * 60)
    print(f"File: {filepath}\n")
    print("Root keys:", list(data.keys()))
    print()

    def _describe_array(name: str, arr: np.ndarray) -> None:
        print(f"[{name}]")
        print(f"  shape: {arr.shape}")
        print(f"  dtype: {arr.dtype}")
        if arr.size > 0 and np.issubdtype(arr.dtype, np.floating):
            print(f"  min: {arr.min():.4g}, max: {arr.max():.4g}")
        print()

    for k, v in data.items():
        if isinstance(v, np.ndarray):
            _describe_array(k, v)
        else:
            print(f"[{k}] type={type(v)} value={v}")
            print()

    if "positions_um" in data and "time_ms" in data and "Vm_mV" in data:
        pos = np.asarray(data["positions_um"])
        tms = np.asarray(data["time_ms"])
        vm = np.asarray(data["Vm_mV"])
        print("Summary:")
        print(f"  positions: {pos.shape[0]} points x 3 (x,y,z um)")
        print(f"  time points: {len(tms)}")
        print(f"  Vm_mV: {vm.shape} (positions x time)")
        print("=" * 60)


def plot_vm_hist_at_time(
    filepath: str,
    target_t_ms: float = 0.2,
    bins: int = 100,
) -> None:
    """
    특정 시간(ms)에 해당하는 Vm 분포 히스토그램을 그립니다.

    - target_t_ms와 가장 가까운 time_ms 인덱스를 사용
    - x축: Vm (mV), y축: count
    """
    data = _load_multipos_dict(filepath)

    if "time_ms" not in data or "Vm_mV" not in data:
        raise KeyError(f"'time_ms' 또는 'Vm_mV'가 npy에 없습니다. keys={list(data.keys())}")

    time_ms = np.asarray(data["time_ms"])
    Vm_mV = np.asarray(data["Vm_mV"])

    if Vm_mV.ndim != 2:
        raise ValueError(f"Vm_mV shape expected (n_positions, n_times), got {Vm_mV.shape}")

    # target_t_ms에 가장 가까운 시간 인덱스 찾기
    idx = int(np.argmin(np.abs(time_ms - target_t_ms)))
    actual_t = float(time_ms[idx])

    vm_slice = Vm_mV[:, idx]

    print(f"File: {filepath}")
    print(f"Target time: {target_t_ms} ms → nearest index {idx} (time_ms={actual_t:.4f} ms)")
    print(f"Vm_mV slice shape: {vm_slice.shape}")
    print(f"Vm range: {vm_slice.min():.4g} ~ {vm_slice.max():.4g} mV")

    plt.figure(figsize=(8, 5))
    plt.hist(vm_slice, bins=bins, edgecolor="black", alpha=0.7)
    plt.xlabel("Vm (mV)")
    plt.ylabel("Count")
    plt.title(f"Vm histogram at t = {actual_t:.4f} ms")
    plt.tight_layout()
    plt.show()


def _default_output_path() -> str:
    """프로젝트 루트 기준 기본 npy 경로 추정."""
    script_dir = Path(__file__).resolve().parent
    default_path = script_dir / "output" / "allen_529898751_multipos_results.npy"
    return str(default_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="allen_*_multipos_results.npy 구조 확인 및 특정 시간 Vm 히스토그램 도구",
    )
    parser.add_argument(
        "filepath",
        nargs="?",
        default=_default_output_path(),
        help="대상 npy 파일 경로 (default: ./output/allen_529898751_multipos_results.npy)",
    )
    parser.add_argument(
        "--hist-time",
        type=float,
        default=None,
        help="이 시간이 가장 가까운 time_ms에서 Vm 히스토그램을 그림 (ms 단위, 예: 0.2)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=100,
        help="히스토그램 bin 개수 (default: 100)",
    )

    args = parser.parse_args()

    if args.hist_time is not None:
        # 히스토그램 모드
        plot_vm_hist_at_time(
            filepath=args.filepath,
            target_t_ms=args.hist_time,
            bins=args.bins,
        )
    else:
        # 기본: 구조 출력
        inspect_multipos_npy(args.filepath)

