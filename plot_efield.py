"""
plot_efield.py

지정된 세 위치에서 시간에 따른 E-field(Ex, Ez)를 서브플롯 3개로 보여주는 스크립트.

- 위치(µm): (35, 0, 550), (35, -80, 550), (0, -80, 550)
- 입력:
    - efield/E_field_4cycle.npy       (shape: (n_comp, n_pos, n_time), V/m)
    - efield/E_field_grid_coords.npy  (shape: (n_pos, 3), m)
- 출력: 화면에 figure만 표시 (저장은 하지 않음)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

E_FIELD_VALUES_FILE = os.path.join(SCRIPT_DIR, "efield", "E_field_4cycle.npy")
E_GRID_COORDS_FILE = os.path.join(SCRIPT_DIR, "efield", "E_field_grid_coords.npy")

# 시각화할 세 위치 (µm)
TARGET_POSITIONS_UM = np.array(
    [
        [80,   0.0, 550.0],
        [80, 35, 550.0],
        [0.0,  35, 550.0],
    ],
    dtype=float,
)

# simulate_allen.py 와 동일한 시간 간격
TIME_STEP_US = 50.0
TIME_STEP_MS = TIME_STEP_US / 1000.0  # 0.05 ms


def main() -> None:
    script_dir = Path(SCRIPT_DIR)

    # --- E-field 로드 ---
    if not Path(E_FIELD_VALUES_FILE).exists():
        raise SystemExit(f"E-field 파일을 찾을 수 없습니다: {E_FIELD_VALUES_FILE}")
    if not Path(E_GRID_COORDS_FILE).exists():
        raise SystemExit(f"E-field 좌표 파일을 찾을 수 없습니다: {E_GRID_COORDS_FILE}")

    print("Loading E-field data...")
    E_field_values = np.load(E_FIELD_VALUES_FILE)  # (n_comp, n_pos, n_time)
    coords_m = np.load(E_GRID_COORDS_FILE)        # (n_pos, 3) in m
    coords_um = coords_m * 1e6                    # -> µm

    if E_field_values.ndim != 3:
        raise SystemExit(f"E_field_4cycle.npy shape가 (n_comp, n_pos, n_time)가 아닙니다: {E_field_values.shape}")

    n_comp, n_pos, n_time = E_field_values.shape
    print(f"  E-field shape: {E_field_values.shape}  (n_comp, n_pos, n_time)")
    print(f"  coords shape: {coords_um.shape}")

    if coords_um.shape[0] != n_pos:
        raise SystemExit(f"E-field pos 수({n_pos})와 coords 수({coords_um.shape[0]})가 다릅니다.")

    # 시간축 (ms)
    time_ms_full = np.arange(n_time, dtype=float) * TIME_STEP_MS

    # 시간 범위를 0~0.2 ms로 제한
    t_mask = (time_ms_full >= 0.0) & (time_ms_full <= 2)
    time_ms = time_ms_full[t_mask]

    # E-field 그리드에서 타깃 위치에 가장 가까운 인덱스 찾기
    tree = cKDTree(coords_um)
    indices = []
    for p in TARGET_POSITIONS_UM:
        dist, idx = tree.query(p.reshape(1, -1), k=1)
        indices.append(int(idx[0]))
    indices = np.array(indices, dtype=int)

    print("\nNearest grid indices for target positions:")
    for i, (pos, idx) in enumerate(zip(TARGET_POSITIONS_UM, indices)):
        print(f"  pos {i}: {tuple(pos)} um -> grid index {idx}, coord {tuple(coords_um[idx])} um")

    # 사용할 컴포넌트: Ex, Ey, Ez (또는 가용한 만큼)
    # n_comp >=3 인 경우 [0]=Ex, [1]=Ey, [2]=Ez
    # n_comp == 2 인 경우 [0]=Ex, [1]=Ez 라고 가정, Ey는 0으로 둠
    # n_comp == 1 인 경우 [0]만 사용
    Ex_sel = E_field_values[0, indices, :][:, t_mask]  # (3, n_time_sel)
    Ey_sel = None
    Ez_sel = None

    if n_comp >= 3:
        Ey_sel = E_field_values[1, indices, :][:, t_mask]
        Ez_sel = E_field_values[2, indices, :][:, t_mask]
    elif n_comp == 2:
        Ez_sel = E_field_values[1, indices, :][:, t_mask]
    else:
        # n_comp == 1: Ex만 존재
        pass

    # 공통 y축 범위 계산 (존재하는 모든 컴포넌트 포함)
    vals = [Ex_sel]
    if Ey_sel is not None:
        vals.append(Ey_sel)
    if Ez_sel is not None:
        vals.append(Ez_sel)
    all_min = float(min(v.min() for v in vals))
    all_max = float(max(v.max() for v in vals))
    if all_max > all_min:
        pad = 0.05 * (all_max - all_min)
    else:
        pad = 1.0
    y_min = all_min - pad
    y_max = all_max + pad

    # --- 플롯 ---
    fig, axes = plt.subplots(
        3, 1,
        figsize=(10, 6),
        sharex=True,
        sharey=True,
    )

    for i in range(3):
        ax = axes[i]
        pos = TARGET_POSITIONS_UM[i]

        ax.plot(time_ms, Ex_sel[i, :], label="Ex", color="C0", linewidth=1.5)
        if Ey_sel is not None:
            ax.plot(time_ms, Ey_sel[i, :], label="Ey", color="C2", linewidth=1.0, linestyle=":")
        if Ez_sel is not None:
            ax.plot(time_ms, Ez_sel[i, :], label="Ez", color="C1", linewidth=1.0, linestyle="--")

        ax.set_ylabel("E (V/m)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"pos {i}: (x, y, z) = ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) µm")

        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time (ms)")
    plt.tight_layout()

    # 저장 경로: ./plot/allen_efield_threepos.png
    plot_dir = script_dir / "plot"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / "allen_efield_threepos.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # 화면에도 표시
    plt.show()


if __name__ == "__main__":
    main()

