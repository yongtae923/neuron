# D:\yongtae\neuron\angleoutin\code\3_gen_gradient.py

"""
기능:
- E-field에서 방향 미분(dEx/dx, dEy/dy, dEz/dz) gradient를 생성합니다.

입출력:
- 입력: data/400us_30V_OUT10_IN20/1_E_field_1cycle.npy, 1_E_field_grid_coords.npy, 0_grid_time_spec.json
- 출력: data/400us_30V_OUT10_IN20/3_gradient_1cycle.npy

실행 방법:
- python 3_gen_gradient.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath


SCRIPT_DIR = Path(__file__).resolve().parent
CASE_NAME = os.environ.get("ANGLEOUTIN_CASE", "400us_30V_OUT10_IN20")
DATA_DIR = SCRIPT_DIR.parent / "data" / CASE_NAME
SPEC_PATH = DATA_DIR / "0_grid_time_spec.json"
INPUT_EFIELD_PATH = DATA_DIR / "1_E_field_1cycle.npy"
INPUT_COORDS_PATH = DATA_DIR / "1_E_field_grid_coords.npy"
OUTPUT_DIR = DATA_DIR / "3_gradient"
OUTPUT_PATH = DATA_DIR / "3_gradient_1cycle.npy"
INPUT_EFIELD_2X_PATH = DATA_DIR / "1_E_field_1cycle_2x.npy"
INPUT_EFIELD_10X_PATH = DATA_DIR / "1_E_field_1cycle_10x.npy"
OUTPUT_2X_PATH = DATA_DIR / "3_gradient_1cycle_2x.npy"
OUTPUT_10X_PATH = DATA_DIR / "3_gradient_1cycle_10x.npy"


def _build_axis_and_indices(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    정렬된 고유 x/y/z 축과 각 포인트의 정수 인덱스를 생성합니다.

    반환값:
        x_axis, y_axis, z_axis, ix, iy, iz
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Unexpected coords shape: {coords.shape}, expected (N_spatial, 3)")

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    x_axis = np.unique(x)
    y_axis = np.unique(y)
    z_axis = np.unique(z)

    ix = np.searchsorted(x_axis, x)
    iy = np.searchsorted(y_axis, y)
    iz = np.searchsorted(z_axis, z)

    n_spatial = coords.shape[0]
    expected = x_axis.size * y_axis.size * z_axis.size
    if expected != n_spatial:
        raise ValueError(
            "Grid is not a full Cartesian product or has duplicates. "
            f"Expected Nx*Ny*Nz={expected}, but N_spatial={n_spatial}."
        )

    occupied = np.zeros((x_axis.size, y_axis.size, z_axis.size), dtype=np.uint8)
    occupied[ix, iy, iz] = 1
    if not np.all(occupied):
        raise ValueError("Detected missing grid points after index mapping.")

    return x_axis, y_axis, z_axis, ix, iy, iz


def _build_surface_shell_mask(coil_mask: np.ndarray) -> np.ndarray:
    """
    코일 마스크 바로 바깥 1-voxel(6-이웃) shell을 생성합니다.

    반환값:
        shell_mask: 코일 내부에 인접한 바깥 점에서 True.
    """
    padded = np.pad(coil_mask, 1, mode="constant", constant_values=False)
    neighbor_of_coil = (
        padded[2:, 1:-1, 1:-1]
        | padded[:-2, 1:-1, 1:-1]
        | padded[1:-1, 2:, 1:-1]
        | padded[1:-1, :-2, 1:-1]
        | padded[1:-1, 1:-1, 2:]
        | padded[1:-1, 1:-1, :-2]
    )
    shell_mask = neighbor_of_coil & (~coil_mask)
    return shell_mask


def main() -> None:
    print(f"Case name: {CASE_NAME}")
    if not SPEC_PATH.exists():
        raise FileNotFoundError(f"Missing spec file: {SPEC_PATH}")
    if not INPUT_EFIELD_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_EFIELD_PATH}")
    if not INPUT_COORDS_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_COORDS_PATH}")

    with SPEC_PATH.open("r", encoding="utf-8") as f:
        spec = json.load(f)

    # spec 단위는 um/us이므로, coords와 맞추기 위해 공간값을 m로 변환
    x_min_m = float(spec["space_um"]["x"]["min"]) * 1e-6
    x_max_m = float(spec["space_um"]["x"]["max"]) * 1e-6
    y_min_m = float(spec["space_um"]["y"]["min"]) * 1e-6
    y_max_m = float(spec["space_um"]["y"]["max"]) * 1e-6
    z_min_m = float(spec["space_um"]["z"]["min"]) * 1e-6
    z_max_m = float(spec["space_um"]["z"]["max"]) * 1e-6
    grid_step_m = float(spec["space_um"]["x"]["step"]) * 1e-6
    files_per_cycle = int(spec["data_files"]["per_cycle"])

    coil_polygon_xz_m = np.array(spec["coil_mask_um"]["polygon_xz"], dtype=np.float64) * 1e-6
    coil_y_min_m = float(spec["coil_mask_um"]["y"]["min"]) * 1e-6
    coil_y_max_m = float(spec["coil_mask_um"]["y"]["max"]) * 1e-6

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    coords = np.load(INPUT_COORDS_PATH)  # (N_spatial, 3)

    x_axis, y_axis, z_axis, ix, iy, iz = _build_axis_and_indices(coords)
    nx, ny, nz = x_axis.size, y_axis.size, z_axis.size

    # 데이터 격자/시간이 공통 angleoutin spec과 일치하는지 확인
    axis_tol = 1e-12
    if abs(float(x_axis[0]) - x_min_m) > axis_tol or abs(float(x_axis[-1]) - x_max_m) > axis_tol:
        raise ValueError("X axis range does not match spec.")
    if abs(float(y_axis[0]) - y_min_m) > axis_tol or abs(float(y_axis[-1]) - y_max_m) > axis_tol:
        raise ValueError("Y axis range does not match spec.")
    if abs(float(z_axis[0]) - z_min_m) > axis_tol or abs(float(z_axis[-1]) - z_max_m) > axis_tol:
        raise ValueError("Z axis range does not match spec.")
    if x_axis.size > 1 and abs(float(np.median(np.diff(x_axis))) - grid_step_m) > axis_tol:
        raise ValueError("X grid step does not match spec.")
    if y_axis.size > 1 and abs(float(np.median(np.diff(y_axis))) - grid_step_m) > axis_tol:
        raise ValueError("Y grid step does not match spec.")
    if z_axis.size > 1 and abs(float(np.median(np.diff(z_axis))) - grid_step_m) > axis_tol:
        raise ValueError("Z grid step does not match spec.")
    # 3D 코일 마스크: True는 코일 내부(gradient를 0 처리)
    # spec의 오각기둥 사용: y 범위 + xz 오각형(단위 m)
    pentagon = MplPath(coil_polygon_xz_m)
    xx, zz = np.meshgrid(x_axis, z_axis, indexing="ij")
    xz_points = np.column_stack([xx.ravel(), zz.ravel()])
    xz_inside = pentagon.contains_points(xz_points, radius=1e-15).reshape(x_axis.size, z_axis.size)
    y_in = (y_axis >= coil_y_min_m) & (y_axis <= coil_y_max_m)
    coil_mask = xz_inside[:, None, :] & y_in[None, :, None]
    shell_mask = _build_surface_shell_mask(coil_mask)
    suppress_mask = coil_mask | shell_mask

    n_coil = int(np.sum(coil_mask))
    n_shell = int(np.sum(shell_mask))
    print(f"Coil interior: {n_coil} points masked to 0")
    print(f"Coil surface shell (outside, 1-voxel): {n_shell} points masked to 0")

    jobs = [
        ("1x", INPUT_EFIELD_PATH, OUTPUT_PATH),
        ("2x", INPUT_EFIELD_2X_PATH, OUTPUT_2X_PATH),
        ("10x", INPUT_EFIELD_10X_PATH, OUTPUT_10X_PATH),
    ]

    if all(out_path.exists() for _, _, out_path in jobs):
        print("All output gradient files already exist. Nothing to do.")
        return

    for tag, efield_path, out_path in jobs:
        if out_path.exists():
            print(f"Skip {tag}: output already exists ({out_path})")
            continue

        if not efield_path.exists():
            if tag == "1x":
                raise FileNotFoundError(f"Missing input file: {efield_path}")
            print(f"Skip {tag}: missing input file ({efield_path})")
            continue

        efield = np.load(efield_path, mmap_mode="r")  # (3, N_spatial, Nt)
        if efield.ndim != 3 or efield.shape[0] != 3:
            raise ValueError(f"Unexpected E-field shape ({tag}): {efield.shape}, expected (3, N_spatial, Nt)")
        if coords.shape[0] != efield.shape[1]:
            raise ValueError(f"Mismatch ({tag}): coords N={coords.shape[0]} vs E-field spatial N={efield.shape[1]}")

        nt = efield.shape[2]
        if nt != files_per_cycle:
            raise ValueError(f"Time-step mismatch ({tag}): nt={nt}, spec per_cycle={files_per_cycle}")

        print(f"\n[{tag}] E-field shape: {efield.shape}")
        print(f"[{tag}] Grid shape: Nx={nx}, Ny={ny}, Nz={nz}, Nt={nt}")
        print(f"[{tag}] Computing: dEx/dx, dEy/dy, dEz/dz (edge_order=2)")

        grad_out = np.lib.format.open_memmap(
            out_path,
            mode="w+",
            dtype=np.float32,
            shape=(3, nx, ny, nz, nt),
        )

        ex_grid = np.empty((nx, ny, nz), dtype=np.float32)
        ey_grid = np.empty((nx, ny, nz), dtype=np.float32)
        ez_grid = np.empty((nx, ny, nz), dtype=np.float32)

        for t in range(nt):
            ex_grid[ix, iy, iz] = efield[0, :, t]
            ey_grid[ix, iy, iz] = efield[1, :, t]
            ez_grid[ix, iy, iz] = efield[2, :, t]

            grad_out[0, :, :, :, t] = np.gradient(ex_grid, x_axis, axis=0, edge_order=2).astype(np.float32, copy=False)
            grad_out[1, :, :, :, t] = np.gradient(ey_grid, y_axis, axis=1, edge_order=2).astype(np.float32, copy=False)
            grad_out[2, :, :, :, t] = np.gradient(ez_grid, z_axis, axis=2, edge_order=2).astype(np.float32, copy=False)

            for c in range(3):
                grad_out[c, :, :, :, t][suppress_mask] = 0.0

            if (t + 1) % 10 == 0 or (t + 1) == nt:
                print(f"[{tag}] Processed time step: {t + 1}/{nt}")

        del grad_out
        print(f"[{tag}] Saved: {out_path}")


if __name__ == "__main__":
    main()
