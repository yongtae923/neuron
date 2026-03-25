#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ANSYS Ez txt stack -> NPZ/NPY converter + interactive matplotlib viewer

기능
1) 저장 파일이 이미 있으면 그것을 로드
2) 저장 파일이 없으면 txt들을 읽어서 npz/npy로 저장
3) matplotlib 슬라이더로
   - 시간 t
   - x index
   - y index
   - z index
   를 바꾸면서 3개 평면(xy, xz, yz) 슬라이스를 색으로 표시

기본 축 정의
- 공간:
    x: -200 ~ 200 um, step 5 um
    y: -200 ~ 200 um, step 5 um
    z:  400 ~ 800 um, step 5 um
- 시간:
    txt 파일 개수대로 로드
    기본 dt = 5 us

출력
- NPZ: Ez, x_um, y_um, z_um, t_us, spacing_um
- NPY: Ez_only.npy
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


HEADER_GRID_RE = re.compile(
    r"Grid Output Min:\s*\[([^\]]+)\]\s*Max:\s*\[([^\]]+)\]\s*Grid Size:\s*\[([^\]]+)\]"
)


def parse_um_triplet(text: str) -> np.ndarray:
    parts = text.strip().split()
    vals = []
    for p in parts:
        p = p.strip()
        if p.endswith("um"):
            p = p[:-2]
        vals.append(float(p))
    if len(vals) != 3:
        raise ValueError(f"Expected 3 values, got: {text}")
    return np.array(vals, dtype=np.float64)


def parse_header(lines: list[str]):
    grid_line_idx = None
    for i, line in enumerate(lines[:20]):
        if line.startswith("Grid Output Min:"):
            grid_line_idx = i
            break
    if grid_line_idx is None:
        raise ValueError("Could not find 'Grid Output Min:' header line.")

    m = HEADER_GRID_RE.search(lines[grid_line_idx])
    if not m:
        raise ValueError(f"Failed to parse grid header:\n{lines[grid_line_idx]}")

    min_um = parse_um_triplet(m.group(1))
    max_um = parse_um_triplet(m.group(2))
    step_um = parse_um_triplet(m.group(3))
    data_start_idx = grid_line_idx + 2
    return min_um, max_um, step_um, data_start_idx


def expected_axis(min_um: float, max_um: float, step_um: float) -> np.ndarray:
    n = int(round((max_um - min_um) / step_um)) + 1
    return min_um + np.arange(n, dtype=np.float64) * step_um


def natural_key(path: Path):
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else path.stem


def load_single_txt(txt_path: Path, verbose: bool = False):
    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    min_um, max_um, step_um, data_start_idx = parse_header(lines)

    data = np.loadtxt(txt_path, skiprows=data_start_idx)
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(f"{txt_path} does not appear to contain 4 numeric columns.")

    xyz_m = data[:, :3]
    scalar = data[:, 3]
    xyz_um = xyz_m * 1e6

    x_axis = expected_axis(min_um[0], max_um[0], step_um[0])
    y_axis = expected_axis(min_um[1], max_um[1], step_um[1])
    z_axis = expected_axis(min_um[2], max_um[2], step_um[2])

    nx, ny, nz = len(x_axis), len(y_axis), len(z_axis)
    expected_points = nx * ny * nz

    if scalar.size != expected_points:
        raise ValueError(
            f"{txt_path.name}: point count mismatch. "
            f"Expected {expected_points} from header ({nx}x{ny}x{nz}), got {scalar.size}."
        )

    x_unique = np.unique(np.round(xyz_um[:, 0], 9))
    y_unique = np.unique(np.round(xyz_um[:, 1], 9))
    z_unique = np.unique(np.round(xyz_um[:, 2], 9))

    if len(x_unique) != nx or len(y_unique) != ny or len(z_unique) != nz:
        raise ValueError(
            f"{txt_path.name}: unique coordinate counts do not match header. "
            f"Unique counts = ({len(x_unique)}, {len(y_unique)}, {len(z_unique)}), "
            f"header counts = ({nx}, {ny}, {nz})."
        )

    # 샘플 구조상 z가 가장 빨리 변하고, 그다음 y, 마지막이 x
    field_xyz = scalar.reshape(nx, ny, nz)

    if verbose:
        print(f"[OK] {txt_path.name}")
        print(f"     grid shape = ({nx}, {ny}, {nz})")
        print(f"     x range um = [{x_axis[0]}, {x_axis[-1]}], step {step_um[0]}")
        print(f"     y range um = [{y_axis[0]}, {y_axis[-1]}], step {step_um[1]}")
        print(f"     z range um = [{z_axis[0]}, {z_axis[-1]}], step {step_um[2]}")

    return field_xyz, x_axis, y_axis, z_axis, step_um


def convert_folder(
    input_dir: Path,
    output_prefix: Path,
    dt_us: float = 5.0,
    glob_pattern: str = "*.txt",
    dtype: str = "float32",
    save_npy: bool = True,
    verbose: bool = True,
):
    txt_files = sorted(input_dir.glob(glob_pattern), key=natural_key)
    if not txt_files:
        raise FileNotFoundError(f"No txt files found in: {input_dir}")

    frames = []
    x_axis = y_axis = z_axis = spacing_um = None

    for i, txt_path in enumerate(txt_files):
        field_xyz, x_axis_i, y_axis_i, z_axis_i, spacing_um_i = load_single_txt(
            txt_path, verbose=(verbose and i == 0)
        )

        if i == 0:
            x_axis, y_axis, z_axis, spacing_um = x_axis_i, y_axis_i, z_axis_i, spacing_um_i
        else:
            if (
                x_axis_i.shape != x_axis.shape
                or y_axis_i.shape != y_axis.shape
                or z_axis_i.shape != z_axis.shape
                or not np.allclose(x_axis_i, x_axis)
                or not np.allclose(y_axis_i, y_axis)
                or not np.allclose(z_axis_i, z_axis)
            ):
                raise ValueError(f"Grid mismatch detected in file: {txt_path.name}")

        frames.append(field_xyz.astype(dtype, copy=False))

        if verbose and ((i + 1) % 25 == 0 or (i + 1) == len(txt_files)):
            print(f"Loaded {i + 1}/{len(txt_files)} files")

    ez = np.stack(frames, axis=0)  # (T, X, Y, Z)
    t_us = np.arange(ez.shape[0], dtype=np.float64) * dt_us

    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    npz_path = output_prefix.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        Ez=ez,
        x_um=x_axis.astype(np.float32),
        y_um=y_axis.astype(np.float32),
        z_um=z_axis.astype(np.float32),
        t_us=t_us.astype(np.float32),
        spacing_um=spacing_um.astype(np.float32),
    )

    npy_path = output_prefix.with_name(output_prefix.name + "_only.npy")
    if save_npy:
        np.save(npy_path, ez)

    print("\nSaved:")
    print(f"  NPZ : {npz_path}")
    if save_npy:
        print(f"  NPY : {npy_path}")

    print("\nSummary:")
    print(f"  Ez shape      : {ez.shape}   (T, X, Y, Z)")
    print(f"  dtype         : {ez.dtype}")
    print(f"  x range (um)  : {x_axis[0]} to {x_axis[-1]}  n={len(x_axis)}")
    print(f"  y range (um)  : {y_axis[0]} to {y_axis[-1]}  n={len(y_axis)}")
    print(f"  z range (um)  : {z_axis[0]} to {z_axis[-1]}  n={len(z_axis)}")
    print(f"  dt (us)       : {dt_us}")
    print(f"  total T       : {ez.shape[0]}")

    return npz_path, npy_path if save_npy else None


def load_or_create_dataset(
    input_dir: Path,
    output_prefix: Path,
    dt_us: float = 5.0,
    glob_pattern: str = "*.txt",
    dtype: str = "float32",
    save_npy: bool = True,
    verbose: bool = True,
):
    npz_path = output_prefix.with_suffix(".npz")

    if npz_path.exists():
        print(f"[LOAD] Existing file found: {npz_path}")
    else:
        print("[BUILD] No saved npz found. Converting txt files first...")
        convert_folder(
            input_dir=input_dir,
            output_prefix=output_prefix,
            dt_us=dt_us,
            glob_pattern=glob_pattern,
            dtype=dtype,
            save_npy=save_npy,
            verbose=verbose,
        )

    data = np.load(npz_path)
    ez = data["Ez"]
    x_um = data["x_um"]
    y_um = data["y_um"]
    z_um = data["z_um"]
    t_us = data["t_us"]
    spacing_um = data["spacing_um"]

    print("[READY] Dataset loaded.")
    print(f"        Ez shape = {ez.shape} (T, X, Y, Z)")
    return ez, x_um, y_um, z_um, t_us, spacing_um


def plot_interactive_slices(
    ez: np.ndarray,
    x_um: np.ndarray,
    y_um: np.ndarray,
    z_um: np.ndarray,
    t_us: np.ndarray,
):
    # shape: (T, X, Y, Z)
    T, NX, NY, NZ = ez.shape

    t_idx0 = 0
    x_idx0 = NX // 2
    y_idx0 = NY // 2
    z_idx0 = NZ // 2

    vmin = float(np.nanmin(ez))
    vmax = float(np.nanmax(ez))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-12

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    plt.subplots_adjust(left=0.07, right=0.97, bottom=0.28, top=0.88, wspace=0.25)

    ax_xy, ax_xz, ax_yz = axes

    # xy @ z
    img_xy = ax_xy.imshow(
        ez[t_idx0, :, :, z_idx0].T,
        origin="lower",
        aspect="auto",
        extent=[x_um[0], x_um[-1], y_um[0], y_um[-1]],
        vmin=vmin,
        vmax=vmax,
    )
    ax_xy.set_title("XY slice @ z")
    ax_xy.set_xlabel("x (um)")
    ax_xy.set_ylabel("y (um)")

    # xz @ y
    img_xz = ax_xz.imshow(
        ez[t_idx0, :, y_idx0, :].T,
        origin="lower",
        aspect="auto",
        extent=[x_um[0], x_um[-1], z_um[0], z_um[-1]],
        vmin=vmin,
        vmax=vmax,
    )
    ax_xz.set_title("XZ slice @ y")
    ax_xz.set_xlabel("x (um)")
    ax_xz.set_ylabel("z (um)")

    # yz @ x
    img_yz = ax_yz.imshow(
        ez[t_idx0, x_idx0, :, :].T,
        origin="lower",
        aspect="auto",
        extent=[y_um[0], y_um[-1], z_um[0], z_um[-1]],
        vmin=vmin,
        vmax=vmax,
    )
    ax_yz.set_title("YZ slice @ x")
    ax_yz.set_xlabel("y (um)")
    ax_yz.set_ylabel("z (um)")

    cbar = fig.colorbar(img_xy, ax=axes.ravel().tolist(), shrink=0.95)
    cbar.set_label("Ez field value")

    title = fig.suptitle("", fontsize=12)

    # slider axes
    ax_t = plt.axes([0.12, 0.18, 0.76, 0.03])
    ax_x = plt.axes([0.12, 0.13, 0.76, 0.03])
    ax_y = plt.axes([0.12, 0.08, 0.76, 0.03])
    ax_z = plt.axes([0.12, 0.03, 0.76, 0.03])

    s_t = Slider(ax_t, "t idx", 0, T - 1, valinit=t_idx0, valstep=1)
    s_x = Slider(ax_x, "x idx", 0, NX - 1, valinit=x_idx0, valstep=1)
    s_y = Slider(ax_y, "y idx", 0, NY - 1, valinit=y_idx0, valstep=1)
    s_z = Slider(ax_z, "z idx", 0, NZ - 1, valinit=z_idx0, valstep=1)

    # reset button
    ax_reset = plt.axes([0.01, 0.03, 0.08, 0.08])
    btn_reset = Button(ax_reset, "Reset")

    def update(_=None):
        t_idx = int(s_t.val)
        x_idx = int(s_x.val)
        y_idx = int(s_y.val)
        z_idx = int(s_z.val)

        arr_xy = ez[t_idx, :, :, z_idx].T
        arr_xz = ez[t_idx, :, y_idx, :].T
        arr_yz = ez[t_idx, x_idx, :, :].T

        img_xy.set_data(arr_xy)
        img_xy.set_extent([x_um[0], x_um[-1], y_um[0], y_um[-1]])

        img_xz.set_data(arr_xz)
        img_xz.set_extent([x_um[0], x_um[-1], z_um[0], z_um[-1]])

        img_yz.set_data(arr_yz)
        img_yz.set_extent([y_um[0], y_um[-1], z_um[0], z_um[-1]])

        ax_xy.set_title(f"XY slice @ z={z_um[z_idx]:.1f} um")
        ax_xz.set_title(f"XZ slice @ y={y_um[y_idx]:.1f} um")
        ax_yz.set_title(f"YZ slice @ x={x_um[x_idx]:.1f} um")

        title.set_text(
            f"t={t_idx} ({t_us[t_idx]:.1f} us), "
            f"x={x_idx} ({x_um[x_idx]:.1f} um), "
            f"y={y_idx} ({y_um[y_idx]:.1f} um), "
            f"z={z_idx} ({z_um[z_idx]:.1f} um)"
        )

        fig.canvas.draw_idle()

    def reset(_):
        s_t.reset()
        s_x.reset()
        s_y.reset()
        s_z.reset()

    s_t.on_changed(update)
    s_x.on_changed(update)
    s_y.on_changed(update)
    s_z.on_changed(update)
    btn_reset.on_clicked(reset)

    update()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Convert ANSYS Ez txt stack to npz/npy and view slices interactively."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(r"D:\yongtae\neuron\efield\400us_50Hz_10umspaing_100mA\400us_50Hz_10umspaing_100mA_Ez"),
        help="Folder containing 001.txt, 002.txt, ...",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path(r"D:\yongtae\neuron\util\400us_50Hz_10umspaing_100mA_Ez"),
        help="Output prefix without extension. Example: D:\\...\\Ez_data",
    )
    parser.add_argument(
        "--dt-us",
        type=float,
        default=5.0,
        help="Time step between consecutive txt files in microseconds.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.txt",
        help="Glob pattern for input files.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Storage dtype for Ez tensor.",
    )
    parser.add_argument(
        "--no-npy",
        action="store_true",
        help="Do not save the extra *_only.npy file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console output.",
    )

    args = parser.parse_args()

    ez, x_um, y_um, z_um, t_us, spacing_um = load_or_create_dataset(
        input_dir=args.input_dir,
        output_prefix=args.output_prefix,
        dt_us=args.dt_us,
        glob_pattern=args.glob,
        dtype=args.dtype,
        save_npy=not args.no_npy,
        verbose=not args.quiet,
    )

    print(f"spacing_um = {spacing_um}")
    plot_interactive_slices(ez, x_um, y_um, z_um, t_us)


if __name__ == "__main__":
    main()