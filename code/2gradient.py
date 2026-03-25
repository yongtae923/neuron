# 2gradient.py
"""
1extract_output.npz에서 E-field를 읽어 directional gradient 파일을 만들고,
이미 gradient 파일이 있으면 그것을 로드해서 바로 플롯을 보여주는 스크립트입니다.

입력:
- D:\yongtae\neuron\efield\400us_50Hz_10umspaing_100mA\1extract_output.npz

출력:
- D:\yongtae\neuron\efield\400us_50Hz_10umspaing_100mA\2gradient.npy
- D:\yongtae\neuron\efield\400us_50Hz_10umspaing_100mA\2gradient.npz

입력 배열 형식:
- E.shape = (T, X, Y, Z, 3)
- 마지막 축 순서: Ex, Ey, Ez

출력 배열 형식:
- G.shape = (T, X, Y, Z, 3)
- 마지막 축 순서:
    G[..., 0] = dEx/dx
    G[..., 1] = dEy/dy
    G[..., 2] = dEz/dz

코일 마스크:
- 기존 코드의 코일 형상을 그대로 사용합니다.
- x-z 평면의 좌우 trapezoid prism 2개
- y 범위는 [-6.25, 11.75] um 이었고,
  5 um 격자에 안 맞는 경계는 "코일 바깥 방향" 기준으로 정렬합니다.
  예:
    -6.25 -> -10
    11.75 -> 15
- 즉 실제 마스크 계산에는 outward-rounded 경계를 사용합니다.

플롯:
- matplotlib 슬라이더로 시간 t, x, y, z를 바꿀 수 있습니다.
- 표시 모드는 dEx/dx, dEy/dy, dEz/dz, div(E) 중에서 고를 수 있습니다.
- XY, XZ, YZ 슬라이스를 동시에 보여줍니다.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.widgets import Slider, Button, RadioButtons


# =========================
# 사용자 설정
# =========================
BASE_DIR = Path(r"D:\yongtae\neuron\efield\400us_50Hz_10umspaing_100mA")

INPUT_NPZ = BASE_DIR / "1extract_output.npz"
OUTPUT_NPY = BASE_DIR / "2gradient.npy"
OUTPUT_NPZ = BASE_DIR / "2gradient.npz"

DTYPE = np.float32
GRID_STEP_UM = 5.0

# 기존 코드의 코일 형상
# Right trapezoid (x-z): [(45,800), (45,590), (5,500), (5,800)] um
# Left trapezoid  (x-z): [(-45,800), (-45,590), (-5,500), (-5,800)] um
COIL_RIGHT_TRAPEZOID_XZ_UM = np.array([
    [45.0, 800.0],
    [45.0, 590.0],
    [5.0, 500.0],
    [5.0, 800.0],
], dtype=np.float64)

COIL_LEFT_TRAPEZOID_XZ_UM = np.array([
    [-45.0, 800.0],
    [-45.0, 590.0],
    [-5.0, 500.0],
    [-5.0, 800.0],
], dtype=np.float64)

# 기존 코드의 y 범위
COIL_Y_RAW_UM = (-6.25, 11.75)

GRAD_COMPONENT_ORDER = ("dExdx", "dEydy", "dEzdz")


def outward_round_scalar_um(v: float, step_um: float = GRID_STEP_UM) -> float:
    """
    코일 바깥 방향 기준 outward rounding.
    예:
      -6.25 -> -10
       11.75 ->  15
        5.00 ->   5
    """
    if v > 0:
        return math.ceil(v / step_um) * step_um
    if v < 0:
        return math.floor(v / step_um) * step_um
    return 0.0


def outward_round_points_xz(points_um: np.ndarray, step_um: float = GRID_STEP_UM) -> np.ndarray:
    out = np.empty_like(points_um, dtype=np.float64)
    for i in range(points_um.shape[0]):
        out[i, 0] = outward_round_scalar_um(float(points_um[i, 0]), step_um)
        out[i, 1] = outward_round_scalar_um(float(points_um[i, 1]), step_um)
    return out


def outward_round_interval_um(interval_um: tuple[float, float], step_um: float = GRID_STEP_UM) -> tuple[float, float]:
    lo, hi = interval_um
    return outward_round_scalar_um(lo, step_um), outward_round_scalar_um(hi, step_um)


def build_surface_shell_mask(coil_mask: np.ndarray) -> np.ndarray:
    """
    6-neighbor 기준 1-voxel outside shell.
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


def build_coil_mask(x_um: np.ndarray, y_um: np.ndarray, z_um: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    기존 코드의 코일 형상을 유지하되, 5um 격자에 맞지 않는 경계는 outward rounding 적용.
    """
    left_poly = outward_round_points_xz(COIL_LEFT_TRAPEZOID_XZ_UM, GRID_STEP_UM)
    right_poly = outward_round_points_xz(COIL_RIGHT_TRAPEZOID_XZ_UM, GRID_STEP_UM)
    y_lo, y_hi = outward_round_interval_um(COIL_Y_RAW_UM, GRID_STEP_UM)

    nx, ny, nz = len(x_um), len(y_um), len(z_um)
    coil_mask = np.zeros((nx, ny, nz), dtype=bool)

    y_coil_mask = (y_um >= y_lo) & (y_um <= y_hi)

    left_path = MplPath(left_poly)
    right_path = MplPath(right_poly)

    xx, zz = np.meshgrid(x_um, z_um, indexing="ij")
    xz_points = np.column_stack([xx.ravel(), zz.ravel()])

    left_inside = left_path.contains_points(xz_points, radius=1e-9).reshape(nx, nz)
    right_inside = right_path.contains_points(xz_points, radius=1e-9).reshape(nx, nz)
    inside_any = left_inside | right_inside

    coil_mask[inside_any[:, None, :] & y_coil_mask[None, :, None]] = True

    meta = {
        "left_poly_um": left_poly,
        "right_poly_um": right_poly,
        "y_range_um": np.array([y_lo, y_hi], dtype=np.float32),
    }
    return coil_mask, meta


def load_input_efield():
    if not INPUT_NPZ.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {INPUT_NPZ}")

    data = np.load(INPUT_NPZ, allow_pickle=True)

    if "E" not in data:
        raise KeyError(f"{INPUT_NPZ} 안에 'E' 배열이 없습니다.")

    E = data["E"]
    x_um = data["x_um"]
    y_um = data["y_um"]
    z_um = data["z_um"]
    t_us = data["t_us"]

    if E.ndim != 5 or E.shape[-1] != 3:
        raise ValueError(f"Unexpected E shape: {E.shape}, expected (T, X, Y, Z, 3)")

    if E.shape[1] != len(x_um) or E.shape[2] != len(y_um) or E.shape[3] != len(z_um):
        raise ValueError("E shape와 축 길이가 일치하지 않습니다.")

    return E, x_um, y_um, z_um, t_us


def build_and_save_gradient():
    print("[BUILD] gradient 출력 파일이 없어서 새로 생성합니다.")

    E, x_um, y_um, z_um, t_us = load_input_efield()
    T, NX, NY, NZ, C = E.shape

    print(f"E shape: {E.shape}  (T, X, Y, Z, 3)")
    print(f"Grid shape: X={NX}, Y={NY}, Z={NZ}, T={T}")

    coil_mask, coil_meta = build_coil_mask(x_um, y_um, z_um)
    shell_mask = build_surface_shell_mask(coil_mask)
    suppress_mask = coil_mask | shell_mask

    n_coil = int(np.sum(coil_mask))
    n_shell = int(np.sum(shell_mask))
    print(f"Coil interior masked to 0: {n_coil}")
    print(f"Coil 1-voxel outside shell masked to 0: {n_shell}")
    print("Computing: dEx/dx, dEy/dy, dEz/dz (edge_order=2)")

    G = np.empty((T, NX, NY, NZ, 3), dtype=DTYPE)

    for t in range(T):
        ex_grid = E[t, :, :, :, 0]
        ey_grid = E[t, :, :, :, 1]
        ez_grid = E[t, :, :, :, 2]

        dExdx = np.gradient(ex_grid, x_um, axis=0, edge_order=2).astype(DTYPE, copy=False)
        dEydy = np.gradient(ey_grid, y_um, axis=1, edge_order=2).astype(DTYPE, copy=False)
        dEzdz = np.gradient(ez_grid, z_um, axis=2, edge_order=2).astype(DTYPE, copy=False)

        dExdx[suppress_mask] = 0.0
        dEydy[suppress_mask] = 0.0
        dEzdz[suppress_mask] = 0.0

        G[t, :, :, :, 0] = dExdx
        G[t, :, :, :, 1] = dEydy
        G[t, :, :, :, 2] = dEzdz

        if (t + 1) % 10 == 0 or (t + 1) == T:
            print(f"Processed time step: {t + 1}/{T}")

    divE = (G[..., 0] + G[..., 1] + G[..., 2]).astype(DTYPE, copy=False)

    np.save(OUTPUT_NPY, G)
    np.savez_compressed(
        OUTPUT_NPZ,
        G=G,
        divE=divE,
        x_um=x_um.astype(np.float32),
        y_um=y_um.astype(np.float32),
        z_um=z_um.astype(np.float32),
        t_us=t_us.astype(np.float32),
        coil_mask=coil_mask,
        shell_mask=shell_mask,
        suppress_mask=suppress_mask,
        component_order=np.array(GRAD_COMPONENT_ORDER),
        coil_left_poly_um=coil_meta["left_poly_um"].astype(np.float32),
        coil_right_poly_um=coil_meta["right_poly_um"].astype(np.float32),
        coil_y_range_um=coil_meta["y_range_um"].astype(np.float32),
        note=np.array([
            "G[...,0]=dEx/dx",
            "G[...,1]=dEy/dy",
            "G[...,2]=dEz/dz",
            "coil boundary uses outward rounding to 5um grid",
        ], dtype=object),
    )

    print(f"\nSaved: {OUTPUT_NPY}")
    print(f"Saved: {OUTPUT_NPZ}")
    print(f"G shape: {G.shape}  (T, X, Y, Z, 3)")
    return G, divE, x_um, y_um, z_um, t_us, coil_mask, shell_mask


def load_existing_gradient():
    print("[LOAD] 기존 gradient 파일을 로드합니다.")

    if not OUTPUT_NPZ.exists():
        raise FileNotFoundError(f"출력 npz 파일이 없습니다: {OUTPUT_NPZ}")

    data = np.load(OUTPUT_NPZ, allow_pickle=True)

    G = data["G"]
    if "divE" in data:
        divE = data["divE"]
    else:
        divE = (G[..., 0] + G[..., 1] + G[..., 2]).astype(DTYPE, copy=False)

    x_um = data["x_um"]
    y_um = data["y_um"]
    z_um = data["z_um"]
    t_us = data["t_us"]

    coil_mask = data["coil_mask"] if "coil_mask" in data else None
    shell_mask = data["shell_mask"] if "shell_mask" in data else None

    print(f"G shape: {G.shape}  (T, X, Y, Z, 3)")
    return G, divE, x_um, y_um, z_um, t_us, coil_mask, shell_mask


def get_mode_volume(G: np.ndarray, divE: np.ndarray, mode: str) -> np.ndarray:
    if mode == "dEx/dx":
        return G[..., 0]
    if mode == "dEy/dy":
        return G[..., 1]
    if mode == "dEz/dz":
        return G[..., 2]
    if mode == "div(E)":
        return divE
    raise ValueError(f"Unknown mode: {mode}")


def plot_interactive(G, divE, x_um, y_um, z_um, t_us):
    T, NX, NY, NZ, _ = G.shape

    t_idx0 = 0
    x_idx0 = NX // 2
    y_idx0 = NY // 2
    z_idx0 = NZ // 2
    mode0 = "dEz/dz"

    modes = ("dEx/dx", "dEy/dy", "dEz/dz", "div(E)")

    mode_arrays = {m: get_mode_volume(G, divE, m) for m in modes}
    vmins = {}
    vmaxs = {}
    for m in modes:
        vmins[m] = float(np.nanmin(mode_arrays[m]))
        vmaxs[m] = float(np.nanmax(mode_arrays[m]))
        if np.isclose(vmins[m], vmaxs[m]):
            vmaxs[m] = vmins[m] + 1e-12

    V0 = mode_arrays[mode0]
    xy0 = V0[t_idx0, :, :, z_idx0].T
    xz0 = V0[t_idx0, :, y_idx0, :].T
    yz0 = V0[t_idx0, x_idx0, :, :].T

    fig, axes = plt.subplots(1, 3, figsize=(17, 7))
    plt.subplots_adjust(left=0.08, right=0.88, bottom=0.29, top=0.88, wspace=0.28)

    ax_xy, ax_xz, ax_yz = axes

    img_xy = ax_xy.imshow(
        xy0,
        origin="lower",
        aspect="auto",
        extent=[x_um[0], x_um[-1], y_um[0], y_um[-1]],
        vmin=vmins[mode0],
        vmax=vmaxs[mode0],
    )
    ax_xy.set_xlabel("x (um)")
    ax_xy.set_ylabel("y (um)")

    img_xz = ax_xz.imshow(
        xz0,
        origin="lower",
        aspect="auto",
        extent=[x_um[0], x_um[-1], z_um[0], z_um[-1]],
        vmin=vmins[mode0],
        vmax=vmaxs[mode0],
    )
    ax_xz.set_xlabel("x (um)")
    ax_xz.set_ylabel("z (um)")

    img_yz = ax_yz.imshow(
        yz0,
        origin="lower",
        aspect="auto",
        extent=[y_um[0], y_um[-1], z_um[0], z_um[-1]],
        vmin=vmins[mode0],
        vmax=vmaxs[mode0],
    )
    ax_yz.set_xlabel("y (um)")
    ax_yz.set_ylabel("z (um)")

    cbar = fig.colorbar(img_xy, ax=axes.ravel().tolist(), shrink=0.95)
    cbar.set_label(mode0)

    title = fig.suptitle("", fontsize=12)

    ax_radio = plt.axes([0.90, 0.58, 0.08, 0.18])
    radio = RadioButtons(ax_radio, modes, active=2)

    ax_t = plt.axes([0.12, 0.20, 0.70, 0.03])
    ax_x = plt.axes([0.12, 0.15, 0.70, 0.03])
    ax_y = plt.axes([0.12, 0.10, 0.70, 0.03])
    ax_z = plt.axes([0.12, 0.05, 0.70, 0.03])

    # restrict t slider to 0..50us
    t_max_us = 50.0
    t_max_idx = min(int(round(t_max_us / DT_US)), T - 1)

    s_t = Slider(ax_t, "t idx", 0, t_max_idx, valinit=t_idx0, valstep=1)
    s_x = Slider(ax_x, "x idx", 0, NX - 1, valinit=x_idx0, valstep=1)
    s_y = Slider(ax_y, "y idx", 0, NY - 1, valinit=y_idx0, valstep=1)
    s_z = Slider(ax_z, "z idx", 0, NZ - 1, valinit=z_idx0, valstep=1)

    ax_reset = plt.axes([0.90, 0.45, 0.08, 0.06])
    btn_reset = Button(ax_reset, "Reset")

    def update(_=None):
        mode = radio.value_selected
        V = mode_arrays[mode]

        t_idx = int(s_t.val)
        x_idx = int(s_x.val)
        y_idx = int(s_y.val)
        z_idx = int(s_z.val)

        xy = V[t_idx, :, :, z_idx].T
        xz = V[t_idx, :, y_idx, :].T
        yz = V[t_idx, x_idx, :, :].T

        img_xy.set_data(xy)
        img_xz.set_data(xz)
        img_yz.set_data(yz)

        img_xy.set_clim(vmins[mode], vmaxs[mode])
        img_xz.set_clim(vmins[mode], vmaxs[mode])
        img_yz.set_clim(vmins[mode], vmaxs[mode])
        cbar.update_normal(img_xy)
        cbar.set_label(mode)

        ax_xy.set_title(f"{mode} XY @ z={z_um[z_idx]:.1f} um")
        ax_xz.set_title(f"{mode} XZ @ y={y_um[y_idx]:.1f} um")
        ax_yz.set_title(f"{mode} YZ @ x={x_um[x_idx]:.1f} um")

        title.set_text(
            f"mode={mode}, "
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

    radio.on_clicked(update)
    s_t.on_changed(update)
    s_x.on_changed(update)
    s_y.on_changed(update)
    s_z.on_changed(update)
    btn_reset.on_clicked(reset)

    update()
    plt.show()


def main():
    if OUTPUT_NPY.exists() and OUTPUT_NPZ.exists():
        G, divE, x_um, y_um, z_um, t_us, _coil_mask, _shell_mask = load_existing_gradient()
    else:
        G, divE, x_um, y_um, z_um, t_us, _coil_mask, _shell_mask = build_and_save_gradient()

    plot_interactive(G, divE, x_um, y_um, z_um, t_us)


if __name__ == "__main__":
    main()