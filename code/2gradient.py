# D:\yongtae\neuron\code\2gradient.py
"""
1extract_output.npz에서 E-field를 읽어 directional gradient 파일을 만들고,
이미 gradient 파일이 있으면 그것을 로드해서 바로 플롯을 보여주는 스크립트입니다.

입력:
- D:\yongtae\neuron\efield\30V_OUT10_IN20_CI\1extract_output.npz

출력:
- D:\yongtae\neuron\efield\30V_OUT10_IN20_CI\2gradient.npy
- D:\yongtae\neuron\efield\30V_OUT10_IN20_CI\2gradient.npz

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
- x-z 평면 오각형: (-85,382),(0,-530),(85,-382),(85,max),(-85,max)
- y 범위: [-21.25, 26.75] um
- 5 um 격자에 안 맞는 경계는 "코일 바깥 방향" 기준으로 정렬합니다.
- 경계가 격자 좌표에 걸치면 바깥쪽 좌표도 보수적으로 포함합니다.

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
from scipy.ndimage import gaussian_filter
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons


# =========================
# 사용자 설정
# =========================
BASE_DIR = Path(r"D:\yongtae\neuron\efield\30V_OUT10_IN20_CI")

INPUT_NPZ = BASE_DIR / "1extract_output.npz"
OUTPUT_NPY = BASE_DIR / "2gradient.npy"
OUTPUT_NPZ = BASE_DIR / "2gradient.npz"

DTYPE = np.float32
GRID_STEP_UM = 5.0

# 새 코일 마스크 형상 (x-z 오각형 + y 두께)
# (x,z)=(-85,-382),(0,-530),(85,-382),(85,max),(-85,max)
COIL_POLY_RAW_XZ_UM = np.array([
    [-85.0, -382.0],
    [0.0, -530.0],
    [85.0, -382.0],
], dtype=np.float64)

COIL_Y_RAW_UM = (-21.25, 26.75)

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


def points_in_polygon(x: np.ndarray, z: np.ndarray, poly_xz: np.ndarray) -> np.ndarray:
    """
    Ray casting 기반 point-in-polygon (경계 포함).
    x, z는 동일 shape ndarray.
    """
    inside = np.zeros(x.shape, dtype=bool)
    boundary = np.zeros(x.shape, dtype=bool)

    px = poly_xz[:, 0]
    pz = poly_xz[:, 1]
    n = len(poly_xz)
    eps = 1e-12

    for i in range(n):
        j = (i + 1) % n
        x1, z1 = float(px[i]), float(pz[i])
        x2, z2 = float(px[j]), float(pz[j])

        dx = x2 - x1
        dz = z2 - z1

        cross = (x - x1) * dz - (z - z1) * dx
        on_line = np.abs(cross) <= eps
        within_x = ((x >= min(x1, x2) - eps) & (x <= max(x1, x2) + eps))
        within_z = ((z >= min(z1, z2) - eps) & (z <= max(z1, z2) + eps))
        boundary |= on_line & within_x & within_z

        cond = ((z1 > z) != (z2 > z))
        xinters = np.where(np.abs(z2 - z1) > eps, (x2 - x1) * (z - z1) / (z2 - z1) + x1, x1)
        inside ^= cond & (x < xinters)

    return inside | boundary


def build_coil_polygon_xz(z_max_um: float) -> np.ndarray:
    raw = np.vstack([
        COIL_POLY_RAW_XZ_UM,
        np.array([[85.0, z_max_um], [-85.0, z_max_um]], dtype=np.float64),
    ])
    return outward_round_points_xz(raw, GRID_STEP_UM)


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
    오각형 x-z 단면을 y 방향으로 extrude 한 코일 마스크 생성.
    격자 경계에 좌표가 걸칠 때 바깥쪽도 포함되도록,
    x-z 포인트를 반 스텝 보수 확장한 좌표에서 판정합니다.
    """
    z_max_um = float(np.max(z_um))
    poly_xz = build_coil_polygon_xz(z_max_um)
    y_lo, y_hi = outward_round_interval_um(COIL_Y_RAW_UM, GRID_STEP_UM)

    nx, ny, nz = len(x_um), len(y_um), len(z_um)
    coil_mask = np.zeros((nx, ny, nz), dtype=bool)

    y_coil_mask = (y_um >= y_lo) & (y_um <= y_hi)

    xx, zz = np.meshgrid(x_um, z_um, indexing="ij")

    # 경계가 voxel 사이에 걸치면 바깥쪽 좌표도 포함되도록 half-step 오프셋까지 검사
    h = GRID_STEP_UM * 0.5
    inside_any = (
        points_in_polygon(xx, zz, poly_xz)
        | points_in_polygon(xx + h, zz, poly_xz)
        | points_in_polygon(xx - h, zz, poly_xz)
        | points_in_polygon(xx, zz + h, poly_xz)
        | points_in_polygon(xx, zz - h, poly_xz)
    )

    coil_mask[inside_any[:, None, :] & y_coil_mask[None, :, None]] = True

    meta = {
        "poly_um": poly_xz,
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
    print("Computing: dEx/dx, dEy/dy, dEz/dz (edge_order=2), without mask suppression")

    G = np.empty((T, NX, NY, NZ, 3), dtype=DTYPE)

    for t in range(T):
        ex_grid = E[t, :, :, :, 0]
        ey_grid = E[t, :, :, :, 1]
        ez_grid = E[t, :, :, :, 2]

        dExdx = np.gradient(ex_grid, x_um, axis=0, edge_order=2).astype(DTYPE, copy=False)
        dEydy = np.gradient(ey_grid, y_um, axis=1, edge_order=2).astype(DTYPE, copy=False)
        dEzdz = np.gradient(ez_grid, z_um, axis=2, edge_order=2).astype(DTYPE, copy=False)

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
        component_order=np.array(GRAD_COMPONENT_ORDER),
        coil_poly_um=coil_meta["poly_um"].astype(np.float32),
        coil_left_poly_um=coil_meta["poly_um"].astype(np.float32),
        coil_right_poly_um=coil_meta["poly_um"].astype(np.float32),
        coil_y_range_um=coil_meta["y_range_um"].astype(np.float32),
        note=np.array([
            "G[...,0]=dEx/dx",
            "G[...,1]=dEy/dy",
            "G[...,2]=dEz/dz",
            "coil boundary uses outward rounding to 5um grid",
            "coil x-z polygon: (-85,-382),(0,-530),(85,-382),(85,max),(-85,max)",
            "if boundary crosses voxel coordinates, outside side is included conservatively",
            "gradient computation is not masked; mask is applied in plot only",
            "shell masking disabled",
        ], dtype=object),
    )

    print(f"\nSaved: {OUTPUT_NPY}")
    print(f"Saved: {OUTPUT_NPZ}")
    print(f"G shape: {G.shape}  (T, X, Y, Z, 3)")
    return G, divE, x_um, y_um, z_um, t_us, coil_mask


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

    # 저장된 마스크가 있어도 현재 코드의 코일 정의를 항상 우선 적용
    coil_mask, _ = build_coil_mask(x_um, y_um, z_um)
    hide_mask = coil_mask

    print(f"G shape: {G.shape}  (T, X, Y, Z, 3)")
    return G, divE, x_um, y_um, z_um, t_us, hide_mask


def masked_copy_for_plot(arr3d: np.ndarray, hide_mask: np.ndarray) -> np.ndarray:
    out = arr3d.astype(np.float32, copy=True)
    out[hide_mask] = np.nan
    return out


def set_axes_square(axes):
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        try:
            ax.set_box_aspect(1)
        except Exception:
            pass


def plot_cmap():
    cmap = plt.cm.bwr_r.copy()
    cmap.set_bad(color="black")
    return cmap


def make_time_slider_data(t_us: np.ndarray):
    """
    시간 슬라이더가 전체 시간축을 사용하도록 설정.
    """
    visible = np.arange(len(t_us))

    t_visible = t_us[visible]
    t_min = float(t_visible[0])
    t_max = float(t_visible[-1])

    if len(t_visible) >= 2:
        step = float(np.min(np.diff(t_visible)))
    else:
        step = 1.0

    return visible, t_min, t_max, step


def nearest_visible_time_index(slider_time_us: float, visible_idx: np.ndarray, t_us: np.ndarray) -> int:
    visible_times = t_us[visible_idx]
    j = int(np.argmin(np.abs(visible_times - slider_time_us)))
    return int(visible_idx[j])


def initial_time_index_50us(visible_idx: np.ndarray, t_us: np.ndarray) -> int:
    target_us = 50.0
    return nearest_visible_time_index(target_us, visible_idx, t_us)


def nearest_axis_index(target_value: float, axis_values: np.ndarray) -> int:
    return int(np.argmin(np.abs(axis_values - target_value)))


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


def plot_interactive(G, divE, x_um, y_um, z_um, t_us, hide_mask):
    T, NX, NY, NZ, _ = G.shape

    visible_idx, t_slider_min, t_slider_max, t_slider_step = make_time_slider_data(t_us)

    t_idx0 = initial_time_index_50us(visible_idx, t_us)
    x_idx0 = NX // 2
    y_idx0 = nearest_axis_index(35.0, y_um)
    z_idx0 = NZ // 2
    mode0 = "dEz/dz"

    modes = ("dEx/dx", "dEy/dy", "dEz/dz", "div(E)")
    cmap0 = plot_cmap()

    mode_arrays = {m: get_mode_volume(G, divE, m) for m in modes}
    vmins = {}
    vmaxs = {}
    for m in modes:
        V_plot = np.where(hide_mask[None, ...], np.nan, mode_arrays[m])
        vmins[m] = float(np.nanmin(V_plot))
        vmaxs[m] = float(np.nanmax(V_plot))
        if np.isclose(vmins[m], vmaxs[m]):
            vmaxs[m] = vmins[m] + 1e-12

    # Cache filtered frames to avoid recomputing gaussian on every slider move.
    frame_cache: dict[tuple[str, int, bool], np.ndarray] = {}

    def get_frame(mode: str, t_idx: int, gaussian_enabled: bool) -> np.ndarray:
        key = (mode, t_idx, gaussian_enabled)
        if key in frame_cache:
            return frame_cache[key]

        vol = mode_arrays[mode][t_idx].astype(np.float32, copy=False)
        if gaussian_enabled:
            outside = (~hide_mask).astype(np.float32, copy=False)
            weighted = gaussian_filter(vol * outside, sigma=1.0, mode="nearest")
            weights = gaussian_filter(outside, sigma=1.0, mode="nearest")
            with np.errstate(invalid="ignore", divide="ignore"):
                vol = np.where(weights > 0.0, weighted / weights, np.nan).astype(np.float32, copy=False)

        vol = masked_copy_for_plot(vol, hide_mask)
        frame_cache[key] = vol
        if len(frame_cache) > 12:
            frame_cache.pop(next(iter(frame_cache)))
        return vol

    gaussian0 = False
    V0 = get_frame(mode0, t_idx0, gaussian0)
    xy0 = V0[:, :, z_idx0].T
    xz0 = V0[:, y_idx0, :].T
    yz0 = V0[x_idx0, :, :].T

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
        cmap=cmap0,
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
        cmap=cmap0,
    )
    set_axes_square((ax_xy, ax_xz, ax_yz))
    ax_xz.set_xlabel("x (um)")
    ax_xz.set_ylabel("z (um)")

    img_yz = ax_yz.imshow(
        yz0,
        origin="lower",
        aspect="auto",
        extent=[y_um[0], y_um[-1], z_um[0], z_um[-1]],
        vmin=vmins[mode0],
        vmax=vmaxs[mode0],
        cmap=cmap0,
    )
    ax_yz.set_xlabel("y (um)")
    ax_yz.set_ylabel("z (um)")

    cbar = fig.colorbar(img_xy, ax=axes.ravel().tolist(), shrink=0.95)
    cbar.set_label(mode0)

    title = fig.suptitle("", fontsize=12)

    ax_radio = plt.axes([0.90, 0.58, 0.08, 0.18])
    radio = RadioButtons(ax_radio, modes, active=2)

    ax_gauss = plt.axes([0.90, 0.34, 0.08, 0.08])
    check_gauss = CheckButtons(ax_gauss, ["Gaussian Blur"], [False])

    ax_t = plt.axes([0.12, 0.20, 0.70, 0.03])
    ax_x = plt.axes([0.12, 0.15, 0.70, 0.03])
    ax_y = plt.axes([0.12, 0.10, 0.70, 0.03])
    ax_z = plt.axes([0.12, 0.05, 0.70, 0.03])

    s_t = Slider(ax_t, "t (us)", t_slider_min, t_slider_max, valinit=float(t_us[t_idx0]), valstep=t_slider_step)
    s_x = Slider(ax_x, "x idx", 0, NX - 1, valinit=x_idx0, valstep=1)
    s_y = Slider(ax_y, "y idx", 0, NY - 1, valinit=y_idx0, valstep=1)
    s_z = Slider(ax_z, "z idx", 0, NZ - 1, valinit=z_idx0, valstep=1)

    ax_reset = plt.axes([0.90, 0.24, 0.08, 0.06])
    btn_reset = Button(ax_reset, "Reset")

    def update(_=None):
        mode = radio.value_selected
        V = mode_arrays[mode]

        t_idx = nearest_visible_time_index(float(s_t.val), visible_idx, t_us)
        x_idx = int(s_x.val)
        y_idx = int(s_y.val)
        z_idx = int(s_z.val)
        gaussian_enabled = bool(check_gauss.get_status()[0])

        Vm = get_frame(mode, t_idx, gaussian_enabled)
        xy = Vm[:, :, z_idx].T
        xz = Vm[:, y_idx, :].T
        yz = Vm[x_idx, :, :].T

        img_xy.set_data(xy)
        img_xz.set_data(xz)
        img_yz.set_data(yz)
        cmap = plot_cmap()
        img_xy.set_cmap(cmap)
        img_xz.set_cmap(cmap)
        img_yz.set_cmap(cmap)

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
            f"gaussian={'on' if gaussian_enabled else 'off'}, "
            f"t={t_us[t_idx]:.1f} us, "
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
        if check_gauss.get_status()[0]:
            check_gauss.set_active(0)

    radio.on_clicked(update)
    check_gauss.on_clicked(update)
    s_t.on_changed(update)
    s_x.on_changed(update)
    s_y.on_changed(update)
    s_z.on_changed(update)
    btn_reset.on_clicked(reset)

    update()
    plt.show()


def main():
    if OUTPUT_NPY.exists() and OUTPUT_NPZ.exists():
        G, divE, x_um, y_um, z_um, t_us, hide_mask = load_existing_gradient()
    else:
        G, divE, x_um, y_um, z_um, t_us, coil_mask = build_and_save_gradient()
        hide_mask = coil_mask

    plot_interactive(G, divE, x_um, y_um, z_um, t_us, hide_mask)


if __name__ == "__main__":
    main()