# D:\yongtae\neuron\angleoutin\code\4_plot_gradient.py

"""
기능:
- Gradient(dEx/dx, dEy/dy, dEz/dz, |grad|)를 XY/YZ/ZX 슬라이스로 인터랙티브 시각화합니다.

입출력:
- 입력: data/30V_OUT10_IN20_CI/3_gradient_1cycle.npy, 3_gradient_1cycle_2x.npy, 3_gradient_1cycle_10x.npy, 1_E_field_grid_coords.npy, 0_grid_time_spec.json
- 출력: 화면 표시(파일 저장 없음)

실행 방법:
- python 4_plot_gradient.py
"""

from __future__ import annotations

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons
from matplotlib.path import Path as MplPath
from scipy.ndimage import gaussian_filter

# --- 1. 데이터 로드 (메모리 매핑) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CASE_NAME = os.environ.get("ANGLEOUTIN_CASE", "30V_OUT10_IN20_CI")
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data", CASE_NAME)
SPEC_PATH = os.path.join(DATA_DIR, "0_grid_time_spec.json")
C_PATH = os.path.join(DATA_DIR, "1_E_field_grid_coords.npy")

G_PATHS = {
    "1x": os.path.join(DATA_DIR, "3_gradient_1cycle.npy"),
    "2x": os.path.join(DATA_DIR, "3_gradient_1cycle_2x.npy"),
    "10x": os.path.join(DATA_DIR, "3_gradient_1cycle_10x.npy"),
}

with open(SPEC_PATH, "r", encoding="utf-8") as f:
    spec = json.load(f)

G_MAP: dict[str, np.ndarray] = {}
for scale_label, path in G_PATHS.items():
    if os.path.exists(path):
        G_MAP[scale_label] = np.load(path, mmap_mode="r")

if not G_MAP:
    raise FileNotFoundError(
        "No gradient files found. Expected at least one of: "
        + ", ".join(G_PATHS.values())
    )

DEFAULT_SCALE = "1x" if "1x" in G_MAP else next(iter(G_MAP.keys()))
SCALE_OPTIONS = [s for s in ("1x", "2x", "10x") if s in G_MAP]

G0 = G_MAP[DEFAULT_SCALE]

# shape: (3, nx, ny, nz, T)
coords = np.load(C_PATH)              # shape: (N_spatial, 3)

if not (G0.ndim == 5 and G0.shape[0] == 3):
    raise ValueError("Gradient file shape should be (3, Nx, Ny, Nz, T)")
if not (coords.ndim == 2 and coords.shape[1] == 3):
    raise ValueError("Coords shape should be (N_spatial, 3)")

nx, ny, nz, T = G0.shape[1], G0.shape[2], G0.shape[3], G0.shape[4]

for scale_label, garr in G_MAP.items():
    if not (garr.ndim == 5 and garr.shape[0] == 3):
        raise ValueError(f"Gradient file shape invalid for {scale_label}: {garr.shape}")
    if garr.shape != G0.shape:
        raise ValueError(
            "All gradient files must share same shape. "
            f"{scale_label}={garr.shape}, {DEFAULT_SCALE}={G0.shape}"
        )

# 공통 시간/공간/코일 spec 로드 (um/us)
X_MIN = float(spec["space_um"]["x"]["min"])
X_MAX = float(spec["space_um"]["x"]["max"])
Y_MIN = float(spec["space_um"]["y"]["min"])
Y_MAX = float(spec["space_um"]["y"]["max"])
Z_MIN = float(spec["space_um"]["z"]["min"])
Z_MAX = float(spec["space_um"]["z"]["max"])

GRID_SPACING_UM = float(spec["space_um"]["x"]["step"])
TIME_START_US = float(spec["time_us"]["start"])
DT_US = float(spec["time_us"]["step"])
FILES_PER_CYCLE = int(spec["data_files"]["per_cycle"])

COIL_POLYGON_XZ = np.array(spec["coil_mask_um"]["polygon_xz"], dtype=np.float64)
COIL_Y_MIN = float(spec["coil_mask_um"]["y"]["min"])
COIL_Y_MAX = float(spec["coil_mask_um"]["y"]["max"])

# 축 라벨 표시를 위해 좌표를 um로 변환
coords_um = coords * 1e6
x = coords_um[:, 0]
y = coords_um[:, 1]
z = coords_um[:, 2]

# coords에서 고유 축 생성
xu = np.unique(np.round(x, 6))
yu = np.unique(np.round(y, 6))
zu = np.unique(np.round(z, 6))

xu.sort()
yu.sort()
zu.sort()

if len(xu) != nx or len(yu) != ny or len(zu) != nz:
    raise ValueError(
        "Gradient grid shape and coordinate-derived axis lengths do not match:\n"
        f"grad=(nx={nx}, ny={ny}, nz={nz}), "
        f"coords=(nx={len(xu)}, ny={len(yu)}, nz={len(zu)})"
    )

def _axis_matches_spec(axis_name: str, values: np.ndarray, vmin: float, vmax: float, step: float, tol: float = 1e-3) -> None:
    if values.size < 2:
        raise ValueError(f"{axis_name} axis has too few points: {values.size}")
    actual_min = float(values[0])
    actual_max = float(values[-1])
    actual_step = float(np.median(np.diff(values)))
    if abs(actual_min - vmin) > tol or abs(actual_max - vmax) > tol or abs(actual_step - step) > tol:
        raise ValueError(
            f"{axis_name} axis mismatch. "
            f"actual(min,max,step)=({actual_min:.3f}, {actual_max:.3f}, {actual_step:.3f}), "
            f"spec=({vmin:.3f}, {vmax:.3f}, {step:.3f})"
        )


_axis_matches_spec("X", xu, X_MIN, X_MAX, GRID_SPACING_UM)
_axis_matches_spec("Y", yu, Y_MIN, Y_MAX, GRID_SPACING_UM)
_axis_matches_spec("Z", zu, Z_MIN, Z_MAX, GRID_SPACING_UM)
if T != FILES_PER_CYCLE:
    raise ValueError(f"Time-step mismatch: data T={T}, spec per_cycle={FILES_PER_CYCLE}")


# --- 2. 데이터 처리 도우미 ---
def get_volume_at_t(t: int, mode: str = "mag", scale: str = DEFAULT_SCALE) -> np.ndarray:
    """
    mode: 'mag' or 'dExdx' or 'dEydy' or 'dEzdz'
    return: volume (nx, ny, nz)
    """
    G = G_MAP[scale]
    if mode == "dExdx":
        vol = G[0, :, :, :, t]
    elif mode == "dEydy":
        vol = G[1, :, :, :, t]
    elif mode == "dEzdz":
        vol = G[2, :, :, :, t]
    elif mode == "mag":
        gx = G[0, :, :, :, t]
        gy = G[1, :, :, :, t]
        gz = G[2, :, :, :, t]
        vol = np.sqrt(gx * gx + gy * gy + gz * gz)
    else:
        raise ValueError("mode must be one of: mag, dExdx, dEydy, dEzdz")
    return np.asarray(vol, dtype=np.float32)


def _coil_mask_3d() -> np.ndarray:
    """True inside the pentagonal-prism coil region from spec."""
    pentagon = MplPath(COIL_POLYGON_XZ)
    xx, zz = np.meshgrid(xu, zu, indexing="ij")
    xz_points = np.column_stack([xx.ravel(), zz.ravel()])
    xz_inside = pentagon.contains_points(xz_points, radius=1e-9).reshape(nx, nz)
    y_in = (yu >= COIL_Y_MIN) & (yu <= COIL_Y_MAX)
    return xz_inside[:, None, :] & y_in[None, :, None]


def _build_surface_shell_mask(coil_mask: np.ndarray) -> np.ndarray:
    """Return 6-neighbor one-voxel shell just outside coil mask."""
    padded = np.pad(coil_mask, 1, mode="constant", constant_values=False)
    neighbor_of_coil = (
        padded[2:, 1:-1, 1:-1]
        | padded[:-2, 1:-1, 1:-1]
        | padded[1:-1, 2:, 1:-1]
        | padded[1:-1, :-2, 1:-1]
        | padded[1:-1, 1:-1, 2:]
        | padded[1:-1, 1:-1, :-2]
    )
    return neighbor_of_coil & (~coil_mask)


COIL_MASK_3D = _coil_mask_3d()
SHELL_MASK_3D = _build_surface_shell_mask(COIL_MASK_3D)
SUPPRESS_MASK_3D = COIL_MASK_3D | SHELL_MASK_3D


def estimate_vmin_vmax(mode: str = "mag", scale: str = DEFAULT_SCALE) -> tuple[float, float]:
    """Compute robust color range from 0~0.1 ms, excluding coil and shell."""
    t_indices = [t for t in [0, 1, 2] if t < T]

    vals = []
    for t in t_indices:
        vol = get_volume_at_t(t, mode=mode, scale=scale)
        outside = vol[~SUPPRESS_MASK_3D]
        finite = outside[np.isfinite(outside)]
        if finite.size > 0:
            vals.append(finite)

    if not vals:
        return 0.0, 1.0

    combined = np.concatenate(vals)
    combined = combined[np.isfinite(combined)]
    if combined.size == 0:
        return 0.0, 1.0

    # 이상치 영향 완화를 위해 분위수 기반 범위 사용
    if mode == "mag":
        vmin = 0.0
        vmax = float(np.quantile(combined, 0.999))
        if vmax <= 0:
            vmax = float(np.max(combined))
        if vmax <= 0:
            vmax = 1e-6
    else:
        abs_vals = np.abs(combined)
        abs_max = float(np.quantile(abs_vals, 0.999))
        if abs_max <= 0:
            abs_max = float(np.max(abs_vals))
        if abs_max <= 0:
            abs_max = 1e-6
        vmin, vmax = -abs_max, abs_max

    return vmin, vmax


# --- 3. 인터랙티브 뷰어 ---
fig = plt.figure(figsize=(16, 6))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.15], hspace=0.3, wspace=0.5)

ax_xy = fig.add_subplot(gs[0, 0])
ax_yz = fig.add_subplot(gs[0, 1])
ax_zx = fig.add_subplot(gs[0, 2])
ax_ctrl = fig.add_subplot(gs[0, 3])

cmap = plt.colormaps["RdBu_r"].copy()
cmap.set_bad(color="black")

init_vol = get_volume_at_t(0, mode="mag", scale=DEFAULT_SCALE)
im_xy = ax_xy.imshow(init_vol[:, :, nz // 2].T, origin="lower", aspect="equal", cmap=cmap)
im_yz = ax_yz.imshow(init_vol[nx // 2, :, :].T, origin="lower", aspect="equal", cmap=cmap)
im_zx = ax_zx.imshow(init_vol[:, ny // 2, :].T, origin="lower", aspect="equal", cmap=cmap)

ax_xy.set_title("XY slice (z fixed)")
ax_yz.set_title("YZ slice (x fixed)")
ax_zx.set_title("ZX slice (y fixed)")

ax_xy.set_xlabel("x (um)")
ax_xy.set_ylabel("y (um)")
ax_yz.set_xlabel("y (um)")
ax_yz.set_ylabel("z (um)")
ax_zx.set_xlabel("x (um)")
ax_zx.set_ylabel("z (um)")

ax_xy.set_xlim(0, nx - 1)
ax_xy.set_ylim(0, ny - 1)
ax_yz.set_xlim(0, ny - 1)
ax_yz.set_ylim(0, nz - 1)
ax_zx.set_xlim(0, nx - 1)
ax_zx.set_ylim(0, nz - 1)

cbar = fig.colorbar(im_xy, ax=[ax_xy, ax_yz, ax_zx], shrink=0.9)
cbar.set_label("Gradient (unit per meter)")

ax_ctrl.axis("off")

ax_t = fig.add_subplot(gs[1, 0])
ax_x = fig.add_subplot(gs[1, 1])
ax_y = fig.add_subplot(gs[1, 2])
ax_z = fig.add_subplot(gs[1, 3])

time_end_us = TIME_START_US + (T - 1) * DT_US
t_sl = Slider(ax_t, "t (us)", TIME_START_US, time_end_us, valinit=TIME_START_US, valstep=DT_US)
x_sl = Slider(ax_x, "x (um)", xu[0], xu[-1], valinit=0.0, valstep=GRID_SPACING_UM)
y_sl = Slider(ax_y, "y (um)", yu[0], yu[-1], valinit=0.0, valstep=GRID_SPACING_UM)
z_sl = Slider(ax_z, "z (um)", zu[0], zu[-1], valinit=-450.0, valstep=GRID_SPACING_UM)

rax = fig.add_axes([0.75, 0.7, 0.18, 0.17])
mode_buttons = RadioButtons(rax, ("mag", "dExdx", "dEydy", "dEzdz"), active=0)

sax = fig.add_axes([0.75, 0.56, 0.18, 0.12])
scale_buttons = RadioButtons(sax, SCALE_OPTIONS, active=SCALE_OPTIONS.index(DEFAULT_SCALE))

cax = fig.add_axes([0.75, 0.42, 0.15, 0.12])
auto_scale_check = CheckButtons(cax, ["Auto scale", "Smoothing"], [False, False])

global_scale_cache: dict[str, tuple[float, float]] = {}


def update_view(_=None) -> None:
    mode = mode_buttons.value_selected
    scale = scale_buttons.value_selected

    t_us = float(t_sl.val)
    t = int(round((t_us - TIME_START_US) / DT_US))
    t = max(0, min(T - 1, t))

    x_um = float(x_sl.val)
    y_um = float(y_sl.val)
    z_um = float(z_sl.val)
    xi = int(np.argmin(np.abs(xu - x_um)))
    yi = int(np.argmin(np.abs(yu - y_um)))
    zi = int(np.argmin(np.abs(zu - z_um)))

    vol = get_volume_at_t(t, mode=mode, scale=scale)

    smoothing_enabled = auto_scale_check.get_status()[1]
    if smoothing_enabled:
        vol = gaussian_filter(vol, sigma=1.0, mode="nearest")

    slice_xy = vol[:, :, zi].copy()
    slice_yz = vol[xi, :, :].copy()
    slice_zx = vol[:, yi, :].copy()

    # 스무딩 on/off와 무관하게 coil + shell을 항상 마스킹
    slice_xy[SUPPRESS_MASK_3D[:, :, zi]] = np.nan
    slice_yz[SUPPRESS_MASK_3D[xi, :, :]] = np.nan
    slice_zx[SUPPRESS_MASK_3D[:, yi, :]] = np.nan

    def safe_minmax(arr: np.ndarray) -> tuple[float, float]:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.nan, np.nan
        return float(np.min(finite)), float(np.max(finite))

    auto_scale_enabled = auto_scale_check.get_status()[0]
    if auto_scale_enabled:
        mn_xy, mx_xy = safe_minmax(slice_xy)
        mn_yz, mx_yz = safe_minmax(slice_yz)
        mn_zx, mx_zx = safe_minmax(slice_zx)
        vmin = float(np.nanmin([mn_xy, mn_yz, mn_zx]))
        vmax = float(np.nanmax([mx_xy, mx_yz, mx_zx]))
        if mode == "mag":
            vmin = 0.0
        else:
            abs_max = max(abs(vmin), abs(vmax)) if np.isfinite(vmin) and np.isfinite(vmax) else 1e-6
            vmin, vmax = -abs_max, abs_max
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = 0.0, 1.0
    else:
        key = f"{scale}:{mode}"
        if key not in global_scale_cache:
            global_scale_cache[key] = estimate_vmin_vmax(mode=mode, scale=scale)
        vmin, vmax = global_scale_cache[key]

    im_xy.set_data(slice_xy.T)
    im_yz.set_data(slice_yz.T)
    im_zx.set_data(slice_zx.T)
    for im in (im_xy, im_yz, im_zx):
        im.set_clim(vmin, vmax)

    n_ticks = 5
    for ax, coord_array, x_or_y in [
        (ax_xy, xu, "x"), (ax_xy, yu, "y"),
        (ax_yz, yu, "x"), (ax_yz, zu, "y"),
        (ax_zx, xu, "x"), (ax_zx, zu, "y"),
    ]:
        ticks_idx = np.linspace(0, len(coord_array) - 1, n_ticks, dtype=int)
        labels = [f"{coord_array[i]:.0f}" for i in ticks_idx]
        if x_or_y == "x":
            ax.set_xticks(ticks_idx)
            ax.set_xticklabels(labels)
        else:
            ax.set_yticks(ticks_idx)
            ax.set_yticklabels(labels)

    mode_label = {
        "mag": "|grad|",
        "dExdx": "dEx/dx",
        "dEydy": "dEy/dy",
        "dEzdz": "dEz/dz",
    }[mode]
    ax_xy.set_title(f"XY: {mode_label} ({scale}), z={zu[zi]:.0f}um, t={t_us:.0f}us")
    ax_yz.set_title(f"YZ: {mode_label} ({scale}), x={xu[xi]:.0f}um, t={t_us:.0f}us")
    ax_zx.set_title(f"ZX: {mode_label} ({scale}), y={yu[yi]:.0f}um, t={t_us:.0f}us")

    cbar.update_normal(im_xy)
    fig.canvas.draw_idle()


t_sl.on_changed(update_view)
x_sl.on_changed(update_view)
y_sl.on_changed(update_view)
z_sl.on_changed(update_view)
mode_buttons.on_clicked(update_view)
scale_buttons.on_clicked(update_view)
auto_scale_check.on_clicked(update_view)

update_view()
plt.show()
