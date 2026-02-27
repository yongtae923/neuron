"""
gradient_plot.py

Interactive 3-slice viewer for directional gradient data:
- dEx/dx
- dEy/dy
- dEz/dz
- magnitude: sqrt((dEx/dx)^2 + (dEy/dy)^2 + (dEz/dz)^2)

Data source:
- data/gradient/grad_Exdx_Eydy_Ezdz_1cycle.npy  (shape: 3, Nx, Ny, Nz, T)
- efield/E_field_grid_coords.npy                (shape: N_spatial, 3; meters)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons
from scipy.ndimage import gaussian_filter

# =========================
# 1) Load data (memory-mapped)
# =========================
G_PATH = "data/gradient/grad_Exdx_Eydy_Ezdz_1cycle.npy"
C_PATH = "efield/E_field_grid_coords.npy"

G = np.load(G_PATH, mmap_mode="r")    # shape: (3, nx, ny, nz, T)
coords = np.load(C_PATH)              # shape: (N_spatial, 3)

if not (G.ndim == 5 and G.shape[0] == 3):
    raise ValueError("Gradient file shape should be (3, Nx, Ny, Nz, T)")
if not (coords.ndim == 2 and coords.shape[1] == 3):
    raise ValueError("Coords shape should be (N_spatial, 3)")

nx, ny, nz, T = G.shape[1], G.shape[2], G.shape[3], G.shape[4]

# Time step from existing project convention
DT_MS = 0.05

# Convert to um for axis labels
coords_um = coords * 1e6
x = coords_um[:, 0]
y = coords_um[:, 1]
z = coords_um[:, 2]

# Build unique axes from coords
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

# Grid spacing in um (for slider step)
GRID_SPACING_UM = 5.0

# Coil region [um]: pentagonal prism
# Face (y=-32): (-79.5,561), (0,498), (79.5,561), (79.5,1502), (-79.5,1502) in (x,z)
# Other face at y=+32 with same (x,z)
COIL_Y_MIN, COIL_Y_MAX = -32.0, 32.0
COIL_Z_TIP_MIN = 498.0
COIL_Z_SHOULDER = 561.0
COIL_Z_MAX = 1502.0
COIL_X_HALF_MAX = 79.5

_zz_axis = zu.copy()
_z_in_axis = (_zz_axis >= COIL_Z_TIP_MIN) & (_zz_axis <= COIL_Z_MAX)
_x_limit_axis = np.where(
    _zz_axis <= COIL_Z_SHOULDER,
    COIL_X_HALF_MAX * (_zz_axis - COIL_Z_TIP_MIN) / (COIL_Z_SHOULDER - COIL_Z_TIP_MIN),
    COIL_X_HALF_MAX,
)
_x_limit_axis = np.where(_z_in_axis, _x_limit_axis, -1.0)
_coil_xz_inside = np.abs(xu)[:, None] <= _x_limit_axis[None, :]


# =========================
# 2) Data helpers
# =========================
def get_volume_at_t(t: int, mode: str = "mag") -> np.ndarray:
    """
    mode: 'mag' or 'dExdx' or 'dEydy' or 'dEzdz'
    return: volume (nx, ny, nz)
    """
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
    """True inside the pentagonal-prism coil region."""
    y_in = (yu >= COIL_Y_MIN) & (yu <= COIL_Y_MAX)
    return _coil_xz_inside[:, None, :] & y_in[None, :, None]


def estimate_vmin_vmax(mode: str = "mag") -> tuple[float, float]:
    """Compute robust color range from 0~0.1 ms, excluding coil region."""
    t_indices = [t for t in [0, 1, 2] if t < T]
    coil_mask = _coil_mask_3d()

    vals = []
    for t in t_indices:
        vol = get_volume_at_t(t, mode=mode)
        outside = vol[~coil_mask]
        finite = outside[np.isfinite(outside)]
        if finite.size > 0:
            vals.append(finite)

    if not vals:
        return 0.0, 1.0

    combined = np.concatenate(vals)
    combined = combined[np.isfinite(combined)]
    if combined.size == 0:
        return 0.0, 1.0

    # Use robust percentile range to avoid outlier-dominated scaling.
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


# =========================
# 3) Interactive viewer
# =========================
fig = plt.figure(figsize=(16, 6))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.15], hspace=0.3, wspace=0.5)

ax_xy = fig.add_subplot(gs[0, 0])
ax_yz = fig.add_subplot(gs[0, 1])
ax_zx = fig.add_subplot(gs[0, 2])
ax_ctrl = fig.add_subplot(gs[0, 3])

cmap = plt.colormaps["RdBu_r"].copy()
cmap.set_bad(color="black")

init_vol = get_volume_at_t(0, mode="mag")
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

t_sl = Slider(ax_t, "t (ms)", 0.0, 1.0, valinit=0.05, valstep=DT_MS)
x_sl = Slider(ax_x, "x (um)", xu[0], xu[-1], valinit=85.0, valstep=GRID_SPACING_UM)
y_sl = Slider(ax_y, "y (um)", yu[0], yu[-1], valinit=40.0, valstep=GRID_SPACING_UM)
z_sl = Slider(ax_z, "z (um)", zu[0], zu[-1], valinit=550.0, valstep=GRID_SPACING_UM)

rax = fig.add_axes([0.75, 0.7, 0.18, 0.17])
mode_buttons = RadioButtons(rax, ("mag", "dExdx", "dEydy", "dEzdz"), active=0)

cax = fig.add_axes([0.75, 0.5, 0.15, 0.15])
auto_scale_check = CheckButtons(cax, ["Auto scale", "Smoothing"], [False, False])

global_scale_cache: dict[str, tuple[float, float]] = {}


def update_view(_=None) -> None:
    mode = mode_buttons.value_selected

    t_ms = float(t_sl.val)
    t = int(round(t_ms / DT_MS))
    t = max(0, min(T - 1, t))

    x_um = float(x_sl.val)
    y_um = float(y_sl.val)
    z_um = float(z_sl.val)
    xi = int(np.argmin(np.abs(xu - x_um)))
    yi = int(np.argmin(np.abs(yu - y_um)))
    zi = int(np.argmin(np.abs(zu - z_um)))

    vol = get_volume_at_t(t, mode=mode)

    smoothing_enabled = auto_scale_check.get_status()[1]
    if smoothing_enabled:
        vol = gaussian_filter(vol, sigma=1.0, mode="nearest")

    slice_xy = vol[:, :, zi].copy()
    slice_yz = vol[xi, :, :].copy()
    slice_zx = vol[:, yi, :].copy()

    z_coord = zu[zi]
    x_coord = xu[xi]
    y_coord = yu[yi]

    if COIL_Z_TIP_MIN <= z_coord <= COIL_Z_MAX:
        z_idx = int(np.argmin(np.abs(zu - z_coord)))
        x_lim = _x_limit_axis[z_idx]
        x_mask = np.abs(xu) <= x_lim
        y_mask = (yu >= COIL_Y_MIN) & (yu <= COIL_Y_MAX)
        slice_xy[np.ix_(x_mask, y_mask)] = np.nan
    if abs(x_coord) <= COIL_X_HALF_MAX:
        y_mask = (yu >= COIL_Y_MIN) & (yu <= COIL_Y_MAX)
        z_mask = _z_in_axis & (abs(x_coord) <= _x_limit_axis)
        slice_yz[np.ix_(y_mask, z_mask)] = np.nan
    if COIL_Y_MIN <= y_coord <= COIL_Y_MAX:
        slice_zx[_coil_xz_inside] = np.nan

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
        if mode not in global_scale_cache:
            global_scale_cache[mode] = estimate_vmin_vmax(mode=mode)
        vmin, vmax = global_scale_cache[mode]

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
    ax_xy.set_title(f"XY: {mode_label}, z={zu[zi]:.0f}um, t={t_ms:.2f}ms")
    ax_yz.set_title(f"YZ: {mode_label}, x={xu[xi]:.0f}um, t={t_ms:.2f}ms")
    ax_zx.set_title(f"ZX: {mode_label}, y={yu[yi]:.0f}um, t={t_ms:.2f}ms")

    cbar.update_normal(im_xy)
    fig.canvas.draw_idle()


t_sl.on_changed(update_view)
x_sl.on_changed(update_view)
y_sl.on_changed(update_view)
z_sl.on_changed(update_view)
mode_buttons.on_clicked(update_view)
auto_scale_check.on_clicked(update_view)

update_view()
plt.show()
