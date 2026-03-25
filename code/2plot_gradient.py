# 2_plot_gradient.py (powershell)
"""
Interactive 3-slice viewer for directional gradient data (2-cycle):
- dEx/dx
- dEy/dy
- dEz/dz
- magnitude: sqrt((dEx/dx)^2 + (dEy/dy)^2 + (dEz/dz)^2)

Data source:
- efield/400us_50Hz_10umspaing_100mA/grad_2cycle.npy
- efield/400us_50Hz_10umspaing_100mA/E_field_grid_coords.npy
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.widgets import Slider, RadioButtons, CheckButtons
from scipy.ndimage import gaussian_filter


# =========================
# 1) Load data (memory-mapped)
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent.parent
G_PATH = SCRIPT_DIR / "efield" / "400us_50Hz_10umspaing_100mA" / "grad_2cycle.npy"
C_PATH = SCRIPT_DIR / "efield" / "400us_50Hz_10umspaing_100mA" / "E_field_grid_coords.npy"

G = np.load(G_PATH, mmap_mode="r")   # shape: (3, nx, ny, nz, T)
coords = np.load(C_PATH)              # shape: (N_spatial, 3), meters

if not (G.ndim == 5 and G.shape[0] == 3):
	raise ValueError("Gradient file shape should be (3, Nx, Ny, Nz, T)")
if not (coords.ndim == 2 and coords.shape[1] == 3):
	raise ValueError("Coords shape should be (N_spatial, 3)")

nx, ny, nz, nt = G.shape[1], G.shape[2], G.shape[3], G.shape[4]

# Requested time convention
DT_MS = 0.05
T_END_MS = 1.0

# Convert to um for display
coords_um = coords * 1e6
xu = np.unique(np.round(coords_um[:, 0], 6))
yu = np.unique(np.round(coords_um[:, 1], 6))
zu = np.unique(np.round(coords_um[:, 2], 6))
xu.sort()
yu.sort()
zu.sort()

if len(xu) != nx or len(yu) != ny or len(zu) != nz:
	raise ValueError(
		"Gradient grid shape and coordinate-derived axis lengths do not match:\n"
		f"grad=(nx={nx}, ny={ny}, nz={nz}), "
		f"coords=(nx={len(xu)}, ny={len(yu)}, nz={len(zu)})"
	)

# Requested spatial domain/spacing
EXPECTED_MIN = np.array([-200.0, -200.0, 400.0], dtype=float)
EXPECTED_MAX = np.array([200.0, 200.0, 800.0], dtype=float)
GRID_SPACING_UM = 5.0


def _check_domain() -> None:
	mins = np.array([xu[0], yu[0], zu[0]], dtype=float)
	maxs = np.array([xu[-1], yu[-1], zu[-1]], dtype=float)
	if not (np.allclose(mins, EXPECTED_MIN, atol=1e-6) and np.allclose(maxs, EXPECTED_MAX, atol=1e-6)):
		print("[Warn] Coordinate domain differs from requested bounds.")
		print(f"       Requested min/max: {EXPECTED_MIN} / {EXPECTED_MAX}")
		print(f"       Actual    min/max: {mins} / {maxs}")


_check_domain()


# Coil region (um): 2 trapezoids in x-z, extruded along y in [-5, 10]
COIL_RIGHT_TRAPEZOID_XZ = np.array([
	[45.0, 800.0],
	[45.0, 590.0],
	[5.0, 500.0],
	[5.0, 800.0],
], dtype=float)

COIL_LEFT_TRAPEZOID_XZ = np.array([
	[-45.0, 800.0],
	[-45.0, 590.0],
	[-5.0, 500.0],
	[-5.0, 800.0],
], dtype=float)

# y range updated to [-6.25, 11.75] um
COIL_Y_MIN, COIL_Y_MAX = -6.25, 11.75


def _build_surface_shell_mask(mask: np.ndarray) -> np.ndarray:
	"""True for points adjacent (6-neighbor) to coil interior."""
	padded = np.pad(mask, 1, mode="constant", constant_values=False)
	neighbor = (
		padded[2:, 1:-1, 1:-1]
		| padded[:-2, 1:-1, 1:-1]
		| padded[1:-1, 2:, 1:-1]
		| padded[1:-1, :-2, 1:-1]
		| padded[1:-1, 1:-1, 2:]
		| padded[1:-1, 1:-1, :-2]
	)
	return neighbor & (~mask)


def _build_coil_masks() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	xx, zz = np.meshgrid(xu, zu, indexing="ij")
	xz_points = np.column_stack([xx.ravel(), zz.ravel()])

	left_path = MplPath(COIL_LEFT_TRAPEZOID_XZ)
	right_path = MplPath(COIL_RIGHT_TRAPEZOID_XZ)
	left_inside = left_path.contains_points(xz_points).reshape(nx, nz)
	right_inside = right_path.contains_points(xz_points).reshape(nx, nz)
	xz_inside_any = left_inside | right_inside

	y_in = (yu >= COIL_Y_MIN) & (yu <= COIL_Y_MAX)
	coil_mask_3d = xz_inside_any[:, None, :] & y_in[None, :, None]
	shell_mask_3d = _build_surface_shell_mask(coil_mask_3d)
	return left_inside, right_inside, xz_inside_any, coil_mask_3d, shell_mask_3d


LEFT_XZ_INSIDE, RIGHT_XZ_INSIDE, COIL_XZ_INSIDE, COIL_MASK_3D, COIL_SHELL_MASK_3D = _build_coil_masks()


# =========================
# 2) Data helpers
# =========================
def get_volume_at_t(t_idx: int, mode: str = "mag") -> np.ndarray:
	"""
	mode: 'mag' or 'dExdx' or 'dEydy' or 'dEzdz'
	return: volume (nx, ny, nz)
	"""
	if mode == "dExdx":
		vol = G[0, :, :, :, t_idx]
	elif mode == "dEydy":
		vol = G[1, :, :, :, t_idx]
	elif mode == "dEzdz":
		vol = G[2, :, :, :, t_idx]
	elif mode == "mag":
		gx = G[0, :, :, :, t_idx]
		gy = G[1, :, :, :, t_idx]
		gz = G[2, :, :, :, t_idx]
		vol = np.sqrt(gx * gx + gy * gy + gz * gz)
	else:
		raise ValueError("mode must be one of: mag, dExdx, dEydy, dEzdz")
	return np.asarray(vol, dtype=np.float32)


def estimate_vmin_vmax(mode: str = "mag") -> tuple[float, float]:
	"""Compute robust color range from early time points, excluding coil region."""
	t_indices = [t for t in [0, 1, 2] if t < nt]
	vals = []

	for t_idx in t_indices:
		vol = get_volume_at_t(t_idx, mode=mode)
		outside = vol[~COIL_MASK_3D]
		finite = outside[np.isfinite(outside)]
		if finite.size > 0:
			vals.append(finite)

	if not vals:
		return 0.0, 1.0

	combined = np.concatenate(vals)
	combined = combined[np.isfinite(combined)]
	if combined.size == 0:
		return 0.0, 1.0

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

t0_idx = 0
init_vol = get_volume_at_t(t0_idx, mode="mag")

im_xy = ax_xy.imshow(
	init_vol[:, :, nz // 2].T,
	origin="lower",
	aspect="equal",
	cmap=cmap,
	extent=[xu[0], xu[-1], yu[0], yu[-1]],
)
im_yz = ax_yz.imshow(
	init_vol[nx // 2, :, :].T,
	origin="lower",
	aspect="equal",
	cmap=cmap,
	extent=[yu[0], yu[-1], zu[0], zu[-1]],
)
im_zx = ax_zx.imshow(
	init_vol[:, ny // 2, :].T,
	origin="lower",
	aspect="equal",
	cmap=cmap,
	extent=[xu[0], xu[-1], zu[0], zu[-1]],
)

ax_xy.set_title("XY slice (z fixed)")
ax_yz.set_title("YZ slice (x fixed)")
ax_zx.set_title("ZX slice (y fixed)")

ax_xy.set_xlabel("x (um)")
ax_xy.set_ylabel("y (um)")
ax_yz.set_xlabel("y (um)")
ax_yz.set_ylabel("z (um)")
ax_zx.set_xlabel("x (um)")
ax_zx.set_ylabel("z (um)")

cbar = fig.colorbar(im_xy, ax=[ax_xy, ax_yz, ax_zx], shrink=0.9)
cbar.set_label("Gradient (unit per meter)")

ax_ctrl.axis("off")

ax_t = fig.add_subplot(gs[1, 0])
ax_x = fig.add_subplot(gs[1, 1])
ax_y = fig.add_subplot(gs[1, 2])
ax_z = fig.add_subplot(gs[1, 3])

t_max_ms = min(T_END_MS, (nt - 1) * DT_MS)
t_init_ms = min(20.0, t_max_ms)
t_sl = Slider(ax_t, "t (ms)", 0.0, t_max_ms, valinit=t_init_ms, valstep=DT_MS)
x_sl = Slider(ax_x, "x (um)", xu[0], xu[-1], valinit=0.0, valstep=GRID_SPACING_UM)
y_sl = Slider(ax_y, "y (um)", yu[0], yu[-1], valinit=0.0, valstep=GRID_SPACING_UM)
z_sl = Slider(ax_z, "z (um)", zu[0], zu[-1], valinit=600.0, valstep=GRID_SPACING_UM)

rax = fig.add_axes([0.75, 0.7, 0.18, 0.17])
mode_buttons = RadioButtons(rax, ("mag", "dExdx", "dEydy", "dEzdz"), active=0)

cax = fig.add_axes([0.75, 0.5, 0.15, 0.15])
check_buttons = CheckButtons(cax, ["Auto scale", "Smoothing"], [False, False])

global_scale_cache: dict[str, tuple[float, float]] = {}


def _safe_minmax(arr: np.ndarray) -> tuple[float, float]:
	finite = arr[np.isfinite(arr)]
	if finite.size == 0:
		return np.nan, np.nan
	return float(np.min(finite)), float(np.max(finite))


def update_view(_=None) -> None:
	mode = mode_buttons.value_selected

	t_ms = float(t_sl.val)
	t_idx = int(round(t_ms / DT_MS))
	t_idx = max(0, min(nt - 1, t_idx))

	xi = int(np.argmin(np.abs(xu - float(x_sl.val))))
	yi = int(np.argmin(np.abs(yu - float(y_sl.val))))
	zi = int(np.argmin(np.abs(zu - float(z_sl.val))))

	vol = get_volume_at_t(t_idx, mode=mode)

	smoothing_enabled = check_buttons.get_status()[1]
	if smoothing_enabled:
		vol = gaussian_filter(vol, sigma=1.0, mode="nearest")

	slice_xy = vol[:, :, zi].copy()
	slice_yz = vol[xi, :, :].copy()
	slice_zx = vol[:, yi, :].copy()

	# Mask coil area as NaN for visualization.
	z_coord = zu[zi]
	x_coord = xu[xi]
	y_coord = yu[yi]

	# Mask coil inside + surface shell in the current slices
	slice_xy_mask = COIL_MASK_3D[:, :, zi] | COIL_SHELL_MASK_3D[:, :, zi]
	slice_yz_mask = COIL_MASK_3D[xi, :, :] | COIL_SHELL_MASK_3D[xi, :, :]
	slice_zx_mask = COIL_MASK_3D[:, yi, :] | COIL_SHELL_MASK_3D[:, yi, :]

	slice_xy[slice_xy_mask] = np.nan
	slice_yz[slice_yz_mask] = np.nan
	slice_zx[slice_zx_mask] = np.nan

	# This original path-based mask can be used for debug/overlay but now not inherently needed
	# if (z_coord >= COIL_LEFT_TRAPEZOID_XZ[:, 1].min()) and (z_coord <= COIL_LEFT_TRAPEZOID_XZ[:, 1].max()):
	#	z_idx = int(np.argmin(np.abs(zu - z_coord)))
	#	x_mask = COIL_XZ_INSIDE[:, z_idx]
	#	y_mask = (yu >= COIL_Y_MIN) & (yu <= COIL_Y_MAX)
	#	slice_xy[np.ix_(x_mask, y_mask)] = np.nan

	# if np.any(COIL_XZ_INSIDE[xi, :]):
	#	y_mask = (yu >= COIL_Y_MIN) & (yu <= COIL_Y_MAX)
	#	z_mask = COIL_XZ_INSIDE[xi, :]
	#	slice_yz[np.ix_(y_mask, z_mask)] = np.nan

	# if COIL_Y_MIN <= y_coord <= COIL_Y_MAX:
	#	slice_zx[COIL_XZ_INSIDE] = np.nan

	auto_scale_enabled = check_buttons.get_status()[0]
	if auto_scale_enabled:
		mn_xy, mx_xy = _safe_minmax(slice_xy)
		mn_yz, mx_yz = _safe_minmax(slice_yz)
		mn_zx, mx_zx = _safe_minmax(slice_zx)
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

	mode_label = {
		"mag": "|grad|",
		"dExdx": "dEx/dx",
		"dEydy": "dEy/dy",
		"dEzdz": "dEz/dz",
	}[mode]
	ax_xy.set_title(f"XY: {mode_label}, z={zu[zi]:.0f}um, t={t_idx * DT_MS:.2f}ms")
	ax_yz.set_title(f"YZ: {mode_label}, x={xu[xi]:.0f}um, t={t_idx * DT_MS:.2f}ms")
	ax_zx.set_title(f"ZX: {mode_label}, y={yu[yi]:.0f}um, t={t_idx * DT_MS:.2f}ms")

	cbar.update_normal(im_xy)
	fig.canvas.draw_idle()


t_sl.on_changed(update_view)
x_sl.on_changed(update_view)
y_sl.on_changed(update_view)
z_sl.on_changed(update_view)
mode_buttons.on_clicked(update_view)
check_buttons.on_clicked(update_view)

update_view()
plt.show()
