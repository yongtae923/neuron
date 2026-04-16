# D:\yongtae\neuron\angleoutin\code\2_plot_efield.py

"""
plot_efield_v2.py

Interactive 3-slice viewer for E-field data (Ex, Ey, Ez, |E|).
- Loads E_field_1cycle.npy and E_field_grid_coords.npy from efield/
- XY / YZ / ZX slices with sliders for time (ms) and position (x,y,z in μm)
- Coil region masked in black; color scale uses RdBu_r (red-white-blue)
- Colorbar limits: computed from 0~0.1 ms, excluding coil region

Windows + Conda:
  conda activate <env>
  conda install numpy matplotlib
  python plot_efield_v2.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons
from matplotlib.path import Path
import json

# =========================
# 1) Load data (memory-mapped)
# =========================
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data", "30V_OUT10_IN20_CI")
SPEC_PATH = os.path.join(DATA_DIR, "0_grid_time_spec.json")
E_path = os.path.join(DATA_DIR, "1_E_field_1cycle.npy")
C_path = os.path.join(DATA_DIR, "1_E_field_grid_coords.npy")

with open(SPEC_PATH, "r", encoding="utf-8") as f:
    spec = json.load(f)

E = np.load(E_path, mmap_mode="r")          # shape: (3, N_spatial, T)
coords = np.load(C_path)                    # shape: (N_spatial, 3)

assert E.ndim == 3 and E.shape[0] == 3, "E_field_1cycle.npy shape should be (3, N_spatial, T)"
assert coords.ndim == 2 and coords.shape[1] == 3, "E_field_grid_coords.npy shape should be (N_spatial, 3)"
assert coords.shape[0] == E.shape[1], "coords N_spatial must match E N_spatial"

T = E.shape[2]

# Read time/space spec from file (units: um, us)
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

# Convert to um for axis labeling (coords are in meters)
coords_um = coords * 1e6
x = coords_um[:, 0]
y = coords_um[:, 1]
z = coords_um[:, 2]

# Coil mask is loaded from spec: polygon in XZ and Y thickness.

# =========================
# 2) Build regular grid index mapping
#    This assumes coords form a Cartesian grid.
# =========================
xu = np.unique(np.round(x, 6))
yu = np.unique(np.round(y, 6))
zu = np.unique(np.round(z, 6))

xu.sort(); yu.sort(); zu.sort()
nx, ny, nz = len(xu), len(yu), len(zu)
N_expected = nx * ny * nz

if N_expected != coords_um.shape[0]:
    raise ValueError(
        f"Grid does not look like a full Cartesian grid.\n"
        f"Unique counts: nx={nx}, ny={ny}, nz={nz}, nx*ny*nz={N_expected}, "
        f"but N_spatial={coords_um.shape[0]}.\n"
        f"대안: (1) 좌표-값을 산점 데이터로 두고 보간해서 그리드화, (2) ParaView용 point cloud로 시각화."
    )


def _axis_matches_spec(axis_name, values, vmin, vmax, step, tol=1e-3):
    if values.size < 2:
        raise ValueError(f"{axis_name} 축 포인트가 부족합니다: {values.size}")
    actual_min = float(values[0])
    actual_max = float(values[-1])
    actual_step = float(np.median(np.diff(values)))
    if abs(actual_min - vmin) > tol or abs(actual_max - vmax) > tol or abs(actual_step - step) > tol:
        raise ValueError(
            f"{axis_name} 축이 spec과 다릅니다. "
            f"actual(min,max,step)=({actual_min:.3f}, {actual_max:.3f}, {actual_step:.3f}), "
            f"spec=({vmin:.3f}, {vmax:.3f}, {step:.3f})"
        )


_axis_matches_spec("X", xu, X_MIN, X_MAX, GRID_SPACING_UM)
_axis_matches_spec("Y", yu, Y_MIN, Y_MAX, GRID_SPACING_UM)
_axis_matches_spec("Z", zu, Z_MIN, Z_MAX, GRID_SPACING_UM)
if T != FILES_PER_CYCLE:
    raise ValueError(f"시간 스텝 수가 spec과 다릅니다. data T={T}, spec per_cycle={FILES_PER_CYCLE}")

# =========================
# 3) Helper to reshape E at time t into (nx, ny, nz)
# =========================
def get_volume_at_t(t, mode="mag"):
    """
    mode: 'mag' or 'Ex' or 'Ey' or 'Ez'
    return: volume (nx, ny, nz)
    """
    if mode == "mag":
        # |E| = sqrt(Ex^2 + Ey^2 + Ez^2) at time t
        ex = E[0, :, t]
        ey = E[1, :, t]
        ez = E[2, :, t]
        val = np.sqrt(ex*ex + ey*ey + ez*ez)
    elif mode == "Ex":
        val = E[0, :, t]
    elif mode == "Ey":
        val = E[1, :, t]
    elif mode == "Ez":
        val = E[2, :, t]
    else:
        raise ValueError("mode must be one of: mag, Ex, Ey, Ez")

    # place into grid (Ansys file order: x,y,z with z varying fastest)
    vol = val.reshape(nx, ny, nz).astype(np.float32)
    return vol

# =========================
# 4) Precompute global vmin/vmax for stable color scale
#    Time range: 0~0.1 ms (indices 0, 1, 2)
# =========================
def _coil_mask_3d():
    """Return boolean mask: True = inside coil (exclude from colorbar calc).
    Coil is a pentagon in XZ plane, extruded along Y.
    Pentagon vertices (XZ): (-85,-382), (0,-530), (85,-382), (85,-250), (-85,-250)
    Y thickness: -21.25 ~ 26.75
    """
    pentagon_path = Path(COIL_POLYGON_XZ)
    
    # Y check: simple range
    y_in = (yu >= COIL_Y_MIN) & (yu <= COIL_Y_MAX)
    
    # XZ check: 각 (x, z) 점이 오각형 내부인지 판정
    x_grid, z_grid = np.meshgrid(xu, zu, indexing='ij')
    grid_xz_flat = np.column_stack([x_grid.ravel(), z_grid.ravel()])  # shape: (nx*nz, 2)
    # Include boundary points with a tiny positive radius.
    xz_in_flat = pentagon_path.contains_points(grid_xz_flat, radius=1e-9)  # shape: (nx*nz,)
    xz_in = xz_in_flat.reshape(nx, nz)  # shape: (nx, nz)
    
    # Broadcast to (nx, ny, nz): point (i,j,k) is in coil if xz_in[i,k] & y_in[j]
    return xz_in[:, None, :] & y_in[None, :, None]


def estimate_vmin_vmax(mode="mag"):
    """Compute vmin/vmax from data in 0~0.1ms range, excluding coil region."""
    t_indices = [0, 1, 2]  # 0, 0.05, 0.1 ms
    t_indices = [t for t in t_indices if t < T]
    
    coil_mask = _coil_mask_3d()  # True = coil (exclude)
    
    all_vals = []
    for t in t_indices:
        vol = get_volume_at_t(t, mode=mode)
        outside_coil = vol[~coil_mask]  # exclude coil region
        finite = outside_coil[np.isfinite(outside_coil)]
        if finite.size > 0:
            all_vals.append(finite)
    
    if not all_vals:
        return 0.0, 1.0  # fallback
    
    combined = np.concatenate(all_vals)
    combined = combined[np.isfinite(combined)]
    if combined.size == 0:
        return 0.0, 1.0
    
    abs_max = float(np.max(np.abs(combined)))
    
    if mode == "mag":
        # Magnitude: 0 to max (always positive)
        vmin = 0.0
        vmax = float(np.max(combined))
        if vmax <= 0:
            vmax = 1e-6  # avoid zero range
    else:
        # Ex, Ey, Ez: symmetric range centered at 0 for RdBu_r
        if abs_max <= 0:
            abs_max = 1e-6
        vmin = -abs_max
        vmax = abs_max
    
    return vmin, vmax

# =========================
# 5) Interactive viewer with matplotlib widgets
# =========================
# Create figure with subplots for controls
fig = plt.figure(figsize=(16, 6))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.15], hspace=0.3, wspace=0.5)

# Plot axes
ax_xy = fig.add_subplot(gs[0, 0])
ax_yz = fig.add_subplot(gs[0, 1])
ax_zx = fig.add_subplot(gs[0, 2])
ax_ctrl = fig.add_subplot(gs[0, 3])  # Control panel

# Initialize colormap with black for NaN (coil region)
cmap = plt.colormaps['RdBu_r'].copy()
cmap.set_bad(color='black')

# Initialize with dummy
init_vol = get_volume_at_t(0, mode="mag")
# Use RdBu_r colormap (red-white-blue)
im_xy = ax_xy.imshow(init_vol[:, :, nz//2].T, origin="lower", aspect="equal", cmap=cmap)
im_yz = ax_yz.imshow(init_vol[nx//2, :, :].T, origin="lower", aspect="equal", cmap=cmap)
im_zx = ax_zx.imshow(init_vol[:, ny//2, :].T, origin="lower", aspect="equal", cmap=cmap)

ax_xy.set_title("XY slice (z fixed)")
ax_yz.set_title("YZ slice (x fixed)")
ax_zx.set_title("ZX slice (y fixed)")

# Axis labels with physical units (μm)
ax_xy.set_xlabel("x (μm)"); ax_xy.set_ylabel("y (μm)")
ax_yz.set_xlabel("y (μm)"); ax_yz.set_ylabel("z (μm)")
ax_zx.set_xlabel("x (μm)"); ax_zx.set_ylabel("z (μm)")

# Set axis limits to physical coordinates (from docs)
ax_xy.set_xlim(0, nx-1); ax_xy.set_ylim(0, ny-1)
ax_yz.set_xlim(0, ny-1); ax_yz.set_ylim(0, nz-1)
ax_zx.set_xlim(0, nx-1); ax_zx.set_ylim(0, nz-1)

# Set tick labels to show physical coordinates
def set_physical_ticks(ax, coord_array, axis='x'):
    """Set tick labels to show physical coordinates in μm"""
    n_ticks = 5
    if axis == 'x':
        ticks = np.linspace(0, len(coord_array)-1, n_ticks, dtype=int)
        labels = [f'{coord_array[i]:.0f}' for i in ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
    elif axis == 'y':
        ticks = np.linspace(0, len(coord_array)-1, n_ticks, dtype=int)
        labels = [f'{coord_array[i]:.0f}' for i in ticks]
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)

# Apply physical tick labels (will be updated in update_view)

# Single shared colorbar (E-field in V/m, from docs)
cbar = fig.colorbar(im_xy, ax=[ax_xy, ax_yz, ax_zx], shrink=0.9)
cbar.set_label("E-field (V/m)")

# Control panel (hide axes)
ax_ctrl.axis('off')

# Create sliders
ax_t = fig.add_subplot(gs[1, 0])
ax_x = fig.add_subplot(gs[1, 1])
ax_y = fig.add_subplot(gs[1, 2])
ax_z = fig.add_subplot(gs[1, 3])

# Time slider: read from spec (us)
time_end_us = TIME_START_US + (T - 1) * DT_US
t_sl = Slider(ax_t, 't (us)', TIME_START_US, time_end_us, valinit=TIME_START_US, valstep=DT_US)

# Spatial sliders: show physical coordinates in μm
x_sl = Slider(ax_x, 'x (μm)', xu[0], xu[-1], valinit=0.0, valstep=GRID_SPACING_UM)
y_sl = Slider(ax_y, 'y (μm)', yu[0], yu[-1], valinit=0.0, valstep=GRID_SPACING_UM)
z_sl = Slider(ax_z, 'z (μm)', zu[0], zu[-1], valinit=-450.0, valstep=GRID_SPACING_UM)

# Radio buttons for mode selection
rax = fig.add_axes([0.75, 0.7, 0.15, 0.15])
mode_buttons = RadioButtons(rax, ('mag', 'Ex', 'Ey', 'Ez'), active=0)

# Checkbox for auto scale
cax = fig.add_axes([0.75, 0.5, 0.15, 0.1])
auto_scale_check = CheckButtons(cax, ['Auto scale'], [False])

# Cache global vmin/vmax per mode
global_scale_cache = {}
current_mode = "mag"
auto_scale_enabled = False

def update_view(val=None):
    global current_mode, auto_scale_enabled
    
    mode = mode_buttons.value_selected
    current_mode = mode
    
    # Convert slider values to indices
    t_us = float(t_sl.val)
    t = int(round((t_us - TIME_START_US) / DT_US))
    t = max(0, min(T-1, t))  # Clamp to valid range
    
    # Find nearest grid indices for physical coordinates
    x_um = float(x_sl.val)
    y_um = float(y_sl.val)
    z_um = float(z_sl.val)
    
    xi = np.argmin(np.abs(xu - x_um))
    yi = np.argmin(np.abs(yu - y_um))
    zi = np.argmin(np.abs(zu - z_um))
    
    auto_scale_enabled = auto_scale_check.get_status()[0]

    vol = get_volume_at_t(t, mode=mode)

    # Extract slices
    slice_xy = vol[:, :, zi].copy()     # (nx, ny)
    slice_yz = vol[xi, :, :].copy()     # (ny, nz)
    slice_zx = vol[:, yi, :].copy()     # (nx, nz)

    # Mask coil region with black (NaN -> displayed as black with bad color)
    z_coord = zu[zi]
    x_coord = xu[xi]
    y_coord = yu[yi]
    
    pentagon_path = Path(COIL_POLYGON_XZ)
    
    # XY slice: for fixed z, mask x values inside polygon and y within coil thickness.
    X_grid, Y_grid = np.meshgrid(xu, yu, indexing='ij')  # shape: (nx, ny), (nx, ny)
    xz_points_xy = np.column_stack([X_grid.ravel(), np.full(nx * ny, z_coord)])
    xz_in_xy = pentagon_path.contains_points(xz_points_xy, radius=1e-9).reshape(nx, ny)
    y_in_xy = (Y_grid >= COIL_Y_MIN) & (Y_grid <= COIL_Y_MAX)
    slice_xy[xz_in_xy & y_in_xy] = np.nan

    # YZ slice: for fixed x, mask z values where (x,z) in polygon and y within thickness.
    Y_grid, Z_grid = np.meshgrid(yu, zu, indexing='ij')  # shape: (ny, nz), (ny, nz)
    xz_points_yz = np.column_stack([np.full(ny * nz, x_coord), Z_grid.ravel()])
    xz_in_yz = pentagon_path.contains_points(xz_points_yz, radius=1e-9).reshape(ny, nz)
    y_in_yz = (Y_grid >= COIL_Y_MIN) & (Y_grid <= COIL_Y_MAX)
    slice_yz[xz_in_yz & y_in_yz] = np.nan

    # ZX slice: only slices within coil thickness in y should be masked.
    if COIL_Y_MIN <= y_coord <= COIL_Y_MAX:
        X_grid, Z_grid = np.meshgrid(xu, zu, indexing='ij')  # shape: (nx, nz), (nx, nz)
        xz_points_zx = np.column_stack([X_grid.ravel(), Z_grid.ravel()])
        xz_in_zx = pentagon_path.contains_points(xz_points_zx, radius=1e-9).reshape(nx, nz)
        slice_zx[xz_in_zx] = np.nan

    # Color scaling (exclude NaN, inf)
    def safe_minmax(arr):
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.nan, np.nan
        return float(np.min(finite)), float(np.max(finite))
    
    if auto_scale_enabled:
        mn_xy, mx_xy = safe_minmax(slice_xy)
        mn_yz, mx_yz = safe_minmax(slice_yz)
        mn_zx, mx_zx = safe_minmax(slice_zx)
        vmin = float(np.nanmin([mn_xy, mn_yz, mn_zx]))
        vmax = float(np.nanmax([mx_xy, mx_yz, mx_zx]))
        if mode == "mag":
            vmin = 0.0  # mag is always >= 0
        else:
            # Ex/Ey/Ez: symmetric around 0
            abs_max = max(abs(vmin), abs(vmax)) if np.isfinite(vmin) and np.isfinite(vmax) else 1e-6
            vmin, vmax = -abs_max, abs_max
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = 0.0, 1.0
    else:
        if mode not in global_scale_cache:
            global_scale_cache[mode] = estimate_vmin_vmax(mode=mode)
        vmin, vmax = global_scale_cache[mode]

    # Update images (transpose to match imshow orientation)
    im_xy.set_data(slice_xy.T)
    im_yz.set_data(slice_yz.T)
    im_zx.set_data(slice_zx.T)

    for im in (im_xy, im_yz, im_zx):
        im.set_clim(vmin, vmax)

    # Update axis tick labels to show physical coordinates
    n_ticks = 5
    for ax, coord_array, x_or_y in [(ax_xy, xu, 'x'), (ax_xy, yu, 'y'),
                                     (ax_yz, yu, 'x'), (ax_yz, zu, 'y'),
                                     (ax_zx, xu, 'x'), (ax_zx, zu, 'y')]:
        ticks_idx = np.linspace(0, len(coord_array)-1, n_ticks, dtype=int)
        ticks_pos = ticks_idx
        labels = [f'{coord_array[i]:.0f}' for i in ticks_idx]
        if x_or_y == 'x':
            ax.set_xticks(ticks_pos)
            ax.set_xticklabels(labels)
        else:
            ax.set_yticks(ticks_pos)
            ax.set_yticklabels(labels)

    # Titles (shortened)
    mode_label = {'mag': '|E|', 'Ex': 'Ex', 'Ey': 'Ey', 'Ez': 'Ez'}[mode]
    ax_xy.set_title(f"XY: {mode_label}, z={zu[zi]:.0f}μm, t={t_us:.0f}us")
    ax_yz.set_title(f"YZ: {mode_label}, x={xu[xi]:.0f}μm, t={t_us:.0f}us")
    ax_zx.set_title(f"ZX: {mode_label}, y={yu[yi]:.0f}μm, t={t_us:.0f}us")

    cbar.update_normal(im_xy)
    fig.canvas.draw_idle()

# Wire callbacks
t_sl.on_changed(update_view)
x_sl.on_changed(update_view)
y_sl.on_changed(update_view)
z_sl.on_changed(update_view)
mode_buttons.on_clicked(update_view)
auto_scale_check.on_clicked(update_view)

# Initial update
update_view()

plt.show()