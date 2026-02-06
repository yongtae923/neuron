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

# =========================
# 1) Load data (memory-mapped)
# =========================
E_path = "efield/E_field_1cycle.npy"
C_path = "efield/E_field_grid_coords.npy"

E = np.load(E_path, mmap_mode="r")          # shape: (3, N_spatial, T)
coords = np.load(C_path)                    # shape: (N_spatial, 3)

assert E.ndim == 3 and E.shape[0] == 3, "E_field_1cycle.npy shape should be (3, N_spatial, T)"
assert coords.ndim == 2 and coords.shape[1] == 3, "E_field_grid_coords.npy shape should be (N_spatial, 3)"
assert coords.shape[0] == E.shape[1], "coords N_spatial must match E N_spatial"

T = E.shape[2]

# Time step: 0.05 ms per step (from docs)
DT_MS = 0.05  # ms

# Convert to um for axis labeling (coords are in meters)
coords_um = coords * 1e6
x = coords_um[:, 0]
y = coords_um[:, 1]
z = coords_um[:, 2]

# Grid spacing: 5 μm (from docs)
GRID_SPACING_UM = 5.0

# Coil region [μm] (from docs)
COIL_X_MIN, COIL_X_MAX = -79.5, 79.5
COIL_Y_MIN, COIL_Y_MAX = -32.0, 32.0
COIL_Z_MIN, COIL_Z_MAX = 498.0, 1502.0

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
    """Return boolean mask: True = inside coil (exclude from colorbar calc)."""
    x_in = (xu >= COIL_X_MIN) & (xu <= COIL_X_MAX)
    y_in = (yu >= COIL_Y_MIN) & (yu <= COIL_Y_MAX)
    z_in = (zu >= COIL_Z_MIN) & (zu <= COIL_Z_MAX)
    # broadcast to (nx, ny, nz): point (i,j,k) is in coil if x_in[i] & y_in[j] & z_in[k]
    return x_in[:, None, None] & y_in[None, :, None] & z_in[None, None, :]


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
cmap = plt.cm.get_cmap('RdBu_r').copy()
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

# Time slider: show time in ms (0.05 ms per step)
t_sl = Slider(ax_t, 't (ms)', 0.0, 1, valinit=0.05, valstep=DT_MS)

# Spatial sliders: show physical coordinates in μm
x_sl = Slider(ax_x, 'x (μm)', xu[0], xu[-1], valinit=-80.0, valstep=GRID_SPACING_UM)
y_sl = Slider(ax_y, 'y (μm)', yu[0], yu[-1], valinit=35.0, valstep=GRID_SPACING_UM)
z_sl = Slider(ax_z, 'z (μm)', zu[0], zu[-1], valinit=550.0, valstep=GRID_SPACING_UM)

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
    t_ms = float(t_sl.val)
    t = int(round(t_ms / DT_MS))
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
    
    # XY slice: mask coil region in x-y plane at fixed z
    if COIL_Z_MIN <= z_coord <= COIL_Z_MAX:
        x_mask = (xu >= COIL_X_MIN) & (xu <= COIL_X_MAX)
        y_mask = (yu >= COIL_Y_MIN) & (yu <= COIL_Y_MAX)
        slice_xy[np.ix_(x_mask, y_mask)] = np.nan
    
    # YZ slice: mask coil region in y-z plane at fixed x
    if COIL_X_MIN <= x_coord <= COIL_X_MAX:
        y_mask = (yu >= COIL_Y_MIN) & (yu <= COIL_Y_MAX)
        z_mask = (zu >= COIL_Z_MIN) & (zu <= COIL_Z_MAX)
        slice_yz[np.ix_(y_mask, z_mask)] = np.nan
    
    # ZX slice: mask coil region in x-z plane at fixed y
    if COIL_Y_MIN <= y_coord <= COIL_Y_MAX:
        x_mask = (xu >= COIL_X_MIN) & (xu <= COIL_X_MAX)
        z_mask = (zu >= COIL_Z_MIN) & (zu <= COIL_Z_MAX)
        slice_zx[np.ix_(x_mask, z_mask)] = np.nan

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
    ax_xy.set_title(f"XY: {mode_label}, z={zu[zi]:.0f}μm, t={t_ms:.2f}ms")
    ax_yz.set_title(f"YZ: {mode_label}, x={xu[xi]:.0f}μm, t={t_ms:.2f}ms")
    ax_zx.set_title(f"ZX: {mode_label}, y={yu[yi]:.0f}μm, t={t_ms:.2f}ms")

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