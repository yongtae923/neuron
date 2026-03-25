# 1_plot_efield.py (powershell)
"""
Interactive 3-slice viewer for the new 2-cycle E-field dataset only.
- Uses:
  - efield/400us_50Hz_10umspaing_100mA/E_field_2cycle.npy
  - efield/400us_50Hz_10umspaing_100mA/E_field_grid_coords.npy
- No coil mask
- Sliders: time (ms), x, y, z (um)
- Radio: mag, Ex, Ey, Ez
- Unit: V/m
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as animation
from matplotlib.path import Path as MplPath
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.animation import FuncAnimation
from pathlib import Path

DATASET = "800us_50Hz_10umspaing_100mA"
BASE_FOLDER = Path(r"D:\yongtae\neuron\efield")
DATA_FOLDER = BASE_FOLDER / DATASET

E_PATH = DATA_FOLDER / "E_field_2cycle.npy"
C_PATH = DATA_FOLDER / "E_field_grid_coords.npy"

# Coil region in um:
# Right trapezoid: (45,800), (45,590), (5,500), (5,800)
# Left trapezoid: (-45,800), (-45,590), (-5,500), (-5,800)
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
COIL_Y_MIN, COIL_Y_MAX = -6.25, 11.75

DT_MS = 0.05


def _axis_step(vals):
    if len(vals) < 2:
        return 1.0
    diffs = np.diff(vals)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return 1.0
    return float(np.min(np.abs(diffs)))


def load_dataset():
    E = np.load(E_PATH, mmap_mode="r")
    coords = np.load(C_PATH)

    if E.ndim != 3 or E.shape[0] != 3:
        raise ValueError(f"Expected E shape (3, N_spatial, T), got {E.shape}")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Expected coords shape (N_spatial, 3), got {coords.shape}")

    n_spatial, nt = E.shape[1], E.shape[2]

    coords_um = coords * 1e6
    xu = np.unique(np.round(coords_um[:, 0], 6)); xu.sort()
    yu = np.unique(np.round(coords_um[:, 1], 6)); yu.sort()
    zu = np.unique(np.round(coords_um[:, 2], 6)); zu.sort()

    nx, ny, nz = len(xu), len(yu), len(zu)
    if nx * ny * nz != n_spatial:
        raise ValueError(
            f"Grid mismatch: nx*ny*nz={nx*ny*nz}, N_spatial={n_spatial}"
        )

    return E, xu, yu, zu, nx, ny, nz, nt


E, xu, yu, zu, nx, ny, nz, nt = load_dataset()


def _build_surface_shell_mask(mask: np.ndarray) -> np.ndarray:
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


def _build_coil_mask_3d() -> tuple[np.ndarray, np.ndarray]:
    xx, zz = np.meshgrid(xu, zu, indexing="ij")
    xz_points = np.column_stack([xx.ravel(), zz.ravel()])

    left_inside = MplPath(COIL_LEFT_TRAPEZOID_XZ).contains_points(xz_points).reshape(nx, nz)
    right_inside = MplPath(COIL_RIGHT_TRAPEZOID_XZ).contains_points(xz_points).reshape(nx, nz)
    xz_inside = left_inside | right_inside

    y_mask = (yu >= COIL_Y_MIN) & (yu <= COIL_Y_MAX)
    coil_mask = xz_inside[:, None, :] & y_mask[None, :, None]
    shell_mask = _build_surface_shell_mask(coil_mask)
    return coil_mask, shell_mask


COIL_MASK_3D, COIL_SHELL_MASK_3D = _build_coil_mask_3d()


def get_volume_at_t(t_idx, mode="mag"):
    t_idx = max(0, min(nt - 1, int(t_idx)))
    if mode == "mag":
        ex = E[0, :, t_idx].reshape(nx, ny, nz).astype(np.float32)
        ey = E[1, :, t_idx].reshape(nx, ny, nz).astype(np.float32)
        ez = E[2, :, t_idx].reshape(nx, ny, nz).astype(np.float32)
        return np.sqrt(ex * ex + ey * ey + ez * ez)
    if mode in ("Ex", "Ey", "Ez"):
        idx = {"Ex": 0, "Ey": 1, "Ez": 2}[mode]
        return E[idx, :, t_idx].reshape(nx, ny, nz).astype(np.float32)
    raise ValueError("mode must be one of: mag, Ex, Ey, Ez")


fig = plt.figure(figsize=(16, 6))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.15], hspace=0.32, wspace=0.5)

ax_xy = fig.add_subplot(gs[0, 0])
ax_yz = fig.add_subplot(gs[0, 1])
ax_zx = fig.add_subplot(gs[0, 2])
ax_ctrl = fig.add_subplot(gs[0, 3])
ax_ctrl.axis("off")

t_init_ms = 0.0
t_init = int(round(t_init_ms / DT_MS))
xi_init = nx // 2
yi_init = ny // 2
zi_init = nz // 2

init_vol = get_volume_at_t(t_init, mode="mag")

im_xy = ax_xy.imshow(
    init_vol[:, :, zi_init].T,
    origin="lower",
    aspect="equal",
    cmap="viridis",
    extent=[xu[0], xu[-1], yu[0], yu[-1]],
)
im_yz = ax_yz.imshow(
    init_vol[xi_init, :, :].T,
    origin="lower",
    aspect="equal",
    cmap="viridis",
    extent=[yu[0], yu[-1], zu[0], zu[-1]],
)
im_zx = ax_zx.imshow(
    init_vol[:, yi_init, :].T,
    origin="lower",
    aspect="equal",
    cmap="viridis",
    extent=[xu[0], xu[-1], zu[0], zu[-1]],
)

ax_xy.set_xlabel("x (um)"); ax_xy.set_ylabel("y (um)")
ax_yz.set_xlabel("y (um)"); ax_yz.set_ylabel("z (um)")
ax_zx.set_xlabel("x (um)"); ax_zx.set_ylabel("z (um)")

cbar = fig.colorbar(im_xy, ax=[ax_xy, ax_yz, ax_zx], shrink=0.9)
cbar.set_label("E-field (V/m)")

ax_t = fig.add_subplot(gs[1, 0])
ax_x = fig.add_subplot(gs[1, 1])
ax_y = fig.add_subplot(gs[1, 2])
ax_z = fig.add_subplot(gs[1, 3])

t_sl = Slider(ax_t, "t (ms)", 0.0, min(1.0, (nt - 1) * DT_MS), valinit=t_init_ms, valstep=DT_MS)
x_sl = Slider(ax_x, "x (um)", xu[0], xu[-1], valinit=xu[xi_init], valstep=_axis_step(xu))
y_sl = Slider(ax_y, "y (um)", yu[0], yu[-1], valinit=yu[yi_init], valstep=_axis_step(yu))
z_sl = Slider(ax_z, "z (um)", zu[0], zu[-1], valinit=zu[zi_init], valstep=_axis_step(zu))

rax = fig.add_axes([0.75, 0.68, 0.16, 0.2])
mode_buttons = RadioButtons(rax, ("mag", "Ex", "Ey", "Ez"), active=0)


def update_view(_=None):
    mode = mode_buttons.value_selected

    t_ms = float(t_sl.val)
    t_idx = int(round(t_ms / DT_MS))
    t_idx = max(0, min(nt - 1, t_idx))

    xi = int(np.argmin(np.abs(xu - float(x_sl.val))))
    yi = int(np.argmin(np.abs(yu - float(y_sl.val))))
    zi = int(np.argmin(np.abs(zu - float(z_sl.val))))

    vol = get_volume_at_t(t_idx, mode=mode)
    slice_xy = vol[:, :, zi]
    slice_yz = vol[xi, :, :]
    slice_zx = vol[:, yi, :]

    # Coil area (interior + surface) masked
    mask_xy = COIL_MASK_3D[:, :, zi] | COIL_SHELL_MASK_3D[:, :, zi]
    mask_yz = COIL_MASK_3D[xi, :, :] | COIL_SHELL_MASK_3D[xi, :, :]
    mask_zx = COIL_MASK_3D[:, yi, :] | COIL_SHELL_MASK_3D[:, yi, :]

    slice_xy = slice_xy.copy()
    slice_yz = slice_yz.copy()
    slice_zx = slice_zx.copy()

    slice_xy[mask_xy] = np.nan
    slice_yz[mask_yz] = np.nan
    slice_zx[mask_zx] = np.nan

    if mode == "mag":
        vmin = 0.0
        outside_max = np.nanmax([np.nanmax(slice_xy), np.nanmax(slice_yz), np.nanmax(slice_zx)])
        vmax = float(outside_max) if np.isfinite(outside_max) and outside_max > 0 else 1e-6
        cmap = "viridis"
    else:
        outside_abs_max = np.nanmax([
            np.abs(slice_xy[~mask_xy]).max() if np.any(~mask_xy) else 0.0,
            np.abs(slice_yz[~mask_yz]).max() if np.any(~mask_yz) else 0.0,
            np.abs(slice_zx[~mask_zx]).max() if np.any(~mask_zx) else 0.0,
        ])
        abs_max = float(outside_abs_max) if np.isfinite(outside_abs_max) and outside_abs_max > 0 else 1e-6
        vmin, vmax = -abs_max, abs_max
        cmap = "RdBu_r"

    im_xy.set_data(slice_xy.T)
    im_yz.set_data(slice_yz.T)
    im_zx.set_data(slice_zx.T)

    for im in (im_xy, im_yz, im_zx):
        im.set_cmap(cmap)
        im.set_clim(vmin, vmax)

    ax_xy.set_title(f"XY: {mode}, z={zu[zi]:.1f} um, t={t_ms:.2f} ms")
    ax_yz.set_title(f"YZ: {mode}, x={xu[xi]:.1f} um, t={t_ms:.2f} ms")
    ax_zx.set_title(f"ZX: {mode}, y={yu[yi]:.1f} um, t={t_ms:.2f} ms")

    cbar.update_normal(im_xy)
    fig.canvas.draw_idle()


t_sl.on_changed(update_view)
x_sl.on_changed(update_view)
y_sl.on_changed(update_view)
z_sl.on_changed(update_view)
mode_buttons.on_clicked(update_view)

# ==== animation helper (Ez 채널 고정, 100us 간격, 1초 per frame) ====

def make_ez_animation(output_path=None, frame_interval_ms=1000, time_step_us=100):
    mode_buttons.set_active(3)  # order: mag, Ex, Ey, Ez
    # time_step_us in microseconds, DT_MS in milliseconds
    step_frames = int(time_step_us / (DT_MS * 1000) + 1e-6)
    if step_frames < 1:
        step_frames = 1

    times = np.arange(0, min(1.0, (nt - 1) * DT_MS), DT_MS * step_frames)
    idxs = np.unique(np.round(times / DT_MS).astype(int))

    def anim_update(i):
        t_idx = idxs[i]
        t_sl.set_val(t_idx * DT_MS)
        update_view()
        return im_xy, im_yz, im_zx

    ani = FuncAnimation(fig, anim_update, frames=len(idxs), interval=frame_interval_ms, blit=False)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        suffix = output_path.suffix.lower()
        if suffix == ".mp4":
            if animation.writers.is_available("ffmpeg"):
                writer = animation.FFMpegWriter(fps=1, bitrate=1800)
                ani.save(str(output_path), writer=writer, dpi=150)
                print(f"Saved animation to: {output_path}")
            else:
                fallback = output_path.with_suffix(".gif")
                print("ffmpeg unavailable, falling back to GIF:", fallback)
                ani.save(str(fallback), writer="pillow", dpi=150, fps=1)
                print(f"Saved animation to: {fallback}")
        elif suffix == ".gif":
            ani.save(str(output_path), writer="pillow", dpi=150, fps=1)
            print(f"Saved animation to: {output_path}")
        else:
            print("Unsupported extension; saving as GIF.")
            fallback = output_path.with_suffix(".gif")
            ani.save(str(fallback), writer="pillow", dpi=150, fps=1)
            print(f"Saved animation to: {fallback}")

    return ani

# 예제: 저장 위치 지정 (mp4 시도, 실패 시 gif)
anim_file = DATA_FOLDER / "Ez_100us_step.mp4"
make_ez_animation(anim_file, frame_interval_ms=1000, time_step_us=100)

update_view()
plt.show()