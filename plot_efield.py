# plot_efield.py
"""
3D E-field quiver at 0.2 ms: npy as-is (no unit conversion), top N_TOP points by |E|.
Exclude points inside coil region [um]. Axes range = full x,y,z from npy.
Saves capture to plot/efield_3d_0.2ms.png; if SHOW_3D=True, opens interactive 3D window (rotate/zoom).
"""

import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")

import numpy as np
import matplotlib
# Agg = save only, no window. Omit for interactive 3D (e.g. TkAgg/Qt5Agg).
SHOW_3D = True
if not SHOW_3D:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
E_FIELD_VALUES_FILE = os.path.join(SCRIPT_DIR, "efield", "E_field_4cycle.npy")
E_GRID_COORDS_FILE = os.path.join(SCRIPT_DIR, "efield", "E_field_grid_coords.npy")
PLOT_DIR = os.path.join(SCRIPT_DIR, "plot")
DT_MS = 0.05
TIME_MS = 0.2
N_TOP = 10000
# Coil region [um]: exclude points inside this box from the plot
COIL_X_MIN, COIL_X_MAX = -79.5, 79.5
COIL_Y_MIN, COIL_Y_MAX = -32.0, 32.0
COIL_Z_MIN, COIL_Z_MAX = 498.0, 1502.0
# Arrow length: scale so typical arrow ~ 2% of spatial extent (shorter = less clutter)
ARROW_LENGTH_FRAC = 0.05


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    E = np.load(E_FIELD_VALUES_FILE)
    coords_m = np.load(E_GRID_COORDS_FILE)
    coords_um = coords_m * 1e6

    n_comp, n_space, n_time = E.shape
    if n_comp >= 3:
        Ex, Ey, Ez = E[0], E[1], E[2]
    else:
        Ex, Ez = E[0], E[1]
        Ey = np.zeros_like(Ex)
    Emag = np.sqrt(Ex**2 + Ey**2 + Ez**2)

    t_idx = int(round(TIME_MS / DT_MS))
    t_idx = max(0, min(t_idx, n_time - 1))
    actual_ms = t_idx * DT_MS

    ex = Ex[:, t_idx]
    ey = Ey[:, t_idx]
    ez = Ez[:, t_idx]
    emag = Emag[:, t_idx]

    x_um = coords_um[:, 0]
    y_um = coords_um[:, 1]
    z_um = coords_um[:, 2]
    inside_coil = (
        (x_um >= COIL_X_MIN) & (x_um <= COIL_X_MAX)
        & (y_um >= COIL_Y_MIN) & (y_um <= COIL_Y_MAX)
        & (z_um >= COIL_Z_MIN) & (z_um <= COIL_Z_MAX)
    )
    outside_coil_idx = np.where(~inside_coil)[0]
    emag_outside = emag[outside_coil_idx]
    top_local = np.argsort(emag_outside)[::-1][:N_TOP]
    top_idx = outside_coil_idx[top_local]
    x = coords_um[top_idx, 0]
    y = coords_um[top_idx, 1]
    z = coords_um[top_idx, 2]
    u = ex[top_idx]
    v = ey[top_idx]
    w = ez[top_idx]
    c = emag[top_idx]

    extent = max(np.ptp(coords_um[:, 0]), np.ptp(coords_um[:, 1]), np.ptp(coords_um[:, 2]))
    if extent <= 0:
        extent = 1.0
    typical = np.percentile(c, 90)
    if typical <= 0:
        typical = 1.0
    scale = (ARROW_LENGTH_FRAC * extent) / typical
    u_plot = u * scale
    v_plot = v * scale
    w_plot = w * scale

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    norm = Normalize(vmin=c.min(), vmax=c.max())
    sm = cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array(c)
    colors = sm.to_rgba(c)
    ax.quiver(x, y, z, u_plot, v_plot, w_plot, colors=colors, alpha=0.9, arrow_length_ratio=0.15)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_zlabel("z (um)")
    ax.set_xlim(coords_um[:, 0].min(), coords_um[:, 0].max())
    ax.set_ylim(coords_um[:, 1].min(), coords_um[:, 1].max())
    ax.set_zlim(coords_um[:, 2].min(), coords_um[:, 2].max())
    ax.set_title(f"E-field at t = {actual_ms} ms, top {N_TOP} by |E| (file units)")
    fig.colorbar(sm, ax=ax, shrink=0.6, label="|E| (file units)")
    ax.view_init(elev=25, azim=45)
    plt.tight_layout()
    fname = os.path.join(PLOT_DIR, "efield_3d_0.2ms.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved: {fname}")
    if SHOW_3D:
        plt.show()  # interactive 3D: drag to rotate, scroll to zoom (close window to exit)
    plt.close()


if __name__ == "__main__":
    main()
