"""
Plot Ez field in XY plane at a fixed z coordinate and time.

Data source:
- efield/400us_50Hz_10umspaing_100mA/E_field_2cycle.npy
- efield/400us_50Hz_10umspaing_100mA/E_field_grid_coords.npy

Displays: XY slice at z=600um, t=0.2ms, with Ez as color
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATASET = "400us_50Hz_10umspaing_100mA"
BASE_FOLDER = Path(r"D:\yongtae\neuron\efield")
DATA_FOLDER = BASE_FOLDER / DATASET

E_PATH = DATA_FOLDER / "E_field_2cycle.npy"
C_PATH = DATA_FOLDER / "E_field_grid_coords.npy"

DT_MS = 0.05
TARGET_TIME_MS = 0.2
TARGET_Z_UM = 600.0


def load_data():
    """Load E-field and coordinates."""
    E = np.load(E_PATH, mmap_mode="r")  # (3, N_spatial, T)
    coords = np.load(C_PATH)  # (N_spatial, 3), meters
    
    if E.ndim != 3 or E.shape[0] != 3:
        raise ValueError(f"Expected E shape (3, N_spatial, T), got {E.shape}")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Expected coords shape (N_spatial, 3), got {coords.shape}")
    
    return E, coords


def build_grid(coords):
    """Build x, y, z axes from coordinates."""
    coords_um = coords * 1e6
    xu = np.unique(np.round(coords_um[:, 0], 6))
    yu = np.unique(np.round(coords_um[:, 1], 6))
    zu = np.unique(np.round(coords_um[:, 2], 6))
    
    xu.sort()
    yu.sort()
    zu.sort()
    
    return xu, yu, zu


def get_z_index(zu, target_z):
    """Find closest z index to target_z."""
    idx = int(np.argmin(np.abs(zu - target_z)))
    actual_z = zu[idx]
    print(f"Target z: {target_z:.1f} um -> Closest z: {actual_z:.1f} um (index {idx})")
    return idx, actual_z


def get_time_index(nt, target_time_ms, dt_ms):
    """Find closest time index to target_time_ms."""
    t_idx = int(round(target_time_ms / dt_ms))
    t_idx = max(0, min(nt - 1, t_idx))
    actual_time = t_idx * dt_ms
    print(f"Target time: {target_time_ms:.2f} ms -> Closest time: {actual_time:.2f} ms (index {t_idx})")
    return t_idx, actual_time


def extract_xy_slice(E, coords, xu, yu, zu, t_idx, z_idx):
    """Extract XY slice at given time and z index."""
    n_spatial = coords.shape[0]
    nx, ny, nz = len(xu), len(yu), len(zu)
    
    if nx * ny * nz != n_spatial:
        raise ValueError(f"Grid mismatch: nx*ny*nz={nx*ny*nz}, N_spatial={n_spatial}")
    
    # Ez component at time t_idx
    ez_flat = E[2, :, t_idx]  # (N_spatial,)
    
    # Convert to meters for comparison
    coords_um = coords * 1e6
    x = coords_um[:, 0]
    y = coords_um[:, 1]
    z = coords_um[:, 2]
    
    # Find points at target z
    z_target = zu[z_idx]
    z_tol = 1.0  # tolerance in um
    z_mask = np.abs(z - z_target) < z_tol
    
    x_slice = x[z_mask]
    y_slice = y[z_mask]
    ez_slice = ez_flat[z_mask]
    
    print(f"Found {np.sum(z_mask)} points at z≈{z_target:.1f} um")
    
    return x_slice, y_slice, ez_slice


def plot_xy_slice(x, y, ez, title=""):
    """Plot XY slice with Ez as color, excluding specified rectangular region."""
    # Exclude rectangle x in [-50, 50], y in [-10, 15] (um)
    exclude_mask = (x >= -50.0) & (x <= 50.0) & (y >= -10.0) & (y <= 15.0)
    include_mask = ~exclude_mask

    x_plot = x[include_mask]
    y_plot = y[include_mask]
    ez_plot = ez[include_mask]

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot with Ez as color
    sc = ax.scatter(x_plot, y_plot, c=ez_plot, cmap="RdBu_r", s=50, edgecolors='k', linewidth=0.5)
    
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Ez (V/m)")
    
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("y (μm)")
    ax.set_title(title if title else f"Ez at z=600 μm, t=0.2 ms (XY Plane)")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Optional: show exclusion region edges for reference
    rect = plt.Rectangle((-50, -10), 100, 25, edgecolor='grey', facecolor='none', linestyle='--', linewidth=1)
    ax.add_patch(rect)
    
    plt.tight_layout()
    return fig, ax


def main():
    print("Loading data...")
    E, coords = load_data()
    print(f"E shape: {E.shape}")
    print(f"Coords shape: {coords.shape}")
    
    print("\nBuilding grid...")
    xu, yu, zu = build_grid(coords)
    print(f"Grid: nx={len(xu)}, ny={len(yu)}, nz={len(zu)}")
    
    print(f"\nFinding time index for t={TARGET_TIME_MS} ms...")
    t_idx, actual_time = get_time_index(E.shape[2], TARGET_TIME_MS, DT_MS)
    
    print(f"\nFinding z index for z={TARGET_Z_UM} um...")
    z_idx, actual_z = get_z_index(zu, TARGET_Z_UM)
    
    print(f"\nExtracting XY slice...")
    x_slice, y_slice, ez_slice = extract_xy_slice(E, coords, xu, yu, zu, t_idx, z_idx)
    
    print(f"\nPlotting...")
    title = f"Ez field at z={actual_z:.0f} μm, t={actual_time:.2f} ms"
    fig, ax = plot_xy_slice(x_slice, y_slice, ez_slice, title=title)
    
    plt.show()


if __name__ == "__main__":
    main()
