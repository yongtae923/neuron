# 2_generate_gradient.py
"""
Compute directional spatial gradients from E-field data (2-cycle).
Coil regions: 2 trapezoidal prisms (left and right).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath


SCRIPT_DIR = Path(__file__).resolve().parent.parent
INPUT_EFIELD_PATH = SCRIPT_DIR / "efield" / "400us_50Hz_10umspaing_100mA" / "E_field_2cycle.npy"
INPUT_COORDS_PATH = SCRIPT_DIR / "efield" / "400us_50Hz_10umspaing_100mA" / "E_field_grid_coords.npy"
OUTPUT_DIR = SCRIPT_DIR / "efield" / "400us_50Hz_10umspaing_100mA"
OUTPUT_PATH = OUTPUT_DIR / "grad_2cycle.npy"

# Coil regions: 2 trapezoidal prisms (x-z plane), y [-5, 10] um
# Left/Right trapezoid definitions updated per request:
# Right trapezoid (x-z): [(45,800), (45,590), (5,500), (5,800)] um
# Left trapezoid is the same with x negated: [(-45,800), (-45,590), (-5,500), (-5,800)] um
COIL_RIGHT_TRAPEZOID_XZ = np.array([
    [45.0e-6, 800.0e-6],
    [45.0e-6, 590.0e-6],
    [5.0e-6, 500.0e-6],
    [5.0e-6, 800.0e-6],
])

COIL_LEFT_TRAPEZOID_XZ = np.array([
    [-45.0e-6, 800.0e-6],
    [-45.0e-6, 590.0e-6],
    [-5.0e-6, 500.0e-6],
    [-5.0e-6, 800.0e-6],
])

# Y range updated to [-6.25, 11.75] um
COIL_Y = [-6.25e-6, 11.75e-6]


def _build_axis_and_indices(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build unique sorted x/y/z axes and per-point integer indices.

    Returns:
        x_axis, y_axis, z_axis, ix, iy, iz
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Unexpected coords shape: {coords.shape}, expected (N_spatial, 3)")

    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    x_axis = np.unique(x)
    y_axis = np.unique(y)
    z_axis = np.unique(z)

    ix = np.searchsorted(x_axis, x)
    iy = np.searchsorted(y_axis, y)
    iz = np.searchsorted(z_axis, z)

    n_spatial = coords.shape[0]
    expected = x_axis.size * y_axis.size * z_axis.size
    if expected != n_spatial:
        raise ValueError(
            "Grid is not a full Cartesian product or has duplicates. "
            f"Expected Nx*Ny*Nz={expected}, but N_spatial={n_spatial}."
        )

    occupied = np.zeros((x_axis.size, y_axis.size, z_axis.size), dtype=np.uint8)
    occupied[ix, iy, iz] = 1
    if not np.all(occupied):
        raise ValueError("Detected missing grid points after index mapping.")

    return x_axis, y_axis, z_axis, ix, iy, iz


def _build_surface_shell_mask(coil_mask: np.ndarray) -> np.ndarray:
    """
    Build 6-neighbor one-voxel shell right outside the coil mask.

    Returns:
        shell_mask: True at points adjacent to coil interior (outside only).
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


def _build_coil_mask(x_axis: np.ndarray, y_axis: np.ndarray, z_axis: np.ndarray) -> np.ndarray:
    """
    Build 3D coil mask for 2 trapezoidal coil regions (in x-z plane).
    
    Returns:
        coil_mask: (nx, ny, nz) bool array, True inside coils.
    """
    nx, ny, nz = x_axis.size, y_axis.size, z_axis.size
    coil_mask = np.zeros((nx, ny, nz), dtype=bool)
    
    # Y range: [-5, 10] um
    y_coil_mask = (y_axis >= COIL_Y[0]) & (y_axis <= COIL_Y[1])
    
    # Create path objects for both trapezoids
    left_path = MplPath(COIL_LEFT_TRAPEZOID_XZ)
    right_path = MplPath(COIL_RIGHT_TRAPEZOID_XZ)
    
    # Create meshgrid for x-z plane
    xx, zz = np.meshgrid(x_axis, z_axis, indexing="ij")
    xz_points = np.column_stack([xx.ravel(), zz.ravel()])
    
    # Check points inside trapezoids
    left_inside = left_path.contains_points(xz_points).reshape(nx, nz)
    right_inside = right_path.contains_points(xz_points).reshape(nx, nz)
    
    # Combine both trapezoids (union)
    inside_any = left_inside | right_inside
    
    # Apply y constraint
    coil_mask[inside_any[:, None, :] & y_coil_mask[None, :, None]] = True
    
    return coil_mask


def main() -> None:
    if not INPUT_EFIELD_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_EFIELD_PATH}")
    if not INPUT_COORDS_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_COORDS_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Memory-map to avoid loading full E array in RAM.
    efield = np.load(INPUT_EFIELD_PATH, mmap_mode="r")  # (3, N_spatial, Nt)
    coords = np.load(INPUT_COORDS_PATH)  # (N_spatial, 3)

    if efield.ndim != 3 or efield.shape[0] != 3:
        raise ValueError(f"Unexpected E-field shape: {efield.shape}, expected (3, N_spatial, Nt)")
    if coords.shape[0] != efield.shape[1]:
        raise ValueError(f"Mismatch: coords N={coords.shape[0]} vs E-field spatial N={efield.shape[1]}")

    x_axis, y_axis, z_axis, ix, iy, iz = _build_axis_and_indices(coords)
    nx, ny, nz = x_axis.size, y_axis.size, z_axis.size
    nt = efield.shape[2]

    # 3D coil mask: True = inside coil (gradient set to 0)
    coil_mask = _build_coil_mask(x_axis, y_axis, z_axis)
    shell_mask = _build_surface_shell_mask(coil_mask)
    suppress_mask = coil_mask | shell_mask

    n_coil = int(np.sum(coil_mask))
    n_shell = int(np.sum(shell_mask))
    print(f"Coil interior: {n_coil} points masked to 0")
    print(f"Coil surface shell (outside, 1-voxel): {n_shell} points masked to 0")

    print(f"E-field shape: {efield.shape}")
    print(f"Grid shape: Nx={nx}, Ny={ny}, Nz={nz}, Nt={nt}")
    print("Computing: dEx/dx, dEy/dy, dEz/dz (edge_order=2)")

    # Single output file as requested.
    # Axis-0 meaning:
    #   0 -> dEx/dx
    #   1 -> dEy/dy
    #   2 -> dEz/dz
    grad_out = np.lib.format.open_memmap(
        OUTPUT_PATH,
        mode="w+",
        dtype=np.float32,
        shape=(3, nx, ny, nz, nt),
    )

    # Reuse buffers for performance and lower peak memory.
    ex_grid = np.empty((nx, ny, nz), dtype=np.float32)
    ey_grid = np.empty((nx, ny, nz), dtype=np.float32)
    ez_grid = np.empty((nx, ny, nz), dtype=np.float32)

    for t in range(nt):
        # Scatter flat spatial values into 3D grid.
        ex_grid[ix, iy, iz] = efield[0, :, t]
        ey_grid[ix, iy, iz] = efield[1, :, t]
        ez_grid[ix, iy, iz] = efield[2, :, t]

        # Directional derivatives only (as requested).
        grad_out[0, :, :, :, t] = np.gradient(ex_grid, x_axis, axis=0, edge_order=2).astype(np.float32, copy=False)
        grad_out[1, :, :, :, t] = np.gradient(ey_grid, y_axis, axis=1, edge_order=2).astype(np.float32, copy=False)
        grad_out[2, :, :, :, t] = np.gradient(ez_grid, z_axis, axis=2, edge_order=2).astype(np.float32, copy=False)

        # Suppress discontinuity spikes near coil boundary.
        # - interior: coil_mask
        # - immediate outside shell: shell_mask
        for c in range(3):
            grad_out[c, :, :, :, t][suppress_mask] = 0.0

        if (t + 1) % 10 == 0 or (t + 1) == nt:
            print(f"Processed time step: {t + 1}/{nt}")

    # Flush memory-mapped output.
    del grad_out

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
