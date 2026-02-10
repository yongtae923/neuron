"""
check_efield_scale.py

Quick sanity checks for E-field amplitude with and without the coil region.

Notes:
- E-field values are assumed to be in V/m.
- E-field coords file is assumed to be in meters (converted to um internally).
- Stats below are computed on |E| of COMPONENT values (not vector magnitude),
  matching previous behavior. Quantiles are estimated by random sampling to
  avoid loading the entire (3, N, T) array into RAM.
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
E_FIELD_VALUES_FILE = os.path.join(SCRIPT_DIR, "efield", "E_field_1cycle.npy")
E_GRID_COORDS_FILE = os.path.join(SCRIPT_DIR, "efield", "E_field_grid_coords.npy")

# Coil region bounds in um (same as simulate_three.py / simulate_allen.py)
COIL_X_MIN, COIL_X_MAX = -79.5, 79.5
COIL_Y_MIN, COIL_Y_MAX = -32.0, 32.0
COIL_Z_MIN, COIL_Z_MAX = 498.0, 1502.0


def _masked_absmax(E: np.ndarray, spatial_mask: np.ndarray, block_size: int = 200_000) -> float:
    """Compute exact max(abs(E)) over a spatial mask by chunking spatial axis."""
    if spatial_mask.ndim != 1:
        raise ValueError("spatial_mask must be 1D")

    n_comp, n_spatial, n_time = E.shape
    if spatial_mask.shape[0] != n_spatial:
        raise ValueError(f"mask length mismatch: {spatial_mask.shape[0]} vs {n_spatial}")

    absmax = 0.0
    for t in range(n_time):
        for s0 in range(0, n_spatial, block_size):
            s1 = min(n_spatial, s0 + block_size)
            m = spatial_mask[s0:s1]
            if not bool(np.any(m)):
                continue
            # E[:, s0:s1, t] is (n_comp, block)
            block_vals = E[:, s0:s1, t]
            # Apply boolean mask on the spatial axis (creates a smaller copy)
            vmax = float(np.max(np.abs(block_vals[:, m])))
            if vmax > absmax:
                absmax = vmax
    return absmax


def _sample_abs_values(
    E: np.ndarray,
    spatial_indices: np.ndarray,
    n_samples: int = 500_000,
    seed: int = 0,
) -> np.ndarray:
    """Randomly sample |E| values from (comp, spatial_indices, time)."""
    rng = np.random.default_rng(seed)
    n_comp, _, n_time = E.shape

    comps = rng.integers(0, n_comp, size=n_samples, dtype=np.int64)
    times = rng.integers(0, n_time, size=n_samples, dtype=np.int64)
    sp = rng.choice(spatial_indices, size=n_samples, replace=True)

    # Fancy indexing returns a dense array of length n_samples.
    vals = np.abs(E[comps, sp, times]).astype(np.float64, copy=False)
    return vals


def _print_stats(label: str, absmax: float, sample_abs: np.ndarray) -> None:
    p50, p99, p999 = (float(np.quantile(sample_abs, q)) for q in (0.50, 0.99, 0.999))
    print(f"\n[{label}]")
    print(f"  max|E|    = {absmax:.6g}  (exact)")
    print(f"  p50|E|    = {p50:.6g}  (sample)")
    print(f"  p99|E|    = {p99:.6g}  (sample)")
    print(f"  p99.9|E|  = {p999:.6g}  (sample)")

    # Rough sanity message (rule-of-thumb)
    if absmax > 1e6:
        print("  [WARN] max|E| >> 1e6 V/m. Unit/scale duplication is very likely.")


def main() -> None:
    # Memory-map to avoid loading multi-GB array into RAM
    E = np.load(E_FIELD_VALUES_FILE, mmap_mode="r")
    coords_m = np.load(E_GRID_COORDS_FILE)  # (N_spatial, 3) in meters
    coords_um = coords_m * 1e6

    print("E shape:", E.shape, "dtype:", E.dtype)
    print("coords shape:", coords_m.shape, "dtype:", coords_m.dtype, "(meters)")

    if E.ndim != 3:
        raise SystemExit(f"Unexpected E-field ndim: {E.ndim}, shape={E.shape}")
    if coords_um.ndim != 2 or coords_um.shape[1] != 3:
        raise SystemExit(f"Unexpected coords shape: {coords_um.shape}")
    if coords_um.shape[0] != E.shape[1]:
        raise SystemExit(f"N_spatial mismatch: coords={coords_um.shape[0]} vs E={E.shape[1]}")

    x = coords_um[:, 0]
    y = coords_um[:, 1]
    z = coords_um[:, 2]
    inside_coil = (
        (x >= COIL_X_MIN) & (x <= COIL_X_MAX)
        & (y >= COIL_Y_MIN) & (y <= COIL_Y_MAX)
        & (z >= COIL_Z_MIN) & (z <= COIL_Z_MAX)
    )
    outside_coil = ~inside_coil

    n_spatial = coords_um.shape[0]
    n_in = int(np.sum(inside_coil))
    n_out = int(np.sum(outside_coil))
    print(f"Coil mask: inside={n_in} ({100.0*n_in/n_spatial:.2f}%), outside={n_out} ({100.0*n_out/n_spatial:.2f}%)")

    # Precompute index arrays for sampling
    idx_all = np.arange(n_spatial, dtype=np.int64)
    idx_out = np.flatnonzero(outside_coil).astype(np.int64)
    idx_in = np.flatnonzero(inside_coil).astype(np.int64)

    # 1) Include coil region (all points)
    absmax_all = float(np.max(np.abs(E)))  # exact but may take time; memmap keeps RAM usage low
    sample_all = _sample_abs_values(E, idx_all, n_samples=500_000, seed=0)
    _print_stats("ALL points (coil included)", absmax_all, sample_all)

    # 2) Exclude coil region (outside only)
    absmax_out = _masked_absmax(E, outside_coil, block_size=200_000)
    sample_out = _sample_abs_values(E, idx_out, n_samples=500_000, seed=1)
    _print_stats("OUTSIDE coil (coil excluded)", absmax_out, sample_out)

    # (Optional) Inside coil stats for comparison
    if idx_in.size > 0:
        absmax_in = _masked_absmax(E, inside_coil, block_size=200_000)
        sample_in = _sample_abs_values(E, idx_in, n_samples=200_000, seed=2)
        _print_stats("INSIDE coil (reference)", absmax_in, sample_in)


if __name__ == "__main__":
    main()
