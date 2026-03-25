"""
show_coords_shapes.py

Print shape and basic info for two E-field grid coordinate files.
Usage:
    cd D:\yongtae\neuron
    conda activate neuronconda
    python util\show_coords_shapes.py
"""
from pathlib import Path
import numpy as np

FILES = [
#    Path("efield") / "old_2025" / "E_field_grid_coords.npy",
#    Path("efield") / "400us_50Hz_10umspaing_100mA" / "E_field_grid_coords.npy",
#    Path("efield") / "400us_50Hz_10umspaing_100mA" / "1x_100mA_grad_Exdx_Eydy_Ezdz_2cycle.npy",
    Path("efield") / "400us_50Hz_10umspaing_100mA" / "E_field_2cycle.npy",
    Path("efield") / "old_2025" / "E_field_1cycle.npy",
]


def analyze(path: Path) -> None:
    print("=" * 60)
    print(f"File: {path}")
    if not path.exists():
        print("  -> NOT FOUND")
        return
    try:
        arr = np.load(str(path))
    except Exception as e:
        print(f"  -> failed to load: {e}")
        return

    print(f"  dtype: {arr.dtype}")
    print(f"  shape: {arr.shape}")

    # Heuristics for common file formats in this project
    if arr.ndim == 2 and arr.shape[1] == 3:
        coords_um = arr * 1e6
        xu = np.unique(np.round(coords_um[:, 0], 6))
        yu = np.unique(np.round(coords_um[:, 1], 6))
        zu = np.unique(np.round(coords_um[:, 2], 6))
        print(f"  Detected: coordinate list (N,3) in meters -> converted to μm for axes")
        print(f"  unique axes lengths: nx={xu.size}, ny={yu.size}, nz={zu.size}")
        expected = xu.size * yu.size * zu.size
        print(f"  expected (nx*ny*nz) = {expected}")
        print(f"  matches full grid: {expected == coords_um.shape[0]}")
    elif arr.ndim == 5 and arr.shape[0] == 3:
        nx, ny, nz, nt = arr.shape[1], arr.shape[2], arr.shape[3], arr.shape[4]
        print(f"  Detected: gradient array (3, nx, ny, nz, nt)")
        print(f"  nx={nx}, ny={ny}, nz={nz}, nt={nt}")
    elif arr.ndim == 3 and arr.shape[0] == 3:
        n_spatial, nt = arr.shape[1], arr.shape[2]
        print(f"  Detected: E-field array (3, N_spatial, T)")
        print(f"  N_spatial={n_spatial}, T={nt}")
    else:
        print("  Unknown array layout; please inspect manually.")


if __name__ == '__main__':
    for p in FILES:
        analyze(p)
    print("=" * 60)
