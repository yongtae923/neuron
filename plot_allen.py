"""
plot_allen.py

Plot Allen multi-position simulation results saved by simulate_allen.py.

- Input: allen_*_multipos_results.npy (dict saved via np.save(..., allow_pickle=True))
- Output: 3D scatter plots of Vm over space for each time frame
- Visualization: 0 mV is transparent; farther from 0 => more opaque/colored
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from tqdm import tqdm


@dataclass
class PlotConfig:
    cmap: str = "RdBu_r"
    gamma: float = 1.0  # alpha = |2x-1|**gamma; bigger => more transparent near center
    point_size: float = 8.0
    elev: float = 25.0
    azim: float = 45.0
    dpi: int = 150


def _find_default_input(script_dir: Path) -> Optional[Path]:
    """Pick the most recently modified allen_*_multipos_results.npy in ./output."""
    outdir = script_dir / "output"
    if not outdir.is_dir():
        return None
    candidates = list(outdir.glob("allen_*_multipos_results.npy"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_results(path: Path) -> dict:
    data = np.load(str(path), allow_pickle=True)
    if isinstance(data, np.ndarray) and data.ndim == 0:
        data = data.item()
    if not isinstance(data, dict):
        raise TypeError(f"Unexpected root type in npy: {type(data)}")
    return data


def _transparent_center_cmap(base_cmap: str, gamma: float = 1.0, n: int = 256) -> ListedColormap:
    """
    Return a colormap where the center (0) is transparent and opacity increases
    with distance from center.

    alpha = |2x-1|**gamma for colormap coordinate x in [0,1]
    """
    base = plt.get_cmap(base_cmap, n)
    xs = np.linspace(0.0, 1.0, n)
    rgba = base(xs)
    rgba[:, 3] = np.abs(2 * xs - 1) ** float(gamma)
    return ListedColormap(rgba, name=f"{base.name}_transparent_center")


def save_vm_plots(
    positions_um: np.ndarray,
    time_ms: np.ndarray,
    Vm_mV: np.ndarray,
    output_dir: Path,
    prefix: str,
    cfg: PlotConfig,
    only_t_index: Optional[int] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    Vm_min, Vm_max = float(np.min(Vm_mV)), float(np.max(Vm_mV))
    Vm_abs_max = max(abs(Vm_min), abs(Vm_max))
    norm = TwoSlopeNorm(vmin=-Vm_abs_max, vcenter=0.0, vmax=Vm_abs_max)
    cmap = _transparent_center_cmap(cfg.cmap, gamma=cfg.gamma, n=256)

    idxs = range(len(time_ms)) if only_t_index is None else [int(only_t_index)]
    for t_idx in tqdm(list(idxs), desc="Save Vm 3D plots", unit="frame", ncols=100):
        t = float(time_ms[t_idx])
        values = Vm_mV[:, t_idx]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        sc = ax.scatter(
            positions_um[:, 0],
            positions_um[:, 1],
            positions_um[:, 2],
            c=values,
            cmap=cmap,
            norm=norm,
            s=cfg.point_size,
            alpha=1.0,  # use colormap alpha for transparency
            linewidths=0,
        )

        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        ax.set_zlabel("z (um)")
        ax.set_title(f"Membrane potential at t={t:.2f} ms")
        fig.colorbar(sc, ax=ax, shrink=0.6, label="Vm (mV)")

        ax.view_init(elev=cfg.elev, azim=cfg.azim)

        plt.tight_layout()
        fname = output_dir / f"{prefix}_Vm_3d_{t:.2f}ms.png"
        plt.savefig(str(fname), dpi=cfg.dpi, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=None, help="Path to allen_*_multipos_results.npy")
    ap.add_argument("--outdir", type=str, default=None, help="Output directory (default: ./plot)")
    ap.add_argument("--prefix", type=str, default=None, help="Output filename prefix (default: cell_id or 'allen')")
    ap.add_argument("--t-index", type=int, default=None, help="Only plot a single time index")

    ap.add_argument("--cmap", type=str, default="RdBu_r")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--point-size", type=float, default=8.0)
    ap.add_argument("--elev", type=float, default=25.0)
    ap.add_argument("--azim", type=float, default=45.0)
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    input_path = Path(args.input) if args.input else _find_default_input(script_dir)
    if input_path is None:
        raise SystemExit("No input provided and no default file found in ./output (allen_*_multipos_results.npy).")
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    data = _load_results(input_path)
    if "positions_um" not in data or "time_ms" not in data or "Vm_mV" not in data:
        raise SystemExit(f"Missing required keys in npy. Found keys: {list(data.keys())}")

    positions_um = np.asarray(data["positions_um"])
    time_ms = np.asarray(data["time_ms"])
    Vm_mV = np.asarray(data["Vm_mV"])

    outdir = Path(args.outdir) if args.outdir else (script_dir / "plot")
    prefix = args.prefix or str(data.get("cell_id", "allen"))
    if not prefix.startswith("allen_"):
        prefix = f"allen_{prefix}"

    cfg = PlotConfig(
        cmap=args.cmap,
        gamma=args.gamma,
        point_size=args.point_size,
        elev=args.elev,
        azim=args.azim,
        dpi=args.dpi,
    )

    save_vm_plots(
        positions_um=positions_um,
        time_ms=time_ms,
        Vm_mV=Vm_mV,
        output_dir=outdir,
        prefix=prefix,
        cfg=cfg,
        only_t_index=args.t_index,
    )

    print(f"Done. Plots in: {outdir}")


if __name__ == "__main__":
    main()

