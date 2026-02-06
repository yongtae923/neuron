# util.py
"""
Utility to inspect the structure of allen_*_multipos_results.npy saved by simulate_allen.py.
"""

import os
import sys
import numpy as np


def describe_array(arr, name, indent=2):
    """Print shape, dtype, min/max for ndarray."""
    prefix = " " * indent
    lines = [
        f"{prefix}{name}: ndarray",
        f"{prefix}  shape: {arr.shape}",
        f"{prefix}  dtype: {arr.dtype}",
    ]
    if arr.size > 0 and np.issubdtype(arr.dtype, np.floating):
        lines.append(f"{prefix}  min: {np.min(arr):.4g}, max: {np.max(arr):.4g}")
    elif arr.size > 0 and arr.ndim == 1 and len(arr) <= 20:
        lines.append(f"{prefix}  sample: {arr.tolist()}")
    elif arr.size > 0 and arr.ndim == 1:
        lines.append(f"{prefix}  sample (first 5): {arr[:5].tolist()}")
    return "\n".join(lines)


def inspect_multipos_npy(filepath):
    """
    Load a multipos_results.npy file (saved with allow_pickle=True) and print its structure.
    """
    if not os.path.isfile(filepath):
        print(f"File not found: {filepath}")
        return

    data = np.load(filepath, allow_pickle=True)

    # np.save of a dict returns 0-dim array with dtype=object
    if isinstance(data, np.ndarray) and data.ndim == 0:
        data = data.item()

    if not isinstance(data, dict):
        print(f"Root type: {type(data)}")
        if hasattr(data, "shape"):
            print(f"  shape: {data.shape}, dtype: {data.dtype}")
        return

    print("=" * 60)
    print("Structure of multipos_results.npy")
    print("=" * 60)
    print(f"File: {filepath}\n")
    print("Root: dict with keys:", list(data.keys()))
    print()

    for key in sorted(data.keys()):
        val = data[key]
        print(f"[{key}]")
        if isinstance(val, np.ndarray):
            print(describe_array(val, key, indent=2))
        elif isinstance(val, dict):
            print("  dict:")
            for k, v in val.items():
                if isinstance(v, (list, tuple)):
                    print(f"    {k}: {type(v).__name__} len={len(v)} = {v}")
                elif isinstance(v, np.ndarray):
                    print(describe_array(v, k, indent=4))
                else:
                    print(f"    {k}: {type(v).__name__} = {v}")
        elif isinstance(val, (list, tuple)):
            print(f"  {type(val).__name__} len={len(val)} = {val}")
        else:
            print(f"  {type(val).__name__} = {val}")
        print()

    # Summary
    if "positions_um" in data and "time_ms" in data:
        pos = data["positions_um"]
        tms = data["time_ms"]
        print("Summary:")
        print(f"  positions: {pos.shape[0]} points x 3 (x,y,z um)")
        print(f"  time points: {len(tms)}")
        if "Vm_mV" in data:
            print(f"  Vm_mV: {data['Vm_mV'].shape} (positions x time)")
        if "vext_mV" in data:
            print(f"  vext_mV: {data['vext_mV'].shape} (positions x time)")
    print("=" * 60)


if __name__ == "__main__":
    default_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "output",
        "allen_529898751_multipos_results.npy"
    )
    path = sys.argv[1] if len(sys.argv) > 1 else default_path
    inspect_multipos_npy(path)
