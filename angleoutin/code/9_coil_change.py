# D:\yongtae\neuron\angleoutin\code\9_coil_change.py

"""
WSL 환경에서 사용합니다.

Run angleoutin pipeline in order for multiple cases:
    1_extract_xyz.py -> 1_multiply_efield.py -> 3_gen_gradient.py -> 7_allen_roi.py

Runs sequentially for:
    30V_OUT10_IN50, 30V_OUT50_IN50, SQ_OUT10_IN10, SQ_OUT50_IN10
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


CASE_NAMES = [
    "SQ_OUT50_IN50",
    "30V_OUT10_IN20_DI"
]


def run_step(script_path: Path, env: dict[str, str]) -> None:
    cmd = [sys.executable, str(script_path)]
    print(f"\n[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, env=env, check=True)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    steps = [
        script_dir / "1_extract_xyz.py",
        script_dir / "1_multiply_efield.py",
        script_dir / "3_gen_gradient.py",
        script_dir / "7_allen_roi.py",
    ]

    for s in steps:
        if not s.exists():
            raise FileNotFoundError(f"Missing script: {s}")

    env = os.environ.copy()

    print("=" * 60)
    print("Angleoutin sequential pipeline (multi-case)")
    print("=" * 60)
    print(f"Cases: {', '.join(CASE_NAMES)}")
    print("Order: 1_extract_xyz.py -> 1_multiply_efield.py -> 3_gen_gradient.py -> 7_allen_roi.py")

    for case_name in CASE_NAMES:
        print(f"\n{'='*60}")
        print(f"Running case: {case_name}")
        print(f"{'='*60}")

        env["ANGLEOUTIN_CASE"] = case_name

        for step in steps:
            run_step(step, env)

    print("\nAll cases completed.")


if __name__ == "__main__":
    main()
