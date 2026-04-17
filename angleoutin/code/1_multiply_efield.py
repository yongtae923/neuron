"""
Create scaled E-field files from 1_E_field_1cycle.npy.

Input:
	- angleoutin/data/30V_OUT10_IN20_CI/1_E_field_1cycle.npy

Output:
	- angleoutin/data/30V_OUT10_IN20_CI/1_E_field_1cycle_2x.npy
	- angleoutin/data/30V_OUT10_IN20_CI/1_E_field_1cycle_10x.npy
"""

from __future__ import annotations

from pathlib import Path
import os
import numpy as np
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
CASE_NAME = os.environ.get("ANGLEOUTIN_CASE", "30V_OUT10_IN20_CI")
DATA_DIR = SCRIPT_DIR.parent / "data" / CASE_NAME

INPUT_PATH = DATA_DIR / "1_E_field_1cycle.npy"
OUTPUT_2X_PATH = DATA_DIR / "1_E_field_1cycle_2x.npy"
OUTPUT_10X_PATH = DATA_DIR / "1_E_field_1cycle_10x.npy"


def main() -> None:
		print("=" * 60)
		print("E-field multiply start")
		print("=" * 60)
		print(f"Case: {CASE_NAME}")
		print(f"Input: {INPUT_PATH}")

		if not INPUT_PATH.exists():
				raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

		print("Loading input npy...")
		efield = np.load(INPUT_PATH)
		print(f"Loaded shape: {efield.shape}, dtype: {efield.dtype}")

		jobs = [
				(2.0, OUTPUT_2X_PATH),
				(10.0, OUTPUT_10X_PATH),
		]

		for scale, out_path in tqdm(jobs, desc="Saving scaled files", unit="file", ncols=80):
			np.save(out_path, efield * scale)
			print(f"Saved ({scale:g}x): {out_path}")

		print("Done.")


if __name__ == "__main__":
		main()
