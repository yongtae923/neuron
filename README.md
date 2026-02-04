# Neuron tES Simulation

A simulation pipeline for analyzing neuronal responses to transcranial electrical stimulation (tES) using NEURON simulator with Ansys-calculated electric field data.

## Contributors

- **Yongtae Kim** (tae.yongtae.kim@gmail.com)
- **Yeeun Seo** (seoyeeun1001@gmail.com)

## Initial Setup

### 1. Install WSL (Windows users)

```bash
wsl --install
```

### 2. Create Conda Environment

```bash
conda create -n neuronconda python=3.10
conda activate neuronconda
```

### 3. Install Dependencies

```bash
pip install neuron numpy pandas matplotlib jupyter
```

## Data (not in Git)

These folders are in `.gitignore` because they are large. You need them on each machine where you run the project.

| Folder         | Contents              | How to get it |
|----------------|-----------------------|----------------|
| `efield/`      | Ansys Ex,Ey,Ez txt    | Export from Ansys, or copy from another PC. |
| `efield_npy/`  | `E_field_*cycle.npy`, `E_field_grid_coords.npy` | Run `python extract_xyz.py` (needs `efield/`), or copy from another PC. |
| `allen_model/` | Allen cell SWC + mod  | Download via Allen SDK / portal, or copy from another PC. |

Scripts that need E-field expect either `efield/` (for extract_xyz) or `efield_npy/` (e.g. `plot_efield.py` can be pointed at `efield_npy/` by changing paths in the script if you use that folder).

## Using on multiple computers

1. **Code**: Clone the repo on each PC (`git clone ...`). Code and small files stay in sync via `git pull`.
2. **Data**: On each PC, put the same data in the repo root:
   - **Option A** – Copy once: Copy `efield/`, `efield_npy/`, and `allen_model/` from one PC to the others (USB, shared folder, or cloud drive).
   - **Option B** – Shared drive: Keep these folders on a shared drive (OneDrive, Dropbox, NAS) and use the same path on each PC (e.g. symlink or set `EFIELD_ROOT` in a small config and point scripts there).
   - **Option C** – Regenerate where possible: On a new PC, install Ansys/Allen sources and run `extract_xyz.py` (and Allen download) again if you have the raw sources.

After the first copy, you only need to re-copy or re-run when the data changes.

## Usage

1. Generate 40-cycle E-field data:
   ```bash
   jupyter notebook Singlecycletoncycle_npyextractor.ipynb
   ```

2. Extract coordinates:
   ```bash
   python extract_coords.py
   ```

3. Run simulation:
   ```bash
   python simulate_tES.py
   ```

Results are saved in `simulate_tES_output/` directory.
