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
