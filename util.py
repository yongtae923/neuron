# util.py
import numpy as np
import os
from neuron import h
from neuron.units import um, ms, mV
from model_allen_neuron import AllenNeuronModel
import math

h.load_file("stdrun.hoc")

# --- Configuration ---
CELL_ID = "529898751"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
E_FIELD_VALUES_FILE = os.path.join(SCRIPT_DIR, "efield", "E_field_1cycle.npy")
E_GRID_COORDS_FILE = os.path.join(SCRIPT_DIR, "efield", "E_field_grid_coords.npy")

TIME_STEP_US = 50.0   # E-field data time step (us)
TIME_STEP_MS = TIME_STEP_US / 1000.0  # 0.05 ms
N_SPATIAL_POINTS = None # 로드 시 결정될 공간 그리드 지점 수

# --- Load E-field data ---

try:
    E_field_values = np.load(E_FIELD_VALUES_FILE)
    print(f"E-field values loaded: {E_field_values.shape}")
    E_grid_coords_M = np.load(E_GRID_COORDS_FILE)
    # Ansys 좌표는 미터(M) 단위이므로, NEURON 단위(μM)로 변환 (1 M = 1e6 uM)
    E_grid_coords_UM = E_grid_coords_M * 1e6
    N_SPATIAL_POINTS = E_grid_coords_UM.shape[0]
    print(f"E-field coords loaded and converted to um: {E_grid_coords_UM.shape}")
except Exception as e:
    print(f"Error: Cannot load E-field values file: {e}")
    exit()

# --- Find nearest spatial index ---
def find_nearest_spatial_index(x_um, y_um, z_um, grid_coords_um):
    """주어진 (x, y, z)에 가장 가까운 그리드 지점의 인덱스를 반환합니다."""
    
    # 3D 유클리드 거리 계산 (NumPy 브로드캐스팅 사용)
    # (x, y, z) - E_grid_coords_UM [Shape (N_spatial, 3)]
    target_coord = np.array([x_um, y_um, z_um])
    
    # 거리 제곱 계산: (x-x0)^2 + (y-y0)^2 + (z-z0)^2
    distances_sq = np.sum((grid_coords_um - target_coord)**2, axis=1)
    # 거리가 가장 작은 인덱스를 찾습니다.
    nearest_index = np.argmin(distances_sq)
    return nearest_index

# --- Global spatial map ---
global_spatial_map = {}

