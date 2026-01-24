# plot_3d_ez_time.py
# 3D 플롯: x, z, 시간축에서 Ez 성분을 색깔로 표시 (마이너스/플러스 구분)

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Headless mode
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

# 파일 경로
script_dir = os.path.dirname(os.path.abspath(__file__))
E_FIELD_VALUES_FILE = os.path.join(script_dir, 'E_field_40cycles.npy')
E_GRID_COORDS_FILE = os.path.join(script_dir, 'E_field_grid_coords.npy')

# 파라미터
TIME_STEP_US = 50.0
TIME_STEP_MS = TIME_STEP_US / 1000.0
TIME_RANGE_MS = [0.0, 0.5]  # 시간 범위 (ms)
Y_SLICE_VALUE = 42.0  # um
Y_SLICE_THICKNESS = 1.0  # um
DOWNSAMPLE_STEP = 1  # 공간 다운샘플링 (1 = 다운샘플링 없음)
TIME_DOWNSAMPLE = 1  # 시간 다운샘플링 (1 = 다운샘플링 없음)
EFIELD_UNIT = "mV/m"  # 표시 단위

print("Loading E-field data...")
e_values = np.load(E_FIELD_VALUES_FILE)  # (2, N_spatial, T)
grid_coords_m = np.load(E_GRID_COORDS_FILE)  # (N_spatial, 3)
print(f"E-field shape: {e_values.shape}, Coords shape: {grid_coords_m.shape}")

# 좌표를 um로 변환
coords_um = grid_coords_m * 1e6

# 시간 배열 생성
time_array_full = np.arange(0, e_values.shape[2]) * TIME_STEP_MS

# 시간 범위 필터링
time_mask = (time_array_full >= TIME_RANGE_MS[0]) & (time_array_full <= TIME_RANGE_MS[1])
time_array = time_array_full[time_mask]
time_indices = np.where(time_mask)[0]
n_time = len(time_array)

print(f"Time range: {TIME_RANGE_MS[0]:.3f} ~ {TIME_RANGE_MS[1]:.3f} ms ({n_time} time points)")

# y 슬라이스 필터링
half = Y_SLICE_THICKNESS / 2.0
mask_y = np.abs(coords_um[:, 1] - Y_SLICE_VALUE) <= half
coords_filtered = coords_um[mask_y]

print(f"Y slice filtering: y = {Y_SLICE_VALUE:.1f} +/- {half:.1f} um (points: {len(coords_filtered)})")

# Ez 데이터 추출 (y 슬라이스 필터링 후)
ez_all_time = e_values[1, mask_y, :]  # (N_filtered, T)

# 다운샘플링
if DOWNSAMPLE_STEP > 1:
    coords_filtered = coords_filtered[::DOWNSAMPLE_STEP]
    ez_all_time = ez_all_time[::DOWNSAMPLE_STEP, :]
    print(f"Spatial downsampling: {DOWNSAMPLE_STEP}x (points: {len(coords_filtered)})")

# 시간 다운샘플링
if TIME_DOWNSAMPLE > 1:
    ez_all_time = ez_all_time[:, ::TIME_DOWNSAMPLE]
    time_array = time_array[::TIME_DOWNSAMPLE]
    time_indices = time_indices[::TIME_DOWNSAMPLE]
    n_time = len(time_array)
    print(f"Time downsampling: {TIME_DOWNSAMPLE}x (time points: {n_time})")

# 시간 범위 필터링 적용
ez_all_time = ez_all_time[:, time_indices]

# 3D 플롯 데이터 생성
n_points = coords_filtered.shape[0]
total_points = n_points * n_time
print(f"Generating plot data... (spatial points: {n_points}, time points: {n_time}, total: {total_points:,} points)")

if total_points > 500000:
    print(f"WARNING: Too many points ({total_points:,}). Rendering may be slow.")

x_coords = []
time_coords = []
z_coords = []
ez_values = []

for i in tqdm(range(n_points), desc="Processing spatial points"):
    x = coords_filtered[i, 0]
    z = coords_filtered[i, 2]
    for t_idx in range(n_time):
        x_coords.append(x)
        time_coords.append(time_array[t_idx])
        z_coords.append(z)
        ez_values.append(ez_all_time[i, t_idx])

print("Converting to arrays...")
x_coords = np.array(x_coords)
time_coords = np.array(time_coords)
z_coords = np.array(z_coords)
ez_values = np.array(ez_values)

# 단위 변환 (V/m → 선택한 단위)
unit_scale = {"V/m": 1.0, "mV/m": 1000.0, "μV/m": 1e6, "V/mm": 0.001}.get(EFIELD_UNIT, 1.0)
ez_values_plot = ez_values * unit_scale

# threshold 설정
threshold_vm = 0.00001  # V/m 기준
threshold = threshold_vm * unit_scale
abs_ez_values = np.abs(ez_values_plot)
mask_above_threshold = abs_ez_values >= threshold

x_coords_filtered = x_coords[mask_above_threshold]
time_coords_filtered = time_coords[mask_above_threshold]
z_coords_filtered = z_coords[mask_above_threshold]
ez_values_plot_filtered = ez_values_plot[mask_above_threshold]

print(f"Points above threshold: {len(x_coords_filtered):,}")

# 플롯 생성
print("Creating 3D plot...")
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Diverging colormap 사용 (마이너스/플러스 구분)
# RdBu_r: 빨강(플러스) - 파랑(마이너스)
vmax = np.max(np.abs(ez_values_plot_filtered)) if len(ez_values_plot_filtered) > 0 else 1.0
vmin = -vmax

# 점 크기 설정
point_sizes = np.ones_like(ez_values_plot_filtered) * 10

# Scatter plot
sc = ax.scatter(x_coords_filtered, time_coords_filtered, z_coords_filtered,
                c=ez_values_plot_filtered, s=point_sizes,
                cmap='RdBu_r', alpha=0.6, vmin=vmin, vmax=vmax)

# 축 레이블
ax.set_xlabel('X (um)', fontsize=12, fontweight='bold')
ax.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
ax.set_zlabel('Z (um)', fontsize=12, fontweight='bold')

# 컬러바
cbar = plt.colorbar(sc, ax=ax, shrink=0.6, label=f'E_z ({EFIELD_UNIT})')
cbar.ax.tick_params(labelsize=10)

# 제목
title = f'E_z Component Distribution (x, z, time)\nTime: {TIME_RANGE_MS[0]:.3f} ~ {TIME_RANGE_MS[1]:.3f} ms, Y = {Y_SLICE_VALUE:.1f} +/- {half:.1f} um'
ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()

# 저장
output_dir = os.path.join(script_dir, "visualize_efield_output")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"efield_3d_ez_time_{TIME_RANGE_MS[0]:.3f}_{TIME_RANGE_MS[1]:.3f}ms.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nSaved: {output_path}")
plt.close()

print("\nDone!")
