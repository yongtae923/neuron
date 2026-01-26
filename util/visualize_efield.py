import argparse
import os
import sys

import numpy as np
from tqdm import tqdm
import matplotlib

# DISPLAY 환경 변수 확인하여 백엔드 자동 선택
def setup_matplotlib_backend():
    """DISPLAY가 사용 가능하면 interactive 모드, 없으면 Agg 백엔드 사용"""
    # Windows 환경에서는 DISPLAY 무시 (자동으로 Windows 백엔드 사용)
    is_windows = sys.platform.startswith('win')
    if is_windows:
        # Windows에서는 기본 백엔드 사용 (TkAgg 등)
        try:
            matplotlib.use('TkAgg')
            print("✅ Windows 환경: Interactive 모드 활성화 (TkAgg)")
            return True
        except Exception:
            try:
                matplotlib.use('Qt5Agg')
                print("✅ Windows 환경: Interactive 모드 활성화 (Qt5Agg)")
                return True
            except Exception:
                matplotlib.use('Agg')
                print("⚠️  Windows 환경: Headless 모드로 전환")
                return False
    
    # Linux/WSL 환경에서만 DISPLAY 확인
    display = os.environ.get('DISPLAY')
    if display:
        # TkAgg 백엔드 시도 (Windows 11 WSLg에서 작동)
        try:
            matplotlib.use('TkAgg')
            # 실제로 연결 가능한지 테스트 (matplotlib import 후)
            import matplotlib.pyplot as plt_test
            fig_test = plt_test.figure()
            plt_test.close(fig_test)
            print(f"✅ Interactive 모드 활성화 (TkAgg, DISPLAY={display})")
            return True
        except Exception as e:
            # TkAgg 실패 시 Qt5Agg 시도
            try:
                matplotlib.use('Qt5Agg')
                import matplotlib.pyplot as plt_test
                fig_test = plt_test.figure()
                plt_test.close(fig_test)
                print(f"✅ Interactive 모드 활성화 (Qt5Agg, DISPLAY={display})")
                return True
            except Exception:
                # 모든 interactive 백엔드 실패 시 Agg 사용
                print(f"⚠️  DISPLAY={display}가 설정되어 있지만 연결할 수 없습니다.")
                print(f"   오류: {str(e)[:100] if 'e' in locals() else 'Unknown error'}")
                print("   → Headless 모드로 전환합니다 (파일 저장만 가능)")
                matplotlib.use('Agg')
                return False
    else:
        # DISPLAY가 없으면 Agg 사용
        matplotlib.use('Agg')
        print("ℹ️  Headless 모드 (파일 저장만 가능, DISPLAY 없음)")
        return False

HAS_DISPLAY = setup_matplotlib_backend()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


DEFAULT_TIME_STEP_US = 50.0

# SimplePyramidal 뉴런 파라미터
SOMA_DIAMETER = 30.0  # um
SOMA_LENGTH = 30.0    # um
AXON_LENGTH = 1000.0  # um

# 뉴런 위치 (simulate_tES.py의 N_POSITIONS와 동일)
NEURON_POSITIONS = [
    (-90.0, 42.0, 561.0),  # Neuron 1 (x, y, z in um)
    (0.0, 42.0, 561.0),    # Neuron 2
    (90.0, 42.0, 561.0)    # Neuron 3
]


def get_neuron_geometry():
    """
    SimplePyramidal 뉴런의 기하학적 정보를 반환합니다.
    Returns:
        list: 각 뉴런의 (soma_x, soma_z_start, soma_z_end, axon_x, axon_z_start, axon_z_end) 정보
    """
    neuron_geoms = []
    for x, y, z_center in NEURON_POSITIONS:
        soma_z_start = z_center - SOMA_LENGTH / 2.0
        soma_z_end = z_center + SOMA_LENGTH / 2.0
        axon_z_start = z_center - AXON_LENGTH / 2.0
        axon_z_end = z_center + AXON_LENGTH / 2.0
        neuron_geoms.append({
            'x': x,
            'y': y,
            'z_center': z_center,
            'soma_z_start': soma_z_start,
            'soma_z_end': soma_z_end,
            'axon_z_start': axon_z_start,
            'axon_z_end': axon_z_end,
            'soma_radius': SOMA_DIAMETER / 2.0
        })
    return neuron_geoms


def plot_neurons_on_3d(ax, units="um", time_value=-1.0):
    """
    3D 플롯에 뉴런을 그립니다 (x-z 평면, time=time_value 위치).
    
    Args:
        ax: matplotlib 3D axes
        units: 좌표 단위 ("um" or "m")
        time_value: 시간축 값 (plot_time_3d에서 사용)
    """
    neuron_geoms = get_neuron_geometry()
    
    # 단위 변환
    scale = 1.0 if units == "um" else 1e6
    
    for i, geom in enumerate(neuron_geoms):
        x = geom['x'] / scale if units == "m" else geom['x']
        z_center = geom['z_center'] / scale if units == "m" else geom['z_center']
        soma_z_start = geom['soma_z_start'] / scale if units == "m" else geom['soma_z_start']
        soma_z_end = geom['soma_z_end'] / scale if units == "m" else geom['soma_z_end']
        axon_z_start = geom['axon_z_start'] / scale if units == "m" else geom['axon_z_start']
        axon_z_end = geom['axon_z_end'] / scale if units == "m" else geom['axon_z_end']
        soma_radius = geom['soma_radius'] / scale if units == "m" else geom['soma_radius']
        
        # Soma를 원으로 그리기 (x-z 평면에서)
        # 원을 그리기 위해 각도를 사용
        theta = np.linspace(0, 2 * np.pi, 50)
        soma_x_circle = x + soma_radius * np.cos(theta)
        soma_z_circle = z_center + soma_radius * np.sin(theta)
        time_circle = np.full_like(theta, time_value)
        
        # 3D 플롯에서 원 그리기 (x-z 평면, y=time_value)
        # 뉴런이 더 잘 보이도록 선 두껍게, 완전 불투명, zorder 높게 설정
        ax.plot(soma_x_circle, time_circle, soma_z_circle, 'r-', linewidth=2, alpha=1.0, zorder=1000)
        # Soma 중심에 점 추가로 더 눈에 띄게
        ax.scatter([x], [time_value], [z_center], c='red', s=100, alpha=1.0, zorder=1001)
        
        # Axon을 선으로 그리기 (x-z 평면에서)
        axon_x_line = np.array([x, x])
        axon_z_line = np.array([axon_z_start, axon_z_end])
        time_line = np.array([time_value, time_value])
        
        ax.plot(axon_x_line, time_line, axon_z_line, 'r-', linewidth=3, alpha=1.0, zorder=1000)


def plot_neurons_on_2d(ax, projection, units="um"):
    """
    2D 플롯에 뉴런을 그립니다.
    
    Args:
        ax: matplotlib axes
        projection: "xy", "xz", "yz"
        units: 좌표 단위 ("um" or "m")
    """
    neuron_geoms = get_neuron_geometry()
    
    # 단위 변환
    scale = 1.0 if units == "um" else 1e6
    
    axis_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    a0, a1 = axis_map[projection]
    
    for i, geom in enumerate(neuron_geoms):
        x = geom['x'] / scale if units == "m" else geom['x']
        y = geom['y'] / scale if units == "m" else geom['y']
        z_center = geom['z_center'] / scale if units == "m" else geom['z_center']
        soma_z_start = geom['soma_z_start'] / scale if units == "m" else geom['soma_z_start']
        soma_z_end = geom['soma_z_end'] / scale if units == "m" else geom['soma_z_end']
        axon_z_start = geom['axon_z_start'] / scale if units == "m" else geom['axon_z_start']
        axon_z_end = geom['axon_z_end'] / scale if units == "m" else geom['axon_z_end']
        soma_radius = geom['soma_radius'] / scale if units == "m" else geom['soma_radius']
        
        if projection == "xz":
            # x-z 평면: soma는 원, axon은 선
            # Soma 원 - 뉴런이 더 잘 보이도록 선 두껍게, 완전 불투명, zorder 높게 설정
            theta = np.linspace(0, 2 * np.pi, 50)
            soma_x_circle = x + soma_radius * np.cos(theta)
            soma_z_circle = z_center + soma_radius * np.sin(theta)
            ax.plot(soma_x_circle, soma_z_circle, 'r-', linewidth=2, alpha=1.0, zorder=1000)
            # Soma 중심에 점 추가
            ax.scatter([x], [z_center], c='red', s=100, alpha=1.0, zorder=1001)
            
            # Axon 선
            ax.plot([x, x], [axon_z_start, axon_z_end], 'r-', linewidth=3, alpha=1.0, zorder=1000)
        elif projection == "xy":
            # x-y 평면: soma는 원, axon은 점 (z 방향이므로)
            theta = np.linspace(0, 2 * np.pi, 50)
            soma_x_circle = x + soma_radius * np.cos(theta)
            soma_y_circle = y + soma_radius * np.sin(theta)
            ax.plot(soma_x_circle, soma_y_circle, 'r-', linewidth=4, alpha=1.0, zorder=1000)
            # Soma 중심에 점 추가
            ax.scatter([x], [y], c='red', s=100, alpha=1.0, zorder=1001)
            # Axon은 z 방향이므로 x-y 평면에서는 점으로만 표시
            ax.plot(x, y, 'ro', markersize=10, alpha=1.0, zorder=1001)
        elif projection == "yz":
            # y-z 평면: soma는 원, axon은 선
            theta = np.linspace(0, 2 * np.pi, 50)
            soma_y_circle = y + soma_radius * np.cos(theta)
            soma_z_circle = z_center + soma_radius * np.sin(theta)
            ax.plot(soma_y_circle, soma_z_circle, 'r-', linewidth=4, alpha=1.0, zorder=1000)
            # Soma 중심에 점 추가
            ax.scatter([y], [z_center], c='red', s=100, alpha=1.0, zorder=1001)
            
            # Axon 선
            ax.plot([y, y], [axon_z_start, axon_z_end], 'r-', linewidth=3, alpha=1.0, zorder=1000)


def load_data(values_path, coords_path):
    print("📂 데이터 로딩 중...")
    e_values = np.load(values_path)  # (2, N_spatial, T)
    grid_coords_m = np.load(coords_path)  # (N_spatial, 3)
    print(f"✅ 데이터 로딩 완료: E-field shape={e_values.shape}, Coords shape={grid_coords_m.shape}")
    return e_values, grid_coords_m


def time_to_index(time_ms, time_step_us, t_max):
    time_step_ms = time_step_us / 1000.0
    idx = int(round(time_ms / time_step_ms))
    return max(0, min(t_max, idx))


def get_component(values, component, t_idx):
    if component == "ex":
        return values[0, :, t_idx]
    if component == "ez":
        return values[1, :, t_idx]
    if component == "mag":
        ex = values[0, :, t_idx]
        ez = values[1, :, t_idx]
        return np.sqrt(ex**2 + ez**2)
    raise ValueError(f"Unknown component: {component}")


def filter_slice(coords, values, axis, center, thickness):
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    half = thickness / 2.0
    mask = np.abs(coords[:, axis_index] - center) <= half
    return coords[mask], values[mask]


def filter_x_range(coords, values, x_min, x_max):
    """x 좌표 범위로 필터링"""
    mask = (coords[:, 0] >= x_min) & (coords[:, 0] <= x_max)
    return coords[mask], values[mask]


def downsample(coords, values, step):
    if step <= 1:
        return coords, values
    return coords[::step], values[::step]


def plot_3d(coords, values, units, title, output_path=None, auto_save=True, efield_unit="mV/m", show_neurons=False):
    print("🎨 플롯 생성 중...")
    with tqdm(total=6, desc="플롯 렌더링", unit="step", leave=False) as pbar:
        fig = plt.figure()
        pbar.update(1)
        ax: Axes3D = fig.add_subplot(111, projection="3d")  # type: ignore
        pbar.update(1)
        
        # 단위 변환 (V/m → 선택한 단위)
        unit_scale = {"V/m": 1.0, "mV/m": 1000.0, "μV/m": 1e6, "V/mm": 0.001}.get(efield_unit, 1.0)
        values_plot = values.copy() * unit_scale
        
        # threshold도 변환된 단위 기준으로 설정
        threshold_vm = 0.00008  # V/m 기준
        max_threshold_vm = 0.0001  # V/m 기준
        threshold = threshold_vm * unit_scale
        max_threshold = max_threshold_vm * unit_scale
        
        # values 복사 후 threshold 이하는 0으로 설정
        abs_values = np.abs(values_plot)
        mask_below_threshold = abs_values <= threshold
        values_plot[mask_below_threshold] = 0.0
        abs_values = np.abs(values_plot)
        
        # threshold 이하는 완전히 제외 (투명하게) - 필터링
        mask_above_threshold = abs_values > threshold
        coords_filtered = coords[mask_above_threshold]
        values_plot_filtered = values_plot[mask_above_threshold]
        abs_values_filtered = abs_values[mask_above_threshold]
        
        max_abs = np.max(abs_values_filtered) if len(abs_values_filtered) > 0 else 1.0
        
        # 초기화 (필터링된 데이터 기준)
        alpha_values = np.zeros_like(values_plot_filtered)
        point_sizes = np.ones_like(values_plot_filtered) * 10
        
        if max_abs > threshold:
            # threshold 이상의 값 처리 (이미 필터링됨)
            if max_abs > max_threshold:
                # max_threshold 이상은 최고로 진하게
                mask_max = abs_values_filtered >= max_threshold
                alpha_values[mask_max] = 1.0
                point_sizes[mask_max] = 400  # 매우 크게
                
                # threshold ~ max_threshold 사이는 정규화 (거의 투명하게)
                mask_mid = (abs_values_filtered > threshold) & (abs_values_filtered < max_threshold)
                if np.any(mask_mid):
                    normalized = (abs_values_filtered[mask_mid] - threshold) / (max_threshold - threshold)
                    alpha_values[mask_mid] = 0.05 + 0.1 * normalized  # 0.05 ~ 0.15 (거의 안 보임)
                    point_sizes[mask_mid] = 20 + 380 * normalized  # 20 ~ 400
            else:
                # max_threshold 미만인 경우 정규화 (거의 투명하게)
                normalized = np.clip((abs_values_filtered - threshold) / (max_abs - threshold), 0.0, 1.0)
                alpha_values = 0.05 + 0.1 * normalized  # 0.05 ~ 0.15 (거의 안 보임)
                point_sizes = 20 + 380 * normalized
        
        sc = ax.scatter(coords_filtered[:, 0], coords_filtered[:, 1], coords_filtered[:, 2], 
                        c=values_plot_filtered, s=point_sizes, 
                        cmap="viridis_r", alpha=alpha_values)
        pbar.update(1)
        ax.set_xlabel(f"x ({units})")
        ax.set_ylabel(f"y ({units})")
        ax.set_zlabel(f"z ({units})")
        pbar.update(1)
        fig.colorbar(sc, ax=ax, shrink=0.6, label=f"E-field ({efield_unit})")
        pbar.update(1)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        # 뉴런 그리기 (y=42um 위치에 x-z 평면, time=0으로 간주)
        if show_neurons:
            # 3D 플롯에서는 y 축이 있으므로, y=42um 위치에 x-z 평면으로 그리기
            # 하지만 plot_3d는 특정 시간 지점이므로 time 축이 없음
            # 대신 y=42um 위치에 x-z 평면으로 그리기
            neuron_geoms = get_neuron_geometry()
            scale = 1.0 if units == "um" else 1e6
            
            for i, geom in enumerate(neuron_geoms):
                x = geom['x'] / scale if units == "m" else geom['x']
                y = geom['y'] / scale if units == "m" else geom['y']
                z_center = geom['z_center'] / scale if units == "m" else geom['z_center']
                soma_z_start = geom['soma_z_start'] / scale if units == "m" else geom['soma_z_start']
                soma_z_end = geom['soma_z_end'] / scale if units == "m" else geom['soma_z_end']
                axon_z_start = geom['axon_z_start'] / scale if units == "m" else geom['axon_z_start']
                axon_z_end = geom['axon_z_end'] / scale if units == "m" else geom['axon_z_end']
                soma_radius = geom['soma_radius'] / scale if units == "m" else geom['soma_radius']
                
                # Soma를 원으로 그리기 (x-z 평면에서, y 고정)
                theta = np.linspace(0, 2 * np.pi, 50)
                soma_x_circle = x + soma_radius * np.cos(theta)
                soma_z_circle = z_center + soma_radius * np.sin(theta)
                y_circle = np.full_like(theta, y)
                
                # 뉴런이 더 잘 보이도록 선 두껍게, 완전 불투명, zorder 높게 설정
                ax.plot(soma_x_circle, y_circle, soma_z_circle, 'r-', linewidth=2, alpha=1.0, zorder=1000)
                # Soma 중심에 점 추가로 더 눈에 띄게
                ax.scatter([x], [y], [z_center], c='red', s=100, alpha=1.0, zorder=1001)
                
                # Axon을 선으로 그리기 (x-z 평면에서, y 고정)
                axon_x_line = np.array([x, x])
                axon_z_line = np.array([axon_z_start, axon_z_end])
                y_line = np.array([y, y])
                
                ax.plot(axon_x_line, y_line, axon_z_line, 'r-', linewidth=3, alpha=1.0, zorder=1000)
        
        plt.tight_layout()
        pbar.update(1)
    
    # 자동 캡처 저장 (3D 플롯의 경우, auto_save가 True이고 output_path가 없을 때)
    if auto_save and output_path is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualize_efield_output")
        os.makedirs(output_dir, exist_ok=True)
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"efield_3d_capture_{timestamp}.png")
    
    if output_path:
        print(f"💾 플롯 저장 중: {output_path}")
        with tqdm(total=1, desc="파일 저장", unit="file", leave=False) as pbar:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            pbar.update(1)
        print(f"✅ 3D 플롯 저장됨: {output_path}")
    
    if HAS_DISPLAY:
        plt.show(block=True)
    elif not output_path:
        print("⚠️  DISPLAY가 없어 플롯을 표시할 수 없습니다. --output 옵션으로 파일 저장하세요.")
    
    plt.close()


def plot_2d_quiver(coords, ex_values, ez_values, units, title, output_path=None, efield_unit="mV/m", 
                   y_slice_value=42.0, y_slice_thickness=1.0, downsample_step=5):
    """
    E-field 방향을 화살표와 색깔로 표현한 2D quiver 플롯 (x-z 평면)
    
    Args:
        coords: 좌표 배열 (N, 3) [x, y, z]
        ex_values: Ex 성분 값 (N,)
        ez_values: Ez 성분 값 (N,)
        units: 좌표 단위 ("um" or "m")
        title: 플롯 제목
        output_path: 출력 파일 경로
        efield_unit: E-field 단위
        y_slice_value: y 슬라이스 중심값 (기본: 42.0 um)
        y_slice_thickness: y 슬라이스 두께 (기본: 1.0 um)
        downsample_step: 다운샘플링 스텝 (기본: 5, 화살표가 너무 많으면 늘리기)
    """
    print("🎨 Quiver 플롯 생성 중...")
    
    # y 슬라이스 필터링
    half = y_slice_thickness / 2.0
    mask_y = np.abs(coords[:, 1] - y_slice_value) <= half
    coords_filtered = coords[mask_y]
    ex_filtered = ex_values[mask_y]
    ez_filtered = ez_values[mask_y]
    
    print(f"📍 Y 슬라이스 필터링: y = {y_slice_value:.1f} ± {half:.1f} {units} (포인트 수: {len(coords_filtered)})")
    
    # 다운샘플링
    if downsample_step > 1:
        coords_filtered = coords_filtered[::downsample_step]
        ex_filtered = ex_filtered[::downsample_step]
        ez_filtered = ez_filtered[::downsample_step]
        print(f"📉 다운샘플링 적용: {downsample_step}배 (포인트 수: {len(coords_filtered)})")
    
    # x-z 좌표 추출
    x_coords = coords_filtered[:, 0]
    z_coords = coords_filtered[:, 2]
    
    # 단위 변환 (V/m → 선택한 단위)
    unit_scale = {"V/m": 1.0, "mV/m": 1000.0, "μV/m": 1e6, "V/mm": 0.001}.get(efield_unit, 1.0)
    ex_plot = ex_filtered * unit_scale
    ez_plot = ez_filtered * unit_scale
    
    # 전기장 크기 계산 (색깔용)
    magnitude = np.sqrt(ex_plot**2 + ez_plot**2)
    
    # threshold 설정 (너무 작은 값은 제외)
    threshold_vm = 0.000001  # V/m 기준
    threshold = threshold_vm * unit_scale
    mask_above_threshold = magnitude >= threshold
    
    if np.sum(mask_above_threshold) == 0:
        print("⚠️  경고: threshold 이상의 전기장이 없습니다. threshold를 낮추세요.")
        return
    
    x_coords = x_coords[mask_above_threshold]
    z_coords = z_coords[mask_above_threshold]
    ex_plot = ex_plot[mask_above_threshold]
    ez_plot = ez_plot[mask_above_threshold]
    magnitude = magnitude[mask_above_threshold]
    
    print(f"📊 플롯할 포인트 수: {len(x_coords)}")
    
    # 플롯 생성
    with tqdm(total=6, desc="Quiver 플롯 렌더링", unit="step", leave=False) as pbar:
        fig, ax = plt.subplots(figsize=(12, 10))
        pbar.update(1)
        
        # 화살표 길이 정규화 (너무 길거나 짧지 않게)
        max_magnitude = np.max(magnitude) if len(magnitude) > 0 else 1.0
        # 화살표 길이를 적절하게 조정 (최대 길이를 데이터 범위의 일정 비율로)
        x_range = np.max(x_coords) - np.min(x_coords) if len(x_coords) > 1 else 1.0
        z_range = np.max(z_coords) - np.min(z_coords) if len(z_coords) > 1 else 1.0
        max_range = max(x_range, z_range)
        
        # 화살표 스케일 조정 (화살표가 너무 길지 않게)
        arrow_scale = max_range / (max_magnitude * 20) if max_magnitude > 0 else 1.0
        
        # Quiver 플롯 (화살표)
        quiver = ax.quiver(x_coords, z_coords, ex_plot, ez_plot, magnitude,
                          cmap='viridis', scale=1.0/arrow_scale, scale_units='xy',
                          angles='xy', width=0.003, alpha=0.8)
        pbar.update(1)
        
        # 컬러바 추가
        cbar = fig.colorbar(quiver, ax=ax, label=f'E-field Magnitude ({efield_unit})')
        pbar.update(1)
        
        # 축 레이블 및 제목
        ax.set_xlabel(f'X ({units})', fontsize=12)
        ax.set_ylabel(f'Z ({units})', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        pbar.update(1)
        
        # 그리드 및 동일 비율
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        pbar.update(1)
        
        plt.tight_layout()
        pbar.update(1)
    
    if output_path:
        print(f"💾 플롯 저장 중: {output_path}")
        with tqdm(total=1, desc="파일 저장", unit="file", leave=False) as pbar:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            pbar.update(1)
        print(f"✅ Quiver 플롯 저장됨: {output_path}")
    
    if HAS_DISPLAY:
        plt.show(block=True)
    elif not output_path:
        print("⚠️  DISPLAY가 없어 플롯을 표시할 수 없습니다. --output 옵션으로 파일 저장하세요.")
    
    plt.close()


def plot_2d(coords, values, units, projection, title, output_path=None, efield_unit="mV/m", show_neurons=False):
    print("🎨 플롯 생성 중...")
    axis_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    a0, a1 = axis_map[projection]
    with tqdm(total=6, desc="플롯 렌더링", unit="step", leave=False) as pbar:
        fig, ax = plt.subplots()
        pbar.update(1)
        
        # 단위 변환 (V/m → 선택한 단위)
        unit_scale = {"V/m": 1.0, "mV/m": 1000.0, "μV/m": 1e6, "V/mm": 0.001}.get(efield_unit, 1.0)
        values_plot = values.copy() * unit_scale
        
        # threshold도 변환된 단위 기준으로 설정
        threshold_vm = 0.00005  # V/m 기준
        threshold = threshold_vm * unit_scale
        
        # values 복사
        abs_values = np.abs(values_plot)
        
        # threshold 이상만 필터링 (이하는 점이 없다고 쳐버림)
        mask_above_threshold = abs_values >= threshold
        coords_filtered = coords[mask_above_threshold]
        values_plot_filtered = values_plot[mask_above_threshold]
        abs_values_filtered = abs_values[mask_above_threshold]
        
        max_abs = np.max(abs_values_filtered) if len(abs_values_filtered) > 0 else 1.0
        
        # 초기화 (필터링된 데이터 기준)
        alpha_values = np.ones_like(values_plot_filtered)  # 모두 진하게
        # 점 크기 정상으로 (값에 비례하지만 작게)
        if max_abs > threshold:
            normalized = (abs_values_filtered - threshold) / (max_abs - threshold)
            point_sizes = 5 + 15 * normalized  # 5 ~ 20 (정상 크기)
        else:
            point_sizes = np.ones_like(values_plot_filtered) * 5
        
        sc = ax.scatter(coords_filtered[:, a0], coords_filtered[:, a1], c=values_plot_filtered, 
                        s=point_sizes, cmap="viridis_r", alpha=alpha_values)
        pbar.update(1)
        ax.set_xlabel(f"{projection[0]} ({units})")
        ax.set_ylabel(f"{projection[1]} ({units})")
        pbar.update(1)
        fig.colorbar(sc, ax=ax, label=f"E-field ({efield_unit})")
        pbar.update(1)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_aspect("equal", adjustable="box")
        
        # 뉴런 그리기
        if show_neurons:
            plot_neurons_on_2d(ax, projection, units)
        
        pbar.update(1)
        plt.tight_layout()
        pbar.update(1)
    
    if output_path:
        print(f"💾 플롯 저장 중: {output_path}")
        with tqdm(total=1, desc="파일 저장", unit="file", leave=False) as pbar:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            pbar.update(1)
        print(f"✅ 2D 플롯 저장됨: {output_path}")
    
    if HAS_DISPLAY:
        plt.show(block=True)
    elif not output_path:
        print("⚠️  DISPLAY가 없어 플롯을 표시할 수 없습니다. --output 옵션으로 파일 저장하세요.")
    
    plt.close()


def plot_time_3d(values, coords_m, component, time_step_us, units, title, output_path=None, 
                  slice_axis=None, slice_value=0.0, slice_thickness=50.0, downsample_step=1,
                  time_downsample=1, x_range=None, time_range=None, efield_unit="mV/m", show_neurons=False):
    """
    시간축을 사용한 3D 플롯: (x, time, z) 공간에서 E-field 시각화
    y축 대신 시간축을 사용합니다.
    
    Args:
        time_downsample: 시간 축 다운샘플링 비율 (1이면 모든 시간 지점 사용)
    """
    # 전체 시간 범위 가져오기
    n_spatial, n_time = values.shape[1], values.shape[2]
    
    # 시간 다운샘플링
    if time_downsample > 1:
        n_time = n_time // time_downsample
        print(f"⏱️  시간 다운샘플링: {time_downsample}배 (시간 지점: {values.shape[2]} → {n_time})")
    
    # 시간 배열 생성 (ms)
    time_step_ms = time_step_us / 1000.0
    time_array_full = np.arange(0, values.shape[2], time_downsample) * time_step_ms
    time_array_full = time_array_full[:n_time]
    
    # 시간 범위 필터링
    if time_range is not None:
        time_min, time_max = time_range
        time_mask = (time_array_full >= time_min) & (time_array_full <= time_max)
        time_array = time_array_full[time_mask]
        time_indices = np.where(time_mask)[0]
        n_time = len(time_array)
        print(f"⏱️  시간 범위 필터링: {time_min} ~ {time_max} ms (시간 지점: {len(time_array_full)} → {n_time})")
    else:
        time_array = time_array_full
        time_indices = np.arange(n_time)
    
    # 좌표 변환
    coords = coords_m if units == "m" else coords_m * 1e6
    
    # x 범위 필터링
    mask_x = None
    if x_range is not None:
        x_min, x_max = x_range
        mask_x = (coords[:, 0] >= x_min) & (coords[:, 0] <= x_max)
        coords = coords[mask_x]
        print(f"📍 X 범위 필터링: {x_min} ~ {x_max} {units} (포인트 수: {len(coords)})")
    
    # E-field 데이터 추출 (x 범위 필터링 후)
    if mask_x is not None:
        if component == "ex":
            field_all_time = values[0, mask_x, :]  # (N_filtered, T)
        elif component == "ez":
            field_all_time = values[1, mask_x, :]  # (N_filtered, T)
        else:  # mag
            ex_all = values[0, mask_x, :]
            ez_all = values[1, mask_x, :]
            field_all_time = np.sqrt(ex_all**2 + ez_all**2)  # (N_filtered, T)
    else:
        if component == "ex":
            field_all_time = values[0, :, :]  # (N_spatial, T)
        elif component == "ez":
            field_all_time = values[1, :, :]  # (N_spatial, T)
        else:  # mag
            ex_all = values[0, :, :]
            ez_all = values[1, :, :]
            field_all_time = np.sqrt(ex_all**2 + ez_all**2)  # (N_spatial, T)
    
    # Slice 필터링
    if slice_axis:
        axis_index = {"x": 0, "y": 1, "z": 2}[slice_axis]
        half = slice_thickness / 2.0
        mask = np.abs(coords[:, axis_index] - slice_value) <= half
        coords = coords[mask]
        # field_all_time도 필터링
        field_all_time = field_all_time[mask, :]
    
    # Downsampling (기본값 1이므로 건너뛰지만, 명시적으로 지정된 경우에만 적용)
    if downsample_step > 1:
        coords = coords[::downsample_step]
        field_all_time = field_all_time[::downsample_step, :]
        print(f"📉 다운샘플링 적용: {downsample_step}배")
    
    # 플롯 데이터 준비: 각 공간 포인트에 대해 모든 시간 지점을 플롯
    n_points = coords.shape[0]
    
    # 다운샘플링 없음 (모든 포인트 사용)
    
    # 시간 다운샘플링 적용
    if time_downsample > 1:
        field_all_time = field_all_time[:, ::time_downsample]
        field_all_time = field_all_time[:, :len(time_array_full)]
    
    # 시간 범위 필터링 적용
    if time_range is not None:
        field_all_time = field_all_time[:, time_indices]
    
    # 3D 플롯 데이터 생성
    total_points = n_points * n_time
    print(f"📊 플롯 데이터 생성 중... (공간 포인트: {n_points}, 시간 지점: {n_time}, 총 {total_points:,}개 점)")
    
    # 너무 많은 점이면 경고
    if total_points > 500000:
        print(f"⚠️  경고: 점이 너무 많습니다 ({total_points:,}개). 렌더링이 느릴 수 있습니다.")
        print(f"   --time-downsample 옵션으로 시간 샘플링을 늘리거나 --downsample으로 공간 샘플링을 늘리세요.")
    
    x_coords = []
    time_coords = []
    z_coords = []
    field_values = []
    
    for i in tqdm(range(n_points), desc="공간 포인트 처리", unit="point"):
        x = coords[i, 0]
        z = coords[i, 2]
        for t_idx in range(n_time):
            x_coords.append(x)
            time_coords.append(time_array[t_idx])
            z_coords.append(z)
            field_values.append(field_all_time[i, t_idx])
    
    print("🔄 배열 변환 중...")
    with tqdm(total=4, desc="배열 변환", unit="step", leave=False) as pbar:
        x_coords = np.array(x_coords)
        pbar.update(1)
        time_coords = np.array(time_coords)
        pbar.update(1)
        z_coords = np.array(z_coords)
        pbar.update(1)
        field_values = np.array(field_values)
        pbar.update(1)
    
    # 3D 플롯 생성
    print("🎨 플롯 생성 중...")
    with tqdm(total=6, desc="플롯 렌더링", unit="step", leave=False) as pbar:
        fig = plt.figure(figsize=(12, 8))
        pbar.update(1)
        ax: Axes3D = fig.add_subplot(111, projection="3d")  # type: ignore
        pbar.update(1)
        
        # 단위 변환 (V/m → 선택한 단위)
        unit_scale = {"V/m": 1.0, "mV/m": 1000.0, "μV/m": 1e6, "V/mm": 0.001}.get(efield_unit, 1.0)
        field_values_plot = field_values.copy() * unit_scale
        
        # threshold도 변환된 단위 기준으로 설정
        threshold_vm = 0.00001  # V/m 기준
        threshold = threshold_vm * unit_scale
        
        # field_values 복사
        abs_field_values = np.abs(field_values_plot)
        
        # threshold 이상만 필터링 (이하는 점이 없다고 쳐버림)
        mask_above_threshold = abs_field_values >= threshold
        
        # 0인 점들은 필터링하여 제외
        x_coords_filtered = x_coords[mask_above_threshold]
        time_coords_filtered = time_coords[mask_above_threshold]
        z_coords_filtered = z_coords[mask_above_threshold]
        field_values_plot_filtered = field_values_plot[mask_above_threshold]
        abs_field_values_filtered = abs_field_values[mask_above_threshold]
        
        max_abs = np.max(abs_field_values_filtered) if len(abs_field_values_filtered) > 0 else 1.0
        
        # 기본 점 크기 정상적으로 (5~20)
        base_size = 5
        
        # 초기화 (필터링된 데이터 기준)
        alpha_values = np.ones_like(field_values_plot_filtered)  # 모두 진하게
        # 점 크기 정상으로 (값에 비례하지만 작게)
        if max_abs > threshold:
            normalized = (abs_field_values_filtered - threshold) / (max_abs - threshold)
            point_sizes = base_size + 15 * normalized  # 5 ~ 20 (정상 크기)
        else:
            point_sizes = np.ones_like(field_values_plot_filtered) * base_size
        
        # Ez 성분의 경우 diverging colormap 사용 (마이너스/플러스 구분)
        if component == "ez":
            # Ez 값을 그대로 사용 (마이너스와 플러스 모두 포함)
            color_values = field_values_plot_filtered
            # Diverging colormap 사용 (0을 중심으로 마이너스/플러스 구분)
            cmap_to_use = "RdBu_r"  # 빨강(플러스) - 파랑(마이너스)
            # 색상 범위를 대칭적으로 설정
            vmax = np.max(np.abs(color_values)) if len(color_values) > 0 else 1.0
            vmin = -vmax
        else:
            # Ex나 magnitude의 경우 기존 방식 사용
            color_values = field_values_plot_filtered
            cmap_to_use = "viridis_r"
            vmin = None
            vmax = None
        
        # scatter accepts arrays for all parameters including alpha and s
        # Note: matplotlib scatter actually accepts arrays for zs and s, but type checker doesn't recognize it
        sc = ax.scatter(x_coords_filtered, time_coords_filtered, z_coords_filtered, 
                        c=color_values, s=point_sizes, 
                        cmap=cmap_to_use, alpha=alpha_values, vmin=vmin, vmax=vmax)  # type: ignore[arg-type, call-overload]
        pbar.update(1)
        ax.set_xlabel(f"x ({units})")
        ax.set_ylabel("Time (ms)")
        ax.set_zlabel(f"z ({units})")
        
        # x축 범위 제한 (x_range가 지정된 경우)
        if x_range is not None:
            x_min, x_max = x_range
            ax.set_xlim(x_min, x_max)
            print(f"📍 X축 범위 제한: {x_min:.1f} ~ {x_max:.1f} {units}")
        
        pbar.update(1)
        # Ez의 경우 라벨에 방향성 표시
        if component == "ez":
            fig.colorbar(sc, ax=ax, shrink=0.6, label=f"E_z ({efield_unit})")
        else:
            fig.colorbar(sc, ax=ax, shrink=0.6, label=f"E-field ({efield_unit})")
        pbar.update(1)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        # 뉴런 그리기 (time=-1 위치에 x-z 평면)
        if show_neurons:
            plot_neurons_on_3d(ax, units, time_value=-1.0)
        
        plt.tight_layout()
        pbar.update(1)
    
    if output_path:
        print(f"💾 플롯 저장 중: {output_path}")
        with tqdm(total=1, desc="파일 저장", unit="file", leave=False) as pbar:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            pbar.update(1)
        print(f"✅ 시간축 3D 플롯 저장됨: {output_path}")
    
    if HAS_DISPLAY:
        print("🖼️  플롯 창 표시 중... (데이터가 많으면 시간이 걸릴 수 있습니다)")
        try:
            plt.show(block=True)
        except Exception as e:
            print(f"⚠️  플롯 표시 중 오류 발생: {e}")
            print("   파일로 저장된 결과를 확인하세요.")
    elif not output_path:
        print("⚠️  DISPLAY가 없어 플롯을 표시할 수 없습니다. --output 옵션으로 파일 저장하세요.")
    
    plt.close()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Visualize E-field data from npy files.")
    parser.add_argument("--values", default=os.path.join(script_dir, "E_field_40cycles.npy"))
    parser.add_argument("--coords", default=os.path.join(script_dir, "E_field_grid_coords.npy"))
    parser.add_argument("--component", choices=["ex", "ez", "mag"], default="mag")
    parser.add_argument("--time-ms", type=float, default=0.0)
    parser.add_argument("--time-index", type=int, default=None)
    parser.add_argument("--time-step-us", type=float, default=DEFAULT_TIME_STEP_US)
    parser.add_argument("--units", choices=["m", "um"], default="um")
    parser.add_argument("--slice-axis", choices=["x", "y", "z"], default=None)
    parser.add_argument("--slice-value", type=float, default=0.0)
    parser.add_argument("--slice-thickness", type=float, default=50.0)
    parser.add_argument("--projection", choices=["xy", "xz", "yz"], default="xy")
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--x-range", type=float, nargs=2, default=None, metavar=('MIN', 'MAX'), help="X-axis range to display (default: None, 전체 범위)")
    parser.add_argument("--output", type=str, default=None, help="Output file path for the plot")
    parser.add_argument("--time-axis", action="store_true", help="Use time axis instead of y-axis for 3D plot (x, time, z)")
    parser.add_argument("--time-downsample", type=int, default=1, help="Time axis downsampling factor for time-axis plot (default: 1)")
    parser.add_argument("--time-range", type=float, nargs=2, default=[0, 0.5], metavar=('MIN', 'MAX'), help="Time range in ms (default: 0 0.5)")
    parser.add_argument("--efield-unit", choices=["V/m", "mV/m", "μV/m", "V/mm"], default="mV/m", help="E-field unit for display (default: mV/m)")
    parser.add_argument("--show-neurons", action="store_true", help="Show SimplePyramidal neurons on the plot (soma as red circle, axon as red line)")
    parser.add_argument("--quiver", action="store_true", help="Plot E-field direction as arrows (quiver plot) on x-z plane")
    parser.add_argument("--quiver-time-ms", type=float, default=0.05, help="Time in ms for quiver plot (default: 0.05 ms, note: t=0 has zero E-field)")
    parser.add_argument("--y-slice", type=float, default=42.0, help="Y slice value for quiver plot (default: 42.0 um)")
    parser.add_argument("--y-slice-thickness", type=float, default=1.0, help="Y slice thickness for quiver plot (default: 1.0 um)")
    parser.add_argument("--quiver-downsample", type=int, default=5, help="Downsampling step for quiver plot arrows (default: 5)")
    args = parser.parse_args()

    values, coords_m = load_data(args.values, args.coords)
    t_max = values.shape[2] - 1
    if args.time_index is not None:
        t_idx = max(0, min(t_max, args.time_index))
    else:
        t_idx = time_to_index(args.time_ms, args.time_step_us, t_max)

    # Quiver 플롯 모드 (E-field 방향 화살표)
    if args.quiver:
        coords = coords_m if args.units == "m" else coords_m * 1e6
        
        # 지정된 시간의 Ex, Ez 값 가져오기
        quiver_t_idx = time_to_index(args.quiver_time_ms, args.time_step_us, t_max)
        quiver_time_ms = quiver_t_idx * args.time_step_us / 1000.0
        
        ex_values = values[0, :, quiver_t_idx]  # Ex at specified time
        ez_values = values[1, :, quiver_t_idx]  # Ez at specified time
        
        # 전기장 크기 확인
        magnitude = np.sqrt(ex_values**2 + ez_values**2)
        max_mag = np.max(magnitude)
        print(f"E-field at t = {quiver_time_ms:.3f} ms (index {quiver_t_idx}): max magnitude = {max_mag:.6e} V/m")
        
        if max_mag < 1e-10:
            print(f"WARNING: E-field is essentially zero at t = {quiver_time_ms:.3f} ms.")
            print(f"  Try a different time point (e.g., --quiver-time-ms 0.05)")
        
        title = f"E-field Direction at t = {quiver_time_ms:.3f} ms\nY = {args.y_slice:.1f} ± {args.y_slice_thickness/2:.1f} {args.units}"
        
        # 출력 경로 설정
        if args.output:
            output_path = args.output
        else:
            output_dir = os.path.join(script_dir, "visualize_efield_output")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"efield_quiver_t{quiver_time_ms:.3f}_y{args.y_slice:.1f}.png")
        
        plot_2d_quiver(coords, ex_values, ez_values, args.units, title, output_path, 
                      args.efield_unit, args.y_slice, args.y_slice_thickness, args.quiver_downsample)
        return

    # 시간축 플롯 모드
    if args.time_axis:
        component_names = {"ex": "Electric Field (E_x)", "ez": "Electric Field (E_z)", "mag": "Electric Field Magnitude"}
        component_name = component_names.get(args.component, args.component.upper())
        title = f"Spatial-Temporal Distribution of {component_name}"
        if args.slice_axis:
            title += f"\nSlice: {args.slice_axis} = {args.slice_value:.1f} ± {args.slice_thickness/2:.1f} {args.units}"
        
        # time-axis 모드에서 x_range가 지정되지 않았으면 기본값 -500~500um 설정
        x_range_to_use = args.x_range
        if x_range_to_use is None:
            # units에 따라 변환 (um 단위로 -500~500)
            if args.units == "um":
                x_range_to_use = [-500.0, 500.0]
            else:  # m 단위
                x_range_to_use = [-500.0e-6, 500.0e-6]
            print(f"📍 Time-axis 모드: X 범위 기본값 설정: {x_range_to_use[0]:.1f} ~ {x_range_to_use[1]:.1f} {args.units}")
        
        # 출력 경로 설정
        if args.output:
            output_path = args.output
        else:
            output_dir = os.path.join(script_dir, "visualize_efield_output")
            os.makedirs(output_dir, exist_ok=True)
            filename = f"efield_{args.component}_time_axis"
            if args.slice_axis:
                filename += f"_{args.slice_axis}{args.slice_value}"
            filename += ".png"
            output_path = os.path.join(output_dir, filename)
        
        plot_time_3d(values, coords_m, args.component, args.time_step_us, args.units, 
                     title, output_path, args.slice_axis, args.slice_value, 
                     args.slice_thickness, args.downsample, args.time_downsample, x_range_to_use, args.time_range, args.efield_unit, args.show_neurons)
    else:
        # 기존 플롯 모드 (특정 시간 지점)
        coords = coords_m if args.units == "m" else coords_m * 1e6
        field = get_component(values, args.component, t_idx)

        # x 범위 필터링 (x_range가 지정된 경우에만)
        if args.x_range is not None:
            x_min, x_max = args.x_range
            coords, field = filter_x_range(coords, field, x_min, x_max)
            print(f"📍 X 범위 필터링: {x_min} ~ {x_max} {args.units} (포인트 수: {len(coords)})")

        if args.slice_axis:
            coords, field = filter_slice(coords, field, args.slice_axis, args.slice_value, args.slice_thickness)

        # 다운샘플링 (기본값 1이므로 건너뛰지만, 명시적으로 지정된 경우에만 적용)
        if args.downsample > 1:
            coords, field = downsample(coords, field, args.downsample)
            print(f"📉 다운샘플링 적용: {args.downsample}배")

        component_names = {"ex": "Electric Field (E_x)", "ez": "Electric Field (E_z)", "mag": "Electric Field Magnitude"}
        component_name = component_names.get(args.component, args.component.upper())
        time_ms = t_idx * args.time_step_us / 1000.0
        title = f"Spatial Distribution of {component_name}\nt = {time_ms:.2f} ms (index: {t_idx})"
        if args.slice_axis:
            title += f"\nSlice: {args.slice_axis} = {args.slice_value:.1f} ± {args.slice_thickness/2:.1f} {args.units}"

        # 출력 경로 설정
        if args.output:
            output_path = args.output
        else:
            # 기본 출력 디렉토리 생성
            output_dir = os.path.join(script_dir, "visualize_efield_output")
            os.makedirs(output_dir, exist_ok=True)
            
            # 파일명 생성
            filename = f"efield_{args.component}_t{t_idx}"
            if args.slice_axis:
                filename += f"_{args.slice_axis}{args.slice_value}_{args.projection}"
            else:
                filename += "_3d"
            filename += ".png"
            output_path = os.path.join(output_dir, filename)

        if args.slice_axis:
            plot_2d(coords, field, args.units, args.projection, title, output_path, args.efield_unit, args.show_neurons)
        else:
            plot_3d(coords, field, args.units, title, output_path, auto_save=True, efield_unit=args.efield_unit, show_neurons=args.show_neurons)


if __name__ == "__main__":
    main()
