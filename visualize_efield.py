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


def plot_3d(coords, values, units, title, output_path=None, auto_save=True, efield_unit="mV/m"):
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


def plot_2d(coords, values, units, projection, title, output_path=None, efield_unit="mV/m"):
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
                  time_downsample=1, x_range=None, time_range=None, efield_unit="mV/m"):
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
        threshold_vm = 0.00005  # V/m 기준
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
        
        # scatter accepts arrays for all parameters including alpha and s
        # Note: matplotlib scatter actually accepts arrays for zs and s, but type checker doesn't recognize it
        sc = ax.scatter(x_coords_filtered, time_coords_filtered, z_coords_filtered, 
                        c=field_values_plot_filtered, s=point_sizes, 
                        cmap="viridis_r", alpha=alpha_values)  # type: ignore[arg-type, call-overload]
        pbar.update(1)
        ax.set_xlabel(f"x ({units})")
        ax.set_ylabel("Time (ms)")
        ax.set_zlabel(f"z ({units})")
        pbar.update(1)
        fig.colorbar(sc, ax=ax, shrink=0.6, label=f"E-field ({efield_unit})")
        pbar.update(1)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
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
    parser.add_argument("--time-range", type=float, nargs=2, default=[0, 30], metavar=('MIN', 'MAX'), help="Time range in ms (default: 0 30)")
    parser.add_argument("--efield-unit", choices=["V/m", "mV/m", "μV/m", "V/mm"], default="mV/m", help="E-field unit for display (default: mV/m)")
    args = parser.parse_args()

    values, coords_m = load_data(args.values, args.coords)
    t_max = values.shape[2] - 1
    if args.time_index is not None:
        t_idx = max(0, min(t_max, args.time_index))
    else:
        t_idx = time_to_index(args.time_ms, args.time_step_us, t_max)

    # 시간축 플롯 모드
    if args.time_axis:
        component_names = {"ex": "Electric Field (E_x)", "ez": "Electric Field (E_z)", "mag": "Electric Field Magnitude"}
        component_name = component_names.get(args.component, args.component.upper())
        title = f"Spatial-Temporal Distribution of {component_name}"
        if args.slice_axis:
            title += f"\nSlice: {args.slice_axis} = {args.slice_value:.1f} ± {args.slice_thickness/2:.1f} {args.units}"
        
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
                     args.slice_thickness, args.downsample, args.time_downsample, args.x_range, args.time_range, args.efield_unit)
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
            plot_2d(coords, field, args.units, args.projection, title, output_path, args.efield_unit)
        else:
            plot_3d(coords, field, args.units, title, output_path, auto_save=True, efield_unit=args.efield_unit)


if __name__ == "__main__":
    main()
