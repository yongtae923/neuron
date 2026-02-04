# extract_xyz.py
"""
Ansys E-field 데이터 추출 및 검증 스크립트

기능:
1. Ex, Ey, Ez 폴더에서 N사이클 데이터 로드 (1사이클=201개 파일, 10ms/50us)
2. 좌표 추출 (X, Y, Z)
3. E-field 값 추출 및 NumPy 배열로 저장
4. 데이터 무결성 검증

출력:
- E_field_Ncycle.npy: Shape (3, N_spatial, N_steps) - Ex, Ey, Ez 성분
- E_field_grid_coords.npy: Shape (N_spatial, 3) - X, Y, Z 좌표
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='numpy')

import numpy as np
import pandas as pd
import os
import glob
import json
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- 1. 경로 및 상수 설정 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = SCRIPT_DIR

# E-field 폴더 경로 (efield 디렉토리 내)
EFIELD_BASE_DIR = os.path.join(BASE_DIR, "efield")
EX_DIR = os.path.join(EFIELD_BASE_DIR, "twin_cons_cathodicfirst_400us_100Hz_single10ms_xyz_Ex")
EY_DIR = os.path.join(EFIELD_BASE_DIR, "twin_cons_cathodicfirst_400us_100Hz_single10ms_xyz_Ey")
EZ_DIR = os.path.join(EFIELD_BASE_DIR, "twin_cons_cathodicfirst_400us_100Hz_single10ms_xyz_Ez")

# --- Cycle 설정: 1사이클 = 201 파일 (10ms, 50us 간격) ---
NUM_CYCLES = 4  # 추출할 사이클 수
STEPS_PER_CYCLE = 201  # 1사이클당 파일 개수
TOTAL_STEPS = STEPS_PER_CYCLE * NUM_CYCLES  # 총 시간 스텝 수

# 출력 파일 경로 (efield 폴더에 저장)
OUTPUT_E_FIELD_FILE = os.path.join(EFIELD_BASE_DIR, f"E_field_{NUM_CYCLES}cycle.npy")
OUTPUT_COORDS_FILE = os.path.join(EFIELD_BASE_DIR, "E_field_grid_coords.npy")

# 상수
BATCH_SIZE = 10  # 배치 단위로 처리할 파일 개수 (체크포인트용)

# 멀티프로세싱 설정: 전체 코어 중 4개 남기고 나머지 사용
TOTAL_CPUS = cpu_count()
NUM_WORKERS = max(1, TOTAL_CPUS - 4)  # 최소 1개는 보장
print(f"CPU 코어: {TOTAL_CPUS}개, 사용: {NUM_WORKERS}개 (4개 남김)")

# 임시 파일 및 체크포인트 디렉토리
TEMP_DIR = os.path.join(EFIELD_BASE_DIR, "_temp_extract")
os.makedirs(TEMP_DIR, exist_ok=True)

print("=" * 60)
print("E-field 데이터 추출 및 검증 스크립트")
print("=" * 60)
print(f"\n사이클 설정: {NUM_CYCLES} cycle(s), {STEPS_PER_CYCLE} steps/cycle → 총 {TOTAL_STEPS} time steps")
print(f"기준 디렉토리: {BASE_DIR}")
print(f"Ex 경로: {EX_DIR}")
print(f"Ey 경로: {EY_DIR}")
print(f"Ez 경로: {EZ_DIR}")
print(f"출력 파일 (E-field): {OUTPUT_E_FIELD_FILE}")
print(f"출력 파일 (좌표): {OUTPUT_COORDS_FILE}")

# 디렉토리 존재 확인
for dir_name, dir_path in [("Ex", EX_DIR), ("Ey", EY_DIR), ("Ez", EZ_DIR)]:
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"오류: {dir_name} 디렉토리를 찾을 수 없습니다: {dir_path}")


# --- 2. 체크포인트 관리 함수 ---
def get_checkpoint_path(component_name):
    """체크포인트 파일 경로 반환"""
    return os.path.join(TEMP_DIR, f"checkpoint_{component_name}.json")

def load_checkpoint(component_name):
    """체크포인트 로드 (처리된 배치 인덱스 리스트 반환)"""
    checkpoint_path = get_checkpoint_path(component_name)
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
            return data.get('processed_batches', [])
    return []

def save_checkpoint(component_name, processed_batches):
    """체크포인트 저장"""
    checkpoint_path = get_checkpoint_path(component_name)
    with open(checkpoint_path, 'w') as f:
        json.dump({'processed_batches': processed_batches}, f)

def get_batch_file_path(component_name, batch_idx):
    """배치 임시 파일 경로 반환"""
    return os.path.join(TEMP_DIR, f"{component_name}_batch_{batch_idx:03d}.npy")

def clear_checkpoint(component_name):
    """체크포인트 및 임시 파일 삭제"""
    checkpoint_path = get_checkpoint_path(component_name)
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    # 해당 성분의 모든 배치 파일 삭제
    pattern = os.path.join(TEMP_DIR, f"{component_name}_batch_*.npy")
    for batch_file in glob.glob(pattern):
        os.remove(batch_file)


# --- 3. 워커 함수 (멀티프로세싱용) ---
def read_single_file(args):
    """
    단일 파일을 읽어서 E-field 값을 반환합니다.
    멀티프로세싱 워커 함수로 사용됩니다.
    
    Args:
        args: (file_path, file_index, num_spatial_points) 튜플
    
    Returns:
        (file_index, field_values): 파일 인덱스와 E-field 값 배열
    """
    file_path, file_index, num_spatial_points = args
    try:
        df = pd.read_csv(file_path, skiprows=2, sep=r'\s+', header=None, engine='python')
        field_values = df.iloc[:, -1].values.astype(np.float32)
        
        if len(field_values) != num_spatial_points:
            raise ValueError(
                f"파일 {os.path.basename(file_path)}의 공간 지점 수가 일치하지 않습니다. "
                f"(예상: {num_spatial_points}, 실제: {len(field_values)})"
            )
        
        return (file_index, field_values)
    except Exception as e:
        raise Exception(f"파일 {os.path.basename(file_path)} 처리 중 오류: {e}")


# --- 4. 배치 단위 데이터 로드 함수 ---
def load_e_field_component(component_dir, total_steps=201, batch_size=20):
    """
    배치 단위로 파일을 로드하고 체크포인트를 사용하여 재시작 가능하게 처리합니다.
    
    Args:
        component_dir: E-field 성분 폴더 경로 (Ex, Ey, 또는 Ez)
        total_steps: 예상 파일 개수 (기본값: 201)
        batch_size: 배치 크기 (기본값: 20)
    
    Returns:
        combined_data: Shape (N_spatial, total_steps) - E-field 값 배열
    """
    file_pattern = os.path.join(component_dir, "*.txt")
    file_list = glob.glob(file_pattern)
    # Sort by time step index (001, 002, ... 201) so order is correct even if 1.txt, 2.txt, ... 201.txt
    def _sort_key(p):
        base = os.path.basename(p)
        try:
            return int(os.path.splitext(base)[0])
        except ValueError:
            return 0
    file_list = sorted(file_list, key=_sort_key)
    
    if len(file_list) == 0:
        raise FileNotFoundError(f"'{component_dir}' 폴더에서 파일을 찾을 수 없습니다.")
    # Load at most total_steps files (e.g. 1 cycle); caller tiles to NUM_CYCLES if needed
    steps_to_load = min(len(file_list), total_steps)
    if len(file_list) < total_steps:
        print(f"   (파일 {len(file_list)}개 → 1사이클 {steps_to_load}개만 로드, 출력 시 {total_steps} 스텝으로 반복)")
    file_list = file_list[:steps_to_load]
    
    # 첫 번째 파일로 공간 지점 수 확인
    try:
        df_temp = pd.read_csv(file_list[0], skiprows=2, sep=r'\s+', header=None, engine='python')
        num_spatial_points = len(df_temp)
    except Exception as e:
        print(f"오류: 첫 번째 파일 로드 중 오류 발생 ({file_list[0]}): {e}")
        raise
    
    component_name = os.path.basename(component_dir)
    print(f"\n-> {component_name} 데이터 로드 시작...")
    print(f"   공간 지점 수: {num_spatial_points:,}")
    print(f"   로드할 시간 스텝 수: {steps_to_load}")
    print(f"   배치 크기: {batch_size}개 파일/배치")
    
    # 체크포인트 로드
    processed_batches = load_checkpoint(component_name)
    num_batches = (steps_to_load + batch_size - 1) // batch_size
    
    if processed_batches:
        print(f"   체크포인트 발견: {len(processed_batches)}/{num_batches} 배치 완료")
        print(f"   이미 처리된 배치: {sorted(processed_batches)}")
    else:
        print(f"   새로 시작: 총 {num_batches}개 배치 처리 예정")
    
    # 배치 단위로 처리
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, steps_to_load)
        batch_files = file_list[start_idx:end_idx]
        
        batch_file_path = get_batch_file_path(component_name, batch_idx)
        
        # 이미 처리된 배치는 스킵
        if batch_idx in processed_batches:
            if os.path.exists(batch_file_path):
                print(f"   배치 {batch_idx+1}/{num_batches} (인덱스 {start_idx}~{end_idx-1}): 이미 완료, 스킵")
                continue
            else:
                print(f"   배치 {batch_idx+1}/{num_batches}: 체크포인트는 있지만 파일이 없음, 재처리")
        
        # 배치 처리 (멀티프로세싱 사용)
        print(f"   배치 {batch_idx+1}/{num_batches} 처리 중 (인덱스 {start_idx}~{end_idx-1})...")
        batch_data = np.zeros((num_spatial_points, len(batch_files)), dtype=np.float32)
        
        # 워커 함수에 전달할 인자 준비
        worker_args = [
            (file_path, local_idx, num_spatial_points)
            for local_idx, file_path in enumerate(batch_files)
        ]
        
        # 멀티프로세싱으로 병렬 처리
        with Pool(processes=NUM_WORKERS) as pool:
            results = list(tqdm(
                pool.imap(read_single_file, worker_args),
                total=len(batch_files),
                desc=f"    배치 {batch_idx+1}",
                unit="file",
                leave=False,
                ncols=80
            ))
        
        # 결과를 배치 데이터에 할당
        for file_index, field_values in results:
            batch_data[:, file_index] = field_values
        
        # 배치 저장
        np.save(batch_file_path, batch_data)
        
        # 체크포인트 업데이트
        processed_batches.append(batch_idx)
        save_checkpoint(component_name, processed_batches)
        print(f"   배치 {batch_idx+1}/{num_batches} 저장 완료")
    
    # 모든 배치를 합치기
    print(f"\n   모든 배치 합치는 중...")
    batch_arrays = []
    for batch_idx in range(num_batches):
        batch_file_path = get_batch_file_path(component_name, batch_idx)
        if not os.path.exists(batch_file_path):
            raise FileNotFoundError(f"배치 파일을 찾을 수 없습니다: {batch_file_path}")
        batch_data = np.load(batch_file_path)
        batch_arrays.append(batch_data)
    
    combined_data = np.concatenate(batch_arrays, axis=1)
    
    # 임시 파일 및 체크포인트 정리
    clear_checkpoint(component_name)
    
    print(f"{component_name} 데이터 로드 완료!")
    return combined_data


def extract_coordinates(file_path):
    """
    Ansys 텍스트 파일에서 X, Y, Z 좌표를 추출합니다.
    
    Args:
        file_path: Ansys 텍스트 파일 경로
    
    Returns:
        grid_coords: Shape (N_spatial, 3) - X, Y, Z 좌표 배열
    """
    print(f"\n-> 좌표 추출 중: {os.path.basename(file_path)}")
    
    try:
        # skiprows=2: 헤더 2줄 스킵
        # sep=r'\s+': 공백/탭으로 분리
        # iloc[:, 0:3]: 첫 3열 (X, Y, Z 좌표)만 추출
        df_coords = pd.read_csv(file_path, skiprows=2, sep=r'\s+', 
                               header=None, engine='python').iloc[:, 0:3]
        
        grid_coords = df_coords.values.astype(np.float64)
        print(f"좌표 추출 완료! Shape: {grid_coords.shape}")
        
        return grid_coords
    
    except Exception as e:
        print(f"오류: 좌표 추출 중 오류 발생: {e}")
        raise


# --- 3. 데이터 검증 함수 ---
def verify_data(E_field_data, grid_coords, total_steps=201):
    """
    추출된 데이터의 무결성을 검증합니다.
    
    Args:
        E_field_data: Shape (3, N_spatial, total_steps) - E-field 데이터
        grid_coords: Shape (N_spatial, 3) - 좌표 데이터
        total_steps: 예상 시간 스텝 수
    """
    print("\n" + "=" * 60)
    print("데이터 검증")
    print("=" * 60)
    
    # 1. Shape 확인
    print("\n--- 1. 배열 Shape 확인 ---")
    print(f"E-field 데이터 Shape: {E_field_data.shape}")
    print(f"   예상: (3, N_spatial, {total_steps})")
    print(f"좌표 데이터 Shape: {grid_coords.shape}")
    print(f"   예상: (N_spatial, 3)")
    
    if E_field_data.shape[0] != 3:
        print(f"오류: E-field 데이터의 첫 번째 차원이 3이 아닙니다: {E_field_data.shape[0]}")
        return False
    
    if E_field_data.shape[2] != total_steps:
        print(f"오류: E-field 데이터의 시간 스텝 수가 예상과 다릅니다: {E_field_data.shape[2]} != {total_steps}")
        return False
    
    if E_field_data.shape[1] != grid_coords.shape[0]:
        print(f"오류: E-field와 좌표 데이터의 공간 지점 수가 일치하지 않습니다: "
              f"{E_field_data.shape[1]} != {grid_coords.shape[0]}")
        return False
    
    print("Shape 검증 통과")
    
    # 2. 데이터 타입 확인
    print("\n--- 2. 데이터 타입 확인 ---")
    print(f"E-field 데이터 타입: {E_field_data.dtype}")
    print(f"좌표 데이터 타입: {grid_coords.dtype}")
    print("데이터 타입 확인 완료")
    
    # 3. 최소/최대값 확인
    print("\n--- 3. 데이터 범위 확인 ---")
    for i, component_name in enumerate(['Ex', 'Ey', 'Ez']):
        component_data = E_field_data[i]
        e_min = np.min(component_data)
        e_max = np.max(component_data)
        e_mean = np.mean(component_data)
        e_std = np.std(component_data)
        print(f"{component_name}:")
        print(f"  최소값: {e_min:.6e}")
        print(f"  최대값: {e_max:.6e}")
        print(f"  평균값: {e_mean:.6e}")
        print(f"  표준편차: {e_std:.6e}")
    
    coord_min = np.min(grid_coords, axis=0)
    coord_max = np.max(grid_coords, axis=0)
    print(f"\n좌표 범위:")
    print(f"  X: [{coord_min[0]:.6e}, {coord_max[0]:.6e}]")
    print(f"  Y: [{coord_min[1]:.6e}, {coord_max[1]:.6e}]")
    print(f"  Z: [{coord_min[2]:.6e}, {coord_max[2]:.6e}]")
    print("데이터 범위 확인 완료")
    
    # 4. NaN/Inf 확인
    print("\n--- 4. 데이터 무결성 확인 ---")
    has_nan = np.any(np.isnan(E_field_data))
    has_inf = np.any(np.isinf(E_field_data))
    coord_has_nan = np.any(np.isnan(grid_coords))
    coord_has_inf = np.any(np.isinf(grid_coords))
    
    if has_nan or has_inf:
        print(f"오류: E-field 데이터에 NaN 또는 Inf가 포함되어 있습니다.")
        if has_nan:
            nan_count = np.sum(np.isnan(E_field_data))
            print(f"   NaN 개수: {nan_count:,}")
        if has_inf:
            inf_count = np.sum(np.isinf(E_field_data))
            print(f"   Inf 개수: {inf_count:,}")
        return False
    else:
        print("E-field 데이터에 NaN 또는 Inf가 없습니다.")
    
    if coord_has_nan or coord_has_inf:
        print(f"오류: 좌표 데이터에 NaN 또는 Inf가 포함되어 있습니다.")
        return False
    else:
        print("좌표 데이터에 NaN 또는 Inf가 없습니다.")
    
    # 5. 시간 일관성 확인 (첫 시간과 마지막 시간의 통계 비교)
    print("\n--- 5. 시간 일관성 확인 ---")
    first_time = E_field_data[:, :, 0]
    last_time = E_field_data[:, :, -1]
    
    for i, component_name in enumerate(['Ex', 'Ey', 'Ez']):
        first_mean = np.mean(np.abs(first_time[i]))
        last_mean = np.mean(np.abs(last_time[i]))
        print(f"{component_name}:")
        print(f"  첫 시간 스텝 평균 절댓값: {first_mean:.6e}")
        print(f"  마지막 시간 스텝 평균 절댓값: {last_mean:.6e}")
    
    print("시간 일관성 확인 완료")
    
    print("\n" + "=" * 60)
    print("모든 검증 통과!")
    print("=" * 60)
    return True


# --- 4. 메인 실행 ---
if __name__ == "__main__":
    try:
        # If final npy files exist, load and verify only (skip heavy extraction)
        if os.path.exists(OUTPUT_E_FIELD_FILE) and os.path.exists(OUTPUT_COORDS_FILE):
            print("\n" + "=" * 60)
            print("기존 npy 파일 발견 — 검증만 수행")
            print("=" * 60)
            E_field_data = np.load(OUTPUT_E_FIELD_FILE)
            grid_coords = np.load(OUTPUT_COORDS_FILE)
            print(f"로드: E_field Shape {E_field_data.shape}, coords Shape {grid_coords.shape}")
            verify_data(E_field_data, grid_coords, TOTAL_STEPS)
            print("\n검증 완료. (추출/저장 생략)")
        else:
            # 4.1. E-field 데이터 로드
            print("\n" + "=" * 60)
            print("1단계: E-field 데이터 로드")
            print("=" * 60)
            
            # Load 1 cycle (STEPS_PER_CYCLE files) per component
            E_x = load_e_field_component(EX_DIR, STEPS_PER_CYCLE, BATCH_SIZE)
            E_y = load_e_field_component(EY_DIR, STEPS_PER_CYCLE, BATCH_SIZE)
            E_z = load_e_field_component(EZ_DIR, STEPS_PER_CYCLE, BATCH_SIZE)
            
            # Shape 확인 (모든 성분이 동일한 공간 지점 수를 가져야 함)
            if not (E_x.shape[0] == E_y.shape[0] == E_z.shape[0]):
                raise ValueError(
                    f"공간 지점 수가 일치하지 않습니다: "
                    f"Ex={E_x.shape[0]}, Ey={E_y.shape[0]}, Ez={E_z.shape[0]}"
                )
            
            # Repeat 1 cycle to NUM_CYCLES along time axis
            if NUM_CYCLES > 1:
                print(f"\n   1사이클({STEPS_PER_CYCLE} 스텝) → {NUM_CYCLES}회 반복하여 {TOTAL_STEPS} 스텝 생성")
                E_x = np.tile(E_x, (1, NUM_CYCLES))
                E_y = np.tile(E_y, (1, NUM_CYCLES))
                E_z = np.tile(E_z, (1, NUM_CYCLES))
            
            # 3개 성분을 스택: Shape (3, N_spatial, TOTAL_STEPS)
            E_field_data = np.stack((E_x, E_y, E_z), axis=0)
            print(f"\n최종 E-field 데이터 Shape: {E_field_data.shape}")
            
            # 4.2. 좌표 추출
            print("\n" + "=" * 60)
            print("2단계: 좌표 추출")
            print("=" * 60)
            
            # Ex 폴더의 첫 번째 파일에서 좌표 추출 (모든 폴더의 좌표는 동일)
            first_ex_file = os.path.join(EX_DIR, "001.txt")
            grid_coords = extract_coordinates(first_ex_file)
            
            # 4.3. 데이터 검증
            verify_data(E_field_data, grid_coords, TOTAL_STEPS)
            
            # 4.4. 파일 저장
            print("\n" + "=" * 60)
            print("3단계: 파일 저장")
            print("=" * 60)
            
            np.save(OUTPUT_E_FIELD_FILE, E_field_data)
            file_size_mb = os.path.getsize(OUTPUT_E_FIELD_FILE) / (1024**2)
            print(f"E-field 데이터 저장 완료: {OUTPUT_E_FIELD_FILE}")
            print(f"   파일 크기: {file_size_mb:.2f} MB")
            
            np.save(OUTPUT_COORDS_FILE, grid_coords)
            coords_size_mb = os.path.getsize(OUTPUT_COORDS_FILE) / (1024**2)
            print(f"좌표 데이터 저장 완료: {OUTPUT_COORDS_FILE}")
            print(f"   파일 크기: {coords_size_mb:.2f} MB")
            
            print("\n" + "=" * 60)
            print("모든 작업 완료!")
            print("=" * 60)
            print(f"\n생성된 파일:")
            print(f"  - {OUTPUT_E_FIELD_FILE}")
            print(f"  - {OUTPUT_COORDS_FILE}")
        
    except FileNotFoundError as e:
        print(f"\n오류: 파일을 찾을 수 없습니다: {e}")
        print("디렉토리 경로와 파일 위치를 확인하세요.")
    except Exception as e:
        print(f"\n오류: 처리 중 예상치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc()
