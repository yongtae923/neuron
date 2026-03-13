# Singlecycletoncycle_npyextractor_3d.py
# 3차원 E-field 데이터 (Ex, Ey, Ez)를 2사이클로 확장하여 저장하는 스크립트

import numpy as np
import os
import glob
import pandas as pd

# --- 1. 경로 및 상수 설정 ---
print("--- 1. 경로 설정 시작 ---")

# 스크립트가 있는 디렉토리를 기준으로 경로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = SCRIPT_DIR

TOTAL_STEPS = 201             # 10ms (1 사이클) 동안의 총 파일 개수
NUM_CYCLES = 2                 # 반복할 사이클 수 (2 사이클)

# Ex, Ey, Ez 파일이 들어있는 디렉토리 경로 (하나의 폴더에 모두 포함)
ANSYS_DIR = os.path.join(BASE_DIR, "ansys", "twin_cons_cathodicfirst_400us_100Hz_single10ms_ExEyEz")

OUTPUT_FILENAME = "E_field_2cycles_3d.npy"
OUTPUT_PATH = os.path.join(BASE_DIR, OUTPUT_FILENAME)

print(f"\n--- 2. 설정된 경로 확인 ---")
print(f"기준 디렉토리: {BASE_DIR}")
print(f"Ansys 데이터 폴더: {ANSYS_DIR}")
print(f"출력 파일: {OUTPUT_PATH}")

# 디렉토리 존재 확인
if not os.path.exists(ANSYS_DIR):
    print(f"⚠️ 경고: Ansys 디렉토리를 찾을 수 없습니다: {ANSYS_DIR}")
    print(f"   현재 작업 디렉토리: {os.getcwd()}")


# --- 3. 데이터 로드 및 통합 함수 정의 ---

def load_e_field_component(ansys_dir, component_name, total_steps=201):
    """
    Ansys 디렉토리에서 특정 컴포넌트(Ex, Ey, Ez)의 데이터를 로드합니다.
    
    Args:
        ansys_dir: Ansys 데이터 디렉토리 경로
        component_name: 컴포넌트 이름 ('Ex', 'Ey', 또는 'Ez')
        total_steps: 예상되는 총 파일 개수 (기본값: 201)
    
    Returns:
        combined_data: Shape (num_spatial_points, total_steps)의 NumPy 배열
    """
    # 파일명 패턴: 001.txt, 002.txt, ... (3자리 숫자)
    file_list = []
    for i in range(1, total_steps + 1):
        file_name = f"{i:03d}.txt"
        file_path = os.path.join(ansys_dir, file_name)
        if os.path.exists(file_path):
            file_list.append(file_path)
        else:
            print(f"⚠️ 경고: 파일을 찾을 수 없습니다: {file_name}")

    if len(file_list) == 0:
        raise FileNotFoundError(f"'{ansys_dir}' 폴더에서 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    
    if len(file_list) != total_steps:
        print(f"⚠️ 경고: 예상 파일 개수 ({total_steps}개)와 실제 파일 개수 ({len(file_list)}개)가 다릅니다.")

    try:
        # 첫 번째 파일을 읽어서 공간 지점 수 확인
        df_temp = pd.read_csv(file_list[0], skiprows=2, sep=r'\s+', header=None, engine='python')
        num_spatial_points = len(df_temp)
    except Exception as e:
        print(f"❌ 첫 번째 파일 로드 중 오류 발생 ({file_list[0]}): {e}")
        raise

    combined_data = np.zeros((num_spatial_points, len(file_list)), dtype=np.float32)

    print(f"\n-> {component_name} 데이터 로드 시작")
    print(f"   디렉토리: {ansys_dir}")
    print(f"   공간 지점 수: {num_spatial_points}, 시간 스텝 수: {len(file_list)}")

    for i, file_path in enumerate(file_list):
        try:
            # 텍스트 파일 읽기 (헤더 2줄 스킵, 공백/탭으로 분리)
            df = pd.read_csv(file_path, skiprows=2, sep=r'\s+', header=None, engine='python')
            # 마지막 열이 E-field 값 (X, Y, Z 좌표 다음)
            field_values = df.iloc[:, -1].values.astype(np.float32)

            if len(field_values) != num_spatial_points:
                raise ValueError(f"파일 {os.path.basename(file_path)}의 공간 지점 수가 일치하지 않습니다. "
                                f"(예상:{num_spatial_points}, 실제:{len(field_values)})")

            combined_data[:, i] = field_values

        except Exception as e:
            print(f"❌ 데이터 처리 중 오류 발생 (파일: {os.path.basename(file_path)}, 인덱스: {i}): {e}")
            combined_data[:, i] = 0

    return combined_data


# --- 4. 데이터 로드, 확장 및 저장 ---
try:
    print("\n--- 3. 데이터 로드 및 2 사이클 확장 ---")
    print(f"⚠️ 주의: 현재 폴더에는 Ex 파일만 확인되었습니다.")
    print(f"   Ey, Ez 파일이 별도로 있는지 확인이 필요합니다.")
    print(f"   일단 Ex 데이터만 로드합니다.\n")

    # Ex 데이터 로드 (현재 확인된 구조)
    print("[1/3] Ex 데이터 로드 중...")
    E_x_1cycle = load_e_field_component(ANSYS_DIR, "Ex", total_steps=TOTAL_STEPS)
    
    # Ey, Ez는 일단 Ex와 동일한 shape로 0으로 초기화 (나중에 실제 데이터로 교체 가능)
    print("\n[2/3] Ey 데이터 (임시: 0으로 초기화)...")
    E_y_1cycle = np.zeros_like(E_x_1cycle)
    
    print("\n[3/3] Ez 데이터 (임시: 0으로 초기화)...")
    E_z_1cycle = np.zeros_like(E_x_1cycle)
    
    print("\n⚠️ 참고: Ey, Ez가 별도 파일로 있다면 load_e_field_component 함수를 수정하여 로드하세요.")

    # 공간 지점 수 일치 확인
    if E_x_1cycle.shape[0] != E_y_1cycle.shape[0] or E_x_1cycle.shape[0] != E_z_1cycle.shape[0]:
        raise ValueError(f"공간 지점 수가 일치하지 않습니다: "
                        f"Ex={E_x_1cycle.shape[0]}, Ey={E_y_1cycle.shape[0]}, Ez={E_z_1cycle.shape[0]}")

    # 데이터 확장 (2 사이클)
    print(f"\n--- 4. 1 사이클 데이터를 {NUM_CYCLES} 사이클로 확장 ---")
    E_x_2cycles = np.tile(E_x_1cycle, (1, NUM_CYCLES))
    E_y_2cycles = np.tile(E_y_1cycle, (1, NUM_CYCLES))
    E_z_2cycles = np.tile(E_z_1cycle, (1, NUM_CYCLES))

    final_time_steps = E_x_2cycles.shape[1]

    print(f"\n--- 5. 최종 데이터 확인 및 저장 ---")
    print(f"✅ Ex (1 사이클) shape: {E_x_1cycle.shape}")
    print(f"✅ Ey (1 사이클) shape: {E_y_1cycle.shape}")
    print(f"✅ Ez (1 사이클) shape: {E_z_1cycle.shape}")
    print(f"\n✅ Ex (2 사이클) shape: {E_x_2cycles.shape}")
    print(f"✅ Ey (2 사이클) shape: {E_y_2cycles.shape}")
    print(f"✅ Ez (2 사이클) shape: {E_z_2cycles.shape}")

    # 3차원 배열로 스택: (3, N_spatial, N_time)
    E_field_2cycles_3d = np.stack((E_x_2cycles, E_y_2cycles, E_z_2cycles), axis=0)

    print(f"\n✅ 최종 저장 배열 (3, Spatial, Time) shape: {E_field_2cycles_3d.shape}")
    print(f"   차원 0: Ex(0), Ey(1), Ez(2)")
    print(f"   차원 1: 공간 그리드 지점 수 ({E_field_2cycles_3d.shape[1]}개)")
    print(f"   차원 2: 시간 스텝 수 ({E_field_2cycles_3d.shape[2]}개 = {TOTAL_STEPS} × {NUM_CYCLES})")

    # .npy 파일로 저장
    np.save(OUTPUT_PATH, E_field_2cycles_3d)
    file_size_mb = os.path.getsize(OUTPUT_PATH) / (1024**2)
    print(f"\n🎉 최종 데이터가 '{OUTPUT_PATH}' 파일로 성공적으로 저장되었습니다.")
    print(f"   파일 크기: {file_size_mb:.2f} MB")

    # 데이터 검증
    print(f"\n--- 6. 데이터 검증 ---")
    print(f"✅ Ex 최대값: {np.max(np.abs(E_x_2cycles)):.6f}")
    print(f"✅ Ey 최대값: {np.max(np.abs(E_y_2cycles)):.6f}")
    print(f"✅ Ez 최대값: {np.max(np.abs(E_z_2cycles)):.6f}")
    print(f"✅ NaN 값: {np.sum(np.isnan(E_field_2cycles_3d))}개")
    print(f"✅ Inf 값: {np.sum(np.isinf(E_field_2cycles_3d))}개")

except FileNotFoundError as e:
    print(f"\n❌ 파일을 찾을 수 없습니다: {e}")
    print("디렉토리 경로와 파일 위치를 다시 한번 확인해 주세요.")
except Exception as e:
    print(f"\n❌ 처리 중 예상치 못한 오류 발생: {e}")
    import traceback
    traceback.print_exc()
