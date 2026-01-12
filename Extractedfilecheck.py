import numpy as np
import os

FILE_PATH = "E_field_40cycles.npy" 
CYCLE_LENGTH = 201

def verify_e_field_data(file_path):
    """E_field_40cycles.npy 파일을 로드하고, 모양, 반복성 및 무결성을 확인합니다."""
    
    try:
        if not os.path.exists(file_path):
             raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        E_field_data = np.load(file_path)
        
        print("--- 1. 배열 기본 정보 확인 ---")
        print(f"파일 로드 성공: {file_path}")
        print(f"배열 Shape (차원): {E_field_data.shape}")
        print(f"데이터 타입: {E_field_data.dtype}")
        
        # 최소값 / 최대값 확인
        e_min = np.min(E_field_data)
        e_max = np.max(E_field_data)
        print(f"최소값 / 최대값: {e_min:.3e} / {e_max:.3e}")

        # N_spatial (공간 그리드 지점 수) 확인
        N_spatial = E_field_data.shape[1]
        
        # 예상 Shape 확인
        if E_field_data.shape[0] == 2 and E_field_data.shape[2] == (CYCLE_LENGTH * 40):
            print("✅ Shape (2, N_spatial, 8040)이 성공적으로 확인되었습니다.")
        else:
            print("❌ Shape이 예상(2, N_spatial, 8040)과 다릅니다. 데이터 추출 과정을 확인하세요.")
            return

        # --- 2. 반복성 확인 (40번 반복되었는지) ---
        print("\n--- 2. 반복성 확인 (Ex 성분 기준) ---")
        Ex_data = E_field_data[0]
        
        # 첫 번째 사이클과 두 번째 사이클 데이터 추출
        first_cycle = Ex_data[:, 0:CYCLE_LENGTH]
        second_cycle = Ex_data[:, CYCLE_LENGTH:2 * CYCLE_LENGTH]
        
        # 두 사이클이 완전히 동일한지 확인
        is_close = np.allclose(first_cycle, second_cycle)
        
        if is_close:
            print("✅ 1주기 데이터가 2주기 데이터와 미세한 부동 소수점 오차 내에서 일치합니다.")
        else:
             print("❌ 1주기와 2주기 데이터가 일치하지 않습니다. 데이터 확장(`np.tile`)에 문제가 있을 수 있습니다.")
             
        # --- 3. 데이터 무결성 확인 ---
        print("\n--- 3. 데이터 무결성 확인 ---")
        has_nan_inf = np.any(np.isnan(E_field_data)) or np.any(np.isinf(E_field_data))
        
        if not has_nan_inf:
            print("✅ 데이터에 NaN (결측값) 또는 Inf (무한대)가 없습니다.")
        else:
            print("❌ 데이터에 NaN 또는 Inf가 포함되어 있습니다. 원본 텍스트 파일이나 처리 코드를 확인하세요.")
            
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print("스크립트 상단의 FILE_PATH 변수를 확인하고, 'pip install numpy'를 실행했는지 확인하세요.")

if __name__ == "__main__":
    verify_e_field_data(FILE_PATH)