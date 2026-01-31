# extract_coords.py
import numpy as np
import pandas as pd
import os

# Ex_001.txt 또는 Ez_001.txt 파일 중 하나만 사용하면 됨
# (예: /mnt/c/Users/YourName/Desktop/My_Ansys_Files/Ex_001.txt)
FIRST_ANSYS_FILE = "001.txt" 

OUTPUT_COORD_FILE = 'E_field_grid_coords.npy'

def extract_coords(file_path, output_path):
    """
    Ansys 텍스트 파일에서 X, Y, Z 좌표를 추출하여 .npy 파일로 저장합니다.
    """
    print(f"좌표 추출을 위해 파일 {file_path}를 로드합니다...")

    try:
        if not os.path.exists(file_path):
             raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        # skiprows=2: 헤더 2줄 스킵. sep=r'\s+': 공백/탭으로 분리.
        # iloc[:, 0:3]: 0열, 1열, 2열 (X, Y, Z 좌표)만 추출
        df_coords = pd.read_csv(file_path, skiprows=2, sep=r'\s+', 
                                header=None, engine='python').iloc[:, 0:3]
        
        # NumPy 배열로 변환 및 저장
        # Shape: (N_spatial, 3)
        grid_coords = df_coords.values.astype(np.float64) 
        
        np.save(output_path, grid_coords)

        print(f"\n✅ 좌표 그리드 추출 성공!")
        print(f"   Shape: {grid_coords.shape}")
        print(f"   저장 파일: {output_path}")

    except FileNotFoundError:
        print(f"\n❌ 오류: 텍스트 파일을 찾을 수 없습니다. 경로를 확인하세요: {file_path}")
    except Exception as e:
        print(f"\n❌ 오류: 좌표 추출 중 문제 발생: {e}")

if __name__ == "__main__":
    extract_coords(FIRST_ANSYS_FILE, OUTPUT_COORD_FILE)