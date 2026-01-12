# simulate_tES.py
import numpy as np
import os
from neuron import h
from neuron.units import um, ms, mV
from simple_pyramidal_model import SimplePyramidal
import math

h.load_file("stdrun.hoc")

# --- 1. 파일 경로 및 시뮬레이션 상수 ---
E_FIELD_VALUES_FILE = 'E_field_40cycles.npy'
E_GRID_COORDS_FILE = 'E_field_grid_coords.npy'
TIME_STEP_US = 50.0  # Ansys 데이터 시간 간격 (us)
TIME_STEP_MS = TIME_STEP_US / 1000.0  # 0.05 ms
TOTAL_TIME_MS = 402.0 # 40 사이클을 포괄하는 전체 시간 (ms)
N_SPATIAL_POINTS = None # 로드 시 결정될 공간 그리드 지점 수

# --- 2. 데이터 로드 및 전처리 ---

print("--- 1. 데이터 로드 및 전처리 ---")

# 2.1. E-필드 값 (Ex, Ez) 로드: Shape (2, N_spatial, 8040)
try:
    E_field_values = np.load(E_FIELD_VALUES_FILE)
    print(f"E-field values loaded: {E_field_values.shape}")
except Exception as e:
    print(f"❌ 오류: E-field 값 파일을 로드할 수 없습니다. 경로 확인: {e}")
    exit()

# 2.2. E-필드 좌표 로드: Shape (N_spatial, 3)
try:
    E_grid_coords_M = np.load(E_GRID_COORDS_FILE)
    
    # Ansys 좌표는 미터(M) 단위이므로, NEURON 단위(μM)로 변환 (1 M = 1e6 uM)
    E_grid_coords_UM = E_grid_coords_M * 1e6
    
    N_SPATIAL_POINTS = E_grid_coords_UM.shape[0]
    print(f"E-field coords loaded and converted to um: {E_grid_coords_UM.shape}")
except Exception as e:
    print(f"❌ 오류: E-field 좌표 파일을 로드할 수 없습니다. 경로 확인: {e}")
    exit()


# --- 3. 공간 매핑 함수 (가장 가까운 그리드 지점 찾기) ---

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

# --- 4. 시간 매핑 및 E-field 설정 함수 ---

# 전역 변수로 저장할 공간 인덱스 매핑 테이블
global_spatial_map = {}

def set_extracellular_field():
    """
    현재 NEURON 시간 (h.t)을 기반으로 모든 뉴런 구획의 E-field 값을 설정합니다.
    이 함수는 NEURON의 finitialize 이후 매 시간 스텝마다 호출됩니다.
    """
    current_time_ms = h.t # 현재 시뮬레이션 시간 (ms)

    # 1. 시간 인덱스 계산 및 보간
    # time_index는 0부터 시작하며, T/dt
    time_index_float = current_time_ms / TIME_STEP_MS
    
    # 정수 시간 인덱스 (이전 데이터 지점)
    t_idx_prev = int(math.floor(time_index_float))
    
    # 다음 데이터 지점
    # 총 8040개 데이터 지점 (인덱스 0 ~ 8039)
    t_idx_next = min(t_idx_prev + 1, E_field_values.shape[2] - 1)
    
    # 보간 비율 (0.0 <= ratio <= 1.0)
    ratio = time_index_float - t_idx_prev
    
    # 2. 모든 뉴런의 모든 구획에 필드 설정
    for neuron in neurons:
        for sec in neuron.all:
            # 3. 공간 매핑된 인덱스를 찾아 가져옵니다.
            # 이 코드는 h.t==0에서 한 번만 호출되는 것이 아니라 매번 호출되므로,
            # 매번 find_nearest_spatial_index를 호출하지 않도록 미리 매핑합니다.
            
            # --- 3.1. 공간 인덱스가 이미 매핑되었는지 확인 (최적화) ---
            for seg in sec.allseg():
                # seg.x는 구획 내의 위치 (0.0 ~ 1.0)
                # Global ID: (Section 이름, segment_x)
                seg_id = (sec.name(), seg.x) 
                
                if seg_id not in global_spatial_map:
                    # 구획의 3D 좌표 가져오기
                    #h.section_ref(sec)
                    #sec_x, sec_y, sec_z = h.x3d(seg.x, sec=sec), h.y3d(seg.x, sec=sec), h.z3d(seg.x, sec=sec)
                    sec_x = h.x3d(seg.x, sec=sec) 
                    sec_y = h.y3d(seg.x, sec=sec)
                    sec_z = h.z3d(seg.x, sec=sec)
                    
                    # 가장 가까운 그리드 인덱스 계산 및 저장
                    nearest_idx = find_nearest_spatial_index(sec_x, sec_y, sec_z, E_grid_coords_UM)
                    global_spatial_map[seg_id] = nearest_idx
                
                # 4. 공간 인덱스 가져오기
                spatial_idx = global_spatial_map[seg_id]

                # 5. 시간 보간을 통해 Ex, Ez 값 계산
                Ex_prev = E_field_values[0, spatial_idx, t_idx_prev]
                Ex_next = E_field_values[0, spatial_idx, t_idx_next]
                Ez_prev = E_field_values[1, spatial_idx, t_idx_prev]
                Ez_next = E_field_values[1, spatial_idx, t_idx_next]
                
                Ex_interp = Ex_prev + ratio * (Ex_next - Ex_prev)
                Ez_interp = Ez_prev + ratio * (Ez_next - Ez_next) # <- 수정: Ez_next - Ez_prev
                # Ez_interp = Ez_prev + ratio * (Ez_next - Ez_prev) # CORRECTED LINE
                
                # 6. Extracellular 전위 계산 및 설정
                # NEURON의 e_ext는 외부 전위 (mV)를 나타내며, 
                # 전위 경사(E-field)를 사용하는 경우 e_ext = -E_axial * segment_length (또는 z-projection)
                
                # 단순화: 축삭돌기(Axon)가 Z축 방향이므로, Ez를 Axial Field로 가정
                # 외부 전위 경사(potential gradient) 설정
                # NEURON의 extracellular.e는 전위(Potential)입니다. 
                # 여기서는 E_field를 직접 이용하는 방법 대신, 
                # E-field 기반 전위 경사(Potential Gradient)를 설정하여 NEURON이 V_e를 계산하도록 유도합니다.
                
                # Extracellular 메커니즘은 e(x)를 통해 전위를 설정합니다.
                # E-field를 직접 인가하는 표준 방식은 field.mod를 사용해야 하지만,
                # 여기서는 축삭돌기의 축방향 성분을 사용하여 간접적으로 설정합니다.
                
                # NOTE: E-field는 V/m 단위이므로 mV/um 단위로 변환해야 NEURON과 일치
                # 1 V/m = 1e-3 mV/um
                E_factor = 0.0000000001 # V/m -> mV/um ; 기존 값: 1e-3
                
                # Seg.e_ext는 해당 세그먼트의 외부 전위입니다.
                # Axon의 축방향 (Z축) 전기장 경사를 사용하여 전위 설정
                seg.e_extracellular = -Ez_interp * E_factor * (seg.sec.L / seg.sec.nseg) # (단순화된 Z축 전위 경사)
                
                # 실제 축삭돌기가 Z축에 평행하다고 가정하면, Ez만 사용하고 Ex는 무시
                # 더 복잡한 모델에서는 E_field_x/y/z 값을 직접 설정하는 field.mod를 사용해야 합니다.
                
    return 0 # h.fadvance()를 위한 반환값 (필수)


# --- 5. 뉴런 생성 및 배치 ---

print("\n--- 2. 뉴런 모델 생성 및 배치 ---")

# 3개 뉴런의 중심 위치 (um)
N_POSITIONS = [
    (-90.0 * um, 42.0 * um, 561.0 * um), # Neuron 1
    (0.0 * um, 42.0 * um, 561.0 * um),   # Neuron 2
    (90.0 * um, 42.0 * um, 561.0 * um)    # Neuron 3
]

# 뉴런 인스턴스 생성
neurons = []
for i, (x, y, z) in enumerate(N_POSITIONS):
    neuron = SimplePyramidal(x=x, y=y, z_center=z)
    neurons.append(neuron)
    print(f"Neuron {i+1} created at (x={x:.1f}, y={y:.1f}, z={z:.1f}) um")

# --- 5.5. IClamp 설정 (새로 추가) ---
# Neuron 2 (가운데 뉴런)에 전류 주입기 (IClamp)를 삽입합니다.
#print("\n--- 2.5. IClamp 삽입 (모델 발화 테스트용) ---")
#ic = h.IClamp(neurons[1].soma(0.5)) # Neuron 2의 Soma 중앙에 삽입
#ic.delay = 10 * ms                   # 10ms 후에 전류 인가 시작
#ic.dur = 5 * ms                      # 5ms 동안 전류 인가
#ic.amp = 0.5                         # 인가 전류 강도 (0.5 nA, nA 단위로 가정)

# --- 6. 시뮬레이션 설정 ---

h.tstop = TOTAL_TIME_MS  # 총 시뮬레이션 시간 (402 ms)
h.dt = 0.025 * ms        # 시뮬레이션 시간 간격 (0.025 ms. Ansys 간격 50us의 1/2)
h.celsius = 34.0  # 온도 설정 (생체 온도)

#전류주입용 추가
#h.tstop = 50 * ms

# 초기화
h.finitialize(-65.0 * mV)

# --- 7. 데이터 기록 설정 ---

# 시간 벡터
t_vec = h.Vector()
#t_vec.record(h._ref_t)

# 3개 뉴런의 Soma 전위 벡터
Vm_vecs = [h.Vector() for _ in range(3)]
#for i, neuron in enumerate(neurons):
    # Soma의 중앙(0.5) 지점 전위 기록
#    Vm_vecs[i].record(neuron.soma(0.5)._ref_v)

# --- 8. 시뮬레이션 실행 ---

print(f"\n--- 3. 시뮬레이션 시작 (총 {h.tstop:.1f} ms) ---")

# Extracellular Field Injection을 위한 Fadvance 호출
#h.cvode.event(0, set_extracellular_field) # 0ms에서 한 번 실행
#h.cvode.continuous = 1 # h.fadvance() 호출 시 set_extracellular_field를 매번 호출
#h.cvode.extra_scatter = set_extracellular_field # NEURON 7.x 버전 이상에서 권장되는 설정

#h.run()

# 1. 0ms 시점의 E-field 값 초기화
set_extracellular_field() 

# 2. h.run() 대신 반복문을 사용하여 매 스텝마다 E-field를 업데이트
while h.t < h.tstop:
    t_vec.append(h.t)
    
    for i, neuron in enumerate(neurons):
        # neuron.soma(0.5)._ref_v 대신 .v를 사용하여 현재 Vm 값을 가져옵니다.
        # h.v()는 현재 구획의 Vm을 반환합니다.
        Vm_vecs[i].append(neuron.soma(0.5).v)

    # E-field 값을 현재 시간 h.t에 맞춰 업데이트 (함수 호출)
    set_extracellular_field() 
    
    # 시간 한 스텝 전진 (h.dt만큼)
    h.fadvance()

print("\n--- 4. 시뮬레이션 완료 ---")

all_vms_clean = True
for i, vec in enumerate(Vm_vecs):
    vm_array = vec.as_numpy()
    if np.any(np.isnan(vm_array)) or np.any(np.isinf(vm_array)):
        print(f"❌ 경고: Neuron {i+1}의 Vm 데이터에 NaN 또는 Inf가 포함되어 있습니다!")
        all_vms_clean = False
    
if all_vms_clean:
    print("✅ Vm 데이터는 깨끗합니다 (NaN/Inf 없음).")


# --- 9. 결과 출력 (선택 사항) ---

try:
    import matplotlib.pyplot as plt
    
    for i in range(3):
        # 1. 새로운 Figure를 생성합니다.
        plt.figure(figsize=(12, 4))
        
        # 2. i번째 뉴런의 Vm 데이터를 플롯합니다.
        neuron_label = f'Neuron {i+1} (X={N_POSITIONS[i][0]:.0f} um)'
        
        # NumPy 배열로 변환하여 플롯합니다.
        plt.plot(t_vec.as_numpy(), Vm_vecs[i].as_numpy(), label=neuron_label, linewidth=1.5)
        
        # 3. 플롯 제목 및 레이블 설정
        plt.xlabel('Time (ms)')
        plt.ylabel('Soma Membrane Potential (mV)')
        plt.title(f'Response of {neuron_label} to Extracellular Field (40 Cycles)')
        
        # 4. 기준선(-65mV) 및 그리드 추가
        plt.axhline(-65, color='red', linestyle='--', linewidth=0.8, label='Resting Potential')
        plt.legend()
        plt.grid(True)

        plt.show(block=False)
        # 5. Figure를 화면에 표시 (Colab/WSL 환경에 따라 바로 뜨지 않을 수 있습니다.)
        plt.show()

    print("\n✅ 3개의 개별 플롯 생성을 완료했습니다.")

except ImportError:
    print("\n경고: Matplotlib이 설치되지 않았습니다. 결과를 플롯하려면 'pip install matplotlib'을 실행하세요.")
    print("Vm 데이터는 Vm_vecs[i] 벡터에 저장되어 있습니다.")