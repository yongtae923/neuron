# simulate_tES.py
import numpy as np
import os
import time
from neuron import h
from neuron.units import um, ms, mV
import math
from tqdm import tqdm

h.load_file("stdrun.hoc")

# --- 0. 모델 선택 및 E-field 적용 방식 ---
# 'simple' 또는 'allen' 중 선택
MODEL_TYPE = 'simple'  # 'simple' 또는 'allen'

# E-field 적용 방식: 'simple' (phi = -(E·r)) 또는 'integrated' (pt3d 기반 적분)
E_FIELD_METHOD = 'integrated'  # 'simple' 또는 'integrated'

# Allen 모델 cell ID 설정 (원하는 cell ID로 변경 가능)
ALLEN_CELL_ID = '486239338'  # 사용할 cell ID를 여기에 입력

# 플롯 타입 선택: 'all' (전체 40 cycles), 'single' (첫 1ms), 'both' (둘 다)
PLOT_TYPE = 'all'  # 'all', 'single', 또는 'both'

# --- 1. 파일 경로 및 시뮬레이션 상수 ---
# 스크립트가 있는 디렉토리를 기준으로 경로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
E_FIELD_VALUES_FILE = os.path.join(SCRIPT_DIR, 'E_field_40cycles.npy')
E_GRID_COORDS_FILE = os.path.join(SCRIPT_DIR, 'E_field_grid_coords.npy')

# 결과 저장 디렉토리
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'simulate_tES_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 디렉토리가 없으면 생성
TIME_STEP_US = 50.0  # Ansys 데이터 시간 간격 (us)
TIME_STEP_MS = TIME_STEP_US / 1000.0  # 0.05 ms
TOTAL_TIME_MS = 402.0 # 40 사이클을 포괄하는 전체 시간 (ms)
N_SPATIAL_POINTS = None # 로드 시 결정될 공간 그리드 지점 수

# FEM E-field 단위 스케일
# - Ansys 결과가 V/m인 경우: 1.0
# - V/mm로 export된 경우: 1000.0 (V/mm → V/m)
E_UNIT_SCALE = 1000.0  # V/mm → V/m 변환 (FEM 데이터가 V/mm 단위)

# E-field 단위 변환 계수 (V/m → mV)
# - E-field는 V/m 단위이므로 mV 단위로 변환해야 NEURON과 일치
# - 1 V/m = 1e-3 mV/um (거리는 이미 um 단위로 계산됨)
E_factor = 1e-3  # V/m → mV 변환 계수

TARGET_E_PEAK = 10.0  # V/m 목표 피크 (E-field 정규화용) - 검증용 10배 증가
# 검증용: 필드 크기를 10배 올리려면 TARGET_E_PEAK = 10.0으로 변경

# --- 2. 데이터 로드 및 전처리 ---

print("--- 1. 데이터 로드 및 전처리 ---")

# 2.1. E-필드 값 (Ex, Ez) 로드: Shape (2, N_spatial, 8040)
try:
    E_field_values = np.load(E_FIELD_VALUES_FILE)
    print(f"E-field values loaded: {E_field_values.shape}")

    # FEM 결과 단위 보정 (E_UNIT_SCALE)
    # - 기본은 1.0 (V/m로 가정)
    # - 만약 Ansys가 V/mm로 export 했다면 E_UNIT_SCALE = 1000.0으로 설정해야 함
    if E_UNIT_SCALE != 1.0:
        E_field_values = E_field_values * E_UNIT_SCALE
        print(f"  ✅ FEM E-field 단위 보정 적용: scale = {E_UNIT_SCALE:.1f} (V/mm → V/m 변환)")
    else:
        print(f"  단위 보정 없음: FEM 데이터가 이미 V/m 단위로 해석됨")
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

def xyz_at_seg(sec, segx):
    """
    세그먼트 위치(segx, 0.0~1.0)에서의 3D 좌표를 반환합니다.
    h.x3d()는 정수 인덱스를 받으므로, pt3d 점들을 선형 보간하여 계산합니다.
    """
    n = int(h.n3d(sec=sec))
    if n < 2:
        return 0.0, 0.0, 0.0
    
    # Section의 양 끝점 좌표
    x0 = h.x3d(0, sec=sec)
    y0 = h.y3d(0, sec=sec)
    z0 = h.z3d(0, sec=sec)
    x1 = h.x3d(n-1, sec=sec)
    y1 = h.y3d(n-1, sec=sec)
    z1 = h.z3d(n-1, sec=sec)
    
    # 선형 보간
    x = x0 + (x1 - x0) * segx
    y = y0 + (y1 - y0) * segx
    z = z0 + (z1 - z0) * segx
    
    return x, y, z

def translate_morphology(all_secs, dx, dy, dz):
    """
    모든 section의 pt3d 좌표를 이동시킵니다.
    Allen 모델을 목표 위치로 이동시키기 위해 사용됩니다.
    """
    for sec in all_secs:
        n = int(h.n3d(sec=sec))
        for i in range(n):
            x = h.x3d(i, sec=sec) + dx
            y = h.y3d(i, sec=sec) + dy
            z = h.z3d(i, sec=sec) + dz
            d = h.diam3d(i, sec=sec)
            h.pt3dchange(i, x, y, z, d, sec=sec)
    h.define_shape()  # shape 재정의

# --- 4. 시간 매핑 및 E-field 설정 함수 ---

# E-field 적용 방식에 따른 함수 선택
def get_E_at(spatial_idx, current_time_ms):
    """특정 공간 인덱스와 시간에서 E-field 값을 반환합니다."""
    time_index_float = current_time_ms / TIME_STEP_MS
    Tmax = E_field_values.shape[2] - 1
    
    # 시간 인덱스 경계 방어 (오프바이원 방지)
    t_idx_prev = int(math.floor(time_index_float))
    if t_idx_prev < 0:
        t_idx_prev = 0
    if t_idx_prev > Tmax:
        t_idx_prev = Tmax
    t_idx_next = min(t_idx_prev + 1, Tmax)
    
    ratio = time_index_float - t_idx_prev
    # ratio도 경계에서 클리핑
    if ratio < 0.0:
        ratio = 0.0
    if ratio > 1.0:
        ratio = 1.0

    Ex_prev = E_field_values[0, spatial_idx, t_idx_prev]
    Ex_next = E_field_values[0, spatial_idx, t_idx_next]
    Ez_prev = E_field_values[1, spatial_idx, t_idx_prev]
    Ez_next = E_field_values[1, spatial_idx, t_idx_next]

    Ex = Ex_prev + ratio * (Ex_next - Ex_prev)
    Ez = Ez_prev + ratio * (Ez_next - Ez_prev)

    # 정규화 스케일 적용
    if hasattr(set_extracellular_field, '_e_scale'):
        Ex *= set_extracellular_field._e_scale
        Ez *= set_extracellular_field._e_scale
    
    return Ex, 0.0, Ez

def interp_phi(arc_list, phi_list, target_arc):
    """arc_list에서 target_arc 위치의 phi 값을 보간합니다."""
    if len(arc_list) == 0 or len(phi_list) == 0:
        return 0.0
    if target_arc <= arc_list[0]:
        return phi_list[0]
    if target_arc >= arc_list[-1]:
        return phi_list[-1]

    # 이진 탐색
    lo, hi = 0, len(arc_list) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if arc_list[mid] <= target_arc:
            lo = mid
        else:
            hi = mid

    a0, a1 = arc_list[lo], arc_list[hi]
    p0, p1 = phi_list[lo], phi_list[hi]
    if a1 == a0:
        return p0
    w = (target_arc - a0) / (a1 - a0)
    return p0 + w * (p1 - p0)

def build_morph_cache(neuron, grid_coords_um):
    """뉴런의 morphology 캐시를 생성합니다 (pt3d 기반)."""
    cache = {}
    topo = {}

    for sec in neuron.all:
        n = int(h.n3d(sec=sec))
        if n < 2:
            cache[sec] = {"n": n, "arc": [0.0], "dl": [], "mid_spidx": []}
            continue

        xs = [h.x3d(i, sec=sec) for i in range(n)]
        ys = [h.y3d(i, sec=sec) for i in range(n)]
        zs = [h.z3d(i, sec=sec) for i in range(n)]
        arc = [h.arc3d(i, sec=sec) for i in range(n)]  # um

        dl = []
        mid_spidx = []
        for i in range(n - 1):
            dx = xs[i+1] - xs[i]
            dy = ys[i+1] - ys[i]
            dz = zs[i+1] - zs[i]
            dl.append((dx, dy, dz))

            mx = 0.5 * (xs[i] + xs[i+1])
            my = 0.5 * (ys[i] + ys[i+1])
            mz = 0.5 * (zs[i] + zs[i+1])

            spidx = find_nearest_spatial_index(mx, my, mz, grid_coords_um)
            mid_spidx.append(spidx)

        cache[sec] = {"n": n, "arc": arc, "dl": dl, "mid_spidx": mid_spidx}

        # 부모 section 정보
        sref = h.SectionRef(sec=sec)
        if sref.has_parent():
            # 우선 parentseg() 정보를 사용 (실제 연결 위치가 가장 정확)
            try:
                pseg = sref.parentseg()
                topo[sec] = (pseg.sec, float(pseg.x))
            except:
                # parentseg가 없으면 기존 방식으로 추정
                parent_sec = sref.parent
                if n > 0:
                    child_x0 = xs[0]
                    child_y0 = ys[0]
                    child_z0 = zs[0]
                    
                    pn = int(h.n3d(sec=parent_sec))
                    min_dist = float('inf')
                    parent_x = 0.0
                    
                    for pi in range(pn):
                        px = h.x3d(pi, sec=parent_sec)
                        py = h.y3d(pi, sec=parent_sec)
                        pz = h.z3d(pi, sec=parent_sec)
                        dist = ((px - child_x0)**2 + (py - child_y0)**2 + (pz - child_z0)**2)**0.5
                        if dist < min_dist:
                            min_dist = dist
                            parc = h.arc3d(pi, sec=parent_sec)
                            total_parc = h.arc3d(pn-1, sec=parent_sec) if pn > 0 else 1.0
                            parent_x = parc / total_parc if total_parc > 0 else 0.0
                    
                    topo[sec] = (parent_sec, parent_x)
                else:
                    topo[sec] = (parent_sec, 1.0)
        else:
            topo[sec] = None

    return cache, topo

def compute_phi_sections(neuron, morph_cache, topo, current_time_ms):
    """섹션 트리를 따라 phi를 누적 적분하여 계산합니다."""
    phi_sec = {}

    def ensure_section_phi(sec):
        if sec in phi_sec:
            return

        parent = topo.get(sec, None)
        if parent is None:
            phi0 = 0.0
        else:
            psec, px = parent
            ensure_section_phi(psec)
            parc, pphi = phi_sec[psec]
            total_arc = parc[-1] if len(parc) > 0 else 0.0
            phi0 = interp_phi(parc, pphi, px * total_arc)

        data = morph_cache[sec]
        n = data["n"]
        if n < 2:
            phi_sec[sec] = ([0.0], [phi0])
            return

        arc = data["arc"]
        dl = data["dl"]
        mid_spidx = data["mid_spidx"]

        phis = [phi0]
        for i in range(n - 1):
            spidx = mid_spidx[i]
            Ex, Ey, Ez = get_E_at(spidx, current_time_ms)
            dx, dy, dz = dl[i]
            dphi = -(Ex * dx + Ey * dy + Ez * dz) * E_factor  # mV
            phis.append(phis[-1] + dphi)

        phi_sec[sec] = (arc, phis)

    # 모든 section에 대해 트리 방향으로 계산
    for sec in neuron.all:
        ensure_section_phi(sec)

    return phi_sec

def apply_phi_to_segments(neuron, phi_sec):
    """계산된 phi를 각 세그먼트의 e_extracellular에 적용합니다."""
    for sec in neuron.all:
        if sec not in phi_sec:
            continue
        arc, phis = phi_sec[sec]
        total_arc = arc[-1] if len(arc) > 0 else 0.0
        # ghost node(0,1)를 제외하고 실제 세그먼트만 순회하기 위해 sec를 직접 순회
        for seg in sec:
            target_arc = seg.x * total_arc
            phi = interp_phi(arc, phis, target_arc)
            seg.e_extracellular = phi

def set_extracellular_field():
    """
    현재 NEURON 시간 (h.t)을 기반으로 모든 뉴런 구획의 E-field 값을 설정합니다.
    E_FIELD_METHOD에 따라 'simple' (phi = -(E·r)) 또는 'integrated' (pt3d 기반 적분) 방식을 사용합니다.
    """
    # 워밍업 시간 이전이면 E-field를 0으로 설정
    if h.t < WARMUP_TIME_MS:
        for neuron in neurons:
            for sec in neuron.all:
                # ghost node(0,1)를 제외하고 실제 세그먼트만 순회
                for seg in sec:
                    seg.e_extracellular = 0.0
        return 0
    
    # 워밍업 시간 이후의 상대 시간 사용
    current_time_ms = h.t - WARMUP_TIME_MS  # E-field 데이터 인덱스 계산용
    
    # E-field 정규화 스케일 계산 (한 번만)
    if not hasattr(set_extracellular_field, '_e_scale_calculated'):
        all_Ex_max = np.max(np.abs(E_field_values[0, :, :]))
        all_Ez_max = np.max(np.abs(E_field_values[1, :, :]))
        measured_peak = max(all_Ex_max, all_Ez_max)
        if measured_peak > 0:
            set_extracellular_field._e_scale = TARGET_E_PEAK / measured_peak
        else:
            set_extracellular_field._e_scale = 1.0
        set_extracellular_field._e_scale_calculated = True
    
    # E-field 적용 방식에 따라 분기
    if E_FIELD_METHOD == 'integrated':
        # 선택지 B: pt3d 기반 적분 방식
        global morph_caches, topos
        for i, neuron in enumerate(neurons):
            phi_sec = compute_phi_sections(neuron, morph_caches[i], topos[i], current_time_ms)
            apply_phi_to_segments(neuron, phi_sec)
    else:
        # 선택지 A: 간단한 방식 (phi = -(E·r))
        # 1. 시간 인덱스 계산 및 보간 (경계 클리핑)
        time_index_float = current_time_ms / TIME_STEP_MS
        Tmax = E_field_values.shape[2] - 1
        t_idx_prev = int(math.floor(time_index_float))
        if t_idx_prev < 0:
            t_idx_prev = 0
        if t_idx_prev > Tmax:
            t_idx_prev = Tmax
        t_idx_next = min(t_idx_prev + 1, Tmax)
        ratio = time_index_float - t_idx_prev
        # ratio도 경계에서 클리핑
        if ratio < 0.0:
            ratio = 0.0
        if ratio > 1.0:
            ratio = 1.0
        
        # 2. 모든 뉴런의 모든 구획에 필드 설정
        for neuron in neurons:
            for sec in neuron.all:
                # ghost node(0,1)를 제외하고 실제 세그먼트만 순회
                for seg in sec:
                    # 공간 매핑
                    seg_id = (id(neuron), sec.name(), seg.x)
                    if seg_id not in global_spatial_map:
                        seg_x, seg_y, seg_z = xyz_at_seg(sec, seg.x)
                        nearest_idx = find_nearest_spatial_index(seg_x, seg_y, seg_z, E_grid_coords_UM)
                        global_spatial_map[seg_id] = nearest_idx
                    
                    spatial_idx = global_spatial_map[seg_id]
                    
                    # 시간 보간
                    Ex_prev = E_field_values[0, spatial_idx, t_idx_prev]
                    Ex_next = E_field_values[0, spatial_idx, t_idx_next]
                    Ez_prev = E_field_values[1, spatial_idx, t_idx_prev]
                    Ez_next = E_field_values[1, spatial_idx, t_idx_next]
                    
                    Ex_interp = Ex_prev + ratio * (Ex_next - Ex_prev)
                    Ez_interp = Ez_prev + ratio * (Ez_next - Ez_prev)
                    
                    # 정규화
                    E_x = Ex_interp * set_extracellular_field._e_scale
                    E_y = 0.0
                    E_z = Ez_interp * set_extracellular_field._e_scale
                    
                    # 세그먼트 중심 좌표
                    seg_x, seg_y, seg_z = xyz_at_seg(sec, seg.x)
                    
                    # Extracellular potential 설정 (절대 전위)
                    phi_mV = -(E_x * seg_x + E_y * seg_y + E_z * seg_z) * E_factor  # mV
                    seg.e_extracellular = phi_mV
                
    return 0 # h.fadvance()를 위한 반환값 (필수)


# --- 5. 뉴런 생성 및 배치 ---

# 3개 뉴런의 중심 위치 (um)
N_POSITIONS = [
    (-90.0 * um, 42.0 * um, 561.0 * um), # Neuron 1
    (0.0 * um, 42.0 * um, 561.0 * um),   # Neuron 2
    (90.0 * um, 42.0 * um, 561.0 * um)    # Neuron 3
]

# 모델 import (미리 import)
from simple_pyramidal_model import SimplePyramidal
from allen_neuron_model import AllenNeuronModel, set_allen_cell_id, ALLEN_DATA_DIR
import re

# Allen 모델 데이터 설정
set_allen_cell_id(ALLEN_CELL_ID)

# 폴더명에서 cell ID 확인 (검증용, 덮어쓰지 않음)
print(f"\n--- 데이터 경로 검증 ---")
print(f"설정한 Cell ID: {ALLEN_CELL_ID}")
print(f"전역 ALLEN_DATA_DIR: {ALLEN_DATA_DIR}")
folder_name = os.path.basename(ALLEN_DATA_DIR)
match = re.search(r'(\d+)$', folder_name)
if match:
    extracted_id = match.group(1)
    print(f"전역 ALLEN_DATA_DIR에서 추출한 ID: {extracted_id}")
    if extracted_id != ALLEN_CELL_ID:
        print(f"⚠️ 경고: 설정한 Cell ID ({ALLEN_CELL_ID})와 전역 ALLEN_DATA_DIR의 ID ({extracted_id})가 다릅니다.")
        print(f"   하지만 AllenNeuronModel은 cell_id 파라미터를 직접 사용하므로 실제로는 올바른 폴더를 사용합니다.")
else:
    print(f"⚠️ 경고: 폴더명에서 ID를 추출할 수 없습니다: {folder_name}")

# 실제 사용될 경로 확인
expected_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'allen_neuron_{ALLEN_CELL_ID}')
print(f"실제 사용될 예상 경로: {expected_data_dir}")
if os.path.exists(expected_data_dir):
    print(f"✅ 예상 경로가 존재합니다.")
    swc_files = [f for f in os.listdir(expected_data_dir) if f.endswith('.swc')]
    print(f"   SWC 파일 개수: {len(swc_files)}")
    if swc_files:
        print(f"   SWC 파일 예시: {swc_files[0]}")
else:
    print(f"❌ 예상 경로가 존재하지 않습니다!")

# --- 6. 시뮬레이션 실행 ---

WARMUP_TIME_MS = 200.0  # 워밍업 시간 (ms) - 초기화 오차 수렴용

# E-field 스케일 리셋
for attr in ['_e_scale_calculated', '_e_scale']:
    if hasattr(set_extracellular_field, attr):
        delattr(set_extracellular_field, attr)

# --- 6.1. 뉴런 생성 및 배치 ---
print("\n--- 2. 뉴런 모델 생성 및 배치 ---")
print(f"사용 모델: {MODEL_TYPE}")

if MODEL_TYPE == 'allen' and ALLEN_CELL_ID:
    print(f"Cell ID: {ALLEN_CELL_ID}")

# 뉴런 인스턴스 생성 (전역 변수 사용)
global neurons, global_spatial_map, morph_caches, topos
# 전역 변수 초기화
global_spatial_map = {}
neurons = []
morph_caches = []
topos = []

for i, (x, y, z) in enumerate(N_POSITIONS):
    if MODEL_TYPE == 'simple':
        neuron = SimplePyramidal(x=x, y=y, z_center=z)
    elif MODEL_TYPE == 'allen':
        # Allen 모델은 먼저 원점에서 생성
        # cell_id를 지정하면 자동으로 해당 폴더의 파일을 찾아서 사용
        print(f"\n--- AllenNeuronModel 생성 (Neuron {i+1}) ---")
        print(f"   전달된 cell_id: {ALLEN_CELL_ID}")
        expected_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'allen_neuron_{ALLEN_CELL_ID}')
        print(f"   예상 데이터 경로: {expected_path}")
        if os.path.exists(expected_path):
            print(f"   ✅ 예상 경로가 존재합니다.")
        else:
            print(f"   ❌ 예상 경로가 존재하지 않습니다!")
        neuron = AllenNeuronModel(x=0, y=0, z=0, cell_id=ALLEN_CELL_ID)
        print(f"   ✅ AllenNeuronModel 생성 완료")
        # 소마 기준으로 재센터링 후 목표 위치로 이동
        sx, sy, sz = xyz_at_seg(neuron.soma, 0.5)  # 현재 soma 좌표
        tx, ty, tz = float(x), float(y), float(z)  # 목표 soma 좌표 (um)
        translate_morphology(neuron.all, tx - sx, ty - sy, tz - sz)
    neurons.append(neuron)
    print(f"Neuron {i+1} created at (x={x:.1f}, y={y:.1f}, z={z:.1f}) um")

# E-field 적용 방식에 따라 morph_cache 생성 (integrated 방식일 때만)
if E_FIELD_METHOD == 'integrated':
    print("\n--- Morphology 캐시 생성 (pt3d 기반 적분 방식) ---")
    for neuron in neurons:
        cache, topo = build_morph_cache(neuron, E_grid_coords_UM)
        morph_caches.append(cache)
        topos.append(topo)
    print("  ✅ Morphology 캐시 생성 완료")

# --- 6.2. 시뮬레이션 설정 ---
h.tstop = WARMUP_TIME_MS + TOTAL_TIME_MS  # 총 시뮬레이션 시간 (워밍업 + 402 ms)
h.dt = 0.025 * ms        # 시뮬레이션 시간 간격 (0.025 ms. Ansys 간격 50us의 1/2)
h.celsius = 34.0  # 온도 설정 (생체 온도)

# 초기화
h.finitialize(-65.0 * mV)

# --- 6.3. 데이터 기록 설정 ---
# 시간 벡터
t_vec = h.Vector()

# 3개 뉴런의 Soma 전위 벡터
Vm_vecs = [h.Vector() for _ in range(3)]
# 3개 뉴런의 Soma vext 벡터 (디버깅용)
Vext_vecs = [h.Vector() for _ in range(3)]
# Distal dendrite 전위 벡터 (검증용 - soma보다 변화가 클 수 있음)
Vm_distal_vecs = [h.Vector() for _ in range(3)]
Vext_distal_vecs = [h.Vector() for _ in range(3)]

# Distal segment 찾기 (각 뉴런마다): 모든 section 끝점 중 soma에서 가장 먼 점 선택
distal_segments = []
for neuron in neurons:
        distal_seg = neuron.soma(0.5)
        try:
            soma_x, soma_y, soma_z = xyz_at_seg(neuron.soma, 0.5)
        except:
            soma_x = soma_y = soma_z = 0.0
        max_dist = -1.0
        # Distal 정의:
        # - Allen 모델: dendrites 리스트가 있으면 dendrites만 대상으로 가장 먼 지점 선택
        # - 그 외: 모든 section 끝점 중 soma에서 가장 먼 지점 선택
        sec_list = getattr(neuron, 'all', [])
        dend_list = getattr(neuron, 'dendrites', [])
        if MODEL_TYPE == 'allen' and len(dend_list) > 0:
            sec_list = dend_list

        for sec in sec_list:
            try:
                tip_x, tip_y, tip_z = xyz_at_seg(sec, 1.0)
                dist = ((tip_x - soma_x)**2 + (tip_y - soma_y)**2 + (tip_z - soma_z)**2)**0.5
                if dist > max_dist:
                    max_dist = dist
                    distal_seg = sec(1.0)
            except:
                continue
        distal_segments.append(distal_seg)

# e_extracellular range 추적 (시간 전체에서 최대값)
e_range_max = [0.0, 0.0, 0.0]

# 실제 resting potential 기록 (워밍업 끝 시점의 Vm 평균)
resting_potentials = []

# --- 6.4. 시뮬레이션 실행 ---
print(f"\n--- 3. 시뮬레이션 시작 (총 {h.tstop:.1f} ms) ---")

# 1. 워밍업 단계: E-field 없이 평형으로 수렴
print(f"워밍업 중... (0 ~ {WARMUP_TIME_MS} ms)")
while h.t < WARMUP_TIME_MS:
    h.fadvance()

print(f"워밍업 완료. E-field 적용 시작... ({WARMUP_TIME_MS} ~ {h.tstop:.1f} ms)")

# 워밍업 끝 시점의 Vm을 실제 resting potential으로 기록
resting_potentials = []
for i, neuron in enumerate(neurons):
    resting_potentials.append(neuron.soma(0.5).v)
print(f"  실제 Resting Potential: Neuron 1={resting_potentials[0]:.2f} mV, Neuron 2={resting_potentials[1]:.2f} mV, Neuron 3={resting_potentials[2]:.2f} mV")

# 2. E-field 적용 단계: 매 스텝마다 E-field를 업데이트
# 중요: 필드 적용 → 기록 → fadvance 순서

# 전체 시뮬레이션 시간 계산 (E-field 적용 부분만)
total_sim_time = h.tstop - WARMUP_TIME_MS
total_steps = int(total_sim_time / h.dt)

# tqdm 진행률 바 초기화
pbar = tqdm(total=total_steps, desc="E-field 시뮬레이션", unit="step", unit_scale=True, ncols=100)

while h.t < h.tstop:
        # 현재 시간 (워밍업 시간을 빼서 E-field 데이터 인덱스 계산)
        t_relative = h.t - WARMUP_TIME_MS
        
        # E-field 값을 현재 시간에 맞춰 업데이트 (먼저 적용)
        if t_relative >= 0:
            set_extracellular_field()
        else:
            # 워밍업 중에는 E-field를 0으로 설정
            for neuron in neurons:
                for sec in neuron.all:
                    # ghost node(0,1)를 제외하고 실제 세그먼트만 순회
                    for seg in sec:
                        seg.e_extracellular = 0.0
        
        # 기록 (필드 적용 후)
        t_vec.append(t_relative)  # 상대 시간 기록
        
        for i, neuron in enumerate(neurons):
            # Soma Vm 기록
            Vm_vecs[i].append(neuron.soma(0.5).v)
            
            # Soma vext 기록 (디버깅용)
            try:
                if hasattr(neuron.soma(0.5), 'vext'):
                    Vext_vecs[i].append(neuron.soma(0.5).vext[0])
                else:
                    Vext_vecs[i].append(0.0)
            except:
                Vext_vecs[i].append(0.0)
            
            # Distal dendrite Vm/vext 기록 (검증용)
            try:
                Vm_distal_vecs[i].append(distal_segments[i].v)
                if hasattr(distal_segments[i], 'vext'):
                    Vext_distal_vecs[i].append(distal_segments[i].vext[0])
                else:
                    Vext_distal_vecs[i].append(0.0)
            except:
                Vm_distal_vecs[i].append(neuron.soma(0.5).v)
                Vext_distal_vecs[i].append(0.0)
            
            # e_extracellular range 추적 (1ms 간격으로만 계산하여 성능 최적화)
            if t_relative >= 0 and int(t_relative / h.dt) % int(1.0 / h.dt) == 0:
                e_ext_values = []
                for sec in neuron.all:
                    # ghost node(0,1)를 제외하고 실제 세그먼트만 사용
                    for seg in sec:
                        e_ext_values.append(seg.e_extracellular)
                if len(e_ext_values) > 0:
                    r = max(e_ext_values) - min(e_ext_values)
                    if r > e_range_max[i]:
                        e_range_max[i] = r
        
        # 시간 한 스텝 전진 (h.dt만큼)
        h.fadvance()
        
        # tqdm 진행률 업데이트 (E-field 적용 구간만)
        if t_relative >= 0:
            current_step = int(t_relative / h.dt)
            if current_step < total_steps:
                pbar.update(1)
            elif current_step >= total_steps:
                pbar.update(total_steps - pbar.n)

# tqdm 진행률 바 종료
pbar.close()

print("\n--- 4. 시뮬레이션 완료 ---")

all_vms_clean = True
for i, vec in enumerate(Vm_vecs):
    vm_array = vec.as_numpy()
    if np.any(np.isnan(vm_array)) or np.any(np.isinf(vm_array)):
        print(f"❌ 경고: Neuron {i+1}의 Vm 데이터에 NaN 또는 Inf가 포함되어 있습니다!")
        all_vms_clean = False
    
if all_vms_clean:
    print("✅ Vm 데이터는 깨끗합니다 (NaN/Inf 없음).")

# --- 검증 체크리스트 ---
print("\n" + "="*60)
print("검증 체크리스트")
print("="*60)

# 체크 1: FEM 필드 크기 확인
print("\n[체크 1] FEM 필드 크기 및 단위 확인")
all_Ex_max = np.max(np.abs(E_field_values[0, :, :]))
all_Ez_max = np.max(np.abs(E_field_values[1, :, :]))
measured_peak = max(all_Ex_max, all_Ez_max)
print(f"  전체 데이터 최대 E-field: {measured_peak:.6f} V/m")
if hasattr(set_extracellular_field, '_e_scale'):
    print(f"  정규화 스케일: {set_extracellular_field._e_scale:.6f}")
    print(f"  정규화 후 목표 피크: {TARGET_E_PEAK:.2f} V/m")

# 기대 범위: tES/tACS 수준은 보통 0.1 ~ 2 V/m
if measured_peak < 1e-2:  # 0.01 V/m 미만
    print("  ⚠️ 경고: E-field가 너무 작습니다 (< 0.01 V/m).")
    print("           tES/tACS 수준(0.1 ~ 2 V/m)보다 10배 이상 작습니다.")
    print("           FEM 결과의 단위를 확인하세요 (V/m vs V/mm vs V/um 등).")
elif measured_peak < 0.1:  # 0.1 V/m 미만
    print("  ⚠️ 주의: E-field가 작습니다 (< 0.1 V/m).")
    print("           tES/tACS 일반 범위(0.1 ~ 2 V/m)보다 작습니다.")
    print("           단위나 스케일을 재확인하는 것을 권장합니다.")
elif measured_peak > 20:  # 20 V/m 초과
    print("  ⚠️ 경고: E-field가 매우 큽니다 (> 20 V/m).")
    print("           tES/tACS 일반 범위(0.1 ~ 2 V/m)보다 10배 이상 큽니다.")
    print("           단위를 확인하세요 (V/m vs V/mm 등).")
else:
    print("  ✅ E-field 크기가 정상 범위입니다 (0.1 ~ 2 V/m, tES/tACS 수준)")

# 체크 2: 뉴런 3개가 정말 다른 공간 포인트를 보고 있는지 확인
print("\n[체크 2] 뉴런 3개 공간 위치 확인")
nearest_indices = []
for i, neuron in enumerate(neurons):
    soma_x, soma_y, soma_z = xyz_at_seg(neuron.soma, 0.5)
    nearest_idx = find_nearest_spatial_index(soma_x, soma_y, soma_z, E_grid_coords_UM)
    nearest_indices.append(nearest_idx)
    
    print(f"  Neuron {i+1}:")
    print(f"    실제 좌표: X={soma_x:.1f}, Y={soma_y:.1f}, Z={soma_z:.1f} um")
    print(f"    기대 좌표: X={N_POSITIONS[i][0]:.1f}, Y={N_POSITIONS[i][1]:.1f}, Z={N_POSITIONS[i][2]:.1f} um")
    print(f"    Grid index: {nearest_idx}")
    
    # E-field 값의 최대값 확인
    Ex_max = np.max(np.abs(E_field_values[0, nearest_idx, :]))
    Ez_max = np.max(np.abs(E_field_values[1, nearest_idx, :]))
    print(f"    Max |Ex|: {Ex_max:.6f} V/m, Max |Ez|: {Ez_max:.6f} V/m")

# Grid index가 모두 다른지 확인
if len(set(nearest_indices)) == 3:
    print("  ✅ 3개 뉴런이 서로 다른 그리드 포인트를 보고 있습니다.")
else:
    print(f"  ⚠️ 경고: {len(set(nearest_indices))}개의 고유한 그리드 포인트만 사용됨 (공간 의존성 문제 가능)")

# 체크 3: 한 뉴런 내부에서 e_extracellular 구배 확인 (시간 전체 최대값)
print("\n[체크 3] 뉴런 내부 e_extracellular 구배 확인 (시간 전체 최대값)")
for i, neuron in enumerate(neurons):
    print(f"  Neuron {i+1}:")
    # 시간 전체에서 추적한 최대 range
    e_ext_range = e_range_max[i]
    print(f"    e_extracellular range (시간 전체 최대값): {e_ext_range:.6f} mV")
    
    if e_ext_range < 0.01:  # 10 uV 미만
        print("    ⚠️ 경고: 구배가 너무 작습니다 (< 10 uV). 분극이 거의 생기지 않을 수 있습니다.")
    elif e_ext_range < 0.1:  # 100 uV 미만
        print("    ⚠️ 주의: 구배가 작습니다 (< 100 uV). Vm 변화가 작을 수 있습니다.")
    else:
        print("    ✅ 구배가 충분합니다 (> 100 uV).")
    
    # vext 최대값 확인
    if len(Vext_vecs[i]) > 0:
        vext_array = Vext_vecs[i].as_numpy()
        vext_max = np.max(np.abs(vext_array))
        print(f"    Max |vext|: {vext_max:.6f} mV")

# 검증 3종 세트: ΔVm 수치로 찍기
print("\n[검증] ΔVm 수치 분석")

# soma peak-to-peak 저장 (각 뉴런별로)
soma_p2p = []
print("  (Soma)")
for i in range(3):
    vm_array = Vm_vecs[i].as_numpy()
    dvm = vm_array - vm_array.mean()
    peak_to_peak = dvm.max() - dvm.min()
    rms_dvm = np.sqrt(np.mean(dvm**2))
    soma_p2p.append(peak_to_peak)  # 리스트에 저장
    
    print(f"  Neuron {i+1}:")
    print(f"    Peak-to-peak ΔVm: {peak_to_peak:.6f} mV ({peak_to_peak*1000:.3f} uV)")
    print(f"    RMS ΔVm: {rms_dvm:.6f} mV ({rms_dvm*1000:.3f} uV)")
    if peak_to_peak < 0.001:  # 1 uV 미만
        print("    ⚠️ 주의: ΔVm이 매우 작습니다 (< 1 uV). 플롯에서 안 보이는 것이 정상입니다.")
    elif peak_to_peak < 0.01:  # 10 uV 미만
        print("    ⚠️ 주의: ΔVm이 작습니다 (< 10 uV). 플롯에서 잘 안 보일 수 있습니다.")
    else:
        print("    ✅ ΔVm이 충분히 큽니다 (> 10 uV).")

print("  (Distal dendrite)")
for i in range(3):
    vm_distal_array = Vm_distal_vecs[i].as_numpy()
    dvm_distal = vm_distal_array - vm_distal_array.mean()
    peak_to_peak_distal = dvm_distal.max() - dvm_distal.min()
    rms_dvm_distal = np.sqrt(np.mean(dvm_distal**2))
    
    # 각 뉴런의 soma 값과 비교
    ratio = (peak_to_peak_distal / soma_p2p[i]) if soma_p2p[i] > 0 else float("inf")
    
    print(f"  Neuron {i+1}:")
    print(f"    Peak-to-peak ΔVm: {peak_to_peak_distal:.6f} mV ({peak_to_peak_distal*1000:.3f} uV)")
    print(f"    RMS ΔVm: {rms_dvm_distal:.6f} mV ({rms_dvm_distal*1000:.3f} uV)")
    print(f"    Distal/Soma 비율: {ratio:.2f}x")
    if ratio > 1.0:
        print(f"    ✅ Distal dendrite에서 변화가 더 큽니다 (soma 대비 {ratio:.2f}배)")
    elif ratio < 1.0:
        print(f"    ⚠️ Distal dendrite에서 변화가 soma보다 작습니다 (soma 대비 {ratio:.2f}배)")

print()

# --- 6.5. 결과 출력 ---
try:
        import matplotlib.pyplot as plt
        
        print(f"\n--- 5. 결과 플롯 생성 및 저장 ---")
        print(f"결과 저장 디렉토리: {OUTPUT_DIR}")
        print(f"플롯 타입: {PLOT_TYPE}")
        
        # --- All 플롯 (전체 40 cycles) ---
        if PLOT_TYPE in ['all', 'both']:
            # 1. 하나의 Figure에 3개의 서브플롯 생성 (세로로 배치)
            # Vm/vext 플롯용
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            if MODEL_TYPE == 'allen' and ALLEN_CELL_ID:
                fig.suptitle(f'Neuron Response to Extracellular Field (40 Cycles) - {MODEL_TYPE.upper()} (Cell ID: {ALLEN_CELL_ID})', fontsize=14, fontweight='bold')
            else:
                fig.suptitle(f'Neuron Response to Extracellular Field (40 Cycles) - {MODEL_TYPE.upper()}', fontsize=14, fontweight='bold')
            
            # 전체 데이터에서 Vm과 vext의 최대/최소값 계산 (축 범위 통일용)
            all_vm_min = float('inf')
            all_vm_max = float('-inf')
            all_vext_min = float('inf')
            all_vext_max = float('-inf')
            
            for i in range(3):
                vm_array = Vm_vecs[i].as_numpy()
                all_vm_min = min(all_vm_min, np.min(vm_array))
                all_vm_max = max(all_vm_max, np.max(vm_array))
                
                if len(Vext_vecs[i]) > 0:
                    vext_array = Vext_vecs[i].as_numpy()
                    all_vext_min = min(all_vext_min, np.min(vext_array))
                    all_vext_max = max(all_vext_max, np.max(vext_array))
            
            # Vm 축 범위에 여유 공간 추가
            vm_range = all_vm_max - all_vm_min
            vm_margin = vm_range * 0.05  # 5% 여유
            vm_ylim = (all_vm_min - vm_margin, all_vm_max + vm_margin)
            
            # vext 축 범위에 여유 공간 추가
            if all_vext_max > all_vext_min:
                vext_range = all_vext_max - all_vext_min
                vext_margin = vext_range * 0.05  # 5% 여유
                vext_ylim = (all_vext_min - vext_margin, all_vext_max + vext_margin)
            else:
                vext_ylim = None
            
            for i in range(3):
                ax = axes[i]
                
                # 2. i번째 뉴런의 Vm 데이터를 플롯합니다.
                neuron_label = f'Neuron {i+1} (X={N_POSITIONS[i][0]:.0f} um)'
            
                # NumPy 배열로 변환
                t_array = t_vec.as_numpy()
                vm_array = Vm_vecs[i].as_numpy()
                
                # 원본 Vm 플롯
                ax.plot(t_array, vm_array, label=f'{neuron_label} (Vm)', linewidth=1.5, color='blue')
                
                # vext도 플롯 (스케일 조정 없이, 축 단위만 조정)
                if len(Vext_vecs[i]) > 0:
                    vext_array = Vext_vecs[i].as_numpy()
                    ax2 = ax.twinx()
                    ax2.plot(t_array, vext_array, label=f'{neuron_label} (vext)', 
                            linewidth=1.0, color='green', alpha=0.7, linestyle='--')
                    ax2.set_ylabel('Extracellular Potential (mV)', color='green')
                    ax2.tick_params(axis='y', labelcolor='green')
                    ax2.spines['right'].set_position(('outward', 0))
                    if vext_ylim:
                        ax2.set_ylim(vext_ylim)
                    ax2.legend(loc='upper right')
                
                # 3. 플롯 제목 및 레이블 설정
                ax.set_xlabel('Time (ms)')
                ax.set_ylabel('Soma Membrane Potential (mV)', color='blue')
                ax.set_title(f'Response of {neuron_label}')
                ax.tick_params(axis='y', labelcolor='blue')
                # Vm 축 범위 통일
                ax.set_ylim(vm_ylim)
                
                # 4. 기준선(실제 resting potential) 및 그리드 추가
                actual_resting = resting_potentials[i]
                ax.axhline(actual_resting, color='red', linestyle='--', linewidth=0.8, label=f'Resting Potential ({actual_resting:.1f} mV)')
                ax.legend(loc='upper left')
                ax.grid(True)
            
            # 5. 서브플롯 간 간격 조정
            plt.tight_layout()
            
            # 6. 파일로 저장
            if MODEL_TYPE == 'allen' and ALLEN_CELL_ID:
                output_filename = f'allen_{ALLEN_CELL_ID}_neuron_response_all_v3.png'
            else:
                output_filename = f'simple_neuron_response_all_v3.png'
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()  # 메모리 절약을 위해 figure 닫기
            print(f"   ✅ 저장됨: {output_filename}")
            print(f"\n✅ All 플롯이 '{OUTPUT_DIR}' 디렉토리에 저장되었습니다.")
        
        # --- Single Cycle 플롯 추가 (첫 1ms만) ---
        if PLOT_TYPE in ['single', 'both']:
            print(f"\n--- 6. Single Cycle 플롯 생성 및 저장 (첫 1ms) ---")
            
            # 첫 1ms만 표시
            SINGLE_CYCLE_TIME_MS = 1.0  # 1ms
            
            # Single cycle 데이터 추출 (첫 1ms)
            t_array_all = t_vec.as_numpy()
            cycle_mask = (t_array_all >= 0) & (t_array_all <= SINGLE_CYCLE_TIME_MS)
            
            if np.any(cycle_mask):
                # 1. 하나의 Figure에 3개의 서브플롯 생성 (세로로 배치)
                fig_single, axes_single = plt.subplots(3, 1, figsize=(12, 10))
                if MODEL_TYPE == 'allen' and ALLEN_CELL_ID:
                    fig_single.suptitle(f'Neuron Response to Extracellular Field (First 1 ms) - {MODEL_TYPE.upper()} (Cell ID: {ALLEN_CELL_ID})', fontsize=14, fontweight='bold')
                else:
                    fig_single.suptitle(f'Neuron Response to Extracellular Field (First 1 ms) - {MODEL_TYPE.upper()}', fontsize=14, fontweight='bold')
                
                # Single cycle 데이터에서 Vm과 vext의 최대/최소값 계산 (축 범위 통일용)
                single_vm_min = float('inf')
                single_vm_max = float('-inf')
                single_vext_min = float('inf')
                single_vext_max = float('-inf')
                
                for i in range(3):
                    vm_array_single = Vm_vecs[i].as_numpy()[cycle_mask]
                    single_vm_min = min(single_vm_min, np.min(vm_array_single))
                    single_vm_max = max(single_vm_max, np.max(vm_array_single))
                    
                    if len(Vext_vecs[i]) > 0:
                        vext_array_single = Vext_vecs[i].as_numpy()[cycle_mask]
                        single_vext_min = min(single_vext_min, np.min(vext_array_single))
                        single_vext_max = max(single_vext_max, np.max(vext_array_single))
                
                # Vm 축 범위에 여유 공간 추가
                single_vm_range = single_vm_max - single_vm_min
                single_vm_margin = single_vm_range * 0.05 if single_vm_range > 0 else 0.1
                single_vm_ylim = (single_vm_min - single_vm_margin, single_vm_max + single_vm_margin)
                
                # vext 축 범위에 여유 공간 추가
                if single_vext_max > single_vext_min:
                    single_vext_range = single_vext_max - single_vext_min
                    single_vext_margin = single_vext_range * 0.05
                    single_vext_ylim = (single_vext_min - single_vext_margin, single_vext_max + single_vext_margin)
                else:
                    single_vext_ylim = None
                
                for i in range(3):
                    ax = axes_single[i]
                    
                    # Single cycle 데이터 추출
                    t_array_single = t_array_all[cycle_mask]
                    vm_array_single = Vm_vecs[i].as_numpy()[cycle_mask]
                    
                    # 2. i번째 뉴런의 Vm 데이터를 플롯합니다.
                    neuron_label = f'Neuron {i+1} (X={N_POSITIONS[i][0]:.0f} um)'
                
                    # 원본 Vm 플롯
                    ax.plot(t_array_single, vm_array_single, label=f'{neuron_label} (Vm)', linewidth=1.5, color='blue')
                    
                    # vext도 플롯 (스케일 조정 없이, 축 단위만 조정)
                    if len(Vext_vecs[i]) > 0:
                        vext_array_single = Vext_vecs[i].as_numpy()[cycle_mask]
                        ax2 = ax.twinx()
                        ax2.plot(t_array_single, vext_array_single, label=f'{neuron_label} (vext)', 
                                linewidth=1.0, color='green', alpha=0.7, linestyle='--')
                        ax2.set_ylabel('Extracellular Potential (mV)', color='green')
                        ax2.tick_params(axis='y', labelcolor='green')
                        ax2.spines['right'].set_position(('outward', 0))
                        if single_vext_ylim:
                            ax2.set_ylim(single_vext_ylim)
                        ax2.legend(loc='upper right')
                    
                    # 3. 플롯 제목 및 레이블 설정
                    ax.set_xlabel('Time (ms)')
                    ax.set_ylabel('Soma Membrane Potential (mV)', color='blue')
                    ax.set_title(f'Response of {neuron_label} (First 1 ms: 0 ~ {SINGLE_CYCLE_TIME_MS:.2f} ms)')
                    ax.tick_params(axis='y', labelcolor='blue')
                    # X축 범위를 첫 1ms로 제한
                    ax.set_xlim(0, SINGLE_CYCLE_TIME_MS)
                    # Vm 축 범위 통일
                    ax.set_ylim(single_vm_ylim)
                    
                    # 4. 기준선(실제 resting potential) 및 그리드 추가
                    actual_resting = resting_potentials[i]
                    ax.axhline(actual_resting, color='red', linestyle='--', linewidth=0.8, label=f'Resting Potential ({actual_resting:.1f} mV)')
                    ax.legend(loc='upper left')
                    ax.grid(True)
                
                # 5. 서브플롯 간 간격 조정
                plt.tight_layout()
                
                # 6. 파일로 저장
                if MODEL_TYPE == 'allen' and ALLEN_CELL_ID:
                    output_filename_single = f'allen_{ALLEN_CELL_ID}_neuron_response_single_cycle_v3.png'
                else:
                    output_filename_single = f'simple_neuron_response_single_cycle_v3.png'
                output_path_single = os.path.join(OUTPUT_DIR, output_filename_single)
                plt.savefig(output_path_single, dpi=300, bbox_inches='tight')
                plt.close()  # 메모리 절약을 위해 figure 닫기
                print(f"   ✅ 저장됨: {output_filename_single}")
                print(f"   Single cycle 시간 범위: 0 ~ {SINGLE_CYCLE_TIME_MS:.2f} ms (첫 1ms)")
                print(f"\n✅ Single cycle 플롯이 '{OUTPUT_DIR}' 디렉토리에 저장되었습니다.")
            else:
                print("   ⚠️ 경고: Single cycle 데이터를 찾을 수 없습니다.")

except ImportError:
    print("\n경고: Matplotlib이 설치되지 않았습니다. 결과를 플롯하려면 'pip install matplotlib'을 실행하세요.")
    print("Vm 데이터는 Vm_vecs[i] 벡터에 저장되어 있습니다.")
