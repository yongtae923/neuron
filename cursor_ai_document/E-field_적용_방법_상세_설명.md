# E-field 적용 방법 상세 설명

## 목차
1. [개요](#개요)
2. [Simple Method (phi = -(E·r))](#simple-method)
3. [Integrated Method (pt3d 기반 적분)](#integrated-method)
4. [두 방법의 차이점 비교](#두-방법의-차이점-비교)
5. [언제 어떤 방법을 사용해야 할까?](#언제-어떤-방법을-사용해야-할까)

---

## 개요

v2에서는 두 가지 E-field 적용 방법을 제공합니다:
- **Simple Method**: 각 세그먼트의 중심 좌표에서 직접 전위 계산
- **Integrated Method**: morphology의 실제 경로를 따라 전위를 적분

두 방법 모두 같은 목표를 달성하지만, 정확도와 계산 방식이 다릅니다.

---

## Simple Method (phi = -(E·r))

### 기본 원리

**물리적 의미**: 전기장(E-field)이 공간에 존재할 때, 어떤 점에서의 전위(phi)는 다음과 같이 계산됩니다:

```
phi = -(E · r)
```

여기서:
- `E`: 전기장 벡터 (V/m)
- `r`: 기준점(원점)으로부터의 위치 벡터 (m)
- `phi`: 전위 (V)

### 작동 방식

#### 1단계: 각 세그먼트의 중심 좌표 찾기
```python
seg_x, seg_y, seg_z = xyz_at_seg(sec, seg.x)
```
- 각 세그먼트의 중심 위치(seg.x, 0.0~1.0)에서 3D 좌표를 계산
- Section의 양 끝점(pt3d)을 선형 보간하여 정확한 위치 계산

#### 2단계: 해당 위치의 E-field 값 찾기
```python
nearest_idx = find_nearest_spatial_index(seg_x, seg_y, seg_z, E_grid_coords_UM)
Ex, Ez = E_field_values[0, nearest_idx, t], E_field_values[1, nearest_idx, t]
```
- 세그먼트 중심 좌표에 가장 가까운 E-field 그리드 포인트를 찾음
- 시간 보간을 통해 현재 시간의 E-field 값 계산

#### 3단계: 전위 계산
```python
phi_mV = -(E_x * seg_x + E_y * seg_y + E_z * seg_z) * 1e-3  # mV
seg.e_extracellular = phi_mV
```
- **핵심 공식**: `phi = -(E·r)`
- 내적(dot product)을 계산하여 전위를 구함
- 단위 변환: V → mV (× 1e-3)

### 장점
- ✅ **빠른 계산**: 각 세그먼트마다 간단한 수식만 계산
- ✅ **이해하기 쉬움**: 직관적인 물리 공식 사용
- ✅ **메모리 효율적**: 추가 캐시 불필요

### 단점
- ❌ **근사적**: 각 세그먼트를 점으로 취급 (실제로는 선분)
- ❌ **연속성 부족**: 인접한 세그먼트 간 전위가 약간 불연속적일 수 있음
- ❌ **부모-자식 관계 무시**: 각 section이 독립적으로 계산됨

### 예시

```
세그먼트 중심: (x=100, y=50, z=200) um
E-field: Ex=0.1 V/m, Ez=0.2 V/m

전위 계산:
phi = -(0.1 × 100 + 0 × 50 + 0.2 × 200) × 1e-3
   = -(10 + 0 + 40) × 1e-3
   = -50 × 1e-3
   = -0.05 mV
```

---

## Integrated Method (pt3d 기반 적분)

### 기본 원리

**물리적 의미**: 전기장을 따라 경로를 적분하면 전위 차이를 계산할 수 있습니다:

```
Δphi = -∫ E · dl
```

여기서:
- `E`: 전기장 벡터
- `dl`: 경로의 미소 길이 벡터
- `Δphi`: 전위 차이

이 방법은 뉴런의 실제 morphology(형태)를 따라 전위를 누적 적분합니다.

### 작동 방식

#### 1단계: Morphology 캐시 생성 (한 번만, 시뮬레이션 시작 시)

```python
build_morph_cache(neuron, grid_coords_um)
```

**각 section에 대해**:
- **pt3d 점들 추출**: Section의 모든 3D 좌표점 (x, y, z)
- **arc 길이 계산**: 각 점까지의 누적 거리 (um)
- **미소 길이 벡터 계산**: 인접한 pt3d 점들 사이의 벡터 (dx, dy, dz)
- **공간 인덱스 매핑**: 각 구간의 중점에서 가장 가까운 E-field 그리드 포인트 찾기
- **부모-자식 관계 저장**: Section 트리 구조 정보

**예시**:
```
Section: dendrite[0]
pt3d 점들:
  점0: (100, 50, 200) um
  점1: (105, 52, 205) um
  점2: (110, 54, 210) um

미소 길이 벡터:
  dl[0] = (5, 2, 5) um  (점0→점1)
  dl[1] = (5, 2, 5) um  (점1→점2)

arc 길이:
  arc[0] = 0 um
  arc[1] = 7.35 um  (점0→점1 거리)
  arc[2] = 14.7 um  (점0→점2 누적 거리)
```

#### 2단계: Section 트리를 따라 전위 적분

```python
compute_phi_sections(neuron, morph_cache, topo, current_time_ms)
```

**재귀적 계산 (부모 → 자식 순서)**:

1. **루트 section (soma)**: 
   - 부모가 없으므로 `phi0 = 0.0` (기준점)

2. **각 section의 전위 계산**:
   ```python
   # 부모 section의 전위를 가져옴
   phi0 = interp_phi(부모_arc, 부모_phi, 연결점)
   
   # 이 section을 따라 전위를 적분
   phis = [phi0]  # 시작 전위
   for 각 pt3d 구간:
       E = get_E_at(구간_중점의_공간인덱스, 현재시간)
       dphi = -(E · dl) × 1e-3  # 전위 차이
       phis.append(phis[-1] + dphi)  # 누적
   ```

3. **자식 section들**: 부모 section의 전위를 이어받아 계속 적분

**핵심 공식**:
```python
dphi = -(Ex * dx + Ey * dy + Ez * dz) * 1e-3  # mV
```

각 pt3d 구간에서:
- E-field와 구간 방향 벡터의 내적을 계산
- 음수를 취하여 전위 차이 계산
- 누적하여 전체 경로의 전위 계산

#### 3단계: 각 세그먼트에 전위 적용

```python
apply_phi_to_segments(neuron, phi_sec)
```

- 각 section의 arc 길이와 전위 배열을 보간
- 각 세그먼트의 위치(seg.x)에 해당하는 전위를 찾아서 적용

### 장점
- ✅ **정확함**: 실제 morphology 경로를 따라 정확히 적분
- ✅ **연속성 보장**: 부모-자식 관계를 고려하여 전위가 연속적
- ✅ **물리적으로 정확**: 전기장의 경로 적분을 정확히 구현

### 단점
- ❌ **느림**: 각 시간 스텝마다 모든 section을 재귀적으로 계산
- ❌ **메모리 사용**: morphology 캐시 저장 필요
- ❌ **복잡함**: 구현이 더 복잡하고 이해하기 어려움

### 예시

```
Section 트리:
  soma (부모 없음)
    └─ dendrite[0] (soma의 0.5 지점에 연결)
         └─ dendrite[1] (dendrite[0]의 1.0 지점에 연결)

계산 순서:
1. soma: phi0 = 0.0
   - soma를 따라 적분 → phi_soma = [0.0, 0.01, 0.02] mV

2. dendrite[0]: 
   - soma의 0.5 지점 전위 가져오기 → phi0 = 0.01 mV
   - dendrite[0]을 따라 적분 → phi_d0 = [0.01, 0.015, 0.02] mV

3. dendrite[1]:
   - dendrite[0]의 1.0 지점 전위 가져오기 → phi0 = 0.02 mV
   - dendrite[1]을 따라 적분 → phi_d1 = [0.02, 0.025, 0.03] mV
```

---

## 두 방법의 차이점 비교

### 계산 방식

| 항목 | Simple Method | Integrated Method |
|------|---------------|-------------------|
| **전위 계산** | `phi = -(E·r)` (절대 좌표 기준) | `Δphi = -∫E·dl` (경로 적분) |
| **기준점** | 원점 (0, 0, 0) | 부모 section의 연결점 |
| **연속성** | 각 section 독립적 | 부모-자식 간 연속적 |
| **계산 단위** | 세그먼트 중심 (점) | pt3d 구간 (선분) |

### 시각적 비교

**Simple Method**:
```
각 세그먼트:
  seg1: phi1 = -(E · r1)  ← 원점 기준
  seg2: phi2 = -(E · r2)  ← 원점 기준
  seg3: phi3 = -(E · r3)  ← 원점 기준
  
문제: seg1과 seg2가 연결되어 있어도 전위가 불연속적일 수 있음
```

**Integrated Method**:
```
Section 트리:
  soma: phi = [0.0, 0.01, 0.02]
    └─ dendrite: phi = [0.02, 0.025, 0.03]  ← soma의 끝점(0.02)에서 시작
  
장점: 부모-자식 간 전위가 완벽히 연속적
```

### 정확도

**Simple Method**:
- 각 세그먼트를 점으로 근사
- E-field가 구간 내에서 일정하다고 가정
- **오차**: 세그먼트 길이와 E-field 변화에 비례

**Integrated Method**:
- 실제 경로를 따라 적분
- 각 pt3d 구간마다 E-field를 샘플링
- **오차**: pt3d 점의 밀도에만 의존 (더 정확)

### 성능

**Simple Method**:
- 시간 복잡도: O(N_segments)
- 각 스텝마다: 세그먼트 수만큼 계산
- **빠름**: 약 10-100배 빠름

**Integrated Method**:
- 시간 복잡도: O(N_sections × N_pt3d)
- 각 스텝마다: 모든 section을 재귀적으로 계산
- **느림**: 하지만 더 정확

---

## 언제 어떤 방법을 사용해야 할까?

### Simple Method를 사용하는 경우
- ✅ 빠른 프로토타이핑이나 테스트
- ✅ E-field가 거의 일정한 경우 (공간적으로 변화가 작음)
- ✅ 계산 시간이 중요한 경우
- ✅ 단순한 morphology (section 수가 적음)

### Integrated Method를 사용하는 경우
- ✅ 정확한 결과가 필요한 경우 (논문, 발표)
- ✅ 복잡한 morphology (많은 분기, 긴 dendrite)
- ✅ E-field가 공간적으로 크게 변화하는 경우
- ✅ 부모-자식 간 전위 연속성이 중요한 경우

### 권장 사항

**일반적으로 Integrated Method를 권장합니다**:
- 더 정확하고 물리적으로 올바름
- 복잡한 Allen 모델에서 특히 중요
- 계산 시간이 오래 걸리더라도 정확도가 우선

**Simple Method는**:
- 빠른 테스트나 디버깅에 유용
- E-field가 매우 작거나 거의 일정할 때 충분할 수 있음

---

## 코드에서의 구현 위치

### Simple Method
```python
# simulate_tES_v2.py, set_extracellular_field() 함수 내
if E_FIELD_METHOD != 'integrated':
    # 355-405번 줄
    for seg in sec:
        seg_x, seg_y, seg_z = xyz_at_seg(sec, seg.x)
        phi_mV = -(E_x * seg_x + E_y * seg_y + E_z * seg_z) * 1e-3
        seg.e_extracellular = phi_mV
```

### Integrated Method
```python
# simulate_tES_v2.py
if E_FIELD_METHOD == 'integrated':
    # 348-353번 줄
    phi_sec = compute_phi_sections(neuron, morph_caches[i], topos[i], current_time_ms)
    apply_phi_to_segments(neuron, phi_sec)

# 보조 함수들:
# - build_morph_cache(): morphology 캐시 생성 (191-260번 줄)
# - compute_phi_sections(): 전위 적분 계산 (262-304번 줄)
# - apply_phi_to_segments(): 세그먼트에 적용 (306-317번 줄)
```

---

## 요약

| 특징 | Simple Method | Integrated Method |
|------|---------------|-------------------|
| **공식** | `phi = -(E·r)` | `Δphi = -∫E·dl` |
| **계산 방식** | 절대 좌표 기준 | 경로 적분 |
| **정확도** | 근사적 | 정확함 |
| **속도** | 빠름 | 느림 |
| **연속성** | 보장 안 됨 | 보장됨 |
| **사용 시기** | 빠른 테스트 | 정확한 결과 필요 시 |

**결론**: Integrated Method가 더 정확하고 물리적으로 올바르지만, Simple Method는 빠르고 간단합니다. 목적에 따라 선택하세요!
