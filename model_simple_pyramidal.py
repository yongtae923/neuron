# simple_pyramidal_model.py
"""
단순화된 피라미드 뉴런 모델 (xyz 버전)

일직선 흥분성 뉴런 모델로, Soma와 Z축 방향 Axon으로 구성됩니다.
Ex, Ey, Ez 모든 성분의 E-field를 처리할 수 있습니다.
"""

from neuron import h
from neuron.units import um, ms, mV

# --- 1. 세포체 및 축삭돌기 파라미터 ---

# L5 피라미드 세포의 대략적인 크기와 특성
SOMA_DIAMETER = 30.0 * um   # 세포체 지름 (um)
SOMA_LENGTH = 30.0 * um     # 세포체 길이 (um)

# 축삭돌기 (Z축 방향)
AXON_LENGTH = 1000.0 * um   # 전체 길이 (um)
AXON_DIAMETER = 1.0 * um    # 축삭돌기 지름 (um)
AXON_SEGMENTS = 50          # 구획 수. Extracellular 계산을 위해 충분히 작게 분할

# --- 2. 기본 세포 클래스 정의 ---
class SimplePyramidal:
    def __init__(self, x, y, z_center):
        """
        단순화된 피라미드 뉴런 모델을 생성합니다.
        
        Args:
            x, y, z_center: 세포체의 중심 위치 (um)
        """
        self.soma = h.Section(name='soma')
        self.axon = h.Section(name='axon')
        self.all = [self.soma, self.axon]

        # 세포 모양 정의
        self.soma.L = SOMA_LENGTH
        self.soma.diam = SOMA_DIAMETER
        
        self.axon.L = AXON_LENGTH
        self.axon.diam = AXON_DIAMETER
        self.axon.nseg = AXON_SEGMENTS

        # 구조 연결: 축삭돌기를 세포체 중앙에 연결
        self.axon.connect(self.soma(0.5))

        # 전위 계산을 위한 좌표 설정 (필수)
        self._set_geometry(x, y, z_center)
        
        # 전기적 특성 (막 메커니즘) 삽입
        self._insert_mechanisms()

    def _set_geometry(self, x, y, z_center):
        """세포의 구획별 3D 좌표를 설정합니다."""
        
        # Z축 시작점 및 끝점 계산
        z_start = z_center - AXON_LENGTH / 2.0
        z_end = z_center + AXON_LENGTH / 2.0
        
        # NEURON의 3D geometry 설정
        self.soma.pt3dadd(x, y, z_center - SOMA_LENGTH/2.0, SOMA_DIAMETER)
        self.soma.pt3dadd(x, y, z_center + SOMA_LENGTH/2.0, SOMA_DIAMETER)

        self.axon.pt3dadd(x, y, z_start, AXON_DIAMETER)
        self.axon.pt3dadd(x, y, z_end, AXON_DIAMETER)
        
    def _insert_mechanisms(self):
        """기본적인 막 전류 메커니즘을 삽입합니다 (흥분성 발화를 위해)."""
        
        for sec in self.all:
            # 축 전류 속성
            sec.Ra = 100  # 축저항 (Ohm-cm)
            sec.cm = 1.0  # 막 커패시턴스 (uF/cm^2)
            
            # 능동 채널 삽입 (발화를 위해 Na, K 채널)
            sec.insert('hh')  # Hodgkin-Huxley (Na, K, Leak)
            
            # 채널 컨덕턴스 설정
            sec.gkbar_hh = 0.036  # K 채널 (S/cm2)
            sec.gnabar_hh = 0.3   # Na 채널 (S/cm2)
            sec.gl_hh = 0.0003    # 누설 컨덕턴스 (S/cm2)
            sec.el_hh = -65.0     # 누설 전위 (mV)
            
            # Extracellular 메커니즘 삽입 (외부 전기장 연동을 위해 필수)
            sec.insert('extracellular')

# --- 3. 모델 테스트 ---
if __name__ == '__main__':
    # 3개의 뉴런 생성 및 위치 확인
    n1 = SimplePyramidal(x=-90, y=42, z_center=561)
    n2 = SimplePyramidal(x=0, y=42, z_center=561)
    n3 = SimplePyramidal(x=90, y=42, z_center=561)
    
    print("\n모델 파일 준비 완료: SimplePyramidal 클래스")
