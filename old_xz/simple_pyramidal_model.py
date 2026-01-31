# simple_pyramidal_model.py
from neuron import h
from neuron.units import um, ms, mV

# --- 1. 세포체 및 축삭돌기 파라미터 ---

# L5 피라미드 세포의 대략적인 크기와 특성 (일반적인 문헌 값 참조)
SOMA_DIAMETER = 30.0 * um   # 세포체 지름 (um)
SOMA_LENGTH = 30.0 * um     # 세포체 길이 (um)

# 축삭돌기 (z축 방향)
AXON_LENGTH = 1000.0 * um   # 전체 길이 (um) - 요청하신 1000um 기준
AXON_DIAMETER = 1.0 * um    # 축삭돌기 지름 (um)
AXON_SEGMENTS = 50          # 구획 수. Extracellular 계산을 위해 충분히 작게 분할

# --- 2. 기본 세포 클래스 정의 ---
class SimplePyramidal:
    def __init__(self, x, y, z_center):
        """
        단순화된 피라미드 뉴런 모델을 생성합니다.
        :param x, y, z_center: 세포체의 중심 위치 (um)
        """
        self.soma = h.Section(name='soma')
        self.axon = h.Section(name='axon')
        self.all = [self.soma, self.axon]

        # 2.1. 세포 모양 정의
        self.soma.L = SOMA_LENGTH
        self.soma.diam = SOMA_DIAMETER
        
        self.axon.L = AXON_LENGTH
        self.axon.diam = AXON_DIAMETER
        self.axon.nseg = AXON_SEGMENTS # 구획 분할

        # 2.2. 구조 연결: 축삭돌기 끝을 세포체 중앙에 연결
        self.axon.connect(self.soma(0.5))

        # 2.3. 전위 계산을 위한 좌표 설정 (필수)
        self._set_geometry(x, y, z_center)
        
        # 2.4. 전기적 특성 (막 메커니즘) 삽입
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
        self.axon.pt3dadd(x, y, z_end, AXON_DIAMETER) # Axon은 Z축을 따라 전체 길이 설정

        # 3D 위치 확인
        # print(f"Axon Z range: {z_start:.1f} um to {z_end:.1f} um")
        
    def _insert_mechanisms(self):
        """기본적인 막 전류 메커니즘을 삽입합니다 (흥분성 발화를 위해)."""
        
        for sec in self.all:
            # 1. 축 전류 속성 (필수)
            sec.Ra = 100 # 축저항 (Ohm-cm)
            sec.cm = 1.0 # 막 커패시턴스 (uF/cm^2)
            
            # 2. 능동 채널 삽입 (발화를 위해 Na, K 채널 추가)
            sec.insert('hh') # Hodgkin-Huxley (Na, K, Leak)
            
            # 3. 채널 컨덕턴스 설정 (임계값 조정)
            sec.gkbar_hh = 0.036 # K 채널 (S/cm2)
            sec.gnabar_hh = 0.3 # Na 채널 (S/cm2) ; 기존 0.12
            sec.gl_hh = 0.0003 # 누설 컨덕턴스 (S/cm2)
            sec.el_hh = -65.0 # 누설 전위 (mV)
            
            # 4. Extracellular 메커니즘 삽입 (외부 전기장 연동을 위해 반드시 필요)
            sec.insert('extracellular') 
            
# --- 3. 모델 테스트 (선택 사항) ---
if __name__ == '__main__':
    # h.load_file('stdrun.hoc') # NEURON 표준 파일 로드
    
    # 3개의 뉴런 생성 및 위치 확인
    n1 = SimplePyramidal(x=-90, y=42, z_center=561)
    n2 = SimplePyramidal(x=0, y=42, z_center=561)
    n3 = SimplePyramidal(x=90, y=42, z_center=561)
    
    # 기본 발화 테스트 (선택 사항)
    # ic = h.IClamp(n2.soma(0.5))
    # ic.delay = 10
    # ic.dur = 1
    # ic.amp = 0.5 # mA
    
    # h.finitialize(-65 * mV)
    # h.dt = 0.025 * ms
    # h.continuerun(50 * ms)
    
    print("\n모델 파일 준비 완료: SimplePyramidal 클래스")