# allen_neuron_model.py
"""
Allen Brain Atlas 뉴런 모델 (xyz 버전)

allen_model 폴더의 데이터를 사용하여 실제 뉴런 morphology와 ephys 파라미터를 로드합니다.
Ex, Ey, Ez 모든 성분의 E-field를 처리할 수 있습니다.
"""

from neuron import h
from neuron.units import um, ms, mV
import os
import json
import glob

# --- 1. 파일 경로 설정 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ALLEN_MODEL_DIR = os.path.join(SCRIPT_DIR, 'allen_model')

# --- 2. Ephys 파라미터 추출 함수 ---
def load_ephys_params(data_dir):
    """
    fit_parameters.json 파일에서 ephys 파라미터를 로드합니다.
    
    Args:
        data_dir: Allen 모델 데이터 디렉토리 (예: allen_model/529898751_layer5_ephys)
    
    Returns:
        dict: ephys 파라미터 딕셔너리
    """
    # 기본값
    params = {
        'vrest': -65.0,  # mV (e_pas 또는 v_init 사용)
        'ra': 100.0,  # Ohm-cm (축저항)
        'cm': {},  # section별 막 커패시턴스 (uF/cm^2)
        'celsius': 34.0,  # 온도
        'ena': 53.0,  # Na reversal potential (mV)
        'ek': -107.0,  # K reversal potential (mV)
        'junction_potential': -14.0,  # mV
        'genome': [],  # section별 메커니즘 파라미터들
    }
    
    # fit_parameters.json 파일 찾기
    fit_params_file = os.path.join(data_dir, 'fit_parameters.json')
    
    if os.path.exists(fit_params_file):
        try:
            with open(fit_params_file, 'r') as f:
                fit_params = json.load(f)
            
            # passive 파라미터 추출
            if 'passive' in fit_params and len(fit_params['passive']) > 0:
                passive = fit_params['passive'][0]
                
                # 축저항
                if 'ra' in passive:
                    params['ra'] = float(passive['ra'])
                
                # 막 커패시턴스 (section별)
                if 'cm' in passive:
                    for cm_entry in passive['cm']:
                        section = cm_entry.get('section', 'unknown')
                        cm_value = float(cm_entry.get('cm', 1.0))
                        params['cm'][section] = cm_value
                
                # Passive reversal potential (resting potential과 유사)
                if 'e_pas' in passive:
                    params['vrest'] = float(passive['e_pas'])
            
            # conditions 파라미터 추출
            if 'conditions' in fit_params and len(fit_params['conditions']) > 0:
                conditions = fit_params['conditions'][0]
                
                # 온도
                if 'celsius' in conditions:
                    params['celsius'] = float(conditions['celsius'])
                
                # 초기 전위 (e_pas보다 우선순위 높음)
                if 'v_init' in conditions:
                    params['vrest'] = float(conditions['v_init'])
                
                # Reversal potentials
                if 'erev' in conditions and len(conditions['erev']) > 0:
                    erev = conditions['erev'][0]
                    if 'ena' in erev:
                        params['ena'] = float(erev['ena'])
                    if 'ek' in erev:
                        params['ek'] = float(erev['ek'])
            
            # fitting 파라미터 추출
            if 'fitting' in fit_params and len(fit_params['fitting']) > 0:
                fitting = fit_params['fitting'][0]
                if 'junction_potential' in fitting:
                    params['junction_potential'] = float(fitting['junction_potential'])
            
            # genome 파라미터 추출 (section별 메커니즘 파라미터)
            if 'genome' in fit_params:
                params['genome'] = fit_params['genome']
            
            print(f"Ephys 파라미터 로드 완료: {os.path.basename(fit_params_file)}")
            print(f"  vrest: {params['vrest']:.2f} mV")
            print(f"  ra: {params['ra']:.2f} Ohm-cm")
            print(f"  celsius: {params['celsius']:.1f} C")
            return params
        except Exception as e:
            print(f"경고: fit_parameters.json 읽기 실패, 기본값 사용: {e}")
            import traceback
            traceback.print_exc()
    
    print("기본 ephys 파라미터 사용")
    return params

# --- 3. SWC 파일 파싱 함수 ---
def parse_swc(swc_file):
    """
    SWC 파일을 파싱하여 morphology 데이터를 반환합니다.
    
    Args:
        swc_file: SWC 파일 경로
    
    Returns:
        dict: 'soma', 'dendrites', 'axon', 'all_points' 키를 가진 딕셔너리
    """
    morphology = {
        'soma': [],
        'dendrites': [],
        'axon': [],
        'all_points': []
    }
    
    with open(swc_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 7:
                continue
            
            point_id = int(parts[0])
            point_type = int(parts[1])  # 1=soma, 2=dendrite, 3=axon, 4=apical
            x = float(parts[2])
            y = float(parts[3])
            z = float(parts[4])
            radius = float(parts[5])
            parent_id = int(parts[6])
            
            point_data = {
                'id': point_id,
                'type': point_type,
                'x': x,
                'y': y,
                'z': z,
                'radius': radius,
                'parent_id': parent_id
            }
            
            morphology['all_points'].append(point_data)
            
            if point_type == 1:  # Soma
                morphology['soma'].append(point_data)
            elif point_type == 2 or point_type == 4:  # Dendrite or Apical
                morphology['dendrites'].append(point_data)
            elif point_type == 3:  # Axon
                morphology['axon'].append(point_data)
    
    return morphology

# --- 4. Allen Neuron 모델 클래스 ---
class AllenNeuronModel:
    def __init__(self, x=0, y=0, z=0, cell_id=None, data_dir=None):
        """
        Allen Brain Atlas 뉴런 모델을 생성합니다.
        
        Args:
            x, y, z: 세포의 원점 위치 (um) - SWC 좌표에 더해짐
            cell_id: Cell ID (문자열 또는 정수, 예: '529898751')
            data_dir: 데이터 폴더 경로 직접 지정 (cell_id보다 우선순위 높음)
        
        Raises:
            FileNotFoundError: 데이터 폴더나 필수 파일을 찾을 수 없을 때
        """
        # data_dir이 지정되면 직접 사용
        if data_dir:
            if not os.path.exists(data_dir):
                raise FileNotFoundError(f"데이터 폴더를 찾을 수 없습니다: {data_dir}")
            model_data_dir = data_dir
        # cell_id가 지정되면 해당 폴더 찾기
        elif cell_id:
            cell_id_str = str(cell_id)
            # allen_model 폴더 내에서 cell_id로 시작하는 폴더 찾기
            pattern = os.path.join(ALLEN_MODEL_DIR, f'{cell_id_str}_*_ephys')
            matching_dirs = glob.glob(pattern)
            
            if not matching_dirs:
                raise FileNotFoundError(
                    f"Cell ID {cell_id_str}에 해당하는 데이터 폴더를 찾을 수 없습니다. "
                    f"검색 경로: {pattern}"
                )
            
            # 여러 개가 있으면 첫 번째 사용
            model_data_dir = matching_dirs[0]
            print(f"Allen 데이터 폴더: {os.path.basename(model_data_dir)}")
        else:
            raise ValueError("cell_id 또는 data_dir 중 하나를 지정해야 합니다.")
        
        # SWC 파일 찾기
        swc_files = [f for f in os.listdir(model_data_dir) if f.endswith('.swc')]
        if not swc_files:
            raise FileNotFoundError(f"SWC 파일을 찾을 수 없습니다: {model_data_dir}")
        
        # reconstruction.swc가 있으면 우선 사용
        if 'reconstruction.swc' in swc_files:
            swc_file = os.path.join(model_data_dir, 'reconstruction.swc')
        else:
            swc_file = os.path.join(model_data_dir, swc_files[0])
        
        print(f"SWC 파일: {os.path.basename(swc_file)}")
        
        # Ephys 파라미터 로드
        self.ephys_params = load_ephys_params(model_data_dir)
        
        # SWC 파일 로드
        self.morphology = parse_swc(swc_file)
        
        # 모든 section을 저장할 리스트
        self.all = []
        self.soma = None
        self.dendrites = []
        self.axon = []
        
        # Morphology 로드
        self._load_morphology(x, y, z)
        
        # 전기적 특성 삽입
        self._insert_mechanisms()
    
    def _load_morphology(self, x_offset, y_offset, z_offset):
        """SWC 파일에서 morphology를 로드하여 NEURON section으로 변환"""
        
        # 모든 점을 딕셔너리로 변환 (id -> point)
        all_points_dict = {p['id']: p for p in self.morphology['all_points']}
        
        # Soma 생성
        soma_points = self.morphology['soma']
        if soma_points:
            self.soma = h.Section(name='soma')
            self.all.append(self.soma)
            
            # Soma의 첫 번째 점을 기준으로 생성
            first_point = soma_points[0]
            soma_radius = first_point['radius'] * 2  # radius to diameter
            
            # Soma를 구 형태로 근사
            self.soma.pt3dadd(
                first_point['x'] + x_offset,
                first_point['y'] + y_offset,
                first_point['z'] + z_offset,
                soma_radius
            )
            self.soma.pt3dadd(
                first_point['x'] + x_offset,
                first_point['y'] + y_offset,
                first_point['z'] + z_offset + soma_radius,
                soma_radius
            )
            self.soma.nseg = 1
        
        # Soma ID 찾기 (연결 지점)
        soma_id = soma_points[0]['id'] if soma_points else None
        
        # Dendrites 생성
        dend_points = self.morphology['dendrites']
        if dend_points:
            dend_sections = self._create_sections_from_tree(
                dend_points, all_points_dict, 'dendrite', x_offset, y_offset, z_offset, soma_id
            )
            self.dendrites.extend(dend_sections)
            self.all.extend(dend_sections)
        
        # Axon 생성
        axon_points = self.morphology['axon']
        if axon_points:
            axon_sections = self._create_sections_from_tree(
                axon_points, all_points_dict, 'axon', x_offset, y_offset, z_offset, soma_id
            )
            self.axon.extend(axon_sections)
            self.all.extend(axon_sections)
    
    def _create_sections_from_tree(self, points, all_points_dict, name_prefix, x_offset, y_offset, z_offset, soma_id):
        """
        부모-자식 관계를 따라 여러 section 생성
        각 경로(path)마다 별도의 section을 생성하여 분기 구조를 정확히 반영
        """
        if not points:
            return []
        
        sections = []
        points_dict = {p['id']: p for p in points}
        
        # 각 점의 자식 리스트 구성
        children_dict = {}
        for point in points:
            parent_id = point['parent_id']
            if parent_id not in children_dict:
                children_dict[parent_id] = []
            children_dict[parent_id].append(point)
        
        # 루트 점들 찾기 (soma에 연결되거나 parent가 points에 없는 점)
        root_points = []
        for point in points:
            parent_id = point['parent_id']
            if parent_id == soma_id or parent_id == -1 or parent_id not in points_dict:
                root_points.append(point)
        
        if not root_points:
            root_points = [points[0]]
        
        # point_id -> section 매핑 (연결을 위해)
        point_to_section = {}
        
        def create_section_from_path(path_points, section_name):
            """점들의 경로로부터 section 생성"""
            if len(path_points) < 2:
                return None
            
            sec = h.Section(name=section_name)
            
            # 모든 점을 pt3dadd로 추가
            for point in path_points:
                diameter = point['radius'] * 2
                sec.pt3dadd(
                    point['x'] + x_offset,
                    point['y'] + y_offset,
                    point['z'] + z_offset,
                    diameter
                )
            
            # 구획 수 설정 (길이에 비례, 최소 1개)
            sec.nseg = max(1, int(sec.L / 10.0))  # 10um당 1개 구획
            
            return sec
        
        def build_paths_from_root(root_point, visited=None):
            """루트에서 시작하여 모든 경로를 재귀적으로 생성"""
            if visited is None:
                visited = set()
            
            if root_point['id'] in visited:
                return
            
            visited.add(root_point['id'])
            
            # 현재 점에서 시작하는 경로 찾기
            current_path = [root_point]
            current_id = root_point['id']
            
            # 자식이 하나인 경우 경로를 이어감 (분기 전까지)
            while current_id in children_dict and len(children_dict[current_id]) == 1:
                child = children_dict[current_id][0]
                if child['id'] in visited:
                    break
                visited.add(child['id'])
                current_path.append(child)
                current_id = child['id']
            
            # 경로로 section 생성
            if len(current_path) >= 2:
                section_name = f'{name_prefix}[{len(sections)}]'
                sec = create_section_from_path(current_path, section_name)
                if sec:
                    sections.append(sec)
                    
                    # 경로의 모든 점을 section에 매핑
                    for point in current_path:
                        point_to_section[point['id']] = sec
                    
                    # 부모 section에 연결
                    parent_id = root_point['parent_id']
                    if parent_id == soma_id and self.soma:
                        sec.connect(self.soma(0.5))
                    elif parent_id in point_to_section:
                        sec.connect(point_to_section[parent_id](1.0))
            
            # 분기점 처리: 여러 자식이 있는 경우 각각에 대해 재귀 호출
            if current_id in children_dict and len(children_dict[current_id]) > 1:
                for child in children_dict[current_id]:
                    if child['id'] not in visited:
                        build_paths_from_root(child, visited)
        
        # 각 루트에서 시작하여 모든 경로 생성
        for root_point in root_points:
            build_paths_from_root(root_point)
        
        return sections
    
    def _insert_mechanisms(self):
        """Ephys 데이터 기반 전기적 특성 삽입"""
        
        # Ephys 파라미터에서 값 가져오기
        vrest = self.ephys_params.get('vrest', -65.0)  # mV
        ra = self.ephys_params.get('ra', 100.0)  # Ohm-cm
        cm_dict = self.ephys_params.get('cm', {})  # section별 막 커패시턴스
        celsius = self.ephys_params.get('celsius', 34.0)  # 온도
        genome = self.ephys_params.get('genome', [])  # 메커니즘 파라미터들
        
        # 전역 온도 설정
        h.celsius = celsius
        
        # Section 이름 매핑 (SWC 타입 -> fit_parameters section 이름)
        def get_section_type(sec):
            """Section 이름에서 타입 추출"""
            sec_name = sec.name()
            if 'soma' in sec_name.lower():
                return 'soma'
            elif 'axon' in sec_name.lower():
                return 'axon'
            elif 'dendrite' in sec_name.lower() or 'dend' in sec_name.lower():
                return 'dend'
            elif 'apical' in sec_name.lower() or 'apic' in sec_name.lower():
                return 'apic'
            return 'soma'  # 기본값
        
        for sec in self.all:
            # 기본 속성
            sec.Ra = ra
            
            # Section 타입에 따른 막 커패시턴스 설정
            sec_type = get_section_type(sec)
            if sec_type in cm_dict:
                sec.cm = cm_dict[sec_type]
            else:
                sec.cm = cm_dict.get('soma', 1.0)  # 기본값
            
            # Hodgkin-Huxley 채널 삽입 (기본 채널)
            sec.insert('hh')
            
            # Ephys 데이터 기반 파라미터 설정
            sec.el_hh = vrest  # Resting potential
            
            # 기본 채널 컨덕턴스
            sec.gnabar_hh = 0.3  # Na 채널 (S/cm2)
            sec.gkbar_hh = 0.036  # K 채널 (S/cm2)
            sec.gl_hh = 0.0003  # 누설 컨덕턴스 (S/cm2)
            
            # Genome 파라미터 적용 (section별 메커니즘 파라미터)
            # fit_parameters.json의 genome 배열에 있는 파라미터들을 적용
            # 실제 Allen 모델은 더 복잡한 메커니즘을 사용하지만,
            # 기본 HH 모델로도 충분한 경우가 많음
            # 필요시 genome 파라미터를 사용하여 추가 메커니즘 삽입 가능
            
            # Extracellular 메커니즘 삽입 (외부 전기장 연동 필수)
            sec.insert('extracellular')
    
    def get_soma_location(self):
        """Soma의 위치를 반환합니다"""
        if self.soma and len(self.soma.psection()['morphology']['pts3d']) > 0:
            pts = self.soma.psection()['morphology']['pts3d']
            if len(pts) >= 3:
                return (pts[0], pts[1], pts[2])
        return (0, 0, 0)

# --- 5. 모델 테스트 ---
if __name__ == '__main__':
    h.load_file('stdrun.hoc')
    
    # 모델 생성 (cell_id 지정)
    print("Allen Neuron 모델 로드 중...")
    neuron = AllenNeuronModel(x=0, y=42, z=561, cell_id='529898751')
    
    print(f"생성된 section 수: {len(neuron.all)}")
    if neuron.soma:
        print(f"Soma 위치: {neuron.get_soma_location()}")
    print(f"Dendrite section 수: {len(neuron.dendrites)}")
    print(f"Axon section 수: {len(neuron.axon)}")
    
    print("\n모델 파일 준비 완료: AllenNeuronModel 클래스")
