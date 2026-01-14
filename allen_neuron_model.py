# allen_neuron_model.py
from neuron import h
from neuron.units import um, ms, mV
import os
import numpy as np
import re

# --- 1. 파일 경로 설정 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 기본 cell ID (하위 호환성을 위해 유지)
_DEFAULT_CELL_ID = '321923685'

def get_allen_cell_id():
    """
    현재 설정된 Allen 모델의 cell ID를 반환합니다.
    
    :return: Cell ID (문자열) 또는 None
    """
    if ALLEN_DATA_DIR:
        folder_name = os.path.basename(ALLEN_DATA_DIR)
        match = re.search(r'(\d+)$', folder_name)
        if match:
            return match.group(1)
    return None

def set_allen_cell_id(cell_id):
    """
    Allen 모델의 cell ID를 설정합니다.
    이 함수를 호출하면 이후 생성되는 AllenNeuronModel이 해당 cell ID의 데이터를 사용합니다.
    
    :param cell_id: Cell ID (문자열 또는 정수, 예: '321923685' 또는 321923685)
    :raises FileNotFoundError: 지정된 cell ID의 데이터 폴더나 필수 파일을 찾을 수 없을 때
    """
    global ALLEN_DATA_DIR, SWC_FILE, NWB_FILE, XML_FILE
    
    cell_id_str = str(cell_id)
    ALLEN_DATA_DIR = os.path.join(SCRIPT_DIR, f'allen_neuron_{cell_id_str}')
    
    if not os.path.exists(ALLEN_DATA_DIR):
        raise FileNotFoundError(f"Allen 데이터 폴더를 찾을 수 없습니다: {ALLEN_DATA_DIR}")
    
    # SWC 파일 자동 찾기
    swc_files = [f for f in os.listdir(ALLEN_DATA_DIR) if f.endswith('.swc')]
    if not swc_files:
        raise FileNotFoundError(f"SWC 파일을 찾을 수 없습니다: {ALLEN_DATA_DIR}")
    # reconstruction.swc가 있으면 우선 사용, 없으면 첫 번째 .swc 파일 사용
    if 'reconstruction.swc' in swc_files:
        SWC_FILE = os.path.join(ALLEN_DATA_DIR, 'reconstruction.swc')
    else:
        SWC_FILE = os.path.join(ALLEN_DATA_DIR, swc_files[0])
    
    # XML 파일 자동 찾기 (여러 가능한 이름 확인)
    xml_candidates = ['ephys_query.xml', 'electrophysiology_query.xml', 'morphology_query.xml']
    XML_FILE = None
    for xml_name in xml_candidates:
        xml_path = os.path.join(ALLEN_DATA_DIR, xml_name)
        if os.path.exists(xml_path):
            XML_FILE = xml_path
            break
    
    # NWB 파일 자동 찾기 (선택사항)
    nwb_files = [f for f in os.listdir(ALLEN_DATA_DIR) if f.endswith('.nwb')]
    if nwb_files:
        # cell_id로 시작하는 nwb 파일 우선 사용
        nwb_with_id = [f for f in nwb_files if f.startswith(cell_id_str)]
        if nwb_with_id:
            NWB_FILE = os.path.join(ALLEN_DATA_DIR, nwb_with_id[0])
        else:
            NWB_FILE = os.path.join(ALLEN_DATA_DIR, nwb_files[0])
    else:
        NWB_FILE = None
    
    print(f"✅ Allen 데이터 설정 완료: Cell ID = {cell_id_str}")
    print(f"   데이터 폴더: {ALLEN_DATA_DIR}")
    print(f"   SWC 파일: {os.path.basename(SWC_FILE)}")
    if XML_FILE:
        print(f"   XML 파일: {os.path.basename(XML_FILE)}")
    if NWB_FILE:
        print(f"   NWB 파일: {os.path.basename(NWB_FILE)}")

# 초기화: 기본 cell ID로 설정
ALLEN_DATA_DIR = os.path.join(SCRIPT_DIR, f'allen_neuron_{_DEFAULT_CELL_ID}')
SWC_FILE = os.path.join(ALLEN_DATA_DIR, 'Nr5a1-Cre_Ai14-172512.06.02.01_491120144_m.swc')
NWB_FILE = os.path.join(ALLEN_DATA_DIR, '321923683_ephys.nwb')
XML_FILE = os.path.join(ALLEN_DATA_DIR, 'ephys_query.xml')

# --- 2. Ephys 파라미터 추출 함수 ---
def extract_ephys_from_nwb(nwb_file):
    """
    NWB 파일에서 전기생리학 파라미터를 추출합니다.
    Returns: dict with ephys parameters
    """
    try:
        from pynwb import NWBHDF5IO
        
        with NWBHDF5IO(nwb_file, 'r') as io:
            nwbfile = io.read()
            
            # Allen Institute NWB 파일 구조에 따라 파라미터 추출
            ephys_params = {}
            
            # 일반적인 ephys features 찾기
            if hasattr(nwbfile, 'units') and nwbfile.units:
                # Units에서 spike 관련 정보 추출 가능
                pass
            
            # Metadata나 processing에서 ephys features 찾기
            if hasattr(nwbfile, 'processing'):
                for proc_name, proc_module in nwbfile.processing.items():
                    if 'ephys' in proc_name.lower():
                        # Ephys processing module에서 파라미터 추출
                        pass
            
            # 기본값 반환 (NWB 구조에 따라 수정 필요)
            return ephys_params
            
    except ImportError:
        print("⚠️ pynwb가 설치되지 않았습니다. 'pip install pynwb'를 실행하세요.")
        return None
    except Exception as e:
        print(f"⚠️ NWB 파일 읽기 오류: {e}")
        return None

def extract_ephys_from_xml(xml_file):
    """
    query.xml 파일에서 전기생리학 파라미터를 추출합니다.
    Returns: dict with ephys parameters
    """
    try:
        import xml.etree.ElementTree as ET
        
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        ephys_params = {}
        
        # ephys-feature 찾기 (Response > specimens > specimen > ephys-features > ephys-feature)
        ephys_feature = root.find('.//ephys-feature')
        if ephys_feature is not None:
            # vrest
            vrest_elem = ephys_feature.find('vrest')
            if vrest_elem is not None and vrest_elem.text:
                try:
                    ephys_params['vrest'] = float(vrest_elem.text)
                except (ValueError, TypeError):
                    pass
            
            # input-resistance-mohm
            input_R_elem = ephys_feature.find('input-resistance-mohm')
            if input_R_elem is not None and input_R_elem.text:
                try:
                    ephys_params['input_resistance'] = float(input_R_elem.text)
                except (ValueError, TypeError):
                    pass
            
            # tau
            tau_elem = ephys_feature.find('tau')
            if tau_elem is not None and tau_elem.text:
                try:
                    ephys_params['tau'] = float(tau_elem.text)
                except (ValueError, TypeError):
                    pass
            
            # sag
            sag_elem = ephys_feature.find('sag')
            if sag_elem is not None and sag_elem.text:
                try:
                    ephys_params['sag'] = float(sag_elem.text)
                except (ValueError, TypeError):
                    pass
            
            # rheobase (threshold-i-long-square)
            threshold_i_elem = ephys_feature.find('threshold-i-long-square')
            if threshold_i_elem is not None and threshold_i_elem.text:
                try:
                    ephys_params['rheobase'] = float(threshold_i_elem.text)
                except (ValueError, TypeError):
                    pass
            
            # threshold-v-long-square
            threshold_v_elem = ephys_feature.find('threshold-v-long-square')
            if threshold_v_elem is not None and threshold_v_elem.text:
                try:
                    ephys_params['threshold_v'] = float(threshold_v_elem.text)
                except (ValueError, TypeError):
                    pass
        
        return ephys_params if len(ephys_params) > 0 else None
        
    except Exception as e:
        print(f"⚠️ XML 파일 읽기 오류: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_ephys_params(nwb_file=None, xml_file=None):
    """
    NWB 또는 XML 파일에서 ephys 파라미터를 로드합니다.
    우선순위: XML > NWB > 기본값 (XML이 더 안정적)
    """
    # 기본값
    default_params = {
        'vrest': -65.0,  # mV
        'input_resistance': 150.0,  # MOhm
        'tau': 30.0,  # ms
        'sag': 0.1,  # sag ratio
        'rheobase': 50.0,  # pA
        'threshold_v': -40.0,  # mV
    }
    
    # XML 파일에서 추출 시도 (우선순위 1: 더 안정적)
    if xml_file and os.path.exists(xml_file):
        xml_params = extract_ephys_from_xml(xml_file)
        if xml_params and len(xml_params) > 0:
            default_params.update(xml_params)
            print(f"✅ XML 파일에서 ephys 파라미터 로드: {xml_file}")
            print(f"   로드된 파라미터: {xml_params}")
            return default_params
    
    # NWB 파일에서 추출 시도 (우선순위 2)
    if nwb_file and os.path.exists(nwb_file):
        nwb_params = extract_ephys_from_nwb(nwb_file)
        if nwb_params and len(nwb_params) > 0:
            default_params.update(nwb_params)
            print(f"✅ NWB 파일에서 ephys 파라미터 로드: {nwb_file}")
            return default_params
    
    # 기본값 사용
    print("⚠️ ephys 파라미터를 기본값으로 사용합니다.")
    return default_params

# --- 3. SWC 파일 파싱 함수 ---
def parse_swc(swc_file):
    """
    SWC 파일을 파싱하여 morphology 데이터를 반환합니다.
    Returns: dict with 'soma', 'dendrites', 'axon' sections
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
    def __init__(self, x=0, y=0, z=0, swc_file=None, nwb_file=None, xml_file=None, cell_id=None, data_dir=None):
        """
        Allen Brain Atlas에서 다운로드한 실제 뉴런 모델을 생성합니다.
        :param x, y, z: 세포의 원점 위치 (um) - SWC 좌표에 더해짐
        :param swc_file: SWC 파일 경로 (None이면 기본 경로 사용)
        :param nwb_file: NWB 파일 경로 (None이면 기본 경로 사용)
        :param xml_file: XML 파일 경로 (query.xml, 선택사항)
        :param cell_id: Cell ID (문자열 또는 정수). 지정하면 해당 cell ID의 데이터를 사용합니다.
        :param data_dir: 데이터 폴더 경로 직접 지정 (cell_id보다 우선순위 높음)
        """
        # data_dir이 지정되면 직접 사용
        if data_dir:
            if not os.path.exists(data_dir):
                raise FileNotFoundError(f"데이터 폴더를 찾을 수 없습니다: {data_dir}")
            
            # data_dir에서 파일 찾기
            swc_files = [f for f in os.listdir(data_dir) if f.endswith('.swc')]
            if not swc_files:
                raise FileNotFoundError(f"SWC 파일을 찾을 수 없습니다: {data_dir}")
            if 'reconstruction.swc' in swc_files:
                swc_file = swc_file or os.path.join(data_dir, 'reconstruction.swc')
            else:
                swc_file = swc_file or os.path.join(data_dir, swc_files[0])
            
            xml_candidates = ['ephys_query.xml', 'electrophysiology_query.xml', 'morphology_query.xml']
            for xml_name in xml_candidates:
                xml_path = os.path.join(data_dir, xml_name)
                if os.path.exists(xml_path):
                    xml_file = xml_file or xml_path
                    break
            
            nwb_files = [f for f in os.listdir(data_dir) if f.endswith('.nwb')]
            if nwb_files:
                nwb_file = nwb_file or os.path.join(data_dir, nwb_files[0])
        
        # cell_id가 지정되면 해당 폴더의 파일을 직접 찾기 (전역 설정 변경 안 함)
        elif cell_id:
            cell_id_str = str(cell_id)
            data_dir = os.path.join(SCRIPT_DIR, f'allen_neuron_{cell_id_str}')
            
            print(f"   📁 실제 사용 데이터 폴더: {data_dir}")
            if not os.path.exists(data_dir):
                raise FileNotFoundError(f"Allen 데이터 폴더를 찾을 수 없습니다: {data_dir}")
            
            # SWC 파일 자동 찾기
            swc_files = [f for f in os.listdir(data_dir) if f.endswith('.swc')]
            if not swc_files:
                raise FileNotFoundError(f"SWC 파일을 찾을 수 없습니다: {data_dir}")
            if 'reconstruction.swc' in swc_files:
                swc_file = swc_file or os.path.join(data_dir, 'reconstruction.swc')
            else:
                swc_file = swc_file or os.path.join(data_dir, swc_files[0])
            print(f"   📄 실제 사용 SWC 파일: {os.path.basename(swc_file)}")
            
            # XML 파일 자동 찾기
            xml_candidates = ['ephys_query.xml', 'electrophysiology_query.xml', 'morphology_query.xml']
            for xml_name in xml_candidates:
                xml_path = os.path.join(data_dir, xml_name)
                if os.path.exists(xml_path):
                    xml_file = xml_file or xml_path
                    break
            if xml_file:
                print(f"   📄 실제 사용 XML 파일: {os.path.basename(xml_file)}")
            
            # NWB 파일 자동 찾기 (선택사항)
            nwb_files = [f for f in os.listdir(data_dir) if f.endswith('.nwb')]
            if nwb_files:
                nwb_with_id = [f for f in nwb_files if f.startswith(cell_id_str)]
                if nwb_with_id:
                    nwb_file = nwb_file or os.path.join(data_dir, nwb_with_id[0])
                else:
                    nwb_file = nwb_file or os.path.join(data_dir, nwb_files[0])
                print(f"   📄 실제 사용 NWB 파일: {os.path.basename(nwb_file)}")
        
        # 파일 경로 설정 (파라미터로 지정된 것이 우선)
        if swc_file is None:
            swc_file = SWC_FILE
        if nwb_file is None:
            nwb_file = NWB_FILE
        if xml_file is None:
            xml_file = XML_FILE  # 기본 XML 파일 경로
        
        # Ephys 파라미터 로드
        self.ephys_params = load_ephys_params(nwb_file=nwb_file, xml_file=xml_file)
        
        # SWC 파일 로드
        self.morphology = parse_swc(swc_file)
        
        # 모든 section을 저장할 리스트
        self.all = []
        self.soma = None
        self.dendrites = []
        self.axon = []
        
        # 1. Morphology 로드
        self._load_morphology(x, y, z)
        
        # 2. 전기적 특성 삽입
        self._insert_mechanisms()
    
    def _load_morphology(self, x_offset, y_offset, z_offset):
        """SWC 파일에서 morphology를 로드하여 NEURON section으로 변환 (부모-자식 관계 처리)"""
        
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
        
        # Dendrites 생성 (부모-자식 관계 처리)
        dend_points = self.morphology['dendrites']
        if dend_points:
            dend_sections = self._create_sections_from_tree(
                dend_points, all_points_dict, 'dendrite', x_offset, y_offset, z_offset, soma_id
            )
            self.dendrites.extend(dend_sections)
            self.all.extend(dend_sections)
        
        # Axon 생성 (부모-자식 관계 처리)
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
        Returns: list of sections
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
            # 루트가 없으면 첫 번째 점을 루트로
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
                    
                    # 경로의 모든 점을 section에 매핑 (연결용)
                    for point in current_path:
                        point_to_section[point['id']] = sec
                    
                    # 부모 section에 연결
                    parent_id = root_point['parent_id']
                    if parent_id == soma_id and self.soma:
                        sec.connect(self.soma(0.5))
                    elif parent_id in point_to_section:
                        # 부모 점이 속한 section의 끝(1.0)에 연결
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
        
        # Ephys 파라미터에서 계산된 값들
        vrest = self.ephys_params.get('vrest', -65.0)  # mV
        input_R = self.ephys_params.get('input_resistance', 150.0)  # MOhm
        tau = self.ephys_params.get('tau', 30.0)  # ms
        
        # 기본 파라미터
        Ra = 100.0  # Ohm-cm (축저항)
        cm = 1.0  # uF/cm^2 (막 커패시턴스)
        
        # Input resistance로부터 막 저항 추정
        # Rm ≈ tau / Cm (단순화, 실제로는 더 복잡)
        # Input resistance가 크면 막 저항이 크고, 누설 컨덕턴스가 작음
        # gl ≈ 1 / (Rm * area) 근사
        
        for sec in self.all:
            # 1. 기본 속성
            sec.Ra = Ra
            sec.cm = cm
            
            # 2. Hodgkin-Huxley 채널 삽입
            sec.insert('hh')
            
            # 3. Ephys 데이터 기반 파라미터 설정
            # vrest를 기준으로 el_hh 설정
            sec.el_hh = vrest  # Resting potential
            
            # 기본 채널 컨덕턴스 (ephys 데이터로 조정 가능)
            sec.gnabar_hh = 0.3  # Na 채널 (S/cm2)
            sec.gkbar_hh = 0.036  # K 채널 (S/cm2)
            
            # Input resistance를 고려한 누설 컨덕턴스
            # 높은 input resistance → 낮은 gl
            # 기본값을 input resistance에 비례하여 조정
            base_gl = 0.0003  # 기본 누설 컨덕턴스 (S/cm2)
            # Input resistance가 150 MOhm일 때 base_gl 사용
            # 더 높은 input resistance면 더 낮은 gl
            gl_factor = 150.0 / max(input_R, 50.0)  # 최소 50 MOhm
            sec.gl_hh = base_gl / gl_factor
            
            # 4. Extracellular 메커니즘 삽입 (외부 전기장 연동 필수)
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
    
    # 모델 생성
    print("Allen Neuron 모델 로드 중...")
    neuron = AllenNeuronModel(x=0, y=42, z=561)
    
    print(f"생성된 section 수: {len(neuron.all)}")
    if neuron.soma:
        print(f"Soma 위치: {neuron.get_soma_location()}")
    print(f"Dendrite section 수: {len(neuron.dendrites)}")
    print(f"Axon section 수: {len(neuron.axon)}")
    
    print("\n모델 파일 준비 완료: AllenNeuronModel 클래스")
