"""
Allen 뉴런 morphology 3D 시각화 도구
SWC 파일을 읽어서 뉴런의 3D 구조를 시각화합니다.
"""
import os
import sys
import numpy as np
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

# DISPLAY 환경 변수 확인하여 백엔드 자동 선택
def setup_matplotlib_backend():
    """DISPLAY가 사용 가능하면 interactive 모드, 없으면 Agg 백엔드 사용"""
    is_windows = sys.platform.startswith('win')
    if is_windows:
        try:
            matplotlib.use('TkAgg')
            print("[OK] Windows 환경: Interactive 모드 활성화 (TkAgg)")
            return True
        except Exception:
            try:
                matplotlib.use('Qt5Agg')
                print("[OK] Windows 환경: Interactive 모드 활성화 (Qt5Agg)")
                return True
            except Exception:
                matplotlib.use('Agg')
                print("[WARN] Windows 환경: Headless 모드로 전환")
                return False
    
    display = os.environ.get('DISPLAY')
    if display:
        try:
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt_test
            fig_test = plt_test.figure()
            plt_test.close(fig_test)
            print(f"[OK] Interactive 모드 활성화 (TkAgg, DISPLAY={display})")
            return True
        except Exception as e:
            try:
                matplotlib.use('Qt5Agg')
                import matplotlib.pyplot as plt_test
                fig_test = plt_test.figure()
                plt_test.close(fig_test)
                print(f"[OK] Interactive 모드 활성화 (Qt5Agg, DISPLAY={display})")
                return True
            except Exception:
                print(f"[WARN] DISPLAY={display}가 설정되어 있지만 연결할 수 없습니다.")
                matplotlib.use('Agg')
                return False
    else:
        matplotlib.use('Agg')
        print("[INFO] Headless 모드 (파일 저장만 가능, DISPLAY 없음)")
        return False

HAS_DISPLAY = setup_matplotlib_backend()
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def parse_swc(swc_file):
    """
    SWC 파일을 파싱하여 morphology 데이터를 반환합니다.
    
    Returns:
        dict: {
            'points': list of dict (id, type, x, y, z, radius, parent_id),
            'points_dict': dict mapping id -> point
        }
    """
    points = []
    points_dict = {}
    
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
            
            points.append(point_data)
            points_dict[point_id] = point_data
    
    return {'points': points, 'points_dict': points_dict}


def get_type_info():
    """SWC 타입 정보를 반환합니다."""
    return {
        1: {'name': 'Soma', 'color': 'red', 'size': 50, 'alpha': 1.0, 'linewidth': 1.5, 'line_alpha': 0.5},
        2: {'name': 'Dendrite', 'color': 'blue', 'size': 2, 'alpha': 0.1, 'linewidth': 0.3, 'line_alpha': 0.15},
        3: {'name': 'Axon', 'color': 'green', 'size': 3, 'alpha': 0.2, 'linewidth': 0.8, 'line_alpha': 0.3},
        4: {'name': 'Apical', 'color': 'purple', 'size': 4, 'alpha': 0.2, 'linewidth': 0.6, 'line_alpha': 0.25}
    }


def visualize_morphology_3d(swc_file, output_path=None, show_connections=True, 
                            color_by_type=True, line_width=1.0, point_size_scale=1.0,
                            view_angle=None, title=None):
    """
    SWC 파일에서 뉴런 morphology를 3D로 시각화합니다.
    
    Args:
        swc_file: SWC 파일 경로
        output_path: 출력 파일 경로 (None이면 자동 생성)
        show_connections: 부모-자식 연결선 표시 여부
        color_by_type: 타입별 색상 구분 여부
        line_width: 연결선 두께
        point_size_scale: 점 크기 스케일
        view_angle: (elev, azim) 튜플로 시점 각도 지정
        title: 플롯 제목
    """
    print(f"[LOAD] SWC 파일 로딩 중: {swc_file}")
    morphology = parse_swc(swc_file)
    points = morphology['points']
    points_dict = morphology['points_dict']
    
    print(f"[OK] {len(points)}개의 점 로드 완료")
    
    # 타입별 통계
    type_info = get_type_info()
    type_counts = {}
    for point in points:
        t = point['type']
        type_counts[t] = type_counts.get(t, 0) + 1
    
    print("\n[STAT] 타입별 점 개수:")
    for t, count in sorted(type_counts.items()):
        if t in type_info:
            print(f"  {type_info[t]['name']} (type {t}): {count}개")
    
    # 3D 플롯 생성
    print("\n[PLOT] 3D 플롯 생성 중...")
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 타입별로 그리기 (soma는 마지막에 그려서 앞쪽에 보이도록)
    if color_by_type:
        # soma(type 1)를 제외한 다른 타입들을 먼저 그리기
        for point_type, info in type_info.items():
            if point_type == 1:  # soma는 나중에
                continue
            type_points = [p for p in points if p['type'] == point_type]
            if not type_points:
                continue
            
            x_coords = [p['x'] for p in type_points]
            y_coords = [p['y'] for p in type_points]
            z_coords = [p['z'] for p in type_points]
            radii = [p['radius'] for p in type_points]
            
            # 점 크기는 반지름에 비례
            sizes = [r * point_size_scale * info['size'] for r in radii]
            
            # 타입별 alpha 사용 (기본값 0.7)
            point_alpha = info.get('alpha', 0.7)
            
            ax.scatter(x_coords, y_coords, z_coords, 
                      c=info['color'], label=info['name'], 
                      s=sizes, alpha=point_alpha, edgecolors='black', linewidths=0.3)
        
        # soma를 마지막에 그리기 (다른 요소들 위에 표시)
        if 1 in type_info:
            soma_info = type_info[1]
            soma_points = [p for p in points if p['type'] == 1]
            if soma_points:
                x_coords = [p['x'] for p in soma_points]
                y_coords = [p['y'] for p in soma_points]
                z_coords = [p['z'] for p in soma_points]
                radii = [p['radius'] for p in soma_points]
                
                sizes = [r * point_size_scale * soma_info['size'] for r in radii]
                point_alpha = soma_info.get('alpha', 0.7)
                
                ax.scatter(x_coords, y_coords, z_coords, 
                          c=soma_info['color'], label=soma_info['name'], 
                          s=sizes, alpha=point_alpha, edgecolors='black', linewidths=0.5, zorder=1000)
    else:
        # 모든 점을 같은 색으로
        x_coords = [p['x'] for p in points]
        y_coords = [p['y'] for p in points]
        z_coords = [p['z'] for p in points]
        radii = [p['radius'] for p in points]
        sizes = [r * point_size_scale * 20 for r in radii]
        
        ax.scatter(x_coords, y_coords, z_coords, 
                  c='blue', s=sizes, alpha=0.7, edgecolors='black', linewidths=0.5)
    
    # 부모-자식 연결선 그리기 (soma 관련 연결선은 나중에)
    if show_connections:
        print("[CONN] 연결선 그리기 중...")
        connection_count = 0
        soma_connections = []  # soma 관련 연결선 저장
        
        for point in points:
            parent_id = point['parent_id']
            if parent_id != -1 and parent_id in points_dict:
                parent = points_dict[parent_id]
                
                # 연결선 색상 및 스타일 결정
                if color_by_type:
                    # 자식의 타입에 따라 색상 및 스타일 결정
                    child_type = point['type']
                    if child_type in type_info:
                        line_color = type_info[child_type]['color']
                        # 타입별 linewidth와 alpha 사용
                        type_linewidth = type_info[child_type].get('linewidth', line_width)
                        type_line_alpha = type_info[child_type].get('line_alpha', 0.3)
                    else:
                        line_color = 'gray'
                        type_linewidth = line_width
                        type_line_alpha = 0.3
                else:
                    line_color = 'gray'
                    type_linewidth = line_width
                    type_line_alpha = 0.3
                
                # soma 관련 연결선은 나중에 그리기 위해 저장
                is_soma_connection = (point['type'] == 1 or parent['type'] == 1)
                if is_soma_connection:
                    soma_connections.append({
                        'parent': parent,
                        'point': point,
                        'line_color': line_color,
                        'linewidth': type_linewidth,
                        'alpha': type_line_alpha
                    })
                else:
                    # soma가 아닌 연결선은 먼저 그리기
                    ax.plot([parent['x'], point['x']], 
                            [parent['y'], point['y']], 
                            [parent['z'], point['z']],
                            color=line_color, alpha=type_line_alpha, linewidth=type_linewidth)
                    connection_count += 1
        
        # soma 관련 연결선을 나중에 그리기
        for conn in soma_connections:
            ax.plot([conn['parent']['x'], conn['point']['x']], 
                    [conn['parent']['y'], conn['point']['y']], 
                    [conn['parent']['z'], conn['point']['z']],
                    color=conn['line_color'], alpha=conn['alpha'], 
                    linewidth=conn['linewidth'], zorder=999)
            connection_count += 1
        
        print(f"[OK] {connection_count}개의 연결선 그려짐")
    
    # 축 레이블 및 제목
    ax.set_xlabel('X (μm)', fontsize=12)
    ax.set_ylabel('Y (μm)', fontsize=12)
    ax.set_zlabel('Z (μm)', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    else:
        cell_id = os.path.basename(os.path.dirname(swc_file))
        ax.set_title(f'Neuron Morphology 3D Visualization\nCell ID: {cell_id}', 
                    fontsize=14, fontweight='bold', pad=20)
    
    # 범례 추가
    if color_by_type:
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fontsize=10)
    
    # 축 비율 설정 (같은 비율로)
    x_coords_all = [p['x'] for p in points]
    y_coords_all = [p['y'] for p in points]
    z_coords_all = [p['z'] for p in points]
    
    max_range = np.array([max(x_coords_all) - min(x_coords_all),
                          max(y_coords_all) - min(y_coords_all),
                          max(z_coords_all) - min(z_coords_all)]).max() / 2.0
    
    mid_x = (max(x_coords_all) + min(x_coords_all)) * 0.5
    mid_y = (max(y_coords_all) + min(y_coords_all)) * 0.5
    mid_z = (max(z_coords_all) + min(z_coords_all)) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 시점 각도 설정
    if view_angle:
        ax.view_init(elev=view_angle[0], azim=view_angle[1])
    else:
        # 기본 시점: 약간 위에서 본 모습
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    # 저장
    if output_path:
        print(f"\n[SAVE] 플롯 저장 중: {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] 저장 완료: {output_path}")
    else:
        # 자동 저장
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'visualize_morphology_output')
        os.makedirs(output_dir, exist_ok=True)
        
        cell_id = os.path.basename(os.path.dirname(swc_file))
        output_path = os.path.join(output_dir, f'neuron_{cell_id}_morphology_3d.png')
        print(f"\n[SAVE] 플롯 저장 중: {output_path}")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] 저장 완료: {output_path}")
    
    # 표시
    if HAS_DISPLAY:
        print("\n[SHOW] 플롯 창 표시 중...")
        plt.show(block=True)
    else:
        print("\n[INFO] DISPLAY가 없어 플롯을 표시할 수 없습니다. 파일로 저장되었습니다.")
    
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Allen 뉴런 morphology 3D 시각화',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 기본 사용 (자동으로 reconstruction.swc 찾기)
  python visualize_neuron_morphology.py --cell-id 529898751
  
  # 특정 SWC 파일 지정
  python visualize_neuron_morphology.py --swc-file allen_neuron_529898751/reconstruction.swc
  
  # 연결선 없이 점만 표시
  python visualize_neuron_morphology.py --cell-id 529898751 --no-connections
  
  # 시점 각도 조정 (elevation, azimuth)
  python visualize_neuron_morphology.py --cell-id 529898751 --view-angle 30 60
        """
    )
    
    parser.add_argument('--swc-file', type=str, default=None,
                       help='SWC 파일 경로 (지정하지 않으면 --cell-id 사용)')
    parser.add_argument('--cell-id', type=str, default=None,
                       help='Cell ID (예: 529898751). --swc-file보다 우선순위 낮음')
    parser.add_argument('--output', type=str, default=None,
                       help='출력 파일 경로 (지정하지 않으면 자동 생성)')
    parser.add_argument('--no-connections', action='store_true',
                       help='부모-자식 연결선 표시 안 함')
    parser.add_argument('--no-color-by-type', action='store_true',
                       help='타입별 색상 구분 안 함 (모두 같은 색)')
    parser.add_argument('--line-width', type=float, default=1.0,
                       help='연결선 두께 (기본값: 1.0)')
    parser.add_argument('--point-size-scale', type=float, default=1.0,
                       help='점 크기 스케일 (기본값: 1.0)')
    parser.add_argument('--view-angle', type=float, nargs=2, metavar=('ELEV', 'AZIM'),
                       help='시점 각도 (elevation, azimuth) 예: --view-angle 30 60')
    parser.add_argument('--title', type=str, default=None,
                       help='플롯 제목')
    
    args = parser.parse_args()
    
    # SWC 파일 경로 결정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.swc_file:
        swc_file = args.swc_file
        if not os.path.isabs(swc_file):
            swc_file = os.path.join(script_dir, swc_file)
    elif args.cell_id:
        cell_dir = os.path.join(script_dir, f'allen_neuron_{args.cell_id}')
        # reconstruction.swc 우선, 없으면 다른 .swc 파일 찾기
        swc_candidates = [
            os.path.join(cell_dir, 'reconstruction.swc'),
            os.path.join(cell_dir, f'H16.06.010.01.01.10.02_682298143_m.swc')
        ]
        
        swc_file = None
        for candidate in swc_candidates:
            if os.path.exists(candidate):
                swc_file = candidate
                break
        
        if not swc_file:
            # 폴더 내 모든 .swc 파일 찾기
            if os.path.exists(cell_dir):
                swc_files = [f for f in os.listdir(cell_dir) if f.endswith('.swc')]
                if swc_files:
                    swc_file = os.path.join(cell_dir, swc_files[0])
        
        if not swc_file or not os.path.exists(swc_file):
            print(f"[ERROR] Cell ID {args.cell_id}의 SWC 파일을 찾을 수 없습니다.")
            print(f"   경로 확인: {cell_dir}")
            sys.exit(1)
    else:
        print("[ERROR] --swc-file 또는 --cell-id를 지정해야 합니다.")
        parser.print_help()
        sys.exit(1)
    
    if not os.path.exists(swc_file):
        print(f"[ERROR] SWC 파일을 찾을 수 없습니다: {swc_file}")
        sys.exit(1)
    
    # 시점 각도 파싱
    view_angle = None
    if args.view_angle:
        view_angle = tuple(args.view_angle)
    
    # 시각화 실행
    visualize_morphology_3d(
        swc_file=swc_file,
        output_path=args.output,
        show_connections=not args.no_connections,
        color_by_type=not args.no_color_by_type,
        line_width=args.line_width,
        point_size_scale=args.point_size_scale,
        view_angle=view_angle,
        title=args.title
    )


if __name__ == '__main__':
    main()
