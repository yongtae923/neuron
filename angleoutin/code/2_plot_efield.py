# D:\yongtae\neuron\angleoutin\code\2_plot_efield.py

"""
기능:
- E-field(Ex/Ey/Ez/|E|)를 XY/YZ/ZX 슬라이스로 인터랙티브 시각화합니다.

입출력:
- 입력: data/30V_OUT10_IN20_CI/1_E_field_1cycle.npy, 1_E_field_grid_coords.npy, 0_grid_time_spec.json
- 출력: 화면 표시(파일 저장 없음)

실행 방법:
- python 2_plot_efield.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons
from matplotlib.path import Path
import json

# --- 1. 데이터 로드 (메모리 매핑) ---
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CASE_NAME = os.environ.get("ANGLEOUTIN_CASE", "30V_OUT10_IN20_CI")
DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "data", CASE_NAME)
SPEC_PATH = os.path.join(DATA_DIR, "0_grid_time_spec.json")
E_path = os.path.join(DATA_DIR, "1_E_field_1cycle.npy")
C_path = os.path.join(DATA_DIR, "1_E_field_grid_coords.npy")

with open(SPEC_PATH, "r", encoding="utf-8") as f:
    spec = json.load(f)

E = np.load(E_path, mmap_mode="r")          # shape: (3, N_spatial, T)
coords = np.load(C_path)                    # shape: (N_spatial, 3)

assert E.ndim == 3 and E.shape[0] == 3, "E_field_1cycle.npy shape should be (3, N_spatial, T)"
assert coords.ndim == 2 and coords.shape[1] == 3, "E_field_grid_coords.npy shape should be (N_spatial, 3)"
assert coords.shape[0] == E.shape[1], "coords N_spatial must match E N_spatial"

T = E.shape[2]

# 설정 파일에서 시간/공간 정보 로드 (단위: um, us)
X_MIN = float(spec["space_um"]["x"]["min"])
X_MAX = float(spec["space_um"]["x"]["max"])
Y_MIN = float(spec["space_um"]["y"]["min"])
Y_MAX = float(spec["space_um"]["y"]["max"])
Z_MIN = float(spec["space_um"]["z"]["min"])
Z_MAX = float(spec["space_um"]["z"]["max"])

GRID_SPACING_UM = float(spec["space_um"]["x"]["step"])
TIME_START_US = float(spec["time_us"]["start"])
DT_US = float(spec["time_us"]["step"])
FILES_PER_CYCLE = int(spec["data_files"]["per_cycle"])

COIL_POLYGON_XZ = np.array(spec["coil_mask_um"]["polygon_xz"], dtype=np.float64)
COIL_Y_MIN = float(spec["coil_mask_um"]["y"]["min"])
COIL_Y_MAX = float(spec["coil_mask_um"]["y"]["max"])

# 축 라벨 표시를 위해 좌표를 um로 변환 (원본은 m)
coords_um = coords * 1e6
x = coords_um[:, 0]
y = coords_um[:, 1]
z = coords_um[:, 2]

# 코일 마스크는 설정 파일의 XZ 오각형 + Y 두께를 사용

# --- 2. 정규 격자 인덱스 구성 ---
# 좌표가 Cartesian grid를 이룬다고 가정
xu = np.unique(np.round(x, 6))
yu = np.unique(np.round(y, 6))
zu = np.unique(np.round(z, 6))

xu.sort(); yu.sort(); zu.sort()
nx, ny, nz = len(xu), len(yu), len(zu)
N_expected = nx * ny * nz

if N_expected != coords_um.shape[0]:
    raise ValueError(
        f"Grid does not look like a full Cartesian grid.\n"
        f"Unique counts: nx={nx}, ny={ny}, nz={nz}, nx*ny*nz={N_expected}, "
        f"but N_spatial={coords_um.shape[0]}.\n"
        f"대안: (1) 좌표-값을 산점 데이터로 두고 보간해서 그리드화, (2) ParaView용 point cloud로 시각화."
    )


def _axis_matches_spec(axis_name, values, vmin, vmax, step, tol=1e-3):
    if values.size < 2:
        raise ValueError(f"{axis_name} 축 포인트가 부족합니다: {values.size}")
    actual_min = float(values[0])
    actual_max = float(values[-1])
    actual_step = float(np.median(np.diff(values)))
    if abs(actual_min - vmin) > tol or abs(actual_max - vmax) > tol or abs(actual_step - step) > tol:
        raise ValueError(
            f"{axis_name} 축이 spec과 다릅니다. "
            f"actual(min,max,step)=({actual_min:.3f}, {actual_max:.3f}, {actual_step:.3f}), "
            f"spec=({vmin:.3f}, {vmax:.3f}, {step:.3f})"
        )


_axis_matches_spec("X", xu, X_MIN, X_MAX, GRID_SPACING_UM)
_axis_matches_spec("Y", yu, Y_MIN, Y_MAX, GRID_SPACING_UM)
_axis_matches_spec("Z", zu, Z_MIN, Z_MAX, GRID_SPACING_UM)
if T != FILES_PER_CYCLE:
    raise ValueError(f"시간 스텝 수가 spec과 다릅니다. data T={T}, spec per_cycle={FILES_PER_CYCLE}")

# --- 3. 시간 t에서 볼륨 구성 도우미 ---
def get_volume_at_t(t, mode="mag"):
    """
    mode: 'mag' or 'Ex' or 'Ey' or 'Ez'
    return: volume (nx, ny, nz)
    """
    if mode == "mag":
        # |E| = sqrt(Ex^2 + Ey^2 + Ez^2)
        ex = E[0, :, t]
        ey = E[1, :, t]
        ez = E[2, :, t]
        val = np.sqrt(ex*ex + ey*ey + ez*ez)
    elif mode == "Ex":
        val = E[0, :, t]
    elif mode == "Ey":
        val = E[1, :, t]
    elif mode == "Ez":
        val = E[2, :, t]
    else:
        raise ValueError("mode must be one of: mag, Ex, Ey, Ez")

    # 격자 배치 (Ansys 순서: x,y,z, z가 가장 빠르게 변화)
    vol = val.reshape(nx, ny, nz).astype(np.float32)
    return vol

# --- 4. 컬러 스케일용 vmin/vmax 사전 계산 ---
# 시간 범위: 0~0.1 ms (인덱스 0, 1, 2)
def _coil_mask_3d():
    """Return boolean mask: True = inside coil (exclude from colorbar calc).
    Coil is a pentagon in XZ plane, extruded along Y.
    Pentagon vertices (XZ): (-85,-382), (0,-530), (85,-382), (85,-250), (-85,-250)
    Y thickness: -21.25 ~ 26.75
    """
    pentagon_path = Path(COIL_POLYGON_XZ)
    
    # Y 범위 조건
    y_in = (yu >= COIL_Y_MIN) & (yu <= COIL_Y_MAX)
    
    # XZ 평면에서 오각형 내부 판정
    x_grid, z_grid = np.meshgrid(xu, zu, indexing='ij')
    grid_xz_flat = np.column_stack([x_grid.ravel(), z_grid.ravel()])  # shape: (nx*nz, 2)
    # 경계점 포함을 위해 작은 radius 사용
    xz_in_flat = pentagon_path.contains_points(grid_xz_flat, radius=1e-9)  # shape: (nx*nz,)
    xz_in = xz_in_flat.reshape(nx, nz)  # shape: (nx, nz)
    
    # (nx, ny, nz)로 확장: xz_in[i,k] and y_in[j]
    return xz_in[:, None, :] & y_in[None, :, None]


def estimate_vmin_vmax(mode="mag"):
    """Compute vmin/vmax from data in 0~0.1ms range, excluding coil region."""
    t_indices = [0, 1, 2]  # 0, 0.05, 0.1 ms
    t_indices = [t for t in t_indices if t < T]
    
    coil_mask = _coil_mask_3d()  # True는 코일 내부(제외)
    
    all_vals = []
    for t in t_indices:
        vol = get_volume_at_t(t, mode=mode)
        outside_coil = vol[~coil_mask]  # 코일 영역 제외
        finite = outside_coil[np.isfinite(outside_coil)]
        if finite.size > 0:
            all_vals.append(finite)
    
    if not all_vals:
        return 0.0, 1.0  # 기본값
    
    combined = np.concatenate(all_vals)
    combined = combined[np.isfinite(combined)]
    if combined.size == 0:
        return 0.0, 1.0
    
    abs_max = float(np.max(np.abs(combined)))
    
    if mode == "mag":
        # Magnitude: 0 to max (always positive)
        vmin = 0.0
        vmax = float(np.max(combined))
        if vmax <= 0:
            vmax = 1e-6  # avoid zero range
    else:
        # Ex, Ey, Ez: symmetric range centered at 0 for RdBu_r
        if abs_max <= 0:
            abs_max = 1e-6
        vmin = -abs_max
        vmax = abs_max
    
    return vmin, vmax

# --- 5. matplotlib 위젯 기반 인터랙티브 뷰어 ---
# 컨트롤 영역 포함 Figure 생성
fig = plt.figure(figsize=(16, 6))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.15], hspace=0.3, wspace=0.5)

# 플롯 축 생성
ax_xy = fig.add_subplot(gs[0, 0])
ax_yz = fig.add_subplot(gs[0, 1])
ax_zx = fig.add_subplot(gs[0, 2])
ax_ctrl = fig.add_subplot(gs[0, 3])  # Control panel

# NaN(코일 영역)을 검정으로 표시하는 컬러맵 설정
cmap = plt.colormaps['RdBu_r'].copy()
cmap.set_bad(color='black')

# 초기 표시용 볼륨
init_vol = get_volume_at_t(0, mode="mag")
# RdBu_r 컬러맵(적-백-청)
im_xy = ax_xy.imshow(init_vol[:, :, nz//2].T, origin="lower", aspect="equal", cmap=cmap)
im_yz = ax_yz.imshow(init_vol[nx//2, :, :].T, origin="lower", aspect="equal", cmap=cmap)
im_zx = ax_zx.imshow(init_vol[:, ny//2, :].T, origin="lower", aspect="equal", cmap=cmap)

ax_xy.set_title("XY slice (z fixed)")
ax_yz.set_title("YZ slice (x fixed)")
ax_zx.set_title("ZX slice (y fixed)")

# 축 라벨(물리 단위: μm)
ax_xy.set_xlabel("x (μm)"); ax_xy.set_ylabel("y (μm)")
ax_yz.set_xlabel("y (μm)"); ax_yz.set_ylabel("z (μm)")
ax_zx.set_xlabel("x (μm)"); ax_zx.set_ylabel("z (μm)")

# 축 한계 설정
ax_xy.set_xlim(0, nx-1); ax_xy.set_ylim(0, ny-1)
ax_yz.set_xlim(0, ny-1); ax_yz.set_ylim(0, nz-1)
ax_zx.set_xlim(0, nx-1); ax_zx.set_ylim(0, nz-1)

# 물리 좌표 tick 라벨 표시
def set_physical_ticks(ax, coord_array, axis='x'):
    """Set tick labels to show physical coordinates in μm"""
    n_ticks = 5
    if axis == 'x':
        ticks = np.linspace(0, len(coord_array)-1, n_ticks, dtype=int)
        labels = [f'{coord_array[i]:.0f}' for i in ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
    elif axis == 'y':
        ticks = np.linspace(0, len(coord_array)-1, n_ticks, dtype=int)
        labels = [f'{coord_array[i]:.0f}' for i in ticks]
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)

# tick 라벨은 update_view에서 갱신

# 공통 colorbar (E-field 단위: V/m)
cbar = fig.colorbar(im_xy, ax=[ax_xy, ax_yz, ax_zx], shrink=0.9)
cbar.set_label("E-field (V/m)")

# 컨트롤 패널 축 숨김
ax_ctrl.axis('off')

# 슬라이더 생성
ax_t = fig.add_subplot(gs[1, 0])
ax_x = fig.add_subplot(gs[1, 1])
ax_y = fig.add_subplot(gs[1, 2])
ax_z = fig.add_subplot(gs[1, 3])

# 시간 슬라이더: spec 기반(us)
time_end_us = TIME_START_US + (T - 1) * DT_US
t_sl = Slider(ax_t, 't (us)', TIME_START_US, time_end_us, valinit=TIME_START_US, valstep=DT_US)

# 공간 슬라이더: 물리 좌표(μm)
x_sl = Slider(ax_x, 'x (μm)', xu[0], xu[-1], valinit=0.0, valstep=GRID_SPACING_UM)
y_sl = Slider(ax_y, 'y (μm)', yu[0], yu[-1], valinit=0.0, valstep=GRID_SPACING_UM)
z_sl = Slider(ax_z, 'z (μm)', zu[0], zu[-1], valinit=-450.0, valstep=GRID_SPACING_UM)

# 모드 선택 라디오 버튼
rax = fig.add_axes([0.75, 0.7, 0.15, 0.15])
mode_buttons = RadioButtons(rax, ('mag', 'Ex', 'Ey', 'Ez'), active=0)

# 자동 스케일 체크박스
cax = fig.add_axes([0.75, 0.5, 0.15, 0.1])
auto_scale_check = CheckButtons(cax, ['Auto scale'], [False])

# 모드별 전역 vmin/vmax 캐시
global_scale_cache = {}
current_mode = "mag"
auto_scale_enabled = False

def update_view(val=None):
    global current_mode, auto_scale_enabled
    
    mode = mode_buttons.value_selected
    current_mode = mode
    
    # 슬라이더 값을 인덱스로 변환
    t_us = float(t_sl.val)
    t = int(round((t_us - TIME_START_US) / DT_US))
    t = max(0, min(T-1, t))  # 유효 범위로 제한
    
    # 물리 좌표에 가장 가까운 격자 인덱스 선택
    x_um = float(x_sl.val)
    y_um = float(y_sl.val)
    z_um = float(z_sl.val)
    
    xi = np.argmin(np.abs(xu - x_um))
    yi = np.argmin(np.abs(yu - y_um))
    zi = np.argmin(np.abs(zu - z_um))
    
    auto_scale_enabled = auto_scale_check.get_status()[0]

    vol = get_volume_at_t(t, mode=mode)

    # 슬라이스 추출
    slice_xy = vol[:, :, zi].copy()     # (nx, ny)
    slice_yz = vol[xi, :, :].copy()     # (ny, nz)
    slice_zx = vol[:, yi, :].copy()     # (nx, nz)

    # 코일 영역을 NaN으로 마스킹(검정 표시)
    z_coord = zu[zi]
    x_coord = xu[xi]
    y_coord = yu[yi]
    
    pentagon_path = Path(COIL_POLYGON_XZ)
    
    # XY: 고정 z에서 오각형 내부 x + 코일 두께 y 범위 마스킹
    X_grid, Y_grid = np.meshgrid(xu, yu, indexing='ij')  # shape: (nx, ny), (nx, ny)
    xz_points_xy = np.column_stack([X_grid.ravel(), np.full(nx * ny, z_coord)])
    xz_in_xy = pentagon_path.contains_points(xz_points_xy, radius=1e-9).reshape(nx, ny)
    y_in_xy = (Y_grid >= COIL_Y_MIN) & (Y_grid <= COIL_Y_MAX)
    slice_xy[xz_in_xy & y_in_xy] = np.nan

    # YZ: 고정 x에서 오각형 내부 z + 코일 두께 y 범위 마스킹
    Y_grid, Z_grid = np.meshgrid(yu, zu, indexing='ij')  # shape: (ny, nz), (ny, nz)
    xz_points_yz = np.column_stack([np.full(ny * nz, x_coord), Z_grid.ravel()])
    xz_in_yz = pentagon_path.contains_points(xz_points_yz, radius=1e-9).reshape(ny, nz)
    y_in_yz = (Y_grid >= COIL_Y_MIN) & (Y_grid <= COIL_Y_MAX)
    slice_yz[xz_in_yz & y_in_yz] = np.nan

    # ZX: y가 코일 두께 범위일 때만 마스킹
    if COIL_Y_MIN <= y_coord <= COIL_Y_MAX:
        X_grid, Z_grid = np.meshgrid(xu, zu, indexing='ij')  # shape: (nx, nz), (nx, nz)
        xz_points_zx = np.column_stack([X_grid.ravel(), Z_grid.ravel()])
        xz_in_zx = pentagon_path.contains_points(xz_points_zx, radius=1e-9).reshape(nx, nz)
        slice_zx[xz_in_zx] = np.nan

    # 컬러 스케일 계산(NaN, inf 제외)
    def safe_minmax(arr):
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return np.nan, np.nan
        return float(np.min(finite)), float(np.max(finite))
    
    if auto_scale_enabled:
        mn_xy, mx_xy = safe_minmax(slice_xy)
        mn_yz, mx_yz = safe_minmax(slice_yz)
        mn_zx, mx_zx = safe_minmax(slice_zx)
        vmin = float(np.nanmin([mn_xy, mn_yz, mn_zx]))
        vmax = float(np.nanmax([mx_xy, mx_yz, mx_zx]))
        if mode == "mag":
            vmin = 0.0  # mag는 항상 0 이상
        else:
            # Ex/Ey/Ez는 0 중심 대칭 스케일
            abs_max = max(abs(vmin), abs(vmax)) if np.isfinite(vmin) and np.isfinite(vmax) else 1e-6
            vmin, vmax = -abs_max, abs_max
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = 0.0, 1.0
    else:
        if mode not in global_scale_cache:
            global_scale_cache[mode] = estimate_vmin_vmax(mode=mode)
        vmin, vmax = global_scale_cache[mode]

    # 이미지 갱신(imshow 방향에 맞춰 transpose)
    im_xy.set_data(slice_xy.T)
    im_yz.set_data(slice_yz.T)
    im_zx.set_data(slice_zx.T)

    for im in (im_xy, im_yz, im_zx):
        im.set_clim(vmin, vmax)

    # 축 tick 라벨 갱신(물리 좌표)
    n_ticks = 5
    for ax, coord_array, x_or_y in [(ax_xy, xu, 'x'), (ax_xy, yu, 'y'),
                                     (ax_yz, yu, 'x'), (ax_yz, zu, 'y'),
                                     (ax_zx, xu, 'x'), (ax_zx, zu, 'y')]:
        ticks_idx = np.linspace(0, len(coord_array)-1, n_ticks, dtype=int)
        ticks_pos = ticks_idx
        labels = [f'{coord_array[i]:.0f}' for i in ticks_idx]
        if x_or_y == 'x':
            ax.set_xticks(ticks_pos)
            ax.set_xticklabels(labels)
        else:
            ax.set_yticks(ticks_pos)
            ax.set_yticklabels(labels)

    # 제목 갱신
    mode_label = {'mag': '|E|', 'Ex': 'Ex', 'Ey': 'Ey', 'Ez': 'Ez'}[mode]
    ax_xy.set_title(f"XY: {mode_label}, z={zu[zi]:.0f}μm, t={t_us:.0f}us")
    ax_yz.set_title(f"YZ: {mode_label}, x={xu[xi]:.0f}μm, t={t_us:.0f}us")
    ax_zx.set_title(f"ZX: {mode_label}, y={yu[yi]:.0f}μm, t={t_us:.0f}us")

    cbar.update_normal(im_xy)
    fig.canvas.draw_idle()

# 콜백 연결
t_sl.on_changed(update_view)
x_sl.on_changed(update_view)
y_sl.on_changed(update_view)
z_sl.on_changed(update_view)
mode_buttons.on_clicked(update_view)
auto_scale_check.on_clicked(update_view)

# 초기 렌더링
update_view()

plt.show()