# D:\yongtae\neuron\code\1extract_xyz.py
"""
Ex, Ey, Ez txt 스택을 하나의 5차원 배열로 묶어 저장하고, 저장된 파일이 있으면 그것을 바로 로드해서 플롯을 보여주는 스크립트입니다.

입력:
- D:\yongtae\neuron\efield\30V_OUT10_IN20_CI\30V_OUT10_IN20_CI_Ex
- D:\yongtae\neuron\efield\30V_OUT10_IN20_CI\30V_OUT10_IN20_CI_Ey
- D:\yongtae\neuron\efield\30V_OUT10_IN20_CI\30V_OUT10_IN20_CI_Ez

출력:
- D:\yongtae\neuron\efield\30V_OUT10_IN20_CI\1extract_output.npy
- D:\yongtae\neuron\efield\30V_OUT10_IN20_CI\1extract_output.npz

배열 형식:
- E.shape = (T, X, Y, Z, 3)
- 마지막 축 3개는 순서대로 Ex, Ey, Ez 입니다.
- 즉 E[t, x, y, z, 0] = Ex
     E[t, x, y, z, 1] = Ey
     E[t, x, y, z, 2] = Ez

동작:
- 출력 파일이 이미 있으면 바로 로드해서 플롯만 보여줍니다.
- 출력 파일이 없으면 txt를 읽어 npy/npz를 저장한 뒤 플롯을 보여줍니다.

플롯:
- matplotlib 슬라이더로 시간 t, x, y, z를 바꿀 수 있습니다.
- 표시 모드는 Ex, Ey, Ez, |E| 중에서 라디오 버튼으로 선택할 수 있습니다.
- XY, XZ, YZ 슬라이스를 동시에 보여줍니다.
- 플롯에서만 코일 interior와 1-voxel outside shell을 NaN 처리해 흰색으로 숨깁니다.
- 컬러바 기준 계산에서도 코일 숨김 영역 값은 제외합니다.
- 시간 슬라이더는 실제 시간값 기준으로 0 ~ 50 us 범위만 사용합니다.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons


# =========================
# 사용자 설정
# =========================
BASE_DIR = Path(r"D:\yongtae\neuron\efield\30V_OUT10_IN20_CI")

EX_DIR = BASE_DIR / "30V_OUT10_IN20_CI_Ex"
EY_DIR = BASE_DIR / "30V_OUT10_IN20_CI_Ey"
EZ_DIR = BASE_DIR / "30V_OUT10_IN20_CI_Ez"

OUTPUT_NPY = BASE_DIR / "1extract_output.npy"
OUTPUT_NPZ = BASE_DIR / "1extract_output.npz"

TXT_GLOB = "*.txt"
DT_US = 5.0
DTYPE = np.float32
GRID_STEP_UM = 5.0

COMPONENT_ORDER = ("Ex", "Ey", "Ez")

# 새 코일 마스크 형상 (x-z 오각형 + y 두께)
# (x,z)=(-85,-382),(0,-530),(85,-382),(85,max),(-85,max)
COIL_Y_RAW_UM = (-21.25, 26.75)

HEADER_GRID_RE = re.compile(
    r"Grid Output Min:\s*\[([^\]]+)\]\s*Max:\s*\[([^\]]+)\]\s*Grid Size:\s*\[([^\]]+)\]"
)


def natural_key(path: Path):
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else path.stem


def parse_um_triplet(text: str) -> np.ndarray:
    parts = text.strip().split()
    vals = []
    for p in parts:
        p = p.strip()
        if p.endswith("um"):
            p = p[:-2]
        vals.append(float(p))
    if len(vals) != 3:
        raise ValueError(f"3개 값이 아니어서 해석할 수 없습니다: {text}")
    return np.array(vals, dtype=np.float64)


def parse_header(lines: list[str]):
    grid_line_idx = None
    for i, line in enumerate(lines[:20]):
        if line.startswith("Grid Output Min:"):
            grid_line_idx = i
            break
    if grid_line_idx is None:
        raise ValueError("헤더에서 'Grid Output Min:' 줄을 찾지 못했습니다.")

    m = HEADER_GRID_RE.search(lines[grid_line_idx])
    if not m:
        raise ValueError(f"그리드 헤더를 파싱하지 못했습니다:\n{lines[grid_line_idx]}")

    min_um = parse_um_triplet(m.group(1))
    max_um = parse_um_triplet(m.group(2))
    step_um = parse_um_triplet(m.group(3))
    data_start_idx = grid_line_idx + 2
    return min_um, max_um, step_um, data_start_idx


def expected_axis(min_um: float, max_um: float, step_um: float) -> np.ndarray:
    n = int(round((max_um - min_um) / step_um)) + 1
    return min_um + np.arange(n, dtype=np.float64) * step_um


def outward_round_scalar_um(v: float, step_um: float = GRID_STEP_UM) -> float:
    if v > 0:
        return math.ceil(v / step_um) * step_um
    if v < 0:
        return math.floor(v / step_um) * step_um
    return 0.0


def outward_round_points(points_um: np.ndarray, step_um: float = GRID_STEP_UM) -> np.ndarray:
    out = np.empty_like(points_um, dtype=np.float64)
    for i in range(points_um.shape[0]):
        out[i, 0] = outward_round_scalar_um(float(points_um[i, 0]), step_um)
        out[i, 1] = outward_round_scalar_um(float(points_um[i, 1]), step_um)
    return out


def outward_round_interval_um(interval_um: tuple[float, float], step_um: float = GRID_STEP_UM) -> tuple[float, float]:
    lo, hi = interval_um
    return outward_round_scalar_um(lo, step_um), outward_round_scalar_um(hi, step_um)


def points_in_polygon(x: np.ndarray, z: np.ndarray, poly_xz: np.ndarray) -> np.ndarray:
    """
    Ray casting 기반 point-in-polygon (경계 포함).
    x, z는 동일 shape ndarray.
    """
    inside = np.zeros(x.shape, dtype=bool)
    boundary = np.zeros(x.shape, dtype=bool)

    px = poly_xz[:, 0]
    pz = poly_xz[:, 1]
    n = len(poly_xz)
    eps = 1e-12

    for i in range(n):
        j = (i + 1) % n
        x1, z1 = float(px[i]), float(pz[i])
        x2, z2 = float(px[j]), float(pz[j])

        dx = x2 - x1
        dz = z2 - z1

        cross = (x - x1) * dz - (z - z1) * dx
        on_line = np.abs(cross) <= eps
        within_x = ((x >= min(x1, x2) - eps) & (x <= max(x1, x2) + eps))
        within_z = ((z >= min(z1, z2) - eps) & (z <= max(z1, z2) + eps))
        boundary |= on_line & within_x & within_z

        cond = ((z1 > z) != (z2 > z))
        xinters = np.where(np.abs(z2 - z1) > eps, (x2 - x1) * (z - z1) / (z2 - z1) + x1, x1)
        inside ^= cond & (x < xinters)

    return inside | boundary


def build_surface_shell_mask(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    neighbor_of_mask = (
        padded[2:, 1:-1, 1:-1]
        | padded[:-2, 1:-1, 1:-1]
        | padded[1:-1, 2:, 1:-1]
        | padded[1:-1, :-2, 1:-1]
        | padded[1:-1, 1:-1, 2:]
        | padded[1:-1, 1:-1, :-2]
    )
    shell_mask = neighbor_of_mask & (~mask)
    return shell_mask


def build_coil_polygon_xz(z_max_um: float) -> np.ndarray:
    raw = np.array([
        [-85.0, -382.0],
        [0.0, -530.0],
        [85.0, -382.0],
        [85.0, z_max_um],
        [-85.0, z_max_um],
    ], dtype=np.float64)
    return outward_round_points(raw, GRID_STEP_UM)


def mode_cmap(mode: str):
    if mode == "|E|":
        cmap = plt.cm.viridis.copy()
    else:
        cmap = plt.cm.bwr_r.copy()
    cmap.set_bad(color="black")
    return cmap


def build_coil_mask_symmetric(x_um: np.ndarray, y_um: np.ndarray, z_um: np.ndarray):
    """
    오각형 x-z 단면을 y 방향으로 extrude 한 코일 마스크 생성.
    격자 경계에 좌표가 걸칠 때 바깥쪽도 포함되도록,
    x-z 포인트를 반 스텝 보수 확장한 좌표에서 판정합니다.
    """
    z_max_um = float(np.max(z_um))
    poly_xz = build_coil_polygon_xz(z_max_um)

    y_lo, y_hi = outward_round_interval_um(COIL_Y_RAW_UM, GRID_STEP_UM)

    nx, ny, nz = len(x_um), len(y_um), len(z_um)
    coil_mask = np.zeros((nx, ny, nz), dtype=bool)

    y_mask = (y_um >= y_lo) & (y_um <= y_hi)

    xx, zz = np.meshgrid(x_um, z_um, indexing="ij")

    # 경계가 voxel 사이에 걸치면 바깥쪽 좌표도 포함되도록 half-step 오프셋까지 검사
    h = GRID_STEP_UM * 0.5
    inside_any = (
        points_in_polygon(xx, zz, poly_xz)
        | points_in_polygon(xx + h, zz, poly_xz)
        | points_in_polygon(xx - h, zz, poly_xz)
        | points_in_polygon(xx, zz + h, poly_xz)
        | points_in_polygon(xx, zz - h, poly_xz)
    )

    coil_mask[inside_any[:, None, :] & y_mask[None, :, None]] = True

    # Plot-only mask: shell is disabled.
    shell_mask = np.zeros_like(coil_mask, dtype=bool)
    hide_mask = coil_mask

    meta = {
        "poly_um": poly_xz.astype(np.float32),
        "y_range_um": np.array([y_lo, y_hi], dtype=np.float32),
    }
    return coil_mask, shell_mask, hide_mask, meta


def read_single_component_txt(txt_path: Path, verbose: bool = False):
    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    min_um, max_um, step_um, data_start_idx = parse_header(lines)

    data = np.loadtxt(txt_path, skiprows=data_start_idx)
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(f"{txt_path} 파일이 4열 데이터 형식이 아닙니다.")

    xyz_m = data[:, :3]
    scalar = data[:, 3]
    xyz_um = xyz_m * 1e6

    x_um = expected_axis(min_um[0], max_um[0], step_um[0])
    y_um = expected_axis(min_um[1], max_um[1], step_um[1])
    z_um = expected_axis(min_um[2], max_um[2], step_um[2])

    nx, ny, nz = len(x_um), len(y_um), len(z_um)
    expected_points = nx * ny * nz

    if scalar.size != expected_points:
        raise ValueError(
            f"{txt_path.name}: 데이터 포인트 수가 맞지 않습니다. "
            f"헤더 기준 기대값 {expected_points} ({nx}x{ny}x{nz}), 실제 {scalar.size}"
        )

    x_unique = np.unique(np.round(xyz_um[:, 0], 9))
    y_unique = np.unique(np.round(xyz_um[:, 1], 9))
    z_unique = np.unique(np.round(xyz_um[:, 2], 9))

    if len(x_unique) != nx or len(y_unique) != ny or len(z_unique) != nz:
        raise ValueError(
            f"{txt_path.name}: unique 좌표 개수가 헤더와 다릅니다. "
            f"unique=({len(x_unique)}, {len(y_unique)}, {len(z_unique)}), "
            f"header=({nx}, {ny}, {nz})"
        )

    field_xyz = scalar.reshape(nx, ny, nz)

    if verbose:
        print(f"[OK] {txt_path.name}")
        print(f"     grid shape = ({nx}, {ny}, {nz})")
        print(f"     x range um = [{x_um[0]}, {x_um[-1]}], step {step_um[0]}")
        print(f"     y range um = [{y_um[0]}, {y_um[-1]}], step {step_um[1]}")
        print(f"     z range um = [{z_um[0]}, {z_um[-1]}], step {step_um[2]}")

    return field_xyz.astype(DTYPE, copy=False), x_um, y_um, z_um, step_um


def get_sorted_txt_files(folder: Path) -> list[Path]:
    files = sorted(folder.glob(TXT_GLOB), key=natural_key)
    if not files:
        raise FileNotFoundError(f"txt 파일이 없습니다: {folder}")
    return files


def build_component_stack(component_dir: Path, component_name: str):
    txt_files = get_sorted_txt_files(component_dir)

    frames = []
    x_um = y_um = z_um = spacing_um = None

    for i, txt_path in enumerate(txt_files):
        frame, x_i, y_i, z_i, s_i = read_single_component_txt(
            txt_path,
            verbose=(i == 0),
        )

        if i == 0:
            x_um, y_um, z_um, spacing_um = x_i, y_i, z_i, s_i
        else:
            if (
                frame.shape != frames[0].shape
                or not np.allclose(x_i, x_um)
                or not np.allclose(y_i, y_um)
                or not np.allclose(z_i, z_um)
            ):
                raise ValueError(f"{component_name}: {txt_path.name} 에서 격자 불일치가 발생했습니다.")

        frames.append(frame)

        if (i + 1) % 10 == 0 or (i + 1) == len(txt_files):
            print(f"{component_name}: {i + 1}/{len(txt_files)} loaded")

    stack = np.stack(frames, axis=0)  # (T, X, Y, Z)
    return stack, x_um, y_um, z_um, spacing_um, txt_files


def build_and_save():
    print("[BUILD] 저장 파일이 없어서 txt를 읽어 새로 생성합니다.")

    ex, x_um, y_um, z_um, spacing_um, ex_files = build_component_stack(EX_DIR, "Ex")
    ey, x2, y2, z2, spacing_um2, ey_files = build_component_stack(EY_DIR, "Ey")
    ez, x3, y3, z3, spacing_um3, ez_files = build_component_stack(EZ_DIR, "Ez")

    if not (len(ex_files) == len(ey_files) == len(ez_files)):
        raise ValueError(
            f"파일 개수가 다릅니다. Ex={len(ex_files)}, Ey={len(ey_files)}, Ez={len(ez_files)}"
        )

    if not (
        np.allclose(x_um, x2) and np.allclose(x_um, x3) and
        np.allclose(y_um, y2) and np.allclose(y_um, y3) and
        np.allclose(z_um, z2) and np.allclose(z_um, z3)
    ):
        raise ValueError("Ex/Ey/Ez 사이의 좌표축이 일치하지 않습니다.")

    if not (
        np.allclose(spacing_um, spacing_um2) and np.allclose(spacing_um, spacing_um3)
    ):
        raise ValueError("Ex/Ey/Ez 사이의 grid spacing이 일치하지 않습니다.")

    E = np.stack([ex, ey, ez], axis=-1).astype(DTYPE, copy=False)
    t_us = np.arange(E.shape[0], dtype=np.float64) * DT_US
    Emag = np.linalg.norm(E, axis=-1).astype(DTYPE, copy=False)

    coil_mask, shell_mask, hide_mask, coil_meta = build_coil_mask_symmetric(x_um, y_um, z_um)

    np.save(OUTPUT_NPY, E)
    np.savez_compressed(
        OUTPUT_NPZ,
        E=E,
        Emag=Emag,
        x_um=x_um.astype(np.float32),
        y_um=y_um.astype(np.float32),
        z_um=z_um.astype(np.float32),
        t_us=t_us.astype(np.float32),
        spacing_um=spacing_um.astype(np.float32),
        component_order=np.array(COMPONENT_ORDER),
        coil_poly_um=coil_meta["poly_um"],
        coil_y_range_um=coil_meta["y_range_um"],
    )

    print("\n[SAVED]")
    print(f"  {OUTPUT_NPY}")
    print(f"  {OUTPUT_NPZ}")
    print("\n[SUMMARY]")
    print(f"  E shape        : {E.shape}   (T, X, Y, Z, 3)")
    print(f"  Emag shape     : {Emag.shape} (T, X, Y, Z)")
    print(f"  dtype          : {E.dtype}")
    print(f"  x range (um)   : {x_um[0]} to {x_um[-1]}  n={len(x_um)}")
    print(f"  y range (um)   : {y_um[0]} to {y_um[-1]}  n={len(y_um)}")
    print(f"  z range (um)   : {z_um[0]} to {z_um[-1]}  n={len(z_um)}")
    print(f"  dt (us)        : {DT_US}")
    print(f"  total T        : {E.shape[0]}")
    print(f"  component order: {COMPONENT_ORDER}")
    print(f"  plot-hide voxels: {int(np.sum(hide_mask))}")

    return E, Emag, x_um, y_um, z_um, t_us, hide_mask


def load_existing():
    print("[LOAD] 기존 출력 파일을 로드합니다.")
    npz = np.load(OUTPUT_NPZ, allow_pickle=True)

    E = npz["E"]
    if "Emag" in npz:
        Emag = npz["Emag"]
    else:
        Emag = np.linalg.norm(E, axis=-1).astype(DTYPE, copy=False)

    x_um = npz["x_um"]
    y_um = npz["y_um"]
    z_um = npz["z_um"]
    t_us = npz["t_us"]

    # 저장된 hide_mask가 있어도 현재 코드의 코일 정의를 항상 우선 적용
    # (코일 형상 변경 후 예전 npz와의 불일치 방지)
    coil_mask, shell_mask, hide_mask, _ = build_coil_mask_symmetric(x_um, y_um, z_um)

    if "component_order" in npz:
        component_order = tuple(npz["component_order"].tolist())
    else:
        component_order = COMPONENT_ORDER

    print("\n[SUMMARY]")
    print(f"  E shape        : {E.shape}   (T, X, Y, Z, 3)")
    print(f"  Emag shape     : {Emag.shape} (T, X, Y, Z)")
    print(f"  dtype          : {E.dtype}")
    print(f"  x range (um)   : {x_um[0]} to {x_um[-1]}  n={len(x_um)}")
    print(f"  y range (um)   : {y_um[0]} to {y_um[-1]}  n={len(y_um)}")
    print(f"  z range (um)   : {z_um[0]} to {z_um[-1]}  n={len(z_um)}")
    print(f"  dt (us)        : {t_us[1] - t_us[0] if len(t_us) > 1 else 0}")
    print(f"  total T        : {E.shape[0]}")
    print(f"  component order: {component_order}")
    print(f"  plot-hide voxels: {int(np.sum(hide_mask))}")

    return E, Emag, x_um, y_um, z_um, t_us, hide_mask


def get_volume(E, Emag, mode: str):
    if mode == "Ex":
        return E[..., 0]
    if mode == "Ey":
        return E[..., 1]
    if mode == "Ez":
        return E[..., 2]
    if mode == "|E|":
        return Emag
    raise ValueError(f"알 수 없는 mode: {mode}")


def masked_copy_for_plot(arr3d: np.ndarray, hide_mask: np.ndarray) -> np.ndarray:
    out = arr3d.astype(np.float32, copy=True)
    out[hide_mask] = np.nan
    return out


def set_axes_square(axes):
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        try:
            ax.set_box_aspect(1)
        except Exception:
            pass


def make_time_slider_data(t_us: np.ndarray):
    """
    시간 슬라이더가 전체 시간축을 사용하도록 설정.
    """
    visible = np.arange(len(t_us))

    t_visible = t_us[visible]
    t_min = float(t_visible[0])
    t_max = float(t_visible[-1])

    if len(t_visible) >= 2:
        step = float(np.min(np.diff(t_visible)))
    else:
        step = 1.0

    return visible, t_min, t_max, step


def nearest_visible_time_index(slider_time_us: float, visible_idx: np.ndarray, t_us: np.ndarray) -> int:
    visible_times = t_us[visible_idx]
    j = int(np.argmin(np.abs(visible_times - slider_time_us)))
    return int(visible_idx[j])


def initial_time_index_50us(visible_idx: np.ndarray, t_us: np.ndarray) -> int:
    target_us = 50.0
    return nearest_visible_time_index(target_us, visible_idx, t_us)


def nearest_axis_index(target_value: float, axis_values: np.ndarray) -> int:
    return int(np.argmin(np.abs(axis_values - target_value)))


def plot_interactive(E, Emag, x_um, y_um, z_um, t_us, hide_mask):
    T, NX, NY, NZ, _ = E.shape

    visible_idx, t_slider_min, t_slider_max, t_slider_step = make_time_slider_data(t_us)

    t_idx0 = initial_time_index_50us(visible_idx, t_us)
    x_idx0 = NX // 2
    y_idx0 = nearest_axis_index(35.0, y_um)
    z_idx0 = NZ // 2
    mode0 = "Ez"

    modes = ("Ex", "Ey", "Ez", "|E|")

    cmap0 = mode_cmap(mode0)

    vmins = {}
    vmaxs = {}
    for m in modes:
        V = get_volume(E, Emag, m)
        V_plot = np.where(hide_mask[None, ...], np.nan, V)
        vmins[m] = float(np.nanmin(V_plot))
        vmaxs[m] = float(np.nanmax(V_plot))
        if np.isclose(vmins[m], vmaxs[m]):
            vmaxs[m] = vmins[m] + 1e-12

    frame_cache: dict[tuple[str, int, bool], np.ndarray] = {}

    def get_frame(mode: str, t_idx: int, gaussian_enabled: bool) -> np.ndarray:
        key = (mode, t_idx, gaussian_enabled)
        if key in frame_cache:
            return frame_cache[key]

        vol = get_volume(E, Emag, mode)[t_idx].astype(np.float32, copy=False)
        if gaussian_enabled:
            outside = (~hide_mask).astype(np.float32, copy=False)
            weighted = gaussian_filter(vol * outside, sigma=1.0, mode="nearest")
            weights = gaussian_filter(outside, sigma=1.0, mode="nearest")
            with np.errstate(invalid="ignore", divide="ignore"):
                vol = np.where(weights > 0.0, weighted / weights, np.nan).astype(np.float32, copy=False)

        vol = masked_copy_for_plot(vol, hide_mask)
        frame_cache[key] = vol
        if len(frame_cache) > 12:
            frame_cache.pop(next(iter(frame_cache)))
        return vol

    gaussian0 = False
    V0 = get_frame(mode0, t_idx0, gaussian0)
    xy0 = V0[:, :, z_idx0].T
    xz0 = V0[:, y_idx0, :].T
    yz0 = V0[x_idx0, :, :].T

    fig, axes = plt.subplots(1, 3, figsize=(17, 7))
    plt.subplots_adjust(left=0.08, right=0.88, bottom=0.29, top=0.88, wspace=0.28)

    ax_xy, ax_xz, ax_yz = axes

    img_xy = ax_xy.imshow(
        xy0,
        origin="lower",
        aspect="equal",
        extent=[x_um[0], x_um[-1], y_um[0], y_um[-1]],
        vmin=vmins[mode0],
        vmax=vmaxs[mode0],
        cmap=cmap0,
    )
    ax_xy.set_xlabel("x (um)")
    ax_xy.set_ylabel("y (um)")

    img_xz = ax_xz.imshow(
        xz0,
        origin="lower",
        aspect="equal",
        extent=[x_um[0], x_um[-1], z_um[0], z_um[-1]],
        vmin=vmins[mode0],
        vmax=vmaxs[mode0],
        cmap=cmap0,
    )
    ax_xz.set_xlabel("x (um)")
    ax_xz.set_ylabel("z (um)")

    img_yz = ax_yz.imshow(
        yz0,
        origin="lower",
        aspect="equal",
        extent=[y_um[0], y_um[-1], z_um[0], z_um[-1]],
        vmin=vmins[mode0],
        vmax=vmaxs[mode0],
        cmap=cmap0,
    )
    set_axes_square((ax_xy, ax_xz, ax_yz))
    ax_yz.set_xlabel("y (um)")
    ax_yz.set_ylabel("z (um)")

    cbar = fig.colorbar(img_xy, ax=axes.ravel().tolist(), shrink=0.95)
    cbar.set_label(mode0)

    title = fig.suptitle("", fontsize=12)

    ax_radio = plt.axes([0.90, 0.58, 0.08, 0.18])
    radio = RadioButtons(ax_radio, modes, active=2)

    ax_gauss = plt.axes([0.90, 0.34, 0.08, 0.08])
    check_gauss = CheckButtons(ax_gauss, ["Gaussian Blur"], [False])

    ax_t = plt.axes([0.12, 0.20, 0.70, 0.03])
    ax_x = plt.axes([0.12, 0.15, 0.70, 0.03])
    ax_y = plt.axes([0.12, 0.10, 0.70, 0.03])
    ax_z = plt.axes([0.12, 0.05, 0.70, 0.03])

    s_t = Slider(ax_t, "t (us)", t_slider_min, t_slider_max, valinit=float(t_us[t_idx0]), valstep=t_slider_step)
    s_x = Slider(ax_x, "x idx", 0, NX - 1, valinit=x_idx0, valstep=1)
    s_y = Slider(ax_y, "y idx", 0, NY - 1, valinit=y_idx0, valstep=1)
    s_z = Slider(ax_z, "z idx", 0, NZ - 1, valinit=z_idx0, valstep=1)

    ax_reset = plt.axes([0.90, 0.24, 0.08, 0.06])
    btn_reset = Button(ax_reset, "Reset")

    def update(_=None):
        mode = radio.value_selected

        t_idx = nearest_visible_time_index(float(s_t.val), visible_idx, t_us)
        x_idx = int(s_x.val)
        y_idx = int(s_y.val)
        z_idx = int(s_z.val)
        gaussian_enabled = bool(check_gauss.get_status()[0])

        V = get_frame(mode, t_idx, gaussian_enabled)

        xy = V[:, :, z_idx].T
        xz = V[:, y_idx, :].T
        yz = V[x_idx, :, :].T

        img_xy.set_data(xy)
        img_xz.set_data(xz)
        img_yz.set_data(yz)
        cmap = mode_cmap(mode)
        img_xy.set_cmap(cmap)
        img_xz.set_cmap(cmap)
        img_yz.set_cmap(cmap)

        img_xy.set_clim(vmins[mode], vmaxs[mode])
        img_xz.set_clim(vmins[mode], vmaxs[mode])
        img_yz.set_clim(vmins[mode], vmaxs[mode])
        cbar.update_normal(img_xy)
        cbar.set_label(mode)

        ax_xy.set_title(f"{mode} XY @ z={z_um[z_idx]:.1f} um")
        ax_xz.set_title(f"{mode} XZ @ y={y_um[y_idx]:.1f} um")
        ax_yz.set_title(f"{mode} YZ @ x={x_um[x_idx]:.1f} um")

        title.set_text(
            f"mode={mode}, "
            f"gaussian={'on' if gaussian_enabled else 'off'}, "
            f"t={t_us[t_idx]:.1f} us, "
            f"x={x_idx} ({x_um[x_idx]:.1f} um), "
            f"y={y_idx} ({y_um[y_idx]:.1f} um), "
            f"z={z_idx} ({z_um[z_idx]:.1f} um)"
        )
        fig.canvas.draw_idle()

    def reset(_):
        s_t.reset()
        s_x.reset()
        s_y.reset()
        s_z.reset()
        if check_gauss.get_status()[0]:
            check_gauss.set_active(0)

    radio.on_clicked(update)
    check_gauss.on_clicked(update)
    s_t.on_changed(update)
    s_x.on_changed(update)
    s_y.on_changed(update)
    s_z.on_changed(update)
    btn_reset.on_clicked(reset)

    update()
    plt.show()


def main():
    if OUTPUT_NPY.exists() and OUTPUT_NPZ.exists():
        E, Emag, x_um, y_um, z_um, t_us, hide_mask = load_existing()
    else:
        E, Emag, x_um, y_um, z_um, t_us, hide_mask = build_and_save()

    plot_interactive(E, Emag, x_um, y_um, z_um, t_us, hide_mask)


if __name__ == "__main__":
    main()