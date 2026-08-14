"""
park_pifm.py
Park Systems SmartScan PiFM TIFF 파일 파서 및 시각화 모듈

사용법:
    from park_pifm import read_park_pifm_tiff, read_info_txt, plot_topo_spectra
"""

import struct
import numpy as np
import tifffile
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def read_park_pifm_tiff(filepath):
    """
    Park Systems PiFM TIFF 파일 파싱

    tag50435 스캔 파라미터 구조 (double, little-endian):
        offset 140: scan_x_um  — X(fast axis) 스캔 크기 (μm)
        offset 148: scan_y_um  — Y(slow axis) 스캔 크기 (μm), 비정사각형 지원
        offset 156: X_stage center (μm)
        offset 164: Y_stage center (μm)

    tag50438 좌표 구조 (12바이트 간격, float32):
        byte+0 : 이전 포인트의 Y_stage (μm) — 1칸 밀려 저장
        byte+4 : 0 (미사용)
        byte+8 : 현재 포인트의 X_stage (μm)
    → X_stage = v2[i], Y_stage = v0[i+1]
    → image_x = X_stage - x_origin, image_y = Y_stage - y_origin

    Returns
    -------
    dict:
        scan_x_nm  : 스캔 크기 X (nm)
        scan_y_nm  : 스캔 크기 Y (nm)
        z_nm       : 토포그래피 이미지 (256×256, nm)
                     부호 반전 적용 (밝은색 = 높은 곳), origin='lower' 용
        coords_nm  : 포인트 좌표 [(x,y), ...] in nm, Y원점=아래
        points     : list of dict
                       amplitude_uV, phase_deg, wavenumber,
                       wavelength_nm, timestamp_us
        n_pts      : 포인트 수
        n_wn       : 스펙트럼 포인트 수
    """
    with tifffile.TiffFile(filepath) as tif:
        page  = tif.pages[0]
        raw34 = bytes(page.tags[50434].value)
        raw35 = bytes(page.tags[50435].value)
        raw38 = bytes(page.tags[50438].value)
        raw39 = bytes(page.tags[50439].value)

    # ── 스캔 크기 및 원점 (tag50435) ──────────────────────────
    # offset 140: scan_x_um (fast axis, X_stage 방향)
    # offset 148: scan_y_um (slow axis, Y_stage 방향) — 비정사각형 스캔 지원
    # offset 156: X_stage center (scan_x 기준)
    # offset 164: Y_stage center (scan_y 기준)
    scan_x_um      = struct.unpack_from('<d', raw35, 140)[0]   # μm
    scan_y_um      = struct.unpack_from('<d', raw35, 148)[0]   # μm
    x_stage_center = struct.unpack_from('<d', raw35, 156)[0]   # μm
    y_stage_center = struct.unpack_from('<d', raw35, 164)[0]   # μm
    x_origin_um    = x_stage_center - scan_x_um / 2
    y_origin_um    = y_stage_center - scan_y_um / 2
    scan_x_nm      = scan_x_um * 1e3
    scan_y_nm      = scan_y_um * 1e3

    # ── Z Height 이미지 (tag50434) ────────────────────────────
    # raw * 1e4 → nm, 부호 반전: 밝은색 = 높은 곳
    # imshow origin='lower' 사용 시 flipud 불필요
    z_nm = np.frombuffer(raw34, dtype=np.float32).reshape(256, 256) * 1e4 * (-1)

    # ── 스펙트럼 구조 자동 감지 (tag50439) ───────────────────
    floats39  = np.frombuffer(raw39, dtype=np.float32)
    small_pos = (floats39 > 0) & (floats39 < 1e-2)
    changes   = np.diff(small_pos.astype(int))
    starts    = np.where(changes == 1)[0] + 1
    if small_pos[0]:
        starts = np.concatenate([[0], starts])
    # starts[1]-starts[0] 대신 최빈값 사용 (블록 내부 spurious 감지 방지)
    diffs          = np.diff(starts)
    unique, counts = np.unique(diffs, return_counts=True)
    step  = int(unique[counts.argmax()])
    n_pts = len(floats39) // step
    n_wn  = step // 5

    # ── 포인트 좌표 (tag50438, byte1200~) ────────────────────
    # v2[i]   = X_stage (fast axis) → image X
    # v0[i+1] = Y_stage (slow axis) → image Y
    v0s = [struct.unpack_from('<f', raw38, 1200 + i*12)[0]     for i in range(n_pts + 1)]
    v2s = [struct.unpack_from('<f', raw38, 1200 + i*12 + 8)[0] for i in range(n_pts)]

    coords_nm = []
    for i in range(n_pts):
        x_nm = (v2s[i]   - x_origin_um) * 1e3
        y_nm = (v0s[i+1] - y_origin_um) * 1e3
        coords_nm.append((x_nm, y_nm))

    # ── 스펙트럼 데이터 ───────────────────────────────────────
    points = []
    for pt in range(n_pts):
        blk = floats39[pt * step:(pt + 1) * step]
        points.append({
            'amplitude_uV':  blk[0 * n_wn:1 * n_wn] * 1e6,
            'phase_deg':     blk[1 * n_wn:2 * n_wn],
            'wavenumber':    blk[2 * n_wn:3 * n_wn],
            'wavelength_nm': blk[3 * n_wn:4 * n_wn],
            'timestamp_us':  blk[4 * n_wn:5 * n_wn],
        })

    return {
        'scan_x_nm': scan_x_nm,
        'scan_y_nm': scan_y_nm,
        'z_nm':      z_nm,
        'coords_nm': coords_nm,
        'points':    points,
        'n_pts':     n_pts,
        'n_wn':      n_wn,
    }


def read_info_txt(filepath):
    """
    SmartScan이 내보내는 info.txt 파일에서 포인트 좌표 읽기

    형식:
        Point#  Position X  Position Y
                μm          μm
        1       0.2354      0.5789
        ...

    Returns
    -------
    list of (x_nm, y_nm) tuples
    """
    lines = None
    for enc in ('utf-8', 'cp949', 'euc-kr', 'latin-1'):
        try:
            with open(filepath, 'r', encoding=enc) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError:
            continue

    coords_nm = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        try:
            int(parts[0])           # 첫 번째 열이 숫자(포인트 번호)인 행만
            x_nm = float(parts[1]) * 1e3   # μm → nm
            y_nm = float(parts[2]) * 1e3
            coords_nm.append((x_nm, y_nm))
        except (ValueError, IndexError):
            continue
    return coords_nm


def save_spectra(data, channel='amplitude_uV', filepath=None):
    """
    스펙트럼 데이터를 Excel(.xlsx) 및 탭 구분 텍스트(.txt)로 저장
    Origin, Excel 모두에서 바로 열 수 있는 형식

    컬럼 구조:
        Wavenumber(cm-1) | Pt1 | Pt2 | ... | PtN

    Parameters
    ----------
    data     : read_park_pifm_tiff() 반환값
    channel  : 'amplitude_uV' or 'phase_deg'
    filepath : 저장 경로 (확장자 제외). None 이면 'spectra_<channel>' 사용
               예) 'result_140' → result_140_amplitude_uV.xlsx / .txt
    """
    import xlsxwriter

    if filepath is None:
        filepath = f'spectra_{channel}'

    ch_unit = {'amplitude_uV': 'Amplitude (μV)', 'phase_deg': 'Phase (deg)'}
    unit    = ch_unit.get(channel, channel)

    wn    = data['points'][0]['wavenumber']
    n_pts = data['n_pts']

    headers   = ['Wavenumber (cm-1)'] + [f'Pt{i+1} {unit}' for i in range(n_pts)]
    col_heads = ['Wavenumber']        + [f'Pt{i+1}'         for i in range(n_pts)]
    col_units = ['cm-1']              + [unit] * n_pts

    # ── Excel 저장 (xlsxwriter) ───────────────────────────────
    xlsx_path = filepath + f'_{channel}.xlsx'
    workbook  = xlsxwriter.Workbook(xlsx_path)
    ws        = workbook.add_worksheet('Spectra')

    for col, h in enumerate(headers):
        ws.write(0, col, h)
        ws.set_column(col, col, max(len(h) + 2, 12))

    for row, wn_val in enumerate(wn):
        ws.write(row + 1, 0, float(wn_val))
        for col, pt in enumerate(data['points']):
            ws.write(row + 1, col + 1, float(pt[channel][row]))

    workbook.close()

    # ── Origin용 탭 구분 텍스트 저장 ─────────────────────────
    # 첫 행: 컬럼명, 두 번째 행: 단위 (Origin LongName/Units 인식)
    txt_path = filepath + f'_{channel}.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\t'.join(col_heads) + '\n')
        f.write('\t'.join(col_units) + '\n')
        for row, wn_val in enumerate(wn):
            vals = [wn_val] + [pt[channel][row] for pt in data['points']]
            f.write('\t'.join(f'{v:.6g}' for v in vals) + '\n')

    print(f'Saved: {xlsx_path}')
    print(f'Saved: {txt_path}')


def plot_topo_spectra(data, coords_nm=None, channel='amplitude_uV',
                      offset_step=120, title=None, save_path=None,
                      xlim=None):
    """
    토포그래피 + 측정 위치 + 스펙트럼 stacking 플롯

    Parameters
    ----------
    data        : read_park_pifm_tiff() 반환값
    coords_nm   : 포인트 좌표 override [(x,y),...] nm, Y원점=아래
                  None 이면 data['coords_nm'] 자동 사용
    channel     : 'amplitude_uV' or 'phase_deg'
    offset_step : 스펙트럼 간 오프셋 간격
    title       : 그래프 제목
    save_path   : 저장 경로 (None 이면 저장 안 함)
    xlim        : 스펙트럼 X축 범위 (cm⁻¹), 예) (900, 1800)
                  None 이면 데이터 전체 범위 자동 사용
    """
    scan_x = data['scan_x_nm']
    scan_y = data['scan_y_nm']
    z_nm   = data['z_nm']
    points = data['points']
    n_pts  = data['n_pts']
    wn     = points[0]['wavenumber']

    if coords_nm is None:
        coords_nm = data.get('coords_nm')
    coords_estimated = coords_nm is None
    if coords_estimated:
        n_side    = int(np.ceil(np.sqrt(n_pts)))
        xs        = np.linspace(scan_x * 0.15, scan_x * 0.85, n_side)
        ys        = np.linspace(scan_y * 0.15, scan_y * 0.85, n_side)
        coords_nm = [(xs[i % n_side], ys[i // n_side]) for i in range(n_pts)]

    ch_label = {
        'amplitude_uV': 'PiFM Amplitude (μV, offset)',
        'phase_deg':    'PiFM Phase (deg, offset)',
    }
    colors = [plt.cm.tab10(i) for i in range(n_pts)]

    fig = plt.figure(figsize=(14, 7))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 1.5], wspace=0.25)
    ax_topo = fig.add_subplot(gs[0])
    ax_spec = fig.add_subplot(gs[1])

    # 토포그래피
    im = ax_topo.imshow(z_nm, cmap='afmhot', origin='lower',
                        extent=[0, scan_x, 0, scan_y],
                        vmin=z_nm.min(), vmax=z_nm.max())
    cbar = plt.colorbar(im, ax=ax_topo, fraction=0.046, pad=0.04)
    cbar.set_label('Height (nm)', fontsize=10)
    ax_topo.set_xlabel('X (nm)', fontsize=11)
    ax_topo.set_ylabel('Y (nm)', fontsize=11)
    ax_topo.set_title(f'Topography  ({scan_x:.0f} × {scan_y:.0f} nm)', fontsize=12)

    for i, (x, y) in enumerate(coords_nm):
        ax_topo.plot(x, y, '+', color=colors[i], markersize=12, markeredgewidth=2)
        ax_topo.text(x + scan_x * 0.02, y + scan_y * 0.02, str(i + 1),
                     color=colors[i], fontsize=11, fontweight='bold')
    if coords_estimated:
        ax_topo.text(scan_x * 0.5, scan_y * 0.03,
                     '※ point positions estimated',
                     ha='center', fontsize=7, color='white', style='italic')

    # 스펙트럼
    for i, pt in enumerate(points):
        offset = (n_pts - 1 - i) * offset_step
        y = pt[channel]
        ax_spec.plot(wn, y + offset, color=colors[i], linewidth=0.9)
        ax_spec.text(wn[-1] + 15, np.median(y) + offset, f'Pt {i+1}',
                     color=colors[i], fontsize=8, va='center', fontweight='bold')

    ax_spec.set_xlabel('Wavenumber (cm⁻¹)', fontsize=11)
    ax_spec.set_ylabel(ch_label.get(channel, channel), fontsize=11)
    ax_spec.set_title(
        f'PiFM {channel.split("_")[0].capitalize()} Spectra', fontsize=12)
    if xlim is not None:
        ax_spec.set_xlim(xlim[0], xlim[1])
    else:
        ax_spec.set_xlim(wn.min(), wn.max() + 80)
    ax_spec.yaxis.set_ticklabels([])

    if title:
        fig.suptitle(title, fontsize=13, y=1.01)
    fig.subplots_adjust(left=0.08, right=0.90, wspace=0.35)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.show()
