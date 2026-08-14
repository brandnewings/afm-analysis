"""
park_pifm.py
Park Systems SmartScan PiFM TIFF 파일 파서 및 시각화 모듈

사용법:
    from park_pifm import read_park_pifm_tiff, read_info_txt, plot_topo_spectra
"""

import struct
import os
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
    # wavenumber 구간(700~2500 cm-1)을 직접 탐색하여 n_wn/step/n_pts 결정
    # small_pos 패턴 기반 방식은 파일에 따라 오탐 가능 → 대체
    floats39 = np.frombuffer(raw39, dtype=np.float32)
    wn_mask  = (floats39 > 700) & (floats39 < 2500)
    _changes = np.diff(wn_mask.astype(int))
    _wn_s    = np.where(_changes == 1)[0] + 1
    _wn_e    = np.where(_changes == -1)[0] + 1
    if wn_mask[0]:  _wn_s = np.concatenate([[0], _wn_s])
    if wn_mask[-1]: _wn_e = np.concatenate([_wn_e, [len(floats39)]])
    _lengths = _wn_e - _wn_s
    n_wn  = int(np.bincount(_lengths).argmax())
    step  = n_wn * 5
    n_pts = len(floats39) // step

    # ── 포인트 좌표 (tag50438, byte1200~) ────────────────────
    # v2[i]   = X_stage (fast axis) → image X
    # v0[i+1] = Y_stage (slow axis, 1칸 밀려 저장) → image Y
    # 파일에 따라 마지막 v0[n_pts] 슬롯이 없을 수 있으므로 범위 체크 후 읽음
    def _safe_f(buf, offset):
        return struct.unpack_from('<f', buf, offset)[0] if offset + 4 <= len(buf) else None

    v0s = [_safe_f(raw38, 1200 + i*12)     for i in range(n_pts + 1)]
    v2s = [_safe_f(raw38, 1200 + i*12 + 8) for i in range(n_pts)]

    coords_nm = []
    for i in range(n_pts):
        y_raw = v0s[i+1] if v0s[i+1] is not None else (v0s[i] or y_origin_um)
        x_nm  = ((v2s[i] or 0) - x_origin_um) * 1e3
        y_nm  = (y_raw          - y_origin_um) * 1e3
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
                      stack_gap=1.1, title=None, save_path=None,
                      xlim=(850, 2400),
                      peaks=None,
                      peak_color='#cc44aa'):
    """
    토포그래피 + 측정 위치 + 스펙트럼 stacking 플롯

    Parameters
    ----------
    data        : read_park_pifm_tiff() 반환값
    coords_nm   : 포인트 좌표 override [(x,y),...] nm, Y원점=아래
                  None 이면 data['coords_nm'] 자동 사용
    channel     : 'amplitude_uV' or 'phase_deg'
    stack_gap   : 정규화된 스펙트럼 간 수직 간격 (1.0 = 피크 높이 단위)
    title       : 그래프 제목
    save_path   : 저장 경로 (None 이면 저장 안 함)
    xlim        : 스펙트럼 X축 범위 (cm⁻¹), 예) (850, 2000)
    peaks       : 수직선으로 표시할 피크 위치 리스트, 예) [1100, 1229, 1728]
                  None 이면 표시 안 함
    peak_color  : 피크 수직선 및 라벨 색상
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

    colors = [plt.cm.tab10(i % 10) for i in range(n_pts)]

    # ── 스펙트럼 정규화 (xlim 범위 기준) ─────────────────────
    mask = (wn >= xlim[0]) & (wn <= xlim[1]) if xlim else np.ones(len(wn), dtype=bool)
    wn_crop = wn[mask]

    amps_norm = []
    for pt in points:
        amp = pt[channel][mask]
        base = amp - np.percentile(amp, 5)
        peak = base.max()
        amps_norm.append(base / peak if peak > 0 else base)

    # ── 레이아웃 ─────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 1.4], wspace=0.28)
    ax_topo = fig.add_subplot(gs[0])
    ax_spec = fig.add_subplot(gs[1])

    # ── 토포그래피 ────────────────────────────────────────────
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

    # ── 스펙트럼 (Pt1 아래, Pt_n 위) ─────────────────────────
    for i in range(n_pts):
        offset = i * stack_gap          # Pt1=0, Pt2=gap, ..., Pt_n=(n-1)*gap
        ax_spec.plot(wn_crop, amps_norm[i] + offset,
                     color=colors[i], linewidth=0.85, label=f'Pt{i + 1}')

    # ── 피크 수직선 + 라벨 (인접 피크는 높이를 교대로 어긋나게 배치) ──
    if peaks:
        y_base  = n_pts * stack_gap + 0.08
        y_step  = 0.48           # 레벨 간 높이 간격
        # 라벨 1개 너비 ≈ 130 cm⁻¹ (fontsize 7.5 기준 실측)
        # 두 라벨 center 간격이 이보다 좁으면 인접으로 판단
        MIN_GAP = 130

        # 각 피크에 레벨 할당:
        # 이전 피크와 인접하면 이전 레벨 +1, 멀면 레벨 0으로 리셋
        MAX_LEVEL = 3
        levels = []
        for k, pk in enumerate(sorted(peaks)):
            if k == 0:
                levels.append(0)
            else:
                gap = pk - sorted(peaks)[k - 1]
                if gap < MIN_GAP:
                    levels.append((levels[-1] + 1) % MAX_LEVEL)
                else:
                    levels.append(0)

        for k, pk in enumerate(sorted(peaks)):
            y_label = y_base + levels[k] * y_step
            ax_spec.axvline(pk, color=peak_color, linewidth=0.9,
                            linestyle='--', alpha=0.75, zorder=0)
            ax_spec.annotate(f'{pk} cm⁻¹',
                             xy=(pk, n_pts * stack_gap),
                             xytext=(pk, y_label),
                             ha='center', va='bottom', fontsize=7.5,
                             color=peak_color, fontweight='bold',
                             arrowprops=dict(arrowstyle='-', color=peak_color,
                                             lw=0.6, alpha=0.5))

    # ── 축 ────────────────────────────────────────────────────
    y_top = n_pts * stack_gap + (2.0 if peaks else 0.2)
    ax_spec.set_xlim(xlim[0], xlim[1])
    ax_spec.set_ylim(-0.3, y_top)
    ax_spec.set_xlabel('Wavenumber (cm⁻¹)', fontsize=11)
    ch_label = {'amplitude_uV': 'PiFM Amplitude (norm., offset)',
                'phase_deg':    'PiFM Phase (norm., offset)'}
    ax_spec.set_ylabel(ch_label.get(channel, channel), fontsize=11)
    ax_spec.set_title(f'PiFM {channel.split("_")[0].capitalize()} Spectra', fontsize=12)
    ax_spec.yaxis.set_ticklabels([])
    ax_spec.spines['top'].set_visible(False)
    ax_spec.spines['right'].set_visible(False)

    # ── 범례: 그래프 위(Pt_n)→아래(Pt1) 순서와 일치하도록 역순 ──
    handles, labels = ax_spec.get_legend_handles_labels()
    ax_spec.legend(handles[::-1], labels[::-1],
                   loc='upper right', fontsize=8.5,
                   framealpha=0.9, edgecolor='#bbbbbb',
                   handlelength=1.2, handletextpad=0.5,
                   borderpad=0.5, labelspacing=0.3)

    if title:
        fig.suptitle(title, fontsize=13, y=1.04)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.show()


def detect_peaks(data, point_indices=None, channel='amplitude_uV',
             xlim=(760, 2400), height=0.08, prominence=0.15,
             distance=20, width=5):
    """
    선택한 포인트들의 평균 스펙트럼에서 피크를 자동 검출하고
    플롯으로 시각화하여 확인할 수 있게 반환한다.

    Parameters
    ----------
    data          : read_park_pifm_tiff() 반환값
    point_indices : 평균에 사용할 포인트 번호 리스트 (1-based).
                    예) [1, 3, 5]  →  Pt1, Pt3, Pt5 평균
                    None (기본값)  →  전체 포인트 평균
    channel       : 'amplitude_uV' or 'phase_deg'
    xlim          : 검출 범위 (cm⁻¹)
    height        : 정규화 스펙트럼 기준 최소 피크 높이 (0~1)
    prominence    : 주변 대비 최소 돌출 정도 (0~1)
    distance      : 인접 피크 간 최소 간격 (포인트 수)

    Returns
    -------
    peaks : list of int  — 검출된 피크 위치 (cm⁻¹), 확인 후 plot_topo_spectra에 전달
    """
    from scipy.signal import find_peaks as _find_peaks

    n_pts = data['n_pts']
    wn    = data['points'][0]['wavenumber']

    if point_indices is None:
        indices_0 = list(range(n_pts))          # 전체 포인트
    else:
        indices_0 = [i - 1 for i in point_indices]   # 1-based → 0-based
        out = [i for i in indices_0 if not (0 <= i < n_pts)]
        if out:
            raise ValueError(f"point_indices out of range (1~{n_pts}): {[i+1 for i in out]}")

    amps = np.array([data['points'][i][channel] for i in indices_0])
    avg  = amps.mean(axis=0)

    mask   = (wn >= xlim[0]) & (wn <= xlim[1])
    wn_c   = wn[mask]
    avg_c  = avg[mask]
    avg_n  = avg_c - np.percentile(avg_c, 5)
    avg_n /= avg_n.max()

    idx, props = _find_peaks(avg_n, height=height,
                              prominence=prominence, distance=distance)
    peaks = [int(round(wn_c[i])) for i in idx]

    # ── 확인용 플롯 ──────────────────────────────────────────
    pt_label = ('all' if point_indices is None
                else ', '.join(f'Pt{i}' for i in point_indices))
    avg_label = f'mean ({pt_label})'

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(wn_c, avg_n, color='#334466', linewidth=0.9, label=avg_label)
    ax.plot(wn_c[idx], avg_n[idx], 'v', color='#cc44aa',
            markersize=7, label='detected peaks')
    for i in idx:
        ax.text(wn_c[i], avg_n[i] + 0.04, f'{int(round(wn_c[i]))}',
                ha='center', va='bottom', fontsize=7.5,
                color='#cc44aa', fontweight='bold')
    ax.set_xlim(xlim)
    ax.set_ylim(-0.05, 1.25)
    ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=11)
    ax.set_ylabel('Amplitude (norm.)', fontsize=11)
    ax.set_title(f'Peak detection — {avg_label}  '
                 f'(height={height}, prominence={prominence}, distance={distance})',
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

    print(f'\n검출된 피크: {peaks}')
    print('→ 이상이 없으면 plot_topo_spectra(..., peaks=peaks) 로 전달하세요.')
    print('→ 수정이 필요하면:  peaks = [1100, 1229, ...]  직접 입력 후 재실행하세요.')
    return peaks


def plot_spectrum(data, point_index, channel='amplitude_uV',
                 xlim=(850, 2400), title=None, save_path=None):
    """
    단일 포인트 스펙트럼을 단독으로 플롯한다.
    피크 위치를 육안으로 읽기 위한 용도.

    Parameters
    ----------
    data          : read_park_pifm_tiff() 반환값
    point_index   : 포인트 번호 (1-based)
    channel       : 'amplitude_uV' or 'phase_deg'
    xlim          : X축 범위 (cm⁻¹)
    title         : 그래프 제목 (None 이면 자동)
    save_path     : 저장 경로 (None 이면 저장 안 함)
    """
    n_pts = data['n_pts']
    if not (1 <= point_index <= n_pts):
        raise ValueError(f"point_index {point_index} out of range (1~{n_pts})")

    wn  = data['points'][point_index - 1]['wavenumber']
    amp = data['points'][point_index - 1][channel]

    mask   = (wn >= xlim[0]) & (wn <= xlim[1])
    wn_c   = wn[mask]
    amp_c  = amp[mask]

    ch_label = {'amplitude_uV': 'PiFM Amplitude (μV)',
                'phase_deg':    'PiFM Phase (deg)'}

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(wn_c, amp_c, color='#334466', linewidth=0.9)
    ax.set_xlim(xlim)
    ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=11)
    ax.set_ylabel(ch_label.get(channel, channel), fontsize=11)
    ax.set_title(title if title else f'Pt{point_index} — {channel}', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.show()


def plot_avg_spectrum(data, point_indices=None, channel='amplitude_uV',
                     xlim=(850, 2400), normalize=False,
                     title=None, save_path=None, filepath=None):
    """
    선택한 포인트들의 평균 스펙트럼을 플롯하고 선택적으로 저장한다.

    Parameters
    ----------
    data          : read_park_pifm_tiff() 반환값
    point_indices : 평균에 사용할 포인트 번호 리스트 (1-based). None → 전체
                    예) [1, 3, 5]
    channel       : 'amplitude_uV' or 'phase_deg'
    xlim          : X축 범위 (cm⁻¹)
    normalize     : True → 0~1 정규화, False → 원본 단위 (기본)
    title         : 그래프 제목 (None 이면 자동)
    save_path     : 그래프 저장 경로 (.png 등). None → 저장 안 함
    filepath      : 데이터 저장 경로 (확장자 제외). None → 저장 안 함
                    예) 'result_152' → result_152_avg_amplitude_uV.xlsx / .txt

    Returns
    -------
    wn_c  : wavenumber 배열 (xlim 범위)
    avg_c : 평균 스펙트럼 값 배열
    """
    n_pts = data['n_pts']
    wn    = data['points'][0]['wavenumber']

    if point_indices is None:
        indices_0 = list(range(n_pts))
    else:
        indices_0 = [i - 1 for i in point_indices]
        out = [i for i in indices_0 if not (0 <= i < n_pts)]
        if out:
            raise ValueError(f"point_indices out of range (1~{n_pts}): {[i+1 for i in out]}")

    amps = np.array([data['points'][i][channel] for i in indices_0])
    avg  = amps.mean(axis=0)

    mask  = (wn >= xlim[0]) & (wn <= xlim[1])
    wn_c  = wn[mask]
    avg_c = avg[mask].copy()

    if normalize:
        base  = avg_c - np.percentile(avg_c, 5)
        peak  = base.max()
        avg_c = base / peak if peak > 0 else base

    pt_label = ('all' if point_indices is None
                else ', '.join(f'Pt{i}' for i in point_indices))

    ch_label = {'amplitude_uV': 'PiFM Amplitude (μV)',
                'phase_deg':    'PiFM Phase (deg)'}
    ylabel = 'Amplitude (norm.)' if normalize else ch_label.get(channel, channel)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(wn_c, avg_c, color='#334466', linewidth=1.0,
            label=f'mean ({pt_label})')
    ax.set_xlim(xlim)
    ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title if title else f'Average Spectrum  [{pt_label}]', fontsize=12)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.show()

    # ── 데이터 저장 ───────────────────────────────────────────
    if filepath:
        import xlsxwriter
        avg_unit  = 'norm.' if normalize else ch_label.get(channel, channel)
        avg_head  = f'Mean ({pt_label})'
        suffix    = f'_avg_{channel}'

        xlsx_path = filepath + suffix + '.xlsx'
        workbook  = xlsxwriter.Workbook(xlsx_path)
        ws        = workbook.add_worksheet('AvgSpectrum')
        ws.write(0, 0, 'Wavenumber (cm-1)')
        ws.write(0, 1, avg_head)
        ws.set_column(0, 0, 18)
        ws.set_column(1, 1, 28)
        for row, (w, a) in enumerate(zip(wn_c, avg_c)):
            ws.write(row + 1, 0, float(w))
            ws.write(row + 1, 1, float(a))
        workbook.close()

        txt_path = filepath + suffix + '.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f'Wavenumber\t{avg_head}\n')
            f.write(f'cm-1\t{avg_unit}\n')
            for w, a in zip(wn_c, avg_c):
                f.write(f'{w:.6g}\t{a:.6g}\n')

        print(f'Saved: {xlsx_path}')
        print(f'Saved: {txt_path}')

    return wn_c, avg_c


def plot_overlay_spectra(scan_list, channel='amplitude_uV',
                         xlim=(850, 2400), normalize=True,
                         title=None, save_path=None, filepath=None):
    """
    여러 스캔의 평균 스펙트럼을 한 그래프에 겹쳐 그린다.

    Parameters
    ----------
    scan_list : list of dict, 각 항목:
        {
          'data'          : read_park_pifm_tiff() 반환값,
          'label'         : 범례 이름 (예: 'Scan 152'),
          'point_indices' : [1, 3] 또는 None (기본: 전체 평균)
        }
    channel   : 'amplitude_uV' or 'phase_deg'
    xlim      : X축 범위 (cm⁻¹)
    normalize : True → 각 스캔을 0~1 정규화하여 비교 (기본), False → 원본 단위
    title     : 그래프 제목 (None 이면 자동)
    save_path : 그래프 저장 경로 (.png 등). None → 저장 안 함
    filepath  : 데이터 저장 경로 (확장자 제외). None → 저장 안 함
                공통 wavenumber 기준으로 각 스캔을 컬럼으로 저장.

    Returns
    -------
    results : list of (label, wn_c, avg_c)

    Usage
    -----
    scan_list = [
        {'data': data1, 'label': 'Sample A',  'point_indices': [1, 2, 3]},
        {'data': data2, 'label': 'Sample B',  'point_indices': None},
        {'data': data3, 'label': 'Reference', 'point_indices': [5]},
    ]
    plot_overlay_spectra(scan_list, normalize=True, save_path='overlay.png')
    """
    ch_label = {'amplitude_uV': 'PiFM Amplitude (μV)',
                'phase_deg':    'PiFM Phase (deg)'}
    ylabel  = 'Amplitude (norm.)' if normalize else ch_label.get(channel, channel)
    colors  = [plt.cm.tab10(i % 10) for i in range(len(scan_list))]
    results = []

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for color, entry in zip(colors, scan_list):
        data          = entry['data']
        label         = entry.get('label', '')
        point_indices = entry.get('point_indices', None)

        n_pts = data['n_pts']
        wn    = data['points'][0]['wavenumber']

        if point_indices is None:
            indices_0 = list(range(n_pts))
        else:
            indices_0 = [i - 1 for i in point_indices]
            out = [i for i in indices_0 if not (0 <= i < n_pts)]
            if out:
                raise ValueError(
                    f"[{label}] point_indices out of range (1~{n_pts}): {[i+1 for i in out]}")

        amps = np.array([data['points'][i][channel] for i in indices_0])
        avg  = amps.mean(axis=0)

        mask  = (wn >= xlim[0]) & (wn <= xlim[1])
        wn_c  = wn[mask]
        avg_c = avg[mask].copy()

        if normalize:
            base  = avg_c - np.percentile(avg_c, 5)
            peak  = base.max()
            avg_c = base / peak if peak > 0 else base

        pt_str     = ('all' if point_indices is None
                      else ', '.join(f'Pt{i}' for i in point_indices))
        line_label = f'{label}  [{pt_str}]' if label else f'[{pt_str}]'

        ax.plot(wn_c, avg_c, color=color, linewidth=1.0, label=line_label)
        results.append((label, wn_c, avg_c))

    ax.set_xlim(xlim)
    ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title if title else f'Overlay — {channel}', fontsize=12)
    ax.legend(fontsize=8.5, framealpha=0.9, edgecolor='#bbbbbb')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.show()

    # ── 데이터 저장 (공통 wavenumber 기준, 각 스캔 컬럼) ────
    if filepath:
        import xlsxwriter

        avg_unit = 'norm.' if normalize else ch_label.get(channel, channel)
        suffix   = f'_overlay_{channel}'
        wn_ref   = results[0][1]

        xlsx_path = filepath + suffix + '.xlsx'
        workbook  = xlsxwriter.Workbook(xlsx_path)
        ws        = workbook.add_worksheet('Overlay')
        headers_x = ['Wavenumber (cm-1)'] + [r[0] for r in results]
        col_units = ['cm-1'] + [avg_unit] * len(results)
        for col, h in enumerate(headers_x):
            ws.write(0, col, h)
            ws.set_column(col, col, max(len(h) + 2, 14))
        for row, wn_val in enumerate(wn_ref):
            ws.write(row + 1, 0, float(wn_val))
            for col, (_, wn_c, avg_c) in enumerate(results):
                val = float(avg_c[row]) if row < len(avg_c) else ''
                ws.write(row + 1, col + 1, val)
        workbook.close()

        txt_path = filepath + suffix + '.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            col_heads = ['Wavenumber'] + [r[0] for r in results]
            f.write('\t'.join(col_heads) + '\n')
            f.write('\t'.join(col_units) + '\n')
            for row, wn_val in enumerate(wn_ref):
                vals = [f'{wn_val:.6g}']
                for _, wn_c, avg_c in results:
                    vals.append(f'{avg_c[row]:.6g}' if row < len(avg_c) else '')
                f.write('\t'.join(vals) + '\n')

        print(f'Saved: {xlsx_path}')
        print(f'Saved: {txt_path}')

    return results


def plot_overlay_normalized(scan_list, channel='amplitude_uV',
                            xlim=(700, 2400),
                            peaks=None,
                            peak_color='#cc44aa',
                            title=None, save_path=None, filepath=None):
    """
    여러 스캔의 평균 스펙트럼을 각 스펙트럼의 최대값으로 정규화(0~1)하여 겹쳐 그린다.
    피크 위치를 수직 점선으로 표시할 수 있다.

    Parameters
    ----------
    scan_list  : list of dict, 각 항목:
        {
          'data'          : read_park_pifm_tiff() 반환값,
          'label'         : 범례 이름 (예: 'Scan 152'),
          'point_indices' : [1, 3] 또는 None (기본: 전체 평균)
        }
    channel    : 'amplitude_uV' or 'phase_deg'
    xlim       : X축 범위 (cm⁻¹)
    peaks      : 수직선으로 표시할 피크 위치 리스트, 예) [1100, 1229, 1728]
                 None 이면 표시 안 함
    peak_color : 피크 수직선 및 라벨 색상
    title      : 그래프 제목 (None 이면 자동)
    save_path  : 그래프 저장 경로 (.png 등). None → 저장 안 함
    filepath   : 데이터 저장 경로 (확장자 제외). None → 저장 안 함
                 공통 wavenumber 기준으로 각 스캔을 컬럼으로 저장.

    Returns
    -------
    results : list of (label, wn_c, avg_norm)
    """
    ch_label = {'amplitude_uV': 'PiFM Amplitude (norm. by max)',
                'phase_deg':    'PiFM Phase (norm. by max)'}
    ylabel  = ch_label.get(channel, channel)
    colors  = [plt.cm.tab10(i % 10) for i in range(len(scan_list))]
    results = []

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for color, entry in zip(colors, scan_list):
        data          = entry['data']
        label         = entry.get('label', '')
        point_indices = entry.get('point_indices', None)

        n_pts = data['n_pts']
        wn    = data['points'][0]['wavenumber']

        if point_indices is None:
            indices_0 = list(range(n_pts))
        else:
            indices_0 = [i - 1 for i in point_indices]
            out = [i for i in indices_0 if not (0 <= i < n_pts)]
            if out:
                raise ValueError(
                    f"[{label}] point_indices out of range (1~{n_pts}): {[i+1 for i in out]}")

        amps = np.array([data['points'][i][channel] for i in indices_0])
        avg  = amps.mean(axis=0)

        mask     = (wn >= xlim[0]) & (wn <= xlim[1])
        wn_c     = wn[mask]
        avg_c    = avg[mask].copy()
        max_val  = avg_c.max()
        avg_norm = avg_c / max_val if max_val > 0 else avg_c

        pt_str     = ('all' if point_indices is None
                      else ', '.join(f'Pt{i}' for i in point_indices))
        line_label = f'{label}  [{pt_str}]' if label else f'[{pt_str}]'

        ax.plot(wn_c, avg_norm, color=color, linewidth=1.0, label=line_label)
        results.append((label, wn_c, avg_norm))

    # ── 피크 수직선 + 라벨 ───────────────────────────────────────
    if peaks:
        y_base  = 1.06
        y_step  = 0.12
        MIN_GAP = 130
        MAX_LEVEL = 3

        levels = []
        for k, pk in enumerate(sorted(peaks)):
            if k == 0:
                levels.append(0)
            else:
                gap = pk - sorted(peaks)[k - 1]
                if gap < MIN_GAP:
                    levels.append((levels[-1] + 1) % MAX_LEVEL)
                else:
                    levels.append(0)

        for k, pk in enumerate(sorted(peaks)):
            y_label = y_base + levels[k] * y_step
            ax.axvline(pk, color=peak_color, linewidth=0.9,
                       linestyle='--', alpha=0.75, zorder=0)
            ax.annotate(f'{pk} cm⁻¹',
                        xy=(pk, 1.0),
                        xytext=(pk, y_label),
                        ha='center', va='bottom', fontsize=7.5,
                        color=peak_color, fontweight='bold',
                        arrowprops=dict(arrowstyle='-', color=peak_color,
                                        lw=0.6, alpha=0.5))

    y_top = 1.5 if peaks else 1.15
    ax.set_xlim(xlim)
    ax.invert_xaxis()
    ax.set_ylim(-0.05, y_top)
    ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title if title else f'Normalized Overlay — {channel}', fontsize=12)
    ax.legend(fontsize=8.5, framealpha=0.9, edgecolor='#bbbbbb')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.show()

    # ── 데이터 저장 ───────────────────────────────────────────────
    if filepath:
        import xlsxwriter

        avg_unit = 'norm. by max'
        suffix   = f'_overlay_norm_{channel}'
        wn_ref   = results[0][1]

        xlsx_path = filepath + suffix + '.xlsx'
        workbook  = xlsxwriter.Workbook(xlsx_path)
        ws        = workbook.add_worksheet('OverlayNorm')
        headers_x = ['Wavenumber (cm-1)'] + [r[0] for r in results]
        col_units = ['cm-1'] + [avg_unit] * len(results)
        for col, h in enumerate(headers_x):
            ws.write(0, col, h)
            ws.set_column(col, col, max(len(h) + 2, 14))
        for row, wn_val in enumerate(wn_ref):
            ws.write(row + 1, 0, float(wn_val))
            for col, (_, wn_c, avg_norm) in enumerate(results):
                val = float(avg_norm[row]) if row < len(avg_norm) else ''
                ws.write(row + 1, col + 1, val)
        workbook.close()

        txt_path = filepath + suffix + '.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            col_heads = ['Wavenumber'] + [r[0] for r in results]
            f.write('\t'.join(col_heads) + '\n')
            f.write('\t'.join(col_units) + '\n')
            for row, wn_val in enumerate(wn_ref):
                vals = [f'{wn_val:.6g}']
                for _, wn_c, avg_norm in results:
                    vals.append(f'{avg_norm[row]:.6g}' if row < len(avg_norm) else '')
                f.write('\t'.join(vals) + '\n')

        print(f'Saved: {xlsx_path}')
        print(f'Saved: {txt_path}')

    return results


def plot_overlay_normalized_stacked(scan_list, channel='amplitude_uV',
                                    xlim=(700, 2400),
                                    stack_gap=1.2,
                                    peaks=None,
                                    peak_color='#cc44aa',
                                    title=None, save_path=None, filepath=None):
    """
    여러 스캔의 평균 스펙트럼을 최대값으로 정규화(0~1)한 뒤 수직으로 쌓아
    피크 위치를 비교하기 쉽게 표시한다.

    Parameters
    ----------
    scan_list  : list of dict, 각 항목:
        {
          'data'          : read_park_pifm_tiff() 반환값,
          'label'         : 범례 이름 (예: 'Sample A'),
          'point_indices' : [1, 3] 또는 None (기본: 전체 평균)
        }
    channel    : 'amplitude_uV' or 'phase_deg'
    xlim       : X축 범위 (cm⁻¹)
    stack_gap  : 정규화된 스펙트럼 간 수직 간격 (1.0 = 정규화 피크 높이 기준)
    peaks      : 수직선으로 표시할 피크 위치 리스트, 예) [1100, 1229, 1728]
                 None 이면 표시 안 함
    peak_color : 피크 수직선 및 라벨 색상
    title      : 그래프 제목 (None 이면 자동)
    save_path  : 그래프 저장 경로 (.png 등). None → 저장 안 함
    filepath   : 데이터 저장 경로 (확장자 제외). None → 저장 안 함
                 저장값은 offset 미포함 정규화값.

    Returns
    -------
    results : list of (label, wn_c, avg_norm)
    """
    n_scans = len(scan_list)
    colors  = [plt.cm.tab10(i % 10) for i in range(n_scans)]
    results = []

    for entry in scan_list:
        data          = entry['data']
        label         = entry.get('label', '')
        point_indices = entry.get('point_indices', None)

        n_pts = data['n_pts']
        wn    = data['points'][0]['wavenumber']

        if point_indices is None:
            indices_0 = list(range(n_pts))
        else:
            indices_0 = [i - 1 for i in point_indices]
            out = [i for i in indices_0 if not (0 <= i < n_pts)]
            if out:
                raise ValueError(
                    f"[{label}] point_indices out of range (1~{n_pts}): {[i+1 for i in out]}")

        amps = np.array([data['points'][i][channel] for i in indices_0])
        avg  = amps.mean(axis=0)

        mask     = (wn >= xlim[0]) & (wn <= xlim[1])
        wn_c     = wn[mask]
        avg_c    = avg[mask].copy()
        max_val  = avg_c.max()
        avg_norm = avg_c / max_val if max_val > 0 else avg_c

        pt_str = ('all' if point_indices is None
                  else ', '.join(f'Pt{i}' for i in point_indices))
        results.append((label, wn_c, avg_norm, pt_str))

    # ── 플롯 ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 3.0 + n_scans * 1.4))

    ytick_pos    = []
    ytick_labels = []

    for i, (color, (label, wn_c, avg_norm, pt_str)) in enumerate(zip(colors, results)):
        offset     = i * stack_gap
        line_label = f'{label}  [{pt_str}]' if label else f'[{pt_str}]'
        ax.plot(wn_c, avg_norm + offset, color=color, linewidth=0.9)
        ax.axhline(offset, color=color, linewidth=0.4, linestyle=':', alpha=0.5)
        ytick_pos.append(offset)
        ytick_labels.append(line_label)

    # ── 피크 수직선 + 라벨 ───────────────────────────────────────
    if peaks:
        top       = (n_scans - 1) * stack_gap + 1.0
        y_base    = top + 0.12
        y_step    = 0.30
        MIN_GAP   = 130
        MAX_LEVEL = 3

        levels = []
        for k, pk in enumerate(sorted(peaks)):
            if k == 0:
                levels.append(0)
            else:
                gap = pk - sorted(peaks)[k - 1]
                levels.append((levels[-1] + 1) % MAX_LEVEL if gap < MIN_GAP else 0)

        for k, pk in enumerate(sorted(peaks)):
            y_label = y_base + levels[k] * y_step
            ax.axvline(pk, color=peak_color, linewidth=0.9,
                       linestyle='--', alpha=0.75, zorder=0)
            ax.annotate(f'{pk} cm⁻¹',
                        xy=(pk, top),
                        xytext=(pk, y_label),
                        ha='center', va='bottom', fontsize=7.5,
                        color=peak_color, fontweight='bold',
                        arrowprops=dict(arrowstyle='-', color=peak_color,
                                        lw=0.6, alpha=0.5))

    # ── 축 설정 ───────────────────────────────────────────────────
    top_total = (n_scans - 1) * stack_gap + 1.0
    y_ceiling = top_total + (1.1 + 0.30 * 3 if peaks else 0.3)
    ax.set_xlim(xlim)
    ax.invert_xaxis()
    ax.set_ylim(-0.3, y_ceiling)
    ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=11)
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_labels, fontsize=9)
    ax.tick_params(axis='y', length=0)
    ax.set_title(title if title else f'Stacked Normalized Overlay — {channel}', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    plt.show()

    # ── 데이터 저장 (offset 미포함 정규화값) ─────────────────────
    if filepath:
        import xlsxwriter

        avg_unit  = 'norm. by max'
        suffix    = f'_overlay_norm_stacked_{channel}'
        wn_ref    = results[0][1]

        xlsx_path = filepath + suffix + '.xlsx'
        workbook  = xlsxwriter.Workbook(xlsx_path)
        ws        = workbook.add_worksheet('OverlayNormStack')
        headers_x = ['Wavenumber (cm-1)'] + [r[0] for r in results]
        col_units = ['cm-1'] + [avg_unit] * len(results)
        for col, h in enumerate(headers_x):
            ws.write(0, col, h)
            ws.set_column(col, col, max(len(h) + 2, 14))
        for row, wn_val in enumerate(wn_ref):
            ws.write(row + 1, 0, float(wn_val))
            for col, (_, wn_c, avg_norm, _pt) in enumerate(results):
                val = float(avg_norm[row]) if row < len(avg_norm) else ''
                ws.write(row + 1, col + 1, val)
        workbook.close()

        txt_path = filepath + suffix + '.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            col_heads = ['Wavenumber'] + [r[0] for r in results]
            f.write('\t'.join(col_heads) + '\n')
            f.write('\t'.join(col_units) + '\n')
            for row, wn_val in enumerate(wn_ref):
                vals = [f'{wn_val:.6g}']
                for _, wn_c, avg_norm, _pt in results:
                    vals.append(f'{avg_norm[row]:.6g}' if row < len(avg_norm) else '')
                f.write('\t'.join(vals) + '\n')

        print(f'Saved: {xlsx_path}')
        print(f'Saved: {txt_path}')

    return [(r[0], r[1], r[2]) for r in results]
