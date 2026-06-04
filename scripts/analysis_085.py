"""
analysis_085.py
park_pifm 모듈을 사용한 PiFM 데이터 분석 예시

실행:
    python analysis_085.py

IPython:
    %run analysis_085.py
"""

from park_pifm import read_park_pifm_tiff, read_info_txt, plot_topo_spectra, save_spectra

# ── 파일 경로 ──────────────────────────────────────────────────
filepath      = 'ps_p1_sb_l1_on_p_points_260514__085.tiff'
filepath_info = 'ps_p1_sb_l1_on_p_points_260514__085_info.txt'  # 있으면 사용

# ── 파일 읽기 ──────────────────────────────────────────────────
data = read_park_pifm_tiff(filepath)
print(f"Scan size : {data['scan_x_nm']:.0f} x {data['scan_y_nm']:.0f} nm")
print(f"Z range   : {data['z_nm'].min():.3f} ~ {data['z_nm'].max():.3f} nm")
print(f"Points    : {data['n_pts']}")
print(f"WN range  : {data['points'][0]['wavenumber'].min():.1f}"
      f" ~ {data['points'][0]['wavenumber'].max():.1f} cm⁻¹")

# ── 포인트 좌표: info.txt 우선, 없으면 raw 파일에서 자동 추출 ──
import os
if os.path.exists(filepath_info):
    coords_nm = read_info_txt(filepath_info)
    print('Coords from info.txt')
else:
    coords_nm = data['coords_nm']
    print('Coords from raw TIFF')

print('Coords (nm):')
for i, (x, y) in enumerate(coords_nm):
    print(f'  Pt{i+1}: X={x:.1f}, Y={y:.1f}')

# ── 데이터 저장 (Excel + Origin txt) ──────────────────────────
save_spectra(data, channel='amplitude_uV', filepath='result_085')
save_spectra(data, channel='phase_deg',    filepath='result_085')

# ── Amplitude 플롯 ─────────────────────────────────────────────
plot_topo_spectra(data,
                  coords_nm=coords_nm,
                  channel='amplitude_uV',
                  offset_step=120,
                  title='ps_p1_sb_l1_on_p  —  #085',
                  save_path='result_085_topo_amplitude.png')

# ── Phase 플롯 ─────────────────────────────────────────────────
plot_topo_spectra(data,
                  coords_nm=coords_nm,
                  channel='phase_deg',
                  offset_step=50,
                  title='ps_p1_sb_l1_on_p  —  #085 (Phase)',
                  save_path='result_085_topo_phase.png')
