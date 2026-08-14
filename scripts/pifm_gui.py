"""
pifm_gui.py — PiFM Analysis GUI
analysis_pifm.ipynb 기능을 tkinter GUI로 구현
"""

import sys
import os
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# park_pifm 모듈 로드 (같은 scripts 폴더)
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from park_pifm import (
    read_park_pifm_tiff, read_info_txt, save_spectra,
    plot_topo_spectra, plot_avg_spectrum,
)


# ─────────────────────────────────────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────────────────────────────────────

def _parse_ints(text):
    """'1,2,3' → [1, 2, 3], 빈 문자열 → None"""
    text = text.strip()
    if not text:
        return None
    return [int(x.strip()) for x in text.split(',') if x.strip().isdigit()]


def _parse_floats(text):
    """'800,1263,1437' → [800.0, 1263.0, 1437.0], 빈 문자열 → None"""
    text = text.strip()
    if not text:
        return None
    result = []
    for x in text.split(','):
        x = x.strip()
        if x:
            try:
                result.append(float(x))
            except ValueError:
                pass
    return result or None


def _auto_file_num(filepath):
    m = re.search(r'__(\d+)\.tiff?$', filepath, re.IGNORECASE)
    return m.group(1) if m else Path(filepath).stem


# ─────────────────────────────────────────────────────────────────────────────
# 피크 선택 대화창
# ─────────────────────────────────────────────────────────────────────────────

class PeakPickerDialog(tk.Toplevel):
    """
    스펙트럼을 보며 마우스 클릭으로 피크를 선택하는 대화창.

    조작법
    ------
    - 왼쪽 클릭  : 피크 추가 (가장 가까운 파수 포인트로 스냅)
    - 오른쪽 클릭: 가장 가까운 피크 제거
    - 줌/팬      : 툴바 버튼 사용 (줌/팬 모드에서는 클릭 동작 안 함)
    - 확인       : dlg.result 에 리스트 반환
    """

    def __init__(self, parent, wn, spectrum, current_peaks=None,
                 xlim=None, title='피크 선택'):
        super().__init__(parent)
        self.title(title)
        self.geometry('860x560')
        self.minsize(640, 420)
        self.result = None

        self._wn    = np.asarray(wn, dtype=float)
        self._spec  = np.asarray(spectrum, dtype=float)
        self._xlim  = xlim
        self._peaks = []          # float list (정수 파수 값)
        self._arts  = {}          # wn_val -> (vline, text)

        self._build()
        self._draw_spectrum()

        # 기존 피크 복원
        for p in (current_peaks or []):
            self._add_peak(float(round(p)), redraw=False)
        self._refresh_list()
        self._canvas.draw()

        self._cid = self._canvas.mpl_connect('button_press_event', self._on_click)
        self.grab_set()
        self.wait_window()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build(self):
        # 안내 문구
        ttk.Label(
            self,
            text='  ● 왼쪽 클릭: 피크 추가   ● 오른쪽 클릭: 가장 가까운 피크 제거'
                 '   ● 줌/팬: 툴바 사용   ● 툴바 모드 해제 후 클릭 가능',
            foreground='#555',
        ).pack(fill='x', padx=6, pady=(4, 0))

        # matplotlib 캔버스
        frm_fig = ttk.Frame(self)
        frm_fig.pack(fill='both', expand=True, padx=6, pady=4)

        self._fig = Figure(figsize=(9, 4.2), tight_layout={'rect': [0, 0, 1, 0.87]})
        self._ax  = self._fig.add_subplot(111)

        self._canvas = FigureCanvasTkAgg(self._fig, master=frm_fig)
        self._canvas.get_tk_widget().pack(fill='both', expand=True)

        # 툴바 (줌·팬·홈)
        frm_tb = ttk.Frame(self)
        frm_tb.pack(fill='x', padx=6)
        self._toolbar = NavigationToolbar2Tk(self._canvas, frm_tb)
        self._toolbar.update()

        # ── 자동 검출 파라미터 ──────────────────────────────────
        frm_auto = ttk.LabelFrame(self, text=' 자동 검출 파라미터 (정규화 스펙트럼 기준, 0~1) ')
        frm_auto.pack(fill='x', padx=6, pady=(4, 2))

        ttk.Label(frm_auto, text='Height:').grid(row=0, column=0, padx=4, pady=3, sticky='w')
        self._e_height = ttk.Entry(frm_auto, width=6)
        self._e_height.insert(0, '0.05')
        self._e_height.grid(row=0, column=1, padx=2)

        ttk.Label(frm_auto, text='Prominence:').grid(row=0, column=2, padx=(8, 4), sticky='w')
        self._e_prom = ttk.Entry(frm_auto, width=6)
        self._e_prom.insert(0, '0.05')
        self._e_prom.grid(row=0, column=3, padx=2)

        ttk.Label(frm_auto, text='최소 간격 (cm⁻¹):').grid(row=0, column=4, padx=(8, 4), sticky='w')
        self._e_dist = ttk.Entry(frm_auto, width=6)
        self._e_dist.insert(0, '20')
        self._e_dist.grid(row=0, column=5, padx=2)

        ttk.Button(frm_auto, text='자동 검출 (추가)',
                   command=lambda: self._auto_detect(clear=False), width=14).grid(
            row=0, column=6, padx=8, pady=3)
        ttk.Button(frm_auto, text='자동 검출 (초기화)',
                   command=lambda: self._auto_detect(clear=True), width=14).grid(
            row=0, column=7, padx=2, pady=3)

        ttk.Separator(frm_auto, orient='vertical').pack_forget()   # spacer trick
        self._pk_smooth = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm_auto, text='Smooth', variable=self._pk_smooth,
                        command=self._redraw_spectrum).grid(row=0, column=8, padx=(12, 2), pady=3)
        ttk.Label(frm_auto, text='W:').grid(row=0, column=9, padx=(0, 1))
        self._pk_sw = ttk.Entry(frm_auto, width=4)
        self._pk_sw.insert(0, '11')
        self._pk_sw.grid(row=0, column=10, padx=1)
        ttk.Label(frm_auto, text='P:').grid(row=0, column=11, padx=(4, 1))
        self._pk_sp = ttk.Entry(frm_auto, width=3)
        self._pk_sp.insert(0, '3')
        self._pk_sp.grid(row=0, column=12, padx=(1, 4))

        # 하단: 피크 목록 + 버튼
        frm_bot = ttk.Frame(self)
        frm_bot.pack(fill='x', padx=6, pady=4)

        frm_list = ttk.LabelFrame(frm_bot, text=' 선택된 피크 (cm⁻¹) ')
        frm_list.pack(side='left', fill='both', expand=True, padx=(0, 6))

        self._listbox = tk.Listbox(
            frm_list, height=3, selectmode='single',
            activestyle='dotbox', width=14,
        )
        self._listbox.pack(side='left', fill='both', expand=True, padx=3, pady=3)

        frm_lb = ttk.Frame(frm_list)
        frm_lb.pack(side='right', padx=3, pady=3)
        ttk.Button(frm_lb, text='선택 삭제', command=self._delete_selected, width=10).pack(pady=2)
        ttk.Button(frm_lb, text='전체 삭제', command=self._clear_all,       width=10).pack(pady=2)

        frm_ok = ttk.Frame(frm_bot)
        frm_ok.pack(side='right')
        ttk.Button(frm_ok, text='확인', command=self._ok,      width=10).pack(pady=3)
        ttk.Button(frm_ok, text='취소', command=self.destroy,  width=10).pack(pady=3)

    # ── 그래프 ────────────────────────────────────────────────────────────────

    def _pk_smooth_args(self):
        try:
            sw = int(self._pk_sw.get())
            sp = int(self._pk_sp.get())
        except (ValueError, AttributeError):
            sw, sp = 11, 3
        return self._pk_smooth.get(), sw, sp

    def _redraw_spectrum(self):
        """Smooth 체크박스 변경 시 스펙트럼만 재렌더링 (피크 마커는 유지)"""
        ax = self._ax
        # 기존 스펙트럼 선만 제거 (vline/text 아티스트는 tag로 구분 불가 → 전체 재그리기)
        ax.clear()
        self._draw_spectrum_lines(ax)
        self._set_axes_style(ax)
        # 피크 마커 복원
        saved_peaks = list(self._peaks)
        self._peaks.clear()
        self._arts.clear()
        for wn_val in saved_peaks:
            self._add_peak(wn_val, redraw=False)
        self._canvas.draw()

    def _draw_spectrum_lines(self, ax):
        """raw(±smooth) 스펙트럼 선을 ax에 그림"""
        use_sm, sw, sp = self._pk_smooth_args()
        if use_sm:
            from park_pifm import smooth_spectrum
            y_sm = smooth_spectrum(self._spec, sw, sp)
            ax.plot(self._wn, self._spec, color='#2060c0', linewidth=0.6, alpha=0.3)
            ax.plot(self._wn, y_sm,        color='#2060c0', linewidth=1.2)
        else:
            ax.plot(self._wn, self._spec, color='#2060c0', linewidth=0.9)

    def _set_axes_style(self, ax):
        ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=9)
        ax.set_ylabel('Intensity', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if self._xlim:
            lo, hi = min(self._xlim), max(self._xlim)
            ax.set_xlim(hi, lo)
            mask = (self._wn >= lo) & (self._wn <= hi)
            if mask.any():
                y = self._spec[mask]
                margin = (y.max() - y.min()) * 0.05
                ax.set_ylim(y.min() - margin, y.max() + margin)
        else:
            ax.invert_xaxis()

    def _draw_spectrum(self):
        ax = self._ax
        ax.clear()
        self._draw_spectrum_lines(ax)
        ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=9)
        ax.set_ylabel('Intensity', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if self._xlim:
            lo, hi = min(self._xlim), max(self._xlim)
            ax.set_xlim(hi, lo)   # X축 반전
            mask = (self._wn >= lo) & (self._wn <= hi)
            if mask.any():
                y = self._spec[mask]
                margin = (y.max() - y.min()) * 0.05
                ax.set_ylim(y.min() - margin, y.max() + margin)
        else:
            ax.invert_xaxis()

    # ── 자동 검출 ─────────────────────────────────────────────────────────────

    def _auto_detect(self, clear=True):
        """scipy.signal.find_peaks 로 피크를 자동 검출하여 현재 목록에 추가/초기화"""
        try:
            from scipy.signal import find_peaks as _fp
        except ImportError:
            messagebox.showerror('오류', 'scipy가 설치되어 있지 않습니다.')
            return
        try:
            height  = float(self._e_height.get())
            prom    = float(self._e_prom.get())
            dist_wn = float(self._e_dist.get())
        except ValueError:
            messagebox.showerror('입력 오류', '파라미터가 올바르지 않습니다 (숫자 입력).')
            return

        # cm⁻¹ 간격 → 포인트 수
        wn_step  = abs(float(self._wn[1]) - float(self._wn[0])) if len(self._wn) > 1 else 1.0
        dist_pts = max(1, int(dist_wn / wn_step))

        # smooth 적용 (활성화된 경우 smoothed 스펙트럼으로 피크 검출)
        use_sm, sw, sp = self._pk_smooth_args()
        if use_sm:
            from park_pifm import smooth_spectrum
            spec = smooth_spectrum(self._spec.astype(float), sw, sp)
        else:
            spec = self._spec.astype(float)

        # min-max 정규화
        lo, hi = spec.min(), spec.max()
        spec_n = (spec - lo) / (hi - lo) if hi > lo else spec

        idx, _ = _fp(spec_n, height=height, prominence=prom, distance=dist_pts)

        if clear:
            self._clear_all()

        for i in idx:
            wn_val = float(round(float(self._wn[i])))
            self._add_peak(wn_val, redraw=False)

        self._refresh_list()
        self._canvas.draw()

    def _add_peak(self, wn_val, redraw=True):
        """피크 마커 + 파수 레이블을 그래프에 추가"""
        if wn_val in self._peaks:
            return
        self._peaks.append(wn_val)

        ax = self._ax
        vline = ax.axvline(wn_val, color='crimson', linestyle='--',
                           alpha=0.75, linewidth=1.0)
        # x = 데이터 좌표, y = axes 좌표 (0~1) → 줌해도 항상 상단에 표시
        text = ax.text(
            wn_val, 1.02, f'{wn_val:.0f}',
            transform=ax.get_xaxis_transform(),
            color='crimson', fontsize=8, fontweight='bold',
            va='bottom', ha='center', rotation=90, clip_on=False,
        )
        self._arts[wn_val] = (vline, text)
        if redraw:
            self._canvas.draw()

    def _remove_peak(self, wn_val):
        """피크 마커 제거"""
        if wn_val in self._peaks:
            self._peaks.remove(wn_val)
        if wn_val in self._arts:
            vline, text = self._arts.pop(wn_val)
            vline.remove()
            text.remove()
            self._canvas.draw()
        self._refresh_list()

    def _refresh_list(self):
        self._listbox.delete(0, 'end')
        for p in sorted(self._peaks, reverse=True):
            self._listbox.insert('end', f'{p:.0f}')

    # ── 이벤트 ────────────────────────────────────────────────────────────────

    def _on_click(self, event):
        if event.inaxes != self._ax:
            return
        if self._toolbar.mode:          # 줌/팬 모드에서는 동작 안 함
            return
        x = event.xdata
        if x is None:
            return

        # 가장 가까운 실제 파수 포인트로 스냅
        idx     = int(np.argmin(np.abs(self._wn - x)))
        wn_snap = float(round(float(self._wn[idx])))

        if event.button == 1:           # 왼쪽 클릭 → 추가
            self._add_peak(wn_snap)
            self._refresh_list()

        elif event.button == 3:         # 오른쪽 클릭 → 가장 가까운 피크 제거
            if self._peaks:
                nearest = min(self._peaks, key=lambda p: abs(p - x))
                self._remove_peak(nearest)

    def _delete_selected(self):
        sel = self._listbox.curselection()
        if not sel:
            return
        sorted_peaks = sorted(self._peaks, reverse=True)
        self._remove_peak(sorted_peaks[sel[0]])

    def _clear_all(self):
        for p in list(self._peaks):
            self._remove_peak(p)

    def _ok(self):
        self.result = sorted(self._peaks)
        self.destroy()


# ─────────────────────────────────────────────────────────────────────────────
# 파일 추가 다이얼로그 (Overlay 탭용)
# ─────────────────────────────────────────────────────────────────────────────

class AddScanDialog(tk.Toplevel):
    def __init__(self, parent, path):
        super().__init__(parent)
        self.title('스캔 추가')
        self.resizable(False, False)
        self.result = None

        ttk.Label(self, text='파일:').grid(row=0, column=0, padx=8, pady=4, sticky='w')
        ttk.Label(self, text=Path(path).name, foreground='#555').grid(
            row=0, column=1, padx=8, pady=4, sticky='w')

        ttk.Label(self, text='레이블:').grid(row=1, column=0, padx=8, pady=4, sticky='w')
        self._lbl = tk.StringVar(value=Path(path).stem[:30])
        ttk.Entry(self, textvariable=self._lbl, width=30).grid(row=1, column=1, padx=8, pady=4)

        ttk.Label(self, text='포인트 (쉼표):').grid(row=2, column=0, padx=8, pady=4, sticky='w')
        self._pts = tk.StringVar(value='1,2,3,4,5')
        ttk.Entry(self, textvariable=self._pts, width=30).grid(row=2, column=1, padx=8, pady=4)
        ttk.Label(self, text='(비우면 전체)').grid(row=2, column=2, padx=4, sticky='w')

        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=3, column=0, columnspan=3, pady=8)
        ttk.Button(btn_frame, text='확인', command=self._ok, width=10).pack(side='left', padx=5)
        ttk.Button(btn_frame, text='취소', command=self.destroy, width=10).pack(side='left', padx=5)

        self.grab_set()
        self.wait_window()

    def _ok(self):
        label = self._lbl.get().strip()
        pts = _parse_ints(self._pts.get())
        self.result = (label, pts)
        self.destroy()


# ─────────────────────────────────────────────────────────────────────────────
# 오버레이 주석 편집 다이얼로그
# ─────────────────────────────────────────────────────────────────────────────

class OverlayAnnotatorDialog(tk.Toplevel):
    """
    오버레이 그래프에 피크 마커와 텍스트 주석을 추가하는 대화창.

    모드
    ----
    피크 표시  : 왼쪽 클릭 → 가장 가까운 파수에 수직선 + 값 레이블
                오른쪽 클릭 → 가장 가까운 피크 제거
    텍스트 추가: 왼쪽 클릭 → 텍스트 입력 → 해당 위치에 표시
                오른쪽 클릭 → 가장 최근 텍스트 제거
    """

    _COLORS = ['#2060c0', '#e03030', '#20a020', '#e07800',
               '#8030c0', '#00a0a0', '#c04080', '#808000']

    def __init__(self, parent, scan_list, channel, xlim, title=None,
                 stack_gap=0.5, initial_peaks=None, save_dir=None):
        super().__init__(parent)
        self.title('오버레이 분석')
        self.geometry('1040x700')
        self.minsize(860, 560)

        self._scan_list   = scan_list
        self._channel     = channel
        self._xlim        = xlim
        self._ov_title    = title
        self._stack_gap   = stack_gap
        self._save_dir    = save_dir

        self._wn          = None
        self._peak_values = []
        self._peak_arts   = {}
        self._text_annots = []

        self._plot_type  = tk.StringVar(value='stacked')
        self._mode       = tk.StringVar(value='peak')
        self._ov_smooth  = tk.BooleanVar(value=False)

        self._build()
        self._draw_overlay()

        for p in (initial_peaks or []):
            self._add_peak(float(round(p)), redraw=False)
        self._canvas.draw()

        self._cid = self._canvas.mpl_connect('button_press_event', self._on_click)
        self.grab_set()
        self.wait_window()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build(self):
        ttk.Label(
            self,
            text='  ● 피크 모드: 왼쪽 클릭=추가, 오른쪽 클릭=제거'
                 '   ● 텍스트 모드: 왼쪽 클릭=텍스트 입력, 오른쪽 클릭=마지막 제거'
                 '   ● 줌/팬: 툴바 사용',
            foreground='#555',
        ).pack(fill='x', padx=6, pady=(4, 0))

        # ── 그래프 유형 선택 ───────────────────────────────────
        frm_type = ttk.LabelFrame(self, text=' 그래프 유형 ')
        frm_type.pack(fill='x', padx=6, pady=(4, 0))

        for val, txt in [('raw', '원본'), ('norm', '정규화'), ('stacked', 'Stacked 정규화')]:
            ttk.Radiobutton(frm_type, text=txt, variable=self._plot_type,
                            value=val, command=self._redraw).pack(
                side='left', padx=10, pady=3)

        ttk.Separator(frm_type, orient='vertical').pack(side='left', fill='y', padx=8, pady=4)
        ttk.Label(frm_type, text='Stack gap:').pack(side='left', padx=(0, 4))
        self._e_gap = ttk.Entry(frm_type, width=6)
        self._e_gap.insert(0, str(self._stack_gap))
        self._e_gap.pack(side='left', padx=2)
        ttk.Button(frm_type, text='적용', command=self._redraw, width=6).pack(
            side='left', padx=6)

        ttk.Separator(frm_type, orient='vertical').pack(side='left', fill='y', padx=8, pady=4)
        ttk.Checkbutton(frm_type, text='Smooth', variable=self._ov_smooth,
                        command=self._redraw).pack(side='left', padx=(0, 4))
        ttk.Label(frm_type, text='W:').pack(side='left', padx=(0, 2))
        self._ov_sw = ttk.Entry(frm_type, width=4)
        self._ov_sw.insert(0, '11')
        self._ov_sw.pack(side='left', padx=2)
        ttk.Label(frm_type, text='P:').pack(side='left', padx=(4, 2))
        self._ov_sp = ttk.Entry(frm_type, width=3)
        self._ov_sp.insert(0, '3')
        self._ov_sp.pack(side='left', padx=(2, 6))

        frm_fig = ttk.Frame(self)
        frm_fig.pack(fill='both', expand=True, padx=6, pady=4)

        self._fig = Figure(figsize=(10, 5), tight_layout={'rect': [0, 0, 1, 0.85]})
        self._ax  = self._fig.add_subplot(111)

        self._canvas = FigureCanvasTkAgg(self._fig, master=frm_fig)
        self._canvas.get_tk_widget().pack(fill='both', expand=True)

        frm_tb = ttk.Frame(self)
        frm_tb.pack(fill='x', padx=6)
        self._toolbar = NavigationToolbar2Tk(self._canvas, frm_tb)
        self._toolbar.update()

        # 컨트롤 행
        frm_ctrl = ttk.Frame(self)
        frm_ctrl.pack(fill='x', padx=6, pady=4)

        # 모드
        frm_mode = ttk.LabelFrame(frm_ctrl, text=' 모드 ')
        frm_mode.pack(side='left', padx=(0, 10))
        ttk.Radiobutton(frm_mode, text='피크 표시',  variable=self._mode,
                        value='peak').pack(side='left', padx=6, pady=3)
        ttk.Radiobutton(frm_mode, text='텍스트 추가', variable=self._mode,
                        value='text').pack(side='left', padx=6, pady=3)

        # 텍스트 입력 (텍스트 모드에서 사용)
        frm_txt = ttk.LabelFrame(frm_ctrl, text=' 텍스트 내용 (텍스트 모드) ')
        frm_txt.pack(side='left', fill='x', expand=True, padx=(0, 10))
        self._text_entry = ttk.Entry(frm_txt, width=36)
        self._text_entry.pack(padx=4, pady=3, fill='x', expand=True)

        # 피크 목록 표시
        frm_pk = ttk.LabelFrame(frm_ctrl, text=' 피크 (cm⁻¹) ')
        frm_pk.pack(side='left', fill='x', expand=True, padx=(0, 10))
        self._peak_disp = ttk.Entry(frm_pk, width=30, state='readonly')
        self._peak_disp.pack(padx=4, pady=3, fill='x', expand=True)

        # 버튼
        ttk.Button(frm_ctrl, text='주석 전체 삭제', command=self._clear_all,
                   width=12).pack(side='left', padx=4)
        ttk.Button(frm_ctrl, text='저장', command=self._save,
                   width=8).pack(side='right', padx=4)

    # ── 그래프 ────────────────────────────────────────────────────────────────

    def _draw_overlay(self):
        ax = self._ax
        ax.clear()

        lo, hi  = min(self._xlim), max(self._xlim)
        n       = len(self._scan_list)
        ptype   = self._plot_type.get()
        try:
            gap = float(self._e_gap.get())
        except (ValueError, AttributeError):
            gap = self._stack_gap

        self._wn = None   # 재계산

        use_sm = self._ov_smooth.get()
        ov_sw, ov_sp = 11, 3
        if use_sm:
            from park_pifm import smooth_spectrum
            try:
                ov_sw = int(self._ov_sw.get())
                ov_sp = int(self._ov_sp.get())
            except (ValueError, AttributeError):
                pass

        for k, item in enumerate(self._scan_list):
            data  = item['data']
            label = item['label']
            pts   = item.get('point_indices') or []
            idx0  = [i - 1 for i in pts] if pts else list(range(data['n_pts']))
            idx0  = [i for i in idx0 if 0 <= i < data['n_pts']]

            wn   = data['points'][0]['wavenumber']
            mask = (wn >= lo) & (wn <= hi)
            wn_c = wn[mask]
            if self._wn is None:
                self._wn = wn_c

            avg = np.mean([data['points'][i][self._channel][mask] for i in idx0], axis=0)

            if ptype in ('norm', 'stacked') and avg.max() > 0:
                avg = avg / avg.max()
            if ptype == 'stacked':
                avg = avg + k * gap

            color = self._COLORS[k % len(self._COLORS)]
            if use_sm:
                y_sm = smooth_spectrum(avg, ov_sw, ov_sp)
                ax.plot(wn_c, avg,  color=color, linewidth=0.6, alpha=0.3)
                ax.plot(wn_c, y_sm, color=color, linewidth=1.3, label=label)
            else:
                ax.plot(wn_c, avg, color=color, linewidth=1.0, label=label)

        # 축 설정
        if ptype == 'stacked':
            y_top = (n - 1) * gap + 1.3
            ax.set_ylim(-0.2, y_top)
            ax.yaxis.set_ticklabels([])
            ylabel  = 'Intensity (norm., offset)'
            suffix  = 'Stacked 정규화'
        elif ptype == 'norm':
            ylabel  = 'Intensity (norm.)'
            suffix  = '정규화'
        else:
            ylabel  = self._channel
            suffix  = '원본'

        ax.set_xlim(hi, lo)
        ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f'{self._ov_title or "Overlay"}  ({suffix})', fontsize=12, pad=22)
        ax.legend(fontsize=8.5, framealpha=0.9, edgecolor='#bbbbbb')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def _redraw(self):
        """그래프 유형 변경 시 주석을 초기화하고 다시 그림"""
        # 기존 주석 artist 제거
        for vline, txt in self._peak_arts.values():
            try: vline.remove()
            except Exception: pass
            try: txt.remove()
            except Exception: pass
        self._peak_arts.clear()
        self._peak_values.clear()
        self._refresh_peak_disp()
        for _, _, art in self._text_annots:
            try: art.remove()
            except Exception: pass
        self._text_annots.clear()

        self._draw_overlay()
        self._canvas.draw()

    # ── 피크 관리 ─────────────────────────────────────────────────────────────

    def _add_peak(self, wn_val, redraw=True):
        if wn_val in self._peak_values:
            return
        self._peak_values.append(wn_val)

        ax = self._ax
        vline = ax.axvline(wn_val, color='#cc44aa', linestyle='--',
                           alpha=0.8, linewidth=1.0)
        txt = ax.text(
            wn_val, 1.02, f'{wn_val:.0f}',
            transform=ax.get_xaxis_transform(),
            color='#cc44aa', fontsize=8, fontweight='bold',
            va='bottom', ha='center', rotation=90, clip_on=False,
        )
        self._peak_arts[wn_val] = (vline, txt)
        self._refresh_peak_disp()
        if redraw:
            self._canvas.draw()

    def _remove_peak(self, wn_val):
        if wn_val in self._peak_values:
            self._peak_values.remove(wn_val)
        if wn_val in self._peak_arts:
            vline, txt = self._peak_arts.pop(wn_val)
            vline.remove()
            txt.remove()
        self._refresh_peak_disp()
        self._canvas.draw()

    def _refresh_peak_disp(self):
        val = ', '.join(f'{p:.0f}' for p in sorted(self._peak_values, reverse=True))
        self._peak_disp.configure(state='normal')
        self._peak_disp.delete(0, 'end')
        self._peak_disp.insert(0, val)
        self._peak_disp.configure(state='readonly')

    # ── 텍스트 관리 ──────────────────────────────────────────────────────────

    def _add_text(self, x, y):
        text_str = self._text_entry.get().strip()
        if not text_str:
            messagebox.showwarning('텍스트 없음', '텍스트 내용을 입력한 후 클릭하세요.',
                                   parent=self)
            return
        art = self._ax.text(
            x, y, text_str,
            fontsize=9, ha='center', color='#333333',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.75, ec='#aaaaaa'),
        )
        self._text_annots.append((x, y, art))
        self._canvas.draw()

    def _remove_last_text(self):
        if self._text_annots:
            _, _, art = self._text_annots.pop()
            art.remove()
            self._canvas.draw()

    # ── 전체 삭제 + 저장 ──────────────────────────────────────────────────────

    def _clear_all(self):
        for p in list(self._peak_values):
            vline, txt = self._peak_arts.pop(p)
            vline.remove()
            txt.remove()
        self._peak_values.clear()
        self._refresh_peak_disp()
        for _, _, art in self._text_annots:
            art.remove()
        self._text_annots.clear()
        self._canvas.draw()

    def _save(self):
        suffix_map = {'raw': '원본', 'norm': '정규화', 'stacked': 'stacked'}
        type_label   = suffix_map.get(self._plot_type.get(), 'overlay')
        default_name = f'overlay_{self._plot_type.get()}_annotated.png'
        init_dir = self._save_dir if self._save_dir else str(Path.home())
        path = filedialog.asksaveasfilename(
            title=f'오버레이 저장  ({type_label})',
            initialdir=init_dir,
            initialfile=default_name,
            defaultextension='.png',
            filetypes=[('PNG 이미지', '*.png'), ('PDF', '*.pdf'), ('SVG', '*.svg'), ('모든 파일', '*.*')],
            parent=self,
        )
        if not path:
            return
        self._fig.savefig(path, dpi=150, bbox_inches='tight')
        p = Path(path)
        messagebox.showinfo('저장 완료',
                            f'저장 위치:\n{p.parent}\n\n파일명:\n  {p.name}',
                            parent=self)

    # ── 이벤트 ────────────────────────────────────────────────────────────────

    def _on_click(self, event):
        if event.inaxes != self._ax:
            return
        if self._toolbar.mode:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        mode = self._mode.get()

        if mode == 'peak':
            if event.button == 1:
                if self._wn is not None:
                    idx     = int(np.argmin(np.abs(self._wn - x)))
                    wn_snap = float(round(float(self._wn[idx])))
                else:
                    wn_snap = float(round(x))
                self._add_peak(wn_snap)
            elif event.button == 3:
                if self._peak_values:
                    nearest = min(self._peak_values, key=lambda p: abs(p - x))
                    self._remove_peak(nearest)

        elif mode == 'text':
            if event.button == 1:
                self._add_text(x, y)
            elif event.button == 3:
                self._remove_last_text()


# ─────────────────────────────────────────────────────────────────────────────
# 메인 앱
# ─────────────────────────────────────────────────────────────────────────────

class PiFMApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('PiFM Analysis')
        self.geometry('860x680')
        self.minsize(760, 580)

        # 상태
        self._single_data = None
        self._single_coords = None
        self._overlay_items = []   # list of {'data', 'label', 'point_indices', 'path'}

        self._build_ui()

    # ── UI 빌드 ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill='both', expand=True, padx=6, pady=6)

        self._tab1 = ttk.Frame(nb)
        self._tab2 = ttk.Frame(nb)
        nb.add(self._tab1, text='  단일 파일 분석  ')
        nb.add(self._tab2, text='  멀티 파일 Overlay  ')

        self._build_tab_single()
        self._build_tab_overlay()

    # ── 단일 파일 탭 ──────────────────────────────────────────────────────────

    def _build_tab_single(self):
        tab = self._tab1

        # 파일 선택
        f_file = ttk.LabelFrame(tab, text=' 파일 ')
        f_file.pack(fill='x', padx=6, pady=4)

        self._s_path = tk.StringVar()
        ttk.Entry(f_file, textvariable=self._s_path, width=62).grid(
            row=0, column=0, padx=4, pady=4, sticky='ew')
        ttk.Button(f_file, text='Browse', command=self._s_browse, width=8).grid(
            row=0, column=1, padx=3)
        ttk.Button(f_file, text='Load', command=self._s_load, width=7).grid(
            row=0, column=2, padx=3)
        f_file.columnconfigure(0, weight=1)

        # 파일 정보
        self._s_info = scrolledtext.ScrolledText(tab, height=4, state='disabled',
                                                  background='#f8f8f8', relief='sunken')
        self._s_info.pack(fill='x', padx=6, pady=2)

        # 설정
        f_cfg = ttk.LabelFrame(tab, text=' 설정 ')
        f_cfg.pack(fill='x', padx=6, pady=4)

        # Row 0
        ttk.Label(f_cfg, text='채널:').grid(row=0, column=0, padx=4, pady=3, sticky='w')
        self._s_ch = ttk.Combobox(f_cfg, values=['amplitude_uV', 'phase_deg'], width=16, state='readonly')
        self._s_ch.set('amplitude_uV')
        self._s_ch.grid(row=0, column=1, padx=4, pady=3, sticky='w')

        ttk.Label(f_cfg, text='X 범위 (cm⁻¹):').grid(row=0, column=2, padx=4, pady=3, sticky='w')
        self._s_xlo = ttk.Entry(f_cfg, width=7)
        self._s_xlo.insert(0, '700')
        self._s_xlo.grid(row=0, column=3, padx=2, pady=3)
        ttk.Label(f_cfg, text='~').grid(row=0, column=4)
        self._s_xhi = ttk.Entry(f_cfg, width=7)
        self._s_xhi.insert(0, '2400')
        self._s_xhi.grid(row=0, column=5, padx=2, pady=3)

        ttk.Label(f_cfg, text='Stack gap:').grid(row=0, column=6, padx=6, pady=3, sticky='w')
        self._s_gap = ttk.Entry(f_cfg, width=6)
        self._s_gap.insert(0, '1.1')
        self._s_gap.grid(row=0, column=7, padx=2, pady=3)

        # Row 1
        ttk.Label(f_cfg, text='포인트 (쉼표):').grid(row=1, column=0, padx=4, pady=3, sticky='w')
        self._s_pts = ttk.Entry(f_cfg, width=28)
        self._s_pts.insert(0, '1,2,3,4,5')
        self._s_pts.grid(row=1, column=1, columnspan=3, padx=4, pady=3, sticky='ew')
        ttk.Label(f_cfg, text='(빈칸 = 전체)').grid(row=1, column=4, columnspan=2, sticky='w')

        # Row 2
        ttk.Label(f_cfg, text='피크 (쉼표):').grid(row=2, column=0, padx=4, pady=3, sticky='w')
        self._s_peaks = ttk.Entry(f_cfg, width=38)
        self._s_peaks.grid(row=2, column=1, columnspan=5, padx=4, pady=3, sticky='ew')
        ttk.Button(f_cfg, text='그래프 선택/검출', command=self._s_pick_peaks, width=14).grid(
            row=2, column=6, columnspan=2, padx=4)

        # Row 3
        ttk.Label(f_cfg, text='출력 폴더:').grid(row=3, column=0, padx=4, pady=3, sticky='w')
        self._s_outdir = ttk.Entry(f_cfg, width=48)
        self._s_outdir.grid(row=3, column=1, columnspan=6, padx=4, pady=3, sticky='ew')
        ttk.Button(f_cfg, text='Browse', command=self._s_browse_out, width=8).grid(
            row=3, column=7, padx=3)

        # Row 4 — Smoothing
        self._s_smooth = tk.BooleanVar(value=False)
        ttk.Checkbutton(f_cfg, text='Smoothing (SG)', variable=self._s_smooth).grid(
            row=4, column=0, columnspan=2, padx=4, pady=3, sticky='w')
        ttk.Label(f_cfg, text='Window:').grid(row=4, column=2, padx=(8, 2), pady=3, sticky='w')
        self._s_sw = ttk.Entry(f_cfg, width=5)
        self._s_sw.insert(0, '11')
        self._s_sw.grid(row=4, column=3, padx=2, pady=3)
        ttk.Label(f_cfg, text='Poly:').grid(row=4, column=4, padx=(8, 2), pady=3, sticky='w')
        self._s_sp = ttk.Entry(f_cfg, width=4)
        self._s_sp.insert(0, '3')
        self._s_sp.grid(row=4, column=5, padx=2, pady=3)

        f_cfg.columnconfigure(1, weight=1)

        # 액션 버튼
        f_btn = ttk.Frame(tab)
        f_btn.pack(fill='x', padx=6, pady=5)
        for text, cmd, width in [
            ('Topo + 스펙트럼',  self._s_plot_topo,  16),
            ('Topo 저장',        self._s_save_topo,  10),
            ('평균 스펙트럼',     self._s_plot_avg,   16),
            ('평균 저장',         self._s_save_avg,   10),
            ('데이터 저장',       self._s_save,       10),
        ]:
            ttk.Button(f_btn, text=text, command=cmd, width=width).pack(side='left', padx=3)

        # 로그
        ttk.Label(tab, text='로그').pack(anchor='w', padx=8)
        self._s_log = scrolledtext.ScrolledText(tab, height=7, state='disabled', foreground='#333')
        self._s_log.pack(fill='both', expand=True, padx=6, pady=2)

    # ── 단일 파일 콜백 ────────────────────────────────────────────────────────

    def _s_browse(self):
        path = filedialog.askopenfilename(
            title='PiFM TIFF 파일 선택',
            filetypes=[('TIFF', '*.tiff *.tif'), ('All', '*.*')])
        if not path:
            return
        self._s_path.set(path)
        # 자동 출력 경로
        file_num = _auto_file_num(path)
        out = str(Path(path).parent.parent.parent / 'output' / file_num)
        self._s_outdir.delete(0, 'end')
        self._s_outdir.insert(0, out)

    def _s_load(self):
        fp = self._s_path.get().strip()
        if not fp:
            messagebox.showwarning('경고', '파일을 먼저 선택하세요.')
            return
        try:
            self._single_data = read_park_pifm_tiff(fp)
            data = self._single_data

            # info.txt 시도
            info_path = str(Path(fp).with_suffix('.txt'))
            if os.path.exists(info_path):
                self._single_coords = read_info_txt(info_path)
                coord_src = 'info.txt'
            else:
                self._single_coords = data['coords_nm']
                coord_src = 'TIFF 자동 추출'

            info_txt = (
                f"스캔 크기 : {data['scan_x_nm']:.0f} × {data['scan_y_nm']:.0f} nm\n"
                f"Z 범위   : {data['z_nm'].min():.3f} ~ {data['z_nm'].max():.3f} nm\n"
                f"포인트 수 : {data['n_pts']}   |   좌표 출처: {coord_src}\n"
                f"파수 범위 : {data['points'][0]['wavenumber'].min():.1f}"
                f" ~ {data['points'][0]['wavenumber'].max():.1f} cm⁻¹"
            )
            self._s_info.configure(state='normal')
            self._s_info.delete('1.0', 'end')
            self._s_info.insert('end', info_txt)
            self._s_info.configure(state='disabled')

            # 포인트 목록 자동 채움
            self._s_pts.delete(0, 'end')
            self._s_pts.insert(0, ','.join(str(i + 1) for i in range(data['n_pts'])))

            self._s_log_msg(f'[로드] {Path(fp).name} — {data["n_pts"]}개 포인트')
        except Exception as e:
            messagebox.showerror('로드 오류', str(e))

    def _s_browse_out(self):
        d = filedialog.askdirectory(title='출력 폴더 선택')
        if d:
            self._s_outdir.delete(0, 'end')
            self._s_outdir.insert(0, d)

    def _s_check(self):
        if self._single_data is None:
            messagebox.showwarning('경고', '먼저 파일을 Load 하세요.')
            return False
        return True

    def _s_args(self):
        file_num = _auto_file_num(self._s_path.get())
        out_dir  = self._s_outdir.get().strip() or str(Path('../output') / file_num)
        os.makedirs(out_dir, exist_ok=True)
        ch       = self._s_ch.get()
        xlim     = (float(self._s_xlo.get()), float(self._s_xhi.get()))
        pts      = _parse_ints(self._s_pts.get())
        peaks    = _parse_floats(self._s_peaks.get())
        gap      = float(self._s_gap.get())
        return file_num, out_dir, ch, xlim, pts, peaks, gap

    def _smooth_args(self):
        """Smoothing 설정을 dict로 반환 — 플롯 함수에 **로 전달"""
        try:
            sw = int(self._s_sw.get())
            sp = int(self._s_sp.get())
        except ValueError:
            sw, sp = 11, 3
        return dict(smooth=self._s_smooth.get(), smooth_window=sw, smooth_poly=sp)

    def _s_save(self):
        if not self._s_check(): return
        file_num, out_dir, ch, *_ = self._s_args()
        ch_safe = ch.replace('_uV', '')
        default_name = f'result_{file_num}_{ch_safe}'
        init_dir = out_dir if out_dir and os.path.isdir(out_dir) else str(Path.home())
        # 두 파일(xlsx + txt) 이 생성되므로 폴더를 선택
        d = filedialog.askdirectory(
            title=f'데이터 저장 폴더 선택  →  {default_name}.xlsx / .txt',
            initialdir=init_dir,
        )
        if not d:
            return
        try:
            save_spectra(self._single_data, channel=ch,
                         filepath=f'{d}/result_{file_num}')
            xlsx = f'{d}/{default_name}.xlsx'
            txt  = f'{d}/{default_name}.txt'
            self._s_log_msg(f'[저장] 데이터 → {xlsx}')
            messagebox.showinfo('저장 완료',
                                f'저장 위치:\n{d}\n\n파일:\n  {default_name}.xlsx\n  {default_name}.txt')
        except Exception as e:
            messagebox.showerror('저장 오류', str(e))

    def _s_save_topo(self):
        if not self._s_check(): return
        file_num, out_dir, ch, xlim, _, peaks, gap = self._s_args()
        ch_safe = ch.replace('_uV', '')
        default_name = f'result_{file_num}_topo_{ch_safe}.png'
        init_dir = out_dir if out_dir and os.path.isdir(out_dir) else str(Path.home())
        path = filedialog.asksaveasfilename(
            title='Topo + 스펙트럼 저장',
            initialdir=init_dir,
            initialfile=default_name,
            defaultextension='.png',
            filetypes=[('PNG 이미지', '*.png'), ('PDF', '*.pdf'), ('SVG', '*.svg'), ('모든 파일', '*.*')],
        )
        if not path:
            return
        try:
            plot_topo_spectra(
                self._single_data,
                coords_nm  = self._single_coords,
                channel    = ch,
                xlim       = xlim,
                stack_gap  = gap,
                peaks      = peaks,
                title      = f'#{file_num}',
                save_path  = path,
                **self._smooth_args(),
            )
            self._s_log_msg(f'[저장] Topo → {path}')
        except Exception as e:
            messagebox.showerror('저장 오류', str(e))

    def _s_save_avg(self):
        if not self._s_check(): return
        file_num, out_dir, ch, xlim, pts, *_ = self._s_args()
        ch_safe = ch.replace('_uV', '')
        default_name = f'result_{file_num}_avg_{ch_safe}.png'
        init_dir = out_dir if out_dir and os.path.isdir(out_dir) else str(Path.home())
        path = filedialog.asksaveasfilename(
            title='평균 스펙트럼 저장',
            initialdir=init_dir,
            initialfile=default_name,
            defaultextension='.png',
            filetypes=[('PNG 이미지', '*.png'), ('PDF', '*.pdf'), ('SVG', '*.svg'), ('모든 파일', '*.*')],
        )
        if not path:
            return
        try:
            plot_avg_spectrum(
                self._single_data,
                point_indices = pts,
                channel       = ch,
                xlim          = xlim,
                title         = f'#{file_num} — avg',
                save_path     = path,
                filepath      = str(Path(path).parent / f'result_{file_num}_avg'),
                **self._smooth_args(),
            )
            self._s_log_msg(f'[저장] 평균 스펙트럼 → {path}')
        except Exception as e:
            messagebox.showerror('저장 오류', str(e))

    def _s_plot_topo(self):
        if not self._s_check(): return
        file_num, _, ch, xlim, pts, peaks, gap = self._s_args()
        try:
            plot_topo_spectra(
                self._single_data,
                coords_nm  = self._single_coords,
                channel    = ch,
                xlim       = xlim,
                stack_gap  = gap,
                peaks      = peaks,
                title      = f'#{file_num}',
                **self._smooth_args(),
            )
            self._s_log_msg('[그래프] Topo + 스펙트럼 표시 (저장하려면 저장 버튼 사용)')
        except Exception as e:
            messagebox.showerror('그래프 오류', str(e))

    def _s_plot_avg(self):
        if not self._s_check(): return
        file_num, _, ch, xlim, pts, *_ = self._s_args()
        try:
            plot_avg_spectrum(
                self._single_data,
                point_indices = pts,
                channel       = ch,
                xlim          = xlim,
                title         = f'#{file_num} — avg',
                **self._smooth_args(),
            )
            self._s_log_msg('[그래프] 평균 스펙트럼 표시 (저장하려면 저장 버튼 사용)')
        except Exception as e:
            messagebox.showerror('그래프 오류', str(e))

    def _s_pick_peaks(self):
        """그래프에서 마우스로 피크 선택 → 단일 파일 탭 피크 필드 업데이트"""
        if not self._s_check():
            return
        _, _, ch, xlim, pts, *_ = self._s_args()
        data = self._single_data

        # 평균 스펙트럼 계산
        wn      = data['points'][0]['wavenumber']
        lo, hi  = min(xlim), max(xlim)
        mask    = (wn >= lo) & (wn <= hi)
        wn_c    = wn[mask]
        idx0    = [(i - 1) for i in pts] if pts else list(range(data['n_pts']))
        idx0    = [i for i in idx0 if 0 <= i < data['n_pts']]
        avg     = np.mean([data['points'][i][ch][mask] for i in idx0], axis=0)

        current = _parse_floats(self._s_peaks.get()) or []
        dlg = PeakPickerDialog(self, wn_c, avg, current_peaks=current,
                               xlim=xlim, title='피크 선택 — 단일 파일')
        if dlg.result is not None:
            self._s_peaks.delete(0, 'end')
            self._s_peaks.insert(0, ','.join(str(int(p)) for p in dlg.result))
            self._s_log_msg(f'[피크] {[int(p) for p in dlg.result]}')

    def _s_log_msg(self, msg):
        self._s_log.configure(state='normal')
        self._s_log.insert('end', msg + '\n')
        self._s_log.see('end')
        self._s_log.configure(state='disabled')

    # ── Overlay 탭 ────────────────────────────────────────────────────────────

    def _build_tab_overlay(self):
        tab = self._tab2

        # 스캔 목록
        f_list = ttk.LabelFrame(tab, text=' 스캔 목록 ')
        f_list.pack(fill='both', expand=True, padx=6, pady=4)

        cols = ('label', 'file', 'points')
        self._ov_tree = ttk.Treeview(f_list, columns=cols, show='headings', height=7)
        self._ov_tree.heading('label',  text='레이블')
        self._ov_tree.heading('file',   text='파일명')
        self._ov_tree.heading('points', text='포인트')
        self._ov_tree.column('label',  width=140, anchor='w')
        self._ov_tree.column('file',   width=310, anchor='w')
        self._ov_tree.column('points', width=110, anchor='w')

        sb = ttk.Scrollbar(f_list, orient='vertical', command=self._ov_tree.yview)
        self._ov_tree.configure(yscrollcommand=sb.set)
        self._ov_tree.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')

        # 목록 조작 버튼
        f_lbtn = ttk.Frame(tab)
        f_lbtn.pack(fill='x', padx=6, pady=2)
        ttk.Button(f_lbtn, text='파일 추가',  command=self._ov_add,    width=12).pack(side='left', padx=3)
        ttk.Button(f_lbtn, text='선택 삭제',  command=self._ov_remove, width=12).pack(side='left', padx=3)
        ttk.Button(f_lbtn, text='전체 삭제',  command=self._ov_clear,  width=12).pack(side='left', padx=3)

        # 설정
        f_cfg = ttk.LabelFrame(tab, text=' 설정 ')
        f_cfg.pack(fill='x', padx=6, pady=4)

        ttk.Label(f_cfg, text='채널:').grid(row=0, column=0, padx=4, pady=3, sticky='w')
        self._ov_ch = ttk.Combobox(f_cfg, values=['amplitude_uV', 'phase_deg'], width=16, state='readonly')
        self._ov_ch.set('amplitude_uV')
        self._ov_ch.grid(row=0, column=1, padx=4, pady=3, sticky='w')

        ttk.Label(f_cfg, text='X 범위 (cm⁻¹):').grid(row=0, column=2, padx=4, pady=3, sticky='w')
        self._ov_xlo = ttk.Entry(f_cfg, width=7)
        self._ov_xlo.insert(0, '750')
        self._ov_xlo.grid(row=0, column=3, padx=2, pady=3)
        ttk.Label(f_cfg, text='~').grid(row=0, column=4)
        self._ov_xhi = ttk.Entry(f_cfg, width=7)
        self._ov_xhi.insert(0, '1800')
        self._ov_xhi.grid(row=0, column=5, padx=2, pady=3)

        ttk.Label(f_cfg, text='Stack gap:').grid(row=0, column=6, padx=6, pady=3, sticky='w')
        self._ov_gap = ttk.Entry(f_cfg, width=6)
        self._ov_gap.insert(0, '0.5')
        self._ov_gap.grid(row=0, column=7, padx=2, pady=3)

        ttk.Label(f_cfg, text='피크 (쉼표):').grid(row=1, column=0, padx=4, pady=3, sticky='w')
        self._ov_peaks = ttk.Entry(f_cfg, width=38)
        self._ov_peaks.grid(row=1, column=1, columnspan=5, padx=4, pady=3, sticky='ew')
        ttk.Button(f_cfg, text='그래프 선택', command=self._ov_pick_peaks, width=10).grid(
            row=1, column=6, padx=4)

        ttk.Label(f_cfg, text='제목:').grid(row=2, column=0, padx=4, pady=3, sticky='w')
        self._ov_title = ttk.Entry(f_cfg, width=36)
        self._ov_title.grid(row=2, column=1, columnspan=3, padx=4, pady=3, sticky='ew')

        ttk.Label(f_cfg, text='출력 폴더:').grid(row=3, column=0, padx=4, pady=3, sticky='w')
        self._ov_outdir = ttk.Entry(f_cfg, width=48)
        self._ov_outdir.insert(0, str(SCRIPT_DIR.parent / 'output' / 'overlay'))
        self._ov_outdir.grid(row=3, column=1, columnspan=6, padx=4, pady=3, sticky='ew')
        ttk.Button(f_cfg, text='Browse', command=self._ov_browse_out, width=8).grid(
            row=3, column=7, padx=3)

        f_cfg.columnconfigure(1, weight=1)

        # 액션 버튼
        f_btn = ttk.Frame(tab)
        f_btn.pack(fill='x', padx=6, pady=5)
        ttk.Button(f_btn, text='오버레이 열기',
                   command=self._ov_open, width=18).pack(side='left', padx=4)

        # 로그
        ttk.Label(tab, text='로그').pack(anchor='w', padx=8)
        self._ov_log = scrolledtext.ScrolledText(tab, height=5, state='disabled', foreground='#333')
        self._ov_log.pack(fill='both', expand=True, padx=6, pady=2)

    # ── Overlay 콜백 ─────────────────────────────────────────────────────────

    def _ov_add(self):
        paths = filedialog.askopenfilenames(
            title='PiFM TIFF 파일 선택 (복수 가능)',
            filetypes=[('TIFF', '*.tiff *.tif'), ('All', '*.*')])
        for path in paths:
            dlg = AddScanDialog(self, path)
            if dlg.result is None:
                continue
            label, pts = dlg.result
            try:
                data = read_park_pifm_tiff(path)
                self._overlay_items.append(
                    {'data': data, 'label': label, 'point_indices': pts, 'path': path})
                pts_str = ','.join(str(p) for p in pts) if pts else 'all'
                self._ov_tree.insert('', 'end', values=(label, Path(path).name, pts_str))
                self._ov_log_msg(f'[추가] {label}: {Path(path).name}')
            except Exception as e:
                messagebox.showerror('로드 오류', f'{Path(path).name}\n{e}')

    def _ov_remove(self):
        sel = self._ov_tree.selection()
        if not sel:
            return
        idx = self._ov_tree.index(sel[0])
        self._ov_tree.delete(sel[0])
        del self._overlay_items[idx]

    def _ov_clear(self):
        for item in self._ov_tree.get_children():
            self._ov_tree.delete(item)
        self._overlay_items.clear()

    def _ov_scan_list(self):
        return [
            {'data': it['data'], 'label': it['label'], 'point_indices': it['point_indices']}
            for it in self._overlay_items
        ]

    def _ov_browse_out(self):
        d = filedialog.askdirectory(title='출력 폴더 선택')
        if d:
            self._ov_outdir.delete(0, 'end')
            self._ov_outdir.insert(0, d)

    def _ov_common_args(self):
        ch       = self._ov_ch.get()
        xlim     = (float(self._ov_xlo.get()), float(self._ov_xhi.get()))
        peaks    = _parse_floats(self._ov_peaks.get())
        gap      = float(self._ov_gap.get())
        title    = self._ov_title.get().strip() or None
        out_dir  = self._ov_outdir.get().strip() or str(SCRIPT_DIR.parent / 'output' / 'overlay')
        os.makedirs(out_dir, exist_ok=True)
        return ch, xlim, peaks, gap, title, out_dir

    def _ov_check(self):
        if not self._overlay_items:
            messagebox.showwarning('경고', '파일을 먼저 추가하세요.')
            return False
        return True

    def _ov_open(self):
        """오버레이 분석 창 열기 — 창 내에서 유형 선택·주석·저장"""
        if not self._ov_check():
            return
        ch, xlim, peaks, gap, title, out_dir = self._ov_common_args()
        OverlayAnnotatorDialog(
            self,
            scan_list     = self._ov_scan_list(),
            channel       = ch,
            xlim          = xlim,
            title         = title,
            stack_gap     = gap,
            initial_peaks = peaks,
            save_dir      = out_dir,
        )

    def _ov_pick_peaks(self):
        """그래프에서 마우스로 피크 선택 → Overlay 탭 피크 필드 업데이트 (첫 번째 스캔 기준)"""
        if not self._ov_check():
            return
        ch   = self._ov_ch.get()
        xlim = (float(self._ov_xlo.get()), float(self._ov_xhi.get()))

        # 첫 번째 스캔의 정규화된 평균 스펙트럼 사용
        item    = self._overlay_items[0]
        data    = item['data']
        pts     = item['point_indices']
        wn      = data['points'][0]['wavenumber']
        lo, hi  = min(xlim), max(xlim)
        mask    = (wn >= lo) & (wn <= hi)
        wn_c    = wn[mask]
        idx0    = [(i - 1) for i in pts] if pts else list(range(data['n_pts']))
        idx0    = [i for i in idx0 if 0 <= i < data['n_pts']]
        avg     = np.mean([data['points'][i][ch][mask] for i in idx0], axis=0)
        if avg.max() > 0:
            avg = avg / avg.max()   # 정규화

        current = _parse_floats(self._ov_peaks.get()) or []
        dlg = PeakPickerDialog(self, wn_c, avg, current_peaks=current,
                               xlim=xlim, title=f'피크 선택 — {item["label"]}')
        if dlg.result is not None:
            self._ov_peaks.delete(0, 'end')
            self._ov_peaks.insert(0, ','.join(str(int(p)) for p in dlg.result))
            self._ov_log_msg(f'[피크] {[int(p) for p in dlg.result]}')

    def _ov_log_msg(self, msg):
        self._ov_log.configure(state='normal')
        self._ov_log.insert('end', msg + '\n')
        self._ov_log.see('end')
        self._ov_log.configure(state='disabled')


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app = PiFMApp()
    app.mainloop()
