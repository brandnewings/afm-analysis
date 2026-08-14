# PiFM Analysis Scripts

Park Systems SmartScan PiFM (Photo-induced Force Microscopy) 데이터 분석 도구 모음.

---

## 파일 구성

| 파일 | 역할 |
|---|---|
| `park_pifm.py` | PiFM TIFF 파싱 · 시각화 **메인 모듈** |
| `pifm_gui.py` | tkinter **GUI 애플리케이션** |
| `txt-to-omnic.py` | 스펙트럼 txt → OMNIC 호환 형식 **변환 CLI** |
| `analysis_pifm.ipynb` | PiFM 데이터 **분석 워크플로우** 노트북 |
| `txt_to_spc_convert.ipynb` | 스펙트럼 txt → JCAMP-DX / 2열 추출 **변환 노트북** |
| `park_pifm_bk.py` | park_pifm.py 구버전 백업 (smoothing 이전) |
| `park_pifm_bk2.py` | park_pifm.py 구버전 백업 2 (overlay 함수 포함) |

---

## `park_pifm.py` — 메인 파싱 · 시각화 모듈

GUI와 스크립트 모두에서 `import`하는 핵심 모듈.

### 의존성
```
numpy, tifffile, matplotlib, scipy (smoothing에만 필요)
```

### 함수 목록

#### 데이터 입출력

| 함수 | 설명 |
|---|---|
| `read_park_pifm_tiff(filepath)` | PiFM TIFF 파일 파싱. TIFF tag 50435(스캔 파라미터)와 tag 50436(스펙트럼 데이터)를 읽어 딕셔너리 반환 |
| `read_info_txt(filepath)` | 같은 폴더의 `*_info.txt`(스캔 메타데이터) 파싱. scan_size, n_pts 등 반환 |
| `save_spectra(data, channel, filepath)` | 스펙트럼 데이터를 **xlsx + txt** 두 파일로 저장. 파일명에 `_uV` 제외 |

#### 전처리

| 함수 | 설명 |
|---|---|
| `smooth_spectrum(y, window=11, poly=3)` | Savitzky-Golay 필터로 스펙트럼 스무딩. 피크 위치·형태 보존. window는 자동으로 홀수 보정 |

#### 시각화

| 함수 | 주요 파라미터 | 설명 |
|---|---|---|
| `plot_topo_spectra(data, ...)` | `coords_nm`, `channel`, `xlim`, `stack_gap`, `peaks`, `smooth` | **Topography 이미지 + stacked 스펙트럼** 2패널 그래프. 포인트 위치 표시, 피크 어노테이션 포함 |
| `plot_avg_spectrum(data, ...)` | `point_indices`, `channel`, `xlim`, `normalize`, `smooth` | 선택 포인트들의 **평균 스펙트럼** 단일 그래프. xlsx/txt 데이터 동시 저장 가능 |
| `plot_spectrum(data, point_index, ...)` | `channel`, `xlim`, `smooth` | 단일 포인트 스펙트럼 그래프 |
| `detect_peaks(data, ...)` | `point_indices`, `height`, `prominence`, `distance` | scipy `find_peaks`로 피크 자동 검출 후 시각화. 검출된 파수 리스트 반환 |
| `plot_overlay_spectra(scan_list, ...)` | `channel`, `xlim`, `peaks`, `smooth` | 여러 스캔의 원본 스펙트럼 **오버레이** |
| `plot_overlay_normalized(scan_list, ...)` | `channel`, `xlim`, `peaks`, `smooth` | 각 스캔을 max 정규화 후 **오버레이** |
| `plot_overlay_normalized_stacked(scan_list, ...)` | `channel`, `xlim`, `stack_gap`, `peaks`, `smooth` | 정규화 + offset 적용 **stacked 오버레이** |

#### `read_park_pifm_tiff` 반환 구조
```python
{
    'n_pts':  int,          # 스펙트럼 포인트 수
    'scan_x': float,        # 스캔 영역 X (nm)
    'scan_y': float,        # 스캔 영역 Y (nm)
    'points': [             # 포인트별 데이터 리스트
        {
            'x': float,                      # 포인트 X 좌표 (nm)
            'y': float,                      # 포인트 Y 좌표 (nm)
            'wavenumber': np.ndarray,        # 파수 배열 (cm⁻¹)
            'amplitude_uV': np.ndarray,      # PiFM 진폭 (μV)
            'phase_deg': np.ndarray,         # PiFM 위상 (deg)
            'z_nm': np.ndarray,              # Topography (nm)
        },
        ...
    ]
}
```

---

## `pifm_gui.py` — GUI 애플리케이션

```
python pifm_gui.py
```

`park_pifm.py`의 기능을 시각적으로 조작하는 tkinter 기반 GUI.

### 클래스 구조

| 클래스 | 설명 |
|---|---|
| `PiFMApp` | 메인 윈도우. **Single File** 탭과 **Overlay** 탭으로 구성 |
| `PeakPickerDialog` | 평균 스펙트럼 위에서 피크를 **자동 검출 + 수동 클릭 선택**하는 대화창 |
| `OverlayAnnotatorDialog` | 오버레이 그래프에 **피크 마커·텍스트 주석**을 추가하고 저장하는 대화창 |
| `AddScanDialog` | Overlay 탭에서 스캔 파일과 사용할 포인트를 지정하는 대화창 |

### Single File 탭 주요 기능
- TIFF 파일 로드 및 메타데이터 확인
- 채널 선택 (Amplitude / Phase)
- 파수 범위(xlim), Stack gap, 포인트 선택
- **Topo + 스펙트럼** 그래프 표시 / PNG·PDF·SVG 저장
- **평균 스펙트럼** 그래프 표시 / 저장 (이미지 + xlsx/txt)
- **데이터 저장** (xlsx + txt)
- **Savitzky-Golay 스무딩** 옵션 (Window / Poly 설정)
- **피크 선택** 대화창 (자동 검출 + 그래프 클릭 수동 선택)

### Overlay 탭 주요 기능
- 복수 TIFF 파일 추가 (포인트 지정 가능)
- **오버레이 열기**: 원본 / 정규화 / Stacked 중 선택하여 주석 편집
- 피크 마커(좌클릭 추가, 우클릭 제거) + 텍스트 주석 추가
- Savitzky-Golay 스무딩 옵션 내장
- PNG·PDF·SVG 저장 (파일 이름 대화창에서 직접 지정)

### `PeakPickerDialog` 사용법
1. **자동 검출** 파라미터 입력 (Height, Prominence, 최소 간격, Smooth)
2. **자동 검출 (초기화)** 또는 **자동 검출 (추가)** 클릭
3. 그래프를 **왼쪽 클릭**으로 피크 수동 추가, **오른쪽 클릭**으로 피크 제거
4. 목록에서 선택 후 **삭제** 또는 **전체 삭제**
5. **확인** 클릭 시 선택된 파수 리스트 반환

---

## `txt-to-omnic.py` — 형식 변환 CLI

Origin 내보내기 txt 파일을 OMNIC에서 열 수 있는 형식으로 변환.

```
python txt-to-omnic.py result.txt              # SPC 형식 (기본)
python txt-to-omnic.py result.txt --fmt jdx    # JCAMP-DX 형식
python txt-to-omnic.py *.txt --fmt jdx         # 일괄 변환
python txt-to-omnic.py result.txt -o output    # 출력 파일명 지정
```

### 지원 출력 형식

| 형식 | 확장자 | 설명 |
|---|---|---|
| SPC | `.spc` | Galactic SPC — OMNIC 직접 지원. 균등 간격 X 모드 (TXVALS 없음) |
| JCAMP-DX | `.dx` | 공개 표준 텍스트 — OMNIC, Origin, SciAps 등 폭넓게 지원. SPC 로딩 실패 시 사용 |

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--fmt` | `spc` | 출력 형식 (`spc` / `jdx`) |
| `--xtype` | `1` | X축 타입 (1=Wavenumber cm⁻¹, 13=Raman Shift) |
| `--ytype` | `0` | Y축 타입 (0=Arbitrary, 2=Absorbance, 1=Transmittance) |
| `--exper` | `4` | 실험 타입 (4=FT-IR/Raman, 11=Raman, 0=General) |

### 입력 txt 형식 (Origin 스타일)
```
Wavenumber    Amplitude    ...    # 1행: 컬럼 이름
cm-1          uV           ...    # 2행: 단위
800.0         1234.5       ...    # 3행~: 데이터 (탭 또는 공백 구분)
...
```
다중 Y 컬럼이 있으면 컬럼별로 별도 파일 생성 (`_01`, `_02`, ...).

---

## 백업 파일

| 파일 | 내용 |
|---|---|
| `park_pifm_bk.py` | smoothing 함수 추가 이전 초기 버전. `read_park_pifm_tiff`, `read_info_txt`, `save_spectra`, `plot_topo_spectra`만 포함 |
| `park_pifm_bk2.py` | overlay 함수군(`plot_overlay_*`)까지 추가된 중간 버전. Savitzky-Golay smoothing 파라미터 미포함 |

---

## 환경 설정

### Conda 환경 (`afm`)
```
conda activate afm
pip install tifffile numpy matplotlib scipy xlsxwriter
```

### 한글 폰트
Windows 기본 내장 **맑은 고딕(Malgun Gothic)** 사용.
다른 OS에서 사용 시 `park_pifm.py`와 `pifm_gui.py` 상단의 `font.family` 값 변경:
```python
matplotlib.rcParams['font.family'] = 'AppleGothic'   # macOS
matplotlib.rcParams['font.family'] = 'NanumGothic'   # Linux
```

---

---

## `analysis_pifm.ipynb` — PiFM 분석 워크플로우 노트북

`park_pifm.py`의 모든 기능을 단계별로 실행하는 인터랙티브 분석 노트북.
GUI 없이 파라미터를 세밀하게 조정하거나 커스텀 분석이 필요할 때 사용.

### 셀 구성 및 실행 순서

| 단계 | 내용 |
|---|---|
| **1. 환경 설정** | `%autoreload 2` + `park_pifm` 함수 임포트 |
| **2. 파일 로드** | TIFF 경로·파일번호 설정, 출력 폴더(`../output/{file_num}`) 자동 생성 |
| **3. 데이터 파싱** | `read_park_pifm_tiff()` → 스캔 크기·Z 범위·포인트 수 확인 |
| **4. 좌표 확인** | `read_info_txt()` 우선, 없으면 TIFF 내 좌표 사용. 각 포인트 X/Y(nm) 출력 |
| **5. 데이터 저장** | `save_spectra()` → xlsx + txt 저장 |
| **6. 피크 검출** | `detect_peaks()` → 자동 검출 결과 확인 후 수동 수정 가능 |
| **7. 단일 포인트** | `plt.plot()` 으로 개별 포인트 스펙트럼 빠른 확인 (`%matplotlib tk`) |
| **8. Topo + 스펙트럼** | `plot_topo_spectra()` → 피크·타이틀·저장 경로 설정하여 PNG 생성 |
| **9. 평균 스펙트럼** | `plot_avg_spectrum()` → 사용할 포인트 번호(1-based) 지정 |
| **10. Overlay** | 여러 TIFF 로드 후 `scan_list` 구성 → 원본·정규화·Stacked 비교 |

### Overlay `scan_list` 구조
```python
scan_list = [
    {'data': data_A, 'label': 'NF',               'point_indices': [6, 7, 8]},
    {'data': data_B, 'label': 'TF',               'point_indices': [1, 2, 3, 4, 5]},
    {'data': data_C, 'label': 'Pristine particle', 'point_indices': [1, 2, 3, 4, 5]},
    ...
]
```

### 주요 팁
- `peaks` 리스트는 `detect_peaks()` 결과를 그대로 사용하거나, 직접 `[1100, 1263, 1437]` 형태로 지정
- `save_path` / `filepath` 인자를 주석 처리하면 저장 없이 미리보기 가능
- `%matplotlib tk` 사용 시 인터랙티브 창에서 줌·팬 가능

---

## `txt_to_spc_convert.ipynb` — 스펙트럼 형식 변환 노트북

Origin 내보내기 txt 파일을 OMNIC 및 기타 소프트웨어에서 읽을 수 있는 형식으로 변환.
CLI 도구(`txt-to-omnic.py`)의 전 단계 작업(컬럼 추출, 포맷 확인)까지 포함.

### 파트 1 — 다중 컬럼 txt → 2열 txt 추출

SmartScan 내보내기 파일은 Index·Amplitude·Phase·Wavenumber 등 여러 컬럼이 혼합되어 있음.
변환 도구에 넣기 전에 원하는 X(파수)·Y(신호) 두 컬럼만 추출.

| 셀 | 내용 |
|---|---|
| **미리보기** | `pd.read_csv()`로 컬럼 구조·헤더 행 수 확인 |
| **단일 파일 변환** | `x_col`, `y_col` 번호 지정 → `_2col.txt` 저장 |
| **일괄 변환** | 폴더 내 모든 `.txt` 파일에 동일 컬럼 설정 적용 |

```
입력 txt 행 구조 (SmartScan 내보내기):
  Row 0: "point XX"          ← 제목행 (skip)
  Row 1: Index  Amplitude  Phase  Wavenumber ...  ← 헤더
  Row 2: (단위행)             ← skip
  Row 3+: 데이터
→ skip_rows = [0, 2], x_col = 3 (Wavenumber), y_col = 1 (Amplitude)
```

### 파트 2 — SpectroChemPy를 이용한 JCAMP-DX 변환

`spectrochempy` 라이브러리로 txt → `.jdx` (JCAMP-DX 4.24) 변환.
OMNIC에서 SPC가 열리지 않을 때 사용하는 대안.

| 셀 | 내용 |
|---|---|
| **설치 확인** | `import spectrochempy as scp` + 버전 출력 |
| **파일 목록** | `Path(folder).glob("*.txt")` 로 대상 파일 일람 |
| **단일 변환** | `scp.NDDataset` 생성 → `write_jcamp()` → `.jdx` |
| **일괄 변환** | 폴더 내 전체 txt 파일 → 개별 `.jdx` 생성 |
| **변환 확인** | `scp.read_jcamp()` 로 재로드 + `ds.plot()` 시각 확인 |

```
의존성: pip install spectrochempy
```

> **CLI 도구와의 차이**
> `txt-to-omnic.py`는 별도 의존성 없이 SPC/JCAMP-DX 모두 지원.
> 이 노트북의 SpectroChemPy 방식은 JCAMP-DX만 지원하나 라이브러리가 포맷 세부 사항을 보장.

---

## 빠른 시작 (스크립트)

```python
from park_pifm import read_park_pifm_tiff, read_info_txt, plot_topo_spectra

data = read_park_pifm_tiff('HS100MG_260514_PiFM Amplitude_Forward_065.tiff')
info = read_info_txt('HS100MG_260514_PiFM Amplitude_Forward_065.txt')  # optional

plot_topo_spectra(
    data,
    channel   = 'amplitude_uV',
    xlim      = (900, 1800),
    stack_gap = 0.6,
    peaks     = [1168, 1436, 1742],   # cm⁻¹
    smooth    = True,
)
```

## 빠른 시작 (GUI)

```
conda activate afm
python pifm_gui.py
```
