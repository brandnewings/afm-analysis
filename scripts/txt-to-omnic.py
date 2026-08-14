#!/usr/bin/env python3
"""
txt-to-omnic.py  —  Origin-style txt → OMNIC 호환 형식 변환기

OMNIC에서 열 수 없는 경우 두 가지 방법을 제공합니다.

방법 1: SPC (균등 간격 X, TXVALS 플래그 없음)
    OMNIC이 직접 지원하는 Galactic SPC 표준 모드.
    이전 버전(txt-to-spc.py)의 TXVALS 플래그 제거 버전.

방법 2: JCAMP-DX (.dx)
    OMNIC 공식 지원 공개 표준 텍스트 포맷.
    SPC로도 안 열릴 경우 이 방법 사용.

사용법:
    python txt-to-omnic.py result.txt             # SPC (균등 간격)
    python txt-to-omnic.py result.txt --fmt jdx   # JCAMP-DX
    python txt-to-omnic.py result.txt -o output   # 출력 기본 이름 지정
    python txt-to-omnic.py *.txt --fmt jdx        # 일괄 변환
"""

import struct
import numpy as np
import argparse
import sys
from pathlib import Path

_HDR_FMT = '<BBBBIddIBBBBI9s9sH8f130s30sIIBBHf48sfIfB187s'
_SUB_FMT = '<BBHfffIIf4s'


def read_txt(path):
    """2-줄 헤더 + 다중 컬럼 탭 구분 파일 읽기"""
    for enc in ('utf-8', 'cp949', 'euc-kr', 'latin-1'):
        try:
            text = Path(path).read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise IOError(f'인코딩 인식 실패: {path}')

    lines     = text.splitlines()
    col_names = lines[0].strip().split('\t') if lines else []
    col_units = lines[1].strip().split('\t') if len(lines) > 1 else []

    rows = []
    for line in lines[2:]:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue

    if not rows:
        raise ValueError(f'유효한 데이터가 없습니다: {path}')

    arr = np.array(rows, dtype=np.float32)
    x   = arr[:, 0]
    ys  = [arr[:, c] for c in range(1, arr.shape[1])]

    if not ys:
        raise ValueError(f'Y 컬럼이 없습니다: {path}')

    return x, ys, col_names, col_units


# ════════════════════════════════════════════════════════════════════════════
# 방법 1: SPC (균등 간격 X, ftflgs=0x00)
# ════════════════════════════════════════════════════════════════════════════

def _make_spc_header(n, first, last, xtype, ytype, exper, comment_bytes):
    """512-byte SPC 헤더 — TXVALS 없음 (균등 간격 X)"""
    hdr = struct.pack(
        _HDR_FMT,
        0x00,            # ftflgs : 0 = 균등 간격 X (TXVALS 없음)
        0x4D,            # fversn : 신형식
        exper,           # fexper
        0x80,            # fexp   : Y = float32
        n,               # fnpts
        float(first),    # ffirst
        float(last),     # flast
        1,               # fnsub
        xtype,           # fxtype
        ytype,           # fytype
        0, 0,            # fztype, fpost
        0,               # fdate
        b'\x00' * 9,    # fres
        b'\x00' * 9,    # fsource
        0,               # fpeakpt
        *(0.0,) * 8,    # fspare
        comment_bytes,   # fcmnt (130 bytes)
        b'\x00' * 30,   # fcatxt
        0, 0,            # flogoff, fmods
        0, 0, 0,         # fprocs, flevel, fsampin
        0.0,             # ffactor
        b'\x00' * 48,   # fmethod
        0.0,             # fzinc
        0,               # fwplanes
        0.0,             # fwinc
        0,               # fwtype
        b'\x00' * 187,  # freserv
    )
    assert len(hdr) == 512
    return hdr


def _make_sub_header(n):
    """32-byte 서브파일 헤더"""
    sub = struct.pack(
        _SUB_FMT,
        0, 0x80, 0,
        0.0, 0.0, 0.0,
        n, 0,
        1.0,
        b'\x00' * 4,
    )
    assert len(sub) == 32
    return sub


def write_spc(x, y, out_path, xtype=1, ytype=0, exper=4, comment=''):
    """
    균등 간격 X SPC 파일 작성 (TXVALS 없음)

    바이너리 레이아웃:
        [512 B] 메인 헤더 (ffirst/flast/fnpts 로 X 정의)
        [ 32 B] 서브파일 헤더
        [n×4 B] Y float32 배열
    """
    n    = len(x)
    cmnt = (comment.encode('latin-1', errors='replace') + b'\x00')[:130]
    cmnt = cmnt.ljust(130, b'\x00')

    with open(out_path, 'wb') as f:
        f.write(_make_spc_header(n, x[0], x[-1], xtype, ytype, exper, cmnt))
        f.write(_make_sub_header(n))
        f.write(y.astype(np.float32).tobytes())


# ════════════════════════════════════════════════════════════════════════════
# 방법 2: JCAMP-DX (.dx)  — OMNIC 공식 지원 공개 표준
# ════════════════════════════════════════════════════════════════════════════

def write_jdx(x, y, out_path, title='', x_unit='1/CM', y_unit='ABSORBANCE'):
    """
    JCAMP-DX 4.24 형식 텍스트 파일 작성

    OMNIC, SciAps, Origin 등에서 열 수 있는 공개 표준.
    Y 데이터는 X++(Y..Y) 압축 없이 (X Y) 쌍으로 저장.
    """
    n      = len(x)
    x_min  = float(x.min())
    x_max  = float(x.max())
    y_min  = float(y.min())
    y_max  = float(y.max())

    lines = [
        f'##TITLE= {title}',
        '##JCAMP-DX= 4.24',
        '##DATA TYPE= INFRARED SPECTRUM',
        '##DATA CLASS= XYDATA',
        '##ORIGIN= ',
        '##OWNER= PUBLIC DOMAIN',
        f'##XUNITS= {x_unit}',
        f'##YUNITS= {y_unit}',
        f'##XFACTOR= 1.0',
        f'##YFACTOR= 1.0',
        f'##FIRSTX= {x_min:.6g}',
        f'##LASTX= {x_max:.6g}',
        f'##MAXY= {y_max:.6g}',
        f'##MINY= {y_min:.6g}',
        f'##NPOINTS= {n}',
        '##XYDATA= (XY..XY)',
    ]

    # 데이터 행: 한 행에 (X Y) 쌍 4개씩
    pairs_per_line = 4
    data_lines = []
    for i in range(0, n, pairs_per_line):
        chunk = [f'{x[j]:.4f} {y[j]:.6g}' for j in range(i, min(i + pairs_per_line, n))]
        data_lines.append('  '.join(chunk))

    lines += data_lines
    lines.append('##END=')

    Path(out_path).write_text('\n'.join(lines) + '\n', encoding='ascii', errors='replace')


# ════════════════════════════════════════════════════════════════════════════
# 공통 변환 함수
# ════════════════════════════════════════════════════════════════════════════

def convert(in_path, out_base=None, fmt='spc', xtype=1, ytype=0, exper=4):
    """
    txt → SPC 또는 JCAMP-DX 변환 (다중 Y 컬럼 지원)

    Parameters
    ----------
    fmt : 'spc' 또는 'jdx'
    """
    in_path = Path(in_path)
    x, ys, names, units = read_txt(in_path)
    n_y = len(ys)

    ext  = '.spc' if fmt == 'spc' else '.dx'
    base = Path(out_base) if out_base else in_path.with_suffix('')

    if n_y == 1:
        out_paths = [base.with_suffix(ext)]
    else:
        digits    = len(str(n_y))
        out_paths = [Path(str(base) + f'_{i+1:0{digits}d}{ext}')
                     for i in range(n_y)]

    x_name = names[0] if names else 'X'
    x_unit = units[0] if units else 'cm-1'

    created = []
    for i, (y, out_path) in enumerate(zip(ys, out_paths)):
        y_name = names[i + 1] if i + 1 < len(names) else f'Col{i+1}'
        y_unit = units[i + 1] if i + 1 < len(units) else ''
        comment = f'{x_name} [{x_unit}]  |  {y_name} [{y_unit}]'

        out_path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == 'spc':
            write_spc(x, y, out_path,
                      xtype=xtype, ytype=ytype, exper=exper, comment=comment)
        else:
            write_jdx(x, y, out_path,
                      title=comment, x_unit='1/CM', y_unit=y_unit or 'ARBITRARY')

        tag = f'[{y_name}]' if n_y > 1 else ''
        print(f'{in_path.name} {tag}  ->  {out_path.name}'
              f'  ({len(x)} pts, X: {x[0]:.1f}~{x[-1]:.1f})')
        created.append(out_path)

    return created


def main():
    ap = argparse.ArgumentParser(
        description='Origin txt → OMNIC 호환 SPC / JCAMP-DX 변환',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\n'.join([
            '형식 선택:',
            '  spc : Galactic SPC 균등 간격 (기본)',
            '  jdx : JCAMP-DX 4.24 텍스트 (가장 호환성 높음)',
            '',
            'xtype 코드: 1=Wavenumber(cm-1)  13=Raman Shift',
            'ytype 코드: 0=Arbitrary  2=Absorbance  1=Transmittance',
            'exper 코드: 4=FT-IR/Raman  11=Raman  0=General',
        ])
    )
    ap.add_argument('files', nargs='+', metavar='FILE')
    ap.add_argument('-o', '--output', default=None, metavar='BASENAME',
                    help='출력 기본 파일명 (단일 입력 시)')
    ap.add_argument('--fmt', choices=['spc', 'jdx'], default='spc',
                    help='출력 형식: spc (기본) 또는 jdx (JCAMP-DX)')
    ap.add_argument('--xtype', type=int, default=1)
    ap.add_argument('--ytype', type=int, default=0)
    ap.add_argument('--exper', type=int, default=4)
    args = ap.parse_args()

    if len(args.files) > 1 and args.output:
        print('경고: 여러 파일 변환 시 -o 옵션은 무시됩니다.', file=sys.stderr)

    ok, fail = 0, 0
    for f in args.files:
        try:
            out_base = args.output if len(args.files) == 1 else None
            created  = convert(f, out_base=out_base, fmt=args.fmt,
                               xtype=args.xtype, ytype=args.ytype,
                               exper=args.exper)
            ok += len(created)
        except Exception as e:
            print(f'오류 [{Path(f).name}]: {e}', file=sys.stderr)
            fail += 1

    if len(args.files) > 1 or fail:
        print(f'\n완료: {ok}개 생성  /  {fail}개 실패')


if __name__ == '__main__':
    main()
