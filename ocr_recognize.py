from __future__ import annotations
import os
import shutil
import sys
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import pytesseract
from pytesseract import TesseractNotFoundError
_easyocr_reader = None

def find_tesseract_executable() -> Optional[str]:
    env = (os.environ.get('TESSERACT_CMD') or '').strip()
    if env:
        p = Path(env)
        if p.is_file():
            return str(p.resolve())
    w = shutil.which('tesseract')
    if w:
        return w
    if sys.platform == 'win32':
        candidates = [Path('C:\\Program Files\\Tesseract-OCR\\tesseract.exe'), Path('C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'), Path(os.environ.get('LOCALAPPDATA', '')) / 'Programs' / 'Tesseract-OCR' / 'tesseract.exe', Path(os.environ.get('ProgramFiles', '')) / 'Tesseract-OCR' / 'tesseract.exe']
        for p in candidates:
            try:
                if p.is_file():
                    return str(p.resolve())
            except OSError:
                continue
    return None

def sanitize_ocr_output(text: str) -> str:
    if not text:
        return ''
    if text.startswith('[Tesseract not found') or text.startswith('(OCR error'):
        return text
    lines_out: list[str] = []
    for line in str(text).splitlines():
        s = line.strip()
        low = s.lower()
        if low in ('none', 'null', 'nan', 'undefined', 'n/a'):
            continue
        if s in ("'None'", '"None"'):
            continue
        lines_out.append(line)
    return '\n'.join(lines_out).strip()

def set_tesseract_cmd(path: str | None) -> None:
    if path and str(path).strip():
        pytesseract.pytesseract.tesseract_cmd = str(Path(path.strip()))
    else:
        found = find_tesseract_executable()
        if found:
            pytesseract.pytesseract.tesseract_cmd = found
        else:
            pytesseract.pytesseract.tesseract_cmd = 'tesseract'

def tesseract_config(psm: int=6) -> str:
    return f'--oem 3 --psm {psm}'

def recognize_tesseract(gray_or_binary: np.ndarray, lang: str='eng') -> str:
    try:
        raw = pytesseract.image_to_string(gray_or_binary, lang=lang, config=tesseract_config(6)).strip()
        return sanitize_ocr_output(raw)
    except TesseractNotFoundError:
        return '[Tesseract not found: install from https://github.com/UB-Mannheim/tesseract/wiki and add to PATH, or set TESSERACT_CMD in ocr_recognize.py]'

def recognize_tesseract_data(gray_or_binary: np.ndarray, lang: str='eng') -> dict:
    return pytesseract.image_to_data(gray_or_binary, lang=lang, config=tesseract_config(6), output_type=pytesseract.Output.DICT)

def get_easyocr_reader(langs: list[str] | None=None):
    global _easyocr_reader
    if _easyocr_reader is not None:
        return _easyocr_reader
    try:
        import easyocr
    except ImportError:
        return None
    langs = langs or ['en']
    _easyocr_reader = easyocr.Reader(langs, gpu=False, verbose=False)
    return _easyocr_reader

def recognize_easyocr(bgr: np.ndarray, reader) -> str:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    lines = reader.readtext(rgb, detail=0, paragraph=True)
    if isinstance(lines, str):
        return sanitize_ocr_output(lines.strip())
    if isinstance(lines, list):
        parts: list[str] = []
        for x in lines:
            if x is None:
                continue
            s = str(x).strip()
            if not s or s.lower() in ('none', 'null', 'nan'):
                continue
            parts.append(s)
        return sanitize_ocr_output('\n'.join(parts))
    s = str(lines).strip()
    return sanitize_ocr_output(s) if s.lower() not in ('none', 'null') else ''

def ensure_tesseract_cmd() -> Optional[str]:
    return os.environ.get('TESSERACT_CMD')

def is_tesseract_working() -> bool:
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False

def tesseract_install_hint() -> str:
    return 'Install Tesseract OCR for Windows:\nhttps://github.com/UB-Mannheim/tesseract/wiki\n\nThen either add the install folder to your system PATH, or in this app open Settings and set the full path to tesseract.exe (e.g. C:\\Program Files\\Tesseract-OCR\\tesseract.exe).'

def guess_language_from_text(text: str) -> str | None:
    t = (text or '').strip()
    if len(t) < 12:
        return None
    try:
        from langdetect import detect
        try:
            code = detect(t)
        except Exception:
            return None
        names = {'en': 'English', 'ar': 'Arabic', 'fr': 'French', 'de': 'German', 'es': 'Spanish', 'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'zh-cn': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean'}
        return names.get(code.lower(), code)
    except ImportError:
        return None
    except Exception:
        return None
