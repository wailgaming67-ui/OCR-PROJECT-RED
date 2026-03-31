from __future__ import annotations
import sys
from pathlib import Path

def _project_root() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent
PROJECT_ROOT = _project_root()
MODELS_DIR = PROJECT_ROOT / 'models'
EAST_MODEL_PATH = MODELS_DIR / 'frozen_east_text_detection.pb'
EAST_WIDTH = 320
EAST_HEIGHT = 320
EAST_CONFIDENCE = 0.5
EAST_NMS_THRESHOLD = 0.4
TEXT_MIN_AREA = 200
TEXT_MIN_ASPECT = 1.2
TEXT_MAX_ASPECT = 15.0
