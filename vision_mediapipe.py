from __future__ import annotations
import logging
import urllib.request
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import config
_tasks_ok: Optional[bool] = None
_import_error: Optional[Exception] = None
HAND_TASK_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'
FACE_TASK_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'

def _models_dir() -> Path:
    d = config.MODELS_DIR / 'mediapipe'
    d.mkdir(parents=True, exist_ok=True)
    return d

def _ensure_task_file(name: str, url: str) -> Path:
    dest = _models_dir() / name
    if dest.is_file() and dest.stat().st_size > 1000:
        return dest
    logging.info('Downloading MediaPipe model %s …', name)
    urllib.request.urlretrieve(url, dest)
    return dest

def _tasks_import_ok() -> bool:
    global _tasks_ok, _import_error
    if _tasks_ok is not None:
        return _tasks_ok
    try:
        from mediapipe.tasks.python.core import base_options
        from mediapipe.tasks.python.vision import HandLandmarker
        _tasks_ok = True
        return True
    except Exception as e:
        _import_error = e
        _tasks_ok = False
        logging.info('MediaPipe Tasks API not available: %s', e)
        return False

def mediapipe_available() -> bool:
    try:
        import mediapipe
    except Exception as e:
        global _import_error
        _import_error = e
        return False
    return _tasks_import_ok()

def last_import_error() -> Optional[Exception]:
    return _import_error

def _bbox_from_norm_landmarks(landmarks: list, w: int, h: int, pad: float=0.02) -> tuple[int, int, int, int]:
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x1 = max(0, int((min(xs) - pad) * w))
    y1 = max(0, int((min(ys) - pad) * h))
    x2 = min(w - 1, int((max(xs) + pad) * w))
    y2 = min(h - 1, int((max(ys) + pad) * h))
    return (x1, y1, x2, y2)

class MediaPipeDrawer:

    def __init__(self, use_hands: bool, use_face: bool) -> None:
        self.use_hands = bool(use_hands and mediapipe_available())
        self.use_face = bool(use_face and mediapipe_available())
        self._hand_lm = None
        self._face_lm = None
        self._init_error: Optional[str] = None
        self.last_primary_wrist_norm: Optional[tuple[float, float]] = None
        if not (self.use_hands or self.use_face):
            return
        if not _tasks_import_ok():
            self.use_hands = self.use_face = False
            return
        try:
            from mediapipe.tasks.python.core import base_options as bo
            from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, HandLandmarker, HandLandmarkerOptions
            from mediapipe.tasks.python.vision.core import vision_task_running_mode as vrm
            rm = vrm.VisionTaskRunningMode.IMAGE
            if self.use_hands:
                path = _ensure_task_file('hand_landmarker.task', HAND_TASK_URL)
                opts = HandLandmarkerOptions(base_options=bo.BaseOptions(model_asset_path=str(path)), running_mode=rm, num_hands=2, min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5)
                self._hand_lm = HandLandmarker.create_from_options(opts)
            if self.use_face:
                path = _ensure_task_file('face_landmarker.task', FACE_TASK_URL)
                opts = FaceLandmarkerOptions(base_options=bo.BaseOptions(model_asset_path=str(path)), running_mode=rm, num_faces=2, min_face_detection_confidence=0.5, min_face_presence_confidence=0.5, min_tracking_confidence=0.5, output_face_blendshapes=False, output_facial_transformation_matrixes=False)
                self._face_lm = FaceLandmarker.create_from_options(opts)
        except Exception as e:
            logging.exception('MediaPipe Tasks init')
            self._init_error = str(e)
            self.use_hands = self.use_face = False
            self._hand_lm = None
            self._face_lm = None

    def annotate(self, bgr: np.ndarray) -> np.ndarray:
        self.last_primary_wrist_norm = None
        if self._init_error or (self._hand_lm is None and self._face_lm is None):
            return bgr
        out = bgr.copy()
        h, w = out.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        from mediapipe.tasks.python.vision.core import image as mp_image_mod
        mp_img = mp_image_mod.Image(image_format=mp_image_mod.ImageFormat.SRGB, data=rgb)
        if self._face_lm is not None:
            try:
                res = self._face_lm.detect(mp_img)
                for face_lms in res.face_landmarks:
                    x1, y1, x2, y2 = _bbox_from_norm_landmarks(face_lms, w, h)
                    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(out, 'face (MP)', (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            except Exception:
                logging.exception('FaceLandmarker.detect')
        if self._hand_lm is not None:
            try:
                res = self._hand_lm.detect(mp_img)
                if res.hand_landmarks:
                    wrist = res.hand_landmarks[0][0]
                    self.last_primary_wrist_norm = (wrist.x, wrist.y)
                for hand_lms in res.hand_landmarks:
                    x1, y1, x2, y2 = _bbox_from_norm_landmarks(hand_lms, w, h, pad=0.03)
                    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(out, 'hand', (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)
            except Exception:
                logging.exception('HandLandmarker.detect')
        return out

    def close(self) -> None:
        if self._hand_lm is not None:
            try:
                self._hand_lm.close()
            except Exception:
                logging.exception('HandLandmarker.close')
            self._hand_lm = None
        if self._face_lm is not None:
            try:
                self._face_lm.close()
            except Exception:
                logging.exception('FaceLandmarker.close')
            self._face_lm = None
