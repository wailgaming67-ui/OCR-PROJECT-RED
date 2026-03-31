from __future__ import annotations
import contextlib
import logging
import queue
import socket
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import config
YOLO_WORLD_WEIGHT = 'yolov8x-worldv2.pt'
YOLO_WORLD_IMGSZ_QUALITY = 1024
LIVE_INFER_MAX_SIDE = 576
LIVE_YOLO_IMGSZ = 576
LIVE_GESTURE_EVERY_N = 2
YOLO_WORLD_PROMPTS: list[str] = ['person', 'computer monitor', 'monitor screen', 'laptop computer', 'mechanical keyboard', 'computer keyboard', 'computer mouse', 'wireless mouse', 'game controller', 'video game controller', 'headphones', 'webcam', 'smartphone', 'cell phone', 'television', 'tv screen', 'computer desk', 'office chair', 'computer tower', 'cup', 'water bottle', 'book', 'notebook', 'pen', 'desk lamp', 'computer speaker', 'microphone', 'tablet', 'smart watch', 'wall clock', 'cable', 'power strip']
GESTURE_TASK_URL = 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task'
Det = tuple[int, int, int, int, str, float]

def _mp_dir() -> Path:
    d = config.MODELS_DIR / 'mediapipe'
    d.mkdir(parents=True, exist_ok=True)
    return d

def _ensure_gesture_model() -> Path:
    dest = _mp_dir() / 'gesture_recognizer.task'
    if dest.is_file() and dest.stat().st_size > 1000:
        return dest
    logging.info('Downloading MediaPipe gesture_recognizer.task …')
    urllib.request.urlretrieve(GESTURE_TASK_URL, dest)
    return dest

def yolo_available() -> bool:
    try:
        import ultralytics
        return True
    except Exception:
        return False

def mediapipe_gesture_available() -> bool:
    try:
        from mediapipe.tasks.python.vision import GestureRecognizer
        return True
    except Exception:
        return False

def _infer_yolo_device_half() -> tuple[object, bool]:
    try:
        import torch
        if torch.cuda.is_available():
            return (0, True)
    except Exception:
        logging.exception('CUDA probe')
    return ('cpu', False)

def _clip_cache_dir() -> Path:
    return Path.home() / '.cache' / 'clip'

def _set_yolo_world_classes_resilient(model: object, classes: list[str]) -> bool:
    attempts = 6
    base = 4.0
    old_timeout = socket.getdefaulttimeout()
    for attempt in range(1, attempts + 1):
        try:
            socket.setdefaulttimeout(600.0)
            model.set_classes(classes)
            socket.setdefaulttimeout(old_timeout)
            logging.info('YOLO-World: set_classes OK (%d text prompts)', len(classes))
            return True
        except Exception as e:
            socket.setdefaulttimeout(old_timeout)
            logging.warning('YOLO-World set_classes attempt %d/%d failed: %s', attempt, attempts, e)
            if attempt < attempts:
                delay = base * 1.6 ** (attempt - 1)
                logging.info('Retrying set_classes in %.1fs …', delay)
                time.sleep(delay)
            else:
                cache = _clip_cache_dir()
                logging.error('YOLO-World open-vocabulary prompts could not be applied. CLIP weights download failed after %d attempts.\nIf a partial file exists, delete the folder and try again: %s', attempts, cache)
                return False
    return False

def _draw_detections_bgr(out: np.ndarray, dets: list[Det], y_text_start: int=22) -> None:
    color = (40, 180, 255)
    y = y_text_start
    for x1, y1, x2, y2, label, conf in dets:
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        tag = f'{label} {conf:.0%}'
        cv2.putText(out, tag, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

def _results_to_dets_scaled(results: object, sx: float, sy: float, conf_min: float=0.2) -> tuple[list[Det], list[str]]:
    names = results.names
    dets: list[Det] = []
    lines: list[str] = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < conf_min:
            continue
        label = names.get(cls_id, str(cls_id))
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1 = int(x1 * sx)
        y1 = int(y1 * sy)
        x2 = int(x2 * sx)
        y2 = int(y2 * sy)
        dets.append((x1, y1, x2, y2, label, conf))
        lines.append(f'Object: {label} ({conf:.0%})')
    return (dets, lines)

class AdvancedPerception:

    def __init__(self, use_gesture_api: bool, use_yolo: bool, *, live_camera: bool=False) -> None:
        self.use_gesture_api = bool(use_gesture_api and mediapipe_gesture_available())
        self.use_yolo = bool(use_yolo and yolo_available())
        self.live_camera = bool(live_camera)
        self._gesture: Optional[object] = None
        self._yolo: Optional[object] = None
        self._yolo_device: object = 'cpu'
        self._yolo_half = False
        self._live_frame_idx = 0
        self._yolo_stop: Optional[threading.Event] = None
        self._yolo_in_q: Optional[queue.Queue[tuple[np.ndarray, float, float]]] = None
        self._yolo_thread: Optional[threading.Thread] = None
        self._yolo_async_lock = threading.Lock()
        self._yolo_async_dets: list[Det] = []
        self._yolo_async_note: str = ''
        if self.live_camera and self.use_yolo:
            self._yolo_stop = threading.Event()
            self._yolo_in_q = queue.Queue(maxsize=1)
            self._yolo_thread = threading.Thread(target=self._yolo_async_loop, name='YOLOWorldLive', daemon=True)
            self._yolo_thread.start()
            logging.info('YOLO-World live mode: async inference, max_side=%d imgsz=%d', LIVE_INFER_MAX_SIDE, LIVE_YOLO_IMGSZ)

    def close(self) -> None:
        if self._yolo_stop is not None:
            self._yolo_stop.set()
        if self._yolo_thread is not None:
            self._yolo_thread.join(timeout=4.0)
            self._yolo_thread = None
        self._yolo_stop = None
        self._yolo_in_q = None
        if self._gesture is not None:
            try:
                self._gesture.close()
            except Exception:
                logging.exception('GestureRecognizer.close')
            self._gesture = None
        self._yolo = None

    def _load_yolo_weights(self) -> None:
        if self._yolo is not None or not self.use_yolo:
            return
        from ultralytics import YOLO
        self._yolo_device, self._yolo_half = _infer_yolo_device_half()
        logging.info('Loading %s (YOLO-World); device=%s half=%s', YOLO_WORLD_WEIGHT, self._yolo_device, self._yolo_half)
        try:
            self._yolo = YOLO(YOLO_WORLD_WEIGHT)
        except Exception as e:
            logging.warning('Could not load %s (%s); trying yolov8l-worldv2.pt', YOLO_WORLD_WEIGHT, e)
            self._yolo = YOLO('yolov8l-worldv2.pt')
        if not _set_yolo_world_classes_resilient(self._yolo, YOLO_WORLD_PROMPTS):
            logging.warning('Using COCO-only labels for this session (no custom text prompts). Fix network or clear %s and restart.', _clip_cache_dir())

    def _yolo_async_loop(self) -> None:
        try:
            self._load_yolo_weights()
        except Exception:
            logging.exception('YOLO async worker: failed to load weights')
            return
        if self._yolo is None:
            return
        assert self._yolo_stop is not None and self._yolo_in_q is not None
        try:
            import torch
        except Exception:
            torch = None
        while not self._yolo_stop.is_set():
            try:
                small_bgr, sx, sy = self._yolo_in_q.get(timeout=0.35)
            except queue.Empty:
                continue
            try:
                ctx = torch.inference_mode() if torch is not None else contextlib.nullcontext()
                with ctx:
                    pred_kw: dict = {'source': small_bgr, 'verbose': False, 'imgsz': LIVE_YOLO_IMGSZ, 'device': self._yolo_device, 'conf': 0.22, 'max_det': 45}
                    if self._yolo_half and self._yolo_device != 'cpu':
                        pred_kw['half'] = True
                    results = self._yolo.predict(**pred_kw)[0]
                dets, lines = _results_to_dets_scaled(results, sx, sy)
                note = ''
                if lines:
                    note = 'YOLO-World: ' + '; '.join(lines[:14])
                    if len(lines) > 14:
                        note += f' … +{len(lines) - 14} more'
                    note += '\n[live / async — latest finished frame]'
                with self._yolo_async_lock:
                    self._yolo_async_dets = dets
                    self._yolo_async_note = note
            except Exception:
                logging.exception('YOLO-World async predict')

    def _enqueue_live_yolo(self, out: np.ndarray) -> None:
        if self._yolo_in_q is None:
            return
        h, w = out.shape[:2]
        m = max(h, w)
        scale = min(1.0, LIVE_INFER_MAX_SIDE / float(m))
        nw, nh = (int(w * scale), int(h * scale))
        if nw < 1 or nh < 1:
            return
        small = cv2.resize(out, (nw, nh), interpolation=cv2.INTER_AREA)
        sx = w / float(nw)
        sy = h / float(nh)
        try:
            self._yolo_in_q.put_nowait((small, sx, sy))
        except queue.Full:
            try:
                self._yolo_in_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._yolo_in_q.put_nowait((small, sx, sy))
            except queue.Full:
                pass

    def _draw_live_yolo_overlay(self, out: np.ndarray, gesture_text_y: int) -> str:
        with self._yolo_async_lock:
            dets = list(self._yolo_async_dets)
            note = self._yolo_async_note
        if dets:
            _draw_detections_bgr(out, dets, y_text_start=gesture_text_y)
        return note

    def _init_gesture(self) -> None:
        if self._gesture is not None or not self.use_gesture_api:
            return
        from mediapipe.tasks.python.core import base_options as bo
        from mediapipe.tasks.python.vision import GestureRecognizer, GestureRecognizerOptions
        from mediapipe.tasks.python.vision.core import vision_task_running_mode as vrm
        path = _ensure_gesture_model()
        opts = GestureRecognizerOptions(base_options=bo.BaseOptions(model_asset_path=str(path)), running_mode=vrm.VisionTaskRunningMode.IMAGE, num_hands=2, min_hand_detection_confidence=0.6, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5)
        self._gesture = GestureRecognizer.create_from_options(opts)

    def _apply_gesture(self, out: np.ndarray, notes: list[str]) -> int:
        y = 22
        if self._gesture is None:
            return y
        try:
            from mediapipe.tasks.python.vision.core import image as mp_image_mod
            rgb = np.ascontiguousarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            mp_img = mp_image_mod.Image(image_format=mp_image_mod.ImageFormat.SRGB, data=rgb)
            res = self._gesture.recognize(mp_img)
            for _, hand_cats in enumerate(res.gestures):
                if not hand_cats:
                    continue
                best = max(hand_cats, key=lambda c: c.score or 0)
                name = (best.category_name or '?').replace('_', ' ')
                sc = float(best.score or 0)
                if sc < 0.35:
                    continue
                notes.append(f'Gesture API: {name} ({sc:.0%})')
                cv2.putText(out, f'{name} ({sc:.0%})', (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 2, cv2.LINE_AA)
                y += 22
        except Exception:
            logging.exception('GestureRecognizer')
        return y

    def _apply_yolo_sync(self, out: np.ndarray, notes: list[str]) -> None:
        self._load_yolo_weights()
        if self._yolo is None:
            return
        try:
            pred_kw: dict = {'source': out, 'verbose': False, 'imgsz': YOLO_WORLD_IMGSZ_QUALITY, 'device': self._yolo_device, 'conf': 0.22, 'max_det': 45}
            if self._yolo_half and self._yolo_device != 'cpu':
                pred_kw['half'] = True
            results = self._yolo.predict(**pred_kw)[0]
            dets, lines = _results_to_dets_scaled(results, 1.0, 1.0)
            if dets:
                _draw_detections_bgr(out, dets)
            if lines:
                summary = 'YOLO-World: ' + '; '.join(lines[:14])
                if len(lines) > 14:
                    summary += f' … +{len(lines) - 14} more'
                summary += '\n[full quality still image]'
                notes.append(summary)
        except Exception:
            logging.exception('YOLO-World')

    def apply(self, bgr: np.ndarray) -> tuple[np.ndarray, str]:
        out = bgr.copy()
        notes: list[str] = []
        self._live_frame_idx += 1
        run_gesture = True
        if self.live_camera and self.use_gesture_api and self.use_yolo:
            run_gesture = self._live_frame_idx % LIVE_GESTURE_EVERY_N == 0
        self._init_gesture()
        gesture_y = 22
        if self.use_gesture_api and run_gesture:
            gesture_y = self._apply_gesture(out, notes)
        if not self.use_yolo:
            return (out, '\n'.join(notes))
        if self.live_camera and self._yolo_in_q is not None:
            self._enqueue_live_yolo(out)
            yolo_note = self._draw_live_yolo_overlay(out, gesture_y)
            if yolo_note:
                notes.append(yolo_note)
        else:
            self._apply_yolo_sync(out, notes)
        return (out, '\n'.join(notes))
