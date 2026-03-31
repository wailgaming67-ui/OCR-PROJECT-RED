from __future__ import annotations
import logging
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import pytesseract
from PySide6.QtCore import QSettings, QSize, Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QCloseEvent, QFont, QImage, QPixmap
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressDialog, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter, QTabWidget, QTextEdit, QToolBar, QVBoxLayout, QWidget
import advanced_perception
import config
import detection
import gestures
import main as pipeline
import ocr_recognize
import vision_mediapipe
LOG_PATH = config.PROJECT_ROOT / 'ocr_app.log'
CAPTURES_DIR = config.PROJECT_ROOT / 'captures'

def setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.FileHandler(LOG_PATH, encoding='utf-8'), logging.StreamHandler(sys.stdout)])

def log_exception(ctx: str, exc: BaseException) -> str:
    logging.error('%s: %s', ctx, exc, exc_info=(type(exc), exc, exc.__traceback__))
    return f'{ctx}: {exc}'
CAM_PREVIEW_MAX_SIDE = 640
CAM_TARGET_FPS = 32
CAM_TARGET_FPS_TEXT = 60
CAM_PREVIEW_MAX_SIDE_TEXT = 512
CAM_OCR_EVERY_N = 12
CAM_OCR_EVERY_N_TEXT = 8
CAM_EASYOCR_EVERY_N = 48
CAM_EASYOCR_EVERY_N_TEXT = 32
CAM_OCR_MAX_BINARY_WIDTH = 960
CAM_OCR_MAX_BINARY_WIDTH_TEXT = 720

def _resize_for_camera_ocr(binary: np.ndarray, max_w: int=CAM_OCR_MAX_BINARY_WIDTH) -> np.ndarray:
    h, w = binary.shape[:2]
    if w <= max_w:
        return binary
    scale = max_w / float(w)
    return cv2.resize(binary, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def numpy_bgr_to_qpixmap(bgr: np.ndarray, max_side: int=1400) -> QPixmap:
    if bgr is None or bgr.size == 0:
        return QPixmap()
    try:
        bgr = np.ascontiguousarray(bgr)
        h, w = bgr.shape[:2]
        if max(h, w) > max_side:
            scale = max_side / float(max(h, w))
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            bgr = np.ascontiguousarray(bgr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())
    except Exception as e:
        logging.exception('numpy_bgr_to_qpixmap: %s', e)
        return QPixmap()

class ImageProcessWorker(QThread):
    finished_ok = Signal(str, object)
    failed = Signal(str)

    def __init__(self, image_path: str, cascade: cv2.CascadeClassifier, east: detection.EASTDetector, use_easyocr: bool, easy_reader, ocr_lang: str, use_mp_face: bool, use_mp_hands: bool, use_gesture_api: bool, use_yolo: bool, text_only: bool, parent=None) -> None:
        super().__init__(parent)
        self._path = image_path
        self._cascade = cascade
        self._east = east
        self._use_easyocr = use_easyocr
        self._easy_reader = easy_reader
        self._ocr_lang = ocr_lang
        self._use_mp_face = use_mp_face
        self._use_mp_hands = use_mp_hands
        self._use_gesture_api = use_gesture_api
        self._use_yolo = use_yolo
        self._text_only = text_only

    def run(self) -> None:
        mp_drawer: vision_mediapipe.MediaPipeDrawer | None = None
        adv: advanced_perception.AdvancedPerception | None = None
        try:
            p = Path(self._path)
            if not p.is_file():
                self.failed.emit(f'File not found:\n{p}')
                return
            bgr = cv2.imread(str(p))
            if bgr is None:
                self.failed.emit(f'Could not decode image (unsupported or corrupt):\n{p}')
                return
            mp_drawer: vision_mediapipe.MediaPipeDrawer | None = None
            mp_fn = None
            if not self._text_only:
                mp_drawer = vision_mediapipe.MediaPipeDrawer(self._use_mp_hands, self._use_mp_face)
                mp_fn = mp_drawer.annotate if mp_drawer.use_hands or mp_drawer.use_face else None
            text, overlay, gray_proc, binary, bgr_vis = pipeline.run_frame(bgr, self._cascade, self._east, self._use_easyocr, self._easy_reader, run_easyocr=True, ocr_lang=self._ocr_lang, mp_annotate=mp_fn, text_only=self._text_only)
            if not self._text_only and (self._use_gesture_api or self._use_yolo):
                adv = advanced_perception.AdvancedPerception(self._use_gesture_api, self._use_yolo, live_camera=False)
                bgr_vis, adv_txt = adv.apply(bgr_vis)
                if adv_txt:
                    text = f'{text}\n\n[Perception API]\n{adv_txt}'
            vis = pipeline.build_pipeline_visualization(bgr_vis, gray_proc, binary, overlay, scale=0.55)
            self.finished_ok.emit(text, vis)
        except Exception as e:
            self.failed.emit(log_exception('Image processing', e))
        finally:
            if adv is not None:
                adv.close()
            if mp_drawer is not None:
                mp_drawer.close()

class CameraWorker(QThread):
    frame_ready = Signal(str, object)
    camera_error = Signal(str)
    camera_closed = Signal()
    wave_captured = Signal(object)

    def __init__(self, camera_index: int, cascade: cv2.CascadeClassifier, east: detection.EASTDetector, use_easyocr: bool, easy_reader, ocr_lang: str, use_mp_face: bool, use_mp_hands: bool, wave_capture: bool, use_gesture_api: bool, use_yolo: bool, text_only: bool, parent=None) -> None:
        super().__init__(parent)
        self._index = camera_index
        self._cascade = cascade
        self._east = east
        self._use_easyocr = use_easyocr
        self._easy_reader = easy_reader
        self._ocr_lang = ocr_lang
        self._use_mp_face = use_mp_face
        self._use_mp_hands = use_mp_hands
        self._gesture_api = use_gesture_api
        self._yolo = use_yolo
        self._text_only = text_only
        self._last_adv = ''
        self._wave_capture = bool(wave_capture and use_mp_hands and (not text_only))
        self._wave: Optional[gestures.WaveGestureDetector] = gestures.WaveGestureDetector() if self._wave_capture else None
        self._running = False
        self._frame_count = 0
        self._last_ocr = ''
        self._last_easyocr = ''
        self._ocr_lock = threading.Lock()
        self._ocr_q: Optional[queue.Queue] = None
        self._ocr_stop: Optional[threading.Event] = None
        self._ocr_thread: Optional[threading.Thread] = None

    def stop(self) -> None:
        self._running = False

    def _ocr_worker_text_mode(self) -> None:
        assert self._ocr_q is not None and self._ocr_stop is not None
        while not self._ocr_stop.is_set():
            try:
                item = self._ocr_q.get(timeout=0.25)
            except queue.Empty:
                continue
            if item is None:
                break
            binary_small, bgr_for_easy = item
            try:
                raw = ocr_recognize.recognize_tesseract(binary_small, lang=self._ocr_lang)
                with self._ocr_lock:
                    if raw.strip():
                        self._last_ocr = raw
                    elif not (self._last_ocr or '').strip():
                        self._last_ocr = raw
            except Exception as ocr_e:
                logging.exception('Camera OCR async: %s', ocr_e)
                with self._ocr_lock:
                    self._last_ocr = f'(OCR error: {ocr_e})'
            if bgr_for_easy is not None and self._use_easyocr and (self._easy_reader is not None):
                try:
                    ez = ocr_recognize.recognize_easyocr(bgr_for_easy, self._easy_reader)
                    with self._ocr_lock:
                        self._last_easyocr = ez
                except Exception as ez_e:
                    logging.exception('Camera EasyOCR async: %s', ez_e)
                    with self._ocr_lock:
                        self._last_easyocr = f'(EasyOCR error: {ez_e})'

    def _format_output(self) -> str:
        if self._text_only:
            with self._ocr_lock:
                last_ocr = self._last_ocr
                last_ez = self._last_easyocr
            parts = ['[Text & language — live @ 60 FPS preview — OCR async]']
        else:
            last_ocr = self._last_ocr
            last_ez = self._last_easyocr
            parts = ['[Live preview — fast mode]']
            if self._wave_capture:
                parts.append('[Gesture] Wave hand left ↔ right (wrist visible) → auto-saves PNG to captures/')
            if self._last_adv:
                parts.append('[Perception API]\n' + self._last_adv)
        ocr_line = '…'
        if last_ocr:
            cleaned = ocr_recognize.sanitize_ocr_output(last_ocr)
            ocr_line = cleaned if cleaned else '…'
        parts.append('[Full frame OCR]\n' + ocr_line)
        if self._use_easyocr and self._easy_reader is not None:
            ez = last_ez
            if ez:
                ez = ocr_recognize.sanitize_ocr_output(ez)
            parts.append('[EasyOCR]\n' + (ez if ez else '(updates less often than preview)'))
        if ocr_recognize.is_tesseract_working() and last_ocr and ('[Tesseract not found' not in last_ocr):
            est = ocr_recognize.guess_language_from_text(last_ocr)
            if est:
                parts.append(f'[Estimated language of recognized text: {est}]')
        return '\n\n'.join(parts)

    def _stop_ocr_worker(self) -> None:
        if self._ocr_thread is None:
            return
        if self._ocr_stop is not None:
            self._ocr_stop.set()
        if self._ocr_q is not None:
            try:
                self._ocr_q.put_nowait(None)
            except Exception:
                pass
        self._ocr_thread.join(timeout=4.0)
        self._ocr_thread = None
        self._ocr_q = None
        self._ocr_stop = None

    def run(self) -> None:
        cap = None
        target_fps = CAM_TARGET_FPS_TEXT if self._text_only else CAM_TARGET_FPS
        frame_period = 1.0 / max(8, min(60, target_fps))
        preview_max = CAM_PREVIEW_MAX_SIDE_TEXT if self._text_only else CAM_PREVIEW_MAX_SIDE
        try:
            if sys.platform == 'win32':
                cap = cv2.VideoCapture(self._index, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(self._index)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self._index)
            if not cap.isOpened():
                self.camera_error.emit(f'Cannot open camera index {self._index}.\nCheck the index in Settings or disconnect other apps using the camera.')
                return
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            if self._text_only:
                try:
                    cap.set(cv2.CAP_PROP_FPS, 60)
                except Exception:
                    pass
            self._running = True
            if self._text_only:
                self._ocr_stop = threading.Event()
                self._ocr_q = queue.Queue(maxsize=1)
                self._ocr_thread = threading.Thread(target=self._ocr_worker_text_mode, name='CameraTextOCR', daemon=True)
                self._ocr_thread.start()
            mp_drawer: vision_mediapipe.MediaPipeDrawer | None = None
            mp_fn = None
            if not self._text_only:
                mp_drawer = vision_mediapipe.MediaPipeDrawer(self._use_mp_hands, self._use_mp_face)
                mp_fn = mp_drawer.annotate if mp_drawer.use_hands or mp_drawer.use_face else None
            adv: advanced_perception.AdvancedPerception | None = None
            if not self._text_only and (self._gesture_api or self._yolo):
                adv = advanced_perception.AdvancedPerception(self._gesture_api, self._yolo, live_camera=True)
            ocr_every = CAM_OCR_EVERY_N_TEXT if self._text_only else CAM_OCR_EVERY_N
            ez_every = CAM_EASYOCR_EVERY_N_TEXT if self._text_only else CAM_EASYOCR_EVERY_N
            try:
                while self._running:
                    loop_start = time.perf_counter()
                    if not self._running:
                        break
                    ok, frame = cap.read()
                    if not ok:
                        self.camera_error.emit('Camera read failed (device disconnected?).')
                        break
                    if not self._running:
                        break
                    self._frame_count += 1
                    try:
                        _, overlay, gray_proc, binary, bgr_vis = pipeline.run_frame(frame, self._cascade, self._east, False, None, preview_only=True, max_side=preview_max, fast_detection=True, mp_annotate=mp_fn, text_only=self._text_only)
                        if adv is not None:
                            bgr_vis, self._last_adv = adv.apply(bgr_vis)
                        else:
                            self._last_adv = ''
                        vis = pipeline.build_pipeline_visualization(bgr_vis, gray_proc, binary, overlay, scale=0.42)
                        if self._wave is not None and self._running:
                            wx = None
                            if mp_drawer.last_primary_wrist_norm:
                                wx = mp_drawer.last_primary_wrist_norm[0]
                            if self._wave.update(time.time(), wx):
                                self.wave_captured.emit(vis.copy())
                        if self._running and self._frame_count % ocr_every == 0:
                            if self._text_only and self._ocr_q is not None:
                                small = _resize_for_camera_ocr(binary, max_w=CAM_OCR_MAX_BINARY_WIDTH_TEXT)
                                bgr_ez = None
                                if self._use_easyocr and self._easy_reader is not None and (self._frame_count % ez_every == 0):
                                    bgr_ez = bgr_vis.copy()
                                try:
                                    self._ocr_q.put_nowait((small, bgr_ez))
                                except queue.Full:
                                    try:
                                        self._ocr_q.get_nowait()
                                    except queue.Empty:
                                        pass
                                    try:
                                        self._ocr_q.put_nowait((small, bgr_ez))
                                    except queue.Full:
                                        pass
                            elif not self._text_only:
                                try:
                                    small = _resize_for_camera_ocr(binary)
                                    if self._running:
                                        raw_ocr = ocr_recognize.recognize_tesseract(small, lang=self._ocr_lang)
                                    if raw_ocr.strip():
                                        self._last_ocr = raw_ocr
                                    elif not (self._last_ocr or '').strip():
                                        self._last_ocr = raw_ocr
                                except Exception as ocr_e:
                                    logging.exception('Camera OCR: %s', ocr_e)
                                    self._last_ocr = f'(OCR error: {ocr_e})'
                        if self._running and (not self._text_only) and self._use_easyocr and (self._easy_reader is not None) and (self._frame_count % ez_every == 0):
                            try:
                                if self._running:
                                    self._last_easyocr = ocr_recognize.recognize_easyocr(bgr_vis, self._easy_reader)
                            except Exception as ez_e:
                                logging.exception('Camera EasyOCR: %s', ez_e)
                                self._last_easyocr = f'(EasyOCR error: {ez_e})'
                        if self._running:
                            self.frame_ready.emit(self._format_output(), vis)
                    except Exception as e:
                        logging.exception('Camera frame: %s', e)
                        self.camera_error.emit(f'Frame processing error:\n{e}')
                        break
                    if not self._running:
                        break
                    elapsed = time.perf_counter() - loop_start
                    sleep_s = frame_period - elapsed
                    if sleep_s > 0:
                        time.sleep(sleep_s)
            finally:
                if adv is not None:
                    adv.close()
                if mp_drawer is not None:
                    mp_drawer.close()
                self._stop_ocr_worker()
        except Exception as e:
            self.camera_error.emit(log_exception('Camera', e))
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    logging.exception('cap.release')
            self.camera_closed.emit()

class SettingsDialog(QDialog):

    def __init__(self, settings: QSettings, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle('Settings')
        self._settings = settings
        lay = QVBoxLayout(self)
        g = QGroupBox('Tesseract OCR')
        fl = QFormLayout(g)
        self._tess = QLineEdit()
        self._tess.setPlaceholderText('C:\\Program Files\\Tesseract-OCR\\tesseract.exe')
        stored = (settings.value('tesseract_cmd', '', str) or '').strip()
        if stored:
            self._tess.setText(stored)
        else:
            auto = ocr_recognize.find_tesseract_executable()
            if auto:
                self._tess.setText(auto)
        browse = QPushButton('Browse…')
        browse.clicked.connect(self._browse_tesseract)
        auto_btn = QPushButton('Auto-detect')
        auto_btn.setToolTip('Search PATH and common install folders for tesseract.exe')
        auto_btn.clicked.connect(self._auto_detect_tesseract)
        test_btn = QPushButton('Test')
        test_btn.setToolTip('Run tesseract --version using the path above')
        test_btn.clicked.connect(self._test_tesseract)
        row = QHBoxLayout()
        row.addWidget(self._tess)
        row.addWidget(browse)
        row.addWidget(auto_btn)
        row.addWidget(test_btn)
        fl.addRow('Executable:', row)
        hint = QLabel('Leave blank to auto-detect on startup. If OCR fails, click <b>Browse</b> and select <b>tesseract.exe</b> from your install folder, then <b>Test</b>.')
        hint.setWordWrap(True)
        fl.addRow(hint)
        self._lang = QLineEdit()
        self._lang.setPlaceholderText('eng  or  ara+eng')
        self._lang.setText(settings.value('ocr_lang', 'eng', str) or 'eng')
        fl.addRow('OCR language(s):', self._lang)
        lay.addWidget(g)
        g2 = QGroupBox('Camera')
        f2 = QFormLayout(g2)
        self._cam = QSpinBox()
        self._cam.setRange(0, 9)
        self._cam.setValue(int(settings.value('camera_index', 0, int)))
        f2.addRow('Device index:', self._cam)
        lay.addWidget(g2)
        self._easy_default = QCheckBox('Remember EasyOCR enabled')
        self._easy_default.setChecked(settings.value('easyocr_default', False, bool))
        lay.addWidget(self._easy_default)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def _browse_tesseract(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, 'Select tesseract.exe', str(Path('C:/Program Files/Tesseract-OCR').expanduser()), 'Executable (tesseract.exe);;All files (*.*)')
        if path:
            self._tess.setText(path)

    def _auto_detect_tesseract(self) -> None:
        found = ocr_recognize.find_tesseract_executable()
        if found:
            self._tess.setText(found)
            QMessageBox.information(self, 'Tesseract', f'Found:\n{found}')
        else:
            QMessageBox.warning(self, 'Tesseract', 'Could not find tesseract.exe.\n\nInstall from the link in the app banner, then either add the install folder to Windows PATH, or use Browse to select tesseract.exe manually.')

    def _test_tesseract(self) -> None:
        raw = self._tess.text().strip()
        try:
            ocr_recognize.set_tesseract_cmd(raw if raw else None)
            ver = pytesseract.get_tesseract_version()
            QMessageBox.information(self, 'Tesseract OK', f'Working. Version: {ver}\n\nPath used:\n{pytesseract.pytesseract.tesseract_cmd}')
        except Exception as e:
            QMessageBox.critical(self, 'Tesseract failed', f'Could not run Tesseract:\n{e}\n\nSet the full path to tesseract.exe (see Browse).')

    def save_to_settings(self) -> None:
        self._settings.setValue('tesseract_cmd', self._tess.text().strip())
        self._settings.setValue('ocr_lang', self._lang.text().strip() or 'eng')
        self._settings.setValue('camera_index', self._cam.value())
        self._settings.setValue('easyocr_default', self._easy_default.isChecked())

class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('OCR Vision — OpenCV + Tesseract')
        self.setMinimumSize(QSize(1000, 700))
        self._settings = QSettings('ProjectRed', 'OCRVision')
        self._cascade: cv2.CascadeClassifier | None = None
        self._east: detection.EASTDetector | None = None
        self._easy_reader = None
        self._easyocr_init_done = False
        self._image_worker: ImageProcessWorker | None = None
        self._camera_worker: CameraWorker | None = None
        self._last_vis: np.ndarray | None = None
        self._last_ui_frame_time: float = 0.0
        self._camera_text_only: bool = False
        self._last_camera_panel_text: Optional[str] = None
        self._last_text_panel_time: float = 0.0
        self._build_ui()
        self._build_menu_toolbar()
        self._apply_settings_to_engine()
        self._refresh_tesseract_banner()
        try:
            self._init_detection()
        except Exception as e:
            QMessageBox.warning(self, 'Detection models', f'Could not load face cascade or EAST:\n{e}\n\nFace/text overlays may be limited.')
            log_exception('init_detection', e)
        if not vision_mediapipe.mediapipe_available():
            e = vision_mediapipe.last_import_error()
            self._log(f'MediaPipe not available — Face/Hands (MediaPipe) need: pip install mediapipe. Haar face + text boxes still work. Import error: {e}')

    def _init_detection(self) -> None:
        self._cascade = detection.load_face_cascade()
        self._east = detection.EASTDetector()
        if not self._east.available:
            self._log(f'EAST model missing — using morphology for text regions. Optional: python download_east_model.py → {config.EAST_MODEL_PATH}')

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        self._tess_banner = QWidget()
        bl = QHBoxLayout(self._tess_banner)
        self._tess_banner_label = QLabel()
        self._tess_banner_label.setWordWrap(True)
        self._tess_banner_label.setOpenExternalLinks(True)
        bt = QPushButton('Configure Tesseract…')
        bt.clicked.connect(self._open_settings)
        bl.addWidget(self._tess_banner_label, 1)
        bl.addWidget(bt)
        outer.addWidget(self._tess_banner)
        split = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(split)
        left = QWidget()
        left_lay = QVBoxLayout(left)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._image_label.setMinimumSize(400, 300)
        self._image_label.setText('Open an image or start the camera.')
        self._image_label.setStyleSheet('color: #888; font-size: 14px;')
        self._scroll.setWidget(self._image_label)
        left_lay.addWidget(self._scroll)
        split.addWidget(left)
        right_tabs = QTabWidget()
        self._text_out = QTextEdit()
        self._text_out.setReadOnly(True)
        self._text_out.setFont(QFont('Consolas', 10))
        right_tabs.addTab(self._text_out, 'Recognized text')
        self._log_out = QTextEdit()
        self._log_out.setReadOnly(True)
        self._log_out.setFont(QFont('Consolas', 9))
        right_tabs.addTab(self._log_out, 'Activity log')
        split.addWidget(right_tabs)
        split.setSizes([700, 400])
        bar = self.statusBar()
        assert bar is not None
        bar.showMessage('Ready')
        self._chk_easy = QCheckBox('EasyOCR (slower)')
        self._chk_easy.setChecked(self._settings.value('easyocr_default', False, bool))
        self._chk_easy.stateChanged.connect(self._on_easy_toggle)

    def _refresh_tesseract_banner(self) -> None:
        if ocr_recognize.is_tesseract_working():
            self._tess_banner.hide()
        else:
            self._tess_banner.show()
            self._tess_banner.setStyleSheet('background-color:#3d2a1f;color:#f0e0d0;padding:10px;border-radius:6px;')
            self._tess_banner_label.setText('<b>No working Tesseract OCR engine.</b> Orange boxes show <i>where</i> text probably is; to <i>read</i> text you must install Tesseract and point to <b>tesseract.exe</b> in Settings (typical: <code>C:\\Program Files\\Tesseract-OCR\\tesseract.exe</code>). <a href="https://github.com/UB-Mannheim/tesseract/wiki">Windows installer</a>')

    def _build_menu_toolbar(self) -> None:
        tb = QToolBar('Main')
        tb.setMovable(False)
        self.addToolBar(tb)
        act_open = QAction('Open image…', self)
        act_open.setShortcut('Ctrl+O')
        act_open.triggered.connect(self._open_image)
        tb.addAction(act_open)
        self._act_cam = QAction('Start camera', self)
        self._act_cam.triggered.connect(self._toggle_camera)
        tb.addAction(self._act_cam)
        act_save = QAction('Save snapshot', self)
        act_save.setShortcut('Ctrl+S')
        act_save.triggered.connect(self._save_snapshot)
        tb.addAction(act_save)
        tb.addSeparator()
        tb.addWidget(QLabel('Mode:'))
        self._combo_vision = QComboBox()
        self._combo_vision.addItem('Full pipeline', 'full')
        self._combo_vision.addItem('Text & language only', 'text_only')
        vm = self._settings.value('vision_mode', 'full', str)
        self._combo_vision.setCurrentIndex(1 if vm == 'text_only' else 0)
        self._combo_vision.setMinimumWidth(200)
        self._combo_vision.setToolTip('Text & language only: Tesseract/EasyOCR + language estimate. No faces, hands, wave capture, gestures, or object detection.')
        self._combo_vision.currentIndexChanged.connect(self._on_vision_mode_changed)
        tb.addWidget(self._combo_vision)
        tb.addSeparator()
        tb.addWidget(self._chk_easy)
        self._chk_mp_face = QCheckBox('Face (MediaPipe)')
        self._chk_mp_face.setChecked(self._settings.value('mp_face', True, bool))
        self._chk_mp_face.setToolTip('Neural face detection (cyan). Green = classic Haar. Requires mediapipe.')
        self._chk_mp_face.stateChanged.connect(lambda: self._settings.setValue('mp_face', self._chk_mp_face.isChecked()))
        tb.addWidget(self._chk_mp_face)
        self._chk_mp_hands = QCheckBox('Hands (MediaPipe)')
        self._chk_mp_hands.setChecked(self._settings.value('mp_hands', True, bool))
        self._chk_mp_hands.setToolTip('Hand regions (magenta). Requires mediapipe.')
        self._chk_mp_hands.stateChanged.connect(lambda: self._settings.setValue('mp_hands', self._chk_mp_hands.isChecked()))
        tb.addWidget(self._chk_mp_hands)
        self._chk_wave = QCheckBox('Wave → capture')
        self._chk_wave.setChecked(self._settings.value('wave_capture', True, bool))
        self._chk_wave.setToolTip('When the camera runs with Hands (MediaPipe), wave your hand left–right to save a snapshot.')
        self._chk_wave.stateChanged.connect(lambda: self._settings.setValue('wave_capture', self._chk_wave.isChecked()))
        tb.addWidget(self._chk_wave)
        self._chk_gesture_api = QCheckBox('Gesture API')
        self._chk_gesture_api.setChecked(self._settings.value('gesture_api', False, bool))
        self._chk_gesture_api.setToolTip('MediaPipe GestureRecognizer (Open Palm, Victory, …). Requires mediapipe.')
        self._chk_gesture_api.stateChanged.connect(lambda: self._settings.setValue('gesture_api', self._chk_gesture_api.isChecked()))
        tb.addWidget(self._chk_gesture_api)
        self._chk_yolo = QCheckBox('YOLO objects')
        self._chk_yolo.setChecked(self._settings.value('yolo_objects', False, bool))
        tip_yolo = 'YOLO-World (yolov8x-worldv2): open-vocabulary — game controller, monitor, laptop, … Uses GPU+FP16 if CUDA is available. First run downloads large weights.'
        if not advanced_perception.yolo_available():
            tip_yolo += '\nInstall: pip install ultralytics'
        self._chk_yolo.setToolTip(tip_yolo)
        self._chk_yolo.stateChanged.connect(lambda: self._settings.setValue('yolo_objects', self._chk_yolo.isChecked()))
        tb.addWidget(self._chk_yolo)
        act_settings = QAction('Settings…', self)
        act_settings.triggered.connect(self._open_settings)
        tb.addAction(act_settings)
        m = self.menuBar()
        file_menu = m.addMenu('&File')
        file_menu.addAction(act_open)
        file_menu.addAction(act_save)
        file_menu.addSeparator()
        quit_a = QAction('Exit', self)
        quit_a.setShortcut('Ctrl+Q')
        quit_a.triggered.connect(self.close)
        file_menu.addAction(quit_a)
        tools_menu = m.addMenu('&Tools')
        tools_menu.addAction(act_settings)
        help_menu = m.addMenu('&Help')
        about = QAction('About', self)
        about.triggered.connect(self._about)
        help_menu.addAction(about)
        self._apply_vision_mode_ui()

    def _is_text_only_mode(self) -> bool:
        return self._combo_vision.currentData() == 'text_only'

    @Slot()
    def _on_vision_mode_changed(self) -> None:
        data = self._combo_vision.currentData()
        self._settings.setValue('vision_mode', data)
        self._apply_vision_mode_ui()

    def _apply_vision_mode_ui(self) -> None:
        text_only = self._is_text_only_mode()
        for w in (self._chk_mp_face, self._chk_mp_hands, self._chk_wave, self._chk_gesture_api, self._chk_yolo):
            w.setEnabled(not text_only)

    def _apply_settings_to_engine(self) -> None:
        cmd = self._settings.value('tesseract_cmd', '', str) or ''
        try:
            ocr_recognize.set_tesseract_cmd(cmd if cmd.strip() else None)
            if not ocr_recognize.is_tesseract_working():
                ocr_recognize.set_tesseract_cmd(None)
        except Exception as e:
            self._log(f'Tesseract path error: {e}')
            try:
                ocr_recognize.set_tesseract_cmd(None)
            except Exception:
                pass

    def _ocr_lang(self) -> str:
        return self._settings.value('ocr_lang', 'eng', str) or 'eng'

    def _camera_index(self) -> int:
        return int(self._settings.value('camera_index', 0, int))

    def _log(self, msg: str) -> None:
        logging.info(msg)
        self._log_out.append(msg)

    @Slot()
    def _on_easy_toggle(self) -> None:
        if self._chk_easy.isChecked() and (not self._easyocr_init_done):
            self._ensure_easyocr()

    def _ensure_easyocr(self) -> bool:
        if self._easy_reader is not None:
            return True
        dlg = QProgressDialog('Loading EasyOCR models (first time may take minutes)…', 'Cancel', 0, 0, self)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.show()
        QApplication.processEvents()
        try:
            reader = ocr_recognize.get_easyocr_reader()
            if reader is None:
                QMessageBox.information(self, 'EasyOCR', 'EasyOCR is not installed.\n\npip install easyocr')
                self._chk_easy.setChecked(False)
                return False
            self._easy_reader = reader
            self._easyocr_init_done = True
            self._log('EasyOCR ready.')
            return True
        except Exception as e:
            QMessageBox.critical(self, 'EasyOCR', f'Failed to load EasyOCR:\n{e}')
            log_exception('EasyOCR init', e)
            self._chk_easy.setChecked(False)
            return False
        finally:
            dlg.close()

    def _ensure_detection(self) -> bool:
        if self._cascade is None or self._east is None:
            try:
                self._init_detection()
            except Exception as e:
                QMessageBox.critical(self, 'Error', str(e))
                return False
        return self._cascade is not None and self._east is not None

    @Slot()
    def _open_image(self) -> None:
        if self._camera_worker is not None and self._camera_worker.isRunning():
            QMessageBox.information(self, 'Camera active', 'Stop the camera before opening an image.')
            return
        if self._image_worker is not None and self._image_worker.isRunning():
            QMessageBox.information(self, 'Busy', 'An image is still processing. Please wait.')
            return
        start = self._settings.value('last_image_dir', str(Path.home()), str)
        path, _ = QFileDialog.getOpenFileName(self, 'Open image', start, 'Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;All files (*.*)')
        if not path:
            return
        self._settings.setValue('last_image_dir', str(Path(path).parent))
        if not self._ensure_detection():
            return
        use_ez = self._chk_easy.isChecked()
        if use_ez and (not self._ensure_easyocr()):
            use_ez = False
        self._set_busy(True)
        self._image_worker = ImageProcessWorker(path, self._cascade, self._east, use_ez, self._easy_reader, self._ocr_lang(), self._chk_mp_face.isChecked(), self._chk_mp_hands.isChecked(), self._chk_gesture_api.isChecked(), self._chk_yolo.isChecked(), self._is_text_only_mode())
        self._image_worker.finished_ok.connect(self._on_image_done)
        self._image_worker.failed.connect(self._on_image_fail)
        self._image_worker.finished.connect(self._on_image_worker_finished)
        self._image_worker.start()

    @Slot(str, object)
    def _on_image_done(self, text: str, vis: object) -> None:
        self._last_vis = vis
        self._text_out.setPlainText(text)
        if isinstance(vis, np.ndarray):
            self._image_label.setPixmap(numpy_bgr_to_qpixmap(vis))
            self._image_label.setStyleSheet('')
        self.statusBar().showMessage('Image processed.', 5000)

    @Slot(str)
    def _on_image_fail(self, err: str) -> None:
        self._log(err)
        QMessageBox.warning(self, 'Image error', err)

    @Slot()
    def _on_image_worker_finished(self) -> None:
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        if busy:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        else:
            while QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()

    @Slot()
    def _toggle_camera(self) -> None:
        if self._camera_worker is not None and self._camera_worker.isRunning():
            self._stop_camera()
            return
        if not self._ensure_detection():
            return
        use_ez = self._chk_easy.isChecked()
        if use_ez and (not self._ensure_easyocr()):
            use_ez = False
        self._camera_worker = CameraWorker(self._camera_index(), self._cascade, self._east, use_ez, self._easy_reader, self._ocr_lang(), self._chk_mp_face.isChecked(), self._chk_mp_hands.isChecked(), self._chk_wave.isChecked(), self._chk_gesture_api.isChecked(), self._chk_yolo.isChecked(), self._is_text_only_mode())
        self._camera_worker.frame_ready.connect(self._on_camera_frame)
        self._camera_worker.camera_error.connect(self._on_camera_error)
        self._camera_worker.camera_closed.connect(self._on_camera_stopped)
        self._camera_worker.wave_captured.connect(self._on_wave_capture)
        self._camera_text_only = self._is_text_only_mode()
        self._last_camera_panel_text = None
        self._last_text_panel_time = 0.0
        self._camera_worker.start()
        self._act_cam.setText('Stop camera')
        self.statusBar().showMessage('Camera running…')

    @Slot(str, object)
    def _on_camera_frame(self, text: str, vis: object) -> None:
        self._last_vis = vis
        now = time.perf_counter()
        if self._camera_text_only:
            if text != self._last_camera_panel_text or now - self._last_text_panel_time >= 0.12:
                self._text_out.setPlainText(text)
                self._last_camera_panel_text = text
                self._last_text_panel_time = now
        else:
            self._text_out.setPlainText(text)
        if isinstance(vis, np.ndarray):
            ui_fps = 60.0 if self._camera_text_only else 33.0
            if now - self._last_ui_frame_time >= 1.0 / ui_fps:
                self._image_label.setPixmap(numpy_bgr_to_qpixmap(vis))
                self._image_label.setStyleSheet('')
                self._last_ui_frame_time = now

    @Slot(object)
    def _on_wave_capture(self, vis: object) -> None:
        if not isinstance(vis, np.ndarray):
            return
        try:
            CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
            name = time.strftime('wave_%Y%m%d_%H%M%S.png')
            path = CAPTURES_DIR / name
            if not cv2.imwrite(str(path), vis):
                raise RuntimeError('cv2.imwrite failed')
            self.statusBar().showMessage(f'Wave captured → {path}', 8000)
            self._log(f'Wave gesture: saved {path}')
        except Exception as e:
            log_exception('wave capture', e)
            self._log(f'Wave capture failed: {e}')

    @Slot(str)
    def _on_camera_error(self, err: str) -> None:
        self._log(err)
        QMessageBox.warning(self, 'Camera', err)
        self._stop_camera()

    @Slot()
    def _on_camera_stopped(self) -> None:
        self._camera_worker = None
        self._camera_text_only = False
        self._act_cam.setText('Start camera')
        self._act_cam.setEnabled(True)
        self.statusBar().showMessage('Camera stopped.', 3000)

    def _stop_camera(self) -> None:
        w = self._camera_worker
        if w is None:
            self._act_cam.setText('Start camera')
            self._act_cam.setEnabled(True)
            return
        if not w.isRunning():
            self._camera_worker = None
            self._act_cam.setText('Start camera')
            self._act_cam.setEnabled(True)
            return
        self._act_cam.setEnabled(False)
        self._act_cam.setText('Stopping…')
        w.stop()

    @Slot()
    def _save_snapshot(self) -> None:
        vis = getattr(self, '_last_vis', None)
        if not isinstance(vis, np.ndarray):
            QMessageBox.information(self, 'Save', 'Nothing to save yet.')
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Save snapshot', str(Path.home() / 'ocr_snapshot.png'), 'PNG (*.png);;JPEG (*.jpg)')
        if not path:
            return
        try:
            ok = cv2.imwrite(path, vis)
            if not ok:
                raise RuntimeError('cv2.imwrite returned False')
            self.statusBar().showMessage(f'Saved: {path}', 5000)
            self._log(f'Saved snapshot: {path}')
        except Exception as e:
            QMessageBox.critical(self, 'Save failed', str(e))
            log_exception('save snapshot', e)

    @Slot()
    def _open_settings(self) -> None:
        dlg = SettingsDialog(self._settings, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        dlg.save_to_settings()
        self._apply_settings_to_engine()
        self._refresh_tesseract_banner()
        self._chk_easy.setChecked(self._settings.value('easyocr_default', False, bool))

    @Slot()
    def _about(self) -> None:
        QMessageBox.about(self, 'About OCR Vision', f'<h3>OCR Vision</h3><p>OpenCV preprocessing, text regions, Haar faces, MediaPipe face/hands, wave gesture capture to <code>captures/</code>, Tesseract + optional EasyOCR, language hint.</p><p>Log file:<br><code>{LOG_PATH}</code></p>')

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._camera_worker is not None and self._camera_worker.isRunning():
            self._camera_worker.stop()
            if not self._camera_worker.wait(15000):
                logging.warning('Camera thread did not finish within 15s on exit')
            self._camera_worker = None
        self._act_cam.setText('Start camera')
        self._act_cam.setEnabled(True)
        if self._image_worker is not None and self._image_worker.isRunning():
            self._image_worker.wait(5000)
        event.accept()

def main() -> None:
    setup_logging()
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setOrganizationName('ProjectRed')
    app.setApplicationName('OCRVision')
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
if __name__ == '__main__':
    main()
