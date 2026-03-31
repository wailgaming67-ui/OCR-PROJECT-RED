from __future__ import annotations
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import config
_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

def load_face_cascade() -> cv2.CascadeClassifier:
    return cv2.CascadeClassifier(_CASCADE_PATH)

def detect_faces(gray: np.ndarray, cascade: cv2.CascadeClassifier) -> list[tuple[int, int, int, int]]:
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]

def _decode_east_raw(scores: np.ndarray, geometry: np.ndarray, conf_threshold: float) -> tuple[list[tuple[int, int, int, int]], list[float]]:
    num_rows, num_cols = scores.shape[2:4]
    rects: list[tuple[int, int, int, int]] = []
    confidences: list[float] = []
    for y in range(num_rows):
        scores_data = scores[0, 0, y]
        x0 = geometry[0, 0, y]
        x1 = geometry[0, 1, y]
        x2 = geometry[0, 2, y]
        x3 = geometry[0, 3, y]
        angles = geometry[0, 4, y]
        for x in range(num_cols):
            if scores_data[x] < conf_threshold:
                continue
            offset_x, offset_y = (x * 4.0, y * 4.0)
            angle = angles[x]
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            h = x0[x] + x2[x]
            w = x1[x] + x3[x]
            end_x = int(offset_x + cos_a * x1[x] + sin_a * x2[x])
            end_y = int(offset_y - sin_a * x1[x] + cos_a * x2[x])
            start_x = int(end_x - w)
            start_y = int(end_y - h)
            rects.append((start_x, start_y, end_x, end_y))
            confidences.append(float(scores_data[x]))
    return (rects, confidences)

def _east_scale_and_nms(rects: list[tuple[int, int, int, int]], confidences: list[float], orig_w: int, orig_h: int, net_w: int, net_h: int, nms_threshold: float) -> list[tuple[int, int, int, int]]:
    r_w = orig_w / float(net_w)
    r_h = orig_h / float(net_h)
    boxes_xywh: list[list[int]] = []
    scores: list[float] = []
    for i, conf in enumerate(confidences):
        sx, sy, ex, ey = rects[i]
        sx = int(sx * r_w)
        sy = int(sy * r_h)
        ex = int(ex * r_w)
        ey = int(ey * r_h)
        bw = max(1, ex - sx)
        bh = max(1, ey - sy)
        boxes_xywh.append([sx, sy, bw, bh])
        scores.append(conf)
    if not boxes_xywh:
        return []
    idx = cv2.dnn.NMSBoxes(boxes_xywh, scores, score_threshold=0.0, nms_threshold=nms_threshold)
    if idx is None or len(idx) == 0:
        return []
    idx = np.asarray(idx).flatten()
    out: list[tuple[int, int, int, int]] = []
    for i in idx:
        x, y, bw, bh = boxes_xywh[int(i)]
        x = max(0, min(x, orig_w - 1))
        y = max(0, min(y, orig_h - 1))
        bw = min(bw, orig_w - x)
        bh = min(bh, orig_h - y)
        out.append((x, y, bw, bh))
    return out

class EASTDetector:

    def __init__(self, model_path: Path | None=None):
        path = model_path or config.EAST_MODEL_PATH
        self.net: Optional[cv2.dnn_Net] = None
        if path.is_file():
            self.net = cv2.dnn.readNet(str(path))
            self.layer_names = ['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3']

    @property
    def available(self) -> bool:
        return self.net is not None

    def detect(self, bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        if self.net is None:
            return []
        h, w = bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(bgr, scalefactor=1.0, size=(config.EAST_WIDTH, config.EAST_HEIGHT), mean=(123.68, 116.78, 103.94), swapRB=True, crop=False)
        self.net.setInput(blob)
        scores, geometry = self.net.forward(self.layer_names)
        rects, confidences = _decode_east_raw(scores, geometry, config.EAST_CONFIDENCE)
        return _east_scale_and_nms(rects, confidences, w, h, config.EAST_WIDTH, config.EAST_HEIGHT, config.EAST_NMS_THRESHOLD)

def detect_text_regions_morphology(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, th = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    grad = cv2.Sobel(th, cv2.CV_8U, 1, 0, ksize=3)
    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(grad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions: list[tuple[int, int, int, int]] = []
    for c in cnts:
        x, y, rw, rh = cv2.boundingRect(c)
        area = rw * rh
        if area < config.TEXT_MIN_AREA or rw < 10 or rh < 8:
            continue
        ar = rw / float(rh)
        if ar < config.TEXT_MIN_ASPECT or ar > config.TEXT_MAX_ASPECT:
            continue
        regions.append((x, y, rw, rh))
    return regions

def merge_text_regions(east: list[tuple[int, int, int, int]], morph: list[tuple[int, int, int, int]], prefer_east: bool) -> list[tuple[int, int, int, int]]:
    if prefer_east and east:
        return east
    if east and (not morph):
        return east
    if morph and (not east):
        return morph
    return east + morph if east else morph
