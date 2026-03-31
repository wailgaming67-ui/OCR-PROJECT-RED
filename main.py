from __future__ import annotations
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import config
import detection
import ocr_recognize
import preprocess

def draw_faces(canvas: np.ndarray, faces: list[tuple[int, int, int, int]]) -> None:
    for x, y, w, h in faces:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(canvas, 'face', (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

def draw_text_boxes(canvas: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> None:
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 128, 0), 2)
        cv2.putText(canvas, f'txt{i}', (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 128, 0), 1)

def build_pipeline_visualization(original: np.ndarray, gray_proc: np.ndarray, binary: np.ndarray, overlay: np.ndarray, scale: float=0.5) -> np.ndarray:
    g3 = cv2.cvtColor(gray_proc, cv2.COLOR_GRAY2BGR)
    b3 = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    h = min(original.shape[0], g3.shape[0], b3.shape[0], overlay.shape[0])

    def crop(img: np.ndarray) -> np.ndarray:
        return img[:h, :]
    row1 = np.hstack([crop(original), crop(overlay)])
    row2 = np.hstack([crop(g3), crop(b3)])
    vis = np.vstack([row1, row2])
    if scale < 1.0:
        vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return vis

def stack_show(original: np.ndarray, gray_proc: np.ndarray, binary: np.ndarray, overlay: np.ndarray, scale: float=0.5) -> None:
    vis = build_pipeline_visualization(original, gray_proc, binary, overlay, scale=scale)
    cv2.imshow('OCR pipeline | top: original + detections | bottom: gray + binary', vis)

def run_frame(bgr: np.ndarray, cascade: cv2.CascadeClassifier, east: detection.EASTDetector, use_easyocr: bool, easy_reader, *, run_easyocr: bool=True, ocr_lang: str='eng', preview_only: bool=False, max_side: int=1280, fast_detection: bool=False, mp_annotate=None, text_only: bool=False) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bgr = preprocess.resize_max_side(bgr, max_side)
    gray_full = preprocess.to_gray(bgr)
    gray_proc, binary = preprocess.pipeline_for_ocr(bgr)
    if text_only:
        faces: list[tuple[int, int, int, int]] = []
    else:
        faces = detection.detect_faces(gray_full, cascade)
    if fast_detection:
        east_boxes: list[tuple[int, int, int, int]] = []
    else:
        east_boxes = east.detect(bgr) if east.available else []
    morph_boxes = detection.detect_text_regions_morphology(gray_proc)
    text_boxes = detection.merge_text_regions(east_boxes, morph_boxes, prefer_east=bool(east_boxes))
    overlay = bgr.copy()
    if not text_only:
        draw_faces(overlay, faces)
    if mp_annotate is not None:
        overlay = mp_annotate(overlay)
    draw_text_boxes(overlay, text_boxes)
    if preview_only:
        return ('', overlay, gray_proc, binary, bgr)
    text_bin = ocr_recognize.recognize_tesseract(binary, lang=ocr_lang)
    crop_texts: list[str] = []
    for x, y, w, h in text_boxes[:20]:
        pad = 4
        x0, y0 = (max(0, x - pad), max(0, y - pad))
        x1, y1 = (min(bgr.shape[1], x + w + pad), min(bgr.shape[0], y + h + pad))
        crop_gray = gray_proc[y0:y1, x0:x1]
        if crop_gray.size == 0:
            continue
        t = ocr_recognize.recognize_tesseract(crop_gray, lang=ocr_lang)
        if t:
            crop_texts.append(t)
    parts = [f'[Full frame OCR]\n{text_bin}']
    if crop_texts:
        parts.append('[Regions]\n' + '\n---\n'.join(crop_texts))
    if use_easyocr and easy_reader is not None and run_easyocr:
        parts.append('[EasyOCR]\n' + ocr_recognize.recognize_easyocr(bgr, easy_reader))
    combined = '\n\n'.join(parts)
    if text_only:
        combined = '[Text & language only — no faces / objects / gestures]\n\n' + combined
    if ocr_recognize.is_tesseract_working() and '[Tesseract not found' not in text_bin:
        blob = '\n'.join([text_bin] + crop_texts)
        est = ocr_recognize.guess_language_from_text(blob)
        if est:
            combined += f'\n\n[Estimated language of recognized text: {est}]'
    return (combined, overlay, gray_proc, binary, bgr)

def main() -> None:
    parser = argparse.ArgumentParser(description='OpenCV + Tesseract OCR demo')
    parser.add_argument('--image', type=str, default=None, help='Path to image file')
    parser.add_argument('--camera', type=int, default=None, help='Camera index (e.g. 0)')
    parser.add_argument('--easyocr', action='store_true', help='Also run EasyOCR (requires pip install easyocr)')
    args = parser.parse_args()
    if args.image is None and args.camera is None:
        parser.print_help()
        print('\nProvide --image <file> or --camera <index>.')
        sys.exit(1)
    cascade = detection.load_face_cascade()
    east = detection.EASTDetector()
    if not east.available:
        print(f'EAST model not found; text regions use morphology only.\nRun: python download_east_model.py  (saves to {config.EAST_MODEL_PATH})')
    easy_reader = None
    if args.easyocr:
        easy_reader = ocr_recognize.get_easyocr_reader()
        if easy_reader is None:
            print('EasyOCR not installed. pip install easyocr')
    if args.image:
        path = Path(args.image)
        if not path.is_file():
            print(f'File not found: {path}')
            sys.exit(1)
        bgr = cv2.imread(str(path))
        if bgr is None:
            print(f'Could not read image: {path}')
            sys.exit(1)
        text, overlay, gray_proc, binary, bgr_vis = run_frame(bgr, cascade, east, args.easyocr, easy_reader)
        stack_show(bgr_vis, gray_proc, binary, overlay, scale=0.55)
        print(text)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
    if sys.platform == 'win32':
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f'Cannot open camera {args.camera}')
        sys.exit(1)
    print('q = quit, s = save snapshot  (fast preview; full OCR every 15 frames)')
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        _, overlay, gray_proc, binary, bgr_vis = run_frame(frame, cascade, east, args.easyocr, easy_reader, preview_only=True, max_side=640, fast_detection=True)
        if n % 15 == 0:
            text = ocr_recognize.recognize_tesseract(binary, lang='eng')
            print(f'[OCR]\n{text[:3000]}')
            if args.easyocr and easy_reader is not None and (n % 60 == 0):
                print('[EasyOCR]\n' + ocr_recognize.recognize_easyocr(bgr_vis, easy_reader)[:2000])
        stack_show(bgr_vis, gray_proc, binary, overlay, scale=0.45)
        key = cv2.waitKey(1) & 255
        if key == ord('q'):
            break
        if key == ord('s'):
            cv2.imwrite('snapshot_ocr.png', overlay)
    cap.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
