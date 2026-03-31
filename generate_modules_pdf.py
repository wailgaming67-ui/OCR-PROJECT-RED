from __future__ import annotations
from pathlib import Path
from fpdf import FPDF
from fpdf.enums import XPos, YPos

class Doc(FPDF):

    def header(self) -> None:
        self.set_font('Helvetica', 'B', 12)
        self.cell(0, 8, 'Project Red - OCR Vision (modules and image filters)', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(2)

    def footer(self) -> None:
        self.set_y(-14)
        self.set_font('Helvetica', 'I', 9)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')

    def section(self, title: str) -> None:
        self.set_font('Helvetica', 'B', 11)
        self.cell(0, 7, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font('Helvetica', '', 10)
        self.ln(1)

    def body(self, text: str) -> None:
        self.multi_cell(0, 5, text)
        self.ln(2)

def main() -> None:
    pdf = Doc()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.section('Overview')
    pdf.body('This document lists third-party Python packages and external tools used by the OCR Vision desktop application (gui_app.py) and related pipeline code. Install core dependencies with: pip install -r requirements.txt')
    pdf.section('Core packages (requirements.txt)')
    pdf.body('opencv-python (import cv2): image and video I/O, color conversion (BGR/RGB), resizing, and drawing. Used throughout camera preview, preprocessing, and OCR input.\n\nnumpy (import numpy as np): ndarray representation of frames and images; required by OpenCV, pytesseract, EasyOCR, and MediaPipe.\n\npytesseract: Python bindings for the Tesseract OCR engine. Calls image_to_string and image_to_data on grayscale or binary images. Requires a separate Tesseract install (tesseract executable on PATH or TESSERACT_CMD / GUI setting).\n\nPillow (PIL): image handling where the stack expects PIL-compatible interfaces (often used with Qt and general image utilities).\n\nPySide6: Qt6 bindings for Python. Provides the desktop GUI (windows, widgets, threads, settings) for OCR Vision.\n\nlangdetect: optional language detection on OCR output text (guess_language_from_text in ocr_recognize.py); needs enough text to be reliable.')
    pdf.section('Optional: EasyOCR')
    pdf.body('Package: easyocr (pip install easyocr). Neural scene-text recognition with many languages; typically pulls in PyTorch. Loaded lazily in ocr_recognize.py via get_easyocr_reader() and recognize_easyocr(). Enable from the GUI when available.')
    pdf.section('Optional: MediaPipe (vision, not OCR)')
    pdf.body('Package: mediapipe. Face and hand landmark detection uses the MediaPipe Tasks API (mediapipe.tasks.python.vision), not the legacy mediapipe.solutions API. Model .task files are cached under models/mediapipe/. May not install on all Python versions (see requirements comments).')
    pdf.section('Optional: Ultralytics / YOLO-World')
    pdf.body('Commented in requirements.txt: ultralytics for open-vocabulary detection in advanced GUI paths; requires PyTorch (CUDA optional for GPU).')
    pdf.section('External non-Python dependency')
    pdf.body('Tesseract OCR: standalone OCR engine (e.g. Windows installer from UB-Mannheim). pytesseract invokes this binary; it is not installed by pip.')
    pdf.section('Project Python modules (local)')
    pdf.body('ocr_recognize.py: Tesseract and EasyOCR helpers, language guessing.\nvision_mediapipe.py: MediaPipe face/hand drawing and task setup.\ngui_app.py: main application wiring camera, OCR, and UI.\ndetection.py, preprocess.py, main.py: pipeline for detection and preprocessing.\nconfig.py, gestures.py, advanced_perception.py: configuration and features.')
    pdf.add_page()
    pdf.section('Image filters - OCR preprocessing (preprocess.py)')
    pdf.body('The function pipeline_for_ocr() builds a chain tuned for Tesseract: it outputs (gray_for_display, binary_for_ocr). Steps run in order:\n\n1) Grayscale (to_gray): Converts BGR camera frames to a single luminance channel. OCR classically works on gray or binary data; color does not help Tesseract here.\n\n2) Bilateral filter (reduce_blur_bilateral): Reduces noise and small speckles while keeping edges fairly sharp. Unlike a plain blur, it limits smoothing across strong edges, which helps keep letter boundaries clear on slightly noisy or soft video.\n\n3) CLAHE (enhance_contrast_clahe): Contrast Limited Adaptive Histogram Equalization boosts local contrast in 8x8 tiles (clip limit 2.0). Purpose: even out shadows, backlight, and uneven lighting so faint strokes stay visible without blowing out bright areas.\n\n4) Unsharp mask (unsharp_mask): Blurs a copy with Gaussian blur, then blends original minus blurred (addWeighted). This sharpens edges and thin strokes so characters look crisper before binarization.\n\n5) Adaptive threshold (adaptive_threshold_ocr): Turns the gray image into black/white. ADAPTIVE_THRESH_GAUSSIAN_C uses a neighborhood (block size 31) so each region gets its own threshold; constant 11 shifts the threshold. Purpose: robust text segmentation when lighting varies across the frame (better than one global threshold).\n\n6) Morphological close (morphological_cleanup): MORPH_CLOSE with a small 2x2 rectangle kernel fills tiny holes and breaks inside characters, producing cleaner glyphs for the OCR engine.')
    pdf.section('Image filters - text region boxes without EAST (detection.py)')
    pdf.body('When the EAST neural model is missing or skipped, detect_text_regions_morphology() proposes rough rectangles around text-like areas (for overlays / merging), using a different pipeline than the full-frame OCR binary above:\n\n1) Black-hat (MORPH_BLACKHAT): With a wide short rectangular kernel (25x5), this highlights dark strokes on a light background (typical printed text).\n\n2) Otsu threshold: THRESH_BINARY + THRESH_OTSU auto-picks a threshold on the black-hat image to separate ink-like regions.\n\n3) Sobel (horizontal gradient): Sobel in x emphasizes vertical edges of character strokes, helping form coherent blobs along text lines.\n\n4) Morphological close: Same kernel connects nearby horizontal fragments into regions.\n\n5) Contours: External contours become candidate boxes; small or badly proportioned regions are discarded using min area and aspect ratio limits in config.\n\nEAST path: When the EAST .pb model is present, text boxes come from a convolutional network (blob mean-subtraction and NMS), not from this morphology chain.')
    pdf.section('Camera OCR resize (gui_app.py)')
    pdf.body('Before sending binary frames to Tesseract or EasyOCR from the live camera, frames may be downscaled (max width about 720-960 px) with INTER_AREA interpolation to limit CPU load while keeping enough detail for OCR.')
    out = Path(__file__).resolve().parent / 'OCR_Vision_Modules.pdf'
    pdf.output(str(out))
    print(f'Wrote {out}')
if __name__ == '__main__':
    main()
