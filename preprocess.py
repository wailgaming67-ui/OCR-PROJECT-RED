from __future__ import annotations
import cv2
import numpy as np

def to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()

def reduce_blur_bilateral(gray: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

def unsharp_mask(gray: np.ndarray, amount: float=1.2, sigma: float=1.0) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    sharpened = cv2.addWeighted(gray, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def enhance_contrast_clahe(gray: np.ndarray, clip_limit: float=2.0) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(gray)

def adaptive_threshold_ocr(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)

def morphological_cleanup(binary: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

def pipeline_for_ocr(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = to_gray(bgr)
    g = reduce_blur_bilateral(g)
    g = enhance_contrast_clahe(g)
    g = unsharp_mask(g)
    binary = adaptive_threshold_ocr(g)
    binary = morphological_cleanup(binary)
    return (g, binary)

def resize_max_side(image: np.ndarray, max_side: int=1600) -> np.ndarray:
    h, w = image.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return image
    scale = max_side / m
    return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
