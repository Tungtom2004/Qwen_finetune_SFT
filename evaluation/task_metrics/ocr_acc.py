# evaluation/task_metrics/ocr_acc.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
from PIL import Image
import numpy as np
import re

"""
OCR Accuracy:
- Given predicted image and expected text (ground truth string),
  run OCR, compare with GT.

Requires either:
- pytesseract + system tesseract installed, OR
- easyocr (optional).

This wrapper defaults to pytesseract if available.
Accuracy is computed as character-level exact match ratio after normalization.
"""

@dataclass
class OCRConfig:
    backend: str = "pytesseract"  # "pytesseract" or "easyocr"
    lang: str = "eng"
    normalize_case: bool = True
    strip_non_alnum: bool = False


def _normalize_text(s: str, normalize_case: bool, strip_non_alnum: bool) -> str:
    s = s.strip()
    if normalize_case:
        s = s.lower()
    if strip_non_alnum:
        s = re.sub(r"[^a-z0-9]+", "", s)
    else:
        s = re.sub(r"\s+", " ", s)
    return s


def _char_accuracy(pred: str, gt: str) -> float:
    if len(gt) == 0:
        return 1.0 if len(pred) == 0 else 0.0
    # simple char-wise match on min length + penalty for length mismatch
    m = min(len(pred), len(gt))
    matches = sum(1 for i in range(m) if pred[i] == gt[i])
    return matches / max(len(gt), len(pred), 1)


class OCRAccuracy:
    def __init__(self, cfg: OCRConfig = OCRConfig()):
        self.cfg = cfg
        self.backend = cfg.backend.lower()

        if self.backend == "pytesseract":
            try:
                import pytesseract  # noqa
            except Exception as e:
                raise ImportError(
                    "pytesseract not available. Install: pip install pytesseract "
                    "and install Tesseract OCR on your system."
                ) from e
            self.pytesseract = pytesseract

        elif self.backend == "easyocr":
            try:
                import easyocr  # noqa
            except Exception as e:
                raise ImportError("easyocr not available. Install: pip install easyocr") from e
            self.easyocr = easyocr
            self.reader = easyocr.Reader([cfg.lang], gpu=False)
        else:
            raise ValueError("Unsupported OCR backend. Use pytesseract or easyocr.")

    def read_text(self, image: Image.Image) -> str:
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.backend == "pytesseract":
            txt = self.pytesseract.image_to_string(image, lang=self.cfg.lang)
            return txt
        else:
            arr = np.asarray(image)
            results = self.reader.readtext(arr, detail=0)
            return " ".join(results)

    def compute(self, images: List[Image.Image], gt_texts: List[str]) -> np.ndarray:
        assert len(images) == len(gt_texts)
        scores = []
        for im, gt in zip(images, gt_texts):
            pred = self.read_text(im)
            pred_n = _normalize_text(pred, self.cfg.normalize_case, self.cfg.strip_non_alnum)
            gt_n = _normalize_text(gt, self.cfg.normalize_case, self.cfg.strip_non_alnum)
            scores.append(_char_accuracy(pred_n, gt_n))
        return np.asarray(scores, dtype=np.float32)
