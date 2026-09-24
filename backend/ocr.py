"""
Slide and Screen Capture OCR module for StudyPilot.
Extracts text from lecture slides / screen capture images using pytesseract and Pillow.
"""

from __future__ import annotations
import os
import shutil
import logging
from typing import Optional, List
from PIL import Image

logger = logging.getLogger(__name__)


class OCRError(Exception):
    """Raised when OCR processing fails."""
    pass


class SlideOCR:
    """OCR processor for lecture slides and screen captures."""

    def __init__(self, tesseract_cmd: Optional[str] = None):
        import pytesseract
        self.pytesseract = pytesseract

        # Determine Tesseract executable path
        default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        default_path_x86 = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

        if tesseract_cmd and os.path.exists(tesseract_cmd):
            self.cmd = tesseract_cmd
        elif os.path.exists(default_path):
            self.cmd = default_path
        elif os.path.exists(default_path_x86):
            self.cmd = default_path_x86
        else:
            self.cmd = shutil.which("tesseract")

        if self.cmd:
            self.pytesseract.pytesseract.tesseract_cmd = self.cmd

    def is_available(self) -> bool:
        if not self.cmd or not os.path.exists(self.cmd):
            return False
        try:
            self.pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def extract_text(self, image_path: str) -> str:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if not self.is_available():
            raise OCRError(
                "Tesseract OCR executable not found or not working. "
                r"Please install Tesseract OCR at 'C:\Program Files\Tesseract-OCR\tesseract.exe' "
                "or add it to system PATH."
            )

        try:
            img = Image.open(image_path)
            extracted = self.pytesseract.image_to_string(img)
            cleaned = extracted.strip()
            return cleaned
        except Exception as e:
            logger.error(f"OCR failed for {image_path}: {e}")
            raise OCRError(f"OCR processing failed: {str(e)}") from e


def extract_slides_text(image_paths: List[str], tesseract_cmd: Optional[str] = None) -> str:
    """Convenience function to run OCR across multiple uploaded slide images."""
    if not image_paths:
        return ""

    ocr = SlideOCR(tesseract_cmd=tesseract_cmd)
    results = []

    for idx, img_path in enumerate(image_paths, 1):
        if not img_path:
            continue
        try:
            text = ocr.extract_text(img_path)
            if text:
                results.append(f"--- Slide {idx} ---\n{text}")
        except Exception as e:
            results.append(f"--- Slide {idx} ---\n[OCR Error: {str(e)}]")

    return "\n\n".join(results)
