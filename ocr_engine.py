from PIL import Image, ImageFilter, ImageEnhance
from metrics import logger

TESSERACT_CONFIG = "--oem 3 --psm 6 -l eng"

_TESSERACT_AVAILABLE = False
try:
    import pytesseract
    pytesseract.get_tesseract_version()
    _TESSERACT_AVAILABLE = True
    logger.info("[ocr_engine] Tesseract available: %s", pytesseract.get_tesseract_version())
except Exception:
    logger.warning(
        "[ocr_engine] Tesseract not found — OCR disabled. "
        "Install: brew install tesseract  (macOS) | apt install tesseract-ocr  (Linux)"
    )


def _preprocess(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = img.filter(ImageFilter.SHARPEN)
    return img


def ocr_image(img: Image.Image) -> str:
    """Run Tesseract OCR on a PIL image. Returns empty string if OCR is unavailable."""
    if not _TESSERACT_AVAILABLE:
        return ""
    try:
        return pytesseract.image_to_string(_preprocess(img), config=TESSERACT_CONFIG).strip()
    except Exception as e:
        logger.warning("[ocr_engine] OCR failed: %s", e)
        return ""


def is_available() -> bool:
    return _TESSERACT_AVAILABLE

