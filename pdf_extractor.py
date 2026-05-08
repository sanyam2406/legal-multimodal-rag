import os
import io
import fitz  # pymupdf
from PIL import Image
from metrics import logger
from ocr_engine import ocr_image

MIN_TEXT_CHARS = 50   # pages with fewer chars are treated as scanned
MIN_OCR_CHARS = 20    # discard OCR results shorter than this (noise/decorations)
MIN_IMAGE_DIM = 100   # skip images smaller than this in either dimension (icons)
RENDER_DPI = 300      # DPI for rendering scanned pages — 300 is the OCR sweet spot


def _render_page(page: fitz.Page) -> Image.Image:
    """Render a PDF page to a grayscale PIL image at RENDER_DPI."""
    scale = RENDER_DPI / 72
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
    return Image.open(io.BytesIO(pix.tobytes("png")))


def _embedded_images(page: fitz.Page, doc: fitz.Document) -> list[Image.Image]:
    """Yield all large embedded images from a page as PIL images."""
    images = []
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        try:
            base = doc.extract_image(xref)
            img = Image.open(io.BytesIO(base["image"]))
            if img.width >= MIN_IMAGE_DIM and img.height >= MIN_IMAGE_DIM:
                images.append(img)
        except Exception as e:
            logger.debug("[pdf_extractor] image xref=%d skip: %s", xref, e)
    return images


def extract_pdf_pages(pdf_path: str) -> list[dict]:
    """
    Parse a PDF page-by-page and return a list of content dicts:
      {"text": str, "source": str, "page_num": int, "source_type": "text"|"ocr_scan"|"ocr"}

    Per-page strategy:
    - If direct text >= MIN_TEXT_CHARS  → emit as source_type="text"
    - If direct text <  MIN_TEXT_CHARS  → render page at 300 DPI, OCR it → source_type="ocr_scan"
    - Embedded images on ANY page       → OCR each image → source_type="ocr"
      (skipped automatically when OCR is unavailable or result is too short)
    """
    source = os.path.basename(pdf_path)
    pages: list[dict] = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error("[pdf_extractor] cannot open %s: %s", pdf_path, e)
        return []

    text_count = scanned_count = image_count = 0

    for page_num, page in enumerate(doc, start=1):
        direct_text = page.get_text("text").strip()
        is_scanned = len(direct_text) < MIN_TEXT_CHARS

        if not is_scanned:
            pages.append({
                "text": direct_text,
                "source": source,
                "page_num": page_num,
                "source_type": "text",
            })
            text_count += 1
        else:
            ocr_text = ocr_image(_render_page(page))
            if len(ocr_text) >= MIN_OCR_CHARS:
                pages.append({
                    "text": ocr_text,
                    "source": source,
                    "page_num": page_num,
                    "source_type": "ocr_scan",
                })
            scanned_count += 1

        for img in _embedded_images(page, doc):
            ocr_text = ocr_image(img)
            if len(ocr_text) >= MIN_OCR_CHARS:
                pages.append({
                    "text": ocr_text,
                    "source": source,
                    "page_num": page_num,
                    "source_type": "ocr",
                })
                image_count += 1

    total_pages = len(doc)
    doc.close()
    logger.info(
        "[pdf_extractor] %s — pages=%d  text=%d  scanned=%d  image_ocr=%d",
        source, total_pages, text_count, scanned_count, image_count,
    )
    return pages
