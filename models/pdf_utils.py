# models/pdf_utils.py
import io
import logging
from typing import List, Dict, Any
from pdfminer.high_level import extract_text as pdfminer_extract_text

# optional OCR dependencies
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

logger = logging.getLogger(__name__)

def _clean_text(s: str) -> str:
    # basic cleaning: collapse whitespace, remove weird control chars
    return " ".join(s.replace("\x0c", " ").split())

def _try_pdfminer(pdf_bytes: bytes) -> Dict[str, Any]:
    """
    Try to extract selectable text via pdfminer. Return dict with keys 'text' and 'pages'.
    """
    try:
        txt = pdfminer_extract_text(io.BytesIO(pdf_bytes))
        if txt and txt.strip():
            pages = [p.strip() for p in txt.split("\f") if p.strip()]
            full = "\n".join([_clean_text(p) for p in pages]) if pages else _clean_text(txt)
            return {"text": full, "pages": pages}
    except Exception as e:
        logger.debug("pdfminer extraction failed: %s", e)
    return {"text": "", "pages": []}

def _try_ocr(pdf_bytes: bytes) -> Dict[str, Any]:
    """
    If OCR libs are installed and Tesseract available, convert each PDF page to image and OCR.
    Returns dict with 'text' and 'pages'.
    """
    if not OCR_AVAILABLE:
        return {"text": "", "pages": []}

    try:
        images = convert_from_bytes(pdf_bytes)
    except Exception as e:
        logger.debug("pdf2image convert_from_bytes failed: %s", e)
        return {"text": "", "pages": []}

    pages_text = []
    for img in images:
        try:
            # psm 3/6 are common; use moderate config. User can change as needed.
            config = "--oem 3 --psm 6"
            ptext = pytesseract.image_to_string(img, config=config) or ""
            pages_text.append(_clean_text(ptext))
        except Exception as e:
            logger.debug("pytesseract failed on one page: %s", e)
            pages_text.append("")

    full = "\n".join(p for p in pages_text if p.strip())
    return {"text": full, "pages": pages_text}

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> Dict[str, Any]:
    """
    Robust PDF text extractor:
      - Try selectable text (pdfminer)
      - Fallback to OCR (pdf2image + pytesseract) if that is available and selectable text is empty
    Returns:
      {"text": full_text, "pages": [page_texts], "used_ocr": bool, "selectable_text_found": bool}
    Notes:
      - If OCR is not available and pdfminer finds nothing, returns empty text.
      - This function is conservative: it prefers selectable text when present.
    """
    # 1) try selectable text
    try:
        res = _try_pdfminer(pdf_bytes)
        if res["text"] and res["text"].strip():
            return {"text": res["text"], "pages": res["pages"], "used_ocr": False, "selectable_text_found": True}
    except Exception as e:
        logger.debug("pdfminer extraction raised: %s", e)

    # 2) fallback to OCR if available
    if OCR_AVAILABLE:
        try:
            ocr_res = _try_ocr(pdf_bytes)
            if ocr_res["text"] and ocr_res["text"].strip():
                return {"text": ocr_res["text"], "pages": ocr_res["pages"], "used_ocr": True, "selectable_text_found": False}
        except Exception as e:
            logger.debug("OCR fallback raised: %s", e)

    # nothing found
    return {"text": "", "pages": [], "used_ocr": False, "selectable_text_found": False}
