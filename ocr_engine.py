"""Tesseract discovery and page-level text extraction with word confidence."""

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import re

from PIL import Image
import pytesseract
import pypdfium2 as pdfium

from preprocessing import DEFAULT_OUTPUT, preprocess_invoice


class OCRError(RuntimeError):
    """OCR cannot run or an OCR subprocess failed."""


class TesseractNotFoundError(OCRError):
    """The separately installed Tesseract executable is unavailable."""


@dataclass(frozen=True)
class OCRResult:
    image_path: Path
    text: str
    mean_confidence: float | None
    word_confidences: tuple[float, ...]
    word_count: int
    raw_text: str | None = None
    method: str = "ocr"
    review_reasons: tuple[str, ...] = ()
    line_evidence: tuple[dict, ...] = ()


def configure_tesseract() -> str:
    """Find Tesseract, preferring an explicit environment override."""
    override = os.environ.get("TESSERACT_CMD")
    if override:
        candidate = Path(os.path.expandvars(override.strip().strip('"'))).expanduser()
        command = str(candidate) if candidate.is_file() else shutil.which(str(candidate))
        if not command:
            raise TesseractNotFoundError(
                f"TESSERACT_CMD points to a missing executable: {override}. "
                "Set it to the full path of tesseract.exe."
            )
    else:
        command = shutil.which("tesseract")
        if not command:
            candidates = [
                Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Tesseract-OCR/tesseract.exe",
                Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Tesseract-OCR/tesseract.exe",
                Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local"))) / "Programs/Tesseract-OCR/tesseract.exe",
            ]
            command = next((str(path) for path in candidates if path.is_file()), None)
        if not command:
            raise TesseractNotFoundError(
                "Tesseract OCR is not installed or could not be found. Install the "
                "Windows Tesseract executable and English language data, then add "
                "its folder to PATH or set $env:TESSERACT_CMD = "
                "'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'. "
                "Installing pytesseract alone does not install Tesseract."
            )
    pytesseract.pytesseract.tesseract_cmd = command
    return command


def extract_text(image_path: Path, *, lang: str = "eng", psm: int = 3,
                 timeout: float = 60) -> OCRResult:
    """OCR a processed page; confidence is Tesseract's 0–100 word score.

    The mean is unweighted across nonempty words with valid confidence; it is
    not a calibrated probability of correctness. Blank pages return None.
    """
    if not 3 <= psm <= 13:
        raise ValueError("psm must be between 3 and 13.")
    if timeout <= 0:
        raise ValueError("OCR timeout must be positive.")
    configure_tesseract()
    image_path = Path(image_path)
    try:
        with Image.open(image_path) as image:
            data = pytesseract.image_to_data(
                image, lang=lang, config=f"--psm {psm}",
                output_type=pytesseract.Output.DICT, timeout=timeout,
            )
    except pytesseract.TesseractNotFoundError as exc:
        raise TesseractNotFoundError("Tesseract could not start. Check TESSERACT_CMD and PATH.") from exc
    except (OSError, RuntimeError) as exc:
        raise OCRError(
            f"OCR failed for '{image_path}': {exc}. Check the executable, "
            f"language data for '{lang}', and timeout ({timeout}s)."
        ) from exc
    lines: dict[tuple[int, int, int, int], list[str]] = {}
    indices: dict[tuple, list[int]] = {}
    confidences = []
    word_count = 0
    for index, value in enumerate(data["text"]):
        word = value.strip()
        if not word:
            continue
        word_count += 1
        key = tuple(data[name][index] for name in ("page_num", "block_num", "par_num", "line_num"))
        lines.setdefault(key, []).append(word)
        indices.setdefault(key, []).append(index)
        try:
            confidence = float(data["conf"][index])
        except (ValueError, TypeError):
            continue
        if 0 <= confidence <= 100:
            confidences.append(confidence)
    original = "\n".join(" ".join(words) for words in lines.values())
    selected, evidence, reasons = [], [], []
    with Image.open(image_path) as image:
        for key, words in lines.items():
            text = " ".join(words)
            chosen = text
            record = {"original": text, "selected": text}
            # TSV geometry keeps a label and its value on the same source line.
            if all(name in data for name in ("left", "top", "width", "height")):
                ids = indices[key]
                bounds = (max(0, min(data['left'][i] for i in ids)-15),
                          max(0, min(data['top'][i] for i in ids)-12),
                          min(image.width, max(data['left'][i]+data['width'][i] for i in ids)+15),
                          min(image.height, max(data['top'][i]+data['height'][i] for i in ids)+12))
                record['bounds'] = bounds
                label = re.match(r'^(?:invoice|bill|date\b|grand\s+total|total\s+amount|total\b)', text, re.I)
                uncertain_dash = bool(re.search(r'\bINV\S*[\u2013\u2014\ufffd]', text, re.I))
                if label or '\ufffd' in text or uncertain_dash:
                    mode = 13 if '\ufffd' in text or uncertain_dash else 7
                    if mode == 13:
                        # Raw-line recognition needs surrounding whitespace;
                        # a tight word union distorts its line normalization.
                        left, top, right, bottom = bounds
                        height = bottom - top
                        bounds = (max(0, left-height//6), max(0, top-height//8),
                                  min(image.width, right+(right-left)//2),
                                  min(image.height, bottom+height//4))
                        record['bounds'] = bounds
                    try:
                        alternate = pytesseract.image_to_string(
                            image.crop(bounds), lang=lang, config=f'--psm {mode}', timeout=timeout).strip()
                    except (OSError, RuntimeError, pytesseract.TesseractNotFoundError) as exc:
                        raise OCRError(f'Line OCR failed: {exc}') from exc
                    record.update(alternate=alternate, psm=mode)
                    if alternate and '\n' not in alternate and '\ufffd' not in alternate:
                        chosen = alternate
                    if chosen != text:
                        reasons.append('Page and line OCR disagree; review source and recognition evidence.')
                record['selected'] = chosen
            selected.append(chosen)
            evidence.append(record)
    return OCRResult(
        image_path=image_path,
        text="\n".join(selected), raw_text=original,
        review_reasons=tuple(dict.fromkeys(reasons)), line_evidence=tuple(evidence),
        mean_confidence=sum(confidences) / len(confidences) if confidences else None,
        word_confidences=tuple(confidences), word_count=word_count,
    )


def process_invoice(source: Path, output_dir: Path = DEFAULT_OUTPUT, *,
                    lang: str = "eng", psm: int = 3, timeout: float = 60,
                    dpi: int = 300, deskew: bool = True,
                    threshold: str = "auto") -> list[OCRResult]:
    """Extract reliable digital text or OCR image pages, in source order."""
    pages = preprocess_invoice(source, output_dir, dpi=dpi, deskew=deskew, threshold=threshold)
    direct = reliable_pdf_text(source) if Path(source).suffix.lower() == '.pdf' else {}
    return [OCRResult(page, direct[index], None, (), len(direct[index].split()),
                      method='pdf_text') if index in direct else
            extract_text(page, lang=lang, psm=psm, timeout=timeout)
            for index, page in enumerate(pages)]


def reliable_pdf_text(source: Path) -> dict[int, str]:
    """Use a conservative per-page gate; mixed/image pages still require OCR.

    Short, damaged or image-backed text layers are not trusted. Text extraction
    is evidence, not certification of an invoice's business correctness.
    """
    result = {}
    with pdfium.PdfDocument(str(source)) as document:
        for index in range(len(document)):
            page = document[index]
            try:
                # Images may contain fields absent from a partial/hidden layer.
                objects = list(page.get_objects())
                if any(obj.type == pdfium.raw.FPDF_PAGEOBJ_IMAGE for obj in objects):
                    continue
                if any(obj.type == pdfium.raw.FPDF_PAGEOBJ_TEXT and
                       pdfium.raw.FPDFTextObj_GetTextRenderMode(obj.raw) in
                       (pdfium.raw.FPDF_TEXTRENDERMODE_INVISIBLE, pdfium.raw.FPDF_TEXTRENDERMODE_CLIP)
                       for obj in objects):
                    continue
                textpage = page.get_textpage()
                try:
                    text = textpage.get_text_bounded().replace('\r\n', '\n')
                finally:
                    textpage.close()
                if (len(text.strip()) >= 20 and len(text.split()) >= 3
                        and not any(c == '\ufffd' or (not c.isprintable() and c not in '\n\r\t') for c in text)
                        and sum(c.isalnum() for c in text) >= 10):
                    result[index] = text
            finally:
                page.close()
    return result
