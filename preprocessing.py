"""OpenCV preprocessing for invoice images and rendered PDF pages."""

from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import pypdfium2 as pdfium

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output" / "processed"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf"}


class PreprocessingError(ValueError):
    """The input cannot be decoded, rendered, or processed."""


def correct_rotation(gray: np.ndarray) -> np.ndarray:
    """Correct small text-line skew (up to 15 degrees), preserving page bounds.

    This is a conservative estimate from near-horizontal edges, not detection
    of upside-down or quarter-turn pages. Pages without useful lines are kept.
    """
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=60,
        minLineLength=max(40, gray.shape[1] // 10), maxLineGap=20,
    )
    if lines is None:
        return gray
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if abs(angle) <= 15:
            angles.append(angle)
    if not angles:
        return gray
    angle = float(np.median(angles))
    if abs(angle) < 0.3:
        return gray
    height, width = gray.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = int(np.ceil(height * sine + width * cosine))
    new_height = int(np.ceil(height * cosine + width * sine))
    matrix[0, 2] += (new_width - width) / 2
    matrix[1, 2] += (new_height - height) / 2
    return cv2.warpAffine(
        gray, matrix, (new_width, new_height),
        flags=cv2.INTER_CUBIC, borderValue=255,
    )


def preprocess_image(image: np.ndarray, *, deskew: bool = True,
                     threshold: str = "auto") -> np.ndarray:
    """Preserve clean grayscale strokes; binarize uneven backgrounds as needed."""
    if image is None or image.size == 0 or min(image.shape[:2]) < 3:
        raise PreprocessingError("Image is empty or too small to process.")
    if threshold not in {"auto", "otsu", "adaptive"}:
        raise ValueError("threshold must be 'auto', 'otsu' or 'adaptive'.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    height, width = gray.shape
    scale = min(max(1.0, 1600 / width), 4000 / max(height, width))
    if abs(scale - 1) > 0.01:
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
    if deskew:
        gray = correct_rotation(gray)
    # Clean digital lettering loses strokes under unconditional Gaussian blur.
    # Preserve grayscale by default; local thresholding is useful for shadows.
    if threshold == "auto":
        background = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
        if float(np.percentile(background, 90) - np.percentile(background, 10)) < 35:
            return gray
        threshold = "adaptive"
    if threshold == "adaptive":
        gray = cv2.medianBlur(gray, 3)
    if threshold == "adaptive":
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 11)
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def preprocess_invoice(source: Path, output_dir: Path = DEFAULT_OUTPUT, *,
                       dpi: int = 300, deskew: bool = True,
                       threshold: str = "auto") -> list[Path]:
    """Save one PNG per page and return paths in document order.

    Unique run names avoid overwriting earlier results. PDF resources are closed
    per page. Unicode Windows paths use byte decoding rather than cv2.imread.
    """
    source, output_dir = Path(source), Path(output_dir)
    if not source.is_file():
        raise FileNotFoundError(f"Invoice file does not exist: {source}")
    if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise PreprocessingError("Supported formats are JPG, JPEG, PNG, and PDF.")
    if not 72 <= dpi <= 600:
        raise ValueError("PDF rendering dpi must be between 72 and 600.")
    if threshold not in {"auto", "otsu", "adaptive"}:
        raise ValueError("threshold must be 'auto', 'otsu' or 'adaptive'.")
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{source.stem}_{uuid4().hex}"
    results: list[Path] = []

    def save(image: np.ndarray, page_number: int) -> None:
        processed = preprocess_image(image, deskew=deskew, threshold=threshold)
        target = output_dir / f"{prefix}_page_{page_number:04d}.png"
        success, encoded = cv2.imencode(".png", processed)
        if not success:
            raise PreprocessingError(f"Could not encode processed page {page_number}.")
        target.write_bytes(encoded.tobytes())
        results.append(target)

    try:
        if source.suffix.lower() == ".pdf":
            with pdfium.PdfDocument(str(source)) as document:
                if len(document) == 0:
                    raise PreprocessingError(f"PDF contains no pages: {source}")
                for index in range(len(document)):
                    page = document[index]
                    try:
                        bitmap = page.render(scale=dpi / 72)
                        try:
                            rgb = np.array(bitmap.to_pil().convert("RGB"))
                            save(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), index + 1)
                        finally:
                            bitmap.close()
                    finally:
                        page.close()
        else:
            image = cv2.imdecode(np.frombuffer(source.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise PreprocessingError(f"Unreadable or corrupt invoice image: {source}")
            save(image, 1)
    except (OSError, cv2.error, pdfium.PdfiumError) as exc:
        raise PreprocessingError(f"Could not process invoice '{source}': {exc}") from exc
    return results
