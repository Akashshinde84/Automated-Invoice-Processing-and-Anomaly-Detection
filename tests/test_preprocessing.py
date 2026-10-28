import cv2
import numpy as np
from PIL import Image
import pytest

from preprocessing import PreprocessingError, correct_rotation, preprocess_image, preprocess_invoice


@pytest.mark.parametrize("extension", [".jpg", ".JPEG", ".png"])
@pytest.mark.parametrize("threshold", ["otsu", "adaptive"])
def test_image_formats(tmp_path, invoice_pixels, extension, threshold):
    source = tmp_path / f"facture_é{extension}"
    source.write_bytes(cv2.imencode(extension.lower(), invoice_pixels)[1].tobytes())
    pages = preprocess_invoice(source, tmp_path / "processed", threshold=threshold)
    assert len(pages) == 1 and pages[0].is_file()
    binary = cv2.imdecode(np.frombuffer(pages[0].read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    assert binary.ndim == 2
    assert set(np.unique(binary)) == {0, 255}
    assert np.count_nonzero(binary == 0) > 1000
    assert binary.shape == invoice_pixels.shape[:2]


def test_pdf_pages(tmp_path, invoice_pixels):
    source = tmp_path / "invoice.pdf"
    page = Image.fromarray(cv2.cvtColor(invoice_pixels, cv2.COLOR_BGR2RGB))
    page.save(source, save_all=True, append_images=[page], resolution=150)
    paths = preprocess_invoice(source, tmp_path / "processed", dpi=150)
    assert len(paths) == 2
    assert "page_0001" in paths[0].name and "page_0002" in paths[1].name
    for path in paths:
        assert path.stat().st_size > 1000


@pytest.mark.parametrize("extension", [".png", ".pdf"])
def test_corrupt_inputs(tmp_path, extension):
    source = tmp_path / f"broken{extension}"
    source.write_bytes(b"not an invoice")
    with pytest.raises(PreprocessingError, match="Unreadable|Could not process"):
        preprocess_invoice(source, tmp_path / "processed")


def test_missing_and_unsupported(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        preprocess_invoice(tmp_path / "missing.png")
    source = tmp_path / "invoice.txt"
    source.write_text("invoice")
    with pytest.raises(PreprocessingError, match="Supported formats"):
        preprocess_invoice(source)


def test_resize_blank_and_empty():
    result = preprocess_image(np.full((300, 400, 3), 255, np.uint8))
    assert result.shape == (1200, 1600)
    assert np.all(result == 255)
    with pytest.raises(PreprocessingError, match="empty or too small"):
        preprocess_image(np.empty((0, 0), np.uint8))


def test_deskew():
    image = np.full((700, 1400), 255, np.uint8)
    for y in range(100, 600, 70):
        cv2.line(image, (100, y), (1200, y), 0, 3)
    matrix = cv2.getRotationMatrix2D((700, 350), 7, 1)
    tilted = cv2.warpAffine(image, matrix, (1400, 700), borderValue=255)
    corrected = correct_rotation(tilted)
    lines = cv2.HoughLinesP(cv2.Canny(corrected, 50, 150), 1, np.pi / 180,
                            60, minLineLength=300, maxLineGap=20)
    assert lines is not None
    angles = [np.degrees(np.arctan2(y2-y1, x2-x1)) for x1, y1, x2, y2 in lines[:, 0]]
    assert abs(np.median(angles)) < 1


def test_does_not_overwrite(invoice_path, tmp_path):
    first = preprocess_invoice(invoice_path, tmp_path / "processed")
    original = first[0].read_bytes()
    second = preprocess_invoice(invoice_path, tmp_path / "processed")
    assert first != second
    assert first[0].read_bytes() == original


def test_invalid_options(invoice_path, tmp_path):
    with pytest.raises(ValueError, match="dpi"):
        preprocess_invoice(invoice_path, tmp_path / "processed", dpi=0)
    with pytest.raises(ValueError, match="threshold"):
        preprocess_invoice(invoice_path, threshold="invalid")
