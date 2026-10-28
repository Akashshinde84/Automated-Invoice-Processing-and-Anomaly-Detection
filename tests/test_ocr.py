from pathlib import Path

from PIL import Image
import pytest

import main
import ocr_engine
from ocr_engine import OCRError, TesseractNotFoundError, configure_tesseract, extract_text, process_invoice


def fake_data():
    return {
        "text": ["", "INVOICE", "INV-1001", "Total:", "100.00"],
        "conf": [-1, 90, 80, 95, 99],
        "page_num": [1]*5, "block_num": [1]*5, "par_num": [1]*5,
        "line_num": [0, 1, 1, 2, 2],
    }


def test_confidence_and_lines(invoice_path, monkeypatch):
    monkeypatch.setattr(ocr_engine, "configure_tesseract", lambda: "tesseract")
    monkeypatch.setattr(ocr_engine.pytesseract, "image_to_data", lambda *a, **k: fake_data())
    result = extract_text(invoice_path)
    assert result.text == "INVOICE INV-1001\nTotal: 100.00"
    assert result.word_count == 4
    assert result.mean_confidence == 91
    assert result.word_confidences == (90, 80, 95, 99)


def test_blank_and_invalid_confidence(invoice_path, monkeypatch):
    monkeypatch.setattr(ocr_engine, "configure_tesseract", lambda: "tesseract")
    data = fake_data()
    data["text"] = [""] * 5
    monkeypatch.setattr(ocr_engine.pytesseract, "image_to_data", lambda *a, **k: data)
    result = extract_text(invoice_path)
    assert result.text == "" and result.mean_confidence is None and result.word_count == 0
    data["text"] = ["", "A", "B", "C", "D"]
    data["conf"] = [-1, "invalid", -1, "nan", "101"]
    assert extract_text(invoice_path).mean_confidence is None


def test_environment_override(tmp_path, monkeypatch):
    executable = tmp_path / "Tesseract OCR" / "tesseract.exe"
    executable.parent.mkdir()
    executable.touch()
    monkeypatch.setenv("TESSERACT_CMD", str(executable))
    monkeypatch.setattr(ocr_engine.pytesseract.pytesseract, "tesseract_cmd", "tesseract")
    assert configure_tesseract() == str(executable)
    assert ocr_engine.pytesseract.pytesseract.tesseract_cmd == str(executable)


def test_invalid_override(monkeypatch, tmp_path):
    monkeypatch.setenv("TESSERACT_CMD", str(tmp_path / "missing.exe"))
    with pytest.raises(TesseractNotFoundError, match="TESSERACT_CMD"):
        configure_tesseract()


def test_missing_tesseract(monkeypatch):
    monkeypatch.delenv("TESSERACT_CMD", raising=False)
    monkeypatch.setattr(ocr_engine.shutil, "which", lambda command: None)
    monkeypatch.setattr(Path, "is_file", lambda path: False)
    with pytest.raises(TesseractNotFoundError, match="Installing pytesseract alone"):
        configure_tesseract()


def test_ocr_timeout(invoice_path, monkeypatch):
    monkeypatch.setattr(ocr_engine, "configure_tesseract", lambda: "tesseract")
    def fail(*args, **kwargs):
        raise RuntimeError("Tesseract process timeout")
    monkeypatch.setattr(ocr_engine.pytesseract, "image_to_data", fail)
    with pytest.raises(OCRError, match="timeout"):
        extract_text(invoice_path)


def test_pdf_pipeline(tmp_path, invoice_pixels, monkeypatch):
    source = tmp_path / "invoice.pdf"
    page = Image.fromarray(invoice_pixels)
    page.save(source, save_all=True, append_images=[page], resolution=150)
    monkeypatch.setattr(ocr_engine, "configure_tesseract", lambda: "tesseract")
    monkeypatch.setattr(ocr_engine.pytesseract, "image_to_data", lambda *a, **k: fake_data())
    results = process_invoice(source, tmp_path / "processed", dpi=150)
    assert len(results) == 2
    assert all("INV-1001" in result.text for result in results)
    assert results[0].image_path != results[1].image_path


def test_cli(invoice_path, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ocr_engine, "configure_tesseract", lambda: "tesseract")
    monkeypatch.setattr(ocr_engine.pytesseract, "image_to_data", lambda *a, **k: fake_data())
    assert main.main([str(invoice_path), "--output-dir", str(tmp_path / "processed"),
                      "--csv", str(tmp_path / "invoices.csv"),
                      "--raw-dir", str(tmp_path / "raw")]) == 0
    output = capsys.readouterr().out
    assert "INV-1001" in output and "91.0/100" in output


def test_cli_error(invoice_path, tmp_path, monkeypatch, capsys):
    def fail():
        raise TesseractNotFoundError("Install Tesseract and set TESSERACT_CMD")
    monkeypatch.setattr(ocr_engine, "configure_tesseract", fail)
    with pytest.raises(SystemExit) as error:
        main.main([str(invoice_path), "--csv", str(tmp_path / "invoices.csv"),
                   "--raw-dir", str(tmp_path / "raw")])
    assert error.value.code == 1
    assert "TESSERACT_CMD" in capsys.readouterr().err


@pytest.mark.parametrize("extension", [".png", ".jpg", ".jpeg", ".pdf"])
def test_real_ocr_known_invoice(tmp_path, invoice_pixels, extension):
    """Integration check runs automatically once real Tesseract is installed."""
    try:
        configure_tesseract()
    except TesseractNotFoundError as exc:
        pytest.skip(str(exc))
    source = tmp_path / f"known_invoice{extension}"
    Image.fromarray(invoice_pixels).save(source, resolution=150)
    results = process_invoice(source, tmp_path / "processed", dpi=150, psm=6)
    text = results[0].text.upper()
    assert "INV-1001" in text
    assert "ACME" in text and "100.00" in text
    assert results[0].word_count >= 8
    assert results[0].mean_confidence is not None
