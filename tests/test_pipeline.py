import json
from pathlib import Path

import pandas as pd
import pytest

import main
from ocr_engine import OCRError, OCRResult
import pipeline
import ocr_engine

SAMPLE = Path(__file__).resolve().parents[1] / "samples/phase3/acme.txt"


def run(sources, tmp_path, **kwargs):
    return pipeline.run_pipeline(sources, csv_path=tmp_path / "invoices.csv",
                                 processed_dir=tmp_path / "processed", raw_dir=tmp_path / "raw", **kwargs)


def test_text_csv_roundtrip_and_unique_documents(tmp_path):
    first, copy, other = (tmp_path / name for name in ("first.txt", "copy.txt", "other.txt"))
    content = SAMPLE.read_bytes()
    first.write_bytes(content)
    copy.write_bytes(content)
    other.write_bytes(content.replace(b"ACME Supplies", b"Another Company"))
    frame = run([first, copy, first, other], tmp_path, text_input=True)
    assert len(frame) == 2
    assert len(json.loads(frame.iloc[0]["source_files"])) == 2
    assert frame.iloc[0]["invoice_number"] == frame.iloc[1]["invoice_number"] == "INV-1001"
    assert frame.iloc[0]["total_amount"] == "137.50"
    assert frame.iloc[0]["status"] == "ok"
    csv = pd.read_csv(tmp_path / "invoices.csv")
    assert json.loads(csv.iloc[0]["line_items"])[0]["description"] == "Printer paper"
    assert Path(frame.iloc[0]["raw_text_file"]).read_text(encoding="utf-8") == content.decode()


def test_multipage_ocr_one_row(tmp_path, invoice_path, monkeypatch):
    lines = SAMPLE.read_text(encoding="utf-8").splitlines()
    def fake_ocr(source, output_dir, **kwargs):
        return [OCRResult(source, "\n".join(lines[:4]), 90, (90,), 1),
                OCRResult(source, "\n".join(lines[4:]), 80, (80, 80), 2)]
    monkeypatch.setattr(pipeline, "process_invoice", fake_ocr)
    frame = run([invoice_path], tmp_path)
    assert len(frame) == 1 and frame.iloc[0]["page_count"] == 2
    assert frame.iloc[0]["total_amount"] == "137.50"
    assert frame.iloc[0]["ocr_mean_confidence"] == pytest.approx(250 / 3)
    assert "Grand Total" in Path(frame.iloc[0]["raw_text_file"]).read_text(encoding="utf-8")


def test_preprocessing_to_csv_with_only_tesseract_mocked(tmp_path, invoice_path, monkeypatch):
    lines = SAMPLE.read_text(encoding="utf-8").splitlines()
    data = {key: [] for key in ("text", "conf", "page_num", "block_num", "par_num", "line_num")}
    for number, line in enumerate(lines, 1):
        for word in line.split():
            for key, value in {"text": word, "conf": 90, "page_num": 1,
                               "block_num": 1, "par_num": 1, "line_num": number}.items():
                data[key].append(value)
    monkeypatch.setattr(ocr_engine, "configure_tesseract", lambda: "tesseract")
    def fake_tesseract(image, **kwargs):
        assert image.mode == "L"
        return data
    monkeypatch.setattr(ocr_engine.pytesseract, "image_to_data", fake_tesseract)
    frame = run([invoice_path], tmp_path)
    assert frame.iloc[0]["status"] == "ok"
    assert frame.iloc[0]["invoice_number"] == "INV-1001"
    assert frame.iloc[0]["total_amount"] == "137.50"
    assert len(list((tmp_path / "processed").glob("*.png"))) == 1
    assert len(json.loads(frame.iloc[0]["line_items"])) == 2


def test_failures_do_not_drop_documents(tmp_path, invoice_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OCRError("Tesseract missing")
    monkeypatch.setattr(pipeline, "process_invoice", fail)
    frame = run([invoice_path], tmp_path)
    assert frame.iloc[0]["status"] == "error"
    assert frame.iloc[0]["invoice_number"] is None
    assert "Tesseract missing" in frame.iloc[0]["warnings"]
    assert frame.iloc[0]["raw_text_file"] is None


def test_missing_fields_written_as_null(tmp_path):
    source = tmp_path / "empty.txt"
    source.write_text("No invoice data", encoding="utf-8")
    frame = run([source], tmp_path, text_input=True)
    assert frame.iloc[0]["status"] == "warning"
    csv = pd.read_csv(tmp_path / "invoices.csv", keep_default_na=False)
    assert csv.iloc[0]["invoice_number"] == "null"
    assert csv.iloc[0]["invoice_date"] == "null"
    assert csv.iloc[0]["line_items"] == "null"
    assert json.loads(csv.iloc[0]["warnings"])


def test_discovery_and_cli_text_mode(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    (source / "sample.txt").write_bytes(SAMPLE.read_bytes())
    (source / "ignore.md").write_text("ignored")
    assert len(pipeline.discover_sources(source, text_input=True)) == 1
    assert main.main([str(source), "--text-input", "--csv", str(tmp_path / "invoices.csv"),
                      "--raw-dir", str(tmp_path / "raw")]) == 0
    with pytest.raises(FileNotFoundError):
        pipeline.discover_sources(tmp_path / "missing")
    with pytest.raises(ValueError):
        pipeline.discover_sources(source / "sample.txt")


def test_empty_directory_does_not_replace_export(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    target = tmp_path / "invoices.csv"
    target.write_text("existing output")
    with pytest.raises(SystemExit):
        main.main([str(source), "--csv", str(target)])
    assert target.read_text() == "existing output"
