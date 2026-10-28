"""Evidence preservation and conservative field extraction regressions."""
import json
from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np
import pytest

import ocr_engine
import pipeline
from parser import parse_invoice
from preprocessing import preprocess_image


@pytest.mark.parametrize('name', ['acme.png', 'acme_multipage.pdf', 'ambiguous.png', 'euro.png'])
def test_authored_sample_fields_against_verified_ground_truth(name):
    root = Path(__file__).resolve().parents[1] / 'samples/phase3'
    truth = json.loads((root/'ground_truth.json').read_text(encoding='utf-8'))['documents'][name]
    invoice = parse_invoice((root/truth['transcript']).read_text(encoding='utf-8'))
    assert invoice.invoice_number == truth['invoice_number']
    assert (invoice.invoice_date.isoformat() if invoice.invoice_date else None) == truth['invoice_date']
    assert invoice.total_amount == Decimal(truth['total_amount'])
    assert invoice.currency == truth['currency']


def test_dash_fallback_rereads_pixels_and_preserves_original(invoice_path, monkeypatch):
    data = {'text': ['INVOICE', 'INV\u20141001'], 'conf': [95, 60],
            'page_num': [1, 1], 'block_num': [1, 1], 'par_num': [1, 1], 'line_num': [1, 1],
            'left': [100, 300], 'top': [120, 120], 'width': [150, 200], 'height': [40, 40]}
    monkeypatch.setattr(ocr_engine, 'configure_tesseract', lambda: 'tesseract')
    monkeypatch.setattr(ocr_engine.pytesseract, 'image_to_data', lambda *a, **kw: data)
    calls = []
    def raw_line(image, **kwargs):
        calls.append(kwargs['config'])
        assert image.width > 415  # retain whitespace for raw-line normalization
        return 'INVOICE INV-1001'
    monkeypatch.setattr(ocr_engine.pytesseract, 'image_to_string', raw_line)
    result = ocr_engine.extract_text(invoice_path)
    assert result.text == 'INVOICE INV-1001'
    assert result.raw_text == 'INVOICE INV\u20141001'
    assert calls == ['--psm 13']
    assert result.review_reasons


@pytest.mark.parametrize('name', ['acme.png', 'acme_multipage.pdf', 'ambiguous.png', 'euro.png'])
def test_real_sample_recognition_matches_source_or_is_unresolved(name, tmp_path):
    try:
        ocr_engine.configure_tesseract()
    except ocr_engine.TesseractNotFoundError as exc:
        pytest.skip(str(exc))
    root = Path(__file__).resolve().parents[1] / 'samples/phase3'
    truth = json.loads((root/'ground_truth.json').read_text(encoding='utf-8'))['documents'][name]
    import hashlib
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == truth['sha256']
    frame = pipeline.run_pipeline([root/name], csv_path=tmp_path/'out.csv',
                                 raw_dir=tmp_path/'raw', processed_dir=tmp_path/'processed')
    row = frame.iloc[0]
    for field in ('invoice_number', 'invoice_date', 'total_amount'):
        actual = row[field]
        if actual is not None:
            assert actual == truth[field]
    if name != 'ambiguous.png':
        assert row.total_amount == truth['total_amount']
        assert row.invoice_date == truth['invoice_date']
        assert row.invoice_number == truth['invoice_number']
    else:
        assert row.invoice_date is None
    assert row.status == 'warning'
    assert json.loads(row.field_review)


def digital_pdf(path, invisible=False):
    """Minimal vector text fixture, independent of optional PDF writers."""
    stream = (f'BT /F1 18 Tf {3 if invisible else 0} Tr 40 760 Td '
              '(Invoice No: DIG-1001) Tj 0 -40 Td (Date: 2026-10-27) Tj '
              '0 -40 Td (Grand Total: USD 137.50) Tj ET').encode()
    objects = [b'<< /Type /Catalog /Pages 2 0 R >>',
               b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
               b'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>',
               b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
               b'<< /Length '+str(len(stream)).encode()+b' >>\nstream\n'+stream+b'\nendstream']
    content = b'%PDF-1.4\n'
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(content))
        content += f'{index} 0 obj\n'.encode()+obj+b'\nendobj\n'
    start = len(content)
    content += b'xref\n0 6\n0000000000 65535 f \n'
    content += b''.join(f'{offset:010d} 00000 n \n'.encode() for offset in offsets[1:])
    content += f'trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n'.encode()
    path.write_bytes(content)


def test_digital_pdf_uses_text_without_tesseract(tmp_path, monkeypatch):
    source = tmp_path / 'digital.pdf'
    digital_pdf(source)
    def forbidden(*args, **kwargs):
        pytest.fail('Digital text should not call Tesseract')
    monkeypatch.setattr(ocr_engine, 'configure_tesseract', forbidden)
    result = ocr_engine.process_invoice(source, tmp_path / 'processed')[0]
    assert result.method == 'pdf_text'
    invoice = parse_invoice(result.text)
    assert invoice.invoice_number == 'DIG-1001'
    assert invoice.invoice_date.isoformat() == '2026-10-27'
    assert invoice.total_amount == Decimal('137.50')
    assert result.mean_confidence is None


def test_hidden_pdf_text_is_not_trusted(tmp_path):
    source = tmp_path / 'hidden.pdf'
    digital_pdf(source, invisible=True)
    assert ocr_engine.reliable_pdf_text(source) == {}


def test_image_pdf_does_not_claim_direct_text(tmp_path, invoice_pixels):
    from PIL import Image
    source = tmp_path / 'scan.pdf'
    Image.fromarray(invoice_pixels).save(source, resolution=150)
    assert ocr_engine.reliable_pdf_text(source) == {}


def test_mixed_pdf_selects_method_per_page(tmp_path, invoice_pixels, monkeypatch):
    import pypdfium2 as pdfium
    from PIL import Image
    digital = tmp_path / 'digital.pdf'
    scan = tmp_path / 'scan.pdf'
    digital_pdf(digital)
    Image.fromarray(invoice_pixels).save(scan, resolution=150)
    mixed = tmp_path / 'mixed.pdf'
    with pdfium.PdfDocument(str(digital)) as doc, pdfium.PdfDocument(str(scan)) as image_doc:
        doc.import_pages(image_doc)
        doc.save(str(mixed))
    monkeypatch.setattr(ocr_engine, 'extract_text', lambda path, **kw:
                        ocr_engine.OCRResult(path, 'scanned page', 90, (90,), 2))
    results = ocr_engine.process_invoice(mixed, tmp_path / 'processed')
    assert [r.method for r in results] == ['pdf_text', 'ocr']
    assert results[1].text == 'scanned page'


def test_clean_preprocessing_preserves_strokes(invoice_pixels):
    expected = cv2.cvtColor(invoice_pixels, cv2.COLOR_BGR2GRAY)
    assert np.array_equal(preprocess_image(invoice_pixels, deskew=False), expected)


def test_auto_threshold_handles_uneven_background():
    image = np.tile(np.linspace(70, 255, 1600).astype(np.uint8), (500, 1))
    cv2.putText(image, 'Invoice 123', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 3)
    result = preprocess_image(image, deskew=False)
    assert set(np.unique(result)) == {0, 255}


def test_multiline_labels_and_no_cross_label_guessing():
    invoice = parse_invoice('Invoice No:\nINV-1001\nDate:\n2026-10-27\nGrand Total:\nUSD 137.50')
    assert invoice.invoice_number == 'INV-1001'
    assert invoice.invoice_date.isoformat() == '2026-10-27'
    assert invoice.total_amount == Decimal('137.50')
    missing = parse_invoice('Invoice No:\nDate: 2026-10-27\nGrand Total: USD 137.50')
    assert missing.invoice_number is None


def test_ambiguous_ocr_amounts_are_never_repaired():
    assert parse_invoice('Grand Total: USD 137.58').total_amount == Decimal('137.58')
    assert parse_invoice('Grand Total: USD 137.5O').total_amount is None
    assert parse_invoice('Grand Total: USD 137.58\nGrand Total: USD 137.50').total_amount is None
    assert parse_invoice('Date: 03/04/2026').invoice_date is None
    assert parse_invoice('Invoice No: INV-10@1').invoice_number is None


def test_uncertain_recognition_keeps_raw_text_and_review(tmp_path, invoice_path, monkeypatch):
    raw = 'Invoice No: INV-1001\nDate: 2026-10-27\nGrand Total: USD 137.58'
    selected = raw.replace('137.58', '137.50')
    result = ocr_engine.OCRResult(invoice_path, selected, 96, (96,), 10,
        raw_text=raw, review_reasons=('OCR disagreement requires review.',),
        line_evidence=({'bounds': (1, 2, 100, 30), 'original': raw, 'selected': selected},))
    monkeypatch.setattr(pipeline, 'process_invoice', lambda *a, **kw: [result])
    frame = pipeline.run_pipeline([invoice_path], csv_path=tmp_path/'out.csv',
                                 raw_dir=tmp_path/'raw', processed_dir=tmp_path/'processed')
    row = frame.iloc[0]
    assert Path(row.raw_text_file).read_text(encoding='utf-8') == raw
    assert row.total_amount == '137.50'
    assert row.status == 'warning'
    assert json.loads(row.field_review)['total_amount'] == 'manual_review_required'
    evidence = json.loads(Path(row.recognition_evidence_file).read_text(encoding='utf-8'))
    assert evidence[0]['original_text'] == raw
    assert evidence[0]['selected_text'] == selected
