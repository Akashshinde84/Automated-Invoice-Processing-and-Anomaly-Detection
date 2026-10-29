"""Browser upload integration without external binaries or real invoice data."""
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from PIL import Image
from reportlab.pdfgen.canvas import Canvas
from streamlit.testing.v1 import AppTest

from analysis_config import AnalysisConfig
from invoice_analysis import analyze_invoices
from reporting import csv_report, pdf_report
from workflow import process_uploads, session_storage_root


def invoice_pdf():
    data = BytesIO()
    canvas = Canvas(data)
    text = canvas.beginText(40, 780)
    for line in ['Invoice No: WEB-1001', 'Vendor: Browser Test Supplies',
                 'Date: 2026-10-27', 'Currency: USD',
                 'Description Qty Price Total', 'Service 1 100.00 100.00',
                 'Subtotal: USD 100.00', 'Tax: USD 0.00',
                 'Shipping: USD 0.00', 'Discount: USD 0.00',
                 'Grand Total: USD 100.00']:
        text.textLine(line)
    canvas.drawText(text)
    canvas.save()
    return data.getvalue()


def test_digital_pdf_real_pipeline_partial_failure_and_reports(tmp_path, monkeypatch):
    import ocr_engine
    def no_ocr():
        raise AssertionError('Digital PDFs should not require the Tesseract executable')
    monkeypatch.setattr(ocr_engine, 'configure_tesseract', no_ocr)
    updates = []
    frame, statuses, run = process_uploads([
        ('../../invoice.pdf', invoice_pdf()), ('bad.png', b'corrupt')], tmp_path,
        progress=lambda value, message: updates.append(value))
    assert len(frame) == 1
    assert frame.iloc[0].invoice_number == 'WEB-1001'
    assert statuses.status.tolist() == ['ok', 'error']
    assert updates[-1] == 1
    assert Path(frame.iloc[0].source_file).resolve().is_relative_to(tmp_path.resolve())
    assert run.is_relative_to(tmp_path)
    analysis = analyze_invoices(frame, AnalysisConfig(as_of='2026-10-29'))
    assert analysis.iloc[0].arithmetic_status == 'CHECKED'
    assert 'isanomaly,anomalyreason' in csv_report(analysis).decode('utf-8-sig')
    assert pdf_report(analysis).startswith(b'%PDF')
    assert 'data_origin' not in frame or not frame.data_origin.eq('synthetic_demo').any()


def test_browser_storage_forces_temporary_session_even_in_local_mode(tmp_path, monkeypatch):
    import streamlit as st
    class State(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__
    monkeypatch.setattr(st, 'session_state', State())
    monkeypatch.setenv('INVOICE_STORAGE_MODE', 'local')
    monkeypatch.setattr('tempfile.mkdtemp', lambda **kwargs: str(tmp_path/'session'))
    assert session_storage_root(tmp_path/'repo') == tmp_path/'repo'
    assert session_storage_root(tmp_path/'repo', temporary=True) == tmp_path/'session'
    assert session_storage_root(tmp_path/'repo', temporary=True) == tmp_path/'session'


def browser_with_files(monkeypatch, files):
    import streamlit as st
    original = st.file_uploader
    def uploader(label, *args, **kwargs):
        if label == 'Choose invoice PDFs or images':
            assert kwargs['accept_multiple_files'] is True
            assert set(kwargs['type']) == {'pdf', 'png', 'jpg', 'jpeg'}
            return [SimpleNamespace(name=name, getvalue=lambda data=data: data) for name, data in files]
        return original(label, *args, **kwargs)
    monkeypatch.setattr(st, 'file_uploader', uploader)
    return AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=45)


@pytest.mark.parametrize('mode', ['local', 'session'])
def test_temporary_keyword_preserves_uploads_across_reruns_and_sessions(tmp_path, monkeypatch, mode):
    import inspect
    import tempfile
    import streamlit as st
    import workflow
    class State(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__
    state = State()
    monkeypatch.setattr(st, 'session_state', state)
    monkeypatch.setenv('INVOICE_STORAGE_MODE', mode)
    original = tempfile.mkdtemp
    calls = []
    def create(**kwargs):
        directory = original(dir=tmp_path, **kwargs)
        calls.append(directory)
        return directory
    monkeypatch.setattr(tempfile, 'mkdtemp', create)
    assert inspect.signature(workflow.session_storage_root).parameters['temporary'].kind is inspect.Parameter.KEYWORD_ONLY
    root = tmp_path/'repository'
    first = workflow.session_storage_root(root, temporary=True)  # Exact app.py call.
    upload = first/'invoice.pdf'
    upload.write_bytes(invoice_pdf())
    before = upload.read_bytes()
    for _ in range(3):
        assert workflow.session_storage_root(root, temporary=True) == first
        assert upload.read_bytes() == before
    assert len(calls) == 1
    if mode == 'local':
        assert workflow.session_storage_root(root) == root
        assert upload.read_bytes() == before
    monkeypatch.setattr(st, 'session_state', State())
    second = workflow.session_storage_root(root, temporary=True)
    assert first != second and second.is_dir()
    assert not (second/'invoice.pdf').exists()
    assert upload.read_bytes() == before  # A new session must not delete old files.


def test_processing_tab_runs_upload_analysis_and_pdf_download(monkeypatch):
    app = browser_with_files(monkeypatch, [('invoice.pdf', invoice_pdf())])
    assert not app.exception
    assert any(b.label == 'Process Invoices' for b in app.tabs[1].button)
    next(b for b in app.button if b.label == 'Process Invoices').click().run(timeout=45)
    assert not app.exception
    assert app.radio[0].value == 'Processed uploads'
    assert app.session_state['processed_invoices'].iloc[0].invoice_number == 'WEB-1001'
    assert not any('SYNTHETIC DEMO ONLY' in warning.value for warning in app.warning)
    assert any(table.value.get('invoice_number', pd.Series(dtype=str)).eq('WEB-1001').any() for table in app.dataframe)
    assert any(b.label == 'Download uploaded analysis CSV' for b in app.get('download_button'))
    next(b for b in app.button if b.label == 'Prepare uploaded PDF report').click().run(timeout=45)
    assert not app.exception
    assert any(b.label == 'Download uploaded PDF report' for b in app.get('download_button'))
    next(w for w in app.selectbox if w.label == 'Navigation').set_value('Invoice Records').run(timeout=45)
    assert not app.exception  # Both review forms must have independent widget keys.
    assert not any('venv' in block.value for block in app.code)


def test_all_failed_upload_shows_error_without_success_or_demo_results(monkeypatch):
    app = browser_with_files(monkeypatch, [('broken.pdf', b'%PDF-invalid')])
    next(b for b in app.button if b.label == 'Process Invoices').click().run(timeout=45)
    assert not app.exception
    assert app.session_state['processed_invoices'].empty
    assert app.radio[0].value == 'Processed uploads'
    assert not app.success
    assert any('No invoices were successfully extracted' in item.value for item in app.info)
    assert not any(b.label == 'Download uploaded analysis CSV' for b in app.get('download_button'))


def test_missing_tesseract_does_not_abort_other_documents(tmp_path, monkeypatch):
    import ocr_engine
    def missing():
        raise ocr_engine.TesseractNotFoundError('Missing Tesseract executable')
    monkeypatch.setattr(ocr_engine, 'configure_tesseract', missing)
    image = BytesIO()
    Image.new('RGB', (200, 200), 'white').save(image, format='PNG')
    frame, statuses, _ = process_uploads([
        ('scan.png', image.getvalue()), ('digital.pdf', invoice_pdf())], tmp_path)
    assert statuses.status.tolist() == ['error', 'ok']
    assert 'packages.txt' in statuses.iloc[0].message
    assert frame.status.tolist() == ['error', 'ok']
    assert frame.iloc[1].invoice_number == 'WEB-1001'


def test_permission_failure_keeps_traceback_in_server_log(tmp_path, monkeypatch, caplog):
    original = Path.mkdir
    def denied(path, *args, **kwargs):
        if 'uploads' in path.parts and path.is_relative_to(tmp_path):
            raise PermissionError(13, 'Access denied', str(path))
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'mkdir', denied)
    with pytest.raises(PermissionError):
        process_uploads([('invoice.pdf', invoice_pdf())], tmp_path)
    logged = next(r for r in caplog.records if r.message == 'Invoice upload storage initialization failed')
    assert logged.exc_info[0] is PermissionError
    assert logged.exc_info[2] is not None
    assert 'WEB-1001' not in caplog.text  # No extracted contents or upload bytes logged.


@pytest.mark.parametrize('failure', [PermissionError, OSError])
def test_upload_ui_storage_error_is_actionable_and_private(monkeypatch, caplog, failure):
    import dashboard_ui
    private_path = '/private/hidden-host/invoices/invoice.pdf'
    def failed(*args, **kwargs):
        # OSError(13, ...) is promoted to PermissionError by Python.
        raise failure(28 if failure is OSError else 13, 'Storage unavailable', private_path)
    monkeypatch.setattr(dashboard_ui, 'process_uploads', failed)
    app = browser_with_files(monkeypatch, [('invoice.pdf', invoice_pdf())])
    next(b for b in app.button if b.label == 'Process Invoices').click().run(timeout=45)
    assert not app.exception
    assert app.error and all(private_path not in item.value for item in app.error)
    assert any('server logs' in item.value for item in app.error)
    assert any(r.name == 'dashboard_ui' and r.exc_info and r.exc_info[0] is failure for r in caplog.records)
    assert not app.success


def test_batch_csv_failure_is_logged_without_fabricated_results(tmp_path, monkeypatch, caplog):
    import workflow
    def cannot_save(*args, **kwargs):
        raise OSError('Disk unavailable')
    monkeypatch.setattr(workflow, 'write_analysis', cannot_save)
    with pytest.raises(OSError, match='Disk unavailable'):
        process_uploads([('invoice.pdf', invoice_pdf())], tmp_path)
    assert any(r.message == 'Invoice upload batch persistence failed' and r.exc_info for r in caplog.records)


def test_real_os_temp_directory_accepts_representative_26kb_pdf(monkeypatch):
    """Regression for the real OS-temp path that mocked mkdtemp tests missed."""
    import tempfile
    import streamlit as st
    class State(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__
    monkeypatch.setattr(st, 'session_state', State())
    with tempfile.TemporaryDirectory(prefix='invoice-real-temp-test-') as parent:
        monkeypatch.setenv('INVOICE_STORAGE_MODE', 'local')
        root = session_storage_root(Path(parent), temporary=True)
        pdf = invoice_pdf()
        pdf += b'\n%' + b'x'*(26600-len(pdf)-2)
        assert len(pdf) == 26600
        frame, statuses, run = process_uploads([('representative.pdf', pdf)], root)
        assert statuses.status.tolist() == ['ok']
        assert frame.iloc[0].invoice_number == 'WEB-1001'
        assert (run/'extracted.csv').is_file()
        assert session_storage_root(Path(parent), temporary=True) == root
        assert Path(frame.iloc[0].source_file).read_bytes() == pdf
        # Test-owned synthetic artifacts only; production never cleans on rerun.
        import shutil
        shutil.rmtree(root)
