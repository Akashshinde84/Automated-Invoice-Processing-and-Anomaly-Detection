"""Fresh checkout and session-isolated deployment regressions."""
from pathlib import Path
import shutil
import pytest
from workflow import session_storage_root,available_models


def test_local_storage_preserved_and_bad_mode_rejected(tmp_path,monkeypatch):
    monkeypatch.setenv('INVOICE_STORAGE_MODE','local')
    assert session_storage_root(tmp_path)==tmp_path
    monkeypatch.setenv('INVOICE_STORAGE_MODE','invalid')
    with pytest.raises(ValueError,match='INVOICE_STORAGE_MODE'):
        session_storage_root(tmp_path)


def test_session_storage_isolated_and_stable(tmp_path,monkeypatch):
    import streamlit as st
    state={}
    monkeypatch.setattr(st,'session_state',state)
    monkeypatch.setattr('tempfile.mkdtemp',lambda **kwargs:str(tmp_path/'private-session'))
    monkeypatch.setenv('INVOICE_STORAGE_MODE','session')
    # Use a minimal state with Streamlit's attribute interface.
    class State(dict):
        __getattr__=dict.__getitem__
        __setattr__=dict.__setitem__
    monkeypatch.setattr(st,'session_state',State())
    path=session_storage_root(tmp_path/'repo')
    assert path==tmp_path/'private-session'
    assert session_storage_root(tmp_path/'repo')==path
    assert not available_models(tmp_path/'absent')


def test_fresh_checkout_all_pages_without_outputs_or_models(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    root=Path(__file__).resolve().parents[1]
    checkout=tmp_path/'checkout';checkout.mkdir()
    for path in root.glob('*.py'):
        shutil.copy2(path,checkout/path.name)
    monkeypatch.setenv('INVOICE_STORAGE_MODE','session')
    assert not (checkout/'output').exists() and not (checkout/'models').exists()
    app=AppTest.from_file(str(checkout/'app.py')).run(timeout=45)
    assert not app.exception
    assert app.radio[0].value=='Synthetic demonstration'
    assert app.metric[0].value=='17'
    for page in ['Dashboard','Upload & Process Invoices','Invoice Records','Anomaly Detection','Reports','Settings']:
        next(w for w in app.selectbox if w.label=='Navigation').set_value(page).run(timeout=45)
        assert not app.exception,page
        assert not app.error,page
    assert not (checkout/'output').exists() and not (checkout/'models').exists()


def test_cloud_system_dependencies_are_minimal():
    root=Path(__file__).resolve().parents[1]
    packages=(root/'packages.txt').read_text().splitlines()
    assert 'tesseract-ocr' in packages and 'tesseract-ocr-eng' in packages
    assert 'poppler-utils' not in packages
    requirements=(root/'requirements.txt').read_text()
    assert 'opencv-python-headless' in requirements
    assert 'matplotlib' not in requirements and 'seaborn' not in requirements
