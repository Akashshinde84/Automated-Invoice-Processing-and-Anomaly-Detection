"""Startup diagnostics must retain the actual import failure in server logs."""
import builtins
import runpy
from pathlib import Path


def test_startup_import_failure_is_logged_not_silenced(monkeypatch,caplog):
    original=builtins.__import__
    def importing(name,*args,**kwargs):
        if name=='dashboard_ui':
            raise ImportError("cannot import name 'invoice_explanation_panel' from 'dashboard_ui'",name='dashboard_ui')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',importing)
    result=runpy.run_path(str(Path(__file__).resolve().parents[1]/'app.py'),run_name='startup_diagnostic_test')
    assert 'dashboard_ui' in result['STARTUP_ERROR']
    assert 'startup logs' in result['STARTUP_ERROR']
    assert 'invoice_explanation_panel' in caplog.text
    assert 'ImportError' in caplog.text
