"""Exercise both dashboard sources, filters, details and the invoice processing tab."""
from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_extracted_records_and_empty_filter():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=30)
    assert not app.exception
    assert app.title[0].value == 'Automated Invoice Processing and Anomaly Detection'
    assert len(app.tabs) == 2
    assert any(metric.label == 'Invoices' for metric in app.metric)
    assert len(app.get('download_button')) == 1
    app.multiselect[0].set_value([]).run()
    assert not app.exception
    assert any('Displaying 0' in item.value for item in app.markdown)


def test_dashboard_synthetic_demo_and_vendor_filter():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=30)
    app.radio[0].set_value('Synthetic demonstration').run(timeout=30)
    assert not app.exception
    assert any('SYNTHETIC DEMO ONLY' in warning.value for warning in app.warning)
    assert app.metric[0].value == '17'
    assert any(metric.label == 'High priority' and int(metric.value) >= 1 for metric in app.metric)
    app.text_input[0].set_value('no such vendor').run()
    assert not app.exception
    assert any('Displaying 0' in item.value for item in app.markdown)
