"""Public terminology and compatibility with existing application integrations."""
import importlib
from pathlib import Path
import re

import pytest


@pytest.mark.parametrize('old,new', [
    ('phase2', 'invoice_analysis'),
    ('phase21', 'model_evaluation'),
    ('phase2_config', 'analysis_config'),
])
def test_existing_imports_resolve_to_canonical_implementation(old, new):
    assert importlib.import_module(old) is importlib.import_module(new)


def test_readme_has_application_sections_without_development_history():
    readme = (Path(__file__).resolve().parents[1]/'README.md').read_text(encoding='utf-8-sig')
    assert readme.startswith('# Automated Invoice Processing and Anomaly Detection using Machine Learning')
    for section in ['Overview', 'Key Features', 'Technologies Used', 'System Architecture',
                    'Installation', 'Configuration', 'How to Run', 'Application Workflow',
                    'Machine Learning Approach', 'Dashboard Features', 'Project Structure',
                    'Testing', 'Limitations', 'Future Enhancements']:
        assert f'## {section}\n' in readme
    assert not re.search(r'phase[\s_.-]*\d|phase-one|development stage', readme, re.I)
    assert not any(marker in readme for marker in ['<<<<<<<', '=======', '>>>>>>>'])


def test_invoice_processing_labels_preserve_dashboard_controls():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=40)
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ['Invoice Validation', 'Invoice Processing']
    assert 'Extracted invoice CSV' in app.radio[0].options
    assert any(widget.label == 'Inspect invoice explanations' for widget in app.selectbox)
    assert any(expander.label == 'Technical details' for expander in app.expander)
