"""Presentation regressions: explanations must preserve existing analysis results."""
import json
from pathlib import Path
import pandas as pd
from dashboard_ui import invoice_explanation_data
from phase2 import analyze_invoices
from phase2_config import AnalysisConfig


def invoice(**changes):
    return dict(document_id='doc-a', invoice_number='INV-001', vendor_name='ACME',
                invoice_date='2026-10-27', currency='USD', total_amount='50000.00',
                warnings='[]', field_review='{}', status='ok', human_verified='true',
                line_items='[]', **changes)


def test_high_amount_panel_preserves_score_and_technical_payload():
    row=analyze_invoices(pd.DataFrame([invoice()]),AnalysisConfig(as_of='2026-10-29')).iloc[0]
    before=row.copy(deep=True)
    view=invoice_explanation_data(row)
    assert view['technical']['risk_score']==20 and row.review_priority=='LOW'
    assert view['technical']['contributions']==json.loads(row.risk_contributions)
    assert view['technical']['validation']==json.loads(row.validation_findings)
    assert view['technical']['ml']==row.ml_explanation
    assert any('50000' in reason and '10000' in reason for reason in view['reasons'])
    assert any('supporting purchase' in action for action in view['actions'])
    pd.testing.assert_series_equal(row,before)


def test_duplicate_and_ocr_actions_preserve_evidence():
    source=invoice()
    source['warnings']='["Amount requires source review"]'
    other={**source,'document_id':'doc-b'}
    row=analyze_invoices(pd.DataFrame([source,other]),AnalysisConfig(as_of='2026-10-29')).iloc[0]
    view=invoice_explanation_data(row)
    assert view['technical']['duplicates'][0]['other_analysis_id']=='row-000002'
    assert view['technical']['duplicates'][0]['provisional']
    assert view['technical']['ocr_warnings']==['Amount requires source review']
    assert any('matching records' in action for action in view['actions'])
    assert any('ambiguous values unresolved' in action for action in view['actions'])


def test_zero_score_retains_standard_review_and_no_invented_findings():
    source=invoice();source['total_amount']='100.00'
    row=analyze_invoices(pd.DataFrame([source]),AnalysisConfig(as_of='2026-10-29')).iloc[0]
    view=invoice_explanation_data(row)
    assert view['technical']['risk_score']==0
    # Arithmetic completeness notes may remain even with zero scoring points.
    assert view['reasons']==list(dict.fromkeys(json.loads(row.anomaly_explanations)))
    assert view['actions']


def test_explanation_panel_dashboard_and_optional_json():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=40)
    app.radio[0].set_value('Synthetic demonstration').run(timeout=40)
    assert not app.exception
    assert len(app.get('progress'))==1
    assert any(m.label=='Invoice ID' for m in app.metric)
    assert any(m.label=='Risk score' for m in app.metric)
    technical=next(e for e in app.expander if e.label=='Technical details')
    assert technical.proto.expanded is False
    assert len(technical.json)==1
    assert any(e.label=='Original values and complete analysis' for e in app.expander)
    headings=' '.join(m.value for m in app.markdown)
    for label in ['Reasons for review','Risk point contributions','Duplicate match details','OCR warnings','Recommended manual review actions','Machine learning assessment']:
        assert label in headings
