"""Upload/review/report integration with source-preserving extraction and analysis."""
from datetime import date
from io import BytesIO
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from pypdf import PdfReader
import pytest

from invoice_analysis import analyze_invoices
from analysis_config import AnalysisConfig
from reporting import csv_report, export_frame, pdf_report, DISCLAIMER
from workflow import (safe_name, validate_upload, process_uploads, apply_corrections, persist_review,
                      read_uploaded_csv, dashboard_data, filter_records, currency_vendor_totals, MAX_FILE_BYTES,
                      available_models,load_dashboard_model,rule_frequencies)

CONFIG=AnalysisConfig(as_of='2026-10-29')


def image_bytes(kind='PNG'):
    stream=BytesIO()
    Image.new('RGB',(100,100),'white').save(stream,format=kind)
    return stream.getvalue()


def record(**kwargs):
    row=dict(document_id='doc-1',source_file=r'C:\private\invoices\original.png',invoice_number='INV-001',
             vendor_name='ACME & Supplies',invoice_date='2026-10-27',currency='USD',total_amount='110.00',
             warnings='["OCR amount needs review"]',field_review='{"total_amount":"manual_review_required"}',
             ocr_mean_confidence=80,raw_text_file=r'C:\private\raw\invoice.txt',raw_text='RAW_OCR_SECRET',
             status='warning',line_items='[{"description":"Paper","quantity":"2","unit_price":"50","line_total":"100","currency":"USD"}]')
    return {**row,**kwargs}


def analysis(rows=None):
    return dashboard_data(analyze_invoices(pd.DataFrame(rows or [record()]),CONFIG))


@pytest.mark.parametrize('name', ['../../invoice.png',r'C:\secret\invoice.png','CON.png','evil<script>.png','..'])
def test_safe_names_are_relative_and_not_reserved(name):
    result=safe_name(name)
    assert '/' not in result and '\\' not in result and '<' not in result
    assert result not in {'.','..','CON.png'}


@pytest.mark.parametrize('name,data', [('a.exe',b'bad'),('a.pdf',b'not PDF'),('a.png',b'%PDF-1.4'),
                                     ('a.jpg',image_bytes()),('a.png',b''),('a.png',b'x'*(MAX_FILE_BYTES+1))],
                         ids=['extension','pdf-content','image-content','wrong-image-type','empty','oversize'])
def test_bad_uploads_are_rejected(name,data):
    with pytest.raises(ValueError):validate_upload(name,data)


@pytest.mark.parametrize('name,kind',[('a.png','PNG'),('a.jpg','JPEG'),('a.JPEG','JPEG')])
def test_valid_image_uploads(name,kind):
    assert validate_upload(name,image_bytes(kind))==name


def test_upload_batch_limits(tmp_path):
    with pytest.raises(ValueError):process_uploads([],tmp_path)
    with pytest.raises(ValueError):process_uploads([('a.png',b'x')]*21,tmp_path)
    with pytest.raises(ValueError):process_uploads([('a.png',b'x'*(MAX_FILE_BYTES))]*6,tmp_path)


def test_processing_isolated_runs_duplicates_and_per_file_errors(tmp_path,monkeypatch):
    import pipeline
    calls=[]
    def fake(sources,**kwargs):
        calls.append(sources[0])
        return pd.DataFrame([record(source_file=str(sources[0]),source_files=json.dumps([str(sources[0])]))])
    monkeypatch.setattr(pipeline,'run_pipeline',fake)
    contents=image_bytes()
    updates=[]
    frame,status,run=process_uploads([('../a.png',contents),('copy.png',contents),('bad.exe',b'bad')],tmp_path,
                                   progress=lambda value,message:updates.append(value))
    assert len(frame)==1 and len(calls)==1
    assert status.status.tolist()==['warning','reused','error']
    assert len(json.loads(frame.iloc[0].source_files))==2
    assert frame.iloc[0].original_filename=='a.png'
    assert (run/'extracted.csv').is_file() and len(updates)==3
    before=calls[0].read_bytes()
    _,_,second=process_uploads([('a.png',contents)],tmp_path)
    assert second!=run and calls[0].read_bytes()==before
    assert all(p.resolve().is_relative_to(tmp_path.resolve()) for p in calls)


def test_csv_upload_schema_and_leading_zero_identifiers():
    data=pd.DataFrame([record(invoice_number='00001')]).to_csv(index=False).encode()
    assert read_uploaded_csv(data).iloc[0].invoice_number=='00001'
    with pytest.raises(ValueError):read_uploaded_csv(b'x,y\n1,2\n')


def test_corrections_copy_original_preserve_warnings_and_create_versions(tmp_path):
    original=pd.DataFrame([record()])
    before=original.copy(deep=True)
    changes={0:dict(invoice_number='INV-1001',total_amount='137.50')}
    corrected=apply_corrections(original,changes,reviewed=True)
    pd.testing.assert_frame_equal(original,before)
    assert corrected.iloc[0].total_amount=='137.50'
    assert json.loads(corrected.iloc[0].original_extracted_values)['total_amount']=='110.00'
    assert corrected.iloc[0].warnings==original.iloc[0].warnings
    assert corrected.iloc[0].user_review_status=='CORE_FIELDS_REVIEWED'
    assert analyze_invoices(corrected,CONFIG).iloc[0].ocr_review_status=='REQUIRED'
    first=persist_review(corrected,tmp_path); second=persist_review(corrected,tmp_path)
    assert first!=second and (first/'corrected.csv').is_file()


@pytest.mark.parametrize('changes',[{0:{'total_amount':'137.5O'}},{0:{'invoice_date':'03/04/2026'}},
                                    {0:{'vendor_name':'1234'}},{0:{'currency':'$'}},{0:{'raw_text':'override'}},
                                    {2:{'invoice_number':'X'}}])
def test_invalid_corrections_rejected(changes):
    with pytest.raises(ValueError):apply_corrections(pd.DataFrame([record()]),changes)


def test_review_confirmation_is_per_record():
    original=pd.DataFrame([record(),record(document_id='doc-2')])
    revised=apply_corrections(original,{0:{'invoice_number':'A'},1:{'invoice_number':'B'}},reviewed={0})
    assert revised.user_review_status.tolist()==['CORE_FIELDS_REVIEWED','CORRECTED_NOT_VERIFIED']


def test_filters_and_currency_totals_do_not_mix_currencies():
    frame=analysis([record(),record(document_id='doc-2',invoice_number='B',currency='EUR',total_amount='900',invoice_date='2026-10-25')])
    assert len(filter_records(frame,currency='USD',start=date(2026,10,26),end=date(2026,10,29)))==1
    assert len(filter_records(frame,vendor='no vendor'))==0
    assert filter_records(frame,anomaly='Flagged').empty
    totals=currency_vendor_totals(frame,'USD')
    assert totals.iloc[0].amount=='110.00' and totals.currency.eq('USD').all()
    assert 'private' not in frame.iloc[0].source_filename


def test_export_privacy_and_csv_formula_protection():
    frame=analysis([record(vendor_name='=HYPERLINK("x")',warnings=json.dumps([r'Cannot open C:\private\secret.pdf']))])
    default=csv_report(frame).decode('utf-8-sig')
    assert 'RAW_OCR_SECRET' not in default and 'C:\\private' not in default
    assert 'raw_text_file' not in export_frame(frame).columns
    restored=pd.read_csv(BytesIO(csv_report(frame)),keep_default_na=False)
    assert restored.iloc[0].vendor_name.startswith("'=")
    private=csv_report(frame,include_private=True).decode('utf-8-sig')
    assert 'RAW_OCR_SECRET' in private
    assert {'anomaly_explanations','risk_score','review_priority','duplicate_matches'}<=set(restored.columns)
    assert isinstance(json.loads(restored.iloc[0].warnings),list)


def test_pdf_report_content_privacy_pagination_and_selection():
    rows=[record(document_id=f'doc-{i}',invoice_number=f'INV-{i}',vendor_name='ACME & <Supplies>') for i in range(8)]
    frame=analysis(rows)
    report=pdf_report(frame,dataset='Selected synthetic sample',model='synthetic_demo',generated='2026-10-29 15:00 IST')
    reader=PdfReader(BytesIO(report))
    content='\n'.join(page.extract_text() for page in reader.pages)
    assert report.startswith(b'%PDF') and len(reader.pages)>=2
    for value in ['INVOICE ANALYSIS REPORT','Selected synthetic sample','INV-0','INV-7','OCR uncertainty','synthetic_demo','not proof of fraud']:
        assert value in content
    assert 'C:\\private' not in content and 'RAW_OCR_SECRET' not in content
    selected=PdfReader(BytesIO(pdf_report(frame.iloc[:1])))
    selected_text='\n'.join(page.extract_text() for page in selected.pages)
    assert 'INV-0' in selected_text and 'INV-7' not in selected_text
    private=PdfReader(BytesIO(pdf_report(frame.iloc[:1],include_private=True)))
    assert 'RAW_OCR_SECRET' in '\n'.join(p.extract_text() for p in private.pages)


def test_empty_report_and_long_explanations():
    frame=analysis()
    report=PdfReader(BytesIO(pdf_report(frame.iloc[:0])))
    assert 'No records match' in report.pages[0].extract_text()
    frame['warnings']=json.dumps(['Long warning '+('review source '*700)])
    data=pdf_report(frame)
    assert len(PdfReader(BytesIO(data)).pages)>=2


def test_pdf_missing_values_are_readable_and_rule_chart_excludes_duplicate_signals():
    frame=analysis([record(invoice_date=None),record(document_id='doc-2')])
    report=PdfReader(BytesIO(pdf_report(frame)))
    content='\n'.join(page.extract_text() for page in report.pages)
    assert 'Unresolved / USD' in content and 'Not scored' in content
    assert 'nan /' not in content and 'outlier=None' not in content
    frequencies=rule_frequencies(frame)
    assert 'ocr_uncertainty' in frequencies.rule.tolist()
    assert not any('EXACT' in code for code in frequencies.rule)


def test_dashboard_navigation_and_reports_downloads():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=30)
    navigation=next(w for w in app.selectbox if w.label=='Navigation')
    assert navigation.options==['Dashboard','Upload & Process Invoices','Invoice Records','Anomaly Detection','Reports','Settings']
    navigation.set_value('Reports').run(timeout=30)
    assert not app.exception
    next(button for button in app.button if button.label=='Prepare PDF report').click().run(timeout=30)
    assert not app.exception
    assert len(app.get('download_button'))==3
    next(w for w in app.selectbox if w.label=='Navigation').set_value('Upload & Process Invoices').run(timeout=30)
    assert not app.exception
    assert any(b.label=='Extract invoice fields' and b.disabled for b in app.button)


def test_dashboard_correction_form_and_settings():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=30)
    next(w for w in app.selectbox if w.label=='Navigation').set_value('Invoice Records').run(timeout=30)
    assert not app.exception
    next(w for w in app.text_input if w.label=='Invoice number').set_value('REVIEW-1001')
    next(w for w in app.checkbox if w.label.startswith('I reviewed')).check()
    next(b for b in app.button if b.label=='Save review and reanalyze').click().run(timeout=30)
    assert not app.exception
    tables=[item.value for item in app.dataframe if 'invoice_number' in item.value.columns]
    assert any(table.invoice_number.eq('REVIEW-1001').any() for table in tables)
    next(w for w in app.selectbox if w.label=='Navigation').set_value('Settings').run(timeout=30)
    assert not app.exception and any('Active model' in item.value for item in app.info)


def test_unregistered_or_tampered_model_never_deserialized(tmp_path,monkeypatch):
    import anomaly_detector
    model=tmp_path/'models'/'fake.joblib'
    model.parent.mkdir()
    model.write_bytes(b'untrusted-pickle')
    def forbidden(path):pytest.fail('Untrusted file must not reach the pickle loader')
    monkeypatch.setattr(anomaly_detector,'load_model',forbidden)
    assert not available_models(tmp_path)
    with pytest.raises(ValueError):load_dashboard_model(model,tmp_path)
    Path(str(model)+'.integrity.json').write_text('{"created_by":"invoice_pipeline","sha256":"wrong"}')
    assert not available_models(tmp_path)
    with pytest.raises(ValueError):load_dashboard_model(model,tmp_path)


def test_missing_ocr_and_bad_file_errors_are_safe(tmp_path,monkeypatch):
    import pipeline
    def missing(*args,**kwargs):raise RuntimeError(r'Tesseract missing at C:\private\machine')
    monkeypatch.setattr(pipeline,'run_pipeline',missing)
    frame,status,run=process_uploads([('invoice.png',image_bytes())],tmp_path)
    assert frame.empty and status.iloc[0].status=='error'
    assert 'private' not in status.iloc[0].message
    assert (run/'extracted.csv').is_file()
