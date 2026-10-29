"""Presentation-only helpers; detection/scoring stay in the existing modules."""
import hashlib
import json
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from workflow import (process_uploads, apply_corrections, persist_review, filter_records,
                      currency_vendor_totals, safe_name, rule_frequencies)
from reporting import csv_report, pdf_report, export_frame, DISCLAIMER, LIMITATIONS
from validator import CORE_FIELDS, text

logger = logging.getLogger(__name__)


def invoice_explanation_data(row):
    """Present existing results without recalculating scores or certifying fields."""
    technical = dict(risk_score=int(row['risk_score']),
                     contributions=json.loads(row['risk_contributions']),
                     validation=json.loads(row['validation_findings']),
                     duplicates=json.loads(row['duplicate_matches']),
                     explanations=json.loads(row['anomaly_explanations']),
                     ocr_warnings=json.loads(row['ocr_warnings']), ml=row['ml_explanation'])
    actions = []
    if row.get('ocr_review_status') == 'REQUIRED':
        actions.append('Compare extracted fields and line items with the original invoice. Save source-supported corrections; leave ambiguous values unresolved.')
    if technical['duplicates']:
        actions.append('Open the matching records and compare vendor, invoice number, date, currency and amount. Confirm whether these are copies, revisions or separate invoices before payment.')
    if technical['validation']:
        actions.append('Resolve the validation findings against the source document and reconcile line items and totals where supporting components are available.')
    codes = {code for item in technical['contributions'] for code in item.get('codes', [])}
    if codes & {'high_amount', 'vendor_high_amount'}:
        actions.append('Verify the amount and currency against the source and supporting purchase or approval documents. Check the reason for the unusually large invoice.')
    if row.get('ml_anomaly') is True or str(row.get('ml_anomaly')).lower() == 'true':
        actions.append('Investigate the unusual model pattern alongside the source and vendor history; a model outlier does not establish fraud.')
    if not actions:
        actions.append('Complete the usual source-document and payment approval checks. No additional anomaly-specific action is currently indicated.')
    return dict(technical=technical, actions=actions,
                reasons=list(dict.fromkeys(technical['explanations'])))


def invoice_explanation_panel(row):
    """Compact, theme-aware invoice explanation with optional raw diagnostics."""
    data = invoice_explanation_data(row)
    detail = data['technical']
    labels = {'validation':'Field validation', 'duplicate_candidate':'Duplicate candidate',
              'amount_rules':'Amount checks', 'ml_pattern':'Machine learning pattern'}
    with st.container(border=True):
        st.subheader('Why this invoice needs review')
        identity, score, priority = st.columns([2, 1, 1])
        identity.metric('Invoice ID', text(row.get('invoice_number')) or 'Unresolved')
        identity.caption(f"Record: {row['analysis_id']}")
        score.metric('Risk score', f"{detail['risk_score']}/100")
        priority.metric('Review priority', row['review_priority'])
        st.progress(detail['risk_score'] / 100, text='Existing heuristic risk score')
        st.caption('Priority is based on the existing scoring policy. LOW does not mean approved. Anomaly flags are review signals, not proof of fraud.')
        st.markdown('**Reasons for review**')
        if data['reasons']:
            for reason in data['reasons']:
                st.write('• ' + str(reason))
        else:
            st.info('No anomaly or validation findings were identified. Standard review still applies.')
        st.markdown('**Risk point contributions**')
        if detail['contributions']:
            st.dataframe(pd.DataFrame([{'Risk factor':labels.get(item['category'], item['category'].replace('_',' ').capitalize()),
                                       'Points':item['points'],
                                       'Signals':', '.join(code.replace('_',' ').lower() for code in item.get('codes', []))}
                                      for item in detail['contributions']]), hide_index=True, width='stretch')
            st.caption('Contributions already include category caps; the overall score is capped at 100. Points are not added again for each duplicate match.')
        else:
            st.caption('No risk points contributed. OCR review can still be required at a score of zero.')
        st.markdown('**Duplicate match details**')
        if detail['duplicates']:
            for match in detail['duplicates']:
                st.write(f"Match: {match['other_analysis_id']} · {match['type'].replace('_',' ').capitalize()} · {match['points']} candidate points")
                if match.get('other_document_id'):
                    st.text('Document ID: ' + match['other_document_id'])
                st.write(match['explanation'])
                if match.get('provisional'):
                    st.caption('Provisional match — underlying OCR fields need review.')
        else:
            st.caption('No duplicate candidates found in this dataset.')
        st.markdown('**OCR warnings**')
        st.caption(f"OCR review status: {str(row.get('ocr_review_status', 'Unknown')).replace('_', ' ').capitalize()}. OCR uncertainty is separate from anomaly risk points.")
        for warning in detail['ocr_warnings']:
            st.write('• ' + str(warning))
        if not detail['ocr_warnings']:
            st.caption('No source OCR warnings recorded; this does not certify extraction accuracy.')
        st.markdown('**Machine learning assessment**')
        st.write(detail['ml'])
        if row.get('ml_training_kind') == 'synthetic_demo':
            st.caption('Synthetic demonstration model: experimental results, not validated real-world fraud detection; contributes zero risk points.')
        st.markdown('**Recommended manual review actions**')
        for number, action in enumerate(data['actions'], 1):
            st.write(f'{number}. {action}')
        with st.expander('Technical details', expanded=False):
            st.json(detail)


def style_dashboard():
    st.markdown('''<style>
    .stApp {background:#0c1424;color:#e8edf5;}
    .block-container {max-width:1320px;padding-top:2rem;padding-bottom:3rem;}
    [data-testid="stSidebar"] {background:#111e32;border-right:1px solid #26384f;}
    [data-testid="stMetric"] {background:#16243a;border:1px solid #30435b;border-radius:12px;padding:14px 17px;}
    [data-testid="stMetricLabel"] {color:#b7c8de;}
    [data-testid="stMetricValue"] {color:#f2f7ff;}
    h1 {font-size:2rem!important;letter-spacing:-.025em;} h2 {font-size:1.45rem!important;}
    [data-testid="stCaptionContainer"] {color:#b8c8dc;}
    [data-baseweb="select"]>div {background:#16243a;color:#edf4ff;}
    .stButton>button,.stDownloadButton>button {border-radius:8px;min-height:42px;}
    @media(max-width:900px){.block-container{padding-left:1rem;padding-right:1rem;}h1{font-size:1.6rem!important;}}
    </style>''', unsafe_allow_html=True)


def dataset_key(frame):
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


def current_review(frame):
    saved = st.session_state.get('review_versions', {}).get(dataset_key(frame))
    return apply_corrections(frame, saved['changes'], reviewed=saved['reviewed']) if saved else frame


def review_editor(frame, root, *, scope='records'):
    if frame.empty:
        st.info('No extracted records are available to review.')
        return
    st.subheader('Review extracted fields')
    st.caption('Corrections create a separate version. Original OCR text, values and warnings are preserved. Core-field review does not certify line items or clear remaining OCR uncertainty.')
    position = st.selectbox('Correction target', list(range(len(frame))), key=f'{scope}_correction_target',
                            format_func=lambda i:f'{i+1} · {text(frame.iloc[i].get("invoice_number")) or "Unresolved number"} · {safe_name(text(frame.iloc[i].get("source_file")))}')
    effective = current_review(frame)
    row = effective.iloc[position]
    with st.form(f'{scope}_field_corrections'):
        left,right = st.columns(2)
        values = {}
        for i,(key,label) in enumerate([('invoice_number','Invoice number'),('vendor_name','Vendor'),
                                       ('invoice_date','Invoice date (YYYY-MM-DD)'),('currency','Currency'),
                                       ('total_amount','Total amount (canonical decimal)')]):
            with left if i%2 == 0 else right:
                values[key] = st.text_input(label, value=text(row.get(key)),key=f'{scope}_review_{dataset_key(frame)[:12]}_{position}_{key}')
        verified = st.checkbox('I reviewed these core fields against the original invoice',key=f'{scope}_review_confirmation_{dataset_key(frame)[:12]}_{position}')
        submitted = st.form_submit_button('Save review and reanalyze', type='primary')
    if submitted:
        try:
            versions = dict(st.session_state.get('review_versions', {}))
            key = dataset_key(frame)
            saved = versions.get(key, {'changes':{}, 'reviewed':set()})
            changes, reviewed = dict(saved['changes']), set(saved['reviewed'])
            changes[position] = values
            reviewed.add(position) if verified else reviewed.discard(position)
            corrected = apply_corrections(frame, changes, reviewed=reviewed)
            persist_review(corrected, root)
            versions[key] = dict(changes=changes, reviewed=reviewed)
            st.session_state.review_versions = versions
            st.success('Review version saved. Original extraction retained; analysis will use the corrected fields.')
            st.rerun()
        except (ValueError,OSError):
            st.error('Review could not be saved. Use valid identifiers, ISO dates, currency codes and decimal amounts; check folder permissions.')


def upload_page(root, config=None):
    st.caption('Hosted storage is temporary. Download reviewed results before restarting or leaving the session. Do not upload confidential invoices to a public demonstration.')
    st.subheader('Upload & Process Invoices')
    st.write('PDF, PNG and JPG/JPEG · 10 MB per file · 20 files / 50 MB per batch · PDF limit: 25 pages')
    files = st.file_uploader('Choose invoice PDFs or images', type=['pdf','png','jpg','jpeg'], accept_multiple_files=True)
    st.caption('Files are saved as new copies. OCR output must be reviewed before relying on amounts or dates.')
    if st.button('Process Invoices', type='primary', disabled=not files):
        progress = st.progress(0.0, text='Validating uploads')
        try:
            frame,statuses,run = process_uploads([(file.name,file.getvalue()) for file in files], root,
                                                progress=lambda value,message:progress.progress(value,text=message))
            st.session_state.processed_invoices = frame
            st.session_state.upload_status = statuses
            st.session_state.processed_dataset = run.name
            st.session_state.prefer_processed = True
            failed = int(statuses.status.eq('error').sum())
            if failed:
                st.warning(f'{failed} file(s) could not be extracted. Review the per-file errors below; failed extractions are not successful invoice results.')
            if statuses.status.isin(['ok', 'warning']).any():
                st.success(f'Extraction complete: {len(frame)} unique document records. Analysis and downloads are below.')
        except ValueError as exc:
            st.error(str(exc))
        except PermissionError:
            logger.exception('Invoice upload batch failed: storage permission denied')
            st.error('Upload storage is not writable. Locally, stop Streamlit and restart it from a normal terminal with your project environment. On Streamlit Cloud, reboot the app and check temporary-storage permissions. Details are in the server logs.')
        except OSError:
            logger.exception('Invoice upload batch failed: storage or persistence error')
            st.error('The upload batch could not be saved. Check available disk space and temporary-storage access, then retry. Details are in the server logs.')
    if 'upload_status' in st.session_state:
        st.dataframe(st.session_state.upload_status, hide_index=True)
    if 'processed_invoices' in st.session_state:
        frame = st.session_state.processed_invoices
        cols = [c for c in ['invoice_number','vendor_name','invoice_date','currency','total_amount','ocr_mean_confidence','warnings','status'] if c in frame]
        preview = export_frame(frame)
        preview['status'] = frame['status']
        st.dataframe(preview[[c for c in cols if c in preview]], hide_index=True)
        review_editor(frame,root,scope='uploads')
        if frame.empty or not frame.status.isin(['ok', 'warning']).any():
            st.info('No invoices were successfully extracted. Check file errors and try again. No demonstration records are substituted.')
            return
        from analysis_config import AnalysisConfig
        from invoice_analysis import analyze_invoices
        from workflow import dashboard_data
        try:
            result = dashboard_data(analyze_invoices(current_review(frame), config or AnalysisConfig()))
            st.subheader('Uploaded invoice analysis')
            st.caption('Rules and duplicate checks use your uploaded records and current sidebar settings. Optional trusted models are available under Invoice Validation → Processed uploads. Anomaly signals are not proof of fraud.')
            st.dataframe(export_frame(result)[['invoice_number','vendor_name','total_amount','currency','isanomaly',
                                 'anomalyreason','ocr_review_status','ml_status']], hide_index=True)
            st.download_button('Download uploaded analysis CSV', csv_report(result),
                               file_name='uploaded_invoice_analysis.csv', mime='text/csv', key='upload_analysis_csv')
            if st.button('Prepare uploaded PDF report', key='upload_prepare_pdf'):
                st.download_button('Download uploaded PDF report',
                    pdf_report(result, dataset=st.session_state.get('processed_dataset','Uploaded invoices'), model='Rules only'),
                    file_name='uploaded_invoice_report.pdf', mime='application/pdf', key='upload_pdf')
        except ImportError:
            logger.exception('Uploaded invoice reporting dependency unavailable')
            st.error('Reporting dependencies are missing. Install requirements.txt and restart the application.')
        except (ValueError, TypeError, OSError, RuntimeError):
            logger.exception('Uploaded invoice analysis or reporting failed')
            st.error('Unable to analyze or report this batch. Check extracted fields and server logs; your extraction preview remains available.')


def additional_filters(frame):
    left,right = st.columns(2)
    with left:
        currency = st.selectbox('Record currency', ['All']+sorted(frame.currency.dropna().astype(str).unique().tolist()))
    with right:
        anomaly = st.selectbox('Anomaly status', ['All','Flagged','Not flagged'])
    start,end = None,None
    if st.checkbox('Filter invoice date range'):
        left,right = st.columns(2)
        with left:
            start = st.date_input('From invoice date')
        with right:
            end = st.date_input('Through invoice date')
        if start>end:
            st.error('Start date must not be after end date.')
            return frame.iloc[:0]
    return filter_records(frame,currency=currency,start=start,end=end,anomaly=anomaly)


def charts(frame):
    if frame.empty:
        st.info('No records match these filters. Adjust the filters to see charts.')
        return
    try:
        import plotly.express as px
    except ImportError:
        st.info('Install Plotly from requirements.txt to enable charts.')
        return
    def show(fig, target):
        fig.update_layout(template='plotly_dark', paper_bgcolor='#0c1424',plot_bgcolor='#0c1424',
                          font=dict(color='#dce7f5'),margin=dict(l=18,r=18,t=45,b=25),height=300)
        target.plotly_chart(fig,width='stretch')
    left,right=st.columns(2)
    show(px.histogram(frame,x='risk_score',nbins=20,title='Risk score distribution',color_discrete_sequence=['#59b6e8']),left)
    priorities=frame.review_priority.value_counts().rename_axis('priority').reset_index(name='invoices')
    show(px.bar(priorities,x='priority',y='invoices',color='priority',title='Review priority',
                color_discrete_map={'LOW':'#68d6ad','MEDIUM':'#f2c66d','HIGH':'#ff8794'}),right)
    currency=st.selectbox('Chart currency',sorted(frame.currency.fillna('Unresolved').astype(str).unique()))
    group=frame[frame.currency==currency].copy()
    group['observed_amount']=pd.to_numeric(group.total_amount,errors='coerce')
    left,right=st.columns(2)
    plotted=group.dropna(subset=['observed_amount']).copy()
    plotted['anomaly_status']=plotted['isanomaly'].map({True:'Flagged',False:'Not flagged'})
    show(px.box(plotted,x='vendor_name',y='observed_amount',points='all',
                title=f'Vendor amount distribution - {currency}'),left)
    show(px.scatter(plotted,x='invoice_date',y='observed_amount',color='anomaly_status',
                    hover_data=['invoice_number','vendor_name'],
                    title=f'Invoice amounts and outliers - {currency}'),right)
    left,right=st.columns(2)
    show(px.histogram(group.dropna(subset=['observed_amount']),x='observed_amount',title=f'Observed amounts · {currency}',color_discrete_sequence=['#59b6e8']),left)
    totals=currency_vendor_totals(frame,currency)
    if not totals.empty:
        totals['display_amount']=pd.to_numeric(totals.amount)
        show(px.bar(totals.sort_values('display_amount',ascending=False).head(12),x='vendor',y='display_amount',title=f'Vendor observed totals · {currency}'),right)
    st.caption('Amounts are unverified observations until reviewed. Charts never combine currencies.')
    frequencies=rule_frequencies(frame)
    left,right=st.columns(2)
    counts=pd.DataFrame({'status':['Candidate','No candidate'],'invoices':[int(frame.duplicate_flag.eq(True).sum()),int(frame.duplicate_flag.eq(False).sum())]})
    show(px.bar(counts,x='status',y='invoices',title='Duplicate candidate records'),left)
    if not frequencies.empty:
        show(px.bar(frequencies,x='rule',y='invoices',title='Rule / validation flags (records)'),right)


def reports_page(frame,dataset,model):
    st.subheader('Reports')
    st.caption('Reports use the current filters. Choose specific records below, or leave the selection empty for all filtered records.')
    selected=st.multiselect('Report records',frame.analysis_id.tolist())
    subset=frame[frame.analysis_id.isin(selected)] if selected else frame
    private=st.checkbox('Explicitly include private local paths and raw columns in reports',value=False)
    st.warning(DISCLAIMER)
    st.caption(LIMITATIONS)
    st.download_button('Download selected report CSV',csv_report(subset,include_private=private),file_name='invoice_review.csv',mime='text/csv')
    if st.button('Prepare PDF report',type='primary'):
        try:
            data=pdf_report(subset,dataset=dataset,model=model,include_private=private)
            st.download_button('Download PDF report',data,file_name='invoice_analysis_report.pdf',mime='application/pdf')
        except ImportError:
            st.error('PDF generation requires ReportLab. Install requirements.txt and restart the app.')
        except (ValueError,OSError,RuntimeError):
            st.error('Report could not be generated. Check report scope and installation; original data is unchanged.')
