"""Integrated invoice processing, validation and review dashboard."""
from dataclasses import replace
from datetime import date
from io import BytesIO
import json
import logging
from pathlib import Path

import streamlit as st

STARTUP_ERROR = None
try:
    from anomaly_detector import load_model, train_model
    from invoice_analysis import analyze_invoices, demo_data, read_records
    from analysis_config import AnalysisConfig
    from workflow import read_uploaded_csv, dashboard_data, safe_name, available_models, load_dashboard_model, session_storage_root
    from reporting import csv_report, redact_paths
    from dashboard_ui import (style_dashboard, upload_page, current_review, review_editor,
                              additional_filters, charts, reports_page, invoice_explanation_panel)
except ImportError as exc:
    logging.getLogger(__name__).exception('Dashboard processing imports failed')
    module = exc.name or 'unknown module'
    STARTUP_ERROR = (f'Processing import failed ({module}). Check the Streamlit startup logs for the exact error. '
                     'Restart using the project virtual environment; install requirements.txt if a dependency is missing. '
                     'No invoice files have been changed.')

ROOT = Path(__file__).resolve().parent


@st.cache_resource
def demo_model(as_of):
    config = AnalysisConfig(as_of=as_of)
    training, _ = demo_data(config)
    return train_model(training, config, training_kind='synthetic_demo')


def main() -> None:
    st.set_page_config(page_title="Invoice Processing", layout="wide")
    if STARTUP_ERROR:
        st.title('Automated Invoice Processing and Anomaly Detection')
        st.error(STARTUP_ERROR)
        return
    style_dashboard()
    try:
        storage_root = session_storage_root(ROOT)
        upload_storage = session_storage_root(ROOT, temporary=True)
    except (ValueError, OSError):
        logging.getLogger(__name__).exception('Invoice session storage initialization failed')
        st.error('Unable to initialize invoice storage. Check INVOICE_STORAGE_MODE and temporary-directory permissions.')
        return
    st.title("Automated Invoice Processing and Anomaly Detection")
    st.caption('INVOICE REVIEW WORKSPACE  ·  Extraction / validation / explainable review. Signals are not confirmed fraud.')
    analysis_tab, ocr_tab = st.tabs(['Invoice Validation', 'Invoice Processing'])
    with st.sidebar:
        st.markdown('### Invoice Review')
        page = st.selectbox('Navigation', ['Dashboard','Upload & Process Invoices','Invoice Records','Anomaly Detection','Reports','Settings'])
        st.header('Analysis settings')
        base = AnalysisConfig()
        as_of = st.date_input('Analysis date', value=date.fromisoformat(base.as_of))
        currency = st.selectbox('Amount threshold currency', list(base.high_amount_limits))
        limits = dict(st.session_state.get('amount_limits', base.high_amount_limits))
        limit = st.number_input('High amount threshold', min_value=0.01, value=float(limits[currency]),key=f'amount_limit_{currency}')
        limits[currency] = str(limit)
        st.session_state.amount_limits = limits
        vendor_similarity = st.slider('Vendor similarity threshold', 50, 100, int(base.vendor_similarity))
        config = replace(base, as_of=as_of.isoformat(), vendor_similarity=vendor_similarity,
                         high_amount_limits=limits)
        st.caption('Thresholds are currency-specific. LOW priority does not mean approved. OCR review is separate from the 0–100 heuristic risk score.')
    with ocr_tab:
        upload_page(upload_storage, config)
    with analysis_tab:
        if page == 'Upload & Process Invoices':
            st.info('Open the Invoice Processing tab to upload, process, review and download invoice results.')
            return
        options = ['Extracted invoice CSV', 'Synthetic demonstration']
        if 'processed_invoices' in st.session_state:
            options.append('Processed uploads')
        if st.session_state.get('prefer_processed'):
            st.session_state.data_source = 'Processed uploads'
            st.session_state.prefer_processed = False
        local_candidates = [] if storage_root != ROOT else [ROOT/'output/ocr_accuracy_final/invoices.csv', ROOT/'output/invoices.csv']
        default_source = 0 if any(path.is_file() for path in local_candidates) else 1
        source = st.radio('Data source', options, index=default_source, horizontal=True,key='data_source')
        model = None
        model_name = 'Rules only'
        dataset = 'Existing extracted invoices'
        try:
            if source == 'Synthetic demonstration':
                st.warning('SYNTHETIC DEMO ONLY: generated training and held-out records are separate from extracted invoices. Demo ML adds zero risk points.')
                _, frame = demo_data(config)
                model = demo_model(config.as_of)
                model_name = 'Built-in numeric synthetic demo'
                dataset = 'Synthetic demonstration · 17 held-out cases'
            else:
                upload = st.file_uploader('Upload extracted invoice CSV', type=['csv']) if source != 'Processed uploads' else None
                if source == 'Processed uploads':
                    frame = st.session_state.processed_invoices
                    dataset = st.session_state.get('processed_dataset','Processed upload run')
                elif upload is not None:
                    frame = read_uploaded_csv(upload.getvalue())
                    dataset = safe_name(upload.name)
                else:
                    candidates = local_candidates
                    existing = next((p for p in candidates if p.is_file()), None)
                    if existing is None:
                        st.info('Upload an extracted CSV or select the synthetic demonstration.')
                        return
                    st.caption(f'Loaded existing sample extraction: {existing.relative_to(ROOT)}')
                    frame = read_records(existing)
                    dataset = safe_name(existing.name)
                artifacts = available_models(ROOT)
                selection = st.selectbox('Saved local model', ['Rules only']+[str(p.relative_to(ROOT)) for p in artifacts])
                if selection != 'Rules only':
                    model = load_dashboard_model(ROOT/selection,ROOT)
                    model_name = safe_name(selection)
                    st.info(f'Model origin: {model["training_kind"]}. Eligible training rows: {model["eligible_rows"]}.')
                st.caption('Only trusted local models created by this project are offered; model files cannot be uploaded.')
            if source == 'Synthetic demonstration':
                artifacts = available_models(ROOT)
                selection = st.selectbox('Saved local model', ['Built-in numeric demo']+[str(p.relative_to(ROOT)) for p in artifacts])
                if selection != 'Built-in numeric demo':
                    model = load_dashboard_model(ROOT/selection,ROOT)
                    model_name = safe_name(selection)
                    st.info(f'Model origin: {model["training_kind"]}; features: {model.get("feature_profile", "numeric_v1")}.')
            if source != 'Synthetic demonstration' and 'data_origin' in frame and frame.data_origin.eq('synthetic_demo').any():
                st.warning('This CSV is labeled synthetic_demo. Its results demonstrate behavior, not real-world accuracy.')
            if model and model.get('feature_profile') == 'vendor_v2':
                st.warning('Experimental vendor-aware model: it underperformed the numeric baseline on synthetic evaluation. Results are review signals only.')
            if page == 'Invoice Records':
                review_editor(frame,upload_storage if source == 'Processed uploads' else storage_root)
            frame = current_review(frame)
            result = dashboard_data(analyze_invoices(frame, config, model))
        except (ValueError, TypeError, OSError) as exc:
            st.error(f'Unable to analyze input: {redact_paths(exc)}')
            return
        st.caption(f'Dataset: {dataset} · Active model: {model_name} · Mode: {"ML-assisted" if model else "Rules only"}')
        if page == 'Settings':
            st.subheader('Settings')
            st.write('Adjust the validated thresholds in the sidebar and choose a trusted local model above. Changes apply to this session; original inputs and model files are unchanged.')
            st.json(config.to_dict())
            st.info(f'Active model: {model_name}. Origin: {model["training_kind"] if model else "none"}. No models can be uploaded.')
            st.caption('PDF charts require Plotly; PDF exports require ReportLab. Install requirements.txt if an optional presentation dependency is missing.')
            return
        columns = st.columns(5)
        for column, label, value in zip(columns,
                ['Invoices', 'Rule flags', 'Duplicate candidates', 'High priority', 'OCR review required'],
                [len(result), int(result.rule_anomaly.sum()), int(result.duplicate_flag.sum()),
                 int((result.review_priority == 'HIGH').sum()), int((result.ocr_review_status == 'REQUIRED').sum())]):
            column.metric(label, value)
        more = st.columns(3)
        more[0].metric('Valid invoices', int(result.validation_status.eq('NO_BLOCKING_FINDINGS').sum()))
        more[1].metric('Invalid invoices', int(result.validation_status.eq('INVALID').sum()))
        more[2].metric('ML outliers', int(result.ml_anomaly.eq(True).sum()))
        st.caption('Valid fields means no blocking validation finding; it does not certify OCR accuracy, arithmetic completeness or payment approval.')
        st.caption('Counts are records, not duplicate groups. Exact-field matches may be different copies/formats of one invoice. No invoices are removed.')
        st.subheader('Anomaly summary')
        st.write(result.groupby(['review_priority', 'ocr_review_status'], dropna=False).size().rename('records').reset_index())
        st.caption(f'ML scored: {int(result.ml_status.isin(["SCORED", "SCORED_DEMO"]).sum())}; ML flagged: {int(result.ml_anomaly.eq(True).sum())}. Missing/unverified fields can cause scoring to be skipped.')
        priorities = st.multiselect('Review priorities', ['LOW', 'MEDIUM', 'HIGH'], default=['LOW', 'MEDIUM', 'HIGH'])
        ocr_status = st.selectbox('OCR review filter', ['All']+sorted(result.ocr_review_status.unique().tolist()))
        query = st.text_input('Vendor contains')
        filtered = result[result.review_priority.isin(priorities)]
        if ocr_status != 'All':
            filtered = filtered[filtered.ocr_review_status == ocr_status]
        if query:
            filtered = filtered[filtered.vendor_name.fillna('').astype(str).str.contains(query, case=False, regex=False)]
        filtered = additional_filters(filtered)
        st.write(f'Displaying {len(filtered)} of {len(result)} invoices')
        visible = ['analysis_id', 'invoice_number', 'vendor_name', 'invoice_date', 'currency', 'total_amount',
                   'risk_score', 'review_priority', 'ocr_review_status', 'review_status', 'duplicate_flag',
                   'arithmetic_status', 'ml_status', 'ml_anomaly', 'ml_decision_score','source_filename','validation_status']
        st.dataframe(filtered[visible], hide_index=True, width='stretch')
        st.download_button('Download filtered analysis CSV', csv_report(filtered),
                           file_name='invoice_analysis.csv', mime='text/csv')
        if page in {'Dashboard','Anomaly Detection'}:
            charts(filtered)
        if page == 'Reports':
            reports_page(filtered,dataset,model_name)
        if not filtered.empty:
            selected = st.selectbox('Inspect invoice explanations', filtered.analysis_id.tolist())
            row = filtered[filtered.analysis_id == selected].iloc[0]
            invoice_explanation_panel(row)
            with st.expander('Original values and complete analysis'):
                st.dataframe(filtered[filtered.analysis_id == selected], hide_index=True)


if __name__ == "__main__":
    main()
