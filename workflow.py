"""Upload, review and dashboard helpers for the integrated extraction and analysis pipeline."""
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timedelta, timezone
from io import BytesIO, StringIO
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import tempfile
from uuid import uuid4

import pandas as pd
from PIL import Image, UnidentifiedImageError
import pypdfium2 as pdfium

from invoice_analysis import read_records, write_analysis
from validator import CORE_FIELDS, iso_date, money, text, validate_record

MAX_FILE_BYTES = 10*1024*1024
MAX_BATCH_BYTES = 50*1024*1024
MAX_FILES = 20
ALLOWED = {'.pdf', '.png', '.jpg', '.jpeg'}
logger = logging.getLogger(__name__)


def session_storage_root(root: Path, *, temporary: bool = False) -> Path:
    """Return configured storage, or force isolated OS-temp storage for uploads.

    Keep the directory path (not a TemporaryDirectory context manager) in
    session_state so Streamlit reruns cannot clean up active uploads. Local
    mode returns root unless temporary=True. Temp files remain until host
    cleanup/recycling; neither reruns nor creating another session delete them.
    """
    import streamlit as st
    mode = 'session' if temporary else os.environ.get('INVOICE_STORAGE_MODE')
    if mode is None:
        try:
            mode = st.secrets.get('INVOICE_STORAGE_MODE', 'local')
        except FileNotFoundError:
            mode = 'local'
    if mode not in {'local', 'session'}:
        raise ValueError('INVOICE_STORAGE_MODE must be local or session.')
    if mode == 'local':
        return Path(root)
    if 'invoice_storage_root' not in st.session_state:
        # Retain for this process lifetime; never remove user data automatically.
        st.session_state.invoice_storage_root = tempfile.mkdtemp(prefix='invoice-session-')
    return Path(st.session_state.invoice_storage_root)


def trusted_model(path, root):
    path=Path(path)
    if not path.resolve().is_relative_to(Path(root).resolve()):
        return False
    try:
        marker=json.loads(Path(str(path)+'.integrity.json').read_text(encoding='utf-8'))
        return marker.get('created_by')=='invoice_pipeline' and marker.get('sha256')==hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError,ValueError):
        return False


def available_models(root):
    root=Path(root)
    # Preserve discovery of existing demonstration artifacts without moving files.
    candidates=sorted((root/'models').glob('*.joblib'))+sorted((root/'output').glob('phase2_demo*/synthetic_model.joblib'))
    return [path for path in candidates if trusted_model(path,root)]


def load_dashboard_model(path,root):
    if not trusted_model(path,root):
        raise ValueError('Model is unregistered or changed. Create a new artifact using the project training CLI.')
    from anomaly_detector import load_model
    return load_model(path)


def safe_name(name):
    name = str(name).replace('\\', '/').split('/')[-1]
    name = re.sub(r'[^A-Za-z0-9._ -]', '_', name).strip(' .')[:110]
    if not name:
        return 'invoice'
    if Path(name).stem.upper() in {'CON','PRN','AUX','NUL',*[f'COM{i}' for i in range(1,10)],*[f'LPT{i}' for i in range(1,10)]}:
        name = 'invoice_'+name
    return name


def validate_upload(name, data):
    name = safe_name(name)
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED:
        raise ValueError('Use PDF, PNG, JPG or JPEG files only.')
    if not data or len(data) > MAX_FILE_BYTES:
        raise ValueError('Each file must contain data and be at most 10 MB.')
    try:
        if suffix == '.pdf':
            if not data.startswith(b'%PDF-'):
                raise ValueError('File content does not match PDF type.')
            with pdfium.PdfDocument(data) as doc:
                if not 1 <= len(doc) <= 25:
                    raise ValueError('PDFs must contain 1 to 25 pages.')
                for i in range(len(doc)):
                    page = doc[i]
                    try:
                        width, height = page.get_size()
                        if width <= 0 or height <= 0 or width*height*(300/72)**2 > 30_000_000:
                            raise ValueError('PDF page dimensions exceed the processing limit.')
                    finally:
                        page.close()
        else:
            with Image.open(BytesIO(data)) as image:
                expected = 'PNG' if suffix == '.png' else 'JPEG'
                if image.format != expected:
                    raise ValueError('File content does not match image type.')
                if image.width*image.height > 25_000_000:
                    raise ValueError('Image exceeds 25 million pixels; resize it before uploading.')
                image.verify()
    except (OSError, pdfium.PdfiumError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError('File is corrupt, encrypted, or cannot be read safely.') from exc
    return name


def process_uploads(uploads, root, progress=None):
    """New isolated run; preserve inputs and expose safe per-file status messages."""
    from pipeline import COLUMNS, run_pipeline
    if not uploads or len(uploads) > MAX_FILES:
        raise ValueError('Choose 1 to 20 invoice files.')
    if sum(len(data) for _, data in uploads) > MAX_BATCH_BYTES:
        raise ValueError('The combined upload must be at most 50 MB.')
    root = Path(root).resolve()
    run = root/'output'/'runs'/f'run_{uuid4().hex}'
    originals = root/'invoices'/'uploads'/run.name
    try:
        originals.mkdir(parents=True, exist_ok=False)
        run.mkdir(parents=True, exist_ok=False)
    except OSError:
        logger.exception('Invoice upload storage initialization failed')
        raise
    records, statuses, seen = [], [], {}
    for i, (name, data) in enumerate(uploads):
        label = safe_name(name)
        try:
            label = validate_upload(name, data)
            target = originals/f'{i+1:03d}_{label}'
            with target.open('xb') as handle:
                handle.write(data)
            digest = hashlib.sha256(data).hexdigest()
            if digest in seen:
                previous = records[seen[digest]]
                aliases = json.loads(previous['source_files'])
                aliases.append(str(target))
                previous['source_files'] = json.dumps(aliases)
                statuses.append(dict(filename=label, status='reused', message='Identical content reused; all source copies preserved.'))
            else:
                with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    frame = run_pipeline([target], csv_path=run/f'extraction_{i+1}.csv',
                                         processed_dir=run/'processed', raw_dir=run/'raw_ocr')
                row = frame.iloc[0].to_dict()
                row['original_filename'] = label
                records.append(row)
                seen[digest] = len(records)-1
                message = ('Extraction failed. Check source readability and OCR timeouts. On Streamlit Cloud, install tesseract-ocr and tesseract-ocr-eng through packages.txt and reboot. PDFs use pypdfium2 from requirements.txt; Poppler is not required.'
                           if row['status'] == 'error' else 'Extracted; inspect OCR warnings before analysis.')
                statuses.append(dict(filename=label, status=row['status'], message=message))
        except ValueError as exc:
            statuses.append(dict(filename=label, status='error', message=str(exc)))
        except (OSError, RuntimeError, MemoryError):
            logger.exception('Invoice upload file processing failed (file index %d)', i+1)
            statuses.append(dict(filename=label, status='error', message='Unable to process this file. Check type, size, readability and OCR setup.'))
        if progress:
            progress((i+1)/len(uploads), f'Processed {i+1} of {len(uploads)} files')
    result = pd.DataFrame(records, columns=COLUMNS+['original_filename'])
    try:
        write_analysis(result, run/'extracted.csv')
        (run/'upload_status.json').write_text(json.dumps(statuses, indent=2), encoding='utf-8')
    except OSError:
        logger.exception('Invoice upload batch persistence failed')
        raise
    return result, pd.DataFrame(statuses), run


def read_uploaded_csv(data):
    if not data or len(data) > MAX_FILE_BYTES:
        raise ValueError('CSV must contain data and be at most 10 MB.')
    frame = read_records(BytesIO(data))
    if len(frame) > 2000 or not set(CORE_FIELDS) <= set(frame.columns):
        raise ValueError('Use an extracted CSV with the five core invoice fields and at most 2000 rows.')
    return frame


def apply_corrections(original, changes, *, reviewed=False):
    """Apply explicit row-position edits to a copy; retain original OCR evidence."""
    corrected = original.copy(deep=True)
    audit = []
    for position, fields in changes.items():
        if not isinstance(position, int) or not 0 <= position < len(original):
            raise ValueError('Correction row does not exist.')
        if not set(fields) <= set(CORE_FIELDS):
            raise ValueError('Only core invoice fields may be corrected here.')
        candidate = original.iloc[position].to_dict()
        for key, value in fields.items():
            candidate[key] = text(value) or None
        valid = validate_record(candidate)
        bad = {'invalid_invoice_number','invalid_vendor','invalid_date','invalid_amount','unresolved_currency'}
        if any(f['code'] in bad and f['field'] in fields for f in valid['findings']):
            raise ValueError('Use readable identifiers/vendor, ISO dates, three-letter currency and canonical decimal amounts.')
        for key, value in fields.items():
            before, after = original.iloc[position][key], text(value) or None
            corrected.iat[position, corrected.columns.get_loc(key)] = after
            if str(before) != str(after):
                audit.append(dict(row=position, field=key, original=str(before), corrected=after))
    corrected['original_extracted_values'] = [json.dumps({key: None if pd.isna(row.get(key)) else str(row.get(key))
        for key in (*CORE_FIELDS, 'warnings', 'field_review')}, ensure_ascii=False) for row in original.to_dict('records')]
    corrected['correction_audit'] = [json.dumps([entry for entry in audit if entry['row'] == i]) for i in range(len(original))]
    reviewed_rows = set(changes) if reviewed is True else set() if reviewed is False else set(reviewed)
    corrected['user_review_status'] = ['CORE_FIELDS_REVIEWED' if i in reviewed_rows and i in changes else
                                      'CORRECTED_NOT_VERIFIED' if i in changes else 'NOT_REVIEWED' for i in range(len(original))]
    # Core-field review does not certify line items or silently dismiss OCR warnings.
    return corrected


def persist_review(frame, root):
    folder = Path(root)/'output'/'runs'/f'review_{uuid4().hex}'
    folder.mkdir(parents=True, exist_ok=False)
    write_analysis(frame, folder/'corrected.csv')
    return folder


def dashboard_data(frame):
    result = frame.copy(deep=True)
    result['source_filename'] = [safe_name(text(value)) for value in result.get('original_filename', result.get('source_file', pd.Series(['unknown']*len(result), index=result.index)))]
    result['validation_status'] = ['INVALID' if any(f.get('points',0)>0 for f in json.loads(value)) else 'NO_BLOCKING_FINDINGS'
                                   for value in result.validation_findings]
    return result


def filter_records(frame, *, vendor='', currency='All', start=None, end=None, anomaly='All'):
    result = frame.copy()
    if vendor:
        result = result[result.vendor_name.fillna('').astype(str).str.contains(vendor, case=False, regex=False)]
    if currency != 'All':
        result = result[result.currency == currency]
    if start or end:
        dates = result.invoice_date.map(iso_date)
        result = result[[bool(d and (not start or d >= start) and (not end or d <= end)) for d in dates]]
    if anomaly != 'All':
        flags = result.rule_anomaly.eq(True) | result.duplicate_flag.eq(True) | result.ml_anomaly.eq(True)
        result = result[flags if anomaly == 'Flagged' else ~flags]
    return result


def currency_vendor_totals(frame, currency):
    group = frame[frame.currency == currency]
    totals = {}
    for row in group.to_dict('records'):
        amount = money(row.get('total_amount'))
        if amount is not None:
            vendor = text(row.get('vendor_name')) or 'Unresolved vendor'
            totals[vendor] = totals.get(vendor, 0)+amount
    return pd.DataFrame([dict(vendor=vendor, amount=str(amount), currency=currency) for vendor, amount in totals.items()])


def rule_frequencies(frame):
    counts={}
    for row in frame.to_dict('records'):
        codes={item['code'] for item in json.loads(row.get('validation_findings','[]'))}
        for contribution in json.loads(row.get('risk_contributions','[]')):
            if contribution['category']=='amount_rules':
                codes.update(contribution['codes'])
        for code in codes:counts[code]=counts.get(code,0)+1
    return pd.DataFrame([dict(rule=code,invoices=count) for code,count in sorted(counts.items())],columns=['rule','invoices'])
