"""Currency-specific Isolation Forest pipelines and explainable amount rules."""
from datetime import date
from decimal import Decimal
from pathlib import Path
from statistics import median
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from phase2_config import AnalysisConfig
from validator import normalized, text, validate_record

FEATURES = ['log_amount', 'item_count', 'day_of_month', 'weekday']
VENDOR_FEATURES = FEATURES + ['vendor_frequency', 'vendor']


def suitable(record, valid):
    """No fitting/scoring unresolved OCR or invalid financial/date records."""
    issues = [f for f in valid['findings'] if f['points'] > 0]
    trusted = valid['human_verified'] or text(record.get('data_origin')) == 'synthetic_demo'
    return (trusted and valid['ocr_review_status'] != 'REQUIRED' and not issues
            and valid['amount'] is not None and valid['amount'] > 0 and valid['date'] is not None
            and bool(valid['currency']) and bool(valid['items'])
            and all(isinstance(i, dict) for i in valid['items'])
            and not any(f['code'] in {'invalid_line_items', 'incomplete_line_item', 'item_currency_unresolved'} for f in valid['findings']))


def features(validated, vendor_frequencies=None):
    matrix = pd.DataFrame([dict(log_amount=np.log1p(float(v['amount'])), item_count=len(v['items']),
                              day_of_month=v['date'].day, weekday=v['date'].weekday())
                         for v in validated], columns=FEATURES)
    if vendor_frequencies is not None:
        matrix['vendor'] = [normalized(v['vendor']) for v in validated]
        matrix['vendor_frequency'] = matrix.vendor.map(vendor_frequencies).fillna(0.0)
        matrix = matrix[VENDOR_FEATURES]
    return matrix


def train_model(invoices: pd.DataFrame, config: AnalysisConfig | None = None, *, training_kind='verified_real', feature_profile='numeric_v1'):
    config = config or AnalysisConfig()
    if feature_profile not in {'numeric_v1', 'vendor_v2'}:
        raise ValueError('Unknown feature profile')
    if training_kind not in {'verified_real', 'synthetic_demo'}:
        raise ValueError('training_kind must explicitly be verified_real or synthetic_demo')
    records = invoices.to_dict('records')
    origins = {text(r.get('data_origin')) for r in records}
    if training_kind == 'synthetic_demo' and origins != {'synthetic_demo'}:
        raise ValueError('Synthetic training must contain only labeled synthetic_demo records.')
    if training_kind == 'verified_real' and any('synthetic' in origin.casefold() for origin in origins):
        raise ValueError('Never mix synthetic records into verified-real training.')
    eligible, seen, seen_sources = [], set(), set()
    for record in records:
        valid = validate_record(record, config)
        if not suitable(record, valid):
            continue
        source_id = text(record.get('document_id'))
        if source_id and source_id in seen_sources:
            continue
        key = (normalized(valid['vendor']), valid['number'].casefold(), valid['date'], valid['amount'], valid['currency'])
        if key in seen:
            continue
        seen.add(key)
        if source_id:
            seen_sources.add(source_id)
        eligible.append(valid)
    models = {}
    for currency in sorted({v['currency'] for v in eligible}):
        group = [v for v in eligible if v['currency'] == currency]
        if len(group) < config.min_training_rows:
            continue
        frequencies = None
        if feature_profile == 'vendor_v2':
            frequencies = pd.Series([normalized(v['vendor']) for v in group]).value_counts(normalize=True).to_dict()
        matrix = features(group, frequencies)
        if feature_profile == 'vendor_v2':
            preprocessing = ColumnTransformer([
                ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')),
                                      ('scale', StandardScaler())]), FEATURES+['vendor_frequency']),
                ('vendor', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                                     ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), ['vendor'])])
            steps = [('preprocess', preprocessing)]
        else:
            steps = [('scale', StandardScaler())]
        estimator = Pipeline(steps + [('isolation_forest', IsolationForest(
            n_estimators=200, contamination=config.contamination, random_state=config.random_seed, n_jobs=1))])
        estimator.fit(matrix)
        models[currency] = dict(pipeline=estimator, rows=len(group),
                                median_amount=str(median(v['amount'] for v in group)),
                                feature_medians=matrix.median(numeric_only=True).to_dict())
        if frequencies is not None:
            models[currency]['vendor_frequencies'] = frequencies
    if not models:
        raise ValueError(f'Need at least {config.min_training_rows} distinct suitable records in one currency; four OCR samples are insufficient.')
    return dict(schema_version=2 if feature_profile == 'vendor_v2' else 1,
                feature_profile=feature_profile, sklearn_version=sklearn.__version__, training_kind=training_kind,
                features=VENDOR_FEATURES if feature_profile == 'vendor_v2' else FEATURES, config=config.to_dict(), models=models,
                input_rows=len(records), eligible_rows=len(eligible),
                excluded_rows=len(records)-len(eligible))


def save_model(bundle, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects existing models.
    integrity = Path(str(path)+'.integrity.json')
    if integrity.exists():
        raise FileExistsError('Model integrity record exists; choose a new path.')
    created = False
    integrity_created = False
    try:
        with path.open('xb') as handle:
            created = True
            joblib.dump(bundle, handle)
        with integrity.open('x',encoding='utf-8') as handle:
            integrity_created = True
            json.dump(dict(created_by='invoice_pipeline',sha256=hashlib.sha256(path.read_bytes()).hexdigest()),handle)
    except Exception:
        if created:
            path.unlink(missing_ok=True)  # Only remove this failed new artifact.
        if integrity_created:
            integrity.unlink(missing_ok=True)
        raise


def load_model(path):
    """Load only a trusted locally created joblib artifact, never an upload."""
    bundle = joblib.load(path)
    if (not isinstance(bundle, dict) or bundle.get('schema_version') not in {1, 2}
            or bundle.get('features') != (VENDOR_FEATURES if bundle.get('schema_version') == 2 else FEATURES)
            or bundle.get('sklearn_version') != sklearn.__version__
            or bundle.get('training_kind') not in {'verified_real', 'synthetic_demo'}
            or not bundle.get('models')):
        raise ValueError('Unsupported or incompatible model artifact; retrain with this environment.')
    if bundle['schema_version'] == 2 and any('vendor_frequencies' not in m for m in bundle['models'].values()):
        raise ValueError('Vendor model is missing fitted frequency preprocessing.')
    return bundle


def ml_result(record, valid, bundle):
    if bundle is None:
        return dict(status='NOT_LOADED', anomaly=None, score=None,
                    explanation='No model loaded; rule checks still run.', training_kind='none')
    kind = bundle['training_kind']
    if not suitable(record, valid):
        return dict(status='SKIPPED_UNVERIFIED_OR_INVALID', anomaly=None, score=None,
                    explanation='ML requires verified, complete, valid fields without unresolved OCR warnings.', training_kind=kind)
    model = bundle['models'].get(valid['currency'])
    if model is None:
        return dict(status='SKIPPED_UNSUPPORTED_CURRENCY', anomaly=None, score=None,
                    explanation='No trained baseline for this currency; no currency conversion is assumed.', training_kind=kind)
    matrix = features([valid], model.get('vendor_frequencies'))
    vendor_note = (' Vendor was unseen in training; unknown-category encoding and frequency zero were used.'
                   if 'vendor_frequencies' in model and normalized(valid['vendor']) not in model['vendor_frequencies'] else '')
    decision = float(model['pipeline'].decision_function(matrix)[0])
    is_anomaly = decision < 0
    return dict(status='SCORED_DEMO' if kind == 'synthetic_demo' else 'SCORED',
                anomaly=is_anomaly, score=decision, training_kind=kind,
                explanation=(f'Isolation Forest decision_function={decision:.6f}; values below zero are unusual '
                             f'against {model["rows"]} {kind} {valid["currency"]} rows. '
                             f'Observed features: {matrix.iloc[0].to_dict()}. '
                             'This is a pattern flag, not a fraud probability or causal explanation.' + vendor_note))


def amount_rules(index, records, validated, config):
    valid = validated[index]
    amount, currency = valid['amount'], valid['currency']
    result = []
    if amount is None or not currency:
        return result
    limit = config.high_amount_limits.get(currency)
    if limit is not None and amount > Decimal(str(limit)):
        result.append(dict(code='high_amount', points=20,
                           message=f'Observed {currency} {amount} exceeds configured {currency} {limit}; verify the extracted amount.'))
    peers, seen, seen_sources = [], set(), set()
    for i, other in enumerate(validated):
        if (i == index or not valid['date'] or not other['date'] or other['date'] >= valid['date']
                or normalized(other['vendor']) != normalized(valid['vendor']) or other['currency'] != currency
                or not suitable(records[i], other)):
            continue
        key = (other['number'], other['date'], other['amount'])
        source_id = text(records[i].get('document_id'))
        if key not in seen and (not source_id or source_id not in seen_sources):
            peers.append(other['amount'])
            seen.add(key)
            if source_id:
                seen_sources.add(source_id)
    if len(peers) >= config.vendor_min_history:
        baseline = median(peers)
        if amount > baseline * Decimal(str(config.vendor_amount_multiplier)):
            result.append(dict(code='vendor_high_amount', points=20,
                message=f'{amount} {currency} exceeds {config.vendor_amount_multiplier} x historical vendor median {baseline} from {len(peers)} distinct earlier verified/demo records.'))
    return result


def detect_anomalies(invoices: pd.DataFrame, config=None, model=None) -> pd.DataFrame:
    from phase2 import analyze_invoices
    return analyze_invoices(invoices, config=config, model=model)
