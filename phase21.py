"""Reproducible synthetic-only ML evaluation; labels never enter training/inference."""
import argparse
from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from anomaly_detector import train_model, save_model, load_model
from phase2 import analyze_invoices, write_analysis
from phase2_config import AnalysisConfig


def verification_data(config=None):
    config = config or AnalysisConfig()
    train_rng = np.random.default_rng(config.random_seed)
    eval_rng = np.random.default_rng(config.random_seed + 1009)
    as_of = date.fromisoformat(config.as_of)
    vendors = [f'Demo Supplier {letter}' for letter in 'ABCDEFGH']

    def invoice(rng, identifier, currency, *, amount=None, count=None, vendor=None):
        amount = Decimal(str(amount if amount is not None else round(float(rng.lognormal(6.2, 0.45)), 2)))
        count = int(count if count is not None else rng.integers(1, 7))
        cents = int(amount*100)
        parts = [cents//count]*count
        parts[-1] += cents-sum(parts)
        items = [dict(description='Synthetic service', quantity='1', unit_price=str(Decimal(part)/100),
                      line_total=str(Decimal(part)/100), currency=currency) for part in parts]
        return dict(document_id=f'synthetic21-{identifier}', invoice_number=identifier,
                    vendor_name=vendor or vendors[int(rng.integers(0, len(vendors)))],
                    invoice_date=(as_of-timedelta(days=int(rng.integers(1, 365)))).isoformat(),
                    total_amount=str(amount), currency=currency, line_items=json.dumps(items),
                    subtotal_amount=str(amount), tax_amount='0', shipping_amount='0', discount_amount='0',
                    warnings='[]', field_review='{}', status='ok', human_verified='false', data_origin='synthetic_demo')

    training = [invoice(train_rng, f'P21-TRAIN-{i:04d}', 'USD' if i < 300 else 'EUR') for i in range(600)]
    evaluation, labels = [], []

    def add(scenario, *, expected=False, duplicate=False, required_code=None, **options):
        i = len(evaluation)
        row = invoice(eval_rng, f'P21-EVAL-{i:04d}', 'USD' if i % 2 == 0 else 'EUR', **options)
        evaluation.append(row)
        labels.append(dict(document_id=row['document_id'], scenario=scenario,
                           expected_ml_outlier=expected, expected_duplicate=duplicate,
                           expected_validation_code=required_code))
        return row

    for _ in range(140):
        add('normal')
    for i in range(40):
        row = add('high_amount', expected=True, amount='50000.00' if i == 0 else str(int(eval_rng.integers(20000, 80001))))
    for _ in range(20):
        add('unusual_item_count', expected=True, count=int(eval_rng.integers(30, 61)))
    for i in range(20):
        add('unseen_vendor', expected=True, vendor=f'Demo New Vendor {i}')
    for _ in range(10):
        first = add('duplicate', duplicate=True)
        second = add('duplicate', duplicate=True)
        identity = second['document_id']
        second.update(first)
        second['document_id'] = identity
    for i in range(10):
        row = add('invalid_date', expected=None, required_code='invalid_date' if i % 2 == 0 else 'future_date')
        row['invoice_date'] = '2026-02-30' if i % 2 == 0 else (as_of+timedelta(days=20)).isoformat()
    for i in range(10):
        row = add('missing_value', expected=None, required_code='missing_field')
        row['total_amount' if i % 2 == 0 else 'vendor_name'] = 'null'
    for _ in range(10):
        row = add('ocr_uncertainty', expected=None, required_code='ocr_uncertainty')
        row['warnings'] = '["Synthetic unverified OCR field"]'
        row['field_review'] = '{"total_amount":"manual_review_required"}'
    for _ in range(10):
        row = add('inconsistent_total', expected=None, required_code='grand_total_mismatch')
        row['subtotal_amount'] = str(Decimal(row['total_amount'])-Decimal('5'))
    return pd.DataFrame(training), pd.DataFrame(evaluation), pd.DataFrame(labels)


def binary_metrics(expected, predicted):
    return dict(confusion_matrix=confusion_matrix(expected, predicted, labels=[False, True]).tolist(),
                matrix_order='rows=expected normal/outlier, columns=predicted normal/outlier',
                precision=float(precision_score(expected, predicted, zero_division=0)),
                recall=float(recall_score(expected, predicted, zero_division=0)), records=len(expected))


def evaluation_boolean(value):
    """Accept typed predictions or their CSV representation without truthy 'False'."""
    if pd.isna(value) or str(value).casefold() in {'', 'null'}:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str) and value.casefold() in {'true', 'false'}:
        return value.casefold() == 'true'
    raise ValueError(f'Invalid evaluation boolean: {value!r}')


def evaluate(analysis, labels):
    if not analysis.document_id.is_unique or not labels.document_id.is_unique:
        raise ValueError('Evaluation IDs must be unique; duplicate invoices need distinct source IDs.')
    if set(analysis.document_id) != set(labels.document_id):
        raise ValueError('Prediction/label IDs do not match exactly.')
    result = analysis.merge(labels, on='document_id', validate='one_to_one')
    for column in ('ml_anomaly', 'duplicate_flag', 'expected_duplicate', 'expected_ml_outlier'):
        result[column] = result[column].map(evaluation_boolean)
    scored = result.ml_status.isin(['SCORED', 'SCORED_DEMO'])
    expected = result.expected_ml_outlier.notna()
    measured = result[scored & expected].copy()
    expected_values = measured.expected_ml_outlier.astype(bool).tolist()
    predictions = measured.ml_anomaly.astype(bool).tolist()
    metrics = binary_metrics(expected_values, predictions)
    metrics.update(scored_rows=int(scored.sum()), skipped_rows=int((~scored).sum()),
                   labeled_but_unscored=int((expected & ~scored).sum()),
                   intended_skip_rows=int((~expected).sum()))
    fp = measured[(measured.expected_ml_outlier == False) & (measured.ml_anomaly == True)]
    fn = measured[(measured.expected_ml_outlier == True) & (measured.ml_anomaly == False)]
    columns = ['document_id', 'scenario', 'total_amount', 'currency', 'vendor_name', 'ml_decision_score']
    metrics['false_positives'] = fp[columns].to_dict('records')
    metrics['false_negatives'] = fn[columns].to_dict('records')
    metrics['by_scenario'] = {name: dict(records=len(group), scored=int(group.ml_status.isin(['SCORED','SCORED_DEMO']).sum()),
                                       predicted_outliers=int(group.ml_anomaly.eq(True).sum()))
                              for name, group in result.groupby('scenario')}
    metrics['duplicates'] = binary_metrics(result.expected_duplicate.astype(bool).tolist(), result.duplicate_flag.astype(bool).tolist())
    metrics['validation_checks'] = [dict(document_id=row.document_id, code=row.expected_validation_code,
        detected=any(f['code'] == row.expected_validation_code for f in json.loads(row.validation_findings)))
        for _, row in result[result.expected_validation_code.notna()].iterrows()]
    return metrics, result


def run_verification(output_dir, model_path, config=None):
    config = config or AnalysisConfig()
    output_dir, model_path = Path(output_dir), Path(model_path)
    baseline_path = model_path.with_name(model_path.stem+'_numeric_baseline.joblib')
    if output_dir.exists() or model_path.exists() or baseline_path.exists():
        raise ValueError('Choose new output and model paths; existing artifacts are protected.')
    training, evaluation, labels = verification_data(config)
    assert set(training.document_id).isdisjoint(evaluation.document_id)
    assert set(training.invoice_number).isdisjoint(evaluation.invoice_number)
    # No labels or scenarios are passed into either model path.
    bundle = train_model(training, config, training_kind='synthetic_demo', feature_profile='vendor_v2')
    baseline = train_model(training, config, training_kind='synthetic_demo')
    baseline_analysis = analyze_invoices(evaluation, config, baseline)
    before = analyze_invoices(evaluation, config, bundle)
    save_model(bundle, model_path)
    save_model(baseline, baseline_path)
    loaded = load_model(model_path)
    after = analyze_invoices(evaluation, config, loaded)
    pd.testing.assert_frame_equal(before, after)
    metrics, comparisons = evaluate(after, labels)
    baseline_metrics, _ = evaluate(baseline_analysis, labels)
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, frame in [('synthetic_training.csv', training), ('synthetic_evaluation.csv', evaluation),
                        ('expected_labels.csv', labels), ('analysis.csv', after),
                        ('evaluation_comparison.csv', comparisons), ('numeric_baseline_analysis.csv', baseline_analysis)]:
        write_analysis(frame, output_dir/name)
    metadata = {key: value for key, value in bundle.items() if key != 'models'}
    metadata['model_settings'] = {'n_estimators':200, 'contamination':config.contamination,
                                  'random_state':config.random_seed, 'n_jobs':1,
                                  'max_samples':'auto (min(256, currency training rows))'}
    metadata['currencies'] = {key: dict(rows=value['rows'], vendor_frequencies=value['vendor_frequencies'])
                              for key, value in bundle['models'].items()}
    metadata['evaluation_seed'] = config.random_seed+1009
    metadata['evaluation_rows'] = len(evaluation)
    metadata['numeric_baseline_model'] = str(baseline_path)
    metadata['persistence_prediction_parity'] = True
    metadata['training_evaluation_ids_disjoint'] = True
    metadata['sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output_dir.glob('*.csv')}
    (output_dir/'model_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (output_dir/'metrics.json').write_text(json.dumps(dict(vendor_v2=metrics, numeric_v1=baseline_metrics), indent=2), encoding='utf-8')
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--as-of', default=AnalysisConfig().as_of)
    args = parser.parse_args(argv)
    try:
        metrics = run_verification(args.output_dir, args.model, AnalysisConfig(as_of=args.as_of))
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print('SYNTHETIC EVALUATION ONLY:', json.dumps({key:metrics[key] for key in ('precision','recall','confusion_matrix','scored_rows','skipped_rows')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
