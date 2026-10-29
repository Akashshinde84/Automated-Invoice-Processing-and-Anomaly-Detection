"""Analyze extracted invoice CSV records without changing their values or OCR pipeline."""
import argparse
import csv
from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal
import json
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from anomaly_detector import amount_rules, load_model, ml_result, save_model, train_model
from duplicate_detector import duplicate_matches
from analysis_config import AnalysisConfig
from validator import CORE_FIELDS, json_value, text, validate_record

ANALYSIS_COLUMNS = [
    'analysis_id', 'validation_findings', 'arithmetic_status', 'duplicate_flag', 'duplicate_matches',
    'rule_anomaly', 'anomaly_explanations', 'ml_status', 'ml_anomaly', 'ml_decision_score',
    'ml_explanation', 'ml_training_kind', 'risk_score', 'risk_contributions', 'review_priority',
    'ocr_review_status', 'review_status', 'ocr_warnings', 'analysis_as_of', 'analysis_mode',
    'isanomaly', 'anomalyreason']


def priority_for_score(score, config=None):
    """Documented queue bands; a rule flag alone does not override the score."""
    config = config or AnalysisConfig()
    if not isinstance(score, (int, np.integer)) or not 0 <= score <= 100:
        raise ValueError('Heuristic score must be an integer from 0 to 100.')
    return 'HIGH' if score >= config.high_priority else 'MEDIUM' if score >= config.medium_priority else 'LOW'


def read_records(source):
    # Preserve identifiers, leading zeros, original decimal strings and literal nulls.
    content = Path(source).read_bytes() if isinstance(source, (str, Path)) else source.read()
    decoded = content.decode('utf-8-sig') if isinstance(content, bytes) else content
    header = next(csv.reader(StringIO(decoded)), [])
    if len(header) != len(set(header)):
        raise ValueError('Input must have unique column names.')
    frame = pd.read_csv(StringIO(decoded), dtype=str, keep_default_na=False)
    return frame


def analyze_invoices(invoices: pd.DataFrame, config=None, model=None):
    config = config or AnalysisConfig()
    if not invoices.columns.is_unique:
        raise ValueError('Input must have unique column names.')
    missing_columns = set(CORE_FIELDS)-set(invoices.columns)
    if missing_columns:
        raise ValueError(f'Missing extracted invoice columns: {sorted(missing_columns)}')
    overlap = set(ANALYSIS_COLUMNS) & set(invoices.columns)
    if overlap:
        raise ValueError(f'Input already contains analysis columns; use the original extracted CSV: {sorted(overlap)}')
    if len(invoices) > 2000:
        raise ValueError('Analysis supports at most 2000 rows per batch; duplicate matching is pairwise.')
    records = invoices.to_dict('records')
    demo_origins = [text(r.get('data_origin')) == 'synthetic_demo' for r in records]
    if any(demo_origins) and not all(demo_origins):
        raise ValueError('Analyze synthetic demonstration and extracted records separately.')
    validated = [validate_record(record, config) for record in records]
    duplicates = duplicate_matches(records, validated, config)
    output = []
    for index, (record, valid, matches) in enumerate(zip(records, validated, duplicates)):
        findings = valid['findings']
        rules = amount_rules(index, records, validated, config)
        ml = ml_result(record, valid, model)
        # Category caps prevent repeated missing fields/pair matches dominating risk.
        contributions = []
        validation_points = min(40, sum(f['points'] for f in findings))
        if validation_points:
            contributions.append(dict(category='validation', points=validation_points,
                                      codes=list(dict.fromkeys(f['code'] for f in findings if f['points']))))
        if matches:
            contributions.append(dict(category='duplicate_candidate', points=max(m['points'] for m in matches),
                                      codes=list(dict.fromkeys(m['type'] for m in matches))))
        if rules:
            contributions.append(dict(category='amount_rules', points=min(30, sum(r['points'] for r in rules)),
                                      codes=[r['code'] for r in rules]))
        if ml['anomaly'] and ml['training_kind'] == 'verified_real':
            contributions.append(dict(category='ml_pattern', points=15, codes=['isolation_forest']))
        risk = min(100, sum(c['points'] for c in contributions))
        priority = priority_for_score(risk, config)
        ocr_required = valid['ocr_review_status'] == 'REQUIRED'
        review = 'NEEDS_OCR_REVIEW' if ocr_required else 'NEEDS_ANALYST_REVIEW' if (findings or rules or matches or ml['anomaly']) else 'NOT_REVIEWED'
        explanations = [f['message'] for f in findings] + [r['message'] for r in rules]
        explanations += [m['explanation'] for m in matches]
        if ml['anomaly']:
            explanations.append(ml['explanation'])
        warnings, good = json_value(record.get('warnings'), list, [])
        if not good:
            warnings = [str(record.get('warnings'))]
        output.append(dict(
            isanomaly=bool(validation_points or rules or matches or ml['anomaly']),
            anomalyreason='; '.join(explanations),
            analysis_id=f'row-{index+1:06d}', validation_findings=json.dumps(findings),
            arithmetic_status=valid['arithmetic_status'], duplicate_flag=bool(matches),
            duplicate_matches=json.dumps(matches), rule_anomaly=bool(rules or validation_points),
            anomaly_explanations=json.dumps(explanations), ml_status=ml['status'], ml_anomaly=ml['anomaly'],
            ml_decision_score=ml['score'], ml_explanation=ml['explanation'], ml_training_kind=ml['training_kind'],
            risk_score=risk, risk_contributions=json.dumps(contributions), review_priority=priority,
            ocr_review_status=valid['ocr_review_status'], review_status=review,
            ocr_warnings=json.dumps(warnings), analysis_as_of=config.as_of,
            analysis_mode='rules_only' if model is None else model['training_kind']))
    result = invoices.copy(deep=True)
    for column in ANALYSIS_COLUMNS:
        result[column] = [row[column] for row in output]
    return result


def write_analysis(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # New files only: neither a prior analysis nor an extracted invoice CSV is overwritten.
    with path.open('x', encoding='utf-8-sig', newline='') as handle:
        frame.to_csv(handle, index=False, na_rep='null')


def demo_data(config=None):
    """Explicitly synthetic independent training and held-out demonstration cases."""
    config = config or AnalysisConfig()
    rng = np.random.default_rng(config.random_seed)
    as_of = date.fromisoformat(config.as_of)
    def record(number, amount, currency='USD', age=3, vendor='Demo Office Supplies', count=1, scenario='baseline'):
        amount = Decimal(str(amount)).quantize(Decimal('0.01'))
        # Split to cents, preserving exact item sums.
        cents = int(amount*100)
        item_cents = [cents//count] * count
        item_cents[-1] += cents-sum(item_cents)
        items = [dict(description=f'Synthetic service {i+1}', quantity='1',
                      unit_price=str(Decimal(c)/100), line_total=str(Decimal(c)/100), currency=currency)
                 for i, c in enumerate(item_cents)]
        return dict(document_id=f'synthetic-{number}', invoice_number=number, vendor_name=vendor,
                    invoice_date=(as_of-timedelta(days=age)).isoformat(), total_amount=str(amount),
                    currency=currency, line_items=json.dumps(items), warnings='[]', field_review='{}',
                    status='ok', data_origin='synthetic_demo', human_verified='false',
                    subtotal_amount=str(amount), tax_amount='0', shipping_amount='0', discount_amount='0',
                    demo_scenario=scenario)
    training = []
    for i in range(600):
        currency = 'USD' if i < 300 else 'EUR'
        training.append(record(f'TRAIN-{i:04d}', round(float(rng.lognormal(6, 0.5)), 2), currency,
                               age=int(rng.integers(15, 365)), count=int(rng.integers(1, 7)),
                               vendor=f'Demo Vendor {i%8}', scenario='training_baseline'))
    holdout = [record('DEMO-001', '400', count=3, scenario='duplicate_original'),
               record('DEMO-001', '400', count=3, scenario='exact_fields_duplicate'),
               record('DEMO-00I', '400', count=3, vendor='Demo Office Supplles', scenario='fuzzy_duplicate'),
               record('DEMO-HIGH', '50000', count=3, scenario='high_amount'),
               record('DEMO-FUTURE', '400', age=-10, scenario='future_date'),
               record('DEMO-BAD-TOTAL', '400', scenario='inconsistent_total'),
               record('DEMO-OCR', '400', scenario='ocr_uncertainty'),
               record('DEMO-MISSING', '400', scenario='missing_fields'),
               record('DEMO-EUR', '500', 'EUR', scenario='different_currency')]
    holdout[1]['document_id'] += '-copy'
    holdout[5]['subtotal_amount'] = '300'
    holdout[6]['warnings'] = '["Synthetic demonstration: unverified OCR amount"]'
    holdout[6]['field_review'] = '{"total_amount":"manual_review_required"}'
    holdout[7]['invoice_number'], holdout[7]['total_amount'] = 'null', 'null'
    # Demonstrate earlier vendor baseline without training on the evaluation rows.
    holdout.extend(record(f'HISTORY-{i}', str(400+i*25), age=20+i, scenario='vendor_history') for i in range(6))
    holdout.append(record('DEMO-NORMAL', '450', count=3, scenario='normal'))
    holdout.append(record('DEMO-COMBINED', '50000', age=-10, scenario='combined_high_priority'))
    holdout[-1]['subtotal_amount'] = '300'
    return pd.DataFrame(training), pd.DataFrame(holdout)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    analyze = sub.add_parser('analyze', help='Analyze an existing extracted invoice CSV')
    analyze.add_argument('input', type=Path)
    analyze.add_argument('--output', type=Path, required=True)
    analyze.add_argument('--model', type=Path)
    train = sub.add_parser('train', help='Train on explicitly verified or synthetic records')
    train.add_argument('input', type=Path)
    train.add_argument('--model', type=Path, required=True)
    train.add_argument('--kind', choices=['verified_real', 'synthetic_demo'], required=True)
    train.add_argument('--feature-profile', choices=['numeric_v1', 'vendor_v2'], default='numeric_v1')
    demo = sub.add_parser('demo', help='Write a separate synthetic training/demo dataset and model')
    demo.add_argument('--output-dir', type=Path, required=True)
    for command in (analyze, train, demo):
        command.add_argument('--config', type=Path)
        command.add_argument('--as-of', help='Explicit YYYY-MM-DD analysis date (default: India local date)')
    args = parser.parse_args(argv)
    try:
        config = AnalysisConfig.load(args.config)
        if args.as_of:
            config = replace(config, as_of=args.as_of)
        if args.command == 'analyze':
            if args.output.exists():
                raise ValueError('Output exists; choose a new filename.')
            model = load_model(args.model) if args.model else None
            result = analyze_invoices(read_records(args.input), config, model)
            write_analysis(result, args.output)
            print(f'Analyzed {len(result)} records -> {args.output}')
            print(result[['analysis_id', 'risk_score', 'review_priority', 'ocr_review_status', 'ml_status']].to_string(index=False))
        elif args.command == 'train':
            if args.model.exists():
                raise ValueError('Model exists; choose a new filename.')
            model = train_model(read_records(args.input), config, training_kind=args.kind,
                                feature_profile=args.feature_profile)
            save_model(model, args.model)
            print(f'Saved {args.kind} model -> {args.model}; eligible={model["eligible_rows"]}, excluded={model["excluded_rows"]}')
        else:
            args.output_dir.mkdir(parents=True, exist_ok=False)
            training, holdout = demo_data(config)
            model = train_model(training, config, training_kind='synthetic_demo')
            write_analysis(training, args.output_dir/'synthetic_training.csv')
            write_analysis(holdout, args.output_dir/'synthetic_holdout.csv')
            save_model(model, args.output_dir/'synthetic_model.joblib')
            # Verify persistence is also the actual demonstration prediction path.
            model = load_model(args.output_dir/'synthetic_model.joblib')
            result = analyze_invoices(holdout, config, model)
            write_analysis(result, args.output_dir/'synthetic_analysis.csv')
            metadata = {k: v for k, v in model.items() if k != 'models'}
            metadata['currencies'] = {k: {name: value for name, value in item.items() if name != 'pipeline'}
                                      for k, item in model['models'].items()}
            (args.output_dir/'model_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
            print(f'SYNTHETIC DEMO ONLY: {len(training)} training rows, {len(holdout)} held-out cases -> {args.output_dir}')
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
