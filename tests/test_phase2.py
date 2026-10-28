"""Phase 2 validation, matching, risk, ML and CLI integration regressions."""
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from anomaly_detector import features, load_model, save_model, train_model
from duplicate_detector import detect_duplicates
from parser import parse_invoice
from phase2 import ANALYSIS_COLUMNS, analyze_invoices, demo_data, main, read_records, write_analysis
from phase2_config import AnalysisConfig
from validator import money, validate_invoice, validate_record

CONFIG = AnalysisConfig(as_of='2026-10-29')


def record(**overrides):
    row = dict(document_id='source-a', invoice_number='INV-001', vendor_name='ACME Supplies',
               invoice_date='2026-10-27', total_amount='110.00', currency='USD', warnings='[]',
               field_review='{}', ocr_mean_confidence='98.123456789', status='ok', human_verified='true',
               line_items=json.dumps([dict(description='Paper', quantity='2', unit_price='50.00',
                                           line_total='100.00', currency='USD')]),
               subtotal_amount='100.00', tax_amount='10.00', shipping_amount='0', discount_amount='0')
    return {**row, **overrides}


def findings(row):
    return {f['code'] for f in validate_record(row, CONFIG)['findings']}


def analyze(rows, model=None, config=CONFIG):
    return analyze_invoices(pd.DataFrame(rows), config, model)


def test_valid_record_and_legacy_invoice_api():
    result = validate_record(record(), CONFIG)
    assert not result['findings']
    assert result['arithmetic_status'] == 'CHECKED'
    assert result['ocr_review_status'] == 'VERIFIED_INPUT'
    assert result['amount'] == Decimal('110.00')
    assert isinstance(validate_invoice(parse_invoice('Grand Total: USD 110')), list)


@pytest.mark.parametrize('value', ['NaN', 'Infinity', '-Infinity', '1e3', '1,000.00', '137.5O', '1.234,56', True])
def test_invalid_amounts_never_repaired(value):
    assert money(value) is None
    assert 'invalid_amount' in findings(record(total_amount=value))


def test_decimal_arithmetic_tolerance_and_no_float_rounding():
    row = record(total_amount='0.30', subtotal_amount='0.30', tax_amount='0',
                 line_items='[{"quantity":"3","unit_price":"0.10","line_total":"0.30","currency":"USD"}]')
    assert not findings(row)
    row['total_amount'] = '0.31'
    assert 'grand_total_mismatch' not in findings(row)
    row['total_amount'] = '0.32'
    assert 'grand_total_mismatch' in findings(row)
    row['line_items'] = '[{"quantity":"3","unit_price":"0.10","line_total":"0.32","currency":"USD"}]'
    assert 'line_total_mismatch' in findings(row)


@pytest.mark.parametrize('field,value,code', [
    ('invoice_number', 'INV-@1', 'invalid_invoice_number'),
    ('vendor_name', '12345', 'invalid_vendor'),
    ('invoice_date', '03/04/2026', 'invalid_date'),
    ('invoice_date', '2026-02-30', 'invalid_date'),
    ('invoice_date', '2026-11-01', 'future_date'),
    ('currency', '$', 'unresolved_currency'),
    ('total_amount', '-10.00', 'nonpositive_amount'),
    ('total_amount', '0', 'nonpositive_amount')])
def test_validation_fields(field, value, code):
    assert code in findings(record(**{field: value}))


@pytest.mark.parametrize('empty', [None, pd.NA, np.nan, '', 'null', 'N/A'])
def test_missing_fields(empty):
    assert 'missing_field' in findings(record(invoice_number=empty, total_amount=empty))


def test_partial_totals_do_not_assume_zero_tax_or_compare_grand_to_items():
    row = record(total_amount='137.50')
    for key in ('subtotal_amount', 'tax_amount', 'shipping_amount', 'discount_amount'):
        row.pop(key)
    result = validate_record(row, CONFIG)
    assert result['arithmetic_status'] == 'PARTIAL'
    assert findings(row) == {'total_not_verifiable'}
    row['line_items'] = '[{"quantity":"2","unit_price":null,"line_total":"100","currency":"USD"}]'
    assert 'incomplete_line_item' in findings(row)
    assert 'grand_total_mismatch' not in findings(row)


def test_mixed_item_currencies_never_summed():
    row = record(line_items='[{"quantity":"2","unit_price":"50","line_total":"999","currency":"EUR"}]')
    codes = findings(row)
    assert 'item_currency_unresolved' in codes
    assert 'subtotal_mismatch' not in codes
    assert 'line_total_mismatch' not in codes


@pytest.mark.parametrize('change', [
    {'warnings': '["unverified OCR total"]'}, {'field_review': '{"total_amount":"manual_review_required"}'},
    {'field_review': '{"invoice_date":"unresolved"}'}, {'ocr_mean_confidence': '84'},
    {'warnings': 'broken-json'}, {'field_review': '[]'}, {'ocr_mean_confidence': 'Infinity'}, {'status': 'error'}])
def test_ocr_review_survives_high_confidence_or_human_marker(change):
    result = analyze([record(**change)]).iloc[0]
    assert result.ocr_review_status == 'REQUIRED'
    assert result.review_status == 'NEEDS_OCR_REVIEW'
    assert result.risk_score == 0  # OCR uncertainty is separate from financial risk


def test_unknown_ocr_provenance_never_marked_verified():
    row = record(human_verified='false', ocr_mean_confidence='99.9')
    assert analyze([row]).iloc[0].ocr_review_status == 'NOT_ASSESSED'


def test_exact_and_fuzzy_duplicates_explained_with_stable_row_ids():
    rows = [record(), record(document_id='source-b', vendor_name=' acme supplies '),
            record(document_id='source-c', invoice_number='INV-00I', vendor_name='ACME Supplles')]
    result = analyze(rows)
    matches = json.loads(result.iloc[0].duplicate_matches)
    assert {m['type'] for m in matches} == {'EXACT_FIELDS', 'POTENTIAL'}
    assert {m['other_analysis_id'] for m in matches} == {'row-000002', 'row-000003'}
    assert all(m['explanation'] for m in matches)
    assert result.duplicate_flag.all()
    assert len(detect_duplicates(pd.DataFrame(rows), CONFIG)) == 3


def test_exact_document_and_identity_conflict():
    assert json.loads(analyze([record(), record()]).iloc[0].duplicate_matches)[0]['type'] == 'EXACT_DOCUMENT'
    other = record(document_id='other', total_amount='200')
    assert json.loads(analyze([record(), other]).iloc[0].duplicate_matches)[0]['type'] == 'IDENTITY_CONFLICT'


def test_duplicates_not_caused_by_missing_keys_or_equal_cross_currency_amounts():
    result = analyze([record(document_id='', invoice_number='null', vendor_name='null'),
                      record(document_id='null', invoice_number='null', vendor_name='null')])
    assert not result.duplicate_flag.any()
    result = analyze([record(), record(document_id='other', invoice_number='DIFFERENT', currency='EUR')])
    assert not result.duplicate_flag.any()


def test_same_identity_cross_currency_is_conflict_not_amount_match():
    result = analyze([record(), record(document_id='other', currency='EUR')])
    matches = json.loads(result.iloc[0].duplicate_matches)
    assert matches[0]['type'] == 'IDENTITY_CONFLICT'


def test_uncertain_duplicate_is_provisional_and_not_dropped():
    result = analyze([record(warnings='["OCR review"]'), record(document_id='b')])
    assert len(result) == 2
    assert all(json.loads(row.duplicate_matches)[0]['provisional'] for _, row in result.iterrows())


def test_rules_config_currency_and_explainable_risk_caps():
    row = record(total_amount='50000', invoice_date='2026-11-01')
    rows = [row]+[record(document_id=str(i), total_amount='50000', invoice_date='2026-11-01') for i in range(10)]
    result = analyze(rows)
    assert result.risk_score.between(0, 100).all()
    first = result.iloc[0]
    assert first.review_priority == 'HIGH'
    assert first.risk_score == min(100, sum(c['points'] for c in json.loads(first.risk_contributions)))
    assert sum(c['points'] for c in json.loads(first.risk_contributions) if c['category'] == 'duplicate_candidate') == 35
    assert 'exceeds configured USD 10000' in first.anomaly_explanations
    config = replace(CONFIG, high_amount_limits={'USD': '100000'})
    assert 'high_amount' not in analyze([row], config=config).iloc[0].risk_contributions


@pytest.mark.parametrize('amount,expected_score,expected_codes', [
    ('9999.99', 0, []), ('10000.00', 0, []), ('50000.00', 20, ['high_amount'])])
def test_single_amount_threshold_flag_uses_documented_priority_policy(amount, expected_score, expected_codes):
    _, holdout = demo_data(CONFIG)
    single = holdout[holdout.demo_scenario == 'high_amount'].copy()
    single['total_amount'] = amount
    single['subtotal_amount'] = amount
    single['line_items'] = json.dumps([dict(quantity='1', unit_price=amount, line_total=amount, currency='USD')])
    result = analyze_invoices(single, CONFIG).iloc[0]
    contributions = json.loads(result.risk_contributions)
    assert result.risk_score == expected_score
    assert result.review_priority == 'LOW'
    assert [code for item in contributions for code in item['codes']] == expected_codes
    assert result.rule_anomaly == bool(expected_codes)
    if expected_codes:
        assert result.review_status == 'NEEDS_ANALYST_REVIEW'
        assert 'exceeds configured USD 10000' in result.anomaly_explanations


def test_same_large_invoice_gets_history_points_only_with_suitable_peers(trained_demo):
    _, holdout, model = trained_demo
    single = holdout[holdout.demo_scenario == 'high_amount']
    isolated = analyze_invoices(single, CONFIG, model).iloc[0]
    assert isolated.risk_score == 20 and isolated.review_priority == 'LOW'
    assert isolated.ml_anomaly and isolated.ml_training_kind == 'synthetic_demo'
    batch = analyze_invoices(holdout, CONFIG, model)
    large = batch[batch.demo_scenario == 'high_amount'].iloc[0]
    assert large.risk_score == 30 and large.review_priority == 'MEDIUM'
    assert json.loads(large.risk_contributions) == [dict(
        category='amount_rules', points=30, codes=['high_amount', 'vendor_high_amount'])]
    assert 'median 462.50 from 6' in large.anomaly_explanations
    assert not any(c['category'] == 'ml_pattern' for c in json.loads(large.risk_contributions))


def test_priority_cutoffs_are_configurable_without_amount_based_override():
    _, holdout = demo_data(CONFIG)
    single = holdout[holdout.demo_scenario == 'high_amount']
    custom = replace(CONFIG, medium_priority=20)
    result = analyze_invoices(single, custom).iloc[0]
    assert result.risk_score == 20
    assert result.review_priority == 'MEDIUM'


def test_vendor_history_requires_distinct_earlier_trusted_same_currency_rows():
    rows = [record(document_id=str(i), invoice_number=f'H-{i}', invoice_date=f'2026-10-{i+1:02d}',
                   total_amount='100', subtotal_amount='100', tax_amount='0') for i in range(5)]
    rows.append(record(document_id='large', invoice_number='BIG', total_amount='600', subtotal_amount='590',
                       line_items='[{"quantity":"2","unit_price":"295","line_total":"590","currency":"USD"}]'))
    assert 'vendor_high_amount' in analyze(rows).iloc[-1].risk_contributions
    rows[0]['warnings'] = '["uncertain"]'
    assert 'vendor_high_amount' not in analyze(rows).iloc[-1].risk_contributions


def test_analysis_preserves_original_columns_index_and_values():
    original = pd.DataFrame([record(invoice_number='0000123')], index=[42])
    before = original.copy(deep=True)
    result = analyze_invoices(original, CONFIG)
    pd.testing.assert_frame_equal(original, before)
    pd.testing.assert_frame_equal(result[original.columns], before)
    assert result.iloc[0].invoice_number == '0000123'
    assert len(analyze_invoices(original.iloc[:0], CONFIG)) == 0
    assert set(ANALYSIS_COLUMNS) <= set(analyze_invoices(original.iloc[:0], CONFIG).columns)


def test_input_schema_and_reanalysis_protection():
    with pytest.raises(ValueError, match='Missing Phase 1'):
        analyze_invoices(pd.DataFrame({'total_amount': ['100']}), CONFIG)
    with pytest.raises(ValueError, match='already contains analysis'):
        analyze_invoices(analyze([record()]), CONFIG)
    with pytest.raises(ValueError, match='2000'):
        analyze([record()]*2001)


@pytest.fixture(scope='module')
def trained_demo():
    training, holdout = demo_data(CONFIG)
    return training, holdout, train_model(training, CONFIG, training_kind='synthetic_demo')


def test_model_persistence_and_heldout_anomaly(trained_demo, tmp_path):
    training, holdout, bundle = trained_demo
    assert len(training) == 600 and len(holdout) == 17
    assert set(training.document_id).isdisjoint(holdout.document_id)
    assert all(item['rows'] == 300 for item in bundle['models'].values())
    assert 'scale' in bundle['models']['USD']['pipeline'].named_steps
    path = tmp_path/'model.joblib'
    save_model(bundle, path)
    loaded = load_model(path)
    before = analyze_invoices(holdout, CONFIG, bundle)
    after = analyze_invoices(holdout, CONFIG, loaded)
    pd.testing.assert_frame_equal(before, after)
    high = after[after.demo_scenario == 'high_amount'].iloc[0]
    assert high.ml_anomaly and high.ml_status == 'SCORED_DEMO'
    assert all(c['category'] != 'ml_pattern' for c in json.loads(high.risk_contributions))
    assert after[after.demo_scenario == 'combined_high_priority'].iloc[0].review_priority == 'HIGH'
    with pytest.raises(FileExistsError):
        save_model(bundle, path)


def test_model_refuses_small_unverified_or_mixed_training(trained_demo):
    training, _, _ = trained_demo
    with pytest.raises(ValueError, match='at least'):
        train_model(pd.DataFrame([record()]*4), CONFIG)
    with pytest.raises(ValueError, match='Never mix synthetic'):
        train_model(training, CONFIG, training_kind='verified_real')
    bad = training.copy()
    bad.loc[0, 'data_origin'] = 'real'
    with pytest.raises(ValueError, match='only labeled'):
        train_model(bad, CONFIG, training_kind='synthetic_demo')
    bad = training.copy()
    bad['warnings'] = '["OCR review"]'
    with pytest.raises(ValueError, match='at least'):
        train_model(bad, CONFIG, training_kind='synthetic_demo')


def test_training_exclusions_and_real_model_risk(trained_demo):
    training, _, _ = trained_demo
    real = training.copy()
    real['data_origin'] = 'verified_test_fixture'
    real['human_verified'] = 'true'
    real.loc[0, 'warnings'] = '["unresolved OCR"]'
    bundle = train_model(real, CONFIG)
    assert bundle['excluded_rows'] == 1
    assert bundle['eligible_rows'] == 599
    high = analyze([record(total_amount='50000', subtotal_amount='49990',
                           line_items='[{"quantity":"2","unit_price":"24995","line_total":"49990","currency":"USD"}]')], bundle).iloc[0]
    assert high.ml_status == 'SCORED'
    assert high.ml_anomaly
    assert any(c['category'] == 'ml_pattern' and c['points'] == 15 for c in json.loads(high.risk_contributions))


def test_ml_skips_unverified_data_and_unsupported_currency(trained_demo):
    _, _, bundle = trained_demo
    result = analyze([record(warnings='["OCR unsure"]')], bundle).iloc[0]
    assert result.ml_status == 'SKIPPED_UNVERIFIED_OR_INVALID'
    assert result.ml_anomaly is None
    row = record(currency='JPY', line_items='[{"quantity":"2","unit_price":"50","line_total":"100","currency":"JPY"}]')
    assert analyze([row], bundle).iloc[0].ml_status == 'SKIPPED_UNSUPPORTED_CURRENCY'


def test_deterministic_model_seed_and_version_guard(trained_demo, tmp_path):
    training, _, bundle = trained_demo
    repeated = train_model(training, CONFIG, training_kind='synthetic_demo')
    matrix = features([validate_record(record(), CONFIG)])
    np.testing.assert_array_equal(bundle['models']['USD']['pipeline'].decision_function(matrix),
                                  repeated['models']['USD']['pipeline'].decision_function(matrix))
    bad = {**bundle, 'sklearn_version': 'incompatible'}
    path = tmp_path/'bad.joblib'
    save_model(bad, path)
    with pytest.raises(ValueError, match='incompatible'):
        load_model(path)


def test_csv_io_cli_preserve_input_and_never_overwrite(tmp_path):
    source = tmp_path/'source.csv'
    pd.DataFrame([record(invoice_number='0000123')]).to_csv(source, index=False)
    original = source.read_bytes()
    output = tmp_path/'analysis.csv'
    assert main(['analyze', str(source), '--output', str(output), '--as-of', CONFIG.as_of]) == 0
    assert read_records(output).iloc[0].invoice_number == '0000123'
    assert source.read_bytes() == original
    with pytest.raises(SystemExit):
        main(['analyze', str(source), '--output', str(output)])
    with pytest.raises(FileExistsError):
        write_analysis(pd.DataFrame(), source)
    assert source.read_bytes() == original


def test_cli_demo_separate_artifacts_and_train_reload(tmp_path):
    output = tmp_path/'demo'
    assert main(['demo', '--output-dir', str(output), '--as-of', CONFIG.as_of]) == 0
    metadata = json.loads((output/'model_metadata.json').read_text())
    assert metadata['training_kind'] == 'synthetic_demo'
    assert len(read_records(output/'synthetic_training.csv')) == 600
    assert len(read_records(output/'synthetic_analysis.csv')) == 17
    with pytest.raises(SystemExit):
        main(['demo', '--output-dir', str(output)])


def test_csv_duplicate_headers_rejected(tmp_path):
    source = tmp_path/'duplicate.csv'
    source.write_text('invoice_number,invoice_number\nA,B\n')
    with pytest.raises(ValueError, match='unique column'):
        read_records(source)


def test_synthetic_and_extracted_analysis_are_separate():
    with pytest.raises(ValueError, match='separately'):
        analyze([record(), record(data_origin='synthetic_demo')])


def test_training_repeated_source_identity_not_counted(trained_demo):
    training, _, _ = trained_demo
    copied = training.copy()
    copied['document_id'] = 'same-source'
    with pytest.raises(ValueError, match='at least'):
        train_model(copied, CONFIG, training_kind='synthetic_demo')


def test_phase1_real_ocr_csv_feeds_phase2_without_losing_review(tmp_path, trained_demo):
    from ocr_engine import configure_tesseract, TesseractNotFoundError
    from pipeline import discover_sources, run_pipeline
    try:
        configure_tesseract()
    except TesseractNotFoundError as exc:
        pytest.skip(str(exc))
    root = Path(__file__).resolve().parents[1]/'samples/phase3'
    extraction = run_pipeline(discover_sources(root), csv_path=tmp_path/'phase1.csv',
                              processed_dir=tmp_path/'processed', raw_dir=tmp_path/'raw')
    assert len(extraction) == 4
    analyzed = analyze_invoices(extraction, CONFIG, trained_demo[2])
    pd.testing.assert_frame_equal(analyzed[extraction.columns], extraction)
    assert analyzed.ocr_review_status.eq('REQUIRED').all()
    assert analyzed.ml_status.eq('SKIPPED_UNVERIFIED_OR_INVALID').all()
    acme = analyzed[analyzed.invoice_number == 'INV-1001']
    assert len(acme) == 2 and acme.duplicate_flag.all()
    assert all(json.loads(row.duplicate_matches)[0]['provisional'] for _, row in acme.iterrows())


@pytest.mark.parametrize('kwargs', [
    {'amount_tolerance': '-1'}, {'amount_tolerance': 'NaN'}, {'vendor_similarity': 101},
    {'contamination': 0}, {'min_training_rows': 4}, {'as_of': '03/04/2026'},
    {'high_amount_limits': {'USD': 'Infinity'}}, {'high_priority': 10},
    {'vendor_min_history': 1}, {'duplicate_date_days': -1}, {'high_amount_limits': None},
    {'random_seed': -1}, {'as_of': '20261029'}])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        AnalysisConfig(**kwargs)
