"""Synthetic evaluation integrity, compatible feature profiles and risk boundaries."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import anomaly_detector
from anomaly_detector import features, load_model, ml_result, save_model, train_model
from invoice_analysis import analyze_invoices, priority_for_score, read_records, main as analysis_main
from model_evaluation import binary_metrics, evaluate, evaluation_boolean, run_verification, verification_data
from analysis_config import AnalysisConfig
from validator import normalized, validate_record

CONFIG = AnalysisConfig(as_of='2026-10-29')


@pytest.fixture(scope='module')
def verification(tmp_path_factory):
    root = tmp_path_factory.mktemp('model_evaluation')
    metrics = run_verification(root/'evaluation', root/'vendor.joblib', CONFIG)
    return root, metrics


def test_reproducible_data_separate_labels_and_no_identity_leakage():
    training, evaluation, labels = verification_data(CONFIG)
    repeated = verification_data(CONFIG)
    for first, second in zip((training, evaluation, labels), repeated):
        pd.testing.assert_frame_equal(first, second)
    assert len(training) == 600 and len(evaluation) == len(labels) == 280
    assert set(training.document_id).isdisjoint(evaluation.document_id)
    assert set(training.invoice_number).isdisjoint(evaluation.invoice_number)
    assert evaluation.document_id.is_unique
    assert not any('expected' in col or col == 'scenario' for col in training.columns)
    assert not any('expected' in col or col == 'scenario' for col in evaluation.columns)
    assert labels.expected_ml_outlier.isna().sum() == 40
    different = verification_data(replace(CONFIG, random_seed=43))
    assert not training.equals(different[0])


def test_pipeline_persistence_category_fit_and_unseen_handling(verification):
    root, _ = verification
    training, evaluation, labels = verification_data(CONFIG)
    model = load_model(root/'vendor.joblib')
    baseline = load_model(root/'vendor_numeric_baseline.joblib')
    assert model['schema_version'] == 2 and baseline['schema_version'] == 1
    assert model['eligible_rows'] == 600 and model['excluded_rows'] == 0
    for currency, fitted in model['models'].items():
        group = training[training.currency == currency]
        assert fitted['rows'] == 300
        expected = group.vendor_name.map(normalized).value_counts(normalize=True).to_dict()
        assert fitted['vendor_frequencies'] == expected
        preprocess = fitted['pipeline'].named_steps['preprocess']
        encoder = preprocess.named_transformers_['vendor'].named_steps['encode']
        assert set(encoder.categories_[0]) == set(expected)
        assert encoder.handle_unknown == 'ignore'
    unseen_id = labels[labels.scenario == 'unseen_vendor'].iloc[0].document_id
    row = evaluation[evaluation.document_id == unseen_id].iloc[0].to_dict()
    valid = validate_record(row, CONFIG)
    result = ml_result(row, valid, model)
    assert result['status'] == 'SCORED_DEMO'
    assert np.isfinite(result['score'])
    assert 'Vendor was unseen in training' in result['explanation']
    assert features([valid], model['models'][row['currency']]['vendor_frequencies']).iloc[0].vendor_frequency == 0
    matrix = features([valid], model['models'][row['currency']]['vendor_frequencies'])
    matrix.loc[0, 'item_count'] = np.nan
    assert np.isfinite(model['models'][row['currency']]['pipeline'].decision_function(matrix)).all()
    row['vendor_name'] = 'null'
    assert ml_result(row, validate_record(row, CONFIG), model)['status'] == 'SKIPPED_UNVERIFIED_OR_INVALID'


def test_metrics_account_for_skips_false_positives_and_false_negatives(verification):
    root, metrics = verification
    assert metrics['scored_rows'] == metrics['records'] == 240
    assert metrics['skipped_rows'] == metrics['intended_skip_rows'] == 40
    assert metrics['labeled_but_unscored'] == 0
    tn, fp = metrics['confusion_matrix'][0]
    fn, tp = metrics['confusion_matrix'][1]
    assert tn+fp+fn+tp == 240
    assert len(metrics['false_positives']) == fp
    assert len(metrics['false_negatives']) == fn
    assert metrics['precision'] == pytest.approx(tp/(tp+fp))
    assert metrics['recall'] == pytest.approx(tp/(tp+fn))
    assert all(check['detected'] for check in metrics['validation_checks'])
    assert metrics['duplicates']['confusion_matrix'] == [[260, 0], [0, 20]]
    comparison = read_records(root/'evaluation/evaluation_comparison.csv')
    skipped = comparison[comparison.scenario.isin(['missing_value','invalid_date','ocr_uncertainty','inconsistent_total'])]
    assert skipped.ml_status.eq('SKIPPED_UNVERIFIED_OR_INVALID').all()
    assert skipped.ml_anomaly.eq('null').all()
    report = json.loads((root/'evaluation/metrics.json').read_text())
    assert 'numeric_v1' in report and 'vendor_v2' in report


def test_label_changes_cannot_change_saved_predictions(verification):
    root, _ = verification
    analysis = pd.read_csv(root/'evaluation/analysis.csv', keep_default_na=False)
    _, _, labels = verification_data(CONFIG)
    original_scores = analysis.ml_decision_score.copy()
    changed = labels.copy()
    mask = changed.expected_ml_outlier.notna()
    changed.loc[mask, 'expected_ml_outlier'] = ~changed.loc[mask, 'expected_ml_outlier'].astype(bool)
    evaluate(analysis, changed)
    pd.testing.assert_series_equal(analysis.ml_decision_score, original_scores)
    with pytest.raises(ValueError, match='IDs do not match'):
        evaluate(analysis, changed.iloc[:-1])


def test_metrics_known_confusion_matrix_and_zero_division():
    result = binary_metrics([False, False, True, True], [False, True, False, True])
    assert result['confusion_matrix'] == [[1, 1], [1, 1]]
    assert result['precision'] == result['recall'] == .5
    assert binary_metrics([True], [False])['precision'] == 0
    assert evaluation_boolean('False') is False
    assert evaluation_boolean('True') is True
    assert evaluation_boolean('null') is None
    with pytest.raises(ValueError):
        evaluation_boolean('not-a-prediction')


def test_legacy_profile_and_artifacts_remain_supported(verification):
    root, _ = verification
    baseline = load_model(root/'vendor_numeric_baseline.joblib')
    assert baseline['features'] == anomaly_detector.FEATURES
    # Older numeric-model artifacts do not carry feature_profile metadata.
    baseline.pop('feature_profile')
    path = root/'legacy.joblib'
    save_model(baseline, path)
    assert load_model(path)['schema_version'] == 1
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_model(baseline, path)
    assert path.read_bytes() == before


def test_failed_new_model_write_is_cleaned_up(tmp_path, monkeypatch):
    def fail(bundle, handle):
        handle.write(b'incomplete')
        raise OSError('simulated serialization failure')
    monkeypatch.setattr(anomaly_detector.joblib, 'dump', fail)
    path = tmp_path/'new.joblib'
    with pytest.raises(OSError):
        save_model({}, path)
    assert not path.exists()


@pytest.mark.parametrize('score,expected', [(0,'LOW'), (24,'LOW'), (25,'MEDIUM'), (59,'MEDIUM'), (60,'HIGH'), (100,'HIGH')])
def test_exact_default_risk_boundaries(score, expected):
    assert priority_for_score(score, CONFIG) == expected


@pytest.mark.parametrize('score', [-1, 101, 25.5])
def test_invalid_risk_values_rejected(score):
    with pytest.raises(ValueError):
        priority_for_score(score, CONFIG)


def test_verification_never_overwrites_prior_artifacts(verification):
    root, _ = verification
    path = root/'vendor.joblib'
    before = path.read_bytes()
    with pytest.raises(ValueError, match='protected'):
        run_verification(root/'evaluation', path, CONFIG)
    assert path.read_bytes() == before


def test_saved_model_selection_in_dashboard_and_export_fields(verification, tmp_path):
    from streamlit.testing.v1 import AppTest
    root, _ = verification
    training, _, _ = verification_data(CONFIG)
    sample = tmp_path/'output/ocr_accuracy_final/invoices.csv'
    sample.parent.mkdir(parents=True)
    training.iloc[:3].to_csv(sample, index=False)
    model = tmp_path/'models/vendor.joblib'
    save_model(load_model(root/'vendor.joblib'), model)
    wrapper = tmp_path/'dashboard_test.py'
    wrapper.write_text('from pathlib import Path\nimport app\napp.ROOT = Path('+repr(str(tmp_path))+')\napp.main()\n')
    app = AppTest.from_file(str(wrapper)).run(timeout=30)
    selection = next(select for select in app.selectbox if select.label == 'Saved local model')
    selection.set_value(str(Path('models')/'vendor.joblib')).run(timeout=30)
    assert not app.exception
    assert any('synthetic_demo' in item.value for item in app.info)
    assert any('synthetic_demo' in item.value for item in app.warning)
    table = next(d.value for d in app.dataframe if 'ml_anomaly' in d.value.columns)
    assert table.ml_status.eq('SCORED_DEMO').all()
    assert {'risk_score','review_priority','ml_anomaly','ml_decision_score'} <= set(table.columns)
    assert len(app.get('download_button')) == 1
    result = analyze_invoices(training.iloc[:3], CONFIG, load_model(model))
    export = pd.read_csv(__import__('io').StringIO(result.to_csv(index=False)))
    assert {'anomaly_explanations','risk_contributions','ocr_review_status','ml_anomaly'} <= set(export.columns)


def test_cli_optional_profile_can_be_trained_and_loaded(verification, tmp_path):
    root, _ = verification
    path = tmp_path/'cli_vendor.joblib'
    assert analysis_main(['train', str(root/'evaluation/synthetic_training.csv'), '--model', str(path),
                        '--kind', 'synthetic_demo', '--feature-profile', 'vendor_v2', '--as-of', CONFIG.as_of]) == 0
    assert load_model(path)['schema_version'] == 2
