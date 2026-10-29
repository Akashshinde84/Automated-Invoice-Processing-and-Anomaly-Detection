"""Project-brief behaviors including statistical and export boundaries."""
from dataclasses import replace
import json

import pandas as pd
import pytest

from analysis_config import AnalysisConfig
from anomaly_detector import amount_rules
from invoice_analysis import analyze_invoices, demo_data
from parser import parse_invoice
from pipeline import invoice_record
from reporting import csv_report
from validator import validate_record

CONFIG = AnalysisConfig(as_of='2026-10-29')


def test_age_boundary_future_and_required_export_columns():
    _, frame = demo_data(CONFIG)
    frame = frame.iloc[:1].copy()
    for value, old, future in [('2026-07-31', False, False), ('2026-07-30', True, False),
                               ('2026-10-30', False, True)]:
        frame['invoice_date'] = value
        result = analyze_invoices(frame, CONFIG)
        codes = {f['code'] for f in json.loads(result.iloc[0].validation_findings)}
        assert ('old_invoice' in codes) == old
        assert ('future_date' in codes) == future
        assert bool(result.iloc[0].isanomaly) == (old or future)
        assert 'isanomaly,anomalyreason' in csv_report(result).decode('utf-8-sig')


def test_zscore_uses_earlier_distinct_same_currency_history():
    _, frame = demo_data(CONFIG)
    base = frame.iloc[0].to_dict()
    history = [dict(base, invoice_number=f'H-{i}', document_id=f'h-{i}',
                    invoice_date=f'2026-10-{i+1:02d}', total_amount=str(v))
               for i, v in enumerate([90, 95, 100, 105, 110])]
    current = dict(base, invoice_number='CURRENT', document_id='current', total_amount='150')
    records = history+[current]
    def codes(rows):
        rows = [dict(row, subtotal_amount=row['total_amount'],
                     line_items=json.dumps([dict(description='Service', quantity='1',
                         unit_price=row['total_amount'], line_total=row['total_amount'], currency=row['currency'])]))
                for row in rows]
        valid = [validate_record(row, CONFIG) for row in rows]
        return {r['code'] for r in amount_rules(len(rows)-1, rows, valid, CONFIG)}
    assert 'vendor_zscore' in codes(records)
    assert 'vendor_zscore' not in codes(history[:4]+[current])
    assert 'vendor_zscore' not in codes([dict(row, total_amount='100') for row in history]+[current])
    assert 'vendor_zscore' not in codes([dict(row, currency='EUR') for row in history]+[current])
    assert 'vendor_zscore' not in codes([dict(row, invoice_date='2026-10-29') for row in history]+[current])
    assert 'vendor_zscore' not in codes([history[0]]*5+[current])


def test_explicit_components_reconcile_and_missing_values_remain_unknown():
    text = '''Invoice No: R-1
Vendor: Test Services
Date: 2026-10-27
Description Qty Price Total
Service 2 50.00 100.00
Subtotal: USD 100.00
Tax: USD 10.00
Shipping: USD 0.00
Discount: USD 0.00
Grand Total: USD 110.00'''
    record = invoice_record(parse_invoice(text))
    valid = validate_record(record, CONFIG)
    assert valid['arithmetic_status'] == 'CHECKED'
    assert not any(f['code'].endswith('mismatch') for f in valid['findings'])
    record['total_amount'] = '111.00'
    assert any(f['code'] == 'grand_total_mismatch' for f in validate_record(record, CONFIG)['findings'])
    record['tax_amount'] = None
    assert validate_record(record, CONFIG)['arithmetic_status'] == 'PARTIAL'


@pytest.mark.parametrize('kwargs', [{'max_invoice_age_days': -1}, {'max_invoice_age_days': 1.5},
                                   {'vendor_zscore_threshold': 0}, {'vendor_zscore_threshold': float('nan')}])
def test_invalid_new_policy(kwargs):
    with pytest.raises(ValueError):
        replace(CONFIG, **kwargs)
