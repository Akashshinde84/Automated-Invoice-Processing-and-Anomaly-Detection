"""Explainable pairwise duplicate candidates; no records are removed."""
from itertools import combinations
from decimal import Decimal
import json

import pandas as pd
from rapidfuzz.fuzz import ratio

from phase2_config import AnalysisConfig
from validator import normalized, text, validate_record


def duplicate_matches(records, validated, config):
    matches = [[] for _ in records]
    for i, j in combinations(range(len(records)), 2):
        left, right = validated[i], validated[j]
        a, b = records[i], records[j]
        same_document = bool(text(a.get('document_id')) and
                             text(a.get('document_id')) == text(b.get('document_id')))
        vendor_a, vendor_b = normalized(left['vendor']), normalized(right['vendor'])
        number_a, number_b = text(left['number']).casefold(), text(right['number']).casefold()
        same_vendor = bool(vendor_a and vendor_a == vendor_b)
        same_number = bool(number_a and number_a == number_b)
        same_currency = bool(left['currency'] and left['currency'] == right['currency'])
        amounts = left['amount'], right['amount']
        dates = left['date'], right['date']
        same_amount = same_currency and all(v is not None for v in amounts) and amounts[0] == amounts[1]
        close_amount = (same_currency and all(v is not None for v in amounts) and
                        abs(amounts[0]-amounts[1]) <= Decimal(str(config.amount_tolerance)))
        same_date = all(dates) and dates[0] == dates[1]
        close_date = all(dates) and abs((dates[0]-dates[1]).days) <= config.duplicate_date_days
        kind, explanation, points = None, '', 0
        if same_document:
            kind, points = 'EXACT_DOCUMENT', 35
            explanation = 'Same nonempty document_id; repeated source-document identity.'
        elif same_vendor and same_number and same_amount and same_date:
            kind, points = 'EXACT_FIELDS', 35
            explanation = 'Normalized vendor, invoice number, date, currency and Decimal amount match exactly.'
        elif same_vendor and same_number:
            kind, points = 'IDENTITY_CONFLICT', 25
            explanation = 'Same vendor and invoice number, but date, amount or currency differs or is unresolved; investigate versions/OCR.'
        elif (vendor_a and vendor_b and ratio(vendor_a, vendor_b) >= config.vendor_similarity
              and close_amount and close_date
              and (not number_a or not number_b or ratio(number_a, number_b) >= config.invoice_similarity)):
            kind, points = 'POTENTIAL', 20
            explanation = (f'Vendor similarity {ratio(vendor_a, vendor_b):.1f} >= {config.vendor_similarity}; '
                           f'same-currency amounts within {config.amount_tolerance}, dates within '
                           f'{config.duplicate_date_days} days, and similar or missing invoice numbers.')
        if kind:
            uncertain = left['ocr_review_status'] == 'REQUIRED' or right['ocr_review_status'] == 'REQUIRED'
            if uncertain:
                explanation += ' Underlying OCR fields require review; this match is provisional.'
            for first, second in ((i, j), (j, i)):
                matches[first].append(dict(other_analysis_id=f'row-{second+1:06d}',
                                           other_document_id=text(records[second].get('document_id')),
                                           type=kind, explanation=explanation, points=points,
                                           provisional=uncertain))
    return matches


def detect_duplicates(invoices: pd.DataFrame, config: AnalysisConfig | None = None) -> pd.DataFrame:
    config = config or AnalysisConfig()
    if len(invoices) > 2000:
        raise ValueError('Duplicate comparison supports at most 2000 rows per batch.')
    records = invoices.to_dict('records')
    matches = duplicate_matches(records, [validate_record(r, config) for r in records], config)
    result = invoices.copy(deep=True)
    result['duplicate_flag'] = [bool(m) for m in matches]
    result['duplicate_matches'] = [json.dumps(m) for m in matches]
    return result
