"""Validate extracted values conservatively without repairing the originals."""
from dataclasses import asdict
from datetime import date
from decimal import Decimal, InvalidOperation, localcontext
import json
import re
import unicodedata

import pandas as pd

from parser import Invoice
from analysis_config import AnalysisConfig

CORE_FIELDS = ('invoice_number', 'vendor_name', 'invoice_date', 'total_amount', 'currency')
NULLS = {'', 'null', 'none', 'nan', 'n/a', 'na', 'unknown', '-'}


def missing(value):
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().casefold() in NULLS
    if isinstance(value, (list, dict, tuple)):
        return False
    return bool(pd.isna(value))


def text(value):
    return '' if missing(value) else unicodedata.normalize('NFKC', str(value)).strip()


def normalized(value):
    return re.sub(r'[^\w]+', '', text(value).casefold(), flags=re.UNICODE)


def money(value):
    """Extraction exports canonical decimals; never guess locale/OCR substitutions."""
    value = text(value)
    if not re.fullmatch(r'[+-]?\d{1,18}(?:\.\d{1,6})?', value):
        return None
    try:
        result = Decimal(value)
        return result if result.is_finite() else None
    except InvalidOperation:
        return None


def iso_date(value):
    value = text(value)
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def json_value(value, expected, default):
    if missing(value):
        return default, True
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
        if parsed is None:
            return default, True
        return (parsed, True) if isinstance(parsed, expected) else (default, False)
    except (ValueError, TypeError):
        return default, False


def validate_record(record: dict, config: AnalysisConfig | None = None) -> dict:
    config = config or AnalysisConfig()
    findings = []
    def add(code, field, message, points=10):
        findings.append(dict(code=code, field=field, message=message, points=points))
    for field in CORE_FIELDS:
        if missing(record.get(field)):
            add('missing_field', field, f'{field} is missing or unresolved.')
    number = text(record.get('invoice_number'))
    if number and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_./-]{0,99}', number):
        add('invalid_invoice_number', 'invoice_number', f'Unsupported invoice number: {number!r}.')
        number = ''
    vendor = text(record.get('vendor_name'))
    if vendor and (len(vendor) > 250 or not any(c.isalpha() for c in vendor) or not all(c.isprintable() for c in vendor)):
        add('invalid_vendor', 'vendor_name', 'Vendor must contain a readable name.')
        vendor = ''
    invoice_date = iso_date(record.get('invoice_date'))
    if text(record.get('invoice_date')) and invoice_date is None:
        add('invalid_date', 'invoice_date', 'Date must be a valid unambiguous YYYY-MM-DD value.', 15)
    if invoice_date and invoice_date > date.fromisoformat(config.as_of):
        add('future_date', 'invoice_date', f'Invoice date is after analysis date {config.as_of}.', 15)
    if invoice_date and (date.fromisoformat(config.as_of) - invoice_date).days > config.max_invoice_age_days:
        add('old_invoice', 'invoice_date', f'Invoice is older than {config.max_invoice_age_days} days as of {config.as_of}.', 10)
    amount = money(record.get('total_amount'))
    nonfinite_literal = (isinstance(record.get('total_amount'), str) and
                         record['total_amount'].strip().casefold() in {'nan', 'infinity', '-infinity', '+infinity'})
    if (text(record.get('total_amount')) or nonfinite_literal) and amount is None:
        add('invalid_amount', 'total_amount', 'Total is not a finite canonical Decimal amount.', 15)
    elif amount is not None and amount <= 0:
        add('nonpositive_amount', 'total_amount', 'Total is zero or negative; check credit-note context.')
    currency = text(record.get('currency')).upper()
    if currency and not re.fullmatch('[A-Z]{3}', currency):
        add('unresolved_currency', 'currency', 'A three-letter currency code is needed; symbols are ambiguous.')
        currency = ''
    warnings, valid_warnings = json_value(record.get('warnings'), list, [])
    reviews, valid_reviews = json_value(record.get('field_review'), dict, {})
    if not valid_warnings or not valid_reviews:
        add('invalid_ocr_metadata', 'warnings/field_review', 'Malformed OCR metadata requires source review.', 0)
    try:
        score = Decimal(text(record.get('ocr_mean_confidence')))
        if not score.is_finite():
            score = None
    except InvalidOperation:
        score = None
    confidence_bad = (not missing(record.get('ocr_mean_confidence')) and
                      (score is None or not 0 <= score <= 100))
    low_confidence = score is not None and score < Decimal(str(config.low_ocr_confidence))
    unresolved = any(str(v).casefold() not in {'verified', 'not_required'} for v in reviews.values())
    required = bool(warnings or unresolved or low_confidence or confidence_bad or
                    not valid_warnings or not valid_reviews or text(record.get('status')) == 'error')
    if required:
        add('ocr_uncertainty', 'ocr', 'OCR warnings, review flags, errors or low/invalid confidence require source review.', 0)
    ocr_status = 'REQUIRED' if required else 'NOT_ASSESSED'
    if text(record.get('data_origin')) == 'synthetic_demo' and not required:
        ocr_status = 'NOT_APPLICABLE_DEMO'
    human_verified = text(record.get('human_verified')).casefold() == 'true'
    if human_verified and not required:
        ocr_status = 'VERIFIED_INPUT'

    items, valid_items = json_value(record.get('line_items'), list, [])
    arithmetic = 'NOT_AVAILABLE'
    if not valid_items:
        add('invalid_line_items', 'line_items', 'Line items must be a JSON list of objects.')
    tolerance = Decimal(str(config.amount_tolerance))
    totals = []
    complete = bool(items) and valid_items
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            add('invalid_line_item', 'line_items', f'Line {index+1} is not an object.')
            complete = False
            continue
        qty, price, total = (money(item.get(name)) for name in ('quantity', 'unit_price', 'line_total'))
        compatible = bool(currency and text(item.get('currency')).upper() == currency)
        if not compatible:
            add('item_currency_unresolved', 'line_items', f'Line {index+1} currency is missing or differs from invoice currency.', 0)
            complete = False
        if qty is None or price is None or total is None:
            add('incomplete_line_item', 'line_items', f'Line {index+1} has missing or invalid numeric values.', 0)
            complete = False
        elif compatible:
            with localcontext() as context:
                context.prec = 60
                expected = qty * price
                if abs(expected-total) > tolerance:
                    add('line_total_mismatch', 'line_items', f'Line {index+1}: quantity x unit price = {expected}, observed line total = {total}.', 20)
            totals.append(total)
    components = ['subtotal_amount', 'tax_amount', 'shipping_amount', 'discount_amount']
    values = [money(record.get(name)) for name in components]
    for name, value in zip(components, values):
        if not missing(record.get(name)) and value is None:
            add('invalid_component', name, f'{name} is not a valid canonical Decimal amount.')
    if items:
        arithmetic = 'PARTIAL'
    if all(value is not None for value in values) and amount is not None and currency:
        subtotal, tax, shipping, discount = values
        expected = subtotal + tax + shipping - discount
        arithmetic = 'CHECKED'
        if abs(expected-amount) > tolerance:
            add('grand_total_mismatch', 'total_amount', f'Explicit subtotal + tax + shipping - discount = {expected}, observed total = {amount}.', 25)
    if complete and values[0] is not None and abs(sum(totals)-values[0]) > tolerance:
        add('subtotal_mismatch', 'subtotal_amount', f'Complete line totals sum to {sum(totals)}, explicit subtotal is {values[0]}.', 20)
    if arithmetic != 'CHECKED':
        add('total_not_verifiable', 'total_amount', 'Grand-total reconciliation needs explicit subtotal, tax, shipping and discount; absent components are not assumed zero.', 0)
    return dict(findings=findings, number=number, vendor=vendor, date=invoice_date,
                amount=amount, currency=currency, items=items, arithmetic_status=arithmetic,
                ocr_review_status=ocr_status, human_verified=human_verified)


def validate_invoice(invoice: Invoice) -> list[str]:
    """Keep the original parser.Invoice entry point usable."""
    record = asdict(invoice)
    record['invoice_date'] = invoice.invoice_date.isoformat() if invoice.invoice_date else None
    record['line_items'] = [dict(description=item.description, quantity=item.quantity,
                                 unit_price=item.unit_price, line_total=item.amount,
                                 currency=item.currency) for item in invoice.line_items]
    return [item['message'] for item in validate_record(record)['findings']]
