from datetime import date
from decimal import Decimal
import json

import pytest

from parser import normalize_text, parse_invoice
from pipeline import invoice_record


@pytest.mark.parametrize("label", ["Invoice No", "Invoice #", "Invoice Number", "Bill No", "Invoice No."])
def test_invoice_labels(label):
    assert parse_invoice(f"{label}: INV-123/26").invoice_number == "INV-123/26"


@pytest.mark.parametrize("text,expected", [
    ("27/10/2026", date(2026, 10, 27)), ("10/27/2026", date(2026, 10, 27)),
    ("2026-10-27", date(2026, 10, 27)), ("27 October 2026", date(2026, 10, 27)),
    ("October 27, 2026", date(2026, 10, 27)), ("Oct 27th, 2026", date(2026, 10, 27)),
    ("01/01/2026", date(2026, 1, 1)),
])
def test_dates(text, expected):
    assert parse_invoice(f"Date: {text}").invoice_date == expected


def test_ambiguous_date():
    text = "Date: 03/04/2026"
    result = parse_invoice(text)
    assert result.invoice_date is None and any("Ambiguous" in w for w in result.warnings)
    assert parse_invoice(text, date_order="dmy").invoice_date == date(2026, 4, 3)
    assert parse_invoice(text, date_order="mdy").invoice_date == date(2026, 3, 4)


@pytest.mark.parametrize("text", ["31/02/2026", "October 2026", "03/04/26", "2026-13-10", "31/12/2026"])
def test_invalid_or_incompatible_date(text):
    assert parse_invoice(f"Date: {text}", date_order="mdy").invoice_date is None


def test_normalization_and_vendor():
    result = parse_invoice("Supplier：  ACME\u00a0Supplies\r\nInvoice No: ＩＮＶ-１２３\rDate: 2026-10-27")
    assert result.vendor_name == "ACME Supplies"
    assert result.invoice_number == "INV-123"
    assert normalize_text("a\t b\r\nc") == "a b\nc"


@pytest.mark.parametrize("amount,currency,expected", [
    ("$1,234.56", "$", "1234.56"), ("EUR 1.234,56", "EUR", "1234.56"),
    ("₹ 1,23,456.78", "INR", "123456.78"), ("£1 234.56", "GBP", "1234.56"),
    ("USD 1,234", "USD", "1234"), ("125.00 CAD", "CAD", "125.00"),
    ("EUR 1'234.50", "EUR", "1234.50"), ("Rs. 100.00", "INR", "100.00"),
])
def test_money(amount, currency, expected):
    invoice = parse_invoice(f"Grand Total: {amount}")
    assert invoice.total_amount == Decimal(expected)
    assert invoice.currency == currency


def test_decimal_configuration():
    assert parse_invoice("Total: EUR 1.234").total_amount is None
    assert parse_invoice("Total: EUR 1.234", decimal_separator=",").total_amount == 1234
    assert parse_invoice("Total: USD 0.125", decimal_separator=".").total_amount == Decimal("0.125")


def test_total_selection():
    text = "Subtotal: USD 100\nTax: USD 20\nTotal: USD 100\nGrand Total: USD 120\nBalance Due: USD 5"
    assert parse_invoice(text).total_amount == 120
    missing = parse_invoice("Subtotal: 100\nTax: 20\nBalance Due: 5")
    assert missing.total_amount is None


def test_conflicts_and_missing():
    result = parse_invoice("Invoice No: A\nInvoice No: B\nGrand Total: USD 100\nGrand Total: USD 200")
    assert result.invoice_number is None and result.total_amount is None
    assert any("Conflicting" in warning for warning in result.warnings)
    empty = invoice_record(parse_invoice("UNLABELLED COMPANY\nDue Date: 2026-10-27"))
    assert empty["vendor_name"] is None and empty["invoice_date"] is None
    assert json.loads(empty["line_items"]) is None


def test_line_items_and_partial_values():
    result = parse_invoice("""Vendor: ACME
Invoice Number: INV-7
Date: 2026-10-27
Description | Quantity | Unit Price | Line Total
Widget model 42 | 2 | USD 10.50 | USD 21.00
Missing values | | | USD 5.00
Grand Total: USD 26.00""")
    assert len(result.line_items) == 2
    item = result.line_items[0]
    assert (item.description, item.quantity, item.unit_price, item.line_total) == ("Widget model 42", 2, Decimal("10.50"), 21)
    partial = result.line_items[1]
    assert partial.quantity is None and partial.unit_price is None and partial.line_total == 5
    assert result.warnings
    serialized = json.loads(invoice_record(result)["line_items"])
    assert serialized[1]["quantity"] is None and serialized[0]["line_total"] == "21.00"


@pytest.mark.parametrize("row", ["Printer paper 2 50.00 100.00", "Printer paper 2 x 50.00 = 100.00"])
def test_simple_rows(row):
    result = parse_invoice("Description Qty Unit Price Amount\n" + row)
    assert len(result.line_items) == 1
    assert result.line_items[0].description == "Printer paper"
    assert result.line_items[0].quantity == 2
    assert result.line_items[0].unit_price == 50
    assert result.line_items[0].line_total == 100


def test_mixed_currencies():
    result = parse_invoice("Description | Qty | Unit Price | Line Total\nService | 2 | USD 10 | EUR 20\nGrand Total: EUR 20")
    assert result.total_amount == 20 and result.currency == "EUR"
    assert result.line_items[0].unit_price is None and result.line_items[0].line_total is None
    assert any("incompatible" in warning for warning in result.warnings)
    conflict = parse_invoice("Grand Total: USD EUR 20")
    assert conflict.total_amount is None


def test_currency_symbol_context_and_negative_total():
    result = parse_invoice("Currency: USD\nDescription | Qty | Unit Price | Line Total\nRefund | 1 | $10 | USD 10\nGrand Total -10")
    assert result.currency == "USD" and result.total_amount == -10
    assert result.line_items[0].unit_price == 10
    assert result.line_items[0].currency == "USD"


def test_explicit_null_labels():
    result = parse_invoice("Invoice No: N/A\nVendor: null")
    assert result.invoice_number is None and result.vendor_name is None


def test_repeated_page_headers_and_identical_items():
    text = "Invoice No: A\nDescription Qty Price Amount\nPaper 1 10 10\nInvoice No: A\nDescription Qty Price Amount\nPaper 1 10 10\nGrand Total: USD 20"
    result = parse_invoice(text)
    assert result.invoice_number == "A" and len(result.line_items) == 2


@pytest.mark.parametrize("options", [{"date_order": "guess"}, {"decimal_separator": ":"}])
def test_bad_configuration(options):
    with pytest.raises(ValueError):
        parse_invoice("", **options)
