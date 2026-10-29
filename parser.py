"""Conservative regex parsing of invoice OCR text, without inferred amounts."""

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
import re
import unicodedata

from dateutil import parser as date_parser


@dataclass
class LineItem:
    description: str | None
    quantity: Decimal | None = None
    unit_price: Decimal | None = None
    amount: Decimal | None = None
    currency: str | None = None

    @property
    def line_total(self) -> Decimal | None:
        """Keep the amount attribute compatible with existing callers."""
        return self.amount


@dataclass
class Invoice:
    invoice_number: str | None = None
    invoice_date: date | None = None
    vendor_name: str | None = None
    total_amount: Decimal | None = None
    line_items: list[LineItem] = field(default_factory=list)
    currency: str | None = None
    warnings: list[str] = field(default_factory=list)
    subtotal_amount: Decimal | None = None
    tax_amount: Decimal | None = None
    shipping_amount: Decimal | None = None
    discount_amount: Decimal | None = None


CURRENCY = r"(?:USD|EUR|GBP|INR|CAD|AUD|JPY|CHF|CNY|Rs\.?|[$€£₹¥])"
NUMBER = r"[+-]?\d+(?:[.,' ]\d+)*"
MONEY = rf"(?:{CURRENCY}\s*)?{NUMBER}(?:\s*{CURRENCY})?"
CURRENCY_RE = re.compile(rf"(?<![A-Za-z]){CURRENCY}(?![A-Za-z])", re.I)
TOTAL_RE = re.compile(r"^(grand\s+total|total\s+amount|invoice\s+total|total)\b\s*[:#=]?\s*(.*)$", re.I)
SUMMARY_RE = re.compile(r"^(?:sub\s*total|tax|vat|gst|balance\s+due|amount\s+due|discount|shipping|paid|total|grand\s+total|invoice\s+total)\b", re.I)
INVOICE_RE = re.compile(r"^(?:invoice\s*(?:number|no\.?|#)|bill\s*(?:number|no\.?|#))\s*[:#=\-]?\s*(.*)$", re.I)
DATE_RE = re.compile(r"^(?:invoice\s+date|bill\s+date|date)\b\s*[:=\-]?\s*(.*)$", re.I)
VENDOR_RE = re.compile(r"^(?:vendor|supplier)(?:\s+name)?\b\s*[:=\-]?\s*(.*)$", re.I)
NULL_LABELS = {"", "n/a", "na", "none", "null", "-", "unknown"}


def _currency_context(currency: str | None, context: str | None) -> str | None:
    if currency == "$" and context in {"USD", "CAD", "AUD"}:
        return context
    if currency == "¥" and context in {"JPY", "CNY"}:
        return context
    return currency or context


def normalize_text(text: str) -> str:
    """Normalize Unicode and horizontal spacing, preserving lines and tables."""
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u200b", "").replace("\ufeff", "")
    return "\n".join(re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n"))


def _currency(text: str, warnings: list[str]) -> str | None:
    mapping = {"€": "EUR", "£": "GBP", "₹": "INR", "RS": "INR", "RS.": "INR"}
    tokens = {mapping.get(token.upper(), token.upper()) for token in CURRENCY_RE.findall(text)}
    if "$" in tokens and len(tokens & {"USD", "CAD", "AUD"}) == 1:
        tokens.remove("$")
    if "¥" in tokens and len(tokens & {"JPY", "CNY"}) == 1:
        tokens.remove("¥")
    if len(tokens) > 1:
        warnings.append("Multiple/incompatible currencies found; amounts are not combined.")
        return None
    return next(iter(tokens), None)


def _number(value: str, decimal_separator: str | None = None) -> Decimal:
    """Parse grouped Western/Indian numbers and decimal-comma amounts strictly."""
    value = value.strip()
    if not re.fullmatch(NUMBER, value):
        raise ValueError("not a numeric value")
    sign = ""
    if value[:1] in {"+", "-"}:
        sign, value = value[0], value[1:]
    if decimal_separator:
        decimal_mark = decimal_separator if decimal_separator in value else None
    elif "." in value and "," in value:
        decimal_mark = "." if value.rfind(".") > value.rfind(",") else ","
    elif "." in value or "," in value:
        mark = "." if "." in value else ","
        if mark == "." and value.count(mark) == 1 and len(value.split(mark)[-1]) == 3:
            raise ValueError("ambiguous dot separator; configure decimal_separator")
        decimal_mark = mark if len(value.split(mark)[-1]) != 3 else None
    else:
        decimal_mark = None
    if decimal_mark:
        if value.count(decimal_mark) != 1:
            raise ValueError("multiple decimal separators")
        integer, fraction = value.rsplit(decimal_mark, 1)
        if not fraction.isdigit():
            raise ValueError("invalid decimal fraction")
    else:
        integer, fraction = value, ""
    groups = re.split(r"[.,' ]", integer)
    if len(groups) > 1:
        separators = re.findall(r"[.,' ]", integer)
        if len(set(separators)) != 1 or not all(group.isdigit() for group in groups):
            raise ValueError("invalid thousands grouping")
        western = 1 <= len(groups[0]) <= 3 and all(len(group) == 3 for group in groups[1:])
        indian = 1 <= len(groups[0]) <= 2 and len(groups[-1]) == 3 and all(len(group) == 2 for group in groups[1:-1])
        if not (western or indian):
            raise ValueError("invalid thousands grouping")
    return Decimal(sign + "".join(groups) + ("." + fraction if fraction else ""))


def _amount(value: str, warnings: list[str], field_name: str,
            decimal_separator: str | None) -> tuple[Decimal | None, str | None]:
    currency = _currency(value, warnings)
    if len(set(CURRENCY_RE.findall(value))) > 1 and currency is None:
        warnings.append(f"{field_name}: conflicting currency labels.")
        return None, None
    raw = CURRENCY_RE.sub("", value).strip()
    try:
        return _number(raw, decimal_separator), currency
    except ValueError:
        warnings.append(f"{field_name}: missing or invalid amount '{value}'.")
        return None, currency


def _date(value: str, order: str, warnings: list[str]) -> date | None:
    numeric = re.fullmatch(r"(\d{1,4})[./-](\d{1,2})[./-](\d{1,4})", value)
    try:
        if numeric:
            a, b, c = map(int, numeric.groups())
            if len(numeric[1]) == 4:
                return date(a, b, c)
            if len(numeric[3]) != 4:
                raise ValueError("a four-digit year is required")
            choices = []
            for day, month in ((a, b), (b, a)):
                try:
                    choices.append(date(c, month, day))
                except ValueError:
                    choices.append(None)
            if order == "dmy":
                if choices[0] is None:
                    raise ValueError("invalid DMY date")
                return choices[0]
            if order == "mdy":
                if choices[1] is None:
                    raise ValueError("invalid MDY date")
                return choices[1]
            valid = set(choices) - {None}
            if len(valid) == 1:
                return valid.pop()
            if len(valid) > 1:
                warnings.append(f"Ambiguous invoice_date '{value}'; configure date_order='dmy' or 'mdy'.")
                return None
            raise ValueError("invalid date")
        # Restrict dateutil to complete month-name dates; never fill missing parts.
        month = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        day = r"\d{1,2}(?:st|nd|rd|th)?"
        if re.fullmatch(rf"(?:{day}[ -]+{month}[ ,.-]+\d{{4}}|{month}[ -]+{day}[ ,.-]+\d{{4}})", value, re.I):
            return date_parser.parse(value, fuzzy=False).date()
        raise ValueError("unsupported or incomplete date")
    except (ValueError, OverflowError) as exc:
        warnings.append(f"Invalid invoice_date '{value}': {exc}.")
        return None


def _unique(values: list, field_name: str, warnings: list[str]):
    distinct = list(dict.fromkeys(values))
    if len(distinct) == 1:
        return distinct[0]
    if len(distinct) > 1:
        warnings.append(f"Conflicting {field_name} values; left null.")
    return None


def parse_invoice(text: str, *, date_order: str = "auto",
                  decimal_separator: str | None = None) -> Invoice:
    """Extract explicit fields and simple row-oriented item tables.

    date_order auto rejects ambiguous numeric dates; dmy/mdy are explicit.
    No totals or missing quantities are calculated. No vendor is inferred from
    an unlabelled heading. Repeated conflicting field values remain null.
    """
    if date_order not in {"auto", "dmy", "mdy"}:
        raise ValueError("date_order must be auto, dmy, or mdy")
    if decimal_separator not in {None, ".", ","}:
        raise ValueError("decimal_separator must be '.' or ','")
    invoice = Invoice()
    seen_components = set()
    warnings = invoice.warnings
    text = normalize_text(text)
    lines = [line for line in text.splitlines() if line]
    # A value may sit immediately below its label. Never cross another known
    # label or table header to fill a missing field.
    joined = []
    index = 0
    labels = (INVOICE_RE, DATE_RE, VENDOR_RE, TOTAL_RE)
    while index < len(lines):
        line = lines[index]
        match = next((pattern.match(line) for pattern in labels if pattern.match(line)), None)
        if match and not match.groups()[-1].strip() and index + 1 < len(lines):
            following = lines[index + 1]
            if (not any(pattern.match(following) for pattern in labels)
                    and not SUMMARY_RE.match(following)
                    and not re.search(r'\b(?:description|quantity|qty|currency)\b', following, re.I)):
                line += ' ' + following
                index += 1
        joined.append(line)
        index += 1
    lines = joined
    document_currency = _currency(text, warnings)
    numbers, dates, vendors, totals = [], [], [], []
    in_table = False
    for line in lines:
        match = INVOICE_RE.match(line)
        if match:
            value = match[1].strip()
            if value.casefold() not in NULL_LABELS and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]*", value):
                numbers.append(value)
            else:
                warnings.append(f"Invalid or missing invoice_number: '{value}'.")
            continue
        match = DATE_RE.match(line)
        if match:
            parsed = _date(match[1], date_order, warnings)
            dates.append(parsed)
            continue
        match = VENDOR_RE.match(line)
        if match:
            vendors.append(match[1] if match[1].casefold() not in NULL_LABELS else None)
            continue
        match = TOTAL_RE.match(line)
        if match:
            # Only these anchored labels qualify; subtotal/tax/balance never do.
            amount, currency = _amount(match[2], warnings, "total_amount", decimal_separator)
            rank = {"grand total": 3, "total amount": 2, "invoice total": 2, "total": 1}[re.sub(r"\s+", " ", match[1].lower())]
            totals.append((rank, amount, _currency_context(currency, document_currency)))
            in_table = False
            continue
        component = re.fullmatch(r'(subtotal|sub total|tax|vat|gst|shipping|discount)\s*[:=]?\s+(.+)', line, re.I)
        if component:
            name = {'subtotal':'subtotal_amount', 'sub total':'subtotal_amount',
                    'tax':'tax_amount', 'vat':'tax_amount', 'gst':'tax_amount',
                    'shipping':'shipping_amount', 'discount':'discount_amount'}[component[1].lower()]
            amount, currency = _amount(component[2], warnings, name, decimal_separator)
            if currency and _currency_context(currency, document_currency) != document_currency:
                warnings.append(f'{name}: incompatible or unresolved currency.')
                amount = None
            if name in seen_components:
                warnings.append(f'{name}: repeated component requires review.')
                amount = None
            setattr(invoice, name, amount)
            seen_components.add(name)
        if SUMMARY_RE.match(line):
            in_table = False
            continue
        if re.search(r"\b(?:description|item)\b", line, re.I) and re.search(r"\b(?:qty|quantity)\b", line, re.I):
            in_table = True
            continue
        # Explicit arithmetic row, or a conventional Description Qty Price Total table.
        explicit = re.fullmatch(rf"(.+?)\s+({NUMBER})\s*[x×]\s*({MONEY})\s*=\s*({MONEY})", line, re.I)
        row = explicit
        if in_table and "|" in line:
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) == 4:
                parts = cells
            else:
                warnings.append(f"Unparsed line item: '{line}'.")
                continue
        else:
            if row is None and in_table:
                # No spaces within numeric tokens here: spaces separate columns.
                compact_number = NUMBER.replace("[.,' ]", "[.,']")
                compact_money = MONEY.replace("[.,' ]", "[.,']")
                row = re.fullmatch(rf"(.+?)\s+({compact_number})\s+({compact_money})\s+({compact_money})", line, re.I)
            if row is None:
                if in_table and not re.fullmatch(r"[-+| ]+", line):
                    warnings.append(f"Unparsed line item: '{line}'.")
                continue
            parts = list(row.groups())
        description, quantity, price, total = parts
        try:
            qty = _number(quantity, decimal_separator)
        except ValueError:
            qty = None
            warnings.append(f"Line item '{description}': missing or invalid quantity.")
        unit_price, unit_currency = _amount(price, warnings, "unit_price", decimal_separator)
        line_total, total_currency = _amount(total, warnings, "line_total", decimal_separator)
        unit_currency = _currency_context(unit_currency, total_currency or document_currency)
        total_currency = _currency_context(total_currency, unit_currency or document_currency)
        if unit_currency and total_currency and unit_currency != total_currency:
            warnings.append(f"Line item '{description}': incompatible unit and total currencies; amounts left null.")
            unit_price, line_total, item_currency = None, None, None
        else:
            item_currency = unit_currency or total_currency or document_currency
        if not description:
            warnings.append("Line item description is missing.")
        invoice.line_items.append(LineItem(description or None, qty, unit_price, line_total, item_currency))
    invoice.invoice_number = _unique(numbers, "invoice_number", warnings)
    invoice.invoice_date = _unique(dates, "invoice_date", warnings)
    invoice.vendor_name = _unique(vendors, "vendor_name", warnings)
    priority = max((entry[0] for entry in totals), default=0)
    total = _unique([(amount, currency) for rank, amount, currency in totals if rank == priority],
                    "total_amount/currency", warnings)
    if total is not None:
        invoice.total_amount, invoice.currency = total
    else:
        invoice.currency = document_currency
    for name in ("invoice_number", "invoice_date", "vendor_name", "total_amount", "currency"):
        if getattr(invoice, name) is None:
            warnings.append(f"Missing or unresolved {name}.")
    if not invoice.line_items:
        warnings.append("No supported line items found.")
    invoice.warnings = list(dict.fromkeys(warnings))
    return invoice
