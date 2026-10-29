"""Validated, serializable thresholds for invoice review, not fraud probability."""
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import json
import re
from pathlib import Path


def local_today():
    return datetime.now(timezone(timedelta(hours=5, minutes=30))).date()


@dataclass(frozen=True)
class AnalysisConfig:
    as_of: str = field(default_factory=lambda: local_today().isoformat())
    amount_tolerance: str = '0.01'
    high_amount_limits: dict = field(default_factory=lambda: {
        'USD': '10000', 'EUR': '10000', 'GBP': '10000', 'INR': '500000'})
    vendor_similarity: float = 90
    invoice_similarity: float = 85
    duplicate_date_days: int = 7
    vendor_min_history: int = 5
    vendor_amount_multiplier: float = 4
    vendor_zscore_threshold: float = 3
    max_invoice_age_days: int = 90
    low_ocr_confidence: float = 85
    min_training_rows: int = 100
    contamination: float = 0.05
    random_seed: int = 42
    medium_priority: int = 25
    high_priority: int = 60

    def __post_init__(self):
        if not isinstance(self.as_of, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', self.as_of):
            raise ValueError('as_of must be YYYY-MM-DD')
        date.fromisoformat(self.as_of)
        if not isinstance(self.high_amount_limits, dict):
            raise ValueError('high_amount_limits must map currency codes to amounts')
        if not isinstance(self.random_seed, int) or not 0 <= self.random_seed < 2**32:
            raise ValueError('random_seed must be an unsigned 32-bit integer')
        try:
            tolerance = Decimal(str(self.amount_tolerance))
            if not tolerance.is_finite() or tolerance < 0:
                raise ValueError('amount_tolerance must be finite and nonnegative')
            for currency, limit in self.high_amount_limits.items():
                if not isinstance(currency, str) or not re_currency(currency):
                    raise ValueError('Amount thresholds require uppercase currency codes')
                value = Decimal(str(limit))
                if not value.is_finite() or value <= 0:
                    raise ValueError('Amount thresholds must be finite and positive')
        except (InvalidOperation, TypeError) as exc:
            raise ValueError('Invalid monetary threshold') from exc
        for name in ('vendor_similarity', 'invoice_similarity', 'low_ocr_confidence'):
            if not 0 <= getattr(self, name) <= 100:
                raise ValueError(f'{name} must be between 0 and 100')
        if not 0 < self.contamination <= 0.5:
            raise ValueError('contamination must be in (0, 0.5]')
        if not 0 < self.medium_priority < self.high_priority <= 100:
            raise ValueError('Priority thresholds must satisfy 0 < medium < high <= 100')
        for name, minimum in [('min_training_rows', 20), ('vendor_min_history', 3), ('duplicate_date_days', 0), ('max_invoice_age_days', 0)]:
            if not isinstance(getattr(self, name), int) or getattr(self, name) < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if not 1 < self.vendor_amount_multiplier < 1000:
            raise ValueError('vendor_amount_multiplier must be between 1 and 1000')
        if not 0 < self.vendor_zscore_threshold < 1000:
            raise ValueError('vendor_zscore_threshold must be between 0 and 1000')

    @classmethod
    def load(cls, path: Path | None = None):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8'))) if path else cls()

    def to_dict(self):
        return asdict(self)


def re_currency(value):
    return len(value) == 3 and value.isascii() and value.isalpha() and value == value.upper()
