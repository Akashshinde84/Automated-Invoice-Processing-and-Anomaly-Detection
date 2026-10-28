"""Future RapidFuzz duplicate invoice detection boundary."""

import pandas as pd


def detect_duplicates(invoices: pd.DataFrame) -> pd.DataFrame:
    """Eventually identify candidate duplicates using fuzzy matching."""
    raise NotImplementedError("Duplicate detection belongs to a future phase.")
