"""Generated invoice fixtures; no private invoice data is required."""

from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def invoice_pixels():
    image = np.full((900, 1700, 3), 255, dtype=np.uint8)
    for index, text in enumerate([
        "INVOICE INV-1001", "ACME SUPPLIES", "Date: 2026-10-27",
        "Paper 2 x 50.00 = 100.00", "Total: 100.00",
    ]):
        cv2.putText(image, text, (100, 160 + index * 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3, cv2.LINE_AA)
    return image


@pytest.fixture
def invoice_path(tmp_path, invoice_pixels):
    target = tmp_path / "invoice.png"
    target.write_bytes(cv2.imencode(".png", invoice_pixels)[1].tobytes())
    return target
