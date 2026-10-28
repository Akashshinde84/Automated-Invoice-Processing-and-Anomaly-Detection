"""Document-level processing and CSV serialization, without ML or validation."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

from ocr_engine import OCRError, process_invoice
from parser import Invoice, parse_invoice
from preprocessing import SUPPORTED_EXTENSIONS

COLUMNS = ["document_id", "source_file", "source_files", "invoice_number", "invoice_date",
           "vendor_name", "total_amount", "currency", "line_items", "warnings",
           "page_count", "ocr_mean_confidence", "raw_text_file", "status",
           "field_review", "recognition_evidence_file"]


def discover_sources(source: Path, *, text_input: bool = False) -> list[Path]:
    """List supported files in a directory (nonrecursive), or one explicit file."""
    source = Path(source)
    extensions = {".txt"} if text_input else SUPPORTED_EXTENSIONS
    if not source.exists():
        raise FileNotFoundError(f"Source does not exist: {source}")
    if source.is_file():
        if source.suffix.lower() not in extensions:
            raise ValueError(f"Unsupported source format: {source.suffix}")
        return [source]
    return sorted((path for path in source.iterdir() if path.is_file() and path.suffix.lower() in extensions),
                  key=lambda path: path.name.casefold())


def invoice_record(invoice: Invoice) -> dict:
    """Use decimal strings in JSON to retain monetary precision and JSON nulls."""
    items = []
    for item in invoice.line_items:
        value = asdict(item)
        value["line_total"] = value.pop("amount")
        for key in ("quantity", "unit_price", "line_total"):
            value[key] = str(value[key]) if value[key] is not None else None
        items.append(value)
    return {
        "invoice_number": invoice.invoice_number,
        "invoice_date": invoice.invoice_date.isoformat() if invoice.invoice_date else None,
        "vendor_name": invoice.vendor_name,
        "total_amount": str(invoice.total_amount) if invoice.total_amount is not None else None,
        "currency": invoice.currency,
        "line_items": json.dumps(items if items else None, ensure_ascii=False),
        "warnings": json.dumps(invoice.warnings, ensure_ascii=False),
    }


def run_pipeline(sources: list[Path], *, csv_path: Path, processed_dir: Path,
                 raw_dir: Path, date_order: str = "auto",
                 decimal_separator: str | None = None, text_input: bool = False,
                 **ocr_options) -> pd.DataFrame:
    """One row per byte-unique document, joining all PDF pages before parsing.

    Byte-identical inputs share a row and retain all source paths. Different
    documents with the same invoice number stay separate. Per-document failures
    produce error rows, never fabricated invoice data. Text mode is an explicit
    parser demo/replay mode and does not claim to perform OCR.
    """
    if date_order not in {"auto", "dmy", "mdy"}:
        raise ValueError("date_order must be auto, dmy, or mdy")
    if decimal_separator not in {None, ".", ","}:
        raise ValueError("decimal_separator must be '.' or ','")
    csv_path, raw_dir = Path(csv_path), Path(raw_dir)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows, seen = [], {}
    for source in sources:
        source = Path(source).resolve()
        try:
            results = []
            content = source.read_bytes()
            identity = hashlib.sha256(content).hexdigest()
        except OSError as exc:
            row = invoice_record(Invoice(warnings=[f"Cannot read source: {exc}"]))
            row.update(source_file=str(source), source_files=json.dumps([str(source)]), status="error")
            rows.append(row)
            continue
        if identity in seen:
            row = seen[identity]
            aliases = json.loads(row["source_files"])
            if str(source) not in aliases:
                aliases.append(str(source))
            row["source_files"] = json.dumps(aliases, ensure_ascii=False)
            continue
        raw_path = raw_dir / f"{identity}.txt"
        row = {"document_id": identity, "source_file": str(source),
               "source_files": json.dumps([str(source)], ensure_ascii=False),
               "page_count": None, "ocr_mean_confidence": None, "raw_text_file": None}
        try:
            if text_input:
                text = content.decode("utf-8-sig")
            else:
                results = process_invoice(source, processed_dir, **ocr_options)
                row["page_count"] = len(results)
                scores = [score for page in results for score in page.word_confidences]
                row["ocr_mean_confidence"] = sum(scores) / len(scores) if scores else None
                text = "\n\n".join(page.text for page in results)
                for number, page in enumerate(results, 1):
                    print(f"\n--- {source.name}: Page {number} | confidence: "
                          f"{page.mean_confidence:.1f}/100 ---" if page.mean_confidence is not None
                          else f"\n--- {source.name}: Page {number} | confidence: unavailable ---")
                    print(page.text or "[No text detected]")
                    print(f"Processed image: {page.image_path}")
            original_text = ("\n\n".join(page.raw_text if page.raw_text is not None else page.text
                                        for page in results) if results else text)
            raw_path.write_text(original_text, encoding="utf-8")
            row["raw_text_file"] = str(raw_path.resolve())
            invoice = parse_invoice(text, date_order=date_order, decimal_separator=decimal_separator)
            review = {}
            if results:
                evidence_path = raw_dir / f"{identity}.evidence.json"
                evidence_path.write_text(json.dumps([
                    {"method": page.method, "original_text": page.raw_text if page.raw_text is not None else page.text,
                     "selected_text": page.text, "mean_confidence": page.mean_confidence,
                     "word_confidences": page.word_confidences, "lines": page.line_evidence,
                     "review_reasons": page.review_reasons} for page in results
                ], ensure_ascii=False, indent=2), encoding="utf-8")
                row['recognition_evidence_file'] = str(evidence_path.resolve())
                if any(page.method == 'ocr' and any('bounds' in line for line in page.line_evidence)
                       for page in results):
                    for field in ('invoice_number', 'invoice_date', 'total_amount'):
                        review[field] = 'manual_review_required' if getattr(invoice, field) is not None else 'unresolved'
                    invoice.warnings.append('OCR fields are unverified: review invoice_number, invoice_date and total_amount against the source; confidence is not proof of correctness.')
                for page in results:
                    invoice.warnings.extend(page.review_reasons)
            row['field_review'] = json.dumps(review)
            row.update(invoice_record(invoice))
            row["status"] = "warning" if invoice.warnings else "ok"
        except (OCRError, OSError, ValueError, UnicodeError) as exc:
            row.update(invoice_record(Invoice(warnings=[f"Document processing failed: {exc}"])))
            row["status"] = "error"
            print(f"Error: {source.name}: {exc}", file=sys.stderr)
        rows.append(row)
        seen[identity] = row
    frame = pd.DataFrame(rows, columns=COLUMNS)
    # Empty scalar values use the explicit CSV token null; JSON retains real null.
    temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", na_rep="null")
    temporary.replace(csv_path)
    return frame
