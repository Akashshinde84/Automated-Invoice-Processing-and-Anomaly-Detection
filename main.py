"""Command-line invoice preprocessing, OCR, parsing, and document CSV export."""

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DIRECTORIES = (
    "invoices", "output", "output/processed", "data", "models", "reports", "tests"
)


def ensure_directories(root: Path = PROJECT_ROOT) -> None:
    """Create project storage directories without changing existing files."""
    for directory in DIRECTORIES:
        (root / directory).mkdir(parents=True, exist_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=PROJECT_ROOT / "invoices",
                        help="Invoice image/PDF or directory (default: invoices/)")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output/processed")
    parser.add_argument("--lang", default="eng", help="Installed Tesseract language code")
    parser.add_argument("--psm", type=int, choices=range(3, 14), default=3)
    parser.add_argument("--dpi", type=int, default=300, help="PDF rendering resolution, 72–600")
    parser.add_argument("--timeout", type=float, default=60, help="OCR timeout per page in seconds")
    parser.add_argument("--threshold", choices=("auto", "otsu", "adaptive"), default="auto")
    parser.add_argument("--no-deskew", action="store_true")
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "output/invoices.csv")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "output/raw_ocr")
    parser.add_argument("--date-order", choices=("auto", "dmy", "mdy"), default="auto")
    parser.add_argument("--decimal-separator", choices=(".", ","), default=None)
    parser.add_argument("--text-input", action="store_true",
                        help="Explicit demo/replay mode: parse UTF-8 .txt files without OCR")
    args = parser.parse_args(argv)
    ensure_directories()
    from pipeline import discover_sources, run_pipeline
    try:
        sources = discover_sources(args.source, text_input=args.text_input)
        if not sources:
            parser.exit(1, f"No supported invoice files found in {args.source}\n")
        frame = run_pipeline(
            sources, csv_path=args.csv, processed_dir=args.output_dir,
            raw_dir=args.raw_dir, date_order=args.date_order,
            decimal_separator=args.decimal_separator, text_input=args.text_input,
            lang=args.lang, psm=args.psm,
            timeout=args.timeout, dpi=args.dpi, deskew=not args.no_deskew,
            threshold=args.threshold,
        )
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Exported {len(frame)} unique source document(s) to {args.csv}")
    if (frame["status"] == "error").any():
        parser.exit(1, "Some invoices failed. See CSV warnings for details.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
