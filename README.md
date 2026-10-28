# Automated Invoice Processing and Anomaly Detection

## OCR accuracy verification (2026-10-29)

The existing pipeline now uses conservative per-page digital PDF text extraction
via pypdfium2. Image-backed, mixed-content, invisible-text and damaged-text pages
fall back to OCR. This gate is a heuristic, not a guarantee of document correctness.
PDF pages are still rendered for inspection. No new OCR dependencies are required.

The default `--threshold auto` preserves clean grayscale strokes and uses local
thresholding with median denoising for uneven backgrounds. Explicit `otsu` and
`adaptive` options remain available. Deskew and image-size limits are retained.
Experiments compared page modes 3, 6 and 11, and cropped-line modes 7 and 13.
Labelled field lines use mode 7; uncertain invoice dashes use mode 13 with
surrounding whitespace. The original PDF test produced U+2014 (em dash), not
corrupt PDF text. Rereading the line pixels fixes the unchanged exact assertion;
no replacement of guessed characters or amounts is performed.

CSV keeps its original columns and adds `field_review` (JSON review flags) and
`recognition_evidence_file`. Real OCR fields require manual review even when
syntax and confidence look good. `raw_text_file` retains original page OCR;
the evidence JSON contains selected text, alternatives, line crop bounds,
original word confidences, extraction method and disagreement warnings.
Mean confidence describes the original page OCR, not the selected crop text.
Monetary parsing still uses Decimal and rejects malformed/ambiguous values.
Immediately adjacent label/value lines are supported without crossing labels.

Verification commands used (outputs were placed in a new directory):

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -ra -p no:cacheprovider --basetemp .setup-tmp/accuracy_tests_final
.\.venv\Scripts\python.exe main.py samples/phase3 --csv output/ocr_accuracy_final/invoices.csv --output-dir output/ocr_accuracy_final/processed --raw-dir output/ocr_accuracy_final/raw_ocr
```

Result: **91 passed**, including all four original real-OCR tests. Windows
sandbox temporary-directory restrictions required normal host permissions for
pytest. Use an unused `--basetemp` path for each subsequent run.
Four sample documents (five pages) produced four warning rows with no errors.
The ACME PNG/PDF core fields and European core fields match visually checked
sources. The ambiguous sample's number and total remain null after OCR, and its
printed `03/04/2026` date deliberately remains unresolved. Some line-item amounts
remain unreadable; do not infer them from totals. All OCR results require review.
`samples/phase3/ground_truth.json` records the visual checks and source hashes;
it is used only for tests and reporting, never production extraction.
The earlier phase verification sections below record historical results.

Phase 3 adds conservative regex extraction and structured CSV generation to
OpenCV preprocessing and Tesseract OCR for JPG, JPEG, PNG, and multi-page PDF
invoices. Python 3.11 or 3.12 is required; the initial environment
uses Python 3.12. Default output paths are relative to the source directory;
relative CLI input and output arguments are relative to your working directory.

## Windows PowerShell setup

```powershell
Set-Location 'C:\Automated Invoice Processing and Anomaly Detection'
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Activation is optional: these commands work without changing execution policy.
Python 3.11 users can substitute `py -3.11` when creating the environment.

## Files

| File | Purpose |
| --- | --- |
| `main.py` | CLI for batch image/PDF processing, parsing options, and CSV export. |
| `preprocessing.py` | OpenCV loading, grayscale, resizing, deskew, blur, binarization, PDF rendering, and processed PNG saving. |
| `ocr_engine.py` | Tesseract discovery, text extraction, page results, word confidence, and errors. |
| `parser.py` | Unicode normalization, regex extraction, explicit dates/currencies, and missing-field warnings. |
| `pipeline.py` | One-row-per-document DataFrames, raw OCR text, JSON line items, and CSV writing. |
| `validator.py` | Placeholder for invoice arithmetic and date checks. |
| `anomaly_detector.py` | Placeholder for vendor Z-scores and Isolation Forest. |
| `duplicate_detector.py` | Placeholder for fuzzy duplicate detection. |
| `app.py` | Runnable Streamlit landing page. |
| `requirements.txt` | Python dependency ranges. |
| `.gitignore` | Excludes environments, generated files, and invoice data. |
| `README.md` | Setup, scope, and verification instructions. |

## Directories

- `invoices/`: incoming invoice images and PDFs.
- `output/`: structured `invoices.csv` export; ML columns are reserved for later phases.
- `output/processed/`: processed page PNGs.
- `output/raw_ocr/`: UTF-8 raw text per source document.
- `samples/phase3/`: synthetic transcripts, PNG invoices, and a two-page PDF.
- `data/`: future working datasets.
- `models/`: future persisted models.
- `reports/`: future reports and visualizations.
- `tests/`: generated invoice fixtures and preprocessing, OCR, and CLI tests.

Empty directories contain `.gitkeep` files so they can be tracked in Git.
Preprocessing, OCR, parsing, and structured CSV exports are implemented.
Mathematical validation, anomaly/duplicate detection, and dashboard
visualizations remain future phases.
Their existing placeholders and the Streamlit landing page are preserved.

## Dependency verification

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import cv2, pytesseract, PIL, pandas, numpy, sklearn, rapidfuzz, streamlit, matplotlib, seaborn, pytest, joblib, dateutil, pypdfium2; print('All dependency imports passed')"
```

`pytesseract` is a wrapper: actual OCR needs the separate
Tesseract OCR Windows executable and appropriate language data. Python package
installation does not install that executable. Check its availability with:

```powershell
tesseract --version
```

If unavailable, install Windows Tesseract with English language data and add its
installation directory to PATH, or set the explicit override shown below.
PDF rendering will use pypdfium2; no Poppler installation is required.

## Test OCR on an actual invoice

Place an invoice in `invoices/`, then run:

```powershell
# Optional when Tesseract is not on PATH:
$env:TESSERACT_CMD = 'C:\Program Files\Tesseract-OCR\tesseract.exe'
& $env:TESSERACT_CMD --version
.\.venv\Scripts\python.exe main.py '.\invoices\invoice.png'
.\.venv\Scripts\python.exe main.py '.\invoices\invoice.pdf' --dpi 300
```

The executable is discovered from `TESSERACT_CMD`, PATH, or common Program Files
and local-user Windows installation folders, in that order. An invalid explicit
override produces an error instead of silently selecting another executable.
The override must identify the executable, without command-line arguments.

Each page prints extracted text, recognized word count, mean word confidence,
and its processed PNG path in `output/processed/`. Confidence ranges from 0 to
100 and is an unweighted mean of valid word scores, not an accuracy probability.
Blank pages return empty text and unavailable mean confidence. Text preserves
Tesseract's detected line order, not the invoice's original table layout.
Original invoices are unchanged; UUID-based output names avoid overwriting
previous runs. Rendering failure on a later PDF page may leave earlier saved
pages for inspection.

```powershell
# Alternative thresholding and a uniform text-block layout:
.\.venv\Scripts\python.exe main.py '.\invoices\invoice.jpg' --threshold adaptive --psm 6
# Other options:
.\.venv\Scripts\python.exe main.py --help
.\.venv\Scripts\python.exe -m pytest -q -ra
```

Defaults are Gaussian blur, Otsu binarization, automatic page segmentation
(`--psm 3`), English, 300 DPI for PDFs, and a 60-second OCR timeout per page.
Use `--lang` for installed languages, `--timeout` to adjust the time limit,
`--output-dir` for another destination, or `--no-deskew` for pages that rotate
incorrectly. Small images are enlarged toward 1600 pixels wide and large
images are reduced to a maximum dimension of 4000 before deskewing. Deskewing
estimates near-horizontal lines up to 15 degrees; it does not correct 90/180
degree orientation. Correct those orientations manually before processing.
Preprocessing cannot recover detail from an illegible scan. Inspect the saved
PNG and try a clearer source or alternate thresholding if recognition is poor.

API: `preprocess_invoice(path)` returns a list of processed page paths;
`extract_text(processed_path)` returns an `OCRResult`; `process_invoice(path)`
combines both and returns one result per page, in source order. Each result
contains `text`, `image_path`, `word_count`, `mean_confidence`, and
`word_confidences`. These replace the phase-one placeholder return contracts.

## Phase 2 verification

Generated sample invoices contain `INV-1001`, `ACME SUPPLIES`, and `100.00`.
Tests cover image formats, Unicode paths, both threshold methods, PDF page
rendering, resizing, skew correction, corrupt/missing files, output preservation,
Tesseract configuration/errors, text/confidence handling, and CLI output/errors.
OCR orchestration tests mock Tesseract; four separate real OCR tests assert
known text for PNG/JPG/JPEG/PDF and automatically run when Tesseract is available.
They skip with a clear reason when it is missing.

Phase 2 result: **23 passed, 4 skipped** (Tesseract executable missing).
The test run required execution outside the sandbox because Windows denied
pytest access to its temporary directories inside the sandbox. Real recognition
accuracy has not been verified on this machine until Tesseract is installed.

## Initial setup results

The following section records Phase 1 setup history; see Phase 3 below for the
current pipeline behavior and verification results.

- Created all requested directories and module skeletons in an empty workspace.
- Created `.venv` with Python 3.12.6 and installed `requirements.txt`.
- Ran `main.py`, checked syntax for all eight Python modules, imported all
  fourteen requested dependencies and all eight project modules successfully.
- Ran `python -m pip check`: no broken requirements found.
- Verified the Streamlit landing page with `streamlit.testing.v1.AppTest`:
  expected title displayed and no application exceptions.
- Windows sandbox temporary-directory restrictions required running environment
  setup and the Streamlit smoke check outside the sandbox; both then completed.
  `.setup-tmp/` is ignored and may contain temporary files from the failed retry.
- Tesseract was not found on PATH. This remains setup work for the OCR phase.

Commands executed included `py -0p`, `py -3.12 --version`,
`py -3.12 -m venv .venv`, an `ensurepip` recovery attempt,
`python -m pip install -r requirements.txt`, `python main.py`,
`python -m pip check`, and Python syntax/import and Streamlit smoke checks.
Use the `.venv\Scripts\python.exe` executable for project commands.

## Phase 3: batch parsing and CSV generation

Process all supported images and PDFs directly inside `invoices/`:

```powershell
.\.venv\Scripts\python.exe main.py .\invoices
```

The source argument is optional and defaults to `invoices/`. Directory scanning
is nonrecursive and ignores unsupported files. A single image or PDF still
works. Outputs default to `output/invoices.csv`, `output/raw_ocr/<document_id>.txt`,
and `output/processed/` page PNGs. All pages of one PDF are joined before parsing
and become one CSV row. Raw text preserves OCR output before parser normalization.

Documents are identified by SHA-256 of their original bytes. Byte-identical
copies share one row and all original paths are preserved in `source_files`.
Different documents with the same invoice number remain separate. This is
source-document deduplication only; fuzzy invoice matching is not implemented.

### Dates and numeric conventions

The default `--date-order auto` accepts ISO, month-name dates, and unambiguous
numeric dates. `03/04/2026` becomes null with an ambiguity warning. Explicitly
configure the convention when known:

```powershell
.\.venv\Scripts\python.exe main.py .\invoices --date-order dmy
.\.venv\Scripts\python.exe main.py .\invoices --date-order mdy
```

A four-digit year and complete valid calendar date are required. DMY/MDY mode
does not silently fall back to the other interpretation. Dates are exported as
YYYY-MM-DD. Future-date/business validation is reserved for the next phase.

Currency symbols and supported codes are retained; `$` is not automatically
assumed to mean USD. A compatible explicit code elsewhere can disambiguate it.
Mixed currencies produce warnings and are never summed or converted. Unit and
line-total amounts with incompatible explicit currencies are left null.
Monetary values use `Decimal` internally, with Western and Indian grouping,
decimal-comma, space, and apostrophe separators supported. For ambiguous values
such as `1.234`, explicitly choose a decimal convention:

```powershell
.\.venv\Scripts\python.exe main.py .\invoices --decimal-separator ','
```

This treats dots as thousands separators and commas as decimal separators.
Use `--decimal-separator '.'` for dot decimals, including three-decimal unit
prices. Without an override a comma followed by three digits is treated as a
thousands separator; a single dot followed by three digits is left unresolved.

### Extraction rules and output schema

Explicit labels support Invoice No/No./#/Number, Bill No, Invoice Date/Date,
Vendor/Supplier (optionally Name), Grand Total, Total Amount, Invoice Total, and
Total. Grand Total has highest precedence; repeated conflicting values at the
same precedence remain null. Subtotal, tax, amount due, and balance due never
substitute for a grand total. Unlabelled vendor headings are not guessed.

Line items support `Description | Quantity | Unit Price | Line Total` tables,
whitespace rows with that column order, and explicit rows such as
`Paper 2 x 50.00 = 100.00`. Headers may use Item/Qty. Delimited partial rows retain
missing cells as null with warnings. Unsupported or wrapped layouts are warned
about rather than reconstructed. Totals and missing row values are not calculated.
This regex parser is not a general document-layout model; always review warnings
and compare important fields with the source.

CSV columns: `document_id`, `source_file`, `source_files`, `invoice_number`,
`invoice_date`, `vendor_name`, `total_amount`, `currency`, `line_items`, `warnings`,
`page_count`, `ocr_mean_confidence`, `raw_text_file`, `status`.
`line_items` and `warnings` are JSON strings; line items contain `description`,
`quantity`, `unit_price`, `line_total`, and `currency`. Decimal item values are
JSON strings to preserve precision. Missing scalars use the CSV token `null`;
missing item cells use JSON null. No recognized items serialize as JSON null.
Use `json.loads(row['line_items'])` after reading CSV to recover the structure.
Pandas' default CSV reader also interprets scalar `null` as a missing value.

Status is `ok`, `warning`, or `error`. Per-document errors are exported and other
documents continue; the CLI exits with code 1 if any document failed. A failed
OCR document has no raw-text path rather than a fabricated transcript. CSV files
are atomically replaced per run, not appended. An empty input directory returns
an error without replacing an existing CSV. Archive earlier exports if needed.

### Synthetic samples and verification

`samples/phase3/` includes three manually authored transcripts, corresponding
PNG invoices, and a two-page PDF. Regenerate missing images with:

```powershell
.\.venv\Scripts\python.exe samples/generate_invoices.py
# Real preprocessing/OCR/parsing (requires the Tesseract executable):
.\.venv\Scripts\python.exe main.py .\samples\phase3 --csv .\output\sample_ocr.csv
# Explicit transcript replay; no preprocessing or OCR is claimed in this mode:
.\.venv\Scripts\python.exe main.py .\samples\phase3 --text-input
.\.venv\Scripts\python.exe -m pytest -q -ra
```

Phase 3 verification: **69 passed, 4 skipped**, with `pip check` reporting no
broken requirements. Full image preprocessing through CSV orchestration is
tested with only Tesseract's recognition response mocked. Transcript replay
executed successfully and generated three rows in `output/invoices.csv` with
raw transcripts in `output/raw_ocr/`. The ACME row has invoice `INV-1001`, date
`2026-10-27`, total `137.50`, currency `USD`, and two line items. The ambiguous
date sample has a null date and an explicit warning.

Actual PNG/PDF execution was also attempted, exporting four setup-error rows
to `output/ocr_setup_check.csv`: Tesseract is still absent. The four real-OCR
integration tests remain skipped; real recognition accuracy is not verified.
Install/configure the executable as described above to complete that check.
Mathematical validation and ML remain unimplemented.
