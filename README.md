<<<<<<< HEAD
# Automated Invoice Processing and Anomaly Detection

## Overview

A Streamlit review workspace that extracts invoice fields from PDF/images using
Tesseract and PDFium, validates fields and Decimal totals, identifies duplicate
candidates and unusual amounts, and optionally uses Isolation Forest. It provides
explainable risk scores, source-preserving corrections, charts, and PDF/CSV reports.
Anomaly flags are review signals, not proof of fraud. Synthetic model performance
is not validated real-world fraud detection accuracy.

**Entry point:** `app.py` in the repository root. Use Python **3.12** for the
prepared deployment. Source paths are resolved relative to the application;
Windows executable/font paths are optional platform fallbacks, not requirements.

## Install and run locally (Windows / VS Code)

```powershell
Set-Location 'C:\path\to\invoice-anomaly-detection'
# Create only when missing; select .venv\Scripts\python.exe in VS Code:
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip check
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENCV_FOR_THREADS_NUM='1'
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

Install Tesseract separately with English language data. Verify `tesseract --version`
and `tesseract --list-langs`. On Windows, discovery also checks common installation
folders; set `TESSERACT_CMD` only when needed. Linux uses the `tesseract` command on
PATH; do not copy a Windows `TESSERACT_CMD` override to the cloud.

`requirements.txt` contains runtime libraries; `requirements-dev.txt` adds pytest
and pypdf for tests. OpenCV uses the headless package: avoid installing the GUI
`opencv-python` package alongside it. PDF rendering uses `pypdfium2`, so Poppler is
not required. `packages.txt` supplies Tesseract, English language data and DejaVu
fonts on Community Cloud. No prebuilt model or existing output is required.

## Features and demonstration

Navigate Dashboard, Upload & Process Invoices, Invoice Records, Anomaly Detection,
Reports, and Settings. On a fresh checkout, **Synthetic demonstration** opens with
17 labeled demonstration cases and a model trained on separate synthetic records.
Use the explanation panel to inspect risk contributions, duplicate details, OCR
warnings and review actions; raw JSON remains under Technical details.

Upload PDF/PNG/JPG/JPEG fixtures from `samples/phase3/`, then open Invoice Records
to review extracted fields. Corrections retain original values and warnings.
Download analysis CSV and a scoped PDF from Reports. The ACME PNG and PDF are two
representations of one synthetic invoice and should be duplicate candidates.
The printed `03/04/2026` date is intentionally ambiguous and must not be guessed.

For a reproducible synthetic ML verification (choose unused destinations):

```powershell
.\.venv\Scripts\python.exe phase21.py --output-dir output/cloud_prep_demo_new --model models/cloud_prep_demo_new.joblib --as-of 2026-10-29
.\.venv\Scripts\python.exe -m pytest tests -q -ra -p no:cacheprovider
```

Saved models and integrity sidecars are ignored by Git. The built-in demonstration
trains at startup when selected; rules-only CSV analysis works without saved
models. Only project-created trusted local artifacts can be selected; user model
uploads are disabled. Pickle/joblib artifacts must never be obtained from strangers.
Retrain models when scikit-learn versions change.

## Storage and public-demo privacy

Local mode preserves the existing `invoices/uploads/` and `output/runs/` workflow.
For **every hosted public demonstration**, set this top-level value in Community
Cloud Advanced settings > Secrets (do not commit a secrets file):

```toml
INVOICE_STORAGE_MODE = "session"
```

Session mode gives each Streamlit browser session a unique OS temporary directory
for original copies, OCR output and corrections, and disables automatic loading of
repository-local extraction CSVs. Model selection still uses trusted repository
artifacts. Downloads are generated in memory. No uploaded invoices are published.
Temporary files may remain on the running host until it is recycled: session
isolation is not authentication, encryption, guaranteed deletion or durable storage.
Do not upload confidential invoices to a public demo. Download required results
before the session/restart/redeploy. No automatic deletion of local user data is
performed. The app does not provide a production multi-user records database.

Local simulation of cloud storage:

```powershell
$env:INVOICE_STORAGE_MODE='session'
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
# For future local persistent runs:
Remove-Item Env:INVOICE_STORAGE_MODE
```

## GitHub preparation (commands for you; not executed automatically)

Review `git status --short`, the safety audit and the exact staged diff before
committing. `.gitignore` excludes environments, secrets, local tooling, caches,
logs, private inputs/outputs, reports, datasets and serialized models. Ignore rules
do not untrack files that were previously committed. This repository already has
Git initialized; skip `git init` here. In a genuinely new directory use `git init -b main`.

After approving the release contents, run:

```powershell
git status --short
git add -- .gitignore README.md requirements.txt requirements-dev.txt packages.txt .streamlit/config.toml *.py phase2_config.example.json samples tests
git diff --cached --stat
git diff --cached --check
git diff --cached
# Only after reviewing and approving that staged content:
git commit -m "Prepare invoice review dashboard for Streamlit Community Cloud"
git branch -M main
# Create an EMPTY GitHub repository first; replace the placeholder URL:
git remote add origin https://github.com/YOUR_USERNAME/invoice-anomaly-detection.git
# If origin already exists, inspect it instead of adding a second origin:
git remote -v
git push -u origin main
```

Keep the remote repository empty initially (no auto-generated README/license) to
avoid unrelated history conflicts. Choose visibility deliberately. No license is
assumed or added by this preparation. Do not force-add ignored invoice/model data.
Screenshots must contain synthetic fixtures only.

## Streamlit Community Cloud deployment (after publication approval)

1. Push only reviewed source, tests, configuration and synthetic samples to GitHub.
2. Sign in to Community Cloud and connect the intended GitHub account/repository.
3. Choose **Create app**, select the repository and branch, and set main file path
   to **`app.py`**. Keep `requirements.txt`, `packages.txt` and `.streamlit/config.toml`
   in the repository root/config locations already provided.
4. In **Advanced settings**, select **Python 3.12** and add the session-storage
   setting from above to Secrets. No API keys are required. Do not copy local logs,
   outputs, Windows paths or credentials into settings.
5. Only after approving deployment, click **Deploy**. Inspect build/runtime logs
   for dependency resolution, Tesseract availability and import errors.
6. Open all six pages, use Synthetic demonstration, process the synthetic samples,
   review corrections, and verify both downloads. Confirm the synthetic-data and
   OCR uncertainty notices remain visible.

These steps follow the official
[dependency instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies),
[deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), and
[secrets guidance](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management).
No cloud app has been deployed as part of preparation.

## Screenshots

No screenshots are committed yet. Before publication, capture the dashboard,
explanation panel, upload review and report pages using **synthetic samples only**.
Redact usernames, local paths and browser/account details. Add reviewed screenshots
under `screenshots/` and link them here; do not reuse confidential invoice images.

## Compatibility, troubleshooting and limitations

- No missing local processing dependency was found. Import failures are logged with
  their traceback; launch using `.venv\Scripts\python.exe -m streamlit`, not an
  unrelated global Streamlit. A healthy HTTP server alone does not prove page imports.
- With missing Tesseract, CSV analysis and synthetic ML remain available; OCR files
  receive a per-file setup error. Verify English language data and remove a stale
  Windows `TESSERACT_CMD` on Linux.
- Upload limits: 10 MB/file, 20 files/50 MB per batch, 25 PDF pages plus decoded-pixel
  limits. OCR is CPU-intensive; use small demo batches on shared hosting.
- Filesystem permissions must allow OS temporary storage in session mode. A missing
  output/model directory is supported; rules-only and built-in demo require no model.
- PDF fonts fall back gracefully; DejaVu fonts are included in Linux system packages.
- OCR can misread line items and punctuation. Core corrections do not certify line
  items, ambiguous dates, or total reconciliation. Missing components are not zero.
- A USD 50000 amount against the USD 10000 threshold contributes 20 risk points,
  yielding LOW on that signal alone. Default MEDIUM starts at 25 and HIGH at 60.
  OCR uncertainty is separate; synthetic ML contributes zero risk points.
- ML has synthetic false positives/negatives, especially for unusual vendor patterns.
  Currency totals are never combined without conversion. No real-world fraud
  accuracy is claimed. Public-demo preparation is not a production security review.
- Linux configuration is prepared and fresh-checkout behavior tested on Windows;
  an actual Community Cloud/Linux build is still required after your approval.

## Historical implementation notes

Current release-preparation verification on Windows/Python 3.12: **228 passed,
0 failed, 0 skipped in 80.02 seconds**. Runtime imports, `pip check`, all six
fresh-checkout pages without local data/models, and local Streamlit startup passed.
No Linux/cloud build or publication was performed.

The phase-specific counts and setup outcomes below are earlier checkpoints, not
the current release verification. See the latest preparation report for new results.


## Phase 3: integrated review workspace and reporting

The professional dashboard extends the existing OCR and analysis modules without
changing their extraction, validation, duplicate, ML inference or scoring policy.
The historical phase sections below remain as implementation/verification history.

### Windows / VS Code setup

Open this project folder in VS Code and use a PowerShell terminal:

```powershell
Set-Location 'C:\path\to\invoice-anomaly-detection'
# Only create the environment if it does not already exist:
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENCV_FOR_THREADS_NUM='1'
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

Open http://localhost:8502. Environment activation is optional. Choose the
`.venv\Scripts\python.exe` interpreter in VS Code. Python 3.11/3.12 are supported.
New presentation dependencies are Plotly (charts), ReportLab (PDF generation),
and pypdf (PDF content verification). The dark theme, accessible text colors,
responsive metric/table layout and 10 MB upload ceiling are configured in
`.streamlit/config.toml`. Deployment preparation is documented above; no external service is required at runtime.

Install Windows Tesseract separately with English language data. The existing
discovery checks TESSERACT_CMD, PATH and common Program Files installation paths.
For example, when installed at the standard location:

```powershell
$env:TESSERACT_CMD='C:\Program Files\Tesseract-OCR\tesseract.exe'
& $env:TESSERACT_CMD --version
& $env:TESSERACT_CMD --list-langs
```

### Navigation and end-to-end workflow

- **Dashboard**: eight KPI cards, original analysis controls, searchable/sortable
  records, per-currency charts and the existing synthetic demonstration.
- **Upload & Process Invoices**: select multiple PDFs/PNG/JPG/JPEG files, extract
  with the existing Phase 1 pipeline, inspect per-file errors and OCR confidence,
  and review core fields before choosing Dashboard/Invoice Records for analysis.
- **Invoice Records**: vendor/date/currency/priority/anomaly filtering, record
  details, validation findings, duplicate explanations, original values and a
  correction form. New review versions never replace the extraction or invoices.
- **Anomaly Detection**: separate rule/ML predictions, 0–100 heuristic scores,
  priorities, explanations, and amount/risk/priority/duplicate/rule/vendor charts.
- **Reports**: use current filters, optionally select specific records, download
  CSV, prepare a paginated PDF, then download it. Empty scopes have clear states.
- **Settings**: inspect active validated settings/model; sidebar controls adjust
  currency-specific amount limits, vendor similarity and analysis date. Thresholds
  persist for this browser session, not as permanent configuration changes.

The original two tabs and existing Phase 2 controls remain available for
compatibility. The Phase 1 OCR tab retains CLI instructions. Choose **Processed
uploads** as the data source after extraction; newly processed runs are selected
automatically on the next analysis page. Uploaded extracted CSVs and the original
17-case synthetic demonstration remain supported.

Uploads are limited to 20 files / 50 MB per batch, 10 MB per file, 25 pages per PDF,
25 million image pixels and 30 million rendered pixels per PDF page. File bytes
are decoded/verified against the extension before processing. Filenames are
sanitized, paths cannot escape the workspace, and every run uses new directories:
`invoices/uploads/run_<id>/` for preserved source copies and
`output/runs/run_<id>/` for extracted CSV, raw text, evidence and processed images.
Identical uploads reuse one extraction while preserving all uploaded copies.
The analysis batch limit remains 2,000 records. Failed files do not delete or
overwrite successful outputs. Processing errors display safe status messages,
not stack traces; full extraction warnings remain available for local inspection.

Review changes are explicit user edits to invoice number/vendor/ISO date/currency/
canonical decimal total. The original core fields and OCR metadata are retained in
`original_extracted_values`, with `correction_audit` and `user_review_status`.
Each save creates `output/runs/review_<id>/corrected.csv`. Review confirmation is
per record. Core review does not certify line items or silently clear OCR warnings,
so a corrected invoice can still require OCR review and remain ML-ineligible.
No permanent deletion or replacement operation is exposed.

### Model integrity, report privacy and interpretation

The dashboard offers only local artifacts with matching `.integrity.json` records
created by the project model saver. Previously generated/verified project models
were enrolled without changing their bytes. Altered, unregistered or out-of-root
model files are not deserialized by the dashboard. This is a local integrity and
provenance boundary, not a cryptographic signature or defense against a person
who controls both files and application code. Models cannot be uploaded. New
models should be created with `phase2.py train` or `phase21.py` as documented below.
Optional charts/PDF dependencies fail with installation guidance; a missing core
dependency shows a setup message rather than an application stack trace.

Reports include dataset identification, India-local generation time, validation
findings, duplicates, risk contributions/priorities, ML status, invoice-level
explanations and OCR/ML limitations. Default PDF/CSV exports use an allowlist,
remove sensitive local paths even inside explanations/JSON, and exclude raw OCR
columns. Spreadsheet formula prefixes are escaped in CSV text fields. Private
paths/raw columns are included only when explicitly requested in Reports; the
application never follows arbitrary CSV raw-text paths. PDF rendering escapes
user text, wraps long explanations, embeds an available Unicode font, and adds
headers/footers/page numbers. Different currencies are never summed together;
the chart currency selector is required for amount/vendor charts.

Valid invoice KPIs mean no blocking field/arithmetic finding, not verified OCR,
complete reconciliation, approved payment or absence of fraud. LOW priority
does not mean approved. All anomaly flags are review signals, not proof of fraud.
Synthetic CSVs/models are labeled; vendor_v2 remains experimental and performed
worse than numeric_v1 in the Phase 2.1 synthetic evaluation. No real-world fraud
accuracy claim is made. The review interface is a local demonstration tool, not
a multiuser authenticated accounting/approval system.

### Demonstration and troubleshooting

1. Select **Synthetic demonstration** to explore known rule/duplicate/risk cases.
2. Upload the four PDF/PNG documents from `samples/phase3/` (these are synthetic
   invoice fixtures), leaving transcript/ground-truth files out of the upload.
3. Inspect OCR output. Visually verify corrections against source documents; the
   ambiguous printed date `03/04/2026` must stay unresolved without issuer context.
4. Save a new review version, choose Dashboard, inspect explanations and duplicate
   candidates, then select Reports and export the filtered CSV/PDF.
5. For the larger ML evaluation, upload
   `output/phase21_final/synthetic_evaluation.csv` and select a registered numeric
   baseline model. Never mix it with real business invoices as training history.

Missing Tesseract: check the executable/languages above; restart after setting
environment variables. PDF rendering does not need Poppler for invoice extraction
because the existing pipeline uses pypdfium2. OCR errors remain possible even with
a correctly installed executable. Port 8502 already in use: use the running app
or stop that process before restarting. Missing Plotly/ReportLab: reinstall
requirements in the project environment. Model absent from the dropdown: create
a new artifact with the project CLI; do not bypass its integrity check. Memory
errors: apply the thread limits above, reduce batch size/page dimensions and run
OCR/testing sequentially. For tests use a fresh temporary-directory name:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -ra -p no:cacheprovider --basetemp .setup-tmp/phase3_test_run
.\.venv\Scripts\python.exe -m pip check
```

Phase 3 verification on Windows: **219 passed in 64.50 seconds**, no failures or
skips, including all 185 existing tests unchanged. Thirty-four new cases cover
uploads, isolated runs, correction provenance, filters/currency boundaries, safe
errors, model-integrity rejection, PDF/CSV privacy/content/pagination and dashboard
navigation/review/downloads. The app was restarted on localhost:8502 and its health
endpoint returned `ok`. An actual four-document/five-page upload-to-report run
produced four analysis rows, a four-page visually inspected PDF, and a private-path-
free public CSV in `output/phase3_final/`. The ambiguous sample's source-supported
number/amount were corrected in a new review version; its date remained unresolved.
All four retained OCR review requirements and were correctly excluded from ML
scoring. The two ACME formats were provisional duplicate candidates, not confirmed
double payments. Prior invoices, outputs, original tests and saved model bytes
retained their hashes. See that directory's verification report for detailed results.

## Phase 2.1: reproducible ML and risk verification

`phase21.py` generates 600 synthetic training invoices and 280 independently
generated evaluation invoices. Training seed is 42; evaluation seed is 1051.
Each currency has 300 training records. Evaluation includes 140 normal invoices,
40 high amounts, 20 unusual item counts, 20 previously unseen vendors, 20 duplicate
records (10 pairs), and 10 each with invalid dates, missing values, OCR uncertainty
or inconsistent totals. These are synthetic expected behaviors, never fraud labels.
Expected labels/scenarios are saved separately in `expected_labels.csv`, keyed by
document ID; they are not passed into training, feature engineering or inference.
Training/evaluation document and invoice IDs are disjoint. Duplicates occur only
inside the evaluation set by design. Both random generators are independent.

The existing numeric-only `numeric_v1` profile remains the default. An optional
`vendor_v2` profile adds normalized vendor identity (one-hot encoding with unknown
categories ignored) and per-currency vendor frequency fitted on training data
only. Numeric features use median imputation and StandardScaler in a fitted
ColumnTransformer/Pipeline. The fitted frequency mapping is saved in the same
versioned bundle. Unseen vendors use frequency zero and an explicit explanation;
they are not automatically anomalies. Essential missing financial/date/vendor
fields and unresolved OCR remain excluded from ML; imputation never invents
missing invoice totals. Internal numeric-feature NaNs can be imputed safely.

Both profiles use Isolation Forest with 200 trees, contamination 0.05, seed 42,
one job, and max_samples=auto (256 per currency here). Negative decision_function
scores are outliers relative to that training baseline. Model save/load prediction
parity is checked. Existing schema-1 numeric models remain loadable without
retraining; vendor-aware models use schema 2. Existing artifacts are never replaced,
and failed serialization removes only the newly created incomplete file.

On this fixed synthetic evaluation, 240 records were scored and 40 invalid or
unverified records correctly skipped. The skipped records are reported separately,
not counted as normal predictions or silently removed from coverage reporting.

| Profile | Precision | Recall | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| numeric_v1 (retained default) | 80.70% | 57.50% | 149 | 11 | 34 | 46 |
| vendor_v2 (experimental) | 61.22% | 37.50% | 141 | 19 | 50 | 30 |

Confusion matrices use expected normal/outlier rows and predicted normal/outlier
columns. Vendor-aware performance was worse on this fixture, so it was **not**
promoted over the numeric default. No hyperparameters were tuned against these
evaluation labels. Numeric v1 detected 39/40 high amounts, 5/20 unusual item counts
and 2/20 unseen vendors; vendor v2 detected 27/40, 3/20 and 0/20 respectively.
The latter's 19 false positives comprise 15 normal and four duplicate records;
its 50 false negatives comprise 13 high amounts, 17 unusual item counts and all
20 unseen vendors. Categorical novelty is handled without crashing but is not
reliably detected by this model. Extra categorical dimensions can dilute numeric
signals, and Isolation Forest does not extrapolate beyond training split ranges.
These are limitations, not a basis for labeling unseen vendors fraudulent.

Rule-based duplicate evaluation is separate: 20/20 duplicate records detected,
zero false positives across 280 records (precision/recall 100% on this fixture
only). All 40 expected validation/OCR findings were detected. Exact false-positive
and false-negative IDs, predictions and scenario counts are exported in
`metrics.json` and `evaluation_comparison.csv`. These metrics do not establish
real-world accuracy or justify automatic payment/fraud decisions.

The risk policy is unchanged: default boundaries are 25 for MEDIUM and 60 for HIGH.
USD 50,000 against USD 10,000 contributes 20/LOW alone and still requires analyst
review; a suitable vendor-history flag raises the amount category to its 30-point
cap/MEDIUM. Synthetic ML contributes zero operational points. Exact score-boundary
and signal-combination tests cover the existing policy. OCR review remains separate.

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENCV_FOR_THREADS_NUM='1'
# Choose unused output/model paths for each run:
.\.venv\Scripts\python.exe phase21.py --output-dir output/phase21_run --model models/phase21_vendor_run.joblib --as-of 2026-10-29
# The command also saves models/phase21_vendor_run_numeric_baseline.joblib.
.\.venv\Scripts\python.exe phase2.py analyze output/phase21_run/synthetic_evaluation.csv --output output/phase21_run/reanalysis.csv --model models/phase21_vendor_run_numeric_baseline.joblib --as-of 2026-10-29
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
.\.venv\Scripts\python.exe -m pytest tests -q -ra -p no:cacheprovider --basetemp .setup-tmp/phase21_test_run
```

The dashboard can select trusted saved local models in both data-source modes.
The main table now includes ML outlier predictions and decision scores alongside
heuristic risk/priority. Uploaded synthetic CSVs are explicitly labeled as such.
For the new evaluation, upload its `synthetic_evaluation.csv` and select the saved
numeric baseline or experimental vendor model. CSV export retains ML results,
explanations, duplicate findings, risk contributions and OCR review status.
The original Phase 1 OCR tab, CSV upload and 17-record Phase 2 demo remain available.
No new dependencies or Phase 3 work were introduced.

## Phase 2: validation, duplicate review and ML demonstration

This phase implements the validation/anomaly objective requested after the OCR
accuracy work. The older sections named Phase 2/3 below describe the historical
OCR development sequence, not the scope of this new phase. There is no separate
project specification. The original OCR/parser/CSV CLI is unchanged.

`phase2.py` reads its extracted CSV with string-preserving types, appends analysis
columns, and never repairs or overwrites original values. It preserves `warnings`,
`field_review`, raw-text paths and source identifiers. New CSV/model destinations
must not already exist. Use a new destination for each run.

```powershell
Set-Location 'C:\path\to\invoice-anomaly-detection'
# Limit numerical-library threads on memory-constrained Windows machines:
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENCV_FOR_THREADS_NUM = '1'

# Separate, explicitly synthetic ML demonstration (600 training / 17 held-out rows):
.\.venv\Scripts\python.exe phase2.py demo --output-dir output/phase2_demo_run --as-of 2026-10-29

# Persist the synthetic pipeline in the existing models directory:
.\.venv\Scripts\python.exe phase2.py train output/phase2_demo_run/synthetic_training.csv --model models/phase2_demo_run.joblib --kind synthetic_demo --as-of 2026-10-29

# Analyze an existing Phase 1 export; --model is optional (rules work alone):
.\.venv\Scripts\python.exe phase2.py analyze output/ocr_accuracy_final/invoices.csv --output output/phase2_run/analysis.csv --model models/phase2_demo_run.joblib --config phase2_config.example.json --as-of 2026-10-29

.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
.\.venv\Scripts\python.exe -m pytest tests -q -ra -p no:cacheprovider --basetemp .setup-tmp/phase2_test_run
.\.venv\Scripts\python.exe -m pip check
```

`--as-of` makes date findings reproducible. Omit it for the current India-local
date. All analysis thresholds are in `phase2_config.example.json`; supply a
custom JSON with `--config`. The dashboard exposes the analysis date, selected
currency's amount threshold and vendor similarity. No extra dependencies were
needed: Phase 2 uses the existing pandas, NumPy, RapidFuzz, scikit-learn, joblib
and Streamlit requirements. Persisted models record the installed scikit-learn
version and must be retrained when that version changes. Only load trusted local
joblib artifacts; the dashboard does not accept model uploads.

### Validation and arithmetic

Validation flags missing/invalid invoice numbers, vendor names, canonical ISO
dates, future dates and nonpositive/nonfinite/malformed amounts. Phase 1 exports
canonical Decimal strings; grouped/locale/OCR-corrupted monetary values are not
guessed during analysis. Arithmetic uses Decimal with a configurable absolute
tolerance. Credit notes are review cases, not automatically invalid transactions.

Quantity x unit price is compared with each line total only when numeric values
and currencies are complete. Grand totals are reconciled only when explicit
`subtotal_amount`, `tax_amount`, `shipping_amount` and `discount_amount` are
provided in the invoice currency. Complete line totals can be checked against an
explicit subtotal. Missing taxes/fees/discounts are never assumed zero, and a
line-item sum is not substituted for a tax-inclusive grand total. The existing
Phase 1 schema does not export these four components: those invoices generally
have PARTIAL/NOT_AVAILABLE arithmetic, rather than fabricated mismatches.

### Duplicate candidates and amount rules

EXACT_DOCUMENT means the nonempty document identifier matches. EXACT_FIELDS
means normalized vendor, case-insensitive invoice number, date, currency and
Decimal amount match. IDENTITY_CONFLICT means vendor/number match while other
fields differ or are missing. POTENTIAL uses vendor/invoice similarity plus
same-currency amounts within tolerance and dates within the configured window.
Missing identity keys alone never create matches. Different currencies are not
numerically compared. Every pair includes the other `analysis_id`, document ID,
reason and a provisional flag when OCR needs review. No invoices are removed:
even an exact-field match may be alternate formats of one legitimate invoice.
Pairwise comparison is limited to 2,000 rows per batch.

Amount rules flag totals over configured per-currency limits, and totals above a
multiple of an earlier vendor median when enough distinct suitable same-currency
records exist in the input. The candidate itself, later dates, duplicate source
IDs and unverified OCR peers cannot inflate that baseline. These are review
heuristics, not proof of fraud or a statistically calibrated vendor risk model.
Analyze synthetic and extracted datasets separately.

### Isolation Forest and training boundaries

Training is explicit, never automatic on uploaded invoices. Real training uses
`--kind verified_real` and requires `human_verified=true`, valid complete fields,
complete item numeric/currency values, and no unresolved OCR warnings/review
flags or financial validation errors. A high OCR confidence alone never verifies
a record. Duplicate source IDs/semantic records are excluded. Minimum training
size defaults to 100 suitable distinct records **per currency**; four samples
cannot train a useful model and are rejected. A separate model is fit per
currency. Features are log(1 + amount), item count, day of month and weekday,
preprocessed with StandardScaler inside a saved scikit-learn Pipeline. Vendor
IDs and invoice numbers are not treated as continuous numbers.

The deterministic synthetic generator creates separate training and held-out
data and labels every row `data_origin=synthetic_demo`. Synthetic and verified
real training cannot be mixed. The demo includes normal, duplicate, high amount,
future date, missing-field, inconsistent-total and uncertain-OCR scenarios.
Synthetic models may be shown against extracted inputs only as explicitly
labeled demonstrations. They add **zero operational risk points**. Invalid or
unverified records, and currencies without a trained baseline, are not scored.
Negative Isolation Forest decision scores mean unusual patterns against the
training baseline; they are not fraud probabilities or causal attributions.
No fraud-detection precision/recall or real-world accuracy claim is made.

### Risk, review status and output

The heuristic risk score is capped at 100. Validation findings contribute
10–25 points each, capped at 40 per invoice; the strongest duplicate candidate
contributes 20 (potential), 25 (identity conflict) or 35 (exact); amount rules
contribute 20 each, capped at 30; a verified-real ML outlier adds 15. The category
contributions and codes are exported. Defaults: LOW <25, MEDIUM 25–59, HIGH >=60.
OCR uncertainty adds no financial-risk points and remains a separate REQUIRED
review status. Missing OCR provenance is NOT_ASSESSED, not verified. LOW never
means approved; there is no automatic fraud or payment approval label.

For example, a valid USD 50,000 invoice analyzed alone with a USD 10,000 limit
triggers `high_amount` (+20): score 20/LOW, but `NEEDS_ANALYST_REVIEW`. LOW is a
relative queue priority, not absence of a flag. The amount rule uses strict `>`;
an amount equal to the limit does not trigger it. Magnitude alone does not
automatically force HIGH. In the full synthetic demo, that same invoice also
triggers `vendor_high_amount` (+20) against six earlier suitable vendor records
with median USD 462.50. The amount-rule category cap yields 30/MEDIUM. Synthetic
ML adds zero points even if it flags the invoice. Filtering dashboard rows after
analysis does not recompute their scores; analyzing a different input batch can
change history/duplicate evidence. Priority cutoffs can be configured explicitly.

Appended fields include `analysis_id`, `validation_findings`, `arithmetic_status`,
`duplicate_flag`, `duplicate_matches`, `rule_anomaly`, `anomaly_explanations`,
`ml_status`, `ml_anomaly`, `ml_decision_score`, `ml_explanation`, `ml_training_kind`,
`risk_score`, `risk_contributions`, `review_priority`, `ocr_review_status`,
`review_status`, `ocr_warnings`, `analysis_as_of`, and `analysis_mode`.
Nested findings/matches/contributions are JSON. Unscored ML results are null,
never false negatives. Row IDs are positional within an analysis batch; original
document IDs are retained. Start from the original extracted CSV when rerunning
analysis; already analyzed input is rejected to protect its columns.

The Streamlit dashboard accepts extracted CSV uploads, defaults to the existing
sample export when available, and provides a separate synthetic-demo mode.
It shows record counts, rule flags, duplicate candidates, priorities, OCR review,
ML coverage, filtering, detailed explanations, original values and a downloadable
filtered analysis CSV. The Phase 1 tab retains OCR usage instructions. Review
statuses are analysis recommendations, not a persisted reviewer-workflow system.

Verified on Windows: **161 passed in 39.66 seconds**, no failures/skips on the
final full run. The first concurrent verification run had two memory-allocation
failures; sequential execution with the thread limits above resolved them
without changing the Phase 1 tests. The actual CLI reprocessed four source
documents/five pages and analyzed all four rows. ACME PNG/PDF were a provisional
exact-field candidate pair (35/MEDIUM each); the ambiguous sample had three
missing fields (30/MEDIUM); the European sample was 0/LOW but still required OCR
review. All four were excluded from ML scoring. The synthetic held-out set
includes a 70/HIGH combined-rule case and a high-amount ML outlier, explicitly
marked as demo data. Streamlit startup/health and interactive dashboard tests
passed; `pip check` reported no broken requirements. Final artifacts are in
`output/phase2_final/`, synthetic data in `output/phase2_demo_verified/`, and the
saved demo model is `models/phase2_synthetic_demo.joblib`.

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
Set-Location 'C:\path\to\invoice-anomaly-detection'
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
| `validator.py` | Conservative field, OCR uncertainty and Decimal arithmetic checks. |
| `anomaly_detector.py` | Amount rules, currency-specific Isolation Forest pipelines and persistence. |
| `duplicate_detector.py` | Exact-field/source, identity-conflict and fuzzy duplicate candidates. |
| `phase2.py` | Analysis orchestration, explainable risk, CSV CLI and synthetic demonstration. |
| `phase2_config.py` | Validated configurable thresholds and India-local reference date. |
| `app.py` | Streamlit review dashboard with retained Phase 1 instructions. |
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
Preprocessing, OCR, parsing, structured CSV exports, validation, duplicate
candidates, anomaly rules, synthetic-demo ML and the review dashboard are
implemented. Historical verification sections below describe earlier states.

## Dependency verification

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import cv2, pytesseract, PIL, pandas, numpy, sklearn, rapidfuzz, streamlit, joblib, dateutil, pypdfium2, plotly, reportlab; print('All dependency imports passed')"
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

At that early checkpoint, actual PNG/PDF execution exported four setup-error rows
to `output/ocr_setup_check.csv`: Tesseract was absent at that checkpoint (it is installed in the current local environment). The four real-OCR
integration tests remain skipped; real recognition accuracy is not verified.
Install/configure the executable as described above to complete that check.
Historical note: validation and ML were unimplemented at that early checkpoint; both are implemented now.
=======
# Automated-Invoice-Processing-and-Anomaly-Detection
>>>>>>> 4d21f48b2559af158266446210a63685059ece2d
