"""Generate a four-page technical report and sample CSV from fresh measurements."""
import argparse
from datetime import date
from html import escape
import json
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.graphics.shapes import Drawing, Line, Rect, Circle, String

from analysis_config import AnalysisConfig, local_today
from pipeline import run_pipeline
from invoice_analysis import analyze_invoices
from model_evaluation import verification_data, evaluate
from anomaly_detector import train_model
from reporting import csv_report


def amount_plot(frame, *, box=False):
    """Vector USD-only charts with the same flags as the exported analysis."""
    group = frame[frame.currency == 'USD'].copy()
    group['total_amount'] = pd.to_numeric(group.total_amount, errors='coerce')
    group = group.dropna(subset=['total_amount']).reset_index(drop=True)
    amounts = group.total_amount.astype(float).to_numpy()
    drawing = Drawing(460, 220)
    top = float(max(amounts)) * 1.1
    y = lambda value: 30 + 160 * value / top
    drawing.add(Line(48, 30, 445, 30))
    drawing.add(Line(48, 30, 48, 195))
    for value in np.linspace(0, top, 5):
        drawing.add(String(0, y(value), f'{value:,.0f}', fontSize=8))
    if box:
        q1, med, q3 = np.percentile(amounts, [25, 50, 75])
        iqr = q3-q1
        inside = amounts[(amounts >= q1-1.5*iqr) & (amounts <= q3+1.5*iqr)]
        drawing.add(Line(240, y(min(inside)), 240, y(max(inside))))
        drawing.add(Rect(210, y(q1), 60, max(1, y(q3)-y(q1)), fillColor=colors.lightblue))
        drawing.add(Line(210, y(med), 270, y(med)))
        for value in amounts[(amounts < q1-1.5*iqr) | (amounts > q3+1.5*iqr)]:
            drawing.add(Circle(240, y(value), 2, fillColor=colors.darkred))
        drawing.add(String(150, 8, 'USD amounts; Tukey 1.5 IQR whiskers', fontSize=9))
    else:
        for i, row in group.iterrows():
            color = colors.darkred if row.isanomaly else colors.steelblue
            drawing.add(Circle(55 + i*380/max(1,len(group)-1), y(float(row.total_amount)), 2, fillColor=color, strokeColor=color))
        drawing.add(String(60, 8, 'Invoice index | red: combined anomaly flag; blue: unflagged', fontSize=9))
    return drawing


def build(output_dir, as_of):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    config = AnalysisConfig(as_of=as_of)
    root = Path(__file__).resolve().parent
    samples = root/'samples'/'phase3'
    truth = json.loads((samples/'ground_truth.json').read_text())['documents']
    extracted = run_pipeline([samples/name for name in truth], csv_path=out/'extracted_invoices.csv',
                             processed_dir=out/'processed', raw_dir=out/'raw_ocr')
    counts = dict(correct=0, incorrect=0, unresolved=0, expected_fields=0, ambiguous_dates=0)
    comparisons = []
    for _, row in extracted.iterrows():
        name = Path(row.source_file).name
        for field in ('invoice_number', 'invoice_date', 'total_amount'):
            expected, actual = truth[name][field], row[field]
            if expected is None:
                counts['ambiguous_dates'] += 1
                continue
            counts['expected_fields'] += 1
            state = 'unresolved' if actual is None or str(actual) in ('nan', '') else 'correct' if str(actual) == str(expected) else 'incorrect'
            counts[state] += 1
            comparisons.append(dict(document=name, field=field, expected=expected, actual=actual, result=state))
    analysis = analyze_invoices(extracted, config)
    (out/'invoice_analysis.csv').write_bytes(csv_report(analysis))
    training, evaluation, labels = verification_data(config)
    model = train_model(training, config, training_kind='synthetic_demo')
    evaluated = analyze_invoices(evaluation, config, model)
    metrics, _ = evaluate(evaluated, labels)
    (out/'synthetic_analysis.csv').write_bytes(csv_report(evaluated))
    (out/'measurements.json').write_text(json.dumps(dict(ocr=counts, comparisons=comparisons, ml=metrics,
        as_of=as_of, sample_documents=len(extracted)), indent=2, default=str))
    styles = getSampleStyleSheet()
    styles['BodyText'].leading = 15
    story = []
    def p(text, style='BodyText'):
        story.extend([Paragraph(escape(text), styles[style]), Spacer(1, 10)])
    def page(title):
        if story:
            story.append(PageBreak())
        p(title, 'Title')
    page('Invoice processing: technical report')
    p(f'Reference date: {as_of}. Reproducible sample extraction and independent synthetic model evaluation.')
    p('1. Architecture and extraction', 'Heading1')
    p('Documents pass through OpenCV grayscale conversion, conservative deskew and adaptive or Otsu binarization. Adaptive processing uses median filtering for noise reduction. Clean lettering can retain grayscale to avoid losing strokes. Tesseract extracts scanned text; digital PDFs use their text layer when available.')
    p('Anchored regular expressions identify invoice number, vendor, date and grand total. Supported table rows produce description, quantity, unit price and line total. Decimal arithmetic retains monetary precision. Ambiguous dates and conflicting values remain unresolved rather than being guessed.')
    p('Every byte-unique document becomes one DataFrame row; byte-identical copies retain source aliases. Different documents with matching invoice identities remain available for duplicate review. Explicit subtotal, tax, shipping and discount values are exported when supported labels are recognized.')
    p('2. OCR accuracy measurement', 'Heading1')
    p(f'Fresh run: {len(extracted)} authored sample documents. Invoice number, date and total are compared with the supplied ground truth. Of {counts["expected_fields"]} fields with definite expected values: {counts["correct"]} correct, {counts["incorrect"]} incorrect and {counts["unresolved"]} unresolved. Exact field recovery: {counts["correct"]/max(1,counts["expected_fields"]):.1%}.')
    p(f'{counts["ambiguous_dates"]} expected ambiguous date(s) are excluded from that denominator. This measures three core fields on a small authored set, not word/character accuracy, vendor accuracy, line-item accuracy or performance on unseen business invoices. Full comparisons are saved in measurements.json.')
    page('Validation and anomaly methodology')
    p('3. Logical and statistical validation', 'Heading1')
    p('Validation checks required fields, finite amounts, future dates and invoice ages greater than the configurable 90-day default. Quantity times unit price is compared with each observed line total. Complete line totals are reconciled with explicit subtotal; subtotal plus tax plus shipping minus discount is reconciled with the grand total. Missing components remain unknown, so arithmetic is marked partial rather than silently passing.')
    p('Vendor Z-scores use (current amount - historical mean) / population standard deviation. The reference group contains at least five distinct earlier eligible invoices for the same normalized vendor and currency. Absolute scores above three trigger review. Zero variance or insufficient history produces no Z-score flag; the separate median multiplier rule still detects large increases. Historical observations never include the current invoice.')
    p('Fuzzy duplicate candidates compare vendor/invoice identifiers and Decimal amounts with date and currency constraints. Byte identity, exact matching and conflicting identities are also reported. Candidates remain records; no invoice is automatically deleted or approved.')
    p('4. Isolation Forest choice', 'Heading1')
    p('Isolation Forest is suited to an unsupervised setting without reliable fraud labels. Random feature splits isolate unusual observations in fewer steps. Separate currency models use log amount, item count, day of month and weekday with saved scaling. Defaults: 200 trees, contamination 0.05 and seed 42. Negative decision scores identify model outliers; contamination is not a fraud prevalence estimate.')
    p('The evaluation uses 600 independently generated training records and 280 separate evaluation records; expected labels do not enter model features. Unverified or invalid source records are skipped. Old age is a review signal and does not itself invalidate model inputs. Synthetic results demonstrate behavior and do not establish real-world fraud accuracy.')
    page('Measured results and distributions')
    p('5. Independent synthetic evaluation', 'Heading1')
    p(f'Numeric baseline results: precision {metrics["precision"]:.3f}; recall {metrics["recall"]:.3f}. Scored records: {metrics["scored_rows"]}; skipped: {metrics["skipped_rows"]}. Metrics apply only to this synthetic evaluation and model signal, independently of combined rule flags.')
    p('Figure 1. USD invoice amount box plot', 'Heading2')
    story.append(amount_plot(evaluated, box=True))
    p('Box plot points beyond the whiskers are distribution outliers; this definition differs from the multivariate model and combined review rules. EUR and USD amounts are never pooled.')
    page('Outlier visualization and deliverables')
    p('Figure 2. USD amounts and combined anomaly flags', 'Heading2')
    story.append(amount_plot(evaluated))
    p('Red observations have at least one scored validation, amount-rule, duplicate or ML finding. This chart uses synthetic evaluation invoices and is distinct from the supplied document extraction. Interactive dashboard plots show the selected data and currency.')
    p('6. Outputs and interpretation', 'Heading1')
    p('invoice_analysis.csv contains the freshly extracted sample invoices plus isanomaly (True/False) and anomalyreason. Separate validation, duplicate, rule and ML columns preserve each signal. OCR uncertainty is tracked separately and does not on its own imply an anomalous payment. synthetic_analysis.csv contains the plotted synthetic evaluation. measurements.json records this run and its accuracy denominator.')
    p('The risk score is a capped heuristic queue priority, not a fraud probability. Review source images, unresolved values and duplicate candidates before making payment decisions. Supported layouts are limited, and unresolved financial components can prevent full reconciliation. A larger independently reviewed real invoice dataset is needed for generalization claims.')
    def footer(canvas, doc):
        canvas.setFont('Helvetica', 9)
        canvas.drawString(42, 25, f'Technical report | {as_of}')
        canvas.drawRightString(A4[0]-42, 25, str(doc.page))
    SimpleDocTemplate(str(out/'technical_report.pdf'), pagesize=A4,
                      leftMargin=42, rightMargin=42, topMargin=38, bottomMargin=40).build(story, onFirstPage=footer, onLaterPages=footer)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--as-of', default=local_today().isoformat())
    args = parser.parse_args()
    print(build(args.output_dir, args.as_of))
