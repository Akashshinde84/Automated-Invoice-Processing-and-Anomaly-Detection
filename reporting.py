"""Privacy-conscious CSV and paginated ReportLab reports for selected records."""
from datetime import datetime, timedelta, timezone
from html import escape
from io import BytesIO
import json
from pathlib import Path
import re

import pandas as pd

from validator import money, text, missing
from workflow import safe_name

DISCLAIMER = 'Anomaly flags are review signals, not proof of fraud. LOW priority does not mean approved.'
LIMITATIONS = ('OCR can misread numbers, dates and amounts. Unverified fields require source review. '
               'Synthetic ML performance is not validated real-world fraud detection accuracy. '
               'Experimental vendor-aware models underperform the numeric baseline on the synthetic evaluation. '
               'Missing financial components can prevent total reconciliation; currencies are not converted or combined.')
PUBLIC_COLUMNS = ['analysis_id','document_id','source_filename','invoice_number','vendor_name','invoice_date',
                  'currency','total_amount','ocr_mean_confidence','warnings','field_review','validation_status',
                  'validation_findings','arithmetic_status','duplicate_flag','duplicate_matches','rule_anomaly',
                  'ml_status','ml_anomaly','ml_decision_score','ml_training_kind','ml_explanation','risk_score',
                  'risk_contributions','review_priority','ocr_review_status','review_status','anomaly_explanations',
                  'analysis_as_of','analysis_mode','user_review_status','original_extracted_values','correction_audit']
JSON_COLUMNS={'warnings','field_review','validation_findings','duplicate_matches','risk_contributions',
              'anomaly_explanations','original_extracted_values','correction_audit'}


def redact_paths(value):
    value = str(value)
    value = re.sub(r'[A-Za-z]:[\\/][^\n\r\"\x27,;<>]*', '[local path redacted]', value)
    value = re.sub(r'\\\\[^\n\r\"\x27,;<>]*', '[local path redacted]', value)
    return re.sub(r'(?<!\w)/(?:[^/\s\"\x27]+/)+[^\s\"\x27,;<>]*', '[local path redacted]', value)


def export_frame(frame, *, include_private=False):
    result = frame.copy(deep=True)
    if 'source_filename' not in result:
        result['source_filename'] = result.get('source_file', pd.Series(['unknown']*len(result), index=result.index)).map(safe_name)
    if not include_private:
        result = result[[c for c in PUBLIC_COLUMNS if c in result.columns]].copy()
        def scrub(value):
            if isinstance(value,str):return redact_paths(value)
            if isinstance(value,list):return [scrub(item) for item in value]
            if isinstance(value,dict):return {key:scrub(item) for key,item in value.items()}
            return value
        def safe_cell(value,column):
            if column in JSON_COLUMNS and isinstance(value,str):
                try:return json.dumps(scrub(json.loads(value)),ensure_ascii=False)
                except (ValueError,TypeError):pass
            return scrub(value)
        for column in result.columns:
            result[column] = result[column].map(lambda value:safe_cell(value,column))
    return result


def csv_report(frame, *, include_private=False):
    safe = export_frame(frame, include_private=include_private)
    def spreadsheet_safe(value):
        if isinstance(value, str) and value.lstrip().startswith(('=', '+', '-', '@')) and money(value) is None:
            return "'"+value
        return value
    for column in safe.columns:
        safe[column] = safe[column].map(spreadsheet_safe)
    return safe.to_csv(index=False, na_rep='null').encode('utf-8-sig')


def pdf_report(frame, *, dataset='Selected invoice analysis', model='Rules only', generated=None, include_private=False):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

    font, bold = 'Helvetica', 'Helvetica-Bold'
    candidates = [(Path('C:/Windows/Fonts/arial.ttf'), Path('C:/Windows/Fonts/arialbd.ttf')),
                  (Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'), Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))]
    for regular, heavy in candidates:
        if regular.is_file() and heavy.is_file():
            if 'InvoiceReport' not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont('InvoiceReport', str(regular)))
                pdfmetrics.registerFont(TTFont('InvoiceReportBold', str(heavy)))
            font, bold = 'InvoiceReport', 'InvoiceReportBold'
            break
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='ReportTitle', fontName=bold, fontSize=25, leading=30, textColor=colors.HexColor('#162C49'), spaceAfter=14))
    styles.add(ParagraphStyle(name='ReportHeading', fontName=bold, fontSize=12, leading=17, textColor=colors.HexColor('#245E8F'), spaceBefore=16, spaceAfter=7, keepWithNext=True))
    styles.add(ParagraphStyle(name='ReportBody', fontName=font, fontSize=9, leading=14, spaceAfter=6, wordWrap='CJK'))
    styles.add(ParagraphStyle(name='ReportSmall', fontName=font, fontSize=8, leading=12, textColor=colors.HexColor('#52657B'), spaceAfter=7))
    safe = export_frame(frame, include_private=include_private)
    clean = (lambda value: str(value)) if include_private else redact_paths
    def p(value, style='ReportBody'):
        return Paragraph(escape(clean(value)), styles[style])
    def display(value):
        return 'Unresolved' if missing(value) else str(value)
    def messages(value):
        try:
            parsed = json.loads(value) if isinstance(value, str) else value
            return parsed if isinstance(parsed, list) else []
        except (ValueError, TypeError):
            return [str(value)]
    generated = generated or datetime.now(timezone(timedelta(hours=5,minutes=30))).isoformat(timespec='minutes')
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=42, leftMargin=42, topMargin=58, bottomMargin=48,
                            title='Invoice Analysis Report', author='Invoice Review Application')
    story = [p('INVOICE ANALYSIS REPORT', 'ReportTitle'), p(dataset, 'ReportHeading'),
             p(f'Generated: {generated} | Model: {model}', 'ReportSmall'), p(DISCLAIMER), p(LIMITATIONS,'ReportSmall')]
    count = len(safe)
    duplicate = int(safe.duplicate_flag.eq(True).sum()) if 'duplicate_flag' in safe else 0
    high = int(safe.review_priority.eq('HIGH').sum()) if 'review_priority' in safe else 0
    ocr = int(safe.ocr_review_status.eq('REQUIRED').sum()) if 'ocr_review_status' in safe else 0
    table = Table([[p('INVOICES'),p('DUPLICATE CANDIDATES'),p('HIGH PRIORITY'),p('OCR REVIEW')],
                   [p(count,'ReportHeading'),p(duplicate,'ReportHeading'),p(high,'ReportHeading'),p(ocr,'ReportHeading')]], colWidths=[112,174,110,115])
    table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#EDF3F8')),('VALIGN',(0,0),(-1,-1),'TOP'),
                               ('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),9)]))
    story += [Spacer(1,10),table,p('Scope and review summary','ReportHeading')]
    priorities = safe.review_priority.value_counts().to_dict() if 'review_priority' in safe else {}
    flags = int(safe.rule_anomaly.eq(True).sum()) if 'rule_anomaly' in safe else 0
    ml = int(safe.ml_anomaly.eq(True).sum()) if 'ml_anomaly' in safe else 0
    story += [p(f'Priorities: {priorities}. Rule-flagged records: {flags}. ML outliers: {ml}. Duplicate counts are records, not groups.'),
              p('Amounts are presented in each invoice currency. No cross-currency grand total is calculated.','ReportSmall')]
    if safe.empty:
        story.append(p('No records match the selected report scope.'))
    for i, row in enumerate(safe.to_dict('records'), 1):
        story.append(p(f'{i:03d}  |  {text(row.get("invoice_number")) or "Unresolved invoice number"}', 'ReportHeading'))
        validation_label={'NO_BLOCKING_FINDINGS':'No blocking field findings','INVALID':'Invalid / incomplete fields'}.get(row.get('validation_status'),'See findings')
        ocr_label={'REQUIRED':'Review required','NOT_ASSESSED':'Not assessed','VERIFIED_INPUT':'Verified input','NOT_APPLICABLE_DEMO':'Synthetic input'}.get(row.get('ocr_review_status'),'Not assessed')
        ml_label={'SKIPPED_UNVERIFIED_OR_INVALID':'Skipped: unverified or invalid fields','NOT_LOADED':'No model loaded',
                  'SCORED_DEMO':'Scored: synthetic demonstration','SCORED':'Scored','SKIPPED_UNSUPPORTED_CURRENCY':'No currency baseline'}.get(row.get('ml_status'),'Not scored')
        outlier='Not scored' if missing(row.get('ml_anomaly')) else 'Outlier' if row.get('ml_anomaly') else 'No outlier flag'
        values = [('Source',display(row.get('source_filename'))),('Vendor',display(row.get('vendor_name'))),
                  ('Date / currency',f'{display(row.get("invoice_date"))} / {display(row.get("currency"))}'),
                  ('Total amount',display(row.get('total_amount'))),
                  ('Risk / priority',f'{row.get("risk_score","n/a")} / {row.get("review_priority","n/a")}'),
                  ('Validation / OCR',f'{validation_label} / {ocr_label}'),
                  ('ML',f'{ml_label} / {outlier}')]
        details = Table([[p(key),p(value)] for key,value in values], colWidths=[110,401], hAlign='LEFT')
        details.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(0,-1),colors.HexColor('#F1F5F9')),
                                    ('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#D7E1EC')),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        story.append(details)
        for label, column in [('Validation findings','validation_findings'),('Duplicate matches','duplicate_matches'),
                              ('Score contributions','risk_contributions'),('Original OCR uncertainty (preserved)','warnings'),
                              ('User corrections','correction_audit')]:
            entries = messages(row.get(column, '[]'))
            if entries:
                story.append(p(label, 'ReportHeading'))
                for entry in entries:
                    if isinstance(entry, dict):
                        if column=='risk_contributions':
                            codes=', '.join(code.replace('_',' ').lower() for code in entry.get('codes',[]))
                            message=f'{entry.get("category","Signal").replace("_"," ").title()}: +{entry.get("points",0)} points ({codes}).'
                        elif column=='correction_audit':
                            message=f'{entry.get("field","Field").replace("_"," ").title()}: {display(entry.get("original"))} -> {display(entry.get("corrected"))} (explicit source review).'
                        else:
                            message = entry.get('message') or entry.get('explanation') or json.dumps(entry)
                    else:
                        message = str(entry)
                    for start in range(0,len(message),1600):
                        story.append(p(message[start:start+1600]))
        if row.get('ml_explanation'):
            story.append(p(row['ml_explanation'],'ReportSmall'))
        if row.get('user_review_status'):
            story.append(p('User review: '+str(row['user_review_status']), 'ReportSmall'))
        if include_private:
            for column in ('source_file','raw_text_file','recognition_evidence_file','raw_text'):
                if column in row:
                    story.append(p(f'{column}: {row[column]}', 'ReportSmall'))
    def footer(canvas, document):
        canvas.setStrokeColor(colors.HexColor('#2D83B5'))
        canvas.line(42, A4[1]-35, A4[0]-42, A4[1]-35)
        canvas.setFont(font,8)
        canvas.setFillColor(colors.HexColor('#52657B'))
        canvas.drawString(42,25,'INVOICE REVIEW  |  Review signals, not confirmed fraud')
        canvas.drawRightString(A4[0]-42,25,f'Page {document.page}')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
