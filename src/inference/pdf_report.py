"""
pdf_report.py
─────────────
Fraud Investigation Report Generator.

Generates a formal forensic investigation report for flagged transactions.
Supports PDF generation via ReportLab (or Markdown/HTML fallback).
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any


def generate_fraud_report_bytes(
    transaction_id: str,
    fraud_proba: float,
    amount: float,
    card1: str,
    timestamp: str,
    top_features: list[tuple[str, float]],
    rule_flags: list[str] | None = None,
) -> tuple[bytes, str]:
    """
    Generate an investigation report.
    Returns (report_bytes, mime_type).
    """
    rule_flags = rule_flags or []
    risk_level = "CRITICAL" if fraud_proba >= 0.85 else ("HIGH" if fraud_proba >= 0.70 else "MEDIUM")

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        story = []
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "TitleStyle",
            parent=styles["Heading1"],
            fontSize=22,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=12,
        )
        story.append(Paragraph("🛡️ Sentinel — Fraud Investigation Report", title_style))
        story.append(Paragraph(f"<b>Generated:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}", styles["Normal"]))
        story.append(Spacer(1, 15))

        # Summary Table
        summary_data = [
            ["Transaction ID", str(transaction_id)],
            ["Fraud Probability", f"{fraud_proba:.4f}"],
            ["Risk Assessment", risk_level],
            ["Transaction Amount", f"${amount:.2f}"],
            ["Card Identifier", str(card1)],
            ["Timestamp", str(timestamp)],
        ]
        if rule_flags:
            summary_data.append(["Triggered Rules", ", ".join(rule_flags)])

        table = Table(summary_data, colWidths=[160, 360])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#0f172a")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ]))
        story.append(table)
        story.append(Spacer(1, 20))

        # Top Driving Features
        story.append(Paragraph("<b>Top Risk Drivers (Feature Importance):</b>", styles["Heading2"]))
        story.append(Spacer(1, 8))

        feat_data = [["Feature", "Impact Value", "Risk Direction"]]
        for feat, val in top_features[:15]:
            direction = "Elevates Risk" if val > 0 else "Reduces Risk"
            feat_data.append([feat, f"{val:+.4f}", direction])

        feat_table = Table(feat_data, colWidths=[240, 140, 140])
        feat_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(feat_table)

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue(), "application/pdf"

    except ImportError:
        # Fallback to Markdown / Plain text report
        md = f"""# 🛡️ Sentinel — Fraud Investigation Report
**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}

---

### Transaction Summary
- **Transaction ID:** {transaction_id}
- **Fraud Probability:** {fraud_proba:.4f}
- **Risk Assessment:** {risk_level}
- **Amount:** ${amount:.2f}
- **Card1:** {card1}
- **Timestamp:** {timestamp}
- **Triggered Rules:** {', '.join(rule_flags) if rule_flags else 'None'}

---

### Top Risk Contributors
| Feature | Impact | Direction |
|---|---|---|
"""
        for feat, val in top_features[:15]:
            direction = "Elevates Risk" if val > 0 else "Reduces Risk"
            md += f"| `{feat}` | {val:+.4f} | {direction} |\n"

        return md.encode("utf-8"), "text/markdown"
