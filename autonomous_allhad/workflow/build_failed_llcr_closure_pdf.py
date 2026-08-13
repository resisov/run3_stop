#!/usr/bin/env python3
"""Build the illustrated English PDF report for the failed LLCR closure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


NAVY = colors.HexColor("#17365D")
BLUE = colors.HexColor("#1F77B4")
RED = colors.HexColor("#B22222")
LIGHT_RED = colors.HexColor("#F7E3E3")
LIGHT_BLUE = colors.HexColor("#E8F0F8")
MID_GRAY = colors.HexColor("#666666")
LIGHT_GRAY = colors.HexColor("#F2F2F2")
DARK = colors.HexColor("#1F1F1F")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=23,
            leading=27,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=12,
            leading=16,
            textColor=MID_GRAY,
            spaceAfter=16,
        ),
        "h1": ParagraphStyle(
            "ReportH1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=NAVY,
            spaceBefore=6,
            spaceAfter=8,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "ReportH2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12.5,
            leading=15,
            textColor=NAVY,
            spaceBefore=8,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "ReportBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13.2,
            textColor=DARK,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "ReportSmall",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.2,
            leading=10.5,
            textColor=DARK,
            alignment=TA_LEFT,
            spaceAfter=4,
        ),
        "caption": ParagraphStyle(
            "FigureCaption",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.8,
            leading=10,
            textColor=MID_GRAY,
            alignment=TA_LEFT,
            spaceBefore=3,
            spaceAfter=7,
        ),
        "decision": ParagraphStyle(
            "DecisionText",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=15,
            textColor=RED,
            alignment=TA_LEFT,
            spaceAfter=0,
        ),
        "quote": ParagraphStyle(
            "ANQuote",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=9,
            leading=13,
            textColor=DARK,
            alignment=TA_JUSTIFY,
            leftIndent=8,
            rightIndent=8,
            spaceAfter=0,
        ),
        "cover_note": ParagraphStyle(
            "CoverNote",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=DARK,
            alignment=TA_LEFT,
            spaceAfter=4,
        ),
    }


def page_header_footer(canvas: Any, document: Any) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#B5B5B5"))
    canvas.setLineWidth(0.5)
    canvas.line(16 * mm, height - 13 * mm, width - 16 * mm, height - 13 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MID_GRAY)
    canvas.drawString(
        16 * mm,
        height - 10 * mm,
        "CMS Work in progress | 2024 (13.6 TeV)",
    )
    canvas.drawRightString(
        width - 16 * mm,
        height - 10 * mm,
        f"Failed LLCR closure | Page {document.page}",
    )
    canvas.line(16 * mm, 12 * mm, width - 16 * mm, 12 * mm)
    canvas.drawString(
        16 * mm,
        8 * mm,
        "Prefit validation study - statistical covariance only",
    )
    canvas.drawRightString(
        width - 16 * mm,
        8 * mm,
        "28 July 2026",
    )
    canvas.restoreState()


def decision_box(text: str, style: ParagraphStyle) -> Table:
    table = Table([[Paragraph(text, style)]], colWidths=[169 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT_RED),
                ("BOX", (0, 0), (-1, -1), 1.2, RED),
                ("LEFTPADDING", (0, 0), (-1, -1), 11),
                ("RIGHTPADDING", (0, 0), (-1, -1), 11),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    return table


def note_box(
    text: str,
    style: ParagraphStyle,
    background: colors.Color = LIGHT_BLUE,
    border: colors.Color = BLUE,
) -> Table:
    table = Table([[Paragraph(text, style)]], colWidths=[169 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.8, border),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def styled_table(
    rows: list[list[Any]],
    widths: list[float],
    font_size: float = 7.5,
    alignments: list[str] | None = None,
) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands: list[tuple[Any, ...]] = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#A8A8A8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    for row in range(1, len(rows)):
        if row % 2 == 0:
            commands.append(("BACKGROUND", (0, row), (-1, row), LIGHT_GRAY))
    if alignments:
        for column, alignment in enumerate(alignments):
            commands.append(
                ("ALIGN", (column, 1), (column, -1), alignment)
            )
            commands.append(
                ("ALIGN", (column, 0), (column, 0), alignment)
            )
    table.setStyle(TableStyle(commands))
    return table


def bullets(items: list[str], style: ParagraphStyle) -> ListFlowable:
    return ListFlowable(
        [
            ListItem(
                Paragraph(item, style),
                leftIndent=12,
                bulletColor=NAVY,
            )
            for item in items
        ],
        bulletType="bullet",
        start="circle",
        leftIndent=18,
        bulletFontName="Helvetica",
        bulletFontSize=7,
        spaceBefore=2,
        spaceAfter=5,
    )


def figure(
    image_path: Path,
    caption: str,
    figure_number: int,
    caption_style: ParagraphStyle,
    width: float = 160 * mm,
) -> list[Any]:
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    image = Image(str(image_path))
    ratio = image.imageHeight / image.imageWidth
    image.drawWidth = width
    image.drawHeight = width * ratio
    return [
        image,
        Paragraph(
            f"<b>Figure {figure_number}.</b> {caption}",
            caption_style,
        ),
    ]


def paired_figures(
    left_path: Path,
    left_caption: str,
    left_number: int,
    right_path: Path,
    right_caption: str,
    right_number: int,
    caption_style: ParagraphStyle,
) -> Table:
    panel_width = 81.5 * mm
    left_image = Image(str(left_path))
    right_image = Image(str(right_path))
    for image in (left_image, right_image):
        ratio = image.imageHeight / image.imageWidth
        image.drawWidth = panel_width
        image.drawHeight = panel_width * ratio
    table = Table(
        [
            [left_image, right_image],
            [
                Paragraph(
                    f"<b>Figure {left_number}.</b> {left_caption}",
                    caption_style,
                ),
                Paragraph(
                    f"<b>Figure {right_number}.</b> {right_caption}",
                    caption_style,
                ),
            ],
        ],
        colWidths=[84.5 * mm, 84.5 * mm],
    )
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 1.5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 1.5),
                ("TOPPADDING", (0, 0), (-1, -1), 1),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return table


def make_document(report_dir: Path, output_path: Path) -> None:
    result = load_json(report_dir / "top_w_closure.json")
    plots = report_dir / "plots"
    style = styles()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=19 * mm,
        bottomMargin=18 * mm,
        title="Failure of the 2024 Lost-Lepton Transfer-Factor Closure",
        author="CMS Run-3 all-hadronic stop analysis",
        subject="Lost-lepton control-region closure and method decision",
    )
    story: list[Any] = []

    story.extend(
        [
            Spacer(1, 8 * mm),
            Paragraph(
                "Failure of the 2024 Lost-Lepton Transfer-Factor Closure",
                style["title"],
            ),
            Paragraph(
                "Prefit validation report and analysis-method decision",
                style["subtitle"],
            ),
            HRFlowable(
                width="100%",
                thickness=1.3,
                color=NAVY,
                spaceBefore=2,
                spaceAfter=12,
            ),
            decision_box(
                "The current MC-derived LLCR-to-zero-lepton "
                "transfer-factor method fails its data validation and "
                "must be retired from the nominal lost-lepton background "
                "estimation.",
                style["decision"],
            ),
            Spacer(1, 7 * mm),
            Paragraph("Executive summary", style["h1"]),
            Paragraph(
                "The combined transfer-factor estimate and a dedicated "
                "Top/W-split implementation were tested in independent "
                "MC folds and orthogonal data validation regions. The "
                "software and normalization chain closes in MC, but the "
                "method does not transfer from the one-lepton control "
                "region to data in the zero-lepton target.",
                style["body"],
            ),
            bullets(
                [
                    "High-dM data residual closure reaches only 65-68% "
                    "of the lost-lepton yield required by data.",
                    "Top/W separation changes the high-dM result by less "
                    "than 0.4 percentage points.",
                    "In two low-dM validation regions, raw MC is within "
                    "1-3% of the data residual, while TF application "
                    "degrades the agreement to 78-90%.",
                    "A held-out low-dM Nb=1 control category is "
                    "underpredicted by 21% with a pull of -14.48.",
                    "The failed method is archived for diagnosis only and "
                    "must not define nominal datacard rates.",
                ],
                style["cover_note"],
            ),
            Spacer(1, 3 * mm),
            note_box(
                "<b>Scope:</b> 1,153 ROOT files, 361,054,245 scanned "
                "events, and 34,735,958 selected events. The selection "
                "authority is <font name='Courier'>"
                "real_subset_worker.py</font>. Nominal intermediates were "
                "not modified. Statistical covariance is included; "
                "detector and modeling systematics are not.",
                style["small"],
            ),
            Spacer(1, 5 * mm),
            Paragraph(
                "CMS Work in progress - 2024 data at 13.6 TeV",
                ParagraphStyle(
                    "CoverFooter",
                    parent=style["small"],
                    alignment=TA_CENTER,
                    textColor=MID_GRAY,
                ),
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            Paragraph("1. Method and validation strategy", style["h1"]),
            Paragraph(
                "The combined method multiplies the background-subtracted "
                "one-lepton data yield by an MC ratio of zero-lepton to "
                "one-lepton target-process yields. The closure observable "
                "is the predicted lost-lepton yield divided by the "
                "zero-lepton data residual after subtraction of Z to "
                "neutrinos, DY, photon+jets, diboson, and QCD multijet.",
                style["body"],
            ),
            note_box(
                "<b>Combined TF:</b> prediction = "
                "TF(combined TTbar + W+jets + single-top) x "
                "(one-lepton data - other MC).<br/>"
                "<b>Top/W split:</b> prediction = mu(Top) x "
                "Top zero-lepton MC + mu(W) x W+jets zero-lepton MC.",
                style["small"],
            ),
            Paragraph("Top/W normalization fit", style["h2"]),
        ]
    )
    high_fit = result["fits"]["highdm"]
    low_fit = result["fits"]["lowdm"]
    fit_rows = [
        ["Regime", "Top normalization", "W+jets normalization", "Correlation"],
        [
            "High-dM",
            f"{high_fit['scale_factors'][0]:.4f}"
            f" +/- {high_fit['scale_factor_uncertainties'][0]:.4f}",
            f"{high_fit['scale_factors'][1]:.4f}"
            f" +/- {high_fit['scale_factor_uncertainties'][1]:.4f}",
            f"{high_fit['scale_factor_correlation']:.3f}",
        ],
        [
            "Low-dM",
            f"{low_fit['scale_factors'][0]:.4f}"
            f" +/- {low_fit['scale_factor_uncertainties'][0]:.4f}",
            f"{low_fit['scale_factors'][1]:.4f}"
            f" +/- {low_fit['scale_factor_uncertainties'][1]:.4f}",
            f"{low_fit['scale_factor_correlation']:.3f}",
        ],
    ]
    story.extend(
        [
            styled_table(
                fit_rows,
                [35 * mm, 45 * mm, 50 * mm, 35 * mm],
                font_size=8,
                alignments=["LEFT", "CENTER", "CENTER", "CENTER"],
            ),
            Spacer(1, 4 * mm),
            *figure(
                plots / "top_w_scale_factors.png",
                "Independent Top and W+jets normalizations extracted "
                "from enriched one-lepton control categories. The "
                "uncertainties shown are statistical.",
                1,
                style["caption"],
                width=128 * mm,
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            Paragraph("2. Technical closure and control diagnostics", style["h1"]),
            Paragraph(
                "The independent A/B-fold MC closure has maximum pulls "
                "below 0.21, and the full-mixture MC pseudodata closure "
                "has maximum pulls below 0.19. These tests validate the "
                "histogram reduction, normalization bookkeeping, bin "
                "mapping, fold independence, and background subtraction. "
                "They cannot validate the generator-to-data transfer.",
                style["body"],
            ),
        ]
    )
    technical_rows = [
        ["MC categorization", "Valid bins", "Chi2 / ndf", "Max. pull"],
        ["High-dM pTmiss", "6 / 7", "0.0029 / 6", "0.054"],
        ["High-dM search bins", "29 / 60", "0.0926 / 29", "0.173"],
        ["Low-dM search bins", "36 / 42", "0.1402 / 36", "0.206"],
    ]
    story.extend(
        [
            styled_table(
                technical_rows,
                [64 * mm, 30 * mm, 42 * mm, 29 * mm],
                font_size=8,
                alignments=["LEFT", "CENTER", "CENTER", "CENTER"],
            ),
            Spacer(1, 5 * mm),
            paired_figures(
                plots / "highdm_control_shape_highdm_nb0.png",
                "High-dM W-enriched one-lepton control shape. The "
                "integrated normalization is fitted, but the recoil "
                "shape remains incompatible (p = 7.95e-4).",
                2,
                plots / "lowdm_nb_control_validation.png",
                "Low-dM Nb control validation. The held-out Nb=1 "
                "category is predicted at 0.788 of the data residual.",
                3,
                style["caption"],
            ),
            Spacer(1, 3 * mm),
            note_box(
                "The low-dM Nb=1 category is not used in the "
                "normalization fit: prediction = 6814.8 +/- 81.2, "
                "data residual = 8649.9 +/- 97.3, and pull = -14.48.",
                style["small"],
                background=LIGHT_RED,
                border=RED,
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            Paragraph("3. Failure in high-dM data residual closure", style["h1"]),
            Paragraph(
                "The high-dM data residual requires substantially more "
                "lost-lepton yield than either TF implementation predicts. "
                "Top/W separation does not improve the result: the "
                "integrated ratios change from 0.6526 to 0.6539 in the "
                "Nb=0 region and from 0.6757 to 0.6794 in the "
                "Top-enriched region.",
                style["body"],
            ),
            paired_figures(
                plots / "top_w_closure_highdm_nb0.png",
                "High-dM Nb=0 closure. The Top/W-split prediction reaches "
                "0.654 of the data residual.",
                4,
                plots / "top_w_closure_highdm_njet3to4_nb1plus.png",
                "High-dM Top-enriched closure. The prediction reaches "
                "0.679 of the data residual and develops a recoil-dependent "
                "deficit.",
                5,
                style["caption"],
            ),
            Spacer(1, 5 * mm),
        ]
    )
    high_rows = [
        ["Region", "Combined TF", "Top/W split", "p-value", "Max. pull"],
        ["High-dM Nb=0", "0.6526", "0.6539 +/- 0.0211", "1.6e-75", "14.51"],
        [
            "High-dM Top enriched",
            "0.6757",
            "0.6794 +/- 0.0078",
            "4.0e-268",
            "19.64",
        ],
    ]
    story.extend(
        [
            styled_table(
                high_rows,
                [48 * mm, 28 * mm, 42 * mm, 27 * mm, 22 * mm],
                font_size=7.7,
                alignments=["LEFT", "CENTER", "CENTER", "CENTER", "CENTER"],
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            Paragraph("4. Transfer-factor degradation in low-dM regions", style["h1"]),
            Paragraph(
                "The low-dM regions provide direct evidence that the "
                "LLCR-derived correction can be harmful. Raw lost-lepton "
                "MC is already close to the data residual, while the "
                "combined TF and the Top/W-split method move the prediction "
                "away from unity.",
                style["body"],
            ),
            paired_figures(
                plots / "top_w_closure_lowdm_met250to300.png",
                "Low-pTmiss validation. Raw MC is at 0.990, while the "
                "combined and split TF predictions are 0.842 and 0.776.",
                6,
                plots / "top_w_closure_lowdm_isr200to300.png",
                "Low-ISR validation. Raw MC is at 1.025, while the "
                "combined and split TF predictions are 0.897 and 0.800.",
                7,
                style["caption"],
            ),
            Spacer(1, 4 * mm),
        ]
    )
    raw_rows = [
        ["Validation region", "Raw LL MC", "Combined TF", "Top/W split"],
        ["High-dM Nb=0", "0.6438", "0.6526", "0.6539"],
        ["High-dM Top enriched", "0.7332", "0.6757", "0.6794"],
        ["Low-dM low ISR", "1.0254", "0.8966", "0.8001"],
        ["Low-dM low pTmiss", "0.9904", "0.8417", "0.7763"],
        ["Low-dM low MET significance", "1.1326", "1.0197", "0.9911"],
    ]
    story.extend(
        [
            styled_table(
                raw_rows,
                [66 * mm, 33 * mm, 33 * mm, 33 * mm],
                font_size=7.8,
                alignments=["LEFT", "CENTER", "CENTER", "CENTER"],
            ),
            Spacer(1, 3 * mm),
            note_box(
                "<b>Key observation:</b> in low-pTmiss and low-ISR "
                "validation regions, the raw MC agrees at the 1-3% level. "
                "The TF methods import an LLCR-specific discrepancy and "
                "degrade the prediction by 10-22 percentage points.",
                style["small"],
                background=LIGHT_RED,
                border=RED,
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            Paragraph("5. Compatible sideband and why it does not rescue the method", style["h1"]),
            Paragraph(
                "The low-MET-significance validation region is compatible "
                "with closure after the Top/W split. It has only three "
                "valid bins, a large integrated uncertainty, and modest "
                "lost-lepton purity in the zero-lepton target. Its "
                "agreement cannot validate a method that fails in all "
                "other constraining validation regions.",
                style["body"],
            ),
            *figure(
                plots / "top_w_closure_lowdm_significance7to10.png",
                "Low-MET-significance validation. The integrated "
                "Top/W-split ratio is 0.991 +/- 0.292. This statistically "
                "weak agreement is retained as a cross-check, not used "
                "to override the failed closure elsewhere.",
                8,
                style["caption"],
                width=145 * mm,
            ),
            Spacer(1, 3 * mm),
            note_box(
                "Selecting only the sideband in which the correction "
                "happens to improve agreement would be an unjustified "
                "a posteriori choice. A valid transfer method must be "
                "portable across independently defined validation regions.",
                style["small"],
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            Paragraph("6. Analysis decision and consequences", style["h1"]),
            decision_box(
                "Retire the current MC-derived LLCR transfer-factor method.",
                style["decision"],
            ),
            Spacer(1, 5 * mm),
            bullets(
                [
                    "Do not use either the combined or Top/W-split TF "
                    "prediction as the nominal lost-lepton central value.",
                    "Do not propagate the failed TF estimate into nominal "
                    "datacards.",
                    "Do not convert the observed 20-50% nonclosure into an "
                    "empirical correction without an independently "
                    "validated model.",
                    "Preserve the code, JSON outputs, and figures as an "
                    "archived failed-method study.",
                    "Use raw lost-lepton MC only as a clearly labeled "
                    "temporary reference. Raw MC does not solve the "
                    "high-dM discrepancy.",
                    "Adopt a replacement only after it improves closure "
                    "relative to raw MC in held-out validation regions.",
                ],
                style["body"],
            ),
            Paragraph("Recommended replacement", style["h2"]),
            Paragraph(
                "The preferred replacement is an event-level lepton-removal "
                "or lepton-embedding estimate. Electron and muon data "
                "events should be treated separately; the identified "
                "lepton momentum should be added to missing transverse "
                "momentum; recoil, angular selections, event categories, "
                "and search bins should be recomputed; and the event "
                "should be weighted by a data-measured probability for "
                "the lepton to fail the veto. Out-of-acceptance leptons "
                "and tau contributions must be explicit auxiliary "
                "components.",
                style["body"],
            ),
            Paragraph("AN-ready conclusion", style["h2"]),
            note_box(
                "The MC-derived lost-lepton transfer-factor method was "
                "tested using independent MC folds and orthogonal data "
                "validation regions. Although the implementation closes "
                "in pure-MC and full-mixture MC pseudodata tests, it fails "
                "the data residual closure in both the high- and low-dM "
                "selections. In the high-dM validation regions, the method "
                "predicts only 65-68% of the lost-lepton residual observed "
                "in data. In two low-dM validation regions, the uncorrected "
                "lost-lepton MC agrees with the data residual at the 1-3% "
                "level, while application of the transfer factor degrades "
                "the agreement to 78-90%. Separating Top and W+jets does "
                "not restore closure and exposes additional inconsistencies "
                "in recoil shape and Nb migration. The transfer-factor "
                "method is therefore rejected as the nominal lost-lepton "
                "background-estimation strategy and is retained only as "
                "a documented failed validation study.",
                style["quote"],
                background=LIGHT_GRAY,
                border=NAVY,
            ),
            Spacer(1, 6 * mm),
            Paragraph(
                "Supporting machine-readable results, source plots, and "
                "the complete Markdown report are stored next to this PDF.",
                style["small"],
            ),
        ]
    )
    document.build(
        story,
        onFirstPage=page_header_footer,
        onLaterPages=page_header_footer,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_dir = args.report_dir.resolve()
    output = args.output.resolve()
    make_document(report_dir, output)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output),
                "bytes": output.stat().st_size,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
