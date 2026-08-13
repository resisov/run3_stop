#!/usr/bin/env python3
"""Build the Korean PDF/HTML report comparing two lost-lepton estimators."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image,
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
BLUE = colors.HexColor("#2F6FAE")
RED = colors.HexColor("#A51C30")
GREEN = colors.HexColor("#2E6B45")
LIGHT_BLUE = colors.HexColor("#EAF2F8")
LIGHT_RED = colors.HexColor("#F8E7EA")
LIGHT_GREEN = colors.HexColor("#EAF4ED")
LIGHT_GRAY = colors.HexColor("#F3F4F6")
MID_GRAY = colors.HexColor("#5F6368")
DARK = colors.HexColor("#202124")
KOREAN_FONT = "KoreanReport"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def register_fonts() -> None:
    font_path = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    if not font_path.is_file():
        raise FileNotFoundError(font_path)
    pdfmetrics.registerFont(TTFont(KOREAN_FONT, str(font_path)))
    pdfmetrics.registerFontFamily(
        KOREAN_FONT,
        normal=KOREAN_FONT,
        bold=KOREAN_FONT,
        italic=KOREAN_FONT,
        boldItalic=KOREAN_FONT,
    )


def report_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleKorean",
            parent=base["Title"],
            fontName=KOREAN_FONT,
            fontSize=22,
            leading=29,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleKorean",
            parent=base["Normal"],
            fontName=KOREAN_FONT,
            fontSize=10.5,
            leading=15,
            textColor=MID_GRAY,
            spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "H1Korean",
            parent=base["Heading1"],
            fontName=KOREAN_FONT,
            fontSize=15,
            leading=20,
            textColor=NAVY,
            spaceBefore=5,
            spaceAfter=7,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2Korean",
            parent=base["Heading2"],
            fontName=KOREAN_FONT,
            fontSize=11.5,
            leading=16,
            textColor=NAVY,
            spaceBefore=6,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "BodyKorean",
            parent=base["BodyText"],
            fontName=KOREAN_FONT,
            fontSize=8.8,
            leading=13.2,
            textColor=DARK,
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "SmallKorean",
            parent=base["BodyText"],
            fontName=KOREAN_FONT,
            fontSize=7.6,
            leading=10.5,
            textColor=DARK,
            alignment=TA_LEFT,
            spaceAfter=3,
        ),
        "caption": ParagraphStyle(
            "CaptionKorean",
            parent=base["BodyText"],
            fontName=KOREAN_FONT,
            fontSize=7.0,
            leading=9.5,
            textColor=MID_GRAY,
            alignment=TA_LEFT,
            spaceBefore=2,
            spaceAfter=5,
        ),
        "box": ParagraphStyle(
            "BoxKorean",
            parent=base["BodyText"],
            fontName=KOREAN_FONT,
            fontSize=9.0,
            leading=13.5,
            textColor=DARK,
            alignment=TA_LEFT,
        ),
        "decision": ParagraphStyle(
            "DecisionKorean",
            parent=base["BodyText"],
            fontName=KOREAN_FONT,
            fontSize=10.5,
            leading=15,
            textColor=RED,
            alignment=TA_LEFT,
        ),
        "table": ParagraphStyle(
            "TableKorean",
            parent=base["BodyText"],
            fontName=KOREAN_FONT,
            fontSize=6.9,
            leading=9.2,
            textColor=DARK,
            alignment=TA_LEFT,
        ),
        "table_header": ParagraphStyle(
            "TableHeaderKorean",
            parent=base["BodyText"],
            fontName=KOREAN_FONT,
            fontSize=6.9,
            leading=9.2,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
    }


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def page_header_footer(canvas: Any, document: Any) -> None:
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#B7BDC5"))
    canvas.setLineWidth(0.45)
    canvas.line(16 * mm, height - 13 * mm, width - 16 * mm, height - 13 * mm)
    canvas.setFont(KOREAN_FONT, 7.0)
    canvas.setFillColor(MID_GRAY)
    canvas.drawString(16 * mm, height - 10 * mm, "CMS Work in progress")
    canvas.drawRightString(
        width - 16 * mm,
        height - 10 * mm,
        f"2024 (13.6 TeV) | Page {document.page}",
    )
    canvas.line(16 * mm, 12 * mm, width - 16 * mm, 12 * mm)
    canvas.drawString(
        16 * mm,
        8 * mm,
        "Lost-lepton background estimation: two-method validation",
    )
    canvas.drawRightString(width - 16 * mm, 8 * mm, "28 July 2026")
    canvas.restoreState()


def box(
    text: str,
    style: ParagraphStyle,
    background: colors.Color,
    border: colors.Color,
    width: float = 169 * mm,
) -> Table:
    table = Table([[para(text, style)]], colWidths=[width])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.9, border),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def bullets(items: list[str], style: ParagraphStyle) -> ListFlowable:
    return ListFlowable(
        [
            ListItem(
                para(item, style),
                leftIndent=12,
                bulletColor=NAVY,
            )
            for item in items
        ],
        bulletType="bullet",
        start="circle",
        leftIndent=18,
        bulletFontName=KOREAN_FONT,
        bulletFontSize=6,
        spaceBefore=1,
        spaceAfter=5,
    )


def styled_table(
    rows: list[list[str]],
    widths: list[float],
    styles: dict[str, ParagraphStyle],
    alignments: list[str] | None = None,
) -> Table:
    converted = []
    for row_index, row in enumerate(rows):
        style = styles["table_header"] if row_index == 0 else styles["table"]
        converted.append([para(str(cell), style) for cell in row])
    table = Table(converted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands: list[tuple[Any, ...]] = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#AAB0B8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    for row_index in range(1, len(converted)):
        if row_index % 2 == 0:
            commands.append(
                ("BACKGROUND", (0, row_index), (-1, row_index), LIGHT_GRAY)
            )
    if alignments:
        for column, alignment in enumerate(alignments):
            commands.append(
                ("ALIGN", (column, 1), (column, -1), alignment)
            )
    table.setStyle(TableStyle(commands))
    return table


def image_with_caption(
    image_path: Path,
    caption: str,
    number: int,
    styles: dict[str, ParagraphStyle],
    width: float = 132 * mm,
) -> list[Any]:
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    image = Image(str(image_path))
    ratio = image.imageHeight / image.imageWidth
    image.drawWidth = width
    image.drawHeight = width * ratio
    return [
        Table(
            [[image]],
            colWidths=[width],
            style=TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            ),
        ),
        para(f"<b>Figure {number}.</b> {caption}", styles["caption"]),
    ]


def paired_figures(
    left_path: Path,
    left_caption: str,
    left_number: int,
    right_path: Path,
    right_caption: str,
    right_number: int,
    styles: dict[str, ParagraphStyle],
) -> Table:
    panel_width = 81 * mm
    images = []
    for path in (left_path, right_path):
        if not path.is_file():
            raise FileNotFoundError(path)
        image = Image(str(path))
        ratio = image.imageHeight / image.imageWidth
        image.drawWidth = panel_width
        image.drawHeight = panel_width * ratio
        images.append(image)
    table = Table(
        [
            images,
            [
                para(
                    f"<b>Figure {left_number}.</b> {left_caption}",
                    styles["caption"],
                ),
                para(
                    f"<b>Figure {right_number}.</b> {right_caption}",
                    styles["caption"],
                ),
            ],
        ],
        colWidths=[84.5 * mm, 84.5 * mm],
    )
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 1),
                ("RIGHTPADDING", (0, 0), (-1, -1), 1),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return table


def raw_ratio(record: dict[str, Any]) -> float:
    mask = record["valid_mask"]
    top = sum(
        value for value, valid in zip(record["top_target"], mask) if valid
    )
    wjets = sum(
        value for value, valid in zip(record["w_target"], mask) if valid
    )
    residual = sum(
        value for value, valid in zip(record["target_residual"], mask) if valid
    )
    return (top + wjets) / residual


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def validate_component_policy(
    top_w: dict[str, Any],
    removal: dict[str, Any],
) -> None:
    if top_w["top_processes"] != ["TT", "ST"]:
        raise RuntimeError(f"unexpected Top policy: {top_w['top_processes']}")
    policy = removal["component_policy"]
    if policy["Top"] != ["TT", "ST"] or policy["W"] != ["WtoLNu"]:
        raise RuntimeError(f"unexpected removal component policy: {policy}")
    if policy["independent_target_components"] != ["Top", "W"]:
        raise RuntimeError(f"unexpected moving components: {policy}")


def build_pdf(
    output_path: Path,
    report_dir: Path,
    top_w: dict[str, Any],
    removal: dict[str, Any],
) -> None:
    styles = report_styles()
    plots = report_dir / "plots"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=19 * mm,
        bottomMargin=18 * mm,
        title="2024 Lost-lepton background estimation: two-method report",
        author="CMS Run-3 all-hadronic stop analysis",
        subject="Transfer-factor and event-level removal closure comparison",
    )
    story: list[Any] = []

    fits = top_w["fits"]
    validation = top_w["validation_regions"]
    data_validation = removal["data_validation"]
    mc_removal = removal["mc_crossfit_closure"]
    normalizations = removal["normalizations"]
    raw_mc_comparison = load_json(
        report_dir / "removal_raw_mc_comparison.json"
    )["records"]

    story.extend(
        [
            Spacer(1, 8 * mm),
            para(
                "2024 Lost-lepton 배경추정 두 방법의 상세 비교",
                styles["title"],
            ),
            para(
                "MC-derived transfer factor와 event-level lepton removal의 "
                "prefit closure, uncertainty, 실패 원인 및 분석 결정",
                styles["subtitle"],
            ),
            HRFlowable(
                width="100%",
                thickness=1.3,
                color=NAVY,
                spaceBefore=1,
                spaceAfter=11,
            ),
            box(
                "<b>최종 판정:</b> 두 방법 모두 현재 nominal lost-lepton "
                "배경추정에 채택하지 않는다. Transfer factor는 MC technical "
                "closure를 통과하지만 data residual closure가 실패한다. "
                "Monolithic lepton removal은 적분 normalization 이후에도 "
                "MC shape closure 자체가 실패한다.",
                styles["decision"],
                LIGHT_RED,
                RED,
            ),
            Spacer(1, 6 * mm),
            para("핵심 요약", styles["h1"]),
            bullets(
                [
                    "모든 단계에서 Top = TT + ST이며 W만 독립 성분이다. "
                    "ST 단독 scale factor, transfer factor 또는 nuisance는 없다.",
                    "Transfer factor의 high-Δm data residual 예측은 관측량의 "
                    "약 65-68%에 그친다.",
                    "Low-Δm low-ISR 및 low-MET에서는 raw LL MC가 1-3% "
                    "수준에서 맞지만 TF 적용 후 0.80 및 0.78까지 악화된다.",
                    "Lepton removal은 high-Δm U_T에서 χ²/ndf = "
                    "11361.1/7, max |pull| = 57.71로 MC shape closure에 실패한다.",
                    "다음 방법은 loss mode별 efficiency/response를 분리하는 "
                    "hybrid embedding이어야 한다.",
                ],
                styles["body"],
            ),
            Spacer(1, 2 * mm),
            box(
                "<b>입력:</b> 두 연구 모두 1,153개 flat ROOT와 "
                "361,054,245개 event를 읽었다. Selection authority는 "
                "<font name='KoreanReport'>real_subset_worker.py</font>이며 "
                "nominal 중간산출물은 수정하지 않았다. Nominal SR data는 "
                "blinded 상태다.",
                styles["box"],
                LIGHT_BLUE,
                BLUE,
            ),
            Spacer(1, 5 * mm),
            para(
                "CMS Work in progress - 2024 data at 13.6 TeV",
                ParagraphStyle(
                    "CoverFooter",
                    parent=styles["small"],
                    alignment=TA_CENTER,
                    textColor=MID_GRAY,
                ),
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            para("1. 공통 입력과 검증 원칙", styles["h1"]),
            para(
                "두 estimator는 같은 물리 dataset과 동일한 selection authority를 "
                "사용한다. Data는 JetMET, target MC는 Top(TT+ST)과 WtoLNu, "
                "Other MC는 Zto2Nu, DY, GJ, VV 및 QCD다. MC normalization은 "
                "physical dataset 전체 generator sum of weights를 분모로 하며 "
                "pileup, b tagging, electron ID와 muon ID nominal weight를 "
                "적용한다.",
                styles["body"],
            ),
            styled_table(
                [
                    ["항목", "Transfer factor", "Event removal"],
                    ["입력 ROOT", "1,153", "1,153"],
                    ["읽은 event", "361,054,245", "361,054,245"],
                    ["선택된 event", "34,735,958", "38,634,909"],
                    ["Top 정책", "TT + ST", "TT + ST"],
                    ["독립 이동 성분", "Top, W", "Top, W"],
                    ["Nominal 수정", "없음", "없음"],
                    ["SR data", "blinded", "blinded"],
                ],
                [57 * mm, 56 * mm, 56 * mm],
                styles,
                ["LEFT", "CENTER", "CENTER"],
            ),
            Spacer(1, 4 * mm),
            para("검증의 세 층", styles["h2"]),
            bullets(
                [
                    "<b>Technical closure:</b> 독립 event-hash fold를 이용해 "
                    "코드, normalization, bin mapping과 subtraction을 검사한다.",
                    "<b>Data residual closure:</b> data zero-lepton yield에서 "
                    "Other MC를 차감한 lost-lepton residual을 직접 비교한다.",
                    "<b>Adoption gate:</b> 적분비뿐 아니라 shape χ², bin pull, "
                    "raw MC 대비 개선 여부를 동시에 요구한다.",
                ],
                styles["body"],
            ),
            box(
                "MC fold closure의 좋은 p-value는 implementation sanity check다. "
                "양쪽 fold가 같은 generator model에서 왔으므로 "
                "simulation-to-data transfer의 증거로 해석하지 않는다.",
                styles["box"],
                LIGHT_GREEN,
                GREEN,
            ),
            PageBreak(),
        ]
    )

    high_fit = fits["highdm"]
    low_fit = fits["lowdm"]
    story.extend(
        [
            para("2. 방법 1: MC-derived transfer factor", styles["h1"]),
            para(
                "Combined estimator는 MC의 zero-lepton/one-lepton yield ratio를 "
                "background-subtracted one-lepton data에 곱한다. Top/W-split "
                "extension은 W-enriched와 Top-enriched control category의 "
                "혼합 방정식을 풀어 μ(Top)과 μ(W)를 별도로 측정한 뒤 "
                "zero-lepton target template에 적용한다.",
                styles["body"],
            ),
            box(
                "<b>Combined:</b> N_pred(i) = TF_combined(i) × "
                "[Data_1l(i) - OtherMC_1l(i)]<br/>"
                "<b>Top/W split:</b> N_pred(i) = μ_Top × Top_0l(i) + "
                "μ_W × W_0l(i)<br/>"
                "<b>Closure:</b> N_pred(i) / "
                "[Data_0l(i) - OtherMC_0l(i)]",
                styles["box"],
                LIGHT_BLUE,
                BLUE,
            ),
            Spacer(1, 4 * mm),
            para("2.1 Top/W normalization 결과", styles["h2"]),
            styled_table(
                [
                    ["Regime", "μ(Top = TT+ST)", "μ(W)", "Correlation"],
                    [
                        "high-Δm",
                        f"{high_fit['scale_factors'][0]:.4f} ± "
                        f"{high_fit['scale_factor_uncertainties'][0]:.4f}",
                        f"{high_fit['scale_factors'][1]:.4f} ± "
                        f"{high_fit['scale_factor_uncertainties'][1]:.4f}",
                        f"{high_fit['scale_factor_correlation']:.3f}",
                    ],
                    [
                        "low-Δm",
                        f"{low_fit['scale_factors'][0]:.4f} ± "
                        f"{low_fit['scale_factor_uncertainties'][0]:.4f}",
                        f"{low_fit['scale_factors'][1]:.4f} ± "
                        f"{low_fit['scale_factor_uncertainties'][1]:.4f}",
                        f"{low_fit['scale_factor_correlation']:.3f}",
                    ],
                ],
                [37 * mm, 48 * mm, 44 * mm, 40 * mm],
                styles,
                ["LEFT", "CENTER", "CENTER", "CENTER"],
            ),
            Spacer(1, 4 * mm),
            *image_with_caption(
                plots / "tf_top_w_scale_factors.png",
                "High-Δm과 low-Δm의 fitted Top(TT+ST) 및 W scale factor. "
                "Low-Δm W는 약 27% 감소한다.",
                1,
                styles,
                width=118 * mm,
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            para("2.2 Transfer-factor technical closure", styles["h1"]),
            styled_table(
                [
                    [
                        "Distribution",
                        "유효 bin",
                        "χ²/ndf",
                        "max |pull|",
                    ],
                    ["high-Δm pTmiss", "6/7", "0.0029/6", "0.054"],
                    ["high-Δm 60 bins", "29/60", "0.0926/29", "0.173"],
                    ["low-Δm 42 bins", "36/42", "0.1402/36", "0.206"],
                ],
                [65 * mm, 31 * mm, 39 * mm, 34 * mm],
                styles,
                ["LEFT", "CENTER", "CENTER", "CENTER"],
            ),
            Spacer(1, 4 * mm),
            para(
                "A/B event-hash fold closure와 full-mixture MC pseudodata "
                "closure는 통과한다. 이는 implementation과 bookkeeping이 "
                "정상임을 보여주지만 data transfer를 검증하지 않는다.",
                styles["body"],
            ),
            para("2.3 Data residual closure", styles["h2"]),
        ]
    )

    vr_order = [
        ("highdm_nb0", "high-Δm, Nb=0"),
        ("highdm_njet3to4_nb1plus", "high-Δm, Top enriched"),
        ("lowdm_isr200to300", "low-Δm, low ISR"),
        ("lowdm_met250to300", "low-Δm, low pTmiss"),
        ("lowdm_significance7to10", "low-Δm, low MET sig."),
    ]
    tf_rows = [
        ["Validation region", "Raw MC", "Combined TF", "Top/W split", "max |pull|"]
    ]
    for key, label in vr_order:
        record = validation[key]
        combined = record["combined_baseline"]["integrated"]["ratio"]
        split = record["split_top_w"]["integrated"]
        tf_rows.append(
            [
                label,
                fmt(raw_ratio(record)),
                fmt(combined),
                f"{split['ratio']:.3f} ± {split['ratio_uncertainty']:.3f}",
                f"{record['split_top_w']['maximum_absolute_pull']:.2f}",
            ]
        )
    story.extend(
        [
            styled_table(
                tf_rows,
                [57 * mm, 25 * mm, 29 * mm, 38 * mm, 25 * mm],
                styles,
                ["LEFT", "CENTER", "CENTER", "CENTER", "CENTER"],
            ),
            Spacer(1, 4 * mm),
            box(
                "High-Δm에서는 Top/W 분리의 변화가 0.4 percentage point "
                "이하다. Low-Δm low-ISR과 low-pTmiss에서는 raw LL MC가 "
                "data residual과 1-3% 수준에서 맞지만 TF 적용 후 오히려 "
                "0.80과 0.78까지 악화된다.",
                styles["box"],
                LIGHT_RED,
                RED,
            ),
            Spacer(1, 4 * mm),
            paired_figures(
                plots / "tf_highdm_top_enriched_data_closure.png",
                "High-Δm Top-enriched data residual closure. Raw MC, combined "
                "TF 및 Top/W-split TF를 직접 비교한다.",
                2,
                plots / "tf_lowdm_met_data_closure.png",
                "Low-Δm low-pTmiss closure. Raw MC는 data residual과 거의 "
                "맞지만 두 TF 예측은 더 낮아진다.",
                3,
                styles,
            ),
            PageBreak(),
        ]
    )

    control = top_w["control_validations"]["lowdm_nb_groups"]
    story.extend(
        [
            para("2.4 독립 control diagnostic과 uncertainty", styles["h1"]),
            para(
                "Low-Δm의 Nb=1 category는 normalization fit에서 제외했다. "
                "따라서 Nb=0 및 Nb≥2 anchor에서 얻은 μ(Top), μ(W)가 새로운 "
                "b-jet category로 전달되는지 검사하는 독립 control이다.",
                styles["body"],
            ),
            styled_table(
                [
                    ["항목", "결과"],
                    [
                        "Prediction",
                        f"{control['prediction'][1]:.1f} ± "
                        f"{math.sqrt(control['prediction_variance'][1]):.1f}",
                    ],
                    [
                        "Data residual",
                        f"{control['observation'][1]:.1f} ± "
                        f"{math.sqrt(control['observation_variance'][1]):.1f}",
                    ],
                    [
                        "Prediction/residual",
                        f"{control['ratio'][1]:.4f} ± "
                        f"{math.sqrt(control['ratio_variance'][1]):.4f}",
                    ],
                    ["Maximum pull", f"{control['maximum_absolute_pull']:.2f}"],
                ],
                [75 * mm, 94 * mm],
                styles,
                ["LEFT", "CENTER"],
            ),
            Spacer(1, 3 * mm),
            *image_with_caption(
                plots / "tf_lowdm_nb1_control_validation.png",
                "유일한 독립 검정인 low-Δm Nb=1만 표시했다. Nb=0과 Nb≥2는 "
                "fit 입력이므로 closure evidence가 아니다. Held-out Nb=1은 "
                "약 21% 과소예측된다.",
                4,
                styles,
                width=112 * mm,
            ),
            para("포함된 통계 uncertainty", styles["h2"]),
            bullets(
                [
                    "Data count, weighted MC sumw2 및 Other-MC subtraction",
                    "μ(Top), μ(W) covariance와 target bin 간 상관",
                    "독립 A/B fold TF의 통계 covariance",
                ],
                styles["body"],
            ),
            para("미포함 systematic", styles["h2"]),
            bullets(
                [
                    "Lepton trigger/reconstruction/ID/isolation",
                    "b tagging, Nb migration, recoil 및 process modeling",
                    "TT/ST composition, Other-MC modeling, signal contamination",
                ],
                styles["body"],
            ),
            box(
                "<b>방법 1 판정:</b> technical MC closure는 통과하지만 data "
                "residual closure가 실패한다. Top/W 분리는 실패를 해결하지 "
                "못하며 low-Δm에서는 오히려 더 나쁘다. Nominal 채택 불가.",
                styles["decision"],
                LIGHT_RED,
                RED,
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            para("3. 방법 2: Event-level lepton removal", styles["h1"]),
            para(
                "Data 1-lepton event의 observed hadronic system과 process "
                "mixture를 더 직접적으로 사용하기 위해 선택된 lepton의 "
                "transverse momentum을 missing momentum에 더한다. 이후 "
                "Δphi, U_T, high-Δm 60 bins, low-Δm 42 bins와 validation "
                "selection을 event-by-event로 다시 계산한다.",
                styles["body"],
            ),
            box(
                "<b>Event transform:</b> pTmiss(rem) = pTmiss + pT(lepton)<br/>"
                "<b>Residual normalization:</b> α(c) = "
                "Σ zero-lepton truth target(c) / Σ removal source(c)<br/>"
                "<b>Component policy:</b> Top = TT + ST; W independent",
                styles["box"],
                LIGHT_BLUE,
                BLUE,
            ),
            para("3.1 Residual lost/pass normalization", styles["h2"]),
            styled_table(
                [
                    ["Regime", "Component", "α"],
                    [
                        "high-Δm",
                        "Top = TT + ST",
                        f"{normalizations['highdm']['Top']['alpha']:.4f} ± "
                        f"{math.sqrt(normalizations['highdm']['Top']['alpha_variance']):.4f}",
                    ],
                    [
                        "high-Δm",
                        "W",
                        f"{normalizations['highdm']['W']['alpha']:.4f} ± "
                        f"{math.sqrt(normalizations['highdm']['W']['alpha_variance']):.4f}",
                    ],
                    [
                        "low-Δm",
                        "Top = TT + ST",
                        f"{normalizations['lowdm']['Top']['alpha']:.4f} ± "
                        f"{math.sqrt(normalizations['lowdm']['Top']['alpha_variance']):.4f}",
                    ],
                    [
                        "low-Δm",
                        "W",
                        f"{normalizations['lowdm']['W']['alpha']:.4f} ± "
                        f"{math.sqrt(normalizations['lowdm']['W']['alpha_variance']):.4f}",
                    ],
                ],
                [48 * mm, 67 * mm, 54 * mm],
                styles,
                ["LEFT", "LEFT", "CENTER"],
            ),
            Spacer(1, 4 * mm),
            para(
                "Data prediction은 removal-transformed Data 1-lepton source에서 "
                "Other-MC source를 차감한 뒤, fitted Top/W mixture로 계산한 "
                "effective lost/pass factor를 곱한다. α는 적분 target/source "
                "ratio이므로 MC 총수율은 거의 정확히 맞도록 강제된다. 따라서 "
                "shape closure가 핵심 검정이다.",
                styles["body"],
            ),
            PageBreak(),
        ]
    )

    removal_rows = [
        [
            "Distribution",
            "Removal χ²/ndf",
            "Removal max |pull|",
            "TF χ²/ndf",
            "TF max |pull|",
        ]
    ]
    for key, label in (
        ("highdm_search60", "high-Δm 60 bins"),
        ("highdm_ut", "high-Δm U_T"),
        ("lowdm_search42", "low-Δm 42 bins"),
    ):
        current = mc_removal[key]["post"]
        old = mc_removal[key]["old_tf_reference"]
        removal_rows.append(
            [
                label,
                f"{current['chi2']:.1f}/{current['ndf']}",
                f"{current['maximum_absolute_pull']:.2f}",
                f"{old['diagonal_chi2']:.3f}/{old['diagonal_ndf']}",
                f"{old['maximum_absolute_pull']:.2f}",
            ]
        )
    story.extend(
        [
            para("3.2 MC cross-fit shape closure", styles["h1"]),
            styled_table(
                removal_rows,
                [48 * mm, 34 * mm, 33 * mm, 32 * mm, 27 * mm],
                styles,
                ["LEFT", "CENTER", "CENTER", "CENTER", "CENTER"],
            ),
            Spacer(1, 4 * mm),
            box(
                "적분 normalization은 정의상 맞지만 shape는 맞지 않는다. "
                "High-Δm U_T max |pull|은 57.71, low-Δm 42-bin max "
                "|pull|은 45.32다. 이 정도의 discrepancy는 단일 α 또는 "
                "normalization nuisance로 해결할 수 없다.",
                styles["box"],
                LIGHT_RED,
                RED,
            ),
            Spacer(1, 4 * mm),
            paired_figures(
                plots / "removal_mc_highdm_ut.png",
                "High-Δm SR U_T MC-only cross-fit. 검은 점은 data가 아니라 "
                "direct Top/W MC SR reference다. 적분은 reference에 맞췄지만 "
                "bin-by-bin migration shape가 크게 다르다.",
                5,
                plots / "removal_mc_lowdm_search42.png",
                "Low-Δm SR 42-bin MC-only cross-fit. 검은 점은 data가 아니라 "
                "direct Top/W MC SR reference이며, 여러 category에서 "
                "coherent shape nonclosure가 관측된다.",
                6,
                styles,
            ),
            PageBreak(),
        ]
    )

    data_rows = [
        [
            "Validation region",
            "Raw Top+W MC",
            "Removal",
            "Removal max |pull|",
        ]
    ]
    for key, label in (
        ("highdm_vr_nb0", "high-Δm, Nb=0"),
        ("highdm_vr_njet3to4_nb1plus", "high-Δm, Top enriched"),
        ("lowdm_vr_isr200to300", "low-Δm, low ISR"),
        ("lowdm_vr_met250to300", "low-Δm, low pTmiss"),
        ("lowdm_vr_significance7to10", "low-Δm, low MET sig."),
    ):
        record = data_validation[key]
        raw_record = raw_mc_comparison[key]
        data_rows.append(
            [
                label,
                f"{raw_record['raw_mc_integrated_ratio']:.3f}",
                f"{record['ratio']:.3f} ± {record['ratio_uncertainty']:.3f}",
                f"{record['maximum_absolute_pull']:.2f}",
            ]
        )
    story.extend(
        [
            para("3.3 Event-removal data validation", styles["h1"]),
            styled_table(
                data_rows,
                [61 * mm, 31 * mm, 46 * mm, 36 * mm],
                styles,
                ["LEFT", "CENTER", "CENTER", "CENTER"],
            ),
            Spacer(1, 4 * mm),
            para(
                "Data-driven prediction이 raw MC와 다른 것 자체는 문제가 "
                "아니다. 판단 기준은 같은 selection에서 Data-Other MC에 더 "
                "가까워지는지다. High-Δm Top-enriched VR은 적분비가 raw "
                "0.741에서 removal 0.807로 개선되지만 max |pull|은 "
                "14.83에서 23.42로 악화된다. Low-pTmiss는 raw 0.993에서 "
                "0.367, low-ISR은 raw 1.026에서 0.542로 크게 악화된다.",
                styles["body"],
            ),
            paired_figures(
                plots / "removal_highdm_top_enriched_data_closure.png",
                "High-Δm Top-enriched VR. 같은 selection에서 raw Top+W MC와 "
                "event-removal prediction을 Data-Other MC에 직접 비교한다. "
                "적분비는 개선되지만 shape pull은 악화된다.",
                7,
                plots / "removal_lowdm_met_data_closure.png",
                "Low-Δm low-pTmiss VR. Raw MC는 data residual과 거의 "
                "일치하지만 removal prediction은 약 37%에 그친다.",
                8,
                styles,
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            para("3.4 Removal failure의 물리적 해석", styles["h1"]),
            para(
                "단일 four-vector removal은 모든 lost-lepton event를 lepton이 "
                "완전히 invisible했던 event처럼 취급한다. 실제 background에는 "
                "acceptance loss, reconstruction failure, ID/isolation failure, "
                "hadronic tau가 섞여 있다.",
                styles["body"],
            ),
            styled_table(
                [
                    ["Loss mode", "합리적인 detector response"],
                    [
                        "Acceptance / reconstruction loss",
                        "Removal-like invisible response가 근사적으로 가능",
                    ],
                    [
                        "ID / isolation failure",
                        "Lepton PF momentum이 남을 수 있으므로 full removal 부적절",
                    ],
                    [
                        "Hadronic tau",
                        "Visible tau decay와 neutrino response를 별도 모델링",
                    ],
                ],
                [66 * mm, 103 * mm],
                styles,
                ["LEFT", "LEFT"],
            ),
            Spacer(1, 4 * mm),
            box(
                "<b>Input materialization limitation:</b> 현재 flat "
                "preselection은 ordinary 1-lepton event에 대해 원래 "
                "pTmiss > 250 GeV를 요구한다. Removal 이후 threshold 위로 "
                "이동할 원래 pTmiss < 250 GeV event가 입력에 없으므로 "
                "특히 low-Δm threshold region에 downward bias가 생길 수 있다.",
                styles["box"],
                LIGHT_RED,
                RED,
            ),
            para("포함된 prototype uncertainty", styles["h2"]),
            bullets(
                [
                    "Data/Other-MC removal source statistics",
                    "Top/W removal-source MC sumw2와 α variance",
                    "Fitted μ(Top), μ(W) covariance",
                    "Zero-lepton validation residual statistics",
                ],
                styles["body"],
            ),
            para("중요한 미포함 항목", styles["h2"]),
            bullets(
                [
                    "Absolute lepton-loss efficiency 및 detector/model systematics",
                    "α와 같은 source histogram의 상관",
                    "Loss-mode별 response, tau response, threshold migration",
                ],
                styles["body"],
            ),
            box(
                "<b>방법 2 판정:</b> Event migration 구현은 완료됐지만 MC "
                "shape closure와 data VR closure가 모두 실패한다. 현재 "
                "monolithic removal estimator는 nominal 채택 불가.",
                styles["decision"],
                LIGHT_RED,
                RED,
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            para("4. 두 방법의 직접 비교와 다음 단계", styles["h1"]),
            styled_table(
                [
                    ["판정 항목", "Transfer factor", "Event removal"],
                    ["Top 정책", "TT + ST", "TT + ST"],
                    ["W 독립 성분", "예", "예"],
                    ["MC technical closure", "통과", "Shape 실패"],
                    ["Data residual closure", "실패", "실패"],
                    ["Raw MC 대비", "Low-Δm에서 악화", "더 크게 악화"],
                    [
                        "핵심 failure",
                        "CR correction의 target 비이식성",
                        "Universal response와 migration 입력 손실",
                    ],
                    ["Nominal 채택", "불가", "불가"],
                ],
                [53 * mm, 58 * mm, 58 * mm],
                styles,
                ["LEFT", "LEFT", "LEFT"],
            ),
            Spacer(1, 5 * mm),
            para("권고하는 hybrid efficiency/response embedding", styles["h2"]),
            bullets(
                [
                    "Data electron 및 muon 1-lepton source를 분리한다.",
                    "Top은 TT+ST로 함께 움직이고 W만 독립으로 유지한다.",
                    "Acceptance, reconstruction, ID, isolation loss 확률을 "
                    "pT(lepton), eta, Nj, Nb, HT의 함수로 측정한다.",
                    "Acceptance/reconstruction loss에만 removal-like response를 "
                    "사용하고 ID/isolation loss에는 PF momentum을 유지하는 "
                    "response template를 사용한다.",
                    "Hadronic tau를 별도 auxiliary component로 둔다.",
                    "원래 MET가 아니라 1-lepton recoil 기준으로 source를 "
                    "materialize하여 below-threshold migration을 보존한다.",
                    "MC truth, flavor split, held-out Nb category와 다섯 data VR을 "
                    "모두 adoption gate로 사용한다.",
                ],
                styles["body"],
            ),
            box(
                "<b>최종 분석 결정:</b><br/>"
                "1. Combined 및 Top/W-split TF를 nominal에서 폐기한다.<br/>"
                "2. 현재 monolithic event-removal estimator도 폐기한다.<br/>"
                "3. 두 구현과 JSON/plot은 failed-method diagnostic으로 보존한다.<br/>"
                "4. Raw LL MC는 임시 reference일 뿐 validated final estimate가 아니다.<br/>"
                "5. 새 hybrid estimator가 독립 validation에서 raw MC보다 "
                "개선될 때만 nominal method로 채택한다.",
                styles["decision"],
                LIGHT_RED,
                RED,
            ),
            Spacer(1, 5 * mm),
            para("재현성", styles["h2"]),
            bullets(
                [
                    "Transfer-factor result: "
                    "lost_lepton_top_w_split_closure_2024_20260728/top_w_closure.json",
                    "Transfer-factor MC closure: "
                    "lost_lepton_closure_2024_20260728/mc_closure.json",
                    "Removal result: "
                    "lost_lepton_removal_closure_2024_20260728/"
                    "removal_closure_results.json",
                    "Removal run manifest: "
                    "lost_lepton_removal_closure_2024_20260728/inputs/"
                    "run_manifest.json",
                    "Selection authority: autonomous_allhad/"
                    "autonomous_allhad/real_subset_worker.py",
                ],
                styles["small"],
            ),
        ]
    )

    document.build(
        story,
        onFirstPage=page_header_footer,
        onLaterPages=page_header_footer,
    )


def build_html(
    output_path: Path,
    top_w: dict[str, Any],
    removal: dict[str, Any],
    raw_mc_comparison: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validation = top_w["validation_regions"]
    removal_validation = removal["data_validation"]
    rows = []
    for removal_key, label in (
        ("highdm_vr_nb0", "high-Δm, Nb=0"),
        ("highdm_vr_njet3to4_nb1plus", "high-Δm, Top enriched"),
        ("lowdm_vr_isr200to300", "low-Δm, low ISR"),
        ("lowdm_vr_met250to300", "low-Δm, low pTmiss"),
        ("lowdm_vr_significance7to10", "low-Δm, low MET significance"),
    ):
        removal_ratio = removal_validation[removal_key]["ratio"]
        raw_record = raw_mc_comparison["records"][removal_key]
        rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{raw_record['raw_mc_integrated_ratio']:.3f}</td>"
            f"<td>{removal_ratio:.3f}</td>"
            f"<td>{raw_record['removal_maximum_absolute_pull']:.2f}</td>"
            "</tr>"
        )
    images = [
        ("tf_top_w_scale_factors.png", "Top/W fitted scale factors"),
        (
            "tf_highdm_top_enriched_data_closure.png",
            "High-Δm: raw MC vs combined TF vs Top/W split",
        ),
        (
            "tf_lowdm_met_data_closure.png",
            "Low-Δm low-pTmiss: raw MC vs two TF estimates",
        ),
        (
            "tf_lowdm_nb1_control_validation.png",
            "Held-out low-Δm Nb=1 only; fit anchors are excluded",
        ),
        ("removal_mc_highdm_ut.png", "Removal MC high-Δm U_T closure"),
        (
            "removal_mc_lowdm_search42.png",
            "Removal MC low-Δm search-bin closure",
        ),
        (
            "removal_highdm_top_enriched_data_closure.png",
            "Removal high-Δm data closure",
        ),
        (
            "removal_lowdm_met_data_closure.png",
            "Removal low-Δm low-pTmiss closure",
        ),
    ]
    gallery = "".join(
        f'<figure><a href="plots/{name}"><img src="plots/{name}"></a>'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
        for name, caption in images
    )
    document = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>2024 Lost-lepton 두 방법 비교</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 1250px; margin: 2rem auto; padding: 0 1rem; color: #202124; }}
h1,h2 {{ color: #17365d; }}
.decision {{ background:#f8e7ea; border-left:5px solid #a51c30; padding:1rem; }}
.note {{ background:#eaf2f8; border-left:5px solid #2f6fae; padding:1rem; }}
table {{ border-collapse:collapse; width:100%; margin:1rem 0 2rem; }}
th,td {{ border:1px solid #b8bec6; padding:.55rem; text-align:left; }}
th {{ background:#17365d; color:white; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:1rem; }}
figure {{ margin:0; border:1px solid #ccd2d8; padding:.5rem; }}
figure img {{ width:100%; }}
figcaption {{ color:#5f6368; font-size:.9rem; }}
a {{ color:#175ea8; }}
</style>
</head>
<body>
<h1>2024 Lost-lepton 배경추정 두 방법의 상세 비교</h1>
<p class="decision"><b>최종 판정:</b> MC-derived transfer factor와 현재의 monolithic event-level lepton-removal은 모두 nominal 배경추정에 채택하지 않는다.</p>
<p class="note"><b>공통 정책:</b> Top = TT + ST이며 W만 독립 성분이다. 두 연구 모두 1,153개 ROOT와 361,054,245개 event를 처리했고 nominal 중간산출물은 수정하지 않았다.</p>
<h2>문서</h2>
<ul>
<li><a href="output/pdf/lost_lepton_two_method_report_2024_ko.pdf">그림 포함 PDF 보고서</a></li>
<li><a href="detailed_report_ko.md">상세 Markdown 원문</a></li>
<li><a href="source_manifest.json">입력 파일 및 SHA-256 manifest</a></li>
</ul>
<h2>Data validation 요약</h2>
<table>
<thead><tr><th>Validation region</th><th>Raw Top+W MC</th><th>Removal</th><th>Removal max |pull|</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
<h2>핵심 차이</h2>
<ul>
<li>Transfer factor: MC technical closure는 통과하지만 data residual closure가 실패한다.</li>
<li>Event removal: 적분 normalization 후에도 MC U_T/search-bin shape closure가 실패한다.</li>
<li>두 방법 모두 low-Δm의 잘 맞는 raw MC를 악화시키므로 nominal central value로 사용할 수 없다.</li>
</ul>
<h2>Figures</h2>
<div class="grid">{gallery}</div>
</body>
</html>
"""
    output_path.write_text(document)


def build_manifest(
    output_path: Path,
    inputs: dict[str, Path],
    output_pdf: Path,
) -> None:
    payload = {
        "schema_version": "lost_lepton_two_method_report_2024_v1",
        "status": "complete",
        "component_policy": {
            "Top": ["TT", "ST"],
            "W": ["WtoLNu"],
            "independent_components": ["Top", "W"],
        },
        "inputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256(path),
            }
            for name, path in inputs.items()
        },
        "outputs": {
            "pdf": str(output_pdf.resolve()),
            "pdf_sha256": sha256(output_pdf),
        },
        "nominal_intermediates_modified": False,
        "selection_authority": (
            "autonomous_allhad/autonomous_allhad/real_subset_worker.py"
        ),
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--top-w", type=Path, required=True)
    parser.add_argument("--tf-mc", type=Path, required=True)
    parser.add_argument("--removal", type=Path, required=True)
    parser.add_argument("--removal-manifest", type=Path, required=True)
    args = parser.parse_args()

    report_dir = args.report_dir.resolve()
    top_w = load_json(args.top_w)
    removal = load_json(args.removal)
    removal_raw_mc = load_json(
        report_dir / "removal_raw_mc_comparison.json"
    )
    validate_component_policy(top_w, removal)
    register_fonts()

    output_pdf = (
        report_dir
        / "output"
        / "pdf"
        / "lost_lepton_two_method_report_2024_ko.pdf"
    )
    build_pdf(output_pdf, report_dir, top_w, removal)
    build_html(
        report_dir / "index.html",
        top_w,
        removal,
        removal_raw_mc,
    )
    build_manifest(
        report_dir / "source_manifest.json",
        {
            "top_w_closure": args.top_w.resolve(),
            "tf_mc_closure": args.tf_mc.resolve(),
            "removal_closure": args.removal.resolve(),
            "removal_run_manifest": args.removal_manifest.resolve(),
            "removal_raw_mc_comparison": (
                report_dir / "removal_raw_mc_comparison.json"
            ).resolve(),
        },
        output_pdf,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "pdf": str(output_pdf),
                "html": str(report_dir / "index.html"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
