#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

from reportlab.lib.colors import Color, HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


PAGE_WIDTH = 960.0
PAGE_HEIGHT = 540.0

INK = HexColor("#202124")
MUTED = HexColor("#5F6368")
LINE = HexColor("#9AA0A6")
SOFT_LINE = HexColor("#DADCE0")
BLUE = HexColor("#3F6F9F")
BLUE_FILL = HexColor("#EAF2FA")
ORANGE = HexColor("#D97818")
ORANGE_FILL = HexColor("#FFF0DF")
GREEN = HexColor("#4D8C57")
GREEN_FILL = HexColor("#E8F3E8")
AMBER = HexColor("#A56A00")
AMBER_FILL = HexColor("#FFF3D6")
PURPLE = HexColor("#7A5A91")
PURPLE_FILL = HexColor("#F1ECF7")
GREY_FILL = HexColor("#F5F6F7")
WHITE = HexColor("#FFFFFF")


def register_fonts() -> None:
    regular_candidates = (
        Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
        Path("/Library/Fonts/Arial Unicode.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    )
    bold_candidates = (
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    )
    regular = next(path for path in regular_candidates if path.is_file())
    bold = next(path for path in bold_candidates if path.is_file())
    pdfmetrics.registerFont(TTFont("ANArial", str(regular)))
    pdfmetrics.registerFont(TTFont("ANArialBold", str(bold)))
    pdfmetrics.registerFontFamily(
        "ANArial",
        normal="ANArial",
        bold="ANArialBold",
        italic="ANArial",
        boldItalic="ANArialBold",
    )


def paragraph(
    drawing: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    size: float = 11.0,
    leading: float | None = None,
    color: Color = INK,
    alignment: int = TA_LEFT,
    font: str = "ANArial",
) -> None:
    style = ParagraphStyle(
        name=f"schematic-{x}-{y}-{size}",
        fontName=font,
        fontSize=size,
        leading=leading or 1.25 * size,
        textColor=color,
        alignment=alignment,
        spaceAfter=0,
        spaceBefore=0,
    )
    item = Paragraph(text, style)
    item.wrapOn(drawing, width, height)
    item.drawOn(drawing, x, y)


def box(
    drawing: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: Color = WHITE,
    stroke: Color = SOFT_LINE,
    radius: float = 8.0,
    line_width: float = 1.2,
) -> None:
    drawing.setFillColor(fill)
    drawing.setStrokeColor(stroke)
    drawing.setLineWidth(line_width)
    drawing.roundRect(x, y, width, height, radius, stroke=1, fill=1)


def arrow(
    drawing: canvas.Canvas,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: Color = LINE,
    width: float = 1.8,
    dashed: bool = False,
) -> None:
    drawing.saveState()
    drawing.setStrokeColor(color)
    drawing.setFillColor(color)
    drawing.setLineWidth(width)
    if dashed:
        drawing.setDash(5, 4)
    drawing.line(x1, y1, x2, y2)
    angle = math.atan2(y2 - y1, x2 - x1)
    head = 7.0
    spread = math.radians(27)
    points = [
        (x2, y2),
        (
            x2 - head * math.cos(angle - spread),
            y2 - head * math.sin(angle - spread),
        ),
        (
            x2 - head * math.cos(angle + spread),
            y2 - head * math.sin(angle + spread),
        ),
    ]
    path = drawing.beginPath()
    path.moveTo(*points[0])
    path.lineTo(*points[1])
    path.lineTo(*points[2])
    path.close()
    drawing.drawPath(path, stroke=0, fill=1)
    drawing.restoreState()


def cell(
    drawing: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    letter: str,
    title: str,
    detail: str,
    fill: Color,
    accent: Color,
) -> None:
    box(
        drawing,
        x,
        y,
        width,
        height,
        fill=fill,
        stroke=accent,
        radius=5,
        line_width=1.4,
    )
    drawing.setFillColor(accent)
    drawing.circle(x + 20, y + height - 20, 12, stroke=0, fill=1)
    drawing.setFillColor(WHITE)
    drawing.setFont("ANArialBold", 12)
    drawing.drawCentredString(x + 20, y + height - 24, letter)
    paragraph(
        drawing,
        f"<b>{title}</b>",
        x + 38,
        y + height - 33,
        width - 46,
        22,
        size=11.0,
        font="ANArial",
    )
    paragraph(
        drawing,
        detail,
        x + 14,
        y + 15,
        width - 28,
        height - 52,
        size=9.4,
        color=INK,
        alignment=TA_CENTER,
    )


def draw(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    drawing = canvas.Canvas(
        str(output),
        pagesize=(PAGE_WIDTH, PAGE_HEIGHT),
        pageCompression=1,
    )
    drawing.setTitle("Fake-photon background estimation schematic")
    drawing.setAuthor("CMS Run-3 all-hadronic stop analysis")
    drawing.setFillColor(WHITE)
    drawing.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, stroke=0, fill=1)

    paragraph(
        drawing,
        "<b>Fake-photon background estimation</b>",
        32,
        501,
        650,
        28,
        size=20.0,
        font="ANArial",
    )
    paragraph(
        drawing,
        "Data-driven transfer factor in the photon control region (GCR)",
        32,
        482,
        650,
        20,
        size=10.5,
        color=MUTED,
    )
    paragraph(
        drawing,
        "<b>Common GCR event selection</b> - only the photon-ID axes change",
        613,
        488,
        315,
        18,
        size=9.4,
        color=MUTED,
        alignment=TA_RIGHT,
    )

    # Inputs.
    box(drawing, 32, 333, 155, 128, fill=GREY_FILL, stroke=SOFT_LINE)
    paragraph(
        drawing,
        "<b>Inputs</b>",
        47,
        433,
        125,
        22,
        size=12,
        font="ANArial",
    )
    drawing.setFillColor(INK)
    drawing.circle(51, 411, 4.5, stroke=0, fill=1)
    paragraph(
        drawing,
        "<b>EGamma data</b>",
        63,
        401,
        108,
        20,
        size=10.2,
    )
    drawing.setFillColor(BLUE)
    drawing.rect(46.5, 380.5, 9, 9, stroke=0, fill=1)
    paragraph(
        drawing,
        "prompt γ MC",
        63,
        374,
        108,
        20,
        size=10.2,
    )
    drawing.setFillColor(ORANGE)
    drawing.rect(46.5, 352.5, 9, 9, stroke=0, fill=1)
    paragraph(
        drawing,
        "e → γ MC",
        63,
        346,
        108,
        20,
        size=10.2,
    )
    arrow(drawing, 187, 397, 212, 397, color=LINE)

    # ABCD photon-ID plane.
    grid_x = 260.0
    grid_y = 233.0
    cell_w = 152.0
    cell_h = 104.0
    gap = 8.0
    paragraph(
        drawing,
        "<b>Charged isolation</b>",
        grid_x,
        450,
        2 * cell_w + gap,
        20,
        size=11,
        alignment=TA_CENTER,
    )
    paragraph(
        drawing,
        "pass medium WP",
        grid_x,
        431,
        cell_w,
        18,
        size=9.3,
        color=MUTED,
        alignment=TA_CENTER,
    )
    paragraph(
        drawing,
        "fail medium WP",
        grid_x + cell_w + gap,
        431,
        cell_w,
        18,
        size=9.3,
        color=MUTED,
        alignment=TA_CENTER,
    )
    paragraph(
        drawing,
        "<b>σ<sub>iηiη</sub></b><br/>pass medium WP",
        201,
        grid_y + cell_h + gap + 27,
        52,
        54,
        size=9.2,
        alignment=TA_CENTER,
    )
    paragraph(
        drawing,
        "<b>σ<sub>iηiη</sub></b><br/>fail medium WP",
        201,
        grid_y + 27,
        52,
        54,
        size=9.2,
        alignment=TA_CENTER,
    )
    cell(
        drawing,
        grid_x,
        grid_y + cell_h + gap,
        cell_w,
        cell_h,
        letter="A",
        title="Target",
        detail="<b>medium photon</b><br/>data used only for validation",
        fill=GREEN_FILL,
        accent=GREEN,
    )
    cell(
        drawing,
        grid_x + cell_w + gap,
        grid_y + cell_h + gap,
        cell_w,
        cell_h,
        letter="B",
        title="Application",
        detail="charged-isolation sideband<br/>shape in every GCR observable",
        fill=AMBER_FILL,
        accent=AMBER,
    )
    cell(
        drawing,
        grid_x,
        grid_y,
        cell_w,
        cell_h,
        letter="C",
        title="Measurement pass",
        detail="fake-factor numerator<br/>fail σ<sub>iηiη</sub>, pass isolation",
        fill=BLUE_FILL,
        accent=BLUE,
    )
    cell(
        drawing,
        grid_x + cell_w + gap,
        grid_y,
        cell_w,
        cell_h,
        letter="D",
        title="Measurement fail",
        detail="fake-factor denominator<br/>fail σ<sub>iηiη</sub>, fail isolation",
        fill=PURPLE_FILL,
        accent=PURPLE,
    )

    # Computation panel.
    panel_x = 606.0
    panel_w = 322.0
    box(drawing, panel_x, 203, panel_w, 258, fill=WHITE, stroke=SOFT_LINE)
    drawing.setFillColor(BLUE)
    drawing.circle(panel_x + 22, 431, 11, stroke=0, fill=1)
    drawing.setFillColor(WHITE)
    drawing.setFont("ANArialBold", 10)
    drawing.drawCentredString(panel_x + 22, 427.5, "1")
    paragraph(
        drawing,
        "<b>Subtract prompt contamination</b>",
        panel_x + 41,
        419,
        panel_w - 55,
        24,
        size=11.2,
    )
    paragraph(
        drawing,
        "For X ∈ {B, C, D}:",
        panel_x + 18,
        394,
        panel_w - 36,
        20,
        size=9.2,
        color=MUTED,
    )
    paragraph(
        drawing,
        "N<super>fake</super><sub>X</sub> = "
        "N<super>data</super><sub>X</sub> - "
        "N<super>prompt γ, MC</super><sub>X</sub> - "
        "N<super>e→γ, MC</super><sub>X</sub>",
        panel_x + 18,
        366,
        panel_w - 36,
        28,
        size=10.0,
        alignment=TA_CENTER,
    )
    drawing.setStrokeColor(SOFT_LINE)
    drawing.setLineWidth(1)
    drawing.line(panel_x + 18, 358, panel_x + panel_w - 18, 358)

    drawing.setFillColor(BLUE)
    drawing.circle(panel_x + 22, 336, 11, stroke=0, fill=1)
    drawing.setFillColor(WHITE)
    drawing.setFont("ANArialBold", 10)
    drawing.drawCentredString(panel_x + 22, 332.5, "2")
    paragraph(
        drawing,
        "<b>Measure the fake factor</b>",
        panel_x + 41,
        324,
        panel_w - 55,
        24,
        size=11.2,
    )
    paragraph(
        drawing,
        "f<sub>fake</sub>(k) = "
        "N<super>fake</super><sub>C,k</sub> / "
        "N<super>fake</super><sub>D,k</sub>, "
        "&nbsp;&nbsp;k = (EB/EE, p<super>γ</super><sub>T</sub> bin)",
        panel_x + 18,
        292,
        panel_w - 36,
        33,
        size=10.0,
        alignment=TA_CENTER,
    )
    drawing.setStrokeColor(SOFT_LINE)
    drawing.line(panel_x + 18, 282, panel_x + panel_w - 18, 282)

    drawing.setFillColor(GREEN)
    drawing.circle(panel_x + 22, 260, 11, stroke=0, fill=1)
    drawing.setFillColor(WHITE)
    drawing.setFont("ANArialBold", 10)
    drawing.drawCentredString(panel_x + 22, 256.5, "3")
    paragraph(
        drawing,
        "<b>Predict the target-region fake yield</b>",
        panel_x + 41,
        248,
        panel_w - 55,
        24,
        size=11.2,
    )
    paragraph(
        drawing,
        "N<super>fake,pred</super><sub>A,i</sub> = "
        "Σ<sub>k</sub> f<sub>fake</sub>(k) "
        "N<super>fake</super><sub>B,i,k</sub>",
        panel_x + 18,
        211,
        panel_w - 36,
        27,
        size=10.4,
        alignment=TA_CENTER,
    )

    # Flow cues: C/D determine f and B supplies the application shapes.
    arrow(
        drawing,
        grid_x + 2 * cell_w + gap + 3,
        grid_y + 0.48 * cell_h,
        panel_x - 7,
        327,
        color=BLUE,
        width=1.6,
    )
    arrow(
        drawing,
        grid_x + 2 * cell_w + gap + 3,
        grid_y + cell_h + gap + 0.48 * cell_h,
        panel_x - 7,
        262,
        color=AMBER,
        width=1.6,
    )
    # Bottom validation and output strip.
    strip_y = 36.0
    strip_h = 158.0
    col_gap = 12.0
    col_w = (PAGE_WIDTH - 64 - 2 * col_gap) / 3.0
    box(drawing, 32, strip_y, col_w, strip_h, fill=GREY_FILL, stroke=SOFT_LINE)
    paragraph(
        drawing,
        "<b>Output</b>",
        48,
        166,
        col_w - 32,
        20,
        size=11.5,
    )
    paragraph(
        drawing,
        "Data-driven fake yield in every available GCR distribution:",
        48,
        135,
        col_w - 32,
        34,
        size=9.5,
    )
    paragraph(
        drawing,
        "<b>U<sub>T</sub>, H<sub>T</sub>, p<super>miss</super><sub>T</sub>, "
        "N<sub>j</sub>, N<sub>b</sub>, N<sub>top</sub>, …</b>",
        48,
        105,
        col_w - 32,
        27,
        size=10.2,
        color=GREEN,
        alignment=TA_CENTER,
    )
    paragraph(
        drawing,
        "Stored separately, then injected downstream without modifying "
        "the nominal intermediate.",
        48,
        55,
        col_w - 32,
        46,
        size=8.9,
        color=MUTED,
    )

    x2 = 32 + col_w + col_gap
    box(drawing, x2, strip_y, col_w, strip_h, fill=GREY_FILL, stroke=SOFT_LINE)
    paragraph(
        drawing,
        "<b>QCD closure test</b>",
        x2 + 16,
        166,
        col_w - 32,
        20,
        size=11.5,
    )
    paragraph(
        drawing,
        "Repeat C/D → B → A using truth-matched fake photons in QCD MC.",
        x2 + 16,
        126,
        col_w - 32,
        42,
        size=9.5,
    )
    paragraph(
        drawing,
        "closure difference → nonclosure uncertainty",
        x2 + 16,
        94,
        col_w - 32,
        26,
        size=9.5,
        color=PURPLE,
        alignment=TA_CENTER,
    )
    paragraph(
        drawing,
        "<b>QCD MC validates the method; it is not added as the nominal "
        "fake prediction.</b>",
        x2 + 16,
        51,
        col_w - 32,
        43,
        size=8.9,
        color=INK,
    )

    x3 = x2 + col_w + col_gap
    box(drawing, x3, strip_y, col_w, strip_h, fill=GREY_FILL, stroke=SOFT_LINE)
    paragraph(
        drawing,
        "<b>Uncertainties</b>",
        x3 + 16,
        166,
        col_w - 32,
        20,
        size=11.5,
    )
    paragraph(
        drawing,
        "• fake-factor statistics<br/>"
        "• prompt-photon normalization (±30%)<br/>"
        "• electron misidentification (±50%)<br/>"
        "• QCD nonclosure",
        x3 + 20,
        73,
        col_w - 40,
        89,
        size=9.4,
        leading=18,
    )
    paragraph(
        drawing,
        "A-region data are never used to fit the prediction.",
        x3 + 16,
        48,
        col_w - 32,
        24,
        size=8.9,
        color=GREEN,
        alignment=TA_CENTER,
    )

    drawing.showPage()
    drawing.save()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create the AN fake-photon background-estimation schematic."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/pdf/fake_photon_background_estimation_schematic.pdf"),
    )
    args = parser.parse_args()
    register_fonts()
    draw(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
