#!/usr/bin/env python3
"""Build the paper-style physics report for the analysis scale-factor campaign."""

from __future__ import annotations

import gzip
import argparse
import hashlib
import json
import math
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    LongTable,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


REPO = Path(__file__).resolve().parents[2]
REPORT_YEAR = "2024"
OUTPUT = REPO / "output/pdf/analysis_sf_measurements_2024_physics_report.pdf"

MET_DIR = REPO / "autonomous_allhad/workflow/met_trigger_measurement"
PHOTON_DIR = REPO / "autonomous_allhad/workflow/photon_trigger_measurement"
ELECTRON_DIR = REPO / "autonomous_allhad/workflow/lowpt_electron_measurement"
MUON_DIR = REPO / "autonomous_allhad/workflow/lowpt_muon_measurement"

PATHS = {
    "met_config": MET_DIR / "config_2024.json",
    "met_result": MET_DIR / "outputs/2024_full/met_trigger_result_adopted.json",
    "photon_config": PHOTON_DIR / "config_2024.json",
    "photon_result": PHOTON_DIR / "outputs/2024_full/photon_trigger_result_adopted.json",
    "electron_config": ELECTRON_DIR / "config_2024.json",
    "electron_result": ELECTRON_DIR / "outputs/2024_full/adopted_result.json",
    "electron_hist": ELECTRON_DIR / "outputs/2024_full/histograms.json",
    "electron_skips": ELECTRON_DIR / "outputs/2024_full/permanent_skips.json",
    "muon_config": MUON_DIR / "config_2024.json",
    "muon_result": MUON_DIR / "outputs/2024_full/adopted_result.json",
    "muon_hist": MUON_DIR / "outputs/2024_full/histograms.json",
    "muon_skips": MUON_DIR / "outputs/2024_full/permanent_skips.json",
    "integration": REPO / "autonomous_allhad/workflow/analysis_sf_integration_validation/summary.json",
    "impact": REPO / "autonomous_allhad/workflow/lowpt_sf_integration_validation/llcr_lowpt_lepton_impact_estimate_2024.json.gz",
    "met_payload": REPO / "analysis/data/AnalysisSF/2024/met_trigger_sf.json.gz",
    "photon_payload": REPO / "analysis/data/AnalysisSF/2024/photon_trigger_sf.json.gz",
    "electron_payload": REPO / "analysis/data/AnalysisSF/2024/veto_electron_5to10_sf.json.gz",
    "muon_payload": REPO / "analysis/data/AnalysisSF/2024/loose_muon_5to10_sf.json.gz",
}

BLUE = colors.HexColor("#155A8A")
DARK_BLUE = colors.HexColor("#103B5C")
PALE_BLUE = colors.HexColor("#EAF3F8")
PALE_GREY = colors.HexColor("#F3F5F6")
MID_GREY = colors.HexColor("#D5DADF")
DARK_GREY = colors.HexColor("#343A40")
ORANGE = colors.HexColor("#C66818")
GREEN = colors.HexColor("#287A52")
RED = colors.HexColor("#A63D40")


def load_json(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


DATA: dict[str, dict[str, Any]] = {}


def register_fonts() -> tuple[str, str, str]:
    candidates = [
        (
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial Italic.ttf"),
        ),
        (
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"),
        ),
    ]
    for regular, bold, italic in candidates:
        if regular.exists() and bold.exists() and italic.exists():
            pdfmetrics.registerFont(TTFont("ReportSans", str(regular)))
            pdfmetrics.registerFont(TTFont("ReportSans-Bold", str(bold)))
            pdfmetrics.registerFont(TTFont("ReportSans-Italic", str(italic)))
            return "ReportSans", "ReportSans-Bold", "ReportSans-Italic"
    return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"


FONT, FONT_BOLD, FONT_ITALIC = register_fonts()


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=9.4,
            leading=13.2,
            textColor=DARK_GREY,
            alignment=TA_JUSTIFY,
            spaceAfter=5,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=7.3,
            leading=9.6,
            textColor=DARK_GREY,
        ),
        "tiny": ParagraphStyle(
            "Tiny",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=6.2,
            leading=7.5,
            textColor=DARK_GREY,
        ),
        "h1": ParagraphStyle(
            "Heading1",
            parent=base["Heading1"],
            fontName=FONT_BOLD,
            fontSize=17,
            leading=20,
            textColor=DARK_BLUE,
            spaceBefore=4,
            spaceAfter=10,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "Heading2",
            parent=base["Heading2"],
            fontName=FONT_BOLD,
            fontSize=12.2,
            leading=15,
            textColor=BLUE,
            spaceBefore=9,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "Heading3",
            parent=base["Heading3"],
            fontName=FONT_BOLD,
            fontSize=10,
            leading=12,
            textColor=DARK_GREY,
            spaceBefore=7,
            spaceAfter=3,
            keepWithNext=True,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=7.4,
            leading=9.6,
            textColor=DARK_GREY,
            spaceBefore=3,
            spaceAfter=8,
            alignment=TA_JUSTIFY,
        ),
        "equation": ParagraphStyle(
            "Equation",
            parent=base["BodyText"],
            fontName="Courier",
            fontSize=8.4,
            leading=11,
            leftIndent=12,
            rightIndent=12,
            borderColor=MID_GREY,
            borderWidth=0.5,
            borderPadding=6,
            backColor=PALE_GREY,
            spaceBefore=5,
            spaceAfter=7,
        ),
        "cover_title": ParagraphStyle(
            "CoverTitle",
            fontName=FONT_BOLD,
            fontSize=28,
            leading=32,
            textColor=DARK_BLUE,
            alignment=TA_LEFT,
        ),
        "cover_sub": ParagraphStyle(
            "CoverSub",
            fontName=FONT,
            fontSize=13,
            leading=18,
            textColor=BLUE,
        ),
        "toc": ParagraphStyle(
            "TOC",
            fontName=FONT,
            fontSize=9,
            leading=13,
            leftIndent=10,
            firstLineIndent=-10,
            textColor=DARK_GREY,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName=FONT,
            fontSize=9.1,
            leading=12.5,
            textColor=DARK_GREY,
            leftIndent=14,
            firstLineIndent=-7,
            bulletIndent=2,
            spaceAfter=3,
        ),
    }


S = styles()


class PhysicsReportDoc(BaseDocTemplate):
    def __init__(self, filename: str):
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=18 * mm,
            rightMargin=18 * mm,
            topMargin=20 * mm,
            bottomMargin=17 * mm,
            title=f"Measurement and Application of {REPORT_YEAR} Trigger and Low-pT Lepton Scale Factors",
            author="Run-3 all-hadronic stop analysis",
            subject="Physics methods, datasets, uncertainties, results, and histogram integration",
        )
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="body")
        self.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=self._draw_page))

    def _draw_page(self, canvas, doc):
        page = canvas.getPageNumber()
        canvas.saveState()
        if page > 1:
            canvas.setStrokeColor(MID_GREY)
            canvas.setLineWidth(0.45)
            canvas.line(self.leftMargin, A4[1] - 12 * mm, A4[0] - self.rightMargin, A4[1] - 12 * mm)
            canvas.setFont(FONT_BOLD, 7.2)
            canvas.setFillColor(DARK_BLUE)
            canvas.drawString(self.leftMargin, A4[1] - 9.5 * mm, "CMS Run-3 all-hadronic stop analysis")
            canvas.setFont(FONT, 7.2)
            canvas.setFillColor(DARK_GREY)
            canvas.drawRightString(A4[0] - self.rightMargin, A4[1] - 9.5 * mm, f"{REPORT_YEAR} analysis scale factors")
            canvas.setStrokeColor(MID_GREY)
            canvas.line(self.leftMargin, 10.5 * mm, A4[0] - self.rightMargin, 10.5 * mm)
            canvas.setFont(FONT, 7.2)
            canvas.drawString(self.leftMargin, 7.3 * mm, "Physics report - generated from adopted machine-readable results")
            canvas.drawRightString(A4[0] - self.rightMargin, 7.3 * mm, str(page))
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name in {"Heading1", "Heading2"}:
            level = 0 if flowable.style.name == "Heading1" else 1
            text = flowable.getPlainText()
            key = f"section-{level}-{self.page}-{abs(hash(text))}"
            self.canv.bookmarkPage(key)
            if level == 0:
                self.canv.addOutlineEntry(text, key, level=0, closed=False)
            self.notify("TOCEntry", (level, text, self.page, key))


def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, S[style])


def H1(text: str) -> Paragraph:
    return P(text, "h1")


def H2(text: str) -> Paragraph:
    return P(text, "h2")


def H3(text: str) -> Paragraph:
    return P(text, "h3")


def bullet(text: str) -> Paragraph:
    return Paragraph(text, S["bullet"], bulletText="-")


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "--"
    number = float(value)
    if not math.isfinite(number):
        return "--"
    if number == 0:
        return "0"
    if abs(number) < 1e-3 or abs(number) >= 1e5:
        return f"{number:.3e}"
    return f"{number:.{digits}f}"


def count_fmt(value: Any) -> str:
    return f"{int(round(float(value))):,}"


def span(edges: Iterable[float]) -> str:
    values = list(edges)
    return ", ".join(f"{value:g}" for value in values)


def styled_table(
    rows: list[list[Any]],
    widths: list[float],
    *,
    font_size: float = 7.2,
    repeat_rows: int = 1,
    align: str = "LEFT",
    long: bool = False,
):
    converted = []
    for row_index, row in enumerate(rows):
        row_style = "tiny" if font_size <= 6.5 else "small"
        converted.append([
            value if hasattr(value, "wrap") else P(str(value), row_style)
            for value in row
        ])
    klass = LongTable if long else Table
    table = klass(converted, colWidths=widths, repeatRows=repeat_rows, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, -1), align),
        ("GRID", (0, 0), (-1, -1), 0.35, MID_GREY),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for index in range(1, len(rows)):
        if index % 2 == 0:
            commands.append(("BACKGROUND", (0, index), (-1, index), PALE_GREY))
    table.setStyle(TableStyle(commands))
    return table


FIGURE_NUMBER = 0


def caption(text: str) -> Paragraph:
    global FIGURE_NUMBER
    FIGURE_NUMBER += 1
    return P(f"<b>Figure {FIGURE_NUMBER}.</b> {text}", "caption")


def single_figure(path: Path, text: str, width: float = 5.85 * inch) -> list[Any]:
    image = Image(str(path), width=width, height=width * path_image_ratio(path))
    image.hAlign = "CENTER"
    return [image, caption(text)]


def path_image_ratio(path: Path) -> float:
    from PIL import Image as PILImage

    with PILImage.open(path) as image:
        return image.height / image.width


def paired_figures(left: Path, right: Path, text: str, width: float = 3.05 * inch) -> list[Any]:
    left_image = Image(str(left), width=width, height=width * path_image_ratio(left))
    right_image = Image(str(right), width=width, height=width * path_image_ratio(right))
    table = Table([[left_image, right_image]], colWidths=[3.18 * inch, 3.18 * inch], hAlign="CENTER")
    table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
    return [table, caption(text)]


def correction_values(payload: dict[str, Any], variation: str) -> list[float]:
    correction = payload["corrections"][0]
    category = correction["data"]
    entry = next(item for item in category["content"] if item["key"] == variation)
    return [float(value) for value in entry["value"]["content"]]


def tnp_bin_geometry(config: dict[str, Any], flat_index: int) -> tuple[float, float, float, float]:
    eta_edges = config["probe_abseta_edges"]
    pt_edges = config["probe_pt_edges_gev"]
    npt = len(pt_edges) - 1
    eta_index, pt_index = divmod(flat_index, npt)
    return eta_edges[eta_index], eta_edges[eta_index + 1], pt_edges[pt_index], pt_edges[pt_index + 1]


def efficiency_range(result: dict[str, Any], key: str) -> tuple[float, float]:
    values = [float(item[key]) for item in result["bins"] if item.get("valid")]
    return min(values), max(values)


def sf_range(result: dict[str, Any]) -> tuple[float, float]:
    return efficiency_range(result, "scale_factor")


def payload_range(payload: dict[str, Any]) -> tuple[float, float]:
    values = correction_values(payload, "nominal")
    return min(values), max(values)


def draw_measurement_logic() -> Table:
    cells = [
        P("<b>Independent reference</b><br/>Tag HLT for resonance TnP; electron HLT for MET; PFHT for photon", "small"),
        P("<b>Denominator</b><br/>Physics-selected probes without requiring the HLT or object requirement being measured", "small"),
        P("<b>Numerator</b><br/>Subset of the denominator passing the target HLT or target reconstruction requirement", "small"),
        P("<b>Data / MC ratio</b><br/>Efficiency in data divided by the corresponding simulated efficiency", "small"),
        P("<b>Correctionlib payload</b><br/>Nominal, up, and down values evaluated during histogram filling", "small"),
    ]
    arrow = P("<font color='#155A8A'><b>-&gt;</b></font>", "h2")
    row: list[Any] = []
    widths: list[float] = []
    for index, cell in enumerate(cells):
        row.append(cell)
        widths.append(1.1 * inch)
        if index < len(cells) - 1:
            row.append(arrow)
            widths.append(0.22 * inch)
    table = Table([row], colWidths=widths, hAlign="CENTER")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), PALE_BLUE),
        ("BACKGROUND", (2, 0), (2, 0), PALE_GREY),
        ("BACKGROUND", (4, 0), (4, 0), PALE_BLUE),
        ("BACKGROUND", (6, 0), (6, 0), PALE_GREY),
        ("BACKGROUND", (8, 0), (8, 0), PALE_BLUE),
        ("BOX", (0, 0), (0, 0), 0.6, BLUE),
        ("BOX", (2, 0), (2, 0), 0.6, MID_GREY),
        ("BOX", (4, 0), (4, 0), 0.6, BLUE),
        ("BOX", (6, 0), (6, 0), 0.6, MID_GREY),
        ("BOX", (8, 0), (8, 0), 0.6, BLUE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return table


def status_box() -> Table:
    integration = DATA["integration"]
    rows = [
        [P("<b>Measurement status</b>", "small"), P("All 61 adopted bins are finite and the measurement-level blocker lists are empty.", "small")],
        [P("<b>Installed products</b>", "small"), P("Four correctionlib v2 json.gz payloads with nominal/up/down variations.", "small")],
        [P("<b>Histogram interface</b>", "small"), P(str(integration["histogram_stage"]), "small")],
        [P("<b>Boundary policy</b>", "small"), P("Analysis payload for 5 &lt; pT &lt; 10 GeV; official electron/muon payloads take over at pT = 10 GeV.", "small")],
        [P("<b>Validation scope</b>", "small"), P("Measurement adoption and software integration are documented here. Independent full-production physics validation remains a separate task.", "small")],
    ]
    table = Table(rows, colWidths=[1.4 * inch, 5.0 * inch], hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PALE_BLUE),
        ("BOX", (0, 0), (-1, -1), 0.8, BLUE),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def dataset_rows() -> list[list[Any]]:
    ecfg, mcfg = DATA["electron_config"], DATA["muon_config"]
    mtr, phr = DATA["met_result"], DATA["photon_result"]
    eskip, mskip = DATA["electron_skips"], DATA["muon_skips"]
    return [
        ["Measurement", "Data", "Simulation", "File accounting"],
        [
            "MET trigger",
            "EGamma0 and EGamma1, Run2024; single-electron reference paths",
            "TTtoLNu2Q, RunIII2024Summer24NanoAODv15; 405.75 pb",
            f"{mtr['files_processed']:,} retained; 0 permanent skips",
        ],
        [
            "Photon trigger",
            "JetMET0 and JetMET1, Run2024; PFHT reference paths",
            "GJ pT-binned samples: 100-200, 200-400, 400-600, and >=600 GeV",
            f"{phr['files_processed']:,}/6,717 retained; 1 GJ file permanently skipped",
        ],
        [
            "Veto electron 5-10 GeV",
            ecfg["campaign_inputs"]["data_dataset_query"],
            ecfg["campaign_inputs"]["mc_datasets"][0],
            f"{eskip['files_retained']:,}/{eskip['files_before_permanent_skips']:,} retained; 6 Run2024G EGamma files skipped",
        ],
        [
            "Loose muon 5-10 GeV",
            mcfg["campaign_inputs"]["data_dataset_query"],
            mcfg["campaign_inputs"]["mc_datasets"][0],
            f"{mskip['files_retained']:,}/{mskip['files_before_permanent_skips']:,} retained; 7 parking-data files skipped",
        ],
    ]


def trigger_bin_rows(result: dict[str, Any], photon: bool = False) -> list[list[Any]]:
    if photon:
        rows = [["|eta|", "pT [GeV]", "N data", "eff(data)", "eff(MC)", "SF", "stat", "PU", "total"]]
        for item in result["bins"]:
            rows.append([
                f"{item['abseta_low']:g}-{item['abseta_high']:g}",
                f"{item['pt_low_gev']:g}-{item['pt_high_gev']:g}",
                count_fmt(item["data_total"]),
                fmt(item["data_efficiency"], 5),
                fmt(item["mc_efficiency"], 5),
                fmt(item["scale_factor"], 5),
                fmt(item["scale_factor_stat_uncertainty"], 5),
                fmt(item["scale_factor_pileup_uncertainty"], 5),
                fmt(item["scale_factor_uncertainty"], 5),
            ])
        return rows
    rows = [["PuppiMET [GeV]", "N data", "eff(data)", "eff(MC)", "SF", "stat", "PU", "total"]]
    for item in result["bins"]:
        rows.append([
            f"{item['low_gev']:g}-{item['high_gev']:g}",
            count_fmt(item["data_total"]),
            fmt(item["data_efficiency"], 5),
            fmt(item["mc_efficiency"], 5),
            fmt(item["scale_factor"], 5),
            fmt(item["scale_factor_stat_uncertainty"], 5),
            fmt(item["scale_factor_pileup_uncertainty"], 5),
            fmt(item["scale_factor_uncertainty"], 5),
        ])
    return rows


def electron_rows() -> list[list[Any]]:
    cfg, result = DATA["electron_config"], DATA["electron_result"]
    rows = [["|eta|", "pT [GeV]", "eff(data)", "eff(MC)", "SF", "stat", "fit", "PU", "total", "chi2/ndf"]]
    for item in result["bins"]:
        eta0, eta1, pt0, pt1 = tnp_bin_geometry(cfg, item["flat_index"])
        data_fit = item["fits"]["nominal"]["data"]
        mc_fit = item["fits"]["nominal"]["mc"]
        rows.append([
            f"{eta0:g}-{eta1:g}", f"{pt0:g}-{pt1:g}",
            fmt(data_fit["efficiency"], 4), fmt(mc_fit["efficiency"], 4),
            fmt(item["scale_factor"], 4), fmt(item["scale_factor_stat_uncertainty"], 4),
            fmt(item["scale_factor_fit_systematic_uncertainty"], 4),
            fmt(item["scale_factor_pileup_uncertainty"], 4), fmt(item["scale_factor_uncertainty"], 4),
            fmt(data_fit["chi2_ndf"], 2),
        ])
    return rows


def muon_rows() -> list[list[Any]]:
    cfg, result = DATA["muon_config"], DATA["muon_result"]
    payload = DATA["muon_payload"]
    combined = correction_values(payload, "nominal")
    combined_up = correction_values(payload, "up")
    combined_down = correction_values(payload, "down")
    rows = [["|eta|", "pT", "eff(data)", "eff(MC)", "iso SF", "iso total", "combined SF", "combined total", "chi2/ndf"]]
    for item, value, up, down in zip(result["bins"], combined, combined_up, combined_down):
        eta0, eta1, pt0, pt1 = tnp_bin_geometry(cfg, item["flat_index"])
        data_fit = item["fits"]["nominal"]["data"]
        mc_fit = item["fits"]["nominal"]["mc"]
        rows.append([
            f"{eta0:g}-{eta1:g}", f"{pt0:g}-{pt1:g}",
            fmt(data_fit["efficiency"], 4), fmt(mc_fit["efficiency"], 4),
            fmt(item["scale_factor"], 4), fmt(item["scale_factor_uncertainty"], 4),
            fmt(value, 4), fmt(max(up - value, value - down), 4), fmt(data_fit["chi2_ndf"], 2),
        ])
    return rows


def cover_story() -> list[Any]:
    story: list[Any] = [Spacer(1, 20 * mm)]
    line = Table([[""]], colWidths=[8 * mm], rowHeights=[113 * mm])
    line.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), BLUE)]))
    title = [
        P("CMS RUN-3 ALL-HADRONIC STOP ANALYSIS", "small"),
        Spacer(1, 10 * mm),
        P("Measurement and Application of 2024 Trigger and Low-pT Lepton Scale Factors", "cover_title"),
        Spacer(1, 6 * mm),
        P("MET trigger, photon trigger, veto-electron 5-10 GeV, and loose-muon 5-10 GeV", "cover_sub"),
        Spacer(1, 17 * mm),
        P("Physics methods, datasets, fit strategy, uncertainty model, correction payloads, and histogram integration", "body"),
        Spacer(1, 10 * mm),
        P(f"Analysis note date: {date.today().isoformat()}<br/>Collision energy: sqrt(s) = 13.6 TeV<br/>Data-taking year: 2024", "small"),
    ]
    cover = Table([[line, title]], colWidths=[10 * mm, 147 * mm], hAlign="LEFT")
    cover.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 8)]))
    story.append(cover)
    story.append(Spacer(1, 12 * mm))
    abstract = (
        "<b>Abstract.</b> Four data-to-simulation efficiency corrections were measured for the 2024 Run-3 all-hadronic stop analysis. "
        "The trigger corrections are obtained with independent reference-trigger counting, while the low-pT lepton corrections use resonance tag-and-probe with simultaneous pass/fail mass fits. "
        "This report records the physics logic, complete input datasets, bin definitions, fit and uncertainty procedures, adopted results, missing-file accounting, correctionlib representation, and the exact handoff into histogram production. "
        "The central design principle is to measure the efficiency actually used by the analysis without conditioning the denominator on that same requirement."
    )
    story.append(P(abstract, "body"))
    story.append(Spacer(1, 5 * mm))
    story.append(P("Internal analysis document. Values are populated from the adopted JSON results and installed correctionlib payloads in this repository.", "small"))
    story.append(PageBreak())
    return story


def report_story() -> list[Any]:
    met_cfg, met = DATA["met_config"], DATA["met_result"]
    pho_cfg, pho = DATA["photon_config"], DATA["photon_result"]
    ele_cfg, ele = DATA["electron_config"], DATA["electron_result"]
    mu_cfg, mu = DATA["muon_config"], DATA["muon_result"]
    ele_hist, mu_hist = DATA["electron_hist"], DATA["muon_hist"]
    integration = DATA["integration"]
    impact = DATA["impact"]
    met_sf_min, met_sf_max = sf_range(met)
    pho_sf_min, pho_sf_max = sf_range(pho)
    ele_sf_min, ele_sf_max = sf_range(ele)
    mu_iso_min, mu_iso_max = sf_range(mu)
    mu_comb_min, mu_comb_max = payload_range(DATA["muon_payload"])

    story = cover_story()
    story.append(H1("Contents"))
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle("TOC1", parent=S["toc"], fontName=FONT_BOLD, leftIndent=0, firstLineIndent=0, spaceBefore=4),
        ParagraphStyle("TOC2", parent=S["toc"], leftIndent=14, firstLineIndent=0, fontSize=8.2),
    ]
    story.append(toc)
    story.append(PageBreak())

    story.append(H1("1. Executive summary"))
    story.append(status_box())
    story.append(Spacer(1, 5 * mm))
    story.append(P(
        "The correction programme closes two distinct holes in the 2024 analysis model. First, the efficiencies of the MET and photon HLT paths are measured in data and simulation rather than assumed to be identical. Second, the veto-electron and loose-muon selections extend to pT above 5 GeV, whereas the pre-existing official electron and muon payload use in this workflow began at 10 GeV and was previously evaluated by clipping sub-threshold objects to that boundary. The new low-pT measurements replace that artificial boundary treatment in the interval 5 &lt; pT &lt; 10 GeV."
    ))
    story.append(P(
        f"The MET correction contains 18 PuppiMET bins and spans {met_sf_min:.3f}-{met_sf_max:.3f}; the photon correction contains 20 (|eta|, pT) bins and spans {pho_sf_min:.3f}-{pho_sf_max:.3f}. The veto-electron correction uses one 5-10 GeV pT bin in each of three detector regions and spans {ele_sf_min:.3f}-{ele_sf_max:.3f}. The measured muon mini-isolation component spans {mu_iso_min:.3f}-{mu_iso_max:.3f}; after multiplication by the official LooseID term, the installed combined correction spans {mu_comb_min:.3f}-{mu_comb_max:.3f}."
    ))
    story.append(P(
        "The four corrections are analysis weights, not event selections. They change the normalization and, where their axes are kinematic, the shape of simulated histograms. Each central value is accompanied by an up/down variation. The measurement itself is adopted; independent end-to-end production validation is deliberately kept separate and is not claimed complete by this report."
    ))
    story.append(draw_measurement_logic())
    story.append(caption("Unified logic of the four measurements. The reference requirement establishes the event sample, the denominator remains unbiased with respect to the requirement being measured, and the numerator is its passing subset. The output is consumed only after conversion to an auditable correctionlib payload."))

    story.append(H1("2. Analysis philosophy"))
    story.append(H2("2.1 Measure the analysis definition, not a nearby proxy"))
    story.append(P(
        "The object and trigger definitions are chosen to reproduce the actual analysis decisions. The electron target is the veto working point, cutBased >= 1 together with miniPFRelIso_all &lt; 0.1. The muon target is the analysis loose definition, LooseID together with miniPFRelIso_all &lt; 0.2. The MET probe is the logical OR of the PFMET/PFMETNoMu IDTight paths used by the analysis. The photon probe is the logical OR of HLT_Photon175 and HLT_Photon200. Measuring a looser, tighter, or differently factorized efficiency would require an extrapolation and is therefore avoided unless the factorization is explicit and externally constrained."
    ))
    story.append(H2("2.2 Independence is the defining requirement"))
    story.append(P(
        "A scale-factor measurement is unbiased only when entry into the denominator does not require the same decision that defines the numerator. For resonance tag-and-probe, the tag fires the event and is trigger-object matched; the probe is not required to match the tag trigger. For the MET measurement, single-electron HLT paths define the reference sample and the MET HLT paths are tested. For the photon measurement, PFHT paths define the reference sample and the photon HLT paths are tested. Residual topology and prescale effects are validation questions, not reasons to place the probe decision in the denominator."
    ))
    story.append(H2("2.3 Statistical stability takes precedence over decorative granularity"))
    story.append(P(
        "Binning is retained only when supported by the data and by stable uncertainty estimates. The veto-electron campaign began with a 4-by-5 counting grid, but the adopted result is exactly rebinned to one 5-10 GeV pT interval and three |eta| regions because no stable pT dependence was resolved and subdivided endcap fits acquired large model uncertainty. The muon sample is sufficiently large to retain 1 GeV pT intervals. This distinction is driven by available information, not by a universal 1 GeV convention."
    ))
    story.append(H2("2.4 Run-2 continuity and Run-3 implementation"))
    story.append(P(
        "AN2019-016 is used as the conceptual Run-2 strategy baseline for control-region and efficiency-correction logic. It is not copied blindly: the collision energy, 2024 NanoAOD campaigns, available data streams, object definitions, trigger menu, pileup correction, and correctionlib interfaces are Run-3 specific. No local Run-3 analysis note was available in this checkout at report generation time; consequently, every operational statement in this document is tied to the 2024 configuration, code, adopted result, or installed payload."
    ))

    story.append(H1("3. Common measurement formalism"))
    story.append(H2("3.1 Efficiency and scale factor"))
    story.append(P(
        "For a bin b, the denominator contains all eligible probes and the numerator contains the subset that passes the requirement. For direct trigger counting the yields are event counts in data and normalized weighted sums in simulation. For tag-and-probe they are fitted signal yields in the pass and fail spectra."
    ))
    story.append(P("epsilon_X(b) = N_pass^X(b) / [N_pass^X(b) + N_fail^X(b)]\n<br/>SF(b) = epsilon_data(b) / epsilon_MC(b)", "equation"))
    story.append(P(
        "The scale factor multiplies the simulated event weight. A value below unity means that the simulated efficiency exceeds that in data for the same bin; a value above unity means the opposite. It is not a correction to data."
    ))
    story.append(H2("3.2 Trigger counting statistics"))
    story.append(P(
        "Data intervals are exact two-sided Clopper-Pearson intervals at 68.2689% coverage. Simulation uses the weighted efficiency and a Wilson interval evaluated with the effective number of entries, N_eff = (sum w)^2 / sum w^2. The larger side of each asymmetric interval is used in the symmetric scale-factor propagation."
    ))
    story.append(P("sigma_stat(SF) = SF * sqrt[(sigma_data / epsilon_data)^2 + (sigma_MC / epsilon_MC)^2]", "equation"))
    story.append(H2("3.3 Resonance tag-and-probe likelihood"))
    story.append(P(
        "The low-pT lepton measurements form opposite-charge tag-probe pairs and fit the J/psi mass interval 2.6-3.6 GeV in 50 bins. Data pass and fail spectra are fit simultaneously. Separate MC pass and fail mass templates describe the resonant signal. A common template shift and width scale allow the simulated line shape to adapt to the data, while independent pass and fail background yields and slopes absorb the nonresonant component. The efficiency is a fit parameter shared by the two spectra."
    ))
    story.append(P(
        "M_pass(m) = N_sig * epsilon * S_pass(m; delta, kappa) + N_bkg,pass * B_pass(m)\n<br/>M_fail(m) = N_sig * (1 - epsilon) * S_fail(m; delta, kappa) + N_bkg,fail * B_fail(m)",
        "equation",
    ))
    story.append(P(
        "The nominal background is exponential. The eight fit parameters are log signal yield, logit efficiency, pass background yield and slope, fail background yield and slope, template shift, and log width scale. A bounded nonlinear least-squares fit minimizes residuals normalized by the pass/fail histogram variances. The covariance matrix is the Moore-Penrose pseudoinverse of J^T J scaled by chi2/ndf; the logit variance is propagated to the efficiency. Simulation is a pure resonant sample and its efficiency is obtained by weighted counting rather than by fitting a background that is not present."
    ))
    story.append(H2("3.4 Event and pair hygiene"))
    for item in [
        "Data are restricted to the 2024 golden-luminosity JSON and required event-quality filters.",
        "Events are deduplicated with (run, luminosityBlock, event) keys before pairs enter the histograms.",
        "Tag and probe must be distinct objects with opposite charge.",
        "The tag is matched to an appropriate trigger object within DeltaR < 0.1 and with the audited filter bit.",
        "Simulation uses genWeight times the nominal 2024 pileup weight; pileup up/down histograms are produced in the same pass.",
    ]:
        story.append(bullet(item))

    story.append(H1("4. Datasets and campaign accounting"))
    story.append(styled_table(dataset_rows(), [1.05 * inch, 1.75 * inch, 2.3 * inch, 1.35 * inch], font_size=6.5))
    story.append(Spacer(1, 4 * mm))
    story.append(P(
        "All campaigns operate on NanoAOD ROOT files and are sharded in groups of 20 files. Only compact numerator/denominator or pass/fail histograms and metadata are retained for reduction. This preserves reproducibility while avoiding repeated reading of full event content during fits, systematic variations, plotting, and payload export."
    ))
    story.append(H2("4.1 Simulation normalization"))
    met_norm_rows = [["MET MC dataset", "files", "xsec [pb]", "Runs gen sumw", "xsec/sumw"]]
    for name, values in met["mc_normalization"].items():
        met_norm_rows.append([name, values["files"], values["xsec_pb"], fmt(values["runs_gen_event_sumw"], 3), fmt(values["factor_without_luminosity"], 3)])
    story.append(styled_table(met_norm_rows, [2.95 * inch, 0.5 * inch, 0.7 * inch, 1.25 * inch, 1.0 * inch], font_size=6.5))
    story.append(Spacer(1, 3 * mm))
    pho_norm_rows = [["Photon MC dataset", "files", "xsec [pb]", "Runs gen sumw", "xsec/sumw"]]
    for name, values in pho["mc_normalization"].items():
        pho_norm_rows.append([name, values["files"], values["xsec_pb"], fmt(values["runs_gen_event_sumw"], 3), fmt(values["factor_without_luminosity"], 3)])
    story.append(styled_table(pho_norm_rows, [2.95 * inch, 0.5 * inch, 0.7 * inch, 1.25 * inch, 1.0 * inch], font_size=6.2))
    story.append(P(
        "The factors above omit integrated luminosity because it cancels in an efficiency ratio. They are nevertheless required to combine distinct simulated samples consistently. Runs.genEventSumw is aggregated from successful unique files; retry outputs cannot silently duplicate the denominator."
    ))

    story.append(H1("5. MET trigger scale factor"))
    story.append(H2("5.1 Physics selection and independence"))
    story.append(P(
        "The reference sample is selected from EGamma0/EGamma1 with the logical OR of HLT_Ele30, Ele32, Ele35, Ele38, and Ele40 WPTight_Gsf. It contains at least one medium electron, zero loose muons, zero selected taus, no isolated-track veto object, at least two cleaned AK4 jets, HT > 300 GeV, the open preselection, event filters, golden luminosity selection, zero veto-map jets, and PuppiMET/CaloMET < 5. Critically, there is no MET threshold and no MET-HLT requirement in the denominator."
    ))
    story.append(P(
        "The numerator is the subset passing any configured PFMET120/130/140 or PFMETNoMu120/130/140 PFMHT IDTight path. The observable is corrected PuppiMET pT. The simulation is semileptonic TTtoLNu2Q, which provides a genuine-neutrino MET topology under the same reference selection. The measured genuine-MET correction is used for all MC in MET-triggered preselection, SR, LLCR, QCDCR, and validation regions; no independent PFHT-QCD correction is installed."
    ))
    story.append(P(f"Binning [GeV]: {span(met_cfg['bin_edges_gev'])}.", "small"))
    story.extend(paired_figures(
        MET_DIR / "plots/2024_full/met_trigger_efficiency_2024.png",
        MET_DIR / "plots/2024_full/met_trigger_scale_factor_2024.png",
        "MET-trigger efficiency in data and normalized TT simulation (left) and the resulting data-to-simulation correction (right). The turn-on is explicitly resolved between 100 and 300 GeV; the high-MET plateau approaches unity.",
    ))
    story.append(P(
        f"The reduction contains {count_fmt(sum(item['data_total'] for item in met['bins']))} denominator events and {count_fmt(sum(item['data_passed'] for item in met['bins']))} passing events in data. All 18 bins are valid. The lowest 100-120 GeV bin gives SF = {met['bins'][0]['scale_factor']:.4f} +/- {met['bins'][0]['scale_factor_uncertainty']:.4f}; the 650-800 GeV bin is {met['bins'][-1]['scale_factor']:.4f} +/- {met['bins'][-1]['scale_factor_uncertainty']:.4f}."
    ))

    story.append(H1("6. Photon trigger scale factor"))
    story.append(H2("6.1 Reference-trigger strategy"))
    story.append(P(
        "The photon measurement is not a resonance tag-and-probe measurement. It uses JetMET0/JetMET1 events accepted by the OR of PFHT180 through PFHT1050 as an independent reference. The denominator requires at least one medium photon, vetoes analysis veto electrons and loose muons, vetoes selected taus and isolated tracks, requires at least two photon-cleaned AK4 jets and HT > 300 GeV, and applies the baseline-like open preselection without a MET threshold. The numerator is the subset passing HLT_Photon175 or HLT_Photon200."
    ))
    story.append(P(
        "G+jet simulation is the correct comparison sample because it supplies prompt photons over the relevant pT range. Four generator-pT samples are normalized with their own cross sections and generator sum weights before their numerator and denominator histograms are combined. The ECAL transition 1.4442 < |eta| < 1.566 is excluded."
    ))
    story.append(P(f"pT edges [GeV]: {span(pho_cfg['pt_edges_gev'])}. |eta| edges: {span(pho_cfg['abseta_edges'])}.", "small"))
    story.extend(paired_figures(
        PHOTON_DIR / "plots/2024_full/photon_trigger_efficiency_2024.png",
        PHOTON_DIR / "plots/2024_full/photon_trigger_scale_factor_2024.png",
        "Photon-trigger efficiency (left) and scale factor (right) in the four detector regions. The analysis domain begins at 220 GeV, above the nominal HLT thresholds, and all 20 bins are retained.",
    ))
    story.append(P(
        f"The data reduction contains {count_fmt(sum(item['data_total'] for item in pho['bins']))} denominator events and {count_fmt(sum(item['data_passed'] for item in pho['bins']))} passing events. The SF range is {pho_sf_min:.4f}-{pho_sf_max:.4f}. One repeatedly unreadable GJ 100-200 GeV ROOT file was explicitly accepted as permanently skipped; the final coverage is 6,716 of 6,717 files and the loss is recorded in the adopted result."
    ))

    story.append(H1("7. Low-pT tag-and-probe method"))
    story.append(H2("7.1 Why J/psi"))
    story.append(P(
        "The analysis lepton threshold starts above 5 GeV, so a Z-based tag-and-probe sample is not optimal for the critical 5-10 GeV interval. J/psi decays provide a narrow, abundant resonance with two reconstructed leptons in the desired momentum range. Requiring an opposite-charge pair and fitting the resonant mass peak separates genuine lepton pairs from combinatorial background without using the probe's target ID or isolation decision."
    ))
    story.append(H2("7.2 Tag, probe, pass, and fail"))
    tnp_rows = [
        ["Element", "Electron measurement", "Muon measurement"],
        ["Data", "EGamma Run2024", "ParkingDoubleMuonLowMass Run2024"],
        ["Tag", "tight electron, miniIso < 0.1", "tight muon, miniIso < 0.1"],
        ["Trigger-object match", "DeltaR < 0.1, filter bit 8192", "DeltaR < 0.1, filter bit 16"],
        ["Probe denominator", "GSF e, convVeto, lostHits <= 1, ECAL fiducial", "muon with LooseID"],
        ["Probe pass", "cutBased >= 1 and miniIso < 0.1", "miniIso < 0.2 conditional on LooseID"],
        ["Pair", "different objects, opposite charge", "different objects, opposite charge"],
        ["Mass", "2.6-3.6 GeV, 50 bins", "2.6-3.6 GeV, 50 bins"],
    ]
    story.append(styled_table(tnp_rows, [1.35 * inch, 2.55 * inch, 2.55 * inch], font_size=7.0))
    story.append(H2("7.3 Reference-trigger audit"))
    story.append(P(
        "Electron tags use Ele8/Ele12 plus PFJet30 paths and filter bit 8192 (bit 13). The PFJet30 leg is an event-side condition only: it is not part of the probe pass/fail definition. Muon tags use DoubleMu4_3_LowMass, DoubleMu2_Jpsi_LowPt, or DoubleMu4_3_Jpsi and filter bit 16 (bit 4). In both channels the trigger-object bit is explicitly present in data and simulation, and the tag - not the probe - is required to match."
    ))
    story.append(H2("7.4 Fit variations"))
    variations = [
        ["Variation", "Change from nominal"],
        ["signal_template_combined", "Use one combined MC signal template for pass and fail"],
        ["background_linear", "Replace exponential backgrounds by linear shapes"],
        ["mass_window_narrow", "Trim 8% of the full interval from each side"],
        ["mass_window_medium", "Trim 4% of the full interval from each side"],
        ["alternate_binning", "Merge adjacent mass bins (rebin factor 2)"],
        ["pileup up/down", "Refit after replacing nominal MC pileup weights"],
    ]
    story.append(styled_table(variations, [1.85 * inch, 4.55 * inch], font_size=7.2))

    story.append(H1("8. Veto-electron scale factor, 5-10 GeV"))
    story.append(P(
        "The electron denominator is a reconstructed GSF electron with 5 < pT < 10 GeV in the ECAL fiducial region, passing conversion veto and lostHits <= 1, but not conditioned on the veto working point or mini-isolation. The passing definition is cutBased >= 1 and miniPFRelIso_all < 0.1. The primary simulation is SPS-JpsiJpsiToMuMuEE from RunIII2024Summer24NanoAODv15. The alternative Winter24 inclusive JpsiToEE sample is reserved for closure and is not mixed into the nominal denominator without normalization."
    ))
    story.append(P(
        f"The compact histograms contain {count_fmt(sum(sum(row) for row in ele_hist['samples']['data']['pass_sumw']))} passing and {count_fmt(sum(sum(row) for row in ele_hist['samples']['data']['fail_sumw']))} failing data pairs. The adopted scale factors are {ele['bins'][0]['scale_factor']:.4f} +/- {ele['bins'][0]['scale_factor_uncertainty']:.4f}, {ele['bins'][1]['scale_factor']:.4f} +/- {ele['bins'][1]['scale_factor_uncertainty']:.4f}, and {ele['bins'][2]['scale_factor']:.4f} +/- {ele['bins'][2]['scale_factor_uncertainty']:.4f} in the three |eta| regions. The endcap uncertainty is retained rather than hidden by smoothing or clipping."
    ))
    story.extend(paired_figures(
        ELECTRON_DIR / "plots/2024_full/efficiency.png",
        ELECTRON_DIR / "plots/2024_full/scale_factor.png",
        "Veto-electron efficiency (left) and adopted data-to-simulation scale factor (right). One 5-10 GeV pT interval is used in each |eta| region because finer pT subdivisions were not statistically stable.",
    ))
    story.extend(single_figure(
        ELECTRON_DIR / "plots/2024_full/mass_fit_bin_00.png",
        "Representative central-electron J/psi simultaneous pass/fail fit. The four panels show data pass, data fail, simulated pass, and simulated fail. The data efficiency is obtained from the common pass/fail signal yield parameter; the MC efficiency is weighted counting in the resonant template sample.",
        width=5.5 * inch,
    ))

    story.append(H1("9. Loose-muon scale factor, 5-10 GeV"))
    story.append(H2("9.1 Factorized measurement"))
    story.append(P(
        "The muon denominator already requires LooseID, so the resonance fit measures miniPFRelIso_all < 0.2 conditional on LooseID. The final analysis correction must cover the full loose-muon definition. Therefore the fitted isolation SF is multiplied bin by bin by the official 2024 NUM_LooseID_DEN_TrackerMuons correction. The installed uncertainty combines the fitted isolation uncertainty and the official LooseID uncertainty in quadrature, with the appropriate multiplicative factors."
    ))
    story.append(P(
        "SF_comb = SF_iso|LooseID * SF_LooseID\n<br/>sigma_comb^2 = (SF_LooseID * sigma_iso)^2 + (SF_iso * sigma_LooseID)^2",
        "equation",
    ))
    story.append(P(
        f"The compact histograms contain {count_fmt(sum(sum(row) for row in mu_hist['samples']['data']['pass_sumw']))} passing and {count_fmt(sum(sum(row) for row in mu_hist['samples']['data']['fail_sumw']))} failing data pairs. Twenty (|eta|, pT) bins are retained. The measured conditional-isolation SF spans {mu_iso_min:.4f}-{mu_iso_max:.4f}; the correction actually applied to analysis histograms spans {mu_comb_min:.4f}-{mu_comb_max:.4f} after the official LooseID multiplication."
    ))
    story.extend(paired_figures(
        MUON_DIR / "plots/2024_full/efficiency.png",
        MUON_DIR / "plots/2024_full/scale_factor.png",
        "Muon mini-isolation efficiency conditional on LooseID (left) and its fitted data-to-simulation scale factor (right). The right panel shows the measured component; the installed analysis payload additionally includes the official LooseID term.",
    ))
    story.extend(single_figure(
        MUON_DIR / "plots/2024_full/scale_factor_heatmap.png",
        "Two-dimensional map of the measured muon mini-isolation component. The colorbar plot was produced in the analysis-standard 12-by-10 layout. Each cell displays the central value and total fitted-component uncertainty; the official LooseID contribution is added during payload export.",
        width=5.65 * inch,
    ))
    story.extend(single_figure(
        MUON_DIR / "plots/2024_full/mass_fit_bin_17.png",
        "Representative forward-muon J/psi pass/fail fit for 2.1 < |eta| < 2.4 and 7 < pT < 8 GeV. This bin has the largest total fitted-component uncertainty and is shown to make the limiting mass-shape information explicit.",
        width=5.5 * inch,
    ))

    story.append(H1("10. Uncertainty model"))
    story.append(H2("10.1 Trigger corrections"))
    story.append(P(
        "For MET and photon triggers, the statistical scale-factor uncertainty is propagated from the exact data interval and the weighted-MC interval. Pileup uncertainty is the larger absolute displacement of the scale factor under the 2024 pileup up/down weights. The reported total is the quadrature sum of statistical and pileup terms."
    ))
    story.append(P("sigma_trigger,total = sqrt[sigma_stat^2 + max(|SF_PUup - SF|, |SF_PUdown - SF|)^2]", "equation"))
    story.append(H2("10.2 Tag-and-probe corrections"))
    story.append(P(
        "For each lepton bin, the statistical term combines the data fit and weighted-MC counting uncertainties. The fit-model systematic is the envelope of the five nonnominal fit configurations relative to nominal. The pileup term is the envelope of the up/down refits. The fit-model and pileup terms are combined in quadrature, followed by the statistical term. For the installed muon payload, the official LooseID uncertainty is then added as described in Sec. 9.1."
    ))
    story.append(P(
        "sigma_fit = max_i |SF_i - SF_nom|\n<br/>sigma_syst = sqrt(sigma_fit^2 + sigma_PU^2)\n<br/>sigma_total = sqrt(sigma_stat^2 + sigma_syst^2)",
        "equation",
    ))
    story.append(H2("10.3 Correlation convention in histogram production"))
    story.append(P(
        "Each installed correction exposes one string variation input: nominal, up, or down. Histogram production therefore creates one coherent up/down nuisance per payload: met_trigger, photon_trigger, veto_electron_5to10, and loose_muon_5to10. Bins inside a payload move together under that nuisance, while the four payload nuisances are independent of each other. This is a conservative implementation convention, not a statement that every fitted bin is physically 100% correlated. A later statistical-model study may decorrelate bins or sources only with an explicit covariance model."
    ))

    story.append(H1("11. Correctionlib payloads and histogram application"))
    payload_rows = [["Component", "correction", "axes", "bins", "SHA256"]]
    axes = {
        "met_trigger": "variation, PuppiMET pT",
        "photon_trigger": "variation, |eta|, photon pT",
        "veto_electron_5to10": "variation, |eta|, electron pT",
        "loose_muon_5to10": "variation, |eta|, muon pT",
    }
    for name, item in integration["payloads"].items():
        payload_rows.append([name, item["correction"], axes[name], item["bins"], item["sha256"][:16] + "..."])
    story.append(styled_table(payload_rows, [1.25 * inch, 1.45 * inch, 2.05 * inch, 0.45 * inch, 1.2 * inch], font_size=6.5))
    story.append(H2("11.1 Exact 5-10 GeV handoff"))
    story.append(P(
        "For veto electrons and loose muons satisfying 5 < pT < 10 GeV, the old official-payload contribution is first set to unity and the analysis-owned low-pT payload is multiplied once. At pT = 10 GeV, the low-pT component is unity and the standard official payload is used. At the object threshold itself, pT = 5 GeV is not selected because the object definitions require pT > 5 GeV. This removes boundary clipping and prevents double counting."
    ))
    boundary_rows = [
        ["pT point", "Object selected?", "Analysis low-pT payload", "Official payload"],
        ["5.000 GeV", "No", "not applied", "not relevant"],
        ["5.001 GeV", "Yes", "applied", "replaced by unity in this interval"],
        ["9.999 GeV", "Yes", "applied", "replaced by unity in this interval"],
        ["10.000 GeV", "Yes", "unity", "applied at native boundary"],
    ]
    story.append(styled_table(boundary_rows, [1.0 * inch, 1.15 * inch, 2.15 * inch, 2.15 * inch], font_size=7.2))
    story.append(H2("11.2 Region masks and nuisance leaves"))
    story.append(P(
        "The photon-trigger correction is active in the photon control region. The MET-trigger correction is active in MET-triggered preselection, SR, LLCR, QCDCR, and validation regions. Object corrections are products over selected veto electrons or loose muons and therefore contribute wherever those objects enter the analysis weight. Histogram production emits the eight explicit leaves met_triggerUp/Down, photon_triggerUp/Down, veto_electron_5to10Up/Down, and loose_muon_5to10Up/Down. Missing required payloads fail closed in production."
    ))

    story.append(H1("12. Validation, failure recovery, and limitations"))
    story.append(H2("12.1 Adoption evidence"))
    validation_rows = [
        ["Measurement", "valid bins", "blockers", "fit / count status"],
        ["MET trigger", "18/18", "0", "finite SF and uncertainty in every bin"],
        ["Photon trigger", "20/20", "0", "finite SF and uncertainty in every bin"],
        ["Veto electron", f"{ele['validation']['valid_bins']}/{ele['validation']['expected_bins']}", len(ele['validation']['blockers']), f"nominal data chi2/ndf {min(x['chi2_ndf'] for x in ele['validation']['fit_diagnostics'] if x['sample']=='data' and x['variation']=='nominal'):.2f}-{max(x['chi2_ndf'] for x in ele['validation']['fit_diagnostics'] if x['sample']=='data' and x['variation']=='nominal'):.2f}"],
        ["Loose muon", f"{mu['validation']['valid_bins']}/{mu['validation']['expected_bins']}", len(mu['validation']['blockers']), f"nominal data chi2/ndf {min(x['chi2_ndf'] for x in mu['validation']['fit_diagnostics'] if x['sample']=='data' and x['variation']=='nominal'):.2f}-{max(x['chi2_ndf'] for x in mu['validation']['fit_diagnostics'] if x['sample']=='data' and x['variation']=='nominal'):.2f}"],
    ]
    story.append(styled_table(validation_rows, [1.25 * inch, 0.8 * inch, 0.65 * inch, 3.75 * inch], font_size=7.2))
    story.append(H2("12.2 Missing-file policy and coverage caveats"))
    story.append(P(
        "Unreadable files do not halt the campaign. Alternate endpoints are attempted; reproducibly unavailable files are recorded and excluded before reduction. Photon simulation loses one of 6,717 files. The electron tag-and-probe sample loses six of 5,440 files, all from Run2024G EGamma0. The muon sample loses seven of 7,714 files; six are from ParkingDoubleMuonLowMass6 Run2024D and one from Run2024G. The electron and muon manifests explicitly set data_lumi_coverage_complete to false because unique luminosity-section coverage was not reconstructed for the missing files. This is a documentation limitation even though efficiency ratios are less sensitive to absolute luminosity than yield measurements."
    ))
    story.append(H2("12.3 What is and is not validated"))
    story.append(P(
        "The adopted results have empty measurement-level blocker lists, all expected bins, trigger-object audits, fit-variation results, and visual plot reviews. The software integration summary records 7/7 analysis-SF tests, 7/7 shape-histogram tests, explicit checks at 5.001, 9.999, and 10.0 GeV, and the presence of all eight Up/Down histogram leaves. These tests establish implementation behavior. They do not replace an independent full-statistics closure of every process, era, control region, transfer factor, and final nuisance correlation. That broader validation is intentionally assigned outside the work reported here."
    ))
    story.append(H2("12.4 Physics-impact estimate"))
    llcr = impact["corrections"][0]["data"]["content"]
    tf = impact["corrections"][1]["data"]["content"]
    llcr_values = {item["key"]: item["value"] for item in llcr}
    tf_values = {item["key"]: item["value"] for item in tf}
    story.append(P(
        f"A composition-level estimate predicts an inclusive LLCR yield ratio of {llcr_values['nominal']:.4f} after/before the new low-pT lepton corrections, with an envelope {llcr_values['down']:.4f}-{llcr_values['up']:.4f}. If the SR numerator is unchanged, the corresponding SR/LLCR transfer-factor ratio is {tf_values['nominal']:.4f}, with envelope {tf_values['down']:.4f}-{tf_values['up']:.4f}. These are diagnostic estimates only. They must not be multiplied on top of the event-level corrections and are not presented as a production result."
    ))

    story.append(H1("13. Conclusions"))
    for item in [
        "The MET and photon trigger efficiencies are measured with independent reference triggers and stored as 18-bin and 20-bin correctionlib payloads.",
        "The 5-10 GeV veto-electron and loose-muon efficiency gap is measured with J/psi tag-and-probe rather than filled by clipping the official 10 GeV boundary.",
        "Electron binning is one pT interval in three |eta| regions because the data do not support stable finer granularity; muons retain 1 GeV bins because the parking sample does.",
        "The muon result is correctly factorized: the fit measures mini-isolation conditional on LooseID, and the installed payload combines it with the official LooseID correction and uncertainty.",
        "All four payloads carry nominal/up/down values and are connected to histogram production with explicit nuisance leaves and an exact 10 GeV handoff.",
        "Missing files and incomplete luminosity-coverage claims are preserved in the analysis record; independent full-production physics validation remains to be completed separately.",
    ]:
        story.append(bullet(item))

    story.append(PageBreak())
    story.append(H1("Appendix A. Complete MET trigger result"))
    story.append(styled_table(trigger_bin_rows(met), [1.0 * inch, 0.7 * inch, 0.75 * inch, 0.75 * inch, 0.65 * inch, 0.65 * inch, 0.65 * inch, 0.65 * inch], font_size=6.2, long=True))
    story.append(P("The stat and PU columns are absolute SF uncertainties; total is their quadrature sum.", "small"))

    story.append(PageBreak())
    story.append(H1("Appendix B. Complete photon trigger result"))
    story.append(styled_table(trigger_bin_rows(pho, photon=True), [0.63 * inch, 0.68 * inch, 0.58 * inch, 0.68 * inch, 0.68 * inch, 0.62 * inch, 0.62 * inch, 0.62 * inch, 0.62 * inch], font_size=6.0, long=True))

    story.append(PageBreak())
    story.append(H1("Appendix C. Complete veto-electron result"))
    story.append(styled_table(electron_rows(), [0.55 * inch, 0.55 * inch, 0.65 * inch, 0.65 * inch, 0.55 * inch, 0.55 * inch, 0.55 * inch, 0.55 * inch, 0.55 * inch, 0.55 * inch], font_size=6.0))
    story.append(P("The ECAL gap 1.4442 < |eta| < 1.566 is excluded even though the adjacent endcap bin is labelled by its stored multibinning edge. Fit is the envelope of line-shape, background, mass-window, and mass-binning alternatives.", "small"))
    story.extend(single_figure(
        ELECTRON_DIR / "plots/2024_full/scale_factor_heatmap.png",
        "Complete veto-electron correction map. The plot is 12-by-10 because it contains a colorbar. The single pT column is the adopted 5-10 GeV interval.",
        width=5.65 * inch,
    ))

    story.append(PageBreak())
    story.append(H1("Appendix D. Complete loose-muon result"))
    story.append(styled_table(muon_rows(), [0.52 * inch, 0.48 * inch, 0.62 * inch, 0.62 * inch, 0.55 * inch, 0.57 * inch, 0.65 * inch, 0.65 * inch, 0.58 * inch], font_size=5.8, long=True))
    story.append(P("iso SF is the fitted mini-isolation correction conditional on LooseID. combined SF is the correction installed for analysis use after multiplying the official LooseID term. combined total includes the official LooseID uncertainty.", "small"))

    story.append(PageBreak())
    story.append(H1("Appendix E. Complete mass-fit gallery"))
    story.append(P(
        "The following pages reproduce every adopted pass/fail fit diagnostic. Each square contains data pass, data fail, simulated pass, and simulated fail panels. The figures carry only the standard CMS left/right labels; bin information and interpretation are provided in the captions."
    ))
    gallery: list[tuple[Path, str]] = []
    for index in range(3):
        eta0, eta1, pt0, pt1 = tnp_bin_geometry(ele_cfg, index)
        gallery.append((ELECTRON_DIR / f"plots/2024_full/mass_fit_bin_{index:02d}.png", f"Electron fit bin {index}: {eta0:g} < |eta| < {eta1:g}, {pt0:g} < pT < {pt1:g} GeV."))
    for index in range(20):
        eta0, eta1, pt0, pt1 = tnp_bin_geometry(mu_cfg, index)
        gallery.append((MUON_DIR / f"plots/2024_full/mass_fit_bin_{index:02d}.png", f"Muon fit bin {index}: {eta0:g} < |eta| < {eta1:g}, {pt0:g} < pT < {pt1:g} GeV."))
    for gallery_index, (path, text) in enumerate(gallery):
        if gallery_index and gallery_index % 2 == 0:
            story.append(PageBreak())
        image = Image(str(path), width=3.15 * inch, height=3.15 * inch)
        image.hAlign = "CENTER"
        story.append(image)
        story.append(caption(text))

    story.append(PageBreak())
    story.append(H1("Appendix F. Reproducibility and provenance"))
    source_rows = [["Role", "Repository path", "SHA256"]]
    source_items = [
        ("MET definition", PATHS["met_config"]),
        ("MET adopted result", PATHS["met_result"]),
        ("Photon definition", PATHS["photon_config"]),
        ("Photon adopted result", PATHS["photon_result"]),
        ("Electron definition", PATHS["electron_config"]),
        ("Electron adopted result", PATHS["electron_result"]),
        ("Muon definition", PATHS["muon_config"]),
        ("Muon adopted result", PATHS["muon_result"]),
        ("Integration audit", PATHS["integration"]),
        ("MET payload", PATHS["met_payload"]),
        ("Photon payload", PATHS["photon_payload"]),
        ("Electron payload", PATHS["electron_payload"]),
        ("Muon payload", PATHS["muon_payload"]),
    ]
    for role, path in source_items:
        source_rows.append([role, str(path.relative_to(REPO)), sha256(path)])
    story.append(styled_table(source_rows, [1.1 * inch, 3.35 * inch, 2.0 * inch], font_size=5.8, long=True))
    story.append(H2("F.1 Primary implementation sources"))
    code_rows = [
        ["Purpose", "Path"],
        ["Unified trigger counting/reduction/export", "autonomous_allhad/workflow/measure_trigger.py"],
        ["Tag-and-probe histogram construction", "autonomous_allhad/workflow/tnp_histograms.py"],
        ["Pass/fail fits and uncertainty envelope", "autonomous_allhad/workflow/tnp_fit.py"],
        ["Correctionlib payload helpers", "autonomous_allhad/workflow/sf_payload.py"],
        ["Runtime payload evaluation", "autonomous_allhad/autonomous_allhad/analysis_scale_factors.py"],
        ["Event-weight composition and 10 GeV handoff", "autonomous_allhad/autonomous_allhad/real_subset_worker.py"],
        ["Histogram entry point", "autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py"],
        ["Unified plotting", "autonomous_allhad/workflow/plot_measurement.py"],
    ]
    story.append(styled_table(code_rows, [2.05 * inch, 4.4 * inch], font_size=6.5))
    story.append(H2("F.2 References"))
    references = [
        "CMS Collaboration, AN2019-016 v9, Run-2 all-hadronic stop-search strategy baseline, cited by the measurement configurations.",
        "2024 measurement configuration JSON files listed in the provenance table above.",
        "Adopted result JSON files and correctionlib v2 json.gz payloads listed in the provenance table above.",
        "CMS NanoAOD trigger-object definitions for the audited filter bits used by the 2024 configurations.",
        "2024 golden luminosity JSON and pileup correction Collisions24_BCDEFGHI_goldenJSON used in the histogram builders.",
    ]
    for index, text in enumerate(references, 1):
        story.append(P(f"[{index}] {text}", "small"))
    story.append(Spacer(1, 5 * mm))
    story.append(P("End of report.", "small"))
    return story


def paths_for_2025() -> dict[str, Path]:
    output_name = "2025_full"
    return {
        "met_config": MET_DIR / "config_2025.json",
        "met_result": MET_DIR / f"outputs/{output_name}/met_trigger_result_adopted.json",
        "photon_config": PHOTON_DIR / "config_2025.json",
        "photon_result": PHOTON_DIR / f"outputs/{output_name}/photon_trigger_result_adopted.json",
        "electron_config": ELECTRON_DIR / "config_2025_id_only_parking_singlemuon.json",
        "electron_result": ELECTRON_DIR / f"outputs/{output_name}/adopted_result.json",
        "electron_hist": ELECTRON_DIR / f"outputs/{output_name}/histograms.json",
        "muon_config": MUON_DIR / "config_2025_id_only_parking_external.json",
        "muon_result": MUON_DIR / f"outputs/{output_name}/adopted_result.json",
        "muon_hist": MUON_DIR / f"outputs/{output_name}/histograms.json",
        "integration": REPO / "autonomous_allhad/workflow/analysis_sf_integration_validation/summary_2025.json",
        "met_payload": REPO / "analysis/data/AnalysisSF/2025/met_trigger_sf.json.gz",
        "photon_payload": REPO / "analysis/data/AnalysisSF/2025/photon_trigger_sf.json.gz",
        "electron_payload": REPO / "analysis/data/AnalysisSF/2025/veto_electron_5to10_sf.json.gz",
        "muon_payload": REPO / "analysis/data/AnalysisSF/2025/loose_muon_5to10_sf.json.gz",
    }


def _join(values: Iterable[Any]) -> str:
    return ", ".join(str(value) for value in values)


def _file_accounting(result: dict[str, Any]) -> str:
    processed = int(result.get("files_processed", 0))
    failed = len(result.get("files_failed") or [])
    skipped = len(result.get("files_permanently_skipped") or [])
    text = f"{processed:,} processed"
    if failed:
        text += f"; {failed:,} unresolved"
    if skipped:
        text += f"; {skipped:,} permanently skipped"
    if not failed and not skipped:
        text += "; no unresolved file"
    return text


def dataset_rows_2025() -> list[list[Any]]:
    met_cfg = DATA["met_config"]
    photon_cfg = DATA["photon_config"]
    electron_cfg = DATA["electron_config"]
    muon_cfg = DATA["muon_config"]
    return [
        ["Measurement", "Collision data", "Simulation", "Retained coverage"],
        [
            "MET trigger",
            _join(met_cfg["campaign_inputs"]["data_dataset_prefixes"]) + "; single-electron reference",
            _join(item["dataset_contains"] for item in met_cfg["campaign_inputs"]["mc_datasets"]),
            _file_accounting(DATA["met_result"]),
        ],
        [
            "Photon trigger",
            _join(photon_cfg["campaign_inputs"]["data_dataset_prefixes"]) + "; PFHT reference",
            "Summer24 GJ bins: " + _join(item["dataset_contains"].split("_Tune", 1)[0] for item in photon_cfg["campaign_inputs"]["mc_datasets"]),
            _file_accounting(DATA["photon_result"]),
        ],
        [
            "Veto electron ID, 5-10 GeV",
            "ParkingSingleMuon, Run2025C-G; independent external-muon reference",
            electron_cfg["campaign_inputs"]["mc_datasets"][0],
            _file_accounting(DATA["electron_hist"]),
        ],
        [
            "Loose muon ID, 5-10 GeV",
            "ParkingSingleMuon, Run2025C-G; disjoint external-muon reference",
            muon_cfg["campaign_inputs"]["mc_datasets"][0],
            _file_accounting(DATA["muon_hist"]),
        ],
    ]


def tnp_rows_2025(kind: str) -> list[list[Any]]:
    result = DATA[f"{kind}_result"]
    payload = DATA[f"{kind}_payload"]
    payload_nominal = correction_values(payload, "nominal")
    payload_up = correction_values(payload, "up")
    payload_down = correction_values(payload, "down")
    eta_edges = result["probe_abseta_edges"]
    pt_edges = result["probe_pt_edges_gev"]
    n_pt = len(pt_edges) - 1
    rows = [["|eta|", "pT [GeV]", "eff(data)", "eff(MC)", "fit SF", "payload SF", "total unc.", "chi2/ndf"]]
    for index, (item, central, up, down) in enumerate(
        zip(result["bins"], payload_nominal, payload_up, payload_down)
    ):
        eta_index, pt_index = divmod(index, n_pt)
        nominal = (item.get("fits") or {}).get("nominal") or {}
        data_fit = nominal.get("data") or {}
        mc_fit = nominal.get("mc") or {}
        rows.append([
            f"{eta_edges[eta_index]:g}-{eta_edges[eta_index + 1]:g}",
            f"{pt_edges[pt_index]:g}-{pt_edges[pt_index + 1]:g}",
            fmt(data_fit.get("efficiency"), 4),
            fmt(mc_fit.get("efficiency"), 4),
            fmt(item.get("scale_factor"), 4),
            fmt(central, 4),
            fmt(max(up - central, central - down), 4),
            fmt(data_fit.get("chi2_ndf"), 2),
        ])
    return rows


def cover_story_2025() -> list[Any]:
    story: list[Any] = [Spacer(1, 20 * mm)]
    line = Table([[""]], colWidths=[8 * mm], rowHeights=[113 * mm])
    line.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), BLUE)]))
    title = [
        P("CMS RUN-3 ALL-HADRONIC STOP ANALYSIS", "small"),
        Spacer(1, 10 * mm),
        P("Measurement and Application of 2025 Trigger and Low-pT Lepton Scale Factors", "cover_title"),
        Spacer(1, 6 * mm),
        P("MET trigger, photon trigger, veto-electron ID, and loose-muon ID", "cover_sub"),
        Spacer(1, 17 * mm),
        P("Physics strategy, data and simulation, reference selections, pass/fail fits, uncertainties, correction payloads, and histogram application", "body"),
        Spacer(1, 10 * mm),
        P(f"Analysis note date: {date.today().isoformat()}<br/>Collision energy: sqrt(s) = 13.6 TeV<br/>Data-taking year: 2025", "small"),
    ]
    cover = Table([[line, title]], colWidths=[10 * mm, 147 * mm], hAlign="LEFT")
    cover.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(cover)
    story.append(Spacer(1, 12 * mm))
    story.append(P(
        "<b>Abstract.</b> Four data-to-simulation efficiency corrections are measured for the 2025 all-hadronic stop analysis. "
        "The MET and photon trigger efficiencies are obtained with independent reference triggers. The 5-10 GeV veto-electron and loose-muon identification efficiencies are obtained from J/psi resonance tag-and-probe samples recorded with ParkingSingleMuon paths external to the measured dilepton pair. "
        "Each correction is stored as a correctionlib v2 json.gz payload with nominal, up, and down evaluations and is propagated to the final histogram weights.",
        "body",
    ))
    story.append(PageBreak())
    return story


def report_story_2025() -> list[Any]:
    global FIGURE_NUMBER
    FIGURE_NUMBER = 0
    met = DATA["met_result"]
    photon = DATA["photon_result"]
    electron = DATA["electron_result"]
    muon = DATA["muon_result"]
    ecfg = DATA["electron_config"]
    mcfg = DATA["muon_config"]
    integration = DATA["integration"]
    story = cover_story_2025()

    toc = TableOfContents()
    toc.levelStyles = [S["h2"], S["h3"]]
    story += [H1("Contents"), toc, PageBreak()]

    story += [H1("1 Physics purpose and analysis role")]
    story.append(P(
        "The signal and control-region event weights must describe the probability that an event present in simulation would also satisfy the online and offline requirements used in data. A scale factor is therefore defined as SF = epsilon(data) / epsilon(MC), with the numerator and denominator efficiencies evaluated in the same kinematic bin. The correction multiplies simulated events only; collision data are not reweighted by these factors."
    ))
    story.append(P(
        "This campaign closes four analysis-specific gaps. The MET and photon paths previously lacked measured Run-3 trigger corrections. The veto-electron and loose-muon object definitions extend down to 5 GeV, while the pre-existing official workflow payloads entered at 10 GeV and therefore clipped lower-pT objects to a boundary bin. The dedicated ID-only measurements replace that clipping in 5 < pT < 10 GeV and hand back to the official payload at 10 GeV."
    ))
    story.append(draw_measurement_logic())
    story.append(Spacer(1, 4 * mm))
    story.append(P(
        "The low-pT strategy follows the physical idea of CMS-DP-2023/081: a J/psi resonance gives substantially more low-pT dielectron signal than a Z resonance and makes the 5-7 GeV interval accessible. The present implementation is a 2025 analysis measurement, not a direct POG result, and future POG validation is needed."
    ))

    story += [H1("2 Samples and common event treatment")]
    story.append(styled_table(dataset_rows_2025(), [1.05 * inch, 1.7 * inch, 2.65 * inch, 1.05 * inch], font_size=6.0, long=True))
    story.append(H2("2.1 Data quality and simulation weighting"))
    story.append(P(
        "Collision data are restricted to the certified 2025 golden luminosity sections. The trigger measurements use EGamma0-3 or JetMET0-1 primary datasets according to the independent reference trigger. The low-pT measurements use ParkingSingleMuon0-15 in Run2025C-G. Duplicate run-luminosity-event triplets are removed across primary-dataset fragments before data histograms are summed."
    ))
    story.append(P(
        "Simulation is weighted by the signed generator weight and the 2025 pileup correction Collisions25_goldenJSON. The trigger samples use semileptonic ttbar or pT-binned gamma+jet simulation. The low-pT samples use Summer24 SPS double-J/psi simulation because these samples contain the measured dilepton resonance together with an independent muon side and match the official 2025 Prompt/Summer24 correction pairing."
    ))

    story += [H1("3 Trigger-efficiency measurements")]
    story.append(H2("3.1 MET trigger"))
    story.append(P(
        "The MET denominator is selected with single-electron paths HLT_Ele30/32/35/38/40_WPTight_Gsf in EGamma data and with the corresponding offline semileptonic topology in TTtoLNu2Q simulation. The probe is the logical OR of the PFMET and PFMETNoMu 120, 130, and 140 GeV tight paths. Because the reference is based on the electron leg rather than MET, it does not condition the denominator on the trigger efficiency being measured. The scale factor is tabulated versus missing transverse momentum from 100 to 800 GeV."
    ))
    met_plot = MET_DIR / "plots/2025_full"
    story += paired_figures(
        met_plot / "met_trigger_efficiency_2025.png",
        met_plot / "met_trigger_scale_factor_2025.png",
        "MET-trigger efficiency in data and simulation, and their ratio. Only llabel and rlabel carry plot-level annotation; the measurement definition is recorded in this caption.",
    )
    story.append(styled_table(trigger_bin_rows(met), [0.8 * inch, 0.75 * inch, 0.72 * inch, 0.72 * inch, 0.65 * inch, 0.62 * inch, 0.62 * inch, 0.62 * inch], font_size=5.9, long=True))

    story.append(H2("3.2 Photon trigger"))
    story.append(P(
        "The photon denominator is selected with the logical OR of PFHT180 through PFHT1050 in JetMET data. A medium-ID photon is reconstructed without imposing the photon HLT decision. The numerator additionally requires HLT_Photon175 or HLT_Photon200. Gamma+jet simulation supplies the corresponding efficiency. The use of PFHT as the reference makes the online photon decision the quantity under test rather than a precondition. Results are measured in five photon-pT intervals and four absolute-eta intervals, excluding the ECAL transition region."
    ))
    photon_plot = PHOTON_DIR / "plots/2025_full"
    story += paired_figures(
        photon_plot / "photon_trigger_efficiency_2025.png",
        photon_plot / "photon_trigger_scale_factor_2025.png",
        "Photon-trigger efficiencies and data-to-simulation scale factors measured with the independent PFHT reference.",
    )
    story.append(styled_table(trigger_bin_rows(photon, photon=True), [0.58 * inch, 0.58 * inch, 0.6 * inch, 0.66 * inch, 0.66 * inch, 0.58 * inch, 0.55 * inch, 0.55 * inch, 0.55 * inch], font_size=5.2, long=True))

    story += [H1("4 Low-pT tag-and-probe measurements")]
    story.append(H2("4.1 Why J/psi and why an external parking trigger"))
    story.append(P(
        "A low invariant mass does not by itself imply a low-pT probe. The analysis explicitly requires the probe to lie in 5 < pT < 10 GeV; the J/psi mass window supplies a narrow resonance with adequate signal yield in that kinematic interval. ParkingSingleMuon records the event through a muon that is distinct from the measured electron or muon pair. In data the event must fire HLT_Mu9_Barrel_L1HP10_IP6 or HLT_Mu10_Barrel_L1HP11_IP6 and contain a tight barrel reference muon with pT > 12 GeV. The identical offline external-muon topology is imposed on simulation, while the parking HLT decision itself is data-only because it is an acquisition condition rather than the efficiency under measurement."
    ))
    story.append(H2("4.2 Electron ID-only definition"))
    story.append(P(
        "The electron denominator is a reconstructed GSF electron in the ECAL fiducial region with conversion veto and at most one lost hit, without a cutBased or mini-isolation requirement. The passing definition is cutBased >= 1 only. Mini-isolation is deliberately absent from both the numerator and the exported correction. The measured bins are 5-7 and 7-10 GeV in |eta| intervals 0-0.8, 0.8-1.44, and 1.44-2.5; probes in the ECAL transition are removed. The endcap payload central value is fixed to unity by analysis policy and its uncertainty is derived from the measured fitted ratio."
    ))
    eplot = ELECTRON_DIR / "plots/2025_full"
    story += paired_figures(
        eplot / "efficiency.png",
        eplot / "scale_factor.png",
        "Electron ID-only efficiencies and scale factors. The probe pT and eta bins are defined in the text, not in a plot title.",
    )
    story.append(styled_table(tnp_rows_2025("electron"), [0.65 * inch, 0.68 * inch, 0.7 * inch, 0.7 * inch, 0.65 * inch, 0.72 * inch, 0.7 * inch, 0.62 * inch], font_size=5.8, long=True))

    story.append(H2("4.3 Muon ID-only definition"))
    story.append(P(
        "The muon denominator is a reconstructed tracker muon with 5 < pT < 10 GeV and |eta| < 2.4, without LooseID or mini-isolation. The numerator requires LooseID only. The measured J/psi tag is a distinct tight muon with pT > 5 GeV and no isolation condition; neither measured leg is trigger matched. The raw 20 MeV mass histogram is summed in adjacent pairs for the nominal 40 MeV fit binning. The final pT and eta binning is obtained only by exact summation of the measured pass/fail histograms, never by averaging fitted scale factors."
    ))
    mplot = MUON_DIR / "plots/2025_full"
    story += paired_figures(
        mplot / "efficiency.png",
        mplot / "scale_factor.png",
        "Muon LooseID-only efficiencies and scale factors from the disjoint external-reference topology.",
    )
    story.append(styled_table(tnp_rows_2025("muon"), [0.65 * inch, 0.68 * inch, 0.7 * inch, 0.7 * inch, 0.65 * inch, 0.72 * inch, 0.7 * inch, 0.62 * inch], font_size=5.8, long=True))

    story.append(H2("4.4 Simultaneous pass/fail resonance fit"))
    story.append(P(
        "For each probe bin, the passing and failing mass spectra are fitted simultaneously. The nominal resonance is a double-sided Crystal Ball line shape and the pass/fail samples share the core response unless that constraint is explicitly varied. Independent smooth backgrounds describe the two categories. A single signal efficiency parameter partitions the fitted resonance yield between pass and fail. The same construction is applied to data and simulation, and the fitted scale factor is the ratio of the two efficiencies."
    ))
    story.append(P(
        "The fit systematic envelope includes an alternate signal form, alternate background form, an independent pass/fail response, restricted mass windows, alternate mass binning, and a simulation-template variation. The nominal fit and every required variation must be finite. The fit plots show data, signal-plus-background fit, and background only; a shared legend is placed outside four square panels."
    ))
    efit = next((eplot / f"mass_fit_bin_{index:02d}.png" for index, item in enumerate(electron["bins"]) if item.get("fits")), None)
    mfit = next((mplot / f"mass_fit_bin_{index:02d}.png" for index, item in enumerate(muon["bins"]) if item.get("fits")), None)
    if efit is not None and mfit is not None:
        story += paired_figures(efit, mfit, "Representative simultaneous pass/fail fits for the electron and muon measurements. The ordinate is Events / 40 MeV.")

    story += [H1("5 Uncertainties and correction representation")]
    story.append(H2("5.1 Trigger uncertainties"))
    story.append(P(
        "Data efficiencies use the binomial count and a Wilson interval. Weighted simulation uses the effective number of entries derived from sumw and sumw2. The statistical uncertainty of epsilon(data)/epsilon(MC) is propagated from the two efficiencies. The pileup uncertainty is the larger absolute shift under the 2025 pileup up/down weights. Statistical and pileup terms are added in quadrature."
    ))
    story.append(H2("5.2 Low-pT fit and pileup uncertainties"))
    story.append(P(
        "For each low-pT bin, the statistical uncertainty is propagated from the fitted data and simulation efficiencies. The fit-model uncertainty is the largest displacement among the required alternate fits. The pileup term is the larger scale-factor shift from the simulation pileup up/down histograms. These terms are combined in quadrature. The electron endcap unity policy retains a symmetric measurement-derived uncertainty that covers both the raw fitted ratio's displacement from one and its statistical precision."
    ))
    payload_rows = [["Component", "Correction name", "Nominal range", "Payload SHA256"]]
    for component, correction_name in (
        ("met", "met_trigger_sf_genuine"),
        ("photon", "photon_trigger_sf"),
        ("electron", "veto_electron_id_5to10_sf"),
        ("muon", "loose_muon_id_5to10_sf"),
    ):
        payload = DATA[f"{component}_payload"]
        low, high = payload_range(payload)
        payload_rows.append([
            component,
            correction_name,
            f"{low:.4f}-{high:.4f}",
            sha256(PATHS[f"{component}_payload"]),
        ])
    story.append(styled_table(payload_rows, [0.8 * inch, 2.0 * inch, 1.0 * inch, 2.65 * inch], font_size=5.8))

    story += [H1("6 Application to analysis histograms")]
    story.append(P(
        "All four payloads are evaluated during the flat-intermediate histogram stage. The central value multiplies the nominal simulated event weight. Each payload also supplies a one-at-a-time Up and Down event-weight variation: met_trigger, photon_trigger, veto_electron_5to10, and loose_muon_5to10. The histogram execution contract hashes the 2025 payload files, so replacing a payload invalidates stale histogram products."
    ))
    story.append(P(
        "The low-pT handoff is open at both sides: the dedicated analysis payload is used only for selected objects with 5 < pT < 10 GeV. In that interval the previously clipped official electron or muon ID contribution is replaced by unity before the dedicated ID-only factor is multiplied. At pT >= 10 GeV the official correction is used and the dedicated factor is unity. This removes boundary clipping and prevents double counting."
    ))
    variation_rows = [["Nominal component", "Up variation", "Down variation"]]
    for component in integration["nominal_components"]:
        variation_rows.append([component, f"{component}Up", f"{component}Down"])
    story.append(styled_table(variation_rows, [2.15 * inch, 2.15 * inch, 2.15 * inch], font_size=7.0))
    story.append(P(
        "The integration validation requires every non-data dataset to record all four components as applied and requires all eight shifted histogram leaves. The campaign year is passed explicitly to the payload resolver, so 2025 histograms cannot silently load the 2024 AnalysisSF directory."
    ))

    story += [H1("7 Interpretation and validation statement")]
    story.append(P(
        "The scale factors quantify residual data-simulation efficiency differences in the measured phase space; they are not corrections to collision data and do not change the object definitions. The resulting uncertainty variations propagate these differences into the signal and control-region predictions. Their impact is expected to be largest in samples selected by the corresponding trigger and in the lost-lepton control region when a reconstructed lepton lies between 5 and 10 GeV."
    ))
    story.append(P(
        "The measurements use the full retained 2025 campaign after local recovery of transient storage failures, the declared fit variations, and explicit correctionlib domain checks. The methodology and numerical results are suitable for analysis-level use. Future POG validation is needed before interpreting these payloads as centrally endorsed CMS recommendations."
    ))

    story += [PageBreak(), H1("Appendix A: machine-readable provenance")]
    source_rows = [["Role", "Repository path", "SHA256"]]
    for role, key in (
        ("MET definition", "met_config"),
        ("MET result", "met_result"),
        ("Photon definition", "photon_config"),
        ("Photon result", "photon_result"),
        ("Electron definition", "electron_config"),
        ("Electron result", "electron_result"),
        ("Muon definition", "muon_config"),
        ("Muon result", "muon_result"),
        ("Histogram integration", "integration"),
        ("MET payload", "met_payload"),
        ("Photon payload", "photon_payload"),
        ("Electron payload", "electron_payload"),
        ("Muon payload", "muon_payload"),
    ):
        path = PATHS[key]
        source_rows.append([role, str(path.relative_to(REPO)), sha256(path)])
    story.append(styled_table(source_rows, [1.1 * inch, 3.35 * inch, 2.0 * inch], font_size=5.6, long=True))
    story.append(H2("A.1 Primary physics and implementation references"))
    references = [
        "CMS Collaboration, AN2019-016 v9, Run-2 all-hadronic stop-search strategy baseline.",
        "CMS Collaboration, CMS-DP-2023/081, Low-pT Electron ID scale factors at 13 TeV using J/psi events.",
        "The 2025 measurement configuration, adopted result, and correctionlib files listed above.",
        "The 2025 golden luminosity JSON and Collisions25_goldenJSON pileup correction used by the histogram builders.",
    ]
    for index, reference in enumerate(references, 1):
        story.append(P(f"[{index}] {reference}", "small"))
    story.append(Spacer(1, 5 * mm))
    story.append(P("End of report.", "small"))
    return story


def main() -> int:
    global REPORT_YEAR, OUTPUT, PATHS, DATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", choices=("2024", "2025"), default="2024")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    REPORT_YEAR = args.year
    if REPORT_YEAR == "2025":
        PATHS = paths_for_2025()
    OUTPUT = (
        args.output.resolve()
        if args.output is not None
        else REPO / f"output/pdf/analysis_sf_measurements_{REPORT_YEAR}_physics_report.pdf"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    missing = [str(path) for path in PATHS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing report inputs:\n" + "\n".join(missing))
    DATA = {
        name: load_json(path)
        for name, path in PATHS.items()
        if path.suffix in {".json", ".gz"}
    }
    document = PhysicsReportDoc(str(OUTPUT))
    document.multiBuild(report_story_2025() if REPORT_YEAR == "2025" else report_story())
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
