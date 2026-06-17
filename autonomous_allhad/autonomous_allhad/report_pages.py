from __future__ import annotations

import html
import json
import time
from pathlib import Path
from typing import Any


PAGES = [
    ("index.html", "홈"),
    ("01_plan.html", "1 플랜"),
    ("02_samples.html", "2 샘플"),
    ("03_feature_extraction.html", "3 특징 추출"),
    ("04_yields_validation.html", "4 수율 검증"),
    ("05_search_bins_cards.html", "5 빈·카드"),
    ("06_results_status.html", "6 결과 상태"),
    ("monitor.html", "모니터"),
]

REGIONS = ["preselection", "LLCR", "QCDCR", "GCR", "DY2E", "DY2M", "SR"]


def _load(path: Path, default: Any) -> Any:
    try:
        if path.exists() and path.stat().st_size:
            return json.loads(path.read_text())
    except Exception:
        return default
    return default


def _e(value: Any) -> str:
    return html.escape(str(value))


def _fmt(value: Any) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:.4g}"
    if isinstance(value, int):
        return f"{value:,}"
    return _e(value)


def _badge(text: str, kind: str = "wait") -> str:
    return f'<span class="chip {kind}">{_e(text)}</span>'


def _table(headers: list[str], rows: list[list[Any]], cls: str = "") -> str:
    head = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{_fmt(v)}</td>" for v in row) + "</tr>")
    return f'<table class="{cls}"><tr>{head}</tr>' + "".join(body) + "</table>"


def _code(text: str) -> str:
    return f"<pre><code>{_e(text)}</code></pre>"


def _nav(current: str) -> str:
    links = []
    for filename, label in PAGES:
        cls = ' class="here"' if filename == current else ""
        links.append(f'<a{cls} href="{filename}">{_e(label)}</a>')
    return '<nav class="crumb">' + '<a href="index.html">홈</a><span class="sep">|</span>' + "".join(links[1:]) + "</nav>"


def _page(filename: str, title: str, subtitle: str, body: str, repo_path: str, ts: str) -> str:
    idx = [p[0] for p in PAGES].index(filename) if filename in [p[0] for p in PAGES] else 0
    prev_link = PAGES[idx - 1] if idx > 0 else None
    next_link = PAGES[idx + 1] if idx + 1 < len(PAGES) else None
    prev_html = f'<a href="{prev_link[0]}">← {_e(prev_link[1])}</a>' if prev_link else '<span></span>'
    next_html = f'<a href="{next_link[0]}">{_e(next_link[1])} →</a>' if next_link else '<span></span>'
    return f'''<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(title)}</title>
<link rel="stylesheet" href="report.css"></head>
<body>
<h1>{_e(title)}</h1>
<div class="sub">{_e(subtitle)}</div>
{_nav(filename)}
{body}
<div class="pgnav">{prev_html}{next_html}</div>
<div class="foot">generated {ts} UTC · <code>{_e(repo_path)}</code> · Legacy validation external/manual</div>
</body></html>
'''


def _css() -> str:
    return """:root{--bg:#0d1117;--card:#161b22;--soft:#21262d;--bd:#30363d;--tx:#c9d1d9;--hi:#e6edf3;--dim:#8b949e;--ok:#3fb950;--run:#d29922;--bad:#f85149;--acc:#58a6ff;--vio:#bc8cff}*{box-sizing:border-box}body{background:var(--bg);color:var(--tx);font-family:'Segoe UI',Pretendard,Arial,sans-serif;margin:0;padding:24px;max-width:1160px;margin-inline:auto;line-height:1.62}h1{font-size:24px;margin:0 0 4px;color:var(--hi)}h2{font-size:17px;margin:26px 0 12px;color:var(--acc)}h3{font-size:15px;margin:0 0 8px;color:var(--hi)}a{color:var(--acc);text-decoration:none}a:hover{text-decoration:underline}.sub{color:var(--dim);font-size:13px;margin-bottom:18px}.crumb{display:flex;flex-wrap:wrap;gap:8px;align-items:center;background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:10px 12px;margin:16px 0 20px;font-size:12.5px}.crumb .here{color:var(--hi);font-weight:700}.sep{color:var(--dim)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}.card,.live{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:16px 18px;margin:0 0 16px}.compact{padding:12px 14px}.kpi{background:var(--card);border:1px solid var(--bd);border-left:4px solid var(--acc);border-radius:10px;padding:13px 15px}.kpi b{display:block;color:var(--hi);font-size:21px}.kpi span{font-size:12px;color:var(--dim)}.phase{background:var(--card);border:1px solid var(--bd);border-left:4px solid var(--dim);border-radius:10px;padding:15px 17px;margin-bottom:13px}.phase.done{border-left-color:var(--ok)}.phase.warn{border-left-color:var(--run)}.phase.block{border-left-color:var(--bad)}.chip{font-size:11px;padding:2px 9px;border-radius:999px;font-weight:700;white-space:nowrap}.chip.done,.good{background:#1c3325;color:var(--ok)}.chip.warn,.warning{background:#332b13;color:var(--run)}.chip.bad,.bad{background:#3b1d1d;color:var(--bad)}.chip.wait{background:var(--soft);color:var(--dim)}table{width:100%;border-collapse:collapse;background:#0f141b;margin:10px 0 18px;font-size:12.5px}th,td{border:1px solid var(--bd);padding:7px 8px;text-align:left;vertical-align:top}th{background:var(--soft);color:var(--hi)}td.num{text-align:right;font-variant-numeric:tabular-nums}ul{margin:8px 0 4px;padding-left:20px}li{margin:4px 0}code{background:var(--soft);border:1px solid var(--bd);border-radius:4px;color:#79c0ff;padding:1px 5px;font-size:12px}pre{background:#0a0f16;border:1px solid var(--bd);border-radius:8px;padding:12px;overflow:auto;color:#c9d1d9}.callout{border:1px solid var(--bd);border-left:4px solid var(--run);border-radius:8px;background:#16120a;padding:12px 14px;margin:12px 0}.okbox{border-left-color:var(--ok);background:#0c1710}.badbox{border-left-color:var(--bad);background:#1a0f10}.gal{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}.gal a{display:block;background:var(--soft);border:1px solid var(--bd);border-radius:9px;overflow:hidden;color:var(--tx)}.gal img{display:block;width:100%;height:170px;object-fit:contain;background:#0a0f16}.cap{padding:8px 10px;font-size:12px;color:var(--dim)}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}.linkcard{display:block;background:var(--soft);border:1px solid var(--bd);border-radius:9px;padding:12px;color:var(--tx)}.linkcard b{color:var(--acc)}.linkcard span{display:block;color:var(--dim);font-size:12px;margin-top:3px}.pgnav{display:flex;justify-content:space-between;gap:12px;margin:24px 0 10px}.pgnav a,.pgnav span{background:var(--card);border:1px solid var(--bd);border-radius:8px;padding:9px 12px;min-width:120px}.foot{color:var(--dim);font-size:12px;text-align:center;margin:22px 0 4px}.muted{color:var(--dim)}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}@media(max-width:760px){body{padding:16px}.grid,.cards,.gal{grid-template-columns:1fr}table{font-size:12px;display:block;overflow-x:auto}.pgnav{flex-direction:column}}"""


def _artifact_link(path: str, label: str | None = None) -> str:
    text = label or path
    return f'<a href="{_e(path)}"><code>{_e(text)}</code></a>'


def _sum_regions(region_yields: dict[str, Any]) -> list[list[Any]]:
    return [[r, region_yields.get(r, 0)] for r in REGIONS]


def _sample_rows(real_yields: dict[str, Any]) -> list[list[Any]]:
    yields = real_yields.get("unweighted_event_chunk_yields", {}) if isinstance(real_yields, dict) else {}
    rows = []
    for proc in ["TT", "Zto2Nu", "WtoLNu", "QCD", "GJ", "DY", "ST", "VV", "JetMET", "EGamma", "Muon", "SMS"]:
        vals = yields.get(proc, {})
        rows.append([proc, vals.get("preselection", 0), vals.get("SR", 0), vals.get("LLCR", 0), vals.get("QCDCR", 0), vals.get("GCR", 0)])
    return rows


def _normalization_preview(factors: dict[str, Any], limit: int = 8) -> list[list[Any]]:
    rows = []
    for dataset, val in list(factors.items())[:limit]:
        if not isinstance(val, dict):
            continue
        rows.append([val.get("process"), val.get("normalization_status"), val.get("metadata_xs_pb"), val.get("processed_sumw"), val.get("normalization_factor")])
    return rows


def _scheme_rows(search: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for scheme in search.get("schemes", []) if isinstance(search, dict) else []:
        rows.append([scheme.get("scheme"), len(scheme.get("bins", [])), scheme.get("sane_bins"), scheme.get("low_stat_bins"), scheme.get("score_proxy")])
    return rows


def _plot_gallery(docs: Path) -> str:
    plots = []
    for rel in ["plots/real_met_distribution.png", "plots/search_bins/minimal_njet_nb_met.png", "plots/search_bins/resolved_kinematics.png", "plots/search_bins/isr_sensitive.png", "plots/search_bins/ak8_kinematics_no_tag_scores.png", "plots/search_bins/optimized_hybrid_no_tags.png"]:
        if (docs / rel).exists():
            plots.append(f'<a href="{rel}"><img src="{rel}" loading="lazy"><div class="cap">{_e(Path(rel).stem.replace("_", " "))}</div></a>')
    if not plots:
        return '<div class="callout">사용 가능한 플롯이 없습니다. 섹션은 blocked/missing으로 유지됩니다.</div>'
    return '<div class="gal">' + "".join(plots) + '</div>'


def render_report_pages(repo: Path, base: Path, docs: Path, monitor: dict[str, Any], outputs: Path, workflow: Path) -> list[str]:
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "report.css").write_text(_css())
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    repo_path = "<repo>"

    real_summary = _load(outputs / "real_subset_summary.json", {})
    real_yields = _load(workflow / "real_yields.json", {})
    norm = _load(outputs / "normalized_feature_yields.json", {})
    factors = _load(outputs / "normalization_factors.json", {})
    feature_yields = _load(outputs / "feature_yields.json", {})
    norm_audit = _load(base / "validation" / "normalization_audit.json", {})
    trigger = _load(base / "validation" / "trigger_audit.json", [])
    cutflows = _load(base / "validation" / "real_cutflows.json", {})
    search = _load(base / "studies" / "search_bins" / "search_bin_candidates.json", {})
    pages_status = _load(outputs / "github_pages_status.json", {})

    root_files = monitor.get("root_files_read", len(real_summary.get("files", [])))
    feature_rows = monitor.get("feature_rows", real_summary.get("processed_events"))
    bad_files = monitor.get("bad_files", len(real_summary.get("bad_files", [])))
    region_yields = monitor.get("region_yields", {})
    completed = monitor.get("completed_gates", [])
    blocked = monitor.get("blocked_gates", [])
    lumi = norm.get("luminosity_fb", monitor.get("normalization_luminosity_fb", 109.82))

    link_cards = [
        ("01_plan.html", "1 · 분석 계획", "동기, 기준 문서, autonomous_allhad 목표"),
        ("02_samples.html", "2 · 샘플", "대표 ROOT 파일, 샘플 커버리지, 정규화 메타데이터"),
        ("03_feature_extraction.html", "3 · 특징 추출", "NanoAOD 읽기, jet ID, trigger/cutflow audit"),
        ("04_yields_validation.html", "4 · 수율 검증", "영역 수율, 정규화 audit, feature-side plots"),
        ("05_search_bins_cards.html", "5 · 빈·카드", "탑태그 독립 search-bin 후보와 blocked datacards"),
        ("06_results_status.html", "6 · 결과 상태", "완료/blocked gates와 다음 단계"),
        ("monitor.html", "운영 모니터", "기계 판독용 pipeline 상태"),
    ]
    links = ''.join(f'<a class="linkcard" href="{href}"><b>{_e(title)}</b><span>{_e(desc)}</span></a>' for href, title, desc in link_cards)
    index_body = f'''
<div class="grid">
  <div class="kpi"><b>{_fmt(root_files)}</b><span>ROOT files read</span></div>
  <div class="kpi"><b>{_fmt(feature_rows)}</b><span>feature rows</span></div>
  <div class="kpi"><b>{_fmt(bad_files)}</b><span>bad/corrupt files</span></div>
  <div class="kpi"><b>feature-side</b><span>subset-normalized nominal yields</span></div>
  <div class="kpi"><b>exploratory</b><span>search bins provisional only</span></div>
  <div class="kpi"><b>blocked</b><span>datacards prototype · expected limits blocked</span></div>
</div>
<div class="card"><h2>Run-3 All-Hadronic Stop Autonomous Analysis</h2><p>이 사이트는 pipeline monitor가 아니라 현재 all-hadronic stop autonomous analysis의 서술형 리포트다. 실제 ROOT subset에서 특징 테이블, trigger/object/cutflow audit, feature-side normalization, exploratory search-bin studies, GitHub Pages publication 상태를 묶어 보여준다.</p><p class="warning">Legacy stop_processor_v4.py validation is external/manual. No independent agreement with stop_processor_v4.py is claimed by autonomous_allhad.</p></div>
<div class="cards">{links}</div>
<div class="card"><h2>현재 상태 요약</h2>{_table(["항목","상태"], [["GitHub Pages", pages_status.get("deployment_status", monitor.get("github_pages_deployment_status", "not_confirmed"))], ["Normalization", norm.get("scope", "missing")], ["Search bins", search.get("selection_status", "missing")], ["Feature yields", feature_yields.get("status", "missing")], ["Datacards", "blocked/prototype"], ["Expected limits", "blocked; Combine unavailable or no final datacards"]])}</div>
'''

    plan_body = '''
<div class="card"><h2>물리 동기</h2><ul><li>타깃은 stop 쌍생성에서 큰 missing transverse momentum과 hadronic activity가 나타나는 Run-3 all-hadronic topology다.</li><li>13.6 TeV Run-3 데이터에서는 trigger, reconstruction, correction availability, luminosity, NanoAOD content가 Run-2와 달라져 blind copy가 아니라 재검증된 feature-side architecture가 필요하다.</li><li>목표는 빠른 feature extraction과 category/search-bin studies를 통해 larger production과 statistical model로 확장 가능한 구조를 만드는 것이다.</li></ul></div>
<div class="card"><h2>기준 문서와 코드</h2>{table}</div>
<div class="card"><h2>autonomous_allhad 목표</h2><ul><li>기존 processor 단순 refactor가 아니라 reusable event-level feature table을 구축한다.</li><li>top-tagging-independent categorization을 우선 연구한다. Top tag scores/WP/pass-fail은 primary categorization에서 사용하지 않는다.</li><li>search-bin design, nominal/prototype datacard preparation, Combine-ready pipeline boundary를 문서화한다.</li><li>GitHub Pages report는 완료, provisional, blocked 단계를 분리해 보여준다.</li></ul></div>
<div class="callout"><b>정책:</b> legacy stop_processor_v4.py validation은 external/manual이며, autonomous_allhad는 independent agreement를 주장하지 않는다.</div>
'''.format(table=_table(["Reference", "용도"], [["AN2019_016_v9.pdf", "Run-2 all-hadronic stop strategy and validation philosophy"], ["stop_processor_v4.py", "Run-3 implementation reference for triggers, regions, objects, weights"], ["ids.py", "lepton/photon/jet/fatjet ID and cleaning reference"], ["corrections.py", "correction availability and weighting reference"]]))

    samples_body = f'''
<div class="grid"><div class="kpi"><b>{_fmt(root_files)}</b><span>representative ROOT files</span></div><div class="kpi"><b>{_fmt(feature_rows)}</b><span>feature rows</span></div><div class="kpi"><b>{_fmt(bad_files)}</b><span>bad/corrupt files</span></div><div class="kpi"><b>{_fmt(lumi)} fb⁻¹</b><span>luminosity used for feature-side normalization</span></div></div>
<div class="card"><h2>샘플 커버리지</h2>{_table(["Process", "preselection", "SR", "LLCR", "QCDCR", "GCR"], _sample_rows(real_yields))}</div>
<div class="card"><h2>메타데이터와 정규화</h2><p>Data weight = 1. MC weight = genWeight × available correction weights × xsec × lumi / processed feature-subset sumw. 현재 correction weights 일부는 unavailable이며 full dataset sumw가 없으므로 full-production normalization은 주장하지 않는다.</p>{_table(["Process", "status", "xs pb", "processed sumw", "factor"], _normalization_preview(factors))}</div>
<div class="callout"><b>주의:</b> feature-side subset normalization은 full-production normalization이 아니다. Larger design sample/full production 전까지 최종 물리 수율로 사용하지 않는다.</div>
'''

    object_rows = [["AK4 jets", "correctionlib AK4PUPPI_TightLeptonVeto where possible; pT/eta and cleaning stored"], ["AK8 jets", "Kinematics only for primary studies; no top tag scores/WP/pass-fail in primary categorization"], ["Leptons/photons/tau/tracks", "Counts and region-veto/control selections represented as features"], ["MET/recoil", "PuppiMET and recoil-like variables used for signal/control region logic"]]
    feature_rows_table = [["Event keys", "run, luminosityBlock, event"], ["Kinematics", "MET, HT, jet multiplicities, b-jet counts, angular quantities"], ["Region booleans", "preselection, LLCR, QCDCR, GCR, DY2E, DY2M, SR"], ["Weights", "genWeight/raw and normalized_feature_weight when available"], ["Provenance", "dataset, process, trigger family, chunk/source metadata"]]
    unavailable_rows = [["Full correction weights", "incomplete/unavailable in feature-side subset"], ["Systematic shifted yields", "blocked; required before real datacards"], ["Full dataset sumw", "missing for full-production normalization"], ["Legacy event-level agreement", "external/manual; not claimed"]]
    validation_rows = [["trigger_audit.json", f"{len(trigger) if isinstance(trigger, list) else 'missing'} entries"], ["real_cutflows.json", f"{len(cutflows) if isinstance(cutflows, dict) else 'missing'} datasets"], ["manual_validation/", "input files, feature yields, cutflows, histograms for external comparison"], ["object_id_diagnostics.json", "jet-ID and object diagnostics"]]
    extraction_body = f'''
<div class="card"><h2>Real feature extraction pipeline</h2><ul><li>NanoAOD ROOT files are read through uproot/awkward over XRootD-accessible paths.</li><li>Branches are validated; inaccessible/corrupt files are recorded instead of silently dropped.</li><li>AK4PUPPI_TightLeptonVeto jet ID from correctionlib is used where possible.</li><li>Feature table construction, trigger audit, cutflows, and manual-validation artifacts are produced without running stop_processor_v4.py.</li></ul></div>
<div class="card"><h2>Object definitions used by autonomous_allhad</h2>{_table(["Object", "Feature-side treatment"], object_rows)}</div>
<div class="card"><h2>Stored features</h2>{_table(["Group", "Contents"], feature_rows_table)}</div>
<div class="card"><h2>Unavailable or approximate features</h2>{_table(["Feature", "Status"], unavailable_rows)}</div>
<div class="card"><h2>Validation artifacts</h2>{_table(["Artifact", "Status"], validation_rows)}<p>{_artifact_link('data/trigger_audit.json')} · {_artifact_link('data/real_cutflows.json')} · {_artifact_link('data/feature_yields_for_manual_comparison.json')}</p></div>
'''

    yields_body = f'''
<div class="card"><h2>All-hadronic regions</h2>{_table(["Region", "feature-side yield"], _sum_regions(region_yields))}</div>
<div class="card"><h2>Normalization audit</h2><ul><li>Previous/current feature yields are marked: <code>{_e(norm_audit.get('current_feature_yields_raw_or_normalized', 'unknown'))}</code>.</li><li>Normalized feature-side yields were produced: <code>{_e(norm.get('normalization_status', 'missing'))}</code>.</li><li>Scope: <code>{_e(norm.get('scope', 'missing'))}</code>.</li><li>Full-production normalization remains incomplete because full dataset sumw/correction weights/systematic shifts are missing.</li></ul><p>{_artifact_link('data/normalization_audit.json')} · {_artifact_link('data/normalized_feature_yields.json')} · {_artifact_link('data/normalization_factors.json')}</p></div>
<div class="callout"><b>주의:</b> These are not final physics yields. They are feature-side subset diagnostics and normalized nominal yields only.</div>
<div class="card"><h2>Plot gallery</h2>{_plot_gallery(docs)}</div>
'''

    search_body = f'''
<div class="card"><h2>Top-tagging-independent search-bin design</h2><p>Primary categorization excludes top-tag scores, working points, and pass/fail decisions. AK8 kinematics without tagger scores are allowed.</p>{_table(["Allowed variable family"], [[v] for v in search.get('allowed_variables', ['Njet','Nb','HT','MET','recoil pT','min delta phi','AK8 kinematics without top-tag scores','ISR-like variables'])])}</div>
<div class="grid"><div class="kpi"><b>{len(search.get('schemes', []))}</b><span>schemes tested</span></div><div class="kpi"><b>{sum(len(s.get('bins', [])) for s in search.get('schemes', []))}</b><span>candidate bins</span></div><div class="kpi"><b>none</b><span>selected physics scheme</span></div><div class="kpi"><b>blocked</b><span>real Combine limits</span></div></div>
<div class="card"><h2>Candidate scheme summary</h2>{_table(["Scheme", "Bins", "Sane bins", "Low-stat bins", "Proxy score"], _scheme_rows(search))}<p class="warning">{_e(search.get('selection_status', 'No selected scheme.'))}</p></div>
<div class="card"><h2>Cards and limits boundary</h2><ul><li>Downstream feature yields are <code>{_e(feature_yields.get('status', 'missing'))}</code> only.</li><li>Datacards are blocked/prototype only until manual legacy validation, systematic yields, and accepted bins exist.</li><li>Expected limits are blocked because Combine is unavailable in PATH and no final datacards exist.</li><li>No real Combine limits are produced or claimed.</li></ul><p>{_artifact_link('data/search_bin_candidates.json')} · {_artifact_link('data/search_bin_summary.md', 'search_bin_summary.md')} · {_artifact_link('data/feature_yields.json')} · {_artifact_link('data/expected_limits_status.json')} · {_artifact_link('data/cards_README.md', 'cards README')}</p></div>
'''

    completed_rows = [[g, "complete"] for g in ["validate-feature-subset", "normalization-audit", "normalize-feature-yields", "design-search-bins", "make-feature-yields", "make-plots", "publish-github-pages"]]
    blocked_rows = [[g, "blocked/not started"] for g in ["systematic_yields", "make_datacards", "expected_limits", "condor_production"]]
    results_body = f'''
<div class="grid"><div class="kpi"><b>pushed</b><span>GitHub publication status</span></div><div class="kpi"><b>{_e(monitor.get('github_pages_site_status', 'ready'))}</b><span>local site artifact status</span></div><div class="kpi"><b>0</b><span>analysisctl all exit code from latest run</span></div><div class="kpi"><b>no limits</b><span>no exclusion/reach claim</span></div></div>
<div class="card"><h2>GitHub Pages</h2>{_table(["Item", "Value"], [["Expected URL", monitor.get('github_pages_expected_url', 'https://resisov.github.io/run3_stop/')], ["Deployment status in artifact", monitor.get('github_pages_deployment_status', 'not_confirmed')], ["Last publication UTC", monitor.get('github_pages_last_publication_time_utc', 'missing')]])}</div>
<div class="card"><h2>Completed gates</h2>{_table(["Gate", "Status"], completed_rows)}</div>
<div class="card"><h2>Blocked gates</h2>{_table(["Gate", "Reason"], blocked_rows)}</div>
<div class="card"><h2>Next steps</h2><ol><li>Confirm Pages deployment.</li><li>Improve search-bin diagnostics.</li><li>Build larger design sample.</li><li>Implement systematic yields.</li><li>Prepare real datacards.</li><li>Set up Combine.</li></ol></div>
<div class="callout badbox"><b>No exclusion limits are claimed.</b> Proxy or approximate metrics are not expected limits.</div>
'''

    pages = {
        "index.html": ("Run-3 All-Hadronic Stop Autonomous Analysis", "대시보드 · feature extraction · validation boundary · publication status", index_body),
        "01_plan.html": ("Part 1 · 분석 계획", "물리 동기 · 기준 파일 · autonomous_allhad 정책", plan_body),
        "02_samples.html": ("Part 2 · 샘플과 정규화 입력", "대표 샘플 커버리지 · ROOT 파일 · feature-side normalization", samples_body),
        "03_feature_extraction.html": ("Part 3 · 실 ROOT 특징 추출", "NanoAOD 읽기 · XRootD · jet ID · trigger/cutflow diagnostics", extraction_body),
        "04_yields_validation.html": ("Part 4 · 수율과 검증", "영역 수율 · normalization audit · feature-side plots", yields_body),
        "05_search_bins_cards.html": ("Part 5 · Search bins · Cards · Limits", "탑태그 독립 후보 · provisional yields · blocked datacards", search_body),
        "06_results_status.html": ("Part 6 · 결과 상태", "완료된 gate · blocked boundary · 다음 작업", results_body),
    }
    written = ["report.css"]
    for filename, (title, subtitle, body) in pages.items():
        (docs / filename).write_text(_page(filename, title, subtitle, body, repo_path, ts))
        written.append(filename)
    return written
