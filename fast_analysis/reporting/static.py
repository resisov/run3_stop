from __future__ import annotations

import html
from pathlib import Path

from ..config.defaults import DEFAULTS
from ..env import collect_environment
from ..paths import PathKind, PathPolicy, configure_eos_runtime_env


def render_index(output=DEFAULTS.report_path, dry_run=False):
    policy = PathPolicy.default()
    output_path = policy.resolve(output, PathKind.OUTPUT)
    env = collect_environment(dry_run=True)
    body = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Fast Stop Analysis Prototype</title>
<style>body{{font-family:Arial,sans-serif;margin:24px;max-width:980px}}code,pre{{background:#f2f4f8;padding:2px 4px}}.tag{{display:inline-block;border:1px solid #999;padding:2px 6px;margin:2px}}</style></head>
<body>
<h1>Run-3 All-Hadronic Stop Fast Analysis</h1>
<p>{''.join(f'<span class="tag">{html.escape(tag)}</span>' for tag in DEFAULTS.maturity)}</p>
<h2>Status</h2><p>Architecture scaffold and EOS policy gate are in place. No full campaign has been submitted.</p><p><b>Legacy regression is currently validated only for cat2_LLCR_highDeltaM. Other regions, including cat1_preselection, are not yet accepted as valid references.</b></p>
<h2>Storage Choice</h2><p>Prototype default is Parquet, with flat ROOT retained as a benchmark fallback.</p>
<h2>Environment</h2><pre>{html.escape(str(env))}</pre>
<h2>Next Milestone</h2><p>Run the three-file architecture benchmark from an EOS-resident manifest, then fill the compact skim implementation against the mapping document.</p>
</body></html>
"""
    if dry_run:
        return output_path
    configure_eos_runtime_env(DEFAULTS.output_root, dry_run=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(body)
    tmp.replace(output_path)
    return output_path
