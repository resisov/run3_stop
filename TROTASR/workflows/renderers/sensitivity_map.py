"""Render compact, immutable group-freeze results; no fit or model changes."""
import hashlib
import json
import textwrap

from TROTASR.utils.paths import internal_path
from TROTASR.utils.io import write_json, sha256
from TROTASR.workflows.sensitivity_map import LABELS, COMPETING


def render(source, grid, output):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm, TwoSlopeNorm
    from matplotlib.lines import Line2D
    import mplhep as hep

    source, grid, output = map(internal_path, (source, grid, output))
    raw = (source/'results.json').read_bytes()
    result = json.loads(raw)
    if result['contract']['manifest_sha256'] != sha256(grid):
        raise ValueError('Sensitivity results and requested mass grid differ')
    if result['schema'] != 'trotasr_sensitivity_map_results_v1':
        raise ValueError('Unsupported sensitivity result schema')
    output.mkdir(parents=True, exist_ok=True)
    # Preserve exactly the data used for these figures, even while fits proceed.
    (output/'input_results.json').write_bytes(raw)
    hep.style.use('CMS')
    colors = plt.get_cmap('tab10').colors
    artifacts = {}
    coverage = {}

    def save(fig, stem):
        for ext in ('png', 'pdf'):
            path = output/(stem+'.'+ext)
            fig.savefig(path, dpi=180)
            artifacts[path.name] = sha256(path)
        plt.close(fig)

    for model in ('T2tt', 'T2bW', 'T2tb'):
        rows = [p for p in result['points'] if p['model'] == model]
        valid = [p for p in rows if p['complete_competing_groups']]
        coverage[model] = dict(complete=len(valid), requested=len(rows))
        if not valid:
            continue
        xs = [p['mStop'] for p in rows]
        ys = [p['mLSP'] for p in rows]
        xlim = (min(xs)-35, max(xs)+50)
        ylim = (-25, max(ys)+60)
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.subplots_adjust(left=.09, right=.64, bottom=.12, top=.88)
        ax.scatter(xs, ys, c='.88', s=28, marker='s')
        ax.scatter([p['mStop'] for p in valid], [p['mLSP'] for p in valid],
            c=[COMPETING.index(p['dominant_group']) for p in valid],
            s=36, marker='s', cmap=ListedColormap(colors),
            norm=BoundaryNorm(np.arange(-.5, 10.5), 10))
        handles = [Line2D([], [], marker='s', linestyle='', color=colors[i],
                         label=textwrap.fill(LABELS[g], 32)) for i, g in enumerate(COMPETING)]
        handles.append(Line2D([], [], marker='s', linestyle='', color='.88', label='Pending / incomplete'))
        ax.legend(handles=handles, bbox_to_anchor=(1.03, 1), loc='upper left',
                  fontsize=11, frameon=False, borderaxespad=0, labelspacing=1.05)
        ax.set(xlabel=r'$m_{\widetilde{t}}$ (GeV)',
               ylabel=r'$m_{\widetilde{\chi}^0_1}$ (GeV)', xlim=xlim, ylim=ylim)
        hep.cms.label(llabel='Work in progress',
                      rlabel=r'220.66 fb$^{-1}$ (13.6 TeV)', data=True, fontsize=17, ax=ax)
        fig.suptitle(model+' | largest conditional limit improvement', fontsize=19, y=.98)
        save(fig, model+'_dominant_group')

        fig, axes = plt.subplots(3, 4, figsize=(20, 13), sharex=True, sharey=True)
        fig.subplots_adjust(left=.06, right=.89, bottom=.08, top=.90, wspace=.13, hspace=.35)
        norm = TwoSlopeNorm(vmin=-10, vcenter=0, vmax=40)
        for ax, group in zip(axes.flat, LABELS):
            selected = [p for p in rows if 'improvement_percent' in p['groups'][group]]
            ax.scatter(xs, ys, c='.9', s=12, marker='s')
            if selected:
                ax.scatter([p['mStop'] for p in selected], [p['mLSP'] for p in selected],
                    c=[p['groups'][group]['improvement_percent'] for p in selected],
                    s=19, marker='s', cmap='RdBu_r', norm=norm)
            ax.set_title(textwrap.fill(LABELS[group], 29), fontsize=13)
            ax.set(xlim=xlim, ylim=ylim)
            ax.tick_params(labelsize=11)
        axes.flat[-1].axis('off')
        cax = fig.add_axes([.92, .17, .015, .61])
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='RdBu_r'), cax=cax,
                     label='Expected median limit improvement (%)', extend='both')
        fig.suptitle(model+' | 2024+2025 high-dM + low-dM | conditional group freezes', fontsize=21, y=.98)
        fig.supxlabel(r'$m_{\widetilde{t}}$ (GeV)', fontsize=22)
        fig.supylabel(r'$m_{\widetilde{\chi}^0_1}$ (GeV)', fontsize=22)
        save(fig, model+'_group_improvements')

    manifest = dict(status='generated_pending_visual_QA', artifacts=artifacts,
        results_sha256=hashlib.sha256(raw).hexdigest(), coverage=coverage,
        complete_points=result['complete_points'], requested=len(result['points']),
        interpolation='none; actual mass-point markers only', non_additive=True,
        ranking='Largest measured conditional improvement; close values need not establish a unique hierarchy.',
        scope=result['contract']['scope'])
    write_json(output/'plot_manifest.json', manifest)
    return manifest
