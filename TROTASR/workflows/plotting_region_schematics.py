"""Original AN TikZ schematics with only the approved Mixed topology update.

No analysis ROOT, event processing, AN source write, or external analysis input.
LaTeX/Poppler are software runtimes. Visual QA is a separate, mandatory step.
"""
import argparse
import hashlib
from pathlib import Path
import re
import subprocess
if __package__ in (None, ''):
    from _bootstrap import bootstrap
    bootstrap()
from TROTASR.utils.paths import ROOT, internal_path
from TROTASR.utils.io import read_json, write_json, sha256


def render(output):
    output = internal_path(output)
    sources = read_json(ROOT/'jsons/region_schematic_sources.json')
    contract = dict(source_manifest_sha256=sha256(ROOT/'jsons/region_schematic_sources.json'),
        code_sha256=sha256(Path(__file__)), selection_sha256=sha256(ROOT/'jsons/selection_contract.json'))
    receipt = output/'plot_manifest.json'
    if output.exists():
        previous = read_json(receipt)
        if previous['contract'] != contract or any(sha256(output/n) != d for n,d in previous['files'].items()):
            raise ValueError('Changed or incomplete schematic output must be retained')
        return previous
    selection = read_json(ROOT/'jsons/selection_contract.json')['region_topology']
    if (selection['highdm']['tag_union'] != 'Nbst >= 1 || Nmix >= 1 || Nres >= 1 || Nw >= 1'
            or selection['lowdm']['topology_veto'] != 'Nbst == 0 && Nmix == 0 && Nres == 0 && Nw == 0'):
        raise ValueError('Changed approved topology definition')
    for row in sources['sources']:
        path = internal_path(ROOT/row['target'])
        text = path.read_text()
        if text.count(row['after']) != 1 or sha256(path) != row['target_sha256']:
            raise ValueError('Changed schematic source')
        restored = text.replace(row['after'], row['before'])
        if hashlib.sha256(restored.encode()).hexdigest() != row['source_sha256']:
            raise ValueError('A change beyond the approved topology condition was introduced')
        if any(token in text for token in ('\\input', '\\include', '\\write18')):
            raise ValueError('External analysis input/command in schematic')
    output.mkdir(parents=True)
    plots = []
    for row in sources['sources']:
        source = internal_path(ROOT/row['target']); name = source.stem
        with (output/(name+'.build.log')).open('x') as stream:
            subprocess.run(['pdflatex','-no-shell-escape','-halt-on-error','-interaction=nonstopmode',
                '-output-directory',str(output),'-jobname',name,str(source)], cwd=output,
                stdout=stream,stderr=subprocess.STDOUT,check=True)
        pdf = output/(name+'.pdf')
        info = subprocess.check_output(['pdfinfo',str(pdf)],text=True)
        pages = int(re.search(r'^Pages:\s+(\d+)',info,re.M).group(1))
        if pages != 1:
            raise ValueError('Original one-page schematic layout changed')
        subprocess.run(['pdftoppm','-png','-r','150','-singlefile',str(pdf),str(output/name)],check=True)
        plots.append(dict(an_reference='figures/'+name+'.pdf',pdf=name+'.pdf',png=name+'.png',
            pages=pages,status='generated_pending_visual_QA'))
    result = dict(status='generated_pending_visual_QA',contract=contract,plots=plots,
        files={p.name:sha256(p) for p in output.iterdir() if p.suffix in ('.png','.pdf')},
        change='Approved Nbst/Nmix/Nres/Nw topology condition only; all other TikZ content byte-preserved',
        actual_event_selection_changed=False,an_source_modified=False,full_workflow_complete=False)
    write_json(receipt,result)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    print(render(parser.parse_args().output)['status'])
