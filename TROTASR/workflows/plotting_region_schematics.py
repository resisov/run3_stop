"""Compile the adopted region schematics to PDF and PNG."""
import argparse
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
    sources = [ROOT/'workflows/renderers/region_schematics'/name for name in (
        'region_orthogonality_schematic.tex',
        'region_orthogonality_schematic_lowdm.tex')]
    contract = dict(sources={str(path.relative_to(ROOT)): sha256(path) for path in sources})
    receipt = output/'plot_manifest.json'
    if output.exists():
        previous = read_json(receipt)
        if previous['contract'] != contract or any(sha256(output/n) != d for n,d in previous['files'].items()):
            raise ValueError('Changed or incomplete schematic output must be retained')
        return previous
    output.mkdir(parents=True)
    plots = []
    for source in sources:
        name = source.stem
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
        an_source_modified=False)
    write_json(receipt,result)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    print(render(parser.parse_args().output)['status'])
