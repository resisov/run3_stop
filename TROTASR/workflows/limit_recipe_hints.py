"""Seed the canonical command-only recovery input; never import a fit result."""
import json
import math
import os
from pathlib import Path
import re
import tempfile

from TROTASR.utils.paths import ROOT
from TROTASR.utils.io import read_json
from TROTASR.workflows.limit_recovery import profile_recovery_command, profile_recovery_configs


def terminal_reference_plans(output, point, attempts, grids=None):
    """Fresh blind fits, guided by accepted same-point/nearby numerical solutions.

    A reference median only selects the nearer *native first trial* (5% or
    50% of the existing interval). It is neither a crossing nor a fit result.
    No fit state, POI bound, Asimov parameter or nuisance value is imported.
    """
    from TROTASR.workflows.limit_recovery import STAGE_LIMITS
    model, mass = point['model'], point['mass']
    key = model + '/' + mass
    def coords(k):
        match = re.fullmatch(r'mStop(\d+)_mLSP(\d+)', k.split('/')[-1])
        return tuple(map(int, match.groups())) if match else None
    xy = coords(key)
    budget = min(2, STAGE_LIMITS['branches'] - sum(a.get('stage') == 'branches' for a in attempts))
    if not xy or budget <= 0 or any(a.get('passed') for a in attempts):
        return []
    known = {tuple(a['command']) for a in attempts if a.get('command')}
    bound = max(float(c[c.index('--rMax')+1]) for c in known)
    if not 0 < bound <= 100000:
        raise ValueError('Invalid current point range')
    workspace = output/key/'workspace.root'
    if grids is None:
        previous = ROOT/'stats/run3/limits'
        frozen = ROOT/'stats/references/era_merged_recoil/limits'
        grids = [(p, read_json(p/'workflow_state.json')) for p in (previous, frozen, output)]
    candidates = []
    for order, (location, grid) in enumerate(grids):
        if grid.get('observed_sr_used') is not False:
            continue
        for other, row in grid['results'].items():
            if row.get('status') != 'complete' or other.split('/')[0] != model:
                continue
            if other != key and location != output:
                continue
            pos = coords(other)
            if not pos:
                continue
            dist = sum(abs(x-y) for x,y in zip(xy,pos))
            if dist > 200:
                continue
            a = row.get('accepted_attempt') or {}
            if a.get('passed') is not True or a.get('exit_code') != 0:
                continue
            q = (row.get('validation') or {}).get('quantiles',{})
            try:
                vals = [float(q[str(v)]) for v in (.025,.16,.5,.84,.975)]
                c = a['command']
                if (not all(math.isfinite(v) and v > 0 for v in vals) or
                        not all(x<y for x,y in zip(vals,vals[1:])) or
                        c[1:3] != ['-M','AsymptoticLimits'] or c[c.index('--run')+1]!='blind'
                        or '--picky' not in c or c[c.index('-n')+1] != '_'+other.split('/')[1]):
                    continue
                algo = c[c.index('--cminDefaultMinimizerAlgo')+1] if '--cminDefaultMinimizerAlgo' in c else 'Migrad'
                strat = int(c[c.index('--cminDefaultMinimizerStrategy')+1])
                if algo not in ('Combined','Migrad') or strat not in (0,1,2):
                    continue
                # Some committed primary receipts omit per-attempt timing.
                # Time is a ranking tiebreaker, not a validity criterion.
                elapsed = (float(a['finished'])-float(a['started'])
                           if a.get('finished') is not None and a.get('started') is not None else math.inf)
                if not math.isfinite(elapsed) or elapsed <= 0:
                    elapsed = math.inf
            except (KeyError,TypeError,IndexError,ValueError):
                continue
            candidates.append((dist,order,elapsed,other,str(location),algo,strat,vals[2]))
    selected = []
    for dist,order,elapsed,other,location,algo,strat,median in sorted(candidates):
        mode = min(('stepping','bisection'),key=lambda m:abs(math.log((.05 if m=='stepping' else .5)*bound/median)))
        # Use the canonical grid's search accuracy, inherited from fit_command.
        # A new attempt name also keeps old tighter-fit logs and products intact.
        spec = dict(name=f'guided_default_{mode}_{algo.lower()}_s{strat}', stage='branches',
                    strategy=strat,tolerance=.001,rmax=bound,algorithm=algo,precision=None,
                    explicit_stats=True,max_calls=2000000)
        cmd = profile_recovery_command(workspace,mass,spec)
        for flag,value in (('--minosAlgo',mode),
                           ('--cminFallbackAlgo',f'Minuit2,{algo},2:0.001')):
            cmd[cmd.index(flag)+1] = value
        if mode=='stepping':
            cmd.append('--strictBounds')
            spec['run3_numerical_adapter']='stepping_full5'
        if tuple(cmd) in known:
            continue
        known.add(tuple(cmd))
        selected.append(dict(spec, command=cmd, reference_point=other, reference_grid=location,
            reference_expected_median=median, reference_distance_GeV=dist,
            first_median_trial=(.05 if mode=='stepping' else .5)*bound,
            reference_used_as_result=False))
        if len(selected) == budget:
            break
    return selected


def fastest_recipes(plan, histories):
    """Rank actual same-point successes at the unchanged refined-primary bound."""
    picked = {}
    for origin, workflow, catalog in histories:
        if (workflow.get('observed_sr_used') is not False
                or catalog.get('schema') != 'canonical_limit_successful_numerical_recipes_v1'
                or catalog.get('numerical_settings_only') is not True
                or catalog.get('fit_values_reused') is not False):
            raise ValueError('Expected a blinded workflow and a command-only catalog')
        for key, prior in catalog['points'].items():
            if key not in plan:
                continue
            row = workflow['results'].get(key, {})
            a = row.get('accepted_attempt') or {}
            if (row.get('status') != 'complete' or a.get('passed') is not True
                    or a.get('exit_code') != 0 or prior.get('name') != a.get('name')):
                continue
            try:
                elapsed = float(a['finished']) - float(a['started'])
                original = prior['command']
                workspace = Path(original[3])
                mass = key.split('/')[1]
                if (not math.isfinite(elapsed) or elapsed <= 0
                        or tuple(workspace.parts[-3:]) != (*key.split('/'), 'workspace.root')):
                    continue
                cmd = list(a['command'])
                if Path(cmd[3]).parts[-3:] != workspace.parts[-3:]:
                    continue
                cmd[3] = str(workspace)
                if cmd != original:
                    continue
                rmax = float(plan[key]['rmax']) * 2.
                for spec in profile_recovery_configs(rmax, 2000000):
                    if profile_recovery_command(workspace, mass, spec) != cmd:
                        continue
                    if key not in picked or elapsed < picked[key]['historical_seconds']:
                        picked[key] = dict(name=prior['name'], command=cmd, historical_seconds=elapsed,
                                           source=str(origin), previous_model_only=True)
            except (IndexError, KeyError, TypeError, ValueError):
                continue
    return picked


def seed_top_pt_recovery_hints(*, publish=False):
    """Create only an absent hint catalog. A controller's final catalog always wins."""
    stats = ROOT/'stats'
    out = stats/'run3/top_pt/limits'
    target = out/'successful_recipes.json'
    if target.exists():
        return dict(status='existing_catalog_preserved')
    state = read_json(out/'workflow_state.json')
    if state.get('observed_sr_used') is not False or state.get('status') != 'running':
        return dict(status='no_running_blinded_grid')
    plan = read_json(out/'range_plan.json')['points']
    waiting = {k: v for k, v in plan.items() if k not in state['results']
               and not (out/k/'workflow.json').exists()}
    histories = ((p, read_json(p/'workflow_state.json'), read_json(p/'successful_recipes.json'))
                 for p in (stats/'run3/limits', stats/'references/era_merged_recoil/limits'))
    points = fastest_recipes(waiting, histories)
    if publish and points:
        doc = dict(schema='canonical_limit_successful_numerical_recipes_v1', points=points,
                   numerical_settings_only=True, fit_values_reused=False,
                   scope='previous-model elapsed-time hints, not current-model results')
        with tempfile.NamedTemporaryFile(mode='w', dir=out, prefix='.recipe-input-') as tmp:
            json.dump(doc, tmp, allow_nan=False)
            tmp.flush()
            os.fsync(tmp.fileno())
            try:
                os.link(tmp.name, target)
            except FileExistsError:
                return dict(status='existing_catalog_preserved')
    return dict(status='seeded' if publish and points else 'preview', points=len(points),
                candidates={k:(v['name'], round(v['historical_seconds'])) for k,v in points.items()})
