"""Transfer the adopted Photon CR model to blinded signal workspaces."""

import fcntl
import math
import re

from TROTASR.utils.paths import ERAS
from TROTASR.utils.io import read_json


MODEL_REVISION = "photon_gnn_recoil_signal_nominal"
LOW = "GCR_lowdm_c1"
SR = "SR_lowdm_c1"
CSEV = "CMS_eff_g_CSEV_13p6TeV_2025"
GNN = tuple(f"gcr_lowdm_nb1_gnn_logshape_bin{i}" for i in range(1, 5))
RECOIL = tuple(f"CMS_NPS26012_sgamma_shape_lowdm_Nb1_bin{i}" for i in range(5))
GAMMA = re.compile(r"PhotonJet_Nb(?:1|2plus)_u\d+_2025$")


def prepare_card(root, source, target, signal_file):
    """Remove signal uncertainty terms and keep only its nominal TH1 contents."""
    rows = [line.split() for line in source.read_text().splitlines()]
    processes, ids = [row[1:] for row in rows if row and row[0] == "process"]
    channels = next(row[1:] for row in rows if row and row[0] == "bin"
                    and len(row) == len(processes) + 1)
    signals = {i for i, value in enumerate(ids) if int(value) <= 0}
    if len(signals) != 12 or any(not channels[i].startswith("SR_") for i in signals):
        raise ValueError("Expected two signal channels per era and no CR signal")
    signal_shapes = {(channels[i], processes[i]) for i in signals}
    written, removed = set(), set()
    n_csev = 0
    files = {}
    try:
        with root.TFile.Open(str(signal_file), "CREATE") as output:
            if not output or output.IsZombie():
                raise OSError(f"Cannot create {signal_file}")
            for row in rows:
                if not row:
                    continue
                if row[0] == "shapes" and row[3] != "FAKE":
                    path = (source.parent / row[3]).resolve()
                    row[3] = str(path)
                    key = (row[2], row[1])
                    if key in signal_shapes:
                        if key in written:
                            raise ValueError(f"Duplicate nominal signal shape: {key}")
                        if path not in files:
                            files[path] = root.TFile.Open(str(path), "READ")
                        hist = files[path].Get(row[4])
                        if not hist or not hist.InheritsFrom("TH1") or hist.GetDimension() != 1:
                            raise ValueError(f"Missing nominal signal TH1: {row[4]}")
                        name = f"{row[2]}_{row[1]}"
                        clone = hist.Clone(name)
                        clone.SetDirectory(0)
                        for i in range(clone.GetNcells()):
                            clone.SetBinError(i, 0.)
                        output.WriteObject(clone, name)
                        row[3:] = [str(signal_file), name]
                        written.add(key)
                elif len(row) > 1 and row[1] in {"lnN", "shape", "shapeN", "shapeN2", "gmN"}:
                    offset = 3 if row[1] == "gmN" else 2
                    if len(row) != len(ids) + offset:
                        raise ValueError(f"Unexpected nuisance row: {row[0]}")
                    if any(row[i + offset] != "-" for i in signals):
                        removed.add(row[0])
                        for i in signals:
                            row[i + offset] = "-"
                    if row[0] == CSEV:
                        for i, (channel, process) in enumerate(zip(channels, processes)):
                            if channel.startswith("GCR_") and GAMMA.fullmatch(process):
                                n_csev += row[i + offset] != "-"
                                row[i + offset] = "-"
                    if all(value == "-" for value in row[offset:]):
                        row.clear()
                elif len(row) > 1 and row[1] == "rateParam":
                    if any(row[2] in ("*", ch) and row[3] in ("*", proc)
                           for ch, proc in signal_shapes):
                        raise ValueError(f"Unexpected signal rateParam: {row[0]}")
                elif len(row) > 1 and row[1] == "autoMCStats":
                    if row != ["*", "autoMCStats", "10", "1", "1"]:
                        raise ValueError("Unexpected autoMCStats configuration")
                    row[:] = ["\n".join(
                        f"{ch} autoMCStats 10 {int(not ch.startswith('SR_'))} 1"
                        for ch in dict.fromkeys(channels))]
    finally:
        for handle in files.values():
            handle.Close()
    if written != signal_shapes or not removed or not n_csev:
        raise ValueError("Incomplete signal/CSEV card transformation")
    # kmax can be '*' already; an explicit count would be stale.
    for row in rows:
        if row and row[0] == "kmax":
            row[:] = ["kmax", "*"]
    with target.open("x") as stream:
        stream.write("\n".join(" ".join(row) for row in rows) + "\n")
    return dict(signal_templates=len(written), removed_signal_terms=len(removed),
                removed_csev_columns=n_csev)


def _processes(w, channel):
    prop = w.function("prop_bin" + channel)
    if not prop or prop.ClassName() != "CMSHistErrorPropagator":
        raise ValueError(f"Missing native histogram likelihood for {channel}")
    return {a.getStringAttribute("combine.process"): (a, b)
            for a, b in zip(prop.wrapperList(), prop.coefList())}


def _yield_vector(w, wrapper, coefficient, bins):
    x = w.var("CMS_th1x")
    saved = x.getVal()
    try:
        result = []
        for i in range(bins):
            x.setVal(i + .5)
            result.append(wrapper.getVal() * coefficient.getVal())
        return result
    finally:
        x.setVal(saved)


def _weights(donor):
    """Read fixed prefit γ+jets Q·Sγ weights, not observed or fitted yields."""
    procs = _processes(donor, LOW)
    return {era: [_yield_vector(donor, *procs[f"PhotonJet_Nb1_u{u}_{era}"], 5)
                  for u in range(5)] for era in ERAS}


def _projection_formula(yields, recoil):
    total = [sum(row[g] for row in yields) for g in range(5)]
    row = yields[recoil]
    if (any(not math.isfinite(v) or v < 0 for v in total + row)
            or sum(row) <= 0 or sum(total) <= 0):
        raise ValueError(f"No supported Photon CR projection in recoil bin {recoil}")

    def weighted_average(values):
        fractions = [v / sum(values) for v in values]
        return "(" + "+".join([f"{fractions[0]:.17g}"] + [
            f"{value:.17g}*exp(@{i})" for i, value in enumerate(fractions[1:])]) + ")"

    return f"{weighted_average(row)}/{weighted_average(total)}"


def _import(root, w, node):
    if getattr(w, "import")(node, root.RooFit.RecycleConflictNodes(), root.RooFit.Silence()):
        raise ValueError(f"Failed to import {node.GetName()}")
    return w.arg(node.GetName())


def _require_nominal_signal(w):
    for channel in ("SR_highdm_c1", SR):
        for era in ERAS:
            wrapper, coefficient = _processes(w, channel)[f"signal_{era}"]
            morph = next(server for server in wrapper.servers()
                         if server.ClassName() == "CMSHistFunc")
            variables = set()
            for node in (morph, coefficient):
                variables.update(v.GetName() for v in node.getVariables() if not v.isConstant())
            if variables - {"r", "CMS_th1x"} or "r" not in variables:
                raise ValueError(f"Signal nuisance survived in {channel}/signal_{era}: {variables}")


def link_workspace(root, base, donor, output):
    """Attach the adopted CR graph and its recoil projection to both models."""
    _require_nominal_signal(base)
    weights = _weights(donor)
    for name in RECOIL:
        base.var(name).setVal(donor.var(name).getVal())
        base.var(name).setConstant(True)

    # Reuse the donor delta coefficients. Their existing native CR servers
    # resolve to this workspace, whose γ+jets CSEV responses were removed.
    functions, coefficients = root.RooArgList(), root.RooArgList()
    native = base.pdf(f"pdf_bin{LOW}_nuis")
    for f, c in zip(native.funcList(), native.coefList()):
        functions.add(f)
        coefficients.add(c)
    x = base.var("CMS_th1x")
    bins = x.getBins()
    for g in range(5):
        coefficients.add(_import(root, base, donor.function(f"qgnn_delta_{g}")))
        hist = root.TH1D(f"cr_linked_bin_{g}", "", bins, x.getMin(), x.getMax())
        hist.SetDirectory(0)
        hist.SetBinContent(g + 1, 1.)
        data = root.RooDataHist(f"cr_linked_data_{g}", "", root.RooArgList(x), hist)
        basis = root.RooHistFunc(f"cr_linked_basis_{g}", "", root.RooArgSet(x), data)
        functions.add(_import(root, base, basis))
    photon = _import(root, base, root.RooRealSumPdf(
        "cr_linked_photon_pdf", "", functions, coefficients, True))
    gnn_args = root.RooArgList()
    for name in GNN:
        parameter = base.var(name)
        if not parameter or parameter.isConstant() or not parameter.getMin() < 0 < parameter.getMax():
            raise ValueError(f"Missing floating relative GNN coefficient: {name}")
        parameter.setVal(0.)
        gnn_args.add(parameter)
    sr_processes = _processes(base, SR)
    replacements = []
    for era in ERAS:
        for u in range(5):
            process = f"ZJet_Nb1_u{u}_{era}"
            if process not in sr_processes:
                raise ValueError(f"Missing SR Z→νν recoil process: {process}")
            _, original = sr_processes[process]
            projection = _import(root, base, root.RooFormulaVar(
                f"cr_linked_recoil_{u}_{era}", "", _projection_formula(weights[era], u), gnn_args))
            varied = _import(root, base, root.RooProduct(
                f"cr_linked_yield_{process}", "", root.RooArgList(original, projection)))
            replacements.append((original, varied))
    for config_name in ("ModelConfig", "ModelConfig_bonly"):
        config = base.obj(config_name)
        pdf = config.GetPdf()
        gcr = next(node for node in pdf.getPdf(LOW).pdfList()
                   if node.GetName().startswith(f"pdf_bin{LOW}_"))
        customized = root.RooCustomizer(pdf, f"cr_linked_{config_name}")
        customized.replaceArg(gcr, photon)
        for original, varied in replacements:
            customized.replaceArg(original, varied)
        model = _import(root, base, customized.build(False))
        nuisances = root.RooArgSet()
        for nuisance in config.GetNuisanceParameters():
            if nuisance.GetName() not in RECOIL:
                nuisances.add(nuisance)
        for name in GNN:
            nuisances.add(base.var(name))
        config.SetPdf(model)
        config.SetNuisanceParameters(nuisances)
    if base.writeToFile(str(output), True):
        raise OSError(f"Could not write {output}")
    return weights


def donor_workspace(run_root):
    """Resolve the current accepted CR-only workspace, not a fitted snapshot."""
    cronly = run_root / "cronly"
    publication = read_json(cronly / "publication.json")
    if (publication.get("kind") != "cronly" or
            publication.get("sr_observed_data_used") is not False):
        raise ValueError("The accepted blinded CR-only receipt is required")
    path = cronly / "current_workspace.root"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def workspace_step(stage, source, directory, run_root, *, products=None):
    """Run each ROOT-intensive conversion step in a single downstream process."""
    products = run_root if products is None else products
    import ROOT as root
    root.gROOT.SetBatch(True)
    root.RooMsgService.instance().setGlobalKillBelow(root.RooFit.ERROR)
    if stage == "prepare":
        if source.parent != products / "grid/datacards":
            raise ValueError("Workspace input is not an accepted Run 3 grid card")
        print(prepare_card(root, source, directory / "card.txt", directory / "signal.root"))
    elif stage == "link":
        root.gSystem.Load("libHiggsAnalysisCombinedLimit")
        donor = donor_workspace(run_root)
        with (donor.parent / "controller.lock").open("rb") as owner:
            fcntl.flock(owner, fcntl.LOCK_SH | fcntl.LOCK_NB)
            marker = read_json(products / "limits/model.json")
            if (marker.get("model_revision") != MODEL_REVISION or
                    not donor.is_file() or donor.stat().st_size <= 0):
                raise ValueError("The adopted CR-only workspace is unavailable")
            with root.TFile.Open(str(source), "READ") as native, root.TFile.Open(
                    str(donor), "READ") as cronly:
                if not native or native.IsZombie() or not cronly or cronly.IsZombie():
                    raise ValueError("Missing or unreadable linked-workspace input")
                link_workspace(root, native.Get("w"), cronly.Get("w"), directory / "linked.root")
    else:
        raise ValueError(f"Unknown workspace step: {stage}")
