"""Transfer the adopted Photon CR model to blinded signal workspaces."""

import fcntl
import math
import re

from TROTASR.utils.paths import ERAS, internal_path
from TROTASR.utils.io import read_json


MODEL_REVISION = "photon_gnn_recoil_signal_nominal"
TOP_PT_MODEL_REVISION = "photon_gnn_both_nb_recoil_signal_nominal"
TOP_PT_CRONLY_REVISION = "tt_top_pt_tail500_photon_gnn_both_nb"
LOW = "GCR_lowdm_c1"
SR = "SR_lowdm_c1"
CSEV = "CMS_eff_g_CSEV_13p6TeV_2025"
PHOTON_GROUPS = ("Nb1", "Nb2plus")
GNN_BY_GROUP = {
    nb: tuple(f"gcr_lowdm_{nb.lower()}_gnn_logshape_bin{i}" for i in range(1, 5))
    for nb in PHOTON_GROUPS
}
RECOIL_BY_GROUP = {
    nb: tuple(f"CMS_NPS26012_sgamma_shape_lowdm_{nb}_bin{i}" for i in range(5))
    for nb in PHOTON_GROUPS
}
GNN = GNN_BY_GROUP["Nb1"]
RECOIL = RECOIL_BY_GROUP["Nb1"]
GAMMA = re.compile(r"PhotonJet_Nb(?:1|2plus)_u\d+_2025$")


def prepare_card(root, source, target, signal_file, *, nominal_only_input=False):
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
    if written != signal_shapes or (not removed and not nominal_only_input) or not n_csev:
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


def _yield_vector(w, wrapper, coefficient, bins, *, offset=0):
    x = w.var("CMS_th1x")
    saved = x.getVal()
    try:
        result = []
        for i in range(offset, offset + bins):
            x.setVal(i + .5)
            result.append(wrapper.getVal() * coefficient.getVal())
        return result
    finally:
        x.setVal(saved)


def _weights(donor, nb="Nb1"):
    """Read fixed prefit γ+jets Q·Sγ weights, not observed or fitted yields."""
    procs = _processes(donor, LOW)
    offset = 5 * PHOTON_GROUPS.index(nb)
    return {era: [_yield_vector(donor, *procs[f"PhotonJet_{nb}_u{u}_{era}"], 5, offset=offset)
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


def _fresh_photon(root, base, donor, groups):
    """Build the accepted five-bin relative deformation on the *new* native yields.

    The non-photon part stays in the native PDF. The additive correction is
    gamma_g*(exp(theta_g)*sum(gamma)/sum(gamma*exp(theta))-1), per era.
    Only parameter bounds come from the approved CR graph.
    """
    x=base.var("CMS_th1x"); procs=_processes(base,LOW)
    def args(seq):
        result=root.RooArgList()
        for item in seq:result.add(item)
        return result
    def formula(name,expression,seq):
        return _import(root,base,root.RooFormulaVar(name,name,expression,args(seq)))
    def total(name,seq):
        return _import(root,base,root.RooAddition(name,name,args(seq)))
    native=base.pdf('pdf_bin'+LOW+'_nuis')
    fs=args(list(native.funcList()));cs=args(list(native.coefList()))
    for nb in groups:
        offset = 5 * PHOTON_GROUPS.index(nb)
        slots = range(offset, offset + 5)
        weights = [_import(root, base, root.RooConstVar(f"qgnn_reference_weight_{nb}", "", 1.))]
        theta = []
        for g, name, reference in zip(list(slots)[1:], GNN_BY_GROUP[nb], GNN):
            old = donor.var(reference)
            parameter = _import(root, base, root.RooRealVar(name, name, 0., old.getMin(), old.getMax()))
            theta.append(parameter)
            weights.append(formula(f"qgnn_weight_bin{g}", "exp(@0)", [parameter]))
        centers = [_import(root, base, root.RooConstVar(f"qgnn_x{g}", "", g + .5)) for g in slots]
        delta = [[] for _ in slots]
        for era in ERAS:
            raw = []
            for g, center in zip(slots, centers):
                products = []
                for u in range(5):
                    wrapper, coefficient = procs[f"PhotonJet_{nb}_u{u}_{era}"]
                    custom = root.RooCustomizer(wrapper, f"qgnn_bin{g}")
                    custom.replaceArg(x, center)
                    fixed = _import(root, base, custom.build(False))
                    products.append(formula(f"qgnn_yield_{era}_{u}_{g}", "@0*@1", [fixed, coefficient]))
                raw.append(total(f"qgnn_raw_{era}_qgnn_bin{g}", products))
            norm = total(f"qgnn_total_{nb}_{era}", raw)
            if not math.isfinite(norm.getVal()) or norm.getVal() <= 0:
                raise ValueError(f"No Photon GNN normalization support in {era}/{nb}")
            weighted = [formula(f"qgnn_weighted_{era}_{g}", "@0*@1", [r, v])
                        for g, r, v in zip(slots, raw, weights)]
            den = total(f"qgnn_weighted_total_{nb}_{era}", weighted)
            for g, terms, r, weight in zip(slots, delta, raw, weights):
                terms.append(formula(f"qgnn_diff_{era}_{g}", "@0*(@1*@2/@3-1)", [r, weight, norm, den]))
        for g, terms in zip(slots, delta):
            hist = root.TH1D(f"qgnn_hist{g}", "", x.getBins(), x.getMin(), x.getMax())
            hist.SetDirectory(0)
            hist.SetBinContent(g + 1, 1.)
            data = root.RooDataHist(f"qgnn_data{g}", "", args([x]), hist)
            fs.add(_import(root, base, root.RooHistFunc(f"qgnn_basis{g}", "", root.RooArgSet(x), data)))
            cs.add(total(f"qgnn_delta_{g}", terms))
        # Check each era independently before summing the CR corrections.
        for parameter in theta:
            for value in (0., .2, -.2):
                parameter.setVal(value)
                for terms in zip(*delta):
                    shifts = [term.getVal() for term in terms]
                    if (not all(map(math.isfinite, shifts)) or abs(sum(shifts)) > 1e-6
                            or (value == 0. and max(map(abs, shifts)) > 1e-6)):
                        raise ValueError("Photon GNN normalization failed: " + parameter.GetName())
            parameter.setVal(0.)
    return _import(root, base, root.RooRealSumPdf("qgnn_cr_native_plus_delta", "", fs, cs, True))


def link_workspace(root, base, donor, output, *, fresh=False, groups=("Nb1",)):
    """Attach the adopted CR graph and its recoil projection to both models."""
    _require_nominal_signal(base)
    weights = {nb: _weights(base if fresh else donor, nb) for nb in groups}
    recoil_names = {name for nb in groups for name in RECOIL_BY_GROUP[nb]}
    gnn_names = [name for nb in groups for name in GNN_BY_GROUP[nb]]
    for name in recoil_names:
        if not fresh:base.var(name).setVal(donor.var(name).getVal())
        base.var(name).setConstant(True)

    if fresh:
        photon = _fresh_photon(root, base, donor, groups)
    else:
        # The frozen model reuses the donor's delta coefficients.
        functions, coefficients = root.RooArgList(), root.RooArgList()
        native = base.pdf(f"pdf_bin{LOW}_nuis")
        for f, c in zip(native.funcList(), native.coefList()):
            functions.add(f)
            coefficients.add(c)
        x = base.var("CMS_th1x")
        for g in range(5):
            coefficients.add(_import(root, base, donor.function(f"qgnn_delta_{g}")))
            hist = root.TH1D(f"cr_linked_bin_{g}", "", x.getBins(), x.getMin(), x.getMax())
            hist.SetDirectory(0)
            hist.SetBinContent(g + 1, 1.)
            data = root.RooDataHist(f"cr_linked_data_{g}", "", root.RooArgList(x), hist)
            basis = root.RooHistFunc(f"cr_linked_basis_{g}", "", root.RooArgSet(x), data)
            functions.add(_import(root, base, basis))
        photon = _import(root, base, root.RooRealSumPdf(
            "cr_linked_photon_pdf", "", functions, coefficients, True))
    sr_processes = _processes(base, SR)
    replacements = []
    for nb in groups:
        gnn_args = root.RooArgList()
        for name in GNN_BY_GROUP[nb]:
            parameter = base.var(name)
            if not parameter or parameter.isConstant() or not parameter.getMin() < 0 < parameter.getMax():
                raise ValueError(f"Missing floating relative GNN coefficient: {name}")
            parameter.setVal(0.)
            gnn_args.add(parameter)
        prefix = "cr_linked_recoil_" if nb == "Nb1" else f"cr_linked_recoil_{nb.lower()}_"
        for era in ERAS:
            for u in range(5):
                process = f"ZJet_{nb}_u{u}_{era}"
                if process not in sr_processes:
                    raise ValueError(f"Missing SR Z→νν recoil process: {process}")
                _, original = sr_processes[process]
                projection = _import(root, base, root.RooFormulaVar(
                    f"{prefix}{u}_{era}", "", _projection_formula(weights[nb][era], u), gnn_args))
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
            if nuisance.GetName() not in recoil_names:
                nuisances.add(nuisance)
        for name in gnn_names:
            nuisances.add(base.var(name))
        config.SetPdf(model)
        config.SetNuisanceParameters(nuisances)
    if base.writeToFile(str(output), True):
        raise OSError(f"Could not write {output}")
    return weights["Nb1"] if groups == ("Nb1",) else weights


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


def _cr_observations(w, *, only=None):
    """Read integer observations from a workspace with no SR channel."""
    from TROTASR.utils.era_merge import CHANNEL_BINS

    counts = {name: [0.] * n for name, n in CHANNEL_BINS.items() if not name.startswith("SR_")}
    category, data = w.cat("CMS_channel"), w.data("data_obs")
    if not category or {str(c.first) for c in category} != set(counts) or not data:
        raise ValueError("Expected a workspace with the six observed CR channels only")
    if only is not None:
        # A frozen donor can have the old high-dM tail binning. Only its
        # unchanged low-dM Photon channel supplies observations to this fit.
        counts = {only: counts[only]}
    seen = set()
    for i in range(data.numEntries()):
        row = data.get(i)
        channel = str(row.getCatLabel("CMS_channel"))
        if only is not None and channel != only:
            continue
        x, weight = float(row.getRealValue("CMS_th1x")), float(data.weight())
        if channel not in counts or not math.isfinite(x) or not math.isfinite(weight) or weight < 0 or weight != int(weight):
            raise ValueError("Invalid observed CR bin")
        index = math.floor(x)
        if not 0 <= index < len(counts[channel]) or x != index + .5:
            if weight == 0:
                continue  # An unused bin of the common CMS_th1x axis.
            raise ValueError("Observed CR bin lies outside its accepted channel")
        if (channel, index) in seen:
            raise ValueError("Duplicate observed CR bin")
        counts[channel][index] = weight
        seen.add((channel, index))
    return counts


def _linked_cr_card_rows(source, observation_root):
    """Resolve all source shapes; change only low-GCR observations and CSEV."""
    from TROTASR.workflows.diagnostic_common import cr_channels

    text = source.read_text()
    expected = cr_channels(text, ERAS)
    if set(expected) != {f"{region}_{mode}_c1" for region in ("LLCR", "QCDCR", "GCR")
                         for mode in ("highdm", "lowdm")}:
        raise ValueError("The Photon GNN link requires the era-merged observed CR card")
    rows = [line.split() for line in text.splitlines()]
    processes, ids = [row[1:] for row in rows if row and row[0] == "process"]
    channels = next(row[1:] for row in rows if row and row[0] == "bin"
                    and len(row) == len(ids) + 1)
    low_source, removed, csev_rows = None, 0, 0
    for row in rows:
        if not row:
            continue
        if row[0] == "shapes":
            row[3] = str(internal_path(source.parent / row[3]))
            if row[1:3] == ["data_obs", LOW]:
                if low_source or len(row) != 5 or row[4] != f"{LOW}/data_obs":
                    raise ValueError("Expected one explicit native observed low Photon CR shape")
                low_source = row[3]
                row[3] = str(observation_root)
        elif row[0] == CSEV:
            csev_rows += 1
            if len(row) != len(processes) + 2 or row[1] not in {"lnN", "shape", "shapeN", "shapeN2"}:
                raise ValueError("Unexpected 2025 Photon CSEV nuisance row")
            for i, (channel, process) in enumerate(zip(channels, processes), 2):
                if channel.startswith("GCR_") and GAMMA.fullmatch(process):
                    removed += row[i] != "-"
                    row[i] = "-"
            if all(value == "-" for value in row[2:]):
                row.clear()
    if not low_source or csev_rows != 1 or not removed:
        raise ValueError("Missing native GCR observation or approved 2025 Photon CSEV columns")
    return rows, low_source, removed


def prepare_cronly_card(root, source, target, observation_root, donor, projection):
    """Carry the actual low Nb=1 Photon GNN observations into the new CR card."""
    from TROTASR.utils.era_merge import write_observation_sum

    rows, low_source, removed = _linked_cr_card_rows(source, observation_root)
    donor_counts = _cr_observations(donor,only=LOW)[LOW]
    record = projection.get("channels", {}).get(LOW, {})
    labels = record.get("labels", [])
    expected = [f"GCR_lowdm_gnn_Nb{nb}_bin{i}" for nb in (1, 2) for i in range(5)]
    if (projection.get("schema") != "trotasr_cronly_prefit_postfit_v1"
            or projection.get("sr_observations_included") is not False
            or tuple(projection.get("years", ())) != ERAS or labels != expected
            or record.get("data") != donor_counts):
        raise ValueError("Frozen low Photon CR observation does not match its approved GNN-bin projection")
    with root.TFile.Open(low_source, "READ") as templates:
        if not templates or templates.IsZombie() or templates.TestBit(root.TFile.kRecovered):
            raise ValueError("Cannot read the selected low Photon CR observation")
        hist = templates.Get(f"{LOW}/data_obs")
        if not hist or hist.GetNbinsX() != 10:
            raise ValueError("The selected low Photon CR does not have the accepted ten-bin layout")
        counts, variances = [], []
        for i in range(6, 11):
            value, variance = float(hist.GetBinContent(i)), float(hist.GetBinError(i)) ** 2
            if (value != donor_counts[i-1] or str(hist.GetXaxis().GetBinLabel(i)) != labels[i-1]
                    or not math.isclose(variance, value, rel_tol=2.e-12, abs_tol=1.e-12)):
                raise ValueError("Unchanged low Photon CR Nb>=2 observation differs from the approved source")
            counts.append(value)
            variances.append(variance)
    # Donor use is restricted to observed Nb=1 GNN-bin counts, never its fit.
    write_observation_sum(observation_root, {
        LOW: dict(values=donor_counts[:5] + counts, variances=donor_counts[:5] + variances)
    }, {LOW: labels})
    with target.open("x") as stream:
        stream.write("\n".join(" ".join(row) for row in rows) + "\n")
    return dict(removed_csev_columns=removed, low_photon_gnn_observed_bins=5)


def _photon_pdf(w):
    pdf = w.pdf("qgnn_cr_native_plus_delta")
    n = 10 if w.var(GNN_BY_GROUP["Nb2plus"][0]) else 5
    if (not pdf or pdf.funcList().getSize() != n + 1 or pdf.coefList().getSize() != n + 1
            or any(pdf.funcList().at(i+1).GetName() != f"qgnn_basis{i}"
                   or pdf.coefList().at(i+1).GetName() != f"qgnn_delta_{i}"
                   for i in range(n))):
        raise ValueError("The accepted low Photon CR GNN graph is missing")
    return pdf


def _vector(w, nodes, count):
    x = w.var("CMS_th1x")
    saved = x.getVal()
    try:
        result = []
        for i in range(count):
            x.setVal(i + .5)
            result.extend(float(node.getVal()) for node in nodes)
        return result
    finally:
        x.setVal(saved)


def _same_vector(first, second, name):
    if len(first) != len(second) or not all(math.isclose(a, b, rel_tol=2.e-9, abs_tol=1.e-7)
                                            for a, b in zip(first, second)):
        raise ValueError("Unchanged Photon CR model differs at " + name)


def _link_cronly_workspace(root, base, donor, output):
    """Attach only the adopted Photon GNN graph to the changed CR likelihood."""
    counts = _cr_observations(base)
    if counts[LOW] != _cr_observations(donor,only=LOW)[LOW]:
        raise ValueError("Native conversion changed the approved low Photon CR observations")
    if base.pdf("qgnn_cr_native_plus_delta"):
        raise ValueError("Native CR fit must not already carry a Photon GNN graph")
    source = _photon_pdf(donor)
    common = set(_processes(base, LOW))
    if common != set(_processes(donor, LOW)):
        raise ValueError("Unchanged low Photon CR process inventory differs from the donor")
    for name in RECOIL:
        original, frozen = base.var(name), donor.var(name)
        if not original or not frozen or not frozen.isConstant() or not math.isfinite(frozen.getVal()):
            raise ValueError("Missing accepted fixed photon recoil coefficient: " + name)
        # These are explicitly fixed prefit constants of the accepted model.
        original.setVal(frozen.getVal())
        original.setConstant(True)
    for name in GNN:
        frozen = donor.var(name)
        if not frozen or frozen.isConstant() or not frozen.getMin() < 0 < frozen.getMax():
            raise ValueError("Missing adopted relative Photon GNN coefficient: " + name)
        frozen.setVal(0.)
    if any(not var.isConstant() and var.GetName() not in GNN and not base.arg(var.GetName())
           for var in source.getVariables()):
        raise ValueError("Photon GNN graft cannot import any other floating parameter from the donor")
    reference_weights, native_weights = _weights(donor), _weights(base)
    for era in ERAS:
        for recoil, (first, second) in enumerate(zip(reference_weights[era], native_weights[era])):
            _same_vector(first, second, f"{era}/recoil{recoil}")
    _import(root, base, source)
    photon = _photon_pdf(base)
    for name in GNN:
        var = base.var(name)
        frozen = donor.var(name)
        if (not var or var.isConstant() or (var.getMin(), var.getMax()) != (frozen.getMin(), frozen.getMax())
                or not photon.dependsOn(var)):
            raise ValueError("Photon GNN coefficient is not linked to the selected workspace: " + name)
        var.setVal(0.)
    _same_vector(_vector(base, [photon], 10), _vector(donor, [source], 10), "prefit nominal")
    for name in GNN:
        for value in (-.20, .20):
            var, frozen = base.var(name), donor.var(name)
            if not var.getMin() < value < var.getMax():
                raise ValueError("Unexpected adopted Photon GNN coefficient range")
            var.setVal(value)
            frozen.setVal(value)
            try:
                _same_vector(_vector(base, [photon], 10), _vector(donor, [source], 10), name)
            finally:
                var.setVal(0.)
                frozen.setVal(0.)
    _install_cr_photon(root,base,photon,counts,output)


def _install_cr_photon(root,base,photon,counts,output,groups=("Nb1",)):
    recoil_names = {name for nb in groups for name in RECOIL_BY_GROUP[nb]}
    for config_name in ("ModelConfig", "ModelConfig_bonly"):
        config = base.obj(config_name)
        if not config:
            raise ValueError("Missing native CR model config: " + config_name)
        pdf = config.GetPdf()
        native = pdf.getPdf(LOW)
        if native and native.GetName().startswith(f"pdf_bin{LOW}_"):
            candidates = [native]
        else:
            candidates = [node for node in native.pdfList()
                          if node.GetName().startswith(f"pdf_bin{LOW}_")]
        if len(candidates) != 1:
            raise ValueError("No unique native low Photon CR graph")
        customized = root.RooCustomizer(pdf, f"cr_gnn_link_{config_name}")
        customized.replaceArg(candidates[0], photon)
        linked = _import(root, base, customized.build(False))
        for channel in counts:
            before, after = pdf.getPdf(channel), linked.getPdf(channel)
            if channel == LOW:
                if not after or (after.GetName() != photon.GetName() and not after.dependsOn(photon)):
                    raise ValueError("ModelConfig does not depend on the adopted low Photon GNN graph")
            elif not after or after.GetName() != before.GetName():
                raise ValueError("Linked model modified a non-Photon CR channel: " + channel)
        nuisances = root.RooArgSet()
        for nuisance in config.GetNuisanceParameters():
            if nuisance.GetName() not in recoil_names:
                nuisances.add(nuisance)
        for nb in groups:
            for name in GNN_BY_GROUP[nb]:
                nuisances.add(base.var(name))
        config.SetPdf(linked)
        config.SetNuisanceParameters(nuisances)
    # The prospective model is always an observed-CR background-only r=0 fit.
    r = base.var("r")
    if not r:
        raise ValueError("Missing native signal-strength coordinate")
    r.setVal(0.)
    if _cr_observations(base) != counts:
        raise ValueError("Photon GNN graph graft modified observations")
    if output.exists() or base.writeToFile(str(output), False):
        raise OSError(f"Cannot create {output}")


def cronly_workspace_step(stage, source, output, run_root):
    """Run native CR card preparation and graph graft under the donor lock."""
    import ROOT as root

    root.gROOT.SetBatch(True)
    root.gSystem.Load("libHiggsAnalysisCombinedLimit")
    root.RooMsgService.instance().setGlobalKillBelow(root.RooFit.ERROR)
    donor = donor_workspace(run_root)
    with (donor.parent / "controller.lock").open("rb") as owner:
        fcntl.flock(owner, fcntl.LOCK_SH | fcntl.LOCK_NB)
        with root.TFile.Open(str(donor), "READ") as cronly:
            if not cronly or cronly.IsZombie() or cronly.TestBit(root.TFile.kRecovered):
                raise ValueError("Frozen CR-only structure workspace is unavailable")
            w = cronly.Get("w")
            if not w:
                raise ValueError("Missing frozen CR-only structure workspace")
            if stage == "prepare":
                publication = read_json(donor.parent / "publication.json")
                from TROTASR.workflows.diagnostic_common import provisional_cronly_products
                products = provisional_cronly_products(donor.parent, publication,
                    required=("workspace", "projection"))
                if not products or products["workspace"] != donor:
                    raise ValueError("CR projection must belong to its approved frozen workspace")
                print(prepare_cronly_card(root, source, output / "linked_card.txt",
                      output / "linked_data_obs.root", w, read_json(products["projection"])))
            elif stage == "link":
                with root.TFile.Open(str(source), "READ") as native:
                    if not native or native.IsZombie() or native.TestBit(root.TFile.kRecovered) or not native.Get("w"):
                        raise ValueError("Missing new native CR-only workspace")
                    b=native.Get("w")
                    if output.parent==run_root/'top_pt':
                        state = read_json(output / "state.json")
                        groups = PHOTON_GROUPS if state["model_revision"] == TOP_PT_CRONLY_REVISION else ("Nb1",)
                        for nb in groups:
                            for name in RECOIL_BY_GROUP[nb]:
                                b.var(name).setConstant(True)
                        counts=_cr_observations(b)
                        if counts[LOW]!=_cr_observations(w,only=LOW)[LOW]:
                            raise ValueError('Changed low GCR observations')
                        photon = _fresh_photon(root, b, w, groups)
                        _install_cr_photon(root, b, photon, counts, output/'workspace.root', groups)
                    else:
                        _link_cronly_workspace(root,b,w,output/"workspace.root")
            else:
                raise ValueError("Unknown linked CR-only workspace stage")


def workspace_step(stage, source, directory, run_root, *, products=None):
    """Run each ROOT-intensive conversion step in a single downstream process."""
    products = run_root if products is None else products
    import ROOT as root
    root.gROOT.SetBatch(True)
    root.RooMsgService.instance().setGlobalKillBelow(root.RooFit.ERROR)
    if stage == "prepare":
        if source.parent != products / "grid/datacards":
            raise ValueError("Workspace input is not an accepted Run 3 grid card")
        print(prepare_card(root, source, directory / "card.txt", directory / "signal.root",nominal_only_input=products==run_root/'top_pt'))
    elif stage == "link":
        root.gSystem.Load("libHiggsAnalysisCombinedLimit")
        donor = donor_workspace(run_root)
        with (donor.parent / "controller.lock").open("rb") as owner:
            fcntl.flock(owner, fcntl.LOCK_SH | fcntl.LOCK_NB)
            marker = read_json(products / "limits/model.json")
            revision = marker.get("model_revision")
            fresh = products == run_root / "top_pt"
            if (revision not in ((MODEL_REVISION, TOP_PT_MODEL_REVISION) if fresh else (MODEL_REVISION,)) or
                    not donor.is_file() or donor.stat().st_size <= 0):
                raise ValueError("The adopted CR-only workspace is unavailable")
            with root.TFile.Open(str(source), "READ") as native, root.TFile.Open(
                    str(donor), "READ") as cronly:
                if not native or native.IsZombie() or not cronly or cronly.IsZombie():
                    raise ValueError("Missing or unreadable linked-workspace input")
                groups = PHOTON_GROUPS if revision == TOP_PT_MODEL_REVISION else ("Nb1",)
                link_workspace(root, native.Get("w"), cronly.Get("w"), directory / "linked.root",
                               fresh=fresh, groups=groups)
    else:
        raise ValueError(f"Unknown workspace step: {stage}")
