import copy
import json
from pathlib import Path

import awkward as ak
import correctionlib
import numpy as np
import pytest
import uproot

from cms_tnp.cli import main
from cms_tnp.condor import finalize_campaign
from cms_tnp.config import eta_axis, load_config, validate_config
from cms_tnp.count import _empty, _fill, count_files
from cms_tnp.fit import apply_fit_config
from cms_tnp.profiles import PROFILES, resolve_profile
from cms_tnp.reduce import merge

from test_reproduce import _payload
from test_root_end_to_end import _root


def _user(profile="electron_jpsi_lowpt"):
    return {"profile": profile, "measurement": "eta_test", "year": "2025"}


@pytest.mark.parametrize("profile", sorted(PROFILES))
@pytest.mark.parametrize("nested", [False, True])
def test_signed_axis_preserves_profile_selections_and_roundtrips(profile, nested):
    user = _user(profile)
    axis = {"eta_edges": [-2.4, 0, 2.4]}
    user.update({"axes": axis} if nested else axis)
    config = resolve_profile(user)
    validate_config(config)
    assert eta_axis(config) == ("eta", [-2.4, 0, 2.4])
    assert config["tag"] == PROFILES[profile]["tag"]
    assert config["probe"] == PROFILES[profile]["probe"]
    assert config["pair"] == PROFILES[profile]["pair"]
    assert config["reference_trigger"] == PROFILES[profile]["reference_trigger"]
    assert resolve_profile(config) == config
    assert "abseta_edges" not in config["axes"]


@pytest.mark.parametrize("overrides", [
    {"eta_edges": [-2.5, 0, 2.5], "abseta_edges": [0, 2.5]},
    {"eta_edges": [-2.5, 0, 2.5], "axes": {"abseta_edges": [0, 2.5]}},
    {"axes": {"eta_edges": [-2.5, 0, 2.5], "abseta_edges": [0, 2.5]}},
    {"eta_edges": [-2.5, 0, 2.5], "axes": {"eta_edges": [-2.5, 2.5]}},
    {"abseta_edges": [-2.5, 0, 2.5]},
    {"eta_edges": [0, 1, 1]},
    {"eta_edges": [2.5, -2.5]},
    {"eta_edges": [0, float("nan")]},
    {"eta_edges": [-float("inf"), 0, float("inf")]},
])
def test_ambiguous_or_invalid_axes_are_rejected(overrides):
    with pytest.raises(ValueError, match="eta_edges"):
        validate_config(resolve_profile({**_user(), **overrides}))


def test_positive_only_signed_axis_is_not_inferred_as_absolute():
    config = resolve_profile({**_user(), "eta_edges": [0, 1, 2.5]})
    assert eta_axis(config) == ("eta", [0, 1, 2.5])
    assert eta_axis(resolve_profile(_user()))[0] == "abseta"


def test_signed_entrypoint_and_doctor(tmp_path, capsys):
    path = tmp_path / "signed.json"
    assert main(["init", "--profile", "photon_z", "--signed-eta", "--output", str(path)]) == 0
    capsys.readouterr()
    user = json.loads(path.read_text())
    assert user["eta_edges"] == [-2.5, -1.4442, 0.0, 1.4442, 2.5]
    assert "abseta_edges" not in user
    assert user["correction"]["name"] == "private_photon_z_signed_eta_sf"
    assert main(["doctor", "--config", str(path)]) == 0
    doctor = json.loads(capsys.readouterr().out)
    assert doctor["version"] == "0.4.0"
    assert doctor["config"]["eta_axis"] == {
        "mode": "signed", "expression": "eta + deltaEtaSC",
        "edges": user["eta_edges"], "correction_input": "eta",
    }
    resolved = tmp_path / "resolved.json"
    assert main(["resolve", "--config", str(path), "--output", str(resolved)]) == 0
    assert load_config(resolved) == load_config(path)
    with pytest.raises(FileExistsError):
        main(["init", "--signed-eta", "--output", str(path)])


def test_signed_filling_uses_standard_boundaries_and_preserves_sumw2():
    # Left edges are included; the last right edge is included by histogramdd.
    etas = [-2.5, -0.6, 0.0, 0.6, 2.5, 3.0]
    pairs = ak.Array([[{"probe": {"bin_eta": eta, "bin_pt": 7., "passing": True}}]
                      for eta in etas])
    weights = ak.Array([1., 2., 3., 4., 5., 6.])
    masses = ak.Array([[3.1]] * len(etas))
    signed = _empty((2, 1, 1))
    _fill(signed, pairs, masses, weights,
          [np.array([-2.5, 0., 2.5]), np.array([5., 10.]), np.array([2.6, 3.6])],
          absolute_eta=False)
    assert signed["pass_sumw"][:, 0, 0].tolist() == [3., 12.]
    assert signed["pass_sumw2"][:, 0, 0].tolist() == [5., 50.]
    absolute = _empty((1, 1, 1))
    _fill(absolute, pairs, masses, weights,
          [np.array([0., 2.5]), np.array([5., 10.]), np.array([2.6, 3.6])])
    for key in signed:
        np.testing.assert_array_equal(signed[key].sum(axis=0), absolute[key][0])


def _shard():
    result = _payload()
    result["processing"] = {
        "files_expected": 1, "files_processed": 1, "files_failed": [],
        "events_read": 1, "pairs_selected": 1,
    }
    return result


def test_merge_rejects_same_edges_with_different_eta_semantics():
    original = _shard()
    signed = copy.deepcopy(original)
    signed["probe_eta_edges"] = signed.pop("probe_abseta_edges")
    with pytest.raises(ValueError, match="eta axis"):
        merge([original, signed])
    new = copy.deepcopy(original)
    new["probe_eta_expression"] = "eta + deltaEtaSC"
    with pytest.raises(ValueError, match="eta axis"):
        merge([original, new])
    changed = copy.deepcopy(new)
    changed["probe_eta_expression"] = "eta"
    with pytest.raises(ValueError, match="eta axis"):
        merge([new, changed])


def test_refit_rejects_axis_mode_or_coordinate_changes():
    payload = _payload()
    config = {
        "measurement": payload["measurement"],
        "probe": {"collection": payload["probe_collection"],
                  "selection": payload["probe_selection"],
                  "pass": payload["pass_selection"], "eta": "eta + deltaEtaSC"},
        "axes": {"eta_edges": payload["probe_abseta_edges"],
                 "pt_edges_gev": payload["probe_pt_edges_gev"]},
        "pair": {"mass_window_gev": [2.6, 3.6]},
        "fit": {**payload["fit"], "mass_bins": 50},
    }
    with pytest.raises(ValueError, match="recount"):
        apply_fit_config(payload, config)
    payload["probe_eta_edges"] = payload.pop("probe_abseta_edges")
    payload["probe_eta_expression"] = "eta + deltaEtaSC"
    assert apply_fit_config(payload, config)["probe_eta_expression"] == "eta + deltaEtaSC"
    config["probe"]["eta"] = "eta"
    with pytest.raises(ValueError, match="eta expression"):
        apply_fit_config(payload, config)


def test_signed_root_to_condor_finalize_and_payload(tmp_path, monkeypatch):
    # Synthetic NanoAOD only: different efficiencies on either detector side.
    paths = {}
    for sample, efficiencies, seed in (("data", (0.60, 0.90), 1), ("mc", (0.80, 0.80), 2)):
        path = tmp_path / f"{sample}.root"
        _root(path, 0.8, seed)
        with uproot.open(path) as source:
            branches = source["Events"].arrays(library="ak", how=dict)
        size = len(branches["event"])
        side = np.where(np.arange(size) < size // 2, -1., 1.)
        # etaSC has the opposite sign to raw eta, testing the stored expression.
        branches["Electron_eta"] = ak.Array(np.column_stack([-0.1 * side] * 2))
        branches["Electron_deltaEtaSC"] = ak.Array(np.column_stack([0.7 * side] * 2))
        rng = np.random.default_rng(seed + 10)
        passed = rng.random(size) < np.where(side < 0, efficiencies[0], efficiencies[1])
        branches["Electron_cutBased"] = ak.Array(np.column_stack([np.full(size, 4), passed.astype(int)]))
        with uproot.recreate(path) as target:
            target["Events"] = branches
        paths[sample] = path
    (tmp_path / "golden.json").write_text(json.dumps({"1": [[1, 1]]}))
    config = resolve_profile({
        **_user(), "eta_edges": [-2.5, 0., 2.5], "pt_edges_gev": [5., 10.],
        "lumimask": "golden.json", "input": {"event_filters": []},
        "reference_trigger": {"paths": [], "apply_to_data": False, "apply_to_mc": False, "match_tag": False},
        "fit": {"signal_model": "gaussian", "alternate_signal_model": "double_gaussian"},
        "weights": {"mc_variations": {"weight_up": "1.1 * genWeight", "weight_down": "0.9 * genWeight"}},
    })
    campaign = tmp_path / "campaign"
    (campaign / "outputs").mkdir(parents=True)
    jobs = []
    for index, (sample, path) in enumerate(paths.items()):
        signed = count_files(config, [str(path)], sample, base_dir=tmp_path)
        absolute_config = copy.deepcopy(config)
        absolute_config["axes"].pop("eta_edges")
        absolute_config["axes"]["abseta_edges"] = [0., 2.5]
        absolute = count_files(absolute_config, [str(path)], sample, base_dir=tmp_path)
        assert signed["status"] == "complete"
        assert signed["probe_eta_expression"] == "eta + deltaEtaSC"
        for variation in signed["samples"]:
            for moment in signed["samples"][variation]:
                np.testing.assert_allclose(
                    np.sum(signed["samples"][variation][moment], axis=0),
                    absolute["samples"][variation][moment][0], rtol=1.e-12, atol=1.e-9,
                )
        name = f"{sample}.json"
        (campaign / "outputs" / name).write_text(json.dumps(signed))
        jobs.append({"job_id": index, "sample": sample, "result": name, "files_expected": 1})
    (campaign / "campaign.json").write_text(json.dumps({"jobs": jobs}))
    # Inspect labels from the real plotting path, preserving normal file output.
    import cms_tnp.plot as plotting
    original_save = plotting._save
    legends, ylabels = [], []

    def capture(fig, path):
        for ax in fig.axes:
            legends.extend(ax.get_legend_handles_labels()[1])
            ylabels.append(ax.get_ylabel())
        return original_save(fig, path)

    monkeypatch.setattr(plotting, "_save", capture)
    output = tmp_path / "results"
    assert finalize_campaign(campaign, output)["status"] == "complete"
    result = json.loads((output / "fit_result.json").read_text())
    assert result["probe_eta_edges"] == [-2.5, 0., 2.5]
    assert all(item["valid"] for item in result["bins"])
    correction = correctionlib.CorrectionSet.from_file(str(output / "scale_factors.json.gz"))["eta_test"]
    assert [item.name for item in correction.inputs] == ["variation", "eta", "pt"]
    assert correction.evaluate("nominal", -0.6, 7.) == pytest.approx(0.60 / 0.80, abs=0.06)
    assert correction.evaluate("nominal", +0.6, 7.) == pytest.approx(0.90 / 0.80, abs=0.06)
    for eta in (-0.6, 0.6):
        assert correction.evaluate("down", eta, 7.) < correction.evaluate("nominal", eta, 7.) < correction.evaluate("up", eta, 7.)
    assert any(r"-2.5<\eta<0" in label for label in legends)
    assert any(r"0<\eta<2.5" in label for label in legends)
    assert r"Electron $\eta$" in ylabels
    assert not any(r"|\eta|" in label for label in legends + ylabels)
    assert all(Path(path).is_file() for path in json.loads((output / "plots/plots.json").read_text())["outputs"])
