import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest


RUNNER = Path(__file__).parents[1] / "workflow/run_asimov_impacts_eos.sh"


@pytest.mark.parametrize("mode", ["recover", "complete", "fail", "bad_checksum", "existing_target"])
@pytest.mark.parametrize("verbosity", [0, 3])
def test_resume_only_missing_and_preserve_on_exit(tmp_path, mode, verbosity):
    source, work, result = [tmp_path / x for x in ("source", "work", "result")]
    for path in (source, work, result):
        path.mkdir()
    workspace = "workspace_mStop1200_mLSP500.root"
    initial = "higgsCombine_initialFit_Test.MultiDimFit.mH1200.root"
    for name in (workspace, initial, "retained.root"):
        (source / name).write_bytes(name.encode())
    if mode == "existing_target":
        (source / "higgsCombine_paramFit_Test_nuisance_a.MultiDimFit.mH1200.root").write_bytes(b"existing")
    env = dict(os.environ, WORKDIR=str(work), RESULTDIR=str(result), OUTDIR=str(result.parent),
               IMPACT_RESUME_DIR=str(source), MASS="1200", IMPACT_EXPECT_SIGNAL="0",
               IMPACT_R_MIN="-20", IMPACT_R_MAX="20", IMPACT_PLOT="0",
               IMPACT_PARALLEL="2", IMPACT_MINIMIZER_STRATEGY="2", MODE=mode)
    env["IMPACT_VERBOSITY"] = str(verbosity)
    env["IMPACT_RESUME_NUISANCES"] = "" if mode == "complete" else "nuisance_a,nuisance_b"
    env["IMPACT_RESUME_WORKSPACE_SHA256"] = hashlib.sha256((source / workspace).read_bytes()).hexdigest()
    env["IMPACT_RESUME_INITIAL_SHA256"] = hashlib.sha256((source / initial).read_bytes()).hexdigest()
    if mode == "bad_checksum":
        env["IMPACT_RESUME_WORKSPACE_SHA256"] = "0" * 64
    body = RUNNER.read_text().split('WORKSPACE="workspace_mStop', 1)[1]
    body = 'WORKSPACE="workspace_mStop' + body
    mocks = r'''
set -euo pipefail
cd "$WORKDIR"
RANGE_ARGS=(--setParameterRanges "r=-20,20")
text2workspace.py() { exit 91; }
combineTool.py() {
    printf '%s\n' "$*" >> calls.log
    if [[ " $* " = *" --doInitialFit "* ]]; then return 92; fi
    if [[ " $* " = *" --doFits "* ]]; then
        [[ " $* " = *" --named nuisance_a,nuisance_b "* ]]
        [[ " $* " = *" --cminDefaultMinimizerStrategy 2 "* ]]
        if [[ "$MODE" = fail ]]; then return 1; fi
        printf 'yes\n' > recovered
        return 0
    fi
    if [[ "$MODE" = complete || -f recovered ]]; then
        printf '{"params":[]}\n' > impacts_mStop1200_mLSP500.json
    elif [[ "$MODE" = recover ]]; then
        printf 'OSError: Failed to open missing ROOT file\n' >&2
        return 1
    else
        printf 'Missing inputs: nuisance_a,nuisance_b\n'
    fi
}
'''
    run = subprocess.run(["bash", "-c", mocks + body], env=env, text=True, capture_output=True)
    expected = 2 if mode == "existing_target" else 1 if mode in {"fail", "bad_checksum"} else 0
    assert run.returncode == expected, run.stderr
    status = json.loads((result / "impact_status.json").read_text())
    assert status["status"] == ("failed" if run.returncode else "fits_complete")
    assert (source / "retained.root").read_bytes() == b"retained.root"
    if mode not in {"bad_checksum", "existing_target"}:
        assert (result / "retained.root").read_bytes() == b"retained.root"
        calls = (result / "calls.log").read_text()
        assert "--doInitialFit" not in calls
        assert calls.count("--doFits") == (0 if mode == "complete" else 1)
        if mode == "recover":
            assert "--doFits" in calls.splitlines()[0]
            assert f"-v {verbosity}" in calls.splitlines()[0]
    if mode == "existing_target":
        assert not (result / "calls.log").exists()
    if mode == "fail":
        assert "Missing inputs:" in (result / "impacts_collect.log").read_text()


def test_existing_submission_writer_records_resume_options(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(RUNNER.parent))
    from build_combined_year_datacards import write_condor_impact_submission

    submit = write_condor_impact_submission(
        str(tmp_path / "card.txt"), tmp_path / "output", RUNNER,
        "mStop1200_mLSP500", "NPS26012_impact_recovery", 0,
        tmp_path / "combine_cmssw_14_1_0_pre4.tgz", "a" * 64,
        extra_environment={"IMPACT_RESUME_DIR": "/eos/example/work", "IMPACT_MINIMIZER_STRATEGY": "2"},
        cpus=2,
    )
    text = submit.read_text()
    assert "IMPACT_R_MIN=-20 IMPACT_R_MAX=20" in text
    assert "IMPACT_RESUME_DIR=/eos/example/work" in text
    assert "IMPACT_MINIMIZER_STRATEGY=2" in text
    assert "request_cpus = 2" in text
