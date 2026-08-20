# Run-3 All-Hadronic Stop Analysis

CMS Run-3 all-hadronic stop search code and 2024 analysis products. The
repository uses the deterministic classic command surface:

- `python -m automation.cli`: deterministic compilation, Coffea execution,
  HTCondor submission, reduction, plotting, templates, datacards, and expected
  limits around the classic analysis code.

The current public 2024 results are at
[resisov.github.io/run3_stop](https://resisov.github.io/run3_stop/nominal_plots_2024_fullselection_v5_dyexclusive_t2models_freebkg_20260728/).

## 1. Prerequisites

Run full production on LXPlus or another Linux host with CMS software, EOS,
XRootD, VOMS, and HTCondor access. A laptop clone is sufficient for source
inspection, unit tests, and some plotting, but not for the complete
NanoAOD-to-limit workflow.

Required software is:

- Python 3.8 with Coffea, Awkward, Uproot, Hist, correctionlib, NumPy, SciPy,
  Matplotlib, PyYAML, cloudpickle, and pytest;
- a valid CMS VOMS proxy for remote NanoAOD and Condor access;
- `condor_submit` and `condor_q` for distributed production;
- ROOT and CMS Combine for datacards and limits;
- Git and, for pull requests, the GitHub CLI `gh`.

The validated production Python currently used by the analysis owner is:

```text
/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python
```

Collaborators must either have read access to that environment or use an
equivalent environment and change `execution.python` in their private config.
There is currently no public lockfile, so a newly built environment must pass
the tests and one-file smoke test below before physics production.

## 2. Clone and enter the repository

```bash
git clone https://github.com/resisov/run3_stop.git
cd run3_stop
git submodule update --init --recursive
```

All commands below run from the repository root unless they explicitly contain
`cd analysis`.

Choose the Python executable for this shell:

```bash
export STOP_PYTHON=/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python
"$STOP_PYTHON" --version
"$STOP_PYTHON" -c 'import awkward, coffea, correctionlib, hist, numpy, uproot; print("analysis imports: OK")'
```

If the shared environment is unavailable, point `STOP_PYTHON` to your compatible
Python environment.

## 3. Restore required large inputs

Git contains the processor, metadata, object definitions, correction bundles,
and adopted JSON scale factors. It intentionally excludes most generated or
large artifacts. A fresh clone does not contain:

- compiled `analysis/data/stop_2024_*.processor` files;
- `condor/analysis.tgz` and `condor/py38.tgz`;
- NanoAOD, intermediate ROOT, `.futures`, merged/scaled histograms, or Combine
  workspaces.

The compiled processors and Condor archive are generated later. MC production
also requires the canonical b-tag efficiency input:

```text
analysis/hists/btageff2024.merged
```

Obtain the current validated file from the production coordinator or canonical
EOS workspace. Do not restore the old file from Git history: it predates later
QCD, GJ, and signal efficiency additions.

```bash
mkdir -p analysis/hists

# Example source. Confirm access and the current approved checksum first.
xrdcp root://eosuser.cern.ch//eos/user/t/taiwoo/run3_stop/decaf/analysis/hists/btageff2024.merged \
  analysis/hists/btageff2024.merged

sha256sum analysis/hists/btageff2024.merged
```

Compare the SHA-256 with the corresponding campaign manifest or append audit.
Never accept this payload based on its filename alone.

Verify files expected from Git:

```bash
test -r analysis/metadata/KNU_2024_v4.json.gz
test -r analysis/data/corrections.coffea
test -r analysis/data/ids.coffea
test -r analysis/data/common.coffea
test -r analysis/data/AnalysisSF/2024/met_trigger_sf.json.gz
test -r analysis/data/AnalysisSF/2024/photon_trigger_sf.json.gz
test -r analysis/data/AnalysisSF/2024/veto_electron_5to10_sf.json.gz
test -r analysis/data/AnalysisSF/2024/loose_muon_5to10_sf.json.gz
```

The Run-2 reference `AN2019_016_v9.pdf` is intentionally local-only. Copy it to
the repository root when a reference-validation stage requires it.

## 4. Create a user-specific configuration

`configs/stop_2024.yaml` is the reference configuration. It contains the
analysis owner's Python, EOS, proxy, and Condor paths, so another user must not
submit it unchanged.

```bash
cp configs/stop_2024.yaml configs/stop_2024.local.yaml
```

Edit at least these fields in `configs/stop_2024.local.yaml`:

```yaml
execution:
  python: /absolute/path/to/your/python
  workers: 8

condor:
  analysis_tarball: /eos/user/<initial>/<user>/run3_stop/condor/analysis.tgz
  python_tarball: /eos/user/<initial>/<user>/run3_stop/condor/py38.tgz
  proxy: /eos/user/<initial>/<user>/run3_stop/proxy/x509up_u<uid>
  initialdir: /eos/user/<initial>/<user>/run3_stop/condor_log
```

Never commit or publish a proxy. Set a short shell variable for the following
commands:

```bash
export STOP_CONFIG=configs/stop_2024.local.yaml
```

## 5. Validate the checkout

Run the repository tests and configured-path validation:

```bash
"$STOP_PYTHON" -m pytest autonomous_allhad/tests
"$STOP_PYTHON" -m automation.cli --help
"$STOP_PYTHON" -m automation.cli validate --config "$STOP_CONFIG"
```

After compiling nominal, require the compiled processor too:

```bash
"$STOP_PYTHON" -m automation.cli validate \
  --config "$STOP_CONFIG" \
  --require-compiled
```

Do not continue while `validate` reports an error.

## 6. Run a one-file smoke test

Use exact metadata keys whenever possible. Prefix matching is useful for
sharded keys; legacy substring matching can select unintended samples.

First print the exact commands without executing them:

```bash
"$STOP_PYTHON" -m automation.cli run \
  --config "$STOP_CONFIG" \
  --shift nominal \
  --dataset-prefix TTto2L2Nu_ \
  --max-datasets 1 \
  --max-files 1 \
  --dry-run
```

Then run the bounded test:

```bash
"$STOP_PYTHON" -m automation.cli run \
  --config "$STOP_CONFIG" \
  --shift nominal \
  --dataset-prefix TTto2L2Nu_ \
  --max-datasets 1 \
  --max-files 1
```

The result is written below `analysis/hists/stop_2024_nominal/`. It is valid
only when the `.futures` file is nonempty and Coffea can load it:

```bash
"$STOP_PYTHON" -c 'from coffea.util import load; load("analysis/hists/stop_2024_nominal/<metadata-key>.futures"); print("futures: OK")'
```

Replace `<metadata-key>` with the exact output filename printed by the run.

## 7. Compile the configured systematic processors

Configured shifts are `nominal`, `jesTotalUp`, `jesTotalDown`,
`metUnclusteredUp`, `metUnclusteredDown`, `jerUp`, and `jerDown`.

```bash
for shift in nominal jesTotalUp jesTotalDown metUnclusteredUp metUnclusteredDown jerUp jerDown; do
  "$STOP_PYTHON" -m automation.cli compile \
    --config "$STOP_CONFIG" \
    --shift "$shift"
done
```

Processors are written under `analysis/data/`. A readable existing processor is
reused unless `--force` is given.

## 8. Prepare and submit HTCondor production

Create a CMS proxy and copy it to your own protected EOS location:

```bash
voms-proxy-init -voms cms -valid 72:00
mkdir -p /eos/user/<initial>/<user>/run3_stop/proxy
cp "$(voms-proxy-info -path)" \
  /eos/user/<initial>/<user>/run3_stop/proxy/x509up_u"$(id -u)"
chmod 600 /eos/user/<initial>/<user>/run3_stop/proxy/x509up_u"$(id -u)"
voms-proxy-info \
  -file /eos/user/<initial>/<user>/run3_stop/proxy/x509up_u"$(id -u)" \
  -timeleft
```

Create `condor/py38.tgz` from the validated runtime and
`condor/analysis.tgz` from the checkout. `setup_condor.sh` documents the
historical packaging procedure but is owner-path-specific; collaborators must
not run it unchanged. Confirm both archives in the private config contain the
same code and payloads used by the smoke test.

Preview one job:

```bash
"$STOP_PYTHON" -m automation.cli submit \
  --config "$STOP_CONFIG" \
  --shift nominal \
  --max-jobs 1 \
  --dry-run
```

Submit one pilot, validate it, and only then submit the remaining nominal jobs:

```bash
"$STOP_PYTHON" -m automation.cli submit \
  --config "$STOP_CONFIG" \
  --shift nominal \
  --max-jobs 1

"$STOP_PYTHON" -m automation.cli status \
  --config "$STOP_CONFIG" \
  --shift nominal

"$STOP_PYTHON" -m automation.cli submit \
  --config "$STOP_CONFIG" \
  --shift nominal
```

Repeat for required systematic shifts. Data are included only in nominal; the
shifted dataset lists are MC-only.

## 9. Monitor, recover, and postprocess

Queue disappearance is not success. The output must exist, be nonempty, and be
readable with Coffea.

```bash
"$STOP_PYTHON" -m automation.cli status --config "$STOP_CONFIG" --shift nominal
"$STOP_PYTHON" -m automation.cli missing --config "$STOP_CONFIG" --shift nominal
```

Preview a recovery, then submit only missing, zero-size, or unreadable outputs:

```bash
"$STOP_PYTHON" -m automation.cli retry \
  --config "$STOP_CONFIG" \
  --shift nominal \
  --max-jobs 10 \
  --dry-run

"$STOP_PYTHON" -m automation.cli retry \
  --config "$STOP_CONFIG" \
  --shift nominal \
  --max-jobs 10
```

After every expected output is readable, reduce, merge, and scale it:

```bash
"$STOP_PYTHON" -m automation.cli postprocess \
  --config "$STOP_CONFIG" \
  --shift nominal
```

Do not use `--allow-partial` for physics results. It is only for explicitly
labeled diagnostics. For a corrupted input ROOT, retry an alternate endpoint,
record the file and quantified loss in the bad-file manifests, and normalize MC
using retained-file sum of weights.

Run `status` and `postprocess` for every shift required by plotting or template
production.

## 10. Produce plots, templates, cards, and expected limits

Run a nominal-only plot check first:

```bash
"$STOP_PYTHON" -m automation.cli plot \
  --config "$STOP_CONFIG" \
  --nominal-only
```

After external JES and unclustered-MET `.scaled` files exist, make the full
configured plots and ROOT template:

```bash
"$STOP_PYTHON" -m automation.cli plot --config "$STOP_CONFIG"
"$STOP_PYTHON" -m automation.cli template --config "$STOP_CONFIG"
"$STOP_PYTHON" -m automation.cli validate-template --config "$STOP_CONFIG"
```

The classic template currently exposes example signal process names
`SMS_2Stop_mStop600`, `SMS_2Stop_mStop1000`, and `SMS_2Stop_mStop1500`. Inspect
the template before choosing one; do not guess a process name.

```bash
export STOP_SIGNAL=SMS_2Stop_mStop1000

"$STOP_PYTHON" -m automation.cli datacard \
  --config "$STOP_CONFIG" \
  --signal "$STOP_SIGNAL"

"$STOP_PYTHON" -m automation.cli limit \
  --config "$STOP_CONFIG" \
  --signal "$STOP_SIGNAL"

"$STOP_PYTHON" -m automation.cli report --config "$STOP_CONFIG"
```

`limit` runs blind expected `AsymptoticLimits`; it does not run an observed
limit. The minimal classic datacard does not infer transfer factors, rate
parameters, or nuisance correlations. For the adopted AN control-region model,
use the validated builders in `autonomous_allhad/workflow/` and
`autonomous_allhad/reports/datacard_control_region_strategy_20260813.md`.

## 11. Output locations

| Stage | Default output |
| --- | --- |
| compiled processor | `analysis/data/stop_2024_<shift>.processor` |
| dataset Coffea result | `analysis/hists/stop_2024_<shift>/*.futures` |
| merged/scaled result | `analysis/hists/stop_2024_<shift>.merged` and `.scaled` |
| classic plots | `analysis/plots/stop_2024/` |
| classic ROOT template | `analysis/templates/templates_metpt.root` |
| classic datacard | `analysis/datacards/stop_2024_shapes.txt` |
| classic expected limit | `analysis/limits/` |
| public site source | `docs/nominal_plots_2024_fullselection_v5_dyexclusive_t2models_freebkg_20260728/` |

Large outputs remain on EOS and are ignored by Git. Compact manifests,
checksums, validation summaries, final plots, and public reports form the
reproducibility record.

## 12. Physics and production safeguards

- Do not modify selections, categories, bins, normalization, transfer factors,
  or nuisance correlations without labeling and validating a physics change.
- Do not mix overlapping sample families. In particular, never combine old
  PT-binned DY with adopted `DYto2E/Mu/Tau-4Jets` production.
- Do not treat a Condor job leaving the queue as successful; validate its exit
  record and output.
- Do not overwrite validated output or repeat completed shards.
- Do not silently skip corrupted ROOT files. Update
  `autonomous_allhad/workflow/bad_files.json`, `bad_files.txt`, and
  `file_validation_summary.json` with the affected dataset, error, alternate
  endpoint attempt, and quantified loss.
- Do not publish credentials, proxies, private tokens, or private EOS paths in
  the generated website.
- Use the established main plotter and adopted style; do not add
  campaign-specific replacement plotters.

Before using a result in the AN, require exact input coverage or documented
loss, finite normalization, required systematic variations, readable
histograms and templates, valid datacards, converged fits, and matching public
checksums.

## 14. GitHub attribution and pull requests

GitHub attributes commits by author email, not by whether the displayed name is
Korean or English. This project uses the CERN address below; it must be added to
and verified on the `resisov` GitHub account:

```bash
git config --global user.email "taiwoo.kim@cern.ch"
git config user.email "taiwoo.kim@cern.ch"
```

Confirm it and the authenticated GitHub account before committing:

```bash
git config --get user.name
git config --get user.email
gh auth status
```

If the GitHub CLI token has expired, reauthenticate in the browser:

```bash
gh auth login -h github.com -w
```

Use a feature branch and pull request for normal development:

```bash
git switch -c codex/<short-topic>
git add <reviewed-files>
git commit -m "Describe the analysis change"
git push -u origin codex/<short-topic>
gh pr create --draft --fill
```

A contribution appears when the author email is linked to the account and the
commit reaches the repository's default branch. Changing only `user.name` does
not fix attribution. Existing commits can be associated by adding and verifying
their email on GitHub; do not rewrite published history without an explicit
decision.

## 15. Principal references and retained products

Principal physics and implementation references are:

- `AN2019_016_v9.pdf` — Run-2 strategy reference, local-only;
- `analysis/processors/stop_processor_v4.py` — current Run-3 processor;
- `analysis/utils/ids.py` — object definitions and working points;
- `analysis/utils/corrections.py` — corrections and scale factors.

Retained records include:

- `analysis/data/AnalysisSF/2024/` — adopted MET/photon trigger and low-pT
  lepton scale factors;
- `autonomous_allhad/reports/` — measurement, closure, and datacard strategy
  reports;
- `autonomous_allhad/validation/` — compact validation records;
- `autonomous_allhad/signals/` — signal discovery, mass grids, and stop cross
  sections;
- `docs/nominal_plots_2024_fullselection_v5_dyexclusive_t2models_freebkg_20260728/`
  — current public 2024 plots, limits, and impacts.

Remaining work is to add the other Run-3 years, propagate and validate the full
systematic model, and regenerate combined cards, limits, and impacts only after
those inputs pass validation.
