# Unified DY normalization measurement

This directory is the single implementation of the adopted DY normalization measurement. It replaces the scattered an_zinv and RZ_UT scripts that previously lived under workflow/.

The adopted result is \(R_Z(N_b)\), not \(R_Z(U_T)\). The categories are \(N_b=1\) and \(N_b\geq2\), measured separately in dielectron and dimuon data for High- and Low-\(\Delta m\). The fit also returns the non-DY sideband normalization \(R_T(N_b)\). Channel-specific \(R_Z\) values are combined with inverse-variance weights.

High- and Low-\(\Delta m\) are not independent workflows. One feature scan fills both regimes. Events whose Low-\(\Delta m\) AK8 topology is ambiguous after lepton removal receive one conditional sparse NanoAOD refinement. The merger then writes one audited measurement JSON containing `rz_high`, `mll_high`, `rz_low`, and `mll_low`; one report command renders both regimes with the same plotting implementation.

## Physics definition

Selections and corrected objects come from the current real_subset_worker.py feature production. The legacy analysis/stop_processor_v4.py and ids.py are not used.

- on-Z: \(81 < m_{\ell\ell} < 101\) GeV.
- off-Z: \(50 < m_{\ell\ell} < 81\) GeV or \(m_{\ell\ell} > 101\) GeV.

For every channel and \(N_b\) category, the two observed counts are fitted as

\[
N_{\mathrm{data}}^{w}=R_Z N_{\mathrm{DY}}^{w}+R_T N_{\mathrm{other}}^{w},
\qquad w\in\{\mathrm{on},\mathrm{off}\}.
\]

Data counts use Poisson likelihoods. Weighted-MC template statistics enter as Gaussian constraints using sumw2. Both scale factors are non-negative. DY uses only the inclusive `DYto2E-4Jets`, `DYto2Mu-4Jets`, and `DYto2Tau-4Jets` samples with `MLL-50`; legacy pT-binned DY samples are rejected to prevent double counting. Z-containing minor backgrounds (TTZ, WZ, ZZ, WWZ, WZZ, ZZZ, WZG) share the Z-like component; all remaining samples enter the non-DY component. MC weights include current normalization and b-tag scale factors.

High-\(\Delta m\) is determined from decorated feature ROOT files. Low-\(\Delta m\) has one exact follow-up: events whose AK8 topology can change after lepton removal are read sparsely from NanoAOD and reconstructed with the canonical JEC, AK4/AK8 ID, cleaning, and topology-veto helpers in real_subset_worker.py.

Post-scaling \(m_{\ell\ell}\) plots visualize fitted inputs. They are not closure tests because the same on/off-Z counts determine \(R_Z\) and \(R_T\).

## Directory contents

| File | Responsibility |
|---|---|
| __main__.py | One CLI with all commands |
| prepare_features.py | Split feature inputs and write the HTCondor submit file |
| feature_stage.py | Select events, evaluate weights, and write feature inputs |
| merge_features.py | Losslessly merge disjoint feature partitions |
| prepare_lowdm.py | Resolve only topology-ambiguous NanoAOD sources and prepare exact-refinement jobs |
| run_lowdm_partition.py | Run one resumable exact-refinement partition |
| lowdm_recovery.py | Canonical sparse NanoAOD topology reconstruction |
| sparse.py | Stable file IDs and bounded sparse-read windows |
| merge_lowdm.py | Enforce complete accounting and build one High/Low-dM measurement artifact |
| model.py | Yield containers and the on/off-Z profile-likelihood fit |
| report.py | Produce \(R_Z\), \(R_T\), and pre/post \(m_{\ell\ell}\) plots |
| validate.py | Check accounting and reproduce the frozen 2024 factors |
| publish.py | Copy only validated report assets into docs/ |
| reference_2024.json | Compact accepted-campaign regression reference |

## Environment

Run from the outer autonomous_allhad/ directory. On lxplus use the established Python 3.8 environment:

~~~bash
cd /eos/home-t/taiwoo/run3_stop/decaf/autonomous_allhad
DY_PYTHON=/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python
DY_REPO=/eos/home-t/taiwoo/run3_stop/decaf
DY_WORK=/eos/home-t/taiwoo/run3_stop/decaf/autonomous_allhad/workflow/dy_estimation_2024
AUTONOMOUS_ALLHAD_SIDECAR_STORE=$DY_REPO/autonomous_allhad/workflow/sidecars_main.sqlite
~~~

The reader first accepts an adjacent legacy JSON sidecar and otherwise uses
the indexed main store above. The current nominal input has no adjacent JSON
sidecars: all 2,919 payloads are losslessly stored and checksum-verified in
`sidecars_main.sqlite`.

Every command has help:

~~~bash
$DY_PYTHON -m autonomous_allhad.dy_estimation report --help
~~~

## Execution order

### 1. Prepare the two feature campaigns

DY_ROOT_LISTS must contain the current decorated nominal feature ROOT lists for background MC and final data. DY_NORMALIZATION is the matching current 2024 normalization JSON. The preparer keeps EGamma data for DY2E, Muon data for DY2M, and excludes signal ROOTs.

~~~bash
DY_ROOT_LISTS="<mc-feature-list> <data-feature-list>"
DY_NORMALIZATION=<normalization-json>

$DY_PYTHON -m autonomous_allhad.dy_estimation prepare-features \
  --repo "$DY_REPO" \
  --input-roots $DY_ROOT_LISTS \
  --normalization "$DY_NORMALIZATION" \
  --output-dir "$DY_WORK/features_ee" \
  --channel DY2E

$DY_PYTHON -m autonomous_allhad.dy_estimation prepare-features \
  --repo "$DY_REPO" \
  --input-roots $DY_ROOT_LISTS \
  --normalization "$DY_NORMALIZATION" \
  --output-dir "$DY_WORK/features_mumu" \
  --channel DY2M
~~~

### 2. Submit to the EOS schedd

~~~bash
condor_submit -name bigbird24 "$DY_WORK/features_ee/submit.sub"
condor_submit -name bigbird24 "$DY_WORK/features_mumu/submit.sub"
~~~

Queue disappearance is not success. Every expected output JSON must exist, have feature_stage_complete status, and close its input/completed ROOT accounting.

### 3. Merge each channel

~~~bash
$DY_PYTHON -m autonomous_allhad.dy_estimation merge-features \
  --inputs "$DY_WORK"/features_ee/outputs/*.json \
  --output "$DY_WORK/features_ee.json"

$DY_PYTHON -m autonomous_allhad.dy_estimation merge-features \
  --inputs "$DY_WORK"/features_mumu/outputs/*.json \
  --output "$DY_WORK/features_mumu.json"
~~~

### 4. Prepare and submit the conditional exact refinement

DY_SHARD_BUNDLE is the current source-record tarball. DY_SOURCE_LIST_DIRS contains feature ROOT source lists used to resolve stable file IDs.

~~~bash
DY_SHARD_BUNDLE=<shard-records.tar.gz>
DY_SOURCE_LIST_DIRS="<source-list-dir-1> <source-list-dir-2>"

$DY_PYTHON -m autonomous_allhad.dy_estimation prepare-exact-refinement \
  --repo "$DY_REPO" \
  --ee "$DY_WORK/features_ee.json" \
  --mumu "$DY_WORK/features_mumu.json" \
  --shard-bundle "$DY_SHARD_BUNDLE" \
  --source-list-dir $DY_SOURCE_LIST_DIRS \
  --output-dir "$DY_WORK/lowdm_exact" \
  --files-per-job 5

condor_submit -name bigbird24 "$DY_WORK/lowdm_exact/submit.sub"
~~~

A partition is idempotent: a complete output with the same stem and closed event accounting is reused. The queue columns are named `manifest_path` and `output_path`; do not use the HTCondor `output` keyword as an item-data variable.

### 5. Merge one High/Low-dM measurement and enforce accounting

~~~bash
$DY_PYTHON -m autonomous_allhad.dy_estimation merge-measurement \
  --ee "$DY_WORK/features_ee.json" \
  --mumu "$DY_WORK/features_mumu.json" \
  --expected "$DY_WORK/lowdm_exact/expected.json" \
  --output "$DY_WORK/dy_measurement.json"
~~~

This fails unless all expected partitions exist, all candidate files are represented, every candidate event is matched, and the failure list is empty. The output contains both the direct High-\(\Delta m\) measurement and the exact-refined Low-\(\Delta m\) measurement.

### 6. Run the frozen-result regression

~~~bash
$DY_PYTHON -m autonomous_allhad.dy_estimation validate \
  --ee "$DY_WORK/features_ee.json" \
  --mumu "$DY_WORK/features_mumu.json" \
  --low-exact "$DY_WORK/dy_measurement.json" \
  --output "$DY_WORK/validation.json"
~~~

Use --verify-hashes only with the frozen July 2026 intermediate JSON files. A new production changes hashes; accounting must still close and physics differences must be reviewed against reference_2024.json.

### 7. Build both reports from the single measurement

~~~bash
$DY_PYTHON -m autonomous_allhad.dy_estimation report \
  --measurement "$DY_WORK/dy_measurement.json" \
  --selection both \
  --output-dir "$DY_WORK/report"
~~~

### 8. Publish only validated assets

From the checkout that owns docs/:

~~~bash
$DY_PYTHON -m autonomous_allhad.dy_estimation publish \
  --high "$DY_WORK/report/highdm" \
  --low "$DY_WORK/report/lowdm" \
  --destination ../docs/dy_rz_nb_run2method_20260731
~~~

The publisher uses a fixed asset allow-list and removes stale files from the High- and Low-dM report directories. Commit and push only after validation succeeds and figures are visually inspected.

## Accepted 2024 regression point

The reference records 2,392 dielectron feature ROOTs, 2,325 dimuon feature ROOTs, and 21,967 Low-dM sparse candidates in 2,541 NanoAOD files. All 509 partitions completed, all 21,967 candidates matched, and 15,458 passed exact topology reconstruction.

| Regime | \(N_b=1\) | \(N_b\geq2\) |
|---|---:|---:|
| High-\(\Delta m\) | \(0.7222\pm0.0469\) | \(0.7014\pm0.0753\) |
| Low-\(\Delta m\) | \(0.6093\pm0.0333\) | \(0.6277\pm0.0760\) |

## Deliberately excluded

This package contains no \(R_Z(U_T)\) fit, no algebraically saturated post-fit closure, no GCR photon-shape measurement, no Combine/datacard construction, and no legacy stop processor or ID implementation. Those are different tasks and must not be mixed into the DY normalization measurement.
