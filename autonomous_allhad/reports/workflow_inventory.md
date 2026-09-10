# Current canonical workflow

2024+2025 · High-dM 73 + frozen Low-dM GNN 30 · lepton veto >10 GeV · DY 71–111 GeV.

EOS repository: `/eos/user/t/taiwoo/run3_stop/decaf`.

| Location | Role |
|---|---|
| `autonomous_allhad/workflow/flat2024_v8/outputs/nominal`, `flat2025_v8/outputs/nominal` | Current intermediate ROOT and existing production metadata; Events/TROTA and Top/W truth remain in the same ROOT. |
| `autonomous_allhad/workflow/histograms/lepton_veto10_20260908` | Canonical histograms, frozen production code, current cards, combined limits and plot exports. |
| `autonomous_allhad/workflow/plot2024`, `plot2025` | Canonical histogram aliases and input/normalization configuration; not a second set of histograms. |
| `autonomous_allhad/reports/background_estimation_lepton_veto10_20260908` | Current histogram-derived TF, RZ, Sgamma and double-ratio inputs. |
| `autonomous_allhad/gnn_lowdm/models`, `configs`, `inputs` | Frozen GNN model, configuration and required input manifests. |
| `analysis/data`, `analysis/hists` | Required corrections, compiled payloads, b-tag and Top/W efficiencies. |
| `autonomous_allhad/workflow/topw_truth_20260909` | Current main-worker submission, job logs and state for in-place ROOT augmentation. No independent truth production implementation. |

Runtime: `/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python`; batch uses the existing `py38.tgz` and validated Combine archive.

The extra `.topw.json` writer and reader have been removed. Completion information belongs in the existing production metadata. ROOT data and physics selections are unchanged.
