# Workspace map

The local analysis root is `TROTASR/` under
`/Users/taiwoomac/Documents/All Hadronic Stop Analysis`. The hep2 analysis root is
`/hep2-scratch/twkim/postprocessing/TROTASR`.

| Location | Contents |
| --- | --- |
| Analysis root, `README.md` | Run commands and datacard review |
| Analysis root, `NOMINAL_CONTINUATION.md` | Approved inputs, official results, figures and update status |
| Analysis root, `jsons/analysis.json` | Six-era configuration |
| hep2 analysis root, `inputs/`, `scales/`, `models/`, `utils/compiled/`, `estimations/`, `results/` | Inputs, calibrations and histograms |
| hep2 analysis root, `stats/run3/` | Cards, templates, workspaces and updated fit results |
| hep2 analysis root, `stats/references/era_merged_recoil/` | Official expected limits and impacts |
| `/hep2-scratch/twkim/postprocessing/Stop_Object_Review` | Object review material |
| Local workspace, `output/pdf/` | Delivered figures, plot inputs and receipts |
| Local workspace, `AN2019_016_v9.pdf` | Physics reference |
| Local workspace, `docs/` and `.github/` | Website files |

`workflows/run_inference.sh` uses `combine_20260914/CMSSW_14_1_0_pre4` and
`trotasr_inference_py39` under `/hep2-scratch/twkim/envs`.

See `AGENTS.md` for analysis policy. It is in the local workspace root and the
hep2 analysis root.
