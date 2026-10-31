# Final canonical workspace

User-approved frozen baseline: 2026-10-31.

| Location | Role |
|---|---|
| `TROTASR/` | Only analysis, plotting and fitting implementation |
| `TROTASR/jsons/analysis.json` | Adopted six-era statistical configuration |
| hep2 `TROTASR/inputs`, `scales`, `models`, `utils/compiled`, `estimations`, `results` | Current input and calibration payloads; large files stay remote |
| hep2 `TROTASR/stats/run3` | Official templates, cards, workspaces, receipts and final results |
| `output/pdf/` | Current delivered figures, compact plot inputs and delivery receipts |
| `AN2019_016_v9.pdf` | Retained physics reference, not an executable analysis copy |
| `docs/`, `.github/` | Existing website publication files, not another analysis implementation |
| hep2 `Stop_Object_Review` | Review material explicitly retained by the user |

The sole hep2 root is `/hep2-scratch/twkim/postprocessing/TROTASR`.
The matching local package is
`/Users/taiwoomac/Documents/All Hadronic Stop Analysis/TROTASR`.

The numerical dependencies used by `TROTASR/workflows/run_inference.sh` are
retained in place: `combine_20260914/CMSSW_14_1_0_pre4` and
`trotasr_inference_py39` under `/hep2-scratch/twkim/envs`. These are dependencies
of the single implementation, not separate analysis checkouts. The unused
`py38` environment was permanently removed with explicit approval.

Do not create additional campaigns, runtimes, source snapshots, restoration
archives or path aliases. Current scientific completion receipts and adopted
reference inputs remain part of the baseline. Their provenance strings do not
authorize reconstruction of retired execution paths.

No scientific source, input, model or figure changes are authorized merely by
review preparation or a status request. Reuse the final products; obtain a new
explicit user instruction before changing or recalculating them. Monitoring
remains stopped. No Git history or Codex conversation database was modified.
