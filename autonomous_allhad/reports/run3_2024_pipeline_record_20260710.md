# Run-3 2024 All-Hadronic Stop Pipeline Record

Generated: 2026-07-10

Scope: 2024 data-analysis pipeline state in this workspace.  The `analysis/`
tree is treated as read-only for this cleanup pass and was not modified.

## Configuration

- Main config: `autonomous_allhad/configs/run3_2024.yaml`
- Year: `2024`
- Luminosity: `109.82 fb^-1`
- Main command interface: `./autonomous_allhad/analysisctl`
- Official all-command: `./autonomous_allhad/analysisctl all --config autonomous_allhad/configs/run3_2024.yaml`
- GitHub Pages workflow: `.github/workflows/pages.yml`

## Official analysisctl status

The official `analysisctl` path has complete feature-side subset artifacts, but
the final full-production physics chain is still blocked.

- ROOT files read in subset path: `14`
- Feature rows: `290006`
- Bad files in subset path: `0`
- `autonomous_allhad/hists.npy`: complete
- `autonomous_allhad/hist_index.json`: complete, `9429` histogram items
- `autonomous_allhad/plots/plot_manifest.json`: complete, `32` plots
- Signal DAS discovery: complete
- FastSim signal files processed: `61`
- Signal mass points with yields: `352`
- Official datacards: blocked, `0` produced
- Official expected limits: blocked, no official Combine-compatible datacards
- Official full-production normalization: blocked

Known blockers:

- Full-production feature table or aggregate finalization is not complete.
- Full dataset signed sumw/Runs denominators are incomplete for final normalization.
- Correction weight products and systematic shifted event weights are not complete.
- Final search bins are not selected through the official gate.
- Systematic templates are unavailable through the official gate.
- Manual legacy validation boundary has not been supplied.

## Flat ROOT production path

The high/low-dM flat-ROOT path is the current substantial production artifact.
It is separate from the official `analysisctl make-datacards` gate.

- Worker: `autonomous_allhad/autonomous_allhad/flat_ntuple_worker.py`
- Nominal campaign: `flat_ntuple_highlowdm_20260704_nominal`
- Nominal Condor cluster: `930070`
- Planned nominal jobs: `4286`
- Nominal output directory:
  `autonomous_allhad/workflow/flat_ntuple_highlowdm_20260704_nominal_outputs`
- Nominal ROOT outputs: `4162`
  - Data ROOT outputs: `2716`
  - MC ROOT outputs: `1443`
  - Signal ROOT outputs: `3`
- Recovery campaign: `highlowdm_recovery10_20260705_nominal`
- Recovery Condor cluster: `931017`
- Recovery jobs/ROOT outputs: `372`
- Final histogram input ROOT list:
  `autonomous_allhad/workflow/highlowdm_full_20260705_root_inputs.txt`
- Final histogram input ROOT count: `4534`

Flat ROOT payload policy:

- Output tree: `Events`
- Stored event weight: raw `gen_weight`
- Luminosity, cross-section, pileup, b-tag, lepton/photon ID/HLT, and top-pT
  weights are deferred to post-skim histogram/template production.
- Nominal object/MET-changing corrections are applied before skim where required.
- Shifted object/MET variations are separate production inputs, not nominal skim
  weights.

## Histogram production

Official subset histograms:

- Source: `autonomous_allhad/outputs/real_feature_table.csv`
- Output: `autonomous_allhad/hists.npy`
- Index: `autonomous_allhad/hist_index.json`
- Histogram items: `9429`
- Data files processed: `3`
- Background files processed: `11`
- Signal files processed: `61`
- Rows read from feature table: `290006`

High/low-dM full flat-ROOT histograms:

- Primary output:
  `autonomous_allhad/workflow/highlowdm_full_20260705_flat_hists.json`
- Latest expanded output:
  `autonomous_allhad/workflow/highlowdm_full_20260707_flat_hists_selected_recoil6_with_nt0_wsplit_lowdm_variables.json`
- Input ROOT count: `4534`
- Events processed: `317400623`
- Chunk outputs in latest expanded pass: `43`
- Region count: `18`
- Search-bin scheme count: `11`
- Recoil bins: `[250, 300, 350, 400, 500, 800, 1500]`

The latest expanded payload and the 2026-07-10 full test run use the same exact
set of 4,534 intermediate ROOT files.  The selected production scheme is
`boosted_an17_selected_recoil6_with_nt0_wsplit_SR`: 9 category blocks with
6 recoil bins each, for 54 high-dM SR bins.

Available post-skim weight variations in the high/low-dM histogram payload:

- `nominal`
- `pileupUp`, `pileupDown`
- `electron_idUp`, `electron_idDown`
- `electron_hltUp`, `electron_hltDown`
- `muon_idUp`, `muon_idDown`
- `muon_hltUp`, `muon_hltDown`
- `photon_idUp`, `photon_idDown`
- b-tag correlated/uncorrelated heavy- and light-flavor variations

Important caveat:

- JEC and MET unclustered shapes are deferred.
- b-tag SF efficiency input `hists/btageff2024.merged` is missing in this
  workspace for many signal points, so several b-tag entries use unity fallback.

## Plot production

Official subset plots:

- Manifest: `autonomous_allhad/plots/plot_manifest.json`
- Plot count: `32`
- Copied public plots: `docs/plots`

High/low-dM web plots:

- Page: `docs/highlowdm_full_20260705/index.html`
- Plot directory: `docs/highlowdm_full_20260705/plots`
- PNG count: `136`
- Summary: `docs/highlowdm_full_20260705/plots/flat_plot_summary.json`

Plot content includes:

- high-dM CR recoil plots
- high-dM SR recoil plots
- inclusive, `N_t = 0`, and `N_t >= 1` diagnostic splits
- AN17 17-bin diagnostic plots
- latest selected 9 x 6 recoil category plot, including separate
  `N_t = 0, N_W = 0` and `N_t = 0, N_W >= 1` blocks
- low-dM one-bin CR/SR plots
- low-dM physical-variable validation plots

The 2026-07-10 full test regenerated 130 PNG and 130 PDF files with the
existing `plot_control_search_bins_style.py::draw_flat_blocks` implementation.
The existing canvas-size policy was retained; the 54-bin SR PNG is 2161 x 1409
pixels.

## Combine and limits

Official `analysisctl` datacards and expected limits are blocked.

Separate high/low-dM Combine outputs exist under `analysis/combine/...`.  The
`analysis/` tree is not modified by this cleanup pass.

Historical 42-bin selected-recoil6 low-dM baseline:

- Combine directory:
  `analysis/combine/highlowdm_full_20260706_selected_recoil6_lowdm`
- Datacards: `352`
- `higgsCombine_*.AsymptoticLimits*.root` outputs: `352`
- Expected-limit JSON: complete, `352 / 352` mass points
- Template ROOT:
  `analysis/combine/highlowdm_full_20260706_selected_recoil6_lowdm/templates_selected_an17_recoil6_lowdm.root`

Channel structure:

- `cat2_LLCR_highDeltaM`: 6 recoil bins
- `cat3_QCDCR_highDeltaM`: 6 recoil bins
- `cat4_GCR_highDeltaM`: 6 recoil bins
- `cat5_DY2E_highDeltaM`: 6 recoil bins
- `cat6_DY2M_highDeltaM`: 6 recoil bins
- `cat7_SR_selected_an17_recoil6`: 42 selected high-dM bins
- `cat8_SR_lowDeltaM_onebin`: 1 low-dM SR bin

High/low-dM web limit outputs:

- Directory: `docs/highlowdm_full_20260705/limits`
- Selected 7 x 6 plus low-dM expected-limit JSON:
  `docs/highlowdm_full_20260705/limits/expected_limits_selected_recoil6_lowdm.json`
- Status: complete, `352 / 352` mass points
- Overlay plot:
  `docs/highlowdm_full_20260705/limits/expected_limit_overlay_run2.png`

The high/low-dM page reports:

- Inclusive high-dM SR: 352 points, 47 expected-excluded points, median expected r 21.4
- High-dM SR `N_t >= 1`: 352 points, 63 expected-excluded points, median expected r 17.4
- High-dM SR `N_t = 0`: 352 points, 20 expected-excluded points, median expected r 69.5
- High-dM CR/SR split by `N_t`: 352 points, 65 expected-excluded points, median expected r 14.8
- High-dM `N_t` split plus low-dM SR: 352 points, 65 expected-excluded points, median expected r 14.7
- Selected high-dM 7 x 6 recoil categories plus low-dM SR: 352 points, 65 expected-excluded points, median expected r 18.3

### Full EOS ROOT-to-web test run, 2026-07-10

A separate full-input test run was completed without modifying analysis/.

- Run directory: autonomous_allhad/workflow/full_run_20260710_root_hist_limit_web
- Full 54-bin histogram payload:
  autonomous_allhad/workflow/highlowdm_full_20260707_flat_hists_selected_recoil6_with_nt0_wsplit_lowdm_variables.json
- Full intermediate ROOT inputs: 4,534
- Histogram events processed: 317,400,623
- Plot records: 130, with 130 PNG and 130 PDF outputs
- Plot implementation: existing
  autonomous_allhad/workflow/plot_control_search_bins_style.py::draw_flat_blocks
- Plot sizes: existing plotting-code canvas dimensions retained
- Combine channels: 7
- High-dM SR channel: cat7_SR_selected_an17_recoil54_nt0_wsplit, 54 bins
- High-dM category structure: 9 category blocks x 6 recoil bins
- Low-dM SR channel: one inclusive bin
- Datacards and mass points: 352
- Combine: v10.0.1, AsymptoticLimits, blind expected
- Valid limit ROOT outputs: 352 / 352; missing 0; invalid 0
- Expected-excluded grid points with median r < 1: 69
- Maximum expected-excluded stop mass for mLSP <= 100 GeV: 1,350 GeV
- One compressed boundary point, mStop2500_mLSP2300, lacked four expected
  quantiles at the default rMax.  It was rerun with rMax=1,000,000 and passed
  the five-quantile ROOT validation.  No additional high-mass optimization was
  performed.
- Machine limit summary:
  autonomous_allhad/workflow/full_run_20260710_root_hist_limit_web/combine_selected_recoil54_nt0_wsplit_lowdm/expected_limits.json
- ROOT template:
  autonomous_allhad/workflow/full_run_20260710_root_hist_limit_web/combine_selected_recoil54_nt0_wsplit_lowdm/templates_selected_an17_recoil54_nt0_wsplit_lowdm.root
- SUS-19-010 overlay implementation: existing heatmap plotting code
  autonomous_allhad/workflow/build_combine_inputs_from_preview.py::plot_contour
- Run-2 reference: CMS SUS-19-010 observed and expected contours
- Overlay display range: mStop 600-1500 GeV, mLSP 0-1500 GeV
- Overlay content: log10(expected 95% CL limit on sigma/sigma_theory) heatmap,
  Run-3 54-bin median expected and +/-1 sigma contours, and Run-2 observed and
  expected contours
- Overlay axis layout: legacy unconstrained axis aspect; equal-aspect forcing is
  disabled so the axes fill the 12 x 10 canvas as in the historical plot
- Overlay size: 12 x 10 inch canvas, 2160 x 1800 PNG
- Execution status: autonomous_allhad/workflow/full_run_20260710_root_hist_limit_web/limit_execution_status.json
- Deployment status: autonomous_allhad/workflow/full_run_20260710_root_hist_limit_web/github_pages_deployment_status.json

For the 157 common points with mStop <= 1,500 GeV, the 54-bin expected r is
lower than the 42-bin baseline at all 157 points.  The median r54/r42 ratio is
0.956.  Expected-excluded grid points increase from 65 to 69.

All generated runtime, histogram, limit, and web outputs for this run used EOS paths. No AFS or system /tmp writes were used by the run.

## Web publication

- Independent full-run page: https://resisov.github.io/run3_stop/full_run_20260710_root_hist_limit_web/
- GitHub Pages workflow: completed successfully
- Workflow run: https://github.com/resisov/run3_stop/actions/runs/29149924493
- Published commit: 293f215bad9af553dbafb68387ad6cb52bfb9966
- Public HTML, SUS-19-010 overlay PNG, 54-bin definition JSON,
  42/54-bin comparison JSON, and expected-limits JSON: HTTP 200 verified
- Public overlay PNG: 2160 x 1800, SHA-256 matches the local generated file
- Public categorization: 54 high-dM bins in 9 category blocks
- Public expected-limit JSON: complete, 352 points, 0 missing, 0 invalid
- Sensitive absolute-path scan: clean
- Main docs/index.html was not modified by this independent-page deployment.

## Cleanup boundary

Preserve for reproducibility:

- `autonomous_allhad/autonomous_allhad/flat_ntuple_worker.py`
- current `autonomous_allhad/workflow/highlowdm_full_20260705*` and
  `highlowdm_full_20260707*` machine-readable outputs
- current `autonomous_allhad/workflow/flat_ntuple_*highlowdm*` ROOT-output and
  Condor manifest directories
- `docs/highlowdm_full_20260705`
- official `docs/index.html`, `docs/monitor.html`, `docs/data`, `docs/plots`
- `.github/workflows/pages.yml`
- `analysis/` tree, untouched by this cleanup pass

Safe cleanup categories:

- duplicate root-level generated HTML files that are superseded by `docs/`
- stale logs, pid files, tmux/nohup/watch snapshots
- obsolete 2025 probe records outside `analysis/`
- older June partial-preview/recovery scratch outputs not used by the current
  high/low-dM pipeline
- local transient work directories whose merged JSON or public plot outputs are
  already preserved
