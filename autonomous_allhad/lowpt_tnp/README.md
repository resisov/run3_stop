# Reproducible low-pT electron and muon ID scale factors

This directory is the release entry point for the Run-3 all-hadronic stop
analysis low-pT lepton tag-and-probe measurements.  It covers environment
creation, frozen input discovery, EOS-only HTCondor execution, local recovery,
histogram reduction, simultaneous pass/fail fitting, correctionlib export, and
the standard mplhep figures.

The only executable interface is:

```bash
python autonomous_allhad/workflow/measure_lowpt_lepton.py --help
```

The implementation remains in the shared `workflow/tnp_*` modules so the
electron and muon campaigns cannot silently drift into independent copies.

## Physics definitions

Both corrections measure identification only.  Isolation is deliberately not
part of the probe pass/fail decision.

| Quantity | Electron | Muon |
|---|---|---|
| Resonance | J/psi to electron pairs | J/psi to muon pairs |
| Probe range | 5 < pT < 10 GeV, ECAL fiducial | 5 < pT < 10 GeV, abs(eta) < 2.4 |
| Denominator | GSF electron, conversion veto, lostHits <= 1; no cutBased or isolation requirement | tracker muon; no LooseID or isolation requirement |
| Numerator | cutBased >= Veto | looseId |
| Measured-leg trigger match | none | none |
| Tag | tight electron, pT > 5 GeV, mini-isolation < 0.1 | tight muon, pT > 5 GeV, no isolation requirement |
| External reference | displaced-muon ParkingSingleMuon HLT, independent of dielectron pair | same HLT plus a third tight barrel muon with pT > 12 GeV, distinct from both J/psi legs |

The data reference is the OR of
`HLT_Mu9_Barrel_L1HP10_IP6` and `HLT_Mu10_Barrel_L1HP11_IP6`.
It is required only in data.  The parking HLT is not applied to simulation.
For the muon measurement, the disjoint third-muon offline topology is imposed
on both data and simulation.

The 2025 result uses 2025 ParkingSingleMuon data and the statistically adequate
2024 SPS double-J/psi simulation requested for the measurement:

- electron: `SPS-JpsiJpsiToMuMuEE`, Summer24 NanoAODv15;
- muon: `SPS-JpsiJpsiTo4Mu_Fil-4Mu`, Summer24 NanoAODv15.

The 2025 target-data histogram and 2024 reference-MC histogram are produced
separately.  Reduction checks every selection and axis that affects the
efficiency before replacing only `mc`, `mc_pileup_up`, and
`mc_pileup_down`.  The provenance remains embedded in the final histogram and
fit JSON.  Future POG validation of this cross-year simulation reference is
required.

The committed fit results are deliberately labeled `validation_pending`.
Electron has four directly valid barrel bins out of six and uses the explicit
unity-central endcap policy described below.  All three nominal muon bins are
valid; the alternate-signal MC fit in the first eta bin has a large chi2/ndf,
which is retained in the uncertainty study and caught by the strict validation
test.  Future POG validation of these choices is required.  This release
packages the measurement and candidate correction payloads; it does not
relabel them as POG-approved.

## Binning and fit model

Electron output axes are abs(eta) `[0, 0.8, 1.44, 2.5]` and pT
`[5, 7, 10]` GeV.  The ECAL transition is excluded before histogram filling.
Muon counting starts on abs(eta) `[0, 0.9, 1.2, 2.1, 2.4]` and pT
`[5, 6, 7, 8, 9, 10]` GeV, then is exactly summed to abs(eta)
`[0, 0.9, 1.2, 2.4]` and pT `[5, 10]` GeV before fitting.

The nominal mass width is 40 MeV.  Pass and fail spectra are fitted
simultaneously with a shared double-sided Crystal Ball signal.  The nominal
background is second-order Chebyshev for electrons and exponential for muons.
The total uncertainty combines fit statistics, alternate signal,
alternate background, independent pass/fail response, narrowed mass window,
alternate 20 MeV binning, and pileup variations.

For electron endcap bins whose failing peak does not satisfy the fit-validity
threshold, the released payload uses central value 1.0.  Its symmetric
uncertainty covers the measured nominal ratio's displacement from unity and
the propagated nominal statistical uncertainty.  This policy is activated
explicitly with `--electron-endcap-unity-fallback`; it is never implicit.

## Environment

From the repository root:

```bash
conda env create -f autonomous_allhad/lowpt_tnp/environment.yml
conda activate lowpt-tnp
export MPLCONFIGDIR="${TMPDIR:-/tmp}/lowpt-tnp-mpl"
python autonomous_allhad/workflow/measure_lowpt_lepton.py doctor --repo .
```

The locked Python 3.8 dependencies are in `requirements-py38.txt`.  The code is
also continuously tested with a modern Python runtime.  A full CERN campaign
additionally requires `dasgoclient`, `xrdcp`, `xrdfs`, HTCondor, an EOS schedd,
and a valid VOMS proxy.  Credentials and proxies are never committed.

## Fast release verification

The committed histogram and fit JSONs allow payload and plot regeneration
without CERN services:

```bash
CLI=autonomous_allhad/workflow/measure_lowpt_lepton.py
FIX=autonomous_allhad/reports/lowpt_id_sf_2025_an_handoff/results

python "$CLI" verify-release \
  --repo . --manifest autonomous_allhad/lowpt_tnp/release.json

python "$CLI" render \
  --kind electron \
  --electron-endcap-unity-fallback \
  --result "$FIX/electron/fit_result.json" \
  --histograms "$FIX/electron/histograms.json" \
  --output-dir /tmp/lowpt-tnp-electron

python "$CLI" render \
  --kind muon \
  --result "$FIX/muon/fit_result.json" \
  --histograms "$FIX/muon/histograms.json" \
  --output-dir /tmp/lowpt-tnp-muon
```

`release.json` hashes the executable sources, locked environments, configs,
golden JSONs, pileup inputs, frozen ROOT-file manifests, histogram and fit
results, correctionlib payloads, and every published PNG/PDF.  Maintainers
regenerate it only after an intentional release change with
`release-manifest`; ordinary users should run `verify-release` first.

Use `reproduce` instead of `render` to rerun every nominal and systematic fit
from the frozen histograms before exporting and plotting.  Electron fitting is
CPU intensive and is expected to take several minutes.

```bash
python "$CLI" reproduce \
  --kind electron \
  --electron-endcap-unity-fallback \
  --histograms "$FIX/electron/histograms.json" \
  --output-dir /tmp/lowpt-tnp-electron-refit
```

Each output directory contains `fit_result.json` when fitting was requested,
a correctionlib JSON.GZ, `plots/`, and `reproduction_manifest.json` with input
hashes, dependency versions, and the git commit.

## Frozen campaign records

The `records/` directory contains deterministic compressed manifests for the
exact released inputs:

| Manifest | Files |
|---|---:|
| `electron_2025_data.json.gz` | 37,320 data ROOT files |
| `muon_2025_data.json.gz` | 37,320 data ROOT files |
| `electron_2024_mc.json.gz` | 52 reference-MC ROOT files |
| `muon_2024_mc.json.gz` | 671 reference-MC ROOT files |

They can be decompressed and audited without changing their scope:

```bash
python "$CLI" select-records \
  --records autonomous_allhad/lowpt_tnp/records/electron_2025_data.json.gz \
  --sample data \
  --config autonomous_allhad/workflow/lowpt_electron_measurement/config_2025_id_only_parking_singlemuon.json \
  --output /tmp/electron_2025_data.json
```

To refresh the scope from DAS, use `build-records --sample data` or
`build-records --sample mc`.  A refreshed manifest is a new campaign and must
not be presented as a bytewise reproduction of the release.

## Full CERN production

All Condor inputs, proxy copies, logs, and outputs must live below EOS.  The
preparer rejects non-EOS paths and rejects any generated AFS reference.  It
always creates exactly 20 ROOT files per full shard.

Create a relocatable Python archive and a source archive:

```bash
LOWPT_USER="$(id -un)"
LOWPT_EOS_BASE="/eos/user/${LOWPT_USER:0:1}/${LOWPT_USER}/lowpt_tnp"
mkdir -p "$LOWPT_EOS_BASE/runtime" "$LOWPT_EOS_BASE/configs" "$LOWPT_EOS_BASE/records"

conda-pack -n lowpt-tnp -o "$LOWPT_EOS_BASE/runtime/python38.tar.gz"
git archive --format=tar.gz --output="$LOWPT_EOS_BASE/runtime/repository.tar.gz" HEAD
cp "$X509_USER_PROXY" "$LOWPT_EOS_BASE/runtime/x509up"
chmod 600 "$LOWPT_EOS_BASE/runtime/x509up"
```

Copy the appropriate config and selected plain record manifest to EOS.  Then
prepare four independent campaigns: 2025 data and 2024 reference MC for each
lepton.  Example:

```bash
python "$CLI" prepare \
  --kind electron \
  --records "$LOWPT_EOS_BASE/records/electron_2025_data.json" \
  --repo "$LOWPT_EOS_BASE" \
  --workdir "$LOWPT_EOS_BASE/electron_2025_data" \
  --python-archive "$LOWPT_EOS_BASE/runtime/python38.tar.gz" \
  --runtime-archive "$LOWPT_EOS_BASE/runtime/repository.tar.gz" \
  --proxy "$LOWPT_EOS_BASE/runtime/x509up" \
  --config "$LOWPT_EOS_BASE/configs/electron_2025.json"
```

Before submission, inspect `campaign.json`, `submit.sub`, `queue.tsv`, and one
shard manifest.  Submit only through the EOS schedd:

```bash
unset LD_LIBRARY_PATH PYTHONPATH
module purge
module load lxbatch/eossubmit
condor_submit "$LOWPT_EOS_BASE/electron_2025_data/submit.sub"
```

Queue disappearance is not success.  Verify the expected shard count, JSON
integrity, `files_processed`, `files_failed`, and every process exit code.

### Recovery

Build the residual manifest only after checking primary outputs:

```bash
python "$CLI" recovery-manifest \
  --records "$LOWPT_EOS_BASE/records/electron_2025_data.json" \
  --workdir "$LOWPT_EOS_BASE/electron_2025_data" \
  --output "$LOWPT_EOS_BASE/electron_2025_data/residual.json"
```

Individual read failures are recovered on local lxplus scratch with `recover`.
The tool tries independent XRootD routes, copies with `xrdcp`, counts locally,
normalizes the output back to the logical ROOT path, and writes only compact
JSON.  Reproducibly bad files can be frozen with `finalize-skips`; data
luminosity coverage is then explicitly marked incomplete when applicable.

### Reduction with the 2024 reference MC

Reduce the 2024 MC shards first.  For the muon reference use
`--target-eta-edges 0 0.9 1.2 2.4 --target-pt-edges 5 10`.
Then reduce 2025 data with:

```bash
python "$CLI" reduce \
  --kind electron \
  --input-dir "$LOWPT_EOS_BASE/electron_2025_data/shard_outputs" \
  --records "$LOWPT_EOS_BASE/records/electron_2025_data.json" \
  --config "$LOWPT_EOS_BASE/configs/electron_2025.json" \
  --mc-reference-histograms "$LOWPT_EOS_BASE/electron_2024_mc/histograms.json" \
  --mc-reference-year 2024 \
  --output "$LOWPT_EOS_BASE/electron_2025/histograms.json"
```

Repeat for muons with the final target edges above.  The reducer refuses
selection, topology, axis, or mass-window mismatches and refuses duplicate
successful ROOT files.

### Fit, trigger audit, validation, export, and plots

```bash
python "$CLI" fit --kind electron --electron-endcap-unity-fallback \
  --histograms "$LOWPT_EOS_BASE/electron_2025/histograms.json" \
  --output "$LOWPT_EOS_BASE/electron_2025/fit_result.json"

python "$CLI" audit-trigger --kind electron --sample data \
  --config "$LOWPT_EOS_BASE/configs/electron_2025.json" \
  --file root://example/data.root \
  --max-events 1000000 \
  --output "$LOWPT_EOS_BASE/electron_2025/data_trigger_audit.json"

python "$CLI" validate --kind electron --electron-endcap-unity-fallback \
  --result "$LOWPT_EOS_BASE/electron_2025/fit_result.json" \
  --histograms "$LOWPT_EOS_BASE/electron_2025/histograms.json" \
  --config "$LOWPT_EOS_BASE/configs/electron_2025.json" \
  --data-trigger-audit "$LOWPT_EOS_BASE/electron_2025/data_trigger_audit.json" \
  --mc-trigger-audit "$LOWPT_EOS_BASE/electron_2025/mc_trigger_audit.json" \
  --output "$LOWPT_EOS_BASE/electron_2025/validated_result.json"
```

After visual review, repeat validation with
`--adopt-after-visual-review --visual-review-note "..."`.  Only an adopted
result may be exported without `--candidate`.

```bash
python "$CLI" export --kind electron --electron-endcap-unity-fallback \
  --result "$LOWPT_EOS_BASE/electron_2025/validated_result.json" \
  --output "$LOWPT_EOS_BASE/electron_2025/veto_electron_5to10_sf.json.gz"

python "$CLI" plot --kind electron \
  --result "$LOWPT_EOS_BASE/electron_2025/validated_result.json" \
  --histograms "$LOWPT_EOS_BASE/electron_2025/histograms.json" \
  --output-dir "$LOWPT_EOS_BASE/electron_2025/plots"
```

Muon commands are identical except `--kind muon` and no endcap-unity flag.

## Plot contract

All plotting is implemented in `workflow/plot_measurement.py`:

- standard figures are 8 by 8 inches;
- heatmaps with color bars are 12 by 10 inches;
- only CMS `llabel` and `rlabel` carry header information;
- no `plt.title` is used;
- the header is `CMS Work in progress` and `<year> (13.6 TeV)`;
- mass panels use `Events / 40 MeV` for the nominal result;
- every mass-fit canvas contains four square panels and one shared legend.

## Release tests

```bash
MPLCONFIGDIR=/tmp/lowpt-tnp-mpl \
python -m pytest -q autonomous_allhad/tests/test_lowpt_tnp_release.py
```

The test checks config semantics, frozen-record integrity, correctionlib
round-trips, electron endcap policy, actual analysis payload names, and complete
PNG/PDF rendering for both leptons.
