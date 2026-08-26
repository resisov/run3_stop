# Standalone low-pT lepton tag-and-probe

This repository reproduces the Run-3 low-pT electron and muon identification
scale-factor measurements without importing code or data from another analysis
repository. A clean clone contains the Python implementation, campaign
configuration, Golden JSONs, pileup corrections, frozen NanoAOD file manifests,
reference histograms, fit results, correctionlib payloads, and publication
plots.

The only user interface is `lowpt-tnp` (equivalently
`python -m lowpt_tnp`). It covers environment checks, DAS discovery, Condor
preparation, shard counting, failed-file recovery, reduction, simultaneous
pass/fail fitting, validation, correctionlib export, and plotting.

## Clone and reproduce without CERN services

```bash
git clone --single-branch --branch lowpt-tnp-standalone \
  https://github.com/resisov/run3_stop.git lowpt_tnp
cd lowpt_tnp

conda env create -f environment.yml
conda activate lowpt-tnp
export MPLCONFIGDIR="${TMPDIR:-/tmp}/lowpt-tnp-mpl"

lowpt-tnp doctor
lowpt-tnp verify-release --manifest release.json
```

Refit all nominal and systematic models, rebuild the candidate correctionlib
files, and redraw every figure:

```bash
lowpt-tnp reproduce \
  --kind electron \
  --electron-endcap-unity-fallback \
  --histograms reference/results/electron/histograms.json \
  --output-dir outputs/electron

lowpt-tnp reproduce \
  --kind muon \
  --histograms reference/results/muon/histograms.json \
  --output-dir outputs/muon
```

These commands use no EOS path, proxy, DAS query, or external analysis checkout.
Each output contains `fit_result.json`, a correctionlib JSON.GZ, `plots/`, and a
`reproduction_manifest.json` recording the input hashes, dependency versions,
and git commit.

## Physics definition

Both measurements are identification-only measurements in
`5 < pT < 10 GeV`. Isolation is not part of the probe pass/fail definition.

| Quantity | Electron | Muon |
|---|---|---|
| Resonance | J/psi to electron pairs | J/psi to muon pairs |
| Probe denominator | GSF electron, conversion veto, `lostHits <= 1`; no cut-based ID or isolation | tracker muon; no LooseID or isolation |
| Probe numerator | `cutBased >= Veto` | `looseId` |
| Tag | tight electron, `pT > 5 GeV`, mini-isolation below 0.1 | tight muon, `pT > 5 GeV`, no isolation |
| Measured-leg trigger match | none | none |
| Independent reference | ParkingSingleMuon displaced-muon trigger | same trigger and a distinct third tight barrel muon with `pT > 12 GeV` |

The data trigger is the OR of
`HLT_Mu9_Barrel_L1HP10_IP6` and
`HLT_Mu10_Barrel_L1HP11_IP6`. It is applied to data only. The parking HLT is
not applied to simulation. In the muon measurement, the disjoint third-muon
topology is imposed on both data and simulation so the measured J/psi pair is
not the trigger reference.

The released 2025 scale factors use:

- 2025 `ParkingSingleMuon` data;
- 2024 Summer24 `SPS-JpsiJpsiToMuMuEE` simulation for electrons;
- 2024 Summer24 `SPS-JpsiJpsiTo4Mu_Fil-4Mu` simulation for muons.

Data and simulation are counted independently. Reduction checks the axes and
selection metadata before replacing only the MC histograms in the 2025 data
result with the frozen 2024 reference-MC histograms.

## Binning, fits, and uncertainties

Electron output axes are `abs(eta) = [0, 0.8, 1.44, 2.5]` and
`pT = [5, 7, 10] GeV`; the ECAL transition is excluded. Muons are initially
counted with `abs(eta) = [0, 0.9, 1.2, 2.1, 2.4]` and
`pT = [5, 6, 7, 8, 9, 10] GeV`, then exactly summed to
`abs(eta) = [0, 0.9, 1.2, 2.4]` and `pT = [5, 10] GeV` before fitting.

Pass and fail mass spectra are fitted simultaneously in 40 MeV bins with a
shared double-sided Crystal Ball signal. The nominal electron background is a
second-order Chebyshev polynomial; the nominal muon background is exponential.
The reported uncertainty combines fit statistics and the following variations:

- alternate signal model;
- alternate background model;
- independent pass/fail response shape;
- narrowed fit window;
- alternate 20 MeV mass binning;
- pileup up/down weights.

For electron endcap bins that do not meet the failing-peak validity threshold,
the released candidate returns a central value of 1.0. Its symmetric
uncertainty covers both the measured ratio's displacement from unity and its
propagated nominal statistical uncertainty. This behavior is enabled only by
`--electron-endcap-unity-fallback`.

The reference results remain `validation_pending`. All three nominal muon bins
are valid; the strict validation test records the high alternate-signal MC
chi-square in the first eta bin. Future POG validation of the cross-year MC
reference and fit choices is required.

## Repository contents

```text
configs/                 exact physics and dataset definitions
data/lumimasks/          2024 and 2025 Golden JSONs
data/pileup/             2024 and 2025 correctionlib pileup payloads
records/                 frozen ROOT-file manifests
reference/results/       frozen histograms and fit results
reference/payloads/      candidate electron and muon correctionlib JSON.GZ
reference/plots/         PNG/PDF publication figures
src/lowpt_tnp/           complete implementation
tests/                   numerical, physics, packaging, and privacy tests
release.json             SHA-256 manifest for every released file
```

Frozen input scope:

| Manifest | Files |
|---|---:|
| `electron_2025_data.json.gz` | 37,320 |
| `muon_2025_data.json.gz` | 37,320 |
| `electron_2024_mc.json.gz` | 52 |
| `muon_2024_mc.json.gz` | 671 |

The manifest paths are global CMS `/store/...` paths. No credential, proxy,
private home directory, AFS path, or user-specific EOS path is committed.

## Full CERN production

This section requires CERN access, `dasgoclient`, XRootD, HTCondor, an EOS
schedd, and a valid VOMS proxy. All transferred inputs and campaign outputs
must be under the operator's EOS area; the preparer rejects AFS and non-EOS
paths. The repository itself can be cloned anywhere outside AFS.

Create operator-owned EOS paths without hard-coding a username:

```bash
LOWPT_USER="$(id -un)"
LOWPT_EOS="/eos/user/${LOWPT_USER:0:1}/${LOWPT_USER}/lowpt_tnp"
mkdir -p "$LOWPT_EOS"/{runtime,configs,records,campaigns}
```

Build the relocatable environment and the self-contained worker archive:

```bash
conda-pack -n lowpt-tnp -o "$LOWPT_EOS/runtime/python38.tar.gz"
lowpt-tnp runtime-archive -o "$LOWPT_EOS/runtime/lowpt-tnp-runtime.tar.gz"
cp "$X509_USER_PROXY" "$LOWPT_EOS/runtime/x509up"
chmod 600 "$LOWPT_EOS/runtime/x509up"
```

Materialize plain JSON records and copy the configs used by the four independent
counting campaigns:

```bash
lowpt-tnp select-records --records records/electron_2025_data.json.gz \
  --sample data --config configs/config_2025_id_only_parking_singlemuon.json \
  --output "$LOWPT_EOS/records/electron_2025_data.json"
lowpt-tnp select-records --records records/electron_2024_mc.json.gz \
  --sample mc --config configs/config_2024_id_only_parking_singlemuon.json \
  --output "$LOWPT_EOS/records/electron_2024_mc.json"
lowpt-tnp select-records --records records/muon_2025_data.json.gz \
  --sample data --config configs/config_2025_id_only_parking_external.json \
  --output "$LOWPT_EOS/records/muon_2025_data.json"
lowpt-tnp select-records --records records/muon_2024_mc.json.gz \
  --sample mc --config configs/config_2024_id_only_parking_external.json \
  --output "$LOWPT_EOS/records/muon_2024_mc.json"
cp configs/*.json "$LOWPT_EOS/configs/"
```

Prepare one campaign at a time. This example is the 2025 electron data leg:

```bash
lowpt-tnp prepare \
  --kind electron \
  --records "$LOWPT_EOS/records/electron_2025_data.json" \
  --workdir "$LOWPT_EOS/campaigns/electron_2025_data" \
  --python-archive "$LOWPT_EOS/runtime/python38.tar.gz" \
  --runtime-archive "$LOWPT_EOS/runtime/lowpt-tnp-runtime.tar.gz" \
  --proxy "$LOWPT_EOS/runtime/x509up" \
  --config "$LOWPT_EOS/configs/config_2025_id_only_parking_singlemuon.json"
```

The preparer enforces exactly 20 ROOT files per full shard, embeds no AFS path,
and transfers only the environment archive, standalone runtime archive, config,
proxy, and the individual shard manifest. Submit via the EOS schedd:

```bash
unset LD_LIBRARY_PATH PYTHONPATH
module purge
module load lxbatch/eossubmit
condor_submit "$LOWPT_EOS/campaigns/electron_2025_data/submit.sub"
```

Queue disappearance is not evidence of success. Check Condor exit status,
expected shard count, JSON integrity, `files_processed`, and `files_failed`.
Use `recovery-manifest` followed by `recover` for individual unreadable ROOT
files; recovery downloads each file into local scratch with `xrdcp` and runs the
same `count` implementation. Reproducibly corrupt files are finalized with
`finalize-skips`, which writes both retained records and an audited skip
manifest.

After reducing the data and MC campaigns separately, use `reduce` with
`--mc-reference-histograms` and `--mc-reference-year 2024` to construct the
final cross-year histogram. Then run `fit`, `validate`, `export`, and `plot`, or
use `render` after fitting. Run `lowpt-tnp COMMAND --help` for every exact
argument.

## Plot contract

All plots use mplhep CMS styling, `llabel` and `rlabel` only, and no
`plt.title`. Standard figures are 8 by 8 inches; heatmaps with a color bar are
12 by 10 inches. Mass-fit figures contain four square panels, a single shared
legend outside the panels, vivid blue fit curves, and `Events / 40 MeV` on the
vertical axis. Both PNG and PDF are produced.

## Release maintenance

Run the complete test suite before changing the reference release:

```bash
python -m pytest -q
lowpt-tnp release-manifest --output release.json
lowpt-tnp verify-release --manifest release.json
```

`release-manifest` hashes the source, tests, CI definition, configurations,
Golden JSONs, pileup payloads, input manifests, frozen results, correctionlib
payloads, and all PNG/PDF figures. `release.json` itself is intentionally not
self-hashed.

