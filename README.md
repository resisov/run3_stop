# cms-tnp

One configuration-driven package for low- and high-pT electron, muon, and photon tag-and-probe measurements.

## Install

```bash
git clone --single-branch --branch global-tnp-standalone \
  https://github.com/resisov/run3_stop.git cms-tnp
cd cms-tnp
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[all]'
cms-tnp doctor
```

The base package depends only on NumPy. `.[all]` installs ROOT I/O, fitting, correctionlib, and plotting support.

## Start from low-pT J/psi

The default is the low-pT electron J/psi profile.

```bash
cms-tnp init
```

This writes `measurement.json`. A ready starting file is also available at `configs/measurement.json`.

Change these three sections:

```json
{
  "profile": "electron_jpsi_lowpt",
  "id": {
    "fields": [],
    "denominator": "(pt > 5) & (pt < 10) & convVeto & (lostHits <= 1)",
    "pass": "cutBased >= 1"
  },
  "pt_edges_gev": [5, 7, 10],
  "samples": {
    "data": ["dataset=/ParkingSingleMuon*/Run2025*-PromptReco-v*/NANOAOD"],
    "mc": ["/Exact/MC/NANOAODSIM"]
  }
}
```

Add every NanoAOD field used by a private ID to `id.fields`, then use its field name in `denominator` or `pass`.

## Profiles

| profile | probe | resonance | intended range |
|---|---|---|---|
| `electron_jpsi_lowpt` | electron | J/psi | low pT; default |
| `muon_jpsi_lowpt` | muon with a distinct reference muon | J/psi | low pT |
| `electron_z` | electron | Z | high pT |
| `muon_z` | muon | Z | high pT |
| `photon_z` | photon with an electron tag | Z | high pT |

Generate another starting configuration with:

```bash
cms-tnp init --profile photon_z --output photon_id.json
```

The profile supplies tag selection, trigger matching, resonance window, fit models, and default eta bins. Any resolved field can still be overridden in the user JSON.

## Object definitions

Each profile resolves into four physics roles:

- `tag`: the well-identified reference object. When trigger-object matching is enabled, only the tag is matched.
- `probe`: the denominator object. `probe.selection` defines all probes and `probe.pass` splits them into passing and failing spectra.
- `spectator`: an optional object independent of the tag-probe pair. The low-pT muon profile requires a third muon as its trigger-reference candidate; the current profile applies only event-level HLT and does not `TrigObj`-match this spectator.
- `pair`: the tag-probe charge, angular, and invariant-mass requirements.

The built-in definitions are:

| profile | tag | probe denominator and numerator | pair | reference trigger |
|---|---|---|---|---|
| `electron_jpsi_lowpt` | Electron: pT > 5 GeV, ECAL fiducial, `cutBased >= 4`, `miniPFRelIso_all < 0.1` | Electron: 5 < pT < 10 GeV, ECAL fiducial, `convVeto`, `lostHits <= 1`; pass `cutBased >= 1` | opposite charge, 2 < m(ee) < 4 GeV | `HLT_Mu9_Barrel_L1HP10_IP6` OR `HLT_Mu10_Barrel_L1HP11_IP6`; event-level Data selection |
| `muon_jpsi_lowpt` | Muon: pT > 5 GeV, abs(eta) < 2.4, `tightId` | Muon: 5 < pT < 10 GeV, abs(eta) < 2.4, `isTracker`; pass `looseId` | opposite charge, 2.6 < m(mumu) < 3.6 GeV; distinct spectator muon with pT > 12 GeV, abs(eta) < 1.5, `tightId` | `HLT_Mu9_Barrel_L1HP10_IP6` OR `HLT_Mu10_Barrel_L1HP11_IP6`; event-level Data selection |
| `electron_z` | Electron: pT > 35 GeV, ECAL fiducial, `cutBased >= 4` | Electron: pT > 10 GeV, ECAL fiducial, `convVeto`, `lostHits <= 1`; pass `cutBased >= 4` | opposite charge, 60 < m(ee) < 120 GeV | `HLT_Ele32_WPTight_Gsf`, matched to the electron tag |
| `muon_z` | Muon: pT > 26 GeV, abs(eta) < 2.4, `tightId`, `miniPFRelIso_all < 0.1` | Muon: pT > 10 GeV, abs(eta) < 2.4, `isTracker`; pass `tightId` | opposite charge, 60 < m(mumu) < 120 GeV | `HLT_IsoMu24`, matched to the muon tag |
| `photon_z` | Electron: pT > 35 GeV, ECAL fiducial, `cutBased >= 4` | Photon: pT > 20 GeV, ECAL fiducial; pass `cutBased >= 3` | deltaR(e,gamma) > 0.2, 60 < m(egamma) < 120 GeV | `HLT_Ele32_WPTight_Gsf`, matched to the electron tag |

Here ECAL fiducial means barrel `abs(etaSC) < 1.4442` or endcap `1.566 < abs(etaSC) < 2.5`, with `etaSC = eta + deltaEtaSC`.

NanoAOD fields are written without their collection prefix in expressions. For example, `mvaID_WP90` in a photon expression reads `Photon_mvaID_WP90`. `id.fields` adds private probe fields, `id.denominator` overrides `probe.selection`, and `id.pass` overrides `probe.pass`.

Inspect the complete configuration after all profile defaults and user overrides have been applied:

```bash
cms-tnp resolve --config photon_id.json --output photon_id.resolved.json
```

The resolved JSON is the authoritative record of the tag, probe, spectator, pair, axes, trigger, and fit definitions used by counting jobs.

## Reference triggers

The reference trigger selects the event independently of the probe requirement under study. For photon ID, `photon_z` therefore uses an electron trigger matched to the electron tag rather than a photon trigger applied to the photon probe.

```json
"reference_trigger": {
  "paths": ["HLT_Ele32_WPTight_Gsf"],
  "apply_to_data": true,
  "apply_to_mc": true,
  "match_tag": true,
  "object_id": 11,
  "filter_bits": 2,
  "max_delta_r": 0.1
}
```

- `paths` lists NanoAOD HLT branches. Available paths are combined with a logical OR.
- `apply_to_data` and `apply_to_mc` control the event-level HLT requirement independently.
- `match_tag` additionally requires a compatible `TrigObj` within `max_delta_r` of the tag.
- `object_id` is the absolute trigger-object PDG identifier: electron `11`, muon `13`, photon `22`.
- `filter_bits` is applied as a bitmask: `(TrigObj_filterBits & filter_bits) != 0`.

If trigger application is enabled but none of the configured HLT branches exists in a ROOT file, that file is recorded as failed. `TrigObj_filterBits` definitions depend on the NanoAOD era and trigger path and must be checked for the selected campaign. A group of OR-ed paths shares one tag-object ID, filter-bit mask, and matching radius in the current configuration.

`reference_trigger` is a tag-side reference selection, not a probe-trigger numerator. Measuring a trigger efficiency itself requires the trigger under study to be represented by a separate probe-side `TrigObj` matching requirement.

## Inputs

Use an exact DAS dataset beginning with `/`, or a DAS dataset query beginning with `dataset=`.

```bash
cms-tnp discover --config measurement.json --sample data --output data_records.json
cms-tnp discover --config measurement.json --sample mc   --output mc_records.json

cms-tnp make-shards --records data_records.json --output-dir data_shards
cms-tnp make-shards --records mc_records.json   --output-dir mc_shards
```

`lumimask` is resolved relative to the configuration file. Data counted without it are retained as a diagnostic result and marked with an adoption blocker.

## Count

One shard is one independent batch job:

```bash
cms-tnp count --config measurement.json --sample data \
  --shard data_shards/shard_00000.json --output outputs/data_00000.json

cms-tnp count --config measurement.json --sample mc \
  --shard mc_shards/shard_00000.json --output outputs/mc_00000.json
```

The command reads NanoAOD and writes only pass/fail mass histograms. It has no dependency on a specific batch system, filesystem, user directory, or analysis repository.

Optional MC branch expressions and correctionlib weights are configured without code changes:

```json
"weights": {
  "mc_nominal": "genWeight",
  "mc_variations": {},
  "corrections": [{
    "file": "puWeights.json.gz",
    "name": "pileup",
    "inputs": [
      {"field": "Pileup_nTrueInt"},
      {"variation": true}
    ],
    "nominal": "nominal",
    "variations": {
      "pileup_up": "up",
      "pileup_down": "down"
    }
  }]
}
```

List correction inputs in the same order as the correctionlib correction.

## Reduce, fit, export, plot

```bash
cms-tnp reduce outputs/data_*.json outputs/mc_*.json \
  --output histograms.json

cms-tnp fit --histograms histograms.json --output fit_result.json

cms-tnp export --result fit_result.json \
  --output scale_factors.json.gz

cms-tnp plot --result fit_result.json --output-dir plots
```

Or repeat fitting, correctionlib export, and plotting together:

```bash
cms-tnp reproduce --histograms histograms.json --output-dir reproduced
```

The correction inputs are `variation`, `abseta`, and `pt`:

```python
import correctionlib

corrections = correctionlib.CorrectionSet.from_file("scale_factors.json.gz")
weight = corrections["private_electron_id_sf"].evaluate("nominal", abs_eta, pt)
```

## Photon ID SF example

Start from the Z profile:

```bash
cms-tnp init --profile photon_z --output photon_id.json
```

For example, the following configuration measures the NanoAOD photon MVA WP90 ID. Replace only the ID expression, pT bins, datasets, year, and golden JSON for a different campaign.

```json
{
  "schema_version": 1,
  "profile": "photon_z",
  "measurement": "private_photon_mvaid_wp90_sf",
  "year": "2024",
  "id": {
    "fields": ["mvaID_WP90"],
    "denominator": "(pt > 20) & ((abs(eta + deltaEtaSC) < 1.4442) | ((abs(eta + deltaEtaSC) > 1.566) & (abs(eta + deltaEtaSC) < 2.5)))",
    "pass": "mvaID_WP90"
  },
  "pt_edges_gev": [20, 35, 50, 100, 200, 500],
  "abseta_edges": [0.0, 1.4442, 2.5],
  "samples": {
    "data": ["dataset=/EGamma*/Run2024*-*/NANOAOD"],
    "mc": ["dataset=/DYto2L-2Jets_MLL-50*/Run3Summer24NanoAOD-*/NANOAODSIM"]
  },
  "lumimask": "golden.json",
  "correction": {
    "name": "private_photon_mvaid_wp90_sf",
    "description": "Data/MC scale factor for the private photon MVA WP90 ID",
    "flow": "clamp"
  }
}
```

`id.denominator` defines the photon probes before the ID under test. `id.pass` defines the numerator. Every additional NanoAOD photon field used by either expression must appear in `id.fields`, without the `Photon_` prefix. A custom boolean branch called `Photon_privateID`, for example, is configured as:

```json
"id": {
  "fields": ["privateID"],
  "denominator": "(pt > 20) & (abs(eta + deltaEtaSC) < 2.5)",
  "pass": "privateID"
}
```

Validate the resolved configuration before processing files:

```bash
cms-tnp resolve --config photon_id.json --output photon_id.resolved.json
cms-tnp doctor --config photon_id.json
```

### Photon end-to-end execution

1. Query DAS and save deterministic Data and MC file lists:

   ```bash
   cms-tnp discover --config photon_id.json --sample data --output data_records.json
   cms-tnp discover --config photon_id.json --sample mc --output mc_records.json
   ```

2. Optionally take two files from each record list and run the complete chain as a smoke test:

   ```bash
   jq -r '.records[:2][].file_path' data_records.json > data_test_files.txt
   jq -r '.records[:2][].file_path' mc_records.json > mc_test_files.txt

   cms-tnp run-local --config photon_id.json \
     --data-files data_test_files.txt \
     --mc-files mc_test_files.txt \
     --output-dir photon_smoke_test
   ```

3. For a full campaign, split the records into independent 20-file shards:

   ```bash
   cms-tnp make-shards --records data_records.json --output-dir data_shards
   cms-tnp make-shards --records mc_records.json --output-dir mc_shards
   ```

4. On the Linux submit host, build the worker environment archive. Install the package non-editably so that its source is included in the archive:

   ```bash
   python3 -m venv worker-env
   worker-env/bin/python -m pip install --upgrade pip
   worker-env/bin/python -m pip install '.[all,batch]'
   worker-env/bin/venv-pack -o cms-tnp-worker.tar.gz
   ```

5. Package every shard, the resolved configuration, golden JSON, worker environment, and proxy into one Condor campaign. Place the campaign and proxy on EOS at CERN; AFS paths are rejected:

   ```bash
   cms-tnp condor-prepare \
     --config photon_id.json \
     --data-shards data_shards \
     --mc-shards mc_shards \
     --environment cms-tnp-worker.tar.gz \
     --proxy /eos/user/USER/analysis/proxy/x509up \
     --campaign-dir /eos/user/USER/analysis/photon_id_campaign \
     --job-flavour workday
   ```

6. At CERN, select the EOS schedd and submit the generated JDL:

   ```bash
   unset LD_LIBRARY_PATH PYTHONPATH
   module purge
   module load lxbatch/eossubmit
   cms-tnp condor-submit \
     --campaign-dir /eos/user/USER/analysis/photon_id_campaign
   ```

7. Monitor both the queue and the expected JSON outputs:

   ```bash
   cms-tnp condor-status \
     --campaign-dir /eos/user/USER/analysis/photon_id_campaign
   ```

   The command remains incomplete until every expected shard output exists, parses correctly, contains no failed ROOT files, and has full file coverage. An empty Condor queue alone is not success.

8. After `condor-status` reports `complete`, merge the shards, fit Data and MC, calculate the SF and uncertainty, write correctionlib, and make the plots:

   ```bash
   cms-tnp condor-finalize \
     --campaign-dir /eos/user/USER/analysis/photon_id_campaign \
     --output-dir photon_results
   ```

9. The final products are:

   ```text
   photon_results/
   ├── summary.json
   ├── histograms.json
   ├── fit_result.json
   ├── scale_factors.json.gz
   └── plots/
   ```

The exported correction name is `private_photon_mvaid_wp90_sf`, with `nominal`, `up`, and `down` variations.

The `photon_z` profile uses an `HLT_Ele32_WPTight_Gsf`-matched electron tag and a photon probe in the Z mass window. The Data dataset must contain this reference trigger, and the DY simulation must contain the same HLT and trigger-object branches. This electron-as-photon method measures the photon-ID part represented by the probe variables. Measure pixel-seed or electron-veto efficiency separately when the proxy does not represent that requirement.

## Changing the fit functions

Add a `fit` block to the measurement configuration. Profile defaults remain active for keys that are not overridden. This example changes the nominal photon fit from Voigt plus exponential to Crystal Ball plus second-order Chebyshev:

```json
"fit": {
  "signal_model": "crystal_ball",
  "background_model": "chebyshev2",
  "alternate_signal_model": "voigt",
  "alternate_background_model": "exponential",
  "crystal_ball_alpha": 1.5,
  "crystal_ball_n": 3.0,
  "natural_width_gev": 1.2476,
  "peak_bounds_gev": [86.0, 96.0],
  "rebin_factors": [1, 2],
  "window_shrink_fraction": 0.05
}
```

Supported models are:

| component | model names |
|---|---|
| signal | `gaussian`, `double_gaussian`, `crystal_ball`, `voigt` |
| background | `exponential`, `linear`, `chebyshev2` |

The first signal, background, and rebin entries define the nominal fit. The alternate signal, alternate background, second rebin factor, and narrowed fit window are refitted independently. The largest SF displacement from the nominal result is used as the fit-model systematic uncertainty and is combined in quadrature with the statistical uncertainty. Every configured variation must fit successfully for the output bin to remain valid.

Changing only these fit models or fit-variation settings does not require new Condor counting. Refit an existing merged histogram with the modified configuration:

```bash
cms-tnp reproduce \
  --histograms photon_results/histograms.json \
  --config photon_fit.json \
  --output-dir photon_results_crystal_ball
```

The same override can be applied during the first Condor finalization:

```bash
cms-tnp condor-finalize \
  --campaign-dir /eos/user/USER/analysis/photon_id_campaign \
  --config photon_fit.json \
  --output-dir photon_results_crystal_ball
```

Changing `pair.mass_window_gev`, `fit.mass_bins`, the ID selections, or the pT/eta axes changes the counted histograms and therefore requires new shard counting. The refit command detects these changes and stops instead of silently reusing incompatible histograms.

To add a new analytic function that is not in the table, implement it in `_signal` or `_background` in `src/cms_tnp/fit.py`, add its name to the corresponding allowed-model set in `src/cms_tnp/config.py`, and add a fit test.

## Expressions

Selections accept names, numbers, comparisons, `&`, `|`, `~`, arithmetic, and:

```text
abs sqrt log exp minimum maximum where
```

Expressions are parsed by a restricted AST interpreter. Python `eval`, imports, attribute access, and arbitrary function calls are not permitted.

## Test

```bash
python -m pip install -e '.[all,test]'
MPLBACKEND=Agg python -m pytest -q
```

The tests cover compact configuration overrides, separate electron-photon collections, low-pT electron and muon J/psi counting, high-pT muon Z counting, correctionlib event weights, ROOT-to-payload execution, simultaneous pass/fail fitting, HTCondor campaign generation and output gating, correctionlib export, and plotting.
