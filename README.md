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
