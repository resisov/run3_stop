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

## Photon IDs

`photon_z` measures a photon-ID numerator with an electron tag. Keep pixel-seed or electron-veto efficiency separate when the electron-as-photon proxy does not represent that requirement.

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

The tests cover compact configuration overrides, separate electron-photon collections, low-pT electron and muon J/psi counting, high-pT muon Z counting, correctionlib event weights, ROOT-to-payload execution, simultaneous pass/fail fitting, correctionlib export, and plotting.
