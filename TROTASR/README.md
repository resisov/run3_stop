# TROTASR

The approved Run 3 model combines 2022, 2022EE, 2023, 2023BPix, 2024 and 2025
in 8 channels and 214 bins. The result includes partial systematics, with SR
observations blinded. The SR uses 118 high-dM bins and 30 low-dM GNN bins
(six Nb × NISR categories with five score bins each). The frozen cards use
`autoMCStats 10 1 1`.

## Analysis workflow

From the repository root, run these commands to see the stages and options.

```bash
python -m TROTASR --help
python -m TROTASR limits --help
```

On hep2, run the CLI from `/hep2-scratch/twkim/postprocessing` through the
installed runtime:

```bash
bash TROTASR/workflows/run_inference.sh -m TROTASR limits --help
```

Use the saved fits and figures listed in [current results](NOMINAL_CONTINUATION.md)
for review. The workflow commands below apply to explicitly authorized runs.
Run `histograms`, `merge`, `background` and `cards` in that order, then use the
saved cards for CR-only fits, impacts and limits. Settings are in
`jsons/analysis.json`.
After changing the serialized ID or correction definitions, run
`python -m TROTASR compile` in the installed hep2 runtime. This refreshes the
two callable archives and their manifest; it does not run event production.
`plot --help` lists the existing renderers. Plotting an incomplete limit grid
requires `plot limits --allow-partial`. Leave missing points unfilled.

### Signal intermediate production

The `produce` stage reads `results/signal_source_inventory.json` and processes
one source NanoAOD file for one target era per worker. It applies the current
era's object corrections and stores raw signed generator weights, resolved and
mixed candidates, and aligned generator truth. Multiplicative scale factors
remain in the histogram stage. The accepted input registry is not replaced.

On lxplus, use `/eos/user/t/taiwoo/run3_stop/decaf` and the EOS batch pool:

```bash
source /etc/profile.d/modules.sh
module load lxbatch/eossubmit
python3 -m TROTASR produce --prepare
python3 -m TROTASR produce --submit
```

The stage uses the installed LCG110, LCG105 and CMSSW15 runtimes. Job definitions,
logs and submission state are under `TROTASR/production`. Submission skips active,
previously submitted, published and collected tasks. Source files are read-only.

On hep2, the same stage collects complete new files:

```bash
bash TROTASR/workflows/run_inference.sh -m TROTASR produce --collect --follow --workers 16
```

This command allows up to 16 concurrent transfers. Collection validates the
transferred file's size and ROOT structure before removing the EOS staging copy.
Files go to `inputs/<era>/signal_<source UUID>.root`, receipts to
`production/collected/` and counts to `production/collection.json`.
`--follow` stops when every task has a receipt and no transfers or errors remain.

Remeasure the new signals' efficiencies on hep2 with the same stage:

```bash
bash TROTASR/workflows/run_inference.sh -m TROTASR produce --efficiencies --workers 8
```

Each original NanoAOD file is read once for all six eras. The measurement uses
all events, stored momenta, current jet ID and era-specific working points,
without event skimming or lepton cleaning. AK4 requires pT>30 GeV and |eta|<2.4.
AK8 Top requires pT>400 GeV and mSD>105 GeV; W requires pT>200 GeV and
60<mSD<105 GeV; both use |eta|<2. The score-only denominator has pT>200 GeV
without a mass cut. Unit counts and separate signed-genWeight Top/W counts
retain the existing axes. Completed per-file counts are reused on resume.
`production/efficiencies/state.json` records progress and the definition.
After all files complete, the stage adds exact new dataset keys to the twelve
installed efficiency payloads while preserving existing entries.

Follow the completed measurement with the signal-only update:

```bash
bash TROTASR/workflows/run_inference.sh -m TROTASR produce --refresh-signals --workers 70
```

This waits for an active efficiency measurement, prepares each era's new
mass-point normalization from the full Runs sums, and builds nominal high-/low-dM
histograms. It then replaces only signal nodes in the current merged inputs,
preserving background and data contents. Large merges use bounded streaming.
Per-file failures retry inside this stage; completed files are reused.
Progress is in `production/histograms/state.json`. No fits or plots are run.

#### Recover failed signal tasks

Recover failed task numbers locally on lxplus with
`python3 -m TROTASR produce --recover TASK [TASK ...] --workers 2`.
Recovery checks the Condor queue, refuses active tasks and skips published or
collected files. It retries each task locally up to three times with at most
two workers. See `production/recovery.json` and `production/logs/`.

### Resume an interrupted analysis stage

For per-file histogram production, pass the current input inventory to the
same stage. The JSON contains `tasks` (each with `input`, `output`, `year`,
`entries` and the input's `size`/`mtime_ns` fingerprint) and
`histogram_options` for the requested object views and weight variations.
Paths are relative to `TROTASR`; outputs stay inside the analysis root.

```bash
python -m TROTASR histograms --inventory PATH --workers 32
python -m TROTASR merge --inventory PATH --year 2024 --output OUTPUT.json.gz
```

The histogram stage reuses complete matching files and retries a failed file
once with a smaller event chunk. It records each attempt in the inventory's
`.state.json`; the merge requires complete coverage of that era's task list.
The requested worker count is a ceiling subject to account, memory and disk
headroom. There is no separate recovery controller.

Run the same `python -m TROTASR <stage>` command again. `cards` selects the
furthest committed extraction, template or card state. `limits` reuses accepted
points. `limits`, `cronly` and `impacts` handle their applicable recovery steps
internally.

Preserve live processes, held locks and accepted results. Record interrupted
diagnostics and continue with the next applicable fit recipe. Check input and
model compatibility, SR blinding, finite values, numerical crossings and
resource limits.

### Validation regions

#### MC and signal contamination

Run the full MC scan on hep2 using the existing ntuples. Check hep2
headroom; this stage does not access EOS or lxplus.

```bash
bash TROTASR/workflows/run_inference.sh -m TROTASR histograms --vr-mc --vr-workers 24
```

The scan reuses completed shards and keeps inventory order. Background batches
use up to the requested ceiling without a per-worker memory reservation.
Signal concurrency caps at 10. Each worker is replaced after its own eight
shards; other workers keep processing. VR MC pauses new submissions below
24 GiB available and kills only its own workers below 16 GiB. That stop
preserves completed receipts and is not automatically restarted.
Read `stats/run3/vr/mc_state.json` for progress and
`stats/run3/vr/mc_summary.json.gz` for the result. Low-dM uses
the 30 SR GNN bins and high-dM the 118 merged SR
MET/topology bins. Per-bin statistics and nominal-cross-section signal
contamination are MC estimates, not CR-fit predictions. No observations
enter this MC stage.

SR/CR/VR signal weights follow the Runs sums of the matching signal inputs. A
replacement signal collection must not supply normalization to those inputs.
Completed weighted VR signal shards are reused only with a matching
normalization record or evidence that they predate the replacement.

To evaluate the tight-angle, single-Mixed low-`mTb` candidate on MC only:

```bash
bash TROTASR/workflows/run_inference.sh -m TROTASR histograms --vr-mc --vr-tight-mixed --vr-workers 4
```

This keeps the high-dM SR preselection and selects only one Mixed tag
with finite `0 <= mTb < 175 GeV`. It stops a shard on any overlap with the
final SR or a CR mask, and separately records the intended overlap with
the broad SR preselection. It does not open observed data or modify the
ordinary VR selection. Read `tight_mixed_state.json` and
`tight_mixed_summary.json.gz` under the same VR results directory. A
`--vr-pilot` result is incomplete and is not a signal-contamination estimate.

#### Observed JetMET data

Count observed JetMET events in the same approved VR bins:

```bash
bash TROTASR/workflows/run_inference.sh -m TROTASR vr-data --workers 4
```

The data stage uses the stored data calibration and no MC weights. It
reads only selection columns until events pass the VR gate and writes
only VR counts to `stats/run3/vr/data_summary.json.gz`. Use these counts to
test the CR-constrained prediction independently of fitting or tuning.
SR stays blinded.

### Corrected nominal TROTA friend trees (MC only)

On hep2, use the existing runtime and accepted six-era MC registry. This reads
the intermediate ROOTs, not NanoAOD. It saves one row-aligned friend ROOT per
source under `inputs/friends/trota_nominal/<era>/`, without changing the source
files, observed data, fits, or any running histogram stage.

```bash
bash TROTASR/workflows/run_inference.sh -m TROTASR trota-friends --workers 16
```

`--workers` is a ceiling; the controller limits actual concurrency to available
resources. Use `--pilot` for one representative background and signal shard per era;
the full stage reuses validated friend files. Read
`inputs/friends/trota_nominal/state.json` for progress. The friend tree stores
nominal corrected Mixed and Resolved tags keyed by `(file_id, entry)`, with a row
for every source event. Once production is complete, nominal SR/CR and VR MC
use the validated friend for registered inputs. Weight-only variations reuse
the nominal tags and selection. JES/JER variations with changed model inputs
still run tag inference. Existing frozen templates, including the stored-tag
non-JME templates, are not regenerated. New signal UUIDs outside this registry
are not covered by these friends.

## Datacard review

Get the five signal cards, the CR-only card and their ROOT files from
[`review_examples/`](review_examples/). The cards come from the six-era model.
The ROOT files contain only the shapes these cards use. The signal regions
use background-only Asimov data. Keep `datacards/` and `templates/` together.

### Build the workspaces

Load Combine v10.5.1 in `CMSSW_14_1_0_pre4`. Run the commands below from
`TROTASR/review_examples/` in your checkout. They use T2tt (1200, 500) for the
signal workspace.

```bash
ulimit -Ss 262144
GRID="$PWD"
OUT=$(mktemp -d "$PWD/nps-26-012.XXXXXX")
MODEL=T2tt
POINT=mStop1200_mLSP500
SIG_WS="$OUT/ws_signal.root"

cd "$GRID/datacards"
text2workspace.py datacard_cronly.txt -m 120 --X-allow-no-signal \
  -o "$OUT/ws_cronly.root"
text2workspace.py "datacard_${MODEL}_${POINT}.txt" -m 120 --verbose 0 \
  -o "$SIG_WS"
cd "$OUT"
```

### CR-only fit

Fit the observed CR data with the signal strength fixed to zero.

```bash
combine -M FitDiagnostics ws_cronly.root -m 120 -n _cronly \
  --skipSBFit --robustFit 1 --rMin 0 --rMax 10 \
  --setParameters r=0 --freezeParameters r \
  --cminDefaultMinimizerStrategy 0 --cminDefaultMinimizerTolerance 0.1 \
  --cminFallbackAlgo Minuit2,Migrad,1:0.1
```

The output is `fitDiagnostics_cronly.root` (`fit_b`).

### Impacts

Use the T2tt (1200, 500) workspace with a signal-plus-background Asimov
dataset at r = 1 and fit r over [0, 20].

```bash
combine -M MultiDimFit -d "$SIG_WS" -m 120 -n _initialFit_Test \
  --algo singles --redefineSignalPOIs r --robustFit 1 --saveFitResult \
  -t -1 --expectSignal 1 --setParameterRanges r=0,20 \
  --cminDefaultMinimizerStrategy 0
combineTool.py -M Impacts -d "$SIG_WS" -m 120 -n Test \
  --redefineSignalPOIs r --allPars --doFits --parallel 1 \
  --robustFit 1 --saveFitResult -t -1 --expectSignal 1 \
  --setParameterRanges r=0,20 --cminDefaultMinimizerStrategy 0
combineTool.py -M Impacts -d "$SIG_WS" -m 120 -n Test \
  --redefineSignalPOIs r --allPars -o impacts_mStop1200_mLSP500.json
plotImpacts.py -i impacts_mStop1200_mLSP500.json -o impacts_T2tt_1200_500
```

### Expected limits

Run a blind expected limit for T2tt (1200, 500) with its accepted options.

```bash
RMAX=20
combine -M AsymptoticLimits "$SIG_WS" -m 120 --run blind -n "_$POINT" \
  --cminDefaultMinimizerStrategy 0 --cminDefaultMinimizerTolerance 0.1 \
  --cminFallbackAlgo Minuit2,Migrad,1:0.1 --minosAlgo bisection --picky \
  --rMax "$RMAX" --rRelAcc 0.0025 --rAbsAcc 0.00001 -v 2
```

The output ROOT tree `limit` contains the five expected quantiles.
