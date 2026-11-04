# TROTASR

The approved Run 3 model combines 2022, 2022EE, 2023, 2023BPix, 2024 and 2025
in 8 channels and 214 bins. The result includes partial systematics, with SR
observations blinded.

## Analysis workflow

From the repository root, run these commands to see the stages and options.

```bash
python -m TROTASR --help
python -m TROTASR limits --help
```

Run `histograms`, `merge`, `background` and `cards` in that order. Use the saved
cards for CR-only fits, impacts and limits. The settings are in
`jsons/analysis.json`. Find the finished fits and figures in
[current results](NOMINAL_CONTINUATION.md).

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
