# Canonical Low-dM hybrid GNN workflow

This directory contains the frozen Low-dM hybrid GNN and one reproducible
workflow. Superseded pilots, rejected architectures, alternative SR binnings,
retry sidecars, and temporary campaigns are not part of the public interface.

## Frozen model

The selected model is `diagonal_v3_h48_l3_sig010`, epoch 17:

- 48-dimensional jet and global embeddings;
- three message-passing layers;
- significance-loss coefficient 0.10;
- deterministic event split of train:validation:test = 2:1:7;
- collision data excluded from supervised training.

The training domain is the Low-dM preselection with `Nb >= 1`, `Nt = 0`,
`NW = 0`, and `Nres = 0`. It does not require `MET/sqrt(HT) >= 10`, a
particular `NISR`, an ISR--MET delta-phi cut, or a High-dM feature veto.
Collision data are never used for supervised training.

There is exactly one adopted SR binning: **30 bins**. It consists of six
categories, `Nb1_NISR0`, `Nb1_NISR1`, `Nb1_NISR2plus`, `Nb2plus_NISR0`,
`Nb2plus_NISR1`, and `Nb2plus_NISR2plus`, with five frozen,
category-specific GNN-score bins. The authoritative edges are in
`config.json`; no second SR-binning definition is supported. The four
inclusive control-region parent groups only define CR-to-SR normalization
sharing and are not an alternative SR binning.

High-dM bins 1--6 are removed before combination, leaving 73 High-dM bins.
The combined model therefore has 73 + 30 = 103 SR bins.

The frozen PyTorch checkpoint, portable NumPy inference model, selection,
training history, and checksums are in
`models/diagonal_v3_h48_l3_sig010/`.

## Environment

The promoted training environment used Python 3.12.4. Create an isolated
Python 3.12 environment and install the exact promoted package versions:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

PyROOT/Combine and the CMS runtime are external dependencies for ROOT template,
datacard, and limit stages. HTCondor workers must use the EOS-hosted `py38.tgz`
runtime and may not reference AFS or local temporary paths.

## Stable code interface

Only eight top-level Python modules are public: `data.py`, `model.py`,
`train.py`, `valid.py`, `eval.py`, `build_datacard.py`, `merge_datacard.py`,
and `plotting.py`. Implementation modules and batch-worker payloads live under
`_implementation/` and are hidden behind those stable commands.

```bash
python -m autonomous_allhad.gnn_lowdm.train --help
python -m autonomous_allhad.gnn_lowdm.valid --help
python -m autonomous_allhad.gnn_lowdm.eval --help
python -m autonomous_allhad.gnn_lowdm.plotting --help

python -m autonomous_allhad.gnn_lowdm.plotting training-curves
python -m autonomous_allhad.gnn_lowdm.plotting roc
python -m autonomous_allhad.gnn_lowdm.plotting shap
```

The supplementary plotting commands read machine-readable frozen artifacts.
New test-score files include event weights and topology IDs so ROC curves can
be reproduced exactly. Legacy score archives without those arrays are rejected
instead of silently producing an unweighted ROC; rerun `eval test` once to
upgrade such an archive. The already validated legacy weighted ROC remains in
the frozen test-result directory.

## Validation

The preserved model passes PyTorch-versus-NumPy inference parity with a maximum
absolute score difference below `2e-6`. The current unit tests cover the adopted
selection, graph geometry and permutation invariance, significance objective,
final 30-bin mapping, cache retries, and TROTA provenance.

## Current limitation

Low-dM object and weight systematic variations have not yet been propagated
through the GNN score and all SR/CR template migrations. Consequently, existing
combined Low-dM limits remain preliminary and must not be treated as the final
systematics-complete statistical result.
