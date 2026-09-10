# Run-3 All-Hadronic Stop Analysis

2024+2025 · High-dM 79−6 = 73 bins + frozen Low-dM GNN 30 bins.

Development branch: `run3_full_analysis`.

- [High-dM canonical code](autonomous_allhad/reports/highdm_canonical_dependencies.md)
- [Low-dM canonical dependencies](autonomous_allhad/reports/lowdm_canonical_dependencies.md)
- [Current workflow locations](autonomous_allhad/reports/workflow_inventory.md)
- [Canonical histogram manifest](autonomous_allhad/reports/lepton_veto10_canonical_20260908.json)

Analysis runtime: `/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python`.
Batch analysis runtime: the existing `py38.tgz`; Combine uses the existing validated Combine runtime.

ROOT inputs and fit products remain on EOS. Local plotting uses compact exports.
Selections: lepton veto >10 GeV; DY dilepton mass 71–111 GeV; SR remains blinded.
Top/W truth augmentation is implemented in the main `flat_ntuple_worker.py`, in the existing intermediate ROOT; completion metadata update the existing production JSON.
