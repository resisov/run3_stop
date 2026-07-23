# Prototype Statistical Model Gate

Maturity: prototype, expected-only, 2024-only, high-DeltaM-only, nominal-only.

The fast-analysis package must not create production datacards until the all-hadronic control strategy is made explicit and validated. The initial candidate channel set is:

- `cat2_LLCR_highDeltaM` for lost-lepton/top/W constraints;
- `cat3_QCDCR_highDeltaM` for multijet validation or constraint;
- `cat4_GCR_highDeltaM` for photon-to-Z validation or constraint;
- `cat5_DY2E_highDeltaM` and `cat6_DY2M_highDeltaM` if needed for transfer-factor validation;
- `cat7_SR_highDeltaM` blinded, with Asimov observation only for expected limits.

Open decisions before cards:

- transfer factors and their provenance;
- whether rate parameters are justified per bin/category;
- nuisance correlations across CR and SR;
- signal contamination in control regions;
- QCD treatment and closure uncertainty;
- whether DY2E/DY2M are required for a defensible model.

Until these are resolved, the CLI `card` and `limit` commands remain scaffolded and return a gated status.
