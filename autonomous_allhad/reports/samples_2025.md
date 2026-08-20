# 2025 data and MC inventory

Machine-readable source: `autonomous_allhad/configs/samples_2025.json`.

## Conditions reference

The adopted reference label is **Run3 25Prompt · Summer24 · NanoAODv15**.
This resolves the apparent data/MC year mismatch: the official 2025 EGM release
explicitly says that its 2025 scale factors must be applied to **Summer24 MC**.
Therefore, using Summer24 MC with 2025 Prompt data is intentional.

The label is a correction-package label, not an exact CMSSW GlobalTag string.
The exact data and MC GlobalTags must still be copied from NanoAOD provenance
before production is frozen.

## Data

The deterministic data expansion contains 56 physical NANOAOD datasets:

- primary datasets: `JetMET0`, `JetMET1`, `EGamma0`–`EGamma3`, `Muon0`, `Muon1`;
- Run2025C: PromptReco v1 and v2;
- Run2025D: PromptReco v1;
- Run2025E: PromptReco v1;
- Run2025F: PromptReco v1 and v2;
- Run2025G: PromptReco v1.

The existing `KNU_2025_v4.json.gz` contains 7,768 split records and 37,933
unique data files: C 7,620; D 8,610; E 4,575; F 9,406; G 7,722.

The C/F v1 and v2 datasets are **not duplicate reprocessings**. CMS DBS run-list
intersections were checked for all eight primary datasets and all 16 v1/v2
pairs had zero overlapping runs. Both versions must be retained.

## MC policy

The starting policy is the validated 2024 Summer24 policy, with exactly one
sample family selected for each physical phase space.

| Process | Selected family | Count | Status |
|---|---|---:|---|
| QCD | `QCD-4Jets_Bin-HT-*` | 11 | adopted HT family |
| Photon+jet | `GJ-4Jets_Bin-HT-*-PTG-*_Par-dRGJ-0p25` | 16 | adopted HT×PTG family |
| DY | flavor-exclusive `DYto2E/2Mu/2Tau-4Jets` | 3 | adopted |
| TT | decay modes plus rare top | 9 | adopted |
| W+jets | 1J/2J PTLNu bins | 10 | adopted after HT rollback |
| Z→νν | 1J/2J PTNuNu bins | 10 | adopted |
| single top | t-, s-, and tW channels | 12 | adopted |
| VV+VVV | WW/WZ/ZZ and WWW/WWZ/WZZ/ZZZ | 7 | adopted |

This gives 78 selected background physical datasets before resolving campaign
versions and the signal grid.

### Adopted QCD HT bins

`40–70`, `70–100`, `100–200`, `200–400`, `400–600`, `600–800`,
`800–1000`, `1000–1200`, `1200–1500`, `1500–2000`, and `≥2000` GeV.

The old `QCD_Bin-PT-*`, QCD0B, and QCDB families must not be mixed with these
inclusive HT samples.

### Adopted photon+jet bins

The 16 selected samples are the available combinations of HT and photon-pT
with `dRGJ > 0.25`:

- PTG 10–100: HT 10–40, 40–100, 100–200, 200–400, 400–600, 600–1000, ≥1000;
- PTG 100–200: HT 40–200, 200–400, 400–600, 600–1000, ≥1000;
- PTG ≥200: HT 40–400, 400–600, 600–1000, ≥1000.

The old `GJ_Bin-PTG-*` family must not be mixed with these samples.

### Produced HT alternatives, not adopted

The following 17 datasets have already been produced and have XSDB cross
sections, but the 2024 shape test was rolled back. They remain comparison
candidates, not default 2025 inputs:

- TT HT: 100–400, 400–800, 800–1500, 1500–2500, ≥2500 GeV;
- W+jets HT×MLNu: HT 40–100, 100–400, 400–800, 800–1500,
  1500–2500, ≥2500 GeV, each split into MLNu 0–120 and ≥120 GeV.

If these are retested, they must replace—not supplement—the TT decay-mode and
W PTLNu families.

## Existing 2025 MC metadata is stale

`analysis/metadata/KNU_2025_v4.json.gz` contains 39,196 unique MC files, but it
predates the adopted sample changes. It still contains QCD pT, GJ PTG-only, and
DY PTLL samples; it has only three FullSim signal anchors and incomplete modern
VVV/rare-top coverage. It must not be submitted as the new 2025 production
input. Regenerate it after this inventory is frozen.

## Correction and production readiness

Available locally:

- 2025 golden JSON covering runs 391658–398903;
- 2025 EGM electron/photon ID payloads;
- 2025 JME JEC/JES/veto-map payloads for C–G;
- 2025 muon payloads;
- processor trigger and MET-filter branches for 2025.

Required before production:

1. Adopt the official preliminary 2025 pileup payload. The current code returns
   unity for 2025 pileup.
2. Adopt the official 2025 BTV payload. Upstream now contains
   `UParTAK4_comb` and `UParTAK4_light` for light, c, and b jets, but the local
   code still returns `None` for 2025.
3. Measure and merge a separate `btageff2025.merged` for the frozen 2025 MC
   list.
4. Fix the 2025 JER keys. The code requests Summer24Prompt24 JRV1 names, while
   the current 2025 JME payload provides the temporary
   `Summer23BPixPrompt23_RunD_JRV1` correction.
5. Measure or validate 2025 MET, photon, and electron trigger scale factors.
   The current electron HLT scale factor is unity and the official EGM release
   does not yet include trigger SFs.
6. Freeze T2tt, T2bW, and T2tb signal coverage and normalization. Do not infer
   it from the three old FullSim anchors.
7. Record exact NanoAOD GlobalTags from provenance.

Until these are resolved, the sample inventory is ready but 2025 production is
not physics-ready.

## Provenance

- 2025 BTV upstream commit: `19718caf08a12945512eaeb51019c9158641a9a4`
  (2026-06-26; added c-jet SFs and variations).
- 2025 LUM upstream commit: `2fcd89bbdd6cc7b4e54eb320324a56a584456d09`
  (2026-06-05; preliminary pileup weights).
- QCD XSDB values: `workflow/qcdht/xsec.json`.
- Photon+jet XSDB values: `workflow/gj4jets/xsec.json`.
- TT/W HT XSDB values: `workflow/twht/xsec.json`.
- Validated sample baseline: `workflow/plot2024/norm.json`.
