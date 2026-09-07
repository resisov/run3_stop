# Main-agent handoff: approved Low-dM double-ratio domain

The 250-GeV starting point was approved at 2026-09-07T22:29:50Z and adopted
for both 2024 and 2025. The final Low-dM template remains **GNN output**.
No event/object selection, GNN score boundary, CR parent, RZ, TF, Sgamma,
nominal Z prediction, or High-dM physics value was changed by this adoption.

**Downstream blocker:** these products are tied to the existing 5-GeV-veto
histogram hashes, not the newly requested 10-GeV lepton veto. The main agent
confirmed both years' frozen `production.argv` lack a veto override, the
builder defaults and GNN object masks use 5 GeV, and the required SF list
contains the 5–10 GeV corrections. Do not use these products as 10-GeV-veto
results or simply remove SF variations and claim the selection is updated.
Card/limit submission is on hold in the main task while it checks for valid
10-GeV canonical inputs. No nominal/SF reprocessing was done here.

## Canonical products

Use `YEAR=2024` or `2025` under the following common root:

```text
Local:
/Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/reports/background_estimation_histonly_20260907

EOS:
/eos/user/t/taiwoo/run3_stop/decaf/autonomous_allhad/reports/background_estimation_histonly_20260907
```

| Product relative to the root | Role |
|---|---|
| `YEAR/zgamma/zgamma_double_ratio.json` | Canonical High/Low-dM double ratio; Low-dM now starts at 250 GeV |
| `YEAR/lowdm_gnn_double_ratio.json` | New central-nonclosure response basis in the frozen GNN bins |
| `YEAR/lowdm_gnn_backgrounds.json` | Unchanged GNN TFs, SR/CR mapping, weight variations, and RZ×Sgamma Z prediction |
| `YEAR/tf_inputs.json` | Unchanged full TF input including process-separated GNN×UT joint histograms |
| `YEAR/rz_high.json`, `YEAR/rz_low.json`, `YEAR/rz_covariance.json` | Unchanged RZ and covariance |
| `YEAR/sgamma/sgamma_ut.json` | Unchanged Nb-only Sgamma factors |
| `YEAR/tf/gnn/` | GNN transfer-factor plots |
| `YEAR/zgamma/` | Current double-ratio plots |

## Response and parent-mapping contract

- Measurement groups remain Nb1 and Nb2plus. The existing final GNN layout is
  six SR categories, each with five score bins (30 bins in total); it is not
  replaced with an UT template or a new two-category SR layout.
- For each Nb group, SR NISR0 maps to CR NISR0; SR NISR1 and NISR2plus both map
  to CR NISR1plus. The four existing CR parents are unchanged. Read physical
  score edges from the product, not from equal array indices: SR and CR edges
  differ.
- Top is TT+ST. Each GNN TF coefficient divides a target-process SR score-bin
  yield by its CR parent's target-process **integral**. Retain CR score
  fractions and shared-denominator covariance; these coefficients are not
  independent same-bin CR ratios.
- Nominal Z prediction is `RZ(Nb) × sum_UT H_Z(GNN,UT) × Sgamma(Nb,UT)`.
  Q is not propagated. Do not renormalize Z to undo the Sgamma-weighted sum.
- The uncertainty source bins are `[250,300,350,400,500,1500]` GeV; the last
  bin is open ended with overflow already folded upstream. Each source-bin
  response contains five GNN-bin yields per SR category, not five UT yields.
- Use `downstream_central_abs_deviation = abs(D-1)` only. The plotted
  `systematic=max(abs(D-1),stat)` is a separate reporting band.
- Existing multipliers are `up=1+delta` and `down=1/(1+delta)` on the selected
  UT component. The product provides the resulting GNN `up` and `down`
  templates and propagated MC sumw2. Its `nominal` is byte-equivalent in values
  to the unchanged stored RZ×Sgamma projection.
- There are five UT-source responses for each of six SR categories. This is
  a response basis, **not** an instruction to create 30 independent nuisance
  parameters. Preserve the existing card's nuisance naming, correlations,
  and shared parent-normalization scheme. Card integration belongs to the
  main agent; this task did not edit cards or run limits/impacts.

The user intends to retire `veto_electron_5to10` and `loose_muon_5to10` ID SF
variations after the lepton veto moves to 10 GeV. They remain in these old-input
products for exact provenance; the main agent's confirmed selection mismatch
must be resolved before producing a 10-GeV-veto card. This factor-adoption
task did not rewrite histograms, alter nominal weights, or start a new
measurement.

## Validation and recovery

`campaign_state.json` records unchanged hashes for 228 protected products,
unchanged High-dM and nominal GNN physics, agreement with the approved
250-start proposal, and all source/output checksums. The four regression
modules pass 18 tests, including central-only response propagation and
rejection of unapproved or inconsistent inputs.

Exact old 300-start files are retained in `YEAR/zgamma_300_previous/`.
`low250_adoption_baseline.json` gives recovery hashes and protected-product
hashes; `before_low250/` retains the former global records. The original
`YEAR/zgamma_250_proposal/` is historical evidence, not the canonical input.

New 250–300 source-bin responses reach 0.4754% (2024) / 1.3756% (2025) of a
GNN-bin nominal yield. Changed common >=300 responses reach 0.1909% / 0.5817%.
These are individual template-response changes, not total uncertainties,
nominal-yield changes, or limit changes.

The 2025 retained set is complete but upstream skipped files include one data
file; complete luminosity coverage is not claimed. Existing normalization and
zero-QCD GNN bins remain untouched. No ROOT/NanoAOD read or web publication
was performed. `sync_receipt.json` records the final synchronization and Git
commit receipt; private approval correspondence is excluded from Git/EOS sync.
