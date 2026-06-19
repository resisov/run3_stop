# Partial Merge Preview 878535

Timestamp UTC: `2026-06-19T12:00:31Z`

**PRELIMINARY ONLY.** This preview is based only on currently completed final shard JSONs.
Process composition may be biased while Condor jobs are still running or idle.
Final results require `complete + complete_with_bad_files == 2110`.
Stale `docs/data/full_normalization_factors.json` must not be trusted for this campaign.

## Shard Status

| metric | count |
|---|---:|
| `expected_shards` | 2110 |
| `final_json_files` | 1240 |
| `complete` | 1239 |
| `complete_with_bad_files` | 1 |
| `failed` | 0 |
| `missing` | 141 |
| `running_checkpoints` | 729 |
| `zero_byte` | 0 |
| `unreadable` | 0 |
| `other_nonterminal` | 0 |
| `completed_for_preview` | 1240 |

## Normalization

- Status: `partial_preview_complete`
- Formula: `DATA factor=1.0; MC normalization_factor = xsec_pb * lumi_pb / sumw, where sumw is retained from currently completed shard payloads only.`
- Luminosity: `109.82` fb^-1 = `109820.0` pb^-1
- Blocked datasets: `0`
- Files attempted in completed shards: `30998`
- Files processed in completed shards: `30997`
- Bad files recorded in completed shards: `1`

## Region Totals

| region | data | background |
|---|---:|---:|
| `cat2_LLCR_highDeltaM` | 1159 | 174705 |
| `cat3_QCDCR_highDeltaM` | 1380 | 2.61504e+06 |
| `cat4_GCR_highDeltaM` | 209 | 420784 |
| `cat5_DY2E_highDeltaM` | 13 | 33028.1 |
| `cat6_DY2M_highDeltaM` | 27 | 50983.4 |
| `cat7_SR_highDeltaM` | 1119 | 290346 |

No official outputs, docs/data full normalization files, hists.npy, datacards, or Combine inputs were written.
