# Photon fake template-fit campaign checkpoint

Updated: 2026-08-10 (Europe/Zurich)

## Active HTCondor campaigns

- Main: cluster `989023` on `bigbird24.cern.ch`
  - EOS: `/eos/user/t/taiwoo/run3_stop/decaf/autonomous_allhad/workflow/photon_fake_template_run2method_20260810`
  - Expected jobs: 5,795
  - Priority order: QCD 120, GJ 110, EGamma 100, DY 80, TT 70, WtoLNu 60, ST 50, VV 40, Zto2Nu 30
- Rare TTVV/VVV supplement: cluster `989028` on `bigbird24.cern.ch`
  - EOS: `/eos/user/t/taiwoo/run3_stop/decaf/autonomous_allhad/workflow/photon_fake_template_run2method_20260810_rare`
  - Expected jobs: 60
  - Priority: 90

Do not infer success from queue disappearance. Run `audit_photon_fake_template_campaign_2024.py` after both queues drain and require every output to be valid.

## Last observed state

- Observed 2026-08-11 02:00 CEST: both recovery/production queues drained. `egamma_00788` passed all recovery checks after reading 2/2 files and 2,427,921 events, selecting 14 compact events, with complete output/metadata, matching checksum, and no bad files. DY2x passed its full-checksum audit at 669/669 valid, zero missing, zero invalid, 1,560,462,096 events read, and 10 selected compact events. The final main allowed-non-DY checksum re-audit is still running; proceed to measurement only after it records 5,456/5,456 valid.
- Observed 2026-08-11 01:29 CEST: replacement DY2x reached 667/669 matching output/metadata pairs with only two production jobs still running and no holds. The EGamma `egamma_00788` recovery job also started running. Wait for both queues to drain, then validate the EGamma retry and run the complete 669-shard DY2x checksum audit.
- Observed 2026-08-11 01:14 CEST: replacement DY2x reached 425/669 matching output/metadata pairs (63.5%), with 60 running and 185 idle production jobs and no holds. The EGamma `egamma_00788` recovery job remains idle without output or metadata.
- Observed 2026-08-10 23:11 CEST: the main allowed-non-DY full-checksum audit finished with 5,455/5,456 valid outputs, zero invalid outputs, and exactly one missing shard: `egamma_00788`. The audit read 12,101,112,501 events and retained 440,079 selected compact events. A single-shard recovery retry for `egamma_00788` was submitted as cluster `989208`; success still requires output, metadata, checksum, complete file coverage, and an empty bad-file list.
- Replacement DY2x cluster `989204` had 2 running and 651 idle jobs with 16/669 matching output/metadata pairs. No holds were observed.
- Observed 2026-08-10 22:56 CEST: main cluster `989023` drained. Raw allowed-process output counts reached EGamma 2,165/2,166, GJ 308/308, QCD 592/592, TT 620/620, WtoLNu 969/969, ST 232/232, VV 61/61, and Zto2Nu 508/508. A full-checksum audit excluding the forbidden legacy DY process is running; do not mark the main campaign complete until it finishes. Replacement DY2x cluster `989204` had 3 running and 658 idle jobs, with 8/669 matching output/metadata pairs including the pilots.
- The rare supplement passed its full-checksum audit: 60/60 valid outputs, zero missing, zero invalid, 75,003,040 events read, and 3,792 selected events.
- Observed 2026-08-10 22:43 CEST: all three correct-DY pilots completed and passed every gate: output and metadata status complete, checksum match, full file coverage, and empty bad-file lists. They read 2,656,248 DY2E, 3,070,039 DY2M, and 2,489,414 DY2Tau events respectively. The remaining 666 non-pilot shards were submitted to `bigbird24.cern.ch` as cluster `989204`; the three valid pilots were not resubmitted.
- At the same observation, the main campaign had only five running jobs left. Accepted output counts were TT 620/620, WtoLNu 968/969, ST 232/232, VV 61/61, and Zto2Nu 505/508. Queue drainage alone is not completion; audit the remaining WtoLNu/Zto2Nu shards and retain the known EGamma incomplete-data exception for explicit recovery.
- Observed 2026-08-10 22:26 CEST: WtoLNu reached 929/969 matching output/metadata pairs, ST 130/232, and VV 10/61. Main queue occupancy increased to 279 running with 441 idle and no holds. Correct-DY pilot cluster `989183` still has three idle jobs and zero output/metadata files.
- Observed 2026-08-10 22:11 CEST: TT reached 620/620 output and metadata files. WtoLNu started and had already reached 276/969 matching output/metadata pairs; the main queue had 69 running and 1,434 idle jobs with no holds. The three correct-DY pilots remain idle without outputs or holds.
- Observed 2026-08-10 20:10 CEST: the three correct-DY pilot jobs in cluster `989183` remain idle with no holds and have not produced outputs or metadata. Main cluster `989023` has 2,188 idle and 25 running jobs; TT advanced to 177 accepted output files, while EGamma/GJ/QCD remain at 2,165/308/592. No active legacy-DY jobs remain.
- Correction recorded 2026-08-10 20:05 CEST: the 339 DY shards in main cluster `989023` were built from obsolete PTLL-binned DY inputs and are invalid for this analysis. Active jobs in the exact DY ProcId range `3066--3404` were removed without touching the other process groups. Any PTLL DY outputs already written by this campaign are explicitly excluded from all measurement, normalization, audit-completion, and plotting inputs.
- The replacement DY campaign is `/eos/user/t/taiwoo/run3_stop/decaf/autonomous_allhad/workflow/photon_fake_template_run2method_20260810_dy2x`. It uses only `DYto2E-4Jets`, `DYto2Mu-4Jets`, and `DYto2Tau-4Jets` from `KNU_2024_dyto2x4jets.json.gz`, normalized with the complete `workflow/plot2024/norm.json`. The prepared manifest contains 9,367 files in 669 shards (2,821 DY2E, 3,194 DY2M, and 3,352 DY2Tau files) and zero PTLL records.
- Pilot cluster `989183` on `bigbird24.cern.ch` contains one representative shard per DY flavor (`dy_00000`, `dy_00217`, and `dy_00446`). Require successful exit, accepted output, metadata, checksum, complete file coverage, and no bad files for all three before submitting the remaining replacement DY shards.
- Observed 2026-08-10 19:51 CEST: the main queue had 2,312 idle and 150 running jobs, with no holds. Accepted output/metadata pairs were EGamma 2,165/2,166, GJ 308/308, QCD 592/592, DY 23/339, and TT 16 produced so far; the rare campaign remains complete at 60/60. The uncovered set is 228 DY shards plus `egamma_00788`; 88 additional DY and 62 TT shards are running.
- Superseded diagnosis: the apparent PTLL-100/PTLL-200 success difference is irrelevant because neither PTLL sample is an allowed DY input. Do not preserve or retry either one for the photon-fake measurement.
- Observed 2026-08-10 19:21 CEST: the uncovered set expanded to 22 shards: the known `egamma_00788` plus 21 DY shards. DY still had 0 accepted outputs. Stop treating this as isolated-file corruption; the common correction/runtime failure is campaign-wide for the initial DY materialization and must be fixed before DY retry.
- Observed 2026-08-10 19:06 CEST: EGamma reached 2,165/2,166 outputs; the remaining shard is the known failed `egamma_00788`. The rare supplement completed 60/60 outputs with matching metadata (TT 25, VV 35), and its queue drained.
- Seven initial DY shards (`dy_00000`, `00001`, `00004`, `00006`, `00011`, `00014`, and `00016`) exited 1 with zero events and no accepted output. Their logs show repeated overflow in `object_corrections_2024.py::_crystal_ball_invcdf` followed by invalid Awkward multiplication; this is a correction/runtime failure, not evidence of ROOT corruption. They are uncovered and must not be treated as bad files or silently skipped. Diagnose and patch the numerical guard before retrying the affected DY shards.
- Observed 2026-08-10 15:35 CEST: EGamma reached 900/2,166 outputs with matching metadata; 3,969 main jobs were idle and 25 were running, while all 60 rare jobs remained idle. No jobs were held.
- `egamma_00788` (`989023.788`) exited 1 with `complete_with_bad_files` and has neither an accepted output nor metadata. The expected-output versus output-plus-queue audit identifies it as the only currently uncovered main shard. Both source files were unavailable through the global redirector during diagnosis; CERN EOS has no replica and the INFN redirector reached its redirect limit. Treat this as transient until a later alternate-endpoint retry proves reproducible; do not silently skip data or claim complete luminosity coverage.
- Observed 2026-08-10 after local recovery: main queue 4 running and 4,251 idle; rare queue 60 idle.
- Full-checksum partial audit: 1,539/5,795 main outputs valid, 0 invalid, 4,256 not yet produced.
- GJ is now 308/308 valid. `gj_00289` was recovered locally after permanently excluding the one reproducibly unavailable input ROOT file already recorded in `bad_files.json`.
- The recovered `gj_00289` shard contains 15/15 valid files, reads 2,018,552 events, selects 5 compact events, and has zero bad-file records. Its source digest is `0b7264a977a96e0dc8b3e2046034b5e8109f8fa9e3e1bf5972825659966500ea`; its output SHA-256 is `e33aa4276fa67dbd4ba41392219479162bade65946d000c35873e9186ed94645`.
- GJ normalization already uses 2,750 processed files out of 2,751 attempted files and sums `Runs.genEventSumw` over those same 2,750 retained files, so no normalization rewrite was required.
- Observed 2026-08-10 13:14 CEST: QCD 592/592, GJ 306/308, EGamma 487/2166.
- `gj_00289` (`989023.2455`) exited 1 with `complete_with_bad_files`; its incomplete result was rejected and it remains a retry/diagnostic target.
- One other GJ job was running; 4,408 main jobs and all 60 rare jobs were idle.
- No held jobs were observed and output/metadata counts matched for published shards.
- Corrected payload SHA-256: `9eb0b58a4d2f7ce006a0d2caa7ef6a654eb99c64472bbd8afcf415b61bb3ff4a`.
- Current `btageff2024.merged` SHA-256: `03524e9ae28110814f336eafc887e60d54b495a7b8dec7cda59bd792f56feaf4`.
- Full smoke: 9/9 files, 2,549,578 events, 11 compact events, zero bad files, 195.73 s.
- Rare normalization audit: 7 datasets and 937/937 files complete.

## Final audited production and measurement

- Final allowed-main audit completed with 5,456/5,456 valid non-DY shards, zero missing, and zero invalid. It reads 12,103,540,422 events and retains 440,093 compact events. The recovered `egamma_00788` is included and valid.
- Correct replacement DY2x completed with 669/669 valid shards, zero missing, and zero invalid. It contains only `DYto2E-4Jets`, `DYto2Mu-4Jets`, and `DYto2Tau-4Jets`, reads 1,560,462,096 events, and retains 10 compact events. Every legacy PTLL-binned DY artifact remains excluded from the audit, normalization, measurement, and plots.
- The rare supplement completed with 60/60 valid shards, zero missing, and zero invalid. It reads 75,003,040 events and retains 3,792 compact events. Its normalization audit covers all 937/937 input files.
- The final Python-3.8 measurement processed 443,895 deduplicated compact records using the allowed main non-DY outputs, all correct DY2x outputs, and the rare supplement. Nominal intermediates were not modified.
- The simulation validation closure fails decisively: the loose-photon prediction is 1,033.38 versus 18.50 truth, or prediction/truth = 55.86. The independent data validation prediction/direct-fit ratio is 2.067.
- Replacing only the nominal GCR fake component changes the integral Data/prediction ratio from 1.416 to 1.394, but the predeclared shape gate does not improve. The replacement therefore does not explain the nominal GCR discrepancy.
- The automatic decision is **do not adopt: closure or Data/MC gate failed**. All generated plots were visually inspected; their extreme factors, sparse EE/high-pT behavior, and closure failure are physical/statistical pathologies rather than missing or malformed rendering.
- Final local report: `autonomous_allhad/reports/photon_fake_template_final_correctdy_20260811/index.html`.
- Final EOS measurement: `autonomous_allhad/workflow/photon_fake_template_final_correctdy_20260811/measurement.json`.
- Final EOS report: `autonomous_allhad/reports/photon_fake_template_final_correctdy_20260811/`. It was not published to the web.

## Full-production same-region follow-up

- The follow-up removes the low-$\Delta\phi$ to high-$\Delta\phi$ transfer and measures one integrated fake normalization per EB/EE and coarse photon-$p_T$ group in the same topology. The $U_T$ shape still comes from the disjoint charged-isolation-fail residual.
- The complete 443,895-event sample was rerun with the same allowed main non-DY, correct DY2E/Mu/Tau, and rare inputs. All three checksum audits are embedded in the machine-readable result and the production-complete gate passes.
- Independent two-fold MC prediction/truth is 0.926 in the $0.30\leq\min\Delta\phi<0.50$ validation region and 1.746 in the nominal GCR. The corresponding coarse-$U_T$ ratios are 1.258/0.187/0.071 and 1.963/0.987/1.536.
- A non-independent full-MC pass/fail structural diagnostic gives integrals 1.225 and 0.700, with coarse-$U_T$ ratios 1.612/0.378/0.060 and 0.773/0.422/1.206. Removing the fold split therefore does not repair the recoil-shape failure.
- Within this evaluator, the integral data/prediction ratio improves from 1.126 to 1.092 in the validation region and from 1.448 to 1.374 in the nominal GCR. This modest prefit improvement is not sufficient because the independent integral and shape gates fail.
- Decision: **do not adopt**. The nominal fake/QCD component remains unchanged. The next defensible candidate, if pursued, is a regularized simultaneous shower-shape fit in a few coarse $U_T$ intervals with its binning and regularization frozen in the validation region.
- Final local report: `autonomous_allhad/reports/photon_fake_same_region_full_correctdy_20260811/index.html`.
- Final EOS report: `autonomous_allhad/reports/photon_fake_same_region_full_correctdy_20260811/`. It was not published to the web.

## Partial end-to-end workflow test

- A same-region shower-shape follow-up was implemented and run on the frozen 393,225-event snapshot. It removes the low-$\Delta\phi$ to high-$\Delta\phi$ factor transfer: one integrated fake normalization is extracted from a $\sigma_{i\eta i\eta}$ fit in each region and EB/EE $\times$ $p_T^\gamma$ group (220--400 and at least 400 GeV), while the $U_T$ shape is taken from the same-region tight-shape charged-isolation-fail residual. The MC test uses deterministic two-fold event-level cross-fitting.
- In `GCR_DPhiVR_High`, the MC cross-fit integral is 14.52 predicted versus 13.39 truth, or 1.084, with an approximate 0.16-sigma integral pull. This apparent integral closure is accidental: the coarse-$U_T$ prediction/truth ratios are 1.69, 0.17, and 0.09 for 250--400, 400--650, and 650--1500 GeV. The truth target has only 6.10 effective events.
- In the nominal GCR, the cross-fit gives 611.83 predicted versus 314.45 truth, or 1.946, with an approximate 2.24-sigma integral pull. The coarse-$U_T$ ratios are 2.19, 1.11, and 1.69; the truth target has 8.47 effective events.
- All eight MC cross-fit normalization components satisfy the current minimum template checks. In partial data, all four nominal-GCR groups are mechanically usable, but the high-$\Delta\phi$ validation EB $p_T^\gamma\geq400$ group has a zero tight-shape sideband fraction and is rejected.
- The partial-data GCR comparison is not interpretable: EGamma is incomplete while GJ/QCD are complete and full-luminosity normalized. Data/MC changes only from 0.464 to 0.470 after the replacement because the full prompt MC dominates. This is not evidence against or for the data estimate.
- Conclusion: the same-region design is far better behaved than the original transfer (46.7 MC nonclosure) but still fails the predeclared $U_T$-shape requirement. It is not adopted. The next full-production study must either condition the isolation-sideband shape on an additional recoil/composition proxy or use a regularized simultaneous $\sigma_{i\eta i\eta}$ fit across coarse $U_T$ bins; it must not fit every $U_T$ normalization freely.
- Same-region report: `autonomous_allhad/reports/photon_fake_same_region_shape_20260810T131652Z/index.html`.
- The truth-transfer closure was rerun on the identical frozen 393,225-event snapshot after broadening the truth-fake sideband from charged-isolation level exactly 1 to all photons failing the medium charged-isolation requirement (levels 0+1). This isolates the effect of the sideband definition; no input, normalization, event selection, or nominal intermediate changed.
- The central prediction/truth ratio improved from 50,936 to 4.23 for the current mapping, from 11.68 to 4.28 for EB/EE-inclusive factors, and from 21.94 to 6.82 for the global factor. The prediction remains an overestimate in every coarse $U_T$ interval: 5.23, 2.81, and 1.50 for 250--400, 400--650, and 650--1500 GeV with the current mapping.
- The broader sideband supplies 889 global raw entries instead of 20, but only 11.60 effective weighted events. The corresponding tight numerator still has 73 raw entries and only 1.39 effective events. No factor passes the predeclared requirement of at least 10 effective events on both sides; the strict-stability mapping consequently has zero coverage.
- Conclusion: broadening the sideband removes the pathological five-order-of-magnitude central result but does not produce a statistically valid or closing transfer factor. The remaining low-$U_T$ excess and strongly $p_T^\gamma$/EB--EE-dependent factors indicate a transferability/composition problem in addition to limited MC statistics. This variant must not replace the nominal GCR fake component.
- Fail-medium report: `autonomous_allhad/reports/photon_fake_truth_transfer_failmedium_20260810T124801Z/index.html`.
- A template-fit-independent truth-transfer diagnostic was completed using the frozen GJ/QCD snapshot. The low-$\Delta\phi$ truth tight/loose factor was applied to high-$\Delta\phi$ truth-fake loose photons.
- Central prediction/truth ratios are 50,936 with the current bin mapping, 11.68 with $p_T^\gamma$-inclusive factors split by EB/EE, and 21.94 with one global inclusive factor. No factor satisfies the predeclared strict requirement of at least 10 effective tight and loose events.
- The low-$\Delta\phi$ global truth factor uses 73 tight and 20 loose raw entries but only 1.39 and 1.96 effective weighted events. The high-$\Delta\phi$ truth target has only 6.10 effective events. Therefore the current charged-isolation transfer cannot be validated or used as a stable estimator; its huge central nonclosure is accompanied by comparably huge statistical uncertainty.
- Detailed report: `autonomous_allhad/reports/photon_fake_truth_transfer_closure_20260810T121110Z/index.html`.
- A second frozen-snapshot follow-up was run at `20260810T121110Z` after the GJ recovery. It contains 1,544 valid outputs: EGamma 644, GJ 308, and QCD 592; all other main processes and all 60 rare jobs are absent.
- The follow-up measurement completed on 393,225 compact events in 1m38s, and plotting/HTML generation completed successfully. The local report is `autonomous_allhad/reports/photon_fake_template_partial_followup_20260810T121110Z/index.html`.
- This snapshot is not physically interpretable: EGamma is only 644/2,166 while GJ and QCD are complete and normalized to the full luminosity. Prompt/electron subtraction therefore produces negative residual fake predictions in the data validation region and GCR.
- MC closure is also poor in the partial snapshot: prediction/truth = 46.66. The nominal GCR Data/MC ratio is 1.416, while the partial template replacement gives 1.743 and worsens every predeclared shape metric. No adoption decision is allowed before complete production.
- Snapshot report: `autonomous_allhad/reports/photon_fake_template_partial_test_20260810T1055Z/index.html`.
- Partial audit: 1,383/5,795 valid main outputs, 0 invalid; rare 0/60. The audit correctly reported both campaigns as incomplete.
- Measurement completed on 387,558 deduplicated compact events in 2m02s.
- Plot and HTML generation completed. The report decision is intentionally `incomplete production: adoption decision deferred`.
- The partial physics result is not usable because only about 22% of EGamma was present while QCD/GJ were nearly complete and MC retained full-luminosity normalization. This caused over-subtraction and negative data residual predictions.
- The partial MC closure also failed strongly; this must be reevaluated with the complete production and may independently reject the method.
- Plotting was hardened to skip invalid NaN factors, use non-overlapping CMS labels on square figures, and defer adoption whenever production audits are incomplete.
- The likelihood saturated-term calculation was changed to a positive-bin mask to avoid harmless `log(0)` runtime warnings.

## Rejected submissions

- `988984`: removed because it used the superseded May b-tag efficiency payload.
- `988987`: removed because the runtime archive omitted current `analysis/data` correction inputs.
- Their outputs/logs were deleted; no mixed-payload output is retained.

## Resume sequence

1. Query clusters `989023` and `989028` on `bigbird24`.
2. Compare expected shard names to output/metadata names and retry only missing/invalid shards.
3. Run full checksum audits for main and rare campaigns.
4. Run `measure_photon_fake_template_2024.py` using both output directories and `normalization_with_rare.json`.
5. Run `plot_photon_fake_template_2024.py` with both production audits, the rare normalization audit, and the reference nominal GCR evaluation.
6. Visually inspect every plot, especially independent-data and MC closure in `U_T`.
7. Adopt only if the predeclared closure and prefit GCR shape gates pass; otherwise report rejection.

Nominal intermediate ROOT files must remain untouched. Event/object selection comes only from `real_subset_worker.py`; do not use legacy `analysis/stop_processor_v4.py` or `ids.py` for this measurement.
