# High-dM canonical code

2026-09-07 · 현재 구성: **High-dM 79−6 = 73 bins + Low-dM GNN 30 bins**, 2024+2025.

## 현재 진입점

| 단계 | 유지할 코드 |
|---|---|
| 오브젝트·event selection·weights | [real_subset_worker.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py>), [analysis_scale_factors.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/analysis_scale_factors.py>), [corrections.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py>) |
| 중간 ROOT·TROTA | [flat_ntuple_worker.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/flat_ntuple_worker.py>), [trota_resolved_2024.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/trota_resolved_2024.py>), [trota_resolved_2024_inplace.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/trota_resolved_2024_inplace.py>) |
| High-dM 카테고리·binning | [highdm_resolved_categories.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/highdm_resolved_categories.py>), [search_bin_categorization.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/search_bin_categorization.py>), [search_bins_2024.json](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/configs/search_bins_2024.json>), [search_bins_2025.json](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/configs/search_bins_2025.json>) |
| 히스토그램 생성·병합 | [run_flat_hists_chunked.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/run_flat_hists_chunked.py>), [build_flat_boosted_recoil_hists.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py>), [merge_flat_hist_chunks_streaming.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/merge_flat_hist_chunks_streaming.py>) |
| 외부 shape·입력 검증 | [apply_flat_external_shape_systematics.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/apply_flat_external_shape_systematics.py>), [validate_flat_ntuple_outputs.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/validate_flat_ntuple_outputs.py>), [validate_histogram_root_inputs.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/validate_histogram_root_inputs.py>) |
| 배경추정 histogram 경계 | [build_flat_boosted_recoil_hists.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py>), [merge_flat_hist_chunks_streaming.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/merge_flat_hist_chunks_streaming.py>) |
| TF, Sgamma, Z/gamma, RZ 측정 | [build_histogram_tf_inputs_2024.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_histogram_tf_inputs_2024.py>), [build_sgamma_ut_report_2024.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_sgamma_ut_report_2024.py>), [build_zgamma_double_ratio_2024.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_zgamma_double_ratio_2024.py>), [dy_estimation/](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/dy_estimation/__main__.py>) |
| 카드·templates | [build_combine_inputs.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py>), [background_process_groups.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/background_process_groups.py>) |
| 연도 결합·limit 제출 준비 | [build_combined_year_datacards.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combined_year_datacards.py>) |
| Condor 환경·impacts | [build_standard_condor_runtime.sh](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_standard_condor_runtime.sh>), [validate_standard_condor_runtime.sh](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/validate_standard_condor_runtime.sh>), [run_asimov_impacts_eos.sh](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/run_asimov_impacts_eos.sh>) |
| CR-only fit·covariance·pulls | [run_cronly_fit_eos.sh](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/run_cronly_fit_eos.sh>), [finalize_cronly_fit_status.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/finalize_cronly_fit_status.py>), [extract_cronly_fit_covariance.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/extract_cronly_fit_covariance.py>), [plot_cronly_nuisance_pulls.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_cronly_nuisance_pulls.py>) |
| 분포·limit 그림 | [plot_control_search_bins_style.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py>), [plot_highdm73_from_limit_templates.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_highdm73_from_limit_templates.py>), [postprocess_limits.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py>) |

연도마다 카드빌더를 복제하지 않는다. 공용 `build_combine_inputs.py`의
`--campaign-year`, `--search-bin-config`를 사용한다.
현재 High-dM-only 카드에는 `--highdm-only --drop-highdm-leading-bins 6`를 사용한다.
공용 빌더의 기존 Low-dM34 경로는 GNN30이 아니다. GNN 결합 경로는
[merge_datacard.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/merge_datacard.py>)와 그 내부 구현이며,
[Low-dM 의존성 보고서](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/reports/lowdm_canonical_dependencies.md>)를 함께 따른다.
이 목록은 코드 분류이며, 새 생산 결과의 검증 완료를 뜻하지 않는다.

## 삭제 금지 범위

- `autonomous_allhad/gnn_lowdm/` 전체와 Low-dM 보고서의 공용 의존성.
- 두 연도 설정, SF payload, compiled corrections, b-tag efficiency, 모델, xsec.
- `sidecar_store.py`, `sidecars_main.sqlite`, normalization JSON 및 입력 metadata.
  이들은 폐기 대상인 “side workflow”가 아니다.
- 검증된 ROOT·히스토그램·datacard·limit·impact·plot 및 campaign 기록.
- 별도 측정·upstream 생산·복구 코드와 연구 보존본. 파일명에 연도나 옛 bin 수가
  있다는 이유만으로 삭제하지 않는다.

`build_nb_recoil_transfer_inputs_2024.py`와 DY feature/sparse 재독기들은
2026-09-07에 폐기했다. 새 측정은 nominal histogram merge가 자동으로 쓰는
`*_background_estimation.json`만 읽는다. `build_an_zinv_factors_2024.py`의
검증된 수식·플로팅 helper와 histogram-only 측정·publication 유틸리티는 보존했다.
`build_lowdm_gnn30_combine_inputs.py`는 현재 6×5 GNN 경로가 아닌 구형 10×3 방식이며,
Low-dM 담당 범위이므로 이번에는 삭제하지 않았다.

## 제거한 구형 side workflow

복구 백업을 먼저 검증한 뒤 아래 **16개 파일, 6,026줄**을 제거했다.
AN17·recoil-only·High-dM54/60·Low-dM42 연결 계열,
구형 free-background 모델, 그 진단·일회성 실행 스크립트다.

- `build_boosted_an17_combine_inputs.py`
- `build_flat_boosted_an17_combine_inputs.py`
- `build_flat_recoil_sr_combine_inputs.py`
- `build_flat_recoil_ntop_split_combine_inputs.py`
- `build_flat_recoil_ntop_split_lowdm_combine_inputs.py`
- `build_flat_selected_recoil6_lowdm_combine_inputs.py`
- `build_highdm54_lowdm42_combine_inputs.py`
- `build_lowdm42_combine_inputs.py`
- `build_nb_recoil_tf_combine_inputs_2024.py`
- `build_free_background_combine_inputs_2024.py`
- `build_signal_tail_bin_merge_diagnostic.py`
- `run_signal_tail_bin_merge_diagnostics.sh`
- `build_highdm_signal_shape_regularization_diagnostic.py`
- `compare_recoil_sr_limits.py`
- `watch_flat_ntuple_pipeline.py`
- `submit_2024_2025_limits_impacts.sh`

정적 검사한 나머지 소스·설정 426개에서 이 삭제 계열로의 호출 참조는 없었다.
과거 SF 검증 summary에 남은 `watch_flat_ntuple_pipeline.py` 이름은
실행 설정이 아닌 당시 검증 기록이므로 그대로 보존했다.
로컬 실행 프로세스와 Low-dM 보호 목록에도 삭제 대상이 없었다.

백업: [highdm_sidecars_20260907.tar.gz](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/reports/cleanup_backups/highdm_sidecars_20260907.tar.gz>)  
SHA256: `ca1e5b6ef3bd67a3a0b614accab2469fb842a0fc72439a29a6fe7918d83bd8bf`  
개별 파일 checksum: [삭제 목록](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/reports/highdm_cleanup_20260907.json>).

백업은 Git에서 제외되는 로컬 보관본이다. 원본 경로별 내용 16개를 모두 검증했으며,
필요하면 별도 복구 디렉토리에 풀 수 있다. 생산 결과는 변경하지 않았다.

삭제 전후 나머지 소스·설정 426개의 내용 checksum, Low-dM 보호 파일 상태,
Low-dM 의존성 보고서가 모두 동일하다. 이 보고서의 파일 링크 34개와 Python 소스
25개의 문법도 확인했다. 생산·fit·limit를 재실행한 검증은 아니다.

## EOS 반영

같은 정리를 `/eos/user/t/taiwoo/run3_stop/decaf`에도 적용했다.
대상 16개 중 원격에 남아 있던 7개를 백업 후 제거했고, 9개는 이미 없었다.
삭제 전후 확인한 원격 공용 소스 129개의 checksum은 동일하다.
캐노니컬 분석 코드·GNN·보정·생산 산출물의 덮어쓰기나 재실행은 하지 않았다.

EOS 복구 백업(삭제 코드 7개 + 갱신 전 README 2개):
`/eos/home-t/taiwoo/run3_stop/decaf/autonomous_allhad/reports/cleanup_backups/highdm_sidecars_eos_20260907.tar.gz`  
SHA256: `68f91d1cde1aa98504a5e8346091024e8dbec80d30e59cbb77871f0dbd6efee9`.

문서의 `/Users/taiwoomac/Documents/All Hadronic Stop Analysis` 링크는 로컬
작업본 기준이다. EOS에서 소스를 찾을 때는 위 EOS 루트를 사용한다.
