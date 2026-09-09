# Workflow 코드·산출물 사용 구분

2026-09-09 EOS 실물 조사. 기준 경로: `/eos/user/t/taiwoo/run3_stop/decaf/autonomous_allhad/workflow`.

판정 대상은 현재 **2024+2025, High-dM 79−6=73 + frozen Low-dM GNN 6×5=30, lepton veto >10 GeV, DY 71<mll<111 GeV** 분석이다. 새 Top/W SF는 아직 production에 적용되지 않았다.

최상위 코드/백업 **67개 = 공용·현 분석 유지 37개 + VR 검증 도구 4개 + 현 분석 미사용 26개**. `results/`의 **22개 디렉토리는 모두 현재 계산에서 미사용**이다. 미사용 판정은 삭제 실행을 뜻하지 않는다. 이번 조사에서는 산출물을 삭제하지 않았다.

## 현재 사용하는 위치

| 위치 | 실제 용도 | 판정 |
|---|---|---|
| `flat2024_v8/outputs/nominal`, `flat2025_v8/outputs/nominal` | Events/TROTA가 들어 있는 현재 intermediate ROOT; 현재 truth 보강의 in-place 대상 | 사용 |
| `histograms/lepton_veto10_20260908` | 두 연도 main/GNN histogram, 현재 카드·결합 limit·그림 | 사용; 약 24 GiB |
| `plot2024`, `plot2025`의 histogram aliases | 위 veto10 campaign으로 연결된 6개 symlink 확인 | 사용 |
| `topw_truth_20260909` | 실행 중인 truth 보강 작업의 argv/queue/log/상태. 실제 worker는 `autonomous_allhad/flat_ntuple_worker.py` | 사용; 조사 시 약 6.1 MiB |
| `topwtageff_20260908_workday` | 완료된 Top/W MC efficiency 생산의 shard/로그/snapshot | 현재 계산은 직접 읽지 않음. 최종 merged는 아래 정식 위치에 설치 완료 |
| `../reports/background_estimation_lepton_veto10_20260908` | 현재 histogram-only TF/RZ/Sgamma/double-ratio 산출물 | 사용 |
| `../../analysis/hists/topwtageff2024.merged`, `topwtageff2025.merged` | 현재 SF 유도에 사용하는 MC efficiency | 사용; 각각 2,245,877 bytes |
| `histograms/dy_window20_20260907` | veto10 이전 DY-window20 campaign | 미사용; 약 42 GiB (화면의 44.4 GB) |
| `studies/object_definition_veto_pt_20260905` | 완료된 lepton pT 5↔10 비교 연구 | 사용자 삭제 완료. 전체 복원하지 않음 |

`topwtageff_20260908_workday`는 2024 샘플을 사용한 2,183개 job의 efficiency 생산이다. summary 상태는 `complete_with_three_skips_handed_off`, 완료 시각은 2026-09-09 02:28:57 UTC였다. `outputs.merged`와 정식 2024/2025 efficiency 두 파일의 SHA-256은 모두 `1d19be3a7fc6e59f1ee9798cb85b7154feca16ad2c1390ce8382453fcaaa1c68`로 같았다. 따라서 연도별 독립 측정 두 건으로 해석하면 안 된다. 전체 폴더는 약 117 MiB였다. 조사 후 이 폴더의 `2024/summary.json` 경로가 사라진 것을 확인했으며 재생성하지 않았다.

`topw_truth_20260909`에는 전체 intermediate ROOT 복제본이 없다. 단, 검증한 pilot ROOT 두 개(`2024/mc_shard_00000.root` 1,322,069 bytes, `2025/mc_shard_00000.root` 1,322,756 bytes)는 남아 있다. 본 작업은 검증 후 기존 `flat2024_v8`/`flat2025_v8` ROOT를 원래 위치에서 교체한다.

## 코드: 현 분석 및 공용 기능 유지 — 37개

아래는 모두 실행 중이라는 뜻이 아니다. 현재 분석을 생성·검증·재현하거나 SF/upstream 입력을 갱신하는 데 쓰이는 코드다. 이름의 `2024`만으로 연도 전용이라고 판단하지 않았다.

| 코드 | 역할·유지 근거 |
|---|---|
| `append_btageff_datasets_2024.py` | b-tag efficiency에 검증된 dataset 추가. 신호/입력 갱신 시 사용하는 upstream 도구. |
| `apply_flat_external_shape_systematics.py` | 생성된 JES/MET shifted histogram을 nominal에 결합하는 canonical 도구; 파일 존재가 해당 systematic 적용 완료를 뜻하지는 않음. |
| `audit_background_histogram_products.py` | 2024/2025 histogram-only 배경추정 산출물 검사 및 요약. |
| `background_process_groups.py` | 카드·분포에서 쓰는 공통 background process grouping. |
| `build_an_zinv_factors_2024.py` | Sgamma 측정이 import하는 RZ/Qgamma/Sgamma 수식·플롯 helper. 연도명 때문에 삭제하면 현재 호출이 깨짐. |
| `build_combine_inputs.py` | 현재 High-dM73 + frozen Low-dM GNN30의 templates와 TF/RZ/Sgamma likelihood 카드. |
| `build_combined_year_datacards.py` | 2024+2025 카드 결합 및 limit/impact 제출 설정 생성. |
| `build_flat_boosted_recoil_hists.py` | 메인 histogram builder. High-dM와 GNN 공용 weight/array 처리에도 연결. |
| `build_histogram_tf_inputs_2024.py` | canonical histogram으로 CR transfer inputs 작성; 두 연도 설정을 받음. |
| `build_impact_label_map_2024.py` | datacard parameter 이름을 impact에 그대로 쓰는 label 도구. 현재 impact 실행 자체는 사용자 지시로 취소됨. |
| `build_sgamma_ut_report_2024.py` | histogram-only Qgamma/Sgamma 측정. audit가 호출. |
| `build_standard_condor_runtime.sh` | 승인된 py38/Combine batch runtime 구축. |
| `build_zgamma_double_ratio_2024.py` | histogram-only Z/gamma double ratio 및 불확정성 산출. |
| `export_tnp_id_correctionlib.py` | low-pT export 기능은 비활성이나 공용 plot_measurement.py가 직접 import. 현재 파일을 단독 삭제할 수 없음. |
| `extract_cronly_fit_covariance.py` | CR-only FitDiagnostics에서 parameter와 full covariance 추출; run_cronly_fit_eos.sh가 호출. |
| `finalize_cronly_fit_status.py` | fit result·카드·로그를 대조하는 CR-only fit 검사. |
| `gnn_background_histograms.py` | frozen GNN histogram을 TF/배경추정/카드와 연결; 공용 builder가 import. |
| `measure_trigger.py` | 현재 MET/photon trigger SF 측정·축약·그림 진입점. |
| `merge_flat_hist_chunks_streaming.py` | 메인 histogram chunk 병합; run_flat_hists_chunked.py가 호출. |
| `merge_flat_ntuple_metadata.py` | 중간 ROOT sidecar 병합 및 normalization bookkeeping. |
| `plot_control_search_bins_style.py` | 현재 공용 CR/SR/search-bin 분포 플로터. |
| `plot_cronly_nuisance_pulls.py` | 현재 스타일의 CR-only nuisance pull plotter. |
| `plot_highdm73_from_limit_templates.py` | 현재 카드 template에서 High-dM 79−6 SR 그림 생성. |
| `plot_measurement.py` | MET/photon SF와 TF 그림이 사용하는 공용 측정 플로터. |
| `plot_recoil_transfer_factors_2024.py` | High-dM recoil 및 Low-dM GNN TF 그림. |
| `postprocess_limits.py` | 현재 공용 limit 수집·보간·contour 엔진. GNN plot wrapper도 import. |
| `prepare_toptag_eff_campaign.py` | Top/W MC efficiency 측정 campaign 준비. |
| `reference_trigger_counts.py` | 현재 MET/photon trigger 측정의 공용 counting/reduction. |
| `run_asimov_impacts_eos.sh` | 현재 결합 카드의 r=1/r=0 Asimov impact 실행기. 현재 impact 작업은 취소 상태. |
| `run_batch.sh` | 기존 argv batch 실행기를 studies 밖으로 이동한 파일. 활성 truth factory 1112824가 사용; 내용/환경 동일. |
| `run_cronly_fit_eos.sh` | CR-only fit 실행 및 covariance 추출 연결. |
| `run_flat_hists_chunked.py` | 메인 histogram의 chunk 분할·제출·재개·병합 관리. |
| `run_toptag_eff_worker.sh` | Top/W MC efficiency batch worker. |
| `sf_payload.py` | 현재 analysis-owned trigger SF correctionlib payload 생성·검증. |
| `validate_flat_ntuple_outputs.py` | 중간 ROOT/metadata integrity 검사. |
| `validate_histogram_root_inputs.py` | 히스토그램 생산 이전의 branch/readability 검사; downstream 카드 단계에서는 재실행하지 않음. |
| `validate_standard_condor_runtime.sh` | 표준 batch 환경 시험. |

## 코드: 별도 CR-only VR 검증 도구 — 4개

현재 batch/limit 계산에서는 실행하지 않는다. VR 검증 기능의 소스이며, 구형 `results/vr_*` 산출물을 현재 결과로 사용한다는 뜻이 아니다.

| 코드 | 역할 |
|---|---|
| `build_vr_met_postfit.py` | CR-only fit의 covariance를 High-dM VR MET로 전파하는 도구. 현재 production에서 실행하지 않음; 새 모델에 대한 적용 검증 없이 최신 VR 결과라고 볼 수 없음. |
| `inspect_vr_postfit_inputs.py` | VR MET 입력 내용 검사. 현재 production 비활성. |
| `plot_vr_met_postfit.py` | VR postfit 그림; 공용 plot style과 grouping을 사용. 현재 production 비활성. |
| `validate_vr_met_postfit.py` | VR observation fit 제외 및 postfit 산출물 검증. 현재 production 비활성. |

## 코드: 현 분석에서 사용하지 않음 — 26개

독립된 구형 측정·시험·backup 코드다. 아래 코드를 제거할 때는 같은 구형 기능의 tests와 상대 import도 함께 정리해야 한다. 현 분석의 핵심 입력/모델로 보존할 이유는 없다. 이 보고서 작성 중 삭제하지 않았다.

| 코드 | 판정 근거 |
|---|---|
| `audit_lowpt_reference_triggers.py` | 폐기된 5–10 GeV lepton 기준-trigger 진단. 현재 veto>10 GeV 경로에서 호출하지 않음. |
| `audit_photon_fake_template_campaign_2024.py` | 과거 photon-fake template campaign 검사. 현 RZ/Sgamma histogram-only 경로에서 호출하지 않음. |
| `build_flat_boosted_recoil_hists.py.orig` | 구형 histogram builder 백업. 실행 대상은 .py이며 .orig는 사용하지 않음. |
| `build_lowdm_gnn30_combine_inputs.py` | 구형 10 category×3 score 재분배. 현 frozen GNN 6×5 모델이 아님. |
| `build_lowdm_only_canonical_inputs.py` | 기존 Low-dM34만 분리하는 카드 wrapper. 현 GNN30 결합에서 사용하지 않음. |
| `build_nb_recoil_transfer_inputs_2024.py` | 중간 feature ROOT를 다시 읽는 구형 TF 생성기. 현 histogram-only TF로 대체; 로컬에는 이미 없음. |
| `build_photon_fake_template_normalization_extension_2024.py` | 과거 photon-fake 연구 normalization 확장. 현재 normalization에 사용하지 않음. |
| `build_tnp_das_records.py` | 과거 low-pT TnP 원본 목록 작성. 현 veto>10 GeV 분석 입력 경로에 없음. |
| `evaluate_photon_fake_same_region_template_2024.py` | 과거 photon-fake shower-shape closure 연구. |
| `evaluate_photon_fake_truth_transfer_2024.py` | 과거 photon-fake truth-transfer closure 연구. |
| `make_exact_nb_fit_seed.py` | 옛 Nb≥2 fit을 exact-Nb fit의 초기값으로 옮기던 일회성 보조 코드. |
| `measure_photon_fake_template_2024.py` | 과거 shower-shape template 방식 photon fake 측정. 현재 채택된 Qgamma/Sgamma 경로와 별개. |
| `plot_control_search_bins_style.pre_lowdm_drop_20260901.py` | 구형 플로터 백업. 현재 .py 파일로 대체됨. |
| `plot_photon_fake_template_2024.py` | 과거 photon-fake template 연구 보고용 플롯. |
| `postprocess_limits.py.pre_regime_20260902` | 구형 limit 플로터 백업. |
| `postprocess_limits.py.pre_topology_20260902` | 구형 limit 플로터 백업. |
| `postprocess_limits.py.pre_topology_final_20260902` | 구형 limit 플로터 백업. |
| `prepare_tnp_measurement_condor.py` | 과거 low-pT TnP batch 준비. 현재 low-pT SF 사용 중단. |
| `publish_free_background_limits_2024.py` | 폐기된 free-background 모델의 2024 결과 게시 코드. |
| `tnp_fit.py` | 5–10 GeV lepton SF용 TnP resonance fit. 현재 분석 경로 비활성. |
| `tnp_histograms.py` | 과거 low-pT TnP pass/fail histogram 작성. |
| `tnp_measurement_reduce.py` | 과거 low-pT TnP partial 병합. |
| `tnp_measurement_shard.py` | 과거 low-pT TnP shard worker. |
| `tnp_recovery.py` | 과거 low-pT TnP 실패 파일 복구. |
| `validate_fullstat_impact_2024.py` | 2024-only·구형 모델 전용 impact 검사. 현 2024+2025 GNN 모델 검증기로 사용할 수 없음. |
| `validate_tnp_adoption.py` | 과거 low-pT SF 채택 검증; 해당 SF는 현 분석에서 제외됨. |

추가로 `._prepare_toptag_eff_campaign.py`, `._run_toptag_eff_worker.sh`는 macOS 메타데이터 파일이며 실행 코드가 아니다. `validate_standard_condor_runtime.sub`는 runtime 검증용 제출 설정이므로 해당 shell과 함께 유지한다.

## results/: 전부 현재 계산에서 미사용 — 22개

용량은 조사 당시 `du -h` 값으로 반올림됐으며 전체 약 64 GiB다. 3,693개 소스/설정과 현재 Condor의 Cmd/Args/Iwd/TransferInput을 대조했으며, 현재 계산이 이 디렉토리들을 읽는 참조는 없었다. 구형 폴더 내부끼리의 상대 참조는 현재 분석 의존성이 아니다.

| 디렉토리 | 용량 | 내용 | 현재 사용 |
|---|---:|---|---|
| `2024` | 913 MiB | 구형 2024 단독 카드·limit | 아니오 |
| `2024_2025` | 28 KiB | 구형 결합 결과 디렉토리 뼈대 | 아니오 |
| `2024_2025_highdm73_lowdm30_contour_20260901` | 5.2 GiB | 9월 1일 High73+GNN30 결합; 현 veto10 campaign 이전 | 아니오 |
| `2024_2025_highdm73_only_contour_20260902` | 5.2 GiB | 9월 2일 High-dM-only contour 연구 | 아니오 |
| `2024_2025_sr_plots_20260904` | 690 KiB | 9월 4일 High73/GNN30 SR 그림; 현 veto10 그림 이전 | 아니오 |
| `2024_highdm79minus6_lowdm30_20260901` | 915 MiB | 이전 2024 High73+GNN30 pilot/grid | 아니오 |
| `2024_highdm79minus6_lowdm30_allsignal_20260901` | 1.1 GiB | 이전 2024 High73+GNN30 신호 전체 grid | 아니오 |
| `2024_lowdm30_allsignal_extra_20260901` | 27 MiB | 이전 Low-dM30 신호 추가 grid | 아니오 |
| `2024_lowdm_diagonal_v3_20260901` | 14 MiB | manifest상 SR20 (4×5)인 초기 diagonal-v3 모델 | 아니오 |
| `2024_lowdm_diagonal_v3_binning_20260901` | 28 MiB | 이전 GNN binning 후보 연구 | 아니오 |
| `2024_lowdm_gnn20_20260829` | 15 MiB | 폐기된 GNN20 (4×5) 결과 | 아니오 |
| `2024_lowdm_gnn30_20260822` | 86 MiB | 8월 22일 구형 GNN30 결과; 현 frozen 6×5 경로 아님 | 아니오 |
| `2024_lowdm_only_20260822` | 284 MiB | 구형 Low-dM-only 카드/limit | 아니오 |
| `2025` | 707 MiB | 구형 2025 단독 카드·limit | 아니오 |
| `RZ_normalized_Sgamma_shape` | 35 GiB | 이전 RZ/Sgamma 모델의 연도별·결합 카드/fit. Low-dM34 포함 | 아니오 |
| `analysis_model` | 15 GiB | High-dM73/79/85 + Low-dM34의 이전 모델 | 아니오 |
| `background_estimation_unified_20260824` | 101 MiB | High73+Low34 shared-CR pilot 모델 | 아니오 |
| `new_signals_2024_highdm73_lowdm34` | 118 MiB | 이전 신규 신호 점검용 High73+Low34 결과 | 아니오 |
| `search_bins` | 1.1 MiB | 이전 highdm79_lowdm34 search-bin 산출물 | 아니오 |
| `vr_postfit_cronly` | 36 MiB | 이전 High73+Low34 CR-only VR postfit | 아니오 |
| `vr_postfit_cronly_exact` | 67 MiB | 이전 High73+Low34 exact-Nb CR-only VR postfit | 아니오 |
| `vr_postfit_cronly_grouped` | 15 MiB | 이전 grouped-CR VR postfit | 아니오 |

## 실제로 드러난 관리 문제와 이번 조치

- 메인 소스, 실행 campaign snapshot, 일회성 진단, backup을 같은 workflow 층에 섞어 두었다.
- 로컬에서 폐기한 `build_nb_recoil_transfer_inputs_2024.py`가 EOS에 남아 있다. 전체 local/EOS/Git 동기화가 완료된 상태가 아니다.
- `studies/.../run_flat_hist_merge_batch.sh`를 새 truth 생산에서도 실행기로 참조했다. 연구 디렉토리 삭제가 현재 작업 시작에 영향을 주는 잘못된 의존성이었다.
- 이 실행기만 `workflow/run_batch.sh`로 이동했다. SHA-256 `4c91cdcbe19e872ca1ff5c09e9662b3cee38f524d6fb57ca8a7a68a8e0eede0c`로 내용·물리·환경은 동일하다.
- factory 1112824의 Cmd와 새로 materialize된 작업이 새 경로를 사용하는 것을 확인했다. 삭제된 실행기 전송 실패(HoldReasonCode=13 및 정확한 이전 경로) 작업만 재개했다. 정상 작업과 다른 계산 오류 10개는 중단·재제출하지 않았다.
- 완료된 연구·efficiency 폴더를 통째로 복원하거나 새 ROOT cache를 로컬에 만들지 않았다.

확인 범위: EOS 최상위 코드 67개, 제한된 소스 1,180개의 import/문자열 참조, 현재 source/config 3,693개, 두 연도 canonical aliases, 현재 EOS schedd 작업, 각 results 디렉토리의 내용과 가능한 manifest. 데이터 이벤트를 읽거나 fit/limit를 재계산한 조사가 아니다.
