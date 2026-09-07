# Low-dM 캐노니컬 워크플로우 의존성 전수 목록

작성 기준: 2026-09-07, 로컬 작업 트리. HEAD `395adc41`. 공개 명령의 dispatch, 재귀 import, 함수 내부 지연 import, Condor payload 복사, 파일 I/O와 공용 코드 연결을 조사했다.

범위: **공개 메인 코드 8개, 내부 구현 Python 34개, 패키지 초기화 파일 2개, 실행 shell 1개**, 직접·재귀적으로 연결되는 공용 Python 9개와 직렬화 보정의 원본 코드 1개. 별도로 공용 결합·그림 도구 7개를 대조했다. 표의 함수 범위는 모듈 전체의 모든 함수가 실행된다는 뜻이 아니다.

외부 ROOT와 EOS runtime은 읽거나 실행하지 않았다. 이벤트 전수처리·학습·카드 생성·리미트 계산은 수행하지 않았으며, 기존 코드와 모델을 수정하지 않았다. 수천 개 입력 ROOT의 개별 경로는 실행 request/manifest가 결정하므로 아래에 이를 결정하는 필드와 파일 패턴을 모두 적었다. Python 배포 패키지 내부의 vendor 의존성 전체는 EOS lockfile 확인 없이는 확정할 수 없다.

## 1. 실행 진입점

| 메인 코드 | 역할 |
|---|---|
| [autonomous_allhad/gnn_lowdm/data.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/data.py>) | 입력 branch, event split, feature 계산, 30-bin mapping |
| [autonomous_allhad/gnn_lowdm/model.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/model.py>) | GNN/MLP 등 model class 정의 |
| [autonomous_allhad/gnn_lowdm/train.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/train.py>) | manifest/cache/학습 dispatch |
| [autonomous_allhad/gnn_lowdm/valid.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/valid.py>) | NumPy·feature·template 검증 dispatch |
| [autonomous_allhad/gnn_lowdm/eval.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/eval.py>) | test/CR/signal/limit 준비 dispatch |
| [autonomous_allhad/gnn_lowdm/build_datacard.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/build_datacard.py>) | Low-dM datacard 작성 |
| [autonomous_allhad/gnn_lowdm/merge_datacard.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/merge_datacard.py>) | 연도별 ROOT와 카드 패키징 |
| [autonomous_allhad/gnn_lowdm/plotting.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/plotting.py>) | 학습·ROC·SHAP·CR·limit·schematic·웹 dispatch |

네 개 dispatcher는 모두 [autonomous_allhad/gnn_lowdm/_implementation/cli.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/cli.py>)의 `dispatch()` → `importlib.import_module()` → 대상 함수 순서로 연결된다.

| 공개 명령 | 실제 실행 파일·함수 | 진입부 확인 |
|---|---|---|
| `train manifest` | [autonomous_allhad/gnn_lowdm/_implementation/build_full_campaign_manifest.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_full_campaign_manifest.py>) → `main()` | --help 통과 |
| `train prepare-cache` | [autonomous_allhad/gnn_lowdm/_implementation/prepare_diagonal_v3_cache_condor.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_diagonal_v3_cache_condor.py>) → `main()` | --help 통과 |
| `train cache-worker` | [autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py>) → `main()` | --help 통과 |
| `train finalize-cache` | [autonomous_allhad/gnn_lowdm/_implementation/finalize_diagonal_v3_cache.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/finalize_diagonal_v3_cache.py>) → `main()` | --help 통과 |
| `train fit` | [autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py>) → `main()` | --help 통과 |
| `valid numpy-parity` | [autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_numpy_inference.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_numpy_inference.py>) → `main()` | --help 통과 |
| `valid feature-parity` | [autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_region_feature_parity.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_region_feature_parity.py>) → `main()` | --help 통과 |
| `valid templates` | [autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_srcr_inputs.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_srcr_inputs.py>) → `main()` | --help 통과 |
| `eval test` | [autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py>) → `main()` | --help 통과 |
| `eval cr-partial` | [autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py>) → `main()` | --help 통과 |
| `eval cr-merge` | [autonomous_allhad/gnn_lowdm/_implementation/merge_diagonal_v3_cr_nnout_partials.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_diagonal_v3_cr_nnout_partials.py>) → `main()` | --help 통과 |
| `eval signal-prepare` | [autonomous_allhad/gnn_lowdm/_implementation/prepare_all_signal_template_condor.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_all_signal_template_condor.py>) → `main()` | --help 통과 |
| `eval signal-partial` | [autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py>) → `main()` | --help 통과 |
| `eval signal-merge` | [autonomous_allhad/gnn_lowdm/_implementation/merge_all_signal_templates.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_all_signal_templates.py>) → `main()` | 실패: 잘못된 build_datacard import |
| `eval limits-prepare` | [autonomous_allhad/gnn_lowdm/_implementation/prepare_highdm79minus6_lowdm30_condor.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_highdm79minus6_lowdm30_condor.py>) → `main()` | --help 통과 |
| `plotting training-curves` | [autonomous_allhad/gnn_lowdm/_implementation/supplementary_plots.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/supplementary_plots.py>) → `main_training_curves()` | --help 통과 |
| `plotting roc` | [autonomous_allhad/gnn_lowdm/_implementation/supplementary_plots.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/supplementary_plots.py>) → `main_roc()` | --help 통과 |
| `plotting shap` | [autonomous_allhad/gnn_lowdm/_implementation/supplementary_plots.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/supplementary_plots.py>) → `main_shap()` | --help 통과 |
| `plotting cr` | [autonomous_allhad/gnn_lowdm/_implementation/plot_control_regions.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_control_regions.py>) → `main()` | --help 통과 |
| `plotting limit` | [autonomous_allhad/gnn_lowdm/_implementation/collect_plot_diagonal_v3_limits.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/collect_plot_diagonal_v3_limits.py>) → `main()` | --help 통과 |
| `plotting combined-limit` | [autonomous_allhad/gnn_lowdm/_implementation/collect_plot_highdm79minus6_lowdm30.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/collect_plot_highdm79minus6_lowdm30.py>) → `main()` | --help 통과 |
| `plotting schematic` | [autonomous_allhad/gnn_lowdm/_implementation/plot_diagonal_v3_model_schematic.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_diagonal_v3_model_schematic.py>) → `main()` | --help 통과 |
| `plotting publish` | [autonomous_allhad/gnn_lowdm/_implementation/publish_lowdm_gnn_web.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/publish_lowdm_gnn_web.py>) → `main()` | --help 통과 |

`--help` 23개 중 22개가 통과했다. 이는 CLI import 확인이며 생산 실행의 성공 증거는 아니다.

## 2. 발견된 의존성·재현성 제한

1. **signal-merge import가 끊어져 있다.** [autonomous_allhad/gnn_lowdm/_implementation/merge_all_signal_templates.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_all_signal_templates.py:18>)는 `._implementation/build_datacard.py`에 해당하는 `.build_datacard`를 찾지만 실제 파일은 상위 [autonomous_allhad/gnn_lowdm/build_datacard.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/build_datacard.py>)다. fallback `from build_datacard`도 repository-root의 공개 명령 실행에서 실패했다.
2. **frozen 모델 패키지와 평가기의 입력 계약이 다르다.** [autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py:311>)는 `selection.json`의 `test_touched`, `score_edges`와 `trials/<best_trial>/best_model.pt`를 요구한다. frozen 패키지는 모델을 직접 `best_model.pt`에 두고, score edges는 `config.json`에 있으며 이미 `test_touched=true`다. CR 평가기 역시 `--selection`의 최상위 `score_edges`를 요구한다. 현재 `config.json`이나 frozen `selection.json`을 그대로 대입해서 재실행할 수 있다고 말할 수 없다.
3. **signal-prepare가 필수 파일을 전부 배치하지 않는다.** payload Python 3개만 복사하며 request에서 참조하는 model, config, xsec, campaign manifest와 `runtime/py38`, `vendor/mt2`는 별도 사전 배치가 필요하다. signal cache 개수를 11개, manifest 이름을 `full_campaign_manifest_2024.json`으로 고정한다.
4. **배포·runtime 계약이 서로 다르다.** 학습 requirements는 Python 3.12 계열, signal worker는 py38 및 mt2 1.2.0, cache worker는 CVMFS LCG_110_swan, limit plotter는 별도 py38 버전 집합이다. 학습 requirements 하나로 전체 workflow가 충족되지 않는다.
5. **로컬 보정 파일 일부가 없다.** 2025 AnalysisSF 4개와 `analysis/hists/btageff2024.merged`, `btageff2025.merged`가 현 repository 위치에 없다. EOS 사본 유무는 이 조사에서 확인하지 않았다. `corrections.coffea`는 존재하지만 원본 소스와 직렬화 payload의 동등성은 확인하지 않았다.
6. **현재 plot dispatcher의 연결이 최종 공용 인터페이스와 완전히 일치하지 않는다.** `cr`은 자체 `plot_control_regions.py`를 쓰며 공용 `plot_control_search_bins_style.py`를 import하지 않는다. `combined-limit`은 2024 luminosity와 구형 Run-2 입력명, Low-dM-only 추가 grid를 사용하는 wrapper다. 공용 `postprocess_limits.py`의 최신 CLI 기본값·topology·xsec 옵션을 전부 전달하지 않는다.
7. **30이라는 총 bin 수만으로 다른 workflow를 같은 것으로 분류하면 안 된다.** 공용 디렉토리의 `build_lowdm_gnn30_combine_inputs.py`는 10 category × 3 score bins 및 기존 34-bin 재분배 코드다. 현 frozen 6 × 5 GNN30의 import 그래프에 연결되지 않으며, 이 보고서에서는 대조 대상으로 분리했다.
8. **2024+2025 결합과 SR 최초 배경-template 작성이 공개 8개 진입점 안에서 끝까지 연결되어 있지 않다.** `merge_datacard.py`는 연도별 패키징이고, signal merge는 기존 Low-dM template/manifest를 입력으로 받아 신호를 추가한다. `limits-prepare`는 주어진 High/Low 카드들을 결합한다. 두 연도 결합 공용 도구는 아래 별도 항목이며 자동으로 호출되지 않는다.
9. **systematic 전파는 파일의 존재와 별개다.** 현 `build_datacard.py`는 signal lumi, CR rateParam, autoMCStats를 작성한다. object/weight systematic GNN score migration까지 완성된 통계모델이라는 의미가 아니다.

## 3. 설정·모델·고정 참조 파일

| 파일 | 용도 | 로컬 상태 |
|---|---|---|
| [autonomous_allhad/gnn_lowdm/config.json](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/config.json>) | 선택 모델·30-bin SR·CR 정의·연도 luminosity | 존재 |
| [autonomous_allhad/gnn_lowdm/configs/diagonal_v3_significance_trials_v1.json](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/configs/diagonal_v3_significance_trials_v1.json>) | train fit --trials-json; 없으면 소스 내 기본 trial 사용 | 존재 |
| [autonomous_allhad/gnn_lowdm/requirements.txt](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/requirements.txt>) | 학습 환경 버전 | 존재 |
| [autonomous_allhad/signals/stop_xsec_13p6TeV.json](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/signals/stop_xsec_13p6TeV.json>) | signal xsec·limit cross-section 색상 | 존재 |
| [autonomous_allhad/gnn_lowdm/models/diagonal_v3_h48_l3_sig010/best_model.pt](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/models/diagonal_v3_h48_l3_sig010/best_model.pt>) | frozen PyTorch checkpoint | 존재; SHA256 일치 |
| [autonomous_allhad/gnn_lowdm/models/diagonal_v3_h48_l3_sig010/diagonal_v3_numpy.npz](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/models/diagonal_v3_h48_l3_sig010/diagonal_v3_numpy.npz>) | NumPy inference weights·scalers | 존재; SHA256 일치 |
| [autonomous_allhad/gnn_lowdm/models/diagonal_v3_h48_l3_sig010/diagonal_v3_numpy.json](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/models/diagonal_v3_h48_l3_sig010/diagonal_v3_numpy.json>) | portable-model metadata; inference는 NPZ를 읽음 | 존재; SHA256 일치 |
| [autonomous_allhad/gnn_lowdm/models/diagonal_v3_h48_l3_sig010/selection.json](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/models/diagonal_v3_h48_l3_sig010/selection.json>) | best trial/epoch 및 test 사용 기록 | 존재; SHA256 일치 |
| [autonomous_allhad/gnn_lowdm/models/diagonal_v3_h48_l3_sig010/summary.json](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/models/diagonal_v3_h48_l3_sig010/summary.json>) | 학습 요약 | 존재; SHA256 일치 |
| [autonomous_allhad/gnn_lowdm/models/diagonal_v3_h48_l3_sig010/training_history.json](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/models/diagonal_v3_h48_l3_sig010/training_history.json>) | supplementary training curves | 존재; SHA256 일치 |
| [autonomous_allhad/gnn_lowdm/models/diagonal_v3_h48_l3_sig010/reference_baseline_best_model.pt](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/models/diagonal_v3_h48_l3_sig010/reference_baseline_best_model.pt>) | 선택적 baseline 비교; frozen inference에 불필요 | 존재; SHA256 일치 |
| [autonomous_allhad/gnn_lowdm/models/diagonal_v3_h48_l3_sig010/MANIFEST.json](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/models/diagonal_v3_h48_l3_sig010/MANIFEST.json>) | 7개 payload checksum·환경 provenance | 존재 |
| [autonomous_allhad/gnn_lowdm/inputs/run2_sus19010_t2bw_contours.json](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/inputs/run2_sus19010_t2bw_contours.json>) | SUS-19-010 Run-2 overlay | 존재 |
| [autonomous_allhad/gnn_lowdm/inputs/run2_sus19010_t2bwc_contours.json](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/inputs/run2_sus19010_t2bwc_contours.json>) | SUS-19-010 Run-2 overlay | 존재 |
| [autonomous_allhad/gnn_lowdm/inputs/run2_sus19010_t2tb_contours.json](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/inputs/run2_sus19010_t2tb_contours.json>) | SUS-19-010 Run-2 overlay | 존재 |
| [autonomous_allhad/gnn_lowdm/inputs/run2_sus19010_t2tt_contours.json](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/inputs/run2_sus19010_t2tt_contours.json>) | SUS-19-010 Run-2 overlay | 존재 |
| [autonomous_allhad/gnn_lowdm/inputs/run2_sus19010_t2ttc_contours.json](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/inputs/run2_sus19010_t2ttc_contours.json>) | SUS-19-010 Run-2 overlay | 존재 |

## 4. 보정 파일과 원본/compiled 연결

`CR eval → region_io.normalized_weight_variations → build_flat_boosted_recoil_hists.flat_arrays_for_weights → real_subset_worker.compute_weight_bundle → analysis/data/corrections.coffea` 순서다. 자체 측정 SF는 `analysis_scale_factors.py`가 correctionlib JSON을 별도로 연다.

`analysis/utils/corrections.py`는 compiled payload의 원본이다. import 끝에 `save(corrections, ...)`가 있어, 이 조사에서는 직접 import하거나 재컴파일하지 않았다. `ids.py`, `ids.coffea`, `common.coffea`, `stop_processor_v4.py`는 현 frozen GNN의 직접 runtime import가 아니다. 이미 만들어진 intermediate ROOT의 upstream provenance와 혼동하지 않아야 한다.

| 파일 | 적용·의존 구분 | 로컬 상태 |
|---|---|---|
| [analysis/data/corrections.coffea](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/data/corrections.coffea>) | nominal MC weight evaluation | 존재 |
| [analysis/utils/corrections.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py>) | compiled payload 재생성 원본 | 존재 |
| [analysis/data/PUweight/2024/puWeights.json.gz](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/data/PUweight/2024/puWeights.json.gz>) | pileup | 존재 |
| [analysis/data/EGammaSF/2024/electron.json.gz](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/data/EGammaSF/2024/electron.json.gz>) | electron ID/reco | 존재 |
| [analysis/data/EGammaSF/2024/photon.json.gz](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/data/EGammaSF/2024/photon.json.gz>) | photon ID/CSEV | 존재 |
| [analysis/data/MuonSF/2024/muon_Z.json.gz](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/data/MuonSF/2024/muon_Z.json.gz>) | muon ID/iso/HLT | 존재 |
| [analysis/data/BTVSF/2024/btagging.json.gz](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/data/BTVSF/2024/btagging.json.gz>) | UParTAK4 b-tag SF | 존재 |
| [analysis/data/EGammaSF/2024/electronHlt.json.gz](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/data/EGammaSF/2024/electronHlt.json.gz>) | electron HLT; 2025는 명시적 미제공 unity 경로 | 존재 |
| [analysis/data/AnalysisSF/2024/met_trigger_sf.json.gz](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/data/AnalysisSF/2024/met_trigger_sf.json.gz>) | analysis-owned SF | 존재 |
| [analysis/data/AnalysisSF/2024/photon_trigger_sf.json.gz](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/data/AnalysisSF/2024/photon_trigger_sf.json.gz>) | analysis-owned SF | 존재 |
| [analysis/data/AnalysisSF/2024/veto_electron_5to10_sf.json.gz](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/data/AnalysisSF/2024/veto_electron_5to10_sf.json.gz>) | analysis-owned SF | 존재 |
| [analysis/data/AnalysisSF/2024/loose_muon_5to10_sf.json.gz](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/data/AnalysisSF/2024/loose_muon_5to10_sf.json.gz>) | analysis-owned SF | 존재 |
| [analysis/hists/btageff2024.merged](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/hists/btageff2024.merged>) | b-tag efficiency; dataset별 조회 | 없음 |
| [analysis/data/JMESF/2024/jetid.json.gz](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/data/JMESF/2024/jetid.json.gz>) | upstream object construction/JEC; 기존 flat CR weight 경로에서는 직접 재계산하지 않음 | 존재 |
| [analysis/data/JMESF/2024/jetvetomaps.json.gz](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/data/JMESF/2024/jetvetomaps.json.gz>) | upstream object construction/JEC; 기존 flat CR weight 경로에서는 직접 재계산하지 않음 | 존재 |
| [analysis/data/JMESF/2024/jet_jerc.json.gz](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/data/JMESF/2024/jet_jerc.json.gz>) | upstream object construction/JEC; 기존 flat CR weight 경로에서는 직접 재계산하지 않음 | 존재 |
| [analysis/data/JMESF/2024/fatJet_jerc.json.gz](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/data/JMESF/2024/fatJet_jerc.json.gz>) | upstream object construction/JEC; 기존 flat CR weight 경로에서는 직접 재계산하지 않음 | 존재 |
| [analysis/data/lumiMask/Cert_Collisions2024_378981_386951_Golden.json](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/data/lumiMask/Cert_Collisions2024_378981_386951_Golden.json>) | upstream data luminosity mask | 존재 |
| [analysis/data/PUweight/2025/puWeights_2025pp_Golden_Summer24_25ns_69200ub.json.gz](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/data/PUweight/2025/puWeights_2025pp_Golden_Summer24_25ns_69200ub.json.gz>) | pileup | 존재 |
| [analysis/data/EGammaSF/2025/electron.json.gz](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/data/EGammaSF/2025/electron.json.gz>) | electron ID/reco | 존재 |
| [analysis/data/EGammaSF/2025/photon.json.gz](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/data/EGammaSF/2025/photon.json.gz>) | photon ID/CSEV | 존재 |
| [analysis/data/MuonSF/2025/muon_Z.json.gz](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/data/MuonSF/2025/muon_Z.json.gz>) | muon ID/iso/HLT | 존재 |
| [analysis/data/BTVSF/2025/btagging.json.gz](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/data/BTVSF/2025/btagging.json.gz>) | UParTAK4 b-tag SF | 존재 |
| [analysis/data/EGammaSF/2025/electronHlt.json.gz](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/data/EGammaSF/2025/electronHlt.json.gz>) | electron HLT; 2025는 명시적 미제공 unity 경로 | 없음 |
| [analysis/data/AnalysisSF/2025/met_trigger_sf.json.gz](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/data/AnalysisSF/2025/met_trigger_sf.json.gz>) | analysis-owned SF | 없음 |
| [analysis/data/AnalysisSF/2025/photon_trigger_sf.json.gz](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/data/AnalysisSF/2025/photon_trigger_sf.json.gz>) | analysis-owned SF | 없음 |
| [analysis/data/AnalysisSF/2025/veto_electron_5to10_sf.json.gz](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/data/AnalysisSF/2025/veto_electron_5to10_sf.json.gz>) | analysis-owned SF | 없음 |
| [analysis/data/AnalysisSF/2025/loose_muon_5to10_sf.json.gz](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/data/AnalysisSF/2025/loose_muon_5to10_sf.json.gz>) | analysis-owned SF | 없음 |
| [analysis/hists/btageff2025.merged](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/hists/btageff2025.merged>) | b-tag efficiency; dataset별 조회 | 없음 |
| [analysis/data/JMESF/2025/jetid.json.gz](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/data/JMESF/2025/jetid.json.gz>) | upstream object construction/JEC; 기존 flat CR weight 경로에서는 직접 재계산하지 않음 | 존재 |
| [analysis/data/JMESF/2025/jetvetomaps.json.gz](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/data/JMESF/2025/jetvetomaps.json.gz>) | upstream object construction/JEC; 기존 flat CR weight 경로에서는 직접 재계산하지 않음 | 존재 |
| [analysis/data/JMESF/2025/jet_jerc.json.gz](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/data/JMESF/2025/jet_jerc.json.gz>) | upstream object construction/JEC; 기존 flat CR weight 경로에서는 직접 재계산하지 않음 | 존재 |
| [analysis/data/JMESF/2025/fatJet_jerc.json.gz](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/data/JMESF/2025/fatJet_jerc.json.gz>) | upstream object construction/JEC; 기존 flat CR weight 경로에서는 직접 재계산하지 않음 | 존재 |
| [analysis/data/lumiMask/Cert_Collisions2025_391658_398903_Golden.json](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/data/lumiMask/Cert_Collisions2025_391658_398903_Golden.json>) | upstream data luminosity mask | 존재 |

RZ는 [autonomous_allhad/gnn_lowdm/_implementation/region_io.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py>)의 `RZ_FACTORS` 상수로 들어간다. 현재 GNN CR 경로가 독립 RZ/Sgamma fit JSON을 읽는다고 해석하면 안 된다. 다른 공용 likelihood 경로의 `--rz-low`, `--rz-high`, `--sgamma`, `--zgamma-double-ratio` 입력은 아래 대조 부록에 분리했다.

추가 실행 의존: `real_subset_worker.local_analysis_dir()`의 기본 동작은 `analysis/data` 전체와 `analysis/hists/btageff*.merged`를 runtime cache로 복사한다. 따라서 함수별 최소 보정 파일뿐 아니라 해당 디렉토리와 쓰기 가능한 cache 경로도 요구한다. EOS 전용 실행에서는 관련 환경변수/작업 경로 계약을 함께 설정해야 한다.

## 5. 단계별 입력·중간 산출물 계약

| 단계 | 실제로 읽는 파일/필드 |
|---|---|
| manifest | --source-directory 아래 shard metadata JSON; 내부 ROOT 경로·dataset_id·sumw·year·luminosity로 full campaign manifest 생성 |
| cache 준비/worker | --manifest; request[inputs]의 root/sidecar; Events + TROTA tree; highdm_resolved_categories.py를 payload로 복사; cache ROOT+metadata JSON 생성 |
| finalize-cache | campaign_manifest.json, request JSON, output shard ROOT+JSON; 완성·누락·source accounting 확인 |
| train fit | --cache의 mc_cache_*.root / signal_cache_*.root 및 동일 stem JSON, campaign_state.json; --campaign-manifest; --xsec; 선택 --trials-json 및 --baseline-checkpoint; resume latest.pt/summary.json |
| eval test | 동일 cache/manifest/xsec; --result/selection.json; --result/trials/<best_trial>/best_model.pt; 기존 test summary. frozen 모델 패키지와 위치·schema 계약 차이 존재 |
| CR inference | --request JSON → request[manifest], [stop_xsec], [repository], [inputs][root,sidecar]; --model NPZ; --selection JSON의 top-level score_edges; Events/TROTA; MC 보정 파일 |
| CR merge | --partials의 CR partial JSON; manifest/request directory 및 선택 expected-inputs; histogram+variance+audit merge |
| signal inference | request[input_root, campaign_manifest, xsec, model, configuration, output]; configuration.sr_binning; request에는 input_sidecar 경로도 기록하지만 worker는 해당 필드를 직접 읽지 않음 |
| signal merge | signal_cache_*.json partials; --existing-low/<topology>/manifest.json 및 template_root; --high-grid campaign JSON; 기존 background/CR template을 복사하고 signal histogram 추가 |
| build_datacard | --template ROOT, --summary JSON(categories.SR/channels), config.json; topology/mass별 nominal signal histograms |
| merge_datacard | --template TOPOLOGY=ROOT, --cards TOPOLOGY=directory; 선택 --update-template/--update-cards; config.json; lowdm_2024.root/lowdm_2025.root와 rewritten cards 생성 |
| limits-prepare / shell | --high-base/<topology>/manifest.json, --low-base/<topology>/manifest.json; 두 영역 cards/datacard_<mass>.txt와 shapes ROOT; mass bundle TXT; EOS CMSSW/Combine |
| limit collect/plot | campaign or input-dir manifest; limits/higgsCombine*.root의 limit tree; optional expected_limits.json 재사용; xsec JSON; Run-2 5개 contour JSON; optional excluded_mass_points.json·overlay/baseline limits·thermal relic JSON |
| training curves | models/diagonal_v3_h48_l3_sig010/training_history.json; config.json.model.selected_epoch |
| ROC | results/diagonal_v3_significance_full_20260831/test/diagonal_v3_test_scores.npz의 scores,labels,weights,signal_topology_id |
| SHAP plot | results/diagonal_v3_significance_full_20260831/global_shap/diagonal_v3_global_gradientshap_values.npz의 shap_values,feature_values,feature_names; 40-global schema |
| SHAP 재계산 | analyze_diagonal_v3_global_shap.py의 별도 main; --cache, --result-dir/selection.json 및 trials checkpoint; 현재 plotting shap은 저장 값의 재플롯만 dispatch |
| CR plot | --input merged histogram JSON; --luminosity-fb 선택; ROOT 직접 읽지 않음 |
| 웹 | --sr-2024/plots/sr/lowdm_sr_gnn_out_30bin.{png,pdf}, --cr-2024/plots/cr/*inclusive.{png,pdf}, --result-2025 동일 구조+validation_summary.json; --highdm73; --lowdm-cr-physics-plots/<year> 그림; CR plot summary JSON |

## 6. 공용 코드 연결

| 코드 | 현 GNN과의 관계 |
|---|---|
| [autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py>) | CR weight branch 목록·flat arrays·signal b-tag dataset anchor |
| [autonomous_allhad/workflow/postprocess_limits.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py>) | 두 limit wrapper가 import하는 유일 공용 contour 엔진 |
| [autonomous_allhad/autonomous_allhad/real_subset_worker.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py>) | compute_weight_bundle과 compiled correction 로더 |
| [autonomous_allhad/autonomous_allhad/analysis_scale_factors.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/analysis_scale_factors.py>) | analysis-owned trigger/low-pT SF JSON 로더 |
| [autonomous_allhad/autonomous_allhad/signal_models.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/signal_models.py>) | T2tt/T2bW/T2tb topology ID·dataset 해석 |
| [autonomous_allhad/autonomous_allhad/highdm_resolved_categories.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/highdm_resolved_categories.py>) | TROTA resolved candidate/event 매핑·overlap veto |
| [autonomous_allhad/autonomous_allhad/dy_ptll_policy.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/dy_ptll_policy.py>) | 공용 flat worker의 DY dataset policy import |
| [autonomous_allhad/autonomous_allhad/search_bin_categorization.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/search_bin_categorization.py>) | 공용 flat worker의 High-dM bin layout import |
| [autonomous_allhad/autonomous_allhad/sidecar_store.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/sidecar_store.py>) | 공용 flat worker metadata reader; adjacent JSON/SQLite fallback |
| [analysis/utils/corrections.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py>) | corrections.coffea의 원본; 직렬화 의존으로 분리 |

공용 파일끼리의 전체 import는 부록 A에 있다. 특히 `region_io`가 공용 flat builder를 함수 안에서 가져오고, 공용 flat builder가 다시 `region_io`를 가져오는 양방향 연결이 있다. 현재 import는 지연되어 있지만 파일 한쪽만 이동하면 깨질 수 있다.

### 공용 도구 대조: 공개 GNN dispatcher에서 자동 연결되지 않는 파일

| 파일 | 관계 |
|---|---|
| [autonomous_allhad/workflow/run_flat_hists_chunked.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/run_flat_hists_chunked.py>) | 공용 histogram chunk orchestration; 직접 GNN import는 아니며 execution contract에 region_io 경로 기록 |
| [autonomous_allhad/workflow/build_combined_year_datacards.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combined_year_datacards.py>) | 2024+2025 카드 결합 후보; build_combine_inputs를 import하며 추가 model contract를 검사 |
| [autonomous_allhad/workflow/build_combine_inputs.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py>) | 공용 histogram likelihood/datacard builder; 공용 연도 결합 도구가 사용 |
| [autonomous_allhad/workflow/background_process_groups.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/background_process_groups.py>) | 공용 배경 process grouping |
| [autonomous_allhad/workflow/plot_control_search_bins_style.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py>) | 사용자가 지정한 공용 distribution plotter; 현 plotting cr은 이 코드를 호출하지 않음 |
| [autonomous_allhad/workflow/plot_highdm73_from_limit_templates.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_highdm73_from_limit_templates.py>) | high-dM73 template 플롯; 공용 style을 import |
| [autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py>) | 이름만 유사한 10×3 score redistribution; frozen 6×5 모델과 별개 |

## 7. 외부 패키지·실행 환경

| 경로 | 요구 환경 |
|---|---|
| 학습·PyTorch·supplementary | Python 3.12.4; requirements.txt의 numpy 2.0.2, awkward 2.13.0, torch 2.13.0, uproot 5.7.6, matplotlib 3.9.1, mplhep 0.3.50, mt2 1.3.1, xxhash 4.0.1; pytest는 검증용 |
| 공용 CR weight | 추가 correctionlib, coffea; compiled/source correction은 hist·coffea lookup_tools/jetmet_tools를 요구. 이 버전들은 GNN requirements에 없음 |
| signal EOS worker | runtime/py38/bin/python3, vendor/mt2/mt2/__init__.py, mt2==1.2.0; 코드가 이미 설치된 runtime을 요구하며 py38.tgz를 직접 풀지 않음 |
| cache EOS worker | /cvmfs/sft.cern.ch/lcg/views/LCG_110_swan/x86_64-el9-gcc13-opt/setup.sh; python3 및 XRootD 명령 |
| 공용 limit plotter CLI | Python 3.8.20; numpy 1.23.5, scipy 1.10.1, matplotlib 3.7.3, mplhep 0.4.1, uproot 4.3.7 |
| template / card packaging | PyROOT (ROOT.TFile/TH1); ROOT 공유 라이브러리와 해당 Python ABI |
| Combine | EOS CMSSW_14_1_0_pre4; /cvmfs/cms.cern.ch/cmsset_default.sh; scramv1, combine, combineCards.py, timeout; Condor schedd |
| SHAP | 현재 자체 GradientSHAP 구현: torch autograd 사용. shap/captum/scikit-learn은 이 import 그래프에서 요구하지 않음 |

제3자 Python 직접 import 전체(대조 공용 도구 포함): `ROOT`, `awkward`, `coffea`, `correctionlib`, `hist`, `matplotlib`, `mplhep`, `mt2`, `numpy`, `scipy`, `torch`, `uproot`.

Python 표준 라이브러리 전체: `__future__`, `argparse`, `array`, `collections`, `concurrent`, `contextlib`, `copy`, `csv`, `dataclasses`, `datetime`, `functools`, `gzip`, `hashlib`, `html`, `importlib`, `itertools`, `json`, `math`, `mmap`, `os`, `pathlib`, `random`, `re`, `resource`, `shutil`, `sqlite3`, `subprocess`, `sys`, `time`, `traceback`, `typing`, `zlib`.

외부 패키지 내부의 의존성 버전·ROOT/CMSSW 공유라이브러리·EOS mount 실재성은 별도 runtime manifest/lockfile이 있어야 확정된다. 이 문서는 프로젝트 소스의 모든 import와 resource 참조를 열거한 것이며, pip lockfile의 대체물이 아니다.

## 부록 A. 파일별 import 전수 목록

각 표에는 함수 내부 import와 EOS standalone fallback도 포함한다. “해결 파일”은 로컬에서 찾은 후보다. fallback은 해당 payload에 파일이 함께 배치되어야 하며 repository에서 후보를 찾았다는 사실만으로 worker import가 성공하는 것은 아니다. `from .build_datacard`처럼 잘못된 relative 경로는 실패로 표시한다.

### A1. GNN 공개·내부 코드

#### [autonomous_allhad/gnn_lowdm/__init__.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/__init__.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| — | 없음 | 패키지 초기화만 수행 |

#### [autonomous_allhad/gnn_lowdm/_implementation/__init__.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/__init__.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| — | 없음 | 패키지 초기화만 수행 |

#### [autonomous_allhad/gnn_lowdm/_implementation/analyze_diagonal_v3_global_shap.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/analyze_diagonal_v3_global_shap.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [11](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/analyze_diagonal_v3_global_shap.py:11>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [13](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/analyze_diagonal_v3_global_shap.py:13>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [14](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/analyze_diagonal_v3_global_shap.py:14>): `module` | `import hashlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [15](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/analyze_diagonal_v3_global_shap.py:15>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [16](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/analyze_diagonal_v3_global_shap.py:16>): `module` | `import time` | 표준 라이브러리 / 외부 설치 패키지 |
| [17](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/analyze_diagonal_v3_global_shap.py:17>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [19](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/analyze_diagonal_v3_global_shap.py:19>): `module` | `import matplotlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [22](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/analyze_diagonal_v3_global_shap.py:22>): `module` | `import matplotlib.pyplot as plt` | 표준 라이브러리 / 외부 설치 패키지 |
| [23](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/analyze_diagonal_v3_global_shap.py:23>): `module` | `import mplhep as hep` | 표준 라이브러리 / 외부 설치 패키지 |
| [24](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/analyze_diagonal_v3_global_shap.py:24>): `module` | `import numpy as np` | 표준 라이브러리 / 외부 설치 패키지 |
| [25](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/analyze_diagonal_v3_global_shap.py:25>): `module` | `import torch` | 표준 라이브러리 / 외부 설치 패키지 |
| [26](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/analyze_diagonal_v3_global_shap.py:26>): `module` | `from matplotlib.lines import Line2D` | 표준 라이브러리 / 외부 설치 패키지 |
| [27](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/analyze_diagonal_v3_global_shap.py:27>): `module` | `from matplotlib.ticker import NullLocator` | 표준 라이브러리 / 외부 설치 패키지 |
| [29](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/analyze_diagonal_v3_global_shap.py:29>): `module` | `from ..data import DIAGONAL_V3_GLOBAL_FEATURE_NAMES, GraphEvents, _read_one, concatenate, event_hash, split_buckets_2_1_7` | [autonomous_allhad/gnn_lowdm/data.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/data.py>) |
| [37](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/analyze_diagonal_v3_global_shap.py:37>): `module` | `from ..model import PhysicsInformedJetGraphClassifier` | [autonomous_allhad/gnn_lowdm/model.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/model.py>) |

#### [autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [4](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py:4>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [6](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py:6>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py:7>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py:8>): `module` | `import os` | 표준 라이브러리 / 외부 설치 패키지 |
| [9](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py:9>): `module` | `import time` | 표준 라이브러리 / 외부 설치 패키지 |
| [10](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py:10>): `module` | `import traceback` | 표준 라이브러리 / 외부 설치 패키지 |
| [11](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py:11>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [12](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py:12>): `module` | `from typing import Any` | 표준 라이브러리 / 외부 설치 패키지 |
| [14](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py:14>): `module` | `import awkward as ak` | 표준 라이브러리 / 외부 설치 패키지 |
| [15](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py:15>): `module` | `import numpy as np` | 표준 라이브러리 / 외부 설치 패키지 |
| [16](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py:16>): `module` | `import uproot` | 표준 라이브러리 / 외부 설치 패키지 |
| [19](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py:19>): `module` | `from . import region_io as base` | [autonomous_allhad/gnn_lowdm/_implementation/region_io.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py>) |
| [20](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py:20>): `module` | `from ..data import split_buckets_2_1_7` | [autonomous_allhad/gnn_lowdm/data.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/data.py>) |
| [21](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py:21>): `module` | `from .diagonal_v3_region_features import feature_arrays` | [autonomous_allhad/gnn_lowdm/_implementation/diagonal_v3_region_features.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/diagonal_v3_region_features.py>) |
| [22](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py:22>): `module` | `from .rank005_numpy import Rank005Numpy` | [autonomous_allhad/gnn_lowdm/_implementation/rank005_numpy.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/rank005_numpy.py>) |
| [24](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py:24>): `module` | `import region_io as base` | [autonomous_allhad/gnn_lowdm/_implementation/region_io.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py>) |
| [25](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py:25>): `module` | `from data import split_buckets_2_1_7` | [autonomous_allhad/gnn_lowdm/data.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/data.py>) |
| [26](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py:26>): `module` | `from diagonal_v3_region_features import feature_arrays` | [autonomous_allhad/gnn_lowdm/_implementation/diagonal_v3_region_features.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/diagonal_v3_region_features.py>) |
| [27](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py:27>): `module` | `from rank005_numpy import Rank005Numpy` | [autonomous_allhad/gnn_lowdm/_implementation/rank005_numpy.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/rank005_numpy.py>) |
| [165](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py:165>): `process_source` | `from build_flat_boosted_recoil_hists import WEIGHT_BRANCHES` | [autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py>) |

#### [autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [10](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py:10>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [12](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py:12>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [13](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py:13>): `module` | `import concurrent.futures` | 표준 라이브러리 / 외부 설치 패키지 |
| [14](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py:14>): `module` | `import datetime as dt` | 표준 라이브러리 / 외부 설치 패키지 |
| [15](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py:15>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [16](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py:16>): `module` | `import os` | 표준 라이브러리 / 외부 설치 패키지 |
| [17](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py:17>): `module` | `import subprocess` | 표준 라이브러리 / 외부 설치 패키지 |
| [18](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py:18>): `module` | `import sys` | 표준 라이브러리 / 외부 설치 패키지 |
| [19](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py:19>): `module` | `import time` | 표준 라이브러리 / 외부 설치 패키지 |
| [20](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py:20>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [21](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py:21>): `module` | `from typing import Any` | 표준 라이브러리 / 외부 설치 패키지 |
| [23](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py:23>): `module` | `import numpy as np` | 표준 라이브러리 / 외부 설치 패키지 |
| [26](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py:26>): `module` | `from autonomous_allhad.autonomous_allhad.highdm_resolved_categories import boosted_overlap_vetoed_ak4_indices, map_candidates_to_events, map_candidates_to_events_rle, select_exclusive_resolved_candidates` | [autonomous_allhad/autonomous_allhad/highdm_resolved_categories.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/highdm_resolved_categories.py>) |
| [33](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py:33>): `module` | `from highdm_resolved_categories import boosted_overlap_vetoed_ak4_indices, map_candidates_to_events, map_candidates_to_events_rle, select_exclusive_resolved_candidates` | [autonomous_allhad/autonomous_allhad/highdm_resolved_categories.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/highdm_resolved_categories.py>) |
| [198](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py:198>): `compute_expanded_nres` | `import awkward as ak` | 표준 라이브러리 / 외부 설치 패키지 |
| [331](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py:331>): `process_source` | `import awkward as ak` | 표준 라이브러리 / 외부 설치 패키지 |
| [332](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py:332>): `process_source` | `import uproot` | 표준 라이브러리 / 외부 설치 패키지 |
| [456](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py:456>): `worker` | `import awkward as ak` | 표준 라이브러리 / 외부 설치 패키지 |
| [457](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py:457>): `worker` | `import uproot` | 표준 라이브러리 / 외부 설치 패키지 |

#### [autonomous_allhad/gnn_lowdm/_implementation/build_full_campaign_manifest.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_full_campaign_manifest.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [1](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_full_campaign_manifest.py:1>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [3](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_full_campaign_manifest.py:3>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [4](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_full_campaign_manifest.py:4>): `module` | `import hashlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [5](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_full_campaign_manifest.py:5>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [6](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_full_campaign_manifest.py:6>): `module` | `import re` | 표준 라이브러리 / 외부 설치 패키지 |
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_full_campaign_manifest.py:7>): `module` | `from collections import Counter, defaultdict` | 표준 라이브러리 / 외부 설치 패키지 |
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_full_campaign_manifest.py:8>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |

#### [autonomous_allhad/gnn_lowdm/_implementation/cli.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/cli.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [3](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/cli.py:3>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [5](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/cli.py:5>): `module` | `import importlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [6](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/cli.py:6>): `module` | `import sys` | 표준 라이브러리 / 외부 설치 패키지 |
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/cli.py:7>): `module` | `from collections.abc import Mapping` | 표준 라이브러리 / 외부 설치 패키지 |
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/cli.py:8>): `module` | `from typing import TypeAlias` | 표준 라이브러리 / 외부 설치 패키지 |

#### [autonomous_allhad/gnn_lowdm/_implementation/collect_plot_diagonal_v3_limits.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/collect_plot_diagonal_v3_limits.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [4](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/collect_plot_diagonal_v3_limits.py:4>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [6](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/collect_plot_diagonal_v3_limits.py:6>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/collect_plot_diagonal_v3_limits.py:7>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/collect_plot_diagonal_v3_limits.py:8>): `module` | `import sys` | 표준 라이브러리 / 외부 설치 패키지 |
| [9](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/collect_plot_diagonal_v3_limits.py:9>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [17](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/collect_plot_diagonal_v3_limits.py:17>): `module` | `from postprocess_limits import DECAY_LABELS, collect_limits, plot_contour, write_json` | [autonomous_allhad/workflow/postprocess_limits.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py>) |

#### [autonomous_allhad/gnn_lowdm/_implementation/collect_plot_highdm79minus6_lowdm30.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/collect_plot_highdm79minus6_lowdm30.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [4](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/collect_plot_highdm79minus6_lowdm30.py:4>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [6](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/collect_plot_highdm79minus6_lowdm30.py:6>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/collect_plot_highdm79minus6_lowdm30.py:7>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/collect_plot_highdm79minus6_lowdm30.py:8>): `module` | `import sys` | 표준 라이브러리 / 외부 설치 패키지 |
| [9](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/collect_plot_highdm79minus6_lowdm30.py:9>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [17](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/collect_plot_highdm79minus6_lowdm30.py:17>): `module` | `from postprocess_limits import DECAY_LABELS, collect_limits, plot_contour` | [autonomous_allhad/workflow/postprocess_limits.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py>) |

#### [autonomous_allhad/gnn_lowdm/_implementation/compare_oof.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/compare_oof.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [1](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/compare_oof.py:1>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [3](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/compare_oof.py:3>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [4](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/compare_oof.py:4>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [5](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/compare_oof.py:5>): `module` | `import math` | 표준 라이브러리 / 외부 설치 패키지 |
| [6](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/compare_oof.py:6>): `module` | `import time` | 표준 라이브러리 / 외부 설치 패키지 |
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/compare_oof.py:7>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/compare_oof.py:8>): `module` | `from typing import Callable` | 표준 라이브러리 / 외부 설치 패키지 |
| [10](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/compare_oof.py:10>): `module` | `import matplotlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [11](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/compare_oof.py:11>): `module` | `import numpy as np` | 표준 라이브러리 / 외부 설치 패키지 |
| [12](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/compare_oof.py:12>): `module` | `import torch` | 표준 라이브러리 / 외부 설치 패키지 |
| [13](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/compare_oof.py:13>): `module` | `import uproot` | 표준 라이브러리 / 외부 설치 패키지 |
| [14](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/compare_oof.py:14>): `module` | `from torch import nn` | 표준 라이브러리 / 외부 설치 패키지 |
| [15](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/compare_oof.py:15>): `module` | `from torch.utils.data import DataLoader` | 표준 라이브러리 / 외부 설치 패키지 |
| [18](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/compare_oof.py:18>): `module` | `import matplotlib.pyplot as plt` | 표준 라이브러리 / 외부 설치 패키지 |
| [21](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/compare_oof.py:21>): `module` | `from ..data import GraphEvents, load_graph_events` | [autonomous_allhad/gnn_lowdm/data.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/data.py>) |
| [22](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/compare_oof.py:22>): `module` | `from ..model import FlattenDNNClassifier, GlobalOnlyClassifier, JetGraphClassifier, JetTransformerClassifier` | [autonomous_allhad/gnn_lowdm/model.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/model.py>) |
| [28](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/compare_oof.py:28>): `module` | `from .train_oof import binary_auc, predict, roc_points, score_bin_table, seed_everything, sha256, tensors` | [autonomous_allhad/gnn_lowdm/_implementation/train_oof.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_oof.py>) |
| [38](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/compare_oof.py:38>): `module` | `from data import GraphEvents, load_graph_events` | [autonomous_allhad/gnn_lowdm/data.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/data.py>) |
| [39](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/compare_oof.py:39>): `module` | `from model import FlattenDNNClassifier, GlobalOnlyClassifier, JetGraphClassifier, JetTransformerClassifier` | [autonomous_allhad/gnn_lowdm/model.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/model.py>) |
| [45](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/compare_oof.py:45>): `module` | `from train_oof import binary_auc, predict, roc_points, score_bin_table, seed_everything, sha256, tensors` | [autonomous_allhad/gnn_lowdm/_implementation/train_oof.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_oof.py>) |

#### [autonomous_allhad/gnn_lowdm/_implementation/diagonal_v3_region_features.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/diagonal_v3_region_features.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [9](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/diagonal_v3_region_features.py:9>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [11](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/diagonal_v3_region_features.py:11>): `module` | `from typing import Any` | 표준 라이브러리 / 외부 설치 패키지 |
| [13](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/diagonal_v3_region_features.py:13>): `module` | `import awkward as ak` | 표준 라이브러리 / 외부 설치 패키지 |
| [14](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/diagonal_v3_region_features.py:14>): `module` | `import numpy as np` | 표준 라이브러리 / 외부 설치 패키지 |
| [17](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/diagonal_v3_region_features.py:17>): `module` | `from . import region_io as base` | [autonomous_allhad/gnn_lowdm/_implementation/region_io.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py>) |
| [18](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/diagonal_v3_region_features.py:18>): `module` | `from ..data import DIAGONAL_V3_GLOBAL_FEATURE_NAMES, _diagonal_v3_features, _pad` | [autonomous_allhad/gnn_lowdm/data.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/data.py>) |
| [24](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/diagonal_v3_region_features.py:24>): `module` | `import region_io as base` | [autonomous_allhad/gnn_lowdm/_implementation/region_io.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py>) |
| [25](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/diagonal_v3_region_features.py:25>): `module` | `from data import DIAGONAL_V3_GLOBAL_FEATURE_NAMES, _diagonal_v3_features, _pad` | [autonomous_allhad/gnn_lowdm/data.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/data.py>) |

#### [autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [10](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py:10>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [12](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py:12>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [13](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py:13>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [14](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py:14>): `module` | `import os` | 표준 라이브러리 / 외부 설치 패키지 |
| [15](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py:15>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [16](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py:16>): `module` | `from typing import Any` | 표준 라이브러리 / 외부 설치 패키지 |
| [18](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py:18>): `module` | `import numpy as np` | 표준 라이브러리 / 외부 설치 패키지 |
| [21](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py:21>): `module` | `from ..data import DIAGONAL_V3_GLOBAL_FEATURE_NAMES, _read_one, split_buckets_2_1_7` | [autonomous_allhad/gnn_lowdm/data.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/data.py>) |
| [26](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py:26>): `module` | `from .rank005_numpy import Rank005Numpy` | [autonomous_allhad/gnn_lowdm/_implementation/rank005_numpy.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/rank005_numpy.py>) |
| [28](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py:28>): `module` | `from data import DIAGONAL_V3_GLOBAL_FEATURE_NAMES, _read_one, split_buckets_2_1_7` | [autonomous_allhad/gnn_lowdm/data.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/data.py>) |
| [33](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py:33>): `module` | `from rank005_numpy import Rank005Numpy` | [autonomous_allhad/gnn_lowdm/_implementation/rank005_numpy.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/rank005_numpy.py>) |
| [186](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py:186>): `main` | `import torch` | 표준 라이브러리 / 외부 설치 패키지 |
| [188](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py:188>): `main` | `from ..model import PhysicsInformedJetGraphClassifier` | [autonomous_allhad/gnn_lowdm/model.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/model.py>) |
| [189](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py:189>): `main` | `from .train_oof import predict, tensors` | [autonomous_allhad/gnn_lowdm/_implementation/train_oof.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_oof.py>) |
| [191](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py:191>): `main` | `from model import PhysicsInformedJetGraphClassifier` | [autonomous_allhad/gnn_lowdm/model.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/model.py>) |
| [192](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py:192>): `main` | `from train_oof import predict, tensors` | [autonomous_allhad/gnn_lowdm/_implementation/train_oof.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_oof.py>) |

#### [autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [4](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py:4>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [6](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py:6>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py:7>): `module` | `import hashlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py:8>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [9](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py:9>): `module` | `from datetime import datetime, timezone` | 표준 라이브러리 / 외부 설치 패키지 |
| [10](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py:10>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [11](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py:11>): `module` | `from typing import Any` | 표준 라이브러리 / 외부 설치 패키지 |
| [13](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py:13>): `module` | `import matplotlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [14](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py:14>): `module` | `import numpy as np` | 표준 라이브러리 / 외부 설치 패키지 |
| [15](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py:15>): `module` | `import torch` | 표준 라이브러리 / 외부 설치 패키지 |
| [16](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py:16>): `module` | `import uproot` | 표준 라이브러리 / 외부 설치 패키지 |
| [19](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py:19>): `module` | `import matplotlib.pyplot as plt` | 표준 라이브러리 / 외부 설치 패키지 |
| [20](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py:20>): `module` | `import mplhep as hep` | 표준 라이브러리 / 외부 설치 패키지 |
| [21](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py:21>): `module` | `from matplotlib.lines import Line2D` | 표준 라이브러리 / 외부 설치 패키지 |
| [23](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py:23>): `module` | `from ..model import PhysicsInformedJetGraphClassifier` | [autonomous_allhad/gnn_lowdm/model.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/model.py>) |
| [24](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py:24>): `module` | `from .plot_full_gnn_results import PROCESS_COLORS, PROCESS_LABELS, PROCESS_ORDER, grouped_process` | [autonomous_allhad/gnn_lowdm/_implementation/plot_full_gnn_results.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_full_gnn_results.py>) |
| [30](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py:30>): `module` | `from .plot_lowdm_category_nn_out import signed_stack` | [autonomous_allhad/gnn_lowdm/_implementation/plot_lowdm_category_nn_out.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_lowdm_category_nn_out.py>) |
| [31](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py:31>): `module` | `from .significance import CATEGORY_NAMES, diagonal_v3_category_ids, evaluate_binning` | [autonomous_allhad/gnn_lowdm/_implementation/significance.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/significance.py>) |
| [32](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py:32>): `module` | `from .train_oof import predict, tensors` | [autonomous_allhad/gnn_lowdm/_implementation/train_oof.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_oof.py>) |
| [33](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py:33>): `module` | `from .tune_full_gnn import background_process_names, load_inputs, physics_weights, validation_metrics` | [autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py>) |
| [157](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py:157>): `plot_templates` | `import mplhep as hep` | 표준 라이브러리 / 외부 설치 패키지 |

#### [autonomous_allhad/gnn_lowdm/_implementation/finalize_diagonal_v3_cache.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/finalize_diagonal_v3_cache.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [4](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/finalize_diagonal_v3_cache.py:4>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [6](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/finalize_diagonal_v3_cache.py:6>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/finalize_diagonal_v3_cache.py:7>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/finalize_diagonal_v3_cache.py:8>): `module` | `import time` | 표준 라이브러리 / 외부 설치 패키지 |
| [9](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/finalize_diagonal_v3_cache.py:9>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [10](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/finalize_diagonal_v3_cache.py:10>): `module` | `from typing import Any` | 표준 라이브러리 / 외부 설치 패키지 |
| [12](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/finalize_diagonal_v3_cache.py:12>): `module` | `import uproot` | 표준 라이브러리 / 외부 설치 패키지 |

#### [autonomous_allhad/gnn_lowdm/_implementation/merge_all_signal_templates.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_all_signal_templates.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [4](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_all_signal_templates.py:4>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [6](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_all_signal_templates.py:6>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_all_signal_templates.py:7>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_all_signal_templates.py:8>): `module` | `import math` | 표준 라이브러리 / 외부 설치 패키지 |
| [9](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_all_signal_templates.py:9>): `module` | `import shutil` | 표준 라이브러리 / 외부 설치 패키지 |
| [10](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_all_signal_templates.py:10>): `module` | `from array import array` | 표준 라이브러리 / 외부 설치 패키지 |
| [11](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_all_signal_templates.py:11>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [12](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_all_signal_templates.py:12>): `module` | `from typing import Any` | 표준 라이브러리 / 외부 설치 패키지 |
| [14](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_all_signal_templates.py:14>): `module` | `import numpy as np` | 표준 라이브러리 / 외부 설치 패키지 |
| [15](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_all_signal_templates.py:15>): `module` | `import ROOT` | 표준 라이브러리 / 외부 설치 패키지 |
| [18](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_all_signal_templates.py:18>): `module` | `from .build_datacard import write_cards` | 해당 상대 경로 없음 |
| [20](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_all_signal_templates.py:20>): `module` | `from build_datacard import write_cards` | [autonomous_allhad/gnn_lowdm/build_datacard.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/build_datacard.py>) |

#### [autonomous_allhad/gnn_lowdm/_implementation/merge_diagonal_v3_cr_nnout_partials.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_diagonal_v3_cr_nnout_partials.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [4](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_diagonal_v3_cr_nnout_partials.py:4>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [6](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_diagonal_v3_cr_nnout_partials.py:6>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_diagonal_v3_cr_nnout_partials.py:7>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_diagonal_v3_cr_nnout_partials.py:8>): `module` | `import os` | 표준 라이브러리 / 외부 설치 패키지 |
| [9](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_diagonal_v3_cr_nnout_partials.py:9>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [10](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_diagonal_v3_cr_nnout_partials.py:10>): `module` | `from typing import Any` | 표준 라이브러리 / 외부 설치 패키지 |
| [13](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_diagonal_v3_cr_nnout_partials.py:13>): `module` | `from .build_diagonal_v3_cr_nnout_partial import merge as merge_histograms` | [autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py>) |
| [15](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_diagonal_v3_cr_nnout_partials.py:15>): `module` | `from build_diagonal_v3_cr_nnout_partial import merge as merge_histograms` | [autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py>) |

#### [autonomous_allhad/gnn_lowdm/_implementation/physics_informed_v2.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/physics_informed_v2.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/physics_informed_v2.py:8>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [10](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/physics_informed_v2.py:10>): `module` | `import numpy as np` | 표준 라이브러리 / 외부 설치 패키지 |
| [12](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/physics_informed_v2.py:12>): `module` | `from ..data import ENGINEERED_EXPANDED_GLOBAL_FEATURE_NAMES, GraphEvents` | [autonomous_allhad/gnn_lowdm/data.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/data.py>) |

#### [autonomous_allhad/gnn_lowdm/_implementation/plot_control_regions.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_control_regions.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [4](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_control_regions.py:4>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [6](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_control_regions.py:6>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_control_regions.py:7>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_control_regions.py:8>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [10](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_control_regions.py:10>): `module` | `import matplotlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [11](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_control_regions.py:11>): `module` | `import numpy as np` | 표준 라이브러리 / 외부 설치 패키지 |
| [15](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_control_regions.py:15>): `module` | `import matplotlib.pyplot as plt` | 표준 라이브러리 / 외부 설치 패키지 |
| [134](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_control_regions.py:134>): `draw` | `import mplhep as hep` | 표준 라이브러리 / 외부 설치 패키지 |

#### [autonomous_allhad/gnn_lowdm/_implementation/plot_diagonal_v3_model_schematic.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_diagonal_v3_model_schematic.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_diagonal_v3_model_schematic.py:8>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [10](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_diagonal_v3_model_schematic.py:10>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [11](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_diagonal_v3_model_schematic.py:11>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [13](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_diagonal_v3_model_schematic.py:13>): `module` | `import matplotlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [16](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_diagonal_v3_model_schematic.py:16>): `module` | `import matplotlib.pyplot as plt` | 표준 라이브러리 / 외부 설치 패키지 |
| [17](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_diagonal_v3_model_schematic.py:17>): `module` | `from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle` | 표준 라이브러리 / 외부 설치 패키지 |

#### [autonomous_allhad/gnn_lowdm/_implementation/plot_full_gnn_results.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_full_gnn_results.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [1](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_full_gnn_results.py:1>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [3](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_full_gnn_results.py:3>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [4](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_full_gnn_results.py:4>): `module` | `import hashlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [5](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_full_gnn_results.py:5>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [6](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_full_gnn_results.py:6>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_full_gnn_results.py:8>): `module` | `import matplotlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [9](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_full_gnn_results.py:9>): `module` | `import numpy as np` | 표준 라이브러리 / 외부 설치 패키지 |
| [10](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_full_gnn_results.py:10>): `module` | `import uproot` | 표준 라이브러리 / 외부 설치 패키지 |
| [13](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_full_gnn_results.py:13>): `module` | `import matplotlib.pyplot as plt` | 표준 라이브러리 / 외부 설치 패키지 |
| [15](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_full_gnn_results.py:15>): `module` | `from .plot_lowdm_category_nn_out import CATEGORIES, signed_stack` | [autonomous_allhad/gnn_lowdm/_implementation/plot_lowdm_category_nn_out.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_lowdm_category_nn_out.py>) |
| [185](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_full_gnn_results.py:185>): `main` | `import mplhep as hep` | 표준 라이브러리 / 외부 설치 패키지 |

#### [autonomous_allhad/gnn_lowdm/_implementation/plot_hyperparameter_losses.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_hyperparameter_losses.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [4](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_hyperparameter_losses.py:4>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [6](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_hyperparameter_losses.py:6>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_hyperparameter_losses.py:7>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_hyperparameter_losses.py:8>): `module` | `import time` | 표준 라이브러리 / 외부 설치 패키지 |
| [9](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_hyperparameter_losses.py:9>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [11](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_hyperparameter_losses.py:11>): `module` | `import matplotlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [14](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_hyperparameter_losses.py:14>): `module` | `import matplotlib.pyplot as plt` | 표준 라이브러리 / 외부 설치 패키지 |
| [15](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_hyperparameter_losses.py:15>): `module` | `import mplhep as hep` | 표준 라이브러리 / 외부 설치 패키지 |
| [16](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_hyperparameter_losses.py:16>): `module` | `from matplotlib.lines import Line2D` | 표준 라이브러리 / 외부 설치 패키지 |

#### [autonomous_allhad/gnn_lowdm/_implementation/plot_lowdm_category_nn_out.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_lowdm_category_nn_out.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [1](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_lowdm_category_nn_out.py:1>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [3](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_lowdm_category_nn_out.py:3>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [4](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_lowdm_category_nn_out.py:4>): `module` | `import hashlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [5](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_lowdm_category_nn_out.py:5>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [6](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_lowdm_category_nn_out.py:6>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_lowdm_category_nn_out.py:8>): `module` | `import matplotlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [9](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_lowdm_category_nn_out.py:9>): `module` | `import numpy as np` | 표준 라이브러리 / 외부 설치 패키지 |
| [10](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_lowdm_category_nn_out.py:10>): `module` | `import uproot` | 표준 라이브러리 / 외부 설치 패키지 |
| [13](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_lowdm_category_nn_out.py:13>): `module` | `import matplotlib.pyplot as plt` | 표준 라이브러리 / 외부 설치 패키지 |
| [177](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_lowdm_category_nn_out.py:177>): `main` | `import mplhep as hep` | 표준 라이브러리 / 외부 설치 패키지 |

#### [autonomous_allhad/gnn_lowdm/_implementation/prepare_all_signal_template_condor.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_all_signal_template_condor.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [4](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_all_signal_template_condor.py:4>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [6](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_all_signal_template_condor.py:6>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_all_signal_template_condor.py:7>): `module` | `import hashlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_all_signal_template_condor.py:8>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [9](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_all_signal_template_condor.py:9>): `module` | `import shutil` | 표준 라이브러리 / 외부 설치 패키지 |
| [10](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_all_signal_template_condor.py:10>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |

#### [autonomous_allhad/gnn_lowdm/_implementation/prepare_diagonal_v3_cache_condor.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_diagonal_v3_cache_condor.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [4](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_diagonal_v3_cache_condor.py:4>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [6](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_diagonal_v3_cache_condor.py:6>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_diagonal_v3_cache_condor.py:7>): `module` | `import hashlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_diagonal_v3_cache_condor.py:8>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [9](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_diagonal_v3_cache_condor.py:9>): `module` | `import shutil` | 표준 라이브러리 / 외부 설치 패키지 |
| [10](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_diagonal_v3_cache_condor.py:10>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [11](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_diagonal_v3_cache_condor.py:11>): `module` | `from typing import Any` | 표준 라이브러리 / 외부 설치 패키지 |

#### [autonomous_allhad/gnn_lowdm/_implementation/prepare_highdm79minus6_lowdm30_condor.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_highdm79minus6_lowdm30_condor.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [9](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_highdm79minus6_lowdm30_condor.py:9>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [11](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_highdm79minus6_lowdm30_condor.py:11>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [12](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_highdm79minus6_lowdm30_condor.py:12>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [13](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_highdm79minus6_lowdm30_condor.py:13>): `module` | `import shutil` | 표준 라이브러리 / 외부 설치 패키지 |
| [14](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_highdm79minus6_lowdm30_condor.py:14>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |

#### [autonomous_allhad/gnn_lowdm/_implementation/publish_lowdm_gnn_web.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/publish_lowdm_gnn_web.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [4](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/publish_lowdm_gnn_web.py:4>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [6](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/publish_lowdm_gnn_web.py:6>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/publish_lowdm_gnn_web.py:7>): `module` | `import hashlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/publish_lowdm_gnn_web.py:8>): `module` | `import html` | 표준 라이브러리 / 외부 설치 패키지 |
| [9](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/publish_lowdm_gnn_web.py:9>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [10](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/publish_lowdm_gnn_web.py:10>): `module` | `import os` | 표준 라이브러리 / 외부 설치 패키지 |
| [11](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/publish_lowdm_gnn_web.py:11>): `module` | `import shutil` | 표준 라이브러리 / 외부 설치 패키지 |
| [12](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/publish_lowdm_gnn_web.py:12>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [13](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/publish_lowdm_gnn_web.py:13>): `module` | `from typing import Any` | 표준 라이브러리 / 외부 설치 패키지 |

#### [autonomous_allhad/gnn_lowdm/_implementation/rank005_numpy.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/rank005_numpy.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/rank005_numpy.py:8>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [10](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/rank005_numpy.py:10>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [12](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/rank005_numpy.py:12>): `module` | `import numpy as np` | 표준 라이브러리 / 외부 설치 패키지 |

#### [autonomous_allhad/gnn_lowdm/_implementation/region_io.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:8>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [10](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:10>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [11](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:11>): `module` | `import gzip` | 표준 라이브러리 / 외부 설치 패키지 |
| [12](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:12>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [13](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:13>): `module` | `import math` | 표준 라이브러리 / 외부 설치 패키지 |
| [14](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:14>): `module` | `import os` | 표준 라이브러리 / 외부 설치 패키지 |
| [15](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:15>): `module` | `import sys` | 표준 라이브러리 / 외부 설치 패키지 |
| [16](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:16>): `module` | `import time` | 표준 라이브러리 / 외부 설치 패키지 |
| [17](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:17>): `module` | `import traceback` | 표준 라이브러리 / 외부 설치 패키지 |
| [18](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:18>): `module` | `from dataclasses import dataclass` | 표준 라이브러리 / 외부 설치 패키지 |
| [19](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:19>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [20](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:20>): `module` | `from typing import Any, Iterable` | 표준 라이브러리 / 외부 설치 패키지 |
| [22](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:22>): `module` | `import awkward as ak` | 표준 라이브러리 / 외부 설치 패키지 |
| [23](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:23>): `module` | `import numpy as np` | 표준 라이브러리 / 외부 설치 패키지 |
| [24](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:24>): `module` | `import uproot` | 표준 라이브러리 / 외부 설치 패키지 |
| [35](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:35>): `module` | `from autonomous_allhad.autonomous_allhad.highdm_resolved_categories import boosted_overlap_vetoed_ak4_indices, map_candidates_to_events, map_candidates_to_events_rle, select_exclusive_resolved_candidates` | [autonomous_allhad/autonomous_allhad/highdm_resolved_categories.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/highdm_resolved_categories.py>) |
| [41](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:41>): `module` | `from autonomous_allhad.autonomous_allhad.signal_models import SIGNAL_TOPOLOGY_IDS, signal_topology` | [autonomous_allhad/autonomous_allhad/signal_models.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/signal_models.py>) |
| [46](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:46>): `module` | `from autonomous_allhad.highdm_resolved_categories import boosted_overlap_vetoed_ak4_indices, map_candidates_to_events, map_candidates_to_events_rle, select_exclusive_resolved_candidates` | [autonomous_allhad/autonomous_allhad/highdm_resolved_categories.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/highdm_resolved_categories.py>) |
| [52](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:52>): `module` | `from autonomous_allhad.signal_models import SIGNAL_TOPOLOGY_IDS, signal_topology` | [autonomous_allhad/autonomous_allhad/signal_models.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/signal_models.py>) |
| [979](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:979>): `normalized_weight_variations` | `from build_flat_boosted_recoil_hists import flat_arrays_for_weights` | [autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py>) |
| [981](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:981>): `normalized_weight_variations` | `from autonomous_allhad.autonomous_allhad.real_subset_worker import compute_weight_bundle` | [autonomous_allhad/autonomous_allhad/real_subset_worker.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py>) |
| [985](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:985>): `normalized_weight_variations` | `from autonomous_allhad.real_subset_worker import compute_weight_bundle` | [autonomous_allhad/autonomous_allhad/real_subset_worker.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py>) |
| [1001](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:1001>): `normalized_weight_variations` | `from build_flat_boosted_recoil_hists import signal_btag_efficiency_dataset` | [autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py>) |
| [1257](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:1257>): `process_source` | `from build_flat_boosted_recoil_hists import WEIGHT_BRANCHES` | [autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py>) |

#### [autonomous_allhad/gnn_lowdm/_implementation/significance.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/significance.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [3](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/significance.py:3>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [5](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/significance.py:5>): `module` | `import itertools` | 표준 라이브러리 / 외부 설치 패키지 |
| [6](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/significance.py:6>): `module` | `import math` | 표준 라이브러리 / 외부 설치 패키지 |
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/significance.py:7>): `module` | `from typing import Any, Iterable` | 표준 라이브러리 / 외부 설치 패키지 |
| [9](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/significance.py:9>): `module` | `import numpy as np` | 표준 라이브러리 / 외부 설치 패키지 |
| [10](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/significance.py:10>): `module` | `import torch` | 표준 라이브러리 / 외부 설치 패키지 |
| [12](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/significance.py:12>): `module` | `from ..data import DIAGONAL_V3_GLOBAL_FEATURE_NAMES, GraphEvents` | [autonomous_allhad/gnn_lowdm/data.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/data.py>) |

#### [autonomous_allhad/gnn_lowdm/_implementation/supplementary_plots.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/supplementary_plots.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [3](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/supplementary_plots.py:3>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [5](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/supplementary_plots.py:5>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [6](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/supplementary_plots.py:6>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/supplementary_plots.py:7>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [9](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/supplementary_plots.py:9>): `module` | `import matplotlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [12](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/supplementary_plots.py:12>): `module` | `import matplotlib.pyplot as plt` | 표준 라이브러리 / 외부 설치 패키지 |
| [13](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/supplementary_plots.py:13>): `module` | `import mplhep as hep` | 표준 라이브러리 / 외부 설치 패키지 |
| [14](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/supplementary_plots.py:14>): `module` | `import numpy as np` | 표준 라이브러리 / 외부 설치 패키지 |
| [16](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/supplementary_plots.py:16>): `module` | `from .analyze_diagonal_v3_global_shap import FEATURE_LABELS, display_feature_view, plot_beeswarm, plot_importance` | [autonomous_allhad/gnn_lowdm/_implementation/analyze_diagonal_v3_global_shap.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/analyze_diagonal_v3_global_shap.py>) |

#### [autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [4](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py:4>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [6](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py:6>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py:7>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py:8>): `module` | `import random` | 표준 라이브러리 / 외부 설치 패키지 |
| [9](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py:9>): `module` | `import time` | 표준 라이브러리 / 외부 설치 패키지 |
| [10](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py:10>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [11](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py:11>): `module` | `from typing import Any` | 표준 라이브러리 / 외부 설치 패키지 |
| [13](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py:13>): `module` | `import matplotlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [14](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py:14>): `module` | `import numpy as np` | 표준 라이브러리 / 외부 설치 패키지 |
| [15](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py:15>): `module` | `import torch` | 표준 라이브러리 / 외부 설치 패키지 |
| [16](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py:16>): `module` | `import uproot` | 표준 라이브러리 / 외부 설치 패키지 |
| [17](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py:17>): `module` | `from torch import nn` | 표준 라이브러리 / 외부 설치 패키지 |
| [18](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py:18>): `module` | `from torch.utils.data import DataLoader, TensorDataset` | 표준 라이브러리 / 외부 설치 패키지 |
| [21](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py:21>): `module` | `import matplotlib.pyplot as plt` | 표준 라이브러리 / 외부 설치 패키지 |
| [22](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py:22>): `module` | `import mplhep as hep` | 표준 라이브러리 / 외부 설치 패키지 |
| [24](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py:24>): `module` | `from ..data import DIAGONAL_V3_GLOBAL_FEATURE_NAMES` | [autonomous_allhad/gnn_lowdm/data.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/data.py>) |
| [25](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py:25>): `module` | `from ..model import PhysicsInformedJetGraphClassifier` | [autonomous_allhad/gnn_lowdm/model.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/model.py>) |
| [26](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py:26>): `module` | `from .physics_informed_v2 import physics_informed_loss_weights` | [autonomous_allhad/gnn_lowdm/_implementation/physics_informed_v2.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/physics_informed_v2.py>) |
| [27](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py:27>): `module` | `from .plot_hyperparameter_losses import save_plot as save_hyperparameter_plot` | [autonomous_allhad/gnn_lowdm/_implementation/plot_hyperparameter_losses.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_hyperparameter_losses.py>) |
| [28](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py:28>): `module` | `from .significance import binning_selection_key, diagonal_v3_category_ids, evaluate_binning, optimize_validation_binning, soft_s_over_sqrt_b_loss, weighted_quantiles` | [autonomous_allhad/gnn_lowdm/_implementation/significance.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/significance.py>) |
| [36](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py:36>): `module` | `from .train_oof import seed_everything, sha256, tensors` | [autonomous_allhad/gnn_lowdm/_implementation/train_oof.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_oof.py>) |
| [37](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py:37>): `module` | `from .tune_full_gnn import background_process_names, load_inputs, physics_weights, validation_metrics, write_json` | [autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py>) |

#### [autonomous_allhad/gnn_lowdm/_implementation/train_oof.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_oof.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [1](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_oof.py:1>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [3](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_oof.py:3>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [4](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_oof.py:4>): `module` | `import hashlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [5](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_oof.py:5>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [6](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_oof.py:6>): `module` | `import math` | 표준 라이브러리 / 외부 설치 패키지 |
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_oof.py:7>): `module` | `import random` | 표준 라이브러리 / 외부 설치 패키지 |
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_oof.py:8>): `module` | `import time` | 표준 라이브러리 / 외부 설치 패키지 |
| [9](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_oof.py:9>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [11](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_oof.py:11>): `module` | `import matplotlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [12](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_oof.py:12>): `module` | `import numpy as np` | 표준 라이브러리 / 외부 설치 패키지 |
| [13](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_oof.py:13>): `module` | `import torch` | 표준 라이브러리 / 외부 설치 패키지 |
| [14](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_oof.py:14>): `module` | `import uproot` | 표준 라이브러리 / 외부 설치 패키지 |
| [15](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_oof.py:15>): `module` | `from torch import nn` | 표준 라이브러리 / 외부 설치 패키지 |
| [16](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_oof.py:16>): `module` | `from torch.utils.data import DataLoader, TensorDataset` | 표준 라이브러리 / 외부 설치 패키지 |
| [19](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_oof.py:19>): `module` | `import matplotlib.pyplot as plt` | 표준 라이브러리 / 외부 설치 패키지 |
| [22](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_oof.py:22>): `module` | `from ..data import GraphEvents, load_graph_events` | [autonomous_allhad/gnn_lowdm/data.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/data.py>) |
| [23](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_oof.py:23>): `module` | `from ..model import GlobalOnlyClassifier, JetGraphClassifier` | [autonomous_allhad/gnn_lowdm/model.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/model.py>) |
| [25](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_oof.py:25>): `module` | `from data import GraphEvents, load_graph_events` | [autonomous_allhad/gnn_lowdm/data.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/data.py>) |
| [26](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_oof.py:26>): `module` | `from model import GlobalOnlyClassifier, JetGraphClassifier` | [autonomous_allhad/gnn_lowdm/model.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/model.py>) |

#### [autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [1](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py:1>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [3](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py:3>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [4](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py:4>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [5](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py:5>): `module` | `import math` | 표준 라이브러리 / 외부 설치 패키지 |
| [6](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py:6>): `module` | `import random` | 표준 라이브러리 / 외부 설치 패키지 |
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py:7>): `module` | `import time` | 표준 라이브러리 / 외부 설치 패키지 |
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py:8>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [10](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py:10>): `module` | `import numpy as np` | 표준 라이브러리 / 외부 설치 패키지 |
| [11](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py:11>): `module` | `import torch` | 표준 라이브러리 / 외부 설치 패키지 |
| [12](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py:12>): `module` | `import uproot` | 표준 라이브러리 / 외부 설치 패키지 |
| [13](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py:13>): `module` | `from torch import nn` | 표준 라이브러리 / 외부 설치 패키지 |
| [14](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py:14>): `module` | `from torch.utils.data import DataLoader, TensorDataset` | 표준 라이브러리 / 외부 설치 패키지 |
| [16](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py:16>): `module` | `from .compare_oof import weighted_binary_auc` | [autonomous_allhad/gnn_lowdm/_implementation/compare_oof.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/compare_oof.py>) |
| [17](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py:17>): `module` | `from ..data import BASE_GLOBAL_FEATURE_NAMES, TOP_TARGETED_GLOBAL_FEATURE_NAMES, GraphEvents, load_graph_events, split_buckets_2_1_7` | [autonomous_allhad/gnn_lowdm/data.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/data.py>) |
| [24](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py:24>): `module` | `from ..model import JetGraphClassifier` | [autonomous_allhad/gnn_lowdm/model.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/model.py>) |
| [25](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py:25>): `module` | `from .train_oof import predict, seed_everything, sha256, tensors` | [autonomous_allhad/gnn_lowdm/_implementation/train_oof.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_oof.py>) |

#### [autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_numpy_inference.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_numpy_inference.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [4](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_numpy_inference.py:4>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [6](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_numpy_inference.py:6>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_numpy_inference.py:7>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_numpy_inference.py:8>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [10](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_numpy_inference.py:10>): `module` | `import numpy as np` | 표준 라이브러리 / 외부 설치 패키지 |
| [11](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_numpy_inference.py:11>): `module` | `import torch` | 표준 라이브러리 / 외부 설치 패키지 |
| [13](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_numpy_inference.py:13>): `module` | `from ..data import _read_one` | [autonomous_allhad/gnn_lowdm/data.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/data.py>) |
| [14](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_numpy_inference.py:14>): `module` | `from ..model import PhysicsInformedJetGraphClassifier` | [autonomous_allhad/gnn_lowdm/model.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/model.py>) |
| [15](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_numpy_inference.py:15>): `module` | `from .rank005_numpy import Rank005Numpy` | [autonomous_allhad/gnn_lowdm/_implementation/rank005_numpy.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/rank005_numpy.py>) |

#### [autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_region_feature_parity.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_region_feature_parity.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [4](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_region_feature_parity.py:4>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [6](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_region_feature_parity.py:6>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_region_feature_parity.py:7>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_region_feature_parity.py:8>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [10](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_region_feature_parity.py:10>): `module` | `import numpy as np` | 표준 라이브러리 / 외부 설치 패키지 |
| [11](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_region_feature_parity.py:11>): `module` | `import uproot` | 표준 라이브러리 / 외부 설치 패키지 |
| [13](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_region_feature_parity.py:13>): `module` | `from . import region_io as base` | [autonomous_allhad/gnn_lowdm/_implementation/region_io.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py>) |
| [14](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_region_feature_parity.py:14>): `module` | `from ..data import DIAGONAL_V3_GLOBAL_FEATURE_NAMES, _read_one` | [autonomous_allhad/gnn_lowdm/data.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/data.py>) |
| [15](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_region_feature_parity.py:15>): `module` | `from .diagonal_v3_region_features import feature_arrays` | [autonomous_allhad/gnn_lowdm/_implementation/diagonal_v3_region_features.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/diagonal_v3_region_features.py>) |

#### [autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_srcr_inputs.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_srcr_inputs.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [4](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_srcr_inputs.py:4>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [6](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_srcr_inputs.py:6>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_srcr_inputs.py:7>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_srcr_inputs.py:8>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [9](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_srcr_inputs.py:9>): `module` | `from typing import Any` | 표준 라이브러리 / 외부 설치 패키지 |
| [11](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_srcr_inputs.py:11>): `module` | `import numpy as np` | 표준 라이브러리 / 외부 설치 패키지 |

#### [autonomous_allhad/gnn_lowdm/build_datacard.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/build_datacard.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [3](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/build_datacard.py:3>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [5](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/build_datacard.py:5>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [6](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/build_datacard.py:6>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/build_datacard.py:7>): `module` | `import re` | 표준 라이브러리 / 외부 설치 패키지 |
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/build_datacard.py:8>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [9](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/build_datacard.py:9>): `module` | `from typing import Any` | 표준 라이브러리 / 외부 설치 패키지 |
| [11](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/build_datacard.py:11>): `module` | `import numpy as np` | 표준 라이브러리 / 외부 설치 패키지 |
| [167](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/build_datacard.py:167>): `signals_from_root` | `import ROOT` | 표준 라이브러리 / 외부 설치 패키지 |

#### [autonomous_allhad/gnn_lowdm/data.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/data.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [1](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/data.py:1>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [3](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/data.py:3>): `module` | `from dataclasses import dataclass` | 표준 라이브러리 / 외부 설치 패키지 |
| [4](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/data.py:4>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [5](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/data.py:5>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [6](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/data.py:6>): `module` | `from typing import Iterable` | 표준 라이브러리 / 외부 설치 패키지 |
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/data.py:8>): `module` | `import awkward as ak` | 표준 라이브러리 / 외부 설치 패키지 |
| [9](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/data.py:9>): `module` | `import numpy as np` | 표준 라이브러리 / 외부 설치 패키지 |
| [10](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/data.py:10>): `module` | `import uproot` | 표준 라이브러리 / 외부 설치 패키지 |
| [539](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/data.py:539>): `_diagonal_v3_features` | `from mt2 import mt2 as compute_mt2` | 표준 라이브러리 / 외부 설치 패키지 |

#### [autonomous_allhad/gnn_lowdm/eval.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/eval.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [3](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/eval.py:3>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [5](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/eval.py:5>): `module` | `from ._implementation.cli import dispatch` | [autonomous_allhad/gnn_lowdm/_implementation/cli.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/cli.py>) |

#### [autonomous_allhad/gnn_lowdm/merge_datacard.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/merge_datacard.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [4](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/merge_datacard.py:4>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [6](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/merge_datacard.py:6>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/merge_datacard.py:7>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/merge_datacard.py:8>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [52](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/merge_datacard.py:52>): `copy_directory` | `import ROOT` | 표준 라이브러리 / 외부 설치 패키지 |
| [111](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/merge_datacard.py:111>): `main` | `import ROOT` | 표준 라이브러리 / 외부 설치 패키지 |

#### [autonomous_allhad/gnn_lowdm/model.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/model.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [1](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/model.py:1>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [3](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/model.py:3>): `module` | `import torch` | 표준 라이브러리 / 외부 설치 패키지 |
| [4](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/model.py:4>): `module` | `from torch import nn` | 표준 라이브러리 / 외부 설치 패키지 |

#### [autonomous_allhad/gnn_lowdm/plotting.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/plotting.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [3](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/plotting.py:3>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [5](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/plotting.py:5>): `module` | `from ._implementation.cli import dispatch` | [autonomous_allhad/gnn_lowdm/_implementation/cli.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/cli.py>) |

#### [autonomous_allhad/gnn_lowdm/train.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/train.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [3](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/train.py:3>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [5](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/train.py:5>): `module` | `from ._implementation.cli import dispatch` | [autonomous_allhad/gnn_lowdm/_implementation/cli.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/cli.py>) |

#### [autonomous_allhad/gnn_lowdm/valid.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/valid.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [3](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/valid.py:3>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [5](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/valid.py:5>): `module` | `from ._implementation.cli import dispatch` | [autonomous_allhad/gnn_lowdm/_implementation/cli.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/cli.py>) |

### A2. 직접 연결 공용 코드와 보정 원본

#### [autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [2](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py:2>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [4](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py:4>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [5](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py:5>): `module` | `import hashlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [6](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py:6>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py:7>): `module` | `import math` | 표준 라이브러리 / 외부 설치 패키지 |
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py:8>): `module` | `import os` | 표준 라이브러리 / 외부 설치 패키지 |
| [9](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py:9>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [10](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py:10>): `module` | `from typing import Any` | 표준 라이브러리 / 외부 설치 패키지 |
| [12](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py:12>): `module` | `import awkward as ak` | 표준 라이브러리 / 외부 설치 패키지 |
| [13](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py:13>): `module` | `import numpy as np` | 표준 라이브러리 / 외부 설치 패키지 |
| [14](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py:14>): `module` | `import uproot` | 표준 라이브러리 / 외부 설치 패키지 |
| [16](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py:16>): `module` | `from autonomous_allhad.signal_models import signal_mass_key, signal_topology` | [autonomous_allhad/autonomous_allhad/signal_models.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/signal_models.py>) |
| [20](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py:20>): `module` | `from autonomous_allhad.sidecar_store import read_root_metadata` | [autonomous_allhad/autonomous_allhad/sidecar_store.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/sidecar_store.py>) |
| [21](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py:21>): `module` | `from autonomous_allhad.analysis_scale_factors import REQUIRED_ANALYSIS_SF_COMPONENTS, REQUIRED_ANALYSIS_SF_VARIATIONS` | [autonomous_allhad/autonomous_allhad/analysis_scale_factors.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/analysis_scale_factors.py>) |
| [26](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py:26>): `module` | `from autonomous_allhad.real_subset_worker import assign_lowdm_search_bin, compute_weight_bundle` | [autonomous_allhad/autonomous_allhad/real_subset_worker.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py>) |
| [27](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py:27>): `module` | `from autonomous_allhad.dy_ptll_policy import dataset_id_prefilter_plan, dy_ptll_dataset_allowed` | [autonomous_allhad/autonomous_allhad/dy_ptll_policy.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/dy_ptll_policy.py>) |
| [28](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py:28>): `module` | `from autonomous_allhad.highdm_resolved_categories import boosted_overlap_vetoed_ak4_indices, map_candidates_to_events, map_candidates_to_events_rle, select_exclusive_resolved_candidates` | [autonomous_allhad/autonomous_allhad/highdm_resolved_categories.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/highdm_resolved_categories.py>) |
| [34](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py:34>): `module` | `from autonomous_allhad.search_bin_categorization import configured_exclusive_bin_count, configured_exclusive_labels, exclusive_category_source_indices, map60_indices_to_adopted55, map_category_sources_to_configured` | [autonomous_allhad/autonomous_allhad/search_bin_categorization.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/search_bin_categorization.py>) |
| [41](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py:41>): `module` | `from gnn_lowdm._implementation.region_io import SELECTION_BRANCHES as BROAD_LOWDM_SELECTION_BRANCHES, build_region_blocks as build_broad_lowdm_region_blocks` | [autonomous_allhad/gnn_lowdm/_implementation/region_io.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py>) |

#### [autonomous_allhad/workflow/postprocess_limits.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [4](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:4>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [6](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:6>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:7>): `module` | `import hashlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:8>): `module` | `from importlib import metadata` | 표준 라이브러리 / 외부 설치 패키지 |
| [9](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:9>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [10](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:10>): `module` | `import math` | 표준 라이브러리 / 외부 설치 패키지 |
| [11](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:11>): `module` | `import re` | 표준 라이브러리 / 외부 설치 패키지 |
| [12](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:12>): `module` | `import sys` | 표준 라이브러리 / 외부 설치 패키지 |
| [13](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:13>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [14](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:14>): `module` | `from typing import Any` | 표준 라이브러리 / 외부 설치 패키지 |
| [16](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:16>): `module` | `import numpy as np` | 표준 라이브러리 / 외부 설치 패키지 |
| [331](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:331>): `parse_limit_file` | `import uproot` | 표준 라이브러리 / 외부 설치 패키지 |
| [346](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:346>): `parse_limit_file` | `import ROOT` | 표준 라이브러리 / 외부 설치 패키지 |
| [503](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:503>): `plot_contour` | `import matplotlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [506](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:506>): `plot_contour` | `import matplotlib.pyplot as plt` | 표준 라이브러리 / 외부 설치 패키지 |
| [507](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:507>): `plot_contour` | `import mplhep as hep` | 표준 라이브러리 / 외부 설치 패키지 |
| [508](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:508>): `plot_contour` | `from matplotlib.colors import LinearSegmentedColormap` | 표준 라이브러리 / 외부 설치 패키지 |
| [509](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:509>): `plot_contour` | `from matplotlib.lines import Line2D` | 표준 라이브러리 / 외부 설치 패키지 |
| [510](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:510>): `plot_contour` | `from matplotlib.patches import Patch` | 표준 라이브러리 / 외부 설치 패키지 |
| [511](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:511>): `plot_contour` | `from matplotlib.ticker import FormatStrFormatter, FuncFormatter, MultipleLocator` | 표준 라이브러리 / 외부 설치 패키지 |
| [512](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:512>): `plot_contour` | `from scipy.interpolate import griddata` | 표준 라이브러리 / 외부 설치 패키지 |

#### [autonomous_allhad/autonomous_allhad/real_subset_worker.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [2](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:2>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [4](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:4>): `module` | `import csv` | 표준 라이브러리 / 외부 설치 패키지 |
| [5](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:5>): `module` | `import contextlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [6](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:6>): `module` | `import gzip` | 표준 라이브러리 / 외부 설치 패키지 |
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:7>): `module` | `import hashlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:8>): `module` | `import html` | 표준 라이브러리 / 외부 설치 패키지 |
| [9](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:9>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [10](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:10>): `module` | `import math` | 표준 라이브러리 / 외부 설치 패키지 |
| [11](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:11>): `module` | `import os` | 표준 라이브러리 / 외부 설치 패키지 |
| [12](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:12>): `module` | `import re` | 표준 라이브러리 / 외부 설치 패키지 |
| [13](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:13>): `module` | `import resource` | 표준 라이브러리 / 외부 설치 패키지 |
| [14](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:14>): `module` | `import shutil` | 표준 라이브러리 / 외부 설치 패키지 |
| [15](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:15>): `module` | `import subprocess` | 표준 라이브러리 / 외부 설치 패키지 |
| [16](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:16>): `module` | `import sys` | 표준 라이브러리 / 외부 설치 패키지 |
| [17](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:17>): `module` | `import time` | 표준 라이브러리 / 외부 설치 패키지 |
| [18](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:18>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [19](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:19>): `module` | `from typing import Any` | 표준 라이브러리 / 외부 설치 패키지 |
| [21](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:21>): `module` | `import awkward as ak` | 표준 라이브러리 / 외부 설치 패키지 |
| [22](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:22>): `module` | `import correctionlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [23](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:23>): `module` | `import numpy as np` | 표준 라이브러리 / 외부 설치 패키지 |
| [25](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:25>): `module` | `from .analysis_scale_factors import REQUIRED_ANALYSIS_SF_COMPONENTS, loose_muon_lowpt_triplet, met_trigger_triplet, photon_trigger_triplet, veto_electron_lowpt_triplet` | [autonomous_allhad/autonomous_allhad/analysis_scale_factors.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/analysis_scale_factors.py>) |
| [32](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:32>): `module` | `from .signal_models import signal_genmodel_branch, signal_mass_from_genmodel, signal_topology_id` | [autonomous_allhad/autonomous_allhad/signal_models.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/signal_models.py>) |
| [37](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:37>): `module` | `import uproot` | 표준 라이브러리 / 외부 설치 패키지 |
| [38](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:38>): `module` | `from coffea.util import load as coffea_load` | 표준 라이브러리 / 외부 설치 패키지 |
| [2795](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:2795>): `build_site` | `import shutil` | 표준 라이브러리 / 외부 설치 패키지 |
| [2891](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:2891>): `main` | `import matplotlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [2893](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:2893>): `main` | `import matplotlib.pyplot as plt` | 표준 라이브러리 / 외부 설치 패키지 |

#### [autonomous_allhad/autonomous_allhad/analysis_scale_factors.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/analysis_scale_factors.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/analysis_scale_factors.py:7>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [9](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/analysis_scale_factors.py:9>): `module` | `from functools import lru_cache` | 표준 라이브러리 / 외부 설치 패키지 |
| [10](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/analysis_scale_factors.py:10>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [11](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/analysis_scale_factors.py:11>): `module` | `from typing import Any` | 표준 라이브러리 / 외부 설치 패키지 |
| [13](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/analysis_scale_factors.py:13>): `module` | `import awkward as ak` | 표준 라이브러리 / 외부 설치 패키지 |
| [14](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/analysis_scale_factors.py:14>): `module` | `import correctionlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [15](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/analysis_scale_factors.py:15>): `module` | `import numpy as np` | 표준 라이브러리 / 외부 설치 패키지 |

#### [autonomous_allhad/autonomous_allhad/signal_models.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/signal_models.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [1](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/signal_models.py:1>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [3](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/signal_models.py:3>): `module` | `import re` | 표준 라이브러리 / 외부 설치 패키지 |

#### [autonomous_allhad/autonomous_allhad/highdm_resolved_categories.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/highdm_resolved_categories.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [1](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/highdm_resolved_categories.py:1>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [3](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/highdm_resolved_categories.py:3>): `module` | `from dataclasses import dataclass` | 표준 라이브러리 / 외부 설치 패키지 |
| [4](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/highdm_resolved_categories.py:4>): `module` | `import math` | 표준 라이브러리 / 외부 설치 패키지 |
| [5](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/highdm_resolved_categories.py:5>): `module` | `from typing import Iterable, Sequence` | 표준 라이브러리 / 외부 설치 패키지 |
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/highdm_resolved_categories.py:7>): `module` | `import numpy as np` | 표준 라이브러리 / 외부 설치 패키지 |

#### [autonomous_allhad/autonomous_allhad/dy_ptll_policy.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/dy_ptll_policy.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [4](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/dy_ptll_policy.py:4>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [6](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/dy_ptll_policy.py:6>): `module` | `from collections import Counter` | 표준 라이브러리 / 외부 설치 패키지 |
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/dy_ptll_policy.py:7>): `module` | `from collections.abc import Iterable, Sequence` | 표준 라이브러리 / 외부 설치 패키지 |
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/dy_ptll_policy.py:8>): `module` | `import re` | 표준 라이브러리 / 외부 설치 패키지 |
| [9](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/dy_ptll_policy.py:9>): `module` | `from typing import Any` | 표준 라이브러리 / 외부 설치 패키지 |

#### [autonomous_allhad/autonomous_allhad/search_bin_categorization.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/search_bin_categorization.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [1](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/search_bin_categorization.py:1>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [3](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/search_bin_categorization.py:3>): `module` | `from typing import Any, Sequence` | 표준 라이브러리 / 외부 설치 패키지 |
| [5](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/search_bin_categorization.py:5>): `module` | `import numpy as np` | 표준 라이브러리 / 외부 설치 패키지 |
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/search_bin_categorization.py:7>): `module` | `from .highdm_resolved_categories import COARSE_NRES_TOPOLOGIES` | [autonomous_allhad/autonomous_allhad/highdm_resolved_categories.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/highdm_resolved_categories.py>) |

#### [autonomous_allhad/autonomous_allhad/sidecar_store.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/sidecar_store.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [10](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/sidecar_store.py:10>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [12](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/sidecar_store.py:12>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [13](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/sidecar_store.py:13>): `module` | `import concurrent.futures` | 표준 라이브러리 / 외부 설치 패키지 |
| [14](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/sidecar_store.py:14>): `module` | `import hashlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [15](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/sidecar_store.py:15>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [16](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/sidecar_store.py:16>): `module` | `import os` | 표준 라이브러리 / 외부 설치 패키지 |
| [17](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/sidecar_store.py:17>): `module` | `import shutil` | 표준 라이브러리 / 외부 설치 패키지 |
| [18](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/sidecar_store.py:18>): `module` | `import sqlite3` | 표준 라이브러리 / 외부 설치 패키지 |
| [19](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/sidecar_store.py:19>): `module` | `import zlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [20](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/sidecar_store.py:20>): `module` | `from functools import lru_cache` | 표준 라이브러리 / 외부 설치 패키지 |
| [21](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/sidecar_store.py:21>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [22](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/sidecar_store.py:22>): `module` | `from typing import Any, Iterable` | 표준 라이브러리 / 외부 설치 패키지 |

#### [analysis/utils/corrections.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [2](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:2>): `module` | `import correctionlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [3](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:3>): `module` | `from correctionlib import convert` | 표준 라이브러리 / 외부 설치 패키지 |
| [4](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:4>): `module` | `import os` | 표준 라이브러리 / 외부 설치 패키지 |
| [5](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:5>): `module` | `import awkward as ak` | 표준 라이브러리 / 외부 설치 패키지 |
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:7>): `module` | `import numpy as np` | 표준 라이브러리 / 외부 설치 패키지 |
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:8>): `module` | `from coffea import lookup_tools, jetmet_tools, util` | 표준 라이브러리 / 외부 설치 패키지 |
| [9](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:9>): `module` | `from coffea.lookup_tools import extractor, dense_lookup` | 표준 라이브러리 / 외부 설치 패키지 |
| [10](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:10>): `module` | `from coffea.jetmet_tools import JECStack, CorrectedJetsFactory, CorrectedMETFactory` | 표준 라이브러리 / 외부 설치 패키지 |
| [12](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:12>): `module` | `import uproot` | 표준 라이브러리 / 외부 설치 패키지 |
| [13](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:13>): `module` | `from coffea.util import save, load` | 표준 라이브러리 / 외부 설치 패키지 |
| [14](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:14>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [16](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:16>): `module` | `import hist` | 표준 라이브러리 / 외부 설치 패키지 |
| [777](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:777>): `module` | `from coffea.lookup_tools.correctionlib_wrapper import correctionlib_wrapper` | 표준 라이브러리 / 외부 설치 패키지 |
| [778](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:778>): `module` | `from coffea.lookup_tools.dense_lookup import dense_lookup` | 표준 라이브러리 / 외부 설치 패키지 |

### A3. 연결 여부 대조용 공용 도구

#### [autonomous_allhad/workflow/run_flat_hists_chunked.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/run_flat_hists_chunked.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [2](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/run_flat_hists_chunked.py:2>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [4](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/run_flat_hists_chunked.py:4>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [5](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/run_flat_hists_chunked.py:5>): `module` | `import hashlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [6](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/run_flat_hists_chunked.py:6>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/run_flat_hists_chunked.py:7>): `module` | `import math` | 표준 라이브러리 / 외부 설치 패키지 |
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/run_flat_hists_chunked.py:8>): `module` | `import os` | 표준 라이브러리 / 외부 설치 패키지 |
| [9](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/run_flat_hists_chunked.py:9>): `module` | `import subprocess` | 표준 라이브러리 / 외부 설치 패키지 |
| [10](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/run_flat_hists_chunked.py:10>): `module` | `import sys` | 표준 라이브러리 / 외부 설치 패키지 |
| [11](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/run_flat_hists_chunked.py:11>): `module` | `import time` | 표준 라이브러리 / 외부 설치 패키지 |
| [12](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/run_flat_hists_chunked.py:12>): `module` | `from collections import Counter` | 표준 라이브러리 / 외부 설치 패키지 |
| [13](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/run_flat_hists_chunked.py:13>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [14](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/run_flat_hists_chunked.py:14>): `module` | `from typing import Any` | 표준 라이브러리 / 외부 설치 패키지 |
| [141](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/run_flat_hists_chunked.py:141>): `load_search_bin_contract` | `from autonomous_allhad.search_bin_categorization import configured_exclusive_bin_count` | [autonomous_allhad/autonomous_allhad/search_bin_categorization.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/search_bin_categorization.py>) |

#### [autonomous_allhad/workflow/build_combined_year_datacards.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combined_year_datacards.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [2](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combined_year_datacards.py:2>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [4](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combined_year_datacards.py:4>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [5](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combined_year_datacards.py:5>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [6](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combined_year_datacards.py:6>): `module` | `import os` | 표준 라이브러리 / 외부 설치 패키지 |
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combined_year_datacards.py:7>): `module` | `import re` | 표준 라이브러리 / 외부 설치 패키지 |
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combined_year_datacards.py:8>): `module` | `import subprocess` | 표준 라이브러리 / 외부 설치 패키지 |
| [9](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combined_year_datacards.py:9>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [11](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combined_year_datacards.py:11>): `module` | `from build_combine_inputs import stable_path, write_parallel_runner` | [autonomous_allhad/workflow/build_combine_inputs.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py>) |

#### [autonomous_allhad/workflow/build_combine_inputs.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [9](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:9>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [11](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:11>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [12](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:12>): `module` | `import hashlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [13](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:13>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [14](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:14>): `module` | `import math` | 표준 라이브러리 / 외부 설치 패키지 |
| [15](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:15>): `module` | `import mmap` | 표준 라이브러리 / 외부 설치 패키지 |
| [16](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:16>): `module` | `import re` | 표준 라이브러리 / 외부 설치 패키지 |
| [17](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:17>): `module` | `import sys` | 표준 라이브러리 / 외부 설치 패키지 |
| [18](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:18>): `module` | `from array import array` | 표준 라이브러리 / 외부 설치 패키지 |
| [19](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:19>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [20](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:20>): `module` | `from typing import Any` | 표준 라이브러리 / 외부 설치 패키지 |
| [22](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:22>): `module` | `import numpy as np` | 표준 라이브러리 / 외부 설치 패키지 |
| [29](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:29>): `module` | `from autonomous_allhad.search_bin_categorization import configured_bin_position_groups, configured_exclusive_mapping, configured_projection_groups` | [autonomous_allhad/autonomous_allhad/search_bin_categorization.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/search_bin_categorization.py>) |
| [35](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:35>): `module` | `from background_process_groups import BACKGROUND_PROCESS_ORDER, background_grouping_contract` | [autonomous_allhad/workflow/background_process_groups.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/background_process_groups.py>) |
| [416](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:416>): `make_hist` | `import ROOT` | 표준 라이브러리 / 외부 설치 패키지 |
| [2122](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:2122>): `overwrite_observations` | `import ROOT` | 표준 라이브러리 / 외부 설치 패키지 |
| [2230](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:2230>): `build_root` | `import ROOT` | 표준 라이브러리 / 외부 설치 패키지 |

#### [autonomous_allhad/workflow/background_process_groups.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/background_process_groups.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [9](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/background_process_groups.py:9>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [11](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/background_process_groups.py:11>): `module` | `from collections import OrderedDict` | 표준 라이브러리 / 외부 설치 패키지 |
| [12](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/background_process_groups.py:12>): `module` | `from typing import Any, Callable` | 표준 라이브러리 / 외부 설치 패키지 |
| [14](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/background_process_groups.py:14>): `module` | `import numpy as np` | 표준 라이브러리 / 외부 설치 패키지 |

#### [autonomous_allhad/workflow/plot_control_search_bins_style.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [4](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:4>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [6](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:6>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:7>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:8>): `module` | `import mmap` | 표준 라이브러리 / 외부 설치 패키지 |
| [9](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:9>): `module` | `import shutil` | 표준 라이브러리 / 외부 설치 패키지 |
| [10](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:10>): `module` | `import sys` | 표준 라이브러리 / 외부 설치 패키지 |
| [11](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:11>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [13](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:13>): `module` | `import numpy as np` | 표준 라이브러리 / 외부 설치 패키지 |
| [22](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:22>): `module` | `from background_process_groups import BACKGROUND_DISPLAY_LABELS, BACKGROUND_PROCESS_ORDER, background_process_for_sample` | [autonomous_allhad/workflow/background_process_groups.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/background_process_groups.py>) |
| [27](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:27>): `module` | `from autonomous_allhad.search_bin_categorization import configured_bin_position_groups, configured_exclusive_mapping` | [autonomous_allhad/autonomous_allhad/search_bin_categorization.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/search_bin_categorization.py>) |
| [813](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:813>): `draw` | `import matplotlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [816](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:816>): `draw` | `import matplotlib.pyplot as plt` | 표준 라이브러리 / 외부 설치 패키지 |
| [817](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:817>): `draw` | `import mplhep as hep` | 표준 라이브러리 / 외부 설치 패키지 |
| [1809](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:1809>): `draw_flat_blocks` | `import matplotlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [1813](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:1813>): `draw_flat_blocks` | `import matplotlib.pyplot as plt` | 표준 라이브러리 / 외부 설치 패키지 |
| [1814](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:1814>): `draw_flat_blocks` | `import mplhep as hep` | 표준 라이브러리 / 외부 설치 패키지 |
| [2515](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:2515>): `write_highdm_distribution_webpage` | `import html` | 표준 라이브러리 / 외부 설치 패키지 |

#### [autonomous_allhad/workflow/plot_highdm73_from_limit_templates.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_highdm73_from_limit_templates.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [4](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_highdm73_from_limit_templates.py:4>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [6](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_highdm73_from_limit_templates.py:6>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [7](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_highdm73_from_limit_templates.py:7>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [8](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_highdm73_from_limit_templates.py:8>): `module` | `from contextlib import ExitStack` | 표준 라이브러리 / 외부 설치 패키지 |
| [9](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_highdm73_from_limit_templates.py:9>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [11](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_highdm73_from_limit_templates.py:11>): `module` | `import numpy as np` | 표준 라이브러리 / 외부 설치 패키지 |
| [12](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_highdm73_from_limit_templates.py:12>): `module` | `import uproot` | 표준 라이브러리 / 외부 설치 패키지 |
| [14](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_highdm73_from_limit_templates.py:14>): `module` | `import plot_control_search_bins_style as style` | [autonomous_allhad/workflow/plot_control_search_bins_style.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py>) |

#### [autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py>)

| 위치·실행 범위 | import | 해결 파일 / 종류 |
|---|---|---|
| [11](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:11>): `module` | `from __future__ import annotations` | 표준 라이브러리 / 외부 설치 패키지 |
| [13](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:13>): `module` | `import argparse` | 표준 라이브러리 / 외부 설치 패키지 |
| [14](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:14>): `module` | `import copy` | 표준 라이브러리 / 외부 설치 패키지 |
| [15](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:15>): `module` | `import hashlib` | 표준 라이브러리 / 외부 설치 패키지 |
| [16](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:16>): `module` | `import json` | 표준 라이브러리 / 외부 설치 패키지 |
| [17](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:17>): `module` | `import math` | 표준 라이브러리 / 외부 설치 패키지 |
| [18](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:18>): `module` | `import re` | 표준 라이브러리 / 외부 설치 패키지 |
| [19](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:19>): `module` | `import sys` | 표준 라이브러리 / 외부 설치 패키지 |
| [20](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:20>): `module` | `from collections import defaultdict` | 표준 라이브러리 / 외부 설치 패키지 |
| [21](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:21>): `module` | `from pathlib import Path` | 표준 라이브러리 / 외부 설치 패키지 |
| [22](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:22>): `module` | `from typing import Any` | 표준 라이브러리 / 외부 설치 패키지 |
| [24](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:24>): `module` | `import numpy as np` | 표준 라이브러리 / 외부 설치 패키지 |
| [25](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:25>): `module` | `import uproot` | 표준 라이브러리 / 외부 설치 패키지 |
| [33](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:33>): `module` | `import build_combine_inputs as canonical` | [autonomous_allhad/workflow/build_combine_inputs.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py>) |

## 부록 B. 파일 I/O·CLI 입력 전수 색인

아래는 AST로 추출한 **읽기/탐색/직렬화/외부 명령 호출부**와 파일 경로를 받는 CLI다. 함수명과 줄번호를 함께 적었으므로 caller → 인자 → resource를 추적할 수 있다. `np.load`, `torch.load`, `ROOT.TFile` 등은 mode에 따라 입력/출력 역할이 다르며, hash 함수의 open은 내용 소비가 아니라 provenance 검사다. 공용 모듈 전체의 dormant/upstream 함수도 포함되어 있다.

### [autonomous_allhad/gnn_lowdm/_implementation/analyze_diagonal_v3_global_shap.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/analyze_diagonal_v3_global_shap.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [94](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/analyze_diagonal_v3_global_shap.py:94>) `sha256` | `path.open('rb')` |
| [108](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/analyze_diagonal_v3_global_shap.py:108>) `cache_paths` | `cache.glob(f'{kind}_cache_*.root')` |
| [112](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/analyze_diagonal_v3_global_shap.py:112>) `cache_paths` | `sidecar_path.read_text()` |
| [472](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/analyze_diagonal_v3_global_shap.py:472>) `main` | `parser.add_argument('--cache', required=True, type=Path)` |
| [473](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/analyze_diagonal_v3_global_shap.py:473>) `main` | `parser.add_argument('--result-dir', required=True, type=Path)` |
| [474](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/analyze_diagonal_v3_global_shap.py:474>) `main` | `parser.add_argument('--output-dir', type=Path)` |
| [497](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/analyze_diagonal_v3_global_shap.py:497>) `main` | `(opts.result_dir / 'selection.json').read_text()` |
| [504](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/analyze_diagonal_v3_global_shap.py:504>) `main` | `torch.load(checkpoint_path, map_location='cpu', weights_only=False)` |

### [autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [162](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py:162>) `process_source` | `Path(sidecar_path).read_text()` |
| [164](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py:164>) `process_source` | `uproot.open(root_path)` |
| [392](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py:392>) `main` | `parser.add_argument('--request', required=True, type=Path)` |
| [393](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py:393>) `main` | `parser.add_argument('--output', required=True, type=Path)` |
| [394](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py:394>) `main` | `parser.add_argument('--model', required=True, type=Path)` |
| [395](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py:395>) `main` | `parser.add_argument('--selection', required=True, type=Path)` |
| [404](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py:404>) `main` | `opts.request.read_text()` |
| [405](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py:405>) `main` | `opts.selection.read_text()` |
| [409](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py:409>) `main` | `Path(request['manifest']).read_text()` |
| [410](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py:410>) `main` | `Path(request['stop_xsec']).read_text()` |

### [autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [174](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py:174>) `validate_trota_provenance` | `Path(sidecar_path).read_text()` |
| [340](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py:340>) `process_source` | `uproot.open(root_path)` |
| [459](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py:459>) `worker` | `request_path.read_text()` |
| [551](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py:551>) `worker` | `uproot.open(output)` |
| [584](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py:584>) `run_request` | `log_path.open('w')` |
| [585](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py:585>) `run_request` | `subprocess.run([sys.executable, str(script), '--worker', str(request_path)], stdout=log, stderr=subprocess.STDOUT, text=True, check=False)` |
| [592](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py:592>) `run_request` | `request_path.read_text()` |
| [606](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py:606>) `run_request` | `sidecar.read_text()` |
| [620](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py:620>) `manager` | `opts.manifest.read_text()` |
| [662](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py:662>) `manager` | `sidecar.read_text()` |
| [761](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py:761>) `manager` | `path.read_text()` |
| [773](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py:773>) `parse_args` | `parser.add_argument('--manifest', type=Path)` |
| [774](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py:774>) `parse_args` | `parser.add_argument('--output', type=Path)` |
| [782](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py:782>) `parse_args` | `parser.add_argument('--worker', type=Path)` |

### [autonomous_allhad/gnn_lowdm/_implementation/build_full_campaign_manifest.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_full_campaign_manifest.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [16](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_full_campaign_manifest.py:16>) `sha256` | `path.open('rb')` |
| [26](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_full_campaign_manifest.py:26>) `main` | `parser.add_argument('--source-directory', required=True, type=Path)` |
| [27](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_full_campaign_manifest.py:27>) `main` | `parser.add_argument('--output', required=True, type=Path)` |
| [37](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_full_campaign_manifest.py:37>) `main` | `source.glob('*.root')` |
| [38](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_full_campaign_manifest.py:38>) `main` | `source.glob('*.json')` |
| [64](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_full_campaign_manifest.py:64>) `main` | `sidecar_path.read_text()` |

### [autonomous_allhad/gnn_lowdm/_implementation/collect_plot_diagonal_v3_limits.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/collect_plot_diagonal_v3_limits.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [22](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/collect_plot_diagonal_v3_limits.py:22>) `main` | `parser.add_argument('--input-dir', type=Path, required=True)` |
| [27](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/collect_plot_diagonal_v3_limits.py:27>) `main` | `(args.input_dir / 'manifest.json').read_text()` |

### [autonomous_allhad/gnn_lowdm/_implementation/collect_plot_highdm79minus6_lowdm30.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/collect_plot_highdm79minus6_lowdm30.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [27](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/collect_plot_highdm79minus6_lowdm30.py:27>) `main` | `parser.add_argument('--campaign', type=Path, required=True)` |
| [28](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/collect_plot_highdm79minus6_lowdm30.py:28>) `main` | `parser.add_argument('--extra-campaign', type=Path, required=True)` |
| [29](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/collect_plot_highdm79minus6_lowdm30.py:29>) `main` | `parser.add_argument('--output', type=Path, required=True)` |
| [30](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/collect_plot_highdm79minus6_lowdm30.py:30>) `main` | `parser.add_argument('--run2-dir', type=Path, required=True)` |
| [39](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/collect_plot_highdm79minus6_lowdm30.py:39>) `main` | `(args.campaign / 'campaign_manifest.json').read_text()` |
| [41](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/collect_plot_highdm79minus6_lowdm30.py:41>) `main` | `(args.extra_campaign / 'campaign_manifest.json').read_text()` |

### [autonomous_allhad/gnn_lowdm/_implementation/compare_oof.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/compare_oof.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [376](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/compare_oof.py:376>) `main` | `parser.add_argument('--signal', nargs='+', required=True, type=Path)` |
| [377](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/compare_oof.py:377>) `main` | `parser.add_argument('--background', nargs='+', required=True, type=Path)` |
| [378](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/compare_oof.py:378>) `main` | `parser.add_argument('--output', required=True, type=Path)` |
| [380](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/compare_oof.py:380>) `main` | `parser.add_argument('--normalization', required=True, type=Path)` |
| [417](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/compare_oof.py:417>) `main` | `opts.normalization.read_text()` |

### [autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [99](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py:99>) `main` | `parser.add_argument('--request', required=True, type=Path)` |
| [101](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py:101>) `main` | `parser.add_argument('--checkpoint', type=Path)` |
| [103](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py:103>) `main` | `parser.add_argument('--local-cache', type=Path)` |
| [104](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py:104>) `main` | `parser.add_argument('--local-output', type=Path)` |
| [105](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py:105>) `main` | `parser.add_argument('--local-manifest', type=Path)` |
| [106](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py:106>) `main` | `parser.add_argument('--local-xsec', type=Path)` |
| [107](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py:107>) `main` | `parser.add_argument('--local-model', type=Path)` |
| [108](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py:108>) `main` | `parser.add_argument('--local-configuration', type=Path)` |
| [110](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py:110>) `main` | `opts.request.read_text()` |
| [142](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py:142>) `main` | `configuration_path.read_text()` |
| [197](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py:197>) `main` | `torch.load(opts.checkpoint, map_location='cpu', weights_only=False)` |
| [216](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py:216>) `main` | `manifest_path.read_text()` |
| [217](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py:217>) `main` | `xsec_path.read_text()` |

### [autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [54](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py:54>) `sha256` | `path.open('rb')` |
| [302](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py:302>) `main` | `parser.add_argument('--cache', required=True, type=Path)` |
| [303](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py:303>) `main` | `parser.add_argument('--campaign-manifest', required=True, type=Path)` |
| [304](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py:304>) `main` | `parser.add_argument('--xsec', required=True, type=Path)` |
| [305](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py:305>) `main` | `parser.add_argument('--result', required=True, type=Path)` |
| [306](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py:306>) `main` | `parser.add_argument('--output', required=True, type=Path)` |
| [314](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py:314>) `main` | `selection_path.read_text()` |
| [318](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py:318>) `main` | `summary_path.read_text()` |
| [344](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py:344>) `main` | `opts.campaign_manifest.read_text()` |
| [345](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py:345>) `main` | `opts.xsec.read_text()` |
| [349](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py:349>) `main` | `torch.load(checkpoint_path, map_location='cpu', weights_only=False)` |

### [autonomous_allhad/gnn_lowdm/_implementation/finalize_diagonal_v3_cache.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/finalize_diagonal_v3_cache.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [66](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/finalize_diagonal_v3_cache.py:66>) `validate_one` | `sidecar_path.read_text()` |
| [72](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/finalize_diagonal_v3_cache.py:72>) `validate_one` | `uproot.open(root_path)` |
| [104](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/finalize_diagonal_v3_cache.py:104>) `main` | `parser.add_argument('--requests', required=True, type=Path)` |
| [105](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/finalize_diagonal_v3_cache.py:105>) `main` | `parser.add_argument('--outputs', required=True, type=Path)` |
| [106](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/finalize_diagonal_v3_cache.py:106>) `main` | `parser.add_argument('--campaign-manifest', required=True, type=Path)` |
| [110](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/finalize_diagonal_v3_cache.py:110>) `main` | `opts.campaign_manifest.read_text()` |
| [113](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/finalize_diagonal_v3_cache.py:113>) `main` | `opts.requests.glob('*.json')` |
| [117](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/finalize_diagonal_v3_cache.py:117>) `main` | `path.read_text()` |

### [autonomous_allhad/gnn_lowdm/_implementation/merge_all_signal_templates.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_all_signal_templates.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [52](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_all_signal_templates.py:52>) `merge_partials` | `path.read_text()` |
| [104](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_all_signal_templates.py:104>) `update_template` | `ROOT.TFile(str(path), 'UPDATE')` |
| [128](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_all_signal_templates.py:128>) `compare_existing_signals` | `ROOT.TFile.Open(str(old_root), 'READ')` |
| [157](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_all_signal_templates.py:157>) `main` | `parser.add_argument('--partials', required=True, type=Path)` |
| [158](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_all_signal_templates.py:158>) `main` | `parser.add_argument('--existing-low', required=True, type=Path)` |
| [159](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_all_signal_templates.py:159>) `main` | `parser.add_argument('--high-grid', required=True, type=Path)` |
| [160](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_all_signal_templates.py:160>) `main` | `parser.add_argument('--output', required=True, type=Path)` |
| [162](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_all_signal_templates.py:162>) `main` | `parser.add_argument('--campaign-year', required=True, choices=('2024', '2025'))` |
| [167](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_all_signal_templates.py:167>) `main` | `args.partials.glob('signal_cache_*.json')` |
| [172](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_all_signal_templates.py:172>) `main` | `args.high_grid.read_text()` |
| [179](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_all_signal_templates.py:179>) `main` | `(source_dir / 'manifest.json').read_text()` |
| [184](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_all_signal_templates.py:184>) `main` | `shutil.copy2(old_template, template)` |
| [193](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_all_signal_templates.py:193>) `main` | `cards_dir.glob('datacard_mStop*_mLSP*.txt')` |

### [autonomous_allhad/gnn_lowdm/_implementation/merge_diagonal_v3_cr_nnout_partials.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_diagonal_v3_cr_nnout_partials.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [44](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_diagonal_v3_cr_nnout_partials.py:44>) `main` | `parser.add_argument('--partials', required=True, type=Path)` |
| [45](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_diagonal_v3_cr_nnout_partials.py:45>) `main` | `parser.add_argument('--retry-partials', type=Path)` |
| [46](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_diagonal_v3_cr_nnout_partials.py:46>) `main` | `parser.add_argument('--output', required=True, type=Path)` |
| [47](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_diagonal_v3_cr_nnout_partials.py:47>) `main` | `parser.add_argument('--manifest', type=Path, help='Optional full-campaign manifest used for exact input-set validation.')` |
| [53](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_diagonal_v3_cr_nnout_partials.py:53>) `main` | `args.partials.glob('*.json')` |
| [55](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_diagonal_v3_cr_nnout_partials.py:55>) `main` | `args.retry_partials.glob('*.json')` |
| [66](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_diagonal_v3_cr_nnout_partials.py:66>) `main` | `path.read_text()` |
| [74](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_diagonal_v3_cr_nnout_partials.py:74>) `main` | `path.read_text()` |
| [94](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_diagonal_v3_cr_nnout_partials.py:94>) `main` | `path.read_text()` |
| [195](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_diagonal_v3_cr_nnout_partials.py:195>) `main` | `args.manifest.read_text()` |

### [autonomous_allhad/gnn_lowdm/_implementation/plot_control_regions.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_control_regions.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [217](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_control_regions.py:217>) `main` | `parser.add_argument('--input', required=True, type=Path)` |
| [218](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_control_regions.py:218>) `main` | `parser.add_argument('--output', required=True, type=Path)` |
| [221](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_control_regions.py:221>) `main` | `opts.input.read_text()` |

### [autonomous_allhad/gnn_lowdm/_implementation/plot_diagonal_v3_model_schematic.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_diagonal_v3_model_schematic.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [384](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_diagonal_v3_model_schematic.py:384>) `main` | `parser.add_argument('--pdf', type=Path, default=Path('output/pdf/lowdm_diagonal_v3_gnn_schematic.pdf'))` |
| [389](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_diagonal_v3_model_schematic.py:389>) `main` | `parser.add_argument('--svg', type=Path, default=Path('output/figures/lowdm_diagonal_v3_gnn_schematic.svg'))` |
| [394](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_diagonal_v3_model_schematic.py:394>) `main` | `parser.add_argument('--png', type=Path, default=Path('output/figures/lowdm_diagonal_v3_gnn_schematic.png'))` |

### [autonomous_allhad/gnn_lowdm/_implementation/plot_full_gnn_results.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_full_gnn_results.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [44](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_full_gnn_results.py:44>) `sha256` | `path.open('rb')` |
| [122](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_full_gnn_results.py:122>) `main` | `parser.add_argument('--result', required=True, type=Path)` |
| [123](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_full_gnn_results.py:123>) `main` | `parser.add_argument('--manifest', required=True, type=Path)` |
| [124](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_full_gnn_results.py:124>) `main` | `parser.add_argument('--output', type=Path)` |
| [142](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_full_gnn_results.py:142>) `main` | `opts.manifest.read_text()` |
| [143](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_full_gnn_results.py:143>) `main` | `test_summary_path.read_text()` |
| [150](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_full_gnn_results.py:150>) `main` | `uproot.open(scores_path)` |
| [266](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_full_gnn_results.py:266>) `main` | `uproot.open(templates_path)` |
| [377](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_full_gnn_results.py:377>) `main` | `selection_path.read_text()` |

### [autonomous_allhad/gnn_lowdm/_implementation/plot_hyperparameter_losses.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_hyperparameter_losses.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [70](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_hyperparameter_losses.py:70>) `load_histories` | `reference_history.read_text()` |
| [74](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_hyperparameter_losses.py:74>) `load_histories` | `(campaign / 'trials').glob('*/training_history.json')` |
| [75](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_hyperparameter_losses.py:75>) `load_histories` | `path.read_text()` |
| [224](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_hyperparameter_losses.py:224>) `campaign_complete` | `path.read_text()` |
| [229](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_hyperparameter_losses.py:229>) `main` | `parser.add_argument('--campaign', required=True, type=Path)` |
| [230](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_hyperparameter_losses.py:230>) `main` | `parser.add_argument('--output', type=Path)` |
| [231](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_hyperparameter_losses.py:231>) `main` | `parser.add_argument('--reference-history', type=Path)` |

### [autonomous_allhad/gnn_lowdm/_implementation/plot_lowdm_category_nn_out.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_lowdm_category_nn_out.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [45](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_lowdm_category_nn_out.py:45>) `sha256` | `path.open('rb')` |
| [97](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_lowdm_category_nn_out.py:97>) `main` | `parser.add_argument('--scores', required=True, type=Path)` |
| [98](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_lowdm_category_nn_out.py:98>) `main` | `parser.add_argument('--category-map', required=True, type=Path)` |
| [99](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_lowdm_category_nn_out.py:99>) `main` | `parser.add_argument('--normalization', required=True, type=Path)` |
| [100](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_lowdm_category_nn_out.py:100>) `main` | `parser.add_argument('--output', required=True, type=Path)` |
| [101](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_lowdm_category_nn_out.py:101>) `main` | `parser.add_argument('--model', choices=('dnn', 'gnn', 'transformer'), default='gnn')` |
| [103](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_lowdm_category_nn_out.py:103>) `main` | `parser.add_argument('--signal-xsec-pb', type=float, default=0.006426)` |
| [109](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_lowdm_category_nn_out.py:109>) `main` | `opts.category_map.read_text()` |
| [116](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_lowdm_category_nn_out.py:116>) `main` | `opts.normalization.read_text()` |
| [118](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_lowdm_category_nn_out.py:118>) `main` | `uproot.open(opts.scores)` |

### [autonomous_allhad/gnn_lowdm/_implementation/prepare_all_signal_template_condor.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_all_signal_template_condor.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [21](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_all_signal_template_condor.py:21>) `sha256` | `path.open('rb')` |
| [91](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_all_signal_template_condor.py:91>) `main` | `parser.add_argument('--staging', required=True, type=Path)` |
| [92](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_all_signal_template_condor.py:92>) `main` | `parser.add_argument('--eos-campaign', required=True)` |
| [93](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_all_signal_template_condor.py:93>) `main` | `parser.add_argument('--cache', required=True, type=Path)` |
| [94](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_all_signal_template_condor.py:94>) `main` | `parser.add_argument('--source-eos', required=True)` |
| [108](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_all_signal_template_condor.py:108>) `main` | `shutil.copy2(path, staging / 'payload' / path.name)` |
| [109](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_all_signal_template_condor.py:109>) `main` | `args.cache.glob('signal_cache_*.root')` |
| [114](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_all_signal_template_condor.py:114>) `main` | `local.with_suffix('.json').read_text()` |
| [148](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_all_signal_template_condor.py:148>) `main` | `path.with_suffix('.json').read_text()` |

### [autonomous_allhad/gnn_lowdm/_implementation/prepare_diagonal_v3_cache_condor.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_diagonal_v3_cache_condor.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [27](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_diagonal_v3_cache_condor.py:27>) `sha256` | `path.open('rb')` |
| [134](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_diagonal_v3_cache_condor.py:134>) `main` | `parser.add_argument('--manifest', required=True, type=Path)` |
| [135](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_diagonal_v3_cache_condor.py:135>) `main` | `parser.add_argument('--staging', required=True, type=Path)` |
| [136](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_diagonal_v3_cache_condor.py:136>) `main` | `parser.add_argument('--eos-campaign', required=True)` |
| [142](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_diagonal_v3_cache_condor.py:142>) `main` | `opts.manifest.read_text()` |
| [167](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_diagonal_v3_cache_condor.py:167>) `main` | `shutil.copy2(cache_worker, staging / 'payload/build_expanded_feature_cache.py')` |
| [168](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_diagonal_v3_cache_condor.py:168>) `main` | `shutil.copy2(finalizer, staging / 'payload/finalize_diagonal_v3_cache.py')` |
| [169](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_diagonal_v3_cache_condor.py:169>) `main` | `shutil.copy2(resolved_worker, staging / 'payload/highdm_resolved_categories.py')` |
| [232](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_diagonal_v3_cache_condor.py:232>) `main` | `(staging / 'payload').iterdir()` |
| [234](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_diagonal_v3_cache_condor.py:234>) `main` | `(staging / 'requests').glob('*.json')` |
| [235](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_diagonal_v3_cache_condor.py:235>) `main` | `(staging / 'pilot_requests').glob('*.json')` |

### [autonomous_allhad/gnn_lowdm/_implementation/prepare_highdm79minus6_lowdm30_condor.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_highdm79minus6_lowdm30_condor.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [21](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_highdm79minus6_lowdm30_condor.py:21>) `read_json` | `path.read_text()` |
| [31](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_highdm79minus6_lowdm30_condor.py:31>) `main` | `parser.add_argument('--high-base', type=Path, required=True)` |
| [32](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_highdm79minus6_lowdm30_condor.py:32>) `main` | `parser.add_argument('--low-base', type=Path, required=True)` |
| [33](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_highdm79minus6_lowdm30_condor.py:33>) `main` | `parser.add_argument('--output', type=Path, required=True)` |
| [57](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_highdm79minus6_lowdm30_condor.py:57>) `main` | `read_json(args.high_base / topology / 'manifest.json')` |
| [58](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_highdm79minus6_lowdm30_condor.py:58>) `main` | `read_json(args.low_base / topology / 'manifest.json')` |
| [109](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_highdm79minus6_lowdm30_condor.py:109>) `main` | `shutil.copy2(worker_source, executable)` |
| [135](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_highdm79minus6_lowdm30_condor.py:135>) `main` | `submit_path.read_text()` |

### [autonomous_allhad/gnn_lowdm/_implementation/publish_lowdm_gnn_web.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/publish_lowdm_gnn_web.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [50](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/publish_lowdm_gnn_web.py:50>) `sha256` | `path.open('rb')` |
| [71](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/publish_lowdm_gnn_web.py:71>) `copy_pair` | `shutil.copy2(source_file, destination_file)` |
| [77](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/publish_lowdm_gnn_web.py:77>) `load` | `path.read_text()` |
| [149](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/publish_lowdm_gnn_web.py:149>) `main` | `parser.add_argument('--docs', required=True, type=Path)` |
| [150](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/publish_lowdm_gnn_web.py:150>) `main` | `parser.add_argument('--sr-2024', required=True, type=Path)` |
| [151](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/publish_lowdm_gnn_web.py:151>) `main` | `parser.add_argument('--cr-2024', required=True, type=Path)` |
| [152](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/publish_lowdm_gnn_web.py:152>) `main` | `parser.add_argument('--result-2025', required=True, type=Path)` |
| [153](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/publish_lowdm_gnn_web.py:153>) `main` | `parser.add_argument('--highdm73', required=True, type=Path)` |
| [154](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/publish_lowdm_gnn_web.py:154>) `main` | `parser.add_argument('--lowdm-cr-physics-plots', '--an-lowdm-plots', dest='lowdm_cr_physics_plots', required=True, type=Path)` |
| [207](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/publish_lowdm_gnn_web.py:207>) `main` | `load(args.result_2025 / 'validation_summary.json')` |
| [208](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/publish_lowdm_gnn_web.py:208>) `main` | `load(args.cr_2024 / 'plots/cr/lowdm_cr_nnout_inclusive_plot_summary.json')` |
| [209](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/publish_lowdm_gnn_web.py:209>) `main` | `load(args.result_2025 / 'plots/cr/lowdm_cr_nnout_inclusive_plot_summary.json')` |

### [autonomous_allhad/gnn_lowdm/_implementation/rank005_numpy.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/rank005_numpy.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [36](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/rank005_numpy.py:36>) `Rank005Numpy.__init__` | `np.load(path)` |

### [autonomous_allhad/gnn_lowdm/_implementation/region_io.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [272](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:272>) `write_json` | `temporary.open('w')` |
| [281](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:281>) `write_text` | `temporary.open('w')` |
| [1251](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:1251>) `process_source` | `Path(sidecar_path).read_text()` |
| [1253](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:1253>) `process_source` | `uproot.open(root_path)` |
| [1403](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:1403>) `worker` | `request_path.read_text()` |
| [1406](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:1406>) `worker` | `Path(request['manifest']).read_text()` |
| [1407](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:1407>) `worker` | `Path(request['stop_xsec']).read_text()` |
| [1512](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:1512>) `make_requests` | `opts.manifest.read_text()` |
| [1567](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:1567>) `collect_partial_paths` | `(path / 'partials').glob('*.json')` |
| [1590](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:1590>) `merge_campaign` | `gzip.open(source_audit_temporary, 'wt')` |
| [1592](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:1592>) `merge_campaign` | `path.read_text()` |
| [1723](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:1723>) `parser` | `action.add_argument('--worker', type=Path)` |
| [1726](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:1726>) `parser` | `result.add_argument('--manifest', type=Path, default=OUTER_PROJECT / 'gnn_lowdm' / 'full_campaign_manifest_2024.json')` |
| [1731](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:1731>) `parser` | `result.add_argument('--stop-xsec', type=Path, default=OUTER_PROJECT / 'signals' / 'stop_xsec_13p6TeV.json')` |
| [1736](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:1736>) `parser` | `result.add_argument('--repository', type=Path, default=REPOSITORY)` |
| [1737](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:1737>) `parser` | `result.add_argument('--output', type=Path)` |
| [1738](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py:1738>) `parser` | `result.add_argument('--input', type=Path)` |

### [autonomous_allhad/gnn_lowdm/_implementation/supplementary_plots.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/supplementary_plots.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [31](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/supplementary_plots.py:31>) `_selected_epoch` | `(PACKAGE / 'config.json').read_text()` |
| [83](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/supplementary_plots.py:83>) `training_curves` | `history_path.read_text()` |
| [140](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/supplementary_plots.py:140>) `roc_curve` | `np.load(scores_path, allow_pickle=False)` |
| [186](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/supplementary_plots.py:186>) `shap_plots` | `np.load(values_path, allow_pickle=False)` |
| [222](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/supplementary_plots.py:222>) `main_training_curves` | `parser.add_argument('--history', type=Path, default=MODEL / 'training_history.json')` |
| [223](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/supplementary_plots.py:223>) `main_training_curves` | `parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)` |
| [230](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/supplementary_plots.py:230>) `main_roc` | `parser.add_argument('--scores', type=Path, default=RESULT / 'test/diagonal_v3_test_scores.npz')` |
| [231](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/supplementary_plots.py:231>) `main_roc` | `parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)` |
| [238](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/supplementary_plots.py:238>) `main_shap` | `parser.add_argument('--values', type=Path, default=RESULT / 'global_shap/diagonal_v3_global_gradientshap_values.npz')` |
| [239](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/supplementary_plots.py:239>) `main_shap` | `parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)` |

### [autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [97](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py:97>) `load_trial_configs` | `path.read_text()` |
| [307](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py:307>) `train_trial` | `summary_path.read_text()` |
| [346](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py:346>) `train_trial` | `torch.load(latest_path, map_location=device, weights_only=False)` |
| [535](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py:535>) `main` | `parser.add_argument('--cache', required=True, type=Path)` |
| [536](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py:536>) `main` | `parser.add_argument('--campaign-manifest', required=True, type=Path)` |
| [537](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py:537>) `main` | `parser.add_argument('--xsec', required=True, type=Path)` |
| [538](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py:538>) `main` | `parser.add_argument('--output', required=True, type=Path)` |
| [539](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py:539>) `main` | `parser.add_argument('--trials-json', type=Path)` |
| [540](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py:540>) `main` | `parser.add_argument('--baseline-checkpoint', type=Path)` |
| [588](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py:588>) `main` | `opts.campaign_manifest.read_text()` |
| [590](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py:590>) `main` | `opts.xsec.read_text()` |
| [668](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py:668>) `main` | `torch.load(opts.output / 'trials' / best_name / 'best_model.pt', map_location='cpu', weights_only=False)` |

### [autonomous_allhad/gnn_lowdm/_implementation/train_oof.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_oof.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [38](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_oof.py:38>) `sha256` | `path.open('rb')` |
| [170](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_oof.py:170>) `main` | `parser.add_argument('--signal', nargs='+', required=True, type=Path)` |
| [171](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_oof.py:171>) `main` | `parser.add_argument('--background', nargs='+', required=True, type=Path)` |
| [172](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_oof.py:172>) `main` | `parser.add_argument('--output', required=True, type=Path)` |

### [autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [54](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py:54>) `load_inputs` | `(opts.cache / 'campaign_state.json').read_text()` |
| [60](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py:60>) `load_inputs.nonempty_cache_paths` | `opts.cache.glob(f'{kind}_cache_*.root')` |
| [64](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py:64>) `load_inputs.nonempty_cache_paths` | `sidecar_path.read_text()` |
| [108](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py:108>) `load_inputs` | `opts.campaign_manifest.read_text()` |
| [187](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py:187>) `physics_weights` | `Path(campaign['manifest']).read_text()` |
| [645](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py:645>) `train_trial` | `summary_path.read_text()` |
| [677](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py:677>) `train_trial` | `torch.load(latest_path, map_location='cpu', weights_only=False)` |
| [915](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py:915>) `tuning_stage` | `opts.trials.read_text()` |
| [950](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py:950>) `tuning_stage` | `torch.load(Path(ranked[0]['checkpoint']), map_location='cpu', weights_only=False)` |
| [1041](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py:1041>) `test_stage` | `selection_path.read_text()` |
| [1046](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py:1046>) `test_stage` | `existing_summary.read_text()` |
| [1049](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py:1049>) `test_stage` | `torch.load(checkpoint_path, map_location='cpu', weights_only=False)` |
| [1081](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py:1081>) `test_stage` | `opts.campaign_manifest.read_text()` |
| [1292](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py:1292>) `main` | `parser.add_argument('--cache', required=True, type=Path)` |
| [1293](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py:1293>) `main` | `parser.add_argument('--campaign-manifest', required=True, type=Path)` |
| [1294](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py:1294>) `main` | `parser.add_argument('--xsec', required=True, type=Path)` |
| [1295](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py:1295>) `main` | `parser.add_argument('--output', required=True, type=Path)` |
| [1299](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py:1299>) `main` | `parser.add_argument('--trials', type=Path)` |
| [1336](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py:1336>) `main` | `opts.xsec.read_text()` |
| [1338](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py:1338>) `main` | `opts.campaign_manifest.read_text()` |

### [autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_numpy_inference.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_numpy_inference.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [20](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_numpy_inference.py:20>) `main` | `parser.add_argument('--cache', required=True, type=Path)` |
| [21](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_numpy_inference.py:21>) `main` | `parser.add_argument('--checkpoint', required=True, type=Path)` |
| [22](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_numpy_inference.py:22>) `main` | `parser.add_argument('--numpy-model', required=True, type=Path)` |
| [23](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_numpy_inference.py:23>) `main` | `parser.add_argument('--output', required=True, type=Path)` |
| [44](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_numpy_inference.py:44>) `main` | `torch.load(args.checkpoint, map_location='cpu', weights_only=False)` |

### [autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_region_feature_parity.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_region_feature_parity.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [20](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_region_feature_parity.py:20>) `main` | `parser.add_argument('--cache', required=True, type=Path)` |
| [21](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_region_feature_parity.py:21>) `main` | `parser.add_argument('--output', required=True, type=Path)` |
| [39](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_region_feature_parity.py:39>) `main` | `uproot.open(opts.cache)` |

### [autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_srcr_inputs.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_srcr_inputs.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [34](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_srcr_inputs.py:34>) `main` | `parser.add_argument('--sr', required=True, type=Path)` |
| [35](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_srcr_inputs.py:35>) `main` | `parser.add_argument('--cr', required=True, type=Path)` |
| [36](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_srcr_inputs.py:36>) `main` | `parser.add_argument('--output', required=True, type=Path)` |
| [38](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_srcr_inputs.py:38>) `main` | `args.sr.read_text()` |
| [39](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_srcr_inputs.py:39>) `main` | `args.cr.read_text()` |

### [autonomous_allhad/gnn_lowdm/build_datacard.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/build_datacard.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [27](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/build_datacard.py:27>) `canonical_sr_labels` | `CONFIG_PATH.read_text()` |
| [169](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/build_datacard.py:169>) `signals_from_root` | `ROOT.TFile.Open(str(template), 'READ')` |
| [207](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/build_datacard.py:207>) `main` | `parser.add_argument('--template', required=True, type=Path)` |
| [208](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/build_datacard.py:208>) `main` | `parser.add_argument('--summary', required=True, type=Path)` |
| [210](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/build_datacard.py:210>) `main` | `parser.add_argument('--output', required=True, type=Path)` |
| [215](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/build_datacard.py:215>) `main` | `args.summary.read_text()` |

### [autonomous_allhad/gnn_lowdm/data.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/data.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [199](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/data.py:199>) `canonical_config` | `CANONICAL_CONFIG_PATH.read_text()` |
| [909](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/data.py:909>) `_read_one` | `uproot.open(path)` |

### [autonomous_allhad/gnn_lowdm/merge_datacard.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/merge_datacard.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [17](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/merge_datacard.py:17>) `canonical_lowdm_layout` | `CONFIG_PATH.read_text()` |
| [77](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/merge_datacard.py:77>) `rewrite_card` | `source.read_text()` |
| [96](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/merge_datacard.py:96>) `main` | `parser.add_argument('--template', action='append', required=True, type=assignment)` |
| [99](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/merge_datacard.py:99>) `main` | `parser.add_argument('--update-template', action='append', default=[], type=assignment)` |
| [102](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/merge_datacard.py:102>) `main` | `parser.add_argument('--cards', action='append', required=True, type=assignment)` |
| [105](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/merge_datacard.py:105>) `main` | `parser.add_argument('--update-cards', action='append', default=[], type=assignment)` |
| [108](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/merge_datacard.py:108>) `main` | `parser.add_argument('--output', required=True, type=Path)` |
| [126](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/merge_datacard.py:126>) `main` | `ROOT.TFile(str(output_root), 'RECREATE')` |
| [133](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/merge_datacard.py:133>) `main` | `ROOT.TFile.Open(str(templates[topology]), 'READ')` |
| [145](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/merge_datacard.py:145>) `main` | `ROOT.TFile.Open(str(update_path), 'READ')` |
| [164](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/merge_datacard.py:164>) `main` | `card_directory.glob('datacard_mStop*_mLSP*.txt')` |
| [169](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/merge_datacard.py:169>) `main` | `card_updates.get(topology, Path('/__missing__')).glob('datacard_mStop*_mLSP*.txt')` |

### [autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [346](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py:346>) `read_json` | `path.read_text()` |
| [352](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py:352>) `write_json` | `temporary.open('w')` |
| [360](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py:360>) `file_sha256` | `path.open('rb')` |
| [370](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py:370>) `load_search_bin_configuration` | `read_json(path)` |
| [2164](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py:2164>) `process_root` | `read_root_metadata(root_path, fallback=norm)` |
| [2171](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py:2171>) `process_root` | `uproot.open(root_path)` |
| [2800](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py:2800>) `expand_roots` | `p.glob('*.root')` |
| [2802](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py:2802>) `expand_roots` | `p.parent.glob(p.name)` |
| [2816](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py:2816>) `main` | `parser.add_argument('--inputs', nargs='+', required=True)` |
| [2818](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py:2818>) `main` | `parser.add_argument('--output', required=True)` |
| [2819](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py:2819>) `main` | `parser.add_argument('--campaign-year', choices=sorted(BTAG_EFFICIENCY_RELATIVE_PATHS), default='2024')` |
| [2859](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py:2859>) `main` | `parser.add_argument('--allow-zero-entry-roots', action='store_true', help='Treat readable flat ROOT files with an empty Events tree as valid zero-contribution inputs.')` |
| [2923](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py:2923>) `main` | `parser.add_argument('--search-bin-config', type=Path, default=None, help='Main search-bin definition. Full histogram production defaults to autonomous_allhad/configs/search_bins_<year>.json.')` |
| [3018](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py:3018>) `main` | `read_json(Path(args.normalization))` |

### [autonomous_allhad/workflow/postprocess_limits.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [192](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:192>) `load_excluded_points` | `exclusion_dir.glob('limit_point_exclusions_*.json')` |
| [197](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:197>) `load_excluded_points` | `path.read_text()` |
| [213](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:213>) `read_json` | `path.read_text()` |
| [222](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:222>) `load_thermal_relic_contour` | `read_json(path)` |
| [250](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:250>) `load_thermal_relic_contour` | `path.read_bytes()` |
| [298](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:298>) `load_stop_pair_xsecs` | `read_json(path)` |
| [333](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:333>) `parse_limit_file` | `uproot.open(path)` |
| [348](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:348>) `parse_limit_file` | `ROOT.TFile.Open(str(path))` |
| [399](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:399>) `collect_limits` | `limit_dir.glob(f'higgsCombine_{mass_key}.AsymptoticLimits*.root')` |
| [403](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:403>) `collect_limits` | `limit_dir.glob(f'higgsCombine_{mass_key}.root')` |
| [404](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:404>) `collect_limits` | `limit_dir.glob(f'higgsCombine_{mass_key}*.root')` |
| [1351](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:1351>) `plot_contour` | `read_json(Path(run2_contours))` |
| [1384](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:1384>) `plot_contour` | `read_json(Path(run2_compressed_contours))` |
| [1554](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:1554>) `main` | `parser.add_argument('--input-dir', type=Path, required=True)` |
| [1555](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:1555>) `main` | `parser.add_argument('--baseline-limits', type=Path)` |
| [1556](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:1556>) `main` | `parser.add_argument('--overlay-limits', type=Path, help='optional expected_limits.json whose expected contour is drawn in blue over the primary result')` |
| [1572](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:1572>) `main` | `parser.add_argument('--thermal-relic', type=Path, help='optional higgsino-stop relic-contour JSON; draws Omega_LSP*h^2=0.12 and hatches the over-abundant side')` |
| [1580](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:1580>) `main` | `parser.add_argument('--campaign-year', choices=('2024', '2025', '2024_2025'), required=True)` |
| [1598](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:1598>) `main` | `parser.add_argument('--signal-xsec', type=Path, default=DEFAULT_STOP_XSEC, help=f'stop-pair cross-section table used by the default xsec view (default: {DEFAULT_STOP_XSEC})')` |
| [1625](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:1625>) `main` | `parser.add_argument('--allow-noncanonical-runtime', action='store_true', help='allow local display-only rendering with the installed plotting stack; the actual versions are recorded in the output manifest')` |
| [1663](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:1663>) `main` | `manifest_path.read_text()` |
| [1687](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:1687>) `main` | `read_json(expected_limits_path)` |
| [1715](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:1715>) `main` | `read_json(args.overlay_limits)` |
| [1731](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py:1731>) `main` | `args.baseline_limits.read_text()` |

### [autonomous_allhad/autonomous_allhad/real_subset_worker.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [173](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:173>) `local_analysis_dir` | `(cached / 'hists').glob('btageff*.merged')` |
| [190](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:190>) `local_analysis_dir` | `shutil.copytree(key / 'analysis' / 'data', data_dir, dirs_exist_ok=True)` |
| [193](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:193>) `local_analysis_dir` | `(key / 'analysis' / 'hists').glob('btageff*.merged')` |
| [197](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:197>) `local_analysis_dir` | `shutil.copy2(source, hists_dir / source.name)` |
| [234](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:234>) `load_analysis_corrections` | `coffea_load('data/corrections.coffea')` |
| [335](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:335>) `get_btag_corrector` | `load_analysis_corrections(repo)` |
| [405](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:405>) `apply_jec` | `load_analysis_corrections(repo)` |
| [480](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:480>) `compute_weight_bundle` | `load_analysis_corrections(repo)` |
| [1020](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:1020>) `open_root_with_xrd_fallback` | `uproot.open(file_path, timeout=timeout)` |
| [1070](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:1070>) `open_root_with_xrd_fallback` | `subprocess.run(cmd, text=True, capture_output=True, timeout=int(os.environ.get('AUTONOMOUS_ALLHAD_XRDCP_TIMEOUT', '600')), env=xrdcp_env)` |
| [1103](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:1103>) `open_root_with_xrd_fallback` | `uproot.open(str(cache_path), timeout=timeout)` |
| [1132](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:1132>) `load_config` | `path.read_text()` |
| [1304](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:1304>) `ak4_tight_lepton_veto_mask` | `correctionlib.CorrectionSet.from_file(str(analysis_data_file(repo, relative_path)))` |
| [1334](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:1334>) `_correction` | `correctionlib.CorrectionSet.from_file(str(path))` |
| [1349](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:1349>) `load_lumimask` | `path.read_text()` |
| [2514](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:2514>) `write_csv` | `path.open('w', newline='')` |
| [2522](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:2522>) `provenance` | `(repo / 'autonomous_allhad').glob('**/*')` |
| [2612](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:2612>) `trigger_audit_for_file` | `uproot.open(file_path, timeout=60)` |
| [2796](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:2796>) `build_site` | `shutil.copy2(s, docs / 'data' / src)` |
| [2834](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:2834>) `main` | `gzip.open(metadata_path, 'rt')` |
| [2835](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py:2835>) `main` | `json.load(f)` |

### [autonomous_allhad/autonomous_allhad/analysis_scale_factors.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/analysis_scale_factors.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [49](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/analysis_scale_factors.py:49>) `_load` | `correctionlib.CorrectionSet.from_file(path)` |

### [autonomous_allhad/autonomous_allhad/sidecar_store.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/sidecar_store.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [75](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/sidecar_store.py:75>) `read_root_metadata` | `adjacent.read_text()` |
| [108](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/sidecar_store.py:108>) `_read_paths` | `path.read_text()` |
| [115](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/sidecar_store.py:115>) `_sha256_file` | `path.open('rb')` |
| [125](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/sidecar_store.py:125>) `_encode_sidecar` | `sidecar.read_bytes()` |
| [288](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/sidecar_store.py:288>) `append_discovered` | `workflow.rglob('*.json')` |
| [302](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/sidecar_store.py:302>) `append_discovered` | `shutil.copy2(store, temporary)` |
| [416](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/sidecar_store.py:416>) `main` | `build.add_argument('--root-list', type=Path, required=True)` |
| [417](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/sidecar_store.py:417>) `main` | `build.add_argument('--output', type=Path, required=True)` |
| [418](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/sidecar_store.py:418>) `main` | `build.add_argument('--delete-sources', action='store_true')` |
| [421](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/sidecar_store.py:421>) `main` | `append.add_argument('--workflow', type=Path, required=True)` |
| [422](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/sidecar_store.py:422>) `main` | `append.add_argument('--store', type=Path, required=True)` |
| [423](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/sidecar_store.py:423>) `main` | `append.add_argument('--delete-sources', action='store_true')` |

### [analysis/utils/corrections.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [43](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:43>) `get_pu_weight` | `correctionlib.CorrectionSet.from_file('data/PUweight/' + year + '/' + filename)` |
| [52](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:52>) `get_met_xy_correction` | `correctionlib.CorrectionSet.from_file('data/JMESF/' + year + '/met_xy_correction.json.gz')` |
| [68](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:68>) `get_jec_correction` | `correctionlib.CorrectionSet.from_file('data/JMESF/' + year + '/jet_jerc.json.gz')` |
| [184](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:184>) `get_jec_uncertainty` | `correctionlib.CorrectionSet.from_file('data/JMESF/' + year + '/jet_jerc.json.gz')` |
| [204](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:204>) `get_fjec_correction` | `correctionlib.CorrectionSet.from_file('data/JMESF/' + year + '/fatJet_jerc.json.gz')` |
| [321](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:321>) `get_jer_sf` | `correctionlib.CorrectionSet.from_file('data/JMESF/' + year + '/jet_jerc.json.gz')` |
| [331](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:331>) `get_fjer_sf` | `correctionlib.CorrectionSet.from_file('data/JMESF/' + year + '/fatJet_jerc.json.gz')` |
| [351](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:351>) `get_mu_highpt_id_sf` | `correctionlib.CorrectionSet.from_file('data/MuonSF/' + year + '/muon_Z.json.gz')` |
| [364](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:364>) `get_mu_loose_iso_sf` | `correctionlib.CorrectionSet.from_file('data/MuonSF/' + year + '/muon_Z.json.gz')` |
| [378](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:378>) `get_mu_hlt_sf` | `correctionlib.CorrectionSet.from_file('data/MuonSF/' + year + '/muon_Z.json.gz')` |
| [395](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:395>) `get_mu_sf` | `correctionlib.CorrectionSet.from_file('data/MuonSF/' + year + '/muon_Z.json.gz')` |
| [419](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:419>) `get_mu_loose_id_sf` | `correctionlib.CorrectionSet.from_file('data/MuonSF/' + year + '/muon_Z.json.gz')` |
| [435](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:435>) `get_mu_medium_id_sf` | `correctionlib.CorrectionSet.from_file('data/MuonSF/' + year + '/muon_Z.json.gz')` |
| [451](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:451>) `get_mu_loose_miniiso_sf` | `correctionlib.CorrectionSet.from_file('data/MuonSF/' + year + '/muon_Z.json.gz')` |
| [467](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:467>) `get_mu_medium_miniiso_sf` | `correctionlib.CorrectionSet.from_file('data/MuonSF/' + year + '/muon_Z.json.gz')` |
| [483](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:483>) `get_mu_tight_id_sf` | `correctionlib.CorrectionSet.from_file('data/MuonSF/' + year + '/muon_Z.json.gz')` |
| [496](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:496>) `get_mu_tight_iso_sf` | `correctionlib.CorrectionSet.from_file('data/MuonSF/' + year + '/muon_Z.json.gz')` |
| [522](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:522>) `get_photon_id_sf` | `correctionlib.CorrectionSet.from_file('data/EGammaSF/' + year + '/photon.json.gz')` |
| [554](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:554>) `_get_photon_veto_sf` | `correctionlib.CorrectionSet.from_file('data/EGammaSF/' + year + '/photon.json.gz')` |
| [585](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:585>) `get_ele_veto_id_sf` | `correctionlib.CorrectionSet.from_file('data/EGammaSF/' + year + '/electron.json.gz')` |
| [616](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:616>) `get_ele_loose_id_sf` | `correctionlib.CorrectionSet.from_file('data/EGammaSF/' + year + '/electron.json.gz')` |
| [647](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:647>) `get_ele_medium_id_sf` | `correctionlib.CorrectionSet.from_file('data/EGammaSF/' + year + '/electron.json.gz')` |
| [678](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:678>) `get_ele_tight_id_sf` | `correctionlib.CorrectionSet.from_file('data/EGammaSF/' + year + '/electron.json.gz')` |
| [709](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:709>) `get_ele_reco_sf` | `correctionlib.CorrectionSet.from_file('data/EGammaSF/' + year + '/electron.json.gz')` |
| [742](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:742>) `get_ele_hlt_sf` | `correctionlib.CorrectionSet.from_file('data/EGammaSF/' + year + '/electronHlt.json.gz')` |
| [796](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:796>) `BTagCorrector.__init__` | `correctionlib.CorrectionSet.from_file('data/BTVSF/' + year + '/btagging.json.gz')` |
| [797](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:797>) `BTagCorrector.__init__` | `correctionlib.CorrectionSet.from_file('data/BTVSF/' + year + '/btagging.json.gz')` |
| [801](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:801>) `BTagCorrector.__init__` | `correctionlib.CorrectionSet.from_file('data/BTVSF/' + year + '/btagging.json.gz')` |
| [802](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:802>) `BTagCorrector.__init__` | `correctionlib.CorrectionSet.from_file('data/BTVSF/' + year + '/btagging.json.gz')` |
| [815](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py:815>) `BTagCorrector.__init__` | `load(filename)` |

### [autonomous_allhad/workflow/run_flat_hists_chunked.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/run_flat_hists_chunked.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [18](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/run_flat_hists_chunked.py:18>) `read_json` | `path.read_text()` |
| [24](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/run_flat_hists_chunked.py:24>) `write_json` | `temporary.open('w')` |
| [32](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/run_flat_hists_chunked.py:32>) `file_sha256` | `path.open('rb')` |
| [145](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/run_flat_hists_chunked.py:145>) `load_search_bin_contract` | `read_json(path)` |
| [604](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/run_flat_hists_chunked.py:604>) `merge_payloads` | `read_json(path)` |
| [806](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/run_flat_hists_chunked.py:806>) `completed_chunk_matches` | `read_json(path)` |
| [842](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/run_flat_hists_chunked.py:842>) `main` | `parser.add_argument('--input-list', required=True)` |
| [844](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/run_flat_hists_chunked.py:844>) `main` | `parser.add_argument('--output', required=True)` |
| [846](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/run_flat_hists_chunked.py:846>) `main` | `parser.add_argument('--campaign-year', choices=sorted(BTAG_EFFICIENCY_RELATIVE_PATHS), default='2024')` |
| [936](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/run_flat_hists_chunked.py:936>) `main` | `parser.add_argument('--search-bin-config', type=Path)` |
| [959](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/run_flat_hists_chunked.py:959>) `main` | `parser.add_argument('--allow-zero-entry-roots', action='store_true', help='Allow the streaming merger to retain explicitly recorded zero-entry ROOTs as zero-contribution inputs.')` |
| [1046](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/run_flat_hists_chunked.py:1046>) `main` | `input_list.read_text()` |
| [1114](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/run_flat_hists_chunked.py:1114>) `main.launch` | `log.open('w')` |
| [1182](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/run_flat_hists_chunked.py:1182>) `main.launch` | `subprocess.Popen(cmd, cwd=str(repo), stdout=handle, stderr=subprocess.STDOUT, env=env)` |
| [1264](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/run_flat_hists_chunked.py:1264>) `main` | `subprocess.run(merge_command, cwd=str(repo), env=env)` |

### [autonomous_allhad/workflow/build_combined_year_datacards.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combined_year_datacards.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [28](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combined_year_datacards.py:28>) `cards_by_mass` | `directory.glob(f'{prefix}*.txt')` |
| [49](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combined_year_datacards.py:49>) `read_source_manifest` | `path.read_text()` |
| [63](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combined_year_datacards.py:63>) `read_combine_runtime` | `path.read_text()` |
| [233](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combined_year_datacards.py:233>) `main` | `parser.add_argument('--left-dir', required=True, type=Path)` |
| [236](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combined_year_datacards.py:236>) `main` | `parser.add_argument('--right-dir', required=True, type=Path)` |
| [239](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combined_year_datacards.py:239>) `main` | `parser.add_argument('--output-dir', required=True, type=Path)` |
| [240](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combined_year_datacards.py:240>) `main` | `parser.add_argument('--combine-cards', default='combineCards.py')` |
| [244](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combined_year_datacards.py:244>) `main` | `parser.add_argument('--cmssw', type=Path, default=DEFAULT_CMSSW)` |
| [245](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combined_year_datacards.py:245>) `main` | `parser.add_argument('--runtime-manifest', type=Path, default=DEFAULT_RUNTIME_MANIFEST)` |
| [252](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combined_year_datacards.py:252>) `main` | `parser.add_argument('--impact-runner', type=Path, default=Path(__file__).with_name('run_asimov_impacts_eos.sh'))` |
| [307](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combined_year_datacards.py:307>) `main` | `temporary.open('w')` |
| [308](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combined_year_datacards.py:308>) `main` | `subprocess.run([args.combine_cards, f'{args.left_label}={left[mass]}', f'{args.right_label}={right[mass]}'], stdout=handle, stderr=subprocess.PIPE, text=True, check=False)` |
| [323](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combined_year_datacards.py:323>) `main` | `temporary.read_text()` |

### [autonomous_allhad/workflow/build_combine_inputs.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [370](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:370>) `read_json` | `path.read_text()` |
| [375](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:375>) `sha256` | `path.open('rb')` |
| [665](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:665>) `extract_top_level_object` | `path.open('rb')` |
| [687](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:687>) `extract_search_scheme` | `path.open('rb')` |
| [719](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:719>) `extract_component_tree` | `path.open('rb')` |
| [2124](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:2124>) `overwrite_observations` | `ROOT.TFile(str(output_root), 'UPDATE')` |
| [2233](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:2233>) `build_root` | `ROOT.TFile(str(output_root), 'RECREATE')` |
| [2704](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:2704>) `main` | `parser.add_argument('--hists', type=Path, required=True)` |
| [2705](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:2705>) `main` | `parser.add_argument('--signal-hists', type=Path, action='append', default=[], help='Canonical signal-only hists.json to overlay on --hists. Matching topology/mass samples replace the older signal; background and data always come from --hists.')` |
| [2717](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:2717>) `main` | `parser.add_argument('--campaign-year', choices=('2024', '2025'), required=True)` |
| [2719](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:2719>) `main` | `parser.add_argument('--sgamma', type=Path, required=True)` |
| [2720](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:2720>) `main` | `parser.add_argument('--rz-high', type=Path, required=True)` |
| [2721](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:2721>) `main` | `parser.add_argument('--rz-low', type=Path, required=True)` |
| [2722](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:2722>) `main` | `parser.add_argument('--zgamma-double-ratio', type=Path, required=True)` |
| [2723](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:2723>) `main` | `parser.add_argument('--exact-input', type=Path, required=True, help='Machine-derived Nb x recoil input product; used only for Low-dM physical CR yields after canonical histogram promotion')` |
| [2732](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:2732>) `main` | `parser.add_argument('--search-bin-config', type=Path, required=True, help='Adopted year-specific High-dM search-bin configuration')` |
| [2738](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:2738>) `main` | `parser.add_argument('--output-dir', type=Path, required=True)` |
| [2780](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:2780>) `main` | `read_json(args.sgamma)` |
| [2781](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:2781>) `main` | `read_json(args.rz_high)` |
| [2782](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:2782>) `main` | `read_json(args.rz_low)` |
| [2783](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:2783>) `main` | `read_json(args.zgamma_double_ratio)` |
| [2784](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:2784>) `main` | `read_json(args.exact_input)` |
| [2785](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py:2785>) `main` | `read_json(args.search_bin_config)` |

### [autonomous_allhad/workflow/plot_control_search_bins_style.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [172](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:172>) `load_json` | `path.read_text()` |
| [355](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:355>) `load_canonical_plot_payload` | `path.open('rb')` |
| [2577](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:2577>) `write_highdm_distribution_webpage` | `shutil.copy2(source, impact_dir / source.name)` |
| [2616](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:2616>) `write_highdm_distribution_webpage` | `shutil.copy2(source, target)` |
| [3166](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:3166>) `add_to_index` | `index.read_text()` |
| [3179](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:3179>) `main` | `parser.add_argument('--preview-dir', required=False, type=Path)` |
| [3180](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:3180>) `main` | `parser.add_argument('--docs-dir', type=Path)` |
| [3181](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:3181>) `main` | `parser.add_argument('--signal-searchbin-yields', default='docs/data/signal_searchbin_yields.json', type=Path)` |
| [3183](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:3183>) `main` | `parser.add_argument('--highdm-distributions', type=Path)` |
| [3185](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:3185>) `main` | `parser.add_argument('--summary-2024', type=Path)` |
| [3186](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:3186>) `main` | `parser.add_argument('--summary-2025', type=Path)` |
| [3187](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:3187>) `main` | `parser.add_argument('--flat-summary-2024', type=Path)` |
| [3188](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:3188>) `main` | `parser.add_argument('--flat-summary-2025', type=Path)` |
| [3189](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:3189>) `main` | `parser.add_argument('--web-dir', type=Path)` |
| [3190](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:3190>) `main` | `parser.add_argument('--impact-png', type=Path)` |
| [3191](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:3191>) `main` | `parser.add_argument('--impact-pdf', type=Path)` |
| [3192](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:3192>) `main` | `parser.add_argument('--impact-json', type=Path)` |
| [3193](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:3193>) `main` | `parser.add_argument('--result-manifest', type=Path)` |
| [3194](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:3194>) `main` | `parser.add_argument('--flat-hists', type=Path)` |
| [3195](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:3195>) `main` | `parser.add_argument('--flat-output-dir', type=Path)` |
| [3196](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:3196>) `main` | `parser.add_argument('--dy-rz-manifest', type=Path)` |
| [3218](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:3218>) `main` | `parser.add_argument('--search-bin-config', type=Path, help='Apply the configured final High-dM bin merges before plotting')` |
| [3295](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py:3295>) `main` | `shutil.copy2(outbase.with_suffix(suffix), plot_dst / outbase.with_suffix(suffix).name)` |

### [autonomous_allhad/workflow/plot_highdm73_from_limit_templates.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_highdm73_from_limit_templates.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [66](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_highdm73_from_limit_templates.py:66>) `main` | `parser.add_argument('--templates', required=True, type=Path, nargs='+', help='one or more same-binning yearly template ROOT files to sum')` |
| [73](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_highdm73_from_limit_templates.py:73>) `main` | `parser.add_argument('--bin-map', required=True, type=Path)` |
| [74](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_highdm73_from_limit_templates.py:74>) `main` | `parser.add_argument('--output', required=True, type=Path)` |
| [79](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_highdm73_from_limit_templates.py:79>) `main` | `args.bin_map.read_text()` |
| [92](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_highdm73_from_limit_templates.py:92>) `main` | `uproot.open(path)` |

### [autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py>)

| 줄·함수 | 참조 표현식 |
|---|---|
| [54](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:54>) `sha256` | `path.open('rb')` |
| [144](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:144>) `background_fractions` | `manifest_path.read_text()` |
| [146](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:146>) `background_fractions` | `uproot.open(score_root)` |
| [193](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:193>) `signal_fractions` | `uproot.open(score_root)` |
| [467](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:467>) `write_gnn_cards` | `path.read_text()` |
| [479](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:479>) `main` | `parser.add_argument('--hists', type=Path, required=True)` |
| [481](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:481>) `main` | `parser.add_argument('--campaign-year', choices=('2024', '2025'), required=True)` |
| [483](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:483>) `main` | `parser.add_argument('--sgamma', type=Path, required=True)` |
| [484](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:484>) `main` | `parser.add_argument('--rz-high', type=Path, required=True)` |
| [485](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:485>) `main` | `parser.add_argument('--rz-low', type=Path, required=True)` |
| [486](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:486>) `main` | `parser.add_argument('--zgamma-double-ratio', type=Path, required=True)` |
| [487](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:487>) `main` | `parser.add_argument('--search-bin-config', type=Path, required=True)` |
| [488](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:488>) `main` | `parser.add_argument('--background-scores', type=Path, required=True)` |
| [489](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:489>) `main` | `parser.add_argument('--signal-scores', type=Path, required=True)` |
| [490](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:490>) `main` | `parser.add_argument('--gnn-selection', type=Path, required=True)` |
| [491](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:491>) `main` | `parser.add_argument('--gnn-checkpoint', type=Path, required=True)` |
| [492](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:492>) `main` | `parser.add_argument('--campaign-manifest', type=Path, required=True)` |
| [493](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:493>) `main` | `parser.add_argument('--output-dir', type=Path, required=True)` |
| [507](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:507>) `main` | `args.gnn_selection.read_text()` |
| [525](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:525>) `main` | `canonical.read_json(args.sgamma)` |
| [526](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:526>) `main` | `canonical.read_json(args.rz_high)` |
| [527](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:527>) `main` | `canonical.read_json(args.rz_low)` |
| [528](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:528>) `main` | `canonical.read_json(args.zgamma_double_ratio)` |
| [529](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py:529>) `main` | `canonical.read_json(args.search_bin_config)` |

## 부록 C. 현재 소스 식별

현재 파일의 SHA256을 기록한다. import 추적은 아래 파일 내용에 대한 결과다. tracked 공용 코드에 다른 작업의 미커밋 변경이 있으므로 HEAD만으로 현재 동작이 전부 재현된다고 가정하면 안 된다.

| 코드 | SHA256 | Git |
|---|---|---|
| [autonomous_allhad/gnn_lowdm/__init__.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/__init__.py>) | `14479d385249cd4d2fed945ec4328ba86f6532029f50563523ba9bcf8b8d0a32` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/_implementation/__init__.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/__init__.py>) | `8c8c1e384b950193ebfd94834a219618cc2872111955dca10cc9cfeecfcfa57a` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/_implementation/analyze_diagonal_v3_global_shap.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/analyze_diagonal_v3_global_shap.py>) | `3170a0aa16526f6c17d3ca31eab0b64f5cbec65b308c992a21fcee70c079f27a` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_diagonal_v3_cr_nnout_partial.py>) | `b6ab837ea8da7300231fc2a95ea4cb7b069dbac9ba67c9c85e2afc6404ad2170` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_expanded_feature_cache.py>) | `4a11efbbd0dd10cfcc4743ff0089eb93e97d63d780e0172320c2f6e3a5b9a436` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/_implementation/build_full_campaign_manifest.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/build_full_campaign_manifest.py>) | `a7325b19b5c04b4687e26f0d23930000bb0afb5a4a75e8ab0d0aceb0e04de800` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/_implementation/cli.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/cli.py>) | `b37a872992c8716aa56ade366180f2a5114b022e3bf687d4ed54f2a7e25fc7b2` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/_implementation/collect_plot_diagonal_v3_limits.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/collect_plot_diagonal_v3_limits.py>) | `ec0730cd31354773f2cdbd679d15f09b25867d175f09ad4e9835cd3f35b200d5` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/_implementation/collect_plot_highdm79minus6_lowdm30.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/collect_plot_highdm79minus6_lowdm30.py>) | `5c8a18ebc3c4ff343c9b647d36a83704aa66c8f146f99151c378a9fc5cdb04bb` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/_implementation/compare_oof.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/compare_oof.py>) | `f807c44ec7903df02734efc5b3e9d6689ef4bec062b719807dd5f2b04863ffa0` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/_implementation/diagonal_v3_region_features.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/diagonal_v3_region_features.py>) | `93fb8ae4b7788ecfcf9c74f5b31939fcf87efbc9dc5b079a2b415e51ad271904` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_all_signal_template_partial.py>) | `b28f5558aa4b73e2dbc251e4b28a5a9407399de9d66c2cb7cc196ff0ce748841` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/evaluate_diagonal_v3_test.py>) | `fe6e4bcc8d242aa97fae9fe31066f3f50fbeaa26b042f0036b6745aecaca00a4` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/_implementation/finalize_diagonal_v3_cache.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/finalize_diagonal_v3_cache.py>) | `90a24cf3c00ddc62f20ee2ca08add687c5b30e10684479f1708b785cab9d47a7` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/_implementation/merge_all_signal_templates.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_all_signal_templates.py>) | `194f076d3200285ee2b2e8f863a1059b1538a69119ed0090da7920e0efbf9b84` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/_implementation/merge_diagonal_v3_cr_nnout_partials.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/merge_diagonal_v3_cr_nnout_partials.py>) | `8350a39f51256f04a18a0734701a61560781066fa68fdf09665c6cca8bd6093b` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/_implementation/physics_informed_v2.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/physics_informed_v2.py>) | `123db98825a55905bb6c748d237ed098d25c5c3c4107cc1752b98c3389400d0a` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/_implementation/plot_control_regions.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_control_regions.py>) | `99251a1f9cbb5c770a474151785e9ad4da88b96cf4e5220814c55918c35eaab4` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/_implementation/plot_diagonal_v3_model_schematic.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_diagonal_v3_model_schematic.py>) | `4ffe615f8aabbab1192c0d374d750369d4d8088a1770320711d18729d326774e` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/_implementation/plot_full_gnn_results.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_full_gnn_results.py>) | `43965a4232b9ffcd0e6e5a3989d06220e0f4c55398090874583ee768b37c777a` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/_implementation/plot_hyperparameter_losses.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_hyperparameter_losses.py>) | `70413cd8600a442f6327cf45c0871edb0c0d2b3895e07e753ad2f2f4e276ef02` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/_implementation/plot_lowdm_category_nn_out.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/plot_lowdm_category_nn_out.py>) | `d316f6b8e9940a5aa25e03deb3a0724b6e50d9d6308d0db9153898ef898a9412` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/_implementation/prepare_all_signal_template_condor.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_all_signal_template_condor.py>) | `2ba656eeab477ef864280f1df655d2a4204f0dcd179071ef955379efc398f7a8` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/_implementation/prepare_diagonal_v3_cache_condor.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_diagonal_v3_cache_condor.py>) | `2b4d2ebb82bffcab14a31cc7dc1a12f2e09958ef757f9e08781145b3506b9dc4` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/_implementation/prepare_highdm79minus6_lowdm30_condor.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/prepare_highdm79minus6_lowdm30_condor.py>) | `e9bd3a097a2d466f9db2d08b1500edce2609e14887da47d7b90b60a21b25a50c` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/_implementation/publish_lowdm_gnn_web.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/publish_lowdm_gnn_web.py>) | `d4e0c724a280604829ce81e834ba3996edfc1c42d0a8adf5ebfd1d17daffe287` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/_implementation/rank005_numpy.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/rank005_numpy.py>) | `d4d8fa8aa9191c2bd412c7b06d05fa2f0e42b7454c61590e5e6dcc2eb2fab870` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/_implementation/region_io.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/region_io.py>) | `021db7bf75acff8ac2f43e2df1c045117c342b02a280dd70de38b63f3ba42d97` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/_implementation/significance.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/significance.py>) | `8bf78637d8b000022fc930da7851c7285f4f4e28d6f6f80bd17b15c5b629f139` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/_implementation/supplementary_plots.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/supplementary_plots.py>) | `8de78930409a233c97e3089503b61f68235a0cc8db3032d391b3b5c6504780ea` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_diagonal_v3_significance.py>) | `2c4413117158d5d6e089fafe8399dfbc28901995dff425200ef6a8d87c7e15ab` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/_implementation/train_oof.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/train_oof.py>) | `d1e98bedbb128236573ecaaeb8b9838e7358d3a3a66c1b9c92d3342a9615d0bc` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/tune_full_gnn.py>) | `2757e33ee2cae026af75f635535a8faec931b85d1d8f2ab1751d1c8870d33161` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_numpy_inference.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_numpy_inference.py>) | `a09e5872b29b059bd3d1d5d03ed63ab4139ad96e3a37a87b46fe1810a1550cf0` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_region_feature_parity.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_region_feature_parity.py>) | `3132eed5e1855d454606ace324e1c703b45f4153c220d2ea73bda2b54b080b2e` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_srcr_inputs.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/validate_diagonal_v3_srcr_inputs.py>) | `433b83c0db040dc9a5139a4627d25d97bf8dbfb0604d197f487e3e12447fbb01` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/build_datacard.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/build_datacard.py>) | `be7d93ce774b2d435cb3287101524fa4accb9e308e992a02418dca6a5b1f09d6` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/data.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/data.py>) | `0c25d4edecb62816e2cc3defc680f715602f4159ba8b7c008a446de362a4e360` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/eval.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/eval.py>) | `7ff6fc78b5e74955953c56903f571171735de2afb8a5aaaed40652cfbec8a046` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/merge_datacard.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/merge_datacard.py>) | `79b48225e3befb85b379ea6e56c0b002b3a0ebb923ab97b6d5bf393f2807c7d1` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/model.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/model.py>) | `c0f63495319e3213c842c98e244dc7fd1c1433c684e181613bbec19b3d75bded` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/plotting.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/plotting.py>) | `1967c96fc2367aca53966485a8d64e8d8708fe77b1dc0efcc7c8cd7aa3de5516` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/train.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/train.py>) | `d2b8cec670b0df2b1b826ab1a01ca571a5b31616668e6de9e1e3b70c4e60173c` | `clean tracked` |
| [autonomous_allhad/gnn_lowdm/valid.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/valid.py>) | `c145f5e6710de69d21c60ff69a757f37127d0dba646879a41cfd0df56650fefc` | `clean tracked` |
| [autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py>) | `bc69498dc430a3c581f58efc76829a21917d90986a300d6cb0e23d35a8ad814e` | `M autonomous_allhad/workflow/build_flat_boosted_recoil_hists.py` |
| [autonomous_allhad/workflow/postprocess_limits.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/postprocess_limits.py>) | `77f0ebdf12794b50efa5d76dbc56e570268451cd7b004299b62fad2e46256221` | `M autonomous_allhad/workflow/postprocess_limits.py` |
| [autonomous_allhad/autonomous_allhad/real_subset_worker.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/real_subset_worker.py>) | `3c76084c7f2ebf8c28e9b1c75aa418a784fb065098cb8033a7dfed3370dc63a7` | `M autonomous_allhad/autonomous_allhad/real_subset_worker.py` |
| [autonomous_allhad/autonomous_allhad/analysis_scale_factors.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/analysis_scale_factors.py>) | `78b0e034c083fade6d1b225e37b12ecae94966297d2e33d5c5823531281aa0b8` | `clean tracked` |
| [autonomous_allhad/autonomous_allhad/signal_models.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/signal_models.py>) | `ad15ac4b4eb19e4f3ea8cd9c9154beea40550cc4946851df53e27884657e0d09` | `clean tracked` |
| [autonomous_allhad/autonomous_allhad/highdm_resolved_categories.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/highdm_resolved_categories.py>) | `10d46148be4589ebb525113433d6d8c63ffe7864d7f66dd043cb83674ee22ae1` | `clean tracked` |
| [autonomous_allhad/autonomous_allhad/dy_ptll_policy.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/dy_ptll_policy.py>) | `9095f026760fa976404a5735dc3adcb515bb42d9043da91293dd706f5a9e9ee7` | `clean tracked` |
| [autonomous_allhad/autonomous_allhad/search_bin_categorization.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/search_bin_categorization.py>) | `319cc061f56895ec26727e39bb6785047f1cdb135bcb4c50b7537f66af1b98ff` | `M autonomous_allhad/autonomous_allhad/search_bin_categorization.py` |
| [autonomous_allhad/autonomous_allhad/sidecar_store.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/autonomous_allhad/sidecar_store.py>) | `de4750a92635a1c256af59dbcaabbf340ae62c2ce175263e507dcad616279595` | `clean tracked` |
| [analysis/utils/corrections.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/analysis/utils/corrections.py>) | `c8ecd046246c68704fac78753fcc1f055e0264f3f5f015b5a05af138cef15ce0` | `clean tracked` |
| [autonomous_allhad/workflow/run_flat_hists_chunked.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/run_flat_hists_chunked.py>) | `8de254888b2123c039d57d666bb083792e7792678c2edf284e4f94bc46737760` | `M autonomous_allhad/workflow/run_flat_hists_chunked.py` |
| [autonomous_allhad/workflow/build_combined_year_datacards.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combined_year_datacards.py>) | `8acadcb4f1b3e1238dd9cf4412f2ef2f9afe8bb4778ea6e83a634041c723b0f9` | `M autonomous_allhad/workflow/build_combined_year_datacards.py` |
| [autonomous_allhad/workflow/build_combine_inputs.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_combine_inputs.py>) | `7845a9152b68057ea15fe427b999245e4bbd3ba3a36b5d3e4390adebb4e04a05` | `M autonomous_allhad/workflow/build_combine_inputs.py` |
| [autonomous_allhad/workflow/background_process_groups.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/background_process_groups.py>) | `afe5af0d1dc478f3294fc236b7083346e5cf3fdc0762bcbaf7eb0e2f04d8e419` | `clean tracked` |
| [autonomous_allhad/workflow/plot_control_search_bins_style.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_control_search_bins_style.py>) | `bf3e2b388916d0431b510876f4216ca72aabf86103c1453c05ec1fccc4dd0324` | `M autonomous_allhad/workflow/plot_control_search_bins_style.py` |
| [autonomous_allhad/workflow/plot_highdm73_from_limit_templates.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/plot_highdm73_from_limit_templates.py>) | `a40810a5203915d7b4ce7f96584c855eab0c8f8095982ea994c59b1448dd149c` | `?? autonomous_allhad/workflow/plot_highdm73_from_limit_templates.py` |
| [autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py>) | `f2bc33c76e79e03463ec0e05aee3e946eb7327691df16d91533e1b9d16bd8526` | `?? autonomous_allhad/workflow/build_lowdm_gnn30_combine_inputs.py` |
| [autonomous_allhad/gnn_lowdm/_implementation/run_highdm79minus6_lowdm30_bundle.sh](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/_implementation/run_highdm79minus6_lowdm30_bundle.sh>) | `6338467f81cb49afd368bca38db5327337e7694bdaa756f79bb766541acc37c8` | `clean tracked` |

## 부록 D. 배포·보존해야 하는 연결과 추가 확인 결과

| 경로·대상 | 읽는 주체 | 구분 |
|---|---|---|
| `/eos/user/t/taiwoo/decaf/analysis/CombinedArea/CMSSW_14_1_0_pre4` | `run_highdm79minus6_lowdm30_bundle.sh` | shell에 고정된 CMSSW·Combine 설치 경로; EOS 실재성 미확인 |
| `$CAMPAIGN/runtime/py38/bin/python3` 및 `lib/` | `prepare_all_signal_template_condor.py`가 생성하는 shell | 사전 설치 필요. 현재 preparer는 `py38.tgz` 자체를 읽거나 배포하지 않음 |
| `$CAMPAIGN/vendor/mt2/mt2/__init__.py` | 동일 signal worker shell | `mt2.__version__ == 1.2.0` 강제 검사 |
| `$CAMPAIGN/model/diagonal_v3_numpy.npz` | signal request → NumPy evaluator | staging의 model 디렉토리는 만들지만 파일을 복사하는 단계는 없음 |
| `$CAMPAIGN/config/{config.json,stop_xsec_13p6TeV.json,full_campaign_manifest_2024.json}` | signal request → evaluator | 모두 사전 배치 필요 |
| `analysis/data/` 전체 | `real_subset_worker.local_analysis_dir()` | 기본 runtime staging은 디렉토리 전체를 복사. 본문 보정표는 그중 물리 함수의 직접 소비 파일 |
| `analysis/hists/btageff*.merged` | 동일 staging과 `BTagCorrector` | 전체 파일을 staging하고 2024/2025 파일 내 dataset key로 효율 조회 |
| `autonomous_allhad/workflow/sidecars_main.sqlite` | `sidecar_store.read_root_metadata()` | 공용 flat worker의 JSON 대체 metadata store. `AUTONOMOUS_ALLHAD_SIDECAR_STORE`로 경로 지정 가능 |
| intermediate shard와 동일 stem `.json` | GNN manifest, cache/CR worker | GNN 자체는 JSON을 직접 읽는 경로가 남아 있음. upstream이 SQLite만 남기는 경우 공용 fallback이 자동 적용되지 않음 |
| cache shard `mc_cache_*.root`, `signal_cache_*.root` + 동일 stem `.json` + `campaign_state.json` | 학습·test·SHAP 입력 | 현재 기본 보존 폴더는 아래 경로. 입력을 재구축하려면 원 intermediate ROOT 및 TROTA provenance가 필요 |

현재 보존 cache 폴더: [inputs/gnn_lowdm_diagonal_v3_cache_20260831](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/inputs/gnn_lowdm_diagonal_v3_cache_20260831>).

request 보존 폴더: [inputs/gnn_lowdm_diagonal_v3_cache_20260831_requests](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/inputs/gnn_lowdm_diagonal_v3_cache_20260831_requests>).

manifest 보존 폴더: [inputs/manifests](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/inputs/manifests>).

ROC와 SHAP의 기본 NPZ 파일은 현재 로컬에 존재한다. NPZ ZIP member 목록만 읽어 확인한 결과, ROC 파일의 key는 `test_indices`, `scores`, `labels`, `category`다. **새 ROC 함수가 요구하는 `weights`, `signal_topology_id`가 없어 기본 파일로는 재플롯할 수 없다.** SHAP 파일에는 `shap_values`, `feature_values`, `labels`, `signal_topology_id`, `feature_names`, `event_hash`가 있어 플로터의 필수 key를 갖췄다. 이 점검에서 event 값 자체를 다시 평가하지 않았다.

보존 테스트 코드는 [test_physics_informed_v2.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/tests/test_physics_informed_v2.py>)와 [test_trota_provenance.py](</Users/taiwoomac/Documents/All Hadronic Stop Analysis/autonomous_allhad/gnn_lowdm/tests/test_trota_provenance.py>)다. 전자는 `data.py`, `model.py`, `physics_informed_v2.py`, `significance.py`, `build_expanded_feature_cache.py`와 numpy/awkward/torch/uproot에, 후자는 `region_io.py`와 pytest에 의존한다. 생산 runtime import와 별도로 분류한다.

보고서 작성 중 변경한 분석 코드·모델·보정 파일: 없음. 생성한 repository 산출물: 이 보고서 1개.
