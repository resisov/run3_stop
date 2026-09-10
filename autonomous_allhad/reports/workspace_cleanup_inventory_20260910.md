# 워크스페이스 정리 대상 전수 조사

작성 기준: 2026-09-10, 로컬 작업 트리. HEAD `561c2721`, 브랜치 `codex/package-lowpt-tnp`.

조사 범위는 로컬 작업 트리와 `.git` 오브젝트 스토어다. EOS와 CVMFS는 읽지 않았다.
**이 조사 과정에서 어떤 파일도 삭제·이동·수정하지 않았다.** 아래는 실행 지시가 아니라
분류와 근거이며, 각 항목의 위험도와 복구 가능성을 함께 적었다.

기계 판독본: [workspace_cleanup_inventory_20260910.json](workspace_cleanup_inventory_20260910.json).
경로는 모두 레포 상대 경로다.

---

## 0. 요약

워크스페이스 총 **6.9 GB**. 그중 추적되는 실제 소스와 데이터는 **185 MB**뿐이다.

| 구획 | 크기 | 파일 수 |
|---|---|---|
| `.git` | **5.20 GB** | pack 15개 4.87 GB |
| 작업 트리 tracked | 185 MB | 1,042 |
| 작업 트리 untracked | **1.50 GB** | 10,894 |
| 작업 트리 ignored | 147 MB | 1,559 |

`.git`의 고유 blob 내용 합은 **1.78 GB**다. pack이 15개로 쪼개져 있어 나머지 약 3.1 GB는
pack 간 중복이며, 히스토리 재작성 없이 회수된다.

무손실·저위험 구간에서 회수 가능한 양은 약 **3.5 GB**, EOS 대조를 전제로 하면 추가로
약 **0.8 GB**다.

크기 표기는 두 출처를 섞어 쓴다. `.git`과 디렉토리 총량은 `du`(KiB 기준) 값이고,
캠페인 산출물 표는 실제 파일 바이트 합(십진 MB)이다. 표마다 출처를 밝혔다.

---

## 1. 선행 조건 (이것부터 정하지 않으면 청소를 시작하지 말 것)

### B1 — 미커밋 삭제 2,367개 · 우선순위 높음

작업 트리에 `-3,004,062` 라인 규모의 **커밋되지 않은 삭제**가 있다.

```
git status --porcelain | grep -c '^ D'   ->  2367
git diff --stat HEAD                     ->  2387 files changed, 883 insertions(+), 3004062 deletions(-)
```

| 삭제된 경로 | 파일 수 |
|---|---|
| `docs/nominal_plots_2024_fullselection_v5_dyexclusive_t2models_freebkg_20260728` | 613 |
| `docs/highdm_distributions_2024_2025_20260720` | 584 |
| `docs/lowdm_gnn_20260901` | 466 |
| `autonomous_allhad/reports` | 550 |
| `autonomous_allhad/workflow` | 51 |
| `autonomous_allhad/lowpt_tnp` | 9 |
| `.github/workflows/lowpt-tnp.yml` | 1 |

추적 파일의 unstaged 삭제이므로 `git restore -- <path>`로 되돌릴 수 있다.
두 가지를 확인해야 한다.

1. **브랜치명이 `codex/package-lowpt-tnp`인데 `autonomous_allhad/lowpt_tnp/`가 통째로 삭제되어 있다.** 의도된 것인지 확인이 필요하다.
2. `docs/`의 플롯 갤러리가 사라져 [docs/index.html](../../docs/index.html)이 참조하는 `plots/limits/*`, `plots/search_bins/*`, `lowdm_gnn_20260901/`이 모두 없다. 작업 트리 기준으로 사이트가 깨져 있다. 다만 Pages 배포는 `master` push에만 걸려 있어(`.github/workflows/pages.yml`) 실제 배포본은 영향받지 않았다.

### B2 — AN 참조 문서의 유일본이 gitignore 안에 있다 · 우선순위 높음

AGENTS.md가 지정한 4대 참조 문서 중 `AN2019_016_v9.pdf`가 **`tmp/pdfs/`에만 존재**한다.
`.gitignore`의 `/tmp/` 규칙에 걸려 추적되지 않는다.

```
find . -name AN2019_016_v9.pdf -not -path './.git/*'
  ->  ./tmp/pdfs/AN2019_016_v9.pdf        (7.0 MB, untracked)
```

`tmp/`를 청소하면 소실된다. 나머지 세 참조 파일은 정상적으로 추적된다
(`analysis/processors/stop_processor_v4.py`, `analysis/utils/ids.py`, `analysis/utils/corrections.py`).

### B3 — `output/` 정책과 코드가 어긋난다 · 우선순위 중간

`.gitignore`는 `/output/`을 "repository-root scratch"로 규정하고 캐노니컬 산출물은
`autonomous_allhad/reports`, `results`, `validation`, `docs` 아래에 두라고 명시한다.
그런데 활성 코드의 **기본 출력 경로**가 `output/`이다.

- [build_analysis_sf_physics_report.py:40](../workflow/build_analysis_sf_physics_report.py) — `OUTPUT = REPO / "output/pdf/analysis_sf_measurements_2024_physics_report.pdf"`
- [plot_diagonal_v3_model_schematic.py:387](../gnn_lowdm/_implementation/plot_diagonal_v3_model_schematic.py) — 기본 `--output` 3종이 모두 `output/` 아래

출력 경로를 옮기든 정책 문구를 고치든 한쪽으로 통일해야 한다.

### B4 — Top/W 효율 payload가 추적되지 않는다 · 우선순위 중간

`.gitignore`가 `analysis/hists/*`를 차단하고 `btageff2024.merged`만 예외 허용한다.
그 결과 `topwtageff2024.merged`와 `topwtageff2025.merged`(각 2.2 MB)가 untracked다.
재생성 비용이 큰 payload이므로 예외를 추가하거나 EOS 사본을 확인한 뒤에 청소해야 한다.

### B5 — `fast_analysis/`는 삭제 금지

import 0건이고 2026-07-24 이후 미변경이지만, AGENTS.md의 **Architecture B 프로토타입**이다.
완료 조건에 "최소 3개 아키텍처 구현 또는 유의미한 프로토타입"이 걸려 있으므로 삭제하면
그 근거가 사라진다. 격리(`archive/`)까지만 검토한다.

---

## 2. 무손실·저위험 — 약 3.5 GB

| ID | 대상 | 회수 | 위험 | 근거 |
|---|---|---|---|---|
| T1 | `git gc --prune=now` | **약 3.1 GB** | 없음 | pack 15개 4.87 GB vs 고유 blob 1.78 GB. 히스토리 재작성 아님 |
| T2 | `.codex-worktrees/signal-histogram-limits/` | **340 MB** | 낮음 | `.git`이 없고 worktree 미등록. 레포 전체의 맨 복사본 잔해 |
| T3 | `git worktree prune` | 메타만 | 없음 | `/private/tmp/` 아래 사라진 worktree 4개 등록 잔존 |
| T4 | `__pycache__`(22) · `.pytest_cache`(4) · `.DS_Store`(17) | 6.2 MB | 없음 | 재생성됨 |

### T2 상세

```
ls -ld .codex-worktrees/signal-histogram-limits/.git   ->  No such file or directory
git worktree list | grep -c signal-histogram-limits    ->  0
du -sk .codex-worktrees/signal-histogram-limits        ->  356616 KB
```

worktree가 아니라 `.git`이 아예 없는 복사본이다. 내부 `docs/` 하나가 268 MB를 차지한다.
과거에 커밋된 적이 있어 히스토리에도 231 MB(3,676 obj)가 남아 있다(T14 참조).

등록된 worktree는 두 개이며 둘 다 유효하다.

| 경로 | 브랜치 | HEAD | dirty |
|---|---|---|---|
| `.codex-worktrees/global-tnp-signed-eta` | `codex/global-tnp-signed-eta` | 2026-09-10 `4716ab12` | 0 |
| `.codex-worktrees/merge-highdm-nb2w2` | `codex/workspace-consolidation` | 2026-08-26 `8de4a158` | 42 |

`merge-highdm-nb2w2`는 299 MB이고 디렉토리명과 브랜치명이 다르다. 42개 미커밋 변경이
있으므로 처리 전에 확인이 필요하다.

### T4 상세

루트 `.pytest_cache/`에 테스트 임시 디렉토리 5개가 남아 있다:
`dy_window20_cli_fix_20260907`, `dy_window20_merge_fix_20260907`,
`dy_window20_pilot_fix_20260907`, `dy_window20_requests_20260907`,
`dy_window20_requests_20260907_run2`.

---

## 3. 캠페인 산출물 — 약 945 MB, 거의 전부 untracked

`autonomous_allhad/workflow/histograms/lepton_veto10_20260908/`
(아래 표는 실제 파일 바이트 합, 십진 MB. 괄호는 파일 수.)

| 하위 | tracked | untracked | 비고 |
|---|---|---|---|
| `cards/` | 0 | **684.0 MB** (1,860) | `manifest.json` 6개만 207 MB, datacard `.txt` 1,848개 |
| `plots/` | 47.9 MB (8) | 77.1 MB (1,016) | 아래 T6 참조 |
| `combined/` | 1.9 MB (30) | 49.7 MB (119) | limit 산출물 + `condor_logs` |
| `2024/` `2025/` | 0 | 61.3 MB (611) | `plot_payload.json` 23 + 20 MB |
| `combined_pilot/` | 0 | 8.5 MB (1) | pilot 잔재 |
| `plot_validation/` | 0 | 6.7 MB (20) | 검증 스크래치 |
| `cards_pilot_v2/` | 0 | 5.4 MB (12) | pilot 잔재 |
| `card_unit_tests/` | 0 | 1.2 MB (441) | `.gitignore` 차단 대상 |
| `impact_step_unit_tests/` | 0 | 0.03 MB (469) | `.gitignore` 차단 대상 |
| `plot_cache/` | 0 | 0.1 MB (1) | `.gitignore` 차단 대상 |
| `impact_code_*` | 7 (각 1파일) | 0 | 진단 잔재 |

`.gitignore`는 `hists.json`, `plot_payload.json`, `hist_chunks/`, `card_unit_tests/`,
`impact_step_unit_tests/`, `plot_cache/`를 이미 차단하도록 작성되어 있다. **git이 무시할 뿐
디스크에서는 계속 증가하는 상태**다.

### T5 — 스크래치 · 약 46 MB · 위험 낮음

`plots/pdf_validation`(19.2 MB, PNG 500장 — 같은 그림의 래스터 사본), `combined_pilot`,
`plot_validation`, `cards_pilot_v2`, `card_unit_tests`, `impact_step_unit_tests`,
`plot_cache`, `impact_code_*` 7개. pilot과 unit-test 산출물이며 캐노니컬 결과가 아니다.

### T6 — 플롯 중복 · 37.5 MB · 결정 필요

`plots/cr_sr_plots_2024_2025.zip`(37.5 MB, 517 files)이 `plots/2024`(24.7 MB) +
`plots/2025`(24.5 MB) + `plots/2024_2025`(0.6 MB)와 같은 내용이다.
**zip 쪽이 tracked이고 낱개본이 untracked**라서 히스토리에는 이미 40 MB blob이 두 벌 들어가 있다.
둘 중 무엇을 정본으로 삼을지 결정해야 한다.

### T7 · T8 — EOS 대조 후에만 · 약 795 MB · 위험 높음

`cards/`(684 MB)와 `2024/` `2025/` `combined/`(111 MB)는 tracked 사본이 거의 없다.
AGENTS.md는 ROOT 입력과 fit 산출물을 EOS에 두고 로컬은 compact export만 쓰도록 규정하므로
원칙적으로 로컬에서 제거 대상이지만, **EOS 사본 존재를 확인하기 전에는 삭제하면 복구할 수 없다.**

---

## 4. 죽은 코드

| ID | 경로 | 크기 | 최종 커밋 | 판정 |
|---|---|---|---|---|
| T9 | `automation/` (5 py) | 196 KB | 2026-06-23 `a124351d` | import 0. 삭제 또는 archive |
| T10 | `condor/` | 80 KB | 2026-07-24 `7fc9ea96` | 레거시 submit. 삭제 또는 archive |
| T11 | `fast_analysis/` (21 py) | 184 KB | 2026-07-24 `7fc9ea96` | **삭제 금지 — B5** |
| T12 | `analysis/distribution_draw_v5.py` | 34 KB | 2026-07-24 | `analysis/plotting/`판과 갈라진 복제본 |
| T13 | `analysis/libs/` | 24 KB | 2026-06-17 `1707e978` | 참조 확인 후 처리 |
| T13 | `analysis/metadata/backup/` | — | 2026-05-26 `c42897fa` | 참조 확인 후 처리 |
| T13 | `configs/stop_2024.yaml` | 4 KB | 2026-06-18 `8ffc8d85` | `spec/baseline_dataflow.md`와 `fast_analysis` 문서에서만 참조 |
| T13 | `setup_condor.sh`, `signal_xsec.txt`, `github-profile-banner.png` | 534 KB | 2026-07~08 | 루트 잔여물 |

T12의 두 파일은 md5가 다르다(`0c4803cb…` vs `29166508…`). 루트본 2026-07-24,
`plotting/`본 2026-08-12. 통합 전에 diff를 확인해야 한다.

### `analysis/`는 죽지 않았다

전체를 레거시로 취급하면 안 된다.

- `analysis/processors/` 2026-09-08 `cc0284da`, `analysis/plotting/` 2026-09-09 `486d24f5` 로 최근까지 수정됨
- [run_flat_hists_chunked.py:38](../workflow/run_flat_hists_chunked.py)의 `EXECUTION_CONTRACT_COMMON_PATHS`가 `analysis/utils/corrections.py`, `analysis/data/corrections.coffea`, `analysis/utils/ids.py`, `analysis/data/ids.coffea`를 sha256으로 고정해 배치 payload로 배포한다
- `EXECUTION_CONTRACT_YEAR_PATHS`가 연도별 correctionlib JSON 8종(2024) / 7종(2025)을 같은 방식으로 고정한다
- `analysis/hists/btageff2024.merged`는 sha256이 코드에 하드코딩된 필수 입력이다

---

## 5. 기록 보존물 — `reports/` 100 MB, 대부분 유지

| 경로 | 크기 | 판정 |
|---|---|---|
| `background_estimation_lepton_veto10_20260908/` | 84 MB (tracked) | **유지.** README가 가리키는 현행 배경추정 입력. `tf_inputs.json` 2개가 61 MB이므로 이 둘만 EOS 이관 검토 가능 |
| `background_estimation_plots_2024_2025_20260908/` | 9.4 MB (tracked) | 유지 |
| `topw_efficiency_an_handoff/` | 2.6 MB | 유지 — AN 핸드오프 근거 |
| `gnn_cms_an_handoff_20260909/` | 1.5 MB | 유지 |
| `object_definitions_20260909/` | 1.3 MB | 유지 |
| `trigger_sf_2025_an_handoff/`, `trigger_sf_2024_an_handoff/` | 2.0 MB | 유지 |
| `higgsino_relic_contours_20260902/` | 848 KB | 유지 |
| `lowdm_canonical_dependencies.md` | **320 KB 단일 마크다운** | 사람이 읽기엔 과대. 기계 판독용 JSON 분리 권장 |

이 구획은 크기 대비 가치가 높다. 일괄 삭제 대상이 아니다.

---

## 6. git 히스토리 — 별건, 승인 필요

고유 blob **1.78 GB** 중 **1.22 GB(29,092 obj)가 `docs/`의 날짜별 플롯 갤러리 16종**이다.

확장자별: PNG 13,890개 **1.40 GB**, PDF 18,689개 158 MB, JSON 1,656개 96 MB, ZIP 3개 88 MB.

| 경로 | 히스토리 크기 |
|---|---|
| `docs/highdm_distributions_2024_2025_20260720` | 303 MB |
| `docs/nominal_plots_..._freebkg_20260728` | 272 MB |
| `.codex-worktrees/signal-histogram-limits` | 231 MB |
| `docs/partial_merge_preview_sf_unc_v2_..._20260622` | 154 MB |
| `docs/nominal_plots_..._lowdm34_..._20260727` | 110 MB |
| `docs/highlowdm_full_20260705` | 79 MB |
| 나머지 갤러리 10종 | 약 230 MB |

단일 최대 blob은 `cr_sr_plots_2024_2025.zip`이 40 MB짜리로 **두 벌** 들어가 있는 것이다.

`git gc`로는 줄지 않는다. `git filter-repo` + force-push가 필요하며, AGENTS.md가
protected 브랜치 force-push를 금지하므로 **사용자 승인 없이 진행하지 않는다.**
`.codex-worktrees/`가 히스토리에 들어간 것은 worktree 디렉토리를 실수로 커밋한 결과이므로,
재발 방지를 위해 `.gitignore`에 `.codex-worktrees/` 추가를 함께 검토한다.

---

## 7. 권장 실행 순서

| # | 대상 | 회수 | 조건 |
|---|---|---|---|
| 1 | B1 — 미커밋 삭제 2,367개 처리 결정 | — | **선행 조건** |
| 2 | B2, B4 — AN pdf · topwtageff payload를 추적 경로로 이동 | — | 소실 방지 |
| 3 | T1 — `git gc --prune=now` | 약 3.1 GB | 위험 0 |
| 4 | T2, T3, T4 — 고아 복사본 · worktree prune · 캐시 | 약 346 MB | 위험 낮음 |
| 5 | T5 — 캠페인 스크래치 | 약 46 MB | 위험 낮음 |
| 6 | T6 — 플롯 중복 해소 | 37.5 MB | 정본 결정 필요 |
| 7 | T9, T10, T12, T13 — 죽은 코드 | 약 0.9 MB | T11 제외 |
| 8 | T7, T8 — 캠페인 산출물 | 약 795 MB | **EOS 대조 완료 후에만** |
| 9 | T14 — `docs/` 히스토리 정리 | 약 1.2 GB | **사용자 승인 필요** |
| 10 | B3 — `output/` 정책 모순 해소 | — | 정책/코드 통일 |

---

## 부록 — 조사 중 발견한 구조 결함 (정리 대상 아님)

정리와 별개로 기록해 둔다.

### `autonomous_allhad` 이름 충돌로 전체 테스트를 한 프로세스에서 수집할 수 없다

레포 루트에서 `autonomous_allhad`는 namespace package(`__file__ is None`)라
`autonomous_allhad.gnn_lowdm`이 풀린다. 반면 `autonomous_allhad/`가 `sys.path`에 들어가면
같은 이름이 내부 regular package로 잡혀 namespace를 가린다. 그런데 `gnn_lowdm`은 두 철자
모두로 import된다.

- 최상위형 — [build_flat_boosted_recoil_hists.py:46](../workflow/build_flat_boosted_recoil_hists.py), `gnn_lowdm/tests/test_score_ut_histograms.py:11`, `tests/test_highdm_veto_pt_threshold.py:152`
- 네임스페이스형 — `gnn_lowdm/{train,valid,eval,plotting}.py`의 dispatch 테이블 전부

두 `sys.path` 구성이 상호배타적이라 결과가 다음과 같이 갈린다.

```
pytest autonomous_allhad/tests                    ->  354 collected
pytest autonomous_allhad/gnn_lowdm/tests          ->  1 error   (No module named 'gnn_lowdm')
pytest autonomous_allhad/tests autonomous_allhad/gnn_lowdm/tests
                                                  ->  2 errors  (No module named 'autonomous_allhad.gnn_lowdm')
```

레포 전체에 `conftest.py`가 하나도 없고, 대신 22개 파일이 `sys.path.insert`로 경로를
직접 조작한다. 루트에 `conftest.py` 하나를 두어 두 이름을 모두 해결해 주는 것이 최소 수정이다.

### 로컬 런타임이 캐노니컬 핀과 어긋난다

| | 요구 | 로컬 실측 |
|---|---|---|
| GNN | `torch==2.13.0`, `numpy==2.0.2`, `uproot==5.7.6` (Python 3.12.4) | torch 2.3.1, numpy 2.0.2, uproot 5.3.10 |
| PyROOT | Combine/템플릿 단계 필수 | homebrew ROOT 6.32.02가 이 인터프리터에서 모듈 로드 실패 |
| TensorFlow | TROTA 추론 | 미설치 |

로컬에서는 히스토그램 이후 단계(카드·limit·플롯 일부)를 실행할 수 없다.
분석·배치 런타임은 `/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python`과 `py38.tgz`다.
