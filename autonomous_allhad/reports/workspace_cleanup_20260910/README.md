# 워크스페이스 정리 검토 및 실행 기록 — 2026-09-10

기존 조사 보고서를 그대로 실행하지 않고 로컬·Git·EOS의 파일을 대조했다.
사용자는 기존 삭제 2,367개가 의도된 것이라고 확인했고, 미참조 Git 객체는
보관 없이 삭제하라고 명시했다. 물리 정본은 메인 에이전트가 확인하는 **현재 사용본**이다.

## 실행 결과

- 고아 복사본 3,783개(357,156,906 bytes)를 제거했다. 모든 비캐시 파일은
  도달 가능한 Git 객체 또는 동일한 로컬 사본과 대조했다. 고유 4개는 Python bytecode였다.
- EOS와 캠페인 파일 2,622개를 대조했다. 2,615개 일치, EOS에 없는 7개 보존.
  그중 대형 카드·fit log·manifest·ROOT 포함 ZIP의 동일 사본 1,882개,
  767,750,184 bytes를 로컬에서 제거했다. EOS 생산 출력은 삭제하지 않았다.
- 재생성 가능한 일반 캐시 36곳과 명시적인 campaign unit-test/font 캐시를 제거했다.
- Git reflog를 만료시키지 않고 `gc --no-prune`로 재압축했다. 미참조 cruft 객체는
  중복이 아니라는 사실을 확인한 뒤, 사용자 별도 지시에 따라 8,620개,
  3,689,686,665 bytes를 영구 삭제했다. 보관 사본은 만들지 않았다.
  삭제 직전 refs/reflogs/index 도달성을 재확인했고, 삭제 후 `git fsck --full --no-dangling`
  통과 및 refs 불변을 확인했다.
- 사라진 임시 worktree 등록 4개를 제거했다. 유효한 worktree 2개는 유지했다.
- 비공개 AN PDF를 루트의 Git-ignore 경로로 옮겼고, EOS와 같은 SHA-256을 확인했다.
  기존 tmp 경로는 호환 symlink다. PDF를 Git에 넣지 않았다.
- Top/W 효율 입력 두 개는 EOS와 해시가 같으며 보존했다. Git-ignore 예외를 추가했다.
- `.gitignore`에 worktree, 개인 복구 기록, 생성 카드·fit 내부 파일·preview 규칙을 보완했다.
- `docs/index.html`의 사라진 링크 11개를 실제 Git 객체가 존재하는 고정 commit 링크로
  교체하고 과거 결과임을 표시했다. 삭제된 플롯을 복원하거나 새 결과로 게시하지 않았다.
- `WORKSPACE_LAYOUT.md`에 로컬·EOS·Git의 역할, 복원 manifest, 공개/비공개 경계를 정리했다.

## 원 보고서에서 수정한 판단

1. `git gc --prune=now`는 무위험 작업이 아니다. 약 3.4 GiB는 중복 pack이 아니라
   미참조 복구 객체였다. 실제 삭제는 사용자 별도 지시 이후에만 수행했다.
2. AN PDF를 Git 추적 경로로 옮기라는 제안은 기존 비공개 정책과 충돌한다.
   scratch 밖으로 보호하되 Git-ignore를 유지했다.
3. 플롯 ZIP은 완전한 중복이 아니다. 517개 항목 중 515개만 낱개 사본과 같고,
   2개 ROOT 템플릿은 로컬 낱개본이 없었다. ZIP 전체 EOS 해시를 확인한 후에만 제거했다.
4. import 0건은 죽은 코드의 증거가 아니다. `analysis/run.py`는 `libs.mycoffea`를 사용하고,
   기존 CLI/config는 `condor`, `setup_condor.sh`, `analysis/distribution_draw_v5.py`를 참조한다.
   이 파일들과 Architecture B 근거인 `fast_analysis/`는 유지했다.
5. `impact_code_*`는 추적되는 실행 snapshot이므로 삭제하지 않았다. pilot/검증 산출물도
   이름만으로 삭제하지 않았다. PDF와 대응이 확인되지 않은 QA 이미지 역시 보존했다.
6. `btageff2024.merged`가 로컬에서 추적된다는 기술은 시작 시점 Git 목록과 맞지 않았다.
   처음부터 로컬에 없었다. 현재 사용 경로를 메인 에이전트에 문의했다.
7. `output/`은 preview 출력 기본 경로로 허용하고, 최종 handoff는 reports로 명시적으로
   승격하는 것으로 정책을 정리했다. 물리 builder 코드는 수정하지 않았다.

## 동기화 상태

EOS canonical checkout은 시작 시 `master` / `c6fd36b9e23b36fcb2e736ade79d915c704d716a`,
로컬·원격 작업 브랜치는 `codex/package-lowpt-tnp` / `561c2721b7dd97ddba1166c8b2ac7de7f07a5701`였다.
브랜치명이나 최신 시각으로 정본을 선택하지 않았다.

3,414개 경로의 초기 대조에서 내용이 다른 파일 19개가 있었다.
[파일별 차이](sync_conflicts.md), [기계 판독 목록](sync_conflicts.json)을 보존했다.
현재 사용본 확인을 `high-dM main Agent`에 요청했다. 해당 작업의 자동 승인 심사가
회신 공유를 차단하여 사용자 발언 원문을 추가 전달한 상태다.
**아직 전체 물리 source/payload 동기화가 완료됐다고 주장하지 않는다.**

기존 미커밋 source 변경과 다른 작업이 갱신 중인 campaign state는 보존했다.
`ids.py`, `corrections.py`와 compiled `.coffea`를 수정하지 않았으므로 재컴파일 작업도 만들지 않았다.
Git 히스토리를 재작성하거나 force-push하지 않았으며, GitHub Pages 배포는 수행하지 않았다.

## 기록과 복원

- [summary.json](summary.json): 측정 수치와 상태.
- [cleanup_actions.json](cleanup_actions.json): 실제 정리 작업.
- [git_object_cleanup.json](git_object_cleanup.json): 사용자 지시로 영구 삭제한 Git 객체 pack의 hash/검증.
- [protected_inputs.json](protected_inputs.json): 참조 파일 및 효율 입력의 존재·hash.
- 캠페인 `local_storage_manifest_20260910.json`: EOS에 보존된 각 파일의 상대 경로·크기·SHA-256.
  대형 카드/fit 작업은 EOS에서 실행한다. 복원 필요 시 동일 상대 경로의 EOS 사본을 사용하고
  hash를 대조한다. ROOT cache를 노트북으로 다시 복사하지 않는다.
- `.workspace-local/cleanup/20260910/`: 이전 수정 patch, 삭제 경로 목록, refs/worktree 목록,
  EOS 충돌 사본과 상세 대조 기록. 비공개이며 Git에 포함하지 않는다.

전체 분석 테스트나 생산·fit 재실행은 이 정리의 검증 범위가 아니다. 파일 hash, Git 무결성,
HTML 링크 대상, JSON 파싱 및 ignore 규칙을 검증한다.
