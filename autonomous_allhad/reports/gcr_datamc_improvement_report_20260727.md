# 2024 high-\(\Delta M\) GCR Data/MC 개선 조사

상태: **조건부 R&D 완료, nominal 채택 전**

기준 선택은
[`real_subset_worker.py`](../autonomous_allhad/real_subset_worker.py)뿐이다.
`analysis/processors/stop_processor_v4.py`와 `ids.py`는 이 조사에서 선택
정의로 사용하지 않았다. 기존 nominal histogram 및 fake-measurement
중간 산출물은 수정하지 않았다.

## 결론

현재 GCR 불일치는 luminosity, 중복 data event, MC scale의 이중 적용,
누락된 \(W\to\ell\nu\), 또는 \(U_T\) 선택 자체의 문제가 아니다.
가장 중요한 \(U_T\)에서는 주로 **photon-channel rate deficit**이며,
그 rate를 지배하는 \(\gamma+\)jets와 QCD prompt-photon 성분의
구성·stitching 및 QCD의 매우 낮은 유효 통계가 우선 원인이다.

nominal GCR \(U_T\)는

\[
N_\text{data}=14\,495,\qquad
N_\text{MC}=10\,239.67,\qquad
\text{Data/MC}=1.41557.
\]

data-driven fake를 QCD 전체가 아니라 truth-fake 부분에만 사용하고,
\(\gamma+\)jets와 QCD truth-prompt를 하나의 prompt pool로 묶어 GCR에서
단일 정규화 계수 \(\alpha_\gamma\)를 fit한 조건부 후보는

\[
\alpha_\gamma=1.38951\pm0.01263\;(\text{data stat. only}),
\]

\[
N_\text{pred}=14\,479.20,\qquad
\text{Data/pred.}=1.00109
\]

를 준다. \(U_T\) bin별 Data/prediction은

```text
1.066, 0.969, 0.948, 0.936, 1.064, 0.931, 1.024, 0.960
```

이고 \(\log(\text{Data/prediction})\) RMS는 nominal 0.329에서 0.054로
감소한다. 다만 이 적분값은 같은 GCR에서 fit했으므로 그 자체가
독립 closure가 아니다. 따라서 이 결과는 아직 prefit MC correction이
아니라 **GCR likelihood constraint의 R&D 모형**이다.

보고서 페이지:
[GCR Data/MC improvement study](gcr_datamc_improvement_20260727/index.html)

주요 그림:

- [nominal 및 조건부 \(U_T\) 비교](gcr_datamc_improvement_20260727/gcr-ut-nominal-vs-prompt-constraint.png)
- [\(U_T\)에 따른 prompt 계수 안정성](gcr_datamc_improvement_20260727/gcr-prompt-scale-vs-ut.png)
- [EB/EE photon-\(p_T\) 진단](gcr_datamc_improvement_20260727/gcr-prompt-scale-vs-photon-pt.png)
- [13개 분포의 nominal/후보 비교](gcr_datamc_improvement_20260727/gcr-all-distributions-metric.png)
- [representative hard-parton radius scan](gcr_datamc_improvement_20260727/gcr-hardparton-dr-radius-scan.png)

## 1. 무엇을 배제했는가

### 전역 MC normalization

2024 luminosity는 \(109\,820~\mathrm{pb}^{-1}\)이다. 65개 background
physical dataset과 3,946개 split에 대해

\[
w_\text{event}
=w_\text{gen}\,w_\text{post-skim SF}\,
\frac{\sigma\,\mathcal L}{\sum w_\text{gen}}
\]

가 정확히 한 번 적용됨을 재계산했다. 저장 factor와 독립 재계산의
최대 상대 차이는 \(10^{-13}\) 수준이며 audit error는 0이다.
따라서 luminosity나 전체 MC에 1.416을 곱하는 조치는 금지한다.

\(\gamma+\)jets cross section과 generator filter efficiency도 독립
확인했다. McM LHE request와 실제 CVMFS gridpack pilot integration의
NLO cross section에 filter efficiency를 한 번 곱한 값은 다음과 같다.

| \(p_T^\gamma\) bin (GeV) | pilot NLO (pb) | filter eff. | 계산값 (pb) | 사용값 (pb) |
|---|---:|---:|---:|---:|
| 100–200 | \(5444\pm20\) | 0.26 | 1415.4 | 1391 |
| 200–400 | \(174.1\pm0.4\) | 0.50 | 87.05 | 88.24 |
| 400–600 | \(5.601\pm0.015\) | 0.68 | 3.8087 | 3.77 |
| \(>600\) | \(0.7158\pm0.0015\) | 0.82 | 0.58696 | 0.576 |

네 bin 모두 사용값과 \(\pm2\)% 이내로 일치한다. 따라서 filter
efficiency의 누락·이중 적용 또는 잘못된 \(\gamma+\)jets cross
section이 \(1.77\times\) 부족을 만든다는 가설도 배제한다.

### data 중복 및 누락

GCR data event key에는 중복이 없다. GCR은 EGamma stream만 사용하며
JetMET/Muon stream과의 중복도 없다. sidecar에 없는 nominal GCR data는
29 events, 약 0.2%뿐이다. 이를 복구하면 data가 증가하므로 현재
Data/MC 불일치는 아주 조금 더 커진다. 수 시간짜리 전체 campaign
재제출을 정당화하지 않는다.

### \(W\to\ell\nu\) 누락

\(W\to\ell\nu\)는 nominal GCR의 `WtoLNu` process로 포함되어 있다.
GCR 적분의 96.1%는 \(\gamma+\)jets와 QCD이므로 작은 electroweak/top
성분의 누락이나 SF로 41.6% deficit을 설명할 수 없다.

### 전역 data/MC 문제

다른 영역의 대표 Data/MC는 QCDCR 1.035, LLCR 0.878, DY2M 0.846으로
GCR의 1.416과 다르다. 전역 luminosity 또는 모든 MC에 공통인
normalization 오류라는 가설과 맞지 않는다.

## 2. \(U_T\)에서는 rate 문제인가 shape 문제인가

nominal \(U_T\)를 적분에 맞춰 shape-only로 비교하면
\(\chi^2/\mathrm{dof}=4.81/7\), \(p=0.683\)이다. nominal low-\(U_T\)
rate로 high-\(U_T\)를 예측해도 각 high-\(U_T\) bin 잔차는
\(0.9\sigma\) 이내이다.

data-driven fake를 고정한 prompt-pool 후보에서는 low-\(U_T\)
\((250\text{--}500~\mathrm{GeV})\) fit이
\(\alpha_\gamma=1.38466\pm0.01331\), high-\(U_T\)
\((500\text{--}1500~\mathrm{GeV})\) fit이
\(1.43232\pm0.04022\)이다. 두 값의 차이는 data 통계만으로
\(1.12\sigma\)이며, low-\(U_T\) factor를 high-\(U_T\)에 그대로
적용하면

\[
N^\text{high-}U_T_\text{pred}=1359.1,\qquad
N^\text{high-}U_T_\text{data}=1404,\qquad
\text{Data/pred.}=1.033.
\]

따라서 핵심 \(U_T\)에 per-bin empirical reweighting은 필요하지 않다.
단일 rate parameter가 더 안정적이고 audit 가능하다.

## 3. 왜 QCD 전체를 fake로 바꾸면 안 되는가

full sidecar truth audit에서 QCD GCR target은

```text
all QCD       4213.66
prompt        3902.54  (92.62%)
truth fake     311.09  ( 7.38%)
```

이다. 즉 GCR의 QCD histogram은 거의 전부 prompt photon이다.
QCD 전체를 data-driven fake로 대체하면 prompt-photon yield까지
삭제되어

```text
nominal Data/MC                   1.4156
truth-fake 부분만 교체            1.3458
QCD 전체를 fake로 교체 (기각)      2.1372
```

가 된다. 따라서 fake measurement는 QCD의 truth-fake 부분에만
대응해야 한다.

## 4. \(\gamma+\)jets–QCD overlap과 낮은 QCD 유효 통계

현재 Run-3 sample은 생성 단계에서 exclusive하지 않다.

- \(\gamma+\)jets는 aMC@NLO `p p > a j [QCD]`와 Frixione isolation
  \(R_0=0.15\)를 사용한다.
- QCD Pythia sample은 `HardQCD:all=on`이며 photon veto 또는 QED
  shower-off 설정이 없다.
- `real_subset_worker.py`에는 두 sample 사이의 generator overlap
  removal이 없다.
- 현재 sidecar의 `Photon_genPartFlav == 1`은 prompt-final-state
  matching일 뿐 direct photon과 fragmentation photon을 구분하지
  못한다.

GCR prompt pool에서 QCD prompt의 적분 비율은 약 42%이고 \(U_T\)
bin별로 21–46%이다. 그러나 QCD 전체 GCR의
\(N_\text{eff}=(\sum w)^2/\sum w^2\)는 약 61뿐이며, QCD prompt
template의 bin별 상대 MC 통계 오차는 10–44%이다. 이 때문에 큰
가중치의 드문 shower-photon event가 jet, \(H_T\), \(N_b\) shape을
쉽게 지배한다.

QCD prompt를 전부 제거하는 것은 overlap removal이 아니다.
그렇게 하면 Data/MC는 2.287로 악화된다. 필요한 것은 selected
prompt photon과 hard parton의 \(\Delta R\) 등을 이용해
direct/fragmentation phase space를 **두 sample 사이에서 상보적으로**
나누는 것이다. 그 경계는 Data/MC가 좋아지도록 tune하지 않고
generator 정의와 stability scan으로 정해야 한다.

이를 실제 NanoAOD에서 시험하기 위해 기존 nominal `feature_GCR`
flat row를 `(file_id, entry)`로 원본 event에 정확히 join하는
representative scan을 실행했다. selection을 재실행하거나 근사하지
않았다. exact nominal GCR 172 events를 172/172 GenPart에 join했고,
prompt-photon 168 events 중 \(\gamma+\)jets 65, QCD 103 events가
hard-parton \(\Delta R\) 계산에 사용되었다.

\(\gamma+\)jets의 최소 \(\Delta R\)는 0.448이어서
\(R=0.15,0.20,0.30,0.40\) 모두 direct keep efficiency가 가중·비가중
100%였다. 반면 `QCD prompt: \(\Delta R<R\)` fragmentation keep은
다음과 같이 극단적으로 radius와 큰 event weight에 의존했다.

| \(R\) | QCD 비가중 keep | QCD \(|w|\)-가중 keep |
|---:|---:|---:|
| 0.15 | 7.77% | 2.37% |
| 0.20 | 8.74% | 2.38% |
| 0.30 | 17.48% | 13.79% |
| 0.40 | 24.27% | 91.97% |

\(R=0.40\)의 92%는 stable한 결과가 아니다. 저-\(\hat p_T\) QCD
event 3개가 representative QCD 총 weight의 약 89%를 차지하며,
그중 PT170–300의 단일 event는 weight 82.79,
\(\Delta R=0.3895\)이다. \(R\)을 0.30에서 0.40으로 옮길 때 이
event 하나가 분류를 바꾸면서 가중 keep이 13.8%에서 92.0%로
폭증한다.

따라서 이 scan은 sample overlap과 QCD high-weight instability를
실측으로 확인했지만 **어떤 radius도 채택하지 않는다**. 다음 단계는
full NanoAOD campaign 재처리가 아니라 모든 exact-GCR flat row만
원본에 join하여, 특히 QCD PT170–300/300–470 전체에서 radius별
\(\sum w^2\), \(N_\text{eff}\), leave-one-out stability를 계산하는
lightweight selected-entry scan이다.

그림:
[representative hard-parton radius scan](gcr_datamc_improvement_20260727/gcr-hardparton-dr-radius-scan.png)

## 5. 모든 분포가 단일 rate로 해결되는 것은 아니다

단일 prompt-pool normalization은 네 개의 1차 metric을 13개 중
12개 분포에서 모두 개선하지만 \(H_T\)에서는 실패한다.

| 분포 | 적분 Data/MC | 적분 정규화 후 shape \(p\) |
|---|---:|---:|
| \(U_T\) | 1.416 | 0.683 |
| \(N_j\) | 1.415 | 0.769 |
| \(N_\mathrm{fatjet}\) | 1.415 | 0.551 |
| \(N_\mathrm{top}\) | 1.415 | 0.510 |
| b-jet \(p_T\) | 1.416 | \(3.3\times10^{-18}\) |
| fatjet \(p_T\) | 1.386 | \(7.9\times10^{-18}\) |
| \(H_T\) | 1.416 | \(6.6\times10^{-18}\) |
| leading-jet \(p_T\) | 1.416 | \(2.2\times10^{-22}\) |
| \(N_b\) | 1.415 | \(6.4\times10^{-9}\) |
| \(N_W\) | 1.415 | \(2.2\times10^{-6}\) |

bin별 Data/MC와 QCD fraction의 상관계수는 b-jet \(p_T\) -0.947,
fatjet \(p_T\) -0.990, \(H_T\) -0.983, leading-jet \(p_T\) -0.974인
반면 \(U_T\)에서는 -0.057이다. 이는 \(U_T\) rate deficit과
jet-related template composition 문제를 분리해서 다뤄야 함을
보여준다.

특히 \(H_T=300\text{--}900~\mathrm{GeV}\)의 Data/MC는 1.515,
\(900\text{--}3000~\mathrm{GeV}\)에서는 1.087이며 차이는 약
\(3.6\sigma\)이다. \(H_T\) 원인이 설명되거나 uncertainty로
포함되기 전에는 “전체 GCR 개선 완료”라고 부르지 않는다.

## 6. 비교한 개선 후보

| 후보 | \(U_T\) Data/pred. | \(\log\)-ratio RMS | 4개 metric이 모두 개선된 분포 |
|---|---:|---:|---:|
| nominal | 1.4156 | 0.3286 | — |
| QCD 전체 \(\to\) DD fake | 2.1372 | 0.6889 | 0/13 |
| truth-fake만 DD로 교체 | 1.3458 | 0.2939 | 13/13 |
| QCD 유지, \(\gamma+\)jets만 fit | 0.9993 | 0.1048 | 13/13 |
| QCD prompt 제거, \(\gamma+\)jets+DD fake fit | 1.0016 | 0.1731 | 11/13 |
| prompt pool + DD fake fit | **1.0011** | **0.0543** | **12/13** |

두-template \(\gamma+\)jets/QCD 자유 fit은 \(U_T\)에서 대략
\(\alpha_{\gamma j}=1.96\), \(\alpha_\mathrm{QCD}=0.76\),
상관계수 -0.982를 주고 다른 변수에서는 계수가 크게 변한다.
이는 두 template이 거의 퇴화되어 있음을 보여주는 진단일 뿐,
물리 correction으로 채택하지 않는다.

## 7. correction 감사에서 발견한 작은 항목

다음은 correctness fix 또는 uncertainty에는 중요하지만 42% deficit의
주원인은 아니다.

- medium photon ID SF는 한 번 적용되며 nominal 총량 효과는 약 +2.5%,
  variation은 약 \(\pm4.7\)%이다.
- \(\gamma+\)jets cross section은 gridpack NLO integration과 McM
  filter efficiency로 독립 검증했으며 네 \(p_T^\gamma\) bin에서
  사용값과 \(\pm2\)% 이내이다.
- photon-trigger SF는 현재 명시적으로 적용되지 않는다. \(p_T>220\)
  GeV plateau에서 orthogonal-trigger data efficiency 측정이 필요하다.
- electron-veto/CSEV SF가 빠져 있다. 예상 부호는 MC를 수% 낮춰
  Data/MC를 악화시키는 쪽이며, 현재 flat skim에는 필요한 photon
  \(R_9\)도 없다.
- GCR b-tag weight가 photon-cleaned jet이 아니라 원래 good-jet
  collection을 사용한다. 예상 크기는 약 1%지만 collection scope를
  일치시켜야 한다.
- pileup 및 b-tag variation의 적분 영향은 수% 이하이다.
- GCR의 nominal high-\(\Delta M\) \(p_T^\text{miss}\) histogram은
  selection이 \(p_T^\text{miss}<250\) GeV인데 binning이 250 GeV부터라
  구조적으로 비어 있다. 이는 물리 deficit이 아니라 plotting/binning
  오류이며 \(0\text{--}250\) GeV binning으로 다시 그려야 한다.

## 8. 채택 조건과 다음 조치

권장 모델은 다음과 같다.

1. 분석 event/object selection은 바꾸지 않는다.
2. \(\gamma+\)jets와 QCD를 generator-level direct/fragmentation
   기준으로 exclusive하게 만든다.
3. data-driven fake는 truth-fake 성분에만 사용한다.
4. exclusive prompt pool에 하나의 GCR normalization nuisance를
   둔다. 현재 \(\alpha_\gamma\simeq1.39\)는 그 조건부 초기값이다.
5. 이 nuisance는 GCR likelihood에서 constrain하며 모든 region/MC에
   곱하는 global correction으로 사용하지 않는다.

채택 전에 다음 validation gate를 모두 통과해야 한다.

- low-\(U_T\) fit으로 high-\(U_T\) holdout closure
- EB와 EE closure
- photon-\(p_T\) strata closure
- \(N_\mathrm{top}=0\)와 \(N_\mathrm{top}\ge1\) closure
- \(N_b\), \(N_j\), \(H_T\), leading-jet \(p_T\) closure
- generator overlap 제거 정책의 radius/definition stability
- 검증된 \(\gamma+\)jets cross section/filter-efficiency provenance를
  campaign metadata에 고정
- photon trigger, CSEV, photon-cleaned b-tag correction 감사
- normalization audit PASS 유지

현재 별도 R&D payload는 EOS의

```text
/eos/user/t/taiwoo/run3_stop/decaf/autonomous_allhad/workflow/
photon_fake_2024_snapshot_complete_20260726T1917Z/
gcr_prompt_constrained_candidate.json
```

에 있으며 상태는 `conditional_r_and_d_not_adopted`이다. nominal
payload에는 어떤 변경도 가하지 않았다. 후보 payload가 기록한 source
nominal SHA-256은
`b51abfd2562e0e8667c0834dc4b6153dad5d4fb3751cf254c79f4a336d240eab`
이다.

## 기계 판독 가능한 산출물

- [요약 JSON](gcr_datamc_improvement_20260727/summary.json)
- [validation record](gcr_datamc_improvement_20260727/validation.md)
- [nominal data/shape audit JSON](gcr_datamc_improvement_20260727/audits/gcr-nominal-data-mc-audit.json)
- [representative hard-parton audit JSON](gcr_datamc_improvement_20260727/audits/gcr-hardparton-dr-representative.json)
- [normalization/correction audit JSON](../validation/gcr_normalization_corrections_audit_20260726.json)
- [reproducible selected-row overlap scanner](../workflow/scan_gcr_prompt_overlap_2024.py)
- EOS `gcr_datamc_improvement_study.json`
- EOS `gcr_photon_strata_audit.json`
- EOS `normalization_audit_all_background_contamination.json`
- EOS `gcr_prompt_constrained_candidate.json`
