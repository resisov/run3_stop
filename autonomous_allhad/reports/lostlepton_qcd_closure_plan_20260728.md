# 2024 Lost-lepton 및 QCD multijet closure 실행 계획

작성일: 2026-07-28  
상태: **실행 전 사전 설계 및 구현 감사 완료**

## 1. 결론

두 background 모두 다음의 서로 다른 검증을 통과해야 한다.

1. **독립 MC closure**: transfer factor를 만든 MC와 prediction을 시험하는 MC가 이벤트 수준에서 독립이어야 한다. 이것은 selection, normalization, weighting, bin mapping, subtraction, 통계 처리의 자기일관성을 검사한다.
2. **Data validation-region closure**: SR과 겹치지 않는 data validation region(VR)에서 control-to-target extrapolation을 시험한다. 이것이 generator mismodelling을 포함한 실제 물리 closure이다.

MC closure만 통과했다고 방법을 채택하지 않는다. 같은 generator의 MC closure는 구현상의 오류는 찾지만, data에서의 lepton-loss 또는 jet-response mismodelling까지 보장하지 못하기 때문이다.

현재 상태의 중요한 결론은 다음과 같다.

- Lost-lepton의 nominal LLCR/SR mask와 raw MC transfer factor는 이미 존재한다. 기존 flat event table을 이용해 첫 번째 독립 MC closure와 일부 data VR closure를 수행할 수 있다.
- QCD의 nominal QCDCR/SR mask와 **보정 전 raw MC transfer factor**는 존재한다.
- 그러나 현재 `autonomous_allhad` 코드에는 AN이 서술한 \(r_{\mathrm{pseudo}}\) template fit, response correction, QCD event smearing이 구현되어 있지 않다. 따라서 현재 transfer-factor 그림은 response-corrected QCD method의 closure 결과가 아니다.
- AN의 두 closure 문단은 현재 결과가 아니라 계획을 서술한 상태이다. 실제 JSON, covariance, pull, data-VR closure plot이 만들어질 때까지 완료로 표시하면 안 된다.

## 2. 분석 정의의 기준

selection의 유일한 기준은
`autonomous_allhad/autonomous_allhad/real_subset_worker.py`이다.
`decaf/analysis/` 아래의 legacy `stop_processor`와 `ids.py`는 사용하지 않는다.

### 2.1 High-\(\Delta m\)

| Region | 핵심 selection |
|---|---|
| SR | zero veto lepton, zero selected tau, \(N_j\geq5\), \(N_b\geq1\), \(p_{\mathrm{T}}^{\mathrm{miss}}>250\) GeV, \(H_{\mathrm{T}}>300\) GeV, \(\Delta\phi(j_{1,2,3,4},p_{\mathrm{T}}^{\mathrm{miss}})>0.5\) |
| LLCR | SR의 hadronic selection을 유지하고 exactly one veto lepton, \(m_{\mathrm{T}}<100\) GeV |
| QCDCR | zero veto lepton, \(N_j\geq5\), \(N_b\geq1\), \(p_{\mathrm{T}}^{\mathrm{miss}}>250\) GeV, \(H_{\mathrm{T}}>300\) GeV, first four jets 중 하나가 \(\Delta\phi<0.5\), first three jets 중 하나가 \(\Delta\phi<0.1\) |

### 2.2 Low-\(\Delta m\)

Low-\(\Delta m\)은 \(N_j\geq2\), boosted-top/W veto, exactly one ISR fat jet,
\(\Delta\phi(\mathrm{ISR},p_{\mathrm{T}}^{\mathrm{miss}})>2\), 그리고
\(p_{\mathrm{T}}^{\mathrm{miss}}/\sqrt{H_{\mathrm{T}}}\geq10\)을 사용한다.

- SR/LLCR angular selection:
  \[
  \Delta\phi(j_1,p_{\mathrm{T}}^{\mathrm{miss}})>0.5,\quad
  \Delta\phi(j_2,p_{\mathrm{T}}^{\mathrm{miss}})>0.15,\quad
  \Delta\phi(j_3,p_{\mathrm{T}}^{\mathrm{miss}})>0.15.
  \]
- QCDCR은 high-\(\Delta m\)과 같이 first three jets 중 하나가
  \(\Delta\phi<0.1\)인 aligned sample이다.
- Low-\(\Delta m\) closure는 adopted 42 search bins와 동일한 bin assignment를 사용한다. High-\(\Delta m\) 정의를 low-\(\Delta m\)에 대신 적용하지 않는다.

## 3. 공통 입력 및 독립성 규칙

### 3.1 필요한 2024 samples

| 용도 | 필수 sample |
|---|---|
| Data | nominal 정책과 동일한 `JetMET` stream. LLCR, QCDCR, SR-like VR 모두 event-key deduplication 후 사용 |
| Lost-lepton target MC | `TT`, `WtoLNu`, `ST` |
| QCD target MC | 전 \(p_{\mathrm{T}}\) bin을 포함하는 `QCD` multijet samples |
| Contamination/subtraction MC | `Zto2Nu`, `DY`, `GJ`, `VV`, 그리고 nominal background model에 포함된 rare processes. 상대 background에는 target이 아닌 `TT`, `WtoLNu`, `ST`, `QCD`도 해당 closure에 맞게 포함 |
| Signal contamination check | 대표적인 T2tt mass points. VR 선택 전에 contamination fraction만 검사하며 data를 보아 region을 조정하지 않음 |

모든 MC는 현재 accepted-file manifest와 full physical-dataset sum-of-weights denominator를 사용한다. shard별 또는 fold별로 normalization denominator를 다시 만들지 않는다. 각 scale factor는 exactly once 적용됐는지 component별 audit를 남긴다.

### 3.2 이벤트 분할

파일 또는 shard 기준으로 나누지 않는다. 다음 stable key를 hash하여 두 fold로 나눈다.

\[
k=(\mathrm{physical\ dataset},\ \mathrm{run},\ \mathrm{luminosityBlock},\ \mathrm{event}).
\]

- `hash(k) mod 2 = 0`: fold A
- `hash(k) mod 2 = 1`: fold B

동일 이벤트가 processing version 또는 shard 중복으로 다시 나타나면 먼저 deduplicate한다. 같은 물리 이벤트의 모든 systematic variation과 모든 QCD smeared copy는 반드시 같은 fold에 속한다.

두 방향을 모두 계산한다.

- A에서 TF를 측정하여 B를 예측: A \(\rightarrow\) B
- B에서 TF를 측정하여 A를 예측: B \(\rightarrow\) A

이벤트 split이 process composition을 우연히 바꾸지 않도록 dataset별, signed-weight별
\(\sum w\), \(\sum w^2\), \(N_{\mathrm{eff}}=(\sum w)^2/\sum w^2\)를 비교한다.

### 3.3 두 종류의 MC closure

각 background마다 두 closure를 따로 만든다.

1. **Pure-target closure**: target background만 사용한다. transfer-factor algebra와 bin mapping을 가장 직접적으로 검사한다.
2. **Full-mixture pseudodata closure**: target과 모든 contamination MC를 합쳐 pseudodata로 만들고, 실제 prediction과 똑같이 other-background를 뺀다. subtraction과 normalization까지 검사한다.

둘 중 하나라도 실패하면 method closure로 인정하지 않는다.

## 4. Lost-lepton closure

### 4.1 Nominal prediction

Lost-lepton target은 `TT + WtoLNu + ST`이다. fold A에서

\[
R^{\mathrm{LL},A}_i =
\frac{N^{\mathrm{LL},A}_{0\ell,i}}
     {N^{\mathrm{LL},A}_{1\ell,i}}
\]

를 만들고 fold B의 pseudodata에 적용한다.

\[
\widehat N^{\mathrm{LL},A\to B}_{0\ell,i}
=R^{\mathrm{LL},A}_i
\left[
N^{\mathrm{all},B}_{1\ell,i}
-N^{\mathrm{other},B}_{1\ell,i}
\right].
\]

비교 대상은 fold B에서 SR selection을 통과한 직접 simulated yield
\(N^{\mathrm{LL},B}_{0\ell,i}\)이다. full-mixture closure에서는 target prediction에
SR의 other-background MC를 더한 뒤 전체 pseudodata와도 비교한다.

Closure ratio는

\[
C_i=\frac{\widehat N_i}{N_i^{\mathrm{direct}}}
\]

로 정의하며 이상적인 값은 1이다.

### 4.2 반드시 분리해서 볼 진단

Nominal 채택 판단은 combined `TT+WtoLNu+ST` prediction으로 하되 다음을 별도 plot과 JSON으로 남긴다.

- `TT`, `WtoLNu`, `ST` 각각의 closure
- electron-tagged LLCR와 muon-tagged LLCR
- generator truth가 준비된 뒤 lost electron, lost muon, hadronic tau origin
- high-\(\Delta m\), low-\(\Delta m\)
- \(p_{\mathrm{T}}^{\mathrm{miss}}\), \(N_j\), \(N_b\)
- high-\(\Delta m\) final category/search-bin index
- low-\(\Delta m\) adopted 42 search bins

현재 flat table에는 generator-level lost-lepton origin과 lepton \(m_{\mathrm{T}}\) 수치가 없다. Nominal combined closure에는 필요하지 않지만, origin breakdown과 \(100<m_{\mathrm{T}}<200\) test를 위해 additive closure columns를 만들어야 한다. Nominal histograms와 selection은 수정하지 않는다.

### 4.3 Data validation regions

Data VR에서는 SR-like zero-lepton target과 matched one-lepton control을 한 쌍으로 만든다. 단지 LLCR 내부의 data/MC를 비교하는 것은 transfer closure가 아니다.

#### High-\(\Delta m\)

1. **LLVR-H-Nb0**  
   Nominal high-\(\Delta m\) SR/LLCR과 같되 \(N_b=0\).
2. **LLVR-H-lowNjet**  
   Nominal 조건과 같되 \(3\leq N_j\leq4\), \(N_b\geq1\).
3. **LLVR-H-mT**  
   \(100<m_{\mathrm{T}}<200\) GeV는 LLCR modelling/stability test로 사용한다. 이것만으로는 zero-lepton closure가 아니므로, 위 zero-lepton VR 결과를 대신할 수 없다.

#### Low-\(\Delta m\)

Low-\(\Delta m\)의 \(N_b=0\) sample 일부는 nominal 42-bin SR에 들어가므로 그대로 data VR로 사용할 수 없다. 다음의 signal-orthogonal sideband를 우선 사용한다.

1. **LLVR-L-lowMET**: \(250<p_{\mathrm{T}}^{\mathrm{miss}}<300\) GeV, 나머지 low-\(\Delta m\) topology 유지.
2. **LLVR-L-lowISR**: \(200<p_{\mathrm{T}}^{\mathrm{ISR}}<300\) GeV, 나머지 조건 유지.
3. **LLVR-L-significance**: \(7<p_{\mathrm{T}}^{\mathrm{miss}}/\sqrt{H_{\mathrm{T}}}<10\), 나머지 조건 유지.

각 후보는 data를 열기 전에 trigger efficiency와 representative signal contamination을 검사한다. Trigger plateau가 아니거나 signal fraction이 1%를 넘으면 해당 VR은 채택하지 않고 이유를 기록한다.

Data closure에서는

\[
\widehat N^{\mathrm{total}}_{\mathrm{target},i}
=R_i^{\mathrm{LL}}
\left[N^{\mathrm{data}}_{1\ell,i}
-N^{\mathrm{other,MC}}_{1\ell,i}\right]
+N^{\mathrm{other,MC}}_{0\ell,i}
\]

를 observed zero-lepton VR data와 비교한다. VR data는 likelihood constraint나 TF tuning에 사용하지 않는다.

### 4.4 Lost-lepton uncertainties

다음 variation을 full prediction에 전파한다.

- electron/muon reconstruction, ID, isolation, trigger scale factors
- tau identification/veto 및 isolated-track veto
- lepton acceptance와 jet--lepton cleaning
- JES, JER, unclustered momentum
- b-tag efficiency와 mistag
- pileup
- `TT : WtoLNu : ST` composition
- top-\(p_{\mathrm{T}}\), renormalization/factorization scale, PDF, generator matching
- LLCR other-background subtraction
- finite MC statistics

Process-specific TF가 서로 크게 다르면 combined TF를 조용히 유지하지 않는다. 먼저 composition nuisance로 cover 가능한지 확인하고, 불가능하면 process-specific transfer component를 likelihood에 두는 방안을 **physics proposal**로 별도 비교한다.

## 5. QCD multijet closure

### 5.1 먼저 필요한 response-correction 모듈

현재 raw QCD transfer factor만으로는 AN에 서술된 method를 검증할 수 없다. 별도 산출물로 다음을 구현한다.

1. reco jet과 gen jet을 match하여
   \[
   r_{\mathrm{jet}}=p_{\mathrm{T,reco}}/p_{\mathrm{T,gen}}
   \]
   을 계산한다.
2. data에서 직접 계산 가능한
   \[
   r_{\mathrm{pseudo}}=
   \frac{p_{\mathrm{T,reco}}}
        {p_{\mathrm{T,reco}}+p_{\mathrm{T}}^{\mathrm{miss}}}
   \]
   를 \(p_{\mathrm{T}}^{\mathrm{miss}}\)에 가장 가까운 jet에 대해 계산한다.
3. QCDCR data에서 non-QCD MC를 빼고, \(r_{\mathrm{jet}}\) 구간별 QCD templates를 \(r_{\mathrm{pseudo}}\)에 fit하여 response scale factors와 covariance를 얻는다.
4. corrected response를 사용하여 QCD MC의 leading jets를 smear하고 \(p_{\mathrm{T}}^{\mathrm{miss}}\), \(H_{\mathrm{T}}\), jet angles, category, search-bin을 매 copy마다 다시 계산한다.
5. smear copy 수가 늘어도 원래 event의 정보량이 늘어난 것으로 세지 않는다. 모든 copy는 original-event ID를 공유한다.

Run-2의 수치와 binning은 초기 reference일 뿐 Run-3에 그대로 고정하지 않는다. 먼저 2024 data/MC의 \(r_{\mathrm{pseudo}}\) fit 안정성, gen-\(p_{\mathrm{T}}\), \(\eta\), jet flavor 의존성을 비교하여 binning을 정한다.

### 5.2 QCD MC closure

Response template fit과 smearing도 fold-independent해야 한다.

1. fold A만으로 response templates, scale factors, \(R_i^{\mathrm{QCD},A}\)를 만든다.
2. fold B의 QCDCR yield를 pseudodata로 사용한다.
3. A에서 만든 response model로
   \[
   R^{\mathrm{QCD},A}_i=
   \frac{N^{\mathrm{QCD,corr},A}_{\mathrm{SR},i}}
        {N^{\mathrm{QCD,corr},A}_{\mathrm{QCDCR},i}}
   \]
   를 계산한다.
4. \(\widehat N^{A\to B}_{\mathrm{SR},i}
   =R^{\mathrm{QCD},A}_iN^{\mathrm{QCD},B}_{\mathrm{QCDCR},i}\)를
   fold B의 direct QCD SR yield와 비교한다.
5. B \(\rightarrow\) A도 반복한다.
6. pure-QCD와 full-mixture pseudodata closure를 모두 수행한다.

다음 세 결과를 한 그림에 겹치지 않고 같은 binning으로 비교한다.

- raw, unsmeared QCD TF
- smeared but response-uncorrected QCD TF
- smeared and response-corrected nominal QCD TF

이 비교가 response model이 closure를 실제로 개선하는지 보여준다.

### 5.3 QCD data validation regions

Data VR마다 aligned control과 non-aligned target을 쌍으로 정의한다.

#### High-\(\Delta m\)

1. **QCDVR-H-Nb0**: \(N_b=0\), 나머지는 matched QCDCR/target selection.
2. **QCDVR-H-lowNjet**: \(3\leq N_j\leq4\), \(N_b\geq1\).
3. **QCDVR-H-intermediate**: control은 first-three-jet \(\min\Delta\phi<0.1\), target은 \(0.1<\min\Delta\phi(j_{1,2,3,4},p_{\mathrm{T}}^{\mathrm{miss}})<0.5\).
4. **QCDVR-H-lowMET**: trigger plateau가 확인되는 lower-\(p_{\mathrm{T}}^{\mathrm{miss}}\) bin.

#### Low-\(\Delta m\)

Low-\(\Delta m\)의 비대칭 angular boundary를 그대로 검증한다.

1. **QCDVR-L-leading-angle**:
   \(0.1<\Delta\phi(j_1,p_{\mathrm{T}}^{\mathrm{miss}})<0.5\),
   \(\Delta\phi(j_{2,3},p_{\mathrm{T}}^{\mathrm{miss}})>0.15\).
2. **QCDVR-L-subleading-angle**:
   \(\Delta\phi(j_1,p_{\mathrm{T}}^{\mathrm{miss}})>0.5\)이고
   \(j_2\) 또는 \(j_3\)가 \(0.1<\Delta\phi<0.15\).
3. **QCDVR-L-lowISR/low-significance**:
   nominal low-\(\Delta m\) topology 바로 아래의 ISR 또는
   \(p_{\mathrm{T}}^{\mathrm{miss}}/\sqrt{H_{\mathrm{T}}}\) sideband.

Data comparison은

\[
\widehat N^{\mathrm{total}}_{\mathrm{target},i}
=R_i^{\mathrm{QCD}}
\left[N^{\mathrm{data}}_{\mathrm{aligned},i}
-N^{\mathrm{other,MC}}_{\mathrm{aligned},i}\right]
+N^{\mathrm{other,MC}}_{\mathrm{target},i}
\]

를 observed target data와 비교한다. Nominal SR data는 열지 않는다.

### 5.4 QCD statistical treatment와 uncertainties

Smeared copy를 독립 Poisson event로 취급하면 QCD MC uncertainty가 인위적으로 작아진다. 따라서 original unsmeared event를 resampling unit으로 하는 Poisson bootstrap을 사용한다.

- 한 original event의 모든 smear copy에는 같은 bootstrap multiplier를 적용한다.
- 각 bootstrap toy에서 response fit, smearing, TF, closure prediction을 다시 계산한다.
- 최소 1000 toys로 full bin covariance를 얻는다.
- random seed는 event key와 toy index에서 결정하여 완전히 재현 가능하게 한다.

Systematic sources:

- response-template fit covariance
- \(r_{\mathrm{jet}}\) template binning 및 gen-\(p_{\mathrm{T}}\), \(\eta\), flavor parameterization
- no-response-correction alternative
- no-smearing alternative
- smearing window와 number-of-smears 안정성
- QCDCR angular boundary 및 intermediate sideband 정의
- JES, JER, unclustered momentum
- trigger efficiency for fake-\(p_{\mathrm{T}}^{\mathrm{miss}}\) events
- pileup, b-tag/mistag
- non-QCD contamination subtraction
- original-event finite MC statistics

No-smearing과 no-response-correction difference를 자동으로 “개선”으로 해석하지 않는다. Data VR closure가 nominal correction을 지지할 때만 nominal method로 채택한다.

## 6. Binning과 low-statistics 정책

Closure를 보기 좋게 만들기 위해 빈 bin을 숨기지 않는다.

1. 먼저 final search bins에서 계산한다.
2. TF denominator의 \(N_{\mathrm{eff}}<25\) 또는 numerator의
   \(N_{\mathrm{eff}}<10\)인 bin은 결과 JSON에 그대로 `insufficient_statistics`로 표시한다.
3. 물리적으로 인접한 \(p_{\mathrm{T}}^{\mathrm{miss}}\) bins 또는 동일 topology bins만 predeclared 순서로 merge한다.
4. merged closure nuisance를 원래 fine bins에 correlation을 유지해 적용한다.
5. QCD smearing의 \(N_{\mathrm{eff}}\)는 smear-copy 수가 아니라 original events와 bootstrap covariance로 계산한다.

## 7. 통계적 판정 기준

각 branch와 validation sample에 대해 다음을 기록한다.

- integral prediction/direct ratio
- binwise closure ratio와 uncertainty
- full-covariance \(\chi^2\) 및 \(p\)-value
- maximum absolute pull
- \(p_{\mathrm{T}}^{\mathrm{miss}}\)에 대한 constant 및 linear residual fit
- A \(\rightarrow\) B와 B \(\rightarrow\) A 차이
- \(\sum w\), \(\sum w^2\), \(N_{\mathrm{eff}}\)

사전 채택 기준:

1. technical/inclusive closure가 combined uncertainty 내에서 1과 일치
2. global \(p>0.05\)
3. 충분한 통계가 있는 bin에서 \(|\mathrm{pull}|<3\)
4. residual slope가 2 standard deviations보다 유의하지 않음
5. 두 fold 방향이 통계적으로 양립
6. 최소 하나의 high-\(\Delta m\), 하나의 low-\(\Delta m\) data VR이 통과

실패 시 uncertainty를 무한히 키워 억지로 채택하지 않는다. 먼저 normalization, duplication, selection, contamination subtraction, bin mapping, effective statistics를 순서대로 재감사한다. 원인이 physics mismodelling이면 method 또는 TF parameterization을 바꾸고 새 closure를 독립적으로 수행한다.

## 8. Nonclosure nuisance 구성

관측 residual
\(\delta_i=(\widehat N_i-N_i^{\mathrm{direct}})/N_i^{\mathrm{direct}}\)에서
bootstrap/Poisson 통계 성분을 분리한다.

\[
\delta^{\mathrm{model}}_i=
\sqrt{\max\left(\delta_i^2-\sigma_{\mathrm{stat},i}^2,0\right)}.
\]

최종 nonclosure amplitude는 독립 MC folds, 채택된 data VRs, response/selection alternatives에서 얻은 \(\delta^{\mathrm{model}}\)의 envelope로 정한다. MC statistical nuisance는 별도로 유지하여 이중 계산하지 않는다.

- topology 전반에서 같은 방향이면 branch-correlated normalization nuisance
- \(p_{\mathrm{T}}^{\mathrm{miss}}\)에 따라 coherent하게 변하면 correlated shape nuisance
- process composition에 따라 변하면 `TT/W/ST` composition nuisance
- 통계가 부족해서 흩어지는 경우에는 nonclosure로 해석하지 않고 bin merge 또는 MC-stat nuisance로 처리

Nuisance correlation choice와 근거는 machine-readable JSON에 저장한다.

## 9. 필수 plot 목록

모든 plot은 `mplhep`을 사용하고 특별한 이유가 없으면 정사각형으로 만든다.

공통 CMS label:

```python
hep.cms.label(
    llabel="Work in progress",
    rlabel="2024 (13.6 TeV)",
    loc=0,
    ax=ax,
)
```

다른 `data=`, `year=`, `com=` 옵션은 사용하지 않는다. plot title은 넣지 않는다.
축 범위는 histogram edge에 정확히 맞춰 좌우 여백을 없앤다.
MET은 \(p_{\mathrm{T}}^{\mathrm{miss}}\), recoil은 \(U_{\mathrm{T}}\)로 표기한다.

### 9.1 AN 본문용 Lost-lepton

1. High-\(\Delta m\) A/B MC closure: prediction과 direct yield, 아래 ratio/pull
2. Low-\(\Delta m\) 42-bin MC closure
3. High-\(\Delta m\) data VR closure
4. Low-\(\Delta m\) data VR closure

### 9.2 AN 본문용 QCD

1. \(r_{\mathrm{pseudo}}\) prefit/postfit response-template distribution
2. raw/smeared/corrected QCD TF comparison
3. High-\(\Delta m\) independent MC closure
4. Low-\(\Delta m\) independent MC closure
5. High-/low-\(\Delta m\) data angular-VR closure

### 9.3 Backup/발표자료

- process/flavor/origin breakdown
- \(N_j\), \(N_b\), \(H_{\mathrm{T}}\), \(p_{\mathrm{T}}^{\mathrm{miss}}\)
- response scale factors와 fit covariance/correlation
- bootstrap pull distributions
- \(N_{\mathrm{eff}}\) per bin
- every systematic/alternative divided by nominal
- contamination fraction before and after subtraction

## 10. 산출물과 provenance

Nominal intermediate ROOT와 nominal histograms는 수정하지 않는다. 별도 디렉터리에 생성한다.

```text
autonomous_allhad/reports/background_closure_2024_<timestamp>/
├── index.html
├── summary.md
├── inputs/
│   ├── accepted_files.json
│   ├── bad_files_snapshot.json
│   ├── normalization_snapshot.json
│   └── event_deduplication.json
├── lost_lepton/
│   ├── fold_assignment.json
│   ├── transfer_factors.json
│   ├── mc_closure.json
│   ├── data_vr_closure.json
│   ├── covariance.json
│   └── plots/
└── qcd/
    ├── response_fit.json
    ├── smearing_config.json
    ├── transfer_factors.json
    ├── mc_closure.json
    ├── data_vr_closure.json
    ├── covariance.json
    └── plots/
```

각 JSON에는 git commit, config hash, input manifest hash, normalization hash,
fold hash algorithm, random-seed policy, selection definition, bin edges, \(\sum w\),
\(\sum w^2\), \(N_{\mathrm{eff}}\), failure/insufficient-statistics status를 포함한다.

## 11. 실행 순서

### Stage 0 — 감사

- event duplication, physical-dataset normalization, applied-SF count 확인
- current transfer-factor plot이 raw MC임을 provenance에 명시
- 기존 AN closure 문장을 `planned/pending results`로 취급

### Stage 1 — Lost-lepton 즉시 closure

- 기존 flat event table에서 event-level A/B split
- pure-target 및 full-mixture MC closure
- high-/low-\(\Delta m\) matched VR mask 생성
- data VR closure와 uncertainty decomposition

### Stage 2 — QCD raw baseline closure

- 기존 flat table로 raw-QCD A/B closure와 data angular-sideband closure
- 이것은 response-corrected method의 결과가 아니라 baseline 진단으로 명시

### Stage 3 — QCD response correction/smearing

- NanoAOD에서 gen/reco response와 \(r_{\mathrm{pseudo}}\) 전용 additive worker 실행
- response fit, deterministic smearing, original-event bootstrap 구현
- corrected QCD A/B 및 data-VR closure

### Stage 4 — 채택 판단

- raw와 corrected QCD 중 data VR closure가 더 좋은 방법만 채택
- lost-lepton combined TF와 process-specific alternative 비교
- 채택된 nonclosure nuisance와 correlation을 datacard proposal로 출력
- nominal histograms/datacards에는 physics review 승인 전 자동 주입하지 않음

## 12. 완료의 정의

다음이 모두 존재해야 “closure 완료”라고 보고한다.

- 두 방향의 independent MC closure
- pure-target와 full-mixture pseudodata closure
- high-/low-\(\Delta m\) data VR closure
- full covariance와 original-event statistical treatment
- normalization/deduplication audit
- search-bin 및 key kinematic plots
- predeclared gate 결과
- nonclosure nuisance magnitude와 correlation proposal
- 실패/저통계 bin의 명시적 목록

Transfer-factor plot 하나 또는 CR data/MC agreement만으로는 closure 완료가 아니다.
