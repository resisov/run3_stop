# 2024 Lost-lepton closure 결과 해석

## 결론

이번 검증으로 구현, 정규화, 이벤트 중복, control-to-target bin 매핑은 정상임을 확인했다. 그러나 데이터 validation region에서 lost-lepton 성분만 분리해 비교하면 high-dM과 대부분의 low-dM 영역에서 큰 nonclosure가 관측된다. 따라서 현재의 결합 `TT + WtoLNu + ST` transfer factor를 nominal lost-lepton 예측으로 바로 채택해서는 안 된다.

QCD multijet은 이 연구의 측정 대상이 아니다. 데이터 validation에서 lost-lepton 성분을 분리하기 위한 비대상 배경 오염 차감에만 포함했다.

## 입력과 재현성

- 2024 입력 ROOT 파일: 1,153개
- 읽은 이벤트: 361,054,245개
- closure 및 validation selection을 통과한 이벤트: 34,735,958개
- 대상 lost-lepton MC: `TT + WtoLNu + ST`
- 데이터: `JetMET`
- 비대상 MC: `Zto2Nu + DY + GJ + VV + QCD`
- selection authority: `real_subset_worker.py`
- fold 분할: 물리 dataset ID와 run, luminosity block, event number의 안정적인 event-level hash
- 파일 또는 shard 기준 분할은 사용하지 않음
- 모든 물리 dataset에서 A/B fold 중복 이벤트 수: 0
- nominal 중간 산출물 수정: 없음
- MC 정규화: 전체 physical dataset의 generator sum of weights를 분모로 사용
- 적용한 nominal event weight: pileup, b-tag, electron ID, muon ID를 포함

## 방법

### 1. 독립 MC closure

대상 MC를 이벤트 단위로 A와 B 두 표본으로 나눈다. A에서

\[
T_i^A = \frac{N_{\mathrm{target},i}^{0\ell,A}}
              {N_{\mathrm{target},i}^{1\ell,A}}
\]

를 측정하여 B의 one-lepton control yield에 적용하고, B의 실제 zero-lepton target yield와 비교한다. 반대 방향인 B-to-A도 수행한 뒤 두 방향을 합친 cross-fit 결과를 사용했다.

### 2. 전체 MC mixture closure

모든 배경 MC를 pseudo-data로 취급한다. one-lepton control region에서 비대상 MC를 차감한 뒤 같은 transfer factor를 적용하고, zero-lepton target region의 비대상 MC는 직접 더한다. 이 검사는 오염 차감과 전체 배경 조합이 코드상 올바른지 확인한다.

### 3. 데이터 validation

데이터에서 lost-lepton closure를 다른 배경으로 희석하지 않기 위해 다음의 잔차를 직접 비교했다.

\[
N_{\mathrm{pred}}^{\mathrm{LL}}
=
T_{\mathrm{MC}}
\left(N_{\mathrm{data}}^{1\ell}
-N_{\mathrm{other\,MC}}^{1\ell}\right)
\]

\[
N_{\mathrm{obs}}^{\mathrm{LL}}
=
N_{\mathrm{data}}^{0\ell}
-N_{\mathrm{other\,MC}}^{0\ell}
\]

전체 배경 예측과 데이터의 비교도 JSON에 보조 지표로 저장했지만, lost-lepton closure 판정에는 위 잔차 비교를 사용했다.

## 주요 결과

### MC 내부 closure

| 구분 | 유효 bin | \(\chi^2/\mathrm{ndf}\) | 최대 \(|\mathrm{pull}|\) |
|---|---:|---:|---:|
| high-dM \(p_{\mathrm{T}}^{\mathrm{miss}}\) | 6/7 | 0.0029/6 | 0.054 |
| high-dM search bins | 29/60 | 0.0926/29 | 0.173 |
| low-dM search bins | 36/42 | 0.1402/36 | 0.206 |

전체 MC mixture pseudo-data closure도 최대 pull이 0.18 이하로 통과했다. 이는 같은 generator에서 나눈 두 대규모 표본의 비교이므로 매우 좋은 p-value 자체를 물리적인 data closure의 증거로 해석하면 안 된다. 이 결과가 보여주는 것은 구현, 정규화, bin mapping 및 비대상 배경 차감의 내부 일관성이다.

search-bin closure에서 high-dM 31개 bin과 low-dM 6개 bin은 유효 통계 기준을 만족하지 못했다. 현재 fine-bin transfer factor를 그대로 사용하기에는 통계가 부족하며, 더 거친 계층적 transfer factor 또는 정규화된 parameterization이 필요하다.

### 데이터 validation

| Validation region | 유효 bin | LL purity: 1-lepton / 0-lepton | 적분 잔차 예측/관측 | p-value | 최대 \(|\mathrm{pull}|\) |
|---|---:|---:|---:|---:|---:|
| high-dM, \(N_b=0\) | 6/7 | 0.928 / 0.425 | 0.653 | \(2.0\times10^{-77}\) | 12.21 |
| high-dM, \(3\leq N_j\leq4,\ N_b\geq1\) | 7/7 | 0.967 / 0.630 | 0.676 | \(7.2\times10^{-264}\) | 19.71 |
| low-dM, \(250<p_{\mathrm{T}}^{\mathrm{miss}}<300\) GeV | 2/2 | 0.927 / 0.451 | 0.842 | \(4.2\times10^{-21}\) | 7.97 |
| low-dM, \(200<p_{\mathrm{T}}^{\mathrm{ISR}}<300\) GeV | 4/4 | 0.926 / 0.411 | 0.897 | \(7.0\times10^{-14}\) | 5.64 |
| low-dM, \(7<S_{\mathrm{MET}}<10\) | 3/3 | 0.937 / 0.422 | 1.020 | 0.945 | 0.53 |

high-dM에서는 lost-lepton 예측이 데이터 잔차의 약 65--68%에 그친다. low-dM의 낮은 MET 및 ISR sideband에서도 각각 약 84%와 90%에 그친다. MET significance sideband만 통계적으로 양립하지만, zero-lepton target purity가 42%에 불과해 큰 비대상 배경 차감에 의존하므로 이것 하나만으로 전체 방법의 closure를 주장할 수 없다.

전체 배경 예측/데이터 비는 같은 영역에서 약 0.81, 0.77, 0.93, 0.96, 1.01이다. 이 값이 lost-lepton 잔차 비보다 1에 가까운 것은 다른 배경이 nonclosure를 희석하기 때문이다. 따라서 전체 Data/MC만 보고 lost-lepton closure가 좋다고 판단해서는 안 된다.

## 추가로 확인된 물리적 문제

결합 transfer factor는 과정 조성에 민감하다. high-dM의 첫 MET bin에서 A-to-B transfer factor는 대략

- ST: 0.62
- TT: 0.76
- W+jets: 0.80

이다. 또한 MET가 증가하면서 one-lepton control sample의 TT 비율은 크게 감소하고 W+jets 및 ST 비율이 증가한다. 따라서 MC가 control과 target의 과정 조성을 정확히 재현하지 못하면 결합 transfer factor가 편향될 수 있다.

MC 내부 closure가 매우 좋은 반면 데이터에서만 큰 nonclosure가 나타나므로, 현재 증거는 단순한 코드 오류나 MC normalization 누락보다 다음 가능성을 지지한다.

1. `TT`, `W+jets`, `ST` 과정 조성의 mismodeling
2. 전자/뮤온 reconstruction, ID, isolation 또는 acceptance의 mismodeling
3. one-lepton control과 zero-lepton target 사이 trigger 효율 차이
4. zero-lepton validation 영역의 낮은 lost-lepton purity와 비대상 배경 차감 불확실성
5. recoil 또는 search-bin에 따른 transfer factor의 과도한 세분화

## 권고

1. 현재 결합 transfer factor를 nominal로 채택하지 않는다.
2. 전자와 뮤온 control region을 분리하여 nonclosure의 lepton-flavor 의존성을 검사한다.
3. `TT`, `W+jets`, `ST`별 transfer factor와 과정 조성 nuisance를 검토한다.
4. 더 높은 lost-lepton purity를 갖는 data validation region을 설계하거나, 비대상 배경을 동시에 constrain하는 likelihood 방식을 사용한다.
5. fine search-bin별 독립 transfer factor 대신 충분한 통계를 확보하는 coarse 또는 hierarchical parameterization을 시험한다.
6. 위 검증 전에 데이터 VR에서 보인 30--50% 차이를 임의의 보정계수로 nominal 예측에 적용하지 않는다.
7. 최종 AN용 결과에는 detector/model systematic과 signal contamination 검사를 추가한다. 이번 결과의 오차는 통계 covariance만 포함한다.

## 판정

- 구현 및 normalization sanity check: **통과**
- event-level fold 독립성 및 중복 검사: **통과**
- 순수 target MC closure: **통과**
- 전체 MC mixture closure: **통과**
- data residual closure: **실패**
- 현 transfer factor의 nominal 채택 가능 여부: **채택 불가**
