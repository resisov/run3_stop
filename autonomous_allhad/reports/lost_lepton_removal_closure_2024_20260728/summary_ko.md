# 2024 lost-lepton event-level removal closure 결과

## 결론

현재 구현한 **단일 four-vector lepton-removal 추정법은 채택하지 않는다.**

이 방법은 1-lepton event에서 선택된 lepton의 transverse momentum을
\(p_{\mathrm{T}}^{\mathrm{miss}}\)에 더한 뒤, zero-lepton selection과 search-bin
assignment를 다시 계산한다. 그러나 MC cross-fit에서 적분값을 target에 맞춘
뒤에도 \(U_{\mathrm{T}}\) 및 search-bin shape closure가 크게 실패한다. Data
validation region에서도 기존 Top/W transfer-factor 결과보다 일관되게 좋아지지
않는다.

Top 성분은 전체 과정에서 다음과 같이 정의했다.

- `Top = TT + ST`
- `W = WtoLNu`
- ST에 대한 독립 scale factor, transfer factor 또는 nuisance는 없다.

## 입력과 보호 조건

- 읽은 flat ROOT 파일: 1,153개
- 읽은 이벤트: 361,054,245개
- source 또는 target/VR 조건에 사용된 이벤트: 38,634,909개
- selection authority:
  `autonomous_allhad/autonomous_allhad/real_subset_worker.py`
- nominal 중간산출물 수정: 없음
- nominal high-/low-\(\Delta m\) SR data target: blinded, histogram yield 0

## 방법

1. Electron 또는 muon이 하나 있는 source event를 선택한다.
2. 선택된 lepton에 대해
   \[
   \vec{p}_{\mathrm{T}}^{\,\mathrm{miss,removed}}
   =
   \vec{p}_{\mathrm{T}}^{\,\mathrm{miss}}
   +
   \vec{p}_{\mathrm{T}}^{\,\ell}
   \]
   로 정의한다.
3. 수정된 \(p_{\mathrm{T}}^{\mathrm{miss}}\)로 \(\Delta\phi\), \(U_{\mathrm{T}}\),
   high-\(\Delta m\) 60 bins 및 low-\(\Delta m\) 42 bins를 다시 계산한다.
4. 서로 독립인 두 event-hash fold를 사용해 한 fold에서 residual
   lost/pass normalization \(\alpha\)를 측정하고 다른 fold를 예측하는
   cross-fit closure를 수행한다.
5. Top과 W는 독립적으로 \(\alpha\)를 측정하되, Top 안에서는 TT와 ST를
   항상 합친다.
6. Data VR 예측은 `Data 1-lepton source − Other MC source`에 fitted
   Top/W mixture로 얻은 effective lost/pass factor를 곱한다.

## Residual normalization

| Regime | Component | \(\alpha\) |
|---|---:|---:|
| high-\(\Delta m\) | Top = TT + ST | \(0.3663 \pm 0.0010\) |
| high-\(\Delta m\) | W | \(0.5297 \pm 0.0113\) |
| low-\(\Delta m\) | Top = TT + ST | \(0.2823 \pm 0.0015\) |
| low-\(\Delta m\) | W | \(0.3134 \pm 0.0021\) |

이 \(\alpha\) 때문에 아래 MC closure의 **적분비는 정의상 거의 1**이다.
따라서 판정에는 적분비가 아니라 shape \(\chi^2\), bin pull 및 분포 모양을
사용해야 한다.

## MC cross-fit shape closure

| Distribution | Removal \(\chi^2/\mathrm{ndf}\) | Removal max \(|pull|\) | 기존 TF \(\chi^2/\mathrm{ndf}\) | 기존 TF max \(|pull|\) |
|---|---:|---:|---:|---:|
| high-\(\Delta m\) 60 bins | \(13362.7/60\) | 51.08 | \(3.899/58\) | 1.19 |
| high-\(\Delta m\) \(U_{\mathrm{T}}\) | \(11361.1/7\) | 57.71 | \(0.004/7\) | 0.05 |
| low-\(\Delta m\) 42 bins | \(5674.2/42\) | 45.32 | \(0.175/42\) | 0.21 |

특히 high-\(\Delta m\) \(U_{\mathrm{T}}\)에서 removal/target은 낮은
\(U_{\mathrm{T}}\)에서 부족하고 중간 및 높은 \(U_{\mathrm{T}}\)에서 과대예측한다.
이는 단일 normalization으로 해결되지 않는 migration/response shape 문제다.

## Data validation regions

| Validation region | 기존 TF pred./obs. | Removal pred./obs. | Removal max \(|pull|\) |
|---|---:|---:|---:|
| high-\(\Delta m\), \(N_b=0\) | 0.654 | 0.598 | 16.06 |
| high-\(\Delta m\), \(3\leq N_j\leq4,\ N_b\geq1\) | 0.679 | 0.807 | 23.42 |
| low-\(\Delta m\), \(200<p_{\mathrm{T}}^{ISR}<300\) GeV | 0.800 | 0.542 | 23.25 |
| low-\(\Delta m\), \(250<p_{\mathrm{T}}^{miss}<300\) GeV | 0.776 | 0.367 | 30.92 |
| low-\(\Delta m\), \(7<p_{\mathrm{T}}^{miss}/\sqrt{H_{\mathrm{T}}}<10\) | 0.991 | 0.273 | 2.72 |

두 번째 high-\(\Delta m\) VR은 적분비의 unity 거리가
\(0.321\to0.193\)으로 줄지만, 최대 pull은 기존 19.64에서 23.42로 증가한다.
따라서 “1/5 VR 개선”은 적분비 하나에만 해당하며 shape closure 개선으로
간주하지 않는다.

## 실패 원인의 물리적 해석

단일 four-vector removal은 모든 lost-lepton event를 “lepton이 완전히
invisible했던 event”처럼 취급한다. 실제 lost-lepton background에는 서로 다른
response가 섞인다.

- acceptance 밖으로 나간 lepton
- reconstruction에 실패한 lepton
- ID 또는 isolation에 실패했지만 PF momentum은 event에 남아 있는 lepton
- hadronic tau decay

ID/isolation loss에서 lepton momentum이 detector response에 남는 경우까지
전부 \(p_{\mathrm{T}}^{miss}\)에 더하면 recoil과 \(\Delta\phi\) migration을
과도하게 만든다. 현재 관측된 \(U_{\mathrm{T}}\) shape distortion과 일치하는
설명이다.

또한 현재 flat preselection은 보통의 1-lepton event에 대해 원래
\(p_{\mathrm{T}}^{miss}>250\) GeV를 요구한다. 따라서 lepton removal 후
250 GeV 위로 이동할 원래 \(p_{\mathrm{T}}^{miss}<250\) GeV event가 입력에 없다.

## 다음에 채택할 방법

다음 iteration은 한 가지 removal response를 모든 lepton-loss mode에 적용하지
않는 hybrid efficiency/response 방법이어야 한다.

1. Data 1-lepton source를 출발점으로 사용한다.
2. acceptance/reconstruction loss와 ID/isolation loss를 분리한다.
3. acceptance/reconstruction loss에만 removal 또는 embedding response를
   적용한다.
4. ID/isolation loss에는 PF momentum을 유지하는 response를 사용한다.
5. electron, muon, hadronic tau를 분리하고
   \(p_{\mathrm{T}}^\ell,\eta^\ell,N_j,N_b,H_{\mathrm{T}}\) 의존 효율을 사용한다.
6. Top은 계속 `TT + ST`로 함께 움직이고 W만 독립 성분으로 둔다.
7. 원래 \(p_{\mathrm{T}}^{miss}\)가 아니라 1-lepton recoil을 기준으로 입력을
   materialize하여 threshold migration을 보존한다.
8. SR 적용 전에 동일한 다섯 orthogonal VR에서 closure gate를 통과시킨다.

## 산출물

- Web report: `index.html`
- Machine-readable result: `removal_closure_results.json`
- Reproduction manifest: `inputs/run_manifest.json`
- MC 및 Data VR figures: `plots/`

