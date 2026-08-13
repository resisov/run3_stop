# Top/W 분리 Lost-lepton closure 결과

## 결론

`Top = TT + single-top`과 `W+jets`를 분리하고, W-enriched 및 Top-enriched one-lepton control category를 동시에 사용하여 두 정규화를 데이터에서 별도로 결정했다. 이 변경은 high-dM closure를 사실상 개선하지 못했고 low-dM closure는 오히려 악화시켰다.

따라서 기존 nonclosure의 주원인은 `Top`과 `W+jets`를 하나의 transfer factor로 합친 것만으로 설명되지 않는다. 특히 low-dM에서는 두 성분의 전역 normalization으로 서로 다른 \(N_b\) category를 동시에 기술할 수 없는 문제가 확인됐다.

이번 결과는 closure용 물리 제안이며 nominal 중간 산출물은 수정하지 않았다. QCD는 다른 배경과 함께 residual 오염 차감에만 사용했으며 QCD closure는 수행하지 않았다.

## 구현한 방법

각 분석 영역에서 Top과 W+jets에 독립적인 control-region normalization을 부여했다.

\[
N_{\mathrm{pred},i}^{\mathrm{LL}}
=
\mu_{\mathrm{Top}}N_{\mathrm{Top},i}^{0\ell,\mathrm{MC}}
+
\mu_WN_{W,i}^{0\ell,\mathrm{MC}}
\]

이는 과정별 transfer factor를 사용한 다음 식과 같다.

\[
N_{\mathrm{pred},i}^{\mathrm{LL}}
=
T_{\mathrm{Top},i}
N_{\mathrm{Top},i}^{1\ell,\mathrm{data\ fit}}
+
T_{W,i}
N_{W,i}^{1\ell,\mathrm{data\ fit}}
\]

### High-dM fit

- W-enriched anchor: \(N_b=0,\ N_j\geq5\)
- Top-enriched anchor: \(N_b\geq1,\ 3\leq N_j\leq4\)
- 두 category의 적분 one-lepton data residual을 동시에 풀어 \(\mu_{\mathrm{Top}}\)과 \(\mu_W\)를 결정

### Low-dM fit

- W-enriched anchor: nominal LLCR의 \(N_b=0\) search bins
- Top-enriched anchor: nominal LLCR의 \(N_b\geq2\) search bins
- \(N_b=1\) category는 fit에 사용하지 않고 독립적인 control-region validation으로 보존

각 anchor에는 Top과 W가 모두 섞여 있다. 따라서 hard assignment를 하지 않고 다음의 \(2\times2\) 혼합 방정식을 풀었다.

\[
\begin{pmatrix}
D_1-O_1\\
D_2-O_2
\end{pmatrix}
=
\begin{pmatrix}
T_1&W_1\\
T_2&W_2
\end{pmatrix}
\begin{pmatrix}
\mu_{\mathrm{Top}}\\
\mu_W
\end{pmatrix}
\]

여기서 \(D\)는 control data, \(O\)는 Top/W 이외의 MC, \(T\)와 \(W\)는 각각 Top 및 W+jets MC yield이다.

## Control-region normalization 결과

| 영역 | \(\mu_{\mathrm{Top}}\) | \(\mu_W\) | 상관계수 |
|---|---:|---:|---:|
| high-dM | \(0.8823\pm0.0101\) | \(1.0504\pm0.0175\) | -0.659 |
| low-dM | \(0.9514\pm0.0178\) | \(0.7300\pm0.0086\) | -0.083 |

오차는 통계만 포함한다.

High-dM에서는 Top을 약 12% 낮추고 W+jets를 약 5% 높이는 결과가 나온다. 두 변화가 서로 일부 상쇄되므로 결합 TF 예측과 분리 예측이 거의 같아진다.

Low-dM에서는 W+jets normalization이 약 27% 낮아진다. 이 값을 zero-lepton target에 적용하면 W 성분이 크게 감소하여 기존 결합 TF보다 더 낮은 lost-lepton 예측이 나온다.

## 독립적인 CR 진단

High-dM의 두 anchor는 적분 yield를 이용해 두 미지수를 결정하므로 적분값은 정의상 맞는다. 따라서 같은 category 내부의 \(p_{\mathrm{T}}^{\mathrm{miss}}\) shape를 별도로 검사했다.

| High-dM one-lepton category | 적분 예측/관측 | shape p-value | 최대 pull |
|---|---:|---:|---:|
| \(N_b=0\), W enriched | 1.000 | \(7.95\times10^{-4}\) | 3.55 |
| \(3\leq N_j\leq4,\ N_b\geq1\), Top enriched | 1.000 | 0.483 | 1.47 |

Top-enriched CR shape는 통계적으로 양립하지만 W-enriched CR에는 recoil-dependent shape 불일치가 남는다. 즉 W normalization 하나로 high-dM W-enriched CR 전체를 설명할 수 없다.

Low-dM에서는 fit에서 제외한 \(N_b=1\) category가 중요한 독립 검사가 된다.

- 예측: \(6814.8\pm81.2\)
- data residual: \(8649.9\pm97.3\)
- 예측/관측: \(0.7878\pm0.0129\)
- pull: -14.48

\(N_b=0\)과 \(N_b\geq2\)에서 얻은 Top/W normalization이 \(N_b=1\) control data를 약 21% 과소예측한다. 이는 단순한 두 과정의 전체 normalization보다 \(N_b\) migration, b-tag modeling, 과정별 세부 조성 또는 kinematic shape에 문제가 있음을 시사한다.

## Data residual closure 비교

아래 적분비는 통계 기준을 만족한 bin만 합산한 lost-lepton 예측/data residual이다.

| Validation region | 기존 combined TF | Top/W 분리 | 변화 |
|---|---:|---:|---:|
| high-dM \(N_b=0\) | 0.6526 | \(0.6539\pm0.0211\) | +0.0013 |
| high-dM \(3\leq N_j\leq4,\ N_b\geq1\) | 0.6757 | \(0.6794\pm0.0078\) | +0.0037 |
| low-dM 낮은 \(p_{\mathrm{T}}^{\mathrm{miss}}\) | 0.8417 | \(0.7763\pm0.0143\) | -0.0654 |
| low-dM 낮은 ISR | 0.8966 | \(0.8001\pm0.0128\) | -0.0966 |
| low-dM 낮은 MET significance | 1.0197 | \(0.9911\pm0.2919\) | -0.0286 |

High-dM에서는 변화가 0.4 percentage point 이하로 사실상 동일하다. 따라서 약 32--35%의 high-dM deficit은 Top/W 결합 TF 때문에 생긴 것이 아니다.

Low-dM에서는 별도로 측정한 낮은 W normalization 때문에 예측이 6--10 percentage point 더 감소한다. 낮은 MET와 낮은 ISR validation에서는 기존보다 명백히 악화된다. MET significance sideband만 1과 양립하지만 오차가 매우 크고 zero-lepton lost-lepton purity가 낮아 전체 방법의 validation으로 사용할 수 없다.

## 통계 검정

| Validation region | Top/W 분리 p-value | 최대 pull |
|---|---:|---:|
| high-dM \(N_b=0\) | \(1.6\times10^{-75}\) | 14.51 |
| high-dM Top-enriched | \(4.0\times10^{-268}\) | 19.64 |
| low-dM 낮은 MET | \(1.4\times10^{-36}\) | 10.58 |
| low-dM 낮은 ISR | \(4.2\times10^{-42}\) | 10.23 |
| low-dM 낮은 MET significance | 0.969 | 0.42 |

Top/W normalization covariance는 target prediction의 모든 bin 사이에 전파했다. 데이터, 다른 MC 차감, Top/W MC template의 통계 오차도 포함했다. detector/model systematic은 아직 포함하지 않았다.

## 물리적 해석

이번 검사는 다음을 명확하게 구분한다.

1. **Top/W 결합 문제:** high-dM에서는 분리 전후 결과가 거의 같으므로 주원인이 아니다.
2. **W-enriched recoil shape:** high-dM \(N_b=0\) CR 자체에 shape 불일치가 남는다.
3. **\(N_b\) category 이동:** low-dM \(N_b=1\) CR을 두 전역 normalization이 설명하지 못한다.
4. **Top 내부 조성:** 현재 Top은 TT와 ST를 묶었다. 두 과정의 transfer factor 차이가 크므로 ST/TT composition도 후속 검사가 필요하다.
5. **lepton flavor 및 효율:** 전자/뮤온을 합쳤으므로 flavor별 reconstruction, ID, isolation 또는 trigger 차이가 아직 가려져 있다.

## 판정

- Top/W 분리 simultaneous fit 구현: **완료**
- normalization 및 covariance 전파: **완료**
- high-dM closure 개선: **아니오**
- low-dM closure 개선: **아니오, 악화**
- 독립 low-dM \(N_b=1\) CR validation: **실패**
- 현재 Top/W 분리 방법의 nominal 채택: **불가**

다음으로 필요한 최소 진단은 전자와 뮤온 분리, TT와 ST 분리, 그리고 \(N_b\) category별 b-tag/migration 검증이다. 현재 결과만으로 30--50%의 경험적 보정계수를 nominal에 적용해서는 안 된다.
