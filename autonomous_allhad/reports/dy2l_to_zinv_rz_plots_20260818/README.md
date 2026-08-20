# DY(2ℓ) 기반 Z→νν 정규화 보정 보고서

## 목적

신호영역의 `Z→νν+jets` 배경은 최종상태에 중성미자만 남으므로 데이터에서
직접 분리해 정규화하기 어렵다. 대신 같은 Z+jets 생성과정을 공유하면서
재구성 가능한 `Z→ee` 및 `Z→μμ` 데이터를 사용해 Z+jets MC의 정규화 보정
계수 `R_Z`를 측정한다. 측정된 `R_Z`는 `Z→νν` MC에 전달한다.

이 측정은 High-Δm과 Low-Δm을 분리하고, 각 영역에서 `N_b=1`과
`N_b≥2`를 분리해 수행한다. `R_Z`는 `U_T`에 따라 다시 나누지 않는다.

## On-Z와 off-Z 영역

- on-Z: `81 < m_ll < 101 GeV`
- off-Z: `50 < m_ll < 81 GeV` 또는 `m_ll > 101 GeV`

on-Z에는 DY가 우세하지만 다른 배경도 남는다. off-Z에는 DY와 다른 배경의
비율이 달라진다. 두 영역을 동시에 사용하면 DY 보정 `R_Z`와 나머지 배경
보정 `R_T`를 함께 구할 수 있다.

## 연립방정식

각 lepton channel `c = ee, μμ`와 b-tag category `b`에 대해 다음 수율을
정의한다.

- `D_on`, `D_off`: 데이터 수율
- `Z_on`, `Z_off`: DY MC 수율
- `T_on`, `T_off`: DY 이외의 모든 MC 수율

데이터 수율은 다음 두 식으로 표현한다.

```text
D_on  = R_Z · Z_on  + R_T · T_on
D_off = R_Z · Z_off + R_T · T_off
```

행렬 형태는

```text
┌ D_on  ┐   ┌ Z_on   T_on  ┐ ┌ R_Z ┐
│       │ = │              │ │     │
└ D_off ┘   └ Z_off  T_off ┘ └ R_T ┘
```

이다. 행렬식

```text
Δ = Z_on · T_off − Z_off · T_on
```

을 이용하면 대수적인 해는

```text
R_Z = (D_on · T_off − D_off · T_on) / Δ
R_T = (Z_on · D_off − Z_off · D_on) / Δ
```

가 된다. 실제 구현은 데이터 수율에 Poisson likelihood를 사용하고, 가중 MC
template의 `sumw2`를 Gaussian constraint로 profile하는 non-negative
likelihood fit이다. 해가 양수인 내부영역에서는 위 연립방정식의 해와 동일한
중앙값을 준다.

`R_T`는 DY 오염을 올바르게 분리하기 위한 보조계수다. `R_T` 자체를
`Z→νν` 예측에 전달하지 않는다.

## ee와 μμ 결합

두 channel에서 각각 `R_Z`를 구한 뒤 통계오차의 역분산으로 결합한다.

```text
w_c = 1 / σ_c²
R_Z(combined) = Σ_c w_c R_Z,c / Σ_c w_c
σ(combined) = 1 / √(Σ_c w_c)
```

## 측정 결과

| 영역 | `N_b=1` | `N_b≥2` |
|---|---:|---:|
| High-Δm | `0.722 ± 0.047` | `0.701 ± 0.075` |
| Low-Δm | `0.609 ± 0.033` | `0.628 ± 0.076` |

따라서 nominal Z+jets MC는 모든 category에서 데이터가 선호하는 값보다
높으며, `R_Z < 1` 보정으로 낮아진다.

## 적용식

DYCR의 적용 후 `m_ll` 확인 플롯은 다음 예측을 표시한다.

```text
N_post(m_ll) = R_Z · N_DY_MC(m_ll) + R_T · N_other_MC(m_ll)
```

신호영역의 invisible-Z 배경에는

```text
N_Z→νν_pred(g,i) = R_Z[k(g)] · S_γ(g,i) · N_Z→νν_MC(g,i)
```

를 적용한다. 여기서 `k(g)`는 해당 High-/Low-Δm 및 `N_b` category이고,
`S_γ(g,i)`는 photon control sample에서 측정한 `U_T` shape 보정이다.
GCR을 포함한 최종 동시피팅에서는 같은 식에 GCR과 공유되는 residual
parameter `ρ_γ(g,i)`가 추가된다.

```text
N_Z→νν_pred(g,i) = ρ_γ(g,i) · R_Z[k(g)] · S_γ(g,i)
                   · N_Z→νν_MC(g,i)
```

## 플롯

### R_Z

The R_Z summary plots show the channel-specific ee and mumu measurements and
their inverse-variance weighted combined result.

- [High-Δm R_Z](highdm/rz_highdm.png)
- [Low-Δm R_Z](lowdm/rz_lowdm.png)

### R_T

The channel-specific R_T factors scale the non-DY contamination in the on-Z
and off-Z matrix inputs. They are not combined or propagated to Z→νν.

- [High-Δm R_T](highdm/rt_highdm.png)
- [Low-Δm R_T](lowdm/rt_lowdm.png)

### R_Z와 R_T 적용 후 DYCR m_ll

- [High-Δm ee, N_b=1](highdm/mll_highdm_dy2e_nb1_post.png)
- [High-Δm ee, N_b≥2](highdm/mll_highdm_dy2e_nb2plus_post.png)
- [High-Δm μμ, N_b=1](highdm/mll_highdm_dy2m_nb1_post.png)
- [High-Δm μμ, N_b≥2](highdm/mll_highdm_dy2m_nb2plus_post.png)
- [Low-Δm ee, N_b=1](lowdm/mll_lowdm_dy2e_nb1_post.png)
- [Low-Δm ee, N_b≥2](lowdm/mll_lowdm_dy2e_nb2plus_post.png)
- [Low-Δm μμ, N_b=1](lowdm/mll_lowdm_dy2m_nb1_post.png)
- [Low-Δm μμ, N_b≥2](lowdm/mll_lowdm_dy2m_nb2plus_post.png)
