# 2024 · 2025 배경추정 플롯 모음

최신 strict lepton veto pT > 10 GeV 및 Low-dM double-ratio 하한 250 GeV 결과입니다.

전체 64개 그림: PNG 64개 + PDF 64개 = 128개 파일.
TF는 연도별 High-dM 3장, Low-dM 4장입니다. Low-dM은 배경별 한 그림에 6개 카테고리를 함께 표시합니다.

## TF 표시 변경

- 빨강: Nb = 1. 파랑: Nb ≥ 2.
- 원: NISR = 0, 사각형: NISR = 1, 삼각형: NISR ≥ 2.
- 카테고리별 원래 GNN 빈 경계와 xerr를 유지합니다. 공통 빈으로 합산하거나 가로 위치를 임의 이동하지 않았습니다.
- 기존 plot_recoil_transfer_factors_2024.py의 렌더링을 확장했습니다. 저장된 TF JSON을 직접 읽는 --plot-only를 사용하므로 TF를 재계산하지 않습니다.
- 모음 폴더의 카테고리별 낱장 복사본 96개를 통합 그림으로 교체했습니다. 원본 검증 캠페인의 파일과 해시는 변경하지 않았습니다.
- 각 연도 tf/plot_manifest.json이 수치 원본 해시와 플롯 코드 해시를 기록합니다.

2025 데이터 luminosity coverage 불완전 표시는 원본과 동일합니다. RZ 보정 후 그림은 독립 closure가 아닙니다.

## 재현

저장소 루트에서 YEAR를 2024 또는 2025로 지정해 실행합니다.

```bash
PYTHONPATH=autonomous_allhad python3 autonomous_allhad/workflow/plot_recoil_transfer_factors_2024.py \
  --input autonomous_allhad/reports/background_estimation_lepton_veto10_20260908/$YEAR/tf/transfer_factors_${YEAR}_nb_recoil.json \
  --campaign-year $YEAR --plot-only \
  --output-dir autonomous_allhad/reports/background_estimation_plots_2024_2025_20260908/$YEAR/tf
```

## 2024

### TF — High-dM

- transfer_factor_qcd_qcdcr_highdm — [PNG](2024/tf/transfer_factor_qcd_qcdcr_highdm.png) · [PDF](2024/tf/transfer_factor_qcd_qcdcr_highdm.pdf)
- transfer_factor_top_llcr_highdm — [PNG](2024/tf/transfer_factor_top_llcr_highdm.png) · [PDF](2024/tf/transfer_factor_top_llcr_highdm.pdf)
- transfer_factor_w_llcr_highdm — [PNG](2024/tf/transfer_factor_w_llcr_highdm.png) · [PDF](2024/tf/transfer_factor_w_llcr_highdm.pdf)

### TF — Low-dM GNN (카테고리 통합)

- transfer_factor_qcd_qcdcr_lowdm_gnn — [PNG](2024/tf/gnn/transfer_factor_qcd_qcdcr_lowdm_gnn.png) · [PDF](2024/tf/gnn/transfer_factor_qcd_qcdcr_lowdm_gnn.pdf)
- transfer_factor_top_llcr_lowdm_gnn — [PNG](2024/tf/gnn/transfer_factor_top_llcr_lowdm_gnn.png) · [PDF](2024/tf/gnn/transfer_factor_top_llcr_lowdm_gnn.pdf)
- transfer_factor_w_llcr_lowdm_gnn — [PNG](2024/tf/gnn/transfer_factor_w_llcr_lowdm_gnn.png) · [PDF](2024/tf/gnn/transfer_factor_w_llcr_lowdm_gnn.pdf)
- transfer_factor_zinv_gcr_lowdm_gnn — [PNG](2024/tf/gnn/transfer_factor_zinv_gcr_lowdm_gnn.png) · [PDF](2024/tf/gnn/transfer_factor_zinv_gcr_lowdm_gnn.pdf)

### DY — High-dM

- mll_highdm_dy2e_nb1 — [PNG](2024/dy_report/highdm/mll_highdm_dy2e_nb1.png) · [PDF](2024/dy_report/highdm/mll_highdm_dy2e_nb1.pdf)
- mll_highdm_dy2e_nb1_post — [PNG](2024/dy_report/highdm/mll_highdm_dy2e_nb1_post.png) · [PDF](2024/dy_report/highdm/mll_highdm_dy2e_nb1_post.pdf)
- mll_highdm_dy2e_nb2plus — [PNG](2024/dy_report/highdm/mll_highdm_dy2e_nb2plus.png) · [PDF](2024/dy_report/highdm/mll_highdm_dy2e_nb2plus.pdf)
- mll_highdm_dy2e_nb2plus_post — [PNG](2024/dy_report/highdm/mll_highdm_dy2e_nb2plus_post.png) · [PDF](2024/dy_report/highdm/mll_highdm_dy2e_nb2plus_post.pdf)
- mll_highdm_dy2m_nb1 — [PNG](2024/dy_report/highdm/mll_highdm_dy2m_nb1.png) · [PDF](2024/dy_report/highdm/mll_highdm_dy2m_nb1.pdf)
- mll_highdm_dy2m_nb1_post — [PNG](2024/dy_report/highdm/mll_highdm_dy2m_nb1_post.png) · [PDF](2024/dy_report/highdm/mll_highdm_dy2m_nb1_post.pdf)
- mll_highdm_dy2m_nb2plus — [PNG](2024/dy_report/highdm/mll_highdm_dy2m_nb2plus.png) · [PDF](2024/dy_report/highdm/mll_highdm_dy2m_nb2plus.pdf)
- mll_highdm_dy2m_nb2plus_post — [PNG](2024/dy_report/highdm/mll_highdm_dy2m_nb2plus_post.png) · [PDF](2024/dy_report/highdm/mll_highdm_dy2m_nb2plus_post.pdf)
- rt_highdm — [PNG](2024/dy_report/highdm/rt_highdm.png) · [PDF](2024/dy_report/highdm/rt_highdm.pdf)
- rz_highdm — [PNG](2024/dy_report/highdm/rz_highdm.png) · [PDF](2024/dy_report/highdm/rz_highdm.pdf)

### DY — Low-dM

- mll_lowdm_dy2e_nb1 — [PNG](2024/dy_report/lowdm/mll_lowdm_dy2e_nb1.png) · [PDF](2024/dy_report/lowdm/mll_lowdm_dy2e_nb1.pdf)
- mll_lowdm_dy2e_nb1_post — [PNG](2024/dy_report/lowdm/mll_lowdm_dy2e_nb1_post.png) · [PDF](2024/dy_report/lowdm/mll_lowdm_dy2e_nb1_post.pdf)
- mll_lowdm_dy2e_nb2plus — [PNG](2024/dy_report/lowdm/mll_lowdm_dy2e_nb2plus.png) · [PDF](2024/dy_report/lowdm/mll_lowdm_dy2e_nb2plus.pdf)
- mll_lowdm_dy2e_nb2plus_post — [PNG](2024/dy_report/lowdm/mll_lowdm_dy2e_nb2plus_post.png) · [PDF](2024/dy_report/lowdm/mll_lowdm_dy2e_nb2plus_post.pdf)
- mll_lowdm_dy2m_nb1 — [PNG](2024/dy_report/lowdm/mll_lowdm_dy2m_nb1.png) · [PDF](2024/dy_report/lowdm/mll_lowdm_dy2m_nb1.pdf)
- mll_lowdm_dy2m_nb1_post — [PNG](2024/dy_report/lowdm/mll_lowdm_dy2m_nb1_post.png) · [PDF](2024/dy_report/lowdm/mll_lowdm_dy2m_nb1_post.pdf)
- mll_lowdm_dy2m_nb2plus — [PNG](2024/dy_report/lowdm/mll_lowdm_dy2m_nb2plus.png) · [PDF](2024/dy_report/lowdm/mll_lowdm_dy2m_nb2plus.pdf)
- mll_lowdm_dy2m_nb2plus_post — [PNG](2024/dy_report/lowdm/mll_lowdm_dy2m_nb2plus_post.png) · [PDF](2024/dy_report/lowdm/mll_lowdm_dy2m_nb2plus_post.pdf)
- rt_lowdm — [PNG](2024/dy_report/lowdm/rt_lowdm.png) · [PDF](2024/dy_report/lowdm/rt_lowdm.pdf)
- rz_lowdm — [PNG](2024/dy_report/lowdm/rz_lowdm.png) · [PDF](2024/dy_report/lowdm/rz_lowdm.pdf)

### Q · Sgamma

- photon_q_normalization — [PNG](2024/sgamma/photon_q_normalization.png) · [PDF](2024/sgamma/photon_q_normalization.pdf)
- sgamma_highdm — [PNG](2024/sgamma/sgamma_highdm.png) · [PDF](2024/sgamma/sgamma_highdm.pdf)
- sgamma_lowdm — [PNG](2024/sgamma/sgamma_lowdm.png) · [PDF](2024/sgamma/sgamma_lowdm.pdf)

### Z/gamma double ratio

- zgamma_double_ratio_highdm — [PNG](2024/zgamma/zgamma_double_ratio_highdm.png) · [PDF](2024/zgamma/zgamma_double_ratio_highdm.pdf)
- zgamma_double_ratio_lowdm — [PNG](2024/zgamma/zgamma_double_ratio_lowdm.png) · [PDF](2024/zgamma/zgamma_double_ratio_lowdm.pdf)

## 2025

### TF — High-dM

- transfer_factor_qcd_qcdcr_highdm — [PNG](2025/tf/transfer_factor_qcd_qcdcr_highdm.png) · [PDF](2025/tf/transfer_factor_qcd_qcdcr_highdm.pdf)
- transfer_factor_top_llcr_highdm — [PNG](2025/tf/transfer_factor_top_llcr_highdm.png) · [PDF](2025/tf/transfer_factor_top_llcr_highdm.pdf)
- transfer_factor_w_llcr_highdm — [PNG](2025/tf/transfer_factor_w_llcr_highdm.png) · [PDF](2025/tf/transfer_factor_w_llcr_highdm.pdf)

### TF — Low-dM GNN (카테고리 통합)

- transfer_factor_qcd_qcdcr_lowdm_gnn — [PNG](2025/tf/gnn/transfer_factor_qcd_qcdcr_lowdm_gnn.png) · [PDF](2025/tf/gnn/transfer_factor_qcd_qcdcr_lowdm_gnn.pdf)
- transfer_factor_top_llcr_lowdm_gnn — [PNG](2025/tf/gnn/transfer_factor_top_llcr_lowdm_gnn.png) · [PDF](2025/tf/gnn/transfer_factor_top_llcr_lowdm_gnn.pdf)
- transfer_factor_w_llcr_lowdm_gnn — [PNG](2025/tf/gnn/transfer_factor_w_llcr_lowdm_gnn.png) · [PDF](2025/tf/gnn/transfer_factor_w_llcr_lowdm_gnn.pdf)
- transfer_factor_zinv_gcr_lowdm_gnn — [PNG](2025/tf/gnn/transfer_factor_zinv_gcr_lowdm_gnn.png) · [PDF](2025/tf/gnn/transfer_factor_zinv_gcr_lowdm_gnn.pdf)

### DY — High-dM

- mll_highdm_dy2e_nb1 — [PNG](2025/dy_report/highdm/mll_highdm_dy2e_nb1.png) · [PDF](2025/dy_report/highdm/mll_highdm_dy2e_nb1.pdf)
- mll_highdm_dy2e_nb1_post — [PNG](2025/dy_report/highdm/mll_highdm_dy2e_nb1_post.png) · [PDF](2025/dy_report/highdm/mll_highdm_dy2e_nb1_post.pdf)
- mll_highdm_dy2e_nb2plus — [PNG](2025/dy_report/highdm/mll_highdm_dy2e_nb2plus.png) · [PDF](2025/dy_report/highdm/mll_highdm_dy2e_nb2plus.pdf)
- mll_highdm_dy2e_nb2plus_post — [PNG](2025/dy_report/highdm/mll_highdm_dy2e_nb2plus_post.png) · [PDF](2025/dy_report/highdm/mll_highdm_dy2e_nb2plus_post.pdf)
- mll_highdm_dy2m_nb1 — [PNG](2025/dy_report/highdm/mll_highdm_dy2m_nb1.png) · [PDF](2025/dy_report/highdm/mll_highdm_dy2m_nb1.pdf)
- mll_highdm_dy2m_nb1_post — [PNG](2025/dy_report/highdm/mll_highdm_dy2m_nb1_post.png) · [PDF](2025/dy_report/highdm/mll_highdm_dy2m_nb1_post.pdf)
- mll_highdm_dy2m_nb2plus — [PNG](2025/dy_report/highdm/mll_highdm_dy2m_nb2plus.png) · [PDF](2025/dy_report/highdm/mll_highdm_dy2m_nb2plus.pdf)
- mll_highdm_dy2m_nb2plus_post — [PNG](2025/dy_report/highdm/mll_highdm_dy2m_nb2plus_post.png) · [PDF](2025/dy_report/highdm/mll_highdm_dy2m_nb2plus_post.pdf)
- rt_highdm — [PNG](2025/dy_report/highdm/rt_highdm.png) · [PDF](2025/dy_report/highdm/rt_highdm.pdf)
- rz_highdm — [PNG](2025/dy_report/highdm/rz_highdm.png) · [PDF](2025/dy_report/highdm/rz_highdm.pdf)

### DY — Low-dM

- mll_lowdm_dy2e_nb1 — [PNG](2025/dy_report/lowdm/mll_lowdm_dy2e_nb1.png) · [PDF](2025/dy_report/lowdm/mll_lowdm_dy2e_nb1.pdf)
- mll_lowdm_dy2e_nb1_post — [PNG](2025/dy_report/lowdm/mll_lowdm_dy2e_nb1_post.png) · [PDF](2025/dy_report/lowdm/mll_lowdm_dy2e_nb1_post.pdf)
- mll_lowdm_dy2e_nb2plus — [PNG](2025/dy_report/lowdm/mll_lowdm_dy2e_nb2plus.png) · [PDF](2025/dy_report/lowdm/mll_lowdm_dy2e_nb2plus.pdf)
- mll_lowdm_dy2e_nb2plus_post — [PNG](2025/dy_report/lowdm/mll_lowdm_dy2e_nb2plus_post.png) · [PDF](2025/dy_report/lowdm/mll_lowdm_dy2e_nb2plus_post.pdf)
- mll_lowdm_dy2m_nb1 — [PNG](2025/dy_report/lowdm/mll_lowdm_dy2m_nb1.png) · [PDF](2025/dy_report/lowdm/mll_lowdm_dy2m_nb1.pdf)
- mll_lowdm_dy2m_nb1_post — [PNG](2025/dy_report/lowdm/mll_lowdm_dy2m_nb1_post.png) · [PDF](2025/dy_report/lowdm/mll_lowdm_dy2m_nb1_post.pdf)
- mll_lowdm_dy2m_nb2plus — [PNG](2025/dy_report/lowdm/mll_lowdm_dy2m_nb2plus.png) · [PDF](2025/dy_report/lowdm/mll_lowdm_dy2m_nb2plus.pdf)
- mll_lowdm_dy2m_nb2plus_post — [PNG](2025/dy_report/lowdm/mll_lowdm_dy2m_nb2plus_post.png) · [PDF](2025/dy_report/lowdm/mll_lowdm_dy2m_nb2plus_post.pdf)
- rt_lowdm — [PNG](2025/dy_report/lowdm/rt_lowdm.png) · [PDF](2025/dy_report/lowdm/rt_lowdm.pdf)
- rz_lowdm — [PNG](2025/dy_report/lowdm/rz_lowdm.png) · [PDF](2025/dy_report/lowdm/rz_lowdm.pdf)

### Q · Sgamma

- photon_q_normalization — [PNG](2025/sgamma/photon_q_normalization.png) · [PDF](2025/sgamma/photon_q_normalization.pdf)
- sgamma_highdm — [PNG](2025/sgamma/sgamma_highdm.png) · [PDF](2025/sgamma/sgamma_highdm.pdf)
- sgamma_lowdm — [PNG](2025/sgamma/sgamma_lowdm.png) · [PDF](2025/sgamma/sgamma_lowdm.pdf)

### Z/gamma double ratio

- zgamma_double_ratio_highdm — [PNG](2025/zgamma/zgamma_double_ratio_highdm.png) · [PDF](2025/zgamma/zgamma_double_ratio_highdm.pdf)
- zgamma_double_ratio_lowdm — [PNG](2025/zgamma/zgamma_double_ratio_lowdm.png) · [PDF](2025/zgamma/zgamma_double_ratio_lowdm.pdf)
