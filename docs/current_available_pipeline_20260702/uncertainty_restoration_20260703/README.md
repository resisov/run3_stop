# Uncertainty restoration 2026-07-03

Restored btag SF weight variations into the current flat recoil histogram payload and rebuilt recoil-bin Combine inputs for SR and SR_Nt1.

JEC/MET shape variations were not attached to the recoil-bin datacards because no compatible valid shifted+nominal recoil-bin source is currently available on disk.

## Impact summary

### SR
- prop_bincat7_SR_recoil_bin5: impact_r = 0.19098222255706787
- prop_bincat7_SR_recoil_bin4: impact_r = 0.15140926837921143
- Lumi_2024: impact_r = 0.14282941818237305
- electron_id: impact_r = 0.11458325386047363
- prop_bincat3_QCDCR_highDeltaM_bin0: impact_r = 0.10625278949737549
- btagSF_bc_correlated: impact_r = 0.09563052654266357
- btagSF_bc_correlated: impact_r = 0.09563052654266357
- btagSF_bc_uncorrelated: impact_r = 0.09065604209899902
- btagSF_light_correlated: impact_r = 0.09474122524261475
- btagSF_light_uncorrelated: impact_r = 0.08793222904205322

### SR_Nt1
- prop_bincat7_SR_recoil_bin5: impact_r = 0.09237146377563477
- prop_bincat7_SR_recoil_bin4: impact_r = 0.07474136352539062
- Lumi_2024: impact_r = 0.04881751537322998
- electron_id: impact_r = 0.03818166255950928
- prop_bincat3_QCDCR_highDeltaM_bin0: impact_r = 0.03711867332458496
- prop_bincat3_QCDCR_highDeltaM_bin1: impact_r = 0.024240612983703613
- btagSF_bc_correlated: impact_r = 0.012243509292602539
- btagSF_bc_uncorrelated: impact_r = 0.010562300682067871
- btagSF_light_correlated: impact_r = 0.007903575897216797
- btagSF_light_uncorrelated: impact_r = 0.008826255798339844
