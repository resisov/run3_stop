# Local / EOS synchronization conflicts

Initial inspection snapshot: 2026-09-10. Existing local deletions were explicitly confirmed by the user.

Resolution: [canonical_decision.json](canonical_decision.json) records the main agent’s current-use decision. EOS dataset/metadata/batch-runner versions and local GNN plotting versions were adopted. Documentation and ignore rules use the reviewed local version. Seven inactive or unproven reference/test variants remain preserved; their initial hashes below are historical evidence, not a claim that all 19 conflicts remain open.

The active EOS checkout and local branch are not interchangeable. A common Git commit alone would not validate these physics inputs.

Concrete differences requiring physics review:

- 2025 jet veto: local `Summer24Prompt25_RunCDEFG_V1`; EOS `Winter25Prompt25_RunCDEFG_V1`. The decompressed payloads differ.
- 2024 metadata: local 5,331 keys; EOS 5,338. The 5,331 common entries are equal; EOS adds TTWW, TTWZ, TTZZ, WWW, WWZ, WZZ, and ZZZ. The dataset list differs correspondingly.
- Baseline processor: local adds `dataset_group`, TT grouping and initialized fallback grouping; EOS retains the previous grouping implementation.
- Trigger measurement: local and EOS use different denominator/object-selection code and interfaces. This is not formatting-only.
- Batch entrypoint: local removes two legacy argv allowlist entries that EOS still permits.

Do not select a side automatically or overwrite frozen campaign code. Differences in tests, plotting, documentation and dependencies are listed below too.

| Path | Local SHA-256 | EOS SHA-256 |
|---|---|---|
| `.gitignore` | `b395daa7824412bbaa8af6d89b7e95f0535079134559c4196f344708e81b9fec` | `3d8c687bc51758399c365fec2c3ab4a80aa0f0d049d390d3d2524afabeee82b1` |
| `README.md` | `1e1fbda1d6006f7ec2b10874798f194b6d69a36c83760119827fa4d287173026` | `dc7d8609f8cdc1f9b0a266f93dabed1ba4c8bf58de2d60eec3ec90fa0ec3b568` |
| `analysis/data/JMESF/2025/jetvetomaps.json.gz` | `9b6f0ad931b31752111e6aceb88e3b806010491473b39e97d4a4e83772ec7151` | `61de83931454ff37c9cc1df476d03dde9bb7871e85558018ce3304d5224c939e` |
| `analysis/datasets/datasets_2024.txt` | `8cdd7f24f080dcf1b54becdb9a7adf3f60fa75f03a6db30b7f929aebfe2b4d6c` | `d9de62e8684159db66788872bf6b3851b8756a806aefffc3f17badedd9fb56b4` |
| `analysis/metadata/KNU_2024_v4.json.gz` | `4f720444db255bbb85ac1ad8d30e4bee56673ba38e60b0614d8fbf0f1cb1ed0c` | `365df1a13c0a61567b70a2136cc083ae78d2fa3c013f0f0fac4be738f934b863` |
| `analysis/metadata/json_maker_lxplus.py` | `873989db5ee27783d698c3d59a5bd66ba3c78b2346a7254cce0f35ec0974c17d` | `7c3a01b49832fb28f794f86aec6956640b7df61d6d7e3db0972bf437483dfe4f` |
| `analysis/processors/stop_processor_v4.py` | `ac975d5f7bf542f3c3bca92da7e017d14160a7193c245b8f5610f10e1641c3bd` | `a9f66c5d19682a57731b9399ff1f123d2c328d2403ec66625a0b139a08c6e4a0` |
| `autonomous_allhad/autonomous_allhad/dy_estimation/README.md` | `624094cbcfff83b8af9ab2ad989a8c5ba1d6b2558cccf07162a2aae53ca7ab8f` | `cfd057cc12af6c7291f6424569c686a878e9d2334086a83cbdfd105b58770b19` |
| `autonomous_allhad/gnn_lowdm/README.md` | `5fb925056e3e526d6c0d6db5f8792078b0b4f1f7f75b7f2cb7113c836716f712` | `e18cfc7eb1a53e8ca81853772d5f7868e8d926e425f2702f3828f0884231c82a` |
| `autonomous_allhad/gnn_lowdm/_implementation/plot_hyperparameter_losses.py` | `a48eb865ee44d2bdfa0aebe92f756683a4a9e5191e06222372b8d3bb8b208b00` | `70413cd8600a442f6327cf45c0871edb0c0d2b3895e07e753ad2f2f4e276ef02` |
| `autonomous_allhad/gnn_lowdm/_implementation/supplementary_plots.py` | `404a6be1a5849f03e097fdb012d8361321e62331659ee3e8a16ccd840afde2eb` | `8de78930409a233c97e3089503b61f68235a0cc8db3032d391b3b5c6504780ea` |
| `autonomous_allhad/gnn_lowdm/plotting.py` | `4c87668644f2cb992d6d1a5987076cc0455cf4aa3668f47667e588131a161c7a` | `4e951bf5f89f16e115c4a5f55f69521143040241dff22ba200296a95a4ae1404` |
| `autonomous_allhad/gnn_lowdm/requirements.txt` | `581fb3e1bb4f7d79d1e4d9edaadd44ed29d6e08319e8bc4348948d0d9ed8aabc` | `70495c798fce18b08d2bd82e31f005138ec7cd1d5ac890f464f07a542921e036` |
| `autonomous_allhad/tests/test_2025_data_policy.py` | `2239bf9355dffa7175ba36c24809f6e4757d24c5a933d8a18ddb80c64af20c95` | `2581b994f4b8530fe52d43058367a76fce86763c50b9bbbf5642c9fa2a3c9fc4` |
| `autonomous_allhad/tests/test_canonical_card_projection.py` | `91c8ef3ba75754c7b00c38a4678a88c6a00ccc404c5e438d747c5d5988dd6cde` | `030749f00322a9d361b67bd0e3109889f3f11e467acefbc4eddb0df6c07fd6b9` |
| `autonomous_allhad/tests/test_trigger_efficiency.py` | `b4aa1f33c622da67c43caf02dfc3c91a2a0a483d0f21dd0ebb0c7e46b3631083` | `8b694d56ba2521bf5afd64d313893aea4a50481ac97b3b4b5faa2b69f3431b9b` |
| `autonomous_allhad/workflow/flat_ntuple_analysis_pipeline.md` | `fdad06a6579e29e56f7193200d85045a45aa0c7b23544f8d08e3a3e3ed9d7230` | `e67d9125062a989d6f10dca7e4a4c00a92766302924e1b9798a3d1704bcaed0b` |
| `autonomous_allhad/workflow/measure_trigger.py` | `6cb5cbcd16f942debb10a9e7bc81d4d7290ef345e45064c7b61bcc1371d88056` | `fe41550aea38b2bc8a5091903a8771822a0e7ca051c12cd40a5b3e44de4f0b56` |
| `autonomous_allhad/workflow/run_batch.sh` | `d366a51d83545d2249b4599ab30c10781b5abdf9b2a4c3b3f91f8c4fb839a1a5` | `4c91cdcbe19e872ca1ff5c09e9662b3cee38f524d6fb57ca8a7a68a8e0eede0c` |
