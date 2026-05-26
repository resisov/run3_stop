## Changes: 2026-02-09 (Update 2025 JECs and JVM)

Merge Request: [!2](https://gitlab.cern.ch/cms-analysis-corrections/JME/Run3-25Prompt-Winter25-NanoAODv15/-/merge_requests/2)

In this MR we update the 2025 `json` files for AK4PFPuppi and AK8PFPuppi jets, as well as the jet veto map `json` file. The new JECs cover the full range of 2025 data, namely eras 2025 C-G.

The new JEC tag version is `Winter25Prompt25_V3`, while for the JVM it is `Winter25Prompt25_RunCDEFG_V1`

In more detail:

- The MC-truth JECs (`L1FastJet`, `L2Relative`, `L3Absolute`) remain unchanged compared to the previous version.
- The residual JECs (`L2L3Residual`) are updated for all eras and now cover the entire range of 2025 data. They were presented in [JME, 9 Dec 2025](https://indico.cern.ch/event/1620043/#4-2025-jecs) and [JERC, 3 Feb 2026](https://indico.cern.ch/event/1642716/#51-closure-for-2025-v3m-jecs-a).
- The JES uncertainties and JER remain unchanged. For now, we use the `Summer23BPixPrompt23_RunD_JRV1_MC` JER tag, until dedicated JER files are derived for 2025.
- The jet veto map (JVM) is also updated, and was presented in [JME, 9 Dec 2025](https://indico.cern.ch/event/1620043/#4-2025-jecs). As usual, the `jetvetomap` histogram is recommended for analyzers.

The corresponding PR in JECDatabase is: [#219](https://github.com/cms-jet/JECDatabase/pull/219)
