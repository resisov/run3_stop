# Manual Legacy Validation Package

Legacy stop_processor_v4.py validation is intentionally external/manual and is not required for the autonomous feature-extraction gate.
No independent agreement with stop_processor_v4.py is claimed by autonomous_allhad.

## ROOT Files

The package lists 14 selected ROOT files in `input_files_for_legacy.json` and `input_files_for_legacy.txt`.
Each JSON record includes the dataset key, process, physical ROOT path, entry count, read status, processing status, and processed entry ranges.

## Event Chunks

Feature extraction uses deterministic stratified chunks with chunk size 2000. Files with fewer entries than the configured stratified coverage are read fully; larger files are sampled in evenly spaced chunks recorded in the manifest.
No random seed or top-tagging discriminator is used for this gate.

## Feature Table Construction

The feature table is built from real NanoAOD ROOT branches using uproot/awkward. It records run/luminosity/event keys, MET/recoil-related quantities, AK4/AK8 kinematics, lepton/photon/tau/track counts, region booleans, nominal weights, and diagnostic provenance fields.
AK4 jet ID uses correctionlib `AK4PUPPI_TightLeptonVeto` where the NanoAOD composition branches are present; fallbacks are labeled in `jet_id_source`.

## Compare These Feature-Side Values Manually

Use `feature_yields_for_manual_comparison.json` for per-process, per-region unweighted and weighted yields.
Use `feature_cutflows_for_manual_comparison.json` for cumulative cutflows and first-zero diagnostics.
Use `feature_histograms_for_manual_comparison.json` for feature-side MET, HT, Njet, Nb, and minimum-dphi histogram counts.

## Feature-Side Only Artifacts

The yields, cutflows, histograms, feature table, trigger audit, object-ID audit, and website are produced by autonomous_allhad only. They are not an independent legacy-processor comparison.
No physics change should be treated as adopted until manual legacy validation is supplied by the analyst.
