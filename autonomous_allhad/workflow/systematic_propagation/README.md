# Separate shape-systematic propagation

This directory is independent of nominal production. `run.py` orchestrates the
existing intermediate, TROTA, Top/W, main histogram and frozen GNN executables;
it does not duplicate their physics selections. Outputs and campaign state are
split into `2024/` and `2025/`. Canonical inputs and normalization are read-only.

The first end-to-end pilot contains nominal and unclustered-MET Up/Down for the
same complete MC input shard in each year. It starts before the nominal skim so
events migrating into the selection are retained. Each shape is histogrammed
with the central weight evaluated on its shifted objects. Main histograms are
nominal-weight-only; the unchanged GNN executable also exports its usual weight
variations, which must not be interpreted as additional kinematic nuisances.

Use the existing miniconda py38 executable for `run.py prepare --repo REPO`.
Submit `pilot.sub` to the EOS schedd after input/hash checks. The user approved
`workday` (28,800 seconds) and the existing TROTA-only LCG_104/TensorFlow child
environment. All other Python execution uses the existing `py38.tgz`.

An individual job is complete only after all stages and output checksums pass.
Full production requires nominal-reference closure first. Pilot ROOTs remain on
EOS for that comparison; no ROOT is copied to the laptop. No result is installed
into the canonical histograms by this runner.

JES/JER require propagation to the raw-NanoAOD-based TROTA model inputs relative
to their nominal reference, without changing the nominal model convention.
That extension, EGM variations, MUO/TAU decomposition, and the JMS/JMR prescription
are subsequent stages, not claimed as implemented by the MET pilot.
