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
Year-specific object payloads are unpacked from the existing production bundle
into job scratch; they do not rely on loose payload files in the live checkout.

An individual job is complete only after all stages and output checksums pass.
Full production requires nominal-reference closure first. Pilot ROOTs remain on
EOS for that comparison; no ROOT is copied to the laptop. No result is installed
into the canonical histograms by this runner.

`validate` checks successful Condor termination, input coverage, product hashes,
Events/TROTA/TopWTruth integrity and identities, and every nominal histogram bin.
It records the result in the existing `campaign_state.json`. The reference
directories contain the original canonical ROOT evaluated with the same frozen
main/GNN commands, normalization and 10 GeV veto. The pilot's `_condor_stdout`
records those commands; only the ROOT/sidecar input and output paths differ.

On lxplus:

```bash
SHAPE_ROOT=/eos/user/t/taiwoo/run3_stop/decaf/autonomous_allhad/workflow/systematic_propagation
SHAPE_PY=/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python
"$SHAPE_PY" "$SHAPE_ROOT/run.py" validate \
  --config "$SHAPE_ROOT/2024/campaign.json" \
  --output "$SHAPE_ROOT/2024/pilot" \
  --reference "$SHAPE_ROOT/2024/nominal_reference" --cluster 1115064
"$SHAPE_PY" "$SHAPE_ROOT/run.py" validate \
  --config "$SHAPE_ROOT/2025/pilot_retry1.json" \
  --output "$SHAPE_ROOT/2025/pilot_retry1" \
  --reference "$SHAPE_ROOT/2025/nominal_reference" --cluster 1115066
```

The submitted configurations retain the hashes of their historical worker
revisions. Do not regenerate those configurations or overwrite completed pilot
products when updating the runner. This validation is for one ST shard, not a
claim that the complete MC campaign or all kinematic nuisances are finished.

JES/JER require propagation to the raw-NanoAOD-based TROTA model inputs relative
to their nominal reference, without changing the nominal model convention.
That extension, EGM variations, MUO/TAU decomposition, and the JMS/JMR prescription
are subsequent stages, not claimed as implemented by the MET pilot.
