# Fast All-Hadronic Stop Analysis Architecture

This package is an additive prototype path. It does not replace the existing Coffea workflow and does not write under `analysis/hists`.

Adopted from the reference workflow:

- vectorized NanoAOD reading with uproot/Awkward;
- compact reusable event data before histograms/cards;
- explicit campaign bookkeeping;
- static HTML status reporting;
- expected-only prototype maturity labels.

Changed for this all-hadronic analysis:

- the current local all-hadronic processor is the physics oracle;
- the fixed runtime is `/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python`;
- no new environment is created and no packages are installed;
- compact format is chosen from installed modules: Parquet if PyArrow is available, otherwise flat ROOT with uproot;
- every project path is centrally validated to remain under approved EOS roots;
- datacards are gated behind a written statistical-model decision, not auto-invented.

Initial choices:

- runtime: existing EOS py38 environment at `/eos/user/t/taiwoo/miniconda3/envs/py38`;
- compact storage: runtime-selected; PyArrow is available in the current fixed env, so Parquet can be benchmarked;
- runtime outputs: `/eos/user/t/taiwoo/run3_stop/decaf/fast_outputs`;
- legacy read-only reference: `/eos/user/t/taiwoo/decaf/analysis/hists/stop_2024_nominal.scaled`;
- first benchmark: one data file, one TT/W file, one T2tt file from an EOS-resident manifest, only after the legacy reference validates;
- job target: 5-15 minutes per chunk after bounded benchmark evidence.

Current status:

- EOS policy gate implemented;
- fixed-environment inventory implemented;
- CLI scaffold implemented;
- compact schema draft implemented;
- mapping document started from the current processor;
- legacy reference validation implemented and currently blocks benchmark/comparison because `recoilpt` contains non-finite values for QCD and W+jets;
- no full campaign submitted;
- no observed SR use;
- no observed limit run.
