# AGENTS.md

## Project mission

This repository contains a CMS Run-3 all-hadronic stop search.

The principal reference files are:

* `AN2019_016_v9.pdf`
* `stop_processor_v4.py`
* `ids.py`
* `corrections.py`

Treat these files as physics references and validation baselines, not as an implementation architecture that must be preserved.

The objective is to design, implement, validate, execute, and publish an improved end-to-end Run-3 all-hadronic stop analysis.

Do not merely refactor the existing processor.

## Reference hierarchy

Use the four reference files as follows.

### `AN2019_016_v9.pdf`

Use this document to understand:

* the Run-2 all-hadronic stop-search strategy;
* physics motivation;
* signal topology;
* background estimation;
* signal and control region logic;
* event categorization;
* search-bin construction;
* discriminating variables;
* statistical treatment;
* validation philosophy.

Do not copy Run-2 definitions blindly into Run 3. Identify which elements remain physically useful and which require redesign for Run-3 data, reconstruction, triggers, taggers, corrections, and available luminosity.

### `stop_processor_v4.py`

Use this file to recover the current Run-3 implementation, including:

* supported years;
* datasets and process groups;
* triggers;
* MET filters;
* object construction;
* object cleaning;
* recoil definitions;
* region selections;
* weights;
* systematic variations;
* histogram definitions;
* output conventions.

### `ids.py`

Use this file as the current implementation reference for:

* electron identification;
* muon identification;
* photon identification;
* AK4 jet identification;
* AK8/fatjet identification;
* veto, loose, medium, and tight working points;
* object cleaning and overlap behavior.

### `corrections.py`

Use this file as the current implementation reference for:

* pileup weights;
* MET XY corrections;
* JEC;
* JER;
* JES uncertainty;
* electron scale factors;
* muon scale factors;
* photon scale factors;
* trigger scale factors;
* b-tagging scale factors;
* b-tagging efficiency treatment;
* top-pT reweighting;
* year-dependent correction availability;
* missing implementations and placeholders.

## Core priorities

Prioritize, in order:

1. physics correctness;
2. reproducibility;
3. expected exclusion sensitivity;
4. control-region constraining power;
5. statistical robustness;
6. systematic robustness;
7. failure recovery;
8. runtime;
9. memory usage;
10. EOS and CVMFS I/O efficiency;
11. HTCondor efficiency;
12. maintainability;
13. auditability;
14. clarity of the final web report.

Do not select a faster implementation if it fails physics validation.

Do not select a more sensitive categorization if the improvement depends on unstable low-statistics bins, poor control-region closure, fragile transfer factors, unjustified nuisance assumptions, or unexplained disagreement with the baseline.

## Physics baseline versus implementation

The reference files define the starting physics intent and current behavior.

They do not define the architecture that must be retained.

You may redesign:

* input discovery;
* branch reading;
* skim format;
* feature storage;
* event processing;
* chunking;
* correction evaluation;
* caching;
* multiprocessing;
* HTCondor job structure;
* retry logic;
* reduction;
* merging;
* histogram representation;
* plotting;
* search-bin evaluation;
* datacard production;
* expected-limit execution;
* report generation;
* GitHub Pages publication.

Do not silently modify physics definitions.

Any change to selections, regions, categories, search bins, background estimation, nuisance correlations, or normalization must be labeled as a physics proposal and compared quantitatively with the baseline.

## Top-tagging-independent categorization

The primary event-categorization optimization must not use top-tagging discriminator values, top-tagging working points, or top-tagging pass/fail decisions.

AK8 jet kinematics may be used, including:

* multiplicity;
* transverse momentum;
* mass;
* angular relations;
* recoil balance.

Top-tag scores must not be used for the primary categorization.

Develop and compare categories using quantities such as:

* jet multiplicity;
* b-jet multiplicity;
* HT;
* MET;
* recoil transverse momentum;
* leading-jet kinematics;
* minimum delta-phi;
* individual jet–MET angular variables;
* resolved hadronic-W reconstruction;
* resolved hadronic-top kinematics;
* invariant masses;
* transverse masses involving b jets;
* MT2-like quantities where justified;
* ISR-sensitive variables;
* event-shape quantities;
* AK8 kinematics without tagger scores.

At minimum, evaluate:

1. a minimal Njet–Nb–MET categorization;
2. a resolved-kinematics categorization;
3. an ISR-sensitive categorization;
4. an AK8-kinematics categorization without tag scores;
5. a statistically optimized hybrid categorization.

## Required architecture candidates

Develop and benchmark at least three architectures.

### Architecture A: optimized faithful rewrite

Reproduce the current Run-3 physics definitions as faithfully as possible while redesigning the software for better performance and robustness.

### Architecture B: feature-table analysis

Produce a reusable event-level feature representation that supports rapid changes to regions, categories, bins, and statistical models without repeatedly processing full NanoAOD inputs.

### Architecture C: redesigned categorization and binning

Use the validated event representation to optimize top-tagging-independent categories, search-bin boundaries, control-region mappings, and the statistical model.

Additional candidates may be evaluated when justified.

## Validation requirements

Validate candidate implementations against the current Run-3 baseline.

### File level

Compare:

* discovered files;
* accepted files;
* skipped files;
* duplicate files;
* expected and retained event counts;
* sum-of-weight bookkeeping.

### Event level

Compare using run, luminosity block, and event number.

Check:

* event acceptance;
* selected objects;
* object indices;
* region assignment;
* category assignment;
* search-bin assignment;
* nominal event weight;
* systematic weights.

### Object level

Check:

* object multiplicities;
* kinematics;
* object IDs;
* cleaning decisions;
* corrected quantities;
* working-point decisions.

### Region level

Check all current high-DeltaM regions, including:

* preselection;
* lost-lepton control region;
* QCD control region;
* photon control region;
* dielectron control region;
* dimuon control region;
* signal region;
* any existing validation regions.

### Yield and shape level

Compare every relevant process, region, category, search bin, and histogram.

### Systematic level

Compare all implemented nominal and shifted quantities, including:

* JES;
* JER;
* MET;
* pileup;
* b tagging;
* electron corrections;
* muon corrections;
* photon corrections;
* top-pT;
* MC statistics;
* normalization uncertainties.

### Statistical level

Compare:

* datacard rates;
* nuisance definitions;
* nuisance correlations;
* control-region constraints;
* expected limits;
* fit convergence;
* nuisance impacts;
* low-statistical-bin behavior.

## Corrupted ROOT file policy

A corrupted, unreadable, truncated, malformed, or structurally invalid ROOT file must not stop the complete campaign.

When a file cannot be read:

1. distinguish a transient storage failure from reproducible corruption;
2. retry from an alternate endpoint when appropriate;
3. if the failure is reproducible, skip the file;
4. record it in a bad-file manifest;
5. exclude it from subsequent submissions;
6. continue processing valid files.

Maintain:

* `autonomous_allhad/workflow/bad_files.json`
* `autonomous_allhad/workflow/bad_files.txt`
* `autonomous_allhad/workflow/file_validation_summary.json`

Each record must include:

* dataset;
* file path;
* failure stage;
* exception type;
* concise error;
* first and last failure times;
* whether alternate access was attempted;
* whether the file was permanently skipped.

Do not silently discard files.

Quantify the lost fraction by:

* file count;
* event count where available;
* generator sum of weights;
* data run and luminosity coverage;
* affected signal mass points.

For MC, ensure that normalization bookkeeping is consistent with the valid retained files.

For data, do not claim complete luminosity coverage when skipped files remove unique luminosity sections.

A dataset must be marked incomplete when the skipped fraction exceeds configurable thresholds.

## Workflow behavior

The workflow must be:

* resumable;
* idempotent;
* deterministic;
* configuration driven;
* auditable.

Completed valid work must not be repeated unnecessarily.

Queue disappearance is not evidence of success.

A job is successful only when:

* it exits successfully;
* the expected output exists;
* the output passes integrity checks;
* required metadata exists;
* campaign state records the output as valid.

Maintain machine-readable state, history, manifests, and summaries inside each
active campaign directory. Do not use a shared monolithic workflow state.

## GitHub Pages report

Create a static analysis website that can be hosted with GitHub Pages.

The website must be generated from machine-readable outputs, not manually invented values.

Use a structure such as:

```text
docs/
├── index.html
├── assets/
├── plots/
├── data/
├── benchmarks/
├── validation/
└── reports/
```

The website must include:

* project overview;
* current pipeline state;
* input and bad-file summary;
* baseline analysis summary;
* architecture candidates;
* performance benchmarks;
* object-level validation;
* event-level validation;
* region and yield comparisons;
* systematic comparisons;
* top-tagging-independent categorization study;
* search-bin study;
* control-region closure;
* transfer-factor studies;
* combined control-region plots;
* expected-limit comparisons;
* final architecture decision;
* unresolved issues;
* reproducibility information;
* git commit and configuration hashes.

The site must clearly distinguish:

* completed results;
* preliminary results;
* failed stages;
* missing samples;
* skipped files;
* proposed physics changes;
* adopted physics changes.

Do not publish credentials, proxies, private tokens, internal secrets, or sensitive paths.

Create a GitHub Actions workflow for GitHub Pages when appropriate:

```text
.github/workflows/pages.yml
```

Do not claim the site is publicly deployed unless the push and GitHub Pages deployment actually succeed.

When repository credentials or GitHub permissions are unavailable, generate the complete `docs/` site and provide the exact remaining publication step.

## Safety and destructive actions

Do not perform destructive operations without necessity.

Do not:

* delete valid production outputs;
* remove large EOS directories;
* rewrite verified results;
* force-push protected branches;
* expose credentials;
* fabricate successful job status;
* fabricate benchmark or limit results.

When a destructive action is necessary, document why and preserve recoverable backups where practical.

## Completion criteria

The project is not complete when code has merely been written.

It is complete only when:

* the four reference files have been fully inspected;
* the baseline has been reverse-engineered;
* machine-readable specifications exist;
* the baseline has been benchmarked;
* at least three architectures have been implemented or meaningfully prototyped;
* the architectures have been tested on identical representative inputs;
* corrupted ROOT files have been skipped and documented;
* event-, object-, region-, yield-, shape-, systematic-, and expected-limit comparisons exist;
* top-tagging-independent categories have been evaluated;
* a final architecture has been selected quantitatively;
* the selected implementation has run end-to-end on the representative subset;
* the static website has been generated;
* GitHub Pages publication has been attempted when credentials permit;
* all unresolved issues are documented.

At completion, report:

* architecture candidates;
* selected architecture;
* quantitative selection reasons;
* physics differences from the baseline;
* categories tested;
* performance results;
* skipped corrupted files;
* normalization effects;
* validation discrepancies;
* control-region results;
* expected-limit results;
* generated website location;
* GitHub Pages status;
* modified and created files;
* unresolved issues.
