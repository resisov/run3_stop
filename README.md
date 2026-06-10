# Run-3 All-Hadronic Stop Analysis

This repository contains the Coffea-based workflow for the CMS Run-3 all-hadronic stop-pair search.

It includes:

* object definitions and event selections
* corrections and scale factors
* nominal and shifted processor production
* local Coffea execution
* HTCondor submission
* histogram reduction, merging, and scaling
* lightweight workflow automation

The automation layer wraps the existing analysis scripts without changing the underlying physics implementation.

---

## Repository Structure

```text
decaf/
├── analysis/
│   ├── processors/
│   │   └── stop_processor_v4.py
│   ├── metadata/
│   │   └── KNU_2024_v4.json.gz
│   ├── datasets/
│   │   ├── datasets_2024.txt
│   │   └── datasets_2024_onlymc.txt
│   ├── data/
│   │   ├── corrections.coffea
│   │   ├── ids.coffea
│   │   └── common.coffea
│   ├── hists/
│   ├── run.py
│   ├── reduce.py
│   ├── merge.py
│   └── scale.py
│
├── automation/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── paths.py
│   └── validation.py
│
├── condor/
│   ├── run_condor_2024.sh
│   ├── run_condor_2024.sub
│   ├── run_condor_jesTotal.sub
│   ├── run_condor_metUnclustered.sub
│   ├── run_condor_jer.sub
│   ├── analysis.tgz
│   └── py38.tgz
│
├── condor_log/
│
├── configs/
│   └── stop_2024.yaml
│
├── .gitignore
├── README.md
└── setup_condor.sh
```

---

## Working Directory

The main EOS working copy is:

```text
/eos/user/t/taiwoo/run3_stop/decaf
```

Move to the repository before running the workflow:

```bash
cd /eos/user/t/taiwoo/run3_stop/decaf
```

---

## Software Environment

The current analysis environment uses:

```text
Python 3.8.20
Coffea 0.7.22
```

The configured Python executable is:

```text
/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python
```

For interactive use:

```bash
source /eos/user/t/taiwoo/miniconda3/etc/profile.d/conda.sh
conda activate py38
```

The automation commands below use the Python executable configured in:

```text
configs/stop_2024.yaml
```

---

## Main Configuration

The 2024 workflow is configured in:

```text
configs/stop_2024.yaml
```

It defines:

* analysis year
* metadata name
* processor names
* external shifts
* dataset-list files
* local output paths
* Condor input files
* Condor log directory
* Python executable

The processor and correction modules remain the source of truth for the physics implementation.

---

## Supported External Shifts

The current 2024 workflow supports:

```text
nominal
jesTotalUp
jesTotalDown
metUnclusteredUp
metUnclusteredDown
jerUp
jerDown
```

The nominal workflow runs both data and MC.

The external shifted workflows use MC-only dataset lists.

---

# 1. Validate the Setup

Run validation before processing or submitting jobs:

```bash
/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python \
  -m automation.cli validate \
  --config configs/stop_2024.yaml
```

Validation checks include:

* processor source
* metadata JSON
* correction pickle files
* dataset-list files
* local workflow scripts
* processor output paths
* histogram output paths
* Condor tarballs
* proxy
* worker script
* Condor log directory

A successful validation prints `[OK]` messages.

---

# 2. Compile a Processor

Compile the nominal processor:

```bash
/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python \
  -m automation.cli compile \
  --config configs/stop_2024.yaml \
  --shift nominal
```

Compile a shifted processor:

```bash
/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python \
  -m automation.cli compile \
  --config configs/stop_2024.yaml \
  --shift jerUp
```

Processor files are written under:

```text
analysis/data/
```

Examples:

```text
analysis/data/stop_2024_nominal.processor
analysis/data/stop_2024_jerUp.processor
analysis/data/stop_2024_jerDown.processor
```

The automation `run` and `submit` commands always recompile the selected processor before execution.

---

# 3. Local Small Test

Always begin with a dry-run:

```bash
/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python \
  -m automation.cli run \
  --config configs/stop_2024.yaml \
  --shift nominal \
  --dataset-prefix TTto2L2Nu_ \
  --max-datasets 1 \
  --max-files 1 \
  --dry-run
```

This resolves:

* one metadata key
* at most one ROOT file for that key
* the processor compile command
* the local run command
* the expected histogram directory

Run the actual test by removing `--dry-run`:

```bash
/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python \
  -m automation.cli run \
  --config configs/stop_2024.yaml \
  --shift nominal \
  --dataset-prefix TTto2L2Nu_ \
  --max-datasets 1 \
  --max-files 1
```

---

## Dataset Selection Modes

The automation CLI supports three mutually exclusive dataset-selection modes.

### Legacy substring mode

```bash
--dataset TT
```

This selects every metadata key containing the text `TT`.

It may also select samples such as:

```text
TTW
TTZ
TTTT
```

Use this mode carefully.

### Exact metadata key

```bash
--dataset-key '<full metadata key>'
```

This selects exactly one metadata key.

### Prefix mode

```bash
--dataset-prefix TTto2L2Nu_
```

This selects metadata keys beginning with the supplied prefix.

For small tests, combine it with:

```bash
--max-datasets 1
--max-files 1
```

The meaning of:

```bash
--max-files 1
```

is:

```text
at most one input file per resolved metadata key
```

It is not a global input-file limit.

---

# 4. Local Postprocessing

Inspect the postprocessing commands first:

```bash
/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python \
  -m automation.cli postprocess \
  --config configs/stop_2024.yaml \
  --shift nominal \
  --dry-run
```

The postprocessing sequence is:

```text
reduce.py
→ merge.py
→ scale.py
```

Run it:

```bash
/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python \
  -m automation.cli postprocess \
  --config configs/stop_2024.yaml \
  --shift nominal
```

The automation calls the three stages explicitly and does not use the legacy `postprocess.py`.

---

# 5. Prepare Condor Inputs

The Condor workflow uses:

```text
condor/analysis.tgz
condor/py38.tgz
analysis/proxy/x509up_u147757
```

The setup helper is:

```text
setup_condor.sh
```

Run it from the repository root when the analysis tarball, Python environment tarball, or proxy must be refreshed:

```bash
bash setup_condor.sh
```

Before submission, verify:

```bash
ls -lh \
  condor/analysis.tgz \
  condor/py38.tgz \
  analysis/proxy/x509up_u147757
```

---

# 6. Condor Dry-Run

Inspect a one-job submission before sending it:

```bash
/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python \
  -m automation.cli submit \
  --config configs/stop_2024.yaml \
  --shift jerUp \
  --max-jobs 1 \
  --dry-run
```

The dry-run prints:

* selected dataset-list entries
* total available job count
* selected job count
* processor compile command
* generated submit description
* `condor_submit` command
* output remap path
* shared Condor log path

Dry-run does not:

* compile the processor
* submit Condor jobs
* create generated submit files
* create output files
* create log files

---

# 7. Submit One Test Job

Submit one real `jerUp` job:

```bash
/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python \
  -m automation.cli submit \
  --config configs/stop_2024.yaml \
  --shift jerUp \
  --max-jobs 1
```

The automation performs:

```text
validation
→ processor compilation
→ generated submit creation
→ condor_submit
```

Record the cluster ID printed by Condor.

Check its status:

```bash
condor_q <CLUSTER_ID>
```

Inspect the cluster log:

```bash
tail -f \
  /eos/user/t/taiwoo/run3_stop/decaf/condor_log/job.<CLUSTER_ID>.log
```

Remove the test cluster if necessary:

```bash
condor_rm <CLUSTER_ID>
```

---

# 8. Condor Job Unit

Each Condor job corresponds to:

```text
one shift × one dataset-list entry
```

A dataset-list entry is a metadata dataset-key token.

It is not necessarily one ROOT file.

The worker receives:

```bash
DATASET="$1"
SHIFT="$2"
PROCESSOR="stop_2024_${SHIFT}"
```

It then runs:

```bash
python run.py \
  -p "stop_2024_${SHIFT}" \
  -m KNU_2024_v4 \
  -w 1 \
  -d "${DATASET}"
```

The shift is therefore propagated through the processor name.

Examples:

```text
stop_2024_nominal
stop_2024_jesTotalUp
stop_2024_jesTotalDown
stop_2024_metUnclusteredUp
stop_2024_metUnclusteredDown
stop_2024_jerUp
stop_2024_jerDown
```

---

# 9. Condor Output

Each worker creates:

```text
out.futures
```

Condor remaps it to:

```text
analysis/hists/stop_2024_<shift>/<dataset>.futures
```

Examples:

```text
analysis/hists/stop_2024_nominal/<dataset>.futures
analysis/hists/stop_2024_jerUp/<dataset>.futures
analysis/hists/stop_2024_jerDown/<dataset>.futures
```

---

# 10. Condor Logs

Condor logs are stored under the top-level directory:

```text
condor_log/
```

All jobs in one submitted cluster share one file:

```text
condor_log/job.<ClusterId>.log
```

The submit configuration uses:

```condor
initialdir = /eos/user/t/taiwoo/run3_stop/decaf/condor_log

output = job.$(ClusterId).log
error  = job.$(ClusterId).log
log    = job.$(ClusterId).log
```

Therefore:

* stdout
* stderr
* Condor event information
* output from all ProcIds

may be interleaved in the same cluster-level log file.

This is intentional to avoid producing thousands of separate log files.

---

# 11. Full Condor Submission

After validating one job, test a small group:

```bash
/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python \
  -m automation.cli submit \
  --config configs/stop_2024.yaml \
  --shift jerUp \
  --max-jobs 10
```

After confirming all test jobs finish successfully, submit the full shift:

```bash
/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python \
  -m automation.cli submit \
  --config configs/stop_2024.yaml \
  --shift jerUp
```

Recommended sequence:

```text
1 job
→ 10 jobs
→ full shift
```

Repeat for:

```text
jerDown
jesTotalUp
jesTotalDown
metUnclusteredUp
metUnclusteredDown
nominal
```

---

# 12. Manual Condor Submit Files

The top-level `condor/` directory contains manually maintained submit files:

```text
condor/run_condor_2024.sub
condor/run_condor_jesTotal.sub
condor/run_condor_metUnclustered.sub
condor/run_condor_jer.sub
```

All shifts reuse the common worker:

```text
condor/run_condor_2024.sh
```

The JER submit file supports:

```text
jerUp
jerDown
```

The automation CLI is preferred over direct manual submission because it provides:

* validation
* processor recompilation
* dry-run
* limited job submission
* generated shift-specific submit files
* explicit subprocess return-code handling

---

# 13. Git Ignore Policy

Analysis outputs and large runtime files must not be committed.

Recommended `.gitignore` entries:

```gitignore
# Coffea outputs
*.futures
*.reduced
*.merged
*.scaled
*.processor

# Large runtime files
*.tgz
*.tar
*.tar.gz
*.root

# Condor logs and generated files
condor_log/
analysis/condor/generated/
```

Because the actual generated submit directory may be under the top-level `condor/`, also ignore it if used:

```gitignore
condor/generated/
```

Before committing:

```bash
git status
git diff --cached --stat
```

Check for unwanted large or generated files:

```bash
git diff --cached --name-only \
  | grep -E '\.(futures|merged|scaled|processor|tgz|root|log)$'
```

---

# 14. Recommended Git Workflow

Avoid using `git add .` without reviewing the result.

Add only the intended source and configuration files:

```bash
git add \
  automation/ \
  configs/stop_2024.yaml \
  analysis/run.py \
  condor/*.sub \
  condor/*.sh \
  setup_condor.sh \
  README.md \
  .gitignore
```

Inspect the staged changes:

```bash
git status
git diff --cached --stat
git diff --cached
```

Commit:

```bash
git commit -m "Add local and Condor analysis automation"
```

Push:

```bash
git push
```

---

# 15. Current Automation Scope

Implemented:

* configuration validation
* processor compilation
* local execution
* local dry-run
* safe metadata-key selection
* limited local tests
* postprocessing orchestration
* Condor submission
* Condor dry-run
* limited Condor submission with `--max-jobs`

Not yet fully automated:

* Condor status summaries
* expected-output completeness checks
* missing-output detection
* failed-job classification
* failed-job resubmission
* automatic postprocessing after production
* plotting
* ROOT template production
* datacard generation
* Combine execution

---

# 16. Recommended Production Workflow

```text
1. Validate the configuration
2. Run one local file
3. Test nominal and shifted processors locally
4. Run one Condor job
5. Run ten Condor jobs
6. Submit a full shift
7. Compare expected and produced outputs
8. Resubmit missing or failed entries
9. Run postprocessing
10. Produce plots and templates
```

---

# Physics Safety

When modifying the analysis:

* do not silently change event selections
* do not silently change histogram names or binning
* do not silently change correction tags or values
* do not treat JES or JER as event weights
* compare data and MC selection logic explicitly
* validate nominal and shifted outputs
* run small tests before full production

The existing processor and correction modules remain the authoritative physics implementation.
