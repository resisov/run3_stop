# Histogram-only DY normalization measurement

This package measures the Run-2-style \(R_Z(N_b)\) and \(R_T(N_b)\) factors
without reopening decorated feature ROOT files or NanoAOD.  The sole input is
the compact `*_background_estimation.json` emitted automatically by the
nominal histogram merger.

Both High- and Low-\(\Delta m\) use exactly two categories:

- `Nb1`: \(N_b=1\);
- `Nb2plus`: \(N_b\geq2\).

No Low-\(\Delta m\) \(N_j\), \(p_T^b\), ISR, or legacy search-bin subdivision
is used.  The nominal on-Z window is \(71<m_{\ell\ell}<111\) GeV.  The off-Z
sideband is \(50<m_{\ell\ell}<71\) GeV or \(m_{\ell\ell}>111\) GeV.

For each channel and category the likelihood is based on

\[
N_{\mathrm{data}}^w = R_Z N_{Z\text{-like}}^w
                     + R_T N_{\mathrm{other}}^w,
\qquad w\in\{\mathrm{on},\mathrm{off}\}.
\]

The histogram producer records the disjoint on/off counts, `sumw2`, and the
validation \(m_{\ell\ell}\) distributions while normalized event weights are
already in memory.  All later operations are JSON-only.

## Execution order

~~~bash
DY_PYTHON=/eos/user/t/taiwoo/miniconda3/envs/py38/bin/python
DY_HIST=<merged_hists_stem>_background_estimation.json
DY_WORK=<output-directory>

$DY_PYTHON -m autonomous_allhad.dy_estimation build-measurement \
  --hist-input "$DY_HIST" \
  --campaign-year 2024 \
  --output "$DY_WORK/dy_measurement.json"

$DY_PYTHON -m autonomous_allhad.dy_estimation report \
  --measurement "$DY_WORK/dy_measurement.json" \
  --selection both \
  --campaign-year 2024 \
  --output-dir "$DY_WORK/report"
~~~

Use `--campaign-year 2025` for 2025.  Plotting remains in `report.py`; no new
plotting implementation is introduced.

## Retired workflow

Feature-ROOT scans, per-channel feature campaigns, sparse NanoAOD recovery,
and their merge/validation commands are retired.  They are not exposed by the
package CLI and must not be used for a new background-estimation recalculation.
