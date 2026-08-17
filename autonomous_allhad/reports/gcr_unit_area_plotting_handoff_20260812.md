# GCR unit-area plotting handoff

Date: 2026-08-12  
Scope: plotting-only handoff for the 2024 photon control region (GCR)  
Status: specification only; no plotting code or published plots were modified

## 1. Requested outcome

Redraw the existing GCR plots using the Run-2 presentation convention:

- data and the total background MC are independently normalized to unit area;
- the relative composition of the stacked MC processes is preserved;
- the lower panel compares the normalized data shape to the normalized total-MC shape;
- the change applies only to plots whose full physics scope is GCR;
- reuse the existing nominal plotter and its current inputs;
- do not create a separate plotting implementation;
- do not modify nominal histogram intermediates.

This is a rendering change, not a physics-weight correction. No event weight, cross section,
normalization payload, fit result, histogram bin content, or Combine input is to be rewritten.

## 2. Existing implementation that must be reused

The current public nominal page identifies the following as its main plotter:

```text
autonomous_allhad/workflow/plot_control_search_bins_style.py
```

Relevant call structure:

```text
draw_highdm_distribution_report
  -> highdm_variable_record
  -> draw_flat_blocks(..., reference_style=True)

draw_flat_report
  -> flat_hist_record / flat_search_record / lowdm_variable_record
  -> draw_flat_blocks(...)
```

The common rendering function is `draw_flat_blocks`. Add the GCR unit-area policy to the
existing record/render path rather than copying this function or writing a sidecar plotter.

Do not use `analysis/plotting/distribution_draw_v5.py` for this page. It is not the main
plotter recorded by the current page manifest.

Current page:

```text
https://resisov.github.io/run3_stop/nominal_plots_2024_fullselection_v5_dyexclusive_t2models_freebkg_20260728/
```

Current local page snapshot:

```text
docs/nominal_plots_2024_fullselection_v5_dyexclusive_t2models_freebkg_20260728/
```

The page manifest records:

- source histogram payload: `hists.json`;
- source histogram SHA-256: `889bcd2411276bb9313664886c1d87080dd4e8ca7a92221ca637fdbc2891e574`;
- luminosity: 109.82 fb^-1;
- collision energy: 13.6 TeV;
- current nominal uncertainty model: MC statistical, luminosity, and stored systematic envelopes.

Use the current canonical render payload associated with this page. Do not fall back to an
older photon-fake snapshot, old PT-binned DY payload, or legacy processor output.

## 3. Physics definition of the plotted GCR

The trusted selection is the one implemented in:

```text
autonomous_allhad/autonomous_allhad/real_subset_worker.py
```

The high-dM GCR requires, in summary:

- EGamma data;
- exactly one medium photon;
- photon pT > 220 GeV;
- photon in ECAL barrel or endcap fiducial acceptance, excluding the transition gap;
- photon `cutBased >= 2` and electron veto;
- no veto electron or muon and zero selected tau;
- photon-cleaned AK4 jets with DeltaR(photon, AK4 jet) >= 0.2;
- photon-cleaned AK8 jets with DeltaR(photon, AK8 jet) >= 0.4;
- at least five photon-cleaned AK4 jets;
- at least one photon-cleaned b jet;
- original missing transverse momentum below 250 GeV;
- photon recoil UT above 250 GeV;
- high-dM opening-angle requirement;
- photon-cleaned HT above 300 GeV.

The `dRGJ > 0.25` string in the new GJ-4Jets sample name is a generator-level sample filter.
It is not an additional reconstructed-event GCR cut.

The GCR data source must remain EGamma only. Do not add JetMET, Muon, or another data
stream to improve the normalization.

## 4. Current MC context

The current nominal histogram snapshot includes the adopted replacements:

- photon+jets: GJ-4Jets samples with HT, photon-pT, and `dRGJ > 0.25` generator filters;
- QCD: HT-binned QCD samples;
- DY: exclusive DYto2E-4Jets, DYto2Mu-4Jets, and DYto2Tau-4Jets samples;
- the remaining standard Top, W, Z-to-invisible, VV/VVV, and rare backgrounds.

Legacy PTLL-binned DY artifacts must not be reintroduced. Do not alter process grouping or
stack order as part of this plotting task.

The latest GCR audit in this task found a raw high-dM GCR normalization disagreement of
approximately Data/MC = 1.75. The exact value must be recomputed from the active render
payload; it must not be hard-coded into the plot. The unit-area rendering is intended to expose
the shape comparison after removing this overall normalization difference.

## 5. Run-2 reference and rationale

The Run-2 analysis note is:

```text
/Users/taiwoomac/Desktop/Working/CMS-AN-2026-090/AN-26-090/AN2019_016_v9.pdf
```

Relevant material:

- Section 7.3, equations 10-12;
- Figures 28 and 29: inclusive photon-control-region shape comparisons;
- Figures 32 and 33: GCR comparisons split by Nb and Nj;
- Figures 36 and 37: the resulting photon shape factors.

Run-2 defined

```text
Q(Nb, Nj) = Ndata(Nb, Nj) / NMC(Nb, Nj)
```

and then formed a shape factor of the form

```text
Sgamma(i) = Ndata(i) / [Q(Nb, Nj) * NMC(i)].
```

Therefore the GCR overall normalization mismatch was removed before interpreting the shape.
The absolute Z-to-invisible normalization was supplied separately by RZ from the dilepton
control regions.

The Run-2 high-dM GCR raw yields were Data = 17,938 and total MC approximately 10,921,
or Data/MC approximately 1.64. Figures 28, 29, 32, and 33 nevertheless normalized data and
the total MC to unit area. The plotted agreement was explicitly a shape comparison.

## 6. Exact unit-area transformation

For one GCR-only plot with data bins `d_i` and MC process bins `m_{p,i}`, define

```text
D = sum_i d_i
M = sum_i sum_p m_{p,i}

d'_i     = d_i / D
m'_{p,i} = m_{p,i} / M
m'_i     = sum_p m'_{p,i}
```

Requirements:

1. Normalize data once by its own integral.
2. Normalize every MC process with the same common factor `1/M`.
3. Never normalize each MC process independently.
4. Preserve the MC process fractions and stack order.
5. Calculate the ratio as `d'_i / m'_i`.
6. Scale data statistical errors by `1/D`.
7. Scale nominal MC statistical errors by `1/M`.
8. Respect the existing overflow/underflow and rebinning policy. Normalize only after the
   final plotted binning and flow-bin treatment have been established.
9. Do not alter or save the normalized arrays back into the source histogram payload.

For a GCR-only plot containing multiple adjacent GCR blocks, such as the Nt split recoil
plot, use one common data integral and one common MC integral across the entire plotted GCR
figure. Do not normalize Nt=0 and Nt>=1 independently. This preserves their relative event
fractions and follows the Run-2 choice to integrate over top/W categories for the shape factor.

## 7. Uncertainty treatment

A unit-area plot is insensitive to a fully correlated global rate change.

The correct shape-only treatment is:

- omit the luminosity normalization uncertainty from the normalized band;
- for every stored systematic Up/Down total shape, normalize that varied total to unit area
  before comparing it with the normalized nominal total;
- retain bin-dependent shape changes;
- retain MC statistical uncertainty, scaled consistently with the normalized MC integral;
- scale data statistical uncertainty by the data integral.

Do not merely multiply the current absolute `background_unc` array by `1/M` if that array
still contains luminosity and other pure normalization components. That would leave rate-only
uncertainties in a plot explicitly designed to remove the rate.

If the current render payload does not retain enough per-source information to reconstruct a
proper normalized uncertainty envelope, document that limitation in the plot summary instead
of inventing a covariance model. The central normalized shapes and data statistical errors can
still be drawn, but the MC band must be labeled precisely.

## 8. Plots in scope

Apply the policy to outputs whose complete scope is GCR.

### High-dM distributions

Current files include:

```text
plots/highdm/cr_gcr_ut.{png,pdf}
plots/highdm/cr_gcr_ht.{png,pdf}
plots/highdm/cr_gcr_jet_pt.{png,pdf}
plots/highdm/cr_gcr_bjet_pt.{png,pdf}
plots/highdm/cr_gcr_fatjet_pt.{png,pdf}
plots/highdm/cr_gcr_nb.{png,pdf}
plots/highdm/cr_gcr_njet.{png,pdf}
plots/highdm/cr_gcr_nfatjet.{png,pdf}
plots/highdm/cr_gcr_ntop.{png,pdf}
plots/highdm/cr_gcr_nw.{png,pdf}
```

UT is the primary physics figure and must be validated first.

### High-dM category/recoil plots

```text
plots/categories/highdm_cr_gcr_recoil.{png,pdf}
plots/categories/highdm_cr_gcr_recoil_ntop_split.{png,pdf}
```

### Low-dM GCR-only plots

Apply the same policy to the individual low-dM GCR recoil/variable plots generated from
`cat4_GCR_lowDeltaM`, including the existing `lowdm_cr_gcr_*` family.

### Explicitly out of scope

Do not independently unit-normalize the GCR block inside a figure that combines GCR with
LLCR, QCDCR, DY2E, or DY2M. Examples include combined all-CR overview figures. Mixing a
unit-normalized GCR block with absolute-yield blocks would have no coherent y-axis meaning.

Do not modify:

- LLCR, QCDCR, DYCR, VR, or SR plots;
- SR blinding;
- DY RZ factors;
- photon-fake estimates;
- nominal histogram files;
- process grouping or stack order;
- Combine inputs or datacards;
- public web content until the user explicitly requests publication.

## 9. Presentation requirements

Retain the current CMS/mplhep style and the user's established label rules:

```python
hep.cms.label(
    llabel="Work in progress",
    rlabel=r"109.82 fb$^{-1}$ (13.6 TeV)",
    ax=ax,
)
```

Do not pass `data=True`, `data=False`, `simulation=True`, or similar label-state options.

Additional requirements:

- no plot title;
- approximately square individual distribution plots;
- y-axis label: `Normalized events`;
- lower-panel label: `Data/MC`;
- UT label: `$U_{T}$ (GeV)`;
- missing-transverse-momentum label: `$p_{T}^{miss}$ (GeV)`;
- no slash-style MET or recoil notation;
- no artificial horizontal padding outside the first and last histogram edges;
- retain the existing process colors, order, and legend styling;
- do not annotate the raw Q value on the main figure unless the user asks;
- save both PNG and PDF.

## 10. Validation and acceptance checks

Before showing any result, verify mechanically for every GCR-only plot:

```text
abs(sum(data_normalized) - 1) < 1e-10
abs(sum(total_mc_normalized) - 1) < 1e-10
total_mc_normalized == sum(normalized_mc_processes)
ratio == data_normalized / total_mc_normalized in every valid bin
```

Also verify:

- the raw integrals extracted before normalization reproduce the current nominal payload;
- all process fractions before and after normalization agree;
- the ratio changes only by the single global factor `M/D` relative to the raw ratio;
- the normalized systematic Up/Down shapes each integrate to one;
- the luminosity-only component is absent from the normalized band;
- no non-GCR plot file changes in a targeted rerender;
- the GCR data process remains EGamma;
- PNG and PDF renders are visually identical in content;
- labels, legend, axes, and ratio panel do not overlap;
- the high-dM UT plot is inspected manually before producing the complete GCR set.

Record the raw `D`, raw `M`, and `D/M` values in a machine-readable rendering summary for
auditability, but do not place them on the main figure.

## 11. Recommended execution order for the plotting agent

1. Read the current `plot_control_search_bins_style.py` completely.
2. Locate the canonical current `hists.json`/render payload used for the public page.
3. Confirm its SHA-256 and sample-policy metadata against `page_summary.json`.
4. Implement the GCR-only unit-area switch inside the existing record/render flow.
5. Draw only `cr_gcr_ut` into a new temporary output directory.
6. Run all mechanical checks in Section 10.
7. Visually inspect the high-dM UT PNG and PDF.
8. After the UT plot is accepted, generate the remaining GCR-only high-dM and low-dM plots.
9. Compare output manifests and prove that non-GCR files are unchanged.
10. Report the local output directory to the user. Do not publish without an explicit request.

## 12. Known interpretation boundary

Unit-area normalization does not resolve or explain the raw GCR normalization disagreement.
It implements the Run-2 shape-comparison convention. The GJets-QCD overlap definition,
missing photon-associated samples, generator cross sections, and any future Q or Sgamma
measurement remain separate physics tasks and must not be silently absorbed into this plotting
change.
