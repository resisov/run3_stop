# Finalization checklist

The current package is complete for internal review. Before freezing AN or
conference-quality figures:

- [ ] Add a standalone QCD photon-origin fraction versus \(U_T\) plot
  (prompt, electron matched, truth-fake).
- [ ] Add the full fake-background uncertainty envelope to the A-region target
  validation and final prefit stack. The current bands are statistical or
  available-template statistical bands.
- [ ] Remove or mark \(p_T^{miss}\) as unavailable in the method schematic
  until a below-250-GeV histogram is produced.
- [ ] Resolve and document the generator-level QCD/\(\gamma+\)jets
  prompt-photon overlap policy.
- [ ] Justify the \(\pm30\%\) prompt and \(\pm50\%\) electron contamination
  variations in the AN.
- [ ] Decide whether the 45.18% nonclosure is used as a normalization-only
  nuisance or receives a shape component.
- [ ] Keep all claims prefit. Do not use the prompt-normalization fitted
  Data/MC result near unity.
- [ ] State explicitly that the nominal intermediate was not modified.
- [ ] Exclude `GCR/met`; it is empty because of the current histogram range,
  not because the selected physics yield is zero.

