# Search for SUSY top squark production in all-hadronic final states

This is TROTASR, the code for CMS NPS-26-012. The search uses six Run 3 eras
(2022–2025), with blinded signal regions and partial systematics.

- [Workflow and datacard review examples](TROTASR/README.md)
- [Find the approved results and figures](TROTASR/NOMINAL_CONTINUATION.md)
- The [six-era inclusive ttbar top-pT prescription](TROTASR/jsons/analysis.json) defines the central correction and its paired uncertainty; the completed contributions are listed in the results above.
- [Object definitions](TROTASR/utils/ids.adl),
  [event selections](TROTASR/utils/event_selections.adl), and
  [corrections](TROTASR/utils/corrections.adl)

The [208-bin card and template release](https://github.com/resisov/run3_stop/releases/tag/trotasr-run3-208bin)
contains the 895 signal cards, one CR-only card, and all 13 ROOT files they
reference. Extract the archive under `TROTASR/` to keep the relative shape
paths intact. The shared observations file uses CR data and MC placeholders
for SR bins. The release contains fit inputs.
