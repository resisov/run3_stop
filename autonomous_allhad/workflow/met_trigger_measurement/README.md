# MET trigger measurement

This directory contains only the configuration and machine outputs for the
Run-3 MET-trigger efficiency and data/MC scale-factor measurement.  All code is
in `../measure_trigger.py`.  It follows AN2019-016 section 5.1.1:

- the signal-trigger numerator is the analysis PFMET/PFMETNoMu OR;
- genuine-pmiss efficiency uses an independent single-electron reference;
- the dedicated measurement selection removes the signal-trigger requirement;
- only EGamma data and semileptonic TT MC are accepted by the code.

The adopted 2024 payload is applied to every MC process in MET-triggered
analysis regions.  There is no separately measured QCD-only correction in this
campaign, and the histogram code must not substitute a missing QCD payload with
unity.

Files and status:

- `config_2024.json`: frozen Run-3 trigger paths, binning, and adoption gates;
- `../measure_trigger.py`: build/count/recover/reduce/export entry point;
- `outputs/`: machine-produced counts, plots, fits, and validation only.
