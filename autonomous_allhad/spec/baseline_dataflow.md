# Baseline Dataflow

1. Dataset names are read from the 2024 dataset list and metadata JSON.
2. NanoAOD Events are processed by `AnalysisProcessor`.
3. Objects are corrected, identified, and cleaned.
4. Region selections fill weighted histograms for nominal and systematics.
5. Shift processors are merged, scaled, converted to templates, datacards, plots, and limits.

Trace: `analysis/processors/stop_processor_v4.py`, `configs/stop_2024.yaml`.
