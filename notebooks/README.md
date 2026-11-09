# Notebooks

The curriculum: each notebook teaches one idea, and the tool notebooks show the **real code** from `tools/`
(via `inspect.getsource`) instead of carrying a second copy of it. They are committed **executed**, with the
outputs of the last run (V3.1 models, November 2026).

| Notebook | What it teaches | Data it needs |
|---|---|---|
| `tyre_degradation.ipynb` | Module 0 — exploratory analysis: cleaning, degradation patterns | FastF1 cache |
| `lap_time_model.ipynb` | Module 1 — XGBoost regression, confounding (fuel vs compound), GroupKFold, physics gates | lap Parquet |
| `overtaking_model.ipynb` | Module 2 — classification, calibration, train/serve skew | lap Parquet |
| `driver_style_fingerprinting.ipynb` | Module 3 — telemetry resampling, data-quality gates, PCA, K-Means, UMAP | FastF1 cache (qualifying telemetry) |
| `undercut_calculator.ipynb` | Tool 1 — two models in one decision; validated on 101 real undercut attempts | lap Parquet |
| `stint_optimizer.ipynb` | Tool 2 — deterministic sweep, U-curves; checked against 193 real one-stops | lap Parquet |
| `race_simulator.ipynb` | Tool 3 — Monte Carlo race engine; 20 races from the real grid | lap Parquet + FastF1 cache (qualifying) |
| `qualy_predictor.ipynb` | Tool 4 — what a correction needs in order to matter | lap Parquet + FastF1 cache (qualifying) |
| `what_if_simulator.ipynb` | Tool 5 — common random numbers, paired comparisons | lap Parquet + FastF1 cache (qualifying) |
| `telemetry_clash.ipynb` | Tool 6 — signal processing, alignment error | FastF1 cache (qualifying telemetry) |

**Data.** `data/laps_2022_2025.parquet` comes from `python -m tools.export_cache`. The FastF1 cache is read
**offline** from `$F1_CACHE_DIR`, `cache_drive/f1_cache` or `cache/` — F1's archive answers 401 to downloads since
October 2026, so only cached sessions load. Models: `models/` or `$F1_MODEL_DIR`.

**Colab.** The first cell clones the repo and reads the models from `MyDrive/f1_models` and the Parquet from
`MyDrive/f1_data/` (override with `F1_MODEL_DIR`, `F1_LAPS`, `F1_CACHE_DIR`).

**Order.** `tyre_degradation` → `lap_time_model` → `overtaking_model` → any tool.
