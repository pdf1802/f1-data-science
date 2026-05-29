# 🏎️ F1 Data Science

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastF1](https://img.shields.io/badge/data-FastF1-E10600.svg)](https://github.com/theOehrly/Fast-F1)

An end-to-end **Formula 1 race strategy system** built as a teaching portfolio. The project trains a set of **science modules** on real FastF1 telemetry (2022–2025) and applies them through **strategy tools** that answer concrete race questions — *should I pit now? is pole pace real race pace? does this driver eat tyres?*

The project is structured as a curriculum: each module teaches one ML/statistics concept (XGBoost regression, classification, unsupervised clustering, Monte Carlo) and each tool demonstrates how the models compose into something useful.

---

## 📚 Curriculum

The repository follows a **two-layer structure**:

- **Science modules** train the underlying models (one notebook per ML concept).
- **Strategy tools** consume those models to answer real race questions.

### Science modules

| # | Notebook | Concept | Output |
|---|----------|---------|--------|
| **0** | [`tyre_degradation.ipynb`](notebooks/tyre_degradation.ipynb) | Exploratory data analysis | Establishes the data cleaning + degradation patterns that justify Module 1 |
| **1** | [`lap_time_model.ipynb`](notebooks/lap_time_model.ipynb) | XGBoost regression | Predicts `LapDelta` from compound, tyre age, team, conditions |
| **2** | [`overtaking_model.ipynb`](notebooks/overtaking_model.ipynb) | XGBoost classification | Predicts overtake probability from gap, tyre delta, DRS state |
| **3** | [`driver_style_fingerprinting.ipynb`](notebooks/driver_style_fingerprinting.ipynb) | Telemetry resampling + KMeans clustering | Driver style signature (6 features → 4 clusters) |

### Strategy tools

| # | Notebook | Question | Uses | Status |
|---|----------|----------|------|--------|
| **1** | [`undercut_calculator.ipynb`](notebooks/undercut_calculator.ipynb) | Will the undercut actually work right now? | M1, M2 | ✅ Complete |
| **2** | [`stint_optimizer.ipynb`](notebooks/stint_optimizer.ipynb) | What's the optimal pit lap for this stint? | M1 | ✅ Complete |
| **3** | [`race_simulator.ipynb`](notebooks/race_simulator.ipynb) | How does the race play out lap-by-lap? | M1, M2 | ✅ Complete |
| **4** | [`qualy_predictor.ipynb`](notebooks/qualy_predictor.ipynb) | Is pole pace real race pace? | M1 | ✅ Complete |
| **5** | [`what_if_simulator.ipynb`](notebooks/what_if_simulator.ipynb) | Counterfactual race outcomes (different strategy, weather, etc.) | M1, M2 | ✅ Complete |
| **6** | *Telemetry Clash* | Who gains in which corners? | M3 | 🚧 Next |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Git, pip

### Install

```bash
git clone https://github.com/pdf1802/f1-data-science.git
cd f1-data-science
pip install -r requirements.txt
```

### Run the notebooks

The notebooks are designed to run in **Google Colab** (each one has a Colab badge at the top) but they also run locally in Jupyter. The FastF1 cache is configured to write to Google Drive in Colab — change `CACHE_DIR` if running locally.

Recommended order for first-time readers:

1. `tyre_degradation.ipynb` — the EDA that motivates everything else
2. `lap_time_model.ipynb` — the workhorse model the tools depend on
3. Any tool notebook — they document how the models compose

---

## 📊 Validation highlights

Models are validated on held-out races, not just train/test splits:

- **Module 1** — strategy recommender tested on Miami 2024 Leclerc one-stop. Predicted optimal pit window matches the team's actual decision within 2 laps.
- **Module 3** — clustering correctly groups HAM + LEC (similar smooth/late-brake style) and isolates the McLaren pair (distinctive aggressive throttle). Documented V1 → V2 iteration after catching a corrupted DRS channel that was driving a spurious 1-driver cluster.
- **Tool 4** — Bahrain 2023 ranking predicts ALO P1 race pace; ALO finished P3 on the road (VER won via driver extraction above car baseline, which is not a feature in the model — see *Known limitations* below).

---

## ⚠️ Known limitations

This section is intentional. Acknowledging limitations is a feature of the project, not a bug.

- **Car confound (Module 3).** Driver style is extracted from telemetry that is partially shaped by the car. A Ferrari with strong front-end will look "smooth" regardless of who is driving it. Mitigated by clustering on inputs that are largely driver-controlled (throttle smoothness, brake application %), but not eliminated.
- **Tool 4 underestimates flying-lap pace on fresh Softs.** The model corrects qualifying times for fuel + compound + degradation, so by construction it predicts a degraded race lap, not a push lap. It is a strong *ranking* tool, not a fastest-lap tool.
- **Compound delta is estimated per team, with a physics clamp.** Negative compound deltas (which are physically implausible) are clamped to zero to prevent corrections from amplifying noise on short stints.
- **FastF1 DRS channel is unreliable for some teams** (notably McLaren in 2024). Module 3 V2 drops the `drs_usage_pct` feature for this reason.
- **No `nGear` / `RPM` features yet.** Both flagged as the most valuable telemetry additions for future Module 3 iterations; currently excluded because the sample size is too small and they would likely overfit.

---

## 🛠️ Stack

| Layer | Tools |
|-------|-------|
| Data | FastF1 (2022–2025 sessions, cached locally) |
| ML | XGBoost, scikit-learn (KMeans, StandardScaler, PCA), UMAP |
| Analysis | pandas, NumPy, scipy.interpolate |
| Visualisation | Matplotlib, Plotly |
| Explainability | SHAP |

---

## 📁 Repository structure

```
f1-data-science/
├── notebooks/                          ← All science modules + strategy tools
│   ├── tyre_degradation.ipynb          ← Module 0 (EDA)
│   ├── lap_time_model.ipynb            ← Module 1
│   ├── overtaking_model.ipynb          ← Module 2
│   ├── driver_style_fingerprinting.ipynb  ← Module 3
│   ├── undercut_calculator.ipynb       ← Tool 1
│   ├── stint_optimizer.ipynb           ← Tool 2
│   ├── race_simulator.ipynb            ← Tool 3
│   ├── qualy_predictor.ipynb           ← Tool 4
│   └── what_if_simulator.ipynb         ← Tool 5
├── models/                             ← Trained .pkl artefacts (gitignored)
├── requirements.txt
├── LICENSE
└── README.md
```

Model artefacts (`.pkl`) are not committed — they are generated by running the module notebooks. Each module notebook ends with a save cell that writes to `models/` (or `/content/drive/MyDrive/f1_models/` in Colab).

---

## 📜 License

MIT — see [LICENSE](LICENSE).

## ⚖️ Disclaimer

Educational and fan-made. Not affiliated with Formula 1, the FIA, or any team. Not for betting.

## 🔗 References

- [FastF1](https://github.com/theOehrly/Fast-F1) — the data layer this entire project depends on
- [OpenF1 API](https://openf1.org) — live timing data
