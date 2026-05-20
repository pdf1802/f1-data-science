# 🏎️ F1 Data Science

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

An end-to-end **Formula 1 race strategy system** powered by machine learning models trained on real telemetry data from [FastF1](https://github.com/theOehrly/Fast-F1). Four ML models enable six interactive strategy tools for analyzing race tactics, driver behavior, and tactical decisions.

## 🎯 Overview

This project combines **data science**, **telemetry analysis**, and **race strategy** to answer critical F1 questions:
- Should I pit now or stay out?
- What's the optimal pit window?
- Does this driving style consume tyres faster?
- Who gains in which corners?

## 📊 ML Models

| # | Model | Type | Input Features | Output |
|---|-------|------|-----------------|--------|
| **1** | Lap Time Predictor | XGBoost Regression | Tyre age, compound, team, track conditions | Lap time delta |
| **2** | Overtaking Probability | XGBoost Classifier | Gap size, tyre degradation, DRS eligibility | Overtake likelihood |
| **3** | Driver Style DNA | KMeans Clustering | Throttle map, braking profile, apex speed patterns | Driver fingerprint |
| **4** | Race Simulator | Monte Carlo | Modules 1 + 2 combined | Race outcome predictions |

## 🛠️ Strategy Tools

Six interactive **Streamlit apps** powered by the models above:

| Tool | Question Answered | Models | Status |
|------|-------------------|--------|--------|
| **Undercut Calculator** | Should I pit now? | 1, 2 | ✅ Ready |
| **Stint Optimizer** | What's the optimal pit lap? | 1 | ✅ Ready |
| **Tyre Whisperer** | Does driving style eat tyres? | 1, 3 | ✅ Ready |
| **Qualy-to-Race Predictor** | Is pole pace real race pace? | 1 | ✅ Ready |
| **Telemetry Clash** | Who gains where? | 3 | ✅ Ready |
| **Chaos Alert** | Safety car pit decision engine | 1, OpenF1 API | ✅ Ready |

## 📁 Project Structure

```
f1-data-science/
├── analysis/
│   ├── notebooks/                    ← Jupyter notebooks (model training + analysis)
│   │   ├── lap_time_model.ipynb
│   │   ├── overtaking_model.ipynb
│   │   ├── driver_style_fingerprinting.ipynb
│   │   ├── race_simulator.ipynb
│   │   └── strategy_tools/           ← Interactive tool notebooks
│   └── app/                          ← Streamlit applications
│       ├── undercut_calc.py
│       ├── stint_optimizer.py
│       ├── tyre_whisperer.py
│       ├── qualy_predictor.py
│       ├── telemetry_clash.py
│       └── chaos_alert.py
├── tools/
│   └── shared.py                     ← Central utilities (shared by all apps)
├── models/                           ← ML model files (.pkl)
│   └── README.md                     ← Instructions for model placement
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Git
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/pdf1802/f1-data-science
cd f1-data-science

# Install dependencies
pip install -r requirements.txt
```

### Add Pre-trained Models

Model files (.pkl) are stored externally (not in Git to avoid large binary files).

1. **Retrieve models** from the project's Google Drive or model registry
2. **Place them** in the `models/` directory (see `models/README.md` for details)
3. Run any Streamlit app:

```bash
streamlit run analysis/app/undercut_calc.py
```

### Train Models from Scratch

Run the notebooks in numerical order:

```bash
jupyter notebook analysis/notebooks/lap_time_model.ipynb
jupyter notebook analysis/notebooks/overtaking_model.ipynb
jupyter notebook analysis/notebooks/driver_style_fingerprinting.ipynb
jupyter notebook analysis/notebooks/race_simulator.ipynb
```

## 📈 Data Sources

- **Historical Race Data**: [FastF1](https://github.com/theOehrly/Fast-F1) — fetched live and cached locally
- **Live Race Data**: [OpenF1 API](https://openf1.org) — free, no authentication required
- **Telemetry Features**: Throttle maps, braking profiles, apex speeds, GPS coordinates

## 🔄 Workflow

```
Raw F1 Telemetry (FastF1)
    ↓
Feature Engineering (analysis/notebooks)
    ↓
ML Models Training (XGBoost, KMeans)
    ↓
Model Serialization (.pkl files)
    ↓
Streamlit Apps (Real-time strategy tools)
    ↓
Race Strategy Decisions
```

## 💻 Technologies Used

| Component | Technology |
|-----------|-----------|
| **Data Processing** | Pandas, NumPy, Polars |
| **ML Models** | XGBoost, scikit-learn |
| **Visualization** | Matplotlib, Plotly, Streamlit |
| **Data Source** | FastF1, OpenF1 API |
| **Notebooks** | Jupyter |
| **Environment** | Python 3.9+ |

## 📋 Requirements

See `requirements.txt` for full dependencies:

```
pandas>=1.5.0
numpy>=1.23.0
xgboost>=1.7.0
scikit-learn>=1.2.0
fastf1>=3.0.0
streamlit>=1.20.0
plotly>=5.0.0
polars>=0.19.0
requests>=2.28.0
```

## 🧪 Testing & Validation

- Models validated on historical F1 seasons (2020–2024)
- Cross-validation with out-of-sample test sets
- Performance benchmarks for each tool available in `analysis/notebooks`

## 📝 Usage Examples

### Example 1: Check if undercut is worth it

```bash
streamlit run analysis/app/undercut_calc.py
# → Select race, driver, pit gap, tyre compound
# → Model predicts lap time gain/loss
```

### Example 2: Analyze driver telemetry patterns

```bash
jupyter notebook analysis/notebooks/driver_style_fingerprinting.ipynb
# → Clusters drivers by driving style
# → Identifies tyre-eating patterns
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add more regression features (fuel load, ambient temperature)
- [ ] Expand driver clustering with additional seasons
- [ ] Real-time live race simulation
- [ ] API deployment (FastAPI)
- [ ] Unit tests for model validation

Please open an **Issue** or **Pull Request** to contribute.

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) file for details.

## ⚖️ Disclaimer

This project is **fan-made and educational**. It is not affiliated with Formula 1, Liberty Media, or any F1 teams. Predictions are for entertainment and learning purposes only and should not be used for actual betting or wagering.

## 🔗 References

- **FastF1**: [theOehrly/Fast-F1](https://github.com/theOehrly/Fast-F1)
- **OpenF1 API**: [openf1.org](https://openf1.org)
- **Official F1**: [Formula1.com](https://www.formula1.com)

---

**Last Updated**: May 2026 | **Status**: Active Development

For questions or suggestions, feel free to open an issue! 🏁
