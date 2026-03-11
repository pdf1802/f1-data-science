# 🏎️ F1 Data Science

An end-to-end F1 race strategy system built on real telemetry data from FastF1.
Four ML models power six interactive strategy tools.

## Modules

| # | Notebook | What it builds |
|---|----------|---------------|
| 1 | `lap_time_model.ipynb` | XGBoost regression → predicts lap time delta from tyre age, compound, team |
| 2 | `overtaking_model.ipynb` | XGBoost classifier → predicts overtake probability from gap and tyre state |
| 3 | `driver_style_fingerprinting.ipynb` | KMeans clustering on telemetry features → driver style DNA |
| 4 | `race_simulator.ipynb` | Monte Carlo race simulator using Modules 1 + 2 |

## Strategy Tools

| Tool | What it answers | Models used |
|------|----------------|-------------|
| **Undercut Calculator** | Should I pit now or stay out? | Module 1 + 2 |
| **Stint Optimizer** | What is the optimal pit lap? | Module 1 |
| **Tyre Whisperer** | Does your driving style eat tyres faster? | Module 1 + 3 |
| **Qualy-to-Race Predictor** | Is pole position real race pace? | Module 1 |
| **Telemetry Clash** | Which corner does Norris gain on Piastri? | Module 3 |
| **Chaos Alert** | Safety Car is out — pit now? | Module 1 + OpenF1 API |

## Repo structure

```
f1-data-science/
├── analysis/
│   ├── notebooks/          ← all Jupyter notebooks (Modules 1–4 + 6 tool notebooks)
│   └── app/                ← Streamlit app files (one per tool)
├── tools/
│   └── shared.py           ← central utility module — all apps import from here
├── models/
│   └── README.md           ← instructions for placing .pkl files (not in git)
├── requirements.txt
└── .gitignore
```

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/f1-data-science
cd f1-data-science
pip install -r requirements.txt
```

Copy your `.pkl` model files into `models/` (see `models/README.md`).

```bash
streamlit run analysis/app/undercut_calc.py
```

## Models

Model files are stored on Google Drive (not in this repo — binary files don't belong in Git).
All models can be reproduced by running the notebooks in order.

## Data

All race data is fetched live from [FastF1](https://github.com/theOehrly/Fast-F1) and cached locally.
Live race data uses the [OpenF1 API](https://openf1.org) (free, no auth required).
