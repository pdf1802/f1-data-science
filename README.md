# 🏎️ F1 Data Science

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastF1](https://img.shields.io/badge/data-FastF1-E10600.svg)](https://github.com/theOehrly/Fast-F1)

An end-to-end **Formula 1 race strategy system** built as a teaching portfolio. It trains **science modules** on real FastF1 lap data (2022–2025), composes them into a **lap-by-lap race engine** that is backtested against 36 real races, and puts that engine on the pit wall: a **Streamlit dashboard** of strategy tools and a live **Race Engineer** that makes the box / stay-out call while a race replays in [f1-race-replay](https://github.com/IAmTomShaw/f1-race-replay) — or lets you take control of the race from any lap.

The project is structured as a curriculum: each module teaches one ML/statistics concept (XGBoost regression and classification, fixed-effects regression, clustering, paired Monte Carlo) and each tool shows how the models compose into something useful. Every number below was re-run for this README; the limitations section is part of the project, not an afterthought.

![Race Engineer window](docs/race_engineer_window.png)

---

## 📚 Curriculum

### Science modules

| # | Notebook | Concept | Output |
|---|----------|---------|--------|
| **0** | [`tyre_degradation.ipynb`](notebooks/tyre_degradation.ipynb) | Exploratory data analysis | The data cleaning + degradation patterns that justify Module 1 |
| **1** | [`lap_time_model.ipynb`](notebooks/lap_time_model.ipynb) | XGBoost regression + fixed-effects reference | `LapDelta` from compound, tyre age and circuit (V3.1: fuel-corrected, clean air, dry races, physics gate) |
| **2** | [`overtaking_model.ipynb`](notebooks/overtaking_model.ipynb) | XGBoost classification, calibrated | P(pass the car ahead this lap) from the real gap, tyres, lap, position, circuit |
| **3** | [`driver_style_fingerprinting.ipynb`](notebooks/driver_style_fingerprinting.ipynb) | Telemetry resampling + KMeans | Driver style signature (5 features → 4 clusters, V2) |

### Strategy tools (dashboard tabs and notebooks)

| # | Notebook | Question | Uses |
|---|----------|----------|------|
| **1** | [`undercut_calculator.ipynb`](notebooks/undercut_calculator.ipynb) | Will the undercut actually work right now? | M1, M2 |
| **2** | [`stint_optimizer.ipynb`](notebooks/stint_optimizer.ipynb) | What's the optimal pit lap for this stint? | M1 |
| **3** | [`race_simulator.ipynb`](notebooks/race_simulator.ipynb) | How does the race play out lap by lap? | M1, M2, engine |
| **4** | [`qualy_predictor.ipynb`](notebooks/qualy_predictor.ipynb) | Is pole pace real race pace? | M1 |
| **5** | [`what_if_simulator.ipynb`](notebooks/what_if_simulator.ipynb) | Same races, different strategy: what changes? | M1, M2, engine |
| **6** | [`telemetry_clash.ipynb`](notebooks/telemetry_clash.ipynb) | Who gains in which corners? | signal processing |

### Race engine and live tools

| Piece | File | What it does |
|-------|------|--------------|
| Circuit profiles | [`tools/circuit_profiles.py`](tools/circuit_profiles.py) | Laps, typical lap, **pit loss measured from real green-flag stops** (Monaco 19 s … Singapore 27 s), Safety Car rate with empirical-Bayes shrinkage |
| Race engine | [`tools/race_engine.py`](tools/race_engine.py) | Race state for S simulations × D drivers and a `step()` that advances one lap; starts from the grid or from **any lap of a real race**, using only that race's laps up to that point |
| Backtest | [`tools/backtest.py`](tools/backtest.py) | From 40 % distance in 36 real races, does the engine predict the finishing order better than "the order stays the same"? |
| Race Engineer | [`tools/race_engineer.py`](tools/race_engineer.py) | Live feed → strategy call for one driver at most once per lap: options raced to the flag on the same random draws |
| Pit-wall window | [`integrations/f1_race_replay/`](integrations/f1_race_replay/README.md) | PySide6 window on f1-race-replay's telemetry stream; stand-in stream from the lap data; **take-control mode** |

---

## 🚀 Quick Start

```bash
git clone https://github.com/pdf1802/f1-data-science.git
cd f1-data-science
pip install -r requirements.txt
pip install -r integrations/f1_race_replay/requirements.txt   # PySide6, only for the Race Engineer window
```

### Data

F1's live-timing archive answers **401** to unauthenticated downloads since October 2026 (FastF1 issue #978), so the project works from one Parquet file exported from a FastF1 cache (43 races, 44,526 laps, 2022–2025; not committed):

```bash
python -m tools.export_cache --out data/laps_2022_2025.parquet     # from $F1_CACHE_DIR
```

### Train the models

```bash
python -m tools.train_models --laps data/laps_2022_2025.parquet
python -m tools.circuit_profiles --laps data/laps_2022_2025.parquet  # refresh tools/circuit_profiles.json
python -m tools.backtest --laps data/laps_2022_2025.parquet          # ~2 min
```

`train_models` keeps dry races, fits the physics reference, evaluates with **GroupKFold by race**, runs the **physics sanity gates** and only overwrites `models/` if every check passes (old files go to `models/backup_<timestamp>/`, plus `training_report.json`).

### Run the dashboard

```bash
streamlit run app.py
```

| Tab | What you can do |
|-----|-----------------|
| 🏁 Race Simulator | Real qualifying grid, editable strategies, 1,000+ Monte Carlo races on the lap-by-lap engine |
| 🔀 What-If | Change one driver's strategy and compare race by race against the baseline (common random numbers) |
| ⏱️ Undercut | Gap projections for 4 leader responses + BOX BOX / STAY OUT verdict |
| 🛞 Stint Optimizer | U-curves for all six legal one-stop compound pairs |
| 📊 Qualy → Race | Qualifying → predicted race-pace order, scored against the real result |
| 📡 Telemetry Clash | Distance-aligned telemetry, cumulative delta and corner-by-corner breakdown |

Circuit profiles set the defaults (laps, pit loss, Safety Car rate) for the chosen race.

### Run the Race Engineer

```bash
python -m integrations.f1_race_replay.replay_stream --race "2023 Bahrain Grand Prix" --speed 30
python -m integrations.f1_race_replay.race_engineer_window --driver ALO
```

Add `--take-control 20` to the stream: the real race runs to lap 20, then the engine races on and the window's **BOX** buttons make the calls. With f1-race-replay itself: `python main.py --telemetry` in that repo, then the window. Details in the [integration README](integrations/f1_race_replay/README.md).

### Run the tests

```bash
python -m pytest -q        # 76 tests, ~35 s
```

The tests run on small **stand-in models** with the same feature contracts as the real ones (built in `tests/conftest.py`), so they check logic and contracts — not prediction quality. `F1_USE_REAL_MODELS=1` runs them against your trained models. The window test runs offscreen and is skipped without PySide6.

### Run the notebooks

They run in **Google Colab** (badge at the top of each) or locally in Jupyter. The V3.1 sections of Modules 1 and 2 read `data/laps_2022_2025.parquet` instead of downloading from FastF1 — the file is not in the repo (export it from a FastF1 cache, see *Data*). Recommended order: `tyre_degradation` → `lap_time_model` → `overtaking_model` → any tool.

---

## 📊 Results (re-run on 26 Oct 2026, xgboost 3.2)

**Module 1 — lap-time model (V3.1).** 36 dry races, 17,442 clean-air laps.

| GroupKFold by race (honest) | Predict the median (bar to beat) | Random 80/20 split (leaky) |
|---|---|---|
| **0.669 s** MAE | 0.698 s (+4.1 %) | 0.562 s |

It beats the baseline on races it has never seen, but only just: compound, tyre age and circuit explain the *shape* of degradation (what the tools need), not every lap. All physics checks pass (fresh SOFT fastest, every compound degrades, SOFT degrades most, fresh HARD not clearly faster than MEDIUM).

**Physics reference (fixed-effects regression, 95 % CI clustered by race).** Fuel burn + track evolution **0.050 ± 0.008 s/lap** — the 0.055 s/lap assumed for the fuel correction is inside the interval. Fresh MEDIUM +0.42 s and HARD +0.38 s vs SOFT (±0.28 s): MEDIUM vs HARD cannot be told apart, so the gate doesn't pretend it can. Degradation SOFT 0.077, MEDIUM 0.048, HARD 0.045 s/lap — these slopes also continue the model beyond the tyre ages it has data for (trees can't extrapolate).

**Module 2 — overtake model (V3.1).** 26,015 car-behind-car situations, 4.0 % overtakes. Log loss **0.090** vs 0.168 for the constant model, ROC-AUC 0.946, PR-AUC 0.572. Calibrated (out of fold) up to ~40 %; slightly optimistic above (says 67 %, 60 % happen). Same situation (0.6 s, 22 laps fresher tyres): **P(pass) 44 % at Monaco vs 91 % in Bahrain**.

**Race engine — backtest on 36 dry races** (real state and real strategies from 40 % distance, pace estimated only from laps already run, 300 races each). Two versions: *in-sample* uses the trained models, which have seen every race — including the second part of the one being tested; *leave one race out* retrains Modules 1 and 2 and the circuit profiles without that race first (`python -m tools.backtest --loro`).

| | Engine, leave one race out | Engine, in-sample | "Order stays the same" |
|---|---|---|---|
| Spearman ρ with the real result | **0.894** | 0.902 | 0.846 |
| Mean absolute position error | **1.50** | 1.41 | 1.95 |
| Races where the engine ranks better | **75 %** (27/36) | 78 % (28/36) | |
| Real position inside the simulated 10–90 % range | **78.9 %** | 80.1 % | |

Out of sample the engine loses a little, as it should, and still beats the baseline clearly. `RACE_FORM_STD = 0.25 s/lap` is the value whose 80 % intervals contain the real position ~80 % of the time (in-sample: 0.12 → 68 %, overconfident; 0.40 → 88 %, too cautious; the log score prefers 0.30 by a hair). Out of sample the baseline wins 9 races: clearly in Melbourne 2023 (ρ 0.61 vs 0.78, three red flags), Singapore 2023 (0.61 vs 0.76) and Monaco 2024 (0.79 vs 0.97, where the order barely changes), narrowly in the other six.

**Race Engineer — live state accuracy.** Cars behind the followed driver have not crossed the line yet when the call is made, so their line time is estimated (`python -m integrations.f1_race_replay.estimate_accuracy`, 2,736 estimates in Bahrain 2023, Monza 2024, Monaco 2025). From recent pace alone — nothing the engineer could not know — the median error is **0.48 s** (90 % under 3.0 s) when that car does not stop in the lap, and **16 s** when it does (its stop is still in the future; a car already in the pit lane is detected through `in_pit`). Using the stand-in stream's `rel_dist` the median falls to 0.17 s, but that is an optimistic bound: the stand-in's `rel_dist` is built from the lap's real duration.

**Module 3** clusters HAM + LEC together (smooth, late braking) and isolates the McLaren pair; the V1 → V2 iteration removed a corrupted DRS channel that created a one-driver cluster.

---

## ⚠️ Known limitations

Acknowledging limitations is a feature of the project, not a bug.

- **One-stop strategies look too good at abrasive tracks.** Degradation is one curve per compound with a circuit offset, so per-circuit degradation is compressed. The Race Engineer often says "no more stops" in Bahrain, where teams really two-stop.
- **Survivorship bias at high tyre ages.** Only tyres that were still good get run long, so the data under-states late-stint degradation; the linear tail beyond the 90th-percentile age is a partial fix.
- **Fresh MEDIUM vs HARD are indistinguishable** in the data (±0.28 s), so the models can't rank them.
- **No fuel effect between strategies.** Fuel is the same for everyone at a given lap, so mirror strategies cost the same.
- **Qualy → Race ≈ qualifying order in V3.1.** The lap model no longer uses the team (it was a race fingerprint), so team-level race-pace corrections are gone; the tab says so.
- **Rivals in the Race Engineer follow a default plan** (stop at the edge of the tyre-age data / two-compound rule), not their real strategy. Calls are "given what we assume they do".
- **Dry races only.** Seven wet races are excluded from training; on INTERMEDIATE / WET the Race Engineer makes no call.
- **2022–2025 era only, and not every race.** 2026 cars and rules are different and deliberately excluded. The cache behind the Parquet holds 43 races on 13 circuits — 8 from 2022, 12 from 2023, 11 from 2024, 12 from 2025 — not the full calendars (e.g. no China, Canada, Spain, Hungary, Netherlands, Mexico, Brazil, Las Vegas, Qatar, USA).
- **Overtake probabilities are slightly optimistic above 40 %** (calibration table above).
- **Car confound (Module 3).** Driver style is extracted from telemetry partly shaped by the car; mitigated by driver-controlled inputs, not eliminated.
- **Telemetry Clash alignment.** Each car's `Distance` is integrated from its own speed trace, so slow corners show short ±200 ms spikes in the running gap. Trends between corners are reliable; single spikes are not.
- **No `nGear` / `RPM` features** in Module 3 yet (too few samples; would likely overfit).
- **Stand-in stream positions are schematic** (constant speed within a lap, on a circle): the Parquet has lap times, not GPS.

### Fixed along the way (each one is a lesson)

- **V2 → V3, fuel/compound confound.** The target `lap time − best lap` still contained fuel: softs run early on a heavy car, so the model learned that *softs are slow* and *tyres get faster with age*. V3 subtracts the fuel effect, drops `LapNumber` (a proxy for the compound), adds a monotonic constraint on tyre age and gates saving on physics checks.
- **V3 → V3.1, race fingerprints.** V3 lost to "predict the median" on unseen races (0.968 vs 0.796 s): track/air temperature are constant within a race and let the trees memorise races. Weather dropped, circuit added, clean-air laps only (softs run in traffic), dry races only.
- **Wrong race in the training data.** `'Great Britain'` was fuzzy-matched by FastF1 to the **Austrian GP** for 2023 and 2024; full event names are now asserted.
- **Module 2 train/serve skew.** `GapAhead` was a lap-time difference in training and a real gap at inference; overtakes included pit stops; `scale_pos_weight` inflated probabilities the simulator consumes.
- **Race Engineer dropped cars.** The live state was built at the leader's lap, so every car that had not crossed that line yet disappeared ("ALO is not running"). Now it is built at the followed driver's lap, with estimated line times for cars behind and lapped cars, and tested for future leakage.
- **Two clocks.** f1-race-replay sends lap end times on the FastF1 session clock and the same moments on the replay clock; comparing the wrong pair would make every lap look either completed or not started.
- **Found by an independent review** (a fresh reviewer who had not written the code, run before release): after a red flag the "out of the race" rule dropped every car behind (it counted seconds; it now counts the field's laps); a car that pitted at the end of lap N rejoined ~20 s too far ahead (a stop's loss is ~4 s on the in-lap and ~19 s on the out-lap — the engine now charges the out-lap part); FastF1's generated placeholder rows after a retirement looked like completed laps; the stand-in stream reported a Safety Car from the start of its lap (now a lap late, never early); and the backtest was in-sample — now also run leave-one-race-out.

---

## 🛠️ Stack

| Layer | Tools |
|-------|-------|
| Data | FastF1 (2022–2025, cached) → Parquet |
| ML / statistics | XGBoost, scikit-learn, fixed-effects OLS with clustered CIs, UMAP |
| Simulation | NumPy (vectorised over simulations × drivers), paired Monte Carlo / common random numbers |
| Analysis | pandas, SciPy |
| Apps | Streamlit (dashboard), PySide6 (Race Engineer window), TCP JSON-lines (f1-race-replay protocol) |
| Visualisation | Matplotlib, Plotly |
| Testing | pytest, Streamlit AppTest, offscreen Qt |
| Explainability | SHAP |

---

## 📁 Repository structure

```
f1-data-science/
├── notebooks/                       ← science modules + strategy tools (Colab badges)
├── tools/                           ← reusable logic imported by the app, notebooks and tests
│   ├── shared.py                    ← model loading, feature contracts, constants
│   ├── training.py · train_models.py← Module 1 + 2 V3.1 pipelines, evaluation, sanity gates
│   ├── export_cache.py              ← FastF1 cache → Parquet
│   ├── circuit_profiles.py (+ .json)← measured per-circuit constants
│   ├── race_engine.py               ← lap-by-lap engine, LiveRace, state from a real race
│   ├── backtest.py                  ← engine vs reality on 36 races
│   ├── race_engineer.py             ← live feed → strategy call
│   └── undercut.py · stint_optimizer.py · race_simulator.py · qualy_predictor.py · telemetry_clash.py
├── integrations/f1_race_replay/     ← Race Engineer window, stand-in stream, take control
├── tests/                           ← pytest suite (stand-in models, synthetic races)
├── docs/                            ← images for the READMEs
├── app.py                           ← Streamlit dashboard
├── models/                          ← trained artefacts (gitignored, see models/README.md)
├── data/                            ← laps Parquet (gitignored)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 📜 License

MIT — see [LICENSE](LICENSE).

## ⚖️ Disclaimer

Educational and fan-made. Not affiliated with Formula 1, the FIA, or any team. Not for betting.

## 🔗 References

- [FastF1](https://github.com/theOehrly/Fast-F1) — the data layer this entire project depends on
- [f1-race-replay](https://github.com/IAmTomShaw/f1-race-replay) — the race replay and telemetry stream the Race Engineer listens to
- [OpenF1 API](https://openf1.org) — live timing data
