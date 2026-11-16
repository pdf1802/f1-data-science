# models/

Trained artefacts — gitignored. Generate them with:

```bash
python -m tools.train_models --laps data/laps_2022_2025.parquet
```

| File | What it is |
|------|------------|
| `lap_time_model.pkl` | Module 1 V3.2 (XGBoost regressor): LapDelta from compound, tyre age, circuit and the race's degradation rate |
| `overtaking_model.pkl` | Module 2 (XGBoost classifier): P(pass the car ahead this lap) |
| `team_encoder.pkl` | Team name → integer (older tools; not a model input since V3.1) |
| `circuits.json` | The CircuitEncoded vocabulary the models were trained with |
| `deg_tail.json` | Tyre-age caps (p90) and linear degradation slopes beyond them |
| `degradation.json` | Degradation prior: global mean, spread between circuits and between years, estimate per circuit and per race, fuel slopes |
| `compound_map.json` | SOFT 0, MEDIUM 1, HARD 2 |
| `training_report.json` | Data used, evaluation, physics reference, sanity checks |
| `backtest_report.json` | Output of `python -m tools.backtest` (`_loro`: leave one race out · `_stops`: one stop or two vs the field) |
| `backup_<timestamp>/` | Previous artefacts, saved before every overwrite |

Set `F1_MODEL_DIR` to load them from somewhere else. With no models here, the dashboard uses the copy in `demo/models/`. Pickles are tied to the xgboost version: retrain after upgrading it.
