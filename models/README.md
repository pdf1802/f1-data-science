# models/

Trained artefacts — gitignored. Generate them with:

```bash
python -m tools.train_models --laps data/laps_2022_2025.parquet
```

| File | What it is |
|------|------------|
| `lap_time_model.pkl` | Module 1 (XGBoost regressor): LapDelta from compound, tyre age, circuit |
| `overtaking_model.pkl` | Module 2 (XGBoost classifier): P(pass the car ahead this lap) |
| `team_encoder.pkl` | Team name → integer (older tools; not a V3.1 model input) |
| `circuits.json` | The CircuitEncoded vocabulary the models were trained with |
| `deg_tail.json` | Tyre-age caps (p90) and linear degradation slopes beyond them |
| `compound_map.json` | SOFT 0, MEDIUM 1, HARD 2 |
| `training_report.json` | Data used, evaluation, physics reference, sanity checks |
| `backtest_report.json` | Output of `python -m tools.backtest` |
| `backup_<timestamp>/` | Previous artefacts, saved before every overwrite |

Set `F1_MODEL_DIR` to load them from somewhere else. Pickles are tied to the xgboost version: retrain after upgrading it.
