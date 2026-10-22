"""
Test fixtures: stand-in models with the SAME feature contracts as the real
Module 1 / Module 2 artefacts, trained on synthetic but physically sensible
data. They let the test suite (and CI) run without the gitignored .pkl files.

They are NOT the real models — never use them to judge prediction quality.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

from tools import shared

TEAMS = ["Alfa Romeo", "AlphaTauri", "Alpine", "Aston Martin", "Ferrari",
         "Haas F1 Team", "Kick Sauber", "McLaren", "Mercedes", "RB",
         "Racing Bulls", "Red Bull Racing", "Williams"]


def _build_standin_models(out_dir, seed=0):
    rng = np.random.default_rng(seed)
    n = 6000
    comp = rng.integers(0, 3, n)
    tyre = rng.integers(1, 40, n)
    lap = np.clip(tyre + rng.integers(0, 30, n), 1, 70)
    team = rng.integers(0, len(TEAMS), n)
    offset = np.array([0.0, 0.35, 0.7])[comp]            # fresh-tyre pace offset
    deg = np.array([0.075, 0.045, 0.025])[comp] * tyre   # degradation per lap
    fuel = 0.035 * (70 - lap)                            # heavier early = slower
    y = offset + deg + fuel * 0.5 + 0.05 * (team % 4) + rng.normal(0, 0.1, n)
    X = pd.DataFrame({"CompoundEncoded": comp, "TyreLife": tyre, "TeamEncoded": team,
                      "LapNumber": lap, "TrackTemp": rng.uniform(20, 50, n),
                      "AirTemp": rng.uniform(15, 35, n), "RainfallEncoded": 0})[shared.LAP_FEATURES]
    lap_model = XGBRegressor(n_estimators=60, max_depth=4).fit(X, y)

    gap = rng.uniform(0, 3, n)
    tl_b, tl_a = rng.integers(1, 40, n), rng.integers(1, 40, n)
    logit = 1.0 - 2.0 * gap - 0.08 * (tl_b - tl_a)
    lbl = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    Xo = shared.build_overtake_features(gap, rng.integers(0, 3, n), rng.integers(0, 3, n),
                                        tl_b, tl_a, rng.integers(1, 60, n), rng.integers(2, 20, n))
    ovt_model = XGBClassifier(n_estimators=60, max_depth=3).fit(Xo, lbl)

    enc = LabelEncoder().fit(TEAMS)
    joblib.dump(lap_model, os.path.join(out_dir, "lap_time_model.pkl"))
    joblib.dump(ovt_model, os.path.join(out_dir, "overtaking_model.pkl"))
    joblib.dump(enc, os.path.join(out_dir, "team_encoder.pkl"))
    with open(os.path.join(out_dir, "compound_map.json"), "w") as f:
        json.dump(shared.COMPOUND_MAP, f)


@pytest.fixture(scope="session", autouse=True)
def standin_models(tmp_path_factory):
    if os.environ.get("F1_USE_REAL_MODELS") == "1":   # opt-in: test with your .pkl files
        yield None
        return
    out = tmp_path_factory.mktemp("models")
    _build_standin_models(str(out))
    old = os.environ.get("F1_MODEL_DIR")
    os.environ["F1_MODEL_DIR"] = str(out)
    shared.reset_models()
    yield str(out)
    shared.reset_models()
    if old is None:
        os.environ.pop("F1_MODEL_DIR", None)
    else:
        os.environ["F1_MODEL_DIR"] = old
