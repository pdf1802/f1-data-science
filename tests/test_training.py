"""
Tests for the V3 training pipelines on synthetic FastF1-shaped laps.

The synthetic race has KNOWN physics (fuel effect, compound offsets, degradation),
so we can check the pipeline recovers it — something real data can't tell us.
"""
import numpy as np
import pandas as pd
import pytest

from tools import shared, training

FUEL = 0.055
OFFSET = {"SOFT": 0.0, "MEDIUM": 0.4, "HARD": 0.8}       # fresh-tyre pace
DEG = {"SOFT": 0.09, "MEDIUM": 0.05, "HARD": 0.03}        # s per lap of age
TEAMS = ["Red Bull Racing", "Ferrari", "Mercedes", "McLaren", "Aston Martin",
         "Alpine", "Williams", "RB", "Kick Sauber", "Haas F1 Team"]


def synthetic_race(season=2024, race="Test Grand Prix", n_laps=50, seed=0):
    """20 cars, mostly SOFT/MEDIUM → HARD one-stoppers, like a real dry race."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(20):
        team = TEAMS[i // 2]
        first = ["SOFT", "MEDIUM"][i % 2] if i < 16 else "HARD"
        second = "HARD" if first != "HARD" else "MEDIUM"
        pit = int(rng.integers(14, 30)) if first == "SOFT" else int(rng.integers(20, 34))
        t, age, comp, stint = 0.0, 1, first, 1
        pace = 90 + 0.08 * (i // 2)
        for lap in range(1, n_laps + 1):
            lt = (pace + OFFSET[comp] + DEG[comp] * age + FUEL * (n_laps - lap)
                  + rng.normal(0, 0.15) + (3.0 if lap == 1 else 0))
            pit_in = lap == pit
            pit_out = lap == pit + 1
            lt += 20 if pit_in else 0
            t += lt
            rows.append({"Driver": f"D{i:02d}", "Team": team, "LapNumber": lap,
                         "LapTime": pd.Timedelta(seconds=lt), "Time": pd.Timedelta(seconds=3600 + t),
                         "Compound": comp, "TyreLife": float(age), "Stint": stint,
                         "PitInTime": pd.Timedelta(seconds=3600 + t) if pit_in else pd.NaT,
                         "PitOutTime": pd.Timedelta(seconds=3600 + t - lt) if pit_out else pd.NaT,
                         "TrackStatus": "1", "Deleted": False,
                         "TrackTemp": 35.0, "AirTemp": 25.0, "Rainfall": False})
            if pit_in:
                comp, age, stint = second, 1, 2
            else:
                age += 1
    df = pd.DataFrame(rows)
    df["Position"] = df.groupby("LapNumber")["Time"].rank(method="first").astype(float)
    df["Season"], df["Race"], df["TotalLaps"] = season, race, n_laps
    return df


@pytest.fixture(scope="module")
def laps():
    return pd.concat([synthetic_race(2023 + k % 3, f"Race {k}", seed=k) for k in range(6)],
                     ignore_index=True)


def test_lap_dataset_drops_in_out_and_lap1(laps):
    df, enc = training.build_lap_dataset(laps, fuel_s_per_lap=FUEL)
    assert (df["LapNumber"] > 1).all()
    assert set(df.columns) >= set(shared.LAP_FEATURES + ["LapDelta", "Group"])
    assert df["LapDelta"].between(0, 5).all()
    assert len(enc.classes_) == len(TEAMS)


def test_fuel_correction_recovers_tyre_physics(laps):
    """With the right fuel correction the model must learn SOFT < MEDIUM < HARD and degradation."""
    df, _ = training.build_lap_dataset(laps, fuel_s_per_lap=FUEL)
    model = training.make_lap_model(n_estimators=200).fit(df[training.LAP_MODEL_FEATURES], df["LapDelta"])
    s = training.lap_model_sanity(model, np.unique(df["TeamEncoded"]))
    assert s["passed"], s["checks"]


def test_without_fuel_correction_the_confound_appears(laps):
    """
    Reproduces the V2 bug on data where we KNOW the truth: with no fuel correction,
    early-race (heavy) laps make SOFT look slow and hide degradation → gate fails.
    """
    df, _ = training.build_lap_dataset(laps, fuel_s_per_lap=0.0)
    model = training.make_lap_model(n_estimators=200).fit(df[training.LAP_MODEL_FEATURES], df["LapDelta"])
    assert not training.lap_model_sanity(model, np.unique(df["TeamEncoded"]))["passed"]


def test_monotone_tyre_life(laps):
    df, _ = training.build_lap_dataset(laps, fuel_s_per_lap=FUEL)
    model = training.make_lap_model(n_estimators=100).fit(df[training.LAP_MODEL_FEATURES], df["LapDelta"])
    curve = training.lap_model_sanity(model, [0])["curve"]
    assert (curve.diff().dropna() >= -1e-9).all().all()     # never faster with age


def test_lap_model_grouped_eval(laps):
    df, _ = training.build_lap_dataset(laps, fuel_s_per_lap=FUEL)
    ev = training.evaluate_lap_model(df, n_splits=3,
                                     make=lambda: training.make_lap_model(n_estimators=50))
    assert ev["grouped_MAE"] < ev["predict_median_MAE"]
    assert len(ev["folds"]) == 3


# ── Module 2 ─────────────────────────────────────────────────────────────────
def _mini(rows):
    df = pd.DataFrame(rows, columns=["Driver", "LapNumber", "Position", "T"])
    df["Time"] = pd.to_timedelta(df.pop("T"), unit="s")
    df["Compound"], df["TyreLife"], df["TrackStatus"] = "MEDIUM", 10.0, "1"
    df["PitInTime"] = df["PitOutTime"] = pd.NaT
    df["Season"], df["Race"] = 2024, "Mini"
    return df


def test_overtake_label_uses_previous_lap_state():
    # Lap 1: A leads B by 0.6 s. Lap 2: B passes A.
    laps = _mini([("A", 1, 1, 100.0), ("B", 1, 2, 100.6),
                  ("A", 2, 2, 191.0), ("B", 2, 1, 190.8)])
    d = training.build_overtake_dataset(laps)
    assert len(d) == 1
    r = d.iloc[0]
    assert r["Driver"] == "B" and r["AheadDriver"] == "A"
    assert r["GapAhead"] == pytest.approx(0.6)        # real gap at the end of lap 1
    assert r["Position"] == 2 and r["LapNumber"] == 2  # position BEFORE the pass
    assert r["Overtake"] == 1


def test_pit_stop_is_not_an_overtake():
    laps = _mini([("A", 1, 1, 100.0), ("B", 1, 2, 100.6),
                  ("A", 2, 2, 215.0), ("B", 2, 1, 190.8)])
    laps.loc[(laps.Driver == "A") & (laps.LapNumber == 2), "PitInTime"] = pd.Timedelta(seconds=215)
    assert len(training.build_overtake_dataset(laps)) == 0


def test_overtake_dataset_on_synthetic_race(laps):
    d = training.build_overtake_dataset(laps)
    assert list(d.columns[:len(shared.OVT_FEATURES)]) == shared.OVT_FEATURES
    assert (d["GapAhead"] >= 0).all()
    assert d["Overtake"].isin([0, 1]).all()


def test_train_models_cli_end_to_end(laps, tmp_path, monkeypatch):
    """Full CLI run on synthetic races; then the dashboard tools must work on the V3 artefacts."""
    import json
    from tools import train_models
    from tools.stint_optimizer import optimize_pit_lap
    from tools.undercut import DriverState, undercut_analysis

    monkeypatch.setattr(training, "load_races", lambda sessions, verbose=True: (laps, []))
    monkeypatch.setattr(shared, "enable_fastf1_cache", lambda *a, **k: None)
    out = tmp_path / "models"
    assert train_models.main(["--out", str(out), "--folds", "3"]) == 0
    report = json.loads((out / "training_report.json").read_text())
    assert report["version"] == "V3" and report["module1"]["sanity"]

    monkeypatch.setenv("F1_MODEL_DIR", str(out))
    shared.reset_models()
    try:
        assert shared.model_features(shared.load_models()[0], shared.LAP_FEATURES) == training.LAP_MODEL_FEATURES
        team = report["teams"][0]
        r = optimize_pit_lap("MEDIUM", "HARD", 50, team, {"track_temp": 35, "air_temp": 25})
        assert 5 <= r["optimal_lap"] <= 45
        a, b = DriverState("A", team, 3, "HARD", 20), DriverState("B", team, 2, "HARD", 20)
        assert undercut_analysis(a, b, 1.0, 20, {"track_temp": 35, "air_temp": 25})["verdict"] in ("BOX BOX", "STAY OUT")
        from tools.race_simulator import simulate
        grid = pd.DataFrame({"Driver": ["A", "B", "C"], "Team": report["teams"][:3], "Grid": [1, 2, 3],
                             "PaceFactor": [1.0, 1.002, 1.004], "StartCompound": "MEDIUM", "StartAge": 1,
                             "Stop1Lap": 20, "Stop1Compound": "HARD", "Stop2Lap": 0, "Stop2Compound": "HARD"})
        assert simulate(grid, {"total_laps": 40}, n_sims=50)["summary"]["Win %"].sum() == pytest.approx(100)
        # second run → old models are backed up, not lost
        assert train_models.main(["--out", str(out), "--folds", "3"]) == 0
        assert any(p.name.startswith("backup_") for p in out.iterdir())
    finally:
        shared.reset_models()


def test_clear_error_when_no_race_loads(monkeypatch):
    monkeypatch.setattr(training, "load_race", lambda *a, **k: (_ for _ in ()).throw(ValueError("401")))
    with pytest.raises(training.NoRacesLoaded, match="401"):
        training.load_races([(2024, "British Grand Prix")], verbose=False)


def test_boolean_columns_survive_parquet(tmp_path):
    """Deleted/Rainfall as text 'False' must not be read as True (a non-empty string is truthy)."""
    laps = synthetic_race(n_laps=30)
    laps["Deleted"] = pd.Series([False, None, True] * (len(laps) // 3), dtype=object)
    expected = len(training.build_lap_dataset(laps)[0])
    as_text = laps.copy()
    for col in ("Deleted", "Rainfall"):
        as_text[col] = as_text[col].astype("string")      # worst case: booleans stored as text
    assert len(training.build_lap_dataset(as_text)[0]) == expected
    assert training.build_lap_dataset(as_text)[0]["RainfallEncoded"].eq(0).all()


def test_export_finds_only_race_sessions_2022_2025(tmp_path):
    from tools.export_cache import races_in_cache
    for p in ["2023/2023-03-05_Bahrain_Grand_Prix/2023-03-05_Race",
              "2024/2024-07-07_British_Grand_Prix/2024-07-06_Qualifying",
              "2024/2024-02-21_Pre-Season_Testing/2024-02-21_Practice_1",
              "2026/2026-03-08_Australian_Grand_Prix/2026-03-08_Race"]:
        (tmp_path / p).mkdir(parents=True)
    assert races_in_cache(str(tmp_path)) == [(2023, "Bahrain Grand Prix")]
