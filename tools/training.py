"""
Training pipelines for Module 1 (lap-time model) and Module 2 (overtake model), V3.

Why V3 exists — what the real-model test of the dashboard revealed
------------------------------------------------------------------
Module 1 V2 predicted that a NEW SOFT is ~1.5 s slower than a new HARD and
that tyres get FASTER with age. Physically backwards. Cause: confounding.
The target was `lap time − driver's best lap`, and the best lap comes late
in the race on an empty tank. Softs are mostly run in the first stint, on a
full tank, so "SOFT" and "slow because heavy" arrived together and the model
blamed the tyre. Same story for TyreLife inside a stint: the car gets lighter
every lap, which hides degradation.

V3 fixes, Module 1
    1. Fuel-corrected target: subtract FUEL_S_PER_LAP × laps of fuel left,
       and drop LapNumber, which is a proxy for the compound (see below).
    2. Monotonic constraint on TyreLife (older tyre can never be faster) —
       domain knowledge the model is not allowed to unlearn.
    3. Full FastF1 event names + an assert on the loaded event. 'Great
       Britain' was silently fuzzy-matched to the AUSTRIAN GP in 2023/2024.
    4. In/out laps removed explicitly (instead of the TyreLife > 1 proxy);
       rainfall per lap instead of "it rained at some point in the race".
    5. GroupKFold by race for evaluation. A random split puts laps of the
       same race in train and test, so the score is optimistic (leakage).
    6. Physics sanity gate: the model is only saved if fresh SOFT < MEDIUM
       < HARD and every compound slows down with age.

V3 fixes, Module 2
    1. Real gap between the cars (session clock at the line), not
       |lap-time difference|. That was a train/serve skew: the tools pass a
       real gap.
    2. Features measured at the END OF LAP t−1, label = "passed the car
       that was directly ahead during lap t". V2 used the same lap's
       position for features and label.
    3. Pit stops and Safety Car laps are not counted as overtakes.
    4. No scale_pos_weight. It rebalances classes for *classification*,
       but it inflates predict_proba — and the simulator and the undercut
       tool use the probabilities directly. We need calibrated probabilities.
    5. Grouped CV reporting log loss / Brier / PR-AUC vs the base rate.

Run everything with:
    python -m tools.train_models
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tools.shared import COMPOUND_MAP, FUEL_S_PER_LAP, LAP_FEATURES, OVT_FEATURES

# V3 drops LapNumber. Once fuel is corrected, LapNumber only carries track
# evolution (small) — but it is a near-perfect PROXY for the compound, because
# softs are run early and hards late. Trees split on LapNumber first and the
# compound effect gets absorbed by it. Removing the proxy forces the model to
# attribute pace differences to the tyre. (Tools still pass LapNumber; the
# prediction helpers use whatever features the saved model was trained on.)
LAP_MODEL_FEATURES = [f for f in LAP_FEATURES if f != "LapNumber"]

SEASONS = (2022, 2023, 2024, 2025)
EVENTS = (
    "British Grand Prix", "Monaco Grand Prix", "Italian Grand Prix",
    "Bahrain Grand Prix", "Belgian Grand Prix", "Singapore Grand Prix",
    "Abu Dhabi Grand Prix", "Japanese Grand Prix",
)
SESSIONS = [(y, e) for y in SEASONS for e in EVENTS]
DRY = list(COMPOUND_MAP)


# ── Loading ──────────────────────────────────────────────────────────────────
def load_race(year: int, event: str, session_type: str = "R") -> pd.DataFrame:
    """Race laps + per-lap weather, tagged with Season / Race / TotalLaps."""
    import fastf1
    s = fastf1.get_session(year, event, session_type)
    loaded = s.event["EventName"]
    if loaded != event:     # the bug that put Austria in the training set
        raise ValueError(f"Asked for '{event}' {year} but FastF1 resolved '{loaded}'")
    s.load(laps=True, telemetry=False, weather=True, messages=False)
    laps = pd.DataFrame(s.laps).copy()        # raises if timing data could not be loaded
    if laps.empty:
        raise ValueError("no laps in this session (data missing from cache and download refused?)")
    w = s.laps.get_weather_data().reset_index(drop=True)
    for col in ("TrackTemp", "AirTemp", "Rainfall"):
        laps[col] = w[col].values if col in w else np.nan
    laps["Season"], laps["Race"] = year, event
    laps["TotalLaps"] = int(laps["LapNumber"].max())
    return laps


class NoRacesLoaded(RuntimeError):
    pass


def load_races(sessions=SESSIONS, verbose: bool = True):
    frames, failed = [], []
    for year, event in sessions:
        try:
            frames.append(load_race(year, event))
            if verbose:
                print(f"  ✅ {year} {event:<22} {len(frames[-1]):>5} laps")
        except Exception as e:      # one broken session must not kill the run
            failed.append((year, event, str(e)))
            if verbose:
                print(f"  ❌ {year} {event:<22} {e}")
    if not frames:
        raise NoRacesLoaded(
            "No race could be loaded. FastF1 failed to download or read every session.\n"
            "  • Since ~15 Oct 2026 livetiming.formula1.com answers 401 to unauthenticated downloads\n"
            "    (github.com/theOehrly/Fast-F1/issues/978).\n"
            "  • Point $F1_CACHE_DIR at an existing FastF1 cache and add --offline, or retry later.")
    return pd.concat(frames, ignore_index=True), failed


def _seconds(s: pd.Series) -> pd.Series:
    return pd.to_timedelta(s).dt.total_seconds()


# ── Module 1 ─────────────────────────────────────────────────────────────────
def build_lap_dataset(laps: pd.DataFrame, fuel_s_per_lap: float = FUEL_S_PER_LAP,
                      max_delta: float = 5.0):
    """
    Clean green-flag racing laps → (features + LapDelta, fitted team encoder).
    LapDelta = fuel-corrected lap time − the driver's best fuel-corrected lap in that race.
    """
    from sklearn.preprocessing import LabelEncoder

    d = laps.copy()
    d["LapTimeSeconds"] = _seconds(d["LapTime"])
    keep = (
        d["LapTimeSeconds"].between(60, 150)
        & d["Compound"].isin(DRY)
        & (d["TrackStatus"].astype(str) == "1")          # green for the whole lap
        & d["PitInTime"].isna() & d["PitOutTime"].isna()  # no in/out laps
        & (d["LapNumber"] > 1)                            # lap 1 = standing start
        & d["TyreLife"].notna()
    )
    if "Deleted" in d:
        deleted = d["Deleted"].map(lambda v: str(v).strip().lower() == "true" if pd.notna(v) else False)
        keep &= ~deleted.astype(bool)
    d = d[keep].copy()

    fuel_laps_left = d["TotalLaps"] - d["LapNumber"]
    d["FuelCorrected"] = d["LapTimeSeconds"] - fuel_s_per_lap * fuel_laps_left
    best = d.groupby(["Season", "Race", "Driver"])["FuelCorrected"].transform("min")
    d["LapDelta"] = d["FuelCorrected"] - best
    d = d[d["LapDelta"] <= max_delta]                     # traffic, mistakes, damage

    enc = LabelEncoder().fit(d["Team"].astype(str))
    d["CompoundEncoded"] = d["Compound"].map(COMPOUND_MAP)
    d["TeamEncoded"] = enc.transform(d["Team"].astype(str))
    d["RainfallEncoded"] = d["Rainfall"].map(
        lambda v: str(v).strip().lower() == "true" if pd.notna(v) else False).astype(int)
    d["TrackTemp"] = d["TrackTemp"].fillna(d["TrackTemp"].median())
    d["AirTemp"] = d["AirTemp"].fillna(d["AirTemp"].median())
    d["Group"] = d["Season"].astype(str) + " " + d["Race"]
    cols = LAP_FEATURES + ["LapDelta", "Group", "Season", "Race", "Driver", "Team"]
    return d[cols].reset_index(drop=True), enc


def make_lap_model(**kw):
    from xgboost import XGBRegressor
    mono = tuple(1 if f == "TyreLife" else 0 for f in LAP_MODEL_FEATURES)
    params = dict(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
                  colsample_bytree=0.9, monotone_constraints=mono, random_state=42, n_jobs=-1)
    params.update(kw)
    return XGBRegressor(**params)


def evaluate_lap_model(df: pd.DataFrame, n_splits: int = 5, make=make_lap_model) -> dict:
    """GroupKFold by race (honest) vs a random split (leaky), MAE in seconds."""
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import GroupKFold, train_test_split

    X, y, g = df[LAP_MODEL_FEATURES], df["LapDelta"], df["Group"]
    folds = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, g)):
        m = make().fit(X.iloc[tr], y.iloc[tr])
        folds.append({"fold": k, "test_races": int(g.iloc[te].nunique()),
                      "MAE": mean_absolute_error(y.iloc[te], m.predict(X.iloc[te]))})
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    random_mae = mean_absolute_error(yte, make().fit(Xtr, ytr).predict(Xte))
    baseline = mean_absolute_error(y, np.full(len(y), y.median()))
    folds = pd.DataFrame(folds)
    return {"folds": folds, "grouped_MAE": float(folds["MAE"].mean()),
            "random_split_MAE": float(random_mae), "predict_median_MAE": float(baseline)}


def lap_model_sanity(model, teams_encoded, track_temp: float = 35.0, air_temp: float = 25.0,
                     lap_number: int = 30) -> dict:
    """
    Physics gate. Averaged over teams so one odd team can't pass/fail it alone.
      1. Fresh tyres (age 2): SOFT < MEDIUM < HARD
      2. Every compound is slower at age 20 than at age 2
    """
    rows = [{"CompoundEncoded": c, "TyreLife": a, "TeamEncoded": t, "LapNumber": lap_number,
             "TrackTemp": track_temp, "AirTemp": air_temp, "RainfallEncoded": 0, "Age": a, "C": c}
            for t in teams_encoded for c in range(3) for a in range(2, 31)]
    grid = pd.DataFrame(rows)
    grid["Delta"] = model.predict(grid[_features_of(model)].astype(float))
    curve = grid.groupby(["C", "Age"])["Delta"].mean().unstack(0)
    curve.columns = [k for k in COMPOUND_MAP]                    # SOFT, MEDIUM, HARD
    fresh = curve.loc[2]
    checks = {
        "fresh SOFT < MEDIUM": bool(fresh["SOFT"] < fresh["MEDIUM"]),
        "fresh MEDIUM < HARD": bool(fresh["MEDIUM"] < fresh["HARD"]),
        **{f"{c} slower at age 20 than age 2": bool(curve.loc[20, c] > curve.loc[2, c]) for c in DRY},
    }
    return {"passed": all(checks.values()), "checks": checks, "curve": curve}


def _features_of(model) -> list[str]:
    names = getattr(model, "feature_names_in_", None)
    return list(names) if names is not None else list(LAP_MODEL_FEATURES)


# ── Module 2 ─────────────────────────────────────────────────────────────────
def build_overtake_dataset(laps: pd.DataFrame, max_gap: float = 10.0) -> pd.DataFrame:
    """
    One row per (car, lap t): state at the end of lap t−1, label = passed the car
    that was directly ahead at the end of lap t−1, on track, during lap t.
    """
    d = laps.copy()
    d = d.dropna(subset=["Position", "Time", "LapNumber"])
    d["T"] = _seconds(d["Time"])                     # session clock when the car crossed the line
    d["Pitted"] = d["PitInTime"].notna() | d["PitOutTime"].notna()
    d["Green"] = d["TrackStatus"].astype(str) == "1"
    key = ["Season", "Race", "LapNumber"]

    prev = d.sort_values(key + ["Position"]).copy()   # lap t−1 standings
    g = prev.groupby(key)
    prev["AheadDriver"] = g["Driver"].shift(1)
    prev["GapAhead"] = prev["T"] - g["T"].shift(1)
    prev["TyreLifeAhead"] = g["TyreLife"].shift(1)
    prev["CompoundAhead"] = g["Compound"].shift(1)
    prev = prev.dropna(subset=["AheadDriver"])
    prev = prev[key + ["Driver", "Position", "TyreLife", "Compound", "AheadDriver",
                       "GapAhead", "TyreLifeAhead", "CompoundAhead"]]
    prev = prev.rename(columns={"Position": "PosPrev"})
    prev["LapNumber"] = prev["LapNumber"] + 1        # align t−1 state with lap t outcome

    cur = d[["Season", "Race", "LapNumber", "Driver", "Position", "Pitted", "Green"]]
    rows = prev.merge(cur, on=["Season", "Race", "LapNumber", "Driver"], how="inner")
    ahead = cur.rename(columns={"Driver": "AheadDriver", "Position": "AheadPos",
                                "Pitted": "AheadPitted", "Green": "AheadGreen"})
    rows = rows.merge(ahead, on=["Season", "Race", "LapNumber", "AheadDriver"], how="inner")

    ok = (rows["Green"] & rows["AheadGreen"] & ~rows["Pitted"] & ~rows["AheadPitted"]
          & rows["Compound"].isin(DRY) & rows["CompoundAhead"].isin(DRY)
          & rows["GapAhead"].between(0, max_gap))
    rows = rows[ok].copy()
    rows["Overtake"] = (rows["Position"] < rows["AheadPos"]).astype(int)

    rows["CompoundEncoded"] = rows["Compound"].map(COMPOUND_MAP)
    rows["CompoundAheadEncoded"] = rows["CompoundAhead"].map(COMPOUND_MAP)
    rows["TyreDelta"] = rows["TyreLife"] - rows["TyreLifeAhead"]     # behind − ahead (contract)
    rows["Position"] = rows["PosPrev"]                                 # position BEFORE the lap
    rows["Group"] = rows["Season"].astype(str) + " " + rows["Race"]
    rows = rows.dropna(subset=OVT_FEATURES)
    return rows[OVT_FEATURES + ["Overtake", "Group", "Driver", "AheadDriver"]].reset_index(drop=True)


def make_overtake_model(**kw):
    from xgboost import XGBClassifier
    params = dict(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
                  min_child_weight=5, eval_metric="logloss", random_state=42, n_jobs=-1)
    params.update(kw)
    return XGBClassifier(**params)


def evaluate_overtake_model(df: pd.DataFrame, n_splits: int = 5, make=make_overtake_model) -> dict:
    """Grouped CV on probability quality — the tools consume probabilities, not labels."""
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
    from sklearn.model_selection import GroupKFold

    X, y, g = df[OVT_FEATURES], df["Overtake"], df["Group"]
    oof = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, g):
        oof[te] = make().fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1]
    base = float(y.mean())
    bins = pd.cut(oof, [0, .02, .05, .1, .2, .4, 1], include_lowest=True)
    calib = pd.DataFrame({"pred": oof, "obs": y}).groupby(bins, observed=True).agg(
        n=("obs", "size"), mean_pred=("pred", "mean"), observed_rate=("obs", "mean"))
    return {"base_rate": base,
            "log_loss": float(log_loss(y, oof)),
            "log_loss_constant": float(log_loss(y, np.full(len(y), base))),
            "brier": float(brier_score_loss(y, oof)),
            "roc_auc": float(roc_auc_score(y, oof)),
            "pr_auc": float(average_precision_score(y, oof)),
            "calibration": calib}


def overtake_sanity(model) -> dict:
    """P(pass) must fall with the gap and rise when the attacker has much fresher tyres."""
    from tools.shared import build_overtake_features
    p = lambda gap, tl_b, tl_a: float(model.predict_proba(
        build_overtake_features(gap, 1, 1, tl_b, tl_a, 30, 8))[:, 1][0])
    close, far = p(0.4, 15, 15), p(3.0, 15, 15)
    fresher, equal = p(0.6, 3, 25), p(0.6, 14, 14)
    checks = {"closer gap → more likely": close > far,
              "fresher tyres → more likely": fresher > equal}
    return {"passed": all(checks.values()), "checks": checks,
            "examples": {"gap 0.4 s, equal tyres": close, "gap 3.0 s, equal tyres": far,
                         "gap 0.6 s, 22 laps fresher": fresher, "gap 0.6 s, equal tyres (age 14)": equal}}
