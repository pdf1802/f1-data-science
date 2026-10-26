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
    6. Physics sanity gate (see lap_model_sanity) before anything is saved.

V3.1 — after the first real training run (Oct 2026)
    The first V3 model fixed degradation but still lost to "predict the
    median" on unseen races, because TrackTemp/AirTemp are almost constant
    within a race and acted as a race FINGERPRINT (memorised, useless on a new
    race). And fresh SOFT still looked slow because softs run in the pack at
    the start (traffic). Experiments on 43 cached races (scratch, documented
    in the Notion page) led to:
    7. Dry races only (< 5 % laps on wet tyres and < 5 % with rain).
    8. Clean-air laps only: > CLEAN_AIR_GAP s to the car ahead at the start
       of the lap (or leading). Traffic is the second confound.
    9. Features: compound, tyre age, CIRCUIT. Temperatures and team removed —
       both made the physics worse and the out-of-race error larger.
   10. Shallow, regularised trees (depth 4, min_child_weight 20).
   11. A transparent fixed-effects regression (estimate_tyre_physics) is fitted
       alongside: it measured fuel + track evolution at −0.050 ± 0.008 s/lap
       (the 0.055 assumption holds) and showed that fresh MEDIUM vs HARD cannot
       be separated with this data (±0.28 s) — "HARD" is a different Pirelli
       compound at different races. The gate therefore checks what the data
       CAN support (below), not a ranking it cannot verify.

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
    6. (V3.1) Circuit as a feature: overtaking is ~10× harder at Monaco than
       at Spa. Log loss 0.095 → 0.090, PR-AUC 0.52 → 0.57 on unseen races.

Run everything with:
    python -m tools.train_models
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tools.shared import (COMPOUND_MAP, FUEL_S_PER_LAP, LAP_FEATURES, OVT_FEATURES,
                          circuit_name)

# LapNumber is out: once fuel is corrected it only carries track evolution
# (small) and it is a near-perfect PROXY for the compound (softs early, hards
# late). TeamEncoded / temperatures are out too (V3.1, see above). The tools
# keep passing every LAP_FEATURES column; predictions use the model's own list.
LAP_MODEL_FEATURES = ["CompoundEncoded", "TyreLife", "CircuitEncoded"]
OVT_MODEL_FEATURES = OVT_FEATURES + ["CircuitEncoded"]

CLEAN_AIR_GAP = 2.0          # s to the car ahead at the start of the lap
WET_MAX_SHARE = 0.05         # a race is "wet" above 5 % wet-tyre laps or 5 % rain laps
HARD_VS_MEDIUM_TOL = 0.15    # s — HARD may not look faster than MEDIUM by more than this

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


# ── Data selection ───────────────────────────────────────────────────────────
def _truthy(v) -> bool:
    """True/False that also survives text round-trips ("False" is a truthy string)."""
    return str(v).strip().lower() == "true" if pd.notna(v) else False


def race_wetness(laps: pd.DataFrame) -> pd.DataFrame:
    g = laps.groupby(["Season", "Race"])
    return pd.DataFrame({
        "wet_tyre_share": g["Compound"].apply(lambda s: s.astype(str).isin(["INTERMEDIATE", "WET"]).mean()),
        "rain_share": g["Rainfall"].apply(lambda s: s.map(_truthy).mean()),
    })


def dry_races(laps: pd.DataFrame, max_share: float = WET_MAX_SHARE):
    """(laps of dry races only, list of excluded (season, race))."""
    w = race_wetness(laps)
    wet = w[(w["wet_tyre_share"] >= max_share) | (w["rain_share"] >= max_share)].index
    idx = pd.MultiIndex.from_frame(laps[["Season", "Race"]])
    return laps[~idx.isin(wet)].copy(), [tuple(x) for x in wet]


def circuit_list(laps: pd.DataFrame) -> list[str]:
    """Sorted circuit names ('Bahrain', 'Monaco', …) — the CircuitEncoded vocabulary."""
    return sorted({circuit_name(r) for r in laps["Race"].astype(str).unique()})


def _encode_circuits(race: pd.Series, circuits: list[str]) -> pd.Series:
    lookup = {c: float(i) for i, c in enumerate(circuits)}
    return race.astype(str).map(lambda r: lookup.get(circuit_name(r), np.nan))


def add_gap_ahead(laps: pd.DataFrame) -> pd.DataFrame:
    """
    GapAheadStart: gap to the car directly ahead on the road when the lap STARTED
    (session clock at the line on the previous lap). PosStart: position then.
    """
    d = laps.copy()
    d["_T"] = _seconds(d["Time"])
    d["_order"] = np.arange(len(d))
    s = d.sort_values(["Season", "Race", "LapNumber", "Position"])
    s["_gap_end"] = s.groupby(["Season", "Race", "LapNumber"])["_T"].diff()
    s = s.sort_values(["Season", "Race", "Driver", "LapNumber"])
    grp = s.groupby(["Season", "Race", "Driver"])
    s["GapAheadStart"] = grp["_gap_end"].shift(1)
    s["PosStart"] = grp["Position"].shift(1)
    return s.sort_values("_order").drop(columns=["_T", "_order", "_gap_end"])


# ── Module 1 ─────────────────────────────────────────────────────────────────
def build_lap_dataset(laps: pd.DataFrame, fuel_s_per_lap: float = FUEL_S_PER_LAP,
                      max_delta: float = 5.0, clean_air_gap: float | None = CLEAN_AIR_GAP,
                      circuits: list[str] | None = None):
    """
    Clean green-flag racing laps → (features + LapDelta, fitted team encoder).
    LapDelta = fuel-corrected lap time − the driver's best fuel-corrected lap in that race.
    df.attrs["circuits"] holds the circuit vocabulary used for CircuitEncoded.
    """
    from sklearn.preprocessing import LabelEncoder

    d = add_gap_ahead(laps) if clean_air_gap is not None else laps.copy()
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
        keep &= ~d["Deleted"].map(_truthy).astype(bool)
    if clean_air_gap is not None:                         # traffic is a confound too
        clean = (d["PosStart"] == 1) | (d["GapAheadStart"] > clean_air_gap)
        keep &= clean.fillna(False).astype(bool)
    d = d[keep].copy()

    fuel_laps_left = d["TotalLaps"] - d["LapNumber"]
    d["FuelCorrected"] = d["LapTimeSeconds"] - fuel_s_per_lap * fuel_laps_left
    best = d.groupby(["Season", "Race", "Driver"])["FuelCorrected"].transform("min")
    d["LapDelta"] = d["FuelCorrected"] - best
    d = d[d["LapDelta"] <= max_delta]                     # mistakes, damage, blue flags

    circuits = circuits or circuit_list(d)
    enc = LabelEncoder().fit(d["Team"].astype(str))
    d["CompoundEncoded"] = d["Compound"].map(COMPOUND_MAP)
    d["TeamEncoded"] = enc.transform(d["Team"].astype(str))
    d["CircuitEncoded"] = _encode_circuits(d["Race"], circuits)
    d["RainfallEncoded"] = d["Rainfall"].map(_truthy).astype(int)
    d["TrackTemp"] = d["TrackTemp"].fillna(d["TrackTemp"].median())
    d["AirTemp"] = d["AirTemp"].fillna(d["AirTemp"].median())
    d["Group"] = d["Season"].astype(str) + " " + d["Race"].astype(str)
    cols = LAP_FEATURES + ["LapDelta", "Group", "Season", "Race", "Driver", "Team"]
    out = d[cols].reset_index(drop=True)
    out.attrs["circuits"] = list(circuits)
    return out, enc


def make_lap_model(**kw):
    from xgboost import XGBRegressor
    feats = kw.pop("features", LAP_MODEL_FEATURES)
    mono = tuple(1 if f == "TyreLife" else 0 for f in feats)
    params = dict(n_estimators=300, learning_rate=0.05, max_depth=4, min_child_weight=20,
                  subsample=0.8, colsample_bytree=0.9, monotone_constraints=mono,
                  random_state=42, n_jobs=-1)
    params.update(kw)
    return XGBRegressor(**params)


def evaluate_lap_model(df: pd.DataFrame, n_splits: int = 5, make=make_lap_model,
                       features=LAP_MODEL_FEATURES) -> dict:
    """
    GroupKFold by race (honest) vs a random split (leaky), MAE in seconds.
    The baseline is "predict the median" fitted on the TRAINING folds only.
    """
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import GroupKFold, train_test_split

    X, y, g = df[features].astype(float), df["LapDelta"], df["Group"]
    pred, base, folds = np.zeros(len(df)), np.zeros(len(df)), []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, g)):
        pred[te] = make().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
        base[te] = y.iloc[tr].median()
        folds.append({"fold": k, "test_races": int(g.iloc[te].nunique()),
                      "MAE": mean_absolute_error(y.iloc[te], pred[te]),
                      "median_MAE": mean_absolute_error(y.iloc[te], base[te])})
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    random_mae = mean_absolute_error(yte, make().fit(Xtr, ytr).predict(Xte))
    return {"folds": pd.DataFrame(folds), "grouped_MAE": float(mean_absolute_error(y, pred)),
            "random_split_MAE": float(random_mae),
            "predict_median_MAE": float(mean_absolute_error(y, base))}


def _features_of(model, default) -> list[str]:
    names = getattr(model, "feature_names_in_", None)
    return list(names) if names is not None else list(default)


def _marginal_predict(model, grid: pd.DataFrame, feats, n_circuits, predict):
    """Average predictions over circuits (and whatever else varies in `grid`)."""
    if "CircuitEncoded" in feats:
        grid = pd.concat([grid.assign(CircuitEncoded=float(c)) for c in range(max(n_circuits, 1))],
                         ignore_index=True)
    return grid.assign(P=predict(grid[feats].astype(float)))


def lap_model_sanity(model, teams_encoded=(0,), n_circuits: int | None = None,
                     track_temp: float = 35.0, air_temp: float = 25.0, lap_number: int = 30) -> dict:
    """
    Physics gate, averaged over teams and circuits so one odd value can't pass/fail it.
    What the data supports (see estimate_tyre_physics):
      1. a fresh SOFT is faster than a fresh MEDIUM and a fresh HARD
      2. every compound is slower at age 20 than at age 2
      3. the SOFT degrades the most
      4. a fresh HARD is not more than HARD_VS_MEDIUM_TOL s faster than a fresh MEDIUM
         (the data cannot rank MEDIUM vs HARD when fresh — but it can rule out
         a clearly faster HARD, which is what V2 claimed)
    """
    feats = _features_of(model, LAP_MODEL_FEATURES)
    grid = pd.DataFrame([{"CompoundEncoded": c, "TyreLife": a, "TeamEncoded": t, "LapNumber": lap_number,
                          "TrackTemp": track_temp, "AirTemp": air_temp, "RainfallEncoded": 0,
                          "CircuitEncoded": 0.0, "Age": a, "C": c}
                         for t in teams_encoded for c in range(3) for a in range(2, 31)])
    grid = _marginal_predict(model, grid, feats, n_circuits or 1, model.predict)
    curve = grid.groupby(["C", "Age"])["P"].mean().unstack(0)
    curve.columns = list(COMPOUND_MAP)                               # SOFT, MEDIUM, HARD
    fresh, deg = curve.loc[2], curve.loc[20] - curve.loc[2]
    checks = {
        "fresh SOFT fastest": bool(fresh["SOFT"] < fresh["MEDIUM"] and fresh["SOFT"] < fresh["HARD"]),
        **{f"{c} slower at age 20 than age 2": bool(deg[c] > 0) for c in DRY},
        "SOFT degrades the most": bool(deg["SOFT"] > deg["MEDIUM"] and deg["SOFT"] > deg["HARD"]),
        f"fresh HARD not >{HARD_VS_MEDIUM_TOL} s faster than MEDIUM":
            bool(fresh["HARD"] > fresh["MEDIUM"] - HARD_VS_MEDIUM_TOL),
    }
    return {"passed": all(checks.values()), "checks": checks, "curve": curve}


def estimate_tyre_physics(laps: pd.DataFrame, clean_air_gap: float = CLEAN_AIR_GAP,
                          max_tyre_age: int = 35) -> pd.DataFrame:
    """
    Transparent reference model, fitted on raw lap times (no fuel correction):

        LapTime = α[driver, race] + φ·LapNumber + β[compound] + γ[compound]·TyreLife + ε

    α absorbs car, driver and circuit. φ = fuel burn + track evolution per lap
    (identified because tyre age resets at a stop while LapNumber keeps going —
    stints on the same compound make it clean). β = fresh offset vs SOFT.
    γ = degradation per lap. 95 % CIs are clustered by race (laps of one race
    are not independent).
    """
    d = add_gap_ahead(laps)
    d["LT"] = _seconds(d["LapTime"])
    clean = ((d["PosStart"] == 1) | (d["GapAheadStart"] > clean_air_gap)).fillna(False).astype(bool)
    d = d[clean & d["Compound"].isin(DRY) & (d["TrackStatus"].astype(str) == "1")
          & d["PitInTime"].isna() & d["PitOutTime"].isna() & (d["LapNumber"] > 1)
          & d["TyreLife"].between(2, max_tyre_age) & d["LT"].between(60, 150)].copy()
    d["key"] = d["Season"].astype(str) + d["Race"].astype(str) + d["Driver"].astype(str)
    d = d[(d["LT"] - d.groupby("key")["LT"].transform("median")).abs() < 3]
    comp = d["Compound"].astype(str)
    X = pd.DataFrame({
        "fuel_and_evolution_per_lap": d["LapNumber"],
        "fresh_MEDIUM_vs_SOFT": (comp == "MEDIUM"),
        "fresh_HARD_vs_SOFT": (comp == "HARD"),
        "deg_SOFT_per_lap": d["TyreLife"] * (comp == "SOFT"),
        "deg_MEDIUM_per_lap": d["TyreLife"] * (comp == "MEDIUM"),
        "deg_HARD_per_lap": d["TyreLife"] * (comp == "HARD"),
    }, index=d.index).astype(float)
    y = d["LT"].astype(float)
    Xd = (X - X.groupby(d["key"]).transform("mean")).to_numpy()
    yd = (y - y.groupby(d["key"]).transform("mean")).to_numpy()
    beta, *_ = np.linalg.lstsq(Xd, yd, rcond=None)
    res = yd - Xd @ beta
    race = (d["Season"].astype(str) + d["Race"].astype(str)).to_numpy()
    bread = np.linalg.inv(Xd.T @ Xd)
    meat = sum(np.outer(Xd[race == r].T @ res[race == r], Xd[race == r].T @ res[race == r])
               for r in np.unique(race))
    se = np.sqrt(np.diag(bread @ meat @ bread))
    out = pd.DataFrame({"estimate": beta, "ci95": 1.96 * se}, index=X.columns)
    out.loc["fuel_and_evolution_per_lap", "estimate"] *= -1   # report as "seconds gained per lap"
    out.attrs.update(laps=len(d), races=int(len(np.unique(race))))
    return out


# ── Module 2 ─────────────────────────────────────────────────────────────────
def build_overtake_dataset(laps: pd.DataFrame, max_gap: float = 10.0,
                           circuits: list[str] | None = None) -> pd.DataFrame:
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
    rows["CircuitEncoded"] = _encode_circuits(rows["Race"], circuits or circuit_list(rows))
    rows["Group"] = rows["Season"].astype(str) + " " + rows["Race"].astype(str)
    rows = rows.dropna(subset=OVT_FEATURES)
    return rows[OVT_MODEL_FEATURES + ["Overtake", "Group", "Driver", "AheadDriver"]].reset_index(drop=True)


def make_overtake_model(**kw):
    from xgboost import XGBClassifier
    params = dict(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
                  min_child_weight=5, eval_metric="logloss", random_state=42, n_jobs=-1)
    params.update(kw)
    return XGBClassifier(**params)


def evaluate_overtake_model(df: pd.DataFrame, n_splits: int = 5, make=make_overtake_model,
                            features=OVT_MODEL_FEATURES) -> dict:
    """Grouped CV on probability quality — the tools consume probabilities, not labels."""
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
    from sklearn.model_selection import GroupKFold

    X, y, g = df[features], df["Overtake"], df["Group"]
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


def overtake_sanity(model, circuits: list[str] | None = None) -> dict:
    """
    P(pass) must fall with the gap and rise with much fresher tyres (averaged over
    circuits) — and, when both are known, Monaco must be harder than Bahrain.
    """
    from tools.shared import build_overtake_features
    feats = _features_of(model, OVT_MODEL_FEATURES)
    n_c = len(circuits) if circuits else 1

    def p(gap, tl_b, tl_a, circuit=None):
        X = build_overtake_features(gap, 1, 1, tl_b, tl_a, 30, 8)
        if circuit is not None and "CircuitEncoded" in feats:
            X["CircuitEncoded"] = float(circuit)
            return float(model.predict_proba(X[feats])[:, 1][0])
        X["CircuitEncoded"] = 0.0
        return float(_marginal_predict(model, X, feats, n_c,
                                       lambda Z: model.predict_proba(Z)[:, 1])["P"].mean())

    close, far = p(0.4, 15, 15), p(3.0, 15, 15)
    fresher, equal = p(0.6, 3, 25), p(0.6, 14, 14)
    checks = {"closer gap → more likely": close > far,
              "fresher tyres → more likely": fresher > equal}
    examples = {"gap 0.4 s, equal tyres": close, "gap 3.0 s, equal tyres": far,
                "gap 0.6 s, 22 laps fresher": fresher, "gap 0.6 s, equal tyres (age 14)": equal}
    if circuits and "Monaco" in circuits and "Bahrain" in circuits and "CircuitEncoded" in feats:
        mon, bah = p(0.6, 3, 25, circuits.index("Monaco")), p(0.6, 3, 25, circuits.index("Bahrain"))
        checks["Monaco harder to pass than Bahrain"] = mon < bah
        examples.update({"Monaco, 0.6 s, 22 laps fresher": mon, "Bahrain, 0.6 s, 22 laps fresher": bah})
    return {"passed": all(checks.values()), "checks": checks, "examples": examples}
