"""
Tool 4 — Qualifying → Race Pace Predictor.

Qualifying time is the cleanest pace signal we have (low fuel, new softs,
max attack). Race pace is a different animal. We bridge the two with three
corrections, all driven by Module 1:

    PredictedRacePace = QualyTime
                      + FUEL_CORRECTION               (same for everyone)
                      + CompoundDelta(team)           (SOFT → race-start compound)
                      + DegAverage(team)              (mean delta over first stint)

Honest caveat worth knowing (and saying in an interview): the fuel term is
a constant, so it never changes the ORDER. Only the team-level terms do, which
means two team-mates can never swap places — the model is "qualifying order
re-shuffled by team race-craft". That is a feature to build on, not hide.

Ported from notebooks/qualy_predictor.ipynb, vectorised (one Module 1 call
for the whole grid instead of ~460), and with an optional comparison to the
actual race result (Spearman rank correlation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tools.shared import (FUEL_CORRECTION, encode_compound, encode_team,
                          enable_fastf1_cache, predict_lap_deltas)

STINT_LAPS = 20
DEFAULT_TRACK_TEMP = 35.0
DEFAULT_AIR_TEMP = 28.0
CORRECTION_LAP = 5      # lap used for the compound comparison (fuel effect equal for both)

# Most common race-start compound per event (keys are FastF1 event names)
START_COMPOUNDS = {
    "Italian Grand Prix": "HARD",
}


def load_qualifying_results(year: int, gp: str, session=None) -> pd.DataFrame:
    """Best lap per driver across Q1–Q3 → Driver | Team | QualyTime | QualyRank."""
    if session is None:
        import fastf1
        enable_fastf1_cache()
        session = fastf1.get_session(year, gp, "Q")
        session.load(laps=True, telemetry=False, weather=False, messages=False)

    laps = session.laps
    laps = laps[laps["LapTime"].notna()]
    idx = laps.groupby("Driver")["LapTime"].idxmin()
    best = laps.loc[idx, ["Driver", "Team", "LapTime"]].copy()
    best["QualyTime"] = best["LapTime"].dt.total_seconds()
    best = best.drop(columns="LapTime").sort_values("QualyTime").reset_index(drop=True)
    best["QualyRank"] = best.index + 1
    best["GapToPole"] = best["QualyTime"] - best["QualyTime"].iloc[0]
    best.attrs["event"] = f"{session.event['EventName']} {year}"
    return best


def predict_race_pace(quali: pd.DataFrame, start_compound: str = "MEDIUM",
                      track_temp: float = DEFAULT_TRACK_TEMP,
                      air_temp: float = DEFAULT_AIR_TEMP,
                      stint_laps: int = STINT_LAPS) -> pd.DataFrame:
    """Apply the three corrections to a qualifying table (from load_qualifying_results)."""
    df = quali.copy()
    df["TeamEncoded"] = df["Team"].map(encode_team)
    teams = df["TeamEncoded"].unique()
    race_c, soft_c = encode_compound(start_compound), encode_compound("SOFT")

    # One frame with every row we need, for every team → one predict() call
    common = {"TrackTemp": track_temp, "AirTemp": air_temp, "RainfallEncoded": 0}
    comp_rows = pd.DataFrame([
        {"TeamEncoded": t, "CompoundEncoded": c, "TyreLife": 1, "LapNumber": CORRECTION_LAP,
         "Kind": kind, **common}
        for t in teams for c, kind in ((soft_c, "soft"), (race_c, "race"))])
    deg_rows = pd.DataFrame([
        {"TeamEncoded": t, "CompoundEncoded": race_c, "TyreLife": age, "LapNumber": age,
         "Kind": "deg", **common}
        for t in teams for age in range(1, stint_laps + 1)])
    rows = pd.concat([comp_rows, deg_rows], ignore_index=True)
    rows["Delta"] = predict_lap_deltas(rows)

    piv = rows[rows.Kind != "deg"].pivot_table(index="TeamEncoded", columns="Kind", values="Delta")
    compound_delta = (piv["race"] - piv["soft"]).clip(lower=0.0)   # a harder tyre is never faster
    deg_avg = rows[rows.Kind == "deg"].groupby("TeamEncoded")["Delta"].mean()

    df["FuelAdjTime"] = df["QualyTime"] + FUEL_CORRECTION
    df["CompoundDelta"] = df["TeamEncoded"].map(compound_delta)
    df["DegAverage"] = df["TeamEncoded"].map(deg_avg)
    df["PredictedRacePace"] = df["FuelAdjTime"] + df["CompoundDelta"] + df["DegAverage"]

    df = df.sort_values("PredictedRacePace").reset_index(drop=True)
    df["PredictedRaceRank"] = df.index + 1
    df["RankChange"] = df["QualyRank"] - df["PredictedRaceRank"]
    df["Direction"] = np.sign(df["RankChange"]).map({1: "↑", -1: "↓", 0: "→"})
    return df


def load_race_result(year: int, gp: str) -> pd.Series | None:
    """Actual finishing position per driver (None if the session can't be loaded)."""
    try:
        import fastf1
        enable_fastf1_cache()
        s = fastf1.get_session(year, gp, "R")
        s.load(laps=False, telemetry=False, weather=False, messages=False)
        res = s.results.set_index("Abbreviation")["Position"].dropna().astype(int)
        return res if len(res) else None
    except Exception:
        return None


def compare_with_actual(pred: pd.DataFrame, actual: pd.Series) -> dict:
    """
    Spearman rank correlation of qualy order vs actual and predicted vs actual.
    The model only earns its keep if it beats the naive 'qualifying order' baseline.
    """
    df = pred.copy()
    df["ActualRank"] = df["Driver"].map(actual)
    df = df.dropna(subset=["ActualRank"])
    rho = lambda a, b: float(pd.Series(a).rank().corr(pd.Series(b).rank()))
    return {
        "n": len(df),
        "spearman_qualy": rho(df["QualyRank"].values, df["ActualRank"].values),
        "spearman_model": rho(df["PredictedRaceRank"].values, df["ActualRank"].values),
        "table": df,
    }
