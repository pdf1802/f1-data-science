"""
Tool 2 — Stint Optimizer.

Sweeps every candidate pit lap for a 1-stop race and returns the total
accumulated LapDelta (+ pit loss) for each, i.e. the "U-curve".

Deterministic on purpose: no noise. Noise answers "what will happen?";
this tool answers "what is optimal in expectation?".

Vectorised: all (pit_lap, race_lap) rows for all pit laps go into ONE
Module 1 call (~2,700 rows) instead of ~2,700 single-row calls.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np
import pandas as pd

from tools.shared import (COMPOUND_MAP, PIT_COST_GREEN, encode_compound,
                          encode_team, predict_lap_deltas)


def _stint_rows(compound: int, start_lap: int, n_laps: int, team_encoded: int,
                config: dict, starting_tyre_age: int = 1) -> pd.DataFrame:
    laps = np.arange(start_lap, start_lap + n_laps)
    return pd.DataFrame({
        "CompoundEncoded": compound,
        "TyreLife": np.arange(starting_tyre_age, starting_tyre_age + n_laps),
        "TeamEncoded": team_encoded,
        "LapNumber": laps,          # real race lap → model sees the fuel load
        "TrackTemp": config["track_temp"],
        "AirTemp": config["air_temp"],
        "RainfallEncoded": config.get("rainfall", 0),
    })


def optimize_pit_lap(compound_1, compound_2, total_laps: int, team: str | int,
                     config: dict, pit_cost: float = PIT_COST_GREEN,
                     earliest_pit: int = 5, latest_pit: int | None = None) -> dict:
    """
    Returns dict with optimal_lap, optimal_cost and a per-pit-lap DataFrame
    (`curve`: PitLap, Stint1, Stint2, PitCost, Total).
    """
    c1, c2 = encode_compound(compound_1), encode_compound(compound_2)
    team_enc = team if isinstance(team, (int, np.integer)) else encode_team(team)
    latest_pit = latest_pit or total_laps - 5

    frames = []
    for p in range(earliest_pit, latest_pit + 1):
        s1 = _stint_rows(c1, 1, p, team_enc, config).assign(PitLap=p, Stint=1)
        s2 = _stint_rows(c2, p + 1, total_laps - p, team_enc, config).assign(PitLap=p, Stint=2)
        frames += [s1, s2]
    rows = pd.concat(frames, ignore_index=True)
    rows["Delta"] = predict_lap_deltas(rows)

    curve = (rows.pivot_table(index="PitLap", columns="Stint", values="Delta", aggfunc="sum")
                 .rename(columns={1: "Stint1", 2: "Stint2"}).reset_index())
    curve["PitCost"] = pit_cost
    curve["Total"] = curve["Stint1"] + curve["Stint2"] + curve["PitCost"]

    best = curve.loc[curve["Total"].idxmin()]
    return {
        "compound_1": compound_1, "compound_2": compound_2,
        "optimal_lap": int(best["PitLap"]),
        "optimal_cost": float(best["Total"]),
        "curve": curve,
        "laps": rows,   # lap-by-lap deltas, for the "why" chart
    }


def pit_window(curve: pd.DataFrame, tolerance_s: float = 1.0) -> tuple[int, int]:
    """Laps whose total is within `tolerance_s` of the optimum (flat U-bottom)."""
    ok = curve[curve["Total"] <= curve["Total"].min() + tolerance_s]["PitLap"]
    return int(ok.min()), int(ok.max())


def compare_strategies(total_laps: int, team: str, config: dict,
                       compounds=("SOFT", "MEDIUM", "HARD"),
                       pit_cost: float = PIT_COST_GREEN) -> pd.DataFrame:
    """
    Every legal 1-stop pair (regulations force two different dry compounds),
    ranked by optimal total cost.
    """
    out = []
    for c1, c2 in permutations(compounds, 2):
        r = optimize_pit_lap(c1, c2, total_laps, team, config, pit_cost)
        lo, hi = pit_window(r["curve"])
        out.append({"Strategy": f"{c1} → {c2}", "Optimal pit lap": r["optimal_lap"],
                    "Window (±1 s)": f"L{lo}–L{hi}", "Total Δ (s)": round(r["optimal_cost"], 2),
                    "_result": r})
    df = pd.DataFrame(out).sort_values("Total Δ (s)").reset_index(drop=True)
    df["Gap to best (s)"] = (df["Total Δ (s)"] - df["Total Δ (s)"].iloc[0]).round(2)
    return df


assert set(COMPOUND_MAP) == {"SOFT", "MEDIUM", "HARD"}
