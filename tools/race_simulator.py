"""
Tools 3 & 5 — Monte Carlo Race Simulator + What-If.

Rewritten from notebooks/race_simulator.ipynb + what_if_simulator.ipynb.
Three design flaws in the notebook version, and how this version fixes them:

1. Overtakes were cosmetic. The notebook swapped `position` after a
   successful Module 2 roll, but the final classification sorted by
   `race_time` — so the swap was thrown away. Track position never mattered.
   → Here a faster car that catches the car ahead is HELD UP (its lap time is
     clamped to `ahead + MIN_FOLLOW_GAP`) unless it wins the Module 2 roll.
     Failing to pass now costs real time, so tyre offsets and pace matter.

2. The Safety Car added +30 s to everyone. Adding a constant to every car
   changes nothing: same order, same gaps. The real effect of an SC is that
   it BUNCHES the field and makes pit stops cheap.
   → Here SC laps compress gaps (`gap *= SC_COMPRESS`, floor `SC_MIN_GAP`),
     pit loss drops to PIT_COST_SC, and cars whose planned stop is close can
     opportunistically box ("free stop").

3. Too deterministic (once 1 and 2 are fixed): i.i.d. per-lap noise averages
   out over 50 laps (σ·√50 ≈ 1 s), so the grid order barely changed. Real
   race-to-race variance comes from CORRELATED sources: a driver's pace on
   the day and the start. → `RACE_FORM_STD` (per driver, per race) and
   `START_STD` (lap-1 launch loss).

4. Slow: one Module 1 call per lap per race (57 × 1,000 = 57,000 calls).
   → Tyre state only depends on (team, compound, tyre age, lap), so we
     predict a lookup TABLE once (~100k rows, one call) and the simulation
     is pure numpy on (n_sims × n_drivers) arrays. Module 2 is called once
     per lap for all sims together.

What-If uses COMMON RANDOM NUMBERS: baseline and modified scenarios consume
the exact same random draws (noise, SC timing, overtake rolls). Differences
between the two are therefore caused by the strategy change, not by luck —
and we can report paired results (in how many of the SAME races did the
change help?).
"""

from __future__ import annotations


import numpy as np
import pandas as pd

from tools.race_engine import (DEFAULT_CONFIG, DIRTY_AIR_PENALTY, MIN_FOLLOW_GAP,  # noqa: F401
                               RACE_FORM_STD, SC_COMPRESS, SC_LAP_FACTOR, SC_MIN_GAP, START_GAP,
                               START_STD, build_setup, draw, initial_state, resolve_config,
                               run_to_flag)

POINTS = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1] + [0] * 30)

GRID_COLUMNS = ["Driver", "Team", "Grid", "PaceFactor", "StartCompound", "StartAge",
                "Stop1Lap", "Stop1Compound", "Stop2Lap", "Stop2Compound"]


# ── Grid helpers ─────────────────────────────────────────────────────────────
def grid_from_qualifying(quali: pd.DataFrame, total_laps: int,
                         start_compound: str = "MEDIUM", next_compound: str = "HARD",
                         pit_lap: int | None = None, pace_spread: float = 1.0) -> pd.DataFrame:
    """
    Editable strategy grid from a qualifying table (Driver, Team, QualyTime, QualyRank).
    PaceFactor = 1 + spread × (QualyTime / pole − 1): qualifying gaps as race pace.
    """
    q = quali.sort_values("QualyRank").reset_index(drop=True)
    pole = q["QualyTime"].min()
    pit_lap = pit_lap or int(round(total_laps * 0.45))
    return pd.DataFrame({
        "Driver": q["Driver"], "Team": q["Team"], "Grid": q["QualyRank"].astype(int),
        "PaceFactor": (1 + pace_spread * (q["QualyTime"] / pole - 1)).round(4),
        "StartCompound": start_compound, "StartAge": 1,
        "Stop1Lap": pit_lap, "Stop1Compound": next_compound,
        "Stop2Lap": 0, "Stop2Compound": next_compound,
    })[GRID_COLUMNS]


def validate_grid(grid: pd.DataFrame, total_laps: int) -> list[str]:
    """Human-readable problems (empty list = OK). Shown in the dashboard."""
    errs = []
    if grid["Driver"].duplicated().any():
        errs.append("Duplicate driver codes.")
    for _, r in grid.iterrows():
        stops = [(int(r[f"Stop{k}Lap"]), r[f"Stop{k}Compound"]) for k in (1, 2)
                 if pd.notna(r[f"Stop{k}Lap"]) and int(r[f"Stop{k}Lap"]) > 0]
        laps = [s[0] for s in stops]
        if any(l >= total_laps for l in laps):
            errs.append(f"{r.Driver}: pit lap must be < {total_laps}.")
        if laps != sorted(laps) or len(set(laps)) != len(laps):
            errs.append(f"{r.Driver}: pit laps must be increasing.")
        used = {str(r.StartCompound).upper()} | {str(c).upper() for _, c in stops}
        if len(used) < 2:
            errs.append(f"{r.Driver}: dry races require two different compounds (rule, not optional).")
    return errs


# ── Core simulation (the engine lives in tools/race_engine.py) ─────────────────
def plans_from_grid(g: pd.DataFrame) -> list[list[tuple[int, str]]]:
    plans = []
    for _, r in g.iterrows():
        plans.append([(int(r[f"Stop{k}Lap"]), r[f"Stop{k}Compound"]) for k in (1, 2)
                      if pd.notna(r.get(f"Stop{k}Lap")) and int(r[f"Stop{k}Lap"]) > 0])
    return plans


def _prepare(grid: pd.DataFrame, cfg: dict) -> dict:
    g = grid.sort_values("Grid").reset_index(drop=True)
    setup = build_setup(g["Driver"], g["Team"], g["PaceFactor"].astype(float), g["StartCompound"],
                        g["StartAge"].astype(int), plans_from_grid(g), cfg)
    setup["grid"] = g
    return setup


def _run(setup: dict, cfg: dict, dr) -> dict:
    state = initial_state(setup, dr.sc_u.shape[0], np.abs(dr.start))   # grid slot + launch loss
    return run_to_flag(state, setup, cfg, dr, record_trace=True)


def summarise(res: dict, grid: pd.DataFrame) -> pd.DataFrame:
    pos, ret = res["positions"], res["retired"]
    D = pos.shape[1]
    out = pd.DataFrame({
        "Driver": grid["Driver"].values, "Team": grid["Team"].values, "Grid": grid["Grid"].values,
        "Win %": (pos == 1).mean(0) * 100,
        "Podium %": (pos <= 3).mean(0) * 100,
        "Points %": (pos <= 10).mean(0) * 100,
        "Exp. position": pos.mean(0),
        "Exp. points": POINTS[pos - 1].mean(0),
        "DNF %": ret.mean(0) * 100,
        "Avg stops": res["n_stops"].mean(0),
    })
    out["Median pos"] = np.median(pos, 0)
    return out.sort_values("Exp. position").reset_index(drop=True)


def position_distribution(res: dict, grid: pd.DataFrame) -> pd.DataFrame:
    """Rows = drivers, cols = P1..PD, values = probability."""
    pos = res["positions"]
    D = pos.shape[1]
    m = np.stack([(pos == k).mean(0) for k in range(1, D + 1)], 1)
    return pd.DataFrame(m, index=grid["Driver"].values, columns=[f"P{k}" for k in range(1, D + 1)])


def simulate(grid: pd.DataFrame, config: dict | None = None, n_sims: int = 500,
             seed: int = 42) -> dict:
    cfg = resolve_config(config)
    p = _prepare(grid, cfg)
    dr = draw(n_sims, cfg["total_laps"], p["D"], cfg, seed)
    res = _run(p, cfg, dr)
    res["grid"] = p["grid"]
    res["summary"] = summarise(res, p["grid"])
    res["config"] = cfg
    return res


def what_if(base_grid: pd.DataFrame, new_grid: pd.DataFrame, focus_driver: str,
            config: dict | None = None, n_sims: int = 500, seed: int = 42) -> dict:
    """
    Same draws for both scenarios (common random numbers) → paired comparison.
    Both grids must contain the same drivers in the same grid order.
    """
    cfg = resolve_config(config)
    pb, pn = _prepare(base_grid, cfg), _prepare(new_grid, cfg)
    if list(pb["grid"]["Driver"]) != list(pn["grid"]["Driver"]):
        raise ValueError("Both scenarios need the same drivers in the same grid order.")
    dr = draw(n_sims, cfg["total_laps"], pb["D"], cfg, seed)
    rb, rn = _run(pb, cfg, dr), _run(pn, cfg, dr)

    i = list(pb["grid"]["Driver"]).index(focus_driver)
    b, n = rb["positions"][:, i], rn["positions"][:, i]
    diff = b - n                                     # >0 → new strategy finished higher
    sb, sn = summarise(rb, pb["grid"]), summarise(rn, pn["grid"])
    comp = sb.set_index("Driver")[["Exp. position", "Win %", "Podium %", "Exp. points"]].join(
        sn.set_index("Driver")[["Exp. position", "Win %", "Podium %", "Exp. points"]],
        lsuffix=" (base)", rsuffix=" (what-if)")
    comp["Δ Exp. position"] = comp["Exp. position (what-if)"] - comp["Exp. position (base)"]
    return {
        "focus": focus_driver,
        "better_pct": float((diff > 0).mean() * 100),
        "same_pct": float((diff == 0).mean() * 100),
        "worse_pct": float((diff < 0).mean() * 100),
        "mean_gain": float(diff.mean()),
        # paired standard error: this is why CRN matters — it is MUCH smaller than unpaired
        "gain_se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
        "gain_hist": pd.Series(diff).value_counts().sort_index(),
        "base": rb, "new": rn, "comparison": comp.sort_values("Exp. position (base)"),
        "base_positions": b, "new_positions": n,
    }
