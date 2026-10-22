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

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tools.shared import (BASE_LAPTIME, DRS_WINDOW, NOISE_STD, PIT_COST_GREEN,
                          PIT_COST_SC, encode_compound, encode_team,
                          lap_time_from_delta, predict_lap_deltas,
                          predict_overtake_probability)

POINTS = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1] + [0] * 30)

START_GAP = 0.25          # s between consecutive grid slots after lap 1
MIN_FOLLOW_GAP = 0.30     # s — can't cross the line closer than this behind a car you failed to pass
DIRTY_AIR_PENALTY = 0.20  # s/lap when within DRS_WINDOW of the car ahead
PIT_NOISE_STD = 0.8       # s — stop-to-stop variation
RACE_FORM_STD = 0.12      # s/lap — per-race pace offset per driver (set-up, track evolution, a good/bad day)
START_STD = 0.6           # s — lap-1 chaos: launch, first-corner positioning
SC_LAP_FACTOR = 1.35      # SC lap ≈ 35 % slower than a racing lap
SC_COMPRESS = 0.35        # each SC lap keeps 35 % of every gap…
SC_MIN_GAP = 0.6          # …but cars queue no closer than this

DEFAULT_CONFIG = {
    "total_laps": 52,
    "track_temp": 35.0,
    "air_temp": 25.0,
    "rainfall": 0,
    "sc_prob_per_lap": 0.012,      # ≈ 45 % chance of at least one SC in 52 laps
    "sc_min_laps": 3,
    "sc_max_laps": 5,
    "sc_pit_window": 8,            # box under SC if planned stop is ≤ N laps away (0 = never)
    "dnf_prob_per_lap": 0.0004,    # ≈ 2 % per car per race
    "base_laptime": BASE_LAPTIME,
}

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


# ── Core simulation ──────────────────────────────────────────────────────────
@dataclass
class _Draws:
    """All randomness for one batch, pre-drawn so two scenarios can share it."""
    lap_noise: np.ndarray    # (S, L, D)
    pit_noise: np.ndarray    # (S, L, D)
    sc_u: np.ndarray         # (S, L)
    sc_len: np.ndarray       # (S, L)
    ovt_u: np.ndarray        # (S, L, D)  indexed by running position
    dnf_u: np.ndarray        # (S, L, D)
    form: np.ndarray         # (S, D)
    start: np.ndarray        # (S, D)


def _draw(n_sims: int, n_laps: int, n_drivers: int, cfg: dict, seed: int) -> _Draws:
    rng = np.random.default_rng(seed)
    S, L, D = n_sims, n_laps, n_drivers
    return _Draws(
        lap_noise=rng.normal(0, NOISE_STD, (S, L, D)),
        pit_noise=rng.normal(0, PIT_NOISE_STD, (S, L, D)),
        sc_u=rng.random((S, L)),
        sc_len=rng.integers(cfg["sc_min_laps"], cfg["sc_max_laps"] + 1, (S, L)),
        ovt_u=rng.random((S, L, D)),
        dnf_u=rng.random((S, L, D)),
        form=rng.normal(0, RACE_FORM_STD, (S, D)),
        start=rng.normal(0, START_STD, (S, D)),
    )


def _delta_table(teams_enc: np.ndarray, n_laps: int, max_age: int, cfg: dict) -> np.ndarray:
    """LapDelta for every (team, compound, tyre age, lap) → array [T, 3, max_age+1, n_laps+1]."""
    T = len(teams_enc)
    t, c, a, l = np.meshgrid(np.arange(T), np.arange(3), np.arange(1, max_age + 1),
                             np.arange(1, n_laps + 1), indexing="ij")
    rows = pd.DataFrame({
        "CompoundEncoded": c.ravel(), "TyreLife": a.ravel(), "TeamEncoded": teams_enc[t.ravel()],
        "LapNumber": l.ravel(), "TrackTemp": cfg["track_temp"], "AirTemp": cfg["air_temp"],
        "RainfallEncoded": cfg.get("rainfall", 0),
    })
    table = np.zeros((T, 3, max_age + 1, n_laps + 1))
    table[t.ravel(), c.ravel(), a.ravel(), l.ravel()] = predict_lap_deltas(rows)
    return table


def _prepare(grid: pd.DataFrame, cfg: dict) -> dict:
    g = grid.sort_values("Grid").reset_index(drop=True)
    D, L = len(g), cfg["total_laps"]
    team_names = g["Team"].tolist()
    uniq = sorted(set(team_names))
    team_idx = np.array([uniq.index(t) for t in team_names])
    stop_laps = np.full((D, 3), 10_000)
    stop_comp = np.zeros((D, 3), dtype=int)
    for i, r in g.iterrows():
        k = 0
        for s in (1, 2):
            lap = r.get(f"Stop{s}Lap")
            if pd.notna(lap) and int(lap) > 0:
                stop_laps[i, k], stop_comp[i, k] = int(lap), encode_compound(r[f"Stop{s}Compound"])
                k += 1
    max_age = L + int(g["StartAge"].max()) + 1
    return {
        "grid": g, "D": D, "L": L, "team_idx": team_idx,
        "pace": g["PaceFactor"].astype(float).to_numpy(),
        "start_comp": np.array([encode_compound(c) for c in g["StartCompound"]]),
        "start_age": g["StartAge"].astype(int).to_numpy(),
        "stop_laps": stop_laps, "stop_comp": stop_comp, "max_age": max_age,
        "table": _delta_table(np.array([encode_team(t) for t in uniq]), L, max_age, cfg),
    }


def _run(p: dict, cfg: dict, dr: _Draws, record_trace: bool = True) -> dict:
    S, D, L = dr.sc_u.shape[0], p["D"], p["L"]
    base = cfg.get("base_laptime", BASE_LAPTIME)
    sims = np.arange(S)[:, None]
    drv = np.arange(D)[None, :]

    cum = np.arange(D)[None, :] * START_GAP + np.abs(dr.start)   # grid slot + launch loss
    comp = np.broadcast_to(p["start_comp"], (S, D)).copy()
    age = np.broadcast_to(p["start_age"], (S, D)).copy()
    nxt = np.zeros((S, D), dtype=int)                    # index of next planned stop
    retired = np.zeros((S, D), dtype=bool)
    sc_left = np.zeros(S, dtype=int)
    sc_count = np.zeros(S, dtype=int)
    overtakes = np.zeros(S, dtype=int)
    n_stops = np.zeros((S, D), dtype=int)
    trace_pos, trace_gap = [], []
    BIG = 1e9

    for lap in range(1, L + 1):
        li = lap - 1
        # --- Safety Car state
        deploy = (sc_left == 0) & (dr.sc_u[:, li] < cfg["sc_prob_per_lap"]) & (lap > 1) & (lap < L - 1)
        sc_left = np.where(deploy, dr.sc_len[:, li], sc_left)
        sc_count += deploy
        sc = sc_left > 0

        # --- Pit decisions (planned, or opportunistic under SC)
        planned_lap = p["stop_laps"][drv, np.minimum(nxt, 2)]
        planned_lap = np.where(nxt >= 2, 10_000, planned_lap)
        has_stop = planned_lap < 10_000
        pit = has_stop & (planned_lap == lap)
        if cfg.get("sc_pit_window", 0) > 0:
            pit |= has_stop & sc[:, None] & (planned_lap - lap <= cfg["sc_pit_window"]) & (planned_lap > lap)
        pit &= ~retired

        # --- Lap times
        delta = p["table"][p["team_idx"][None, :], comp, np.minimum(age, p["max_age"]), lap]
        racing = lap_time_from_delta(delta, p["pace"][None, :], base) + dr.form + dr.lap_noise[:, li]
        sc_time = base * SC_LAP_FACTOR + 0.1 * dr.lap_noise[:, li]
        lt = np.where(sc[:, None], sc_time, racing)
        pit_cost = np.where(sc, PIT_COST_SC, PIT_COST_GREEN)[:, None] + dr.pit_noise[:, li]
        lt = lt + np.where(pit, pit_cost, 0.0)

        # --- Running order at the start of the lap
        order = np.argsort(np.where(retired, BIG, cum), axis=1)
        cum_o = np.take_along_axis(cum, order, 1)
        gap_prev = np.diff(cum_o, axis=1, prepend=cum_o[:, :1])         # gap to car ahead (0 for leader)

        # dirty air (green only)
        lt_o = np.take_along_axis(lt, order, 1)
        close = (np.arange(D)[None, :] > 0) & (gap_prev < DRS_WINDOW) & ~sc[:, None]
        lt_o = lt_o + np.where(close, DIRTY_AIR_PENALTY, 0.0)

        # Module 2 — one call for every in-range pair in every sim
        comp_o, age_o = np.take_along_axis(comp, order, 1), np.take_along_axis(age, order, 1)
        prob = np.zeros((S, D))
        ask = close.copy()
        if ask.any():
            si, pi = np.nonzero(ask)
            prob[si, pi] = predict_overtake_probability(
                gap_prev[si, pi], comp_o[si, pi], comp_o[si, pi - 1], age_o[si, pi],
                age_o[si, pi - 1], np.full(len(si), lap), pi + 1)

        # sequential pass down the order: held up unless the pass succeeds
        cand = cum_o + lt_o
        pit_o = np.take_along_axis(pit, order, 1)
        ret_o = np.take_along_axis(retired, order, 1)
        new = cand.copy()
        front = np.where(pit_o[:, 0] | ret_o[:, 0], -BIG, cand[:, 0])
        for i in range(1, D):
            c = cand[:, i]
            active = ~ret_o[:, i] & ~pit_o[:, i]
            blocked = active & (c < front + MIN_FOLLOW_GAP)
            passed = blocked & ~sc & (gap_prev[:, i] < DRS_WINDOW) & (dr.ovt_u[:, li, i] < prob[:, i])
            overtakes += passed
            new[:, i] = np.where(blocked & ~passed, front + MIN_FOLLOW_GAP, c)
            front = np.where(active, np.maximum(front, new[:, i]), front)

        # --- Safety Car bunching (after the lap, in the NEW order)
        if sc.any():
            o2 = np.argsort(np.where(ret_o, BIG, new), axis=1)
            n2 = np.take_along_axis(new, o2, 1)
            r2 = np.take_along_axis(ret_o, o2, 1)
            gaps = np.diff(n2, axis=1)
            gaps = np.where(sc[:, None], np.maximum(SC_MIN_GAP, gaps * SC_COMPRESS), gaps)
            n2 = np.where(r2, n2, np.concatenate([n2[:, :1], n2[:, :1] + np.cumsum(gaps, 1)], 1))
            np.put_along_axis(new, o2, n2, 1)

        # back to driver indexing
        np.put_along_axis(cum, order, new, 1)

        # --- Tyre / stop bookkeeping, retirements
        comp = np.where(pit, p["stop_comp"][drv, np.minimum(nxt, 2)], comp)
        age = np.where(pit, 1, age + 1)
        nxt += pit
        n_stops += pit
        newly_out = ~retired & (dr.dnf_u[:, li] < cfg.get("dnf_prob_per_lap", 0.0))
        retired |= newly_out
        sc_left = np.maximum(sc_left - 1, 0)

        if record_trace:
            pos = np.argsort(np.argsort(np.where(retired, BIG, cum), 1), 1) + 1
            trace_pos.append(pos[0])
            trace_gap.append(np.where(retired[0], np.nan, cum[0] - cum[0][~retired[0]].min()))

    final = np.where(retired, BIG + cum, cum)
    positions = np.argsort(np.argsort(final, 1), 1) + 1                     # (S, D)
    return {"positions": positions, "retired": retired, "race_time": cum,
            "sc_count": sc_count, "overtakes": overtakes, "n_stops": n_stops,
            "trace_pos": np.array(trace_pos), "trace_gap": np.array(trace_gap)}


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
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    p = _prepare(grid, cfg)
    dr = _draw(n_sims, cfg["total_laps"], p["D"], cfg, seed)
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
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    pb, pn = _prepare(base_grid, cfg), _prepare(new_grid, cfg)
    if list(pb["grid"]["Driver"]) != list(pn["grid"]["Driver"]):
        raise ValueError("Both scenarios need the same drivers in the same grid order.")
    dr = _draw(n_sims, cfg["total_laps"], pb["D"], cfg, seed)
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
