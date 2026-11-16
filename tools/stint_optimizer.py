"""
Tool 2 — Stint Optimizer.

Two questions:

1. WHEN to stop on a one-stop race: sweep every candidate pit lap and return the total
   accumulated LapDelta (+ pit loss) for each — the "U-curve" (optimize_pit_lap).
2. HOW MANY times to stop (V3.2): the best one-, two- and three-stop strategies
   (best_strategies) and how sure we can be before the race (stop_count_outlook): the
   answer depends on how hard the day is on tyres, which is only known ±0.024 s/lap in
   advance — so the tool reports P(two stops is faster), not just a winner.

Deterministic on purpose: no lap noise, no traffic. Noise answers "what will happen?";
this tool answers "what is optimal in expectation, on a clear track?".

Vectorised: every (compound, tyre age, lap) the search needs goes into ONE Module 1 call,
then all strategies are plain array sums.
"""

from __future__ import annotations

from itertools import combinations, permutations, product

import numpy as np
import pandas as pd

from tools.shared import (COMPOUND_MAP, COMPOUND_NAMES, DEG_RANGE, PIT_COST_GREEN, circuit_deg_rate,
                          circuit_deg_sd, encode_circuit, encode_compound, encode_team,
                          predict_lap_deltas, uses_deg_rate)

MIN_STINT = 5        # laps — shortest stint the search considers (as optimize_pit_lap's earliest_pit)


def _stint_rows(compound: int, start_lap: int, n_laps: int, team_encoded: int,
                config: dict, starting_tyre_age: int = 1) -> pd.DataFrame:
    laps = np.arange(start_lap, start_lap + n_laps)
    return pd.DataFrame({
        "CompoundEncoded": compound,
        "TyreLife": np.arange(starting_tyre_age, starting_tyre_age + n_laps),
        "TeamEncoded": team_encoded,
        "LapNumber": laps,          # used by V2 models; V3 deltas are fuel-corrected (same for every plan)
        "TrackTemp": config["track_temp"],
        "AirTemp": config["air_temp"],
        "RainfallEncoded": config.get("rainfall", 0),
        "CircuitEncoded": encode_circuit(config.get("circuit")),   # unknown → average of circuits
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


# ── How many stops? (V3.2) ───────────────────────────────────────────────────
def stint_costs(total_laps: int, team: str | int, config: dict, deg_rate: float | None = None) -> np.ndarray:
    """
    cost[c, start, n] = seconds lost over a stint of n laps on fresh compound c that begins on
    lap `start` (sum of LapDelta, tyre age 1…n). One Module 1 call for every (compound, age, lap).
    """
    L = total_laps
    team_enc = team if isinstance(team, (int, np.integer)) else encode_team(team)
    c, a, l = np.meshgrid(np.arange(3), np.arange(1, L + 1), np.arange(1, L + 1), indexing="ij")
    rows = pd.DataFrame({
        "CompoundEncoded": c.ravel(), "TyreLife": a.ravel(), "TeamEncoded": team_enc, "LapNumber": l.ravel(),
        "TrackTemp": config["track_temp"], "AirTemp": config["air_temp"],
        "RainfallEncoded": config.get("rainfall", 0), "CircuitEncoded": encode_circuit(config.get("circuit")),
    })
    delta = np.zeros((3, L + 2, L + 2))
    delta[c.ravel(), a.ravel(), l.ravel()] = predict_lap_deltas(rows, deg_rate=deg_rate)
    cost = np.zeros((3, L + 2, L + 1))
    for start in range(1, L + 1):                       # walk the diagonal: age 1 on lap `start`, 2 on the next…
        n = L - start + 1
        ages = np.arange(1, n + 1)
        cost[:, start, 1:n + 1] = np.cumsum(delta[:, ages, start + ages - 1], axis=1)
    return cost


def _sequences(n_stints: int, compounds) -> list[tuple[int, ...]]:
    """Compound orders with at least two different dry compounds (the regulations), softest listed first on ties."""
    codes = [encode_compound(c) for c in compounds]
    return [seq for seq in product(codes, repeat=n_stints) if len(set(seq)) >= 2]


def best_strategies(total_laps: int, team: str | int, config: dict, pit_cost: float = PIT_COST_GREEN,
                    max_stops: int = 3, min_stint: int = MIN_STINT, compounds=("SOFT", "MEDIUM", "HARD"),
                    deg_rate: float | None = None, costs: np.ndarray | None = None) -> pd.DataFrame:
    """
    The fastest strategy for each number of stops (1…max_stops): every combination of pit laps
    (stints of at least `min_stint` laps) × every legal compound order. Columns: Stops, Strategy,
    Pit laps, Total Δ (s) (tyre time lost + pit stops), Gap to best (s), and the raw
    `pit_laps` / `compounds` lists.
    Each lap of tyre age costs the same on any race lap for V3+ models (fuel is corrected out), so
    two orders of the same compounds tie; the softer start is listed.
    """
    L = total_laps
    cost = stint_costs(L, team, config, deg_rate) if costs is None else costs
    out = []
    for stops in range(1, max_stops + 1):
        pits = np.array([p for p in combinations(range(min_stint, L - min_stint + 1), stops)
                         if all(b - a >= min_stint for a, b in zip(p, p[1:]))], dtype=int).reshape(-1, stops)
        if not len(pits):
            continue
        starts = np.column_stack([np.ones(len(pits), dtype=int), pits + 1])          # first lap of each stint
        lengths = np.diff(np.column_stack([np.zeros(len(pits), dtype=int), pits, np.full(len(pits), L)]), axis=1)
        best = (np.inf, None, None)
        for seq in _sequences(stops + 1, compounds):
            total = sum(cost[cmp, starts[:, k], lengths[:, k]] for k, cmp in enumerate(seq)) + stops * pit_cost
            i = int(np.argmin(total))
            if total[i] < best[0] - 1e-9:
                best = (float(total[i]), seq, pits[i])
        total, seq, laps = best
        out.append({"Stops": stops, "Strategy": " → ".join(COMPOUND_NAMES[c] for c in seq),
                    "Pit laps": ", ".join(f"L{p}" for p in laps), "Total Δ (s)": round(total, 2),
                    "pit_laps": [int(p) for p in laps], "compounds": [COMPOUND_NAMES[c] for c in seq]})
    df = pd.DataFrame(out)
    df["Gap to best (s)"] = (df["Total Δ (s)"] - df["Total Δ (s)"].min()).round(2)
    return df


def stop_count_outlook(total_laps: int, team: str | int, config: dict, pit_cost: float = PIT_COST_GREEN,
                       max_stops: int = 3, min_stint: int = MIN_STINT, n_nodes: int = 9,
                       deg_rate: float | None = None, deg_sd: float | None = None) -> dict:
    """
    How many stops, given that the day's degradation is uncertain.

    The race's degradation rate γ is known before the race only as N(circuit estimate, race-to-race
    sd²). The optimisation is repeated at `n_nodes` equally likely values of γ; the share of them
    where n stops is fastest is P(n stops is the fastest plan).

    Returns {deg_rate, deg_sd, by_rate (one row per γ: best total per number of stops, Best),
             p_best {stops: probability}, expected (best_strategies at the expected γ)}.
    """
    expected = best_strategies(total_laps, team, config, pit_cost, max_stops, min_stint, deg_rate=deg_rate)
    if not uses_deg_rate():                             # older models: no γ to vary
        best = int(expected.loc[expected["Total Δ (s)"].idxmin(), "Stops"])
        return {"deg_rate": None, "deg_sd": 0.0, "by_rate": pd.DataFrame(), "expected": expected,
                "p_best": {int(s): float(s == best) for s in expected["Stops"]}}
    from scipy.stats import norm
    rate = float(circuit_deg_rate(config.get("circuit")) if deg_rate is None else deg_rate)
    sd = float(circuit_deg_sd(config.get("circuit")) if deg_sd is None else deg_sd)
    cuts = norm.ppf(np.arange(0, n_nodes + 1) / n_nodes)
    z = (norm.pdf(cuts[:-1]) - norm.pdf(cuts[1:])) * n_nodes            # mean of z in each equally likely slice
    rows = []
    for g in np.clip(rate + z * sd, *DEG_RANGE):
        b = best_strategies(total_laps, team, config, pit_cost, max_stops, min_stint, deg_rate=float(g))
        rows.append({"Degradation (s/lap)": float(g), **{f"{int(s)}-stop": t for s, t in zip(b["Stops"], b["Total Δ (s)"])},
                     "Best": int(b.loc[b["Total Δ (s)"].idxmin(), "Stops"])})
    by_rate = pd.DataFrame(rows)
    p_best = {int(s): float((by_rate["Best"] == s).mean()) for s in expected["Stops"]}
    return {"deg_rate": rate, "deg_sd": sd, "by_rate": by_rate, "p_best": p_best, "expected": expected}


def real_stop_counts(laps: pd.DataFrame) -> pd.DataFrame:
    """
    What teams really did: one row per (Season, Race, Driver) that reached the flag (or was
    lapped ≤ 1), with `stops` = tyre changes in the pit lane during the race — red-flag changes
    are free and are not stops — and `sc_stops` = how many of them were made under a Safety Car
    or VSC (cheap stops: a Safety Car can turn a one-stop race into a two-stop one).
    """
    rows = []
    for (season, race), r in laps.groupby(["Season", "Race"]):
        L = int(r["TotalLaps"].iloc[0])
        for drv, g in r.groupby("Driver"):
            if g["LapNumber"].max() < L - 1:
                continue
            g = g.sort_values("LapNumber")
            fresh = g["TyreLife"].shift(-1) < g["TyreLife"]             # the next lap is on a younger tyre
            swap = g["Compound"].shift(-1).notna() & (g["Compound"].shift(-1) != g["Compound"])
            pits = g[g["PitInTime"].notna() & (g["LapNumber"] < L) & (fresh | swap)]   # drive-throughs are not stops
            status = pits["TrackStatus"].astype(str)
            racing = pits[~status.str.contains("5")]
            rows.append({"Season": season, "Race": race, "Driver": drv, "stops": len(racing),
                         "sc_stops": int(racing["TrackStatus"].astype(str).str.contains("[467]").sum())})
    return pd.DataFrame(rows)


assert set(COMPOUND_MAP) == {"SOFT", "MEDIUM", "HARD"}
