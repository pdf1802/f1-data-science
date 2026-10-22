"""
Tool 1 — Undercut Calculator.

Question it answers: "Car A is X s behind car B. If A pits NOW, does it come
out ahead once B responds?"

Ported from notebooks/undercut_calculator.ipynb with three changes:

1. Deterministic by default. The notebook added N(0, 0.8) to every pit stop,
   so running the same input twice could flip the verdict. A decision tool
   must give the same answer for the same input. (`pit_noise_std` keeps the
   option for sensitivity analysis.)
2. One Module 1 call for all scenarios. Lap times do not depend on the gap,
   so we predict every (scenario, driver, lap) row at once and only the gap
   bookkeeping — which DOES depend on the previous lap (dirty air) — loops.
3. Dirty air is symmetric: whoever is within 2 s behind pays the penalty,
   including the original leader once the undercut has worked.

Gap convention: gap = time(A) − time(B). Positive → A still behind.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from tools.shared import (BASE_LAPTIME, DRS_WINDOW, PIT_COST_GREEN, PIT_COST_SC,
                          encode_compound, encode_team, lap_time_from_delta,
                          predict_lap_deltas, predict_overtake_probability)

VERDICT_GAP = 1.5            # worst-case gap must close below this…
VERDICT_PROB = 0.25          # …AND mean overtake probability must exceed this
N_PROJ_LAPS = 5
DIRTY_AIR_THRESHOLD = 2.0    # s
DIRTY_AIR_PENALTY = 0.35     # s/lap — domain knowledge, not in Module 1
DEFAULT_PACE_FACTOR = 1.02   # used when a team has no pace factor

SCENARIOS = {                 # name → laps until the leader reacts
    "Stay out": None,
    "Leader reacts next lap": 1,
    "Leader reacts in 3 laps": 3,
    "Leader never reacts": 99,
}
WORST_CASE = "Leader reacts next lap"


@dataclass
class DriverState:
    driver: str
    team_name: str
    position: int
    compound: int | str
    tyre_age: int

    def __post_init__(self):
        self.compound = encode_compound(self.compound)
        self.team_encoded = encode_team(self.team_name)


def _plan(driver: DriverState, pit_on: int | None, new_compound: int,
          current_lap: int, n_laps: int) -> pd.DataFrame:
    """Lap-by-lap tyre state for one driver. pit_on = projection lap index (1-based) or None."""
    rows, comp, age = [], driver.compound, driver.tyre_age
    for k in range(1, n_laps + 1):
        pitted = pit_on is not None and k == pit_on
        if pitted:
            comp, age = new_compound, 1
        else:
            age += 1
        rows.append({"k": k, "LapNumber": current_lap + k, "CompoundEncoded": comp,
                     "TyreLife": age, "TeamEncoded": driver.team_encoded, "Pitted": pitted})
    return pd.DataFrame(rows)


def undercut_analysis(chaser: DriverState, leader: DriverState, current_gap: float,
                      current_lap: int, config: dict, new_compound="MEDIUM",
                      leader_compound=None, sc_active: bool = False,
                      n_laps: int = N_PROJ_LAPS, pit_noise_std: float = 0.0,
                      rng: np.random.Generator | None = None) -> dict:
    """
    Projects every scenario in SCENARIOS and returns the verdict + evidence.

    config: track_temp, air_temp, rainfall (0/1), optional pace_factors {team: f}.
    """
    new_c = encode_compound(new_compound)
    lead_c = encode_compound(leader_compound) if leader_compound is not None else new_c
    pit_cost = PIT_COST_SC if sc_active else PIT_COST_GREEN
    pf = config.get("pace_factors", {})
    rng = rng or np.random.default_rng(0)

    # 1) build every row for every scenario, predict once
    frames = []
    for name, delay in SCENARIOS.items():
        a_pit = None if delay is None else 1
        b_pit = None if delay is None or delay + 1 > n_laps else delay + 1
        frames.append(_plan(chaser, a_pit, new_c, current_lap, n_laps).assign(Scenario=name, Car="A"))
        frames.append(_plan(leader, b_pit, lead_c, current_lap, n_laps).assign(Scenario=name, Car="B"))
    rows = pd.concat(frames, ignore_index=True)
    rows["TrackTemp"] = config["track_temp"]
    rows["AirTemp"] = config["air_temp"]
    rows["RainfallEncoded"] = config.get("rainfall", 0)
    factor = np.where(rows["Car"] == "A", pf.get(chaser.team_name, DEFAULT_PACE_FACTOR),
                      pf.get(leader.team_name, DEFAULT_PACE_FACTOR))
    rows["LapTime"] = lap_time_from_delta(predict_lap_deltas(rows), factor, BASE_LAPTIME)
    rows["PitLoss"] = np.where(rows["Pitted"],
                               pit_cost + (rng.normal(0, pit_noise_std, len(rows)) if pit_noise_std else 0.0),
                               0.0)

    # 2) gap bookkeeping (sequential because of dirty air)
    curves, details = {}, {}
    for name in SCENARIOS:
        a = rows[(rows.Scenario == name) & (rows.Car == "A")].reset_index(drop=True)
        b = rows[(rows.Scenario == name) & (rows.Car == "B")].reset_index(drop=True)
        gap, curve = current_gap, [current_gap]
        for k in range(n_laps):
            la, lb = a.LapTime[k] + a.PitLoss[k], b.LapTime[k] + b.PitLoss[k]
            if 0 < gap < DIRTY_AIR_THRESHOLD:
                la += DIRTY_AIR_PENALTY            # A stuck behind B
            elif -DIRTY_AIR_THRESHOLD < gap < 0:
                lb += DIRTY_AIR_PENALTY            # B stuck behind A
            gap += la - lb
            curve.append(round(float(gap), 3))
        curves[name] = curve
        details[name] = (a, b)

    # 3) overtake probabilities on the worst-case path, with the REAL tyre state each lap
    a, b = details[WORST_CASE]
    curve = curves[WORST_CASE]
    probs = {}
    for k in range(1, n_laps + 1):
        g = curve[k]
        lap = current_lap + k
        if g <= 0:
            probs[lap] = 1.0                       # already ahead
        elif g < DRS_WINDOW:
            probs[lap] = round(predict_overtake_probability(
                g, a.CompoundEncoded[k - 1], b.CompoundEncoded[k - 1],
                a.TyreLife[k - 1], b.TyreLife[k - 1], lap, chaser.position), 4)
        else:
            probs[lap] = None                      # out of range: model not asked

    min_gap = min(curve[1:])
    valid = [p for p in probs.values() if p is not None]
    avg_prob = float(np.mean(valid)) if valid else 0.0
    cond_gap, cond_prob = min_gap < VERDICT_GAP, avg_prob > VERDICT_PROB

    # "Net gain" = where A ends up vs staying out, under each response
    final_stay = curves["Stay out"][-1]
    gains = {n: round(final_stay - c[-1], 3) for n, c in curves.items() if n != "Stay out"}

    return {
        "verdict": "BOX BOX" if (cond_gap and cond_prob) else "STAY OUT",
        "confidence": round(avg_prob, 4),
        "laps_in_drs_window": len(valid),   # 0 → Module 2 was never asked (gap never < 1 s)
        "min_gap": round(min_gap, 3),
        "condition_gap": cond_gap,
        "condition_prob": cond_prob,
        "curves": curves,
        "overtake_probs": probs,
        "gain_vs_stay_out": gains,
        "laps": list(range(current_lap, current_lap + n_laps + 1)),
        "pit_cost": pit_cost,
        "lap_table": rows,
    }


def undercut_sweep(chaser: DriverState, leader: DriverState, current_lap: int,
                   config: dict, gaps=np.arange(0.5, 6.01, 0.5), **kw) -> pd.DataFrame:
    """Verdict across a range of starting gaps — 'up to what gap does the undercut work?'"""
    out = []
    for g in gaps:
        r = undercut_analysis(chaser, leader, float(g), current_lap, config, **kw)
        out.append({"Gap (s)": round(float(g), 2), "Verdict": r["verdict"],
                    "Min gap (s)": r["min_gap"], "Avg P(overtake)": r["confidence"],
                    "Undercut gain, worst case (s)": r["gain_vs_stay_out"][WORST_CASE]})
    return pd.DataFrame(out)


__all__ = ["DriverState", "undercut_analysis", "undercut_sweep", "SCENARIOS",
           "WORST_CASE", "VERDICT_GAP", "VERDICT_PROB", "replace"]
