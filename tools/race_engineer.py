"""
The Race Engineer — strategy calls for one driver, from a live race feed.

Pure Python (no GUI): feed it the payloads that f1-race-replay broadcasts on
its telemetry stream (or any feed in that shape) and, once per completed lap,
it answers "box now, wait, or stay out?" for the driver you follow:

    eng = RaceEngineer(driver="VER", circuit="Bahrain Grand Prix", total_laps=57)
    for payload in stream:
        call = eng.update(payload)        # None until a new lap is completed
        if call:
            print(call["headline"]); print(call["options"])

How a call is made
    1. Rebuild the race state at the end of the last completed lap: order, real
       gaps (session clock at the line), tyres, stops done — and pace, estimated
       from the laps run so far only (race_engine.state_from_real_race).
    2. Rivals follow a simple, documented default plan (default_plan): stop when
       the tyre reaches the age range covered by the data, or when the
       two-compound rule still has to be met.
    3. Each option for our driver is simulated to the flag on the SAME random
       draws (paired Monte Carlo) → P(win / podium / points), expected position,
       and the gain over "stay out" with a 95 % CI.
    4. Undercut alert against the car ahead when it is within 3 s.

The engine behind it is backtested (tools/backtest.py): from 40 % distance it
beats "the order stays the same" in 75 % of 36 real races.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from tools import race_engine as eng
from tools.shared import COMPOUND_MAP, COMPOUND_NAMES, load_models

DRY = ("SOFT", "MEDIUM", "HARD")
DEFAULT_CAPS = {"SOFT": 19, "MEDIUM": 28, "HARD": 34}   # p90 tyre age in the training data
ALERT_GAP = 3.0                                         # s — undercut alert range


# ── Feed → FastF1-shaped laps ────────────────────────────────────────────────
def _compound_name(code) -> str | None:
    try:
        return COMPOUND_NAMES.get(int(round(float(code))))
    except (TypeError, ValueError):
        return None


def laps_from_payload(payload: dict, circuit: str, total_laps: int) -> tuple[pd.DataFrame, float]:
    """
    f1-race-replay payload → (laps table in FastF1 shape, current session time).
    The replay sends the lap times of the WHOLE race in every payload; only laps
    completed by the current time are kept, so nothing from the future leaks in.
    """
    t_now = float(payload.get("session_data", {}).get("time_s")
                  or (payload.get("frame") or {}).get("t") or 0.0)
    rows = []
    for code, entries in (payload.get("lap_times") or {}).items():
        for e in entries:
            end = e.get("end_time_s")
            if end is None or end > t_now or e.get("time_s", -1) is None:
                continue
            lt = e.get("time_s", -1.0)
            rows.append({
                "Driver": code, "LapNumber": int(e["lap"]),
                "LapTime": pd.to_timedelta(lt, unit="s") if lt and lt > 0 else pd.NaT,
                "Time": pd.to_timedelta(end, unit="s"),
                "Compound": _compound_name(e.get("tyre")),
                "TyreLife": float(e["tyre_life"]) if e.get("tyre_life") else np.nan,
                "PitInTime": pd.to_timedelta(end, unit="s") if e.get("is_pit_entry") else pd.NaT,
                "PitOutTime": pd.to_timedelta(end - max(lt, 0), unit="s") if e.get("is_out_lap") else pd.NaT,
                "TrackStatus": "1", "Team": "",
            })
    laps = pd.DataFrame(rows)
    if laps.empty:
        return laps, t_now
    laps["Position"] = laps.groupby("LapNumber")["Time"].rank(method="first")
    laps["Season"], laps["Race"], laps["TotalLaps"] = 0, circuit, int(total_laps)
    return laps, t_now


# ── Plans ────────────────────────────────────────────────────────────────────
def _caps() -> dict:
    from tools import shared
    load_models()
    return (shared._DEG_TAIL or {}).get("cap") or DEFAULT_CAPS


def default_plan(compound: str, age: int, used: set, lap: int, total_laps: int,
                 caps: dict | None = None) -> list[tuple[int, str]]:
    """
    What a rival is assumed to do: one more stop when the tyre would run past the
    age range the model has data for, or when the two-compound rule is not met
    yet; onto HARD (MEDIUM if already on HARD). Deliberately simple and visible.
    """
    caps = caps or DEFAULT_CAPS
    remaining = total_laps - lap
    left_on_tyre = max(caps.get(compound, 25) - age, 1)
    rule_met = len(set(used) | {compound}) >= 2
    if remaining <= left_on_tyre and rule_met:
        return []
    stop = lap + max(1, min(left_on_tyre, remaining - 3))
    if stop >= total_laps:
        return []
    return [(stop, "HARD" if compound != "HARD" else "MEDIUM")]


def plan_after_box(box_lap: int, compound: str, total_laps: int, caps: dict) -> list[tuple[int, str]]:
    """Box on box_lap onto `compound`; add one later stop only if that tyre cannot reach the flag."""
    plan = [(box_lap, compound)]
    if total_laps - box_lap > caps.get(compound, 25) + 5:
        nxt = "HARD" if compound != "HARD" else "MEDIUM"
        plan.append((box_lap + caps.get(compound, 25), nxt))
    return plan


# ── The engineer ─────────────────────────────────────────────────────────────
class RaceEngineer:
    def __init__(self, driver: str, circuit: str | None, total_laps: int | None = None,
                 n_sims: int = 400, seed: int = 7):
        from tools.circuit_profiles import profile_for
        self.driver = driver
        self.circuit = circuit
        prof = profile_for(circuit) or {}
        self.total_laps = int(total_laps or prof.get("laps") or 57)
        self.n_sims, self.seed = n_sims, seed
        self.last_lap_done = -1
        self.last_call: dict | None = None

    def update(self, payload: dict) -> dict | None:
        """New call once per newly completed lap (by the leader); None otherwise."""
        if payload.get("session_data", {}).get("total_laps"):
            self.total_laps = int(payload["session_data"]["total_laps"])
        laps, _ = laps_from_payload(payload, self.circuit or "Unknown", self.total_laps)
        if laps.empty:
            return None
        lap_done = int(laps["LapNumber"].max())
        if lap_done <= self.last_lap_done or lap_done < 3 or lap_done >= self.total_laps - 1:
            return None
        self.last_lap_done = lap_done
        current = self._current_tyres(payload)
        self.last_call = self.analyse(laps, lap_done, current)
        return self.last_call

    @staticmethod
    def _current_tyres(payload: dict) -> dict:
        out = {}
        for code, d in ((payload.get("frame") or {}).get("drivers") or {}).items():
            c = _compound_name(d.get("tyre"))
            if c in DRY:
                out[code] = (c, max(int(round(float(d.get("tyre_life") or 1))), 1))
        return out

    def analyse(self, laps: pd.DataFrame, lap_done: int, current_tyres: dict | None = None) -> dict:
        caps = _caps()
        used = laps.dropna(subset=["Compound"]).groupby("Driver")["Compound"].agg(lambda s: set(s))
        # current tyres for everyone (feed first, else last completed lap)
        last = laps[laps["LapNumber"] == lap_done].set_index("Driver")
        tyres = {}
        for d in last.index:
            if current_tyres and d in current_tyres:
                tyres[d] = current_tyres[d]
            else:
                c = last.loc[d, "Compound"] if last.loc[d, "Compound"] in DRY else "MEDIUM"
                a = int(last.loc[d, "TyreLife"]) + 1 if pd.notna(last.loc[d, "TyreLife"]) else 1
                tyres[d] = (c, a)
        plans = {d: default_plan(c, a, used.get(d, set()), lap_done, self.total_laps, caps)
                 for d, (c, a) in tyres.items()}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setup, cfg, state, info = eng.state_from_real_race(
                laps, 0, self.circuit or "Unknown", lap_done, {"dnf_prob_per_lap": 0.0},
                future_plans="none", current_tyres=tyres, plans=plans)
        live = eng.LiveRace(setup, cfg, state, seed=self.seed)
        standings = live.standings()
        if self.driver not in setup["drivers"]:
            return {"lap": lap_done, "standings": standings, "options": None,
                    "headline": f"{self.driver} is not running", "alert": None}

        comp, age = tyres[self.driver]
        rule_met_now = len(used.get(self.driver, set()) | {comp}) >= 2
        nxt = lap_done + 1
        options = {"Stay out (default plan)": plans[self.driver]}
        for c in DRY:
            if c == comp and not rule_met_now and len(used.get(self.driver, set())) < 2:
                continue                                   # would not satisfy the two-compound rule
            options[f"Box now → {c}"] = plan_after_box(nxt, c, self.total_laps, caps)
        best_alt = "HARD" if comp != "HARD" else "MEDIUM"
        if nxt + 3 < self.total_laps - 1:
            options[f"Box in 3 laps → {best_alt}"] = plan_after_box(nxt + 3, best_alt, self.total_laps, caps)
        if rule_met_now and plans[self.driver]:
            options["No more stops"] = []
        table = live.evaluate(self.driver, options, n_sims=self.n_sims, seed=self.seed + lap_done)
        table = table.sort_values(["Exp. position", "P(podium)"], ascending=[True, False]).reset_index(drop=True)
        best = table.iloc[0]
        stay = table[table["Option"] == "Stay out (default plan)"].iloc[0]
        gain = stay["Exp. position"] - best["Exp. position"]
        ci = best["Gain 95% CI"] if best["Option"] != stay["Option"] else 0.0
        if best["Option"] == stay["Option"] or gain <= ci:
            headline = "STAY OUT — no option is clearly better than the current plan"
        else:
            headline = f"{best['Option'].upper()} — +{gain:.1f} places expected vs staying out"
        return {"lap": lap_done, "total_laps": self.total_laps, "driver": self.driver,
                "tyres": {"compound": comp, "age": age, "rule_met": rule_met_now},
                "standings": standings, "options": table, "headline": headline,
                "alert": self._undercut_alert(standings, cfg), "pace": info["pace"],
                "rival_plans": plans}

    def _undercut_alert(self, standings: pd.DataFrame, cfg: dict) -> dict | None:
        from tools.undercut import DriverState, undercut_analysis
        st = standings.reset_index(drop=True)
        idx = st.index[st["Driver"] == self.driver]
        if len(idx) == 0 or idx[0] == 0:
            return None
        me, ahead = st.iloc[idx[0]], st.iloc[idx[0] - 1]
        gap = float(me["Gap"] - ahead["Gap"])
        if not np.isfinite(gap) or gap > ALERT_GAP:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # (team is not a V3.1 model input; the feed does not carry it)
            chaser = DriverState(me["Driver"], str(me["Team"]), int(me["Pos"]), me["Compound"],
                                 int(me["TyreAge"]) + 1)
            leader = DriverState(ahead["Driver"], str(ahead["Team"]), int(ahead["Pos"]), ahead["Compound"],
                                 int(ahead["TyreAge"]) + 1)
            fresh = "HARD" if me["Compound"] != "HARD" else "MEDIUM"
            r = undercut_analysis(chaser, leader, gap, self.last_lap_done,
                                  {"track_temp": cfg["track_temp"], "air_temp": cfg["air_temp"],
                                   "circuit": self.circuit}, fresh, fresh)
        return {"target": ahead["Driver"], "gap": round(gap, 2), "verdict": r["verdict"],
                "min_gap": r["min_gap"], "p_overtake": r["confidence"],
                "in_drs_window": r["laps_in_drs_window"] > 0}
