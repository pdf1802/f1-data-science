"""
The Race Engineer — strategy calls for one driver, from a live race feed.

Pure Python (no GUI): feed it the payloads that f1-race-replay broadcasts on
its telemetry stream (or any feed in that shape) and, every time the driver you
follow crosses the line, it answers "box now, wait, or stay out?":

    eng = RaceEngineer(driver="VER", circuit="Bahrain Grand Prix", total_laps=57)
    for payload in stream:
        call = eng.update(payload)        # None until VER completes a new lap
        if call:
            print(call["headline"]); print(call["options"])

How a call is made
    1. Snapshot of the race at the end of the lap N our driver has just completed.
       Cars ahead on the road crossed that line already (real times). Cars behind
       — and lapped cars — have not yet: their line time is ESTIMATED from where
       they are now (frame `rel_dist`) and their recent pace. Nothing that has not
       happened yet at the current replay time is used (no future leakage).
    2. Pace per driver from the laps run so far (race_engine.state_from_real_race).
    3. Rivals follow a simple, documented default plan (default_plan): stop when
       the tyre reaches the age range covered by the data, or when the
       two-compound rule still has to be met.
    4. Each option for our driver is simulated to the flag on the SAME random
       draws (paired Monte Carlo) → P(win / podium / points), expected position,
       and the gain over "stay out" with a 95 % CI.
    5. Undercut alert against the car ahead when it is within 3 s.

Every call has the same keys whether the driver is running or not (see CALL_KEYS).

The engine behind it is backtested (tools/backtest.py): from 40 % distance it
beats "the order stays the same" in 75 % of 36 real races.

Clocks. f1-race-replay sends lap end times on the FastF1 session clock
(`end_time_s`, `line_time_s`) and, when it can align them, on the replay clock
of `frame.t` (`replay_line_time_s`, `replay_end_time_s`). Laps it rebuilds from
frames carry `end_time_s` already on the replay clock. `line_time` picks the
replay-clock value first, so "completed by now" compares like with like.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from tools import race_engine as eng
from tools.shared import COMPOUND_NAMES, load_models

DRY = ("SOFT", "MEDIUM", "HARD")
TYRE_NAMES = {**COMPOUND_NAMES, 3: "INTERMEDIATE", 4: "WET"}   # f1-race-replay's tyre ints
DEFAULT_CAPS = {"SOFT": 19, "MEDIUM": 28, "HARD": 34}   # p90 tyre age in the training data
ALERT_GAP = 3.0            # s — undercut alert range
MIN_CALL_LAP = 3           # no calls on the opening laps (pace not known yet)
# Out of the race: since the car's last line crossing, at least OUT_WITNESSES other cars have
# each crossed the line OUT_FIELD_LAPS times (≥ 2 full laps went by; even a pit stop or a repair
# takes less). Counting the field's laps instead of seconds keeps everyone in during a red flag.
OUT_FIELD_LAPS, OUT_WITNESSES = 3, 3
STOPPED_S = 20.0           # s of replay time without moving → parked (retired), unless red flag
STAY = "Stay out (default plan)"
OPTION_COLUMNS = ["Option", "P(win)", "P(podium)", "P(points)", "Exp. position",
                  "Gain vs first (places)", "Gain 95% CI"]
CALL_KEYS = ("lap", "total_laps", "driver", "running", "tyres", "standings", "options", "headline",
             "alert", "pace", "rival_plans", "track_status", "t", "estimated")


# ── Reading the feed ─────────────────────────────────────────────────────────
def _num(v) -> float | None:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _has_replay_clock(entry: dict) -> bool:
    return _num(entry.get("replay_line_time_s")) is not None or _num(entry.get("replay_end_time_s")) is not None


def line_time(entry: dict, offset: float = 0.0) -> float | None:
    """
    When the car crossed the line at the end of this lap, on the replay clock.
    `offset` (see clock_offset) is subtracted from end_time_s when the entry has no replay_* field.
    """
    for key in ("replay_line_time_s", "replay_end_time_s"):
        v = _num(entry.get(key))
        if v is not None:
            return v
    v = _num(entry.get("end_time_s"))
    return None if v is None else v - offset


def is_lap(entry: dict) -> bool:
    """FastF1 generates placeholder rows (e.g. where a car retired): no lap was completed there."""
    return entry.get("lap") is not None and not entry.get("fastf1_generated")


def clock_offset(payload: dict) -> float:
    """
    Seconds between the FastF1 session clock and the replay clock, for entries without replay_*
    fields. f1-race-replay adds those fields when it can align the two clocks; when it cannot,
    end_time_s is session time (~1 h ahead) and every lap would look unfinished. Estimated from
    where the cars are now: a car on lap c at rel_dist r crossed the line ending lap c-1 about
    r × (its lap c-1 time) ago. 0 when no entry needs it, or when the clocks already agree
    (laps rebuilt from frames carry end_time_s on the replay clock).
    """
    lap_times = payload.get("lap_times") or {}
    t_now, est = payload_time(payload), []
    for d, fd in ((payload.get("frame") or {}).get("drivers") or {}).items():
        c, r = _num(fd.get("lap")), _num(fd.get("rel_dist"))
        if c is None or r is None:
            continue
        prev = next((e for e in lap_times.get(d) or [] if e.get("lap") == int(c) - 1), None)
        if prev is None or _has_replay_clock(prev):
            continue
        end, lt = _num(prev.get("end_time_s")), _num(prev.get("time_s"))
        if end is not None and lt and lt > 0:
            est.append(end - (t_now - r * lt))
    off = float(np.median(est)) if est else 0.0
    return off if abs(off) > 60 else 0.0


def payload_time(payload: dict) -> float:
    """Current replay time (frame clock)."""
    t = _num((payload.get("session_data") or {}).get("time_s"))
    if t is None:
        t = _num((payload.get("frame") or {}).get("t"))
    return t or 0.0


def tyre_name(code) -> str | None:
    v = _num(code)
    return TYRE_NAMES.get(int(round(v))) if v is not None else None


def track_status(payload: dict) -> str:
    """FastF1 code: '1' green, '2' yellow, '4' SC, '5' red, '6'/'7' VSC. f1-race-replay sends 'GREEN' by default."""
    s = str(payload.get("track_status") or "1").upper()
    return "1" if s in ("GREEN", "") else s


def completed_laps(payload: dict, driver: str, t_now: float | None = None,
                   offset: float | None = None) -> list[tuple[int, float]]:
    """(lap, line time) of every lap `driver` has completed by now — cheap, no DataFrame."""
    t_now = payload_time(payload) if t_now is None else t_now
    offset = clock_offset(payload) if offset is None else offset
    out = []
    for e in (payload.get("lap_times") or {}).get(driver) or []:
        t = line_time(e, offset)
        if t is not None and t <= t_now and is_lap(e):
            out.append((int(e["lap"]), t))
    return sorted(out)


def out_by_field(crossings: dict[str, np.ndarray]) -> set:
    """Cars whose last crossing is followed by OUT_FIELD_LAPS crossings of OUT_WITNESSES other cars."""
    out = set()
    for d, ts in crossings.items():
        if not len(ts):
            continue
        last = ts.max()
        counts = sorted((int((o > last).sum()) for e, o in crossings.items() if e != d), reverse=True)
        if len(counts) >= OUT_WITNESSES and counts[OUT_WITNESSES - 1] >= OUT_FIELD_LAPS:
            out.add(d)
    return out


def crossings_from_payload(payload: dict, t_now: float, offset: float) -> dict[str, np.ndarray]:
    return {d: np.array([t for e in es if is_lap(e) and (t := line_time(e, offset)) is not None and t <= t_now])
            for d, es in (payload.get("lap_times") or {}).items()}


def status_by_lap(payload: dict, upto_lap: int) -> dict[int, str]:
    """
    Track status per (leader) lap from `status_laps`, PAST ONLY: f1-race-replay
    precomputes it for the whole race, so periods that start after `upto_lap` are
    ignored and the current one is cut at `upto_lap` (its end is the future).
    """
    out = {}
    for s in payload.get("status_laps") or []:
        a, b = s.get("start_lap"), s.get("end_lap")
        if a is None or int(a) > upto_lap:
            continue
        for lap in range(int(a), min(int(b if b is not None else a), upto_lap) + 1):
            out[lap] = str(s.get("status"))
    return out


def laps_from_payload(payload: dict, circuit: str, total_laps: int,
                      offset: float | None = None) -> tuple[pd.DataFrame, float]:
    """
    f1-race-replay payload → (laps completed so far, FastF1-shaped; current replay time).
    The replay sends the lap times of the WHOLE race in every payload; only laps
    completed by the current time are kept, so nothing from the future leaks in.
    `Time` is the line time on the replay clock.
    """
    t_now = payload_time(payload)
    offset = clock_offset(payload) if offset is None else offset
    rows = []
    for code, entries in (payload.get("lap_times") or {}).items():
        for e in entries:
            end = line_time(e, offset)
            if end is None or end > t_now or not is_lap(e):
                continue
            lt = _num(e.get("time_s"))
            rows.append({
                "Driver": code, "LapNumber": int(e["lap"]),
                "LapTime": pd.to_timedelta(lt, unit="s") if lt and lt > 0 else pd.NaT,
                "Time": pd.to_timedelta(end, unit="s"),
                "Compound": tyre_name(e.get("tyre")),
                "TyreLife": float(e["tyre_life"]) if _num(e.get("tyre_life")) else np.nan,
                "PitInTime": pd.to_timedelta(end, unit="s") if e.get("is_pit_entry") else pd.NaT,
                "PitOutTime": pd.to_timedelta(end - max(lt or 0, 0), unit="s") if e.get("is_out_lap") else pd.NaT,
                "Team": "",
            })
    laps = pd.DataFrame(rows)
    if laps.empty:
        return laps, t_now
    status = status_by_lap(payload, int(laps["LapNumber"].max()))
    laps["TrackStatus"] = laps["LapNumber"].map(status).fillna("1").astype(str)
    laps["Position"] = laps.groupby("LapNumber")["Time"].rank(method="first")
    laps["Season"], laps["Race"], laps["TotalLaps"] = 0, circuit, int(total_laps)
    return laps, t_now


# ── Snapshot at the end of lap N ─────────────────────────────────────────────
def _sec(s):
    return pd.to_timedelta(s).dt.total_seconds()


def _recent_lap(g: pd.DataFrame, fallback: float) -> float:
    """Median of a driver's last 3 normal laps (no pit in/out, not lap 1)."""
    ok = g[(g["LapNumber"] > 1) & g["PitInTime"].isna() & g["PitOutTime"].isna()].sort_values("LapNumber")
    lt = _sec(ok["LapTime"]).dropna()
    return float(lt.tail(3).median()) if len(lt) else fallback


def running_drivers(laps: pd.DataFrame, t_now: float, frame_drivers: dict | None = None,
                    stopped: set | None = None) -> list[str]:
    """Cars still in the race: not left behind by the field (out_by_field) and not parked."""
    if laps.empty:
        return []
    crossings = {d: _sec(g["Time"]).to_numpy() for d, g in laps.groupby("Driver")}
    gone = out_by_field(crossings)
    return [d for d in crossings
            if d not in gone and not (frame_drivers and d not in frame_drivers) and not (stopped and d in stopped)]


def snapshot_at_lap(laps: pd.DataFrame, n: int, t_now: float, frame_drivers: dict | None = None,
                    status_now: str = "1", stopped: set | None = None
                    ) -> tuple[pd.DataFrame, dict, list[str], list[str]]:
    """
    The race at the end of lap `n`, using only what has happened by `t_now`.

    Returns (laps up to lap n with a lap-n row for every running car,
             tyres for lap n+1 {driver: (compound, age)}, running cars, estimated cars).
    Lap-n rows of cars that have not crossed that line yet are estimated
    (LapTime NaT, so they never feed the pace estimate).
    """
    frame_drivers = frame_drivers or {}
    running = running_drivers(laps, t_now, frame_drivers, stopped)
    field = _sec(laps.loc[laps["LapNumber"] > 1, "LapTime"]).median()
    field = float(field) if np.isfinite(field) else 95.0
    by_driver = {d: g.sort_values("LapNumber") for d, g in laps.groupby("Driver")}
    keep = laps[(laps["LapNumber"] <= n) & laps["Driver"].isin(running)]
    extra, tyres, estimated = [], {}, []
    for d in running:
        g = by_driver[d]
        last = g.iloc[-1]
        last_lap, last_t = int(last["LapNumber"]), float(last["Time"].total_seconds())
        pace = _recent_lap(g, field)
        fd = frame_drivers.get(d) or {}
        cur_lap = int(_num(fd.get("lap")) or last_lap + 1)
        rel = _num(fd.get("rel_dist"))

        # in the pit lane before crossing line n: the stop ends lap n (its out-lap loss is added
        # by state_from_real_race) and the car leaves on the next compound
        pitting = bool(fd.get("in_pit")) and last_lap < n and cur_lap <= n

        # line time at the end of lap n
        if not (g["LapNumber"] == n).any():
            if last_lap > n:                                  # gap in the data: back off from a later lap
                t_n = last_t - (last_lap - n) * pace
            elif rel is not None and cur_lap == last_lap + 1 and 0.0 <= rel < 1.0:
                t_n = t_now + ((n - cur_lap) + (1.0 - rel)) * pace
            else:
                t_n = max(last_t + (n - last_lap) * pace, t_now + 0.1)
            estimated.append(d)
            extra.append({"Driver": d, "LapNumber": n, "LapTime": pd.NaT, "Time": pd.to_timedelta(t_n, unit="s"),
                          "Compound": last["Compound"], "TyreLife": np.nan,
                          "PitInTime": pd.to_timedelta(t_n, unit="s") if pitting else pd.NaT,
                          "PitOutTime": pd.NaT, "Team": "", "TrackStatus": status_now})

        # tyres for lap n+1
        nxt = g[g["LapNumber"] == n + 1]
        f_comp, f_life = tyre_name(fd.get("tyre")), _num(fd.get("tyre_life"))
        if len(nxt):
            comp, age = nxt.iloc[0]["Compound"], nxt.iloc[0]["TyreLife"]
        elif pitting:
            old = f_comp or str(last["Compound"])
            comp, age = ("HARD" if old != "HARD" else "MEDIUM"), 1
        elif f_comp and f_life and cur_lap <= n + 1:
            comp, age = f_comp, f_life + (n + 1 - cur_lap)
        else:
            comp = last["Compound"]
            age = (last["TyreLife"] + (n + 1 - last_lap)) if pd.notna(last["TyreLife"]) else np.nan
        comp = str(comp) if comp is not None and pd.notna(comp) else "MEDIUM"
        tyres[d] = (comp, max(int(round(age)), 1) if pd.notna(age) else 1)

    snap = pd.concat([keep, pd.DataFrame(extra)], ignore_index=True) if extra else keep.copy()
    if snap.empty:
        return snap, tyres, running, estimated
    for col in ("Season", "Race", "TotalLaps"):
        snap[col] = laps[col].iloc[0]
    snap["Position"] = snap.groupby("LapNumber")["Time"].rank(method="first")
    return snap, tyres, running, estimated


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


def make_headline(table: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Sort the options (best expected position first) and turn them into one call.
    Gains are paired against the first option evaluated, which is always STAY;
    an option is only called if its gain over staying out exceeds its 95 % CI —
    otherwise the honest call is "stay out", not a coin flip dressed as advice.
    """
    table = table.sort_values(["Exp. position", "P(podium)"], ascending=[True, False]).reset_index(drop=True)
    best = table.iloc[0]
    stay = table[table["Option"] == STAY].iloc[0]
    gain = stay["Exp. position"] - best["Exp. position"]
    if best["Option"] == stay["Option"] or gain <= best["Gain 95% CI"]:
        return table, "STAY OUT — no option is clearly better than the current plan"
    return table, f"{best['Option'].upper()} — +{gain:.1f} places expected vs staying out"


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
        self.reset()

    def reset(self) -> None:
        """Forget the race so far (new driver, or the replay jumped back in time)."""
        self.last_lap_done = -1
        self.last_call: dict | None = None
        self._out_reported = False
        self._t_seen = -np.inf
        self._motion: dict[str, tuple[float, float]] = {}      # driver → (distance, time it last changed)

    def _track_motion(self, payload: dict, t_now: float) -> None:
        for d, fd in ((payload.get("frame") or {}).get("drivers") or {}).items():
            dist = _num(fd.get("dist"))
            if dist is None:
                dist = _num(fd.get("fraction"))
            if dist is None:
                continue
            prev = self._motion.get(d)
            if prev is None or abs(dist - prev[0]) > 1e-3:
                self._motion[d] = (dist, t_now)

    def stopped(self, t_now: float) -> set:
        """Cars whose position has not changed for STOPPED_S of replay time (parked / retired)."""
        return {d for d, (_, since) in self._motion.items() if t_now - since > STOPPED_S}

    def update(self, payload: dict) -> dict | None:
        """A new call each time the followed driver completes a lap; None otherwise."""
        if (payload.get("session_data") or {}).get("total_laps"):
            self.total_laps = int(payload["session_data"]["total_laps"])
        if not self.circuit:
            self.circuit = (payload.get("session_info") or {}).get("event_name") or None
        t_now = payload_time(payload)
        if t_now < self._t_seen - 1.0:                      # the replay was rewound
            self.reset()
        self._t_seen = t_now
        self._track_motion(payload, t_now)
        offset = clock_offset(payload)
        mine = completed_laps(payload, self.driver, t_now, offset)
        if not mine:
            return None
        n, my_t = mine[-1]
        status = track_status(payload)
        if n >= self.total_laps - 1:                        # last lap / chequered flag: nothing to call
            self.last_lap_done = max(self.last_lap_done, n)
            return None
        if n <= self.last_lap_done:
            return self._check_out(payload, t_now, offset)
        self._out_reported = False
        if "5" in status:                                   # red flag: wait for the restart
            return None
        self.last_lap_done = n
        if n < MIN_CALL_LAP:
            return None
        laps, t_now = laps_from_payload(payload, self.circuit or "Unknown", self.total_laps, offset)
        self.last_call = self.analyse(laps, n, t_now, payload)
        return self.last_call

    def _check_out(self, payload: dict, t_now: float, offset: float) -> dict | None:
        """One 'not running' call when the followed driver stops crossing the line (never under red flag)."""
        if self._out_reported or "5" in track_status(payload):
            return None
        parked = self.driver in self.stopped(t_now)
        if not parked and self.driver not in out_by_field(crossings_from_payload(payload, t_now, offset)):
            return None                                     # cheap check first: still lapping
        laps, t_now = laps_from_payload(payload, self.circuit or "Unknown", self.total_laps, offset)
        if self.driver in running_drivers(laps, t_now, (payload.get("frame") or {}).get("drivers"),
                                          self.stopped(t_now)):
            return None
        self._out_reported = True
        lead = int(laps["LapNumber"].max())
        self.last_call = self.analyse(laps, lead, t_now, payload)
        return self.last_call

    def _blank(self, n: int, headline: str, **kw) -> dict:
        call = {"lap": n, "total_laps": self.total_laps, "driver": self.driver, "running": False,
                "tyres": None, "standings": pd.DataFrame(), "options": pd.DataFrame(columns=OPTION_COLUMNS),
                "headline": headline, "alert": None, "pace": {}, "rival_plans": {}, "track_status": "1",
                "t": None, "estimated": []}
        call.update(kw)
        return call

    def analyse(self, laps: pd.DataFrame, n: int, t_now: float, payload: dict | None = None) -> dict:
        """The call at the end of lap `n` (laps = completed laps only, see laps_from_payload)."""
        payload = payload or {}
        frame_drivers = (payload.get("frame") or {}).get("drivers") or {}
        status = track_status(payload)
        stopped = self.stopped(t_now) if "5" not in status else set()
        if n >= self.total_laps:
            return self._blank(n, "Race finished", t=t_now, track_status=status)
        snap, tyres, running, estimated = snapshot_at_lap(laps, n, t_now, frame_drivers, status, stopped)
        if snap.empty or not running:
            return self._blank(n, "No cars running", t=t_now, track_status=status)
        caps = _caps()
        used = snap.dropna(subset=["Compound"]).groupby("Driver")["Compound"].agg(lambda s: set(s) & set(DRY))
        plans = {d: default_plan(c if c in DRY else "MEDIUM", a, used.get(d, set()), n, self.total_laps, caps)
                 for d, (c, a) in tyres.items()}
        weather = (payload.get("frame") or {}).get("weather") or {}
        config = {"dnf_prob_per_lap": 0.0}
        for key in ("track_temp", "air_temp"):
            if _num(weather.get(key)) is not None:
                config[key] = float(weather[key])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setup, cfg, state, info = eng.state_from_real_race(
                snap, snap["Season"].iloc[0], snap["Race"].iloc[0], n, config,
                future_plans="none", current_tyres=tyres, plans=plans, track_status=status)
        live = eng.LiveRace(setup, cfg, state, seed=self.seed)
        standings = live.standings()
        standings["Estimated"] = standings["Driver"].isin(estimated)
        base = dict(t=t_now, track_status=status, estimated=estimated, pace=info["pace"], rival_plans=plans)
        if self.driver not in setup["drivers"]:
            return self._blank(n, f"{self.driver} is not running", standings=standings, **base)

        comp, age = tyres[self.driver]
        if comp not in DRY:
            return self._blank(n, f"{self.driver} on {comp} tyres — the models cover dry races only",
                               running=True, standings=standings,
                               tyres={"compound": comp, "age": age, "rule_met": True}, **base)
        my_used = used.get(self.driver, set())
        rule_met_now = len(my_used | {comp}) >= 2
        nxt = n + 1
        options = {STAY: plans[self.driver]}
        for c in DRY:
            if c == comp and not rule_met_now and len(my_used) < 2:
                continue                                   # would not satisfy the two-compound rule
            options[f"Box now → {c}"] = plan_after_box(nxt, c, self.total_laps, caps)
        best_alt = "HARD" if comp != "HARD" else "MEDIUM"
        if nxt + 3 < self.total_laps - 1:
            options[f"Box in 3 laps → {best_alt}"] = plan_after_box(nxt + 3, best_alt, self.total_laps, caps)
        if rule_met_now and plans[self.driver]:
            options["No more stops"] = []
        table = live.evaluate(self.driver, options, n_sims=self.n_sims, seed=self.seed + n)
        table, headline = make_headline(table)
        return {"lap": n, "total_laps": self.total_laps, "driver": self.driver, "running": True,
                "tyres": {"compound": comp, "age": age, "rule_met": rule_met_now},
                "standings": standings, "options": table, "headline": headline,
                "alert": self._undercut_alert(standings, cfg, n, sc_active="4" in status), **base}

    def _undercut_alert(self, standings: pd.DataFrame, cfg: dict, n: int, sc_active: bool = False) -> dict | None:
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
            r = undercut_analysis(chaser, leader, gap, n,
                                  {"track_temp": cfg["track_temp"], "air_temp": cfg["air_temp"],
                                   "circuit": self.circuit}, fresh, fresh, sc_active=sc_active)
        return {"target": ahead["Driver"], "gap": round(gap, 2), "verdict": r["verdict"],
                "min_gap": r["min_gap"], "p_overtake": r["confidence"],
                "in_drs_window": r["laps_in_drs_window"] > 0}
