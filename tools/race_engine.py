"""
Lap-by-lap race engine — the core shared by the Monte Carlo simulator, the What-If
tool, the live "Race Engineer" and the backtests.

One idea: a race is a STATE (cumulative times, tyres, planned stops, Safety Car…)
for S parallel simulations × D drivers, and `step()` advances every simulation
by ONE lap. Everything else is built on top:

    simulate a race from the grid   → initial_state + step × L        (race_simulator)
    "what if I box now?"            → copy the live state S times, change one
                                      driver's plan, step to the flag (LiveRace.evaluate)
    replay-then-take-control        → state from a real race at lap N (from_real_race)

Vectorised over simulations; Module 1 is called once per setup (lookup table),
Module 2 once per lap for all simulations.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tools.shared import (BASE_LAPTIME, COMPOUND_NAMES, DRS_WINDOW, FUEL_S_PER_LAP, NOISE_STD,
                          PIT_COST_GREEN, PIT_COST_SC, encode_circuit, encode_compound,
                          encode_team, lap_time_from_delta, load_models, model_features,
                          predict_lap_deltas, predict_overtake_probability, LAP_FEATURES)

START_GAP = 0.25          # s between consecutive grid slots after lap 1
MIN_FOLLOW_GAP = 0.30     # s — can't cross the line closer than this behind a car you failed to pass
DIRTY_AIR_PENALTY = 0.20  # s/lap when within DRS_WINDOW of the car ahead
PIT_NOISE_STD = 0.8       # s — stop-to-stop variation
# s/lap — how far a driver's pace can drift from what has been seen so far. Calibrated on
# 36 real races (tools/backtest.py): at 0.25 the simulated 10–90 % range contains the real
# finishing position 79.5 % of the time (calibrated); 0.12 gave 67 % (overconfident).
RACE_FORM_STD = 0.25
START_STD = 0.6           # s — lap-1 chaos: launch, first-corner positioning
SC_LAP_FACTOR = 1.35      # SC lap ≈ 35 % slower than a racing lap
SC_COMPRESS = 0.35        # each SC lap keeps 35 % of every gap…
SC_MIN_GAP = 0.6          # …but cars queue no closer than this
SC_PIT_RATIO = PIT_COST_SC / PIT_COST_GREEN   # a stop under SC costs ~72 % of a green one
MAX_STOPS = 4
NO_STOP = 10_000
BIG = 1e9

DEFAULT_CONFIG = {
    "total_laps": 52,
    "track_temp": 35.0,
    "air_temp": 25.0,
    "rainfall": 0,
    "sc_prob_per_lap": 0.012,
    "sc_min_laps": 3,
    "sc_max_laps": 5,
    "sc_pit_window": 8,            # box under SC if planned stop is ≤ N laps away (0 = never)
    "dnf_prob_per_lap": 0.0004,    # ≈ 2 % per car per race
    "base_laptime": BASE_LAPTIME,
    "pit_loss": PIT_COST_GREEN,
    "race_form_std": RACE_FORM_STD,
}


def resolve_config(config: dict | None) -> dict:
    """DEFAULT_CONFIG ← circuit profile (if the circuit is known) ← explicit config."""
    from tools.circuit_profiles import profile_for
    config = dict(config or {})
    prof = profile_for(config.get("circuit")) or {}
    from_profile = {}
    if prof:
        from_profile["base_laptime"] = prof["base_lap_s"]
        from_profile["sc_prob_per_lap"] = prof["sc_per_lap"]
        from_profile["total_laps"] = prof["laps"]
        if prof.get("pit_loss_s"):
            from_profile["pit_loss"] = prof["pit_loss_s"]
    return {**DEFAULT_CONFIG, **from_profile, **config}


# ── Randomness ───────────────────────────────────────────────────────────────
@dataclass
class Draws:
    """All randomness for one batch, pre-drawn so two scenarios can share it (CRN)."""
    lap_noise: np.ndarray    # (S, L, D)
    pit_noise: np.ndarray    # (S, L, D)
    sc_u: np.ndarray         # (S, L)
    sc_len: np.ndarray       # (S, L)
    ovt_u: np.ndarray        # (S, L, D)  indexed by running position
    dnf_u: np.ndarray        # (S, L, D)
    form: np.ndarray         # (S, D)
    start: np.ndarray        # (S, D)


def draw(n_sims: int, n_laps: int, n_drivers: int, cfg: dict, seed: int) -> Draws:
    rng = np.random.default_rng(seed)
    S, L, D = n_sims, max(n_laps, 1), n_drivers
    return Draws(
        lap_noise=rng.normal(0, NOISE_STD, (S, L, D)),
        pit_noise=rng.normal(0, PIT_NOISE_STD, (S, L, D)),
        sc_u=rng.random((S, L)),
        sc_len=rng.integers(cfg["sc_min_laps"], cfg["sc_max_laps"] + 1, (S, L)),
        ovt_u=rng.random((S, L, D)),
        dnf_u=rng.random((S, L, D)),
        form=rng.normal(0, cfg.get("race_form_std", RACE_FORM_STD), (S, D)),
        start=rng.normal(0, START_STD, (S, D)),
    )


# ── Setup: everything that does not change during a race ─────────────────────
def _delta_table(teams_enc: np.ndarray, n_laps: int, max_age: int, cfg: dict) -> np.ndarray:
    """LapDelta for every (team, compound, tyre age, lap) → array [T, 3, max_age+1, n_laps+1]."""
    T = len(teams_enc)
    t, c, a, l = np.meshgrid(np.arange(T), np.arange(3), np.arange(1, max_age + 1),
                             np.arange(1, n_laps + 1), indexing="ij")
    rows = pd.DataFrame({
        "CompoundEncoded": c.ravel(), "TyreLife": a.ravel(), "TeamEncoded": teams_enc[t.ravel()],
        "LapNumber": l.ravel(), "TrackTemp": cfg["track_temp"], "AirTemp": cfg["air_temp"],
        "RainfallEncoded": cfg.get("rainfall", 0),
        "CircuitEncoded": encode_circuit(cfg.get("circuit")),
    })
    table = np.zeros((T, 3, max_age + 1, n_laps + 1))
    table[t.ravel(), c.ravel(), a.ravel(), l.ravel()] = predict_lap_deltas(rows)
    return table


def model_has_fuel() -> bool:
    """V2 lap models predict deltas that include fuel; V3+ deltas are fuel-corrected."""
    lap_model = load_models()[0]
    return "LapNumber" in model_features(lap_model, LAP_FEATURES)


def build_setup(drivers, teams, pace, start_comp, start_age, plans, cfg: dict) -> dict:
    """
    plans: one list of (lap, compound) per driver — the stops still to come, in order.
    A stop on lap X means lap X is the in-lap (it carries the pit loss) and the new
    tyres are used from lap X+1.
    """
    D, L = len(drivers), cfg["total_laps"]
    uniq = sorted(set(teams))
    stop_laps = np.full((D, MAX_STOPS), NO_STOP)
    stop_comp = np.zeros((D, MAX_STOPS), dtype=int)
    for i, plan in enumerate(plans):
        for k, (lap, comp) in enumerate(sorted(plan)[:MAX_STOPS]):
            stop_laps[i, k], stop_comp[i, k] = int(lap), encode_compound(comp)
    max_age = L + int(np.max(start_age)) + 1
    return {
        "drivers": list(drivers), "teams": list(teams), "D": D, "L": L,
        "team_idx": np.array([uniq.index(t) for t in teams]),
        "pace": np.asarray(pace, dtype=float),
        "start_comp": np.array([encode_compound(c) for c in start_comp]),
        "start_age": np.asarray(start_age, dtype=int),
        "stop_laps": stop_laps, "stop_comp": stop_comp, "max_age": max_age,
        "table": _delta_table(np.array([encode_team(t) for t in uniq]), L, max_age, cfg),
        "fuel_in_model": model_has_fuel(),
    }


# ── State ────────────────────────────────────────────────────────────────────
def initial_state(setup: dict, n_sims: int, start_offsets: np.ndarray | None = None) -> dict:
    """State before lap 1 (or before lap `lap + 1` once fields are overwritten)."""
    S, D = n_sims, setup["D"]
    cum = np.broadcast_to(np.arange(D) * START_GAP, (S, D)).astype(float).copy()
    if start_offsets is not None:
        cum = cum + start_offsets
    return {
        "lap": 0, "cum": cum,
        "comp": np.broadcast_to(setup["start_comp"], (S, D)).copy(),
        "age": np.broadcast_to(setup["start_age"], (S, D)).copy(),
        "nxt": np.zeros((S, D), dtype=int),
        "stop_laps": np.broadcast_to(setup["stop_laps"], (S, D, MAX_STOPS)).copy(),
        "stop_comp": np.broadcast_to(setup["stop_comp"], (S, D, MAX_STOPS)).copy(),
        "retired": np.zeros((S, D), dtype=bool),
        "sc_left": np.zeros(S, dtype=int),
        "sc_count": np.zeros(S, dtype=int),
        "overtakes": np.zeros(S, dtype=int),
        "n_stops": np.zeros((S, D), dtype=int),
    }


def broadcast_state(state: dict, n_sims: int) -> dict:
    """Copy a 1-simulation state into n_sims identical simulations."""
    out = {}
    for k, v in state.items():
        out[k] = v if k == "lap" else np.repeat(np.asarray(v)[:1], n_sims, axis=0).copy()
    return out


def positions(state: dict) -> np.ndarray:
    """(S, D) running positions, retired cars last."""
    key = np.where(state["retired"], BIG + state["cum"], state["cum"])
    return np.argsort(np.argsort(key, 1), 1) + 1


# ── One lap ──────────────────────────────────────────────────────────────────
def step(state: dict, setup: dict, cfg: dict, dr: Draws, li: int) -> dict:
    """
    Advance every simulation by one lap (in place). `li` indexes the lap inside
    the Draws arrays. Returns per-lap events for the live view.
    """
    lap = state["lap"] + 1
    L, D = setup["L"], setup["D"]
    S = state["cum"].shape[0]
    sims, drv = np.arange(S)[:, None], np.arange(D)[None, :]
    base = cfg.get("base_laptime", BASE_LAPTIME)
    circuit_enc = encode_circuit(cfg.get("circuit"))
    cum, comp, age, nxt, retired = (state["cum"], state["comp"], state["age"],
                                    state["nxt"], state["retired"])

    # --- Safety Car
    deploy = (state["sc_left"] == 0) & (dr.sc_u[:, li] < cfg["sc_prob_per_lap"]) & (lap > 1) & (lap < L - 1)
    state["sc_left"] = np.where(deploy, dr.sc_len[:, li], state["sc_left"])
    state["sc_count"] += deploy
    sc = state["sc_left"] > 0

    # --- Pit decisions (planned, or opportunistic under SC)
    k = np.minimum(nxt, MAX_STOPS - 1)
    planned = np.where(nxt >= MAX_STOPS, NO_STOP, state["stop_laps"][sims, drv, k])
    has_stop = planned < NO_STOP
    pit = has_stop & (planned == lap)
    if cfg.get("sc_pit_window", 0) > 0:
        pit |= has_stop & sc[:, None] & (planned - lap <= cfg["sc_pit_window"]) & (planned > lap)
    pit &= ~retired
    new_comp = state["stop_comp"][sims, drv, k]

    # --- Lap times
    delta = setup["table"][setup["team_idx"][None, :], comp, np.minimum(age, setup["max_age"]), lap]
    racing = lap_time_from_delta(delta, setup["pace"][None, :], base) + dr.form + dr.lap_noise[:, li]
    if not setup.get("fuel_in_model", True):           # V3 deltas are fuel-free: add it back (same for all)
        racing = racing + cfg.get("fuel_s_per_lap", FUEL_S_PER_LAP) * (L - lap)
    sc_time = base * SC_LAP_FACTOR + 0.1 * dr.lap_noise[:, li]
    lt = np.where(sc[:, None], sc_time, racing)
    pit_loss = cfg.get("pit_loss", PIT_COST_GREEN)
    pit_cost = np.where(sc, pit_loss * SC_PIT_RATIO, pit_loss)[:, None] + dr.pit_noise[:, li]
    lt = lt + np.where(pit, pit_cost, 0.0)

    # --- Running order at the start of the lap
    order = np.argsort(np.where(retired, BIG, cum), axis=1)
    cum_o = np.take_along_axis(cum, order, 1)
    gap_prev = np.diff(cum_o, axis=1, prepend=cum_o[:, :1])         # gap to car ahead (0 for leader)
    lt_o = np.take_along_axis(lt, order, 1)
    close = (np.arange(D)[None, :] > 0) & (gap_prev < DRS_WINDOW) & ~sc[:, None]
    lt_o = lt_o + np.where(close, DIRTY_AIR_PENALTY, 0.0)

    # Module 2 — one call for every in-range pair in every simulation
    comp_o, age_o = np.take_along_axis(comp, order, 1), np.take_along_axis(age, order, 1)
    prob = np.zeros((S, D))
    if close.any():
        si, pi = np.nonzero(close)
        prob[si, pi] = predict_overtake_probability(
            gap_prev[si, pi], comp_o[si, pi], comp_o[si, pi - 1], age_o[si, pi],
            age_o[si, pi - 1], np.full(len(si), lap), pi + 1, circuit=circuit_enc)

    # sequential pass down the order: held up unless the pass succeeds
    cand = cum_o + lt_o
    pit_o = np.take_along_axis(pit, order, 1)
    ret_o = np.take_along_axis(retired, order, 1)
    new = cand.copy()
    passed_o = np.zeros((S, D), dtype=bool)
    front = np.where(pit_o[:, 0] | ret_o[:, 0], -BIG, cand[:, 0])
    for i in range(1, D):
        c = cand[:, i]
        active = ~ret_o[:, i] & ~pit_o[:, i]
        blocked = active & (c < front + MIN_FOLLOW_GAP)
        passed = blocked & ~sc & (gap_prev[:, i] < DRS_WINDOW) & (dr.ovt_u[:, li, i] < prob[:, i])
        passed_o[:, i] = passed
        state["overtakes"] += passed
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

    np.put_along_axis(cum, order, new, 1)
    passed_d = np.zeros((S, D), dtype=bool)
    np.put_along_axis(passed_d, order, passed_o, 1)

    # --- Tyres, stops, retirements
    state["comp"] = np.where(pit, new_comp, comp)
    state["age"] = np.where(pit, 1, age + 1)
    state["nxt"] = nxt + pit
    state["n_stops"] += pit
    newly_out = ~retired & (dr.dnf_u[:, li] < cfg.get("dnf_prob_per_lap", 0.0))
    state["retired"] = retired | newly_out
    state["sc_left"] = np.maximum(state["sc_left"] - 1, 0)
    state["lap"] = lap
    return {"lap": lap, "sc": sc, "sc_deployed": deploy, "pit": pit, "passed": passed_d,
            "retired_now": newly_out, "lap_time": lt}


def run_to_flag(state: dict, setup: dict, cfg: dict, dr: Draws, record_trace: bool = False) -> dict:
    """Step until the chequered flag. Draws are indexed from the current lap."""
    start = state["lap"]
    trace_pos, trace_gap = [], []
    for li in range(setup["L"] - start):
        step(state, setup, cfg, dr, li)
        if record_trace:
            ret, cum = state["retired"][0], state["cum"][0]
            trace_pos.append(positions(state)[0])
            trace_gap.append(np.where(ret, np.nan, cum - cum[~ret].min()))
    return {"positions": positions(state), "retired": state["retired"], "race_time": state["cum"],
            "sc_count": state["sc_count"], "overtakes": state["overtakes"], "n_stops": state["n_stops"],
            "trace_pos": np.array(trace_pos), "trace_gap": np.array(trace_gap)}


# ── A live race: one realised race you can step, plus "what if" look-ahead ────
class LiveRace:
    """
    One race that unfolds lap by lap (the "real" one on the pit wall), plus
    Monte Carlo look-ahead from its current state.

        race = LiveRace.from_grid(grid, {"circuit": "Bahrain Grand Prix"})
        race.step()                       # one lap
        race.standings()                  # timing tower
        race.evaluate("VER", {"Box now → HARD": [(race.lap + 1, "HARD")],
                              "Stay out": race.plan("VER")})
    """

    def __init__(self, setup: dict, cfg: dict, state: dict, seed: int = 0):
        self.setup, self.cfg, self.state = setup, cfg, state
        self.seed = seed
        self._dr = draw(1, setup["L"] - state["lap"], setup["D"], cfg, seed)
        self._li = 0
        self.events: list[dict] = []

    # constructors ---------------------------------------------------------------
    @classmethod
    def from_grid(cls, grid: pd.DataFrame, config: dict | None = None, seed: int = 0):
        from tools.race_simulator import plans_from_grid
        cfg = resolve_config(config)
        g = grid.sort_values("Grid").reset_index(drop=True)
        setup = build_setup(g["Driver"], g["Team"], g["PaceFactor"], g["StartCompound"],
                            g["StartAge"], plans_from_grid(g), cfg)
        start = np.abs(np.random.default_rng(seed + 1).normal(0, START_STD, (1, setup["D"])))
        return cls(setup, cfg, initial_state(setup, 1, start), seed)

    @classmethod
    def from_real_race(cls, laps: pd.DataFrame, season: int, race: str, at_lap: int,
                       config: dict | None = None, seed: int = 0, future_plans: str = "actual"):
        """State of a real race at the end of `at_lap` (see state_from_real_race)."""
        setup, cfg, state, info = state_from_real_race(laps, season, race, at_lap, config, future_plans)
        live = cls(setup, cfg, state, seed)
        live.info = info
        return live

    # stepping ---------------------------------------------------------------------
    @property
    def lap(self) -> int:
        return int(self.state["lap"])

    @property
    def finished(self) -> bool:
        return self.lap >= self.setup["L"]

    def step(self, n: int = 1) -> list[dict]:
        out = []
        for _ in range(n):
            if self.finished:
                break
            ev = step(self.state, self.setup, self.cfg, self._dr, self._li)
            self._li += 1
            out.append(self._describe(ev))
        self.events.extend(out)
        return out

    def _describe(self, ev: dict) -> dict:
        d = self.setup["drivers"]
        return {"lap": ev["lap"],
                "safety_car": bool(ev["sc"][0]), "sc_deployed": bool(ev["sc_deployed"][0]),
                "pitted": [d[i] for i in np.nonzero(ev["pit"][0])[0]],
                "overtakes": [d[i] for i in np.nonzero(ev["passed"][0])[0]],
                "retired": [d[i] for i in np.nonzero(ev["retired_now"][0])[0]]}

    # reading the state ------------------------------------------------------------
    def standings(self) -> pd.DataFrame:
        s = self.state
        cum, ret = s["cum"][0], s["retired"][0]
        pos = positions(s)[0]
        lead = cum[~ret].min() if (~ret).any() else 0.0
        df = pd.DataFrame({
            "Pos": pos, "Driver": self.setup["drivers"], "Team": self.setup["teams"],
            "Gap": np.where(ret, np.nan, cum - lead),
            "Compound": [COMPOUND_NAMES[int(c)] for c in s["comp"][0]],
            "TyreAge": s["age"][0] - 1, "Stops": s["n_stops"][0], "Retired": ret,
        }).sort_values("Pos").reset_index(drop=True)
        df["Interval"] = df["Gap"].diff()
        return df

    def plan(self, driver: str) -> list[tuple[int, str]]:
        """Remaining planned stops of a driver."""
        i = self.setup["drivers"].index(driver)
        nxt = int(self.state["nxt"][0, i])
        laps, comps = self.state["stop_laps"][0, i], self.state["stop_comp"][0, i]
        return [(int(laps[k]), COMPOUND_NAMES[int(comps[k])]) for k in range(nxt, MAX_STOPS)
                if laps[k] < NO_STOP and laps[k] > self.lap]

    def set_plan(self, driver: str, plan: list[tuple[int, str]]) -> None:
        _set_plan(self.state, self.setup["drivers"].index(driver), plan)

    def box(self, driver: str, compound: str) -> None:
        """Pit at the end of the next lap (next lap is the in-lap); later plan unchanged."""
        nxt_lap = self.lap + 1
        rest = [(l, c) for l, c in self.plan(driver) if l > nxt_lap]
        self.set_plan(driver, [(nxt_lap, compound)] + rest)

    # look-ahead -------------------------------------------------------------------
    def evaluate(self, driver: str, options: dict[str, list[tuple[int, str]]],
                 n_sims: int = 500, seed: int | None = None) -> pd.DataFrame:
        """
        Finishing-position distribution of `driver` for each option (a plan of the
        remaining stops), all options on the SAME random draws — paired comparison
        against the first option.
        """
        i = self.setup["drivers"].index(driver)
        remaining = self.setup["L"] - self.lap
        dr = draw(n_sims, remaining, self.setup["D"], self.cfg, self.seed + 1000 + self.lap
                  if seed is None else seed)
        # Every option runs in ONE batch: block k of the simulations follows option k, and every
        # block uses the same draws (tiled) — identical to looping over the options, but the
        # models are called once per lap instead of once per lap per option.
        names = list(options)
        K = len(names)
        st = broadcast_state(self.state, n_sims * K)
        for k, name in enumerate(names):
            _set_plan(st, i, options[name], rows=slice(k * n_sims, (k + 1) * n_sims))
        pos_all = run_to_flag(st, self.setup, self.cfg, tile_draws(dr, K))["positions"][:, i].reshape(K, n_sims)
        rows, ref = [], None
        for name, pos in zip(names, pos_all):
            if ref is None:
                ref = pos
            diff = ref - pos                                          # >0: this option finishes higher
            rows.append({"Option": name, "P(win)": (pos == 1).mean(), "P(podium)": (pos <= 3).mean(),
                         "P(points)": (pos <= 10).mean(), "Exp. position": pos.mean(),
                         "Gain vs first (places)": diff.mean(),
                         "Gain 95% CI": 1.96 * diff.std(ddof=1) / np.sqrt(n_sims) if n_sims > 1 else np.nan})
        return pd.DataFrame(rows)


def tile_draws(dr: Draws, k: int) -> Draws:
    """The same draws repeated k times along the simulation axis (one block per option)."""
    if k == 1:
        return dr
    return Draws(**{f: np.concatenate([getattr(dr, f)] * k, axis=0) for f in dr.__dataclass_fields__})


def _set_plan(state: dict, i: int, plan: list[tuple[int, str]], rows=slice(None)) -> None:
    """Replace the remaining stops of driver i (in simulations `rows`, default all) with `plan`."""
    idx = np.arange(state["cum"].shape[0])[rows]
    nxt = state["nxt"][idx, i]
    plan = sorted(plan)
    for k0 in np.unique(nxt):                    # usually one value: every simulation is at the same stop
        s = idx[nxt == k0]
        state["stop_laps"][s, i, k0:] = NO_STOP
        for j, (lap, comp) in enumerate(plan):
            if k0 + j < MAX_STOPS:
                state["stop_laps"][s, i, k0 + j] = int(lap)
                state["stop_comp"][s, i, k0 + j] = encode_compound(comp)


# ── From a real race ─────────────────────────────────────────────────────────
def _sec(s):
    return pd.to_timedelta(s).dt.total_seconds()


def state_from_real_race(laps: pd.DataFrame, season: int, race: str, at_lap: int,
                         config: dict | None = None, future_plans: str = "actual",
                         current_tyres: dict | None = None, plans: dict | None = None,
                         track_status: str | None = None):
    """
    Engine state at the END of lap `at_lap` of a real race (laps = FastF1-shaped table).

    pace: what a pit wall knows at that moment — each driver's median racing lap so
          far, corrected for fuel and for tyre state with Module 1, relative to the
          quickest. Nothing after `at_lap` is used.
    future_plans: "actual" → the stops each driver really made afterwards (backtests);
                  "none"   → no more stops planned (you decide).
    current_tyres: {driver: (compound, tyre_life)} — live use, when lap at_lap+1 does not exist yet.
    plans: {driver: [(lap, compound), …]} — overrides the remaining stops per driver.
    track_status: live track status right now (FastF1 code, "4" = Safety Car). When given,
                  it decides whether the SC is out instead of the leader's lap row.
    """
    r = laps[(laps["Season"] == season) & (laps["Race"] == race)].copy()
    if r.empty:
        raise ValueError(f"No laps for {season} {race}")
    L = int(r["TotalLaps"].iloc[0])
    if not 1 <= at_lap < L:
        raise ValueError(f"at_lap must be in 1..{L - 1}")
    r["LT"] = _sec(r["LapTime"])
    r["T"] = _sec(r["Time"])
    now = r[r["LapNumber"] == at_lap].dropna(subset=["T", "Position"]).sort_values("Position")
    nxt_rows = r[r["LapNumber"] == at_lap + 1].set_index("Driver")
    if len(nxt_rows):                                                    # history: still running next lap
        drivers = [d for d in now["Driver"] if d in nxt_rows.index]
    else:                                                                # live: lap at_lap+1 not run yet
        drivers = list(now["Driver"])
    now = now.set_index("Driver").loc[drivers]

    cfg = resolve_config({**(config or {}), "circuit": race, "total_laps": L})
    def tyre_now(d):
        if current_tyres and d in current_tyres:
            c, a = current_tyres[d]
        elif d in nxt_rows.index:
            c, a = nxt_rows.loc[d, "Compound"], nxt_rows.loc[d, "TyreLife"]
        else:                                    # live, no pit on the last lap: one lap older
            c, a = now.loc[d, "Compound"], (now.loc[d, "TyreLife"] + 1 if pd.notna(now.loc[d, "TyreLife"]) else 1)
        c = str(c)
        return (c if c in ("SOFT", "MEDIUM", "HARD") else "MEDIUM"), (int(a) if pd.notna(a) else 1)
    comp, age = map(list, zip(*[tyre_now(d) for d in drivers]))

    # pace known at lap N: fuel- and tyre-corrected median racing lap per driver
    hist = r[(r["LapNumber"] >= 2) & (r["LapNumber"] <= at_lap) & (r["TrackStatus"].astype(str) == "1")
             & r["PitInTime"].isna() & r["PitOutTime"].isna() & r["Compound"].isin(["SOFT", "MEDIUM", "HARD"])
             & r["Driver"].isin(drivers)].dropna(subset=["LT", "TyreLife"])
    hist = hist[hist["LT"] < hist["LT"].median() + 5]
    feats = pd.DataFrame({"CompoundEncoded": hist["Compound"].map({"SOFT": 0, "MEDIUM": 1, "HARD": 2}),
                          "TyreLife": hist["TyreLife"], "TeamEncoded": 0, "LapNumber": hist["LapNumber"],
                          "TrackTemp": cfg["track_temp"], "AirTemp": cfg["air_temp"], "RainfallEncoded": 0,
                          "CircuitEncoded": encode_circuit(race)})
    fuel_in_model = model_has_fuel()
    corr = hist["LT"] - predict_lap_deltas(feats)
    if not fuel_in_model:
        corr = corr - cfg.get("fuel_s_per_lap", FUEL_S_PER_LAP) * (L - hist["LapNumber"])
    med = corr.groupby(hist["Driver"]).median()
    best = med.min() if len(med) else cfg["base_laptime"]
    base = float(best) if len(med) else cfg["base_laptime"]
    pace = [1 + (med.get(d, med.median() if len(med) else best) - best) / base for d in drivers]
    cfg["base_laptime"] = base

    plans_out = []
    for d in drivers:
        if plans and d in plans:
            plans_out.append(list(plans[d]))
        elif future_plans == "actual":
            dl = r[(r["Driver"] == d) & (r["LapNumber"] > at_lap)].sort_values("LapNumber")
            stops = []
            for _, row in dl[dl["PitInTime"].notna()].iterrows():
                after = dl[dl["LapNumber"] == row["LapNumber"] + 1]
                c = str(after["Compound"].iloc[0]) if len(after) else "HARD"
                if row["LapNumber"] < L:
                    stops.append((int(row["LapNumber"]), c if c in ("SOFT", "MEDIUM", "HARD") else "HARD"))
            plans_out.append(stops)
        else:
            plans_out.append([])

    setup = build_setup(drivers, [str(now.loc[d, "Team"]) for d in drivers], pace, comp, age, plans_out, cfg)
    setup["fuel_in_model"] = fuel_in_model
    state = initial_state(setup, 1)
    state["lap"] = at_lap
    state["cum"] = (now["T"].to_numpy(float) - now["T"].min())[None, :]
    stops_done = r[(r["LapNumber"] <= at_lap) & r["PitInTime"].notna()].groupby("Driver").size()
    state["n_stops"] = np.array([[int(stops_done.get(d, 0)) for d in drivers]])
    if track_status is None:
        status_row = nxt_rows.loc[drivers[0]] if drivers[0] in nxt_rows.index else now.loc[drivers[0]]
        track_status = str(status_row["TrackStatus"])
    if str(track_status).find("4") >= 0:                                 # SC already out
        state["sc_left"] = np.array([2])
    info = {"season": season, "race": race, "at_lap": at_lap, "total_laps": L,
            "pace": dict(zip(drivers, np.round(pace, 4))), "base_laptime": base}
    return setup, cfg, state, info


def actual_result(laps: pd.DataFrame, season: int, race: str) -> pd.Series:
    """Finishing order of drivers who were running at the flag (lapped cars included)."""
    r = laps[(laps["Season"] == season) & (laps["Race"] == race)]
    L = int(r["TotalLaps"].iloc[0])
    last = r.sort_values("LapNumber").groupby("Driver").tail(1)
    running = last[last["LapNumber"] >= L - 3].copy()                    # finished or lapped ≤ 3
    running["key"] = -running["LapNumber"] * 1e6 + _sec(running["Time"])
    return running.sort_values("key")["Driver"].reset_index(drop=True).pipe(
        lambda s: pd.Series(np.arange(1, len(s) + 1), index=s.values))
