"""Lap-by-lap engine: stepping, plans, look-ahead and starting from a real race."""
import numpy as np
import pandas as pd
import pytest

from tools import race_engine as eng
from tools.race_simulator import grid_from_qualifying

from test_training import synthetic_race


@pytest.fixture
def grid():
    teams = ["Red Bull Racing", "Red Bull Racing", "Ferrari", "Ferrari", "McLaren", "McLaren"]
    q = pd.DataFrame({"Driver": list("ABCDEF"), "Team": teams,
                      "QualyTime": 90 + np.arange(6) * 0.2, "QualyRank": np.arange(1, 7)})
    return grid_from_qualifying(q, 30, "MEDIUM", "HARD", 14)


def test_stepping_live_race_matches_running_to_the_flag(grid):
    live = eng.LiveRace.from_grid(grid, {"total_laps": 30}, seed=3)
    twin = eng.LiveRace.from_grid(grid, {"total_laps": 30}, seed=3)
    while not live.finished:
        live.step()
    res = eng.run_to_flag(twin.state, twin.setup, twin.cfg, twin._dr)
    assert (eng.positions(live.state) == res["positions"]).all()
    assert live.lap == 30 and len(live.events) == 30


def test_events_report_pit_stops(grid):
    live = eng.LiveRace.from_grid(grid, {"total_laps": 30, "sc_prob_per_lap": 0, "dnf_prob_per_lap": 0})
    ev = live.step(14)[-1]
    assert set(ev["pitted"]) == set("ABCDEF")           # everyone planned lap 14
    st = live.standings()
    assert (st["Compound"] == "HARD").all() and (st["TyreAge"] == 0).all()


def test_box_and_plan(grid):
    live = eng.LiveRace.from_grid(grid, {"total_laps": 30})
    live.step(5)
    assert live.plan("A") == [(14, "HARD")]
    live.box("A", "SOFT")
    assert live.plan("A") == [(6, "SOFT"), (14, "HARD")]


def test_evaluate_is_paired(grid):
    live = eng.LiveRace.from_grid(grid, {"total_laps": 30})
    live.step(8)
    same = live.plan("C")
    out = live.evaluate("C", {"plan": same, "plan again": same}, n_sims=60)
    assert out.loc[1, "Gain vs first (places)"] == 0 and out.loc[1, "Gain 95% CI"] == 0
    silly = live.evaluate("C", {"plan": same, "three extra stops": same + [(10, "SOFT"), (12, "SOFT"), (16, "SOFT")]},
                          n_sims=60)
    assert silly.loc[1, "Exp. position"] > silly.loc[0, "Exp. position"]


def test_state_from_real_race():
    laps = synthetic_race(2024, "Test Grand Prix", n_laps=40, seed=1)
    setup, cfg, state, info = eng.state_from_real_race(laps, 2024, "Test Grand Prix", at_lap=10)
    real = laps[laps["LapNumber"] == 10].sort_values("Position")["Driver"].tolist()
    assert setup["drivers"] == real                                   # running order at lap 10
    assert (np.diff(state["cum"][0]) >= 0).all() and state["lap"] == 10
    # the stops still to come are the ones that really happened
    d0 = real[0]
    future = laps[(laps.Driver == d0) & (laps.LapNumber > 10) & laps.PitInTime.notna()]["LapNumber"].tolist()
    i = setup["drivers"].index(d0)
    assert [l for l in state["stop_laps"][0, i] if l < eng.NO_STOP] == [int(x) for x in future]
    assert pytest.approx(min(info["pace"].values())) == 1.0
    live = eng.LiveRace.from_real_race(laps, 2024, "Test Grand Prix", at_lap=10)
    live.step(30)
    assert live.finished
    res = eng.actual_result(laps, 2024, "Test Grand Prix")
    assert sorted(res.values) == list(range(1, 21))


def test_circuit_profile_feeds_the_config():
    cfg = eng.resolve_config({"circuit": "Singapore Grand Prix"})
    assert cfg["pit_loss"] > 25 and cfg["total_laps"] == 62
    assert eng.resolve_config({"circuit": "Singapore Grand Prix", "total_laps": 40})["total_laps"] == 40
    assert eng.resolve_config({"circuit": "Nowhere"})["pit_loss"] == eng.DEFAULT_CONFIG["pit_loss"]


def test_backtest_runs_on_a_race():
    from tools.backtest import backtest_race
    laps = synthetic_race(2024, "Test Grand Prix", n_laps=40, seed=2)
    res = backtest_race(laps, 2024, "Test Grand Prix", n_sims=50)
    assert res is not None and res["drivers"] == 20
    assert -1 <= res["rho_sim"] <= 1 and 0 <= res["cover80"] <= 1


def test_batched_evaluate_equals_one_run_per_option(grid):
    """evaluate() races every option in one batch; it must equal racing them one by one."""
    live = eng.LiveRace.from_grid(grid, {"total_laps": 30}, seed=4)
    live.step(6)
    options = {"plan": live.plan("B"), "box now": [(7, "HARD")], "two stops": [(9, "SOFT"), (20, "HARD")]}
    batched = live.evaluate("B", options, n_sims=40, seed=11)
    i = live.setup["drivers"].index("B")
    dr = eng.draw(40, live.setup["L"] - live.lap, live.setup["D"], live.cfg, 11)
    for k, plan in enumerate(options.values()):
        st = eng.broadcast_state(live.state, 40)
        eng._set_plan(st, i, plan)
        pos = eng.run_to_flag(st, live.setup, live.cfg, dr)["positions"][:, i]
        assert batched.loc[k, "Exp. position"] == pytest.approx(pos.mean())


def test_state_from_real_race_uses_nothing_after_lap_n():
    """Who retires on lap N+1 and lap N+1's track status are the future: they must not matter."""
    from race_factory import make_race
    laps = make_race(30, retire={"D07": 10}, sc_laps=(11,), season=2024, race="Test Grand Prix")
    setup, cfg, state, info = eng.state_from_real_race(laps, 2024, "Test Grand Prix", at_lap=10)
    assert "D07" in setup["drivers"]                         # completed lap 10; retiring is the future
    assert state["sc_left"][0] == 0                          # the SC of lap 11 is not known at lap 10


def test_pit_on_lap_n_owes_its_out_lap():
    from race_factory import make_race
    laps = make_race(30, pit_split=(4.0, 19.0), season=2024, race="Test Grand Prix")   # D04 pits on lap 14
    setup, cfg, state, info = eng.state_from_real_race(laps, 2024, "Test Grand Prix", at_lap=14)
    assert cfg["pit_loss"] - 6 < info["rejoin_s"]["D04"] < cfg["pit_loss"] - 2
    assert set(info["rejoin_s"]) == set(laps[(laps.LapNumber == 14) & laps.PitInTime.notna()]["Driver"])


# ── V3.2: the day's degradation is uncertain, and the same for every car ─────
def test_draws_of_a_seed_did_not_change_when_deg_z_was_added():
    a, b = eng.draw(4, 6, 5, eng.DEFAULT_CONFIG, 11), eng.draw(4, 6, 5, eng.DEFAULT_CONFIG, 11)
    assert a.deg_z.shape == (4,) and (a.deg_z == b.deg_z).all()
    rng = np.random.default_rng(11)                                 # first draw of the stream is still lap_noise
    assert np.allclose(a.lap_noise, rng.normal(0, eng.NOISE_STD, (4, 6, 5)))


def test_degradation_setup_spreads_the_day_around_the_circuit_rate():
    cfg = eng.resolve_config({"circuit": "Bahrain Grand Prix", "total_laps": 30})
    d = eng.degradation_setup(np.array([0]), 30, 40, cfg)
    assert d["deg_tables"].shape[0] == eng.DEG_NODES and len(d["deg_edges"]) == eng.DEG_NODES - 1
    nodes = d["deg_nodes"]
    assert (np.diff(nodes) > 0).all() and nodes[eng.DEG_NODES // 2] == pytest.approx(d["deg_rate"])
    old = d["deg_tables"][:, 0, 1, 25, 5] - d["deg_tables"][:, 0, 1, 2, 5]      # MEDIUM: age 25 vs age 2
    assert old[-1] > old[0]                                         # a harsher day costs more on old tyres
    assert (np.diff(d["deg_tables"][:, 0, :, 1:, 5], axis=2) >= -1e-9).all()    # every day is monotone in age
    known = eng.degradation_setup(np.array([0]), 30, 40, {**cfg, "deg_sd": 0.0})
    assert known["deg_tables"] is None and np.allclose(known["table"], d["table"])


def test_a_harsher_day_makes_the_race_longer(grid):
    def race_time(z):
        live = eng.LiveRace.from_grid(grid, {"total_laps": 30, "circuit": "Italian Grand Prix",
                                             "sc_prob_per_lap": 0.0, "dnf_prob_per_lap": 0.0}, seed=3)
        live._dr.deg_z[:] = z
        while not live.finished:
            live.step()
        return float(live.state["cum"].min())
    assert race_time(2.0) > race_time(0.0) > race_time(-2.0)


def test_state_from_real_race_reports_the_live_degradation():
    laps = synthetic_race(2024, "Italian Grand Prix", n_laps=50, seed=1, deg_scale=2.0)
    early = eng.state_from_real_race(laps, 2024, "Italian Grand Prix", 3)[3]["degradation"]
    late_setup, late_cfg, _, late_info = eng.state_from_real_race(laps, 2024, "Italian Grand Prix", 40)
    late = late_info["degradation"]
    assert early["mean"] == pytest.approx(early["prior_mean"], abs=0.01)
    assert late["mean"] > late["prior_mean"] + 0.02 and late["sd"] < early["sd"]      # the race showed its hand
    assert late_cfg["deg_rate"] == late["mean"] and late_setup["deg_rate"] == late["mean"]
    fixed = eng.state_from_real_race(laps, 2024, "Italian Grand Prix", 40, {"deg_rate": 0.05, "deg_sd": 0.0})
    assert fixed[0]["deg_rate"] == 0.05 and fixed[3]["degradation"] is None           # the caller's value wins
