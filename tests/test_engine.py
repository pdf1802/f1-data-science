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
