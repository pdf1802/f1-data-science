"""
Unit tests for the tools/ package. They run on stand-in models (see conftest.py),
so they test LOGIC and CONTRACTS, not prediction quality.

    pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from tools import shared
from tools.race_simulator import grid_from_qualifying, simulate, validate_grid, what_if
from tools.stint_optimizer import compare_strategies, optimize_pit_lap
from tools.undercut import WORST_CASE, DriverState, undercut_analysis

CFG = {"track_temp": 35, "air_temp": 25, "rainfall": 0}


# ── shared ───────────────────────────────────────────────────────────────────
def test_tyre_delta_sign_matches_training():
    f = shared.build_overtake_features(0.5, 0, 2, tyre_life_behind=5, tyre_life_ahead=25,
                                       lap_number=30, position=4)
    assert list(f.columns) == shared.OVT_FEATURES
    assert f["TyreDelta"].iloc[0] == -20       # behind − ahead: attacker on fresher tyres


def test_overtake_probability_scalar_and_vector():
    p = shared.predict_overtake_probability(0.5, 0, 2, 5, 25, 30, 4)
    assert isinstance(p, float) and 0 <= p <= 1
    v = shared.predict_overtake_probability(np.array([0.2, 0.9]), 0, 2, 5, 25, 30, 4)
    assert v.shape == (2,)


def test_encode_team_fallbacks():
    assert shared.encode_team("Ferrari") == shared.encode_team("Scuderia Ferrari")  # substring
    with pytest.warns(UserWarning):
        shared.encode_team("Totally Unknown Racing")


def test_encode_compound():
    assert shared.encode_compound("soft") == 0 and shared.encode_compound(2) == 2
    with pytest.raises(ValueError):
        shared.encode_compound("INTER")


def test_lap_time_formula():
    assert shared.lap_time_from_delta(1.0, 1.01, 95.0) == pytest.approx(95 + 1 + 0.95)


# ── stint optimizer ──────────────────────────────────────────────────────────
def test_stint_curve_shape_and_optimum():
    r = optimize_pit_lap("MEDIUM", "HARD", 57, "Ferrari", CFG)
    cur = r["curve"]
    assert cur["PitLap"].min() == 5 and cur["PitLap"].max() == 52
    assert r["optimal_cost"] == pytest.approx(cur["Total"].min())
    # each pit lap should simulate exactly 57 laps
    assert (r["laps"].groupby("PitLap").size() == 57).all()


def test_compare_strategies_has_all_legal_pairs():
    df = compare_strategies(50, "McLaren", CFG)
    assert len(df) == 6 and df["Gap to best (s)"].iloc[0] == 0


# ── undercut ─────────────────────────────────────────────────────────────────
def _pair():
    return (DriverState("VER", "Red Bull Racing", 3, "HARD", 20),
            DriverState("NOR", "McLaren", 2, "HARD", 20))


def test_undercut_is_deterministic():
    a, b = _pair()
    r1 = undercut_analysis(a, b, 1.5, 20, CFG)
    r2 = undercut_analysis(a, b, 1.5, 20, CFG)
    assert r1["curves"] == r2["curves"] and r1["verdict"] == r2["verdict"]


def test_undercut_pit_lap_costs_pit_loss():
    a, b = _pair()
    r = undercut_analysis(a, b, 1.5, 20, CFG)
    jump = r["curves"][WORST_CASE][1] - r["curves"][WORST_CASE][0]
    assert 18 < jump < 28               # ≈ 23.5 s pit loss ± tyre offset


def test_huge_gap_means_stay_out():
    a, b = _pair()
    assert undercut_analysis(a, b, 15.0, 20, CFG)["verdict"] == "STAY OUT"


# ── race simulator ───────────────────────────────────────────────────────────
@pytest.fixture
def quali():
    teams = ["Red Bull Racing", "Red Bull Racing", "Ferrari", "Ferrari", "McLaren", "McLaren",
             "Mercedes", "Mercedes"]
    return pd.DataFrame({"Driver": list("ABCDEFGH"), "Team": teams,
                         "QualyTime": 90 + np.arange(8) * 0.15, "QualyRank": np.arange(1, 9)})


def test_grid_validation_two_compound_rule(quali):
    g = grid_from_qualifying(quali, 40, "MEDIUM", "HARD", 18)
    assert validate_grid(g, 40) == []
    g.loc[0, "Stop1Compound"] = "MEDIUM"
    assert any("two different compounds" in e for e in validate_grid(g, 40))


def test_simulation_outputs_are_consistent(quali):
    g = grid_from_qualifying(quali, 40, "MEDIUM", "HARD", 18)
    r = simulate(g, {"total_laps": 40}, n_sims=200, seed=1)
    pos = r["positions"]
    assert pos.shape == (200, 8)
    assert (np.sort(pos, 1) == np.arange(1, 9)).all()        # a permutation in every race
    s = r["summary"]
    assert s["Win %"].sum() == pytest.approx(100)


def test_simulation_is_reproducible(quali):
    g = grid_from_qualifying(quali, 30, "MEDIUM", "HARD", 14)
    a = simulate(g, {"total_laps": 30}, n_sims=50, seed=7)["positions"]
    b = simulate(g, {"total_laps": 30}, n_sims=50, seed=7)["positions"]
    assert (a == b).all()


def test_faster_car_gains_positions(quali):
    """Overtakes must matter: a much faster car from the back should climb."""
    g = grid_from_qualifying(quali, 40, "MEDIUM", "HARD", 18)
    g.loc[g["Driver"] == "H", "PaceFactor"] = 0.99
    s = simulate(g, {"total_laps": 40, "dnf_prob_per_lap": 0}, n_sims=200, seed=3)["summary"]
    assert s.set_index("Driver").loc["H", "Exp. position"] < 4


def test_what_if_identical_strategy_is_identical(quali):
    """Common random numbers: same strategy ⇒ exactly the same race, every time."""
    g = grid_from_qualifying(quali, 30, "MEDIUM", "HARD", 14)
    w = what_if(g, g.copy(), "C", {"total_laps": 30}, n_sims=100, seed=5)
    assert w["same_pct"] == 100 and w["gain_se"] == 0


def test_dnfs_are_excluded_from_race_result():
    from tools.qualy_predictor import classified_finishers
    res = pd.DataFrame({"Abbreviation": ["HAM", "VER", "RUS"],
                        "ClassifiedPosition": ["1", "2", "R"], "Position": [1.0, 2.0, 19.0]})
    out = classified_finishers(res)
    assert out.to_dict() == {"HAM": 1, "VER": 2}
