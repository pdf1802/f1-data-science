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


# ── validation helpers on real-race-shaped data ──────────────────────────────
def _duel(a_pit=10, b_pit=11, a_gain=1.5, sc_lap=None, n=20):
    """Two cars: B leads, A 1.0 s behind at the end of lap 9. A pits at the end of lap a_pit, B at b_pit.
    In-lap +4 s, out-lap +19 s; fresh tyres are a_gain s/lap faster than the old ones."""
    rows = []
    for d, pit, start in (("AAA", a_pit, 1.0), ("BBB", b_pit, 0.0)):
        t, comp, age = 3600.0 + start, "MEDIUM", 5
        for lap in range(1, n + 1):
            lt = 90.0 + (0.0 if comp == "MEDIUM" else -a_gain) + (4.0 if lap == pit else 0.0) + (19.0 if lap == pit + 1 else 0.0)
            t += lt
            rows.append({"Season": 2099, "Race": "Bahrain Grand Prix", "Driver": d, "Team": "Ferrari",
                         "LapNumber": float(lap), "Time": pd.Timedelta(seconds=t), "Compound": comp,
                         "TyreLife": float(age), "Stint": 1.0 if lap <= pit else 2.0,
                         "PitInTime": pd.Timedelta(seconds=t) if lap == pit else pd.NaT,
                         "TrackStatus": "4" if lap == sc_lap else "1", "TrackTemp": 35.0, "AirTemp": 25.0,
                         "TotalLaps": n})
            if lap == pit:
                comp, age = "HARD", 1
            else:
                age += 1
    return pd.DataFrame(rows)


def test_undercut_attempts_finds_the_duel_and_its_outcome():
    from tools.undercut import undercut_attempts
    att = undercut_attempts(_duel())
    assert len(att) == 1
    x = att.iloc[0]
    assert (x.A, x.B, x.lap, x.reaction) == ("AAA", "BBB", 9, 1) and x.gap == pytest.approx(1.0)
    # at the end of lap 12 (B's out-lap): A paid 4 + 19 and gained 1.5 twice; B paid 4 + 19 and gained 1.5 once
    assert x.final_gap == pytest.approx(1.0 + (4 + 19 - 1.5 - 1.5) - (4 + 19 - 1.5)) and x.worked
    assert not undercut_attempts(_duel(a_gain=0.5)).iloc[0].worked              # small tyre gain: it fails
    assert undercut_attempts(_duel(sc_lap=11)).empty                            # Safety Car: not a green undercut
    assert undercut_attempts(_duel(b_pit=15)).empty                             # B reacted too late to count


def test_real_strategies_override_the_default_plan():
    from tools.race_simulator import with_real_strategies
    duel = _duel()
    q = pd.DataFrame({"Driver": ["BBB", "AAA"], "Team": ["Ferrari", "Ferrari"], "QualyTime": [90.0, 90.1],
                      "QualyRank": [1, 2]})
    g = with_real_strategies(grid_from_qualifying(q, 20), duel, 2099, "Bahrain Grand Prix")
    a = g.set_index("Driver").loc["AAA"]
    assert (a.StartCompound, a.StartAge, a.Stop1Lap, a.Stop1Compound, a.Stop2Lap) == ("MEDIUM", 5, 10, "HARD", 0)
    assert validate_grid(g, 20) == []


def test_style_features_survive_a_98_percent_throttle_ceiling_and_dead_channels():
    grid = np.linspace(0, 1, 300)
    lap = pd.DataFrame({"RelativeDistance": grid, "Speed": 100 + 200 * np.abs(np.sin(6 * grid)),
                        "Throttle": np.where(np.sin(6 * grid) ** 2 > 0.3, 99.0, 20.0),
                        "Brake": (np.sin(6 * grid) ** 2 < 0.1).astype(float), "DRS": 0.0})
    capped = lap.assign(Throttle=lap.Throttle.clip(upper=98.0))           # sensor tops out at 98 %
    f, g = shared.extract_style_features(lap), shared.extract_style_features(capped)
    assert f["full_throttle_pct"] == pytest.approx(g["full_throttle_pct"]) and f["full_throttle_pct"] > 0.5
    assert shared.style_lap_ok(lap) == (True, "")
    assert not shared.style_lap_ok(lap.assign(Throttle=0.0))[0]           # RUS, Miami 2025 Q
    assert not shared.style_lap_ok(lap.assign(Throttle=np.nan))[0]
