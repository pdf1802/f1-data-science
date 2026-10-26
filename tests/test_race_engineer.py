"""
The live Race Engineer on synthetic f1-race-replay payloads.

What matters for a live tool, and what these tests pin down:
- every running car is in the call, including cars that have not crossed the
  line yet and lapped cars (the old bug dropped them: "ALO is not running");
- nothing from the future is used, although every payload carries the whole race;
- at most one call per lap of the followed driver (one per lap when payloads keep coming), and the
  same keys every time.
"""
import copy

import numpy as np
import pandas as pd
import pytest

from integrations.f1_race_replay.replay_stream import RaceFeed
from tools import race_engineer as re_
from tools.race_engineer import CALL_KEYS, OPTION_COLUMNS, STAY, RaceEngineer

from race_factory import RACE, line, make_race, stream

N_SIMS = 40


@pytest.fixture(scope="module")
def lapped():
    """D19 is 4 s/lap slower: lapped from about lap 13."""
    return RaceFeed(make_race(30, slow={"D19": 4.0}), 2099, RACE)


def call_at(feed, driver, lap, after=0.2, **kw):
    """The call made the moment `driver` completes `lap`."""
    eng = RaceEngineer(driver, RACE, n_sims=N_SIMS, **kw)
    eng.last_lap_done = lap - 1
    return eng.update(feed.payload(line(feed, driver, lap) + after)), eng


# ── the bug ──────────────────────────────────────────────────────────────────
def test_cars_behind_and_lapped_cars_are_in_the_call(lapped):
    call, _ = call_at(lapped, "D05", 20)
    st = call["standings"]
    assert call["running"] and call["lap"] == 20 and call["tyres"]["compound"] in ("SOFT", "MEDIUM", "HARD")
    assert len(st) == 20                                            # nobody dropped
    t = line(lapped, "D05", 20) + 0.2
    not_there_yet = {d for d in lapped.tl if line(lapped, d, 20) > t}
    assert set(st.loc[st["Estimated"], "Driver"]) == not_there_yet
    assert "D19" in not_there_yet and st.set_index("Driver").loc["D19", "Gap"] > 85   # a lap down


def test_estimated_gaps_are_close_to_what_really_happened(lapped):
    call, _ = call_at(lapped, "D05", 20)
    st = call["standings"].set_index("Driver")
    leader = min(lapped.tl, key=lambda d: line(lapped, d, 20))
    for d in st.index[st["Estimated"]]:
        real_gap = line(lapped, d, 20) - line(lapped, leader, 20)
        assert abs(st.loc[d, "Gap"] - real_gap) < 2.0, d


def test_the_leader_and_a_lapped_car_can_be_followed(lapped):
    lead, _ = call_at(lapped, "D00", 20)
    assert lead["running"] and lead["standings"].iloc[0]["Driver"] == "D00"
    back, _ = call_at(lapped, "D19", 15)
    assert back["running"] and back["standings"].iloc[-1]["Driver"] == "D19"
    assert len(back["standings"]) == 20


# ── no future leakage ────────────────────────────────────────────────────────
def test_the_future_part_of_the_payload_changes_nothing(lapped):
    t = line(lapped, "D07", 18) + 0.2
    p = lapped.payload(t)
    q = copy.deepcopy(p)
    for entries in q["lap_times"].values():
        for e in entries:
            if e["replay_line_time_s"] > t:                          # rewrite the future
                e["time_s"] *= 1.5
                e["tyre"], e["tyre_life"] = 0, 40
                e["is_pit_entry"] = not e["is_pit_entry"]
                for k in ("end_time_s", "line_time_s", "replay_end_time_s", "replay_line_time_s"):
                    e[k] += 50.0
    q["status_laps"] = q["status_laps"] + [{"status": "4", "start_lap": 19, "end_lap": 25}]
    a = RaceEngineer("D07", RACE, n_sims=N_SIMS)
    b = RaceEngineer("D07", RACE, n_sims=N_SIMS)
    a.last_lap_done = b.last_lap_done = 17
    ca, cb = a.update(p), b.update(q)
    pd.testing.assert_frame_equal(ca["options"], cb["options"])
    pd.testing.assert_frame_equal(ca["standings"], cb["standings"])
    assert ca["headline"] == cb["headline"]


def test_laps_from_payload_keeps_only_completed_laps(lapped):
    t = line(lapped, "D10", 12) + 0.1
    laps, t_now = re_.laps_from_payload(lapped.payload(t), RACE, 30)
    assert t_now == pytest.approx(t)
    assert (laps["Time"].dt.total_seconds() <= t).all()
    done = sum(1 for tl in lapped.tl.values() for x in tl["line"] if x <= t)
    assert len(laps) == done


def test_replay_clock_is_used_not_the_session_clock(lapped):
    """f1-race-replay: end_time_s is FastF1 session time; replay_* is the frame clock."""
    t = line(lapped, "D03", 8) + 0.1
    p = lapped.payload(t)
    e = p["lap_times"]["D03"][0]
    assert e["end_time_s"] - e["replay_end_time_s"] == pytest.approx(lapped.t0) and lapped.t0 > 3000
    a, _ = re_.laps_from_payload(p, RACE, 30)
    # laps rebuilt from frames only carry end_time_s, already on the replay clock
    q = copy.deepcopy(p)
    for entries in q["lap_times"].values():
        for e in entries:
            e["end_time_s"] = e.pop("replay_end_time_s")
            del e["replay_line_time_s"], e["line_time_s"]
    b, _ = re_.laps_from_payload(q, RACE, 30)
    pd.testing.assert_frame_equal(a, b)


def test_status_laps_are_cut_at_the_present():
    p = {"status_laps": [{"status": "4", "start_lap": 5, "end_lap": 9},
                         {"status": "6", "start_lap": 20, "end_lap": 21}]}
    assert re_.status_by_lap(p, 7) == {5: "4", 6: "4", 7: "4"}       # SC still out: its end is unknown
    assert re_.status_by_lap(p, 12) == {k: "4" for k in range(5, 10)}
    assert 20 not in re_.status_by_lap(p, 19)


# ── one call per lap, same keys every time ───────────────────────────────────
def test_exactly_one_call_per_lap():
    feed = RaceFeed(make_race(14), 2099, RACE)
    eng = RaceEngineer("D04", RACE, n_sims=10)
    calls = stream(feed, eng, 0.0, feed.t_end + 30, dt=3.0)
    assert [c["lap"] for c in calls] == list(range(re_.MIN_CALL_LAP, 13))   # 3 … total_laps-2
    assert all(set(c) == set(CALL_KEYS) for c in calls)


def test_a_retired_driver_gets_one_not_running_call_with_the_same_keys():
    feed = RaceFeed(make_race(30, retire={"D05": 12}), 2099, RACE)
    eng = RaceEngineer("D05", RACE, n_sims=10)
    calls = stream(feed, eng, line(feed, "D05", 11) - 1, line(feed, "D00", 18), dt=2.0)
    out = [c for c in calls if not c["running"]]
    assert len(out) == 1
    c = out[0]
    assert set(c) == set(CALL_KEYS) and c["tyres"] is None and "not running" in c["headline"]
    assert list(c["options"].columns) == OPTION_COLUMNS and c["options"].empty
    assert "D05" not in set(c["standings"]["Driver"]) and len(c["standings"]) == 19


def test_rewinding_the_replay_starts_again(lapped):
    eng = RaceEngineer("D02", RACE, n_sims=10)
    stream(lapped, eng, line(lapped, "D02", 9) - 1, line(lapped, "D02", 10) - 1, dt=2.0)
    assert eng.update(lapped.payload(line(lapped, "D02", 10) + 1))["lap"] == 10
    c = eng.update(lapped.payload(line(lapped, "D02", 5) + 1))      # user jumps back to lap 5
    assert c is not None and c["lap"] == 5


def test_red_flag_holds_the_call(lapped):
    eng = RaceEngineer("D02", RACE, n_sims=10)
    eng.last_lap_done = 9
    p = lapped.payload(line(lapped, "D02", 10) + 0.5)
    assert eng.update({**p, "track_status": "5"}) is None
    assert eng.update(p)["lap"] == 10                              # made once the session resumes


# ── options and headline ─────────────────────────────────────────────────────
def test_options_respect_the_two_compound_rule(lapped):
    first, _ = call_at(lapped, "D04", 6)                             # SOFT, not stopped yet
    opts = set(first["options"]["Option"])
    assert first["tyres"] == {"compound": "SOFT", "age": 9, "rule_met": False}
    assert "Box now → SOFT" not in opts and "No more stops" not in opts
    assert {STAY, "Box now → MEDIUM", "Box now → HARD"} <= opts
    after, _ = call_at(lapped, "D04", 20)                            # stopped on lap 14 → HARD
    assert after["tyres"]["compound"] == "HARD" and after["tyres"]["rule_met"]
    assert "Box now → HARD" in set(after["options"]["Option"])


def test_wet_tyres_get_no_dry_strategy(lapped):
    t = line(lapped, "D06", 8) + 0.2
    p = copy.deepcopy(lapped.payload(t))
    p["frame"]["drivers"]["D06"]["tyre"] = 3.0                        # INTERMEDIATE
    eng = RaceEngineer("D06", RACE, n_sims=10)
    eng.last_lap_done = 7
    c = eng.update(p)
    assert c["running"] and c["tyres"]["compound"] == "INTERMEDIATE" and c["options"].empty
    assert "dry races only" in c["headline"]


def _table(rows):
    return pd.DataFrame(rows, columns=OPTION_COLUMNS)


def test_headline_only_calls_an_option_that_clearly_beats_staying_out():
    base = [STAY, 0, .2, 1, 5.0, 0.0, 0.0]
    clear = _table([base, ["Box now → HARD", 0, .4, 1, 3.0, 2.0, 0.3]])
    t, h = re_.make_headline(clear)
    assert t.iloc[0]["Option"] == "Box now → HARD" and h.startswith("BOX NOW → HARD — +2.0")
    noisy = _table([base, ["Box now → HARD", 0, .25, 1, 4.9, 0.1, 0.3]])
    assert re_.make_headline(noisy)[1].startswith("STAY OUT")
    assert re_.make_headline(_table([base]))[1].startswith("STAY OUT")


def test_default_plan():
    assert re_.default_plan("SOFT", 18, {"SOFT"}, 20, 57) == [(21, "HARD")]      # at the edge of the data
    assert re_.default_plan("HARD", 5, {"SOFT"}, 40, 57) == []                     # reaches the flag, rule met
    assert re_.default_plan("HARD", 5, {"HARD"}, 50, 57) == [(54, "MEDIUM")]       # rule not met yet


# ── found by the independent review ──────────────────────────────────────────
def test_after_a_red_flag_nobody_is_dropped():
    """A 25-minute stoppage: the first call after the restart must still have all 20 cars."""
    feed = RaceFeed(make_race(30, red_flag=(15 * 91.0, 1500.0)), 2099, RACE)
    eng = RaceEngineer("D05", RACE, n_sims=10)
    red_lap = next(lap for lap, st in zip(feed.r[feed.r.Driver == "D05"].LapNumber, feed.r[feed.r.Driver == "D05"].TrackStatus)
                   if st == "5")
    calls = stream(feed, eng, line(feed, "D05", int(red_lap)) - 60, line(feed, "D05", int(red_lap)) + 2,
                   dt=2.0, override={"track_status": "1"})   # the real replay says green after the restart
    first = [c for c in calls if c["lap"] == red_lap]
    assert first and first[0]["running"] and len(first[0]["standings"]) == 20


def test_fastf1_placeholder_row_is_not_a_lap():
    """A retired car's generated row (no lap time, 150 s later) must not trigger a call for that lap."""
    feed = RaceFeed(make_race(30, retire={"D05": 12}, placeholder=True), 2099, RACE)
    eng = RaceEngineer("D05", RACE, n_sims=10)
    calls = stream(feed, eng, line(feed, "D05", 11) - 1, line(feed, "D00", 18), dt=2.0)
    assert [c["lap"] for c in calls if c["running"]] == [10, 11, 12]
    assert len([c for c in calls if not c["running"]]) == 1
    laps, _ = re_.laps_from_payload(feed.payload(1e5), RACE, 30)
    assert laps[laps.Driver == "D05"]["LapNumber"].max() == 12


def test_a_car_in_the_pit_lane_is_charged_its_stop():
    """A rival in the pit lane before crossing line N: it stops at the end of lap N, on new tyres."""
    feed = RaceFeed(make_race(30, pit_split=(4.0, 19.0)), 2099, RACE)  # D04 boxes at the end of lap 14
    t = line(feed, "D04", 14) - 0.5                                     # D04 in the pit lane, line 14 not crossed
    leader_done = [d for d in feed.tl if line(feed, d, 14) < t]
    me = max(leader_done, key=lambda d: line(feed, d, 14))             # someone who just crossed line 14
    eng = RaceEngineer(me, RACE, n_sims=10)
    eng.last_lap_done = 13
    p = feed.payload(t)
    assert p["frame"]["drivers"]["D04"]["in_pit"]
    c = eng.update(p)
    row = c["standings"].set_index("Driver").loc["D04"]
    assert "D04" in c["estimated"] and row["Compound"] == "HARD" and row["TyreAge"] == 0
    gap_real = line(feed, "D04", 15) - min(line(feed, d, 15) for d in feed.tl)
    assert abs(row["Gap"] - gap_real) < 6.0                            # where it really rejoined, roughly


def test_clocks_are_aligned_when_the_replay_could_not():
    """Without replay_* fields, end_time_s is the session clock (~1 h ahead): estimate the offset."""
    feed = RaceFeed(make_race(20), 2099, RACE)
    t = line(feed, "D07", 9) + 30
    p = copy.deepcopy(feed.payload(t))
    for entries in p["lap_times"].values():
        for e in entries:
            e.pop("replay_end_time_s"), e.pop("replay_line_time_s")
    assert re_.clock_offset(p) == pytest.approx(feed.t0, abs=2.0)
    a, _ = re_.laps_from_payload(feed.payload(t), RACE, 20)
    b, _ = re_.laps_from_payload(p, RACE, 20)
    far = lambda df: set(map(tuple, df.loc[(df.Time.dt.total_seconds() - t).abs() > 3, ["Driver", "LapNumber"]].values))
    assert far(a) == far(b)


def test_a_car_on_its_out_lap_still_owes_most_of_its_stop():
    """D04's in-lap is 14 (+4 s); its out-lap 15 costs +19 s more — known as soon as lap 14 ended in the pits."""
    feed = RaceFeed(make_race(30, pit_split=(4.0, 19.0)), 2099, RACE)
    me = "D00"                                                     # the leader finishes lap 15…
    assert line(feed, "D04", 14) < line(feed, me, 15) < line(feed, "D04", 15)   # …while D04 is on its out-lap
    eng = RaceEngineer(me, RACE, n_sims=10)
    eng.last_lap_done = 14
    c = eng.update(feed.payload(line(feed, me, 15) + 0.2))
    st = c["standings"].set_index("Driver")
    assert "D04" in c["estimated"]
    real_gap = line(feed, "D04", 15) - min(line(feed, d, 15) for d in feed.tl)
    assert abs(st.loc["D04", "Gap"] - real_gap) < 4.0
