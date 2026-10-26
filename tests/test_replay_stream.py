"""The stand-in f1-race-replay stream and the take-control mode."""
import json
import socket
import time

import numpy as np
import pytest

from integrations.f1_race_replay import replay_stream as rs
from integrations.f1_race_replay.replay_stream import RaceFeed, StreamServer, TakeControlFeed
from tools.race_engineer import RaceEngineer

from race_factory import RACE, line, make_race, stream

N = 12                                                   # take-control lap


@pytest.fixture(scope="module")
def laps():
    return make_race(30, slow={"D19": 4.0}, sc_laps=(6,))


@pytest.fixture(scope="module")
def feed(laps):
    return RaceFeed(laps, 2099, RACE)


# ── same shape and clocks as the real payload ────────────────────────────────
def test_payload_has_the_real_fields(feed):
    p = json.loads(json.dumps(feed.payload(500.0)))               # JSON round trip, like the socket
    assert {"frame", "track_status", "session_data", "lap_times", "status_laps", "frame_index",
            "total_frames", "circuit_length_m", "driver_colors"} <= set(p)
    d = p["frame"]["drivers"]["D03"]
    assert {"position", "lap", "rel_dist", "dist", "x", "y", "tyre", "tyre_life", "in_pit"} <= set(d)
    assert {"time_s", "lap", "leader", "total_laps"} <= set(p["session_data"])
    e = p["lap_times"]["D03"][0]
    assert {"lap", "time_s", "end_time_s", "replay_end_time_s", "tyre", "tyre_life",
            "is_pit_entry", "is_out_lap"} <= set(e)
    assert p["frame"]["t"] == pytest.approx(500.0) and p["frame_index"] == 500 * rs.FPS


def test_positions_follow_progress_on_track(feed):
    t = line(feed, "D00", 15) + 30
    drv = feed.payload(t)["frame"]["drivers"]
    order = sorted(drv, key=lambda d: drv[d]["position"])
    progress = [drv[d]["lap"] + drv[d]["rel_dist"] for d in order]
    assert sorted(drv[d]["position"] for d in drv) == list(range(1, 21))
    assert progress == sorted(progress, reverse=True)
    assert order[-1] == "D19"                                     # the lapped backmarker
    assert drv["D00"]["lap"] - drv["D19"]["lap"] >= 1


def test_safety_car_is_reported_late_never_early(feed):
    """The Parquet has one status per lap: report the leader's last COMPLETED lap (never ahead of time)."""
    on_sc_lap = (line(feed, "D00", 5) + line(feed, "D00", 6)) / 2  # leader on lap 6, the SC lap
    after = (line(feed, "D00", 6) + line(feed, "D00", 7)) / 2      # leader on lap 7: lap 6 is history
    assert feed.payload(on_sc_lap)["track_status"] == "1"
    assert feed.payload(after)["track_status"] == "4"
    assert {"status": "4", "start_lap": 6, "end_lap": 6} in feed.payload(after)["status_laps"]


def test_fastf1_placeholder_rows_are_flagged_and_not_on_the_timeline():
    f = RaceFeed(make_race(20, retire={"D05": 12}, placeholder=True), 2099, RACE)
    assert f.tl["D05"]["lap"][-1] == 12
    e = f.payload(1e5)["lap_times"]["D05"]
    gen = [x for x in e if x["fastf1_generated"]]
    assert [x["lap"] for x in gen] == [13] and gen[0]["time_s"] == -1.0 and not gen[0]["is_accurate"]


# ── take control ─────────────────────────────────────────────────────────────
def test_take_control_is_the_real_race_until_lap_n(laps, feed):
    tc = TakeControlFeed(laps, 2099, RACE, N, seed=3)
    t = tc.T0 - 20                                                # leader about to finish lap N
    assert tc.payload(t)["frame"] == feed.payload(t)["frame"]
    assert tc.payload(t)["take_control"]["active"] is False


def test_after_lap_n_laps_are_simulated_and_never_from_the_future(laps):
    tc = TakeControlFeed(laps, 2099, RACE, N, seed=3)
    t = tc.T0 + 400
    p = tc.payload(t)
    entries = [e for es in p["lap_times"].values() for e in es]
    sim = [e for e in entries if e["source"] == "simulated"]
    assert sim and all(e["lap"] > N for e in sim)
    assert all(e["replay_line_time_s"] <= t for e in sim)
    assert p["take_control"]["active"] and p["take_control"]["engine_lap"] > N
    # every car still running has a next line crossing beyond t
    drv = p["frame"]["drivers"]
    assert all(0 <= d["rel_dist"] < 1 for d in drv.values())


def test_take_control_is_deterministic(laps):
    a = TakeControlFeed(laps, 2099, RACE, N, seed=5)
    b = TakeControlFeed(laps, 2099, RACE, N, seed=5)
    t = a.T0 + 900
    assert a.payload(t)["lap_times"] == b.payload(t)["lap_times"]


def test_boxing_mid_lap_equals_having_planned_it(laps):
    """A box command re-runs the current lap with the same draws: same as if planned from the start."""
    a = TakeControlFeed(laps, 2099, RACE, N, seed=2)
    assert a.live.plan("D06") == [(16, "HARD")]                    # default plan: stop on lap 16
    a.payload(line(a, "D06", N) + 400)                             # engine has run ahead
    t = line(a, "D06", N + 2) + 20                                 # 20 s into lap 15
    prog = a._progress("D06", t)
    assert prog["lap"] == N + 3 and prog["rel"] < rs.PIT_DEADLINE
    msg = a.command({"cmd": "box", "driver": "D06", "compound": "SOFT"}, t)
    assert f"BOX lap {prog['lap']} → SOFT" in msg

    b = TakeControlFeed(laps, 2099, RACE, N, seed=2)
    b.live.set_plan("D06", [(prog["lap"], "SOFT")])
    end = a.T0 + 30 * 100
    pa, pb = a.payload(end), b.payload(end)
    for d in ("D06", "D00", "D10"):
        la = [e["replay_line_time_s"] for e in pa["lap_times"][d]]
        lb = [e["replay_line_time_s"] for e in pb["lap_times"][d]]
        assert la == pytest.approx(lb)
    stop = next(e for e in pa["lap_times"]["D06"] if e["lap"] == prog["lap"])
    after = next(e for e in pa["lap_times"]["D06"] if e["lap"] == prog["lap"] + 1)
    assert stop["is_pit_entry"] and after["tyre"] == 0 and after["tyre_life"] == 1 and after["is_out_lap"]


def test_box_after_the_pit_entry_means_next_lap(laps):
    tc = TakeControlFeed(laps, 2099, RACE, N, seed=2)
    lap = N + 3
    tc.payload(tc.T0 + 600)                                        # engine has run past that lap
    tl = tc.tl["D04"]
    i = tl["lap"].index(lap)
    t = tl["line"][i - 1] + 0.95 * (tl["line"][i] - tl["line"][i - 1])     # 95 % into lap `lap`
    msg = tc.command({"cmd": "box", "driver": "D04", "compound": "MEDIUM"}, t)
    assert f"lap {lap + 1}" in msg and "next lap" in msg


def test_a_command_right_after_crossing_the_line_is_accepted(laps):
    """The engine may not have run the lap the car just started: command() must step it first."""
    tc = TakeControlFeed(laps, 2099, RACE, N, seed=2)
    tc.payload(tc.T0 + 300)
    lap = tc.live.lap                                              # last lap the engine has run
    t = line(tc, "D00", lap) + 0.05                                # the leader has just finished it
    msg = tc.command({"cmd": "box", "driver": "D00", "compound": "MEDIUM"}, t)
    assert f"BOX lap {lap + 1}" in msg


def test_a_car_that_pitted_on_the_take_over_lap_rejoins_behind(laps):
    """The in-lap only carries ~4 s of a stop; the out-lap part is charged when the engine takes over."""
    real = make_race(30, pit_split=(4.0, 19.0))
    tc = TakeControlFeed(real, 2099, RACE, 12, seed=1)              # D02 boxes at the end of lap 12
    tc.payload(tc.T0 + 400)
    sim = line(tc, "D02", 13) - line(tc, "D02", 12)
    actual = (real[(real.Driver == "D02") & (real.LapNumber == 13)]["LapTime"].dt.total_seconds().iloc[0])
    assert abs(sim - actual) < 4.0                                 # out-lap includes the stop, as in reality


def test_stay_out_cancels_the_planned_stops_and_bad_commands_are_refused(laps):
    tc = TakeControlFeed(laps, 2099, RACE, N, seed=2)
    t = tc.T0 + 100
    tc.payload(t)
    assert "stay out" in tc.command({"cmd": "stay_out", "driver": "D03"}, t)
    assert tc.live.plan("D03") == []
    assert "not in the simulated race" in tc.command({"cmd": "box", "driver": "XXX"}, t)
    assert "unknown compound" in tc.command({"cmd": "box", "driver": "D03", "compound": "WET"}, t)
    assert rs.parse_terminal_command("box d03 soft") == {"cmd": "box", "driver": "d03", "compound": "soft"}
    assert rs.parse_terminal_command("stay D03") == {"cmd": "stay_out", "driver": "D03"}


def test_the_race_engineer_keeps_working_after_the_take_over(laps):
    tc = TakeControlFeed(laps, 2099, RACE, N, seed=4)
    eng = RaceEngineer("D08", RACE, n_sims=10)
    calls = stream(tc, eng, tc.T0 - 200, tc.T0 + 8 * 95, dt=4.0)
    laps_called = [c["lap"] for c in calls]
    assert laps_called == list(range(laps_called[0], laps_called[0] + len(calls)))
    assert max(laps_called) > N + 4 and all(c["running"] for c in calls)


def test_take_control_runs_to_the_flag(laps):
    tc = TakeControlFeed(laps, 2099, RACE, 25, seed=1)
    p = tc.payload(1e5)
    assert np.isfinite(tc.t_end)
    drv = p["frame"]["drivers"]
    assert all(d["lap"] == 30 for d in drv.values())                # everyone took the flag (no DNFs)


# ── the socket ───────────────────────────────────────────────────────────────
def test_server_broadcasts_lines_and_queues_client_commands():
    server = StreamServer(port=0)
    try:
        c = socket.create_connection(("localhost", server.port), timeout=2)
        c.sendall(b'{"cmd": "box", "driver": "VER", "compound": "HARD"}\n')
        assert server.commands.get(timeout=2) == {"cmd": "box", "driver": "VER", "compound": "HARD"}
        for _ in range(50):                                        # wait until accepted
            if server.clients:
                break
            time.sleep(0.02)
        server.broadcast({"hello": 1})
        assert json.loads(c.makefile().readline()) == {"hello": 1}
        c.close()
    finally:
        server.close()


def test_take_control_does_not_send_real_retirements_from_the_future():
    real = make_race(30, retire={"D05": 15}, placeholder=True)
    tc = TakeControlFeed(real, 2099, RACE, 10, seed=1)
    entries = tc.payload(1e5)["lap_times"]["D05"]
    assert not [e for e in entries if e.get("fastf1_generated")]   # he races on in the simulation
    assert max(e["lap"] for e in entries) == 30
