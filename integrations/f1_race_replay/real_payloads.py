"""
Real f1-race-replay payloads, without watching the replay.

    python -m integrations.f1_race_replay.real_payloads --race "2023 Bahrain Grand Prix" --driver ALO

Builds f1-race-replay's replay window for a cached race (the same object that runs
when you watch it) and asks it for the payload it would broadcast at any replay
time: `_broadcast_telemetry_state()` with a capture instead of the socket. The
payloads are JSON round-tripped, like over the socket. Used to check the Race
Engineer on the REAL stream (until now it was developed on the stand-in stream):

  * every lap of the followed driver gets a call, nobody is dropped
  * how good the estimated line times of the cars behind are when `rel_dist` is the
    real, distance-based value (the stand-in's is time-based)

Needs f1-race-replay (see run_real_replay.py) and a display (the window is created
but never shown; on a headless Linux box use `xvfb-run`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pandas as pd

from integrations.f1_race_replay import run_real_replay as rrr
from integrations.f1_race_replay.estimate_accuracy import MODES

FPS = 25


class _Capture:
    """Stands in for TelemetryStreamServer: keeps the last payload instead of sending it."""

    def __init__(self):
        self.last = None

    def broadcast(self, data):
        self.last = data


class RealReplay:
    """f1-race-replay's window for one cached race; payload(t) = what it streams at replay time t."""

    def __init__(self, race: str, cache=None, replay_dir=None, offline: bool = True):
        cache = (cache or rrr.default_cache()).resolve()
        replay_dir = (replay_dir or rrr.default_replay_dir()).resolve()
        year_s, _, event = race.partition(" ")
        self.year, self.event = int(year_s), event
        rrr.configure_fastf1(cache, offline)
        import fastf1
        rnd = int(fastf1.get_event(self.year, event)["RoundNumber"])
        sys.path.insert(0, str(rrr.REPO_ROOT))
        sys.path.insert(0, str(replay_dir))
        cwd = os.getcwd()
        os.chdir(replay_dir)                                       # computed_data/, images/
        try:
            import main as rr_main
            rrr.patch_f1_race_replay(rr_main, offline)
            from src.f1_data import get_race_telemetry, load_session
            from src.interfaces.race_replay import F1RaceReplayWindow
            session = load_session(self.year, rnd, "R")
            tele = get_race_telemetry(session, session_type="R")
            example = session.laps.pick_fastest().get_telemetry()
            self.window = F1RaceReplayWindow(
                frames=tele["frames"], track_statuses=tele["track_statuses"], example_lap=example,
                drivers=session.drivers, title=race, total_laps=tele["total_laps"],
                circuit_rotation=rr_main.get_circuit_rotation(session), visible_hud=False,
                session_info={"event_name": session.event["EventName"], "year": self.year,
                              "circuit_length_m": float(example["Distance"].max())},
                session=session, enable_telemetry=False, driver_colors=tele["driver_colors"],
                race_control_messages=tele.get("race_control_messages", []))
        finally:
            os.chdir(cwd)
        self.window.set_visible(False)
        self.capture = _Capture()
        self.window.telemetry_stream = self.capture
        self.total_laps = int(tele["total_laps"])
        self.n_frames = len(tele["frames"])
        self.t_end = self.n_frames / FPS
        self._lap_times_json = None

    def payload(self, t: float) -> dict:
        w = self.window
        w.frame_index = float(min(max(t, 0.0) * FPS, self.n_frames - 1))
        w._broadcast_telemetry_state()
        p = dict(self.capture.last)
        lap_times = p.pop("lap_times")
        if self._lap_times_json is None:                           # constant for the race: round-trip once
            self._lap_times_json = json.loads(json.dumps(lap_times))
        p = json.loads(json.dumps(p))
        p["lap_times"] = self._lap_times_json
        return p

    def line_times(self, driver: str) -> dict[int, float]:
        """{lap: replay-clock line time} of a driver's completed laps (the truth for estimates)."""
        from tools.race_engineer import is_lap, line_time
        out = {}
        for e in (self._lap_times_json or self.payload(0.0)["lap_times"]).get(driver) or []:
            t = line_time(e)
            if t is not None and is_lap(e):
                out[int(e["lap"])] = t
        return out


def engineer_calls(rp: RealReplay, driver: str, circuit: str, n_sims: int = 300, dt: float = 2.0) -> list[dict]:
    """Stream the whole race into a RaceEngineer like the window does; return its calls."""
    from tools.race_engineer import RaceEngineer
    eng, calls, t = RaceEngineer(driver, circuit, n_sims=n_sims), [], 0.0
    while t <= rp.t_end:
        c = eng.update(rp.payload(t))
        if c:
            calls.append(c)
        t += dt
    return calls


def estimate_errors(rp: RealReplay, driver: str, circuit: str) -> pd.DataFrame:
    """At each line crossing of `driver`, the estimated lap-n line time of every car not there yet vs the truth."""
    from tools import race_engineer as re_
    mine = rp.line_times(driver)
    truth = {d: rp.line_times(d) for d in rp.payload(0.0)["lap_times"]}
    rows = []
    for n in range(4, rp.total_laps - 1):
        if n not in mine:
            continue
        t = mine[n] + 0.2
        p = rp.payload(t)
        done, _ = re_.laps_from_payload(p, circuit, rp.total_laps)
        trust = re_.REL_TRUST
        for mode, rel_trust in MODES:
            re_.REL_TRUST = rel_trust
            try:
                snap, _, _, est = re_.snapshot_at_lap(done, n, t, p["frame"]["drivers"])
            finally:
                re_.REL_TRUST = trust
            at_n = snap[snap["LapNumber"] == n].set_index("Driver")["Time"].dt.total_seconds()
            for d in est:
                if n not in truth.get(d, {}):
                    continue
                e_n = next((e for e in p["lap_times"][d] if e.get("lap") == n and re_.is_lap(e)), {})
                pit = bool(e_n.get("is_pit_entry") or e_n.get("is_out_lap"))     # as in estimate_accuracy
                rows.append({"lap": n, "driver": d, "mode": mode, "error_s": at_n[d] - truth[d][n],
                             "seconds_to_line": truth[d][n] - t, "pit_lap": bool(pit)})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--race", default="2023 Bahrain Grand Prix")
    ap.add_argument("--driver", default="ALO")
    ap.add_argument("--cache", type=rrr.Path, default=None)
    ap.add_argument("--replay-dir", type=rrr.Path, default=None)
    ap.add_argument("--sims", type=int, default=300)
    args = ap.parse_args(argv)
    t0 = time.time()
    rp = RealReplay(args.race, args.cache, args.replay_dir)
    circuit = args.race.partition(" ")[2]
    print(f"Real f1-race-replay window ready in {time.time() - t0:.0f} s ({rp.n_frames} frames)", flush=True)

    calls = engineer_calls(rp, args.driver, circuit, n_sims=args.sims)
    laps = [c["lap"] for c in calls]
    sizes = [len(c["standings"]) for c in calls if c["running"]]
    print(f"\nRace Engineer on the REAL stream, following {args.driver}: {len(calls)} calls, laps "
          f"{laps[0]}–{laps[-1]}, consecutive: {laps == list(range(laps[0], laps[-1] + 1))}, cars in the "
          f"tower {min(sizes)}–{max(sizes)}, estimated per call (median) "
          f"{pd.Series([len(c['estimated']) for c in calls]).median():.0f}")
    for c in calls[::8]:
        print(f"  L{c['lap']:>2}  {c['headline']}")

    from integrations.f1_race_replay.estimate_accuracy import summary
    df = estimate_errors(rp, args.driver, circuit)
    table, near = summary(df.assign(race=args.race))
    print("\n|error| of the estimated line time on the REAL stream (s)")
    print(table.to_string())
    print("\nmedian |error| for cars within 10 s of the line (no stop):")
    print(near.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
