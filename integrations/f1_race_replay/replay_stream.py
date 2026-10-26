"""
A stand-in for f1-race-replay's telemetry stream, fed from the lap Parquet.

    python -m integrations.f1_race_replay.replay_stream --race "2023 Bahrain Grand Prix" --speed 30

f1-race-replay (github.com/IAmTomShaw/f1-race-replay) broadcasts one JSON object
per line on localhost:9999 while it replays a race. Its replay needs car
position data, which cannot be downloaded at the moment (F1 live timing answers
401 since Oct 2026). This server sends the SAME message shape, built from lap
data only, so the Race Engineer window can be developed and demoed without it.

Fields sent (a subset of the real payload, same names and meaning):
    frame: {t, drivers: {CODE: {position, lap, tyre, tyre_life, in_pit}}}
    session_data: {time_s, lap, leader, total_laps}, track_status, lap_times
    session_info: {event_name, year}   ← not in the real payload (proposed upstream)
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time

import numpy as np
import pandas as pd

from tools.shared import COMPOUND_MAP, REPO_ROOT

PORT = 9999


def _sec(s):
    return pd.to_timedelta(s).dt.total_seconds()


class RaceFeed:
    """Builds f1-race-replay-shaped payloads for any session time of one race."""

    def __init__(self, laps: pd.DataFrame, season: int, race: str):
        r = laps[(laps["Season"] == season) & (laps["Race"] == race)].copy()
        if r.empty:
            raise ValueError(f"No laps for {season} {race}")
        r["end"] = _sec(r["Time"])
        r["lt"] = _sec(r["LapTime"])
        r = r.dropna(subset=["end"]).sort_values(["Driver", "LapNumber"])
        self.r, self.season, self.race = r, season, race
        self.total_laps = int(r["TotalLaps"].iloc[0])
        first = r[r["LapNumber"] == 1]
        self.t_start = float((first["end"] - first["lt"].fillna(first["lt"].median())).min())
        self.t_end = float(r["end"].max())
        tyre = r["Compound"].map(lambda c: COMPOUND_MAP.get(str(c), {"INTERMEDIATE": 3, "WET": 4}.get(str(c), -1)))
        self.lap_times = {}
        for code, g in r.assign(tyre=tyre).groupby("Driver"):
            self.lap_times[code] = [{
                "lap": int(x.LapNumber), "time_s": float(x.lt) if pd.notna(x.lt) else -1.0,
                "end_time_s": float(x.end), "tyre": int(x.tyre),
                "tyre_life": int(x.TyreLife) if pd.notna(x.TyreLife) else 0,
                "is_pit_entry": bool(pd.notna(x.PitInTime)), "is_out_lap": bool(pd.notna(x.PitOutTime)),
            } for x in g.itertuples()]

    def payload(self, t: float) -> dict:
        r = self.r
        done = r[r["end"] <= t]
        if done.empty:
            lap_of = {d: 0 for d in r["Driver"].unique()}
            last_cross = {d: self.t_start for d in lap_of}
        else:
            lastrow = done.groupby("Driver").tail(1).set_index("Driver")
            lap_of = lastrow["LapNumber"].astype(int).to_dict()
            last_cross = lastrow["end"].to_dict()
        running = [d for d in r["Driver"].unique()
                   if d in lap_of and (lap_of[d] >= self.total_laps or
                                       not r[(r.Driver == d) & (r.LapNumber == lap_of[d] + 1)].empty)]
        order = sorted(running, key=lambda d: (-lap_of[d], last_cross.get(d, 0)))
        drivers = {}
        for pos, d in enumerate(order, 1):
            cur = r[(r.Driver == d) & (r.LapNumber == min(lap_of[d] + 1, self.total_laps))]
            row = cur.iloc[0] if len(cur) else None
            in_pit = bool(row is not None and (pd.notna(row.PitOutTime) and t - last_cross.get(d, t) < 25))
            drivers[d] = {
                "position": pos, "lap": min(lap_of[d] + 1, self.total_laps),
                "tyre": float(COMPOUND_MAP.get(str(row.Compound), -1)) if row is not None else -1.0,
                "tyre_life": float(row.TyreLife) if row is not None and pd.notna(row.TyreLife) else 0.0,
                "in_pit": in_pit,
            }
        leader = order[0] if order else ""
        status = "GREEN"
        cur_rows = r[(r["end"] > t)].groupby("Driver").head(1)
        ts = cur_rows["TrackStatus"].astype(str)
        if ts.str.contains("4").any():
            status = "SC"
        elif ts.str.contains("6|7").any():
            status = "VSC"
        return {
            "frame_index": int(t), "frame": {"t": float(t), "drivers": drivers},
            "track_status": status, "playback_speed": 1.0, "is_paused": False,
            "session_data": {"time_s": float(t), "lap": drivers.get(leader, {}).get("lap", 1),
                             "leader": leader, "total_laps": self.total_laps},
            "lap_times": self.lap_times,
            "session_info": {"event_name": self.race, "year": int(self.season)},
        }


class StreamServer:
    """Same protocol as f1-race-replay: TCP, one JSON object per line, many clients."""

    def __init__(self, port: int = PORT):
        self.port, self.clients, self.lock = port, [], threading.Lock()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("localhost", port))
        self.sock.listen(5)
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                c, _ = self.sock.accept()
            except OSError:
                return
            with self.lock:
                self.clients.append(c)

    def broadcast(self, data: dict):
        msg = (json.dumps(data) + "\n").encode()
        with self.lock:
            for c in list(self.clients):
                try:
                    c.sendall(msg)
                except OSError:
                    self.clients.remove(c)

    def close(self):
        self.sock.close()


def parse_race(text: str) -> tuple[int, str]:
    """'2023 Bahrain Grand Prix' → (2023, 'Bahrain Grand Prix')."""
    year, _, name = text.strip().partition(" ")
    return int(year), name


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--laps", default=os.path.join(REPO_ROOT, "data", "laps_2022_2025.parquet"))
    ap.add_argument("--race", default="2023 Bahrain Grand Prix")
    ap.add_argument("--speed", type=float, default=30.0, help="race seconds per real second")
    ap.add_argument("--start-lap", type=int, default=1)
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args(argv)
    season, race = parse_race(args.race)
    feed = RaceFeed(pd.read_parquet(args.laps), season, race)
    server = StreamServer(args.port)
    lap1 = feed.r[feed.r["LapNumber"] == max(args.start_lap - 1, 1)]["end"].min()
    t = feed.t_start if args.start_lap <= 1 else float(lap1)
    print(f"Streaming {season} {race} ({feed.total_laps} laps) on localhost:{args.port} at {args.speed}× — Ctrl+C to stop")
    try:
        while t <= feed.t_end + 5:
            server.broadcast(feed.payload(t))
            time.sleep(0.25)
            t += 0.25 * args.speed
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
