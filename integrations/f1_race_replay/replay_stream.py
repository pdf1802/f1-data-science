"""
A stand-in for f1-race-replay's telemetry stream, fed from the lap Parquet —
plus a "take control" mode where the race engine drives the race after lap N.

    python -m integrations.f1_race_replay.replay_stream --race "2023 Bahrain Grand Prix" --speed 30
    python -m integrations.f1_race_replay.replay_stream --race "2023 Bahrain Grand Prix" --take-control 20

f1-race-replay (github.com/IAmTomShaw/f1-race-replay) broadcasts one JSON object
per line on localhost:9999 while it replays a race. Its replay needs car
position data, which cannot be downloaded at the moment (F1 live timing answers
401 since Oct 2026). This server sends the SAME message shape, built from lap
data only, so the Race Engineer window (and f1-race-replay's own insight
windows) can be developed and demoed without it.

Fields sent — same names, meaning and clocks as the real payload:
    frame: {t, lap, drivers: {CODE: {position, lap, rel_dist, dist, fraction, x, y,
            tyre, tyre_life, in_pit}}, weather}
    session_data: {time, time_s, lap, leader, total_laps}, track_status (FastF1 code),
    lap_times (end_time_s on the session clock + replay_*_time_s on the replay clock),
    status_laps, circuit_length_m, driver_colors, frame_index, total_frames
Not in the real payload:
    session_info: {event_name, year}         ← proposed upstream (the window needs the circuit)
    take_control: {...}                      ← only in take-control mode
Positions are SCHEMATIC: cars move round a circle at constant speed within each
lap (x, y, rel_dist), because the Parquet has lap times, not GPS.

Take control (--take-control N): real laps up to lap N, then the lap-by-lap
engine (tools/race_engine.py) races on from the real state at the end of lap N.
Clients can send commands back on the same socket, one JSON object per line:
    {"cmd": "box", "driver": "VER", "compound": "HARD"}   pit this lap (next lap if
                                                            past 85 % of the lap)
    {"cmd": "stay_out", "driver": "VER"}                    cancel the planned stops
or type them on the terminal: `box VER HARD`, `stay VER`.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import queue
import socket
import sys
import threading
import time

import numpy as np
import pandas as pd

from tools.shared import COMPOUND_MAP, REPO_ROOT, circuit_name, team_colour

PORT = 9999
FPS = 25                     # f1-race-replay's frame rate (frame_index = t × FPS)
PIT_DEADLINE = 0.85          # past this fraction of the lap, "box" means next lap
TYRE_INT = {**COMPOUND_MAP, "INTERMEDIATE": 3, "WET": 4}

# Official lap lengths (m) — only used to turn laps into metres for the schematic map.
CIRCUIT_LENGTH_M = {
    "Abu Dhabi": 5281, "Australian": 5278, "Austrian": 4318, "Azerbaijan": 6003, "Bahrain": 5412,
    "Belgian": 7004, "British": 5891, "Italian": 5793, "Japanese": 5807, "Miami": 5412,
    "Monaco": 3337, "Saudi Arabian": 6174, "Singapore": 4940, "Chinese": 5451, "Dutch": 4259,
    "Emilia Romagna": 4909, "Hungarian": 4381, "Mexico City": 4304, "Qatar": 5419,
    "Las Vegas": 6201, "São Paulo": 4309, "Spanish": 4657, "Canadian": 4361, "United States": 5513,
}


def _sec(s):
    return pd.to_timedelta(s).dt.total_seconds()


def _status_code(text) -> str:
    """FastF1 per-lap TrackStatus string ('124', '41'…) → the code that matters most."""
    s = str(text)
    for code in ("5", "4", "6", "7"):
        if code in s:
            return "6" if code == "7" else code
    return "1"


def _merge_status(codes: dict[int, str]) -> list[dict]:
    """{lap: code} → f1-race-replay's status_laps: [{status, start_lap, end_lap}] (non-green only)."""
    out = []
    for lap in sorted(codes):
        c = codes[lap]
        if c == "1":
            continue
        if out and out[-1]["status"] == c and out[-1]["end_lap"] == lap - 1:
            out[-1]["end_lap"] = lap
        else:
            out.append({"status": c, "start_lap": lap, "end_lap": lap})
    return out


class RaceFeed:
    """Builds f1-race-replay-shaped payloads for any replay time of one real race."""

    def __init__(self, laps: pd.DataFrame, season: int, race: str):
        r = laps[(laps["Season"] == season) & (laps["Race"] == race)].copy()
        if r.empty:
            raise ValueError(f"No laps for {season} {race}")
        r["end"] = _sec(r["Time"])
        r["lt"] = _sec(r["LapTime"])
        r = r.dropna(subset=["end"]).sort_values(["Driver", "LapNumber"])
        self.r, self.season, self.race = r, int(season), race
        self.total_laps = int(r["TotalLaps"].iloc[0])
        first = r[r["LapNumber"] == 1]
        self.t0 = float((first["end"] - first["lt"].fillna(first["lt"].median())).min())   # session clock
        self.length_m = float(CIRCUIT_LENGTH_M.get(circuit_name(race) or "", 5000))
        self.typical_lap = float(r.loc[r["LapNumber"] > 1, "lt"].median())
        self.colors = {d: team_colour(str(t)) for d, t in r.groupby("Driver")["Team"].first().items()}
        self.weather = {"track_temp": float(r["TrackTemp"].iloc[0]) if "TrackTemp" in r else None,
                        "air_temp": float(r["AirTemp"].iloc[0]) if "AirTemp" in r else None,
                        "rain_state": "RAINING" if "Rainfall" in r and r["Rainfall"].astype(bool).any() else "DRY"}

        # FastF1's generated placeholder rows (e.g. where a car retired) are sent like the real
        # replay does — flagged fastf1_generated — but are not line crossings on the timeline
        gen = r["FastF1Generated"].fillna(False).astype(bool) if "FastF1Generated" in r else pd.Series(False, r.index)
        acc = r["IsAccurate"].fillna(False).astype(bool) if "IsAccurate" in r else pd.Series(True, r.index)
        self.generated = {d: [{"lap": int(x.LapNumber), "time_s": -1.0, "end_time_s": float(x.end),
                               "line_time_s": float(x.end), "replay_end_time_s": float(x.end - self.t0),
                               "replay_line_time_s": float(x.end - self.t0), "tyre": TYRE_INT.get(str(x.Compound), -1),
                               "tyre_life": 0, "is_pit_entry": False, "is_pit_affected": False, "is_out_lap": False,
                               "fastf1_generated": True, "is_accurate": False, "source": "official"}
                              for x in g.itertuples()] for d, g in r[gen].groupby("Driver")}
        r = r[~gen]

        # one timeline per driver, on the REPLAY clock (0 = race start), lap by lap
        self.tl: dict[str, dict] = {}
        for d, g in r.groupby("Driver"):
            lt1 = g["lt"].iloc[0] if g["LapNumber"].iloc[0] == 1 and pd.notna(g["lt"].iloc[0]) else None
            start = (g["end"].iloc[0] - lt1 - self.t0) if lt1 is not None else 0.0
            self.tl[d] = {
                "start": float(start),
                "lap": g["LapNumber"].astype(int).tolist(),
                "line": (g["end"] - self.t0).tolist(),
                "tyre": [TYRE_INT.get(str(c), -1) for c in g["Compound"]],
                "life": [int(x) if pd.notna(x) else 0 for x in g["TyreLife"]],
                "pit_in": g["PitInTime"].notna().tolist(),
                "pit_out": g["PitOutTime"].notna().tolist(),
                "time_s": [float(x) if pd.notna(x) else -1.0 for x in g["lt"]],
                "accurate": acc[g.index].tolist(),
                "source": ["official"] * len(g),
            }
        self.r = r
        # track status by (leader) lap, like f1-race-replay's status_laps
        lead_rows = r[r["Position"] == 1].drop_duplicates("LapNumber").set_index("LapNumber")["TrackStatus"]
        self.status = {int(k): _status_code(v) for k, v in lead_rows.items()}
        self.t_end = float(max(tl["line"][-1] for tl in self.tl.values()))
        self.t_end_est = self.t_end                               # for total_frames while t_end is unknown

    # ── timeline queries ────────────────────────────────────────────────────
    def _progress(self, d: str, t: float) -> dict:
        tl = self.tl[d]
        line = np.asarray(tl["line"])
        c = int(np.searchsorted(line, t, side="right"))           # laps completed by t
        done_lap = tl["lap"][c - 1] if c else 0
        if c < len(line):                                         # on a lap that will be completed
            prev = line[c - 1] if c else tl["start"]
            rel = float(np.clip((t - prev) / max(line[c] - prev, 1e-6), 0.0, 0.9999)) if t >= prev else 0.0
            i, cur, state = c, tl["lap"][c], "running"
        elif done_lap >= self.total_laps:                         # chequered flag
            i, cur, rel, state = c - 1, self.total_laps, 0.0, "finished"
        else:                                                     # stopped on track (retired / not simulated yet)
            i, cur, state = c - 1, done_lap + 1, "stopped"
            rel = float(min((t - line[c - 1]) / self.typical_lap, 0.5)) if c else 0.0
        i = max(i, 0)
        in_pit = (tl["pit_in"][i] and rel > 0.92) or (tl["pit_out"][i] and rel < 0.08 and state == "running")
        return {"lap": int(cur), "rel": rel, "done": int(done_lap), "i": i, "state": state, "in_pit": bool(in_pit),
                "last_line": float(line[c - 1]) if c else tl["start"]}

    def _entries(self, d: str, t: float) -> list[dict]:
        tl, out = self.tl[d], []
        for k, lap in enumerate(tl["lap"]):
            line = tl["line"][k]
            if tl["source"][k] == "simulated" and line > t:       # simulated laps exist only once run
                continue
            out.append({
                "lap": lap, "time_s": tl["time_s"][k],
                "end_time_s": line + self.t0, "line_time_s": line + self.t0,           # session clock
                "replay_end_time_s": line, "replay_line_time_s": line,               # replay clock
                "tyre": tl["tyre"][k], "tyre_life": tl["life"][k],
                "is_pit_entry": bool(tl["pit_in"][k]), "is_pit_affected": bool(tl["pit_in"][k]),
                "is_out_lap": bool(tl["pit_out"][k]), "fastf1_generated": False,
                "is_accurate": bool(tl["accurate"][k]), "source": tl["source"][k],
            })
        return out + self.generated.get(d, [])

    def payload(self, t: float) -> dict:
        prog = {d: self._progress(d, t) for d in self.tl}
        # order like f1-race-replay: by lap, then distance into the lap; finishers by crossing time
        key = {d: (p["lap"] + p["rel"] - (1e-7 * p["last_line"] if p["state"] == "finished" else 0.0))
               for d, p in prog.items()}
        order = sorted(prog, key=lambda d: -key[d])
        R = self.length_m / (2 * math.pi)
        drivers = {}
        for pos, d in enumerate(order, 1):
            p, tl = prog[d], self.tl[d]
            frac = (p["lap"] - 1) + p["rel"]
            ang = 2 * math.pi * p["rel"]
            drivers[d] = {
                "position": pos, "lap": p["lap"], "rel_dist": round(p["rel"], 4),
                "dist": frac * self.length_m, "fraction": frac,
                "x": R * math.cos(ang), "y": R * math.sin(ang),
                "tyre": float(tl["tyre"][p["i"]]), "tyre_life": float(tl["life"][p["i"]]),
                "in_pit": p["in_pit"],
            }
        leader = order[0] if order else ""
        lead_lap = drivers[leader]["lap"] if leader else 1
        upto = max((p["done"] for p in prog.values()), default=0)
        status = self._status_now(lead_lap)
        hh, mm, ss = int(t // 3600), int(t % 3600 // 60), int(t % 60)
        return {
            "frame_index": int(t * FPS), "total_frames": int((self.t_end if np.isfinite(self.t_end) else self.t_end_est) * FPS),
            "frame": {"t": round(float(t), 3), "lap": lead_lap, "drivers": drivers, "weather": self.weather},
            "track_status": status, "playback_speed": 1.0, "is_paused": False,
            "circuit_length_m": self.length_m, "driver_colors": self.colors,
            "session_data": {"time": f"{hh:02}:{mm:02}:{ss:02}", "time_s": float(t), "lap": lead_lap,
                             "leader": leader, "total_laps": self.total_laps},
            "lap_times": {d: self._entries(d, t) for d in self.tl},
            # whole race, like the real replay (consumers must not read the future from it)
            "status_laps": _merge_status({k: v for k, v in self.status.items() if self._status_known(k, upto)}),
            "session_info": {"event_name": self.race, "year": self.season},
        }

    def _status_now(self, lead_lap: int) -> str:
        """
        Track status right now. The Parquet only has one status string per lap, so a Safety Car
        that comes out mid-lap would be visible from the START of that lap — before it happened.
        Report what the leader's last COMPLETED lap showed instead: up to a lap late, never early.
        (The real replay knows the exact times.)
        """
        return self.status.get(lead_lap - 1, "1")

    def _status_known(self, lap: int, upto: int) -> bool:
        return True                                               # real race: precomputed for every lap


class TakeControlFeed(RaceFeed):
    """
    Real race up to lap `from_lap`; after that, the lap-by-lap engine races on
    from the real state, rivals follow the Race Engineer's default plan, and
    clients can box any driver.
    """

    def __init__(self, laps: pd.DataFrame, season: int, race: str, from_lap: int, seed: int = 0):
        super().__init__(laps, season, race)
        from tools import race_engine as eng
        from tools.race_engineer import DRY, _caps, default_plan
        L, N = self.total_laps, int(from_lap)
        if not 1 <= N < L - 1:
            raise ValueError(f"--take-control must be between 1 and {L - 2}")
        r = self.r
        # what each car that completed lap N runs on lap N+1, and which compounds it has used
        now = r[r["LapNumber"] == N].set_index("Driver")
        nxt = r[r["LapNumber"] == N + 1].set_index("Driver")
        used = r[(r["LapNumber"] <= N) & r["Compound"].isin(DRY)].groupby("Driver")["Compound"].agg(set)
        plans = {}
        self.caps = _caps()
        for d in now.index:
            row, extra = (nxt.loc[d], 0) if d in nxt.index else (now.loc[d], 1)
            c = str(row["Compound"])
            a = int(row["TyreLife"]) + extra if pd.notna(row["TyreLife"]) else 1
            plans[d] = default_plan(c if c in DRY else "MEDIUM", a, used.get(d, set()), N, L, self.caps)
        setup, cfg, state, info = eng.state_from_real_race(
            r, season, race, N, {"dnf_prob_per_lap": 0.0}, future_plans="none", plans=plans)
        self.live = eng.LiveRace(setup, cfg, state, seed=seed)
        self.N, self.L = N, L
        drv = setup["drivers"]
        t_n = r[r["LapNumber"] == N].set_index("Driver")["end"]
        self.T0 = float(t_n.loc[drv].min() - self.t0)              # replay time the leader crosses lap N
        for d in self.tl:                                         # real data stops at lap N
            tl = self.tl[d]
            keep = sum(1 for lap in tl["lap"] if lap <= N) if d in drv else len(tl["lap"])
            for k in tl:
                if isinstance(tl[k], list):
                    del tl[k][keep:]
        self.status = {k: v for k, v in self.status.items() if k <= N}
        self.commanded: set[str] = set()
        self.log: list[str] = [f"Engine in control after lap {N}: rivals on default plans, no DNFs"]
        self._before: dict[int, tuple] = {}                       # lap → (state, draw index) before running it
        self.t_end = float("inf")                                 # known once the engine reaches the flag
        self.t_end_est = self.T0 + (L - N) * self.typical_lap

    # ── engine stepping ─────────────────────────────────────────────────────
    def _record_lap(self) -> None:
        live, s = self.live, self.live.state
        lap = live.lap + 1
        self._before[lap] = (copy.deepcopy(s), live._li)
        comp0, age0, cum0 = s["comp"][0].copy(), s["age"][0].copy(), s["cum"][0].copy()
        stops0 = s["n_stops"][0].copy()
        ev = live.step()[0]
        cum1 = live.state["cum"][0]
        for i, d in enumerate(live.setup["drivers"]):
            tl = self.tl[d]
            pitted = bool(live.state["n_stops"][0, i] > stops0[i])
            tl["lap"].append(lap)
            tl["line"].append(self.T0 + float(cum1[i]))
            tl["accurate"].append(True)
            tl["tyre"].append(int(comp0[i]))
            tl["life"].append(int(age0[i]))
            tl["pit_in"].append(pitted)
            tl["pit_out"].append(bool(tl["pit_in"][-2]) if len(tl["pit_in"]) > 1 else False)
            tl["time_s"].append(tl["line"][-1] - tl["line"][-2])   # includes a pending out-lap loss
            tl["source"].append("simulated")
        self.status[lap] = "4" if ev["safety_car"] else "1"
        if ev["sc_deployed"]:
            self.log.append(f"Lap {lap}: SAFETY CAR")
        if live.finished:
            self.t_end = max(tl["line"][-1] for tl in self.tl.values())

    def _run_until(self, t: float) -> None:
        """Step until every car's next line crossing lies beyond t."""
        cum = self.live.state["cum"][0]
        while not self.live.finished and self.T0 + float(cum.min()) <= t:
            self._record_lap()
            cum = self.live.state["cum"][0]

    def _rewind(self, lap: int) -> None:
        """Forget laps >= lap and restore the engine to the end of lap-1."""
        st, li = self._before[lap]
        self.live.state, self.live._li = copy.deepcopy(st), li
        for k in [k for k in self._before if k >= lap]:
            del self._before[k]
        for d in self.live.setup["drivers"]:
            tl = self.tl[d]
            keep = sum(1 for x in tl["lap"] if x < lap)
            for key in tl:
                if isinstance(tl[key], list):
                    del tl[key][keep:]
        self.status = {k: v for k, v in self.status.items() if k < lap}
        self.t_end = float("inf")

    # ── commands ────────────────────────────────────────────────────────────
    def command(self, cmd: dict, t: float) -> str:
        """Apply a client command at replay time t; returns a log line."""
        self._run_until(t)                                        # the car may have just started a lap
        d = str(cmd.get("driver", "")).upper()
        if d not in self.live.setup["drivers"]:
            return self._say(f"{d or '?'}: not in the simulated race")
        p = self._progress(d, t)
        if p["state"] != "running":
            return self._say(f"{d}: not running")
        action = cmd.get("cmd")
        if action == "box":
            comp = str(cmd.get("compound", "HARD")).upper()
            if comp not in ("SOFT", "MEDIUM", "HARD"):
                return self._say(f"{d}: unknown compound {comp}")
            lap = p["lap"] if p["rel"] < PIT_DEADLINE else p["lap"] + 1
            lap = max(lap, self.N + 1)
            if lap >= self.L:
                return self._say(f"{d}: too late to box")
            plan = [(lap, comp)]
            msg = f"{d}: BOX lap {lap} → {comp}" + (" (pit entry missed, next lap)" if lap > p["lap"] else "")
        elif action == "stay_out":
            lap, plan, msg = max(p["lap"], self.N + 1), [], f"{d}: stay out, no planned stops"
        else:
            return self._say(f"unknown command {action!r}")
        if self.live.lap >= lap:                                   # that lap was already run: re-run it
            self._rewind(lap)
        self.live.set_plan(d, plan)
        self.commanded.add(d)
        self._run_until(t)
        return self._say(msg)

    def _say(self, msg: str) -> str:
        self.log.append(msg)
        return msg

    # ── payload ─────────────────────────────────────────────────────────────
    def payload(self, t: float) -> dict:
        self._run_until(t)
        out = super().payload(t)
        out["take_control"] = {"from_lap": self.N, "active": t >= self.T0, "engine_lap": self.live.lap,
                               "drivers": self.live.setup["drivers"], "commanded": sorted(self.commanded),
                               "plans": {d: self.live.plan(d) for d in self.live.setup["drivers"]},
                               "log": self.log[-12:]}
        return out

    def _status_now(self, lead_lap: int) -> str:
        if lead_lap > self.N:                                     # the engine decides an SC at the start of a lap
            return self.status.get(lead_lap, "1")
        return super()._status_now(lead_lap)

    def _status_known(self, lap: int, upto: int) -> bool:
        return lap <= upto                                        # simulated laps: only once run


# ── TCP server: same protocol as f1-race-replay, plus commands from clients ──
class StreamServer:
    """One JSON object per line to every client; lines sent BY clients are queued as commands."""

    def __init__(self, port: int = PORT, host: str = "localhost"):
        self.port, self.clients, self.lock = port, [], threading.Lock()
        self.commands: queue.Queue = queue.Queue()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.port = self.sock.getsockname()[1]                     # port=0 → a free port (tests)
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
            threading.Thread(target=self._read, args=(c,), daemon=True).start()

    def _read(self, c):
        buf = b""
        while True:
            try:
                chunk = c.recv(4096)
            except OSError:
                return
            if not chunk:
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    self.commands.put(json.loads(line))
                except ValueError:
                    pass

    def broadcast(self, data: dict):
        msg = (json.dumps(data) + "\n").encode()
        with self.lock:
            for c in list(self.clients):
                try:
                    c.sendall(msg)
                except OSError:
                    self.clients.remove(c)

    def close(self):
        try:
            self.sock.shutdown(socket.SHUT_RDWR)                  # wakes the thread blocked in accept()
        except OSError:
            pass
        self.sock.close()
        with self.lock:
            for c in self.clients:
                c.close()
            self.clients = []


def parse_race(text: str) -> tuple[int, str]:
    """'2023 Bahrain Grand Prix' → (2023, 'Bahrain Grand Prix')."""
    year, _, name = text.strip().partition(" ")
    return int(year), name


def parse_terminal_command(line: str) -> dict | None:
    """'box VER HARD' / 'stay VER' → command dict."""
    parts = line.strip().split()
    if len(parts) >= 2 and parts[0].lower() == "box":
        return {"cmd": "box", "driver": parts[1], "compound": parts[2] if len(parts) > 2 else "HARD"}
    if len(parts) >= 2 and parts[0].lower() in ("stay", "stay_out"):
        return {"cmd": "stay_out", "driver": parts[1]}
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--laps", default=os.path.join(REPO_ROOT, "data", "laps_2022_2025.parquet"))
    ap.add_argument("--race", default="2023 Bahrain Grand Prix")
    ap.add_argument("--speed", type=float, default=30.0, help="race seconds per real second")
    ap.add_argument("--start-lap", type=int, default=1, help="start streaming from this lap")
    ap.add_argument("--take-control", type=int, default=None, metavar="N",
                    help="real race to lap N, then the engine races on and accepts box commands")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--hz", type=float, default=4.0, help="payloads per real second")
    args = ap.parse_args(argv)
    season, race = parse_race(args.race)
    laps = pd.read_parquet(args.laps)
    feed = (TakeControlFeed(laps, season, race, args.take_control, seed=args.seed)
            if args.take_control else RaceFeed(laps, season, race))
    server = StreamServer(args.port)
    t = 0.0
    if args.start_lap > 1:
        lines = [tl["line"][tl["lap"].index(args.start_lap - 1)] for tl in feed.tl.values()
                 if args.start_lap - 1 in tl["lap"]]
        t = float(min(lines)) if lines else 0.0
    mode = f", engine takes over after lap {args.take_control}" if args.take_control else ""
    print(f"Streaming {season} {race} ({feed.total_laps} laps) on localhost:{server.port} "
          f"at {args.speed:g}×{mode} — Ctrl+C to stop")
    if args.take_control:
        print("Commands: 'box VER HARD', 'stay VER' (or send JSON from a client)")
        threading.Thread(target=lambda: [server.commands.put(c) for c in
                                         map(parse_terminal_command, sys.stdin) if c],
                         daemon=True).start()
    dt = 1.0 / args.hz
    try:
        while t <= feed.t_end + 5:
            while not server.commands.empty():
                cmd = server.commands.get()
                if isinstance(feed, TakeControlFeed) and isinstance(cmd, dict):
                    print(feed.command(cmd, t))
            server.broadcast(feed.payload(t))
            time.sleep(dt)
            t += dt * args.speed
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
