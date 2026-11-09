"""
Circuit outlines from a FastF1 cache — the shape of each track, for drawing cars on it.

    python -m tools.circuit_outlines                 # writes tools/circuit_outlines.json

For every circuit in the lap Parquet, the fastest qualifying lap found in the cache
(2025, else 2024, else the race) gives the GPS line (FastF1 X/Y, 1/10 m). It is
resampled to N points EVENLY SPACED BY DISTANCE, starting at the timing line, so
point k is k/N of the way round: a car's fraction of the lap (rel_dist) maps
straight onto the outline. The JSON is small and committed; the stand-in stream
sends it as `track_geometry`, like f1-race-replay does.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

from tools.shared import REPO_ROOT, circuit_name, enable_fastf1_cache

OUT = os.path.join(REPO_ROOT, "tools", "circuit_outlines.json")
N_POINTS = 400


def resample_closed(x, y, n: int = N_POINTS) -> tuple[np.ndarray, np.ndarray, float]:
    """Resample a lap's X/Y to n points evenly spaced in distance; returns (x, y, length in metres)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    seg = np.hypot(np.diff(x), np.diff(y))
    keep = np.concatenate([[True], seg > 0])                 # drop repeated samples
    x, y = x[keep], y[keep]
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    grid = np.linspace(0, s[-1], n, endpoint=False)
    return np.interp(grid, s, x), np.interp(grid, s, y), float(s[-1]) / 10.0


def principal_axis_rotation(x, y) -> float:
    """Degrees that make the track's longest axis horizontal (display only)."""
    pts = np.column_stack([x, y]) - np.mean(np.column_stack([x, y]), axis=0)
    _, vecs = np.linalg.eigh(np.cov(pts.T))
    vx, vy = vecs[:, -1]
    return -math.degrees(math.atan2(vy, vx))


def outline_from_session(session) -> dict:
    lap = session.laps.pick_fastest()
    pos = lap.get_pos_data()
    x, y, length = resample_closed(pos["X"], pos["Y"])
    return {"x": np.round(x, 1).tolist(), "y": np.round(y, 1).tolist(), "length_m": round(length, 1),
            "rotation_deg": round(principal_axis_rotation(x, y), 1),
            "source": f"{session.event.year} {session.event['EventName']} {session.name}, {lap['Driver']} fastest lap"}


def build(races: list[tuple[int, str]]) -> dict:
    import fastf1
    out = {}
    by_circuit: dict[str, list[tuple[int, str]]] = {}
    for season, race in races:
        by_circuit.setdefault(circuit_name(race), []).append((season, race))
    for circuit, events in sorted(by_circuit.items()):
        tried = []
        for season, race in sorted(events, reverse=True):
            for kind in ("Q", "R"):
                tried.append(f"{season} {kind}")
                try:
                    s = fastf1.get_session(season, race, kind)
                    s.load(laps=True, telemetry=True, weather=False, messages=False)
                    out[circuit] = outline_from_session(s)
                    break
                except Exception:
                    continue
            if circuit in out:
                break
        print(f"{circuit:<15} {out[circuit]['source'] if circuit in out else 'not in the cache (' + ', '.join(tried) + ')'}")
    return out


def load_outlines() -> dict:
    try:
        with open(OUT) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def outline_for(event: str | None) -> dict | None:
    name = circuit_name(event)
    return load_outlines().get(name) if name else None


def main(argv=None):
    import pandas as pd
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--laps", default=os.path.join(REPO_ROOT, "data", "laps_2022_2025.parquet"))
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    import fastf1
    fastf1.set_log_level("ERROR")
    print("FastF1 cache:", enable_fastf1_cache())
    laps = pd.read_parquet(args.laps, columns=["Season", "Race"])
    races = sorted({(int(s), str(r)) for s, r in laps.drop_duplicates().itertuples(index=False)})
    outlines = build(races)
    with open(args.out, "w") as f:
        json.dump(outlines, f, separators=(",", ":"))
    print(f"{len(outlines)} outlines → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
