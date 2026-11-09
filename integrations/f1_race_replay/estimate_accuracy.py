"""
How accurate are the Race Engineer's estimated line times?

    python -m integrations.f1_race_replay.estimate_accuracy

When the followed driver crosses the line at the end of lap N, the cars behind (and
lapped cars) have not crossed it yet: their lap-N line time is estimated. This script
replays real races through the stand-in stream, makes that estimate at every lap, and
compares it with the time the car really crossed.

Three ways to estimate, on purpose:
  engineer              what the Race Engineer does: last crossing + recent pace (rel_dist unused)
  rel_dist last 10 %    the previous version: the frame's rel_dist in the last 10 % of the lap
  rel_dist everywhere   rel_dist all round the lap

On this stand-in stream rel_dist looks good only because the stand-in builds it from
the lap's REAL duration (elapsed time / lap time), so part of the answer is baked in.
On real f1-race-replay frames (distance-based) it is worse than pace alone:
see real_payloads.py, which runs the same measurement on the real replay.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

from integrations.f1_race_replay.replay_stream import RaceFeed
from tools import race_engineer as re_
from tools.shared import REPO_ROOT

MODES = (("engineer", re_.REL_TRUST), ("rel_dist last 10 %", 0.9), ("rel_dist everywhere", 0.0))
DEFAULT = [("2023 Bahrain Grand Prix", "ALO"), ("2024 Italian Grand Prix", "LEC"), ("2025 Monaco Grand Prix", "NOR")]


def measure(laps: pd.DataFrame, season: int, race: str, driver: str) -> pd.DataFrame:
    feed = RaceFeed(laps, season, race)
    tl, rows = feed.tl[driver], []
    for n in range(4, feed.total_laps - 1):
        if n not in tl["lap"]:
            continue
        t = tl["line"][tl["lap"].index(n)] + 0.2
        p = feed.payload(t)
        done, _ = re_.laps_from_payload(p, race, feed.total_laps)
        trust = re_.REL_TRUST
        for mode, rel_trust in MODES:
            re_.REL_TRUST = rel_trust
            try:
                snap, _, _, est = re_.snapshot_at_lap(done, n, t, p["frame"]["drivers"])
            finally:
                re_.REL_TRUST = trust
            at_n = snap[snap["LapNumber"] == n].set_index("Driver")["Time"].dt.total_seconds()
            for d in est:
                other = feed.tl[d]
                if n not in other["lap"]:
                    continue
                k = other["lap"].index(n)
                rows.append({"race": f"{season} {race}", "lap": n, "driver": d, "mode": mode,
                             "error_s": at_n[d] - other["line"][k], "seconds_to_line": other["line"][k] - t,
                             "pit_lap": bool(other["pit_in"][k] or other["pit_out"][k])})
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(abs_error=df["error_s"].abs(),
                   case=df["pit_lap"].map({False: "no stop in that lap", True: "lap with a stop"}))
    g = df.groupby(["mode", "case"])["abs_error"]
    out = pd.DataFrame({"n": g.size(), "median_s": g.median(), "p90_s": g.quantile(0.9)})
    near = df[~df["pit_lap"] & (df["seconds_to_line"] <= 10)].groupby("mode")["abs_error"].median()
    return out.round(2), near.round(3)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--laps", default=os.path.join(REPO_ROOT, "data", "laps_2022_2025.parquet"))
    args = ap.parse_args(argv)
    laps = pd.read_parquet(args.laps)
    parts = []
    for text, driver in DEFAULT:
        year, _, race = text.partition(" ")
        parts.append(measure(laps, int(year), race, driver))
        print(f"{text} ({driver}): {len(parts[-1]) // 3} estimates", flush=True)
    table, near = summary(pd.concat(parts, ignore_index=True))
    print("\n|error| of the estimated line time (s)")
    print(table.to_string())
    print("\nmedian |error| for cars within 10 s of the line (no stop):")
    print(near.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
