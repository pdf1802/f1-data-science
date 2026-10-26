"""
Per-circuit race constants, measured from the lap data instead of assumed.

    python -m tools.circuit_profiles --laps data/laps_2022_2025.parquet

Writes tools/circuit_profiles.json (committed: small, derived statistics, no raw
data), which the simulator and the dashboard use as defaults:

    laps           race distance (most common TotalLaps among dry races)
    base_lap_s     median green-flag racing lap (no in/out laps), recent seasons first
    pit_loss_s     median time lost to a green-flag stop:
                   (in-lap + out-lap) − 2 × the driver's median green lap in that race
    sc_per_lap     Safety Car deployments per racing lap, shrunk towards the
                   all-circuit rate (empirical Bayes) — with 2–4 races per circuit
                   one chaotic race would otherwise dominate
    races          dry races behind each number

Before this, every circuit used 95 s laps, 23.5 s pit loss and the same SC rate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from tools.shared import REPO_ROOT, circuit_name

PROFILE_PATH = os.path.join(os.path.dirname(__file__), "circuit_profiles.json")
PRIOR_RACES = 2.0          # shrinkage strength: the global rate counts as this many races


def _sec(s):
    return pd.to_timedelta(s).dt.total_seconds()


def sc_deployments(race_laps: pd.DataFrame) -> int:
    """Number of Safety Car periods (status '4') in one race, counted at the leader's laps."""
    sc_by_lap = (race_laps.assign(SC=race_laps["TrackStatus"].astype(str).str.contains("4"))
                 .groupby("LapNumber")["SC"].any().sort_index())
    return int((sc_by_lap & ~sc_by_lap.shift(1, fill_value=False)).sum())


def pit_losses(race_laps: pd.DataFrame) -> pd.Series:
    """Seconds lost per green-flag stop: in-lap + out-lap − 2 × driver's median green lap."""
    d = race_laps.copy()
    d["LT"] = _sec(d["LapTime"])
    green = (d["TrackStatus"].astype(str) == "1")
    normal = green & d["PitInTime"].isna() & d["PitOutTime"].isna() & (d["LapNumber"] > 1)
    ref = d[normal].groupby("Driver")["LT"].median()
    d = d.sort_values(["Driver", "LapNumber"])
    nxt = d.groupby("Driver")[["LT", "PitOutTime", "TrackStatus"]].shift(-1)
    stop = d["PitInTime"].notna() & nxt["PitOutTime"].notna() & green & (nxt["TrackStatus"].astype(str) == "1")
    loss = d.loc[stop, "LT"] + nxt.loc[stop, "LT"] - 2 * d.loc[stop, "Driver"].map(ref)
    return loss[(loss > 10) & (loss < 45)]       # drop slow stops / damage / drive-throughs


def build_profiles(laps: pd.DataFrame) -> dict:
    from tools.training import dry_races
    dry, _ = dry_races(laps)
    rows = []
    for (season, race), g in dry.groupby(["Season", "Race"]):
        lt = _sec(g["LapTime"])
        normal = (g["TrackStatus"].astype(str) == "1") & g["PitInTime"].isna() & g["PitOutTime"].isna() \
            & (g["LapNumber"] > 1)
        rows.append({"circuit": circuit_name(race), "season": season, "laps": int(g["TotalLaps"].iloc[0]),
                     "base_lap_s": float(lt[normal].median()), "sc": sc_deployments(g),
                     "pit_losses": pit_losses(g).tolist()})
    r = pd.DataFrame(rows)
    global_rate = r["sc"].sum() / r["laps"].sum()
    profiles = {}
    for c, g in r.groupby("circuit"):
        g = g.sort_values("season")
        stops = [x for xs in g["pit_losses"] for x in xs]
        shrunk = (g["sc"].sum() + PRIOR_RACES * global_rate * g["laps"].mean()) / \
                 (g["laps"].sum() + PRIOR_RACES * g["laps"].mean())
        profiles[c] = {
            "laps": int(g["laps"].mode().max()),
            "base_lap_s": round(float(g["base_lap_s"].iloc[-2:].mean()), 2),   # recent cars are faster
            "pit_loss_s": round(float(np.median(stops)), 1) if len(stops) >= 5 else None,
            "pit_stops_measured": len(stops),
            "sc_per_lap": round(float(shrunk), 4),
            "sc_deployments": int(g["sc"].sum()),
            "races": [f"{s}" for s in g["season"]],
        }
    profiles["_global"] = {"sc_per_lap": round(float(global_rate), 4),
                           "pit_loss_s": round(float(np.median([x for xs in r["pit_losses"] for x in xs])), 1)}
    return profiles


def load_profiles() -> dict:
    if not os.path.isfile(PROFILE_PATH):
        return {}
    with open(PROFILE_PATH) as f:
        return json.load(f)


def profile_for(event: str | None) -> dict | None:
    """Profile of a circuit by event or circuit name ('British Grand Prix' / 'British'), or None."""
    name = circuit_name(event)
    if name is None:
        return None
    for k, v in load_profiles().items():
        if k.lower() == name.lower():
            return v
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--laps", default=os.path.join(REPO_ROOT, "data", "laps_2022_2025.parquet"))
    ap.add_argument("--out", default=PROFILE_PATH)
    args = ap.parse_args(argv)
    profiles = build_profiles(pd.read_parquet(args.laps))
    with open(args.out, "w") as f:
        json.dump(profiles, f, indent=1)
    table = pd.DataFrame({k: v for k, v in profiles.items() if not k.startswith("_")}).T
    print(table[["laps", "base_lap_s", "pit_loss_s", "pit_stops_measured", "sc_per_lap", "sc_deployments"]]
          .to_string())
    print(f"\nall circuits: {profiles['_global']}  →  {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
