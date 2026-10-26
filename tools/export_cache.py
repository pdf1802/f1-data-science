"""
Export every RACE session found in a FastF1 cache to one Parquet file.

    python -m tools.export_cache                       # uses $F1_CACHE_DIR (or <repo>/cache)
    python -m tools.export_cache --out data/laps.parquet

Why: since ~15 Oct 2026 F1's live-timing archive answers 401, so the cache is
the only source of lap data. Loading 50 races through FastF1 takes minutes and
needs the cache folder; one Parquet file (a few MB) can be shared, versioned
outside git and loaded in a second — which is what the experiments need.

Nothing is filtered here (wet races included): filtering is a modelling decision
and belongs in tools/training.py, where it is documented and tested.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time

import pandas as pd

from tools import shared, training

SEASONS = range(2022, 2026)           # ground-effect era; 2026 is a different car
RACE_DIR = re.compile(r"^\d{4}-\d{2}-\d{2}_Race$")


def races_in_cache(cache_dir: str, seasons=SEASONS) -> list[tuple[int, str]]:
    """(season, event name) for every folder that holds a cached Race session."""
    found = []
    for year in seasons:
        ydir = os.path.join(cache_dir, str(year))
        if not os.path.isdir(ydir):
            continue
        for ev_dir in sorted(os.listdir(ydir)):
            full = os.path.join(ydir, ev_dir)
            if not os.path.isdir(full) or not any(RACE_DIR.match(s) for s in os.listdir(full)):
                continue
            # "2023-03-05_Bahrain_Grand_Prix" → "Bahrain Grand Prix"
            event = ev_dir.split("_", 1)[1].replace("_", " ")
            if "Grand Prix" in event:          # skips testing sessions
                found.append((year, event))
    return found


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(shared.REPO_ROOT, "data", "laps_2022_2025.parquet"))
    args = ap.parse_args(argv)

    import fastf1
    logging.getLogger("fastf1").setLevel(logging.ERROR)
    cache = shared.enable_fastf1_cache()
    fastf1.Cache.offline_mode(True)                      # never hit the network
    sessions = races_in_cache(cache)
    print(f"Cache: {cache}\nRaces found in cache (2022–2025): {len(sessions)}\n")
    if not sessions:
        print("❌ No race folders found. Is F1_CACHE_DIR pointing at the folder that contains 2022/, 2023/…?")
        return 2

    t0 = time.time()
    try:
        laps, failed = training.load_races(sessions)
    except training.NoRacesLoaded as e:
        print(f"\n❌ {e}")
        return 2

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for col in laps.columns:                             # Parquet wants one type per column
        if laps[col].dtype == object:
            vals = set(laps[col].dropna().map(type))
            # True/False/None must stay boolean: as text, "False" would count as true
            laps[col] = laps[col].astype("boolean" if vals <= {bool} else "string")
    laps.to_parquet(args.out, index=False)
    report = {"races_loaded": int(laps.groupby(["Season", "Race"]).ngroups),
              "laps": len(laps), "failed": failed}
    with open(os.path.splitext(args.out)[0] + "_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n✅ {report['races_loaded']} races · {len(laps):,} laps → {args.out}")
    print(f"   {len(failed)} failed · {os.path.getsize(args.out) / 1e6:.1f} MB · {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
