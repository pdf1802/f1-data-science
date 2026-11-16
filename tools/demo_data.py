"""
Demo bundle — everything the dashboard needs to run with no FastF1 cache and no training.

Why. FastF1 downloads from F1's live-timing archive, which answers 401 since Oct 2026, and the
trained models and the lap Parquet are not in git. A fresh clone (or a free cloud host) would
show an empty dashboard. `demo/` holds a small, read-only copy of what the tabs read:

    demo/laps.parquet        the race laps (same file as data/laps_2022_2025.parquet)
    demo/models/             Module 1 + Module 2 + encoders + degradation prior (≈ 1 MB)
    demo/qualifying.parquet  best qualifying lap per driver, every event in the local cache
    demo/results.parquet     finishing order of those events (cars running at the flag)
    demo/telemetry.parquet   fastest-lap telemetry of a few drivers in a few sessions

Build it on a machine that has the data (it only reads the local cache, never downloads):

    python -m tools.demo_data                    # → demo/

Lookup order everywhere: your own data first (data/, models/, the FastF1 cache), the bundle last.
`F1_DEMO=1` forces the bundle (what the public demo sets).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import pandas as pd

from tools.shared import REPO_ROOT, circuit_name

DEMO_DIR = os.environ.get("F1_DEMO_DIR", os.path.join(REPO_ROOT, "demo"))
MODEL_FILES = ("lap_time_model.pkl", "overtaking_model.pkl", "team_encoder.pkl", "compound_map.json",
               "circuits.json", "deg_tail.json", "degradation.json", "training_report.json")
TELEMETRY_COLS = ["Distance", "Speed", "Throttle", "Brake", "X", "Y"]
# sessions × how many of the fastest drivers get their telemetry exported
TELEMETRY_SESSIONS = [(2024, "Bahrain Grand Prix", "Q"), (2024, "Monaco Grand Prix", "Q"),
                      (2024, "Italian Grand Prix", "Q"), (2023, "Bahrain Grand Prix", "Q")]
TELEMETRY_DRIVERS = 8


def path(name: str) -> str:
    return os.path.join(DEMO_DIR, name)


def available() -> bool:
    return os.path.isfile(path("laps.parquet"))


def has_fastf1_cache() -> bool:
    """A local FastF1 cache with at least one season folder."""
    from tools.shared import DEFAULT_CACHE_DIR
    d = os.environ.get("F1_CACHE_DIR", DEFAULT_CACHE_DIR)
    return os.path.isdir(d) and any(n.isdigit() for n in os.listdir(d))


def forced() -> bool:
    """
    Use the bundle without trying FastF1 first: F1_DEMO=1, or there is no local cache at all
    (a fresh clone, the public demo) — no point waiting for a download that cannot happen.
    """
    return os.environ.get("F1_DEMO", "") == "1" or not has_fastf1_cache()


def _same_event(df: pd.DataFrame, year: int, gp: str) -> pd.DataFrame:
    """Rows of one event; 'Bahrain', 'bahrain grand prix' and 'Bahrain Grand Prix' all match."""
    want = (circuit_name(gp) or "").lower()
    return df[(df["Season"] == int(year)) & (df["Race"].map(lambda r: (circuit_name(r) or "").lower()) == want)]


def _read(name: str) -> pd.DataFrame | None:
    return pd.read_parquet(path(name)) if os.path.isfile(path(name)) else None


def events(kind: str = "qualifying") -> list[tuple[int, str]]:
    """(season, event name) pairs the bundle has — kind: 'qualifying' or 'telemetry'."""
    df = _read(f"{kind}.parquet")
    if df is None:
        return []
    pairs = df[["Season", "Race"]].drop_duplicates().sort_values(["Season", "Race"], ascending=[False, True])
    return [(int(s), r) for s, r in pairs.itertuples(index=False)]


def qualifying(year: int, gp: str) -> pd.DataFrame | None:
    """Same table as qualy_predictor.load_qualifying_results, from the bundle (None if absent)."""
    df = _read("qualifying.parquet")
    if df is None:
        return None
    q = _same_event(df, year, gp)
    if q.empty:
        return None
    event = q["Race"].iloc[0]
    q = q[["Driver", "Team", "QualyTime"]].sort_values("QualyTime").reset_index(drop=True)
    q["QualyRank"] = q.index + 1
    q["GapToPole"] = q["QualyTime"] - q["QualyTime"].iloc[0]
    q.attrs["event"] = f"{event} {year}"
    q.attrs["event_name"] = event
    return q


def race_result(year: int, gp: str) -> pd.Series | None:
    """Classified finishing position per driver, from the bundle (None if absent)."""
    df = _read("results.parquet")
    if df is None:
        return None
    r = _same_event(df, year, gp)
    return pd.Series(r["Position"].astype(int).values, index=r["Driver"].values) if len(r) else None


def telemetry_drivers(year: int, gp: str, session: str) -> list[str]:
    df = _read("telemetry.parquet")
    if df is None:
        return []
    t = _same_event(df, year, gp)
    t = t[t["Session"] == session]
    return list(t.groupby("Driver")["LapTime"].first().sort_values().index)


def fastest_lap(year: int, gp: str, session: str, driver: str) -> dict | None:
    """
    {"telemetry": DataFrame (Time as timedelta, Distance, Speed, Throttle, Brake, X, Y),
     "lap_time": s, "compound": str, "event_name": str} — or None if the bundle has no such lap.
    """
    df = _read("telemetry.parquet")
    if df is None:
        return None
    t = _same_event(df, year, gp)
    t = t[(t["Session"] == session) & (t["Driver"] == driver)]
    if t.empty:
        return None
    tel = t[TELEMETRY_COLS].reset_index(drop=True)
    tel.insert(0, "Time", pd.to_timedelta(t["TimeS"].to_numpy(), unit="s"))
    return {"telemetry": tel, "lap_time": float(t["LapTime"].iloc[0]), "compound": str(t["Compound"].iloc[0]),
            "event_name": str(t["Race"].iloc[0])}


# ── Build ────────────────────────────────────────────────────────────────────
def build(out_dir: str = DEMO_DIR, laps_path: str | None = None, model_dir: str | None = None,
          verbose: bool = True) -> dict:
    """Write the bundle from the local data, models and FastF1 cache. Returns what was exported."""
    import fastf1

    from tools import shared
    from tools.qualy_predictor import load_qualifying_results
    from tools.race_engine import actual_result
    say = print if verbose else (lambda *a, **k: None)
    laps_path = laps_path or os.path.join(REPO_ROOT, "data", "laps_2022_2025.parquet")
    model_dir = model_dir or shared.resolve_model_dir()
    os.makedirs(os.path.join(out_dir, "models"), exist_ok=True)
    shutil.copyfile(laps_path, os.path.join(out_dir, "laps.parquet"))
    for f in MODEL_FILES:
        src = os.path.join(model_dir, f)
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(out_dir, "models", f))
    shared.enable_fastf1_cache()
    all_laps = pd.read_parquet(laps_path)
    laps = all_laps[["Season", "Race"]].drop_duplicates()
    quali, results = [], []
    for season, race in sorted(laps.itertuples(index=False)):
        try:
            q = load_qualifying_results(int(season), race)
            quali.append(q[["Driver", "Team", "QualyTime"]].assign(Season=int(season), Race=q.attrs["event_name"]))
        except Exception as e:
            say(f"  no qualifying for {season} {race}: {type(e).__name__}")
            continue
        # the finishing order comes from the lap data itself (cars running at the flag): the
        # official classification needs a download that is not available offline
        res = actual_result(all_laps, int(season), race)
        results.append(pd.DataFrame({"Season": int(season), "Race": q.attrs["event_name"],
                                     "Driver": res.index, "Position": res.values}))
    tel = []
    for season, race, kind in TELEMETRY_SESSIONS:
        try:
            s = fastf1.get_session(season, race, kind)
            s.load(laps=True, telemetry=True, weather=False, messages=False)
            best = s.laps[s.laps["LapTime"].notna()].groupby("Driver")["LapTime"].min().sort_values()
            for drv in best.index[:TELEMETRY_DRIVERS]:
                lap = s.laps.pick_drivers(drv).pick_fastest()
                t = lap.get_telemetry()
                out = t[TELEMETRY_COLS].astype(float).round(2)
                out.insert(0, "TimeS", (t["Time"] - t["Time"].iloc[0]).dt.total_seconds().round(3))
                tel.append(out.assign(Season=season, Race=s.event["EventName"], Session=kind, Driver=drv,
                                      LapTime=lap["LapTime"].total_seconds(), Compound=str(lap["Compound"])))
        except Exception as e:
            say(f"  no telemetry for {season} {race} {kind}: {type(e).__name__}")
    done = {"laps": laps_path, "models": model_dir, "qualifying_events": len(quali), "result_events": len(results),
            "telemetry_laps": len(tel)}
    for name, parts in (("qualifying.parquet", quali), ("results.parquet", results), ("telemetry.parquet", tel)):
        if parts:
            pd.concat(parts, ignore_index=True).to_parquet(os.path.join(out_dir, name), index=False)
    with open(os.path.join(out_dir, "bundle.json"), "w") as f:
        json.dump({k: (os.path.basename(v) if isinstance(v, str) else v) for k, v in done.items()}, f, indent=1)
    say(f"Demo bundle written to {out_dir}: {done['qualifying_events']} qualifying sessions, "
        f"{done['result_events']} results, {done['telemetry_laps']} telemetry laps")
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEMO_DIR)
    ap.add_argument("--laps", default=None)
    ap.add_argument("--models", default=None)
    args = ap.parse_args(argv)
    build(args.out, args.laps, args.models)
    return 0


if __name__ == "__main__":
    sys.exit(main())
