"""
Run the REAL f1-race-replay offline, on a FastF1 cache you already have.

    python -m integrations.f1_race_replay.run_real_replay --list
    python -m integrations.f1_race_replay.run_real_replay --race "2023 Bahrain Grand Prix"
    python -m integrations.f1_race_replay.race_engineer_window --driver ALO      (second terminal)

Why this exists: since mid-October 2026 F1's live-timing archive answers 401 to
unauthenticated downloads (FastF1 issue #978, f1-race-replay issue #354), so
f1-race-replay can only replay sessions that are already in a local FastF1
cache — and it needs the full session there (car data AND position data). This
launcher points it at your cache, switches FastF1 to offline mode (no 401s, no
waiting on timeouts) and covers the two calls that need the internet even for a
cached race:

  * circuit rotation (MultiViewer API) → fallback: rotate the track so its longest
    axis is horizontal (principal axis of the fastest lap's X/Y)
  * championship standings (Ergast) → f1-race-replay already shows none when it fails

Everything else is f1-race-replay itself, unchanged: the arcade replay with its
telemetry stream on localhost:9999, which the Race Engineer window listens to.
f1-race-replay is not part of this repo: clone it next to it
(`git clone https://github.com/IAmTomShaw/f1-race-replay ../f1-race-replay`,
`pip install -r ../f1-race-replay/requirements.txt`) or pass --replay-dir.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NEEDED = ("car_data.ff1pkl", "position_data.ff1pkl", "timing_app_data.ff1pkl", "_extended_timing_data.ff1pkl")


def default_cache() -> Path:
    """$F1_CACHE_DIR, else cache_drive/f1_cache, else cache/ — the same order as tools/shared.py."""
    env = os.environ.get("F1_CACHE_DIR")
    if env:
        return Path(env)
    for p in (REPO_ROOT / "cache_drive" / "f1_cache", REPO_ROOT / "cache"):
        if p.is_dir():
            return p
    return REPO_ROOT / "cache"


def default_replay_dir() -> Path:
    env = os.environ.get("F1_RACE_REPLAY_DIR")
    return Path(env) if env else REPO_ROOT.parent / "f1-race-replay"


def replayable_sessions(cache: Path, session: str = "Race") -> list[tuple[int, str, Path]]:
    """(year, event name, folder) of every cached session that has everything a replay needs."""
    out = []
    for folder in sorted(cache.glob(f"*/*/*_{session}")):
        if all((folder / f).exists() for f in NEEDED):
            year = int(folder.parent.parent.name)
            event = folder.parent.name.split("_", 1)[1].replace("_", " ")
            out.append((year, event, folder))
    return out


def principal_axis_rotation(x, y) -> float:
    """Degrees to rotate a track so its longest axis is horizontal (stand-in for MultiViewer's rotation)."""
    import numpy as np
    pts = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 10:
        return 0.0
    pts -= pts.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov(pts.T))
    vx, vy = vecs[:, -1]                                      # eigenvector of the largest eigenvalue
    return -math.degrees(math.atan2(vy, vx))


def configure_fastf1(cache: Path, offline: bool) -> None:
    import fastf1
    if hasattr(fastf1.Cache, "configure"):                    # FastF1 >= 3.6
        try:
            fastf1.Cache.configure(cache_dir=str(cache))
        except TypeError:
            fastf1.Cache.enable_cache(str(cache))
    else:
        fastf1.Cache.enable_cache(str(cache))
    if offline:
        fastf1.Cache.offline_mode(True)


def patch_f1_race_replay(rr_main, offline: bool) -> None:
    """Make f1-race-replay's main module work on a cache-only session."""
    rr_main.enable_cache = lambda: None                       # already configured, on OUR cache path
    original = rr_main.get_circuit_rotation

    def get_circuit_rotation(session):
        if not offline:
            try:
                return original(session)
            except Exception as e:                            # noqa: BLE001 — any network failure
                print(f"Circuit rotation unavailable ({type(e).__name__}); using the track's principal axis")
        lap = session.laps.pick_fastest()
        pos = lap.get_pos_data()
        rot = principal_axis_rotation(pos["X"], pos["Y"])
        print(f"Circuit rotation: {rot:.0f}° (principal axis — MultiViewer's value needs the internet)")
        return rot

    rr_main.get_circuit_rotation = get_circuit_rotation


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--race", default="2023 Bahrain Grand Prix", help='e.g. "2023 Bahrain Grand Prix"')
    ap.add_argument("--cache", type=Path, default=default_cache(), help="FastF1 cache folder (default: "
                    "$F1_CACHE_DIR or cache_drive/f1_cache)")
    ap.add_argument("--replay-dir", type=Path, default=default_replay_dir(),
                    help="f1-race-replay clone (default: $F1_RACE_REPLAY_DIR or ../f1-race-replay)")
    ap.add_argument("--speed", type=float, default=1.0, help="initial playback speed")
    ap.add_argument("--online", action="store_true", help="let FastF1 try the internet (it answers 401 today)")
    ap.add_argument("--list", action="store_true", help="list the races in the cache that can be replayed")
    args = ap.parse_args(argv)

    cache = args.cache.resolve()
    races = replayable_sessions(cache)
    if args.list or not races:
        print(f"Replayable races in {cache} (car + position data cached):")
        for year, event, _ in races:
            print(f"  {year} {event}")
        if not races:
            print("  none — f1-race-replay needs a session loaded with telemetry before the 401 started")
        return 0 if races else 1

    year_s, _, event = args.race.strip().partition(" ")
    year = int(year_s)
    if not any(y == year and e.lower() == event.lower() for y, e, _ in races):
        print(f"{args.race} is not fully cached. Choose one of:")
        for y, e, _ in races:
            print(f"  {y} {e}")
        return 1

    replay_dir = args.replay_dir.resolve()
    if not (replay_dir / "main.py").exists() or not (replay_dir / "src" / "f1_data.py").exists():
        print(f"f1-race-replay not found in {replay_dir}.\n"
              "  git clone https://github.com/IAmTomShaw/f1-race-replay ../f1-race-replay\n"
              "  pip install -r ../f1-race-replay/requirements.txt\n"
              "or pass --replay-dir / set F1_RACE_REPLAY_DIR.")
        return 1

    offline = not args.online
    configure_fastf1(cache, offline)
    import fastf1
    rnd = int(fastf1.get_event(year, event)["RoundNumber"])

    # f1-race-replay uses paths relative to its own folder (computed_data/, images/)
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(replay_dir))
    os.chdir(replay_dir)
    import logging
    logging.getLogger("fastf1").setLevel(logging.WARNING)
    import main as rr_main                                    # f1-race-replay's main.py
    patch_f1_race_replay(rr_main, offline)
    print(f"f1-race-replay: {year} {event} (round {rnd}) from {cache}"
          f"{' — offline' if offline else ''}. Telemetry stream on localhost:9999:\n"
          f"  python -m integrations.f1_race_replay.race_engineer_window --driver ALO")
    rr_main.main(year, rnd, args.speed, session_type="R", visible_hud=True, show_telemetry_viewer=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
