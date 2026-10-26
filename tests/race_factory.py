"""Synthetic FastF1-shaped races for the live-feed tests (replay stream, Race Engineer, window)."""
import numpy as np
import pandas as pd

OFFSET = {"SOFT": 0.0, "MEDIUM": 0.4, "HARD": 0.8}
DEG = {"SOFT": 0.09, "MEDIUM": 0.05, "HARD": 0.03}
TEAMS = ["Red Bull Racing", "Ferrari", "Mercedes", "McLaren", "Aston Martin",
         "Alpine", "Williams", "RB", "Kick Sauber", "Haas F1 Team"]
RACE = "Bahrain Grand Prix"          # a circuit the stand-in models know (fast: no marginalising)


def make_race(n_laps=30, slow=None, retire=None, season=2099, race=RACE, seed=0, sc_laps=()):
    """
    20 cars 2 s apart, one stop each (SOFT/MEDIUM → HARD), times on the FastF1
    session clock (lights out at 3600 s).
    slow:    {driver: extra s/lap}  — e.g. a backmarker who gets lapped
    retire:  {driver: last lap completed}
    sc_laps: leader laps run under the Safety Car (TrackStatus '4', slow laps)
    """
    rng = np.random.default_rng(seed)
    slow, retire = slow or {}, retire or {}
    rows = []
    for i in range(20):
        d = f"D{i:02d}"
        first = ["SOFT", "MEDIUM"][i % 2]
        pit = 10 + i % 8
        t, age, comp, stint = 2.0 * i, 3 if first == "SOFT" else 1, first, 1
        pace = 90 + 0.1 * i + slow.get(d, 0.0)
        for lap in range(1, n_laps + 1):
            if lap > retire.get(d, n_laps):
                break
            sc = lap in sc_laps
            lt = (pace + OFFSET[comp] + DEG[comp] * age + rng.normal(0, 0.1)
                  + (3.0 if lap == 1 else 0.0) + (20.0 if lap == pit else 0.0) + (30.0 if sc else 0.0))
            t += lt
            rows.append({"Driver": d, "Team": TEAMS[i // 2], "LapNumber": float(lap),
                         "LapTime": pd.Timedelta(seconds=lt), "Time": pd.Timedelta(seconds=3600 + t),
                         "Compound": comp, "TyreLife": float(age), "Stint": float(stint),
                         "PitInTime": pd.Timedelta(seconds=3600 + t) if lap == pit else pd.NaT,
                         "PitOutTime": pd.Timedelta(seconds=3600 + t - lt) if lap == pit + 1 else pd.NaT,
                         "TrackStatus": "4" if sc else "1", "TrackTemp": 35.0, "AirTemp": 25.0,
                         "Rainfall": False})
            if lap == pit:
                comp, age, stint = "HARD", 1, 2
            else:
                age += 1
    df = pd.DataFrame(rows)
    df["Position"] = df.groupby("LapNumber")["Time"].rank(method="first").astype(float)
    df["Season"], df["Race"], df["TotalLaps"] = season, race, n_laps
    return df


def line(feed, driver, lap):
    """Replay-clock time `driver` crosses the line at the end of `lap`."""
    tl = feed.tl[driver]
    return tl["line"][tl["lap"].index(lap)]


def stream(feed, engineer, t_from, t_to, dt=2.0):
    """Feed payloads every dt seconds; return the calls made."""
    calls, t = [], t_from
    while t <= t_to:
        c = engineer.update(feed.payload(t))
        if c:
            calls.append(c)
        t += dt
    return calls
