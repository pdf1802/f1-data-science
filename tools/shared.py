"""
Central utility module for the F1 Data Science project.

Every tool and notebook imports from here — models are loaded once,
functions are defined once.

Usage (from any app or notebook):
    from tools.shared import load_models, predict_lap_time_batch, encode_team

Model loading strategy:
    We support two environments automatically:
        1. Local machine -> looks for ./models/ relative to project root
        2. Google Colab  -> looks for /content/drive/MyDrive/f1_models
    Set MODEL_DIR explicitly to override this.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from scipy.interpolate import interp1d   # FIX: import at top, not inside a function

warnings.filterwarnings("ignore")

# ── Path resolution ─────────────────────────────────────────────────────────
# We try local first, then Colab. Same code runs on your laptop AND in Colab.

_LOCAL_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
_COLAB_MODEL_DIR = "/content/drive/MyDrive/f1_models"


def resolve_model_dir() -> str:
    """Find where the .pkl files actually live."""
    if os.path.isdir(_LOCAL_MODEL_DIR) and os.listdir(_LOCAL_MODEL_DIR):
        return os.path.abspath(_LOCAL_MODEL_DIR)
    if os.path.isdir(_COLAB_MODEL_DIR):
        return _COLAB_MODEL_DIR
    raise FileNotFoundError(
        "No model directory found. Expected either:\n"
        f"  Local : {os.path.abspath(_LOCAL_MODEL_DIR)}\n"
        f"  Colab : {_COLAB_MODEL_DIR}\n"
        "Copy your .pkl files into one of these locations."
    )


# FIX: was `_resolve_model_dir()` (underscore) — function is `resolve_model_dir`.
# This NameError crashed the whole module on import.
MODEL_DIR = resolve_model_dir()

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_LAPTIME = 95.0   # Bahrain 2024 average racing lap (seconds).
                      # Anchor for all LapDelta predictions.

# FIX: was PIT_COST_DEFAUT (typo)
PIT_COST_DEFAULT = 23.5   # seconds lost vs staying out (Bahrain average)
PIT_COST_SC      = 17.0   # under Safety Car — field slows to ~80 km/h
PIT_COST_VSC     = 19.0   # under Virtual Safety Car

COMPOUND_MAP   = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
COMPOUND_NAMES = {v: k for k, v in COMPOUND_MAP.items()}

# Official team hex colours — used consistently across all charts and apps
TEAM_COLOURS = {
    "Red Bull Racing": "#3671C6",
    "Ferrari":         "#E8002D",
    "Mercedes":        "#27F4D2",
    "McLaren":         "#FF8000",
    "Aston Martin":    "#358C75",
    "Alpine":          "#FF87BC",
    "Williams":        "#64C4FF",
    "RB":              "#6692FF",
    "Kick Sauber":     "#52E252",
    "Haas F1 Team":    "#B6BABD",
}

# Driver style feature columns — must match Module 3 output exactly
STYLE_FEATURE_COLS = [
    "brake_application_pct",
    "full_throttle_pct",
    "coasting_pct",
    "throttle_smoothness",
    "min_speed_ratio",
]

# Cluster names from Module 3
CLUSTER_NAMES = {
    0: "Committed Throttle",
    1: "Aggressive Braker",
    2: "Balanced Midfield",
    3: "Smooth Precision",
}

# ── Model loading ─────────────────────────────────────────────────────────────
# Module-level globals with lazy loading. load_models() is called explicitly —
# never on import — so importing shared.py in a notebook that doesn't need
# models doesn't crash.

_LAP_MODEL    = None
_OVT_MODEL    = None
_TEAM_ENCODER = None


def load_models():
    """
    Load all .pkl models from MODEL_DIR.
    Safe to call multiple times — returns cached objects after first load.

    Returns
    -------
    lap_model    : XGBRegressor   — predicts LapDelta
    ovt_model    : XGBClassifier  — predicts overtake probability
    team_encoder : LabelEncoder   — converts team name → integer
    """
    global _LAP_MODEL, _OVT_MODEL, _TEAM_ENCODER

    if _LAP_MODEL is None:
        _LAP_MODEL = joblib.load(os.path.join(MODEL_DIR, "lap_time_model.pkl"))
        print("Lap time model loaded.")
    if _OVT_MODEL is None:
        # FIX: was "overtake_model.pkl" — Module 2 saves "overtaking_model.pkl"
        _OVT_MODEL = joblib.load(os.path.join(MODEL_DIR, "overtaking_model.pkl"))
        print("Overtake model loaded.")
    if _TEAM_ENCODER is None:
        _TEAM_ENCODER = joblib.load(os.path.join(MODEL_DIR, "team_encoder.pkl"))
        print("Team encoder loaded.")

    return _LAP_MODEL, _OVT_MODEL, _TEAM_ENCODER


# ── Encoding helpers ──────────────────────────────────────────────────────────

def encode_team(team_name: str) -> int:
    """
    Convert a team name string to the integer the model was trained on.

    XGBoost stores TeamEncoded as a numeric feature. Passing a raw string
    crashes; passing the wrong integer silently gives wrong predictions.
    This function is the single source of truth for that mapping.

    Falls back to 0 for unknown teams — check get_known_teams() first if
    exact accuracy matters.
    """
    if _TEAM_ENCODER is None:
        load_models()
    try:
        return int(_TEAM_ENCODER.transform([team_name])[0])
    except Exception:   # FIX: bare `except:` → `except Exception:` (never swallow everything)
        return 0

def encode_compound(compound: str) -> int:
    """Convert tyre compound string → integer."""
    if compound not in COMPOUND_MAP:
        raise ValueError(
            f"Unknown compound: {compound}. "
            f"Valid options: {list(COMPOUND_MAP.keys())}"
        )
    return COMPOUND_MAP[compound]


def get_known_teams() -> list[str]:
    """Return the list of team names the model was trained on."""
    if _TEAM_ENCODER is None:
        load_models()
    return list(_TEAM_ENCODER.classes_)


# ── Core prediction functions ─────────────────────────────────────────────────

def predict_lap_time_batch(rows: list[dict],
                           pace_factors: dict,
                           base_laptime: float = BASE_LAPTIME,
                           add_noise: bool = False) -> np.ndarray:
    """
    Predict lap times for one or more drivers in a single model call.

    The most-used function in the project. Every tool that projects future
    lap times calls this.

    Parameters
    ----------
    rows : list of dicts, each with keys:
        CompoundEncoded, TyreLife, TeamEncoded, LapNumber,
        TrackTemp, AirTemp, RainfallEncoded, team_name
    pace_factors : dict {team_name: float} — from calculate_pace_factors()
    base_laptime : float — base anchor in seconds (default 95.0 = Bahrain)
    add_noise    : bool  — add ±0.15s gaussian noise (True for simulation,
                           False for strategy tools where you want clean curves)

    Returns
    -------
    np.ndarray of predicted lap times, one per row, in seconds
    """
    if _LAP_MODEL is None:
        load_models()

    feature_cols = ["CompoundEncoded", "TyreLife", "TeamEncoded",
                    "LapNumber", "TrackTemp", "AirTemp", "RainfallEncoded"]

    df = pd.DataFrame(rows)[feature_cols].copy()
    deltas = _LAP_MODEL.predict(df)

    # FIX: parenthesis was misplaced — .get needs (key, default) and the
    # list comprehension must close before np.array wraps it.
    factors = np.array([pace_factors.get(r.get("team_name"), 1.0) for r in rows])

    times = (base_laptime + deltas) * factors

    if add_noise:
        times += np.random.normal(0, 0.15, size=len(rows))
    return times


def predict_overtake_probability(gap: float,
                                 tyre_delta: int,
                                 compound_behind: int,
                                 compound_ahead: int,
                                 tyre_life_behind: int,
                                 tyre_life_ahead: int,
                                 lap_number: int,
                                 position: int) -> float:
    """
    Predict the probability that the car behind overtakes the car ahead.

    Returns
    -------
    float in [0, 1] — probability of an overtake this lap
    """
    if _OVT_MODEL is None:
        load_models()

    features = pd.DataFrame([{
        "GapAhead":             gap,
        "TyreDelta":            tyre_delta,
        "CompoundEncoded":      compound_behind,
        "CompoundAheadEncoded": compound_ahead,
        "TyreLife":             tyre_life_behind,
        "TyreLifeAhead":        tyre_life_ahead,
        "LapNumber":            lap_number,
        "Position":             position,
    }])

    return float(_OVT_MODEL.predict_proba(features)[0][1])


# ── Pace factors ──────────────────────────────────────────────────────────────

def calculate_pace_factors(season: int, n_races: int = 5) -> dict:
    # FIX: parameter renamed n_race → n_races (loop below uses n_races)
    """
    Derive {team: factor} from the first N races of a season using FastF1.

    factor = 1.000 for the fastest team. Everyone else > 1.000.
    A factor of 1.008 means ~0.76s/lap slower at 95s base pace.

    Requires FastF1 cache. Downloads data on first call.
    """
    try:
        import fastf1
    except ImportError:
        # FIX: "pip installed fastf1" → "pip install fastf1"
        raise ImportError("fastf1 not installed. Run: pip install fastf1")

    team_times = {}
    team_counts = {}

    for round_num in range(1, n_races + 1):
        try:
            session = fastf1.get_session(season, round_num, "R")
            session.load(telemetry=False)
            laps = session.laps.pick_quicklaps()
            laps = laps[laps["LapTime"].notna()]
            # FIX: .dt_total_seconds() → .dt.total_seconds()
            race_avg = laps.groupby("Team")["LapTime"].mean().dt.total_seconds()

            for team, avg_time in race_avg.items():
                team_times[team]  = team_times.get(team, 0) + avg_time
                team_counts[team] = team_counts.get(team, 0) + 1
        except Exception as e:
            print(f"  Round {round_num} skipped: {e}")

    season_avg = {
        team: team_times[team] / team_counts[team]
        for team in team_times
        if team_counts[team] >= 2
    }

    if not season_avg:
        raise RuntimeError(f"Could not load any races for season {season}.")

    fastest = min(season_avg.values())
    return {team: avg / fastest for team, avg in season_avg.items()}


# ── Telemetry utilities ───────────────────────────────────────────────────────
# FIX: these two were accidentally indented INSIDE calculate_pace_factors,
# which made them local functions you couldn't import. Now at module level.

def resample_telemetry(tel, n_points: int = 300) -> pd.DataFrame:
    """
    Interpolate a telemetry DataFrame onto a fixed distance grid.

    Every circuit has a different lap length → different telemetry point
    counts. We normalise to n_points on a 0→1 relative-distance grid so
    features are comparable across circuits.

    Parameters
    ----------
    tel      : FastF1 telemetry DataFrame (must have RelativeDistance col)
    n_points : number of grid points (300 = enough resolution, not too noisy)

    Returns
    -------
    DataFrame with columns Speed, Throttle, Brake, DRS on a uniform grid
    """
    grid     = np.linspace(0, 1, n_points)
    dist     = tel["RelativeDistance"].values
    channels = ["Speed", "Throttle", "Brake", "DRS"]
    result   = {"RelativeDistance": grid}

    for ch in channels:
        values = tel[ch].astype(float).values
        f = interp1d(dist, values, bounds_error=False, fill_value="extrapolate")
        result[ch] = f(grid)

    return pd.DataFrame(result)


def extract_style_features(resampled_tel: pd.DataFrame) -> dict:
    """
    Extract 5 driving-style features from a resampled telemetry lap.

    These are the exact features used in Module 3 clustering.
    Must stay in sync with STYLE_FEATURE_COLS.

    Returns
    -------
    dict with keys matching STYLE_FEATURE_COLS
    """
    throttle = resampled_tel["Throttle"].values   # 0–100
    brake    = resampled_tel["Brake"].values       # 0.0 or 1.0
    speed    = resampled_tel["Speed"].values

    # 1. Brake pedal applied % of lap — main tyre killer
    brake_application_pct = float(brake.mean())

    # 2. Full throttle % — commitment on straights (threshold: 98%)
    full_throttle_pct = float((throttle > 98).mean())

    # 3. Coasting % — neither throttle nor brake (smooth transition)
    coasting_pct = float(((throttle < 5) & (brake < 0.5)).mean())

    # 4. Throttle smoothness — low std = smooth, high std = aggressive
    throttle_smoothness = float(np.std(np.diff(throttle)))

    # 5. Minimum speed ratio — how slow they go in the slowest corner
    min_speed_ratio = float(np.percentile(speed, 5) / (np.mean(speed) + 1e-6))

    return {
        "brake_application_pct": brake_application_pct,
        "full_throttle_pct":     full_throttle_pct,
        "coasting_pct":          coasting_pct,
        "throttle_smoothness":   throttle_smoothness,
        "min_speed_ratio":       min_speed_ratio,
    }