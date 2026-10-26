"""
Central utility module for the F1 Data Science project.

Every tool and the Streamlit app import from here: models are loaded once,
feature contracts are defined once, physics constants live in one place.

Usage:
    from tools.shared import load_models, predict_lap_times, encode_team

Model location (first match wins):
    1. $F1_MODEL_DIR                         (explicit override)
    2. <repo>/models/                        (local machine)
    3. /content/drive/MyDrive/f1_models      (Google Colab)

Nothing is loaded at import time, so importing this module never crashes
just because the .pkl files are missing. The error appears only when a
model is actually needed, with a message saying where it looked.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_LOCAL_MODEL_DIR = os.path.join(REPO_ROOT, "models")
_COLAB_MODEL_DIR = "/content/drive/MyDrive/f1_models"
DEFAULT_CACHE_DIR = os.environ.get("F1_CACHE_DIR", os.path.join(REPO_ROOT, "cache"))

REQUIRED_MODEL_FILES = ("lap_time_model.pkl", "overtaking_model.pkl", "team_encoder.pkl")


def resolve_model_dir() -> str:
    """Return the first directory that actually contains the model files."""
    candidates = [os.environ.get("F1_MODEL_DIR"), _LOCAL_MODEL_DIR, _COLAB_MODEL_DIR]
    for path in candidates:
        if path and os.path.isfile(os.path.join(path, "lap_time_model.pkl")):
            return os.path.abspath(path)
    raise FileNotFoundError(
        "Model files not found. Looked in:\n"
        + "\n".join(f"  - {p}" for p in candidates if p)
        + "\nRun the module notebooks (they save the .pkl files) and copy them "
        "into models/, or set the F1_MODEL_DIR environment variable."
    )


def models_available() -> bool:
    try:
        resolve_model_dir()
        return True
    except FileNotFoundError:
        return False


def enable_fastf1_cache(cache_dir: str = DEFAULT_CACHE_DIR) -> str:
    """Enable the FastF1 on-disk cache (idempotent). Returns the path used."""
    import fastf1

    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    return cache_dir


# ── Physics constants (single source of truth) ───────────────────────────────
BASE_LAPTIME = 95.0       # s — anchor all LapDelta predictions are added to
PIT_COST_GREEN = 23.5     # s lost vs staying out (green flag)
PIT_COST_SC = 17.0        # s lost under Safety Car (field is slow, pit lane relatively cheap)
PIT_COST_VSC = 19.0       # s lost under Virtual Safety Car
DRS_WINDOW = 1.0          # s — gap under which an overtake attempt is possible
NOISE_STD = 0.15          # s — per-lap Gaussian noise used by the Monte Carlo
FUEL_CORRECTION = 3.5     # s — qualifying fuel (~8 kg) → race start (~105 kg)
# Lap-time cost of carrying fuel, per lap of fuel still on board.
# ~0.03 s per kg × ~1.7 kg burnt per lap ≈ 0.055 s/lap. An ASSUMPTION, not
# learned: Module 1 V3 subtracts it from the target so the model cannot
# confuse "early in the race (heavy car)" with "this compound is slow".
FUEL_S_PER_LAP = 0.055

# Backwards-compatible alias (older notebooks import this name)
PIT_COST_DEFAULT = PIT_COST_GREEN

COMPOUND_MAP = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
COMPOUND_NAMES = {v: k for k, v in COMPOUND_MAP.items()}
COMPOUND_COLOURS = {"SOFT": "#DA291C", "MEDIUM": "#FFD12E", "HARD": "#F0F0EC"}

# Module 1 inputs the TOOLS provide. A fitted model uses a subset of them —
# see model_features(): V2 used the first 7, V3 uses Compound, TyreLife, Circuit.
LAP_FEATURES = [
    "CompoundEncoded", "TyreLife", "TeamEncoded",
    "LapNumber", "TrackTemp", "AirTemp", "RainfallEncoded", "CircuitEncoded",
]

# Module 2 (V2) feature contract
OVT_FEATURES = [
    "GapAhead", "TyreDelta", "CompoundEncoded", "CompoundAheadEncoded",
    "TyreLife", "TyreLifeAhead", "LapNumber", "Position",
]

TEAM_COLOURS = {
    "Red Bull Racing": "#3671C6",
    "Ferrari":         "#E8002D",
    "Mercedes":        "#27F4D2",
    "McLaren":         "#FF8000",
    "Aston Martin":    "#229971",
    "Alpine":          "#FF87BC",
    "Williams":        "#64C4FF",
    "RB":              "#6692FF",
    "Racing Bulls":    "#6692FF",
    "AlphaTauri":      "#5E8FAA",
    "Kick Sauber":     "#52E252",
    "Alfa Romeo":      "#C92D4B",
    "Haas F1 Team":    "#B6BABD",
}


def team_colour(team: str) -> str:
    return TEAM_COLOURS.get(team, "#888888")


# Driver style feature columns — must match Module 3 V2 (DRS dropped)
STYLE_FEATURE_COLS = [
    "brake_application_pct",
    "full_throttle_pct",
    "coasting_pct",
    "throttle_smoothness",
    "min_speed_ratio",
]

# Cluster names from Module 3 V2 (driver_style_fingerprinting.ipynb)
CLUSTER_NAMES = {
    0: "McLaren Mechanical Grip",
    1: "Committed Throttle",
    2: "Late Brakers",
    3: "Smooth Precision",
}

# ── Model loading (lazy + cached) ────────────────────────────────────────────
_LAP_MODEL = None
_OVT_MODEL = None
_TEAM_ENCODER = None
_CIRCUITS: list[str] = []      # V3: circuits the models were trained on (circuits.json)
_DEG_TAIL: dict = {}           # V3.1: linear degradation beyond the data (deg_tail.json)


def load_models():
    """
    Load the three Module 1/2 artefacts. Safe to call repeatedly — cached.

    Returns (lap_model, ovt_model, team_encoder).
    """
    global _LAP_MODEL, _OVT_MODEL, _TEAM_ENCODER, _CIRCUITS, _DEG_TAIL
    if _LAP_MODEL is None:
        model_dir = resolve_model_dir()
        _LAP_MODEL = joblib.load(os.path.join(model_dir, "lap_time_model.pkl"))
        _OVT_MODEL = joblib.load(os.path.join(model_dir, "overtaking_model.pkl"))
        _TEAM_ENCODER = joblib.load(os.path.join(model_dir, "team_encoder.pkl"))
        cmap_path = os.path.join(model_dir, "compound_map.json")
        if os.path.isfile(cmap_path):
            with open(cmap_path) as f:
                saved = json.load(f)
            if saved != COMPOUND_MAP:
                raise ValueError(f"compound_map.json {saved} != shared.COMPOUND_MAP {COMPOUND_MAP}")
        circ_path = os.path.join(model_dir, "circuits.json")
        if os.path.isfile(circ_path):
            with open(circ_path) as f:
                _CIRCUITS = list(json.load(f))
        tail_path = os.path.join(model_dir, "deg_tail.json")
        if os.path.isfile(tail_path):
            with open(tail_path) as f:
                _DEG_TAIL = json.load(f)
    return _LAP_MODEL, _OVT_MODEL, _TEAM_ENCODER


def reset_models() -> None:
    """Forget cached models (used by tests to swap in stand-in models)."""
    global _LAP_MODEL, _OVT_MODEL, _TEAM_ENCODER, _CIRCUITS, _DEG_TAIL
    _LAP_MODEL = _OVT_MODEL = _TEAM_ENCODER = None
    _CIRCUITS = []
    _DEG_TAIL = {}


# ── Circuits ─────────────────────────────────────────────────────────────────
def circuit_name(event: str | None) -> str | None:
    """'Bahrain Grand Prix' / 'bahrain' → 'Bahrain' (the key used in circuits.json)."""
    if event is None or (isinstance(event, float) and np.isnan(event)):
        return None
    name = str(event).replace("Grand Prix", "").strip()
    return name[:1].upper() + name[1:] if name else None


def get_known_circuits() -> list[str]:
    load_models()
    return list(_CIRCUITS)


def encode_circuit(event: str | None) -> float:
    """
    Circuit → integer the V3 models were trained on, or NaN if unknown.
    NaN is not an error: predictions then AVERAGE over every known circuit
    ("a typical track"), which is the honest answer for a track we have no data on.
    """
    load_models()
    name = circuit_name(event)
    if name is None:
        return np.nan
    for i, c in enumerate(_CIRCUITS):
        if c.lower() == name.lower():
            return float(i)
    return np.nan


def _predict_marginal(feats: list[str], df: pd.DataFrame, predict) -> np.ndarray:
    """
    predict(df[feats]) — but rows with an unknown circuit are predicted once per
    known circuit and averaged (marginalised). Models without CircuitEncoded are
    called directly.
    """
    if "CircuitEncoded" not in feats:
        return predict(df[feats].astype(float))
    df = df.copy()
    if "CircuitEncoded" not in df:
        df["CircuitEncoded"] = np.nan
    known = df["CircuitEncoded"].notna().to_numpy()
    out = np.empty(len(df))
    if known.any():
        out[known] = predict(df.loc[known, feats].astype(float))
    if (~known).any():
        n_c = max(len(_CIRCUITS), 1)
        sub = df.loc[~known]
        rep = pd.concat([sub.assign(CircuitEncoded=float(i)) for i in range(n_c)], ignore_index=True)
        out[~known] = np.asarray(predict(rep[feats].astype(float))).reshape(n_c, len(sub)).mean(0)
    return out


# ── Encoding helpers ─────────────────────────────────────────────────────────
def get_known_teams() -> list[str]:
    _, _, enc = load_models()
    return list(enc.classes_)


def encode_team(team_name: str) -> int:
    """
    Team name → integer the model was trained on.

    FastF1 names drift between seasons ("RB" vs "Racing Bulls"), so we try an
    exact match, then a substring match, then fall back to a midfield team.
    A wrong integer is a *silent* bug (predictions change, nothing crashes),
    so the fallback warns loudly instead of quietly returning 0.
    """
    _, _, enc = load_models()
    known = list(enc.classes_)
    if team_name in known:
        return int(enc.transform([team_name])[0])
    low = str(team_name).lower()
    for k in known:
        if low in k.lower() or k.lower() in low:
            return int(enc.transform([k])[0])
    midfield = known[len(known) // 2]
    warnings.warn(f"Unknown team '{team_name}' — encoding as '{midfield}'")
    return int(enc.transform([midfield])[0])


def encode_compound(compound) -> int:
    """'SOFT' / 'medium' / 2 → integer. Raises on unknown strings."""
    if isinstance(compound, (int, np.integer)):
        return int(compound)
    key = str(compound).upper()
    if key not in COMPOUND_MAP:
        raise ValueError(f"Unknown compound: {compound}. Valid: {list(COMPOUND_MAP)}")
    return COMPOUND_MAP[key]


# ── Core prediction functions ────────────────────────────────────────────────
def predict_lap_deltas(features: pd.DataFrame | Iterable[dict]) -> np.ndarray:
    """
    Vectorised Module 1 call: one row per (driver, lap) → LapDelta seconds.

    Always call this with *all* the rows you need at once. One predict() on
    3,000 rows costs about the same as one predict() on 1 row — the overhead
    is per call, not per row.
    """
    lap_model, _, _ = load_models()
    df = pd.DataFrame(features)
    predict = lambda d: _predict_marginal(model_features(lap_model, LAP_FEATURES), d, lap_model.predict)
    return apply_deg_tail(df, predict, _DEG_TAIL)


def apply_deg_tail(df: pd.DataFrame, predict, tail: dict) -> np.ndarray:
    """
    Trees cannot extrapolate: past the tyre ages they saw often, their prediction
    goes FLAT — a 43-lap-old HARD looks as fast as a 30-lap-old one, which makes
    absurdly long stints look free. (The few very old tyres in the data are also
    survivors: stints get extended when the tyres are holding up.)
    Beyond each compound's cap (90th percentile of the training ages) we predict
    at the cap and continue with the linear degradation slope measured by the
    fixed-effects reference model (training.estimate_tyre_physics).
    """
    if not tail:
        return np.asarray(predict(df))
    df = df.copy()
    extra = np.zeros(len(df))
    comp = df["CompoundEncoded"].to_numpy(dtype=float)
    age = df["TyreLife"].to_numpy(dtype=float)
    for name, code in COMPOUND_MAP.items():
        cap, slope = tail["cap"].get(name), tail["slope"].get(name)
        if cap is None or slope is None:
            continue
        over = (comp == code) & (age > cap)
        extra[over] = slope * (age[over] - cap)
        age[over] = cap
    df["TyreLife"] = age
    return np.asarray(predict(df)) + extra


def model_features(model, default: list[str]) -> list[str]:
    """
    The columns a fitted model was trained on, in order. Lets the tools always
    pass the full LAP_FEATURES set while V2 (7 features, with LapNumber) and
    V3 (6 features, without it) models both work.
    """
    names = getattr(model, "feature_names_in_", None)
    return list(names) if names is not None else list(default)


def lap_time_from_delta(delta, pace_factor=1.0, base_laptime: float = BASE_LAPTIME):
    """
    LapDelta → absolute lap time.

    pace_factor scales the base lap (1.008 ≈ +0.76 s at 95 s). We use the
    additive form base + delta + (f-1)*base, which is what the Undercut and
    What-If tools were validated with. (The multiplicative (base+delta)*f
    differs by delta*(f-1) ≈ 0.02 s — negligible, but now it's consistent.)
    """
    return base_laptime + np.asarray(delta) + (np.asarray(pace_factor) - 1.0) * base_laptime


def predict_lap_times(rows: list[dict], pace_factors: dict,
                      base_laptime: float = BASE_LAPTIME,
                      add_noise: bool = False, rng=None) -> np.ndarray:
    """
    Predict absolute lap times for many drivers in a single model call.

    rows: dicts with the LAP_FEATURES keys plus 'team_name'.
    add_noise: True for Monte Carlo, False for deterministic strategy curves.
    """
    deltas = predict_lap_deltas(rows)
    factors = np.array([pace_factors.get(r.get("team_name"), 1.0) for r in rows])
    times = lap_time_from_delta(deltas, factors, base_laptime)
    if add_noise:
        rng = rng or np.random.default_rng()
        times = times + rng.normal(0, NOISE_STD, size=len(rows))
    return times


# Kept for old imports
predict_lap_time_batch = predict_lap_times


def build_overtake_features(gap, compound_behind, compound_ahead,
                            tyre_life_behind, tyre_life_ahead,
                            lap_number, position) -> pd.DataFrame:
    """
    Build the Module 2 feature frame. Accepts scalars or equal-length arrays.

    IMPORTANT — sign convention: in training (overtaking_model.ipynb)
        TyreDelta = TyreLife(car behind) - TyreLife(car ahead)
    so TyreDelta < 0 means the attacker is on FRESHER tyres. The notebooks'
    simulators computed ahead - behind, i.e. the opposite sign. Computing it
    here, in one place, makes that mistake impossible for every caller.
    """
    tl_b = np.atleast_1d(np.asarray(tyre_life_behind, dtype=float))
    tl_a = np.atleast_1d(np.asarray(tyre_life_ahead, dtype=float))
    n = max(len(tl_b), len(tl_a), np.size(gap))
    bc = lambda v: np.broadcast_to(np.asarray(v, dtype=float), (n,))
    return pd.DataFrame({
        "GapAhead":             bc(gap),
        "TyreDelta":            bc(tl_b) - bc(tl_a),
        "CompoundEncoded":      bc(compound_behind),
        "CompoundAheadEncoded": bc(compound_ahead),
        "TyreLife":             bc(tl_b),
        "TyreLifeAhead":        bc(tl_a),
        "LapNumber":            bc(lap_number),
        "Position":             bc(position),
    })[OVT_FEATURES]


def predict_overtake_probability(gap, compound_behind, compound_ahead,
                                 tyre_life_behind, tyre_life_ahead,
                                 lap_number, position, circuit=None):
    """
    P(car behind passes car ahead this lap). Scalars → float, arrays → array.

    circuit: event/circuit name ('Monaco Grand Prix', 'Monaco') or an already
    encoded value. V3 models know the circuit (Monaco ~0.6 % vs Spa ~6 % per
    situation); unknown → averaged over known circuits. V2 models ignore it.
    """
    _, ovt_model, _ = load_models()
    feats = build_overtake_features(gap, compound_behind, compound_ahead,
                                    tyre_life_behind, tyre_life_ahead,
                                    lap_number, position)
    enc = circuit if isinstance(circuit, (int, float, np.integer, np.floating)) else encode_circuit(circuit)
    feats["CircuitEncoded"] = enc
    probs = _predict_marginal(model_features(ovt_model, OVT_FEATURES), feats,
                              lambda X: ovt_model.predict_proba(X)[:, 1])
    return float(probs[0]) if np.ndim(gap) == 0 and len(probs) == 1 else probs


# ── Pace factors ─────────────────────────────────────────────────────────────
def pace_factors_from_times(best_times: pd.Series) -> dict:
    """{team: best_time} → {team: time / fastest}. Fastest team = 1.000."""
    best_times = best_times.dropna()
    fastest = best_times.min()
    return {team: float(t / fastest) for team, t in best_times.items()}


def calculate_pace_factors(season: int, n_races: int = 5) -> dict:
    """
    Season race pace: mean quick-lap time per team over the first N races,
    normalised to the fastest team. Slow (downloads N race sessions).
    """
    import fastf1

    team_times, team_counts = {}, {}
    for round_num in range(1, n_races + 1):
        try:
            session = fastf1.get_session(season, round_num, "R")
            session.load(telemetry=False, weather=False, messages=False)
            laps = session.laps.pick_quicklaps()
            race_avg = laps.groupby("Team")["LapTime"].mean().dt.total_seconds()
            for team, t in race_avg.items():
                team_times[team] = team_times.get(team, 0.0) + t
                team_counts[team] = team_counts.get(team, 0) + 1
        except Exception as e:  # one missing round shouldn't kill the season
            print(f"  Round {round_num} skipped: {e}")

    season_avg = pd.Series({t: team_times[t] / team_counts[t]
                            for t in team_times if team_counts[t] >= 2})
    if season_avg.empty:
        raise RuntimeError(f"Could not load any races for season {season}.")
    return pace_factors_from_times(season_avg)


# ── Telemetry utilities (Module 3) ───────────────────────────────────────────
def resample_telemetry(tel: pd.DataFrame, n_points: int = 300,
                       channels=("Speed", "Throttle", "Brake", "DRS")) -> pd.DataFrame:
    """
    Interpolate telemetry onto a fixed 0→1 relative-distance grid.

    Every circuit/lap has a different number of samples; after this, index i
    means the same fraction of the lap for every lap, so laps are comparable.
    """
    grid = np.linspace(0, 1, n_points)
    dist = tel["RelativeDistance"].values
    out = {"RelativeDistance": grid}
    for ch in channels:
        f = interp1d(dist, tel[ch].astype(float).values,
                     bounds_error=False, fill_value="extrapolate")
        out[ch] = f(grid)
    return pd.DataFrame(out)


def extract_style_features(resampled_tel: pd.DataFrame) -> dict:
    """
    The 5 Module 3 V2 style features. Definitions are copied *exactly* from
    driver_style_fingerprinting.ipynb — any drift here would put new drivers
    in the wrong cluster without any error.
    """
    throttle = resampled_tel["Throttle"].values
    brake = resampled_tel["Brake"].values
    speed = resampled_tel["Speed"].values
    return {
        "brake_application_pct": float(brake.mean()),
        "full_throttle_pct":     float((throttle > 98).mean()),
        "coasting_pct":          float(((throttle < 2) & (brake < 0.5)).mean()),
        "throttle_smoothness":   float(np.std(np.diff(throttle))),
        "min_speed_ratio":       float(speed.min() / speed.max()),
    }
