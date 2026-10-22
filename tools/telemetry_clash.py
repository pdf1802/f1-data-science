"""
Tool 6 — Telemetry Clash.

Compares two drivers' fastest laps on a shared distance grid, detects corners
from GPS curvature, and computes speed + cumulative-time deltas.

Public API:
    load_session(year, gp, session_type)            -> fastf1 Session
    compare_drivers(year, gp, session_type, a, b)   -> results dict
    analyze_corners(results)                        -> per-corner DataFrame
    plot_clash(results)                             -> matplotlib Figure

Sign convention throughout: positive deltas mean driver_b is faster / ahead.
Colour convention: RED = driver_a faster, BLUE = driver_b faster.
"""

import fastf1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

COLOUR_A = "firebrick"
COLOUR_B = "steelblue"
# speed_delta = B - A. "RdBu" runs red (low) → blue (high), so negative
# (A faster) is red and positive (B faster) is blue — matching the title.
# The previous "RdBu_r" was reversed and painted B's gains red.
DELTA_CMAP = "RdBu"


def load_session(year, gp, session_type):
    """Load a session with what Tool 5 needs (laps + telemetry, no weather)."""
    session = fastf1.get_session(year, gp, session_type)
    session.load(laps=True, telemetry=True, weather=False)
    return session


def lap_fraction(tel):
    """
    Distance covered as a fraction of THIS lap, guaranteed to end at 1.0.

    FastF1's RelativeDistance often stops short of 1.0 (e.g. 0.9948) even
    though the last sample's Time equals the official LapTime. Resampling onto
    a 0→1 grid then *extrapolates* the final metres, and because each driver
    stops short by a different amount, that extrapolation invented ~160 ms of
    fake gap at Silverstone 2024. Normalising by the lap's own final distance
    pins the last grid point to the real finish line.
    """
    d = tel["Distance"].values.astype(float)
    d = d - d[0]
    return d / d[-1]


def resample_lap(tel, n_points=300):
    """Resample telemetry onto an n_points lap-fraction grid (incl. GPS X/Y)."""
    grid = np.linspace(0, 1, n_points)
    dist = lap_fraction(tel)
    out = {}
    for ch in ["Speed", "Throttle", "Brake", "X", "Y"]:
        f = interp1d(dist, tel[ch].astype(float).values,
                     bounds_error=False, fill_value="extrapolate")
        out[ch] = f(grid)
    return pd.DataFrame(out, index=grid)


def detect_corners(grid_a, grid_b, sigma=2, prominence_frac=0.1, min_distance=10):
    """
    Detect corners from the averaged GPS path via curvature peak detection.

    Curvature κ = |x'y'' - y'x''| / (x'² + y'²)^{3/2}. Straights have κ≈0,
    hairpins have large κ, so corners are the peaks of κ along the lap.
    """
    ref_X = (grid_a["X"].values + grid_b["X"].values) / 2
    ref_Y = (grid_a["Y"].values + grid_b["Y"].values) / 2
    ref_X_smooth = gaussian_filter1d(ref_X, sigma=sigma)
    ref_Y_smooth = gaussian_filter1d(ref_Y, sigma=sigma)

    dx, dy = np.gradient(ref_X_smooth), np.gradient(ref_Y_smooth)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = np.sqrt((dx**2 + dy**2) ** 3) + 1e-9
    curvature = gaussian_filter1d(numerator / denominator, sigma=sigma)

    peaks, _ = find_peaks(curvature,
                          prominence=curvature.max() * prominence_frac,
                          distance=min_distance)
    return {"peaks": peaks, "ref_X_smooth": ref_X_smooth,
            "ref_Y_smooth": ref_Y_smooth, "curvature": curvature}


def compute_deltas(grid_a, grid_b, tel_a, tel_b, n_points=300):
    """Speed delta + exact cumulative time delta (via FastF1 Time channel)."""
    grid = np.linspace(0, 1, n_points)
    speed_delta = grid_b["Speed"].values - grid_a["Speed"].values

    def resample_time(tel):
        t = (tel["Time"] - tel["Time"].iloc[0]).dt.total_seconds().values
        f = interp1d(lap_fraction(tel), t, bounds_error=False, fill_value="extrapolate")
        return f(grid)

    # A took longer to reach this point → B is ahead → positive
    cumulative_dt = resample_time(tel_a) - resample_time(tel_b)
    return {"speed_delta": speed_delta, "cumulative_dt": cumulative_dt,
            "total_dt_ms": cumulative_dt[-1] * 1000}


def compare_drivers(year, gp, session_type, driver_a, driver_b,
                    n_points=300, session=None):
    """Full Tool 5 pipeline. Pass `session` to reuse an already-loaded one."""
    if session is None:
        session = load_session(year, gp, session_type)

    lap_a = session.laps.pick_drivers(driver_a).pick_fastest()
    lap_b = session.laps.pick_drivers(driver_b).pick_fastest()
    for code, lap in ((driver_a, lap_a), (driver_b, lap_b)):
        if lap is None or (hasattr(lap, "empty") and lap.empty):
            raise ValueError(f"No timed lap found for {code} in this session.")

    tel_a = lap_a.get_telemetry()
    tel_b = lap_b.get_telemetry()

    grid_a = resample_lap(tel_a, n_points)
    grid_b = resample_lap(tel_b, n_points)
    corners = detect_corners(grid_a, grid_b)
    deltas = compute_deltas(grid_a, grid_b, tel_a, tel_b, n_points)
    actual_dt_ms = (lap_a["LapTime"] - lap_b["LapTime"]).total_seconds() * 1000

    return {
        "meta": {"year": year, "gp": gp, "session": session_type,
                 "driver_a": driver_a, "driver_b": driver_b,
                 "event_name": session.event["EventName"],
                 "lap_time_a": lap_a["LapTime"].total_seconds(),
                 "lap_time_b": lap_b["LapTime"].total_seconds(),
                 "compound_a": lap_a["Compound"], "compound_b": lap_b["Compound"],
                 "lap_length_m": (tel_a["Distance"].max() + tel_b["Distance"].max()) / 2},
        "grids": {"a": grid_a, "b": grid_b},
        "corners": corners,
        "deltas": deltas,
        "validation": {"predicted_dt_ms": deltas["total_dt_ms"],
                       "actual_dt_ms": actual_dt_ms,
                       "error_ms": deltas["total_dt_ms"] - actual_dt_ms},
    }


def analyze_corners(results, max_half_width=12, brake_thresh=0.5, throttle_thresh=90):
    """
    Per-corner diagnostics: apex speed, brake point, throttle-on point and net
    time gained by driver B across each corner zone. Positive values favour B.
    """
    meta = results["meta"]
    peaks = results["corners"]["peaks"]
    grid_a, grid_b = results["grids"]["a"], results["grids"]["b"]
    cum_dt = results["deltas"]["cumulative_dt"]
    a, b = meta["driver_a"], meta["driver_b"]

    n_points = len(cum_dt)
    mpp = meta["lap_length_m"] / (n_points - 1)  # metres per grid point

    spd_a, spd_b = grid_a["Speed"].values, grid_b["Speed"].values
    brk_a, brk_b = grid_a["Brake"].values, grid_b["Brake"].values
    thr_a, thr_b = grid_a["Throttle"].values, grid_b["Throttle"].values

    def brake_point_m(brake, left, apex):
        window = np.where(brake[left:apex] > brake_thresh)[0]
        return (apex - (left + window[0])) * mpp if len(window) else np.nan

    def throttle_point_m(throttle, apex, right):
        window = np.where(throttle[apex:right] > throttle_thresh)[0]
        return window[0] * mpp if len(window) else np.nan

    def diff(x, y):
        return np.nan if (np.isnan(x) or np.isnan(y)) else round(x - y, 1)

    rows = []
    for i, p in enumerate(peaks):
        prev_gap = p - peaks[i - 1] if i > 0 else max_half_width * 2
        next_gap = peaks[i + 1] - p if i < len(peaks) - 1 else max_half_width * 2
        left = max(0, p - min(max_half_width, prev_gap // 2))
        right = min(n_points - 1, p + min(max_half_width, next_gap // 2))

        apex_a = left + int(np.argmin(spd_a[left:right]))
        apex_b = left + int(np.argmin(spd_b[left:right]))
        bp_a, bp_b = brake_point_m(brk_a, left, apex_a), brake_point_m(brk_b, left, apex_b)
        tp_a, tp_b = throttle_point_m(thr_a, apex_a, right), throttle_point_m(thr_b, apex_b, right)

        rows.append({
            "Corner": f"C{i + 1}",
            "Lap%": round(p / n_points * 100, 1),
            f"Apex {a}": round(spd_a[apex_a], 1),
            f"Apex {b}": round(spd_b[apex_b], 1),
            "ΔApex (km/h)": round(spd_b[apex_b] - spd_a[apex_a], 1),
            "ΔBrake (m)": diff(bp_a, bp_b),       # >0: B brakes later
            "ΔThrottle (m)": diff(tp_a, tp_b),    # >0: B back on power earlier
            "Time gain (ms)": round((cum_dt[right] - cum_dt[left]) * 1000, 1),
        })
    return pd.DataFrame(rows)


def plot_clash(results, figsize=(16, 8)):
    """Render the 3-panel Tool 5 figure from a compare_drivers() results dict."""
    meta, corners, deltas, val = (results["meta"], results["corners"],
                                  results["deltas"], results["validation"])
    a, b = meta["driver_a"], meta["driver_b"]
    peaks = corners["peaks"]
    ref_X_smooth, ref_Y_smooth = corners["ref_X_smooth"], corners["ref_Y_smooth"]
    speed_delta, cumulative_dt = deltas["speed_delta"], deltas["cumulative_dt"]
    n_points = len(speed_delta)
    x_axis = np.linspace(0, 100, n_points)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1.2, 1])

    ax_map = fig.add_subplot(gs[:, 0])
    points = np.array([ref_X_smooth, ref_Y_smooth]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    max_abs = np.abs(speed_delta).max()
    lc = LineCollection(segments, cmap=DELTA_CMAP,
                        norm=plt.Normalize(-max_abs, max_abs), linewidth=5)
    lc.set_array(speed_delta[:-1])
    ax_map.add_collection(lc)
    ax_map.set_xlim(ref_X_smooth.min() - 200, ref_X_smooth.max() + 200)
    ax_map.set_ylim(ref_Y_smooth.min() - 200, ref_Y_smooth.max() + 200)
    ax_map.scatter(ref_X_smooth[peaks], ref_Y_smooth[peaks], c="black", s=80, zorder=3)
    for i, p in enumerate(peaks):
        ax_map.annotate(f"C{i + 1}", (ref_X_smooth[p], ref_Y_smooth[p]),
                        xytext=(8, 8), textcoords="offset points",
                        fontsize=10, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.85))
    ax_map.set_aspect("equal")
    ax_map.axis("off")
    ax_map.set_title(f"{meta['event_name']} {meta['year']} {meta['session']} - {a} vs {b}\n"
                     f"RED = {a} faster   |   BLUE = {b} faster", fontsize=12)
    plt.colorbar(lc, ax=ax_map, shrink=0.6, label=f"Speed delta (km/h) - {b} minus {a}")

    ax_time = fig.add_subplot(gs[0, 1])
    ax_time.plot(x_axis, cumulative_dt * 1000, color="dimgray", linewidth=2)
    ax_time.axhline(0, color="black", linewidth=0.8)
    ax_time.fill_between(x_axis, 0, cumulative_dt * 1000, where=(cumulative_dt > 0),
                         color=COLOUR_B, alpha=0.25, label=f"{b} ahead")
    ax_time.fill_between(x_axis, 0, cumulative_dt * 1000, where=(cumulative_dt < 0),
                         color=COLOUR_A, alpha=0.25, label=f"{a} ahead")
    for p in peaks:
        ax_time.axvline(p / n_points * 100, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax_time.set_xlabel("Lap distance (%)")
    ax_time.set_ylabel(f"Cumulative time delta (ms)\n{b} ahead = positive")
    ax_time.set_title(f"Running gap - final {val['predicted_dt_ms']:+.0f} ms "
                      f"(actual {val['actual_dt_ms']:+.0f}, err {val['error_ms']:+.1f})", fontsize=10)
    ax_time.legend(loc="best", fontsize=9)
    ax_time.grid(alpha=0.3)

    ax_speed = fig.add_subplot(gs[1, 1])
    ax_speed.plot(x_axis, speed_delta, color="dimgray", linewidth=1.2)
    ax_speed.axhline(0, color="black", linewidth=0.8)
    ax_speed.fill_between(x_axis, 0, speed_delta, where=(speed_delta > 0), color=COLOUR_B, alpha=0.25)
    ax_speed.fill_between(x_axis, 0, speed_delta, where=(speed_delta < 0), color=COLOUR_A, alpha=0.25)
    for p in peaks:
        ax_speed.axvline(p / n_points * 100, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax_speed.set_xlabel("Lap distance (%)")
    ax_speed.set_ylabel("Speed delta (km/h)")
    ax_speed.set_title("Instantaneous speed difference")
    ax_speed.grid(alpha=0.3)

    plt.tight_layout()
    return fig
