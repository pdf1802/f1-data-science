"""
Tool 5 — Telemetry Clash.

Compares two drivers' fastest laps on a shared distance grid, detects corners
from GPS curvature, and computes speed + cumulative-time deltas.

Public API:
    compare_drivers(year, gp, session_type, driver_a, driver_b) -> results dict
    plot_clash(results) -> matplotlib Figure

Sign convention throughout: positive deltas mean driver_b is faster / ahead.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fastf1
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from matplotlib.collections import LineCollection

def resample_lap(tel,n_points= 300):
    """Resample telemetry onto an n_points relative-distance grid (incl. GPS X/Y)."""
    grid = np.linspace(0,1,n_points)
    channels = ["Speed","Throttle","Brake", "X", "Y"]
    dist = tel["RelativeDistance"].values
    out = {}
    for ch in channels:
        values = tel[ch].astype(float).values
        f = interp1d(dist, values, bounds_error=False, fill_value="extrapolate")
        out[ch] = f(grid)
    return pd.DataFrame(out, index=grid)


def detect_corners(grid_a, grid_b, sigma=2, prominence_frac=0.1, min_distance=10):
    """Detect corners from the averaged GPS path via curvature peak detection."""
    ref_X = (grid_a["X"].values + grid_b["X"].values) / 2
    ref_Y = (grid_a["Y"].values + grid_b["Y"].values) / 2
    ref_X_smooth = gaussian_filter1d(ref_X, sigma=sigma)
    ref_Y_smooth = gaussian_filter1d(ref_Y, sigma=sigma)

    dx, dy = np.gradient(ref_X_smooth), np.gradient(ref_Y_smooth)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = np.sqrt((dx**2 + dy**2)**3) + 1e-9
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
        f = interp1d(tel["RelativeDistance"].values, t,
                     bounds_error=False, fill_value="extrapolate")
        return f(grid)

    cumulative_dt = resample_time(tel_a) - resample_time(tel_b)
    return {"speed_delta": speed_delta, "cumulative_dt": cumulative_dt,
            "total_dt_ms": cumulative_dt[-1] * 1000}   


def compare_drivers(year, gp, session_type, driver_a, driver_b, n_points=300):
    """Full Tool 5 pipeline. Positive deltas mean driver_b is faster/ahead."""
    session = fastf1.get_session(year, gp, session_type)
    session.load()

    lap_a = session.laps.pick_driver(driver_a).pick_fastest()
    lap_b = session.laps.pick_driver(driver_b).pick_fastest()
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
                 "lap_length_m": (tel_a["Distance"].max() + tel_b["Distance"].max()) / 2},
        "laps": {"a": lap_a, "b": lap_b},
        "grids": {"a": grid_a, "b": grid_b},
        "corners": corners,
        "deltas": deltas,
        "validation": {"predicted_dt_ms": deltas["total_dt_ms"],
                       "actual_dt_ms": actual_dt_ms,
                       "error_ms": deltas["total_dt_ms"] - actual_dt_ms},
    }


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
    lc = LineCollection(segments, cmap="RdBu_r",
                        norm=plt.Normalize(-max_abs, max_abs), linewidth=5)
    lc.set_array(speed_delta[:-1])
    ax_map.add_collection(lc)
    ax_map.set_xlim(ref_X_smooth.min() - 200, ref_X_smooth.max() + 200)
    ax_map.set_ylim(ref_Y_smooth.min() - 200, ref_Y_smooth.max() + 200)
    ax_map.scatter(ref_X_smooth[peaks], ref_Y_smooth[peaks], c="black", s=80, zorder=3)
    for i, p in enumerate(peaks):
        ax_map.annotate(f"C{i+1}", (ref_X_smooth[p], ref_Y_smooth[p]),
                        xytext=(8, 8), textcoords="offset points",
                        fontsize=10, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.85))
    ax_map.set_aspect("equal"); ax_map.axis("off")
    ax_map.set_title(f"{meta['event_name']} {meta['year']} {meta['session']} - {a} vs {b}\n"
                     f"RED = {a} faster   |   BLUE = {b} faster", fontsize=12)
    plt.colorbar(lc, ax=ax_map, shrink=0.6, label=f"Speed delta (km/h) - {b} minus {a}")

    ax_time = fig.add_subplot(gs[0, 1])
    ax_time.plot(x_axis, cumulative_dt * 1000, color="steelblue", linewidth=2)
    ax_time.axhline(0, color="black", linewidth=0.8)
    ax_time.fill_between(x_axis, 0, cumulative_dt * 1000, where=(cumulative_dt > 0),
                         color="steelblue", alpha=0.2, label=f"{b} ahead")
    ax_time.fill_between(x_axis, 0, cumulative_dt * 1000, where=(cumulative_dt < 0),
                         color="firebrick", alpha=0.2, label=f"{a} ahead")
    for i, p in enumerate(peaks):
        ax_time.axvline(p / n_points * 100, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax_time.set_xlabel("Lap distance (%)")
    ax_time.set_ylabel(f"Cumulative time delta (ms)\n{b} ahead = positive")
    ax_time.set_title(f"Running gap - final {val['predicted_dt_ms']:+.0f} ms "
                      f"(actual {val['actual_dt_ms']:+.0f}, err {val['error_ms']:+.1f})", fontsize=10)
    ax_time.legend(loc="best", fontsize=9); ax_time.grid(alpha=0.3)

    ax_speed = fig.add_subplot(gs[1, 1])
    ax_speed.plot(x_axis, speed_delta, color="purple", linewidth=1.2)
    ax_speed.axhline(0, color="black", linewidth=0.8)
    ax_speed.fill_between(x_axis, 0, speed_delta, where=(speed_delta > 0), color="steelblue", alpha=0.2)
    ax_speed.fill_between(x_axis, 0, speed_delta, where=(speed_delta < 0), color="firebrick", alpha=0.2)
    for p in peaks:
        ax_speed.axvline(p / n_points * 100, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax_speed.set_xlabel("Lap distance (%)"); ax_speed.set_ylabel("Speed delta (km/h)")
    ax_speed.set_title("Instantaneous speed difference"); ax_speed.grid(alpha=0.3)

    plt.tight_layout()
    return fig