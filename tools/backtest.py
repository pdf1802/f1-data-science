"""
Backtest the race engine against what really happened.

    python -m tools.backtest --laps data/laps_2022_2025.parquet

For every dry race: take the REAL state at 40 % distance (order, gaps, tyres),
give every driver the stops they REALLY made afterwards, estimate pace only from
the laps run so far, simulate to the flag many times and compare with the real
classification. Strategy is given — what is tested is the engine (pace,
degradation, traffic, overtakes, Safety Cars), i.e. what the Race Engineer
relies on to compare options.

Baseline to beat: "the order at 40 % distance is the final order".
Metrics (only drivers who were classified at the flag):
    Spearman ρ   rank correlation, expected simulated position vs real position
    MAE          mean |expected position − real position|
    log score    mean log P(real position) under the simulated distribution
                 (+1 smoothing) — rewards being right AND honest about uncertainty
    80 % cover   share of drivers whose real position falls inside the simulated
                 10–90 % range (≈ 80 % = calibrated, much less = overconfident)

It also calibrates RACE_FORM_STD (how much a driver's pace can differ from what
was seen so far): the value whose 80 % intervals really contain ~80 % of results
(the log score is reported too; it is flat between 0.25 and 0.4).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from tools import race_engine as eng
from tools.shared import REPO_ROOT


def _ranks_among(pos: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """(S, D) positions → ranks among the `keep` columns only (S, K)."""
    sub = pos[:, keep]
    return np.argsort(np.argsort(sub, 1), 1) + 1


def backtest_race(laps, season, race, frac=0.4, n_sims=300, form_std=None, seed=0) -> dict | None:
    L = int(laps[(laps["Season"] == season) & (laps["Race"] == race)]["TotalLaps"].iloc[0])
    at = max(8, int(round(frac * L)))
    cfg = {"dnf_prob_per_lap": 0.0}
    if form_std is not None:
        cfg["race_form_std"] = form_std
    try:
        setup, cfg, state, info = eng.state_from_real_race(laps, season, race, at, cfg)
    except Exception:
        return None
    actual = eng.actual_result(laps, season, race)
    drivers = setup["drivers"]
    keep = np.array([d in actual.index for d in drivers])
    if keep.sum() < 5:
        return None
    st = eng.broadcast_state(state, n_sims)
    dr = eng.draw(n_sims, setup["L"] - at, setup["D"], cfg, seed)
    pos = eng.run_to_flag(st, setup, cfg, dr)["positions"]
    ranks = _ranks_among(pos, keep)                               # (S, K)
    kd = [d for d, k in zip(drivers, keep) if k]
    real = actual.loc[kd].rank().to_numpy()                       # real order among the same drivers
    now = np.arange(1, len(kd) + 1)                               # order at lap N (drivers sorted by position)
    exp = ranks.mean(0)
    K = len(kd)
    probs = np.array([((ranks[:, j] == real[j]).sum() + 1) / (n_sims + K) for j in range(K)])
    lo, hi = np.percentile(ranks, 10, axis=0), np.percentile(ranks, 90, axis=0)
    rho = lambda a, b: float(pd.Series(a).corr(pd.Series(b), method="spearman"))
    return {"season": season, "race": race, "at_lap": at, "drivers": K,
            "rho_sim": rho(exp, real), "rho_now": rho(now, real),
            "mae_sim": float(np.abs(exp - real).mean()), "mae_now": float(np.abs(now - real).mean()),
            "log_score": float(np.log(probs).mean()),
            "cover80": float(((real >= lo) & (real <= hi)).mean())}


def run(laps: pd.DataFrame, form_stds=(0.0, 0.12, 0.2, 0.25, 0.3, 0.4), n_sims=300, frac=0.4) -> tuple[pd.DataFrame, pd.DataFrame]:
    from tools.training import dry_races
    dry, _ = dry_races(laps)
    races = dry.groupby(["Season", "Race"]).size().index.tolist()
    rows = []
    for fs in form_stds:
        for s, r in races:
            res = backtest_race(dry, s, r, frac=frac, n_sims=n_sims, form_std=fs)
            if res:
                rows.append({**res, "form_std": fs})
    df = pd.DataFrame(rows)
    summary = df.groupby("form_std")[["rho_sim", "rho_now", "mae_sim", "mae_now", "log_score", "cover80"]].mean()
    summary["races"] = df.groupby("form_std").size()
    summary["sim_beats_baseline_rho"] = df.assign(w=df.rho_sim > df.rho_now).groupby("form_std")["w"].mean()
    return df, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--laps", default=os.path.join(REPO_ROOT, "data", "laps_2022_2025.parquet"))
    ap.add_argument("--sims", type=int, default=300)
    ap.add_argument("--frac", type=float, default=0.4, help="fraction of the race already run")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "models", "backtest_report.json"))
    args = ap.parse_args(argv)
    df, summary = run(pd.read_parquet(args.laps), n_sims=args.sims, frac=args.frac)
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string())
    by_log = float(summary["log_score"].idxmax())
    best = float((summary["cover80"] - 0.8).abs().idxmin())
    print(f"\nRACE_FORM_STD — best log score: {by_log}; 80 % interval closest to 80 % coverage: {best}"
          f" (used: {eng.RACE_FORM_STD})")
    print(df[df.form_std == best][["season", "race", "at_lap", "drivers", "rho_sim", "rho_now", "mae_sim",
                                    "mae_now", "cover80"]].round(2).to_string(index=False))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary.round(4).reset_index().to_dict(orient="records"),
                   "best_form_std_coverage": best, "best_form_std_log_score": by_log,
                   "frac": args.frac, "sims": args.sims}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
