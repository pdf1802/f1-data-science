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

In-sample vs out-of-sample. By default the trained models (Modules 1 and 2) and the
circuit profiles have seen every race — including the second part of the race being
backtested. `--loro` (leave one race out) retrains both modules and the profiles
without each race before backtesting it: the honest version, ~3 minutes.

It also calibrates RACE_FORM_STD (how much a driver's pace can differ from what
was seen so far): the value whose 80 % intervals really contain ~80 % of results
(the log score is reported too; it prefers slightly wider intervals, 0.30).
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


def train_without(laps: pd.DataFrame, season: int, race: str, out_dir: str,
                  race_table: pd.DataFrame | None = None) -> dict:
    """
    Modules 1 + 2 (the train_models pipeline, no CV) and circuit profiles fitted on every race
    but one. Models are written to out_dir (load with F1_MODEL_DIR); the profiles are returned.
    race_table: degradation.race_physics of all the races, computed once — each row uses only
    its own race's laps, so dropping the held-out row is exactly "never saw that race".
    """
    import joblib

    from tools import training
    from tools.circuit_profiles import build_profiles
    from tools.shared import COMPOUND_MAP
    rest = laps[~((laps["Season"] == season) & (laps["Race"] == race))]
    dry, _ = training.dry_races(rest)
    circuits = training.circuit_list(dry)
    if race_table is not None:
        race_table = race_table.drop(index=f"{season} {race}", errors="ignore")
    fit = training.fit_lap_pipeline(dry, circuits, race_table=race_table)
    ovt = training.build_overtake_dataset(dry, circuits=circuits)
    ovt_model = training.make_overtake_model().fit(ovt[training.OVT_MODEL_FEATURES], ovt["Overtake"])
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(fit["model"], os.path.join(out_dir, "lap_time_model.pkl"))
    joblib.dump(ovt_model, os.path.join(out_dir, "overtaking_model.pkl"))
    joblib.dump(fit["encoder"], os.path.join(out_dir, "team_encoder.pkl"))
    for name, obj in (("compound_map.json", COMPOUND_MAP), ("circuits.json", circuits),
                      ("deg_tail.json", fit["tail"]), ("degradation.json", fit["prior"])):
        with open(os.path.join(out_dir, name), "w") as f:
            json.dump(obj, f)
    return build_profiles(rest)


def run_loro(laps: pd.DataFrame, form_std: float = eng.RACE_FORM_STD, n_sims=300, frac=0.4,
             verbose: bool = True) -> pd.DataFrame:
    """Leave-one-race-out: every race backtested with models and profiles that never saw it."""
    import tempfile

    from tools import shared
    from tools.circuit_profiles import use_profiles
    from tools import degradation
    from tools.training import dry_races
    dry, _ = dry_races(laps)
    races = dry.groupby(["Season", "Race"]).size().index.tolist()
    race_table = degradation.race_physics(dry)          # once: each race's row uses only that race
    old = os.environ.get("F1_MODEL_DIR")
    rows = []
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for k, (s, r) in enumerate(races, 1):
                out = os.path.join(tmp, f"race{k}")
                profiles = train_without(laps, s, r, out, race_table=race_table)
                os.environ["F1_MODEL_DIR"] = out
                shared.reset_models()
                with use_profiles(profiles):
                    res = backtest_race(dry, s, r, frac=frac, n_sims=n_sims, form_std=form_std)
                if res:
                    rows.append({**res, "form_std": form_std})
                if verbose:
                    print(f"  {k:2d}/{len(races)} {s} {r}: " + (f"rho {res['rho_sim']:.2f} vs {res['rho_now']:.2f}"
                                                               if res else "skipped"), flush=True)
    finally:
        if old is None:
            os.environ.pop("F1_MODEL_DIR", None)
        else:
            os.environ["F1_MODEL_DIR"] = old
        shared.reset_models()
    return pd.DataFrame(rows)


def run_stop_counts(laps: pd.DataFrame, train=None, verbose: bool = True, frac: float = 0.4) -> pd.DataFrame:
    """
    Does the tyre model get the DECISION right — one stop or two? Leave one race out: Module 1,
    the degradation prior and the circuit profile (pit loss) are fitted without the race, then
    the Stint Optimizer says which is faster on a clear track and with what probability, and that
    is compared with what the finishers really did.

    One row per race: deg_rate (the circuit's prior γ; the global mean for a circuit seen only in
    that race), gap_s = best one-stop − best two-stop (> 0: two stops are faster), p_two,
    real_two = share of finishers who stopped twice or more, real_two_green = the same counting
    only green-flag stops, sc = the race had a Safety Car / VSC stop.
    The same question is asked twice more, with better knowledge of the day (models still never
    saw the race): deg_live / gap_live_s — γ updated with the laps up to `frac` of the distance,
    as the Race Engineer does; deg_final / gap_final_s — γ updated with the whole race (hindsight:
    the ceiling for any estimate of the day).
    `train(laps, season, race, out_dir)` defaults to train_without (pass another to compare models).
    """
    import tempfile

    from tools import degradation, shared
    from tools.circuit_profiles import use_profiles
    from tools.stint_optimizer import best_strategies, real_stop_counts, stop_count_outlook
    from tools.training import dry_races
    dry, _ = dry_races(laps)
    races = dry.groupby(["Season", "Race"]).size().index.tolist()
    if train is None:
        race_table = degradation.race_physics(dry)
        train = lambda lp, s, r, out: train_without(lp, s, r, out, race_table=race_table)
    real = real_stop_counts(dry)
    old = os.environ.get("F1_MODEL_DIR")
    rows = []
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for k, (s, r) in enumerate(races, 1):
                out = os.path.join(tmp, f"race{k}")
                profiles = train(laps, s, r, out)
                os.environ["F1_MODEL_DIR"] = out
                shared.reset_models()
                L = int(dry[(dry["Season"] == s) & (dry["Race"] == r)]["TotalLaps"].iloc[0])
                with use_profiles(profiles):
                    cfg = eng.resolve_config({"circuit": r, "total_laps": L})
                team = shared.get_known_teams()[0]
                o = stop_count_outlook(L, team, cfg, cfg["pit_loss"], max_stops=2)
                best = o["expected"].set_index("Stops")["Total Δ (s)"]
                rr = real[(real["Season"] == s) & (real["Race"] == r)]
                later = {}
                this = dry[(dry["Season"] == s) & (dry["Race"] == r)]
                for name, upto in (("live", int(round(frac * L))), ("final", L)):
                    est = eng.live_degradation(this[this["LapNumber"] <= upto], r)
                    if est:
                        b = best_strategies(L, team, cfg, cfg["pit_loss"], max_stops=2,
                                            deg_rate=est["mean"]).set_index("Stops")["Total Δ (s)"]
                        later.update({f"deg_{name}": est["mean"], f"gap_{name}_s": float(b[1] - b[2])})
                rows.append({"season": s, "race": r, "deg_rate": o["deg_rate"], "pit_loss": cfg["pit_loss"],
                             "gap_s": float(best[1] - best[2]), "p_two": o["p_best"].get(2, 0.0), **later,
                             "real_two": float((rr["stops"] >= 2).mean()),
                             "real_two_green": float((rr["stops"] - rr["sc_stops"] >= 2).mean()),
                             "sc": bool(rr["sc_stops"].sum() > 0), "finishers": len(rr)})
                if verbose:
                    x = rows[-1]
                    print(f"  {k:2d}/{len(races)} {s} {r}: two-stop {x['gap_s']:+.1f} s, P(two) {x['p_two']:.0%} · "
                          f"real {x['real_two']:.0%} of {x['finishers']} finishers", flush=True)
    finally:
        if old is None:
            os.environ.pop("F1_MODEL_DIR", None)
        else:
            os.environ["F1_MODEL_DIR"] = old
        shared.reset_models()
    return pd.DataFrame(rows)


# Races where the number of stops was set by the rules, not by the tyres: no test of a tyre model.
MANDATED_STOPS = {(2025, "Monaco Grand Prix")}       # two stops were compulsory at Monaco in 2025


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    """P(a random two-stop race has a larger score than a random one-stop race); ties count half."""
    pos, neg = score[y], score[~y]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def summarise_stop_counts(df: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    """
    How good is the optimizer's one-vs-two-stop answer, at each level of knowledge of the day
    (before the race / at `frac` of the distance / hindsight)? Races in MANDATED_STOPS are left out.
    Per level: agreement with the field's majority at "which is faster", the same for the trivial
    "always one stop", the AUC of the gap with a bootstrap 95 % interval (36 races are few: the
    intervals are wide and overlap), and the same without Bahrain — the circuit the fix was aimed at.
    The pre-race probability is scored with the Brier score against the base rate.
    """
    df = df[~df.apply(lambda r: (int(r["season"]), r["race"]) in MANDATED_STOPS, axis=1)].reset_index(drop=True)
    two = (df["real_two"] >= 0.5).to_numpy()
    rng = np.random.default_rng(seed)
    out = {"races": int(len(df)), "real_two_stop_races": int(two.sum()),
           "always_one_stop_agree": float((~two).mean()),
           "brier_p_two": float(((df["p_two"] - two) ** 2).mean()),
           "brier_base_rate": float(((two.mean() - two) ** 2).mean())}
    not_bahrain = (df["race"] != "Bahrain Grand Prix").to_numpy()
    for name, col in (("before", "gap_s"), ("live", "gap_live_s"), ("hindsight", "gap_final_s")):
        if col not in df:
            continue
        gap = df[col].to_numpy()
        said = gap > 0
        boots = []
        for _ in range(n_boot):
            i = rng.integers(0, len(df), len(df))
            boots.append(_auc(two[i], gap[i]))
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        out[name] = {"said_two_stop": int(said.sum()), "agree": float((two == said).mean()),
                     "two_stop_races_found": float((said & two).sum() / max(two.sum(), 1)),
                     "one_stop_races_found": float((~said & ~two).sum() / max((~two).sum(), 1)),
                     "auc": _auc(two, gap), "auc_ci95": [float(lo), float(hi)],
                     "auc_without_bahrain": _auc(two[not_bahrain], gap[not_bahrain]),
                     "two_stop_found_without_bahrain": float((said & two & not_bahrain).sum()
                                                             / max((two & not_bahrain).sum(), 1)),
                     "spearman_gap_vs_share": float(df[col].corr(df["real_two"], method="spearman"))}
    return out


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby("form_std")[["rho_sim", "rho_now", "mae_sim", "mae_now", "log_score", "cover80"]].mean()
    summary["races"] = df.groupby("form_std").size()
    summary["sim_beats_baseline_rho"] = df.assign(w=df.rho_sim > df.rho_now).groupby("form_std")["w"].mean()
    return summary


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
    return df, summarise(df)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--laps", default=os.path.join(REPO_ROOT, "data", "laps_2022_2025.parquet"))
    ap.add_argument("--sims", type=int, default=300)
    ap.add_argument("--frac", type=float, default=0.4, help="fraction of the race already run")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "models", "backtest_report.json"))
    ap.add_argument("--loro", action="store_true",
                    help="leave one race out: retrain models + profiles without each race (out-of-sample)")
    ap.add_argument("--stops", action="store_true",
                    help="one stop or two? the Stint Optimizer's answer vs what the field did, leave one race out")
    args = ap.parse_args(argv)
    if args.stops:
        print("One stop or two — leave one race out…")
        df = run_stop_counts(pd.read_parquet(args.laps))
        summary = summarise_stop_counts(df)
        print(json.dumps(summary, indent=2))
        out = args.out.replace(".json", "_stops.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump({"summary": summary, "races": df.round(4).to_dict(orient="records")}, f, indent=2)
        return 0
    if args.loro:
        print(f"Leave-one-race-out at RACE_FORM_STD = {eng.RACE_FORM_STD}: retraining for every race…")
        df = run_loro(pd.read_parquet(args.laps), n_sims=args.sims, frac=args.frac)
        summary = summarise(df)
        pd.set_option("display.width", 200)
        print(summary.round(3).to_string())
        out = args.out.replace(".json", "_loro.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as f:
            json.dump({"summary": summary.round(4).reset_index().to_dict(orient="records"),
                       "races": df.round(4).to_dict(orient="records"), "frac": args.frac, "sims": args.sims,
                       "leave_one_race_out": True}, f, indent=2)
        return 0
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
