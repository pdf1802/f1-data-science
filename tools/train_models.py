"""
Retrain Module 1 + Module 2 (V3.1) and write the artefacts the dashboard uses.

    python -m tools.train_models --laps data/laps_2022_2025.parquet   # recommended (no FastF1 needed)
    python -m tools.train_models --offline                           # every race found in the FastF1 cache
    python -m tools.train_models --quick --offline                   # 2-race smoke test, saves nothing
    python -m tools.train_models --laps ... --fuel 0.045             # sensitivity check on the fuel assumption

What it does
    1. Loads laps (Parquet from tools.export_cache, or the FastF1 cache)
    2. Keeps dry races only and fixes the circuit vocabulary
    3. Fits the transparent physics reference (fixed-effects regression)
    4. Builds both datasets, evaluates with GroupKFold by race, trains on all data
    5. Physics sanity gates — if one fails, NOTHING is overwritten (--force overrides)
    6. Backs up models/ to models/backup_<timestamp>/, saves the new artefacts
       (+ circuits.json, training_report.json)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from tools import shared, training


def _hr(title):
    print(f"\n{'═' * 72}\n{title}\n{'═' * 72}")


def _load(args):
    """→ (laps, failed sessions, source description)"""
    if args.laps:
        laps = pd.read_parquet(args.laps)
        if args.quick:
            keep = laps.groupby(["Season", "Race"]).ngroup() < 2
            laps = laps[keep]
        return laps, [], f"Parquet {args.laps}"

    import logging

    import fastf1
    logging.getLogger("fastf1").setLevel(logging.WARNING)
    cache = shared.enable_fastf1_cache()
    if args.offline:
        fastf1.Cache.offline_mode(True)
    from tools.export_cache import races_in_cache
    sessions = races_in_cache(cache) or training.SESSIONS
    if args.quick:
        sessions = [(2023, "Bahrain Grand Prix"), (2024, "Abu Dhabi Grand Prix")]
    print(f"FastF1 cache: {cache}{'  (offline mode)' if args.offline else ''} · {len(sessions)} races")
    laps, failed = training.load_races(sessions)
    return laps, failed, f"FastF1 cache {cache}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--laps", help="Parquet written by `python -m tools.export_cache` (skips FastF1)")
    ap.add_argument("--out", default=os.path.join(shared.REPO_ROOT, "models"))
    ap.add_argument("--fuel", type=float, default=shared.FUEL_S_PER_LAP, help="s per lap of fuel on board")
    ap.add_argument("--quick", action="store_true", help="2 races only — pipeline smoke test, not saved")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--force", action="store_true", help="save even if a sanity gate fails")
    ap.add_argument("--offline", action="store_true",
                    help="never hit the network — build everything from the FastF1 cache")
    args = ap.parse_args(argv)
    t0 = time.time()

    _hr("1/6  Loading laps")
    try:
        laps, failed, source = _load(args)
    except training.NoRacesLoaded as e:
        print(f"\n❌ {e}")
        return 2
    print(f"{source}: {laps.groupby(['Season', 'Race']).ngroups} races, {len(laps):,} laps")
    if failed:
        print(f"⚠️  {len(failed)} sessions could not be loaded — training continues without them.")

    _hr("2/6  Dry races + circuits")
    laps, wet = training.dry_races(laps)
    for s, r in wet:
        print(f"  ☔ excluded (wet): {s} {r}")
    circuits = training.circuit_list(laps)
    print(f"{laps.groupby(['Season', 'Race']).ngroups} dry races · {len(circuits)} circuits: {', '.join(circuits)}")

    _hr("3/6  Physics reference (fixed-effects regression, 95 % CI clustered by race)")
    physics = training.estimate_tyre_physics(laps)
    print(physics.round(4).to_string())
    print(f"({physics.attrs['laps']:,} clean-air laps, {physics.attrs['races']} races)")

    # ── Module 1 ─────────────────────────────────────────────────────────────
    _hr(f"4/6  Module 1 — lap-time model (fuel {args.fuel} s/lap, clean air > {training.CLEAN_AIR_GAP} s)")
    lap_df, enc = training.build_lap_dataset(laps, fuel_s_per_lap=args.fuel, circuits=circuits)
    print(f"Clean-air laps: {len(lap_df):,} from {lap_df['Group'].nunique()} races · features {training.LAP_MODEL_FEATURES}")
    folds = min(args.folds, lap_df["Group"].nunique())
    ev1 = training.evaluate_lap_model(lap_df, n_splits=folds) if folds >= 2 else None
    if ev1:
        gain = 1 - ev1["grouped_MAE"] / ev1["predict_median_MAE"]
        print(f"\nMAE, GroupKFold by race : {ev1['grouped_MAE']:.3f} s   ← the honest number")
        print(f"MAE, predict the median : {ev1['predict_median_MAE']:.3f} s   ← the bar to beat ({gain:+.1%})")
        print(f"MAE, random 80/20 split : {ev1['random_split_MAE']:.3f} s   ← optimistic (same-race leakage)")
    lap_model = training.make_lap_model().fit(lap_df[training.LAP_MODEL_FEATURES], lap_df["LapDelta"])
    tail = training.deg_tail(lap_df, physics)
    print(f"\nBeyond the data (age > {tail['cap']}) degradation continues linearly at "
          f"{ {k: round(v, 3) for k, v in tail['slope'].items()} } s/lap (physics reference) — trees can't extrapolate.")
    s1 = training.lap_model_sanity(lap_model, n_circuits=len(circuits), tail=tail)
    print("\nPredicted LapDelta (s) by tyre age, averaged over circuits:")
    print(s1["curve"].loc[[2, 5, 10, 15, 20, 25, 30, 35, 40]].round(2).to_string())
    for name, ok in s1["checks"].items():
        print(f"  {'✅' if ok else '❌'} {name}")
    beats_median = bool(ev1 and ev1["grouped_MAE"] < ev1["predict_median_MAE"])
    if ev1:
        print(f"  {'✅' if beats_median else '❌'} beats 'predict the median' on unseen races")

    # ── Module 2 ─────────────────────────────────────────────────────────────
    _hr("5/6  Module 2 — overtake model")
    ovt_df = training.build_overtake_dataset(laps, circuits=circuits)
    print(f"Car-behind-car situations: {len(ovt_df):,} | overtakes: {ovt_df['Overtake'].sum():,} "
          f"({ovt_df['Overtake'].mean():.1%})")
    folds2 = min(args.folds, ovt_df["Group"].nunique())
    ev2 = training.evaluate_overtake_model(ovt_df, n_splits=folds2) if folds2 >= 2 else None
    if ev2:
        print(f"\nLog loss  : {ev2['log_loss']:.4f}  (always predicting the base rate: {ev2['log_loss_constant']:.4f})")
        print(f"Brier     : {ev2['brier']:.4f}")
        print(f"ROC-AUC   : {ev2['roc_auc']:.3f}")
        print(f"PR-AUC    : {ev2['pr_auc']:.3f}  (base rate {ev2['base_rate']:.3f})")
        print("\nCalibration (out-of-fold) — mean_pred should track observed_rate:")
        print(ev2["calibration"].round(3).to_string())
    ovt_model = training.make_overtake_model().fit(ovt_df[training.OVT_MODEL_FEATURES], ovt_df["Overtake"])
    s2 = training.overtake_sanity(ovt_model, circuits)
    for k, v in s2["examples"].items():
        print(f"  P(pass) {k:<36} {v:.1%}")
    for name, ok in s2["checks"].items():
        print(f"  {'✅' if ok else '❌'} {name}")

    # ── Gate + save ──────────────────────────────────────────────────────────
    _hr("6/6  Sanity gate + save")
    passed = s1["passed"] and s2["passed"] and (beats_median or ev1 is None)
    if args.quick:
        print("--quick: pipeline OK, nothing saved (2 races is not a real model).")
        return 0
    if not passed and not args.force:
        print("❌ A check failed — models/ left untouched. Paste this log to investigate,\n"
              "   or re-run with --force to save anyway.")
        return 1
    print("✅ All checks passed" if passed else "⚠️  Saving despite a failed check (--force)")

    os.makedirs(args.out, exist_ok=True)
    existing = [f for f in shared.REQUIRED_MODEL_FILES + ("compound_map.json", "circuits.json",
                                                          "deg_tail.json", "training_report.json")
                if os.path.isfile(os.path.join(args.out, f))]
    if existing:
        bk = os.path.join(args.out, f"backup_{datetime.now():%Y%m%d_%H%M%S}")
        os.makedirs(bk)
        for f in existing:
            shutil.copy2(os.path.join(args.out, f), bk)
        print(f"Old models backed up to {bk}")
    joblib.dump(lap_model, os.path.join(args.out, "lap_time_model.pkl"))
    joblib.dump(ovt_model, os.path.join(args.out, "overtaking_model.pkl"))
    joblib.dump(enc, os.path.join(args.out, "team_encoder.pkl"))
    with open(os.path.join(args.out, "compound_map.json"), "w") as f:
        json.dump(shared.COMPOUND_MAP, f)
    with open(os.path.join(args.out, "circuits.json"), "w") as f:
        json.dump(circuits, f, indent=1)
    with open(os.path.join(args.out, "deg_tail.json"), "w") as f:
        json.dump(tail, f, indent=1)
    report = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "version": "V3.1", "source": source, "fuel_s_per_lap": args.fuel,
        "clean_air_gap_s": training.CLEAN_AIR_GAP,
        "races_used": int(lap_df["Group"].nunique()), "wet_races_excluded": wet,
        "failed_sessions": failed, "circuits": circuits,
        "physics_reference": physics.round(4).to_dict(orient="index"),
        "deg_tail": tail,
        "module1": {"features": training.LAP_MODEL_FEATURES, "n_laps": len(lap_df),
                    **({k: v for k, v in ev1.items() if k != "folds"} if ev1 else {}),
                    "sanity": s1["checks"],
                    "curve": s1["curve"].loc[[2, 10, 20, 30]].round(3).to_dict()},
        "module2": {"features": training.OVT_MODEL_FEATURES, "n_rows": len(ovt_df),
                    **({k: v for k, v in ev2.items() if k != "calibration"} if ev2 else {}),
                    "sanity": s2["checks"], "examples": s2["examples"]},
        "teams": list(enc.classes_),
    }
    with open(os.path.join(args.out, "training_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved models + circuits.json + training_report.json to {args.out}  ({time.time() - t0:.0f} s)")
    print("Restart the dashboard (Ctrl+C, then streamlit run app.py) to load them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
