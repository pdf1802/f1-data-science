"""
Retrain Module 1 + Module 2 (V3) and write the artefacts the dashboard uses.

    python -m tools.train_models                 # full run: 32 races, ~10–30 min first time
    python -m tools.train_models --quick         # 2 races, just to check the pipeline works
    python -m tools.train_models --fuel 0.04     # sensitivity check on the fuel assumption

What it does
    1. Loads the races (FastF1 cache in <repo>/cache, or $F1_CACHE_DIR)
    2. Builds both datasets, evaluates with GroupKFold by race
    3. Trains the final models on all data
    4. Physics sanity gates — if one fails, NOTHING is overwritten (use --force to override)
    5. Backs up the current models/ to models/backup_<timestamp>/ and saves the new ones
       + models/training_report.json
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

from tools import shared, training


def _hr(title):
    print(f"\n{'═' * 70}\n{title}\n{'═' * 70}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(shared.REPO_ROOT, "models"))
    ap.add_argument("--fuel", type=float, default=shared.FUEL_S_PER_LAP, help="s per lap of fuel on board")
    ap.add_argument("--quick", action="store_true", help="2 races only — pipeline smoke test, not saved")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--force", action="store_true", help="save even if a sanity gate fails")
    ap.add_argument("--offline", action="store_true",
                    help="never hit the network — build everything from the FastF1 cache "
                         "(use with $F1_CACHE_DIR when F1's servers refuse downloads)")
    args = ap.parse_args(argv)

    import fastf1
    import logging
    logging.getLogger("fastf1").setLevel(logging.WARNING)
    cache = shared.enable_fastf1_cache()
    if args.offline:
        fastf1.Cache.offline_mode(True)
    print(f"FastF1 cache: {cache}{'  (offline mode)' if args.offline else ''}")
    t0 = time.time()

    sessions = [(2023, "Bahrain Grand Prix"), (2024, "Abu Dhabi Grand Prix")] if args.quick else training.SESSIONS
    _hr(f"1/5  Loading {len(sessions)} races")
    try:
        laps, failed = training.load_races(sessions)
    except training.NoRacesLoaded as e:
        print(f"\n❌ {e}")
        return 2
    if failed:
        print(f"\n⚠️  {len(failed)} sessions failed — training continues without them.")

    # ── Module 1 ─────────────────────────────────────────────────────────────
    _hr(f"2/5  Module 1 — lap-time model (fuel correction {args.fuel} s/lap)")
    lap_df, enc = training.build_lap_dataset(laps, fuel_s_per_lap=args.fuel)
    print(f"Clean laps: {len(lap_df):,} from {lap_df['Group'].nunique()} races")
    print(lap_df["Team"].value_counts().rename("laps").to_string())
    folds = min(args.folds, lap_df["Group"].nunique())
    ev1 = training.evaluate_lap_model(lap_df, n_splits=folds) if folds >= 2 else None
    if ev1:
        print(f"\nMAE, GroupKFold by race : {ev1['grouped_MAE']:.3f} s   ← the honest number")
        print(f"MAE, random 80/20 split : {ev1['random_split_MAE']:.3f} s   ← optimistic (same-race leakage)")
        print(f"MAE, predict the median : {ev1['predict_median_MAE']:.3f} s   ← the bar to beat")
    lap_model = training.make_lap_model().fit(lap_df[training.LAP_MODEL_FEATURES], lap_df["LapDelta"])
    s1 = training.lap_model_sanity(lap_model, np.unique(lap_df["TeamEncoded"]))
    print("\nPredicted LapDelta (s) by tyre age, averaged over teams:")
    print(s1["curve"].loc[[2, 5, 10, 15, 20, 25, 30]].round(2).to_string())
    for name, ok in s1["checks"].items():
        print(f"  {'✅' if ok else '❌'} {name}")

    # ── Module 2 ─────────────────────────────────────────────────────────────
    _hr("3/5  Module 2 — overtake model")
    ovt_df = training.build_overtake_dataset(laps)
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
    ovt_model = training.make_overtake_model().fit(ovt_df[shared.OVT_FEATURES], ovt_df["Overtake"])
    s2 = training.overtake_sanity(ovt_model)
    for k, v in s2["examples"].items():
        print(f"  P(pass) {k:<34} {v:.1%}")
    for name, ok in s2["checks"].items():
        print(f"  {'✅' if ok else '❌'} {name}")

    # ── Gate + save ──────────────────────────────────────────────────────────
    _hr("4/5  Sanity gate")
    passed = s1["passed"] and s2["passed"]
    if args.quick:
        print("--quick: pipeline OK, nothing saved (2 races is not a real model).")
        return 0
    if not passed and not args.force:
        print("❌ A physics check failed — models/ left untouched. Paste this log to investigate,\n"
              "   or re-run with --force to save anyway.")
        return 1
    print("✅ All checks passed" if passed else "⚠️  Saving despite a failed check (--force)")

    _hr("5/5  Saving")
    os.makedirs(args.out, exist_ok=True)
    existing = [f for f in shared.REQUIRED_MODEL_FILES if os.path.isfile(os.path.join(args.out, f))]
    if existing:
        bk = os.path.join(args.out, f"backup_{datetime.now():%Y%m%d_%H%M%S}")
        os.makedirs(bk)
        for f in existing + ["compound_map.json"]:
            if os.path.isfile(os.path.join(args.out, f)):
                shutil.copy2(os.path.join(args.out, f), bk)
        print(f"Old models backed up to {bk}")
    joblib.dump(lap_model, os.path.join(args.out, "lap_time_model.pkl"))
    joblib.dump(ovt_model, os.path.join(args.out, "overtaking_model.pkl"))
    joblib.dump(enc, os.path.join(args.out, "team_encoder.pkl"))
    with open(os.path.join(args.out, "compound_map.json"), "w") as f:
        json.dump(shared.COMPOUND_MAP, f)
    report = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "version": "V3", "fuel_s_per_lap": args.fuel,
        "races_used": int(lap_df["Group"].nunique()), "failed_sessions": failed,
        "module1": {"n_laps": len(lap_df), **({k: v for k, v in ev1.items() if k != "folds"} if ev1 else {}),
                    "sanity": s1["checks"]},
        "module2": {"n_rows": len(ovt_df),
                    **({k: v for k, v in ev2.items() if k != "calibration"} if ev2 else {}),
                    "sanity": s2["checks"], "examples": s2["examples"]},
        "teams": list(enc.classes_),
    }
    with open(os.path.join(args.out, "training_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved models + training_report.json to {args.out}  ({time.time() - t0:.0f} s total)")
    print("Restart the dashboard (Ctrl+C, then streamlit run app.py) to load them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
