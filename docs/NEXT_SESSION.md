# Next session — plan (agreed 11 Nov 2026)

Read this first, then `README.md` (Results, Known limitations) and `integrations/f1_race_replay/README.md`.
Branch `dashboard-v1` (= `origin/dashboard-v1`); merge to `main` only after asking Paco. Push: Claude has no
write access to the GitHub repo, so commits go to Paco's PC (`D:\Proyectos\f1-data-science`, CRLF,
`core.autocrlf=true`) via a bundle + `git merge --ff-only`, and he pushes. Data: `data/laps_2022_2025.parquet`
and `models/` (not in git, on his PC); FastF1 cache `cache_drive/f1_cache`, **offline only** (401 since Oct 2026).
Paco wants to understand everything: explain each step, document in Notion (page "F1 project", sections 24+).

## 1. Per-circuit degradation — the main work (use the strongest model for this)

**Why.** The one-stop bias is behind most findings of the last session: "NO MORE STOPS" in Bahrain, the
Stint Optimizer (nobody one-stopped in Bahrain in four seasons, the optimizer says one stop), What-If
(the one-stop "wins" +0.9 places for ALO). Module 1 has `CircuitEncoded`, but the trees compress the
degradation differences between circuits.

**Steps.**
1. *Measure first.* From the Parquet: fuel-corrected, clean-air, green-flag stint laps → degradation slope
   (s/lap) per compound × circuit, with bootstrap CIs. Compare with the slopes Module 1 implies
   (`shared.predict_lap_deltas` over tyre age at each circuit). Quantify the compression.
2. *Choose a fix with Paco* (explain the trade-offs): (a) a per-circuit degradation multiplier estimated
   from the data, shrunk to the global mean when a circuit has few stints (hierarchical / empirical Bayes),
   applied on top of Module 1; (b) a tyre-age × circuit interaction feature (e.g. a per-circuit abrasiveness
   score × TyreLife) with monotone constraints, retrained; (c) both. Leave-one-race-out to check it doesn't
   memorise races (the V3 lesson).
3. *Validate on decisions, not just MAE:* share of real finishers by number of stops per circuit vs what the
   engine/optimizer recommend; Bahrain should come out as a two-stop race; re-run `stint_optimizer`,
   `what_if_simulator`, `race_simulator` notebooks and the backtest (`python -m tools.backtest`) — nothing may
   get worse; physics gates in `train_models` must still pass.
4. Update the Race Engineer limitations, README results, notebooks' conclusions.

## 2. Public demo

Streamlit Community Cloud (free): the dashboard live, link at the top of the README and on Paco's CV.
- Parquet (~3 MB) + models (~1 MB) as a GitHub **release asset**, downloaded on start (or committed under
  `demo/` if small enough — decide with Paco).
- The tabs that use FastF1 (qualifying, telemetry) can't download (401): export the qualifying results and the
  two telemetry laps needed for a handful of races to small files, and use those when there is no cache.
- `requirements.txt` that installs on the cloud (no PySide6 there).

## 3. Professional polish

- GitHub Actions: `pytest` on every push with the stand-in models (`tests/conftest.py` already supports
  running without data) + badge in the README.
- `pyproject.toml` (project metadata, ruff config), `ruff check` clean, `environment.yml` for the `f1` env.
- A 30-second GIF at the top of the README: take control + track map (record from the offscreen renders or
  a screen capture on Paco's PC).

## 4. If time is left

- Qualy → Race: a race-craft prior per team from the *previous* races of the season (out of sample).
- Undercut: model the out-lap warm-up and the spread of stop times (the projection's weak part, 2.2 s).

## Pending outside the code

- Paco posts the two issues in IAmTomShaw/f1-race-replay (`integrations/f1_race_replay/UPSTREAM_PROPOSAL.md`):
  Issue A (bug: offline crash in `get_circuit_rotation`) first, then Issue B (`session_info` in the payload).
