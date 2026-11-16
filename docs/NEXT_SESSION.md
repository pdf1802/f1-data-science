# Next session — where things stand (16 Nov 2026)

Read this first, then `README.md` (Results, Known limitations) and `notebooks/degradation_by_circuit.ipynb`.
Branch `dashboard-v1`. Push: Claude has no write access to the GitHub repo, so commits go to Paco's PC
(`D:\Proyectos\f1-data-science`, CRLF, `core.autocrlf=true`) via a bundle + `git merge --ff-only`, and he pushes.
Data: `data/laps_2022_2025.parquet` and `models/` (not in git, on his PC; a copy for the demo is in `demo/`);
FastF1 cache `cache_drive/f1_cache`, **offline only** (401 since Oct 2026). Paco wants to understand everything:
explain each step, document in Notion (page "F1 project", sections 29+).

## Done on 15–16 Nov (the plan of 11 Nov)

1. **Per-race degradation (Module 1 V3.2)** — `tools/degradation.py`, rate as a model input, uncertain in the
   engine, updated live; n-stop Stint Optimizer; Race Engineer plans from `optimal_plan`. Validated leave one
   race out on 35 races (Miami 2025 excluded: placeholder tyre data): engine unchanged (ρ 0.893 vs 0.892 with
   V3.1); Bahrain two-stop 4/4 before the race; one-vs-two AUC 0.59 → 0.74 before / 0.84 live / 0.86 hindsight,
   but outside Bahrain the pre-race call finds no two-stop race and its probability has no skill (Brier).
   An independent review found five real problems (data, fuel shrinkage, a STAY OUT on a box lap, prior
   uncertainty, overclaims); all fixed and re-run — see README "Fixed along the way".
2. **Demo bundle** (`demo/`, `tools/demo_data.py`) and `docs/DEPLOY.md` — the deployment itself needs Paco's account.
3. **Polish** — dashboard redesign (`ui/theme.py`), GitHub Actions (ruff + pytest), `pyproject.toml`,
   `environment.yml`, GIF at the top of the README.

## Open, in order of value

1. **The remaining one-stop lean** (several seconds; 3 of 11 two-stop races found outside Bahrain even in
   hindsight, see README). Ideas, cheapest first:
   - *Tyre management:* stints that end up long are driven slower from lap one. Test a feature "planned stint
     length" (known to a planner) — but watch survivorship: stints get extended when the tyres hold up.
   - *Cliff risk:* model the variance of late-stint laps, not just the mean; a risk-averse planner stops earlier.
   - *Track position:* the optimizer is clear-track; let the engine (with traffic) score the top plans of each
     stop count instead of the deterministic sum. Monaco is the test.
2. **Rebuild the demo bundle on Paco's PC** (`python -m tools.demo_data`): his cache has more qualifying sessions
   than the 22 exported here. Then deploy (docs/DEPLOY.md) and put the link in the README and CV.
3. **Fresh SOFT pace** is not learned where the SOFT is hardly raced (plain average: level with the HARD). A
   physics prior on the compound offsets (the fixed-effects reference as a base margin) is the natural fix.
4. **Qualy → Race:** a race-craft prior per team from the *previous* races of the season (out of sample).
5. **Undercut:** model the out-lap warm-up and the spread of stop times (the projection's weak part, 2.1 s).
6. The Race Engineer window (PySide6) still has its old colours; the dashboard's palette is in `ui/theme.py`.

## Pending outside the code

- Paco posts the two issues in IAmTomShaw/f1-race-replay (`integrations/f1_race_replay/UPSTREAM_PROPOSAL.md`):
  Issue A (bug: offline crash in `get_circuit_rotation`) first, then Issue B (`session_info` in the payload).
- Paco deploys the demo (needs his Streamlit / GitHub login).
