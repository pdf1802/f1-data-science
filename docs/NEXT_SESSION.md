# Next session — where things stand (16 Nov 2026)

Read this first, then `README.md` (Results, Known limitations) and `notebooks/degradation_by_circuit.ipynb`.
Branch `dashboard-v1`. Push: Claude has no write access to the GitHub repo, so commits go to Paco's PC
(`D:\Proyectos\f1-data-science`, CRLF, `core.autocrlf=true`) via a bundle + `git merge --ff-only`, and he pushes.
Data: `data/laps_2022_2025.parquet` and `models/` (not in git, on his PC; a copy for the demo is in `demo/`);
FastF1 cache `cache_drive/f1_cache`, **offline only** (401 since Oct 2026). Paco wants to understand everything:
explain each step, document in Notion (page "F1 project", sections 28+).

## Done on 15–16 Nov (the plan of 11 Nov)

1. **Per-race degradation (Module 1 V3.2)** — `tools/degradation.py`, rate as a model input, uncertain in the
   engine, updated live; n-stop Stint Optimizer; Race Engineer plans from `optimal_plan`. Validated leave one
   race out (engine not worse; Bahrain two-stop 4/4; AUC of one-vs-two 0.63 → 0.69 / 0.78 live / 0.84 hindsight).
2. **Demo bundle** (`demo/`, `tools/demo_data.py`) and `docs/DEPLOY.md` — the deployment itself needs Paco's account.
3. **Polish** — dashboard redesign (`ui/theme.py`), GitHub Actions (ruff + pytest), `pyproject.toml`,
   `environment.yml`, GIF at the top of the README.

## Open, in order of value

1. **The remaining one-stop lean** (several seconds, see README). Ideas, cheapest first:
   - *Tyre management:* stints that end up long are driven slower from lap one. Test a feature "planned stint
     length" (known to a planner) — but watch survivorship: stints get extended when the tyres hold up.
   - *Cliff risk:* model the variance of late-stint laps, not just the mean; a risk-averse planner stops earlier.
   - *Track position:* the optimizer is clear-track; let the engine (with traffic) score the top plans of each
     stop count instead of the deterministic sum. Monaco is the test.
2. **Rebuild the demo bundle on Paco's PC** (`python -m tools.demo_data`): his cache has more qualifying sessions
   than the 22 exported here. Then deploy (docs/DEPLOY.md) and put the link in the README and CV.
3. **Qualy → Race:** a race-craft prior per team from the *previous* races of the season (out of sample).
4. **Undercut:** model the out-lap warm-up and the spread of stop times (the projection's weak part, 2.1 s).
5. The Race Engineer window (PySide6) still has its old colours; the dashboard's palette is in `ui/theme.py`.

## Pending outside the code

- Paco posts the two issues in IAmTomShaw/f1-race-replay (`integrations/f1_race_replay/UPSTREAM_PROPOSAL.md`):
  Issue A (bug: offline crash in `get_circuit_rotation`) first, then Issue B (`session_info` in the payload).
- Paco deploys the demo (needs his Streamlit / GitHub login).
