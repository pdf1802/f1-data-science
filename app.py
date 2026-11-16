"""
F1 Pit Wall — Streamlit dashboard for the f1-data-science project.

    streamlit run app.py

Every tab is a thin UI over a function in tools/. No modelling logic lives
here: the same functions are imported by the notebooks and the tests.

Caching strategy
    st.cache_resource → things that must exist once per process (models)
    st.cache_data     → pure function results keyed by their inputs
                        (FastF1 sessions, simulations). Re-running with the
                        same inputs is instant; changing one input re-computes.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tools import demo_data, shared
from tools.shared import COMPOUND_COLOURS, team_colour
from ui import theme as T

st.set_page_config(page_title="F1 Pit Wall", page_icon="🏁", layout="wide", initial_sidebar_state="collapsed")
T.inject_css()

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
NEUTRAL, GOOD, BAD = T.NEUTRAL, T.GOOD, T.BAD
base_layout, show = T.style, T.show


# ── Cached wrappers ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def get_models():
    return shared.load_models()


@st.cache_data(show_spinner="Loading qualifying…", max_entries=16)
def cached_quali(year: int, gp: str) -> pd.DataFrame:
    from tools.qualy_predictor import load_qualifying_results
    return load_qualifying_results(year, gp)


@st.cache_data(show_spinner="Loading race result…", max_entries=16)
def cached_race_result(year: int, gp: str):
    from tools.qualy_predictor import load_race_result
    return load_race_result(year, gp)


@st.cache_data(show_spinner="Loading telemetry and computing deltas…", max_entries=8)
def cached_clash(year, gp, session_type, a, b):
    from tools.telemetry_clash import compare_drivers
    return compare_drivers(year, gp, session_type, a, b)


@st.cache_data(show_spinner="Simulating races…", max_entries=16)
def cached_simulate(grid: pd.DataFrame, config: dict, n_sims: int, seed: int):
    from tools.race_simulator import simulate
    r = simulate(grid, config, n_sims, seed)
    return {k: r[k] for k in ("summary", "positions", "trace_pos", "trace_gap",
                              "sc_count", "overtakes", "grid")}


@st.cache_data(show_spinner="Simulating both scenarios with the same random draws…", max_entries=8)
def cached_what_if(base: pd.DataFrame, new: pd.DataFrame, focus: str, config: dict,
                   n_sims: int, seed: int):
    from tools.race_simulator import what_if
    r = what_if(base, new, focus, config, n_sims, seed)
    del r["base"], r["new"]          # (a bare r.pop() expression would be rendered by Streamlit "magic")
    return r


LAPS_PATH = shared.laps_path()


@st.cache_data(show_spinner="Loading the lap data…")
def cached_laps() -> pd.DataFrame | None:
    if LAPS_PATH is None:
        return None
    from tools.training import dry_races
    return dry_races(pd.read_parquet(LAPS_PATH))[0]


@st.cache_resource(show_spinner=False, max_entries=8)
def cached_feed(season: int, race: str):
    """The stand-in stream of a real race (same payloads the Race Engineer window receives)."""
    from integrations.f1_race_replay.replay_stream import RaceFeed
    return RaceFeed(cached_laps(), season, race)


@st.cache_data(show_spinner="Asking the engineer…", max_entries=256)
def cached_engineer_call(season: int, race: str, driver: str, lap: int, n_sims: int):
    """The Race Engineer's call when `driver` crosses the line at the end of `lap` — no future data."""
    from tools.race_engineer import RaceEngineer
    feed = cached_feed(season, race)
    tl = feed.tl[driver]
    t = tl["line"][tl["lap"].index(lap)] + 0.2
    return RaceEngineer(driver, race, n_sims=n_sims).update(feed.payload(t))


@st.cache_data(show_spinner="Searching one, two and three stops…", max_entries=64)
def cached_outlook(total_laps: int, circuit: str | None, pit_cost: float, track_temp: float, air_temp: float):
    from tools.stint_optimizer import stop_count_outlook
    cfg = {"track_temp": track_temp, "air_temp": air_temp, "circuit": circuit}
    return stop_count_outlook(total_laps, shared.get_known_teams()[0], cfg, pit_cost)


@st.cache_data(show_spinner=False)
def cached_real_stops() -> pd.DataFrame | None:
    from tools.stint_optimizer import real_stop_counts
    laps = cached_laps()
    return None if laps is None else real_stop_counts(laps)


@st.cache_resource(show_spinner=False)
def model_info() -> dict:
    """Which model generation is loaded + the physics gate from tools/training.py."""
    from tools.training import lap_model_sanity
    lap_model, _, enc = get_models()
    feats = shared.model_features(lap_model, shared.LAP_FEATURES)
    teams = range(len(enc.classes_)) if "TeamEncoded" in feats else (0,)
    circuits = shared.get_known_circuits()
    rates = [shared.circuit_deg_rate(c) for c in circuits] if "DegRate" in feats and circuits else None
    sanity = lap_model_sanity(lap_model, teams, n_circuits=len(circuits), circuit_deg=rates,
                              deg_sd=shared.deg_race_sd())
    version = ("V2" if "LapNumber" in feats else "V3.2" if "DegRate" in feats
               else "V3.1" if "CircuitEncoded" in feats else "V3")
    return {"version": version, "sanity": sanity}


AVERAGE_CIRCUIT = "Average of all circuits"


def circuit_picker(col, key: str, default: str = "Bahrain"):
    """Circuit selector for the model-based tabs → event name, or None (= averaged)."""
    known = shared.get_known_circuits()
    if not known:                                     # V2 models: no circuit feature
        return None
    options = known + [AVERAGE_CIRCUIT]
    choice = col.selectbox("Circuit", options, options.index(default) if default in known else len(known),
                           key=key, help="The models know the circuit: tyre degradation and how hard it is "
                                         "to overtake differ a lot (Monaco vs Bahrain).")
    return None if choice == AVERAGE_CIRCUIT else choice


def models_ready() -> bool:
    try:
        get_models()
        return True
    except FileNotFoundError:
        return False


def need_models():
    st.warning(
        "This tool needs the trained models. Train them with `python -m tools.train_models` "
        "(needs the lap Parquet), or use the demo bundle in `demo/`.", icon="🧰")


def fmt_lap(seconds: float) -> str:
    """86.203 → '1:26.203' (fits in a metric tile; '86.203 s' got truncated)."""
    m, sec = divmod(float(seconds), 60)
    return f"{int(m)}:{sec:06.3f}"


def has_fastf1_cache() -> bool:
    """Your own FastF1 cache is in use (otherwise: the demo bundle)."""
    return not demo_data.forced()


def event_picker(key: str, kind: str = "qualifying", default=(2024, "Bahrain Grand Prix")):
    """
    Season + event. With a FastF1 cache: every race in the lap data (and you can type another).
    Without one: the events of the demo bundle.
    """
    laps = cached_laps()
    if has_fastf1_cache() and laps is not None:
        events = sorted({(int(a), b) for a, b in laps[["Season", "Race"]].drop_duplicates().itertuples(index=False)},
                        key=lambda e: (-e[0], e[1]))
        free = True
    else:
        events, free = demo_data.events(kind), False
    if not events:
        events, free = [default], True
    labels = [f"{y} {g}" for y, g in events]
    want = f"{default[0]} {default[1]}"
    choice = st.selectbox("Event", labels, labels.index(want) if want in labels else 0, key=f"{key}_ev",
                          accept_new_options=free,
                          help="Type 'YEAR Event name' for a session of your FastF1 cache that is not listed."
                          if free else "The public demo ships a few sessions; clone the repo to use your own cache.")
    if choice in labels:
        return events[labels.index(choice)]
    year, _, gp = str(choice).partition(" ")
    return (int(year), gp) if year.isdigit() and gp else events[0]


# ── Header ───────────────────────────────────────────────────────────────────
_ready = models_ready()
_chips = []
if _ready:
    _info = model_info()
    _ok = _info["sanity"]["passed"]
    _chips.append(("Tyre model", f"{_info['version']} · physics checks {'pass' if _ok else 'FAIL'}", "ok" if _ok else "bad"))
    _chips.append(("Race engine", "beats 'order stays the same' in 78 % of 36 races", ""))
else:
    _chips.append(("Models", "not found", "bad"))
_laps_all = cached_laps()
if _laps_all is not None:
    _chips.append(("Data", f"{_laps_all.groupby(['Season', 'Race']).ngroups} dry races, 2022–2025", ""))
_demo = _ready and os.path.abspath(shared.resolve_model_dir()).startswith(os.path.abspath(demo_data.DEMO_DIR))
_chips.append(("Mode", "demo bundle" if _demo or not has_fastf1_cache() else "your local data", "warn" if _demo else ""))
T.header("F1 Pit Wall",
         "Race strategy from real lap data: a tyre model and an overtaking model, a lap-by-lap race engine "
         "checked against 36 real races, and an engineer that makes the call.", _chips)
if _ready and not _info["sanity"]["passed"]:
    with st.expander("The tyre model fails a physics check — tyre and stint numbers are not trustworthy"):
        for _k, _v in _info["sanity"]["checks"].items():
            st.markdown(f"{'✅' if _v else '❌'} {_k}")
        st.caption("Fix: `python -m tools.train_models`.")

tabs = st.tabs(["Race engineer", "How many stops", "Undercut", "Race simulator", "What-if",
                "Qualifying → race", "Telemetry"])
TAB = {"engineer": tabs[0], "stint": tabs[1], "undercut": tabs[2], "sim": tabs[3], "whatif": tabs[4],
       "qualy": tabs[5], "clash": tabs[6]}


# ═════════════════════════════ 1. RACE SIMULATOR ════════════════════════════
with TAB["sim"]:
    st.subheader("Race simulator")
    st.caption("A thousand races from the grid: tyres that degrade as this circuit usually does (and a "
               "different amount each day), overtakes that change track position, Safety Cars that bunch "
               "the field and make stops cheap, retirements. Edit any driver's strategy in the table.")
    if not models_ready():
        need_models()
    else:
        from tools.race_simulator import (DEFAULT_CONFIG, grid_from_qualifying,
                                          position_distribution, validate_grid)
        from tools.circuit_profiles import profile_for
        year, gp = event_picker("sim")
        prof = profile_for(gp) or {}
        if prof:
            st.caption(f"Circuit profile (measured on {len(prof['races'])} dry races): {prof['laps']} laps · "
                       f"typical lap {prof['base_lap_s']:.1f} s · pit loss {prof['pit_loss_s']} s · "
                       f"Safety Car {prof['sc_per_lap'] * 100:.1f} % per lap")
        c1, c2 = st.columns([1, 3])
        total_laps = c1.number_input("Race laps", 20, 80, int(prof.get("laps", 52)), key=f"sim_l_{gp}")
        # everyone starts on the plan that is fastest here on a clear track (How many stops tab)
        _exp = cached_outlook(int(total_laps), gp, float(prof.get("pit_loss_s") or shared.PIT_COST_GREEN),
                              35.0, 25.0)["expected"]
        _exp = _exp[_exp["Stops"] <= 2].sort_values("Total Δ (s)").reset_index(drop=True)
        _labels = [f"{'One stop' if n == 1 else 'Two stops'}: {strat} ({laps_})"
                   for n, strat, laps_ in zip(_exp["Stops"], _exp["Strategy"], _exp["Pit laps"])]
        _plan = _exp.iloc[_labels.index(c2.selectbox(
            "Everyone's plan to start with", _labels, key=f"sim_plan_{gp}_{total_laps}",
            help="The fastest one- and two-stop plans for this circuit; edit any driver in the table."))]
        start_c, next_c, pit_lap = _plan["compounds"][0], _plan["compounds"][1], _plan["pit_laps"][0]

        first_visit = "grid" not in st.session_state and "grid_tried" not in st.session_state
        st.session_state["grid_tried"] = True
        if st.button("Load grid from qualifying", key="load_grid") or first_visit:
            try:
                q = cached_quali(year, gp)
                _g = grid_from_qualifying(q, total_laps, start_c, next_c, pit_lap)
                if len(_plan["pit_laps"]) > 1:
                    _g["Stop2Lap"], _g["Stop2Compound"] = _plan["pit_laps"][1], _plan["compounds"][2]
                st.session_state["grid"] = _g
                st.session_state["grid_event"] = q.attrs.get("event", f"{gp} {year}")
                st.session_state["grid_circuit"] = q.attrs.get("event_name", gp)
            except Exception as e:
                if not first_visit:
                    st.error(f"Couldn't load qualifying for {gp} {year}: {e}")

        if "grid" not in st.session_state:
            st.info("Load a qualifying session to build the starting grid.")
        else:
            st.markdown(f"**Grid — {st.session_state['grid_event']}** · PaceFactor = qualifying "
                        "time ÷ pole (1.004 ≈ +0.4 s per lap). Stop lap 0 = no stop.")
            grid = st.data_editor(
                st.session_state["grid"], key="grid_editor", hide_index=True,
                width="stretch", disabled=["Driver", "Team", "Grid"],
                column_config={
                    "PaceFactor": st.column_config.NumberColumn(format="%.4f", step=0.001),
                    "StartCompound": st.column_config.SelectboxColumn(options=COMPOUNDS),
                    "Stop1Compound": st.column_config.SelectboxColumn(options=COMPOUNDS),
                    "Stop2Compound": st.column_config.SelectboxColumn(options=COMPOUNDS),
                    "StartAge": st.column_config.NumberColumn(min_value=1, max_value=10),
                    "Stop1Lap": st.column_config.NumberColumn(min_value=0, max_value=total_laps - 1),
                    "Stop2Lap": st.column_config.NumberColumn(min_value=0, max_value=total_laps - 1),
                })

            with st.expander("Race conditions"):
                d1, d2, d3, d4 = st.columns(4)
                track_temp = d1.slider("Track temp (°C)", 15, 55, 35)
                air_temp = d2.slider("Air temp (°C)", 10, 40, 25)
                sc_prob = d3.slider("SC probability / lap", 0.0, 0.05,
                                    float(prof.get("sc_per_lap", DEFAULT_CONFIG["sc_prob_per_lap"])),
                                    0.001, format="%.3f")
                sc_window = d4.slider("Box under SC if stop ≤ N laps away", 0, 15,
                                      DEFAULT_CONFIG["sc_pit_window"])
                e1, e2, e3 = st.columns(3)
                dnf = e1.slider("DNF probability / lap", 0.0, 0.002, DEFAULT_CONFIG["dnf_prob_per_lap"],
                                0.0001, format="%.4f")
                n_sims = e2.select_slider("Simulations", [200, 500, 1000, 2000], 1000)
                seed = e3.number_input("Random seed", 0, 10_000, 42)
            p_any_sc = 1 - (1 - sc_prob) ** max(total_laps - 3, 0)
            st.caption(f"→ P(at least one Safety Car) ≈ {p_any_sc:.0%}")

            config = {"total_laps": int(total_laps), "track_temp": track_temp, "air_temp": air_temp,
                      "sc_prob_per_lap": sc_prob, "sc_pit_window": sc_window, "dnf_prob_per_lap": dnf,
                      "circuit": st.session_state.get("grid_circuit")}
            errs = validate_grid(grid, total_laps)
            for e in errs:
                st.error(e)

            if st.button("Run simulation", type="primary", disabled=bool(errs)):
                st.session_state["grid"] = grid
                st.session_state["sim_config"] = config
                st.session_state["sim"] = cached_simulate(grid, config, int(n_sims), int(seed))
                st.session_state["sim_params"] = (int(n_sims), int(seed))

            if "sim" in st.session_state:
                r = st.session_state["sim"]
                s = r["summary"]
                k1, k2, k3, k4 = st.columns(4)
                fav = s.sort_values("Win %", ascending=False).iloc[0]
                k1.metric("Favourite", fav["Driver"], f"{fav['Win %']:.1f}% wins", delta_color="off", delta_arrow="off")
                k2.metric("Races with a Safety Car", f"{(r['sc_count'] > 0).mean():.0%}")
                k3.metric("Overtakes / race (avg)", f"{r['overtakes'].mean():.1f}")
                k4.metric("Simulations", f"{len(r['sc_count']):,}")

                left, right = st.columns([2, 3])
                with left:
                    top = s.sort_values("Win %", ascending=True).tail(10)
                    fig = go.Figure(go.Bar(
                        x=top["Win %"], y=top["Driver"], orientation="h",
                        marker=dict(color=[team_colour(t) for t in top["Team"]], cornerradius=4),
                        text=[f"{v:.1f}%" for v in top["Win %"]], textposition="outside",
                        hovertemplate="%{y}: %{x:.1f}% wins<extra></extra>"))
                    fig.update_xaxes(range=[0, float(top["Win %"].max()) * 1.18 + 1])     # room for the labels
                    show(base_layout(fig, "Win probability (top 10)", 420, hovermode="closest",
                                     xaxis_title="%", showlegend=False))
                with right:
                    dist = position_distribution(r, r["grid"])
                    order = s["Driver"].tolist()
                    dist = dist.loc[order]
                    fig = go.Figure(go.Heatmap(
                        z=dist.values * 100, x=dist.columns, y=dist.index, colorscale=T.HEAT,
                        zmin=0, zmax=max(40.0, float(dist.values.max() * 100)), colorbar=dict(title="%"),
                        hovertemplate="%{y} finishes %{x}: %{z:.1f}%<extra></extra>"))
                    fig.update_yaxes(autorange="reversed")
                    show(base_layout(fig, "Finishing position distribution", 420, hovermode="closest"))

                st.dataframe(s.style.format({c: "{:.1f}" for c in s.columns if s[c].dtype.kind == "f"}),
                             hide_index=True, width="stretch")

                with st.expander("One simulated race, lap by lap"):
                    g = r["grid"]
                    pick = st.multiselect("Drivers", g["Driver"].tolist(), g["Driver"].tolist()[:5])
                    fig = go.Figure()
                    for d in pick:
                        i = g.index[g["Driver"] == d][0]
                        fig.add_trace(go.Scatter(
                            x=np.arange(1, len(r["trace_gap"]) + 1), y=r["trace_gap"][:, i], name=d,
                            mode="lines", line=dict(width=2, color=team_colour(g.loc[i, "Team"]))))
                    fig.update_yaxes(autorange="reversed")
                    show(base_layout(fig, "Gap to leader (s) — simulation #1", 380,
                                     xaxis_title="Lap", yaxis_title="Gap to leader (s)"))
                    st.caption("Team-mates share a colour; hover to identify. Flat steps = Safety Car bunching, "
                               "spikes = pit stops.")


# ═════════════════════════════ 2. WHAT-IF ═══════════════════════════════════
with TAB["whatif"]:
    st.subheader("What-If: change one strategy, keep the luck identical")
    st.caption("Both scenarios replay the exact same random draws (lap noise, Safety Car timing, "
               "overtake rolls) — *common random numbers*. Any difference is caused by the strategy, "
               "not by luck, so we can compare race-by-race.")
    if not models_ready():
        need_models()
    elif "grid" not in st.session_state:
        st.info("Build a grid in the Race Simulator tab first — it becomes the baseline here.")
    else:
        base = st.session_state["grid"].reset_index(drop=True)
        cfg = st.session_state.get("sim_config", {"total_laps": 52})
        L = cfg["total_laps"]
        focus = st.selectbox("Driver", base["Driver"].tolist(), index=min(3, len(base) - 1))
        row = base[base["Driver"] == focus].iloc[0]
        st.markdown(f"Baseline: **{row.StartCompound}** → L{int(row.Stop1Lap)} **{row.Stop1Compound}**"
                    + (f" → L{int(row.Stop2Lap)} **{row.Stop2Compound}**" if row.Stop2Lap else ""))
        c = st.columns(5)
        ns = c[0].selectbox("Start tyre", COMPOUNDS, COMPOUNDS.index(row.StartCompound), key="wi_s")
        p1 = c[1].number_input("Stop 1 lap", 0, L - 1, int(row.Stop1Lap), key="wi_p1")
        c1_ = c[2].selectbox("Stop 1 tyre", COMPOUNDS, COMPOUNDS.index(row.Stop1Compound), key="wi_c1")
        p2 = c[3].number_input("Stop 2 lap (0 = none)", 0, L - 1, int(row.Stop2Lap), key="wi_p2")
        c2_ = c[4].selectbox("Stop 2 tyre", COMPOUNDS, COMPOUNDS.index(row.Stop2Compound), key="wi_c2")

        new = base.copy()
        i = new.index[new["Driver"] == focus][0]
        new.loc[i, ["StartCompound", "Stop1Lap", "Stop1Compound", "Stop2Lap", "Stop2Compound"]] = \
            [ns, p1, c1_, p2, c2_]
        from tools.race_simulator import validate_grid
        errs = validate_grid(new, L)
        for e in errs:
            st.error(e)
        if new.equals(base):
            st.caption("Change at least one field above — identical strategies give identical races "
                       "(that is the point of common random numbers).")
        n_sims, seed = st.session_state.get("sim_params", (1000, 42))
        if st.button("Compare strategies", type="primary", disabled=bool(errs)):
            st.session_state["wi"] = cached_what_if(base, new, focus, cfg, n_sims, seed)

        if "wi" in st.session_state and st.session_state["wi"]["focus"] == focus:
            w = st.session_state["wi"]
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Finished higher", f"{w['better_pct']:.1f}%")
            k2.metric("Same position", f"{w['same_pct']:.1f}%")
            k3.metric("Finished lower", f"{w['worse_pct']:.1f}%")
            lo, hi = w["mean_gain"] - 1.96 * w["gain_se"], w["mean_gain"] + 1.96 * w["gain_se"]
            k4.metric("Mean places gained", f"{w['mean_gain']:+.2f}", f"95% CI [{lo:+.2f}, {hi:+.2f}]",
                      delta_color="off", delta_arrow="off")
            if lo > 0:
                st.success(f"The new strategy is better for {focus} on average — the 95% CI excludes 0.")
            elif hi < 0:
                st.error(f"The new strategy is worse for {focus} on average — the 95% CI excludes 0.")
            else:
                st.info("No significant difference: the CI includes 0. More simulations would narrow it.")

            h = w["gain_hist"]
            fig = go.Figure(go.Bar(
                x=h.index, y=h.values / h.values.sum() * 100,
                marker=dict(color=[GOOD if x > 0 else BAD if x < 0 else NEUTRAL for x in h.index],
                            cornerradius=4),
                hovertemplate="%{x:+d} places: %{y:.1f}% of races<extra></extra>"))
            show(base_layout(fig, f"{focus}: places gained per race (same race, two strategies)", 340,
                             hovermode="closest", xaxis_title="Places gained (+ = better)",
                             yaxis_title="% of races", showlegend=False))
            comp = w["comparison"]
            st.dataframe(comp.style.format("{:.2f}"), width="stretch")


# ═════════════════════════════ 3. UNDERCUT ══════════════════════════════════
with TAB["undercut"]:
    st.subheader("Undercut calculator")
    st.caption("If the chasing car pits now, does it come out ahead once the leader responds? "
               "The verdict uses the worst case (leader reacts next lap).")
    if not models_ready():
        need_models()
    else:
        from tools.undercut import (VERDICT_GAP, VERDICT_PROB, WORST_CASE, DriverState,
                                    undercut_analysis, undercut_sweep)
        teams = shared.get_known_teams()

        def driver_inputs(col, label, code, team, pos, comp, age, key):
            col.markdown(f"**{label}**")
            a, b = col.columns(2)
            code = a.text_input("Driver", code, key=f"{key}_c").upper()
            team = b.selectbox("Team", teams, teams.index(team) if team in teams else 0, key=f"{key}_t")
            a, b, c = col.columns(3)
            pos = a.number_input("Position", 1, 20, pos, key=f"{key}_p")
            comp = b.selectbox("Tyre", COMPOUNDS, COMPOUNDS.index(comp), key=f"{key}_cp")
            age = c.number_input("Tyre age", 1, 50, age, key=f"{key}_a")
            return DriverState(code, team, pos, comp, age)

        lc, rc = st.columns(2)
        chaser = driver_inputs(lc, "Chaser (considering the stop)", "VER", "Red Bull Racing", 3, "HARD", 14, "uc_a")
        leader = driver_inputs(rc, "Car ahead", "NOR", "McLaren", 2, "MEDIUM", 14, "uc_b")
        uc_circuit = circuit_picker(st, "uc_circuit")
        c = st.columns(6)
        gap = c[0].number_input("Gap (s)", 0.1, 30.0, 2.0, 0.1)
        lap = c[1].number_input("Current lap", 1, 70, 14)
        new_c = c[2].selectbox("Chaser fits", COMPOUNDS, 1)
        lead_c = c[3].selectbox("Leader responds with", COMPOUNDS, 2)
        tt = c[4].number_input("Track °C", 10, 60, 40)
        sc_on = c[5].toggle("Safety Car", False)
        cfg = {"track_temp": tt, "air_temp": tt - 12, "rainfall": 0, "circuit": uc_circuit}

        r = undercut_analysis(chaser, leader, gap, lap, cfg, new_c, lead_c, sc_on)
        if r["verdict"] == "BOX BOX":
            T.pit_board("Box box", f"{chaser.driver} comes out close enough to pass {leader.driver} even if "
                        f"{leader.driver} answers on the next lap.", tone=T.YELLOW)
        else:
            T.pit_board("Stay out", f"The undercut on {leader.driver} does not work from {gap:.1f} s back.",
                        tone=T.PAINT)
        v2, v3, v4 = st.columns(3)
        v2.metric("Worst-case min gap", f"{r['min_gap']:.2f} s",
                  f"needs < {VERDICT_GAP} s", delta_color="off", delta_arrow="off")
        if r["laps_in_drs_window"]:
            v3.metric("Avg P(overtake)", f"{r['confidence']:.0%}", f"needs > {VERDICT_PROB:.0%}",
                      delta_color="off", delta_arrow="off")
        else:
            v3.metric("Avg P(overtake)", "—", "never within 1 s", delta_color="off", delta_arrow="off",
                      help="Module 2 is only asked on laps where the gap is inside the 1 s DRS window. "
                           "In the worst case the chaser never gets that close, so there is no attempt.")
        v4.metric("Pit loss used", f"{r['pit_cost']:.1f} s")

        fig = go.Figure()
        styles = {"Stay out": (NEUTRAL, "dot"), WORST_CASE: (BAD, "solid"),
                  "Leader reacts in 3 laps": (T.YELLOW, "solid"), "Leader never reacts": (GOOD, "dash")}
        for name, curve in r["curves"].items():
            col, dash = styles[name]
            fig.add_trace(go.Scatter(x=r["laps"], y=curve, name=name, mode="lines+markers",
                                     line=dict(color=col, width=2, dash=dash), marker=dict(size=8)))
        fig.add_hline(0, line=dict(color=NEUTRAL, width=1))
        fig.add_hrect(0, VERDICT_GAP, fillcolor=GOOD, opacity=0.08, line_width=0)
        # Zoom on the battle: the ~24 s pit-stop spikes would squash the 0–3 s region to a line.
        y_top = max(gap, max(r["curves"]["Stay out"])) + 3
        y_bot = min(-2.0, min(min(c) for c in r["curves"].values()) - 0.5)
        fig.update_yaxes(range=[y_bot, y_top])
        show(base_layout(fig, f"Gap {chaser.driver} → {leader.driver} (positive = {chaser.driver} behind)", 420,
                         xaxis_title="Lap", yaxis_title="Gap (s)"))
        st.caption(f"Zoomed on the fight: lines that leave the top of the chart are the lap a car is in the "
                   f"pit lane (+{r['pit_cost']:.1f} s). Shaded band = striking distance (< 1.5 s); below 0 = "
                   "the chaser is ahead. 'Never reacts' is only within the 5-lap horizon — the leader still owes a stop.")

        with st.expander("Up to what gap does the undercut work?"):
            st.caption("'Undercut gain' = how much better the chaser ends up after 5 laps than if nobody had "
                       "pitted, in the worst case (both cars stopped). Positive means fresh tyres paid off — "
                       "but it only becomes a position if the gap closes enough to pass.")
            sw = undercut_sweep(chaser, leader, lap, cfg, new_compound=new_c, leader_compound=lead_c,
                                sc_active=sc_on)
            st.dataframe(sw, hide_index=True, width="stretch")
        with st.expander("How it works"):
            st.markdown(
                "- **Module 1** predicts every lap for both cars in every scenario (one batched call).\n"
                "- Pit loss: measured per circuit from real green-flag stops (cheaper under a Safety Car). "
                "Deterministic, so the same inputs always give the same verdict.\n"
                "- **Dirty air:** whoever runs < 2 s behind loses 0.35 s/lap (domain knowledge, not learned).\n"
                "- **Module 2** is asked for P(overtake) only on laps inside the 1 s DRS window, "
                "with each car's *actual* tyre state that lap.\n"
                "- **BOX BOX** only if the worst case closes below 1.5 s *and* the average overtake "
                "probability beats 25 %.")


# ═════════════════════════════ 4. STINT OPTIMIZER ═══════════════════════════
with TAB["stint"]:
    st.subheader("How many stops, and when")
    st.caption("Every pit lap and every legal compound order for one, two and three stops, on a clear track. "
               "How hard a day is on tyres is only known roughly before the race, so the answer comes with a "
               "probability.")
    if not models_ready():
        need_models()
    else:
        from tools.circuit_profiles import profile_for
        from tools.stint_optimizer import compare_strategies
        teams = shared.get_known_teams()
        c = st.columns([2, 1, 1])
        so_circuit = circuit_picker(c[0], "so_circuit")
        so_prof = profile_for(so_circuit) or {}
        laps_ = c[1].number_input("Race laps", 20, 80, int(so_prof.get("laps", 57)), key=f"so_l_{so_circuit}")
        pc = c[2].number_input("Pit loss (s)", 15.0, 35.0, float(so_prof.get("pit_loss_s") or shared.PIT_COST_GREEN),
                               0.5, key=f"so_pc_{so_circuit}", help="Measured per circuit from real green-flag stops.")
        tt, at, team = 35.0, 25.0, teams[0]
        if {"TrackTemp", "TeamEncoded"} & set(shared.model_features(get_models()[0], shared.LAP_FEATURES)):
            c = st.columns(3)                         # only older models use these inputs
            team = c[0].selectbox("Team", teams, key="so_t")
            tt = c[1].slider("Track °C", 15, 55, 35, key="so_tt")
            at = c[2].slider("Air °C", 10, 40, 25, key="so_at")

        o = cached_outlook(int(laps_), so_circuit, float(pc), float(tt), float(at))
        exp = o["expected"]
        best = exp.loc[exp["Total Δ (s)"].idxmin()]
        runner = exp[exp["Stops"] != best["Stops"]].sort_values("Total Δ (s)").iloc[0]
        words = {1: "One stop", 2: "Two stops", 3: "Three stops"}
        p = o["p_best"].get(int(best["Stops"]), 1.0)
        why = (f"{best['Strategy']}, stopping on {best['Pit laps'].replace(', ', ' and ')} — "
               f"{runner['Gap to best (s)']:.1f} s faster than the best {words[int(runner['Stops'])].lower().replace(' ', '-')} plan.")
        ctx = (f"The fastest plan on {round(p * 100)} % of the days this circuit can throw at you "
               f"(tyres lose {o['deg_rate']:.3f} ± {o['deg_sd']:.3f} s per lap of age here).") if o["deg_rate"] else ""
        T.pit_board(words[int(best["Stops"])], why, ctx, tone=T.YELLOW)

        left, right = st.columns([3, 2])
        with left:
            st.dataframe(exp[["Stops", "Strategy", "Pit laps", "Total Δ (s)", "Gap to best (s)"]],
                         hide_index=True, width="stretch")
            if len(o["by_rate"]):
                by = o["by_rate"]
                fig = go.Figure()
                for n, colour in ((2, T.YELLOW), (3, T.PURPLE)):
                    if f"{n}-stop" in by:
                        fig.add_trace(go.Scatter(x=by["Degradation (s/lap)"], y=by[f"{n}-stop"] - by["1-stop"],
                                                 name=f"{words[n]} vs one stop", mode="lines+markers",
                                                 line=dict(color=colour, width=3), marker=dict(size=7)))
                fig.add_hline(0, line=dict(color=T.NEUTRAL, width=1))
                fig.add_vline(o["deg_rate"], line=dict(color=T.NEUTRAL, dash="dot"),
                              annotation_text="usual here", annotation_font_color=T.MUTED)
                show(base_layout(fig, "More stops pay off when tyres degrade faster", 360,
                                 xaxis_title="Degradation today (s per lap of tyre age)",
                                 yaxis_title="Seconds vs the best one-stop"))
                st.caption("Each dot is an equally likely day at this circuit. Below zero, the extra stop is "
                           "faster than staying on one stop.")
        with right:
            real = cached_real_stops()
            here = real[real["Race"].map(shared.circuit_name) == so_circuit] if real is not None and so_circuit else None
            if here is not None and len(here):
                st.markdown("**What the field really did here**")
                tab = (here.assign(n=here["stops"].clip(upper=3))
                           .pivot_table(index="Season", columns="n", values="Driver", aggfunc="count", fill_value=0))
                tab = tab.reindex(columns=sorted(set(tab.columns) | {1, 2, 3}), fill_value=0)
                tab = tab.rename(columns={0: "No stop", 1: "1 stop", 2: "2 stops", 3: "3 or more"})
                tab["Stops under SC"] = here.groupby("Season")["sc_stops"].sum()
                st.dataframe(tab.reset_index().astype({"Season": str}), hide_index=True, width="stretch")
                st.caption("Finishers by number of tyre stops (red-flag changes are free and not counted). "
                           "A Safety Car makes a stop cheap and adds stops the plan did not have.")
            T.note("The search assumes a clear track and drivers who push the same on any plan. Real teams also "
                   "weigh traffic after the stop and tyre management, and where nobody can pass (Monaco) track "
                   "position beats tyre life. Checked against 36 real races, one at a time and never seen in "
                   "training: the size of the gap ranks races well once the day's degradation is known, but the "
                   "tool still favours one stop by a few seconds — read a small margin as a toss-up.")

        with st.expander("The one-stop sweep: total time by pit lap"):
            df = compare_strategies(int(laps_), team, {"track_temp": tt, "air_temp": at, "circuit": so_circuit},
                                    pit_cost=pc)
            fig = go.Figure()
            for _, row in df.iterrows():
                cur = row["_result"]["curve"]
                c1, c2 = row["Strategy"].split(" → ")
                fig.add_trace(go.Scatter(
                    x=cur["PitLap"], y=cur["Total"] - df["Total Δ (s)"].min(), name=row["Strategy"],
                    mode="lines", line=dict(width=3 if row.name == 0 else 2, color=COMPOUND_COLOURS[c1],
                                            dash="solid" if c2 == "HARD" else "dot" if c2 == "MEDIUM" else "dash")))
            fig.update_yaxes(range=[-1, 25])
            show(base_layout(fig, "Time lost vs the best one-stop, by pit lap", 420,
                             xaxis_title="Pit lap", yaxis_title="Seconds vs best"))
            st.caption("Pit too early and the second stint is long and slow; too late and you crawl on dead "
                       "tyres. Line colour = first tyre, line style = second tyre (solid HARD, dotted MEDIUM, "
                       "dashed SOFT). Two orders of the same pair cost the same: fuel is identical for every "
                       "plan, so only stint lengths matter.")
            st.dataframe(df.drop(columns="_result"), hide_index=True, width="stretch")


# ═════════════════════════════ 5. QUALY → RACE ══════════════════════════════
with TAB["qualy"]:
    st.subheader("Qualifying → race pace")
    st.caption("Qualifying time + fuel load + switch from SOFT to the race tyre + average degradation "
               "over the first stint. Compared against the actual result when it's available.")
    if not models_ready():
        need_models()
    else:
        from tools.qualy_predictor import compare_with_actual, predict_race_pace
        year, gp = event_picker("qp")
        c = st.columns(3)
        sc_ = c[0].selectbox("Race start tyre", COMPOUNDS, 1, key="qp_c")
        tt = c[1].slider("Track °C", 15, 55, 35, key="qp_tt")
        at = c[2].slider("Air °C", 10, 40, 28, key="qp_at")
        if st.button("Predict", type="primary", key="qp_go"):
            st.session_state["qp_key"] = (year, gp)
        if "qp_key" in st.session_state:
            y, g = st.session_state["qp_key"]
            try:
                q = cached_quali(y, g)
            except Exception as e:
                st.error(f"Couldn't load qualifying for {g} {y}: {e}")
                q = None
            if q is not None:
                pred = predict_race_pace(q, sc_, tt, at)
                actual = cached_race_result(y, g)
                cmpd = compare_with_actual(pred, actual) if actual is not None else None

                k = st.columns(3)
                k[0].markdown(f"**Event**  \n{q.attrs.get('event', g)}")
                if cmpd:
                    k[1].metric("ρ qualy order", f"{cmpd['spearman_qualy']:.3f}",
                                help="Spearman rank correlation between the qualifying order and the "
                                     f"classified race result ({cmpd['n']} finishers, "
                                     f"{cmpd['n_excluded']} DNF/unclassified excluded). The naive baseline.")
                    k[2].metric("ρ model", f"{cmpd['spearman_model']:.3f}",
                                f"{cmpd['spearman_model'] - cmpd['spearman_qualy']:+.3f} vs baseline")
                else:
                    k[1].metric("Actual result", "not available")

                fig = go.Figure()
                for _, row in pred.iterrows():
                    col = team_colour(row["Team"])
                    fig.add_trace(go.Scatter(
                        x=["Qualifying", "Predicted race"], y=[row["QualyRank"], row["PredictedRaceRank"]],
                        mode="lines+markers+text", line=dict(color=col, width=2), marker=dict(size=9),
                        text=[row["Driver"], row["Driver"]], textposition=["middle left", "middle right"],
                        name=row["Driver"], showlegend=False,
                        hovertemplate=f"{row['Driver']} ({row['Team']})<br>%{{x}}: P%{{y}}<extra></extra>"))
                fig.update_yaxes(range=[len(pred) + 0.5, 0.5], dtick=1, title="Position")
                show(base_layout(fig, "Qualifying rank → predicted race-pace rank", 560, hovermode="closest"))
                if "TeamEncoded" in shared.model_features(get_models()[0], shared.LAP_FEATURES):
                    st.info("Every correction is team-level, so team-mates can never swap order — the model "
                            "re-shuffles teams, not drivers. A driver-level race-pace feature is the obvious "
                            "next step.", icon="🔍")
                else:
                    st.info("Module 1 V3.1 has no team effect: adding the team made the tyre physics worse "
                            "and the error on unseen races larger. So the fuel, compound and degradation "
                            "corrections are the same for every car and **the predicted order equals the "
                            "qualifying order**. That is an honest result, not a bug: from qualifying alone "
                            "the data gives no reliable reason to re-order the grid. What re-orders a race is "
                            "pace seen *during* it — which is what the Race Engineer uses (it beats 'order "
                            "stays the same' in 75 % of the backtested races).", icon="🔍")
                cols = ["QualyRank", "Driver", "Team", "QualyTime", "FuelAdjTime", "CompoundDelta",
                        "DegAverage", "PredictedRacePace", "PredictedRaceRank", "Direction"]
                show_df = pred[cols].copy()
                if cmpd:
                    show_df["ActualRank"] = show_df["Driver"].map(actual)
                st.dataframe(show_df.round(3), hide_index=True, width="stretch")


# ═════════════════════════════ 6. TELEMETRY CLASH ═══════════════════════════
with TAB["clash"]:
    st.subheader("Telemetry — where is the lap time?")
    st.caption("The fastest laps of two drivers on the same distance grid: where one pulls away, the speed "
               "traces and a corner-by-corner breakdown. No model here, only signal processing.")
    year, gp = event_picker("tc", kind="telemetry")
    c = st.columns(3)
    if has_fastf1_cache():
        session_type = c[0].selectbox("Session", ["Q", "R", "SQ", "FP1", "FP2", "FP3"])
        da = c[1].text_input("Driver A", "VER").upper()
        db = c[2].text_input("Driver B", "LEC").upper()
    else:                                            # demo bundle: the laps it ships
        session_type = c[0].selectbox("Session", ["Q"])
        have = demo_data.telemetry_drivers(year, gp, session_type) or ["VER", "LEC"]
        da = c[1].selectbox("Driver A", have, have.index("VER") if "VER" in have else 0)
        db = c[2].selectbox("Driver B", have, have.index("LEC") if "LEC" in have else min(1, len(have) - 1))
    if st.button("Compare", type="primary", key="tc_go"):
        st.session_state["tc_key"] = (year, gp, session_type, da, db)
    if "tc_key" in st.session_state:
        from tools.telemetry_clash import analyze_corners, plot_clash
        try:
            res = cached_clash(*st.session_state["tc_key"])
        except Exception as e:
            st.error(f"Couldn't build the comparison: {e}")
            res = None
        if res is not None:
            m, v = res["meta"], res["validation"]
            a, b = m["driver_a"], m["driver_b"]
            k = st.columns(4)
            k[0].metric(f"{a} lap", fmt_lap(m["lap_time_a"]), m["compound_a"], delta_color="off", delta_arrow="off")
            k[1].metric(f"{b} lap", fmt_lap(m["lap_time_b"]), m["compound_b"], delta_color="off", delta_arrow="off")
            faster = a if v["actual_dt_ms"] < 0 else b
            k[2].metric("Faster", faster, f"{abs(v['actual_dt_ms']):.0f} ms", delta_color="off", delta_arrow="off")
            k[3].metric("Recon. error", f"{v['error_ms']:+.1f} ms",
                        help="Cumulative delta at the line minus the official lap-time difference. "
                             "A pipeline sanity check: the endpoint is pinned by construction, "
                             "the shape in between is what carries information.")
            with plt.style.context("dark_background"), plt.rc_context(
                    {"figure.facecolor": T.ASPHALT, "axes.facecolor": T.ASPHALT, "savefig.facecolor": T.ASPHALT,
                     "axes.edgecolor": T.LINE, "grid.color": T.LINE, "text.color": T.PAINT,
                     "axes.labelcolor": T.MUTED, "xtick.color": T.MUTED, "ytick.color": T.MUTED}):
                st.pyplot(plot_clash(res), width="stretch")
            st.caption("Sharp, short spikes in slow corners are mostly an **alignment artefact**: each car's "
                       "Distance channel is integrated from its own speed trace, so the same physical point can "
                       "sit a few metres apart. At 90 km/h, 5 m ≈ 200 ms. Read the trend between corners, "
                       "not the spikes.")
            corners = analyze_corners(res)
            st.markdown(f"**Corner by corner** — positive values favour **{b}**")
            st.dataframe(
                corners.style.background_gradient(subset=["Time gain (ms)"], cmap="RdBu",
                                                  vmin=-60, vmax=60).format(precision=1),
                hide_index=True, width="stretch")
            if len(corners):
                best_c = corners.loc[corners["Time gain (ms)"].abs().idxmax()]
                who = b if best_c["Time gain (ms)"] > 0 else a
                st.caption(f"Biggest single swing: **{best_c['Corner']}** — {who} gains "
                           f"{abs(best_c['Time gain (ms)']):.0f} ms.")


# ═════════════════════════════ 7. RACE ENGINEER ═════════════════════════════
with TAB["engineer"]:
    st.subheader("The pit-wall call at any lap of a real race")
    st.caption("Pick a race, a driver and a lap. The engineer sees only what had happened when that driver "
               "crossed the line, races every option to the flag on the same simulated races, and makes the "
               "call. Then compare it with what the team really did.")
    laps_df = cached_laps()
    if not models_ready():
        need_models()
    elif laps_df is None:
        st.warning("Needs the lap Parquet (`data/laps_2022_2025.parquet`) — `python -m tools.export_cache`, "
                   "or the demo bundle in `demo/`.", icon="🧰")
    else:
        from tools.race_engineer import MIN_CALL_LAP
        from tools.race_engine import actual_result
        races = sorted({(int(a), b) for a, b in laps_df[["Season", "Race"]].drop_duplicates().itertuples(index=False)},
                       reverse=True)
        labels = [f"{a} {b}" for a, b in races]
        c = st.columns([2, 1, 3, 1])
        choice = c[0].selectbox("Race", labels, labels.index("2023 Bahrain Grand Prix")
                                if "2023 Bahrain Grand Prix" in labels else 0, key="re_race")
        season, race = races[labels.index(choice)]
        feed = cached_feed(season, race)
        res = actual_result(laps_df, season, race)
        drivers = list(res.index) + sorted(set(feed.tl) - set(res.index))
        driver = c[1].selectbox("Driver", drivers, drivers.index("ALO") if "ALO" in drivers else 0, key="re_drv")
        done = [l for l in feed.tl[driver]["lap"] if MIN_CALL_LAP <= l <= feed.total_laps - 2]
        if not done:
            st.info(f"{driver} did not complete enough laps for a call.")
        else:
            lap = c[2].select_slider("Lap", done, value=min(20, done[-1]) if 20 in done else done[len(done) // 2],
                                     key="re_lap")
            sims = c[3].selectbox("Sims", [100, 300, 1000], 1, key="re_sims")
            call = cached_engineer_call(season, race, driver, int(lap), int(sims))
            if call is None:
                st.info("No call at this lap (the engineer needs a few laps of pace first).")
            else:
                h = call["headline"]
                st_ = call["standings"]
                me = st_[st_.Driver == driver]
                ty = call.get("tyres") or {}
                verdict, _, reason = h.partition(" — ")
                deg = call.get("degradation")
                today = ""
                if deg:
                    drift = deg["mean"] - deg["prior_mean"]
                    how = "as usual here" if abs(drift) < 0.01 else ("harder on tyres than usual" if drift > 0
                                                                     else "easier on tyres than usual")
                    today = (f"Today is {how}: {deg['mean']:.3f} ± {deg['sd']:.3f} s lost per lap of tyre age "
                             f"(this circuit: {deg['prior_mean']:.3f}), from {deg['n_laps']} clean laps so far.")
                plan = (call.get("rival_plans") or {}).get(driver) or []
                plan_txt = " then ".join(f"lap {l_} for {c_.title()}s" for l_, c_ in plan)
                if h.startswith("STAY"):
                    reason = (f"Keep to the plan: box on {plan_txt}." if plan
                              else "No more stops: these tyres go to the flag.")
                else:
                    reason = reason[:1].upper() + reason[1:]
                T.pit_board(verdict, reason,
                            f"End of lap {call['lap']} of {call['total_laps']}. {driver} is "
                            f"P{int(me.Pos.iloc[0]) if len(me) else '?'} on {ty.get('compound', '?').title()}s, "
                            f"{ty.get('age', 1) - 1} laps old. {today}",
                            tone=T.PAINT if h.startswith("STAY") else T.YELLOW)

                # what the team really did after this lap
                r = laps_df[(laps_df.Season == season) & (laps_df.Race == race) & (laps_df.Driver == driver)]
                r = r.sort_values("LapNumber")
                nxt = r[(r.LapNumber > lap) & r.PitInTime.notna() & (r.LapNumber < feed.total_laps)]
                if len(nxt):
                    pl = int(nxt.LapNumber.iloc[0])
                    after = r[r.LapNumber == pl + 1].Compound
                    late = pl - (int(lap) + 1)          # "Box now" at the end of lap n = in-lap n+1
                    when = ("the same lap as \"Box now\"" if late == 0 else
                            f"{late} lap{'s' if late != 1 else ''} after \"Box now\" would have")
                    team = f"boxed at the end of lap {pl} ({when}) " \
                           f"onto {after.iloc[0].title() if len(after) else '?'}"
                else:
                    team = "made no more stops"
                fin = res.get(driver)
                k = st.columns(3)
                k[0].metric("The engineer's plan", ("Box lap " + ", ".join(str(l_) for l_, _ in plan)) if plan
                            else "No more stops")
                k[1].metric("The team", team.split(" (")[0].replace("boxed at the end of", "box").capitalize())
                k[2].metric("Real result", f"P{int(fin)}" if pd.notna(fin) else "DNF")
                st.caption(f"In reality {driver} {team}.")

                left, right = st.columns([3, 2])
                with left:
                    st.markdown("**Options — every one raced to the flag on the same simulated races**")
                    opt = call["options"].copy()
                    for col in ("P(win)", "P(podium)", "P(points)"):
                        opt[col] = (opt[col] * 100).round(0).astype(int).astype(str) + " %"
                    opt["vs stay out"] = [("—" if o.startswith("Stay out") else f"{g:+.2f} ± {ci:.2f}")
                                          for o, g, ci in zip(opt.Option, opt["Gain vs first (places)"],
                                                              opt["Gain 95% CI"])]
                    st.dataframe(opt[["Option", "P(win)", "P(podium)", "P(points)", "Exp. position", "vs stay out"]]
                                 .round(2), hide_index=True, width="stretch")
                    a = call.get("alert")
                    if a:
                        st.info(f"Undercut check on {a['target']} ({a['gap']:.1f} s ahead): **{a['verdict']}** · "
                                f"P(pass) {a['p_overtake']:.0%}")
                    T.note("Every car, rivals included, is given the plan that is fastest to the flag on a clear "
                           "track at today's degradation; real teams also react to each other. Dry races only.")
                with right:
                    st.markdown("**Timing at the end of the lap** (~ = estimated)")
                    plans = call.get("rival_plans") or {}
                    tower = pd.DataFrame({
                        "Pos": st_.Pos.astype(int),
                        "Driver": [("~" if e else "") + d for d, e in zip(st_.Driver, st_.get("Estimated", False))],
                        "Gap": ["LEADER" if i == 0 else f"+{g:.1f}" for i, g in enumerate(st_.Gap)],
                        "Tyre": [f"{c_[0]} ({int(a_)})" for c_, a_ in zip(st_.Compound, st_.TyreAge)],
                        "Stops": st_.Stops.astype(int),
                        "Stops to come (assumed)": [" · ".join(f"L{l_}→{c_[0]}" for l_, c_ in plans.get(d, [])) or "—"
                                                for d in st_.Driver]})
                    st.dataframe(tower, hide_index=True, width="stretch", height=560)
