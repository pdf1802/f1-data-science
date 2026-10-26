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

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tools import shared
from tools.shared import COMPOUND_COLOURS, team_colour

st.set_page_config(page_title="F1 Pit Wall", page_icon="🏎️", layout="wide")

COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]
YEARS = [2025, 2024, 2023, 2022]
NEUTRAL = "#8a8f98"
GOOD, BAD = "#2fbf71", "#e5484d"


# ── Cached wrappers ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def get_models():
    shared.enable_fastf1_cache()
    return shared.load_models()


@st.cache_data(show_spinner="Loading qualifying from FastF1…", max_entries=16)
def cached_quali(year: int, gp: str) -> pd.DataFrame:
    from tools.qualy_predictor import load_qualifying_results
    shared.enable_fastf1_cache()
    return load_qualifying_results(year, gp)


@st.cache_data(show_spinner="Loading race result…", max_entries=16)
def cached_race_result(year: int, gp: str):
    from tools.qualy_predictor import load_race_result
    return load_race_result(year, gp)


@st.cache_data(show_spinner="Loading telemetry and computing deltas…", max_entries=8)
def cached_clash(year, gp, session_type, a, b):
    from tools.telemetry_clash import compare_drivers
    shared.enable_fastf1_cache()
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


@st.cache_resource(show_spinner=False)
def model_info() -> dict:
    """Which model generation is loaded + the physics gate from tools/training.py."""
    from tools.training import lap_model_sanity
    lap_model, _, enc = get_models()
    feats = shared.model_features(lap_model, shared.LAP_FEATURES)
    teams = range(len(enc.classes_)) if "TeamEncoded" in feats else (0,)
    sanity = lap_model_sanity(lap_model, teams, n_circuits=len(shared.get_known_circuits()))
    version = "V2" if "LapNumber" in feats else "V3.1" if "CircuitEncoded" in feats else "V3"
    return {"version": version, "sanity": sanity}


AVERAGE_CIRCUIT = "Average of all circuits"


def circuit_picker(col, key: str, default: str = "Bahrain"):
    """Circuit selector for the model-based tabs → event name, or None (= averaged)."""
    known = shared.get_known_circuits()
    if not known:                                     # V2 models: no circuit feature
        return None
    options = known + [AVERAGE_CIRCUIT]
    choice = col.selectbox("Circuit", options, options.index(default) if default in known else len(known),
                           key=key, help="The V3.1 models know the circuit: degradation and overtaking "
                                         "difficulty differ a lot (Monaco vs Spa).")
    return None if choice == AVERAGE_CIRCUIT else choice


def models_ready() -> bool:
    try:
        get_models()
        return True
    except FileNotFoundError:
        return False


def need_models():
    st.warning(
        "This tool needs the trained models (`lap_time_model.pkl`, `overtaking_model.pkl`, "
        "`team_encoder.pkl`, `compound_map.json`). Train them with `notebooks/lap_time_model.ipynb` "
        "and `notebooks/overtaking_model.ipynb`, then put them in `models/` or set `F1_MODEL_DIR`.",
        icon="🧰")


def base_layout(fig: go.Figure, title: str = "", height: int = 380, **kw) -> go.Figure:
    fig.update_layout(title=dict(text=title, x=0, y=0.98, yanchor="top", font=dict(size=15)),
                      height=height, margin=dict(l=10, r=10, t=90 if title else 40, b=10),
                      hovermode=kw.pop("hovermode", "x unified"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0), **kw)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig


def fmt_lap(seconds: float) -> str:
    """86.203 → '1:26.203' (fits in a metric tile; '86.203 s' got truncated)."""
    m, sec = divmod(float(seconds), 60)
    return f"{int(m)}:{sec:06.3f}"


def show(fig):
    st.plotly_chart(fig, width="stretch", theme="streamlit")


def event_picker(key: str, default_gp: str = "British Grand Prix", default_year: int = 2024):
    c1, c2 = st.columns([1, 3])
    year = c1.selectbox("Season", YEARS, index=YEARS.index(default_year), key=f"{key}_y")
    gp = c2.text_input("Grand Prix (FastF1 event name or country)", default_gp, key=f"{key}_gp",
                       help="Use the full event name, e.g. 'British Grand Prix'. "
                            "Short names like 'Great Britain' can be fuzzy-matched to the wrong race.")
    return year, gp


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏎️ F1 Pit Wall")
    st.caption("Strategy & performance tools built on FastF1 + XGBoost.")
    if models_ready():
        _md = shared.resolve_model_dir()
        _label = "repo `models/`" if _md == shared._LOCAL_MODEL_DIR else "`$F1_MODEL_DIR`" \
            if os.environ.get("F1_MODEL_DIR") else f"`{os.path.basename(_md)}`"
        st.success(f"Models loaded from {_label}", icon="✅")
        _info = model_info()
        if _info["sanity"]["passed"]:
            st.caption(f"Module 1 {_info['version']} · physics check ✅")
        else:
            st.warning(f"Module 1 {_info['version']} fails the physics check — tyre/stint numbers "
                       "are not trustworthy.", icon="⚠️")
            with st.expander("Details"):
                for _k, _ok in _info["sanity"]["checks"].items():
                    st.markdown(f"{'✅' if _ok else '❌'} {_k}")
                st.caption("Fix: `python -m tools.train_models` (Module 1 V3: fuel-corrected target).")
    else:
        st.error("Models not found — model-based tabs are disabled.", icon="⚠️")
    st.markdown(
        "**Models**\n"
        "- *Module 1* — LapDelta regressor (tyre, fuel, team, weather)\n"
        "- *Module 2* — overtake classifier (gap, tyre offset, compounds)\n\n"
        "**Known limitations** are listed in the README — read them before "
        "trusting a number.")
    st.caption("Data: FastF1 (F1 live timing). Not affiliated with Formula 1.")

st.title("F1 Pit Wall")

tabs = st.tabs(["🏁 Race Simulator", "🔀 What-If", "⏱️ Undercut", "🛞 Stint Optimizer",
                "📊 Qualy → Race", "📡 Telemetry Clash"])


# ═════════════════════════════ 1. RACE SIMULATOR ════════════════════════════
with tabs[0]:
    st.subheader("Monte Carlo race simulator")
    st.caption("Thousands of races with tyre degradation (Module 1), overtakes that actually change "
               "track position (Module 2), Safety Cars that bunch the field and make stops cheap, "
               "and DNFs. Edit any driver's strategy in the table.")
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
        c1, c2, c3, c4 = st.columns(4)
        total_laps = c1.number_input("Race laps", 20, 80, int(prof.get("laps", 52)))
        start_c = c2.selectbox("Default start tyre", COMPOUNDS, index=1)
        next_c = c3.selectbox("Default 2nd tyre", COMPOUNDS, index=2)
        pit_lap = c4.number_input("Default pit lap", 2, 78, int(round(52 * 0.45)))

        if st.button("Load grid from qualifying", key="load_grid"):
            try:
                q = cached_quali(year, gp)
                st.session_state["grid"] = grid_from_qualifying(q, total_laps, start_c, next_c, pit_lap)
                st.session_state["grid_event"] = q.attrs.get("event", f"{gp} {year}")
                st.session_state["grid_circuit"] = q.attrs.get("event_name", gp)
            except Exception as e:
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
                    show(base_layout(fig, "Win probability (top 10)", 420, hovermode="closest",
                                     xaxis_title="%", showlegend=False))
                with right:
                    dist = position_distribution(r, r["grid"])
                    order = s["Driver"].tolist()
                    dist = dist.loc[order]
                    fig = go.Figure(go.Heatmap(
                        z=dist.values * 100, x=dist.columns, y=dist.index, colorscale="Blues",
                        zmin=0, zmax=100, colorbar=dict(title="%"),
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
with tabs[1]:
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
with tabs[2]:
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
            st.success(f"### BOX BOX — {chaser.driver} should pit now", icon="🟢")
        else:
            st.error(f"### STAY OUT — the undercut doesn't work here", icon="🔴")
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
                  "Leader reacts in 3 laps": ("#f5a524", "solid"), "Leader never reacts": (GOOD, "dash")}
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
                "- Pit loss: 23.5 s green, 17 s under SC. Deterministic, so the same inputs always give "
                "the same verdict.\n"
                "- **Dirty air:** whoever runs < 2 s behind loses 0.35 s/lap (domain knowledge, not learned).\n"
                "- **Module 2** is asked for P(overtake) only on laps inside the 1 s DRS window, "
                "with each car's *actual* tyre state that lap.\n"
                "- **BOX BOX** only if the worst case closes below 1.5 s *and* the average overtake "
                "probability beats 25 %.")


# ═════════════════════════════ 4. STINT OPTIMIZER ═══════════════════════════
with tabs[3]:
    st.subheader("Stint optimizer — the 1-stop U-curve")
    st.caption("Pit too early and the second stint is long and slow; too late and you crawl on dead tyres. "
               "The total time is U-shaped; the bottom is the optimal lap.")
    if not models_ready():
        need_models()
    else:
        from tools.stint_optimizer import compare_strategies
        teams = shared.get_known_teams()
        from tools.circuit_profiles import profile_for
        so_circuit = circuit_picker(st, "so_circuit")
        so_prof = profile_for(so_circuit) or {}
        c = st.columns(5)
        laps_ = c[0].number_input("Race laps", 20, 80, int(so_prof.get("laps", 57)), key=f"so_l_{so_circuit}")
        team = c[1].selectbox("Team", teams, teams.index("Red Bull Racing") if "Red Bull Racing" in teams else 0,
                              key="so_t")
        tt = c[2].slider("Track °C", 15, 55, 35, key="so_tt")
        at = c[3].slider("Air °C", 10, 40, 25, key="so_at")
        pc = c[4].number_input("Pit loss (s)", 15.0, 35.0, float(so_prof.get("pit_loss_s") or shared.PIT_COST_GREEN),
                               0.5, key=f"so_pc_{so_circuit}", help="Measured per circuit from real green-flag stops.")
        df = compare_strategies(int(laps_), team, {"track_temp": tt, "air_temp": at, "circuit": so_circuit},
                                pit_cost=pc)

        best = df.iloc[0]
        st.markdown(f"#### Best 1-stop: {best['Strategy']}")
        k2, k3 = st.columns(2)
        k2.metric("Optimal pit lap", f"L{best['Optimal pit lap']}", best["Window (±1 s)"], delta_color="off", delta_arrow="off")
        pair = lambda st_: frozenset(st_.split(" → "))
        others = df[df["Strategy"].map(pair) != pair(best["Strategy"])]
        k3.metric("Margin over next compound pair", f"{others['Gap to best (s)'].iloc[0]:.2f} s",
                  help="SOFT→HARD and HARD→SOFT (with the stop at the mirrored lap) cost the same in the "
                       "V3 tyre model: fuel is identical for every plan, so only stint lengths matter. "
                       "Teams usually start on the softer tyre for launch grip — something the model "
                       "does not know.")

        fig = go.Figure()
        for _, row in df.iterrows():
            cur = row["_result"]["curve"]
            c1, c2 = row["Strategy"].split(" → ")
            is_best = row.name == 0
            fig.add_trace(go.Scatter(
                x=cur["PitLap"], y=cur["Total"] - df["Total Δ (s)"].min(), name=row["Strategy"],
                mode="lines", line=dict(width=3 if is_best else 2, color=COMPOUND_COLOURS[c1],
                                        dash="solid" if c2 == "HARD" else "dot" if c2 == "MEDIUM" else "dash")))
        fig.update_yaxes(range=[-1, 25])
        show(base_layout(fig, "Time lost vs best strategy, by pit lap", 440,
                         xaxis_title="Pit lap", yaxis_title="Seconds vs best"))
        st.caption("Line colour = first tyre, line style = second tyre (solid HARD, dotted MEDIUM, dashed SOFT).")
        st.dataframe(df.drop(columns="_result"), hide_index=True, width="stretch")

        with st.expander("Why is this the optimum? Lap-by-lap degradation of the best strategy"):
            lt = best["_result"]["laps"]
            lt = lt[lt["PitLap"] == best["Optimal pit lap"]]
            fig = go.Figure(go.Scatter(
                x=lt["LapNumber"], y=lt["Delta"], mode="markers",
                marker=dict(size=8, color=[COMPOUND_COLOURS[shared.COMPOUND_NAMES[int(v)]]
                                           for v in lt["CompoundEncoded"]],
                            line=dict(width=1, color="#444")),
                hovertemplate="Lap %{x}: +%{y:.2f} s<extra></extra>"))
            fig.add_vline(best["Optimal pit lap"] + 0.5, line=dict(dash="dot", color=NEUTRAL))
            show(base_layout(fig, "Predicted LapDelta (s above the driver's best)", 320,
                             hovermode="closest", xaxis_title="Lap", yaxis_title="LapDelta (s)"))
            if model_info()["version"] != "V2":
                st.caption("V3 model: deltas are fuel-corrected, so this is tyre performance only — each "
                           "stint should start low and rise. The optimiser trades that rise against the pit loss.")
            else:
                st.caption("V2 model: deltas still contain fuel, and fuel is confounded with the compound "
                           "(softs run early, on a heavy car). Retrain with `python -m tools.train_models`.")


# ═════════════════════════════ 5. QUALY → RACE ══════════════════════════════
with tabs[4]:
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
with tabs[5]:
    st.subheader("Telemetry clash — where is the lap time?")
    st.caption("Fastest laps of two drivers resampled onto the same distance grid. Cumulative time delta, "
               "speed traces and a corner-by-corner breakdown. No ML needed — pure signal processing.")
    year, gp = event_picker("tc")
    c = st.columns(3)
    session_type = c[0].selectbox("Session", ["Q", "R", "SQ", "FP1", "FP2", "FP3"])
    da = c[1].text_input("Driver A", "VER").upper()
    db = c[2].text_input("Driver B", "NOR").upper()
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
            st.pyplot(plot_clash(res), width="stretch")
            st.caption("Sharp, short spikes in slow corners are mostly an **alignment artefact**: each car's "
                       "Distance channel is integrated from its own speed trace, so the same physical point can "
                       "sit a few metres apart. At 90 km/h, 5 m ≈ 200 ms. Read the trend between corners, "
                       "not the spikes.")
            corners = analyze_corners(res)
            st.markdown(f"**Corner by corner** — positive values favour **{b}**")
            st.dataframe(
                corners.style.background_gradient(subset=["Time gain (ms)"], cmap="RdBu",
                                                  vmin=-60, vmax=60),
                hide_index=True, width="stretch")
            if len(corners):
                best_c = corners.loc[corners["Time gain (ms)"].abs().idxmax()]
                who = b if best_c["Time gain (ms)"] > 0 else a
                st.caption(f"Biggest single swing: **{best_c['Corner']}** — {who} gains "
                           f"{abs(best_c['Time gain (ms)']):.0f} ms.")
