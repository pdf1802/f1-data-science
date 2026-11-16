"""
Look of the dashboard — one place for colours, type and chart styling.

"Tarmac and paint": a dark asphalt base; pit-board yellow for the things you act on; the tyre
compounds' own colours (red / yellow / white) and the timing screen's green and purple carry the
data. The one loud element is the pit board (pit_board): the call, in the lettering a driver
reads at 300 km/h. Everything around it stays quiet.

Streamlit's theme (colours, bundled fonts) is in .streamlit/config.toml; this module adds the
CSS that the theme cannot express and the Plotly styling shared by every chart.
"""

from __future__ import annotations

import html

import plotly.graph_objects as go
import streamlit as st

ASPHALT, GARAGE, LINE = "#131A22", "#1B2530", "#2B3947"
PAINT, MUTED = "#EDF1F4", "#93A1AF"
YELLOW, RED, GREEN, PURPLE, BLUE = "#FFD12E", "#E5484D", "#2FBF71", "#B58CFF", "#58A6FF"
GOOD, BAD, NEUTRAL = GREEN, RED, "#8794A2"
SERIES = [YELLOW, BLUE, GREEN, PURPLE, RED, PAINT]            # categorical order for charts
HEAT = [[0.0, GARAGE], [0.5, "#7A6A2A"], [1.0, YELLOW]]       # sequential: none → a lot
DIVERGING = [[0.0, BLUE], [0.5, GARAGE], [1.0, YELLOW]]       # two-sided: one driver ← → the other
FONT = "Barlow, 'Segoe UI', sans-serif"
DISPLAY = "'Barlow Condensed', Barlow, 'Segoe UI', sans-serif"

CSS = f"""
<style>
.block-container {{ padding-top: 2.2rem; max-width: 1320px; }}
h1, h2, h3 {{ letter-spacing: .01em; }}
[data-testid="stMetricValue"] {{ font-family: {DISPLAY}; font-weight: 600; font-variant-numeric: tabular-nums; }}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; }}
[data-testid="stDataFrame"] {{ font-variant-numeric: tabular-nums; }}
button[kind="primary"], button[kind="primary"] p {{ color: {ASPHALT} !important; font-weight: 600; }}
.stTabs [data-baseweb="tab"] p {{ font-family: {DISPLAY}; font-size: 1.12rem; font-weight: 600; letter-spacing: .02em; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 1.4rem; border-bottom: 1px solid {LINE}; }}

.pw-head {{ display: flex; align-items: baseline; gap: 1.1rem; flex-wrap: wrap; margin-bottom: .2rem; }}
.pw-title {{ font-family: {DISPLAY}; font-weight: 700; font-size: 2.5rem; line-height: 1; letter-spacing: .01em; }}
.pw-title span {{ color: {YELLOW}; }}
.pw-lede {{ color: {MUTED}; max-width: 68ch; margin: .1rem 0 .7rem 0; }}
.pw-chips {{ display: flex; gap: .45rem; flex-wrap: wrap; margin: 0 0 1.1rem 0; }}
.pw-chip {{ border: 1px solid {LINE}; border-radius: 999px; padding: .12rem .7rem; font-size: .84rem; color: {MUTED};
           background: {GARAGE}; white-space: nowrap; }}
.pw-chip b {{ color: {PAINT}; font-weight: 600; }}
.pw-chip.ok b {{ color: {GREEN}; }} .pw-chip.warn b {{ color: {YELLOW}; }} .pw-chip.bad b {{ color: {RED}; }}

/* The pit board: the call, as the driver reads it from the car. */
.pw-board {{ background: #0B1016; border: 1px solid {LINE}; border-left: .55rem solid var(--tone, {YELLOW});
            border-radius: .3rem; padding: .85rem 1.2rem .8rem 1.1rem; margin: .5rem 0 1rem 0; }}
.pw-board .call {{ font-family: {DISPLAY}; font-weight: 700; font-size: clamp(1.9rem, 4.2vw, 3.1rem); line-height: 1.02;
                  letter-spacing: .03em; color: var(--tone, {YELLOW}); text-transform: uppercase; }}
.pw-board .why {{ color: {PAINT}; margin-top: .3rem; font-size: 1.02rem; }}
.pw-board .ctx {{ color: {MUTED}; margin-top: .15rem; font-size: .9rem; }}
.pw-note {{ color: {MUTED}; font-size: .9rem; border-left: 2px solid {LINE}; padding: .1rem 0 .1rem .8rem; margin: .4rem 0 1rem 0;
           max-width: 86ch; }}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header(title: str, lede: str, chips: list[tuple[str, str, str]]) -> None:
    """Page header. chips: (label, value, tone) with tone in '', 'ok', 'warn', 'bad'."""
    first, _, rest = title.partition(" ")
    chip_html = "".join(f"<span class='pw-chip {tone}'>{html.escape(k)} <b>{html.escape(v)}</b></span>"
                        for k, v, tone in chips)
    st.markdown(f"<div class='pw-head'><div class='pw-title'>{html.escape(first)} <span>{html.escape(rest)}</span></div></div>"
                f"<div class='pw-lede'>{html.escape(lede)}</div><div class='pw-chips'>{chip_html}</div>",
                unsafe_allow_html=True)


def pit_board(call: str, why: str = "", context: str = "", tone: str = YELLOW) -> None:
    """The verdict of a tool, pit-board style. tone: YELLOW (act), PAINT (hold), GREEN / RED."""
    parts = [f"<div class='call'>{html.escape(call)}</div>"]
    if why:
        parts.append(f"<div class='why'>{html.escape(why)}</div>")
    if context:
        parts.append(f"<div class='ctx'>{html.escape(context)}</div>")
    st.markdown(f"<div class='pw-board' style='--tone:{tone}'>{''.join(parts)}</div>", unsafe_allow_html=True)


def note(text: str) -> None:
    """A quiet aside: how to read what is above, or what the tool does not know."""
    st.markdown(f"<div class='pw-note'>{text}</div>", unsafe_allow_html=True)


def style(fig: go.Figure, title: str = "", height: int = 380, **kw) -> go.Figure:
    """The shared chart look: transparent on the page, quiet grid, legend above, bundled font."""
    fig.update_layout(
        title=dict(text=title, x=0, y=0.98, yanchor="top", font=dict(size=16, family=DISPLAY, color=PAINT)),
        height=height, margin=dict(l=64, r=18, t=86 if title else 36, b=54),
        font=dict(family=FONT, color=PAINT, size=13), colorway=SERIES,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hovermode=kw.pop("hovermode", "x unified"),
        hoverlabel=dict(bgcolor=GARAGE, bordercolor=LINE, font=dict(family=FONT, color=PAINT)),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, font=dict(color=MUTED)), **kw)
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=LINE, tickfont=dict(color=MUTED),
                     title_font=dict(color=MUTED), automargin=True)
    fig.update_yaxes(gridcolor="rgba(237,241,244,0.07)", zeroline=False, tickfont=dict(color=MUTED),
                     title_font=dict(color=MUTED), automargin=True)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False})
