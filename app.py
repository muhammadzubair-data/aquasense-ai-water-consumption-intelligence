"""AquaSense AI — Smart Water Consumption, Potential Leak Intelligence & Demand Forecasting (Streamlit).

Synthetic demonstration data by default. Run: streamlit run app.py
"""
from __future__ import annotations

import hashlib
import html
import io
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from aquasense.core import (MIN_MODEL_HOURS, NIGHT_HOURS, REQUIRED, TOTAL, add_anomalies, daily, detect_anomalies,
                            episode_scores, evaluate_forecast, flag_scores, forecast_summary, generate_data,
                            hourly_profile, kpis, leak_signals, leak_windows, meter_summary, night_flow, peak_periods,
                            potential_leaks, prepare, weekday_hour)

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
WEBSITE = "https://zubairlabs.com"
REPO = "https://github.com/muhammadzubair-data/aquasense-ai-water-consumption-intelligence"
NAVY, AQUA, TURQ, EMERALD, INDIGO, AMBER, ROSE, SLATE = ("#071A33", "#0EA5E9", "#14B8A6", "#10B981", "#6366F1",
                                                        "#F59E0B", "#F43F5E", "#94A3B8")
SEQ = [AQUA, TURQ, EMERALD, INDIGO, AMBER, ROSE, "#22D3EE", "#84CC16", "#A855F7", "#F97316", "#64748B", "#EC4899",
       "#0284C7", "#0D9488", "#15803D", "#4338CA"]
WATER_SCALE = [[0, "#E0F7FA"], [.35, "#5EEAD4"], [.7, "#0EA5E9"], [1, "#1E3A8A"]]
MAX_UPLOAD_MB = 25

st.set_page_config(page_title="AquaSense AI | ZubairLabs", page_icon=str(ASSETS / "icon.svg"), layout="wide",
                   initial_sidebar_state="auto",
                   menu_items={"About": "AquaSense AI by ZubairLabs — water consumption analytics on synthetic data. "
                                        "Anomalies and potential-leak indicators are not confirmed leaks.",
                               "Report a bug": REPO + "/issues"})

WAVE = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1200 120' preserveAspectRatio='none'>"
        "<path d='M0 60 C150 110 350 10 600 60 C850 110 1050 10 1200 60 L1200 120 L0 120 Z' fill='%2322D3EE' fill-opacity='0.16'/>"
        "<path d='M0 80 C200 40 400 120 600 80 C800 40 1000 120 1200 80 L1200 120 L0 120 Z' fill='%2310B981' fill-opacity='0.14'/></svg>")
st.markdown(f"""<style>
.block-container {{max-width: 1440px; padding-top: 3.2rem; padding-bottom: 3rem;}}
.aq-hero {{position: relative; overflow: hidden; padding: 24px 28px 34px 28px; border-radius: 20px; margin-bottom: 16px; color: #F8FAFC;
  background: url("{WAVE}") bottom/100% 64px no-repeat,
              radial-gradient(760px 260px at 0% 0%, rgba(34,211,238,.30), transparent 60%),
              radial-gradient(700px 300px at 100% 100%, rgba(16,185,129,.28), transparent 60%),
              linear-gradient(120deg, #04101F 0%, #071A33 55%, #0B2D4A 100%);
  border: 1px solid rgba(148,163,184,.22); box-shadow: 0 12px 32px rgba(2,8,23,.22);}}
.aq-hero .eyebrow {{letter-spacing: .2em; font-size: 11px; font-weight: 700; color: #22D3EE; text-transform: uppercase;}}
.aq-hero h1 {{font-size: 32px; margin: 6px 0 4px 0; padding: 0; color: #F8FAFC; font-weight: 800;}}
.aq-hero p {{margin: 0; color: #CBD5E1; max-width: 1000px;}}
.pill {{display: inline-block; margin: 10px 6px 0 0; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 600;
  background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.18); color: #E2E8F0;}}
.pill.warn {{background: rgba(245,158,11,.18); border-color: rgba(245,158,11,.5); color: #FDE68A;}}
.pill.ok {{background: rgba(16,185,129,.18); border-color: rgba(16,185,129,.5); color: #A7F3D0;}}
.pill.aqua {{background: rgba(14,165,233,.2); border-color: rgba(14,165,233,.55); color: #BAE6FD;}}
.kpi-grid {{display: grid; grid-template-columns: repeat(auto-fit, minmax(132px, 1fr)); gap: 12px; margin: 6px 0 14px 0;}}
.kpi {{border-radius: 16px; padding: 14px 16px; border: 1px solid rgba(148,163,184,.28); position: relative; overflow: hidden;
  background: linear-gradient(160deg, rgba(14,165,233,.10), rgba(20,184,166,.05) 55%, rgba(16,185,129,.08));}}
.kpi:before {{content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
  background: linear-gradient(180deg, #22D3EE, {TURQ}, {EMERALD});}}
.kpi .label {{font-size: 11.5px; font-weight: 700; opacity: .7; text-transform: uppercase; letter-spacing: .06em;}}
.kpi .value {{font-size: 26px; font-weight: 800; margin-top: 4px; line-height: 1.1;}}
.kpi .sub {{font-size: 12px; opacity: .65; margin-top: 4px;}}
.note {{border-left: 4px solid {AQUA}; padding: 10px 14px; border-radius: 10px; background: rgba(14,165,233,.09); margin: 6px 0 12px 0;}}
.note.warn {{border-left-color: {AMBER}; background: rgba(245,158,11,.10);}}
.note.ok {{border-left-color: {EMERALD}; background: rgba(16,185,129,.09);}}
.card {{border-radius: 14px; padding: 12px 16px; margin-bottom: 10px; border: 1px solid rgba(148,163,184,.28);
  background: linear-gradient(90deg, rgba(14,165,233,.10), rgba(16,185,129,.05));}}
.card b {{font-size: 15px;}}
.card .why {{opacity: .78; font-size: 13.2px; margin-top: 3px;}}
.footer {{opacity: .6; font-size: 12px; text-align: center; margin-top: 28px;}}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def esc(s) -> str:
    return html.escape(str(s))


def kpi_row(items):
    st.markdown("<div class='kpi-grid'>" + "".join(
        f"<div class='kpi'><div class='label'>{esc(a)}</div><div class='value'>{esc(b)}</div><div class='sub'>{esc(c)}</div></div>"
        for a, b, c in items) + "</div>", unsafe_allow_html=True)


def note(text, kind=""):
    st.markdown(f"<div class='note {kind}'>{text}</div>", unsafe_allow_html=True)


def chart(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="", title_font=dict(size=15))
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def csv_bytes(df: pd.DataFrame) -> bytes:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object or str(out[c].dtype).startswith(("string", "str", "category")):
            out[c] = out[c].astype(str).map(lambda v: "'" + v if v[:1] in ("=", "+", "-", "@") else v)
    return out.to_csv(index=False).encode()


def rnd(df: pd.DataFrame, n: int = 2) -> pd.DataFrame:
    num = df.select_dtypes("number").columns
    return df.assign(**{c: df[c].round(n) for c in num})


def m3(liters: float) -> str:
    return f"{liters / 1000:,.1f} m³"


def lph(v: float) -> str:
    return f"{v:,.1f} L/h"


PCT = st.column_config.ProgressColumn


# ------------------------------------------------------------------ data source and caching
ss = st.session_state


@st.cache_data(show_spinner=False, max_entries=6)
def load_demo(profile: str, days: int, meters: int, seed: int) -> pd.DataFrame:
    return generate_data(days, meters, seed, truth=True, profile=profile)


@st.cache_data(show_spinner=False, max_entries=8, ttl=3600)
def demo_anomalies(key: tuple, contamination: float, method: str) -> pd.DataFrame:
    d = load_demo(*key)[REQUIRED]
    return detect_anomalies(d, contamination) if method == "context" else _original_anoms(d, contamination)


@st.cache_data(show_spinner=False, max_entries=8, ttl=3600)
def demo_leaks(key: tuple, window: int, ratio: float, min_excess: float, min_hours: int):
    return potential_leaks(load_demo(*key)[REQUIRED], window, ratio, min_excess, min_hours)


@st.cache_resource(show_spinner=False, max_entries=16, ttl=3600)
def demo_forecast(key: tuple, meter: str, horizon: int):
    return evaluate_forecast(load_demo(*key)[REQUIRED], meter, horizon)


def _original_anoms(d: pd.DataFrame, contamination: float) -> pd.DataFrame:
    a = add_anomalies(d, contamination)
    a["expected"] = np.nan
    a["excess"] = np.nan
    a["direction"] = "—"
    return a[["timestamp", "meter_id", "liters", "expected", "excess", "anomaly", "anomaly_score", "direction"]]


def session_cached(name: str, args: tuple, fn):
    """Results for uploaded data live in this browser session only (bounded), never in a shared cache."""
    store = ss.setdefault("upload_results", OrderedDict())
    key = (ss.get("upload_key"), name, args)
    if key not in store:
        store[key] = fn()
        while len(store) > 12:
            store.popitem(last=False)
    store.move_to_end(key)
    return store[key]


with st.sidebar:
    st.markdown("**Data source**")
    has_upload = "upload_df" in ss
    options = ["Synthetic demo"] + ([f"Uploaded: {ss['upload_name']}"] if has_upload else [])
    use_upload = bool(has_upload and ss.get("use_upload"))
    ss["source_choice"] = options[-1] if use_upload else options[0]

    def _pick_source():
        ss["use_upload"] = ss["source_choice"] != "Synthetic demo"

    st.radio("Data source", options, key="source_choice", on_change=_pick_source, label_visibility="collapsed")
    if not use_upload:
        prof_label = st.selectbox("Usage profile", ["Realistic residential", "Original (project v1)"],
                                  help="Realistic: low night flow, morning/evening peaks, weekend shift and short "
                                       "legitimate high-use bursts. Original: the first version's generator, unchanged.")
        c1, c2 = st.columns(2)
        days = c1.select_slider("Days", [30, 60, 90, 180], value=90)
        n_meters = c2.select_slider("Meters", [4, 8, 12, 16], value=8)
        seed = st.number_input("Seed", 0, 9999, 42, help="Same seed, same data and results.")
    else:
        st.caption("Analysing your uploaded file in this session only.")

if use_upload:
    data = ss["upload_df"]
    truth = None
    source_label, is_demo = ss["upload_name"], False
    demo_key = None
else:
    profile = "realistic" if prof_label.startswith("Realistic") else "original"
    demo_key = (profile, int(days), int(n_meters), int(seed))
    full = load_demo(*demo_key)
    data, truth = full[REQUIRED], full
    source_label = f"Synthetic demo · {profile} profile · {days} days · {n_meters} meters · seed {seed}"
    is_demo = True

meters_all = sorted(data.meter_id.unique())
model_meters = [m for m in meters_all if (data.meter_id == m).sum() >= MIN_MODEL_HOURS]
with st.sidebar:
    focus_opts = [TOTAL] + model_meters
    if ss.get("focus_meter") not in focus_opts:
        ss["focus_meter"] = model_meters[0] if model_meters else TOTAL
    st.selectbox("Focus meter", focus_opts, key="focus_meter",
                 help="Used by the anomaly, leak and forecasting views. 'ALL METERS' is the network total.")
    with st.expander("About these results", icon="💧"):
        st.markdown("- Data is **synthetic** unless you upload your own.\n"
                    "- Anomalies are **unusual readings**, not leaks.\n"
                    "- Potential-leak indicators are **signals to check**, not confirmed leaks.\n"
                    "- Nothing here is validated for real infrastructure decisions.")
    st.caption("Theme: ⋮ menu (top right) → Light, Dark or System")
    st.caption(f"[ZubairLabs]({WEBSITE}) · [Source code]({REPO})")
focus = ss["focus_meter"]


def get_anomalies(contamination: float, method: str = "context") -> pd.DataFrame:
    if is_demo:
        return demo_anomalies(demo_key, contamination, method)
    return session_cached("anoms", (contamination, method), lambda: detect_anomalies(data, contamination)
                          if method == "context" else _original_anoms(data, contamination))


def get_leaks(window=12, ratio=1.35, min_excess=8.0, min_hours=6):
    if is_demo:
        return demo_leaks(demo_key, window, ratio, min_excess, min_hours)
    return session_cached("leaks", (window, ratio, min_excess, min_hours),
                          lambda: potential_leaks(data, window, ratio, min_excess, min_hours))


def get_forecast(meter: str, horizon: int = 24):
    with st.spinner(f"Training and back-testing the forecast for {meter}…"):
        if is_demo:
            return demo_forecast(demo_key, meter, horizon)
        return session_cached("fc", (meter, horizon), lambda: evaluate_forecast(data, meter, horizon))


def hero(eyebrow, title, text):
    pills = ("<span class='pill warn'>Synthetic demonstration data</span>" if is_demo
             else "<span class='pill ok'>Your uploaded file · this session only</span>")
    pills += "<span class='pill aqua'>Signals, not confirmed leaks</span><span class='pill'>Not validated for operations</span>"
    st.markdown(f"<div class='aq-hero'><div class='eyebrow'>{esc(eyebrow)}</div><h1>{esc(title)}</h1>"
                f"<p>{esc(text)}</p>{pills}</div>", unsafe_allow_html=True)


def footer():
    st.markdown(f"<div class='footer'>AquaSense AI · built by <a href='{WEBSITE}' target='_blank'>ZubairLabs</a> · "
                f"{esc(source_label)}</div>", unsafe_allow_html=True)


def meter_series_chart(d: pd.DataFrame, meter: str, title: str, days_back: int | None = None):
    s = (d.groupby("timestamp")["liters"].sum() if meter == TOTAL else d[d.meter_id == meter].set_index("timestamp")["liters"])
    if days_back:
        s = s[s.index > s.index.max() - pd.Timedelta(days=days_back)]
    fig = go.Figure(go.Scatter(x=s.index, y=s.values, mode="lines", line=dict(color=AQUA, width=1.4), name="Observed"))
    fig.update_layout(title=title, yaxis_title="L/h")
    return fig


# ------------------------------------------------------------------ page 1: command center
def page_command():
    hero("ZubairLabs · Smart utilities", "Water Intelligence Command Center",
         "Network consumption, meter statistics and the signals that deserve a closer look, in one place.")
    k = kpis(data)
    _, ep = get_leaks()
    an = get_anomalies(0.01)
    higher = ep[ep.priority == "Higher"]
    kpi_row([("Total consumption", m3(k["total_liters"]), f"{k['days']:.0f} days · {k['readings']:,} readings"),
             ("Average per day", m3(k["mean_daily_liters"]), f"peak {m3(k['peak_daily_liters'])} on {k['peak_day']}"),
             ("Meters", f"{k['meters']}", f"mean {lph(k['mean_lph'])} per meter"),
             ("Peak hour", f"{k['peak_hour']:02d}:00", "network average"),
             ("Night flow 02–05h", lph(k["night_lph"]), f"{k['night_share']:.0%} of average flow"),
             ("Potential-leak indicators", f"{len(higher)} higher",
              f"{len(ep) - len(higher)} lower · {int(an.anomaly.sum()):,} unusual readings")])
    c1, c2 = st.columns([1.5, 1])
    with c1:
        dd = daily(data)
        dd["rolling7"] = dd.liters.rolling(7, min_periods=1).mean()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=dd.date, y=dd.liters / 1000, name="Daily total", marker_color="rgba(14,165,233,.45)"))
        fig.add_trace(go.Scatter(x=dd.date, y=dd.rolling7 / 1000, name="7-day average", line=dict(color=EMERALD, width=3)))
        for r in higher.itertuples():
            fig.add_vrect(x0=r.start, x1=r.end, fillcolor=ROSE, opacity=.12, line_width=0)
        fig.update_layout(title="Daily network consumption (m³) · shaded = higher-priority potential leak",
                          yaxis_title="m³", legend=dict(orientation="h", y=-0.15))
        chart(fig)
    with c2:
        ms = meter_summary(data)
        flagged_m = set(higher.meter_id)
        ms["status"] = np.where(ms.meter_id.isin(flagged_m), "Higher-priority indicator", "No higher-priority indicator")
        fig = px.bar(ms.sort_values("total_liters"), x="total_liters", y="meter_id", orientation="h", color="status",
                     color_discrete_map={"Higher-priority indicator": ROSE, "No higher-priority indicator": AQUA},
                     labels={"total_liters": "Litres", "meter_id": ""}, title="Consumption by meter")
        fig.update_layout(legend=dict(orientation="h", y=-0.2))
        chart(fig)
    c3, c4 = st.columns([1, 1])
    with c3:
        hp = hourly_profile(data.groupby("timestamp", as_index=False)["liters"].sum().assign(meter_id="net"), by="day_type")
        fig = px.line(hp, x="hour", y="mean", color="day_type", markers=True, color_discrete_sequence=[AQUA, AMBER],
                      title="Average network flow by hour", labels={"mean": "L/h", "hour": "Hour", "day_type": ""})
        fig.update_xaxes(dtick=2)
        chart(fig, 340)
    with c4:
        st.markdown("#### Needs a closer look")
        if len(higher):
            for r in higher.sort_values("excess_liters", ascending=False).head(4).itertuples():
                st.markdown(f"<div class='card'><b>{esc(r.meter_id)} · potential leak indicator (higher priority)</b>"
                            f"<div class='why'>{r.start:%d %b %H:%M} → {r.end:%d %b %H:%M} · {r.hours} h above normal · "
                            f"≈ {r.excess_liters:,.0f} L extra · night flow raised in {r.night_hours_elevated} of the "
                            f"night hours checked</div></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='card'><b>No higher-priority indicators</b><div class='why'>No sustained rise with "
                        "raised night flow at the current settings.</div></div>", unsafe_allow_html=True)
        top = an[an.anomaly].groupby("meter_id").size().sort_values(ascending=False).head(2)
        for m, n_ in top.items():
            st.markdown(f"<div class='card'><b>{esc(m)} · {n_} unusual readings</b><div class='why'>Short-lived "
                        "departures from this meter's normal pattern. Often legitimate use.</div></div>",
                        unsafe_allow_html=True)
    note("Indicators and anomalies are statistical signals. Confirming a leak needs a site visit, a step test or "
         "pressure data — none of which this demo has.", "warn")
    footer()


# ------------------------------------------------------------------ page 2: consumption analytics
def page_consumption():
    hero("Consumption analytics", "Consumption Analytics",
         "Hourly and daily usage, peak periods, weekly rhythm and meter comparisons.")
    c1, c2 = st.columns([2, 1.3])
    sel = c1.multiselect("Meters", meters_all, placeholder="All meters")
    t0, t1 = data.timestamp.min().date(), data.timestamp.max().date()
    rng = c2.date_input("Date range", (t0, t1), min_value=t0, max_value=t1)
    d = data[data.meter_id.isin(sel)] if sel else data
    if isinstance(rng, tuple) and len(rng) == 2:
        d = d[(d.timestamp.dt.date >= rng[0]) & (d.timestamp.dt.date <= rng[1])]
    if d.empty:
        st.info("No readings match these filters.", icon="🔎")
        return
    k = kpis(d)
    kpi_row([("Consumption", m3(k["total_liters"]), f"{k['readings']:,} readings"),
             ("Average day", m3(k["mean_daily_liters"]), f"busiest {k['peak_day']}"),
             ("Peak hour", f"{k['peak_hour']:02d}:00", "highest average flow"),
             ("Night flow", lph(k["night_lph"]), "02:00–04:59 average per meter")])
    dm = daily(d, by_meter=True)
    fig = px.area(dm, x="date", y="liters", color="meter_id", color_discrete_sequence=SEQ,
                  title="Daily consumption by meter (litres)", labels={"liters": "L", "date": "", "meter_id": ""})
    chart(fig, 360)
    c3, c4 = st.columns(2)
    with c3:
        hp = hourly_profile(d)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(hp.hour) + list(hp.hour)[::-1], y=list(hp.p90) + list(hp.p10)[::-1],
                                 fill="toself", fillcolor="rgba(20,184,166,.18)", line=dict(width=0), mode="lines",
                                 hoverinfo="skip", name="10th–90th percentile"))
        fig.add_trace(go.Scatter(x=hp.hour, y=hp["mean"], mode="lines+markers", line=dict(color=AQUA, width=3), name="Mean"))
        fig.update_layout(title="Hourly profile per meter-hour", xaxis=dict(title="Hour", dtick=2), yaxis_title="L/h",
                          legend=dict(orientation="h", y=-0.2))
        chart(fig)
    with c4:
        wh = weekday_hour(d)
        fig = go.Figure(go.Heatmap(z=wh.values, x=list(wh.columns), y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][:len(wh)],
                                   colorscale=WATER_SCALE, colorbar=dict(title="L/h"),
                                   hovertemplate="%{y} %{x}:00 → %{z:.0f} L/h<extra></extra>"))
        fig.update_layout(title="Weekly rhythm: average network flow", xaxis=dict(title="Hour", dtick=2))
        chart(fig)
    c5, c6 = st.columns([1, 1.2])
    with c5:
        st.markdown("**Top 10 network peak hours**")
        pk = peak_periods(d)
        pk["timestamp"] = pk.timestamp.dt.strftime("%a %d %b %Y %H:%M")
        st.dataframe(rnd(pk, 1), hide_index=True, width="stretch",
                     column_config={"liters": PCT("litres", min_value=0.0, max_value=float(pk.liters.max()), format="%.0f")})
    with c6:
        dt = daily(d).assign(weekday=lambda x: x.date.dt.day_name().str[:3])
        fig = px.box(dt, x="weekday", y="liters", category_orders={"weekday": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]},
                     color_discrete_sequence=[TURQ], title="Daily network total by weekday", labels={"liters": "L", "weekday": ""})
        chart(fig, 360)
    st.download_button("Download daily totals by meter (CSV)", csv_bytes(rnd(dm, 2)), "aquasense_daily_by_meter.csv",
                       "text/csv", icon="⬇️")
    footer()


# ------------------------------------------------------------------ page 3: anomalies
def page_anomalies():
    hero("Anomaly detection", "Anomaly Detection",
         "Readings that depart from each meter's normal pattern for that hour, found with an Isolation Forest. "
         "Unusual is not the same as a leak: a garden hose or a filled tank looks the same.")
    c1, c2, c3 = st.columns([1.3, 1, 1])
    method = c1.radio("Method", ["Context-aware (recommended)", "Original v1 (raw reading, hour, weekday)"],
                      help="Context-aware compares each reading with the same hour on earlier days of the same type.")
    rate = c2.select_slider("Expected anomaly rate", [0.005, 0.01, 0.02, 0.03, 0.05], value=0.01,
                            format_func=lambda v: f"{v:.1%}", help="Isolation Forest contamination: an assumption, not a finding.")
    dirn = c3.selectbox("Show", ["All unusual readings", "Higher than usual", "Lower than usual"])
    mkey = "context" if method.startswith("Context") else "original"
    an = get_anomalies(rate, mkey)
    flagged = an[an.anomaly]
    if dirn != "All unusual readings" and mkey == "context":
        flagged = flagged[flagged.direction == dirn.lower()]
    kpi_row([("Unusual readings", f"{len(flagged):,}", f"of {len(an):,} scored"),
             ("Meters affected", f"{flagged.meter_id.nunique()}", f"of {len(meters_all)}"),
             ("Highest score", f"{flagged.anomaly_score.max():.3f}" if len(flagged) else "—", "Isolation Forest"),
             ("Extra volume", f"{flagged.excess.clip(lower=0).sum():,.0f} L" if mkey == "context" else "—",
              "above typical, flagged hours")])
    m = focus if focus != TOTAL else (flagged.meter_id.value_counts().index[0] if len(flagged) else meters_all[0])
    if focus == TOTAL:
        st.caption(f"Network total has no per-meter model; showing {m}. Pick a meter in the sidebar to change it.")
    part = an[an.meter_id == m]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=part.timestamp, y=part.liters, mode="lines", line=dict(color=AQUA, width=1.1), name="Observed"))
    if mkey == "context":
        fig.add_trace(go.Scatter(x=part.timestamp, y=part.expected, mode="lines", line=dict(color=SLATE, width=1, dash="dot"),
                                 name="Typical for the hour"))
    fp = flagged[flagged.meter_id == m]
    fig.add_trace(go.Scatter(x=fp.timestamp, y=fp.liters, mode="markers", name="Unusual reading",
                             marker=dict(color=ROSE, size=9, line=dict(color="white", width=1))))
    last = part.timestamp.max()
    fig.update_layout(title=f"{m}: readings and unusual points (drag the slider to change the window)", yaxis_title="L/h",
                      legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
                      xaxis=dict(range=[last - pd.Timedelta(days=21), last], rangeslider=dict(visible=True, thickness=.08)))
    chart(fig, 440)
    c4, c5 = st.columns(2)
    with c4:
        if mkey == "context":
            vol = flagged.groupby("meter_id").excess.apply(lambda e: e.clip(lower=0).sum()).reindex(meters_all, fill_value=0)
            fig = px.bar(x=vol.index, y=vol.values, color_discrete_sequence=[TURQ],
                         title="Extra volume in unusual readings, per meter (L)", labels={"x": "", "y": "L"})
        else:
            cnt = flagged.groupby("meter_id").size().reindex(meters_all, fill_value=0)
            fig = px.bar(x=cnt.index, y=cnt.values, color_discrete_sequence=[TURQ], title="Unusual readings per meter",
                         labels={"x": "", "y": ""})
        chart(fig, 320)
        st.caption("The model flags the same share of readings on every meter (the expected rate), so counts are "
                   "equal by design; the volume involved is what differs.")
    with c5:
        hh = flagged.groupby(flagged.timestamp.dt.hour).size().reindex(range(24), fill_value=0)
        fig = px.bar(x=hh.index, y=hh.values, color_discrete_sequence=[INDIGO], title="When they happen (hour of day)",
                     labels={"x": "Hour", "y": ""})
        chart(fig, 320)
    st.markdown("**Unusual readings, highest score first**")
    show = flagged.sort_values("anomaly_score", ascending=False).head(300).copy()
    if mkey == "context":
        show["explanation"] = [f"{abs(e):.1f} L/h {d_} (typical {x:.1f})" for e, d_, x in zip(show.excess, show.direction, show.expected)]
    st.dataframe(rnd(show.drop(columns=[c for c in ["excess_3h"] if c in show]), 3), hide_index=True, width="stretch", height=300)
    if is_demo and truth is not None:
        with st.expander("Synthetic check: what did the flags catch?", icon="🧪"):
            st.markdown(synthetic_flag_table(an))
            st.caption("Bursts are legitimate short high-use events the generator added (realistic profile only). "
                       "Most flags land on bursts or random noise, not leaks — which is why flags alone can't confirm a leak.")
    st.download_button("Download scored readings (CSV)", csv_bytes(rnd(an, 4)), "aquasense_anomalies.csv", "text/csv", icon="⬇️")
    footer()


def synthetic_flag_table(an: pd.DataFrame) -> str:
    leak = flag_scores(an, truth, "anomaly")
    rows = [f"| Injected leak hours flagged | {leak['flagged_in_leak']} of {leak['leak_hours']} ({leak['recall']:.0%}) |"]
    if "injected_burst" in truth:
        b = flag_scores(an, truth.assign(injected_leak=truth.injected_burst), "anomaly")
        rows.append(f"| Legitimate burst hours flagged | {b['flagged_in_leak']} of {b['leak_hours']} ({b['recall']:.0%}) |")
        both = flag_scores(an, truth.assign(injected_leak=truth.injected_burst | truth.injected_leak), "anomaly")
        rows.append(f"| Flags that fall on a leak or burst | {both['flagged_in_leak']} of {both['flagged_hours']} ({both['precision']:.0%}) |")
    else:
        rows.append(f"| Flags that fall on a leak hour | {leak['flagged_in_leak']} of {leak['flagged_hours']} ({leak['precision']:.0%}) |")
    return "| Check | Result |\n|:--|:--|\n" + "\n".join(rows)


# ------------------------------------------------------------------ page 4: potential leaks
def page_leaks():
    hero("Potential leak intelligence", "Potential Leak Intelligence",
         "Sustained consumption above what is normal for each hour, grouped into episodes and ranked by whether "
         "night flow rose too. These are potential indicators to check on site, not confirmed leaks.")
    with st.expander("Indicator settings", expanded=False, icon="⚙️"):
        c1, c2, c3, c4 = st.columns(4)
        window = c1.slider("Rolling window (hours)", 6, 24, 12, 2)
        ratio = c2.slider("Above-normal ratio", 1.1, 2.5, 1.35, 0.05)
        min_excess = c3.slider("Minimum excess (L/h)", 2.0, 30.0, 8.0, 1.0)
        min_hours = c4.slider("Minimum episode length (h)", 2, 24, 6, 1)
    hourly, ep = get_leaks(window, ratio, min_excess, min_hours)
    higher = ep[ep.priority == "Higher"]
    kpi_row([("Episodes", f"{len(ep)}", f"{len(higher)} higher · {len(ep) - len(higher)} lower priority"),
             ("Meters affected", f"{ep.meter_id.nunique()}", f"of {len(meters_all)}"),
             ("Estimated extra volume", f"{ep.excess_liters.sum():,.0f} L", "all episodes, above typical"),
             ("Longest episode", f"{int(ep.hours.max())} h" if len(ep) else "—", "consecutive hours")])
    note("<b>Higher priority</b> = at least 12 hours above normal <i>and</i> raised night flow (02:00–05:00), the pattern "
         "a continuous leak leaves. <b>Lower priority</b> = daytime or evening rises that night flow does not support; "
         "often legitimate use. Neither is a confirmed leak.", "warn")
    if ep.empty:
        st.info("No sustained elevated consumption at these settings.", icon="💧")
    else:
        show = ep.copy()
        show["start"] = show.start.dt.strftime("%Y-%m-%d %H:%M")
        show["end"] = show.end.dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(rnd(show.sort_values(["priority", "excess_liters"], ascending=[True, False]), 2), hide_index=True,
                     width="stretch", column_config={"peak_ratio": st.column_config.NumberColumn("peak ratio", format="%.2f×")})
    opts = list(ep.episode) if len(ep) else []
    lab = {r.episode: f"#{r.episode} · {r.meter_id} · {r.start:%d %b %H:%M} · {r.priority}" for r in ep.itertuples()}
    default = int(higher.episode.iloc[0]) if len(higher) else (opts[0] if opts else None)
    pick = st.selectbox("Inspect episode", opts, index=opts.index(default) if default in opts else 0,
                        format_func=lambda k: lab[k]) if opts else None
    if pick is not None:
        r = ep[ep.episode == pick].iloc[0]
        h = hourly[(hourly.meter_id == r.meter_id) & (hourly.timestamp >= r.start - pd.Timedelta(days=3))
                   & (hourly.timestamp <= r.end + pd.Timedelta(days=2))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=h.timestamp, y=h.expected, name="Typical for the hour", mode="lines",
                                 line=dict(color=SLATE, dash="dot")))
        fig.add_trace(go.Scatter(x=h.timestamp, y=h.liters, name="Observed", mode="lines", line=dict(color=AQUA, width=1.6)))
        fig.add_trace(go.Scatter(x=h.timestamp, y=h.rolling_mean, name=f"{window}-h mean", mode="lines",
                                 line=dict(color=ROSE if r.priority == "Higher" else AMBER, width=3)))
        fig.add_vrect(x0=r.start, x1=r.end, fillcolor=ROSE if r.priority == "Higher" else AMBER, opacity=.12, line_width=0,
                      annotation_text=f"{r.priority} priority", annotation_position="top left")
        if is_demo and truth is not None:
            w = leak_windows(truth)
            for x in w[w.meter_id == r.meter_id].itertuples():
                fig.add_vrect(x0=x.start, x1=x.end, line=dict(color=EMERALD, dash="dash", width=2), fillcolor="rgba(0,0,0,0)",
                              annotation_text="injected leak (synthetic truth)", annotation_position="bottom left")
        fig.update_layout(title=f"Episode #{pick} · {r.meter_id}", yaxis_title="L/h", legend=dict(orientation="h", y=-0.15))
        chart(fig, 420)
    nf = night_flow(data)
    piv = nf.pivot_table(index="meter_id", columns="date", values="ratio")
    fig = go.Figure(go.Heatmap(z=piv.values, x=piv.columns, y=piv.index, colorscale=[[0, "#E0F7FA"], [.25, "#5EEAD4"],
                                                                                    [.5, "#0EA5E9"], [1, "#BE123C"]],
                               zmin=0.5, zmax=3, colorbar=dict(title="× typical"),
                               hovertemplate="%{y} %{x|%d %b}: night flow %{z:.2f}× typical<extra></extra>"))
    fig.update_layout(title="Night-flow check: 02:00–04:59 flow compared with each meter's typical night")
    chart(fig, 360)
    if is_demo and truth is not None:
        with st.expander("Synthetic check against injected leaks", expanded=True, icon="🧪"):
            s_all, s_hi = episode_scores(ep, truth, window), episode_scores(higher, truth, window)
            orig = leak_signals(data)
            o = flag_scores(orig, truth, "leak_signal")
            n_ = flag_scores(hourly, truth, "in_episode")
            st.markdown(
                "| Check | All episodes | Higher priority only |\n|:--|--:|--:|\n"
                f"| Injected leaks found | {s_all['leaks_found']} of {s_all['injected_leaks']} | {s_hi['leaks_found']} of {s_hi['injected_leaks']} |\n"
                f"| Episodes not matching a leak | {s_all['false_episodes']} | {s_hi['false_episodes']} |\n"
                f"| Mean delay after leak start | {s_all['mean_delay_hours']:.1f} h | {s_hi['mean_delay_hours']:.1f} h |\n\n"
                f"Hour level · this method: {n_['flagged_in_leak']} of {n_['flagged_hours']} flagged hours are leak hours "
                f"(precision {n_['precision']:.0%}, recall {n_['recall']:.0%}). Original v1 rule: precision "
                f"{o['precision']:.0%}, recall {o['recall']:.0%}.")
            st.caption("Synthetic leaks are clean step changes (+27 L/h for 40 hours). Real leaks grow, fluctuate and "
                       "hide in noise; these numbers are not real-world detection rates.")
    c1, c2 = st.columns(2)
    c1.download_button("Download episodes (CSV)", csv_bytes(rnd(ep, 3)), "aquasense_potential_leak_episodes.csv",
                       "text/csv", icon="⬇️", width="stretch")
    c2.download_button("Download hourly indicator values (CSV)", csv_bytes(rnd(hourly.drop(columns=["episode"]), 3)),
                       "aquasense_leak_indicator_hourly.csv", "text/csv", icon="⬇️", width="stretch")
    footer()


# ------------------------------------------------------------------ page 5: forecasting
def page_forecast():
    hero("Demand forecasting", "Water Demand Forecasting",
         "Short-term hourly forecasts from a Random Forest on lagged consumption, checked against simple baselines "
         "with a chronological, leakage-safe backtest.")
    c1, c2 = st.columns([1.2, 1])
    opts = [TOTAL] + model_meters
    m = c1.selectbox("Series", opts, index=opts.index(focus) if focus in opts else 0)
    horizon = c2.select_slider("Forecast horizon (hours)", [6, 12, 24, 48, 72], value=24)
    try:
        ev = get_forecast(m, int(horizon))
    except ValueError as exc:
        st.warning(str(exc), icon="⚠️")
        footer()
        return
    day = ev.day_ahead.set_index("method")
    one = ev.one_step.set_index("method")
    rf = day.iloc[0] if len(day) else None
    best_base = day.iloc[1:].MAE.min() if len(day) > 1 else np.nan
    best_name = day.iloc[1:].MAE.idxmin() if len(day) > 1 else "—"
    nxt = ev.future.forecast_liters.sum() if len(ev.future) else np.nan
    kpi_row([(f"Recursive MAE ({horizon} h)", f"{rf.MAE:.2f} L/h" if rf is not None else "—", f"RMSE {rf.RMSE:.2f}" if rf is not None else ""),
             ("Best simple baseline", f"{best_base:.2f} L/h", best_name.split(":")[0][:34]),
             ("Improvement", f"{(1 - rf.MAE / best_base):+.0%}" if rf is not None and best_base else "—",
              "lower error than baseline" if rf is not None and rf.MAE < best_base else "baseline is as good or better"),
             ("One-step MAE", f"{one.iloc[0].MAE:.2f} L/h", "real previous hour known"),
             (f"Next {horizon} h total", f"{nxt:,.0f} L" if not np.isnan(nxt) else "—", f"from {ev.meta['last_observed']:%d %b %H:%M}")])
    if rf is not None and rf.MAE >= best_base:
        note(f"On this series a simple baseline ({esc(best_name)}) is as accurate as the Random Forest. That usually means "
             "consumption follows a fixed daily pattern with random noise, and there is little more to learn.", "warn")
    hist = ev.history[ev.history.index > ev.history.index.max() - pd.Timedelta(days=7)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Observed (last 7 days)", line=dict(color=AQUA, width=1.6)))
    f = ev.future
    if len(f):
        fig.add_trace(go.Scatter(x=list(f.timestamp) + list(f.timestamp)[::-1], y=list(f.high) + list(f.low)[::-1],
                                 fill="toself", fillcolor="rgba(16,185,129,.18)", line=dict(width=0), mode="lines",
                                 name="10th–90th percentile of past errors", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=f.timestamp, y=f.forecast_liters, name="Forecast", line=dict(color=EMERALD, width=3)))
    fig.update_layout(title=f"{m}: observed and forecast hourly consumption", yaxis_title="L/h", legend=dict(orientation="h", y=-0.15))
    chart(fig, 420)
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Day-ahead backtest (recursive, same as the live forecast)**")
        st.dataframe(rnd(ev.day_ahead, 3), hide_index=True, width="stretch")
        st.markdown("**One step ahead (the original app's metric)**")
        st.dataframe(rnd(ev.one_step, 3), hide_index=True, width="stretch")
        st.caption(f"Train {ev.meta['train_hours']:,} h up to {ev.meta['train_end']:%d %b %H:%M} · holdout "
                   f"{ev.meta['test_hours']:,} h · {ev.meta['origins']} forecast origins at 23:00.")
    with c4:
        b = ev.by_horizon
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=b.horizon, y=b.MAE, name="Random Forest", mode="lines+markers", line=dict(color=EMERALD, width=3)))
        fig.add_trace(go.Scatter(x=b.horizon, y=b.MAE_naive_24h, name="Same hour, last observed day", mode="lines",
                                 line=dict(color=AMBER, dash="dash")))
        fig.update_layout(title="Error by hours ahead", xaxis_title="Hours ahead", yaxis_title="MAE (L/h)",
                          legend=dict(orientation="h", y=-0.2))
        chart(fig, 360)
    c5, c6 = st.columns(2)
    with c5:
        bt = ev.backtest
        if len(bt):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=bt.timestamp, y=bt.actual, name="Actual", line=dict(color=AQUA, width=1.3)))
            fig.add_trace(go.Scatter(x=bt.timestamp, y=bt.forecast, name="Day-ahead forecast", line=dict(color=EMERALD, width=1.3)))
            fig.update_layout(title="Holdout period: actual vs day-ahead forecasts", yaxis_title="L/h",
                              legend=dict(orientation="h", y=-0.2))
            chart(fig, 340)
    with c6:
        imp = ev.importance
        fig = px.bar(imp.sort_values("importance"), x="importance", y="label", orientation="h",
                     color_discrete_sequence=[TURQ], title="What the model relies on (impurity importance)",
                     labels={"importance": "", "label": ""})
        chart(fig, 340)
    c7, c8, c9 = st.columns(3)
    c7.download_button("Download forecast (CSV)", csv_bytes(rnd(ev.future, 2)), "aquasense_forecast.csv", "text/csv",
                       icon="⬇️", width="stretch")
    c8.download_button("Download backtest (CSV)", csv_bytes(rnd(ev.backtest, 3)), "aquasense_backtest.csv", "text/csv",
                       icon="⬇️", width="stretch")
    c9.download_button("Download metrics (JSON)", json.dumps(forecast_summary(ev), indent=2, default=str),
                       "aquasense_forecast_metrics.json", "application/json", icon="⬇️", width="stretch")
    footer()


# ------------------------------------------------------------------ page 6: meter intelligence
def page_meters():
    hero("Meter intelligence", "Meter Intelligence",
         "Compare usage patterns across meters: volume, daily shape, night flow, variability and recent change.")
    ms = meter_summary(data)
    _, ep = get_leaks()
    ms = ms.merge(ep.groupby("meter_id").agg(episodes=("episode", "size"),
                                             higher=("priority", lambda p: int((p == "Higher").sum()))),
                  on="meter_id", how="left").fillna({"episodes": 0, "higher": 0})
    hi = ms.sort_values("night_ratio").iloc[-1]
    kpi_row([("Meters", f"{len(ms)}", f"{ms.hours.sum():,} meter-hours"),
             ("Largest share", ms.iloc[0].meter_id, f"{ms.iloc[0].share:.1%} of volume"),
             ("Highest night ratio", hi.meter_id, f"{hi.night_ratio:.2f} × its average"),
             ("Biggest weekly change", ms.loc[ms.last7_vs_prev7.abs().idxmax()].meter_id if ms.last7_vs_prev7.notna().any() else "—",
              f"{ms.last7_vs_prev7.abs().max():+.1%} last 7 vs previous 7 days" if ms.last7_vs_prev7.notna().any() else "")])
    st.dataframe(rnd(ms, 3), hide_index=True, width="stretch",
                 column_config={"share": PCT("share", min_value=0.0, max_value=1.0, format="percent"),
                                "last7_vs_prev7": st.column_config.NumberColumn("last 7 vs prev 7", format="percent"),
                                "night_ratio": st.column_config.NumberColumn("night ratio", format="%.2f"),
                                "episodes": st.column_config.NumberColumn("leak episodes", format="%d"),
                                "higher": st.column_config.NumberColumn("higher priority", format="%d")})
    st.caption("Night ratio = average 02:00–04:59 flow ÷ the meter's overall average. Load factor = average ÷ peak "
               "hourly average. A rising night ratio is the classic first sign of a continuous leak.")
    sel = st.multiselect("Compare meters", meters_all, default=meters_all[:min(4, len(meters_all))], max_selections=8)
    c1, c2 = st.columns(2)
    if sel:
        d = data[data.meter_id.isin(sel)]
        prof = d.groupby(["meter_id", d.timestamp.dt.hour.rename("hour")])["liters"].mean().reset_index()
        prof["index"] = prof.liters / prof.groupby("meter_id").liters.transform("mean")
        with c1:
            fig = px.line(prof, x="hour", y="index", color="meter_id", color_discrete_sequence=SEQ, markers=True,
                          title="Daily shape (1.0 = the meter's own average)", labels={"index": "× average", "hour": "Hour", "meter_id": ""})
            fig.update_xaxes(dtick=2)
            chart(fig)
        with c2:
            dm = daily(d, by_meter=True)
            fig = px.line(dm, x="date", y="liters", color="meter_id", color_discrete_sequence=SEQ,
                          title="Daily totals", labels={"liters": "L", "date": "", "meter_id": ""})
            chart(fig)
    else:
        st.info("Pick at least one meter to compare.", icon="👆")
    c3, c4 = st.columns(2)
    with c3:
        fig = px.scatter(ms, x="mean_lph", y="night_ratio", size="total_liters", color="variability_cv", text="meter_id",
                         color_continuous_scale=WATER_SCALE, size_max=40,
                         title="Average flow vs night ratio (size = volume, colour = variability)",
                         labels={"mean_lph": "Average L/h", "night_ratio": "Night ratio", "variability_cv": "CV"})
        fig.update_traces(textposition="top center", cliponaxis=False)
        chart(fig)
    with c4:
        full = data.groupby(["meter_id", data.timestamp.dt.hour.rename("hour")])["liters"].mean().unstack()
        norm = full.div(full.mean(axis=1), axis=0)
        fig = go.Figure(go.Heatmap(z=norm.values, x=list(norm.columns), y=list(norm.index), colorscale=WATER_SCALE,
                                   colorbar=dict(title="× avg"), hovertemplate="%{y} %{x}:00 → %{z:.2f}× average<extra></extra>"))
        fig.update_layout(title="Usage fingerprint: every meter by hour", xaxis=dict(title="Hour", dtick=2))
        chart(fig)
    st.download_button("Download meter summary (CSV)", csv_bytes(rnd(ms, 4)), "aquasense_meter_summary.csv", "text/csv", icon="⬇️")
    footer()


# ------------------------------------------------------------------ page 7: data explorer
def page_data():
    hero("Data explorer", "Data Explorer",
         "Upload hourly water-meter readings, see exactly how they were cleaned, filter them and download the results. "
         "Uploads stay in this browser session and are never stored.")
    up = st.file_uploader("Water-meter CSV", type=["csv"], max_upload_size=MAX_UPLOAD_MB,
                          key=f"uploader_{ss.get('uploader_gen', 0)}",
                          help="Columns: timestamp, meter_id, liters (alternatives such as datetime, meter, litres, m3 are recognised).")
    if up is not None:
        raw_bytes = up.getvalue()
        key = hashlib.sha256(raw_bytes).hexdigest()[:16]
        if ss.get("upload_key") != key:
            try:
                if len(raw_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
                    raise ValueError(f"The file is larger than {MAX_UPLOAD_MB} MB.")
                raw = pd.read_csv(io.BytesIO(raw_bytes), low_memory=False)
                clean, rep = prepare(raw)
            except (ValueError, pd.errors.ParserError, UnicodeDecodeError, pd.errors.EmptyDataError) as exc:
                st.error(f"This file could not be used: {exc}", icon="⚠️")
            else:
                ss.update(upload_df=clean, upload_report=rep, upload_name=up.name, upload_key=key, use_upload=True)
                ss.pop("upload_results", None)
                st.rerun()
    if "upload_df" in ss:
        c1, c2 = st.columns([3, 1])
        c1.success(f"Using **{esc(ss['upload_name'])}** for every page in this session "
                   f"({len(ss['upload_df']):,} hourly readings, {ss['upload_df'].meter_id.nunique()} meters).", icon="✅")
        if c2.button("Remove upload", icon="🗑️", width="stretch"):
            for k in ("upload_df", "upload_report", "upload_name", "upload_key", "upload_results", "use_upload"):
                ss.pop(k, None)
            ss["uploader_gen"] = ss.get("uploader_gen", 0) + 1
            st.rerun()
        rep = ss["upload_report"]
        with st.expander("Validation report", expanded=not is_demo, icon="🧾"):
            st.table(rep.as_frame().set_index("check"))
            for n_ in rep.notes:
                st.caption("• " + n_)
    with st.expander("Upload instructions, data contract and samples", icon="📄"):
        st.markdown("""
| column | meaning | rules |
|:--|:--|:--|
| `timestamp` | start of the reading interval | any common date-time format; time zones are converted and dropped |
| `meter_id` | meter or device identifier | 1–64 letters, digits, spaces or `- _ . : / #` |
| `liters` | volume used in that interval | numeric, not negative; `m3` and `gallons` columns are converted |

- Hourly or more frequent readings. Sub-hourly readings are summed into hourly totals; daily data is not supported.
- Exact duplicates are removed; two different readings for the same meter and time are averaged (and counted).
- Gaps of up to 3 hours are interpolated; longer gaps stay missing and models skip them.
- At least 2 days per meter for analytics and 4 days (96 hours) for the models. Up to 200 meters, 500,000 rows, 25 MB.
- Do not upload customer names, addresses or account numbers.
""")
        s1, s2 = st.columns(2)
        sample = ROOT / "data" / "synthetic_water_meters.csv"
        if sample.exists():
            s1.download_button("Original sample (14 days, 4 meters)", sample.read_bytes(), "aquasense_sample_original.csv",
                               "text/csv", icon="⬇️", width="stretch")
        s2.download_button("Realistic sample (21 days, 4 meters)",
                           csv_bytes(generate_data(21, 4, 42, profile="realistic")), "aquasense_sample_realistic.csv",
                           "text/csv", icon="⬇️", width="stretch")
    st.markdown(f"#### Explore · {esc(source_label)}")
    f1, f2 = st.columns([2, 1.3])
    sel = f1.multiselect("Meters ", meters_all, placeholder="All meters")
    t0, t1 = data.timestamp.min().date(), data.timestamp.max().date()
    rng = f2.date_input("Dates", (t0, t1), min_value=t0, max_value=t1)
    d = data[data.meter_id.isin(sel)] if sel else data
    if isinstance(rng, tuple) and len(rng) == 2:
        d = d[(d.timestamp.dt.date >= rng[0]) & (d.timestamp.dt.date <= rng[1])]
    if d.empty:
        st.info("No readings match these filters.", icon="🔎")
        footer()
        return
    k = kpis(d)
    kpi_row([("Readings", f"{len(d):,}", f"{d.meter_id.nunique()} meters"), ("Volume", m3(k["total_liters"]), "filtered"),
             ("Mean", lph(k["mean_lph"]), "per meter-hour"), ("Period", f"{k['days']:.0f} days", f"{k['first']:%d %b} → {k['last']:%d %b}")])
    st.dataframe(rnd(d.head(1000), 2), hide_index=True, width="stretch", height=280)
    st.caption(f"Showing the first {min(1000, len(d)):,} of {len(d):,} readings. Downloads include all of them.")
    fig = meter_series_chart(d, TOTAL if not sel or len(sel) > 1 else sel[0], "Hourly consumption (filtered)")
    chart(fig, 320)
    dl = st.columns(4)
    dl[0].download_button("Filtered readings", csv_bytes(d), "aquasense_readings.csv", "text/csv", icon="⬇️", width="stretch")
    dl[1].download_button("Daily totals", csv_bytes(rnd(daily(d, by_meter=True), 2)), "aquasense_daily.csv", "text/csv",
                          icon="⬇️", width="stretch")
    dl[2].download_button("Meter summary", csv_bytes(rnd(meter_summary(d), 4)), "aquasense_meters.csv", "text/csv",
                          icon="⬇️", width="stretch")
    if "upload_report" in ss and not is_demo:
        dl[3].download_button("Validation report", csv_bytes(ss["upload_report"].as_frame()), "aquasense_validation.csv",
                              "text/csv", icon="⬇️", width="stretch")
    else:
        dl[3].button("Validation report", disabled=True, width="stretch", help="Available after an upload")
    footer()


# ------------------------------------------------------------------ page 8: methodology
def page_method():
    hero("Methodology & limitations", "Methodology & Limitations",
         "How each number is produced, what the synthetic data contains, how the models were evaluated and what this "
         "demo cannot tell you.")
    st.markdown("""
#### 1 · Data
- **Synthetic generator** (`generate_data`): hourly readings per meter with a random baseline of 18–42 L/h and Gaussian
  noise (σ = 3.2). Every fourth meter gets one **injected leak**: +27 L/h for 40 consecutive hours in the second half of the period.
- **Original profile** is the first version's generator, reproduced exactly. Its daily curve peaks around 01:00 and 13:00,
  so night flow equals average flow, which is unrealistic for homes.
- **Realistic residential profile** keeps the same random draws (same meters, leaks and noise) but applies a home-like day:
  low night flow, morning and evening peaks, later weekend mornings, and 1–3 short **legitimate bursts** per meter
  (e.g. irrigation, +35–60 L/h for 2–4 h). Bursts make it possible to test whether a method mistakes normal heavy use for a leak.
- **Uploads** go through `prepare()`: header mapping, unit conversion, timestamp parsing, meter-ID checks, duplicate handling,
  hourly resampling and gap handling, all reported in the validation report.

#### 2 · Consumption analytics
Totals, daily and hourly profiles, weekday × hour patterns, peak hours, and per-meter statistics (share, night ratio,
load factor, coefficient of variation, last 7 days vs the previous 7).

#### 3 · Anomaly detection (Isolation Forest)
- **Context-aware (default):** per meter, features are the reading, its gap from the *typical value for that hour*
  (median of the same hour on earlier days of the same type, past data only) and the 3-hour mean of that gap.
  `IsolationForest(n_estimators=200, contamination=rate, random_state=42)`.
- **Original v1:** reading, hour and weekday as raw features (kept for comparison).
- The contamination rate is an **assumption** about how many readings are unusual; the model always flags that share.

#### 4 · Potential-leak indicators
- Expected flow = typical value for the hour (as above). A reading is in a signal when the trailing *w*-hour mean exceeds
  the expected mean by a ratio **and** by a minimum number of L/h.
- Consecutive signal hours form an episode (one quiet hour bridged); short episodes are dropped.
- **Higher priority:** ≥ 12 h *and* raised night flow in at least two 02:00–04:59 hours, the signature of a continuous leak.
- Still only an indicator: a filled pool, guests or a broken appliance can look the same.

#### 5 · Forecasting
- Features (past data only, on a regular hourly grid): hour, weekday, previous hour, same hour yesterday, trailing 24-h mean.
- `RandomForestRegressor(n_estimators=100, min_samples_leaf=3, random_state=42)`.
- **Chronological holdout:** the last 20% of hours. The model is trained on the first 80% only.
- **One step ahead:** each holdout hour predicted with the real previous hour (the original metric, optimistic).
- **Recursive day-ahead backtest:** from every 23:00 in the holdout, forecast the next *h* hours feeding predictions back in,
  exactly like the live forecast. Compared with *same hour on the last observed day* and *hour-of-day average of the last 14 training days*.
- The live forecast uses a model refit on all hours; its band is the 10th–90th percentile of backtest errors per hours-ahead.
""")
    if is_demo and truth is not None:
        st.markdown("#### 6 · Synthetic results at the current settings")
        _, ep = get_leaks()
        s_all, s_hi = episode_scores(ep, truth), episode_scores(ep[ep.priority == "Higher"], truth)
        an = get_anomalies(0.01)
        a = flag_scores(an, truth, "anomaly")
        rows = [("Injected leaks", f"{s_all['injected_leaks']}"),
                ("Found by any episode", f"{s_all['leaks_found']} (mean delay {s_all['mean_delay_hours']:.1f} h)"),
                ("Found by higher-priority episodes", f"{s_hi['leaks_found']} with {s_hi['false_episodes']} non-matching"),
                ("Lower-priority episodes not matching a leak", f"{s_all['false_episodes'] - s_hi['false_episodes']}"),
                ("Leak hours among anomaly flags (1% rate)", f"{a['flagged_in_leak']} of {a['flagged_hours']} flags")]
        st.table(pd.DataFrame(rows, columns=["check", "result"]).set_index("check"))
    st.markdown("""
#### Limitations
- **Synthetic data only.** The leaks are clean steps added to a known pattern; results are not real-world detection or forecasting accuracy.
- **No confirmed leak labels, pressure, acoustic or district-meter data.** Confirming a leak needs field work.
- **Meters are independent.** No network topology, so leaks between a district meter and customer meters cannot be found.
- **Contamination and thresholds are assumptions** that trade missed events against false alarms.
- **Recursive forecasts compound errors**, and the band is empirical, not a calibrated probability.
- **Not validated** for billing, operations, safety or customer decisions.
""")
    footer()


PAGES = [st.Page(page_command, title="Water Intelligence Command Center", icon="💧", url_path="command", default=True),
         st.Page(page_consumption, title="Consumption Analytics", icon="📊", url_path="consumption"),
         st.Page(page_anomalies, title="Anomaly Detection", icon="🔍", url_path="anomalies"),
         st.Page(page_leaks, title="Potential Leak Intelligence", icon="🚰", url_path="leaks"),
         st.Page(page_forecast, title="Water Demand Forecasting", icon="📈", url_path="forecast"),
         st.Page(page_meters, title="Meter Intelligence", icon="🔢", url_path="meters"),
         st.Page(page_data, title="Data Explorer", icon="🗂️", url_path="data"),
         st.Page(page_method, title="Methodology & Limitations", icon="📘", url_path="method")]
st.navigation(PAGES).run()
