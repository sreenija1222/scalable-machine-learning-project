from __future__ import annotations
import os
import calendar
import json
from datetime import date
from typing import Any, Dict, Optional
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

if "HOPSWORKS_PROJECT" in st.secrets:
    os.environ["HOPSWORKS_PROJECT"] = str(st.secrets["HOPSWORKS_PROJECT"])
if "HOPSWORKS_API_KEY" in st.secrets:
    os.environ["HOPSWORKS_API_KEY"] = str(st.secrets["HOPSWORKS_API_KEY"])
if "HOPSWORKS_MODEL_VERSION" in st.secrets:
    os.environ["HOPSWORKS_MODEL_VERSION"] = str(st.secrets["HOPSWORKS_MODEL_VERSION"])

from inference_service import predict_one, get_input_schema
from storage  import upsert_entry, fetch_entries, fetch_entry_by_date, delete_entry



today = date.today()
st.set_page_config(page_title="Wellbeing Explorer", layout="wide")

#labels
MOOD_LABELS = {0: "Low", 1: "Medium", 2: "High"}
ENERGY_LABELS = {0: "Low", 1: "Medium", 2: "High"}

#low=1, med=2,high=3
LEVEL_MAP = {0: 1, 1: 2, 2: 3}

#calender coloring
PINK_SCALE = {
    0: "lightpink",
    1: "hotpink",
    2: "deeppink",
}

#values for the settings bars
ORDINAL_0_5_TEXT = {
    0: "0 – Not at all",
    1: "1 – Very little",
    2: "2 – Low",
    3: "3 – Moderate",
    4: "4 – High",
    5: "5 – Very high",
}

schema = get_input_schema() or {}
PHASE_OPTIONS = schema.get("phase_values", ["Menstruation", "Late Follicular", "Ovulation", "Luteal"])
SLEEP_MIN_MIN, SLEEP_MIN_MAX = schema.get("sleep_minutes_range", [0, 900])
HR_MIN, HR_MAX = schema.get("resting_hr_range", [35, 110])



#helpers
def date_key(d: Any) -> str:
    return pd.Timestamp(d).strftime("%Y-%m-%d")

def safe_int(x: Any, default: int = 0) -> int:
    if x is None:
        return default
    try:
        if isinstance(x, float) and pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default

def clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))

def build_payload(
    chosen_date: date,
    phase: str,
    sleep_minutes: int,
    resting_hr: int,
    stress_num: int,
    cramps_num: int,
    headaches_num: int,
    sleepissue_num: int,
    include_yesterday: bool,
    lag1_mood: Optional[int],
    lag1_energy: Optional[int],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "phase": phase,
        "is_weekend": int(pd.Timestamp(chosen_date).weekday() >= 5),
        "sleep_duration_minutes": int(sleep_minutes),
        "resting_heart_rate__value": int(resting_hr),
        "stress_num": int(stress_num),
        "cramps_num": int(cramps_num),
        "headaches_num": int(headaches_num),
        "sleepissue_num": int(sleepissue_num),
    }
    if include_yesterday and lag1_mood is not None and lag1_energy is not None:
        payload["lag1_mood"] = int(lag1_mood)
        payload["lag1_energy"] = int(lag1_energy)
    return payload

def month_grid(year: int, month: int):
    """Return list of weeks; each week is list of 7 day numbers (0 means blank)."""
    cal = calendar.Calendar(firstweekday=0) 
    return cal.monthdayscalendar(year, month)


# header
st.title("Wellbeing Explorer")
st.write("Log a day, see predicted mood & energy, and review patterns across your month.")
st.caption("Exploration only — not diagnosis or medical advice.")


# calender controls
st.markdown("### View controls")
control1, control2, control3, control4 = st.columns([1, 1, 1.2, 0.8])
today = date.today()
with control1:
    month = st.selectbox(
        "Month",
        list(range(1, 13)),
        index=today.month - 1,
        format_func=lambda m: calendar.month_name[m],
        key="month_select",
    )
with control2:
    year = st.selectbox(
        "Year",
        [today.year - 1, today.year, today.year + 1],
        index=1,
        key="year_select",
    )
with control3:
    target_view = st.selectbox(
        "Calendar dots show",
        ["Mood", "Energy"],
        index=0,
        key="calendar_target",
    )
with control4:
    show_debug = st.checkbox("Evaluate", value=False)

left, right = st.columns([1.2, 0.8], gap="large")




#loading saved enetries for selected month
today = date.today()
start_month = pd.Timestamp(year=year, month=month, day=1)
end_month = (start_month + pd.offsets.MonthEnd(1)).normalize()

load = fetch_entries(start_month.strftime("%Y-%m-%d"), end_month.strftime("%Y-%m-%d"))

if not load.empty:
    load["entry_date"] = pd.to_datetime(load["entry_date"])
    load = load.sort_values("entry_date")
    load["sleep_hours"] = (load["sleep_duration_minutes"] / 60.0).astype(float)
    if "stress_num" in load.columns:
        load["stress_num"] = pd.to_numeric(load["stress_num"], errors="coerce")
    load["mood_label"] = load["mood_pred"].map(MOOD_LABELS)
    load["energy_label"] = load["energy_pred"].map(ENERGY_LABELS)

    load["entry_date_display"] = load["entry_date"].dt.strftime("%Y-%m-%d")

    #levels 1,2,3
    load["mood_level"] = load["mood_pred"].map(LEVEL_MAP)
    load["energy_level"] = load["energy_pred"].map(LEVEL_MAP)

by_date = {}
if not load.empty:
    for _, r in load.iterrows():
        by_date[r["entry_date"].date()] = r.to_dict()

# calendar + trends
with left:
    #month calendar
    st.subheader("Calendar")
    st.caption("Dots appear only on days you saved. Darker pink = higher level.")

    weeks = month_grid(year, month)
    xs, ys, vals, texts = [], [], [], []
    day_numbers = []

    for w_i, week in enumerate(weeks):
        for d_i, day_num in enumerate(week):
            if day_num == 0:
                continue
            dt = date(year, month, day_num)
            row = by_date.get(dt)

            # only plot if saved
            if row is None:
                continue

            if target_view == "Mood":
                val = row.get("mood_pred", None)
                label = row.get("mood_label", None)
            else:
                val = row.get("energy_pred", None)
                label = row.get("energy_label", None)

            if val is None or (isinstance(val, float) and pd.isna(val)):
                continue

            val = int(val)
            xs.append(d_i)
            ys.append(w_i)
            vals.append(val)  
            day_numbers.append(day_num)

            sleep_h = row.get("sleep_duration_minutes", 0)
            sleep_h = round(float(sleep_h) / 60.0, 2) if sleep_h is not None else None
            stress = row.get("stress_num", None)

            hover = (
                f"<b>{dt.strftime('%Y-%m-%d')}</b>"
                f"<br>Phase: {row.get('phase', '—')}"
                f"<br>Predicted: {label}"
                f"<br>Sleep: {sleep_h}h"
                f"<br>Stress: {stress}"
            )
            texts.append(hover)

    PINKS_STEP = [
    [0.00, "lightpink"], [0.2499, "lightpink"],
    [0.25, "hotpink"],   [0.7499, "hotpink"],
    [0.75, "deeppink"],  [1.00, "deeppink"],
]
   

    fig_month = go.Figure()

    fig_month.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            marker=dict(
                size=18,
                color=vals,
                cmin=0,
                cmax=2,
                colorscale=PINKS_STEP,
                showscale=True,
                colorbar=dict(
                title="value",
                tickmode="array",
                tickvals=[0, 1, 2],
                ticktext=["low", "medium", "high"],
                thickness=18,
                len=0.78,
                x=0.92,
                xanchor="left",
                y=0.5,
                yanchor="middle",
            ),
            ),
            text=[str(n) for n in day_numbers],
            textposition="middle center",
            textfont=dict(size=10, color="white"),
            hovertext=texts,
            hovertemplate="%{hovertext}<extra></extra>",
            name="Saved days",
        )
    )

    fig_month.add_trace(
        go.Scatter(
            x=[0, 6],
            y=[0, max(0, len(weeks) - 1)],
            mode="markers",
            marker=dict(opacity=0),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig_month.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(7)),
            ticktext=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            range=[-0.5, 6.5],
            title="",
            showgrid=True,
            zeroline=False,
            domain=[0.0, 0.86],
        ),
        yaxis=dict(
            autorange="reversed",
            tickmode="array",
            tickvals=list(range(len(weeks))),
            ticktext=[f"Week {i+1}" for i in range(len(weeks))],
            range=[len(weeks) - 0.5, -0.5],
            title="",
            showgrid=True,
            zeroline=False,
        ),
        title=f"{calendar.month_name[month]} {year} — {target_view} dots",
    )

    st.plotly_chart(fig_month, use_container_width=True)


#trends
    st.subheader("Trends")
    if load.empty:
        st.info("Save a few days to see trends.")
    else:
        trends = load.copy()
        trends["entry_date"] = pd.to_datetime(trends["entry_date"])
        trends = trends.sort_values("entry_date")

        trends["sleep_hours"] = pd.to_numeric(trends["sleep_duration_minutes"], errors="coerce") / 60.0
        trends["stress_num"] = pd.to_numeric(trends["stress_num"], errors="coerce")
        trends["resting_hr"] = pd.to_numeric(trends["resting_heart_rate__value"], errors="coerce")

        LABEL_TO_LEVEL = {"Low": 1, "Medium": 2, "High": 3}
        PRED_TO_LEVEL = {0: 1, 1: 2, 2: 3}

        def to_level(pred_series: pd.Series, label_series: pd.Series) -> pd.Series:
            pred_num = pd.to_numeric(pred_series, errors="coerce")
            lvl = pred_num.map(PRED_TO_LEVEL)
            lab = label_series.astype(str).map(LABEL_TO_LEVEL)
            return lvl.fillna(lab)

        trends["mood_level"] = to_level(trends.get("mood_pred"), trends.get("mood_label"))
        trends["energy_level"] = to_level(trends.get("energy_pred"), trends.get("energy_label"))

        trends_mood = trends.dropna(subset=["mood_level"]).copy()
        trends_energy = trends.dropna(subset=["energy_level"]).copy()

        # sleep
        trends_sleep = trends.copy()
        trends_sleep["entry_date"] = pd.to_datetime(trends_sleep["entry_date"], errors="coerce")

        if "sleep_hours" not in trends_sleep.columns:
            trends_sleep["sleep_hours"] = pd.to_numeric(trends_sleep["sleep_duration_minutes"], errors="coerce") / 60.0
        else:
            trends_sleep["sleep_hours"] = pd.to_numeric(trends_sleep["sleep_hours"], errors="coerce")

        trends_sleep = trends_sleep.sort_values("entry_date").reset_index(drop=True)
        x_sl = trends_sleep["entry_date"].to_list()
        y_sl = trends_sleep["sleep_hours"].astype(float).to_list()

        fig_sleep = go.Figure()
        fig_sleep.add_trace(go.Scatter(
            x=x_sl,
            y=y_sl,
            mode="lines+markers",
            name="Sleep (hours)",
            connectgaps=False,
        ))
        fig_sleep.update_layout(height=260, title="Sleep (hours)", xaxis_title="", yaxis_title="Hours")
        fig_sleep.update_xaxes(tickformat="%Y-%m-%d")
        st.plotly_chart(fig_sleep, use_container_width=True)

        #stress
        trends_stress = trends.copy()
        trends_stress["entry_date"] = pd.to_datetime(trends_stress["entry_date"], errors="coerce")
        trends_stress["stress_num"] = pd.to_numeric(trends_stress["stress_num"], errors="coerce")
        trends_stress = trends_stress.sort_values("entry_date").reset_index(drop=True)

        x_s = trends_stress["entry_date"].to_list()
        y_s = trends_stress["stress_num"].astype(float).to_list()

        fig_stress = go.Figure()
        fig_stress.add_trace(go.Scatter(
            x=x_s,
            y=y_s,
            mode="lines+markers",
            name="Stress (0–5)",
            connectgaps=False,
        ))
        fig_stress.update_layout(height=260, title="Stress (0–5)", xaxis_title="", yaxis_title="Stress")
        fig_stress.update_yaxes(range=[-0.2, 5.2], tickmode="array", tickvals=[0,1,2,3,4,5])
        fig_stress.update_xaxes(tickformat="%Y-%m-%d")
        st.plotly_chart(fig_stress, use_container_width=True)


        #resting heart rate
        trends_heartrate = trends.copy()
        trends_heartrate["entry_date"] = pd.to_datetime(trends_heartrate["entry_date"], errors="coerce")
        trends_heartrate["resting_hr"] = pd.to_numeric(trends_heartrate["resting_heart_rate__value"], errors="coerce")
        trends_heartrate = trends_heartrate.sort_values("entry_date").reset_index(drop=True)

        x_hr = trends_heartrate["entry_date"].to_list()
        y_hr = trends_heartrate["resting_hr"].astype(float).to_list()

        fig_hr = go.Figure()
        fig_hr.add_trace(go.Scatter(
            x=x_hr,
            y=y_hr,
            mode="lines+markers",
            name="Resting HR (bpm)",
            connectgaps=False,
        ))
        fig_hr.update_layout(height=260, title="Resting heart rate (bpm)", xaxis_title="", yaxis_title="BPM")
        fig_hr.update_xaxes(tickformat="%Y-%m-%d")
        st.plotly_chart(fig_hr, use_container_width=True)


        #mood
        trends_mood = trends.dropna(subset=["mood_level"]).copy()
        trends_mood["entry_date"] = pd.to_datetime(trends_mood["entry_date"], errors="coerce")
        trends_mood = trends_mood.sort_values("entry_date").reset_index(drop=True)

        x_m = trends_mood["entry_date"].to_list()
        y_m = trends_mood["mood_level"].astype(float).to_list()

        fig_mood = go.Figure()
        fig_mood.add_trace(go.Scatter(
            x=x_m,
            y=y_m,
            mode="lines+markers",
            line=dict(shape="linear"),
            marker=dict(size=12, color="red", symbol="diamond"),
            name="Mood",
            connectgaps=False,
        ))
        fig_mood.update_layout(height=260, title="Mood level", xaxis_title="", yaxis_title="Level")
        fig_mood.update_yaxes(
            range=[0.75, 3.25],
            tickmode="array",
            tickvals=[1, 2, 3],
            ticktext=["Low", "Medium", "High"],
        )
        fig_mood.update_xaxes(tickformat="%Y-%m-%d")
        st.plotly_chart(fig_mood, use_container_width=True)


        #energy
        trends_energy = trends.dropna(subset=["energy_level"]).copy()
        trends_energy["entry_date"] = pd.to_datetime(trends_energy["entry_date"], errors="coerce")
        trends_energy = trends_energy.sort_values("entry_date").reset_index(drop=True)

        x_e = trends_energy["entry_date"].to_list()
        y_e = trends_energy["energy_level"].astype(float).to_list()

        fig_energy = go.Figure()
        fig_energy.add_trace(go.Scatter(
            x=x_e,
            y=y_e,
            mode="lines+markers",
            line=dict(shape="linear"),
            marker=dict(size=12, color="green", symbol="circle"),
            name="Energy",
            connectgaps=False,
        ))
        fig_energy.update_layout(height=260, title="Energy level", xaxis_title="", yaxis_title="Level")
        fig_energy.update_yaxes(
            range=[0.75, 3.25],
            tickmode="array",
            tickvals=[1, 2, 3],
            ticktext=["Low", "Medium", "High"],
        )
        fig_energy.update_xaxes(tickformat="%Y-%m-%d")
        st.plotly_chart(fig_energy, use_container_width=True)

#log/edit + live prediction
with right:
    st.subheader("Log a day (or edit a saved day)")
    st.caption("Pick any date. If it’s already saved, we’ll load it automatically. Predictions update live.")

    chosen_date = st.date_input("Date", value=today)
    key = date_key(chosen_date)
    existing = fetch_entry_by_date(key) or {}

    def pref(col: str, default: Any):
        v = existing.get(col, default)
        return default if (isinstance(v, float) and pd.isna(v)) else v

    phase_default = pref("phase", PHASE_OPTIONS[0])
    phase_idx = PHASE_OPTIONS.index(phase_default) if phase_default in PHASE_OPTIONS else 0
    phase = st.selectbox("Cycle phase", PHASE_OPTIONS, index=phase_idx)

    sleep_entry_mode = st.radio("Enter sleep as", ["Hours & minutes", "Minutes"], horizontal=True, key="sleep_mode")
    if sleep_entry_mode == "Hours & minutes":
        default_minutes = clamp(safe_int(pref("sleep_duration_minutes", 420)), int(SLEEP_MIN_MIN), int(SLEEP_MIN_MAX))
        default_h = default_minutes // 60
        default_m = default_minutes % 60
        sleep_h = st.number_input("Hours of sleep", min_value=0, max_value=24, value=int(default_h), step=1)
        sleep_m = st.number_input("Minutes", min_value=0, max_value=59, value=int(default_m), step=5)
        sleep_minutes = clamp(int(sleep_h) * 60 + int(sleep_m), int(SLEEP_MIN_MIN), int(SLEEP_MIN_MAX))
    else:
        sleep_minutes = st.slider(
            "Sleep duration (minutes)",
            int(SLEEP_MIN_MIN),
            int(SLEEP_MIN_MAX),
            value=clamp(safe_int(pref("sleep_duration_minutes", 420)), int(SLEEP_MIN_MIN), int(SLEEP_MIN_MAX)),
            step=5,
        )

    resting_hr = st.slider(
        "Resting heart rate (bpm)",
        int(HR_MIN),
        int(HR_MAX),
        value=clamp(safe_int(pref("resting_heart_rate__value", 62)), int(HR_MIN), int(HR_MAX)),
        step=1,
    )

    stress_num = st.select_slider("Stress", options=list(range(6)),
                                  value=clamp(safe_int(pref("stress_num", 2)), 0, 5),
                                  format_func=lambda x: ORDINAL_0_5_TEXT[x])
    cramps_num = st.select_slider("Cramps", options=list(range(6)),
                                  value=clamp(safe_int(pref("cramps_num", 0)), 0, 5),
                                  format_func=lambda x: ORDINAL_0_5_TEXT[x])
    headaches_num = st.select_slider("Headaches", options=list(range(6)),
                                     value=clamp(safe_int(pref("headaches_num", 0)), 0, 5),
                                     format_func=lambda x: ORDINAL_0_5_TEXT[x])
    sleepissue_num = st.select_slider("Sleep issues", options=list(range(6)),
                                      value=clamp(safe_int(pref("sleepissue_num", 0)), 0, 5),
                                      format_func=lambda x: ORDINAL_0_5_TEXT[x])

    st.markdown("#### Yesterday (optional)")
    include_yesterday = st.checkbox("Include yesterday’s mood & energy", value=False)
    lag1_mood = lag1_energy = None
    if include_yesterday:
        lag1_mood = st.selectbox("Yesterday’s mood", [0, 1, 2], index=1, format_func=lambda x: MOOD_LABELS[x])
        lag1_energy = st.selectbox("Yesterday’s energy", [0, 1, 2], index=1, format_func=lambda x: ENERGY_LABELS[x])

    payload = build_payload(
        chosen_date=chosen_date,
        phase=phase,
        sleep_minutes=sleep_minutes,
        resting_hr=resting_hr,
        stress_num=stress_num,
        cramps_num=cramps_num,
        headaches_num=headaches_num,
        sleepissue_num=sleepissue_num,
        include_yesterday=include_yesterday,
        lag1_mood=lag1_mood,
        lag1_energy=lag1_energy,
    )

    pred_out = None
    try:
        pred_out = predict_one(payload)
        mcol, ecol = st.columns(2)
        mcol.metric("Predicted mood", MOOD_LABELS[int(pred_out["mood_pred"])])
        ecol.metric("Predicted energy", ENERGY_LABELS[int(pred_out["energy_pred"])])
    except Exception as e:
        st.error("Prediction failed for these inputs.")
        st.exception(e)

    if show_debug and pred_out:
            st.markdown("**Model evaluation**")
            st.code(json.dumps(pred_out, indent=2), language="json")

    st.divider()
    button1, button2 = st.columns(2)

    if button1.button("Save this day", type="primary", use_container_width=True):
        try:
            out = pred_out or predict_one(payload)
            upsert_entry({
                "entry_date": key,
                **payload,
                "mood_pred": int(out["mood_pred"]),
                "energy_pred": int(out["energy_pred"]),
                "route": out.get("route"),
            })
            st.success("Saved ✔")
            st.rerun()
        except Exception as e:
            st.error("Save failed.")
            st.exception(e)

    if button2.button("Delete this day", use_container_width=True):
        try:
            delete_entry(key)
            st.warning("Deleted ✔")
            st.rerun()
        except Exception as e:
            st.error("Delete failed.")
            st.exception(e)

    with st.expander("History"):
        if load.empty:
            st.write("No saved entries for this month yet.")
        else:
            st.dataframe(
                load[
                    [
                        "entry_date_display",
                        "phase",
                        "sleep_duration_minutes",
                        "sleep_hours",
                        "resting_heart_rate__value",
                        "mood_label",
                        "energy_label",
                    ]
                ].rename(
                    columns={
                        "entry_date_display": "entry date",
                        "sleep_duration_minutes": "sleep duration (min)",
                        "sleep_hours": "sleep hours",
                        "resting_heart_rate__value": "resting heart rate",
                        "mood_label": "mood (pred)",
                        "energy_label": "energy (pred)",
                    }
                ),
                use_container_width=True,
            )

