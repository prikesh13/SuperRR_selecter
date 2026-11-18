"""
3_Batting_Analytics.py
======================
Comprehensive batting analytics under pressure: Top 10 batters,
Strike Rate Delta, Average Delta, Survival, and Player-to-Player head-to-head comparison.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from theme import apply_theme, render_header, metric_card, RR_BLUE, RR_PINK, RR_GOLD, RR_ROYAL_BLUE

apply_theme()
render_header("Batting Analytics Under Pressure", "Identifying the most clutch batters who elevate their scoring when the stakes are highest")

@st.cache_data
def load_batting_data():
    top10_bat = pd.read_csv(DATA_DIR / "top10_batters.csv")
    bat_pii = pd.read_csv(DATA_DIR / "batting_pii.csv")
    bat_roles = pd.read_csv(DATA_DIR / "batting_roles.csv")
    df = bat_pii.merge(bat_roles, on="player_id", how="left")
    return top10_bat, df

top10_bat, all_bat = load_batting_data()

st.subheader("Official Top 10 Batters Under Pressure (Batting PII)")

# Display Leaderboard Table
display_cols = [
    "batting_rank", "player_name", "batting_pii", "hp_balls",
    "hp_strike_rate", "normal_strike_rate", "pressure_sr_delta",
    "hp_batting_avg", "hp_boundary_pct", "batting_reliability"
]
table_df = top10_bat[display_cols].copy().rename(columns={
    "batting_rank": "Rank",
    "player_name": "Batter",
    "batting_pii": "PII Score",
    "hp_balls": "HP Balls",
    "hp_strike_rate": "HP Strike Rate",
    "normal_strike_rate": "Normal SR",
    "pressure_sr_delta": "SR Delta (Clutch)",
    "hp_batting_avg": "HP Average",
    "hp_boundary_pct": "Boundary %",
    "batting_reliability": "Reliability (Weight)"
})
st.dataframe(table_df.style.background_gradient(subset=["PII Score", "SR Delta (Clutch)"], cmap="RdPu"), use_container_width=True)

st.divider()

# Charts
c1, c2 = st.columns(2)

with c1:
    st.subheader("Pressure Strike Rate Delta (High Pressure vs Normal)")
    st.caption("Positive values indicate the batter accelerates significantly under pressure")
    fig_delta = px.bar(
        top10_bat.sort_values("pressure_sr_delta", ascending=True),
        y="player_name",
        x="pressure_sr_delta",
        orientation="h",
        color="pressure_sr_delta",
        color_continuous_scale=[RR_ROYAL_BLUE, RR_PINK],
        labels={"player_name": "", "pressure_sr_delta": "SR Difference (+/-)"},
        text="pressure_sr_delta"
    )
    fig_delta.update_traces(texttemplate='%{text:+.1f}', textposition='outside')
    fig_delta.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig_delta, use_container_width=True)

with c2:
    st.subheader("High-Pressure Boundary Efficiency vs Survival")
    st.caption("Top right corner represents batters who score boundaries without giving away their wicket")
    fig_scatter = px.scatter(
        all_bat,
        x="hp_boundary_pct",
        y="pressure_survival_rate",
        size="hp_balls",
        color="batting_pii",
        hover_name="player_name",
        color_continuous_scale="Viridis",
        labels={
            "hp_boundary_pct": "Boundary % under Pressure",
            "pressure_survival_rate": "Pressure Survival Rate",
            "batting_pii": "PII Score"
        }
    )
    fig_scatter.update_layout(height=450)
    st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# Head-to-Head Batter Comparison
st.subheader("⚔️ Head-to-Head Batter Comparison Engine")
comp_col1, comp_col2 = st.columns(2)

player_list = sorted(all_bat["player_name"].dropna().unique().tolist())
p1_default_idx = player_list.index("Rinku Singh") if "Rinku Singh" in player_list else 0
p2_default_idx = player_list.index("Tim David") if "Tim David" in player_list else 1

with comp_col1:
    p1 = st.selectbox("Select Player 1", player_list, index=p1_default_idx)
with comp_col2:
    p2 = st.selectbox("Select Player 2", player_list, index=p2_default_idx)

if p1 and p2:
    p1_row = all_bat[all_bat["player_name"] == p1].iloc[0]
    p2_row = all_bat[all_bat["player_name"] == p2].iloc[0]

    radar_metrics = [
        "HP Strike Rate",
        "Normal Strike Rate",
        "HP Boundary %",
        "HP Survival %",
        "Consistency Score",
        "Batting PII"
    ]

    p1_vals = [
        min(p1_row["hp_strike_rate"] / 2.2, 100),
        min(p1_row["normal_strike_rate"] / 2.0, 100),
        min(p1_row["hp_boundary_pct"] * 3.0, 100),
        p1_row["pressure_survival_rate"] * 100,
        p1_row.get("batting_consistency_score", 50),
        p1_row["batting_pii"]
    ]

    p2_vals = [
        min(p2_row["hp_strike_rate"] / 2.2, 100),
        min(p2_row["normal_strike_rate"] / 2.0, 100),
        min(p2_row["hp_boundary_pct"] * 3.0, 100),
        p2_row["pressure_survival_rate"] * 100,
        p2_row.get("batting_consistency_score", 50),
        p2_row["batting_pii"]
    ]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=p1_vals, theta=radar_metrics, fill='toself', name=p1, line_color=RR_PINK))
    fig_radar.add_trace(go.Scatterpolar(r=p2_vals, theta=radar_metrics, fill='toself', name=p2, line_color=RR_ROYAL_BLUE))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=420)

    st.plotly_chart(fig_radar, use_container_width=True)
