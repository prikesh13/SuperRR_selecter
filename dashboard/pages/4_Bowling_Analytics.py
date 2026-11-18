"""
4_Bowling_Analytics.py
======================
Comprehensive bowling analytics under pressure: Top 10 bowlers,
Economy Delta, Wicket-Taking Rate, Dot Ball Efficiency, and Phase Performance.
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
render_header("Bowling Analytics Under Pressure", "Identifying the most economical and threatening wicket-takers in clutch situations")

@st.cache_data
def load_bowling_data():
    top10_bowl = pd.read_csv(DATA_DIR / "top10_bowlers.csv")
    bowl_pii = pd.read_csv(DATA_DIR / "bowling_pii.csv")
    bowl_roles = pd.read_csv(DATA_DIR / "bowling_roles.csv")
    phase_metrics = pd.read_csv(DATA_DIR / "phase_bowling_metrics.csv")
    df = bowl_pii.merge(bowl_roles, on="player_id", how="left")
    return top10_bowl, df, phase_metrics

top10_bowl, all_bowl, phase_metrics = load_bowling_data()

st.subheader("Official Top 10 Bowlers Under Pressure (Bowling PII)")

# Display Leaderboard Table
display_cols = [
    "bowling_rank", "player_name", "bowling_pii", "hp_balls",
    "hp_economy", "normal_economy", "pressure_economy_delta",
    "hp_wicket_rate", "hp_dot_pct", "bowling_reliability"
]
table_df = top10_bowl[display_cols].copy().rename(columns={
    "bowling_rank": "Rank",
    "player_name": "Bowler",
    "bowling_pii": "PII Score",
    "hp_balls": "HP Balls",
    "hp_economy": "HP Economy",
    "normal_economy": "Normal Econ",
    "pressure_economy_delta": "Econ Delta (+ Imp)",
    "hp_wicket_rate": "HP Wkt/Over",
    "hp_dot_pct": "HP Dot %",
    "bowling_reliability": "Reliability"
})
st.dataframe(table_df.style.background_gradient(subset=["PII Score", "Econ Delta (+ Imp)"], cmap="Blues"), use_container_width=True)

st.divider()

# Charts
c1, c2 = st.columns(2)

with c1:
    st.subheader("Pressure Economy Delta (Normal Economy − HP Economy)")
    st.caption("Positive values indicate the bowler tightens up and becomes more economical under pressure")
    fig_delta = px.bar(
        top10_bowl.sort_values("pressure_economy_delta", ascending=True),
        y="player_name",
        x="pressure_economy_delta",
        orientation="h",
        color="pressure_economy_delta",
        color_continuous_scale=[RR_PINK, RR_ROYAL_BLUE],
        labels={"player_name": "", "pressure_economy_delta": "Economy Improvement (+/- rpo)"},
        text="pressure_economy_delta"
    )
    fig_delta.update_traces(texttemplate='%{text:+.2f}', textposition='outside')
    fig_delta.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig_delta, use_container_width=True)

with c2:
    st.subheader("HP Economy vs HP Wicket Rate (All Qualified Bowlers)")
    st.caption("Lower-right quadrant represents the ideal T20 bowler: low economy with high wicket frequency")
    fig_scatter = px.scatter(
        all_bowl,
        x="hp_economy",
        y="hp_wicket_rate",
        size="hp_balls",
        color="bowling_pii",
        hover_name="player_name",
        color_continuous_scale="Viridis",
        labels={
            "hp_economy": "HP Economy (rpo)",
            "hp_wicket_rate": "HP Wickets / Over",
            "bowling_pii": "PII Score"
        }
    )
    fig_scatter.update_layout(height=450)
    st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# Head-to-Head Bowler Radar Comparison
st.subheader("⚔️ Head-to-Head Bowler Comparison Engine")
comp_col1, comp_col2 = st.columns(2)

player_list = sorted(all_bowl["player_name"].dropna().unique().tolist())
p1_default_idx = player_list.index("Haris Rauf") if "Haris Rauf" in player_list else 0
p2_default_idx = player_list.index("Wanindu Hasaranga") if "Wanindu Hasaranga" in player_list else 1

with comp_col1:
    b1 = st.selectbox("Select Bowler 1", player_list, index=p1_default_idx)
with comp_col2:
    b2 = st.selectbox("Select Bowler 2", player_list, index=p2_default_idx)

if b1 and b2:
    b1_row = all_bowl[all_bowl["player_name"] == b1].iloc[0]
    b2_row = all_bowl[all_bowl["player_name"] == b2].iloc[0]

    radar_metrics = [
        "Economy Control (Inv)",
        "Dot Ball %",
        "Wicket Threat",
        "Boundary Prevention",
        "Consistency Score",
        "Bowling PII"
    ]

    # Inverted economy: 6 rpo -> 100, 12 rpo -> 0
    b1_vals = [
        max(0, min(100, (14 - b1_row["hp_economy"]) * 12.5)),
        min(100, b1_row["hp_dot_pct"] * 2.0),
        min(100, (b1_row["hp_wicket_rate"] or 0) * 40.0),
        max(0, min(100, (30 - b1_row["hp_boundary_pct"]) * 4.0)),
        b1_row.get("bowling_consistency_score", 50),
        b1_row["bowling_pii"]
    ]

    b2_vals = [
        max(0, min(100, (14 - b2_row["hp_economy"]) * 12.5)),
        min(100, b2_row["hp_dot_pct"] * 2.0),
        min(100, (b2_row["hp_wicket_rate"] or 0) * 40.0),
        max(0, min(100, (30 - b2_row["hp_boundary_pct"]) * 4.0)),
        b2_row.get("bowling_consistency_score", 50),
        b2_row["bowling_pii"]
    ]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(r=b1_vals, theta=radar_metrics, fill='toself', name=b1, line_color=RR_ROYAL_BLUE))
    fig_radar.add_trace(go.Scatterpolar(r=b2_vals, theta=radar_metrics, fill='toself', name=b2, line_color=RR_PINK))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=420)

    st.plotly_chart(fig_radar, use_container_width=True)
