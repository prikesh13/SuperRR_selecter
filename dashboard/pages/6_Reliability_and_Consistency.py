"""
6_Reliability_and_Consistency.py
================================
Statistical shrinkage, James-Stein adjustment, sample size uncertainty,
and 4-quadrant consistency analysis (Elite, Streaky, Reliable, Inconsistent).
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from theme import apply_theme, render_header, metric_card, RR_BLUE, RR_PINK, RR_GOLD, RR_ROYAL_BLUE

apply_theme()
render_header("Reliability & Consistency Analysis", "Overcoming small-sample bias via statistical shrinkage and match-level stability")

st.markdown("""
### Why Statistical Shrinkage is Critical

A batter who faces 15 pressure balls across two matches and hits three sixes may boast an observed strike rate of $240$.
Ranking such a player above a seasoned finisher who maintains a strike rate of $190$ over 200 balls would be a **catastrophic scouting failure**.

To eliminate small-sample bias, we apply **James-Stein Statistical Shrinkage**:

$$\\text{Adjusted Score} = \\left( \\frac{n}{n + k} \\right) \\cdot \\text{Player Observed Score} + \\left( \\frac{k}{n + k} \\right) \\cdot \\text{Baseline Population Score}$$

Where:
- $n = $ Effective high-pressure sample size (deliveries faced / bowled).
- $k = $ Reliability shrinkage constant ($k_{\\text{bat}} = 100$, $k_{\\text{bowl}} = 60$).
- Baseline = Empirical population mean across all qualifying professional matches.
""")

st.divider()

@st.cache_data
def load_reliability_data():
    bat = pd.read_csv(DATA_DIR / "batting_pii.csv")
    bowl = pd.read_csv(DATA_DIR / "bowling_pii.csv")
    return bat, bowl

bat_df, bowl_df = load_reliability_data()

c1, c2 = st.columns(2)

with c1:
    st.subheader("Batting 4-Quadrant Consistency Map")
    st.caption("Elite quadrant (Top-Right): High HP Strike Rate with low match-to-match variance")
    fig_bat_quad = px.scatter(
        bat_df,
        x="hp_sr_mean",
        y="batting_consistency_score",
        color="bat_perf_quadrant",
        size="hp_balls",
        hover_name="player_name",
        color_discrete_map={
            "Elite": RR_PINK,
            "Reliable": RR_ROYAL_BLUE,
            "Streaky": RR_GOLD,
            "Inconsistent": "#9E9E9E"
        },
        labels={
            "hp_sr_mean": "Mean HP Match Strike Rate",
            "batting_consistency_score": "Consistency Score (100 - CV)",
            "bat_perf_quadrant": "Quadrant"
        }
    )
    fig_bat_quad.update_layout(height=450)
    st.plotly_chart(fig_bat_quad, use_container_width=True)

with c2:
    st.subheader("Bowling 4-Quadrant Consistency Map")
    st.caption("Elite quadrant (Top-Left): Economical under pressure with highly consistent spells")
    fig_bowl_quad = px.scatter(
        bowl_df,
        x="hp_econ_mean",
        y="bowling_consistency_score",
        color="bowl_perf_quadrant",
        size="hp_balls",
        hover_name="player_name",
        color_discrete_map={
            "Elite": RR_PINK,
            "Reliable": RR_ROYAL_BLUE,
            "Streaky": RR_GOLD,
            "Inconsistent": "#9E9E9E"
        },
        labels={
            "hp_econ_mean": "Mean HP Economy (rpo)",
            "bowling_consistency_score": "Consistency Score (100 - CV)",
            "bowl_perf_quadrant": "Quadrant"
        }
    )
    fig_bowl_quad.update_layout(height=450)
    st.plotly_chart(fig_bowl_quad, use_container_width=True)

st.divider()

st.subheader("Observed vs Reliability-Adjusted Batting Strike Rate")
st.caption("Demonstrating how shrinkage prevents low-sample inflation while rewarding proven volume")

top_bat_comp = bat_df.sort_values("hp_balls", ascending=False).head(15)
fig_shrink = px.bar(
    top_bat_comp,
    x="player_name",
    y=["hp_strike_rate", "adj_hp_strike_rate"],
    barmode="group",
    color_discrete_map={
        "hp_strike_rate": "#90CAF9",
        "adj_hp_strike_rate": RR_PINK
    },
    labels={"value": "Strike Rate", "player_name": "Batter", "variable": "Metric"}
)
fig_shrink.update_layout(height=400)
st.plotly_chart(fig_shrink, use_container_width=True)
