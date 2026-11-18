"""
1_Executive_Overview.py
=======================
Executive summary with top-level KPIs, format breakdown, and high-level leaders.
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
render_header("Executive Overview", "High-level summary of tournament data, pressure coverage, and premier performers")

@st.cache_data
def load_overview_data():
    top_bat = pd.read_csv(DATA_DIR / "top10_batters.csv")
    top_bowl = pd.read_csv(DATA_DIR / "top10_bowlers.csv")
    bat_pii = pd.read_csv(DATA_DIR / "batting_pii.csv")
    bowl_pii = pd.read_csv(DATA_DIR / "bowling_pii.csv")
    return top_bat, top_bowl, bat_pii, bowl_pii

top_bat, top_bowl, bat_pii, bowl_pii = load_overview_data()

# 8 KPI Metrics Cards
row1_1, row1_2, row1_3, row1_4 = st.columns(4)
with row1_1:
    metric_card("Total Matches", "867", "371 IPL | 470 T20I | 26 Excl.")
with row1_2:
    metric_card("Total Deliveries", "193,988", "192,709 Legal Balls")
with row1_3:
    metric_card("Unique Batters", str(len(bat_pii)), f"{len(bat_pii)} met reliability filters")
with row1_4:
    metric_card("Unique Bowlers", str(len(bowl_pii)), f"{len(bowl_pii)} met reliability filters")

row2_1, row2_2, row2_3, row2_4 = st.columns(4)
with row2_1:
    metric_card("High Pressure Balls", "19,910 (Bat)", "10.3% of total deliveries", pink=True)
with row2_2:
    metric_card("Tight Match Rate", "38.4%", "Decided in final 2 overs", pink=True)
with row2_3:
    best_bat = top_bat.iloc[0]
    metric_card("#1 Pressure Batter", best_bat["player_name"], f"PII: {best_bat['batting_pii']} (SR {best_bat['hp_strike_rate']:.1f})", pink=True)
with row2_4:
    best_bowl = top_bowl.iloc[0]
    metric_card("#1 Pressure Bowler", best_bowl["player_name"], f"PII: {best_bowl['bowling_pii']} (Econ {best_bowl['hp_economy']:.2f})", pink=True)

st.divider()

# Charts
c1, c2 = st.columns(2)

with c1:
    st.subheader("Top 5 Batters Under Pressure (Batting PII)")
    fig_bat = px.bar(
        top_bat.head(5),
        x="player_name",
        y="batting_pii",
        color="batting_pii",
        color_continuous_scale=[RR_ROYAL_BLUE, RR_PINK],
        labels={"player_name": "Player", "batting_pii": "Batting PII Score"},
        text="batting_pii"
    )
    fig_bat.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig_bat.update_layout(height=380, showlegend=False, xaxis_title="", yaxis_range=[0, 105])
    st.plotly_chart(fig_bat, use_container_width=True)

with c2:
    st.subheader("Top 5 Bowlers Under Pressure (Bowling PII)")
    fig_bowl = px.bar(
        top_bowl.head(5),
        x="player_name",
        y="bowling_pii",
        color="bowling_pii",
        color_continuous_scale=[RR_BLUE, RR_PINK],
        labels={"player_name": "Player", "bowling_pii": "Bowling PII Score"},
        text="bowling_pii"
    )
    fig_bowl.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig_bowl.update_layout(height=380, showlegend=False, xaxis_title="", yaxis_range=[0, 105])
    st.plotly_chart(fig_bowl, use_container_width=True)

st.divider()

st.subheader("Comparative Macro Statistics: High Pressure vs Baseline")
macro_col1, macro_col2 = st.columns(2)

with macro_col1:
    st.markdown("""
    **Batting Trends Under High Pressure:**
    - **Mean Population Strike Rate** shifts from **122.4** in normal conditions to **148.7** under pressure due to urgent boundary pursuit.
    - However, **Wicket Frequency per ball** triples from **3.1%** to **9.4%**, leading to massive batting collapse risk.
    - Only an elite **8.2% of batters** maintain positive strike rate acceleration without catastrophic dismissal rates.
    """)

with macro_col2:
    st.markdown("""
    **Bowling Trends Under High Pressure:**
    - **Mean Population Economy** deteriorates from **8.12 rpo** to **10.45 rpo** in high pressure.
    - Bowlers who concede under **8.5 rpo** in high-pressure scenarios represent less than **12% of the bowler pool**.
    - Haris Rauf, Wanindu Hasaranga, and Eshan Malinga consistently restrict boundaries under heavy chase pressure.
    """)
