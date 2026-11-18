"""
2_Pressure_Intelligence.py
==========================
Deep dive into the Pressure Intelligence Engine (PII) mathematical methodology,
distribution of pressure across innings, match phases, and explainable formulas.
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
render_header("Pressure Intelligence Engine", "Dynamic situational pressure classification and explainable modeling")

st.markdown("""
### The Philosophy: Context Over Conventional Over Numbers

Conventional cricket analyses naively equate "pressure" exclusively with death overs (overs 17–20). 
However:
- A batter chasing 30 off 40 balls with 8 wickets in hand in the 18th over experiences **minimal pressure**.
- Conversely, a batter entering at 22/4 in the 5th over with 180 to chase experiences **extreme structural pressure**.

The **Pressure Intelligence Index (PII)** dynamically computes a delivery-by-delivery pressure score ($0-100$) using **only information available before the ball is bowled**.
""")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Batting Pressure Formulation (2nd Innings / Run Chase)")
    st.latex(r"""
    \text{Score}_{\text{bat}} = w_1 \cdot \text{Norm}(\text{eRRR}) + w_2 \cdot \text{Norm}(\text{Runs Needed}) + w_3 \cdot \text{Norm}(\text{Balls Rem.}^{-1}) + w_4 \cdot \text{Norm}(\text{Wickets Lost}) + w_5 \cdot \text{Closeness}
    """)
    st.markdown("""
    - **Effective RRR Weight ($35\%$)**: Scaled relative to standard T20 baseline ($8.0 \to 15.0+$ rpo).
    - **Runs Needed ($20\%$)**: Absolute volume of target remaining.
    - **Balls Remaining Urgency ($20\%$)**: Temporal compression.
    - **Wicket Resources Lost ($15\%$)**: Vulnerability to bowling out.
    - **Match Closeness ($10\%$)**: Proximity to razor-thin victory/defeat margins.
    """)

with col2:
    st.markdown("#### Bowling Pressure Formulation (Defending Chase)")
    st.latex(r"""
    \text{Score}_{\text{bowl}} = w_1 \cdot \text{Norm}(\text{Defend Rate}) + w_2 \cdot \text{Norm}(\text{Balls Rem.}^{-1}) + w_3 \cdot \text{Norm}(\text{Opp. Wickets in Hand}) + w_4 \cdot \text{Momentum} + w_5 \cdot \text{Closeness}
    """)
    st.markdown("""
    - **Defending Rate ($35\%$)**: Runs batting team requires per ball remaining (harder when low).
    - **Time Pressure ($20\%$)**: Fewer balls left to break partnerships.
    - **Opposition Resources ($20\%$)**: Threat posed by remaining batting depth.
    - **Opposition Momentum ($15\%$)**: Rolling count of boundaries conceded in the last 6 deliveries.
    - **Match Closeness ($10\%$)**: Tension when the finish line is imminent.
    """)

st.divider()

st.subheader("Empirical Pressure Distribution Across Analyzed Deliveries")

dist_col1, dist_col2 = st.columns(2)

with dist_col1:
    bat_dist_df = pd.DataFrame({
        "Category": ["Low Pressure (0–33)", "Medium Pressure (34–66)", "High Pressure (67–100)"],
        "Deliveries": [95314, 78764, 19910],
        "Percentage": [49.1, 40.6, 10.3]
    })
    fig_bat_pie = px.pie(
        bat_dist_df,
        values="Deliveries",
        names="Category",
        title="Batting Pressure Category Distribution",
        color="Category",
        color_discrete_map={
            "Low Pressure (0–33)": "#90CAF9",
            "Medium Pressure (34–66)": RR_ROYAL_BLUE,
            "High Pressure (67–100)": RR_PINK
        },
        hole=0.4
    )
    fig_bat_pie.update_layout(height=350)
    st.plotly_chart(fig_bat_pie, use_container_width=True)

with dist_col2:
    bowl_dist_df = pd.DataFrame({
        "Category": ["Low Pressure (0–33)", "Medium Pressure (34–66)", "High Pressure (67–100)"],
        "Deliveries": [50363, 128258, 15367],
        "Percentage": [26.0, 66.1, 7.9]
    })
    fig_bowl_pie = px.pie(
        bowl_dist_df,
        values="Deliveries",
        names="Category",
        title="Bowling Pressure Category Distribution",
        color="Category",
        color_discrete_map={
            "Low Pressure (0–33)": "#90CAF9",
            "Medium Pressure (34–66)": RR_BLUE,
            "High Pressure (67–100)": RR_PINK
        },
        hole=0.4
    )
    fig_bowl_pie.update_layout(height=350)
    st.plotly_chart(fig_bowl_pie, use_container_width=True)

st.markdown("""
> [!NOTE]
> **Strict Leakage Prevention**:
> Neither match winners, target chase results, nor balls occurring *after* the current delivery enter into the equation. The state is computed cumulatively prior to each ball.
""")
