"""
7_RR_Scouting_Recommendations.py
================================
Actionable Rajasthan Royals scouting recommendations, squad fit,
and evidence-based player targets for IPL 2027 auction planning.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from theme import apply_theme, render_header, metric_card, RR_BLUE, RR_PINK, RR_GOLD, RR_ROYAL_BLUE

apply_theme()
render_header("Rajasthan Royals Auction Scouting Dossiers", "Evidence-based player recommendations to strengthen RR's high-pressure execution for IPL 2027")

st.markdown("""
> [!IMPORTANT]
> **Scouting Objective & Principles**:
> These scouting recommendations are **grounded strictly in historical statistical evidence** from 2022 to 2026. 
> We do not speculate on player auction prices, auction dynamics, or availability. Our objective is to identify **who RR can trust when the match is on the line**.
""")

@st.cache_data
def load_scouting_data():
    scout_bat = pd.read_csv(DATA_DIR / "scouting_batters.csv")
    scout_bowl = pd.read_csv(DATA_DIR / "scouting_bowlers.csv")
    return scout_bat, scout_bowl

scout_bat, scout_bowl = load_scouting_data()

tab1, tab2 = st.tabs(["🏏 Top Batting Targets", "🎯 Top Bowling Targets"])

with tab1:
    st.subheader("Priority Batting Targets for Rajasthan Royals")
    
    # Download button for scouting report
    st.download_button(
        label="📥 Download Batting Scouting Dossier (CSV)",
        data=scout_bat.to_csv(index=False),
        file_name="rr_batting_scouting_targets.csv",
        mime="text/csv"
    )
    
    for _, row in scout_bat.iterrows():
        with st.expander(f"Rank #{int(row['rank'])}: {row['player_name']} — PII: {row['batting_pii']} ({row['primary_role']})", expanded=(row['rank'] <= 3)):
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                st.markdown(f"**Teams**: {row['teams']}")
                st.markdown(f"**Primary Role**: {row['primary_role']}")
                st.markdown(f"**Avg Batting Pos**: {row['avg_batting_position']}")
                st.markdown(f"**Reliability Score**: `{row['reliability_score']}`")
            with c2:
                st.markdown(f"**HP Balls Faced**: {row['pressure_balls_faced']}")
                st.markdown(f"**HP Strike Rate**: `{row['pressure_strike_rate']}`")
                st.markdown(f"**Normal Strike Rate**: `{row['normal_strike_rate']}`")
                st.markdown(f"**Clutch Delta**: `{row['pressure_sr_delta']}`")
            with c3:
                st.markdown(f"**Strategic Value Proposition for RR**:")
                st.info(row['rr_squad_role'])
                st.markdown(f"✅ **Key Strengths**: {row['strengths']}")
                st.markdown(f"⚠️ **Weaknesses / Risk Factors**: {row['weaknesses']}")
                st.markdown(f"📊 **Evidence Strength**: {row['reliability_evidence']}")

with tab2:
    st.subheader("Priority Bowling Targets for Rajasthan Royals")
    
    st.download_button(
        label="📥 Download Bowling Scouting Dossier (CSV)",
        data=scout_bowl.to_csv(index=False),
        file_name="rr_bowling_scouting_targets.csv",
        mime="text/csv"
    )
    
    for _, row in scout_bowl.iterrows():
        with st.expander(f"Rank #{int(row['rank'])}: {row['player_name']} — PII: {row['bowling_pii']} ({row['primary_phase_role']})", expanded=(row['rank'] <= 3)):
            c1, c2, c3 = st.columns([1, 1, 2])
            with c1:
                st.markdown(f"**Teams**: {row['teams']}")
                st.markdown(f"**Discipline**: {row['pace_or_spin']} ({row['bowling_style']})")
                st.markdown(f"**Phase Role**: {row['primary_phase_role']}")
                st.markdown(f"**Reliability Score**: `{row['reliability_score']}`")
            with c2:
                st.markdown(f"**HP Balls Bowled**: {row['pressure_balls_bowled']}")
                st.markdown(f"**HP Economy**: `{row['pressure_economy']} rpo`")
                st.markdown(f"**Normal Economy**: `{row['normal_economy']} rpo`")
                st.markdown(f"**Economy Delta**: `{row['pressure_economy_delta']} rpo`")
            with c3:
                st.markdown(f"**Strategic Value Proposition for RR**:")
                st.info(row['rr_squad_role'])
                st.markdown(f"✅ **Key Strengths**: {row['strengths']}")
                st.markdown(f"⚠️ **Weaknesses / Risk Factors**: {row['weaknesses']}")
                st.markdown(f"📊 **Evidence Strength**: {row['reliability_evidence']}")
