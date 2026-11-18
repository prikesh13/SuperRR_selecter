"""
5_Player_Explorer.py
====================
Deep-dive single player dossier and exploratory filtering across roles, formats, and pressure scenarios.
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
render_header("Player Explorer", "Search, filter, and inspect detailed situational profiles for any player")

@st.cache_data
def load_all_players():
    bat = pd.read_csv(DATA_DIR / "batting_pii.csv")
    bowl = pd.read_csv(DATA_DIR / "bowling_pii.csv")
    bat_roles = pd.read_csv(DATA_DIR / "batting_roles.csv")
    bowl_roles = pd.read_csv(DATA_DIR / "bowling_roles.csv")
    bat = bat.merge(bat_roles, on="player_id", how="left")
    bowl = bowl.merge(bowl_roles, on="player_id", how="left")
    return bat, bowl

bat_df, bowl_df = load_all_players()

search_type = st.radio("Choose Discipline:", ["Batters", "Bowlers"], horizontal=True)

if search_type == "Batters":
    st.subheader("Filter Batters")
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        roles = ["All"] + sorted(bat_df["batting_role"].dropna().unique().tolist())
        selected_role = st.selectbox("Role Filter", roles)
    with f_col2:
        min_balls = st.slider("Minimum HP Balls Faced", 50, 250, 100)
    with f_col3:
        quadrants = ["All"] + sorted(bat_df["bat_perf_quadrant"].dropna().unique().tolist())
        selected_quad = st.selectbox("Consistency Quadrant", quadrants)

    filtered = bat_df[bat_df["hp_balls"] >= min_balls].copy()
    if selected_role != "All":
        filtered = filtered[filtered["batting_role"] == selected_role]
    if selected_quad != "All":
        filtered = filtered[filtered["bat_perf_quadrant"] == selected_quad]

    st.write(f"Showing **{len(filtered)}** qualified batters:")
    st.dataframe(filtered[[
        "batting_rank", "player_name", "batting_role", "batting_pii",
        "hp_balls", "hp_strike_rate", "normal_strike_rate",
        "pressure_sr_delta", "batting_consistency_score", "bat_perf_quadrant"
    ]].sort_values("batting_pii", ascending=False), use_container_width=True)

    st.divider()
    st.subheader("Individual Player Dossier")
    sel_player = st.selectbox("Select Batter to Inspect", filtered["player_name"].tolist())
    if sel_player:
        p_row = filtered[filtered["player_name"] == sel_player].iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            metric_card("Batting PII", f"{p_row['batting_pii']:.1f}", f"Rank #{int(p_row['batting_rank'])}", pink=True)
        with m2:
            metric_card("HP Strike Rate", f"{p_row['hp_strike_rate']:.1f}", f"Normal: {p_row['normal_strike_rate']:.1f}")
        with m3:
            metric_card("Clutch SR Delta", f"{p_row['pressure_sr_delta']:+.1f}", "vs Normal Conditions", pink=True)
        with m4:
            metric_card("Consistency", f"{p_row.get('batting_consistency_score', 0):.0f}/100", f"Tier: {p_row['bat_perf_quadrant']}")

else:
    st.subheader("Filter Bowlers")
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        roles = ["All"] + sorted(bowl_df["bowling_phase_role"].dropna().unique().tolist())
        selected_role = st.selectbox("Phase Role Filter", roles)
    with f_col2:
        min_balls = st.slider("Minimum HP Balls Bowled", 30, 200, 60)
    with f_col3:
        styles = ["All"] + sorted(bowl_df["bowl_type"].dropna().unique().tolist())
        selected_style = st.selectbox("Pace or Spin", styles)

    filtered = bowl_df[bowl_df["hp_balls"] >= min_balls].copy()
    if selected_role != "All":
        filtered = filtered[filtered["bowling_phase_role"] == selected_role]
    if selected_style != "All":
        filtered = filtered[filtered["bowl_type"] == selected_style]

    st.write(f"Showing **{len(filtered)}** qualified bowlers:")
    st.dataframe(filtered[[
        "bowling_rank", "player_name", "bowling_phase_role", "bowl_type",
        "bowling_pii", "hp_balls", "hp_economy", "normal_economy",
        "pressure_economy_delta", "bowling_consistency_score", "bowl_perf_quadrant"
    ]].sort_values("bowling_pii", ascending=False), use_container_width=True)

    st.divider()
    st.subheader("Individual Player Dossier")
    sel_player = st.selectbox("Select Bowler to Inspect", filtered["player_name"].tolist())
    if sel_player:
        p_row = filtered[filtered["player_name"] == sel_player].iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            metric_card("Bowling PII", f"{p_row['bowling_pii']:.1f}", f"Rank #{int(p_row['bowling_rank'])}", pink=True)
        with m2:
            metric_card("HP Economy", f"{p_row['hp_economy']:.2f}", f"Normal: {p_row['normal_economy']:.2f}")
        with m3:
            metric_card("Econ Delta (+ Imp)", f"{p_row['pressure_economy_delta']:+.2f} rpo", "Improvement under pressure", pink=True)
        with m4:
            metric_card("Consistency", f"{p_row.get('bowling_consistency_score', 0):.0f}/100", f"Tier: {p_row['bowl_perf_quadrant']}")
