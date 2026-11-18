"""
App.py
======
Main entry point for SupeRR Selector Season 2 - Rajasthan Royals Scouting & Pressure Intelligence System.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add src and dashboard to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR / "dashboard"))

from theme import apply_theme, render_header, metric_card, RR_BLUE, RR_PINK

st.set_page_config(
    page_title="SupeRR Selector | RR Pressure Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_theme()

st.sidebar.image("https://thumb.wikimedia.org/wikipedia/commons/thumb/6/69/Rajasthan_Royals_Logo.png/330px-Rajasthan_Royals_Logo.png", width=110)
st.sidebar.title("SupeRR Selector")
st.sidebar.markdown("**Season 2 Challenge**\n\n*Pressure Performance Intelligence & IPL Auction Scouting System*")
st.sidebar.divider()
st.sidebar.info("Navigate through the analytics modules above using the sidebar pages.")

render_header(
    "SupeRR Selector — Pressure Performance Intelligence",
    "Rajasthan Royals Decision-Support & Scouting System (IPL 2022–2026 Data)"
)

st.markdown("""
### Welcome to the Rajasthan Royals Pressure Performance Intelligence System

In modern T20 cricket, champions are crowned not by aggregate runs or wickets scored in comfortable phases, but by **who executes when the match is on the line**.

This platform evaluates historical ball-by-ball performance across **5 IPL Seasons (2022–2026)** and **134 T20I bilateral tournaments**, isolating high-pressure scenarios from baseline comfort to answer the franchise's most critical question:

> **"Who can Rajasthan Royals trust when 18 runs are needed off 10 balls, or when defending 12 runs in the 20th over?"**

---
""")

col1, col2, col3, col4 = st.columns(4)
with col1:
    metric_card("Analyzed Matches", "867", "IPL + T20I (2022-2026)", pink=False)
with col2:
    metric_card("Total Deliveries", "193,988", "Cleaned & Validated balls", pink=False)
with col3:
    metric_card("Top Ranked Batter", "Rinku Singh", "Batting PII: 91.95 (SR 196.4)", pink=True)
with col4:
    metric_card("Top Ranked Bowler", "Haris Rauf", "Bowling PII: 97.91 (Econ 7.56)", pink=True)

st.markdown("""
### Core Analytical Pillars:
1. **Explainable Pressure Engine (PII)**: Dynamic calculation of batting and bowling pressure based on required run rates, resources remaining, stage, and closeness — *with zero future information leakage*.
2. **Pressure Delta Analysis**: Quantifies whether a player elevates or declines under pressure compared to normal situations.
3. **Statistical Shrinkage & Reliability Adjustment**: Penalises small-sample variance using James-Stein empirical Bayesian priors to ensure evidence-backed rankings.
4. **Match-Level Consistency**: Evaluates consistency across individual match appearances using coefficient of variation and quadrant mapping.
5. **Rajasthan Royals Auction Shortlist**: Evidence-based scouting dossiers with tactical fit, strengths, and weaknesses for IPL 2027 auction planning.

---
👉 **Select a page from the sidebar to begin exploring.**
""")
