"""
scouting_engine.py
==================
Generates RR scouting cards and natural-language player profiles.

For every Top-10 ranked player, produces:
  - Structured scouting card (dict/DataFrame)
  - Readable explanation of why they are ranked
  - Role fit for Rajasthan Royals squad

No auction prices, salary expectations, or availability claims are made.
All text is based strictly on calculated statistics.
"""

import numpy as np
import pandas as pd
from typing import Optional


def _fmt(value, fmt=".1f", fallback="N/A"):
    """Safe formatted string for a numeric value."""
    try:
        if pd.isna(value):
            return fallback
        return f"{value:{fmt}}"
    except (TypeError, ValueError):
        return fallback


def generate_batter_scouting_card(
    row: pd.Series,
    batting_role_df: pd.DataFrame,
    ctpm: pd.DataFrame,
) -> dict:
    """
    Generate a structured scouting card for a single batter.

    Parameters
    ----------
    row : one row from the top-10 batting PII DataFrame
    batting_role_df : output of batting_model.classify_batting_role()
    ctpm : competition_team_player_mapping DataFrame

    Returns
    -------
    dict suitable for display in dashboard / export to CSV
    """
    pid = row["player_id"]

    # Look up player details from CTPM
    player_rows = ctpm[ctpm["player_id"] == pid].drop_duplicates("comp_id")
    player_name = row.get("player_name", "Unknown")

    teams = player_rows["team_name"].unique().tolist()
    comps = player_rows["comp_name"].unique().tolist()

    # Batting role
    role_row = batting_role_df[batting_role_df["player_id"] == pid]
    batting_role = role_row["batting_role"].values[0] if len(role_row) else "Unknown"
    avg_bat_order = role_row["avg_bat_order"].values[0] if len(role_row) else np.nan

    # Determine strength and weakness
    hp_sr    = row.get("hp_strike_rate", np.nan)
    norm_sr  = row.get("normal_strike_rate", np.nan)
    sr_delta = row.get("pressure_sr_delta", np.nan)
    hp_avg   = row.get("hp_batting_avg", np.nan)
    hp_bnd   = row.get("hp_boundary_pct", np.nan)
    hp_dot   = row.get("hp_dot_pct", np.nan)
    survival = row.get("pressure_survival_rate", np.nan)
    conscore = row.get("batting_consistency_score", np.nan)
    reliability = row.get("batting_reliability", np.nan)
    pii     = row.get("batting_pii", np.nan)
    rank    = int(row.get("batting_rank", 0))
    hp_balls = row.get("hp_balls", 0)
    hp_innings = row.get("hp_innings", 0)
    pct_100 = row.get("pct_above_100", np.nan)

    # Strengths / weaknesses
    strengths = []
    weaknesses = []

    if pd.notna(sr_delta) and sr_delta > 5:
        strengths.append(f"SR improves by {sr_delta:+.1f} under pressure vs normal")
    elif pd.notna(sr_delta) and sr_delta < -10:
        weaknesses.append(f"SR drops by {abs(sr_delta):.1f} under pressure")

    if pd.notna(hp_sr) and hp_sr > 130:
        strengths.append(f"Exceptional pressure SR of {hp_sr:.1f}")
    elif pd.notna(hp_sr) and hp_sr < 90:
        weaknesses.append(f"Below-average pressure SR ({hp_sr:.1f})")

    if pd.notna(hp_bnd) and hp_bnd > 20:
        strengths.append(f"High boundary rate under pressure ({hp_bnd:.1f}%)")
    if pd.notna(hp_dot) and hp_dot > 45:
        weaknesses.append(f"High dot-ball rate under pressure ({hp_dot:.1f}%)")

    if pd.notna(conscore) and conscore > 60:
        strengths.append(f"Consistent pressure performer (consistency score {conscore:.0f}/100)")
    elif pd.notna(conscore) and conscore < 35:
        weaknesses.append("Inconsistent across pressure innings")

    if pd.notna(pct_100) and pct_100 > 50:
        strengths.append(f"{pct_100:.0f}% of pressure innings with SR > 100")

    # Reliability note
    if pd.notna(reliability):
        if reliability >= 0.65:
            evidence = f"Strong evidence ({hp_balls:.0f} pressure balls across {hp_innings:.0f} innings)"
        elif reliability >= 0.4:
            evidence = f"Moderate evidence ({hp_balls:.0f} pressure balls)"
        else:
            evidence = f"Limited evidence ({hp_balls:.0f} pressure balls — interpret with caution)"
    else:
        evidence = "Unknown"

    # RR role fit
    if batting_role == "Opener":
        rr_role = "Aggressive T20 opener capable of explosive starts under powerplay pressure"
    elif batting_role == "Top-order":
        rr_role = "Reliable top-order anchor who can build innings and handle early-wicket pressure"
    elif batting_role == "Middle-order":
        rr_role = "Middle-order stabiliser who can accelerate when wickets fall"
    elif batting_role == "Finisher":
        rr_role = "Death-overs finisher — ideal for final 4-over assault and difficult run chases"
    else:
        rr_role = "Versatile batting option"

    card = {
        "rank":                   rank,
        "player_name":            player_name,
        "player_id":              pid,
        "teams":                  ", ".join(teams[:3]),
        "primary_role":           batting_role,
        "avg_batting_position":   _fmt(avg_bat_order, ".1f"),
        "batting_pii":            _fmt(pii, ".1f"),
        "reliability_score":      _fmt(reliability, ".2f"),
        "reliability_evidence":   evidence,
        "pressure_balls_faced":   int(hp_balls) if pd.notna(hp_balls) else "N/A",
        "pressure_innings":       int(hp_innings) if pd.notna(hp_innings) else "N/A",
        "normal_strike_rate":     _fmt(norm_sr, ".1f"),
        "pressure_strike_rate":   _fmt(hp_sr, ".1f"),
        "pressure_sr_delta":      _fmt(sr_delta, "+.1f"),
        "pressure_avg":           _fmt(hp_avg, ".1f"),
        "pressure_boundary_pct":  _fmt(hp_bnd, ".1f"),
        "pressure_dot_pct":       _fmt(hp_dot, ".1f"),
        "pressure_survival_rate": _fmt(survival * 100 if pd.notna(survival) else np.nan, ".1f") + "%",
        "consistency_score":      _fmt(conscore, ".0f") + "/100",
        "pct_innings_sr_above_100": _fmt(pct_100, ".0f") + "%",
        "strengths":              " | ".join(strengths) if strengths else "See profile",
        "weaknesses":             " | ".join(weaknesses) if weaknesses else "None identified",
        "rr_squad_role":          rr_role,
    }

    return card


def generate_bowler_scouting_card(
    row: pd.Series,
    bowling_role_df: pd.DataFrame,
    ctpm: pd.DataFrame,
) -> dict:
    """
    Generate a structured scouting card for a single bowler.
    """
    pid = row["player_id"]

    player_rows = ctpm[ctpm["player_id"] == pid].drop_duplicates("comp_id")
    player_name = row.get("player_name", "Unknown")
    teams = player_rows["team_name"].unique().tolist()

    role_row = bowling_role_df[bowling_role_df["player_id"] == pid]
    bowl_phase_role = role_row["bowling_phase_role"].values[0] if len(role_row) else "Unknown"
    bowl_type = role_row["bowl_type"].values[0] if len(role_row) else "Unknown"
    bowl_style = role_row["bowling_type"].values[0] if len(role_row) else "Unknown"

    hp_econ   = row.get("hp_economy", np.nan)
    norm_econ = row.get("normal_economy", np.nan)
    econ_delta = row.get("pressure_economy_delta", np.nan)
    hp_dot    = row.get("hp_dot_pct", np.nan)
    hp_wktr   = row.get("hp_wicket_rate", np.nan)
    hp_bnd    = row.get("hp_boundary_pct", np.nan)
    bp_prev   = row.get("pressure_boundary_prevention", np.nan)
    conscore  = row.get("bowling_consistency_score", np.nan)
    reliability = row.get("bowling_reliability", np.nan)
    pii       = row.get("bowling_pii", np.nan)
    rank      = int(row.get("bowling_rank", 0))
    hp_balls  = row.get("hp_balls", 0)
    hp_matches = row.get("hp_match_count", 0)
    hp_wkts   = row.get("hp_wickets", 0)
    pct_8     = row.get("pct_below_8", np.nan)

    strengths = []
    weaknesses = []

    if pd.notna(econ_delta) and econ_delta > 1:
        strengths.append(f"Economy improves by {econ_delta:.1f} rpo under pressure")
    elif pd.notna(econ_delta) and econ_delta < -1.5:
        weaknesses.append(f"Economy worsens by {abs(econ_delta):.1f} rpo under pressure")

    if pd.notna(hp_econ) and hp_econ < 7.5:
        strengths.append(f"Excellent pressure economy ({hp_econ:.1f} rpo)")
    elif pd.notna(hp_econ) and hp_econ > 11:
        weaknesses.append(f"Expensive under pressure ({hp_econ:.1f} rpo)")

    if pd.notna(hp_dot) and hp_dot > 40:
        strengths.append(f"Locks down scoring — {hp_dot:.1f}% dot balls under pressure")

    if pd.notna(hp_wktr) and hp_wktr > 1.5:
        strengths.append(f"Takes wickets under pressure ({hp_wktr:.2f} per over)")

    if pd.notna(bp_prev) and bp_prev > 5:
        strengths.append(f"Reduces boundary rate by {bp_prev:.1f}% under pressure")
    elif pd.notna(hp_bnd) and hp_bnd > 20:
        weaknesses.append(f"Concedes many boundaries under pressure ({hp_bnd:.1f}%)")

    if pd.notna(conscore) and conscore > 60:
        strengths.append(f"Consistent under pressure (consistency {conscore:.0f}/100)")
    elif pd.notna(conscore) and conscore < 35:
        weaknesses.append("Inconsistent economy across pressure spells")

    if pd.notna(reliability):
        if reliability >= 0.65:
            evidence = f"Strong evidence ({hp_balls:.0f} pressure balls)"
        elif reliability >= 0.4:
            evidence = f"Moderate evidence ({hp_balls:.0f} pressure balls)"
        else:
            evidence = f"Limited evidence ({hp_balls:.0f} balls — interpret with caution)"
    else:
        evidence = "Unknown"

    # RR role fit
    if "Death" in bowl_phase_role:
        rr_role = f"{bowl_type} death bowler — go-to option in overs 17–20 for RR"
    elif "Powerplay" in bowl_phase_role:
        rr_role = f"{bowl_type} powerplay specialist — take early wickets for RR"
    elif "Middle" in bowl_phase_role:
        rr_role = f"{bowl_type} middle-overs controller — stem the flow in overs 7–15"
    else:
        rr_role = f"Versatile {bowl_type.lower()} option across all phases"

    card = {
        "rank":                        rank,
        "player_name":                 player_name,
        "player_id":                   pid,
        "teams":                       ", ".join(teams[:3]),
        "bowling_style":               bowl_style,
        "pace_or_spin":                bowl_type,
        "primary_phase_role":          bowl_phase_role,
        "bowling_pii":                 _fmt(pii, ".1f"),
        "reliability_score":           _fmt(reliability, ".2f"),
        "reliability_evidence":        evidence,
        "pressure_balls_bowled":       int(hp_balls) if pd.notna(hp_balls) else "N/A",
        "pressure_wickets":            int(hp_wkts) if pd.notna(hp_wkts) else "N/A",
        "normal_economy":              _fmt(norm_econ, ".2f"),
        "pressure_economy":            _fmt(hp_econ, ".2f"),
        "pressure_economy_delta":      _fmt(econ_delta, "+.2f"),
        "pressure_dot_pct":            _fmt(hp_dot, ".1f"),
        "pressure_wicket_rate":        _fmt(hp_wktr, ".2f"),
        "pressure_boundary_pct":       _fmt(hp_bnd, ".1f"),
        "pressure_boundary_prevention":_fmt(bp_prev, "+.1f"),
        "consistency_score":           _fmt(conscore, ".0f") + "/100",
        "pct_spells_below_8_econ":     _fmt(pct_8, ".0f") + "%",
        "strengths":                   " | ".join(strengths) if strengths else "See profile",
        "weaknesses":                  " | ".join(weaknesses) if weaknesses else "None identified",
        "rr_squad_role":               rr_role,
    }

    return card


def build_scouting_report(
    top10_bat: pd.DataFrame,
    top10_bowl: pd.DataFrame,
    batting_role_df: pd.DataFrame,
    bowling_role_df: pd.DataFrame,
    ctpm: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build complete scouting DataFrames for top batters and bowlers.

    Returns
    -------
    (batting_scouting_df, bowling_scouting_df)
    """
    bat_cards = []
    for _, row in top10_bat.iterrows():
        card = generate_batter_scouting_card(row, batting_role_df, ctpm)
        bat_cards.append(card)

    bowl_cards = []
    for _, row in top10_bowl.iterrows():
        card = generate_bowler_scouting_card(row, bowling_role_df, ctpm)
        bowl_cards.append(card)

    return pd.DataFrame(bat_cards), pd.DataFrame(bowl_cards)
