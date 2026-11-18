"""
ranking_engine.py
=================
Builds the Pressure Intelligence Index (PII) scoring system for
batters and bowlers, and produces the final Top 10 rankings.

BATTING PII (0–100)
-------------------
Components and weights:
  1. Pressure Strike Rate (reliability-adjusted)   — 30%
     Higher SR under pressure = better
  2. Pressure SR Delta (vs normal)                 — 20%
     Improves under pressure = good signal
  3. Pressure Average (reliability-adjusted)       — 15%
     Scoring AND surviving under pressure
  4. Boundary Efficiency under pressure            — 15%
     Scoring at a high rate from boundaries
  5. Pressure Survival Rate                        — 10%
     Not getting out under pressure
  6. Batting Consistency Score                     — 10%
     Repeating performance match after match

BOWLING PII (0–100)
-------------------
  1. Pressure Economy (reliability-adjusted, inverted)  — 30%
     Lower economy = higher score
  2. Pressure Economy Delta (normal − pressure econ)   — 20%
     Gets more economical under pressure = good
  3. Pressure Dot Ball %                               — 20%
     Restricting scoring under pressure
  4. Pressure Wicket Rate                              — 15%
     Taking wickets when it matters
  5. Boundary Prevention under pressure                — 10%
     Keeping boundaries down
  6. Bowling Consistency Score                         — 5%
     Reliability across matches

Weight Justification:
  Batting: SR under pressure is the most direct measure of run-scoring
  effectiveness. Delta captures improvement — a player who raises their
  game under pressure is more valuable than one who maintains average.
  Bowling: Economy under pressure and its delta are dominant because
  restricting runs is the primary job of a bowler defending a target.
  Dot % is weighted highly as it compounds pressure on batters.

Sensitivity analysis: weights are stored as a dict so they can be
varied in tests (see test_ranking_engine.py).
"""

import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Scoring weights (adjustable for sensitivity analysis)
# ---------------------------------------------------------------------------

BATTING_WEIGHTS = {
    "adj_hp_strike_rate":          0.30,
    "pressure_sr_delta":           0.20,
    "adj_hp_batting_avg":          0.15,
    "hp_boundary_pct":             0.15,
    "pressure_survival_rate_norm": 0.10,
    "batting_consistency_score":   0.10,
}

BOWLING_WEIGHTS = {
    "hp_economy_inverted":         0.30,
    "pressure_economy_delta":      0.20,
    "hp_dot_pct":                  0.20,
    "hp_wicket_rate":              0.15,
    "pressure_boundary_prevention":0.10,
    "bowling_consistency_score":   0.05,
}


# ---------------------------------------------------------------------------
# Normalise a series to 0–100 (min-max scaling)
# ---------------------------------------------------------------------------

def _minmax(s: pd.Series, invert: bool = False) -> pd.Series:
    """
    Normalise series to [0, 100].
    If invert=True, lower original values map to higher scores.
    """
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(50.0, index=s.index)
    normalised = (s - lo) / (hi - lo) * 100
    if invert:
        normalised = 100 - normalised
    return normalised


# ---------------------------------------------------------------------------
# Build Batting PII
# ---------------------------------------------------------------------------

def build_batting_pii(
    batting_metrics: pd.DataFrame,
    consistency_df: pd.DataFrame,
    name_map: Optional[dict] = None,
    weights: dict = BATTING_WEIGHTS,
) -> pd.DataFrame:
    """
    Compute Batting PII scores for all qualified batters.

    Parameters
    ----------
    batting_metrics : output of reliability_model.apply_batting_reliability()
    consistency_df  : output of consistency_model.build_batting_consistency()
    name_map        : dict {player_id → player_name}
    weights         : weight dict (can be overridden for sensitivity analysis)

    Returns
    -------
    DataFrame with columns: player_id, player_name, batting_pii, rank, ...
    """
    df = batting_metrics.merge(
        consistency_df[["player_id", "batting_consistency_score",
                        "bat_perf_quadrant", "hp_match_count",
                        "pct_above_100", "pct_above_80"]],
        on="player_id",
        how="left"
    ).copy()

    # --- Fill missing consistency for players with insufficient match data ---
    pop_consistency = consistency_df["batting_consistency_score"].mean()
    df["batting_consistency_score"] = df["batting_consistency_score"].fillna(
        pop_consistency * 0.5   # penalise insufficient match data
    )

    # --- Survival rate normalisation (convert to 0–100) ---
    df["pressure_survival_rate_norm"] = df["pressure_survival_rate"] * 100

    # --- Economy inverted: not applicable for batting; skip ---
    # Clamp pressure_sr_delta (can be very negative or positive)
    df["pressure_sr_delta_clamped"] = df["pressure_sr_delta"].clip(lower=-40, upper=40)

    # --- Min-max normalise each component ---
    components = {
        "adj_hp_strike_rate":          _minmax(df["adj_hp_strike_rate"]),
        "pressure_sr_delta":           _minmax(df["pressure_sr_delta_clamped"]),
        "adj_hp_batting_avg":          _minmax(df["adj_hp_batting_avg"].fillna(
                                           df["adj_hp_batting_avg"].median())),
        "hp_boundary_pct":             _minmax(df["hp_boundary_pct"].fillna(0)),
        "pressure_survival_rate_norm": _minmax(df["pressure_survival_rate_norm"]),
        "batting_consistency_score":   df["batting_consistency_score"],
    }

    # --- Weighted sum → Batting PII ---
    pii = pd.Series(0.0, index=df.index)
    for comp_name, comp_scores in components.items():
        pii += weights[comp_name] * comp_scores

    df["batting_pii"] = pii.clip(lower=0, upper=100).round(2)

    # --- Apply reliability penalty for low-evidence players ---
    # Players with reliability < 0.4 get a 20% penalty on their PII
    low_rel = df["batting_reliability"] < 0.4
    df.loc[low_rel, "batting_pii"] = (
        df.loc[low_rel, "batting_pii"] * df.loc[low_rel, "batting_reliability"] / 0.4
    ).clip(lower=0).round(2)

    # --- Add player names ---
    if name_map:
        df["player_name"] = df["player_id"].map(name_map).fillna("Unknown")

    # --- Rank ---
    df = df.sort_values("batting_pii", ascending=False).reset_index(drop=True)
    df["batting_rank"] = df.index + 1

    return df


# ---------------------------------------------------------------------------
# Build Bowling PII
# ---------------------------------------------------------------------------

def build_bowling_pii(
    bowling_metrics: pd.DataFrame,
    consistency_df: pd.DataFrame,
    name_map: Optional[dict] = None,
    weights: dict = BOWLING_WEIGHTS,
) -> pd.DataFrame:
    """
    Compute Bowling PII scores for all qualified bowlers.
    """
    df = bowling_metrics.merge(
        consistency_df[["player_id", "bowling_consistency_score",
                        "bowl_perf_quadrant", "hp_match_count",
                        "pct_below_8", "pct_below_10"]],
        on="player_id",
        how="left"
    ).copy()

    pop_consistency = consistency_df["bowling_consistency_score"].mean()
    df["bowling_consistency_score"] = df["bowling_consistency_score"].fillna(
        pop_consistency * 0.5
    )

    # --- Economy inverted: lower economy → higher score ---
    # Cap at 20 rpo to avoid outliers skewing normalisation
    df["hp_economy_capped"] = df["hp_economy"].clip(upper=20)
    df["hp_economy_inverted"] = _minmax(df["hp_economy_capped"], invert=True)

    # Clamp pressure_economy_delta
    df["pressure_economy_delta_clamped"] = df["pressure_economy_delta"].clip(
        lower=-5, upper=5)

    # Clamp boundary prevention
    df["pressure_bp_clamped"] = df["pressure_boundary_prevention"].clip(
        lower=-15, upper=15)

    # Clamp wicket rate
    df["hp_wicket_rate_capped"] = df["hp_wicket_rate"].fillna(0).clip(upper=5)

    # --- Min-max normalise ---
    components = {
        "hp_economy_inverted":         df["hp_economy_inverted"],
        "pressure_economy_delta":      _minmax(df["pressure_economy_delta_clamped"]),
        "hp_dot_pct":                  _minmax(df["hp_dot_pct"].fillna(0)),
        "hp_wicket_rate":              _minmax(df["hp_wicket_rate_capped"]),
        "pressure_boundary_prevention":_minmax(df["pressure_bp_clamped"]),
        "bowling_consistency_score":   df["bowling_consistency_score"],
    }

    pii = pd.Series(0.0, index=df.index)
    for comp_name, comp_scores in components.items():
        pii += weights[comp_name] * comp_scores

    df["bowling_pii"] = pii.clip(lower=0, upper=100).round(2)

    # --- Reliability penalty ---
    low_rel = df["bowling_reliability"] < 0.4
    df.loc[low_rel, "bowling_pii"] = (
        df.loc[low_rel, "bowling_pii"] * df.loc[low_rel, "bowling_reliability"] / 0.4
    ).clip(lower=0).round(2)

    if name_map:
        df["player_name"] = df["player_id"].map(name_map).fillna("Unknown")

    df = df.sort_values("bowling_pii", ascending=False).reset_index(drop=True)
    df["bowling_rank"] = df.index + 1

    return df


# ---------------------------------------------------------------------------
# Extract Top 10 with no duplicates
# ---------------------------------------------------------------------------

def get_top_10_batters(batting_pii_df: pd.DataFrame) -> pd.DataFrame:
    """Return top 10 unique players by Batting PII."""
    top = (
        batting_pii_df
        .drop_duplicates(subset=["player_id"])
        .nlargest(10, "batting_pii")
        .reset_index(drop=True)
    )
    top["rank"] = top.index + 1
    return top


def get_top_10_bowlers(bowling_pii_df: pd.DataFrame) -> pd.DataFrame:
    """Return top 10 unique players by Bowling PII."""
    top = (
        bowling_pii_df
        .drop_duplicates(subset=["player_id"])
        .nlargest(10, "bowling_pii")
        .reset_index(drop=True)
    )
    top["rank"] = top.index + 1
    return top


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def sensitivity_analysis(batting_metrics, consistency_bat,
                          bowling_metrics, consistency_bowl,
                          name_map) -> dict:
    """
    Test alternative weight configurations and check rank stability.

    Returns a dict mapping config_name → top 5 player names.
    """
    configs = {
        "default": (BATTING_WEIGHTS, BOWLING_WEIGHTS),
        "sr_heavy": (
            {**BATTING_WEIGHTS, "adj_hp_strike_rate": 0.45,
             "pressure_sr_delta": 0.15, "adj_hp_batting_avg": 0.10},
            BOWLING_WEIGHTS
        ),
        "consistency_heavy": (
            {**BATTING_WEIGHTS, "batting_consistency_score": 0.25,
             "adj_hp_strike_rate": 0.20},
            {**BOWLING_WEIGHTS, "bowling_consistency_score": 0.15,
             "hp_economy_inverted": 0.25}
        ),
        "delta_heavy": (
            {**BATTING_WEIGHTS, "pressure_sr_delta": 0.35,
             "adj_hp_strike_rate": 0.20},
            {**BOWLING_WEIGHTS, "pressure_economy_delta": 0.35,
             "hp_economy_inverted": 0.20}
        ),
    }

    results = {}
    for config_name, (bat_w, bowl_w) in configs.items():
        try:
            bat_pii = build_batting_pii(batting_metrics, consistency_bat,
                                        name_map, weights=bat_w)
            bowl_pii = build_bowling_pii(bowling_metrics, consistency_bowl,
                                         name_map, weights=bowl_w)
            results[config_name] = {
                "top5_bat": bat_pii.head(5)["player_name"].tolist()
                            if "player_name" in bat_pii else [],
                "top5_bowl": bowl_pii.head(5)["player_name"].tolist()
                             if "player_name" in bowl_pii else [],
            }
        except Exception as e:
            results[config_name] = {"error": str(e)}

    return results
