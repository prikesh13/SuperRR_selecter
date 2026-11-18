"""
consistency_model.py
====================
Builds a composite consistency rating for batters and bowlers.

Consistency is measured as performance STABILITY across multiple matches,
not just aggregate volume.

Four quadrant classification:
  - High Performance + High Consistency  → "Elite"
  - High Performance + Low Consistency   → "Streaky"
  - Average Performance + High Consistency → "Reliable"
  - Low Performance + Low Consistency    → "Inconsistent"
"""

import numpy as np
import pandas as pd


def build_batting_consistency(master_bbb: pd.DataFrame,
                               min_matches: int = 5) -> pd.DataFrame:
    """
    Per-player batting consistency under high pressure.

    Metrics:
      - hp_sr_std        : std deviation of pressure SR across matches
      - hp_sr_cv         : coefficient of variation (std/mean)
      - pct_above_100    : % of pressure innings with SR > 100
      - pct_above_80     : % of pressure innings with SR > 80
      - consistency_score: 100-point scale (higher = more consistent)
      - perf_quadrant    : Elite / Streaky / Reliable / Inconsistent
    """
    df = master_bbb.copy()
    df = df[(df["bat_pressure_label"] == "High") & df["is_legal"]].copy()

    # Dismissed flag
    df["batter_dismissed"] = (
        df["is_wicket"] &
        (df["dismissed_player_id"].fillna(-1).astype(int) ==
         df["striker_id"].astype(int))
    )

    # Per (player, match, innings) aggregation
    match_inn = (
        df.groupby(["striker_id", "match_id", "innings"])
        .agg(
            hp_runs=("batter_runs", "sum"),
            hp_balls=("is_legal", "sum"),
            hp_dismissals=("batter_dismissed", "sum"),
        )
        .reset_index()
    )
    match_inn = match_inn[match_inn["hp_balls"] >= 6].copy()
    match_inn["match_hp_sr"] = match_inn["hp_runs"] / match_inn["hp_balls"] * 100

    player_cons = (
        match_inn.groupby("striker_id")
        .agg(
            hp_match_count=("match_id", "nunique"),
            hp_sr_mean=("match_hp_sr", "mean"),
            hp_sr_std=("match_hp_sr", "std"),
            hp_sr_median=("match_hp_sr", "median"),
            hp_runs_total=("hp_runs", "sum"),
            hp_balls_total=("hp_balls", "sum"),
            pct_above_100=("match_hp_sr", lambda x: (x > 100).mean() * 100),
            pct_above_80=("match_hp_sr", lambda x: (x > 80).mean() * 100),
            pct_above_50=("match_hp_sr", lambda x: (x > 50).mean() * 100),
        )
        .reset_index()
        .rename(columns={"striker_id": "player_id"})
    )

    # CV (lower = more consistent)
    player_cons["hp_sr_cv"] = (
        player_cons["hp_sr_std"] /
        player_cons["hp_sr_mean"].replace(0, np.nan)
    ).fillna(1.0)

    # Consistency score (0–100): lower CV = higher score
    # Anchored so CV = 0 → 100, CV = 1 → 0
    player_cons["batting_consistency_score"] = (
        (1 - player_cons["hp_sr_cv"].clip(upper=1.5) / 1.5) * 100
    ).clip(lower=0)

    # Filter minimum matches
    player_cons = player_cons[player_cons["hp_match_count"] >= min_matches].copy()

    # Quadrant classification
    sr_median  = player_cons["hp_sr_mean"].median()
    cons_median = player_cons["batting_consistency_score"].median()

    def _bat_quadrant(row):
        high_perf = row["hp_sr_mean"] >= sr_median
        high_cons = row["batting_consistency_score"] >= cons_median
        if high_perf and high_cons:
            return "Elite"
        elif high_perf and not high_cons:
            return "Streaky"
        elif not high_perf and high_cons:
            return "Reliable"
        else:
            return "Inconsistent"

    player_cons["bat_perf_quadrant"] = player_cons.apply(_bat_quadrant, axis=1)

    return player_cons


def build_bowling_consistency(master_bbb: pd.DataFrame,
                               min_matches: int = 5) -> pd.DataFrame:
    """
    Per-player bowling consistency under high pressure.

    Metrics:
      - hp_econ_std      : std deviation of pressure economy across matches
      - hp_econ_cv       : coefficient of variation
      - pct_below_8      : % of pressure spells with economy < 8
      - consistency_score: 100-point scale (higher = more consistent)
      - perf_quadrant    : Elite / Streaky / Reliable / Inconsistent
    """
    df = master_bbb.copy()
    df = df[df["bowl_pressure_label"] == "High"].copy()

    # Per (bowler, match) aggregation
    match_spell = (
        df.groupby(["bowler_id", "match_id"])
        .agg(
            hp_runs=("bowler_conceded_runs", "sum"),
            hp_balls=("is_legal", "sum"),
            hp_wickets=("is_bowler_wicket", "sum"),
        )
        .reset_index()
    )
    match_spell = match_spell[match_spell["hp_balls"] >= 6].copy()
    match_spell["match_hp_economy"] = (
        match_spell["hp_runs"] / (match_spell["hp_balls"] / 6)
    )

    player_cons = (
        match_spell.groupby("bowler_id")
        .agg(
            hp_match_count=("match_id", "nunique"),
            hp_econ_mean=("match_hp_economy", "mean"),
            hp_econ_std=("match_hp_economy", "std"),
            hp_econ_median=("match_hp_economy", "median"),
            hp_wickets_total=("hp_wickets", "sum"),
            hp_balls_total=("hp_balls", "sum"),
            pct_below_8=("match_hp_economy", lambda x: (x < 8).mean() * 100),
            pct_below_10=("match_hp_economy", lambda x: (x < 10).mean() * 100),
            pct_above_14=("match_hp_economy", lambda x: (x > 14).mean() * 100),
        )
        .reset_index()
        .rename(columns={"bowler_id": "player_id"})
    )

    player_cons["hp_econ_cv"] = (
        player_cons["hp_econ_std"] /
        player_cons["hp_econ_mean"].replace(0, np.nan)
    ).fillna(1.0)

    player_cons["bowling_consistency_score"] = (
        (1 - player_cons["hp_econ_cv"].clip(upper=1.5) / 1.5) * 100
    ).clip(lower=0)

    player_cons = player_cons[player_cons["hp_match_count"] >= min_matches].copy()

    # Quadrant (for bowling: lower economy = higher performance)
    econ_median = player_cons["hp_econ_mean"].median()
    cons_median = player_cons["bowling_consistency_score"].median()

    def _bowl_quadrant(row):
        # Lower economy = better performance
        good_econ = row["hp_econ_mean"] <= econ_median
        high_cons = row["bowling_consistency_score"] >= cons_median
        if good_econ and high_cons:
            return "Elite"
        elif good_econ and not high_cons:
            return "Streaky"
        elif not good_econ and high_cons:
            return "Reliable"
        else:
            return "Inconsistent"

    player_cons["bowl_perf_quadrant"] = player_cons.apply(_bowl_quadrant, axis=1)

    return player_cons
