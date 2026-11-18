"""
bowling_model.py
================
Computes bowling performance metrics separately for:
  - High Pressure situations (bowl_pressure_label == 'High')
  - Normal (Low) situations

Then derives Pressure Delta and bowling-specific pressure metrics.

Key rules applied:
  - Run-outs NOT credited as bowler wickets.
  - Byes and leg-byes NOT charged to bowler.
  - Wides and no-balls counted as extra deliveries (not legal balls).
  - Minimum sample thresholds enforced.
"""

import numpy as np
import pandas as pd

# Minimum legal balls bowled under pressure to include in rankings
MIN_PRESSURE_BALLS = 60
MIN_PRESSURE_SPELLS = 5   # distinct match appearances under pressure


def compute_bowling_metrics(master_bbb: pd.DataFrame,
                             min_pressure_balls: int = MIN_PRESSURE_BALLS,
                             min_pressure_spells: int = MIN_PRESSURE_SPELLS
                             ) -> pd.DataFrame:
    """
    Compute comprehensive bowling metrics for every bowler.
    Returns a player-level DataFrame with normal and pressure metrics.
    """
    df = master_bbb.copy()

    # Bowling pressure label already on each delivery
    df["is_high_pressure"] = df["bowl_pressure_label"] == "High"
    df["is_normal"] = df["bowl_pressure_label"] == "Low"

    def _aggregate_bowling(subset, label):
        """Aggregate bowling stats for a given subset of deliveries."""
        cols = [
            "player_id", f"{label}_economy", f"{label}_wickets", f"{label}_balls",
            f"{label}_runs_conceded", f"{label}_matches", f"{label}_bowling_avg",
            f"{label}_bowling_sr", f"{label}_dot_pct", f"{label}_boundary_pct",
            f"{label}_four_pct", f"{label}_six_pct", f"{label}_wide_rate",
            f"{label}_noball_rate", f"{label}_wicket_rate"
        ]
        if subset.empty:
            return pd.DataFrame(columns=cols)

        grp = subset.groupby("bowler_id")

        agg = grp.agg(
            runs_conceded=("bowler_conceded_runs", "sum"),
            legal_balls=("is_legal", "sum"),
            total_balls=("is_legal", "count"),
            wickets=("is_bowler_wicket", "sum"),
            fours_conceded=("is_four", "sum"),
            sixes_conceded=("is_six", "sum"),
            wides=("is_wide", "sum"),
            noballs=("is_noball", "sum"),
            dots=("is_batter_dot", "sum"),
            matches=("match_id", "nunique"),
        ).reset_index()

        agg.rename(columns={"bowler_id": "player_id"}, inplace=True)

        agg["overs_bowled"] = agg["legal_balls"] / 6

        agg[f"{label}_economy"] = np.where(
            agg["legal_balls"] > 0,
            agg["runs_conceded"] / agg["overs_bowled"], np.nan)

        agg[f"{label}_wickets"] = agg["wickets"]
        agg[f"{label}_balls"] = agg["legal_balls"]
        agg[f"{label}_runs_conceded"] = agg["runs_conceded"]
        agg[f"{label}_matches"] = agg["matches"]

        agg[f"{label}_bowling_avg"] = np.where(
            agg["wickets"] > 0,
            agg["runs_conceded"] / agg["wickets"], np.nan)

        agg[f"{label}_bowling_sr"] = np.where(
            agg["wickets"] > 0,
            agg["legal_balls"] / agg["wickets"], np.nan)

        agg[f"{label}_dot_pct"] = np.where(
            agg["legal_balls"] > 0,
            agg["dots"] / agg["legal_balls"] * 100, np.nan)

        agg[f"{label}_boundary_pct"] = np.where(
            agg["legal_balls"] > 0,
            (agg["fours_conceded"] + agg["sixes_conceded"]) / agg["legal_balls"] * 100,
            np.nan)

        agg[f"{label}_four_pct"] = np.where(
            agg["legal_balls"] > 0,
            agg["fours_conceded"] / agg["legal_balls"] * 100, np.nan)

        agg[f"{label}_six_pct"] = np.where(
            agg["legal_balls"] > 0,
            agg["sixes_conceded"] / agg["legal_balls"] * 100, np.nan)

        agg[f"{label}_wide_rate"] = np.where(
            agg["total_balls"] > 0,
            agg["wides"] / agg["total_balls"] * 100, np.nan)

        agg[f"{label}_noball_rate"] = np.where(
            agg["total_balls"] > 0,
            agg["noballs"] / agg["total_balls"] * 100, np.nan)

        # Wicket rate per over
        agg[f"{label}_wicket_rate"] = np.where(
            agg["legal_balls"] > 0,
            agg["wickets"] / agg["overs_bowled"], np.nan)

        # Drop generic columns
        for col in ["runs_conceded", "legal_balls", "total_balls", "wickets",
                    "fours_conceded", "sixes_conceded", "wides", "noballs",
                    "dots", "overs_bowled", "matches"]:
            agg.drop(columns=[col], inplace=True, errors="ignore")

        return agg

    # --- HIGH PRESSURE metrics ---
    hp_df = df[df["is_high_pressure"]].copy()
    hp_metrics = _aggregate_bowling(hp_df, "hp")

    # --- NORMAL (Low) pressure metrics ---
    normal_df = df[df["is_normal"]].copy()
    normal_metrics = _aggregate_bowling(normal_df, "normal")

    # --- OVERALL ---
    all_metrics = _aggregate_bowling(df, "overall")

    # Merge
    merged = hp_metrics.merge(normal_metrics, on="player_id", how="outer")
    merged = merged.merge(all_metrics, on="player_id", how="outer")

    # Apply minimum sample filter
    merged = merged[
        merged["hp_balls"].fillna(0) >= min_pressure_balls
    ].copy()

    # --- Pressure Delta metrics ---
    # For economy: LOWER is better for bowler, so delta = normal - hp
    # Positive delta = bowler is MORE economical under pressure (good)
    merged["pressure_economy_delta"] = (
        merged["normal_economy"] - merged["hp_economy"]
    )

    # For dot ball %: HIGHER is better for bowler
    merged["pressure_dot_delta"] = (
        merged["hp_dot_pct"] - merged["normal_dot_pct"]
    )

    # For boundary %: LOWER is better for bowler
    merged["pressure_boundary_prevention"] = (
        merged["normal_boundary_pct"] - merged["hp_boundary_pct"]
    )

    # Wicket-taking improvement under pressure
    merged["pressure_wicket_rate_delta"] = (
        merged["hp_wicket_rate"] - merged["normal_wicket_rate"]
    )

    return merged


def compute_phase_bowling_metrics(master_bbb: pd.DataFrame) -> pd.DataFrame:
    """
    Compute economy and wicket rate by phase (Powerplay / Middle / Death)
    for each bowler under high pressure.
    """
    df = master_bbb.copy()
    df = df[df["bowl_pressure_label"] == "High"].copy()

    grp = df.groupby(["bowler_id", "phase"])

    phase_metrics = grp.agg(
        hp_runs=("bowler_conceded_runs", "sum"),
        hp_legal_balls=("is_legal", "sum"),
        hp_wickets=("is_bowler_wicket", "sum"),
        hp_dots=("is_batter_dot", "sum"),
        hp_boundaries=("is_boundary", "sum"),
        hp_matches=("match_id", "nunique"),
    ).reset_index()

    phase_metrics.rename(columns={"bowler_id": "player_id"}, inplace=True)

    phase_metrics["hp_overs"] = phase_metrics["hp_legal_balls"] / 6
    phase_metrics["hp_economy_phase"] = np.where(
        phase_metrics["hp_legal_balls"] > 0,
        phase_metrics["hp_runs"] / phase_metrics["hp_overs"], np.nan)

    phase_metrics["hp_dot_pct_phase"] = np.where(
        phase_metrics["hp_legal_balls"] > 0,
        phase_metrics["hp_dots"] / phase_metrics["hp_legal_balls"] * 100, np.nan)

    phase_metrics["hp_wicket_rate_phase"] = np.where(
        phase_metrics["hp_legal_balls"] > 0,
        phase_metrics["hp_wickets"] / phase_metrics["hp_overs"], np.nan)

    return phase_metrics


def compute_match_level_bowling_consistency(master_bbb: pd.DataFrame,
                                             min_matches: int = 5
                                             ) -> pd.DataFrame:
    """
    Per bowler, compute match-level consistency of economy under high pressure.
    """
    df = master_bbb.copy()
    df = df[df["bowl_pressure_label"] == "High"].copy()

    hp_match = (
        df.groupby(["bowler_id", "match_id"])
        .agg(
            hp_runs=("bowler_conceded_runs", "sum"),
            hp_balls=("is_legal", "sum"),
            hp_wickets=("is_bowler_wicket", "sum"),
        )
        .reset_index()
    )
    hp_match = hp_match[hp_match["hp_balls"] >= 6].copy()
    hp_match["match_hp_economy"] = hp_match["hp_runs"] / (hp_match["hp_balls"] / 6)

    consistency = (
        hp_match.groupby("bowler_id")
        .agg(
            hp_match_count=("match_id", "nunique"),
            hp_econ_mean=("match_hp_economy", "mean"),
            hp_econ_std=("match_hp_economy", "std"),
            pct_below_8=("match_hp_economy", lambda x: (x < 8).mean() * 100),
            pct_below_10=("match_hp_economy", lambda x: (x < 10).mean() * 100),
        )
        .reset_index()
        .rename(columns={"bowler_id": "player_id"})
    )

    consistency["hp_econ_cv"] = (
        consistency["hp_econ_std"] / consistency["hp_econ_mean"].replace(0, np.nan)
    )

    # Consistency score: 100 - CV*100 (higher = more consistent)
    consistency["bowling_consistency_score"] = (
        (1 - consistency["hp_econ_cv"].fillna(1)).clip(lower=0) * 100
    )

    consistency = consistency[
        consistency["hp_match_count"] >= min_matches
    ].copy()

    return consistency


def classify_bowling_role(master_bbb: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each bowler's primary phase specialisation based on
    where they bowl most of their legal deliveries.

    Roles:
      Powerplay specialist  : > 50% balls in overs 1–6
      Death specialist      : > 40% balls in overs 16–20
      Middle-over specialist: > 50% balls in overs 7–15
    """
    df = master_bbb.copy()
    df = df[df["is_legal"]].copy()

    df["is_pp"]     = df["overid"].between(1, 6)
    df["is_mid"]    = df["overid"].between(7, 15)
    df["is_death"]  = df["overid"].between(16, 20)

    phase_balls = (
        df.groupby("bowler_id")
        .agg(
            total_balls=("is_legal", "sum"),
            pp_balls=("is_pp", "sum"),
            mid_balls=("is_mid", "sum"),
            death_balls=("is_death", "sum"),
        )
        .reset_index()
        .rename(columns={"bowler_id": "player_id"})
    )

    phase_balls["pp_pct"]    = phase_balls["pp_balls"] / phase_balls["total_balls"]
    phase_balls["mid_pct"]   = phase_balls["mid_balls"] / phase_balls["total_balls"]
    phase_balls["death_pct"] = phase_balls["death_balls"] / phase_balls["total_balls"]

    def _role(row):
        if row["pp_pct"] > 0.50:
            return "Powerplay Specialist"
        elif row["death_pct"] > 0.40:
            return "Death Specialist"
        elif row["mid_pct"] > 0.50:
            return "Middle-over Specialist"
        else:
            return "All-phase Bowler"

    phase_balls["bowling_phase_role"] = phase_balls.apply(_role, axis=1)

    # Add pace/spin
    pace_spin = (
        df.groupby("bowler_id")["pace_or_spin"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
        .rename(columns={"bowler_id": "player_id", "pace_or_spin": "bowl_type"})
    )

    bowling_type_map = (
        df.groupby("bowler_id")["bowling_type"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
        .rename(columns={"bowler_id": "player_id"})
    )

    phase_balls = phase_balls.merge(pace_spin, on="player_id", how="left")
    phase_balls = phase_balls.merge(bowling_type_map, on="player_id", how="left")

    return phase_balls
