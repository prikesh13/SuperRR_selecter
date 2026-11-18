"""
batting_model.py
================
Computes batting performance metrics separately for:
  - High Pressure situations
  - Non-High-Pressure (normal) situations

Then derives Pressure Delta metrics.

All metrics are calculated per player across all their appearances.
Minimum sample thresholds are enforced BEFORE surfacing results.
"""

import numpy as np
import pandas as pd

# Minimum balls faced under pressure to include a batter in pressure rankings
MIN_PRESSURE_BALLS = 100
MIN_PRESSURE_INNINGS = 5


def compute_batting_metrics(master_bbb: pd.DataFrame,
                             min_pressure_balls: int = MIN_PRESSURE_BALLS,
                             min_pressure_innings: int = MIN_PRESSURE_INNINGS
                             ) -> pd.DataFrame:
    """
    Compute comprehensive batting metrics for every player across all matches.

    Returns a player-level DataFrame with normal and pressure metrics.
    """
    df = master_bbb.copy()

    # Only look at deliveries where this player is the striker
    # (master_bbb contains all deliveries — filter to striker perspective)
    bat_df = df.copy()

    # Dismissed flag: wicket where the dismissed player is the striker
    bat_df["batter_dismissed"] = (
        bat_df["is_wicket"] &
        (
            bat_df["dismissed_player_id"].fillna(-1).astype(int) ==
            bat_df["striker_id"].astype(int)
        )
    )

    # Classify pressure from the batter's perspective
    bat_df["is_high_pressure"] = bat_df["bat_pressure_label"] == "High"
    bat_df["is_normal"] = bat_df["bat_pressure_label"] == "Low"

    def _aggregate(subset, label):
        """Aggregate batting stats for a given subset of deliveries."""
        cols = [
            "player_id", f"{label}_runs", f"{label}_balls", f"{label}_matches",
            f"{label}_fours", f"{label}_sixes", f"{label}_boundaries", f"{label}_dots",
            f"{label}_dismissals", f"{label}_innings", f"{label}_strike_rate",
            f"{label}_batting_avg", f"{label}_boundary_pct", f"{label}_four_pct",
            f"{label}_six_pct", f"{label}_dot_pct", f"{label}_dismissal_rate"
        ]
        if subset.empty:
            return pd.DataFrame(columns=cols)

        grp = subset.groupby("striker_id")

        agg = grp.agg(
            total_runs=("batter_runs", "sum"),
            balls_faced=("is_legal", "sum"),
            innings_count=("innings", lambda x: subset.loc[x.index].groupby(
                ["match_id", "innings"]).ngroups),
            matches_played=("match_id", "nunique"),
            fours=("is_four", "sum"),
            sixes=("is_six", "sum"),
            boundaries=("is_boundary", "sum"),
            dots=("is_batter_dot", "sum"),
            dismissals=("batter_dismissed", "sum"),
        ).reset_index()

        agg.rename(columns={"striker_id": "player_id"}, inplace=True)

        # Innings count (number of innings where player faced ≥1 ball)
        innings_map = (
            subset[subset["is_legal"]]
            .groupby("striker_id")[["match_id", "innings"]]
            .apply(lambda x: x.drop_duplicates().shape[0])
            .reset_index()
            .rename(columns={"striker_id": "player_id", 0: "innings_count"})
        )
        agg = agg.drop(columns=["innings_count"], errors="ignore")
        agg = agg.merge(innings_map, on="player_id", how="left")

        # Derived
        agg[f"{label}_strike_rate"] = np.where(
            agg["balls_faced"] > 0,
            agg["total_runs"] / agg["balls_faced"] * 100, np.nan)

        agg[f"{label}_batting_avg"] = np.where(
            agg["dismissals"] > 0,
            agg["total_runs"] / agg["dismissals"], np.nan)

        agg[f"{label}_boundary_pct"] = np.where(
            agg["balls_faced"] > 0,
            agg["boundaries"] / agg["balls_faced"] * 100, np.nan)

        agg[f"{label}_four_pct"] = np.where(
            agg["balls_faced"] > 0,
            agg["fours"] / agg["balls_faced"] * 100, np.nan)

        agg[f"{label}_six_pct"] = np.where(
            agg["balls_faced"] > 0,
            agg["sixes"] / agg["balls_faced"] * 100, np.nan)

        agg[f"{label}_dot_pct"] = np.where(
            agg["balls_faced"] > 0,
            agg["dots"] / agg["balls_faced"] * 100, np.nan)

        agg[f"{label}_dismissal_rate"] = np.where(
            agg["balls_faced"] > 0,
            agg["dismissals"] / agg["innings_count"], np.nan)

        # Rename generic columns with label prefix
        agg = agg.rename(columns={
            "total_runs":     f"{label}_runs",
            "balls_faced":    f"{label}_balls",
            "matches_played": f"{label}_matches",
            "fours":          f"{label}_fours",
            "sixes":          f"{label}_sixes",
            "boundaries":     f"{label}_boundaries",
            "dots":           f"{label}_dots",
            "dismissals":     f"{label}_dismissals",
            "innings_count":  f"{label}_innings",
        })

        return agg

    # --- HIGH PRESSURE metrics ---
    hp_df = bat_df[bat_df["is_high_pressure"] & bat_df["is_legal"]].copy()
    hp_metrics = _aggregate(hp_df, "hp")

    # --- NORMAL (Low) pressure metrics ---
    normal_df = bat_df[bat_df["is_normal"] & bat_df["is_legal"]].copy()
    normal_metrics = _aggregate(normal_df, "normal")

    # --- OVERALL metrics (all conditions) ---
    all_df = bat_df[bat_df["is_legal"]].copy()
    overall_metrics = _aggregate(all_df, "overall")

    # Merge all
    merged = hp_metrics.merge(normal_metrics, on="player_id", how="outer")
    merged = merged.merge(overall_metrics, on="player_id", how="outer")

    # --- Apply sample size filter ---
    merged = merged[
        merged["hp_balls"].fillna(0) >= min_pressure_balls
    ].copy()

    # --- Pressure Delta metrics ---
    merged["pressure_sr_delta"] = (
        merged["hp_strike_rate"] - merged["normal_strike_rate"]
    )
    merged["pressure_avg_delta"] = (
        merged["hp_batting_avg"] - merged["normal_batting_avg"]
    )
    merged["pressure_boundary_efficiency"] = (
        merged["hp_boundary_pct"] / merged["normal_boundary_pct"].replace(0, np.nan)
    )
    merged["pressure_dot_increase"] = (
        merged["hp_dot_pct"] - merged["normal_dot_pct"]
    )

    # --- Pressure Survival Rate ---
    # How often does the batter survive a pressure innings (not get out)?
    merged["pressure_survival_rate"] = (
        1 - merged["hp_dismissal_rate"].fillna(0)
    ).clip(lower=0)

    return merged


def compute_match_level_batting_consistency(master_bbb: pd.DataFrame,
                                             min_matches: int = 5
                                             ) -> pd.DataFrame:
    """
    For each player, compute match-level batting consistency under pressure.

    Returns a DataFrame with one row per player containing:
      - std_dev of pressure strike rate across matches
      - fraction of pressure matches where SR > 100
      - consistency score (inverse of CV)
    """
    df = master_bbb.copy()
    df = df[df["is_legal"]].copy()

    # Tag high-pressure deliveries
    df["is_hp"] = df["bat_pressure_label"] == "High"

    # Dismissed flag
    df["batter_dismissed"] = (
        df["is_wicket"] &
        (df["dismissed_player_id"].fillna(-1).astype(int) ==
         df["striker_id"].astype(int))
    )

    # Per (player, match) high-pressure stats
    hp_match = (
        df[df["is_hp"]]
        .groupby(["striker_id", "match_id"])
        .agg(
            hp_runs=("batter_runs", "sum"),
            hp_balls=("is_legal", "sum"),
            hp_dismissals=("batter_dismissed", "sum"),
        )
        .reset_index()
    )
    hp_match = hp_match[hp_match["hp_balls"] >= 6].copy()  # min 6 pressure balls in match
    hp_match["match_hp_sr"] = hp_match["hp_runs"] / hp_match["hp_balls"] * 100

    # Player-level consistency
    consistency = (
        hp_match.groupby("striker_id")
        .agg(
            hp_match_count=("match_id", "nunique"),
            hp_sr_mean=("match_hp_sr", "mean"),
            hp_sr_std=("match_hp_sr", "std"),
            pct_above_100=("match_hp_sr", lambda x: (x > 100).mean() * 100),
            pct_above_80=("match_hp_sr", lambda x: (x > 80).mean() * 100),
        )
        .reset_index()
        .rename(columns={"striker_id": "player_id"})
    )

    # Coefficient of variation (lower = more consistent)
    consistency["hp_sr_cv"] = (
        consistency["hp_sr_std"] / consistency["hp_sr_mean"].replace(0, np.nan)
    )

    # Consistency score: 100 - CV*100 (higher = more consistent), capped
    consistency["batting_consistency_score"] = (
        (1 - consistency["hp_sr_cv"].fillna(1)).clip(lower=0) * 100
    )

    consistency = consistency[
        consistency["hp_match_count"] >= min_matches
    ].copy()

    return consistency


def classify_batting_role(master_bbb: pd.DataFrame,
                           playing_eleven: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each player's batting role based on actual batting order
    observed across matches.

    Roles:
      Opener       : avg batting order ≤ 2
      Top-order    : avg batting order 3–4
      Middle-order : avg batting order 5–7
      Finisher     : avg batting order ≥ 8
    """
    df = master_bbb.copy()

    # Use striker batting order per delivery (most common per player)
    order_map = (
        df[df["is_legal"]]
        .groupby("striker_id")["batting_order"]
        .median()
        .reset_index()
        .rename(columns={"striker_id": "player_id", "batting_order": "avg_bat_order"})
    )

    def _role(order):
        if order <= 2:
            return "Opener"
        elif order <= 4:
            return "Top-order"
        elif order <= 7:
            return "Middle-order"
        else:
            return "Finisher"

    order_map["batting_role"] = order_map["avg_bat_order"].apply(_role)
    return order_map
