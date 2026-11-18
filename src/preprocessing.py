"""
preprocessing.py
================
Cleans, enriches and integrates all six raw datasets into a single
master ball-by-ball DataFrame plus player-match summary tables.

Design principles (from Phase 1 findings):
  - `runs` column already contains batter runs only.
  - Legal delivery = not wide and not no-ball (Overstep).
  - Byes and leg-byes charged to team, not bowler, not batter.
  - Run-outs credited to no bowler.
  - Super-over deliveries (innings > 2) excluded from main analysis.
  - Abandoned matches (no BBB rows) automatically excluded.
  - Emerging-team Asia Cup competitions excluded.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Competition IDs to EXCLUDE (emerging/junior events, not relevant to IPL 2027)
# ---------------------------------------------------------------------------
EXCLUDE_COMP_IDS = {2670, 3289}   # Emerging Teams Asia Cup 2024 & 2025

# IPL competition IDs (match_format = 'T20')
IPL_COMP_IDS = {1358, 1715, 2193, 2725, 3298}

# Bowling types → pace or spin flag (supplement pace_or_spin column)
PACE_STYLES = {
    "Right-arm Fast", "Right-arm fast-medium", "Right-arm medium-fast",
    "Right-arm medium", "Left-arm Fast", "Left-arm fast-medium",
    "Left-arm medium-fast", "Left-arm medium",
}
SPIN_STYLES = {
    "Right-arm Leg-Break", "Right-arm offbreak", "Slow left-arm orthodox",
    "Left-arm Chinaman",
}

# Match phases by over number (1-indexed overid)
PHASE_MAP = {
    **{o: "Powerplay" for o in range(1, 7)},
    **{o: "Middle" for o in range(7, 16)},
    **{o: "Death" for o in range(16, 21)},
}

# Non-bowler dismissal types
NON_BOWLER_DISMISSALS = {"RunOut", "Retire Out", "Obstructing the field",
                          "Obstracting the field"}


# ===========================================================================
# STEP 1 — Build enriched master BBB
# ===========================================================================

def enrich_bbb(bbb: pd.DataFrame, matches: pd.DataFrame,
               competitions: pd.DataFrame,
               ctpm: pd.DataFrame) -> pd.DataFrame:
    """
    Merge ball-by-ball with match and competition metadata.
    Returns a copy of bbb with extra columns.
    """
    # --- Competition metadata ---
    comp_meta = competitions[["comp_id", "comp_name", "match_format", "season"]].copy()

    # Normalise season strings: '2022/23' → '2022' (use start year)
    comp_meta["season_year"] = (
        comp_meta["season"]
        .str.split("/").str[0]
        .astype(int)
    )

    # --- Match metadata (slim) ---
    match_meta = matches[[
        "match_id", "comp_id", "match_date", "stage",
        "home_team_id", "away_team_id",
        "toss_win_team_id", "toss_opted",
        "win_team_id", "result",
        "first_inn_revised_overs", "second_inn_revised_overs", "revised_target",
        "knock_out",
    ]].copy()

    # Merge competitions into matches
    match_meta = match_meta.merge(comp_meta, on="comp_id", how="left")

    # Merge into bbb
    df = bbb.copy()
    df = df.merge(match_meta, on="match_id", how="left")

    # --- Player name lookup from CTPM ---
    # Build a unique player_id → player_name map (use most common name per ID)
    name_map = (
        ctpm.groupby("player_id")["player_name"]
        .agg(lambda x: x.mode().iloc[0])
        .to_dict()
    )

    # --- Competition format flags ---
    df["is_ipl"] = df["comp_id"].isin(IPL_COMP_IDS)
    df["is_t20i"] = df["match_format"] == "T20I"

    return df


# ===========================================================================
# STEP 2 — Exclusions
# ===========================================================================

def apply_exclusions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Remove records that must not enter analysis.

    Exclusions:
      1. Super-over deliveries (innings > 2).
      2. Emerging-team competition matches.
      3. Rows without match metadata (should be zero).

    Returns enriched df and an exclusion log dict.
    """
    total = len(df)
    log = {}

    # Super overs
    super_overs = df["innings"] > 2
    log["super_over_deliveries"] = int(super_overs.sum())
    df = df[~super_overs].copy()

    # Excluded competitions
    excl_comp = df["comp_id"].isin(EXCLUDE_COMP_IDS)
    log["emerging_team_deliveries"] = int(excl_comp.sum())
    df = df[~excl_comp].copy()

    # Rows with no match metadata (comp_id null)
    no_meta = df["comp_id"].isna()
    log["no_meta_deliveries"] = int(no_meta.sum())
    df = df[~no_meta].copy()

    log["total_removed"] = total - len(df)
    log["retained"] = len(df)

    print(f"[Exclusions] Super-over rows removed : {log['super_over_deliveries']}")
    print(f"[Exclusions] Emerging-team rows removed: {log['emerging_team_deliveries']}")
    print(f"[Exclusions] No-metadata rows removed : {log['no_meta_deliveries']}")
    print(f"[Exclusions] Total retained           : {log['retained']:,}")
    return df, log


# ===========================================================================
# STEP 3 — Delivery-level feature engineering
# ===========================================================================

def engineer_delivery_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add computed columns at the individual delivery level.
    """
    df = df.copy()

    # --- Normalise dismissal type spelling ---
    df["dismissal_type"] = df["dismissal_type"].str.strip()
    df.loc[df["dismissal_type"] == "Obstracting the field",
           "dismissal_type"] = "Obstructing the field"

    # --- Extra type flags ---
    df["is_wide"]   = df["extras_type"] == "wide"
    df["is_noball"] = df["extras_type"] == "Overstep"
    df["is_bye"]    = df["extras_type"] == "byes"
    df["is_legbye"] = df["extras_type"] == "legbyes"
    df["is_other_extra"] = df["extras_type"] == "other"
    df["is_penalty"] = df["extras_type"] == "Penalty"

    # --- Legal delivery flag ---
    # A legal ball is one that counts towards the over (not wide, not no-ball)
    df["is_legal"] = ~(df["is_wide"] | df["is_noball"])

    # --- Run accounting ---
    # batter_runs: runs scored by batter off the bat
    # (already in `runs` for normal, legbye, bye deliveries;
    #  for wides: 0; for no-balls: bat runs in `runs`)
    df["batter_runs"] = df["runs"].astype(int)

    # Extra runs (wides, no-balls, byes, legbyes, penalties)
    df["extra_runs"] = df["extras1_runs"].astype(int) + df["extras_2_runs"].astype(int)

    # Total delivery runs (what the team scores off this ball)
    df["total_delivery_runs"] = df["batter_runs"] + df["extra_runs"]

    # Bowler-conceded runs: batter runs + no-ball extras + wide extras
    # Byes and leg-byes are NOT charged to the bowler
    df["bowler_conceded_runs"] = np.where(
        df["is_legbye"] | df["is_bye"],
        df["extra_runs"],   # only the penalty noball (if stacked) — handled below
        df["total_delivery_runs"]
    )
    # Correct: byes and legbyes extras are not charged to bowler
    df["bowler_conceded_runs"] = np.where(
        df["is_legbye"] | df["is_bye"],
        0,
        df["total_delivery_runs"]
    )
    # Penalty runs (5-run penalties) go to team, not bowler
    df["bowler_conceded_runs"] = np.where(
        df["is_penalty"],
        0,
        df["bowler_conceded_runs"]
    )

    # --- Wicket attribution ---
    df["is_bowler_wicket"] = (
        df["is_wicket"] &
        ~df["dismissal_type"].isin(NON_BOWLER_DISMISSALS) &
        df["dismissal_type"].notna()
    )

    # --- Dot ball ---
    df["is_dot_ball"] = (df["is_legal"]) & (df["batter_runs"] == 0) & (~df["is_wicket"])
    # (a wicket delivery with 0 runs is usually a dot, but we keep them separate)
    df["is_batter_dot"] = (df["batter_runs"] == 0) & df["is_legal"]

    # --- Boundary flags (already in data as is_four, is_six) ---
    df["is_boundary"] = df["is_four"] | df["is_six"]

    # --- Match phase ---
    df["phase"] = df["overid"].map(PHASE_MAP).fillna("Unknown")

    # --- Over number (0-indexed for some computations) ---
    df["over_0idx"] = df["overid"] - 1

    # --- Ball in innings (1-indexed, legals only running count per innings) ---
    # We'll compute this during pressure engine; placeholder here
    df["delivery_number"] = df["innings_ball_num"]  # total deliveries inc. extras

    return df


# ===========================================================================
# STEP 4 — Team innings totals (for pressure computation)
# ===========================================================================

def compute_innings_totals(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every (match_id, innings), compute:
      - total team runs
      - total wickets
      - total legal deliveries (overs equivalent)
      - run rate
    Returns a summary DataFrame with one row per (match_id, innings).
    """
    grp = df.groupby(["match_id", "innings"])

    innings_totals = grp.agg(
        team_runs=("total_delivery_runs", "sum"),
        wickets_fallen=("is_wicket", "sum"),
        legal_balls=("is_legal", "sum"),
        total_balls=("over_ball_num", "count"),
    ).reset_index()

    innings_totals["overs_complete"] = innings_totals["legal_balls"] // 6
    innings_totals["balls_in_partial"] = innings_totals["legal_balls"] % 6
    innings_totals["run_rate"] = (
        innings_totals["team_runs"] / (innings_totals["legal_balls"] / 6)
    ).replace([np.inf, -np.inf], np.nan)

    return innings_totals


# ===========================================================================
# STEP 5 — Build delivery-level running state (for pressure)
# ===========================================================================

def compute_running_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each delivery, compute the match state AT THE TIME OF THAT DELIVERY
    (before the ball is bowled — using cumulative values up to previous ball).

    Columns added:
      - cumulative_runs_before : team runs scored before this delivery
      - cumulative_wickets_before : wickets fallen before this delivery
      - cumulative_legal_balls_before : legal balls bowled before this delivery
      - balls_remaining_in_innings : legal balls left after this delivery
      - wickets_remaining_before : wickets in hand before this delivery
    """
    df = df.sort_values(["match_id", "innings", "innings_ball_num"]).copy()

    # Cumulative totals shifted by 1 so they represent state BEFORE the delivery
    df["cumul_runs"] = (
        df.groupby(["match_id", "innings"])["total_delivery_runs"]
        .cumsum()
        .shift(1, fill_value=0)
    )
    df["cumul_wickets"] = (
        df.groupby(["match_id", "innings"])["is_wicket"]
        .cumsum()
        .astype(int)
        .shift(1, fill_value=0)
    )
    df["cumul_legal_balls"] = (
        df.groupby(["match_id", "innings"])["is_legal"]
        .cumsum()
        .astype(int)
        .shift(1, fill_value=0)
    )

    df["wickets_remaining"] = 10 - df["cumul_wickets"]
    # Standard T20: 120 legal balls per innings (overs can vary with DLS, handled later)
    df["legal_balls_remaining"] = (120 - df["cumul_legal_balls"]).clip(lower=0)

    return df


# ===========================================================================
# STEP 6 — Innings 1 required run rate proxy (for 1st innings pressure)
# ===========================================================================

def tag_innings1_pressure_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    For first-innings deliveries, we cannot know the target.
    Instead, we construct a 'virtual required run rate' based on a
    benchmark T20 par score of 160 (a neutral baseline).

    This gives us a directional pressure signal for batting first
    without using future information.
    """
    PAR_SCORE = 160  # neutral benchmark

    inn1 = df["innings"] == 1
    runs_still_needed = (PAR_SCORE - df["cumul_runs"]).clip(lower=0)
    balls_left = df["legal_balls_remaining"].clip(lower=1)

    df.loc[inn1, "virtual_rrr"] = (runs_still_needed / balls_left * 6).clip(upper=36)
    df.loc[~inn1, "virtual_rrr"] = np.nan  # only for 1st innings

    return df


# ===========================================================================
# STEP 7 — Second-innings required run rate
# ===========================================================================

def compute_second_innings_rrr(df: pd.DataFrame,
                                innings_totals: pd.DataFrame) -> pd.DataFrame:
    """
    For second-innings deliveries, compute the ACTUAL required run rate
    at the time of each delivery.

    Uses first-innings total (from innings_totals) as the target.
    """
    # First innings team scores per match
    inn1_scores = (
        innings_totals[innings_totals["innings"] == 1][["match_id", "team_runs"]]
        .rename(columns={"team_runs": "inn1_score"})
    )

    df = df.merge(inn1_scores, on="match_id", how="left")

    # For DLS matches, use revised_target if available
    df["target"] = np.where(
        df["revised_target"].notna() & (df["innings"] == 2),
        df["revised_target"],
        df["inn1_score"] + 1   # normal chase: need inn1_score + 1
    )

    # Runs needed before this delivery
    runs_needed = (df["target"] - df["cumul_runs"]).clip(lower=0)
    balls_left = df["legal_balls_remaining"].clip(lower=1)

    df["actual_rrr"] = np.where(
        df["innings"] == 2,
        (runs_needed / balls_left * 6).clip(upper=36),
        np.nan
    )

    df["runs_needed"] = np.where(df["innings"] == 2, runs_needed, np.nan)

    return df


# ===========================================================================
# STEP 8 — Recent boundaries (last 6 legal balls)
# ===========================================================================

def compute_recent_boundaries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count boundary deliveries (4s and 6s) in the last 6 legal balls
    of the SAME innings (rolling window).  Used as a bowling pressure signal.
    """
    df = df.sort_values(["match_id", "innings", "innings_ball_num"]).copy()

    # Only legal boundary deliveries
    df["_boundary_legal"] = df["is_boundary"] & df["is_legal"]

    df["recent_boundaries_6"] = (
        df.groupby(["match_id", "innings"])["_boundary_legal"]
        .transform(lambda s: s.shift(1).rolling(6, min_periods=1).sum())
        .fillna(0)
        .astype(int)
    )

    df.drop(columns=["_boundary_legal"], inplace=True)
    return df


# ===========================================================================
# STEP 9 — Player-match batting summary
# ===========================================================================

def build_batting_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-player per-match batting statistics.
    Only considers deliveries where the player is the striker.
    """
    bat = df.copy()

    grp = bat.groupby(["match_id", "striker_id", "innings",
                        "is_ipl", "is_t20i", "comp_id", "comp_name",
                        "season_year", "match_date", "phase"])

    summary = grp.agg(
        batter_runs=("batter_runs", "sum"),
        balls_faced=("is_legal", "sum"),
        fours=("is_four", "sum"),
        sixes=("is_six", "sum"),
        boundaries=("is_boundary", "sum"),
        dots=("is_batter_dot", "sum"),
        dismissed=("is_wicket", lambda x: (
            x & bat.loc[x.index, "dismissed_player_id"].fillna(-1).eq(
                bat.loc[x.index, "striker_id"])
        ).any()),
    ).reset_index()

    summary.rename(columns={"striker_id": "player_id"}, inplace=True)

    # Derived metrics
    summary["strike_rate"] = np.where(
        summary["balls_faced"] > 0,
        summary["batter_runs"] / summary["balls_faced"] * 100,
        np.nan
    )
    summary["boundary_pct"] = np.where(
        summary["balls_faced"] > 0,
        summary["boundaries"] / summary["balls_faced"] * 100,
        np.nan
    )
    summary["four_pct"] = np.where(
        summary["balls_faced"] > 0,
        summary["fours"] / summary["balls_faced"] * 100,
        np.nan
    )
    summary["six_pct"] = np.where(
        summary["balls_faced"] > 0,
        summary["sixes"] / summary["balls_faced"] * 100,
        np.nan
    )
    summary["dot_pct"] = np.where(
        summary["balls_faced"] > 0,
        summary["dots"] / summary["balls_faced"] * 100,
        np.nan
    )

    return summary


# ===========================================================================
# STEP 10 — Player-match bowling summary
# ===========================================================================

def build_bowling_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-player per-match bowling statistics.
    Only considers deliveries where the player is the bowler.
    """
    bow = df.copy()

    grp = bow.groupby(["match_id", "bowler_id", "innings",
                        "is_ipl", "is_t20i", "comp_id", "comp_name",
                        "season_year", "match_date", "phase"])

    summary = grp.agg(
        runs_conceded=("bowler_conceded_runs", "sum"),
        legal_balls=("is_legal", "sum"),
        total_balls=("is_legal", "count"),
        wickets=("is_bowler_wicket", "sum"),
        fours_conceded=("is_four", "sum"),
        sixes_conceded=("is_six", "sum"),
        wides=("is_wide", "sum"),
        noballs=("is_noball", "sum"),
        dots=("is_batter_dot", "sum"),
    ).reset_index()

    summary.rename(columns={"bowler_id": "player_id"}, inplace=True)

    summary["overs_bowled"] = summary["legal_balls"] / 6

    summary["economy"] = np.where(
        summary["legal_balls"] > 0,
        summary["runs_conceded"] / summary["overs_bowled"],
        np.nan
    )
    summary["bowling_sr"] = np.where(
        summary["wickets"] > 0,
        summary["legal_balls"] / summary["wickets"],
        np.nan
    )
    summary["bowling_avg"] = np.where(
        summary["wickets"] > 0,
        summary["runs_conceded"] / summary["wickets"],
        np.nan
    )
    summary["dot_pct"] = np.where(
        summary["legal_balls"] > 0,
        summary["dots"] / summary["legal_balls"] * 100,
        np.nan
    )
    summary["boundary_pct_conceded"] = np.where(
        summary["legal_balls"] > 0,
        (summary["fours_conceded"] + summary["sixes_conceded"]) / summary["legal_balls"] * 100,
        np.nan
    )

    return summary


# ===========================================================================
# STEP 11 — Add player names to summaries
# ===========================================================================

def add_player_names(df: pd.DataFrame, ctpm: pd.DataFrame,
                     bbb: pd.DataFrame) -> pd.DataFrame:
    """
    Add player_name column using BBB striker/bowler name as primary source
    (100% coverage confirmed in EDA), falling back to CTPM.
    """
    # Build from BBB directly
    batter_names = (
        bbb[["striker_id", "striker_name"]]
        .drop_duplicates("striker_id")
        .rename(columns={"striker_id": "player_id", "striker_name": "player_name"})
    )
    bowler_names = (
        bbb[["bowler_id", "bowler_name"]]
        .drop_duplicates("bowler_id")
        .rename(columns={"bowler_id": "player_id", "bowler_name": "player_name"})
    )
    name_map = pd.concat([batter_names, bowler_names]).drop_duplicates("player_id")

    df = df.merge(name_map, on="player_id", how="left")
    return df


# ===========================================================================
# MASTER PIPELINE
# ===========================================================================

def run_preprocessing_pipeline(raw: dict) -> dict:
    """
    Orchestrate all preprocessing steps.

    Parameters
    ----------
    raw : dict returned by data_loader.load_all()

    Returns
    -------
    dict with keys:
        'master_bbb'       — enriched delivery-level DataFrame
        'innings_totals'   — per (match, innings) summary
        'batting_summary'  — per (player, match, phase) batting stats
        'bowling_summary'  — per (player, match, phase) bowling stats
        'exclusion_log'    — dict of removed row counts
    """
    bbb   = raw["bbb"]
    matches = raw["matches"]
    comps   = raw["competitions"]
    ctpm    = raw["ctpm"]

    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)

    print("\n[Step 1] Enriching BBB with match/competition metadata …")
    df = enrich_bbb(bbb, matches, comps, ctpm)

    print("[Step 2] Applying exclusions …")
    df, excl_log = apply_exclusions(df)

    print("[Step 3] Engineering delivery-level features …")
    df = engineer_delivery_features(df)

    print("[Step 4] Computing innings totals …")
    innings_totals = compute_innings_totals(df)

    print("[Step 5] Computing running match state …")
    df = compute_running_state(df)

    print("[Step 6] Tagging 1st-innings pressure proxy …")
    df = tag_innings1_pressure_proxy(df)

    print("[Step 7] Computing 2nd-innings required run rate …")
    df = compute_second_innings_rrr(df, innings_totals)

    print("[Step 8] Computing recent boundary counts …")
    df = compute_recent_boundaries(df)

    print("[Step 9] Building batting summary …")
    bat_summary = build_batting_summary(df)
    bat_summary = add_player_names(bat_summary, ctpm, bbb)

    print("[Step 10] Building bowling summary …")
    bowl_summary = build_bowling_summary(df)
    bowl_summary = add_player_names(bowl_summary, ctpm, bbb)

    print(f"\n[Pipeline Complete]")
    print(f"  Master BBB rows      : {len(df):>8,}")
    print(f"  Innings totals rows  : {len(innings_totals):>8,}")
    print(f"  Batting summary rows : {len(bat_summary):>8,}")
    print(f"  Bowling summary rows : {len(bowl_summary):>8,}")

    return {
        "master_bbb": df,
        "innings_totals": innings_totals,
        "batting_summary": bat_summary,
        "bowling_summary": bowl_summary,
        "exclusion_log": excl_log,
    }
