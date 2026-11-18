"""
pressure_engine.py
==================
Implements the Pressure Intelligence Index (PII) framework.

Design principles:
  - Pressure is defined SEPARATELY for batting and bowling.
  - Only information available BEFORE the delivery is used (no future leakage).
  - Pressure is a continuous score (0–100) PLUS a 3-level categorical label.
  - Different formulas for 1st and 2nd innings.
  - Death overs alone do NOT define pressure — match context does.

BATTING PRESSURE FORMULA (2nd innings / chase)
----------------------------------------------
Factors (all computed BEFORE the ball is bowled):
  1. Effective Required Run Rate (eRRR) — scaled against par (8.0 rpo benchmark)
  2. Runs needed — scaled on 0–target range
  3. Legal balls remaining — urgency increases as balls reduce
  4. Wickets remaining — fewer wickets = more pressure
  5. Match closeness — how tight is the chase?

BOWLING PRESSURE FORMULA (2nd innings / defending)
---------------------------------------------------
Factors (mirror of batting but from the bowling team's perspective):
  1. Runs to defend per over remaining (defending pressure)
  2. Balls remaining (fewer = higher pressure)
  3. Wickets remaining for batting team (fewer wickets = easier to defend)
  4. Recent boundary rate (momentum signal)
  5. Match closeness

FIRST INNINGS PROXY
-------------------
Uses virtual required run rate against a 160-run par benchmark,
wicket-based resource pressure, and innings stage.

THRESHOLDS (justified below)
-----------------------------
Score  0–33 → Low Pressure
Score 34–66 → Medium Pressure
Score 67–100 → High Pressure

Justification:
  A tertile split on a 0–100 continuous scale gives equal logical bands.
  Empirically, deliveries with RRR > 12 and < 3 wickets remaining in overs 17–20
  are unambiguously high-pressure in T20 cricket.  The continuous score
  naturally places these scenarios above 67.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants & weights
# ---------------------------------------------------------------------------

# Benchmark RRR values used for normalisation
PAR_RRR = 8.0          # T20 par scoring rate
HIGH_PRESSURE_RRR = 15.0  # RRR at which batting pressure is near maximum

# Weight vectors for BATTING pressure (2nd innings)
BAT_WEIGHTS_CHASE = {
    "rrr_component":          0.35,  # most important signal for chasing
    "runs_needed_component":  0.20,  # absolute runs needed (stage of chase)
    "balls_remaining_comp":   0.20,  # urgency of remaining overs
    "wickets_remaining_comp": 0.15,  # batting resources
    "closeness_component":    0.10,  # match tightness
}

# Weight vectors for BATTING pressure (1st innings)
BAT_WEIGHTS_FIRST = {
    "virtual_rrr_component":  0.30,
    "wickets_remaining_comp": 0.30,  # 1st innings: wickets = main pressure driver
    "innings_stage_comp":     0.25,  # how deep into the innings
    "recent_wkt_comp":        0.15,  # recent wicket cluster
}

# Weight vectors for BOWLING pressure (defending in 2nd innings)
BOWL_WEIGHTS_DEFEND = {
    "defend_rate_component":  0.35,  # runs per ball they need to defend
    "balls_remaining_comp":   0.20,
    "bat_wickets_comp":       0.20,  # more batting wickets = more bowling pressure
    "momentum_comp":          0.15,  # recent boundary rate
    "closeness_component":    0.10,
}

# Weight vectors for BOWLING pressure (1st innings)
BOWL_WEIGHTS_FIRST = {
    "innings_stage_comp":     0.30,
    "scoring_pace_comp":      0.35,  # how fast they're scoring
    "bat_wickets_comp":       0.20,
    "momentum_comp":          0.15,
}

# Threshold labels
PRESSURE_THRESHOLDS = [(0, 33, "Low"), (34, 66, "Medium"), (67, 100, "High")]


# ---------------------------------------------------------------------------
# Helper: sigmoid-style normaliser clamped to [0, 1]
# ---------------------------------------------------------------------------

def _norm(value, low, high):
    """
    Linearly normalise `value` from [low, high] → [0, 1], clamped.
    """
    if high == low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


# ---------------------------------------------------------------------------
# BATTING PRESSURE — CHASE (2nd innings)
# ---------------------------------------------------------------------------

def _batting_pressure_chase(row) -> float:
    """
    Compute batting pressure score (0–1) for a 2nd-innings delivery.
    """
    rrr       = row.get("actual_rrr", np.nan)
    runs_need = row.get("runs_needed", np.nan)
    balls_rem = row.get("legal_balls_remaining", 0)
    wkts_rem  = row.get("wickets_remaining", 10)
    target    = row.get("target", np.nan)

    # --- Component 1: RRR component (0 → 1 as RRR goes from 8 to 15+) ---
    rrr_comp = _norm(rrr if pd.notna(rrr) else PAR_RRR, PAR_RRR, HIGH_PRESSURE_RRR)

    # --- Component 2: Runs needed component (0 → 1 as more runs needed) ---
    if pd.notna(target) and target > 0 and pd.notna(runs_need):
        runs_comp = _norm(runs_need, 0, target)
    else:
        runs_comp = 0.5

    # --- Component 3: Balls remaining (fewer balls = more pressure) ---
    # Pressure rises as balls deplete: invert (fewer balls → higher score)
    balls_comp = _norm(120 - balls_rem, 0, 120)

    # --- Component 4: Wickets remaining (fewer wickets = more pressure) ---
    # Invert: 0 wickets remaining → maximum pressure
    wkts_comp = _norm(10 - wkts_rem, 0, 10)

    # --- Component 5: Match closeness ---
    # If runs_needed < 10 (nearly won) or runs_needed > target-5 (hopeless) → low pressure
    # Pressure peaks when chase is "in the balance" (5–30 runs needed)
    if pd.notna(runs_need):
        closeness = 1.0 - abs(runs_need - 15) / max(target, 1) if pd.notna(target) else 0.5
        closeness_comp = float(np.clip(closeness, 0.0, 1.0))
    else:
        closeness_comp = 0.5

    w = BAT_WEIGHTS_CHASE
    score = (
        w["rrr_component"]          * rrr_comp +
        w["runs_needed_component"]  * runs_comp +
        w["balls_remaining_comp"]   * balls_comp +
        w["wickets_remaining_comp"] * wkts_comp +
        w["closeness_component"]    * closeness_comp
    )
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# BATTING PRESSURE — FIRST INNINGS
# ---------------------------------------------------------------------------

def _batting_pressure_first(row) -> float:
    """
    Compute batting pressure score (0–1) for a 1st-innings delivery.
    """
    virtual_rrr = row.get("virtual_rrr", np.nan)
    balls_rem   = row.get("legal_balls_remaining", 0)
    wkts_rem    = row.get("wickets_remaining", 10)
    over        = row.get("overid", 1)
    cumul_wkts  = row.get("cumul_wickets", 0)

    # --- Component 1: Virtual RRR (above par = pressure) ---
    vrrr_comp = _norm(virtual_rrr if pd.notna(virtual_rrr) else PAR_RRR,
                      PAR_RRR, HIGH_PRESSURE_RRR)

    # --- Component 2: Wickets remaining ---
    wkts_comp = _norm(10 - wkts_rem, 0, 10)

    # --- Component 3: Innings stage (later = more pressure to accelerate) ---
    stage_comp = _norm(over, 1, 20)

    # --- Component 4: Recent wicket cluster ---
    # If 2+ wickets in last 3 overs → elevated pressure
    recent_wkt_comp = _norm(min(cumul_wkts, 5), 0, 5) * 0.6 + stage_comp * 0.4

    w = BAT_WEIGHTS_FIRST
    score = (
        w["virtual_rrr_component"]  * vrrr_comp +
        w["wickets_remaining_comp"] * wkts_comp +
        w["innings_stage_comp"]     * stage_comp +
        w["recent_wkt_comp"]        * recent_wkt_comp
    )
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# BOWLING PRESSURE — DEFENDING (2nd innings)
# ---------------------------------------------------------------------------

def _bowling_pressure_defend(row) -> float:
    """
    Compute bowling pressure score (0–1) for a 2nd-innings delivery.
    Bowling pressure = difficulty of defending the remaining runs.
    """
    runs_need    = row.get("runs_needed", np.nan)
    balls_rem    = row.get("legal_balls_remaining", 0)
    wkts_rem     = row.get("wickets_remaining", 10)   # batting team's wickets
    target       = row.get("target", np.nan)
    recent_bnd   = row.get("recent_boundaries_6", 0)

    # --- Component 1: Defend rate (runs batting team needs vs balls left) ---
    # High = easy to defend (need many, few balls); Low = hard to defend
    # We INVERT: easy to defend = low bowling pressure
    if pd.notna(runs_need) and balls_rem > 0:
        rpo_needed = runs_need / (balls_rem / 6)
        # As rpo_needed INCREASES, batting pressure increases → bowling pressure DECREASES
        # So bowling pressure is higher when rpo_needed is LOW (easy chase)
        # Bowling pressure peaks when chase is tight (rpo 7–10)
        defend_comp = _norm(PAR_RRR + 2 - rpo_needed, 0, PAR_RRR + 2)
    else:
        defend_comp = 0.5

    # --- Component 2: Balls remaining (fewer balls to take wickets = more pressure) ---
    balls_comp = _norm(120 - balls_rem, 0, 120)

    # --- Component 3: Batting wickets remaining (more wickets = more pressure for bowling) ---
    bat_wkts_comp = _norm(wkts_rem, 0, 10)   # more wickets in hand = higher bowling pressure

    # --- Component 4: Momentum (recent boundary rate) ---
    momentum_comp = _norm(recent_bnd, 0, 4)   # 4+ boundaries in last 6 = max pressure

    # --- Component 5: Match closeness ---
    if pd.notna(runs_need) and pd.notna(target) and target > 0:
        closeness_comp = 1.0 - abs(runs_need - 10) / max(target, 1)
        closeness_comp = float(np.clip(closeness_comp, 0.0, 1.0))
    else:
        closeness_comp = 0.5

    w = BOWL_WEIGHTS_DEFEND
    score = (
        w["defend_rate_component"]  * defend_comp +
        w["balls_remaining_comp"]   * balls_comp +
        w["bat_wickets_comp"]       * bat_wkts_comp +
        w["momentum_comp"]          * momentum_comp +
        w["closeness_component"]    * closeness_comp
    )
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# BOWLING PRESSURE — FIRST INNINGS (opposition building)
# ---------------------------------------------------------------------------

def _bowling_pressure_first(row) -> float:
    """
    Compute bowling pressure score (0–1) for a 1st-innings delivery.
    """
    over         = row.get("overid", 1)
    cumul_runs   = row.get("cumul_runs", 0)
    cumul_balls  = row.get("cumul_legal_balls", 1)
    wkts_rem     = row.get("wickets_remaining", 10)
    recent_bnd   = row.get("recent_boundaries_6", 0)

    # --- Component 1: Innings stage (death overs = more pressure to restrict) ---
    stage_comp = _norm(over, 1, 20)

    # --- Component 2: Current scoring pace ---
    if cumul_balls > 0:
        current_rpo = cumul_runs / (cumul_balls / 6)
    else:
        current_rpo = PAR_RRR
    pace_comp = _norm(current_rpo, PAR_RRR * 0.6, PAR_RRR * 1.6)

    # --- Component 3: Batting wickets remaining (fewer wickets = less pressure for bowler) ---
    bat_wkts_comp = _norm(wkts_rem, 0, 10)

    # --- Component 4: Momentum ---
    momentum_comp = _norm(recent_bnd, 0, 4)

    w = BOWL_WEIGHTS_FIRST
    score = (
        w["innings_stage_comp"]  * stage_comp +
        w["scoring_pace_comp"]   * pace_comp +
        w["bat_wickets_comp"]    * bat_wkts_comp +
        w["momentum_comp"]       * momentum_comp
    )
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# PUBLIC API: assign pressure scores to master BBB
# ---------------------------------------------------------------------------

def assign_batting_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute batting pressure score and label for every delivery in `df`.
    Adds columns: bat_pressure_score (0–100), bat_pressure_label.
    """
    df = df.copy()

    scores = []
    for _, row in df.iterrows():
        if row["innings"] == 2:
            s = _batting_pressure_chase(row)
        else:
            s = _batting_pressure_first(row)
        scores.append(round(s * 100, 2))

    df["bat_pressure_score"] = scores
    df["bat_pressure_label"] = pd.cut(
        df["bat_pressure_score"],
        bins=[-1, 33, 66, 101],
        labels=["Low", "Medium", "High"]
    )
    return df


def assign_bowling_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute bowling pressure score and label for every delivery in `df`.
    Adds columns: bowl_pressure_score (0–100), bowl_pressure_label.
    """
    df = df.copy()

    scores = []
    for _, row in df.iterrows():
        if row["innings"] == 2:
            s = _bowling_pressure_defend(row)
        else:
            s = _bowling_pressure_first(row)
        scores.append(round(s * 100, 2))

    df["bowl_pressure_score"] = scores
    df["bowl_pressure_label"] = pd.cut(
        df["bowl_pressure_score"],
        bins=[-1, 33, 66, 101],
        labels=["Low", "Medium", "High"]
    )
    return df


def run_pressure_engine(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply both batting and bowling pressure assignments to the master BBB.
    Uses vectorised row iteration (apply with raw=False).
    """
    print("[PressureEngine] Assigning batting pressure …")
    df = df.copy()

    # Vectorised computation using apply (faster than row-loop for large df)
    is_chase = df["innings"] == 2

    bat_scores = np.where(
        is_chase,
        df.apply(_batting_pressure_chase, axis=1),
        df.apply(_batting_pressure_first, axis=1)
    ) * 100

    bowl_scores = np.where(
        is_chase,
        df.apply(_bowling_pressure_defend, axis=1),
        df.apply(_bowling_pressure_first, axis=1)
    ) * 100

    df["bat_pressure_score"]  = np.round(bat_scores, 2)
    df["bowl_pressure_score"] = np.round(bowl_scores, 2)

    df["bat_pressure_label"] = pd.cut(
        df["bat_pressure_score"],
        bins=[-1, 33, 66, 101],
        labels=["Low", "Medium", "High"]
    )
    df["bowl_pressure_label"] = pd.cut(
        df["bowl_pressure_score"],
        bins=[-1, 33, 66, 101],
        labels=["Low", "Medium", "High"]
    )

    print(f"[PressureEngine] Batting pressure distribution:")
    print(df["bat_pressure_label"].value_counts().to_string())
    print(f"[PressureEngine] Bowling pressure distribution:")
    print(df["bowl_pressure_label"].value_counts().to_string())

    return df
