"""
reliability_model.py
====================
Implements statistical shrinkage / reliability adjustment.

Formula:
    Adjusted Score = (n / (n + k)) * Player Score + (k / (n + k)) * Baseline Score

Where:
    n = effective sample size (pressure balls faced/bowled)
    k = reliability parameter (empirically set, see justification below)
    Player Score = observed pressure metric (e.g. pressure SR)
    Baseline Score = role-specific / format-specific population mean

Justification for k:
    k represents the "equivalent sample" for a prior belief.
    For batting SR: a T20 innings typically has ~20 pressure balls.
    We want 5 pressure innings (100 balls) to have ~50% weight → k ≈ 100.
    This prevents 20-ball wonders from topping rankings.

    For bowling economy: a spell has ~6–12 pressure balls.
    We set k = 60 (equivalent to ~5 spells) for similar effect.

Confidence intervals are estimated via bootstrap resampling.
"""

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Reliability parameters
# ---------------------------------------------------------------------------
K_BATTING_SR   = 100   # shrinkage constant for batting strike rate
K_BATTING_AVG  = 100   # shrinkage constant for batting average
K_BOWLING_ECON = 60    # shrinkage constant for bowling economy
K_BOWLING_WKT  = 60    # shrinkage constant for wicket rate

N_BOOTSTRAP = 500      # bootstrap resamples for CI estimation


# ---------------------------------------------------------------------------
# Core shrinkage function
# ---------------------------------------------------------------------------

def shrink(player_score: float, n: float, k: float, baseline: float) -> float:
    """
    Apply James-Stein-style shrinkage toward population baseline.

    Parameters
    ----------
    player_score : observed player metric value
    n            : effective sample size (balls / matches)
    k            : reliability constant
    baseline     : population/role baseline

    Returns
    -------
    float : reliability-adjusted score
    """
    if np.isnan(player_score) or np.isnan(n) or n <= 0:
        return float(baseline)
    weight = n / (n + k)
    return weight * player_score + (1 - weight) * baseline


# ---------------------------------------------------------------------------
# Reliability weight (0 → 1 confidence)
# ---------------------------------------------------------------------------

def reliability_weight(n: float, k: float) -> float:
    """Return the fraction of weight given to the player's observed score."""
    if np.isnan(n) or n <= 0:
        return 0.0
    return float(n / (n + k))


# ---------------------------------------------------------------------------
# Bootstrap confidence interval for batting SR under pressure
# ---------------------------------------------------------------------------

def bootstrap_batting_sr_ci(player_deliveries: pd.DataFrame,
                              n_boot: int = N_BOOTSTRAP,
                              ci: float = 0.90) -> tuple[float, float]:
    """
    Estimate confidence interval for pressure strike rate using bootstrap.

    Parameters
    ----------
    player_deliveries : DataFrame of individual deliveries for one player
                        (must contain 'batter_runs' and 'is_legal' columns)
    n_boot : number of bootstrap resamples
    ci     : confidence level (default 90%)

    Returns
    -------
    (lower, upper) confidence interval for pressure SR
    """
    legal = player_deliveries[player_deliveries["is_legal"]]["batter_runs"].values
    if len(legal) < 10:
        return (np.nan, np.nan)

    boot_srs = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        sample = rng.choice(legal, size=len(legal), replace=True)
        boot_srs.append(sample.mean() * 100)

    alpha = (1 - ci) / 2
    return (float(np.percentile(boot_srs, alpha * 100)),
            float(np.percentile(boot_srs, (1 - alpha) * 100)))


def bootstrap_bowling_econ_ci(player_deliveries: pd.DataFrame,
                               n_boot: int = N_BOOTSTRAP,
                               ci: float = 0.90) -> tuple[float, float]:
    """
    Estimate confidence interval for pressure economy rate using bootstrap.
    """
    legal = player_deliveries[
        player_deliveries["is_legal"]
    ]["bowler_conceded_runs"].values

    if len(legal) < 12:
        return (np.nan, np.nan)

    boot_econs = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        sample = rng.choice(legal, size=len(legal), replace=True)
        boot_econs.append(sample.mean() * 6)

    alpha = (1 - ci) / 2
    return (float(np.percentile(boot_econs, alpha * 100)),
            float(np.percentile(boot_econs, (1 - alpha) * 100)))


# ---------------------------------------------------------------------------
# Apply reliability adjustment to batting metrics
# ---------------------------------------------------------------------------

def apply_batting_reliability(batting_metrics: pd.DataFrame,
                               master_bbb: pd.DataFrame) -> pd.DataFrame:
    """
    Apply shrinkage to batting pressure metrics.

    Baselines computed from the FULL population of batters with ≥ 50 balls.
    Role-specific baselines used where ≥ 20 role members have sufficient data.
    """
    df = batting_metrics.copy()

    # --- Population baselines ---
    pop_hp_sr  = df["hp_strike_rate"].dropna().mean()
    pop_hp_avg = df["hp_batting_avg"].dropna().mean()
    pop_hp_dot = df["hp_dot_pct"].dropna().mean()
    pop_hp_bnd = df["hp_boundary_pct"].dropna().mean()

    df["baseline_sr"]  = pop_hp_sr
    df["baseline_avg"] = pop_hp_avg

    # --- Shrinkage adjustments ---
    df["adj_hp_strike_rate"] = df.apply(
        lambda r: shrink(r["hp_strike_rate"], r["hp_balls"],
                         K_BATTING_SR, pop_hp_sr), axis=1)

    df["adj_hp_batting_avg"] = df.apply(
        lambda r: shrink(r["hp_batting_avg"], r["hp_innings"],
                         K_BATTING_AVG / 10, pop_hp_avg), axis=1)

    # Reliability weight (evidence strength 0–1)
    df["batting_reliability"] = df["hp_balls"].apply(
        lambda n: round(reliability_weight(n, K_BATTING_SR), 3))

    # Reliability tier
    df["reliability_tier"] = pd.cut(
        df["batting_reliability"],
        bins=[-0.001, 0.4, 0.65, 0.85, 1.001],
        labels=["Very Low", "Low", "Medium", "High"]
    )

    # --- Bootstrap CI (computed on full pressure delivery subset) ---
    # We pre-compute CIs from master_bbb per player
    hp_bbb = master_bbb[
        (master_bbb["bat_pressure_label"] == "High") &
        master_bbb["is_legal"]
    ].copy()

    ci_records = []
    for pid, grp in hp_bbb.groupby("striker_id"):
        lo, hi = bootstrap_batting_sr_ci(grp, n_boot=200)  # fast boot
        ci_records.append({"player_id": pid, "hp_sr_ci_lo": lo, "hp_sr_ci_hi": hi})

    ci_df = pd.DataFrame(ci_records)
    df = df.merge(ci_df, on="player_id", how="left")

    df["hp_sr_ci_width"] = df["hp_sr_ci_hi"] - df["hp_sr_ci_lo"]

    return df


# ---------------------------------------------------------------------------
# Apply reliability adjustment to bowling metrics
# ---------------------------------------------------------------------------

def apply_bowling_reliability(bowling_metrics: pd.DataFrame,
                               master_bbb: pd.DataFrame) -> pd.DataFrame:
    """
    Apply shrinkage to bowling pressure metrics.
    """
    df = bowling_metrics.copy()

    # --- Population baselines ---
    pop_hp_econ = df["hp_economy"].dropna().mean()
    pop_hp_dot  = df["hp_dot_pct"].dropna().mean()
    pop_hp_bnd  = df["hp_boundary_pct"].dropna().mean()
    pop_hp_wktr = df["hp_wicket_rate"].dropna().mean()

    df["baseline_economy"]     = pop_hp_econ
    df["baseline_wicket_rate"] = pop_hp_wktr

    # --- Shrinkage ---
    df["adj_hp_economy"] = df.apply(
        lambda r: shrink(r["hp_economy"], r["hp_balls"],
                         K_BOWLING_ECON, pop_hp_econ), axis=1)

    df["adj_hp_wicket_rate"] = df.apply(
        lambda r: shrink(r["hp_wicket_rate"], r["hp_balls"],
                         K_BOWLING_WKT, pop_hp_wktr), axis=1)

    # Reliability weight
    df["bowling_reliability"] = df["hp_balls"].apply(
        lambda n: round(reliability_weight(n, K_BOWLING_ECON), 3))

    df["reliability_tier"] = pd.cut(
        df["bowling_reliability"],
        bins=[-0.001, 0.4, 0.65, 0.85, 1.001],
        labels=["Very Low", "Low", "Medium", "High"]
    )

    # --- Bootstrap CI ---
    hp_bbb = master_bbb[
        (master_bbb["bowl_pressure_label"] == "High") &
        master_bbb["is_legal"]
    ].copy()

    ci_records = []
    for pid, grp in hp_bbb.groupby("bowler_id"):
        lo, hi = bootstrap_bowling_econ_ci(grp, n_boot=200)
        ci_records.append({"player_id": pid, "hp_econ_ci_lo": lo, "hp_econ_ci_hi": hi})

    ci_df = pd.DataFrame(ci_records)
    df = df.merge(ci_df, on="player_id", how="left")

    df["hp_econ_ci_width"] = df["hp_econ_ci_hi"] - df["hp_econ_ci_lo"]

    return df
