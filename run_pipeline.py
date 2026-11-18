"""
run_pipeline.py
===============
Master orchestration script. Runs all phases in order and saves
processed datasets to data/processed/.

Usage:
    python run_pipeline.py

Outputs (in data/processed/):
    master_bbb.parquet
    innings_totals.csv
    batting_summary.csv
    bowling_summary.csv
    batting_metrics.csv
    bowling_metrics.csv
    batting_pii.csv
    bowling_pii.csv
    top10_batters.csv
    top10_bowlers.csv
    scouting_batters.csv
    scouting_bowlers.csv
"""

import sys
import time
from pathlib import Path

# Allow imports from src/
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import numpy as np

from data_loader import load_all
from preprocessing import run_preprocessing_pipeline
from pressure_engine import run_pressure_engine
from batting_model import (
    compute_batting_metrics,
    compute_match_level_batting_consistency,
    classify_batting_role,
)
from bowling_model import (
    compute_bowling_metrics,
    compute_phase_bowling_metrics,
    compute_match_level_bowling_consistency,
    classify_bowling_role,
)
from reliability_model import apply_batting_reliability, apply_bowling_reliability
from consistency_model import build_batting_consistency, build_bowling_consistency
from ranking_engine import (
    build_batting_pii,
    build_bowling_pii,
    get_top_10_batters,
    get_top_10_bowlers,
    sensitivity_analysis,
)
from scouting_engine import build_scouting_report

PROCESSED = PROJECT_ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


def build_name_map(bbb: pd.DataFrame, ctpm: pd.DataFrame) -> dict:
    """Build player_id → player_name lookup from BBB (highest coverage)."""
    batter = bbb[["striker_id", "striker_name"]].drop_duplicates("striker_id")
    bowler = bbb[["bowler_id", "bowler_name"]].drop_duplicates("bowler_id")
    batter.columns = ["player_id", "player_name"]
    bowler.columns = ["player_id", "player_name"]
    names = pd.concat([batter, bowler]).drop_duplicates("player_id")
    # fill from ctpm for any gaps
    ctpm_map = ctpm.drop_duplicates("player_id")[["player_id", "player_name"]]
    all_names = pd.concat([names, ctpm_map]).drop_duplicates("player_id", keep="first")
    return dict(zip(all_names["player_id"], all_names["player_name"]))


def run():
    t0 = time.time()
    print("\n" + "="*70)
    print("  SupeRR Selector Season 2 — Full Analytics Pipeline")
    print("="*70)

    # -----------------------------------------------------------------------
    # Phase 1–2: Load + Preprocess
    # -----------------------------------------------------------------------
    raw = load_all()
    processed = run_preprocessing_pipeline(raw)

    master_bbb    = processed["master_bbb"]
    innings_totals = processed["innings_totals"]

    # -----------------------------------------------------------------------
    # Phase 3: Pressure Engine
    # -----------------------------------------------------------------------
    print("\n[Phase 3] Running Pressure Intelligence Engine …")
    master_bbb = run_pressure_engine(master_bbb)

    # -----------------------------------------------------------------------
    # Phase 4: Batting Metrics
    # -----------------------------------------------------------------------
    print("\n[Phase 4] Computing batting metrics …")
    batting_metrics = compute_batting_metrics(master_bbb)
    print(f"  Qualified batters: {len(batting_metrics)}")

    batting_role = classify_batting_role(master_bbb, raw["playing_eleven"])

    # -----------------------------------------------------------------------
    # Phase 5: Bowling Metrics
    # -----------------------------------------------------------------------
    print("\n[Phase 5] Computing bowling metrics …")
    bowling_metrics = compute_bowling_metrics(master_bbb)
    print(f"  Qualified bowlers: {len(bowling_metrics)}")

    phase_bowl = compute_phase_bowling_metrics(master_bbb)
    bowling_role = classify_bowling_role(master_bbb)

    # -----------------------------------------------------------------------
    # Phase 6: Reliability Adjustment
    # -----------------------------------------------------------------------
    print("\n[Phase 6] Applying reliability adjustments …")
    batting_metrics = apply_batting_reliability(batting_metrics, master_bbb)
    bowling_metrics = apply_bowling_reliability(bowling_metrics, master_bbb)

    # -----------------------------------------------------------------------
    # Phase 7: Consistency Model
    # -----------------------------------------------------------------------
    print("\n[Phase 7] Building consistency models …")
    bat_consistency  = build_batting_consistency(master_bbb)
    bowl_consistency = build_bowling_consistency(master_bbb)
    print(f"  Batters with consistency data: {len(bat_consistency)}")
    print(f"  Bowlers with consistency data: {len(bowl_consistency)}")

    # -----------------------------------------------------------------------
    # Phase 8: PII Scoring
    # -----------------------------------------------------------------------
    print("\n[Phase 8] Building PII scores …")
    name_map = build_name_map(raw["bbb"], raw["ctpm"])

    batting_pii = build_batting_pii(batting_metrics, bat_consistency, name_map)
    bowling_pii = build_bowling_pii(bowling_metrics, bowl_consistency, name_map)

    top10_bat  = get_top_10_batters(batting_pii)
    top10_bowl = get_top_10_bowlers(bowling_pii)

    print("\n" + "="*70)
    print("  TOP 10 BATTERS UNDER PRESSURE (Batting PII)")
    print("="*70)
    display_cols = ["rank", "player_name", "batting_pii", "hp_balls",
                    "hp_strike_rate", "normal_strike_rate",
                    "pressure_sr_delta", "batting_reliability"]
    avail = [c for c in display_cols if c in top10_bat.columns]
    print(top10_bat[avail].to_string(index=False))

    print("\n" + "="*70)
    print("  TOP 10 BOWLERS UNDER PRESSURE (Bowling PII)")
    print("="*70)
    display_cols_b = ["rank", "player_name", "bowling_pii", "hp_balls",
                      "hp_economy", "normal_economy",
                      "pressure_economy_delta", "bowling_reliability"]
    avail_b = [c for c in display_cols_b if c in top10_bowl.columns]
    print(top10_bowl[avail_b].to_string(index=False))

    # -----------------------------------------------------------------------
    # Phase 9: Scouting Report
    # -----------------------------------------------------------------------
    print("\n[Phase 9] Generating scouting cards …")
    # Merge consistency into top 10
    top10_bat = top10_bat.merge(
        bat_consistency[["player_id", "batting_consistency_score",
                         "bat_perf_quadrant", "hp_match_count",
                         "pct_above_100", "pct_above_80"]],
        on="player_id", how="left"
    )
    top10_bowl = top10_bowl.merge(
        bowl_consistency[["player_id", "bowling_consistency_score",
                          "bowl_perf_quadrant", "hp_match_count",
                          "pct_below_8", "pct_below_10"]],
        on="player_id", how="left"
    )

    scouting_bat, scouting_bowl = build_scouting_report(
        top10_bat, top10_bowl, batting_role, bowling_role, raw["ctpm"]
    )

    # -----------------------------------------------------------------------
    # Phase 10: Sensitivity Analysis
    # -----------------------------------------------------------------------
    print("\n[Phase 10] Sensitivity analysis of weight configurations …")
    sensitivity = sensitivity_analysis(
        batting_metrics, bat_consistency,
        bowling_metrics, bowl_consistency,
        name_map
    )
    for config, result in sensitivity.items():
        print(f"  [{config}] Top-5 bat: {result.get('top5_bat', [])}")
        print(f"           Top-5 bowl: {result.get('top5_bowl', [])}")

    # -----------------------------------------------------------------------
    # Save processed datasets
    # -----------------------------------------------------------------------
    print("\n[Saving] Writing processed datasets …")

    master_bbb.to_parquet(PROCESSED / "master_bbb.parquet", index=False)
    innings_totals.to_csv(PROCESSED / "innings_totals.csv", index=False)
    batting_metrics.to_csv(PROCESSED / "batting_metrics.csv", index=False)
    bowling_metrics.to_csv(PROCESSED / "bowling_metrics.csv", index=False)
    batting_pii.to_csv(PROCESSED / "batting_pii.csv", index=False)
    bowling_pii.to_csv(PROCESSED / "bowling_pii.csv", index=False)
    top10_bat.to_csv(PROCESSED / "top10_batters.csv", index=False)
    top10_bowl.to_csv(PROCESSED / "top10_bowlers.csv", index=False)
    scouting_bat.to_csv(PROCESSED / "scouting_batters.csv", index=False)
    scouting_bowl.to_csv(PROCESSED / "scouting_bowlers.csv", index=False)
    bat_consistency.to_csv(PROCESSED / "batting_consistency.csv", index=False)
    bowl_consistency.to_csv(PROCESSED / "bowling_consistency.csv", index=False)
    bowling_role.to_csv(PROCESSED / "bowling_roles.csv", index=False)
    batting_role.to_csv(PROCESSED / "batting_roles.csv", index=False)
    phase_bowl.to_csv(PROCESSED / "phase_bowling_metrics.csv", index=False)

    elapsed = time.time() - t0
    print(f"\n[Done] Pipeline complete in {elapsed:.1f}s")
    print(f"[Done] All outputs saved to {PROCESSED}")

    return {
        "master_bbb": master_bbb,
        "batting_pii": batting_pii,
        "bowling_pii": bowling_pii,
        "top10_bat": top10_bat,
        "top10_bowl": top10_bowl,
        "scouting_bat": scouting_bat,
        "scouting_bowl": scouting_bowl,
    }


if __name__ == "__main__":
    run()
