"""
tests/test_ranking_engine.py
============================
Unit tests for ranking engine, top 10 uniqueness, PII boundaries, and reliability weighting.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ranking_engine import (
    build_batting_pii,
    build_bowling_pii,
    get_top_10_batters,
    get_top_10_bowlers,
    BATTING_WEIGHTS,
    BOWLING_WEIGHTS,
)


def test_batting_pii_no_duplicates_and_bounded():
    # Mock batting metrics with 15 players
    players = []
    for i in range(1, 16):
        players.append({
            "player_id": i,
            "adj_hp_strike_rate": 120 + i * 5,
            "pressure_sr_delta": i * 2,
            "adj_hp_batting_avg": 20 + i,
            "hp_boundary_pct": 10 + i,
            "pressure_survival_rate": 0.5 + (i / 50),
            "batting_reliability": min(1.0, 0.3 + (i / 20)),
        })
    batting_metrics = pd.DataFrame(players)
    consistency_df = pd.DataFrame([
        {"player_id": i, "batting_consistency_score": 50 + i, "bat_perf_quadrant": "Elite",
         "hp_match_count": 10, "pct_above_100": 60, "pct_above_80": 80}
        for i in range(1, 16)
    ])
    name_map = {i: f"Player_{i}" for i in range(1, 16)}

    pii_df = build_batting_pii(batting_metrics, consistency_df, name_map)
    top10 = get_top_10_batters(pii_df)

    # Validate constraints
    assert len(top10) == 10
    assert top10["player_id"].nunique() == 10
    assert (top10["batting_pii"] >= 0).all() and (top10["batting_pii"] <= 100).all()
    # Check monotonic rank
    assert list(top10["rank"]) == list(range(1, 11))


def test_bowling_pii_no_duplicates_and_bounded():
    # Mock bowling metrics with 15 players
    players = []
    for i in range(1, 16):
        players.append({
            "player_id": i,
            "hp_economy": 10 - (i * 0.2),
            "pressure_economy_delta": i * 0.1,
            "hp_dot_pct": 30 + i,
            "hp_wicket_rate": 0.5 + (i * 0.1),
            "pressure_boundary_prevention": i * 0.2,
            "bowling_reliability": min(1.0, 0.3 + (i / 20)),
        })
    bowling_metrics = pd.DataFrame(players)
    consistency_df = pd.DataFrame([
        {"player_id": i, "bowling_consistency_score": 50 + i, "bowl_perf_quadrant": "Elite",
         "hp_match_count": 10, "pct_below_8": 50, "pct_below_10": 80}
        for i in range(1, 16)
    ])
    name_map = {i: f"Bowler_{i}" for i in range(1, 16)}

    pii_df = build_bowling_pii(bowling_metrics, consistency_df, name_map)
    top10 = get_top_10_bowlers(pii_df)

    assert len(top10) == 10
    assert top10["player_id"].nunique() == 10
    assert (top10["bowling_pii"] >= 0).all() and (top10["bowling_pii"] <= 100).all()
    assert list(top10["rank"]) == list(range(1, 11))
