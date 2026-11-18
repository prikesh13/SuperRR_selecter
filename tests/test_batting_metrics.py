"""
tests/test_batting_metrics.py
==============================
Unit tests for batting metrics, pressure deltas, survival rate, and sample filtering.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from batting_model import compute_batting_metrics, classify_batting_role


def test_batting_metrics_strike_rate_and_delta():
    # Construct mock master_bbb with 1 player facing high pressure and normal balls
    records = []
    # 100 high pressure balls, scored 200 runs (SR = 200)
    for i in range(100):
        records.append({
            "match_id": (i // 20) + 1,
            "innings": 2,
            "innings_ball_num": i + 1,
            "striker_id": 999,
            "is_legal": True,
            "batter_runs": 2,
            "is_wicket": False,
            "dismissed_player_id": np.nan,
            "is_four": False,
            "is_six": False,
            "is_boundary": False,
            "is_batter_dot": False,
            "bat_pressure_label": "High",
        })
    # 100 normal balls, scored 100 runs (SR = 100)
    for i in range(100):
        records.append({
            "match_id": (i // 20) + 1,
            "innings": 2,
            "innings_ball_num": i + 101,
            "striker_id": 999,
            "is_legal": True,
            "batter_runs": 1,
            "is_wicket": False,
            "dismissed_player_id": np.nan,
            "is_four": False,
            "is_six": False,
            "is_boundary": False,
            "is_batter_dot": False,
            "bat_pressure_label": "Low",
        })

    mock_df = pd.DataFrame(records)
    metrics = compute_batting_metrics(mock_df, min_pressure_balls=50, min_pressure_innings=1)

    assert len(metrics) == 1
    row = metrics.iloc[0]
    assert row["player_id"] == 999
    assert row["hp_runs"] == 200
    assert row["hp_balls"] == 100
    assert abs(row["hp_strike_rate"] - 200.0) < 1e-4
    assert abs(row["normal_strike_rate"] - 100.0) < 1e-4
    assert abs(row["pressure_sr_delta"] - 100.0) < 1e-4


def test_batting_metrics_filters_small_sample():
    records = []
    # Only 10 high pressure balls (below threshold 50)
    for i in range(10):
        records.append({
            "match_id": 1,
            "innings": 2,
            "innings_ball_num": i + 1,
            "striker_id": 888,
            "is_legal": True,
            "batter_runs": 2,
            "is_wicket": False,
            "dismissed_player_id": np.nan,
            "is_four": False,
            "is_six": False,
            "is_boundary": False,
            "is_batter_dot": False,
            "bat_pressure_label": "High",
        })

    mock_df = pd.DataFrame(records)
    metrics = compute_batting_metrics(mock_df, min_pressure_balls=50, min_pressure_innings=1)
    assert len(metrics) == 0


def test_classify_batting_role():
    records = [
        {"striker_id": 1, "batting_order": 1, "is_legal": True},
        {"striker_id": 1, "batting_order": 2, "is_legal": True},
        {"striker_id": 2, "batting_order": 3, "is_legal": True},
        {"striker_id": 3, "batting_order": 6, "is_legal": True},
        {"striker_id": 4, "batting_order": 8, "is_legal": True},
    ]
    mock_df = pd.DataFrame(records)
    roles = classify_batting_role(mock_df, pd.DataFrame())

    role_map = dict(zip(roles["player_id"], roles["batting_role"]))
    assert role_map[1] == "Opener"
    assert role_map[2] == "Top-order"
    assert role_map[3] == "Middle-order"
    assert role_map[4] == "Finisher"
