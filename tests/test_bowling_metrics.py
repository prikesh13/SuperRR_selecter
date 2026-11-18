"""
tests/test_bowling_metrics.py
==============================
Unit tests for bowling metrics, economy rate, pressure economy delta, and wicket rate.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bowling_model import compute_bowling_metrics, classify_bowling_role


def test_bowling_economy_and_delta():
    records = []
    # 60 high pressure legal balls (10 overs), 60 runs conceded (Economy = 6.0)
    for i in range(60):
        records.append({
            "match_id": (i // 12) + 1,
            "innings": 2,
            "bowler_id": 101,
            "bowler_conceded_runs": 1,
            "is_legal": True,
            "is_bowler_wicket": (i % 20 == 0),
            "is_four": False,
            "is_six": False,
            "is_wide": False,
            "is_noball": False,
            "is_batter_dot": False,
            "bowl_pressure_label": "High",
        })
    # 60 normal legal balls (10 overs), 90 runs conceded (Economy = 9.0)
    for i in range(60):
        records.append({
            "match_id": (i // 12) + 1,
            "innings": 2,
            "bowler_id": 101,
            "bowler_conceded_runs": 1.5,
            "is_legal": True,
            "is_bowler_wicket": False,
            "is_four": False,
            "is_six": False,
            "is_wide": False,
            "is_noball": False,
            "is_batter_dot": False,
            "bowl_pressure_label": "Low",
        })

    mock_df = pd.DataFrame(records)
    metrics = compute_bowling_metrics(mock_df, min_pressure_balls=30, min_pressure_spells=1)

    assert len(metrics) == 1
    row = metrics.iloc[0]
    assert row["player_id"] == 101
    assert abs(row["hp_economy"] - 6.0) < 1e-4
    assert abs(row["normal_economy"] - 9.0) < 1e-4
    # Positive delta means normal_economy - hp_economy = 9 - 6 = +3 (improvement)
    assert abs(row["pressure_economy_delta"] - 3.0) < 1e-4


def test_classify_bowling_role():
    records = [
        # Bowler 1: Powerplay specialist (>50% balls in 1-6)
        {"bowler_id": 1, "overid": 2, "is_legal": True, "pace_or_spin": "Pace", "bowling_type": "Right-arm Fast"},
        {"bowler_id": 1, "overid": 4, "is_legal": True, "pace_or_spin": "Pace", "bowling_type": "Right-arm Fast"},
        {"bowler_id": 1, "overid": 18, "is_legal": True, "pace_or_spin": "Pace", "bowling_type": "Right-arm Fast"},
        # Bowler 2: Death specialist (>40% balls in 16-20)
        {"bowler_id": 2, "overid": 17, "is_legal": True, "pace_or_spin": "Pace", "bowling_type": "Right-arm Fast"},
        {"bowler_id": 2, "overid": 19, "is_legal": True, "pace_or_spin": "Pace", "bowling_type": "Right-arm Fast"},
        # Bowler 3: Middle specialist (>50% balls in 7-15)
        {"bowler_id": 3, "overid": 9, "is_legal": True, "pace_or_spin": "Spin", "bowling_type": "Right-arm Leg-Break"},
        {"bowler_id": 3, "overid": 11, "is_legal": True, "pace_or_spin": "Spin", "bowling_type": "Right-arm Leg-Break"},
    ]
    mock_df = pd.DataFrame(records)
    roles = classify_bowling_role(mock_df)
    role_map = dict(zip(roles["player_id"], roles["bowling_phase_role"]))

    assert role_map[1] == "Powerplay Specialist"
    assert role_map[2] == "Death Specialist"
    assert role_map[3] == "Middle-over Specialist"
