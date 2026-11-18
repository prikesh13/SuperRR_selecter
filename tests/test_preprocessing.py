"""
tests/test_preprocessing.py
============================
Unit tests for preprocessing pipeline correctness.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocessing import (
    engineer_delivery_features,
    compute_innings_totals,
    NON_BOWLER_DISMISSALS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_delivery(
    extras_type=None, runs=0, extras1_runs=0, extras_2_runs=0,
    is_wicket=False, dismissal_type=None, is_four=False, is_six=False,
    overid=1, over_ball_num=1, innings_ball_num=1, innings=1,
    match_id=1, striker_id=1, bowler_id=2, batting_team_id=10,
    bowling_team_id=20, dismissed_player_id=None,
):
    return {
        "match_id": match_id,
        "innings": innings,
        "overid": overid,
        "over_ball_num": over_ball_num,
        "innings_ball_num": innings_ball_num,
        "batting_team_id": batting_team_id,
        "bowling_team_id": bowling_team_id,
        "striker_id": striker_id,
        "non_striker_id": 3,
        "bowler_id": bowler_id,
        "batting_type": "RHB",
        "bowling_type": "Right-arm Fast",
        "batting_order": 1,
        "runs": runs,
        "is_four": is_four,
        "is_six": is_six,
        "is_wicket": is_wicket,
        "dismissal_type_id": None,
        "dismissal_type": dismissal_type,
        "dismissed_player_id": dismissed_player_id,
        "extras_type": extras_type,
        "extras1_runs": extras1_runs,
        "extras_2_runs": extras_2_runs,
        "pace_or_spin": "Pace",
        "bowl_speed": 140.0,
        "stroke": "Drive",
        "delivery_style": "Normal",
        "over_or_around": "Over",
        "on_or_off": "Off",
        "delivery_ist_time": "2022-04-02T14:00:00",
        # enriched fields (normally added by enrich_bbb)
        "comp_id": 1358,
        "comp_name": "IPL 2022",
        "match_format": "T20",
        "season": "2022",
        "season_year": 2022,
        "is_ipl": True,
        "is_t20i": False,
        "match_date": pd.Timestamp("2022-04-02"),
        "stage": "League",
        "home_team_id": 10,
        "away_team_id": 20,
        "toss_win_team_id": 10.0,
        "toss_opted": "Fielding",
        "win_team_id": 20.0,
        "result": "Team B won by 5 wkts",
        "first_inn_revised_overs": None,
        "second_inn_revised_overs": None,
        "revised_target": None,
        "knock_out": 0,
    }


# ---------------------------------------------------------------------------
# Tests: legal delivery detection
# ---------------------------------------------------------------------------

class TestLegalDeliveries:
    def test_normal_delivery_is_legal(self):
        df = pd.DataFrame([make_delivery(extras_type=None)])
        df = engineer_delivery_features(df)
        assert df["is_legal"].iloc[0] == True

    def test_wide_is_not_legal(self):
        df = pd.DataFrame([make_delivery(extras_type="wide", extras1_runs=1)])
        df = engineer_delivery_features(df)
        assert df["is_legal"].iloc[0] == False

    def test_noball_is_not_legal(self):
        df = pd.DataFrame([make_delivery(extras_type="Overstep", extras1_runs=1)])
        df = engineer_delivery_features(df)
        assert df["is_legal"].iloc[0] == False

    def test_legbye_is_legal(self):
        df = pd.DataFrame([make_delivery(extras_type="legbyes", extras1_runs=1)])
        df = engineer_delivery_features(df)
        assert df["is_legal"].iloc[0] == True

    def test_bye_is_legal(self):
        df = pd.DataFrame([make_delivery(extras_type="byes", extras1_runs=4)])
        df = engineer_delivery_features(df)
        assert df["is_legal"].iloc[0] == True


# ---------------------------------------------------------------------------
# Tests: batter runs
# ---------------------------------------------------------------------------

class TestBatterRuns:
    def test_batter_runs_on_normal_delivery(self):
        df = pd.DataFrame([make_delivery(runs=4, is_four=True)])
        df = engineer_delivery_features(df)
        assert df["batter_runs"].iloc[0] == 4

    def test_batter_runs_on_wide_is_zero(self):
        df = pd.DataFrame([make_delivery(extras_type="wide", runs=0, extras1_runs=1)])
        df = engineer_delivery_features(df)
        assert df["batter_runs"].iloc[0] == 0

    def test_batter_runs_on_legbye_is_zero(self):
        df = pd.DataFrame([make_delivery(extras_type="legbyes", runs=0, extras1_runs=2)])
        df = engineer_delivery_features(df)
        assert df["batter_runs"].iloc[0] == 0

    def test_batter_runs_on_noball_with_bat_hit(self):
        df = pd.DataFrame([make_delivery(extras_type="Overstep", runs=6, is_six=True, extras1_runs=1)])
        df = engineer_delivery_features(df)
        assert df["batter_runs"].iloc[0] == 6


# ---------------------------------------------------------------------------
# Tests: bowler conceded runs
# ---------------------------------------------------------------------------

class TestBowlerConcededRuns:
    def test_legbye_not_charged_to_bowler(self):
        df = pd.DataFrame([make_delivery(extras_type="legbyes", runs=0, extras1_runs=1)])
        df = engineer_delivery_features(df)
        assert df["bowler_conceded_runs"].iloc[0] == 0

    def test_bye_not_charged_to_bowler(self):
        df = pd.DataFrame([make_delivery(extras_type="byes", runs=0, extras1_runs=4)])
        df = engineer_delivery_features(df)
        assert df["bowler_conceded_runs"].iloc[0] == 0

    def test_wide_charged_to_bowler(self):
        df = pd.DataFrame([make_delivery(extras_type="wide", runs=0, extras1_runs=1)])
        df = engineer_delivery_features(df)
        assert df["bowler_conceded_runs"].iloc[0] == 1

    def test_noball_plus_runs_charged_to_bowler(self):
        df = pd.DataFrame([make_delivery(extras_type="Overstep", runs=4, extras1_runs=1)])
        df = engineer_delivery_features(df)
        assert df["bowler_conceded_runs"].iloc[0] == 5  # 4 bat + 1 noball

    def test_penalty_not_charged_to_bowler(self):
        df = pd.DataFrame([make_delivery(extras_type="Penalty", runs=0, extras1_runs=5)])
        df = engineer_delivery_features(df)
        assert df["bowler_conceded_runs"].iloc[0] == 0

    def test_normal_six_charged_to_bowler(self):
        df = pd.DataFrame([make_delivery(runs=6, is_six=True)])
        df = engineer_delivery_features(df)
        assert df["bowler_conceded_runs"].iloc[0] == 6


# ---------------------------------------------------------------------------
# Tests: wicket attribution
# ---------------------------------------------------------------------------

class TestWicketAttribution:
    def test_caught_is_bowler_wicket(self):
        df = pd.DataFrame([make_delivery(is_wicket=True, dismissal_type="Caught",
                                          dismissed_player_id=1)])
        df = engineer_delivery_features(df)
        assert df["is_bowler_wicket"].iloc[0] == True

    def test_bowled_is_bowler_wicket(self):
        df = pd.DataFrame([make_delivery(is_wicket=True, dismissal_type="Bowled",
                                          dismissed_player_id=1)])
        df = engineer_delivery_features(df)
        assert df["is_bowler_wicket"].iloc[0] == True

    def test_lbw_is_bowler_wicket(self):
        df = pd.DataFrame([make_delivery(is_wicket=True, dismissal_type="LBW",
                                          dismissed_player_id=1)])
        df = engineer_delivery_features(df)
        assert df["is_bowler_wicket"].iloc[0] == True

    def test_runout_is_not_bowler_wicket(self):
        df = pd.DataFrame([make_delivery(is_wicket=True, dismissal_type="RunOut",
                                          dismissed_player_id=1)])
        df = engineer_delivery_features(df)
        assert df["is_bowler_wicket"].iloc[0] == False

    def test_retire_out_is_not_bowler_wicket(self):
        df = pd.DataFrame([make_delivery(is_wicket=True, dismissal_type="Retire Out",
                                          dismissed_player_id=1)])
        df = engineer_delivery_features(df)
        assert df["is_bowler_wicket"].iloc[0] == False

    def test_stump_is_bowler_wicket(self):
        df = pd.DataFrame([make_delivery(is_wicket=True, dismissal_type="Stump",
                                          dismissed_player_id=1)])
        df = engineer_delivery_features(df)
        assert df["is_bowler_wicket"].iloc[0] == True


# ---------------------------------------------------------------------------
# Tests: total delivery runs
# ---------------------------------------------------------------------------

class TestTotalDeliveryRuns:
    def test_total_runs_normal(self):
        df = pd.DataFrame([make_delivery(runs=1)])
        df = engineer_delivery_features(df)
        assert df["total_delivery_runs"].iloc[0] == 1

    def test_total_runs_wide(self):
        df = pd.DataFrame([make_delivery(extras_type="wide", runs=0, extras1_runs=1)])
        df = engineer_delivery_features(df)
        assert df["total_delivery_runs"].iloc[0] == 1

    def test_total_runs_noball_plus_bat(self):
        df = pd.DataFrame([make_delivery(extras_type="Overstep", runs=4, extras1_runs=1)])
        df = engineer_delivery_features(df)
        assert df["total_delivery_runs"].iloc[0] == 5


# ---------------------------------------------------------------------------
# Tests: phase classification
# ---------------------------------------------------------------------------

class TestPhaseClassification:
    def test_powerplay_overs(self):
        for over in range(1, 7):
            df = pd.DataFrame([make_delivery(overid=over)])
            df = engineer_delivery_features(df)
            assert df["phase"].iloc[0] == "Powerplay", f"Over {over} should be Powerplay"

    def test_middle_overs(self):
        for over in range(7, 16):
            df = pd.DataFrame([make_delivery(overid=over)])
            df = engineer_delivery_features(df)
            assert df["phase"].iloc[0] == "Middle", f"Over {over} should be Middle"

    def test_death_overs(self):
        for over in range(16, 21):
            df = pd.DataFrame([make_delivery(overid=over)])
            df = engineer_delivery_features(df)
            assert df["phase"].iloc[0] == "Death", f"Over {over} should be Death"


# ---------------------------------------------------------------------------
# Tests: boundary flags
# ---------------------------------------------------------------------------

class TestBoundaryFlags:
    def test_four_is_boundary(self):
        df = pd.DataFrame([make_delivery(runs=4, is_four=True)])
        df = engineer_delivery_features(df)
        assert df["is_boundary"].iloc[0] == True

    def test_six_is_boundary(self):
        df = pd.DataFrame([make_delivery(runs=6, is_six=True)])
        df = engineer_delivery_features(df)
        assert df["is_boundary"].iloc[0] == True

    def test_one_is_not_boundary(self):
        df = pd.DataFrame([make_delivery(runs=1)])
        df = engineer_delivery_features(df)
        assert df["is_boundary"].iloc[0] == False


# ---------------------------------------------------------------------------
# Tests: dot ball
# ---------------------------------------------------------------------------

class TestDotBall:
    def test_zero_legal_is_dot(self):
        df = pd.DataFrame([make_delivery(runs=0)])
        df = engineer_delivery_features(df)
        assert df["is_batter_dot"].iloc[0] == True

    def test_one_run_is_not_dot(self):
        df = pd.DataFrame([make_delivery(runs=1)])
        df = engineer_delivery_features(df)
        assert df["is_batter_dot"].iloc[0] == False

    def test_wide_is_not_legal_dot(self):
        df = pd.DataFrame([make_delivery(extras_type="wide", extras1_runs=1)])
        df = engineer_delivery_features(df)
        # Wide is not legal, so is_batter_dot should be False
        assert df["is_batter_dot"].iloc[0] == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
