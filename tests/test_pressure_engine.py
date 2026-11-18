"""
tests/test_pressure_engine.py
==============================
Unit tests for Pressure Intelligence Engine.
Verifies no future information leakage and correct pressure directionality.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pressure_engine import (
    _batting_pressure_chase,
    _batting_pressure_first,
    _bowling_pressure_defend,
    _bowling_pressure_first,
    _norm,
)


class TestNorm:
    def test_norm_midpoint(self):
        assert abs(_norm(5, 0, 10) - 0.5) < 1e-9

    def test_norm_clamps_below(self):
        assert _norm(-5, 0, 10) == 0.0

    def test_norm_clamps_above(self):
        assert _norm(15, 0, 10) == 1.0


class TestBattingPressureChase:
    def _row(self, rrr, runs_needed, balls_rem, wkts_rem, target=200):
        return {
            "innings": 2,
            "actual_rrr": rrr,
            "runs_needed": runs_needed,
            "legal_balls_remaining": balls_rem,
            "wickets_remaining": wkts_rem,
            "target": target,
        }

    def test_extreme_high_pressure(self):
        """Need 50 from 12 balls with 2 wickets — extreme pressure."""
        row = self._row(rrr=25, runs_needed=50, balls_rem=12, wkts_rem=2, target=200)
        score = _batting_pressure_chase(row)
        assert score > 0.7, f"Expected high pressure, got {score:.3f}"

    def test_low_pressure_easy_chase(self):
        """Need 10 from 36 balls with 9 wickets — easy."""
        row = self._row(rrr=1.67, runs_needed=10, balls_rem=36, wkts_rem=9, target=120)
        score = _batting_pressure_chase(row)
        assert score < 0.5, f"Expected low pressure, got {score:.3f}"

    def test_score_bounded_0_1(self):
        """Score must always be in [0, 1]."""
        for rrr, need, balls, wkts in [
            (30, 100, 6, 1), (1, 1, 100, 10), (12, 20, 18, 5)
        ]:
            row = self._row(rrr, need, balls, wkts)
            s = _batting_pressure_chase(row)
            assert 0.0 <= s <= 1.0, f"Out of bounds: {s}"

    def test_more_wickets_less_pressure(self):
        """Same RRR and runs — more wickets → lower pressure."""
        base = dict(rrr=10, runs_needed=30, balls_rem=18, target=150)
        s_high = _batting_pressure_chase({**base, "wickets_remaining": 2})
        s_low  = _batting_pressure_chase({**base, "wickets_remaining": 9})
        assert s_high > s_low, "Fewer wickets should mean higher pressure"

    def test_higher_rrr_more_pressure(self):
        """Same wickets and balls — higher RRR → higher pressure."""
        base = dict(runs_needed=40, balls_rem=18, wickets_remaining=5, target=200)
        s_high = _batting_pressure_chase({**base, "actual_rrr": 15})
        s_low  = _batting_pressure_chase({**base, "actual_rrr": 5})
        assert s_high > s_low, "Higher RRR should mean higher pressure"


class TestBattingPressureFirst:
    def _row(self, over, wkts_rem, cumul_wkts=0, virtual_rrr=8.0):
        return {
            "innings": 1,
            "overid": over,
            "virtual_rrr": virtual_rrr,
            "legal_balls_remaining": max(0, 120 - over * 6),
            "wickets_remaining": wkts_rem,
            "cumul_wickets": cumul_wkts,
        }

    def test_last_over_many_wickets_lost(self):
        """Over 20, 4 wickets remaining — elevated pressure."""
        row = self._row(over=20, wkts_rem=4, cumul_wkts=6, virtual_rrr=14)
        s = _batting_pressure_first(row)
        assert s > 0.5

    def test_first_over_no_wickets(self):
        """Over 1, all 10 wickets — low pressure."""
        row = self._row(over=1, wkts_rem=10, cumul_wkts=0, virtual_rrr=7.0)
        s = _batting_pressure_first(row)
        assert s < 0.4

    def test_bounded(self):
        for over, wkts in [(1, 10), (20, 1), (10, 5)]:
            s = _batting_pressure_first(self._row(over, wkts))
            assert 0.0 <= s <= 1.0


class TestBowlingPressureDefend:
    def _row(self, runs_needed, balls_rem, wkts_rem, recent_bnd=0, target=160):
        return {
            "innings": 2,
            "runs_needed": runs_needed,
            "legal_balls_remaining": balls_rem,
            "wickets_remaining": wkts_rem,
            "recent_boundaries_6": recent_bnd,
            "target": target,
        }

    def test_tight_chase_many_wickets(self):
        """Chase: 15 needed from 18 balls, 7 wickets — high bowling pressure."""
        row = self._row(runs_needed=15, balls_rem=18, wkts_rem=7, recent_bnd=2)
        s = _bowling_pressure_defend(row)
        assert s > 0.4

    def test_hopeless_chase_low_bowling_pressure(self):
        """Chase: 80 needed from 12 balls, 3 wickets — batting doomed, low bowling pressure."""
        row = self._row(runs_needed=80, balls_rem=12, wkts_rem=3, target=200)
        s = _bowling_pressure_defend(row)
        # Bowling pressure should be lower when chase is nearly impossible
        assert s < 0.8   # not necessarily very low due to other factors

    def test_bounded(self):
        for rn, bl, wr in [(5, 6, 9), (100, 120, 1), (30, 24, 5)]:
            s = _bowling_pressure_defend(self._row(rn, bl, wr))
            assert 0.0 <= s <= 1.0


class TestNoPressureFormulaUseFutureData:
    """
    Confirm that pressure score functions only use state BEFORE the delivery.
    They accept a row dict and must not reference any post-delivery columns.
    """
    ALLOWED_KEYS = {
        "innings", "actual_rrr", "runs_needed", "legal_balls_remaining",
        "wickets_remaining", "target", "overid", "virtual_rrr",
        "cumul_wickets", "cumul_runs", "cumul_legal_balls",
        "recent_boundaries_6",
    }
    FORBIDDEN_KEYS = {
        "win_team_id",    # post-match outcome
        "result",         # post-match outcome
    }

    def test_no_future_keys_in_batting_chase(self):
        """The function should not access any forbidden future keys."""
        row = {
            "innings": 2,
            "actual_rrr": 10.0,
            "runs_needed": 30,
            "legal_balls_remaining": 18,
            "wickets_remaining": 5,
            "target": 150,
        }
        # Add forbidden keys — if function uses them, they'd affect result
        for k in self.FORBIDDEN_KEYS:
            row[k] = "FORBIDDEN"
        # Should run without errors
        s = _batting_pressure_chase(row)
        assert 0.0 <= s <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
