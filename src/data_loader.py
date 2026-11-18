"""
data_loader.py
==============
Loads and validates all six raw CSV datasets for the SupeRR Selector project.

All paths are relative to the project root (SupeRR_Selector/).
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def _csv_path(filename: str) -> Path:
    return RAW_DATA_DIR / filename


def load_ball_by_ball() -> pd.DataFrame:
    """
    Load ball_by_ball.csv.

    Key notes:
    - `runs` column = batter runs only (not total delivery runs).
    - `extras1_runs` = extras for the delivery.
    - `extras_type` is null for legal deliveries.
    - `innings` 3 and 4 are super-over deliveries.
    """
    df = pd.read_csv(
        _csv_path("ball_by_ball.csv"),
        low_memory=False,
        dtype={
            "match_id": "int32",
            "innings": "int8",
            "overid": "int8",
            "over_ball_num": "int8",
            "innings_ball_num": "int16",
            "batting_team_id": "int32",
            "bowling_team_id": "int32",
            "striker_id": "int32",
            "non_striker_id": "int32",
            "bowler_id": "int32",
            "batting_order": "int8",
            "runs": "int8",
            "is_four": bool,
            "is_six": bool,
            "is_wicket": bool,
            "extras1_runs": "int8",
            "extras_2_runs": "int8",
        },
    )
    return df


def load_matches() -> pd.DataFrame:
    """Load matches.csv with date parsing."""
    df = pd.read_csv(_csv_path("matches.csv"), low_memory=False)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    return df


def load_competitions() -> pd.DataFrame:
    """Load competitions.csv."""
    df = pd.read_csv(_csv_path("competitions.csv"))
    return df


def load_teams() -> pd.DataFrame:
    """Load teams.csv."""
    df = pd.read_csv(_csv_path("teams.csv"))
    return df


def load_playing_eleven() -> pd.DataFrame:
    """Load playing_eleven.csv."""
    df = pd.read_csv(_csv_path("playing_eleven.csv"), low_memory=False)
    return df


def load_competition_team_player_mapping() -> pd.DataFrame:
    """Load competition_team_player_mapping.csv."""
    df = pd.read_csv(_csv_path("competition_team_player_mapping.csv"))
    return df


def load_all() -> dict[str, pd.DataFrame]:
    """
    Load all six datasets and return as a dictionary.

    Returns
    -------
    dict with keys:
        'bbb', 'matches', 'competitions', 'teams',
        'playing_eleven', 'ctpm'
    """
    print("[DataLoader] Loading all datasets …")
    data = {
        "bbb": load_ball_by_ball(),
        "matches": load_matches(),
        "competitions": load_competitions(),
        "teams": load_teams(),
        "playing_eleven": load_playing_eleven(),
        "ctpm": load_competition_team_player_mapping(),
    }
    for name, df in data.items():
        print(f"  {name:12s}: {df.shape[0]:>7,} rows × {df.shape[1]} cols")
    print("[DataLoader] Done.\n")
    return data
