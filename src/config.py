"""Configuration constants for hoops-edge-predictor."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
FEATURES_DIR = PROJECT_ROOT / "features"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"

# S3 lakehouse
S3_BUCKET = "hoops-edge"
S3_REGION = "us-east-1"
SILVER_PREFIX = "silver"
GOLD_PREFIX = "gold"

# Silver table paths (relative to SILVER_PREFIX)
TABLE_FCT_GAMES = "fct_games"
TABLE_FCT_GAME_TEAMS = "fct_pbp_game_teams_flat"  # full boxscore with opponent stats
TABLE_FCT_RATINGS = "fct_ratings_adjusted"
TABLE_FCT_LINES = "fct_lines"

# Rolling average parameters
ROLLING_WINDOW = 15  # number of games for exponential decay
EWM_SPAN = 15  # span parameter for pandas ewm (matches ROLLING_WINDOW)

# The 37 features in EXACT order expected by the model
FEATURE_ORDER: list[str] = json.loads(
    (ARTIFACTS_DIR / "feature_order.json").read_text()
)
assert len(FEATURE_ORDER) >= 10, f"Expected >=10 features, got {len(FEATURE_ORDER)}"

# ── Column mappings from S3 Parquet schemas ──────────────────────────

# fct_games columns
GAMES_COLS = {
    "game_id": "gameId",
    "home_team_id": "homeTeamId",
    "away_team_id": "awayTeamId",
    "home_score": "homeScore",  # fallback: homePoints
    "away_score": "awayScore",  # fallback: awayPoints
    "neutral_site": "neutralSite",
    "start_date": "startDate",  # fallback: startTime, date
    "season": "season",
}

# fct_ratings_adjusted columns
RATINGS_COLS = {
    "team_id": "teamid",
    "offense_rating": "offenserating",  # fallback: offensiveRating
    "defense_rating": "defenserating",  # fallback: defensiveRating
    "net_rating": "netrating",
    "pace": "pace",  # may be available as pass-through from API
}

# fct_pbp_game_teams_flat columns
BOXSCORE_COLS = {
    "game_id": "gameid",
    "team_id": "teamid",
    "opponent_id": "opponentid",
    "is_home": "ishometeam",
    "start_date": "startdate",
    # Team offense
    "team_fg_made": "team_fg_made",
    "team_fg_att": "team_fg_att",
    "team_3fg_made": "team_3fg_made",
    "team_3fg_att": "team_3fg_att",
    "team_ft_made": "team_ft_made",
    "team_ft_att": "team_ft_att",
    "team_reb_off": "team_reb_off",
    "team_reb_def": "team_reb_def",
    # Opponent (for computing defensive four factors)
    "opp_fg_made": "opp_fg_made",
    "opp_fg_att": "opp_fg_att",
    "opp_3fg_made": "opp_3fg_made",
    "opp_3fg_att": "opp_3fg_att",
    "opp_ft_made": "opp_ft_made",
    "opp_ft_att": "opp_ft_att",
    "opp_reb_off": "opp_reb_off",
    "opp_reb_def": "opp_reb_def",
}

# fct_lines columns
LINES_COLS = {
    "game_id": "gameId",
    "provider": "provider",
    "spread": "spread",
    "over_under": "overUnder",
    "home_moneyline": "homeMoneyline",
    "away_moneyline": "awayMoneyline",
}

# BARTHAG exponent (BartTorvik's Pythagorean formula)
BARTHAG_EXPONENT = 11.5
