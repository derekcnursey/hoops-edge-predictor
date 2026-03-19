"""Configuration constants for hoops-edge-predictor."""

from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
FEATURES_DIR = PROJECT_ROOT / "features"
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
SITE_DATA_DIR = PROJECT_ROOT / "site" / "public" / "data"
TREE_REGRESSOR_PATH = CHECKPOINTS_DIR / "regressor_lgbm_l2.pkl"
TORVIK_TREE_REGRESSOR_PATH = CHECKPOINTS_DIR / "regressor_lgbm_l2_torvik.pkl"
TREE_REGRESSOR_TEAM_AB_ELITE_TAIL_ROUND64_V1_PATH = (
    CHECKPOINTS_DIR / "regressor_lgbm_l2_team_ab_elite_tail_round64_v1.pkl"
)
TORVIK_TREE_REGRESSOR_TEAM_AB_ELITE_TAIL_ROUND64_V1_PATH = (
    CHECKPOINTS_DIR / "regressor_lgbm_l2_torvik_team_ab_elite_tail_round64_v1.pkl"
)
MEAN_MODEL_VARIANT = os.getenv(
    "HOOPS_MEAN_MODEL_VARIANT",
    "team_ab_elite_tail_round64_v1",
).strip().lower()
BRACKET_MATCHUP_MODEL_VARIANT = os.getenv(
    "HOOPS_BRACKET_MATCHUP_MODEL_VARIANT",
    "team_ab_elite_tail_round64_v1",
).strip().lower()

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
TABLE_FCT_LINES_REPAIRED = "fct_lines_repaired_v1"
RESEARCH_LINES_TABLE = os.getenv("HOOPS_RESEARCH_LINES_TABLE", TABLE_FCT_LINES_REPAIRED)
HRB_LIVE_ODDS_ENABLED = os.getenv("HOOPS_HRB_LIVE_ODDS_ENABLED", "1") != "0"
HRB_MATCH_LOOKAHEAD_DAYS = int(os.getenv("HOOPS_HRB_LOOKAHEAD_DAYS", "7"))
HRB_MATCH_LOOKBACK_HOURS = int(os.getenv("HOOPS_HRB_LOOKBACK_HOURS", "6"))
HRB_MATCH_TIME_TOLERANCE_HOURS = float(os.getenv("HOOPS_HRB_TIME_TOLERANCE_HOURS", "6"))
HRB_HTTP_TIMEOUT_SECS = int(os.getenv("HOOPS_HRB_HTTP_TIMEOUT_SECS", "20"))
DAILY_ETL_LOOKBACK_DAYS = int(os.getenv("HOOPS_DAILY_ETL_LOOKBACK_DAYS", "4"))
FINAL_SCORES_LOOKBACK_DAYS = int(os.getenv("HOOPS_FINAL_SCORES_LOOKBACK_DAYS", "4"))
DAILY_ETL_MAX_CONCURRENCY = int(os.getenv("HOOPS_DAILY_ETL_MAX_CONCURRENCY", "6"))
DAILY_ETL_RATE_LIMIT_PER_SEC = int(os.getenv("HOOPS_DAILY_ETL_RATE_LIMIT_PER_SEC", "6"))
DAILY_ETL_FANOUT_BATCH_SIZE = int(os.getenv("HOOPS_DAILY_ETL_FANOUT_BATCH_SIZE", "40"))
LIVE_FEATURE_DRIFT_WARN_Z = float(os.getenv("HOOPS_LIVE_FEATURE_DRIFT_WARN_Z", "4.0"))
LIVE_FEATURE_FILL_WARN_RATE = float(os.getenv("HOOPS_LIVE_FEATURE_FILL_WARN_RATE", "0.2"))
LIVE_FEATURE_DRIFT_TOP_N = int(os.getenv("HOOPS_LIVE_FEATURE_DRIFT_TOP_N", "5"))
RATINGS_ASOF_STALE_WARN_DAYS = int(os.getenv("HOOPS_RATINGS_ASOF_STALE_WARN_DAYS", "3"))
HRB_SPREAD_DISLOCATION_WARN = float(os.getenv("HOOPS_HRB_SPREAD_DISLOCATION_WARN", "5.0"))

# Rolling average parameters
ROLLING_WINDOW = 15  # number of games for exponential decay
EWM_SPAN = 15  # span parameter for pandas ewm (matches ROLLING_WINDOW)

# Production no-garbage-time flag — must be consistent across train/inference
NO_GARBAGE = True

# Efficiency source: "gold" (gold-layer adj efficiencies) or "torvik" (Torvik daily ratings)
EFFICIENCY_SOURCE = "gold"
SUPPORTED_EFFICIENCY_SOURCES = {"gold", "torvik"}
GOLD_PRIORREG_K5_V1_RATINGS_TABLE = "team_adjusted_efficiencies_no_garbage_priorreg_k5_v1"
PRODUCTION_GOLD_RATINGS_TABLE = "team_adjusted_efficiencies_no_garbage_softkeep25_priorreg_k5_v1"
TEAM_AB_DEFAULT_EFFICIENCY_SOURCE = "torvik"
TEAM_AB_EFFICIENCY_SOURCE = os.getenv(
    "HOOPS_TEAM_AB_EFFICIENCY_SOURCE",
    TEAM_AB_DEFAULT_EFFICIENCY_SOURCE,
).strip().lower()
TEAM_AB_GOLD_RATINGS_TABLE = os.getenv(
    "HOOPS_TEAM_AB_GOLD_RATINGS_TABLE",
    PRODUCTION_GOLD_RATINGS_TABLE,
).strip()
BRACKET_TEAM_AB_DEFAULT_EFFICIENCY_SOURCE = TEAM_AB_DEFAULT_EFFICIENCY_SOURCE
BRACKET_TEAM_AB_EFFICIENCY_SOURCE = os.getenv(
    "HOOPS_BRACKET_TEAM_AB_EFFICIENCY_SOURCE",
    BRACKET_TEAM_AB_DEFAULT_EFFICIENCY_SOURCE,
).strip().lower()
BRACKET_TEAM_AB_GOLD_RATINGS_TABLE = os.getenv(
    "HOOPS_BRACKET_TEAM_AB_GOLD_RATINGS_TABLE",
    TEAM_AB_GOLD_RATINGS_TABLE,
).strip()
if TEAM_AB_EFFICIENCY_SOURCE not in SUPPORTED_EFFICIENCY_SOURCES:
    raise ValueError(
        "Unsupported HOOPS_TEAM_AB_EFFICIENCY_SOURCE "
        f"{TEAM_AB_EFFICIENCY_SOURCE!r}; expected one of "
        f"{sorted(SUPPORTED_EFFICIENCY_SOURCES)}."
    )
if BRACKET_TEAM_AB_EFFICIENCY_SOURCE not in SUPPORTED_EFFICIENCY_SOURCES:
    raise ValueError(
        "Unsupported HOOPS_BRACKET_TEAM_AB_EFFICIENCY_SOURCE "
        f"{BRACKET_TEAM_AB_EFFICIENCY_SOURCE!r}; expected one of "
        f"{sorted(SUPPORTED_EFFICIENCY_SOURCES)}."
    )
PRODUCTION_MU_BLEND_ENABLED = True
PRODUCTION_MU_BLEND_START_DAY = 0   # Nov 1
PRODUCTION_MU_BLEND_END_DAY = 75    # Jan 15
NCAA_TOURNAMENT_TORVIK_OVERRIDE_ENABLED = (
    os.getenv("HOOPS_NCAA_TOURNAMENT_TORVIK_OVERRIDE_ENABLED", "0") == "1"
)
NCAA_TOURNAMENT_PRIMARY_WEIGHT = float(
    os.getenv("HOOPS_NCAA_TOURNAMENT_PRIMARY_WEIGHT", "0.35")
)
NCAA_TOURNAMENT_MARKET_BLEND_ENABLED = (
    os.getenv("HOOPS_NCAA_TOURNAMENT_MARKET_BLEND_ENABLED", "0") == "1"
)
NCAA_TOURNAMENT_MARKET_WEIGHT = float(
    os.getenv("HOOPS_NCAA_TOURNAMENT_MARKET_WEIGHT", "0.55")
)
NCAA_TOURNAMENT_DISPLAY_USE_POSTPROCESS = (
    os.getenv("HOOPS_NCAA_TOURNAMENT_DISPLAY_USE_POSTPROCESS", "0") == "1"
)

# Seasons to exclude from training and evaluation (e.g. COVID-shortened 2021)
EXCLUDE_SEASONS: list[int] = [2021]

# Production four-factor adjustment parameters (a0.85_p10 config)
ADJUST_FF = True
ADJUST_ALPHA = 0.85
ADJUST_PRIOR = 10

# Four-factor adjustment method: "none", "multiplicative", "iterative"
ADJUST_FF_METHOD = "multiplicative"

# Extra feature groups included in the 54-feature production model
EXTRA_FEATURES = [
    "rest_days",
    "sos",
    "conf_strength",
    "form_delta",
    "tov_rate",
    "margin_std",
]

# The 53 features in EXACT order expected by the model
# (V3: removed home_team_home, away_def_def_rebound_pct, home_def_def_rebound_pct)
# (V6: added home_team_hca, home_team_efg_home_split, away_team_efg_away_split)
FEATURE_ORDER: list[str] = json.loads(
    (ARTIFACTS_DIR / "feature_order.json").read_text()
)
assert len(FEATURE_ORDER) == 53, f"Expected 53 features, got {len(FEATURE_ORDER)}"
FEATURE_ORDER_NO_FORM_V1: list[str] = [
    feature
    for feature in FEATURE_ORDER
    if feature not in {"home_form_delta", "away_form_delta"}
]
assert len(FEATURE_ORDER_NO_FORM_V1) == 51, (
    f"Expected 51 no-form-v1 features, got {len(FEATURE_ORDER_NO_FORM_V1)}"
)
FEATURE_ORDER_SWAP_SAFE_V2: list[str] = json.loads(
    (ARTIFACTS_DIR / "feature_order_swap_safe_v2.json").read_text()
)
assert len(FEATURE_ORDER_SWAP_SAFE_V2) == 53, (
    f"Expected 53 swap-safe-v2 features, got {len(FEATURE_ORDER_SWAP_SAFE_V2)}"
)

# Benchmark-winning point-prediction baseline, promoted into production mu path.
HGBR_PARAMS = {
    "loss": "absolute_error",
    "learning_rate": 0.05,
    "max_depth": 6,
    "max_iter": 300,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "random_state": 42,
}

LGBM_REG_L2_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.02019353198222356,
    "num_leaves": 110,
    "max_depth": 9,
    "min_child_samples": 60,
    "subsample": 0.6469253206264255,
    "colsample_bytree": 0.4684413154048659,
    "reg_alpha": 1.561017503434852,
    "reg_lambda": 0.004257796528177626,
    "n_estimators": 5000,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}

# Optional inference-time sigma sharpening. Disabled by default.
_sigma_cap = os.getenv("HOOPS_SIGMA_CAP_MAX")
SIGMA_CAP_MAX: float | None = float(_sigma_cap) if _sigma_cap else None
_sigma_mode = os.getenv("HOOPS_SIGMA_CALIBRATION_MODE", "").strip().lower()
SIGMA_CALIBRATION_MODE: str | None = _sigma_mode or ("cap" if SIGMA_CAP_MAX is not None else None)
_sigma_scale = os.getenv("HOOPS_SIGMA_SCALE")
SIGMA_SCALE: float | None = float(_sigma_scale) if _sigma_scale else None
_sigma_affine_a = os.getenv("HOOPS_SIGMA_AFFINE_A")
SIGMA_AFFINE_A: float | None = float(_sigma_affine_a) if _sigma_affine_a else None
_sigma_affine_b = os.getenv("HOOPS_SIGMA_AFFINE_B")
SIGMA_AFFINE_B: float | None = float(_sigma_affine_b) if _sigma_affine_b else None
_sigma_shrink_alpha = os.getenv("HOOPS_SIGMA_SHRINK_ALPHA")
SIGMA_SHRINK_ALPHA: float | None = float(_sigma_shrink_alpha) if _sigma_shrink_alpha else None
_sigma_shrink_target = os.getenv("HOOPS_SIGMA_SHRINK_TARGET")
SIGMA_SHRINK_TARGET: float | None = float(_sigma_shrink_target) if _sigma_shrink_target else None

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
