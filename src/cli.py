"""Click CLI for hoops-edge-predictor."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import click
import numpy as np
import pandas as pd

from . import config
from .efficiency_blend import blend_enabled
from .features import (
    build_features,
    get_feature_matrix,
    get_targets,
    load_boxscores,
    load_efficiency_ratings,
    load_games,
    load_lines,
)
from .live_audits import audit_hrb_lines, audit_live_feature_drift, audit_ratings_asof
from .line_selection import select_preferred_lines
from .tournament_adjustments import needs_secondary_mu_features
from .trainer import load_scaler

_ET = ZoneInfo("America/New_York")
_CRITICAL_FEATURE_COLS = [
    "home_team_adj_oe",
    "away_team_adj_oe",
    "home_team_adj_de",
    "away_team_adj_de",
    "home_tov_rate",
    "away_tov_rate",
    "home_def_tov_rate",
    "away_def_tov_rate",
]


def _format_matchup(row: pd.Series) -> str:
    away = row.get("awayTeam") or row.get("awayTeamId") or "?"
    home = row.get("homeTeam") or row.get("homeTeamId") or "?"
    return f"{away} @ {home}"


def _normalize_raw_games_for_preflight(df: pd.DataFrame) -> pd.DataFrame:
    """Project raw game rows down to prediction-relevant fields for conflict checks."""
    if df.empty:
        return df
    out = df.copy()
    rename = {}
    for target, candidates in [
        ("gameId", ["gameId", "gameid"]),
        ("homeTeamId", ["homeTeamId", "hometeamid"]),
        ("awayTeamId", ["awayTeamId", "awayteamid"]),
        ("homeScore", ["homeScore", "homePoints", "homescore"]),
        ("awayScore", ["awayScore", "awayPoints", "awayscore"]),
        ("neutralSite", ["neutralSite", "neutralsite"]),
        ("startDate", ["startDate", "startTime", "date", "startdate"]),
    ]:
        for cand in candidates:
            if cand in out.columns:
                rename[cand] = target
                break
    out = out.rename(columns=rename)
    keep_cols = [c for c in ["gameId", "homeTeamId", "awayTeamId", "homeScore", "awayScore", "neutralSite", "startDate"] if c in out.columns]
    return out[keep_cols].copy()


def _run_prediction_preflight(season: int, game_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run small integrity checks before generating a live slate."""
    from . import s3_reader

    click.echo("\nPreflight integrity check...")

    # Raw schedule duplicate audit
    games_raw_tbl = s3_reader.read_silver_table(config.TABLE_FCT_GAMES, season=season)
    games_raw = games_raw_tbl.to_pandas() if games_raw_tbl.num_rows else pd.DataFrame()
    games_raw_norm = _normalize_raw_games_for_preflight(games_raw)
    if not games_raw_norm.empty and "startDate" in games_raw_norm.columns:
        raw_start = pd.to_datetime(games_raw_norm["startDate"], utc=True, errors="coerce")
        slate_date = pd.Timestamp(game_date).tz_localize(_ET)
        raw_slate_mask = raw_start.dt.tz_convert(_ET).dt.normalize() == slate_date.normalize()
        games_raw_norm = games_raw_norm.loc[raw_slate_mask].copy()
    raw_game_dup_rows = 0
    raw_game_dup_keys = 0
    raw_game_conflicting_keys = 0
    if not games_raw_norm.empty and "gameId" in games_raw_norm.columns:
        raw_game_dup_mask = games_raw_norm.duplicated(subset=["gameId"], keep=False)
        raw_game_dup_rows = int(raw_game_dup_mask.sum())
        raw_game_dup_keys = int(games_raw_norm.loc[raw_game_dup_mask, "gameId"].nunique())
        if raw_game_dup_keys:
            compare_cols = [c for c in games_raw_norm.columns if c != "gameId"]
            raw_game_conflicting_keys = int(
                games_raw_norm.loc[raw_game_dup_mask, ["gameId", *compare_cols]]
                .fillna("__NA__")
                .groupby("gameId", sort=False)[compare_cols]
                .apply(lambda g: len(g.drop_duplicates()) > 1)
                .sum()
            )

    games = load_games(season)
    slate_games = games.copy()
    if not slate_games.empty and "startDate" in slate_games.columns:
        slate_start = pd.to_datetime(slate_games["startDate"], utc=True, errors="coerce")
        slate_date = pd.Timestamp(game_date)
        slate_mask = slate_start.dt.tz_convert(_ET).dt.normalize() == slate_date.tz_localize(_ET)
        slate_games = slate_games.loc[slate_mask].copy()

    click.echo(
        "  Games table: "
        f"{len(games_raw_norm)} raw slate rows, {raw_game_dup_keys} duplicate gameId key(s), "
        f"{raw_game_conflicting_keys} conflicting duplicate key(s)"
    )

    # Boxscore duplicate audit after loader dedupe.
    boxscores = load_boxscores(season)
    boxscore_dup_keys = 0
    if not boxscores.empty and {"gameid", "teamid"}.issubset(boxscores.columns):
        boxscore_dup_keys = int(
            boxscores.duplicated(subset=["gameid", "teamid"], keep=False).sum()
        )
    click.echo(
        f"  Team-game boxscores: {len(boxscores)} rows after load, {boxscore_dup_keys} duplicate key row(s)"
    )

    # Efficiency duplicate audit, raw vs protected load.
    ratings_raw_tbl = s3_reader.read_gold_table(config.PRODUCTION_GOLD_RATINGS_TABLE, season=season)
    ratings_raw = ratings_raw_tbl.to_pandas() if ratings_raw_tbl.num_rows else pd.DataFrame()
    raw_rating_dup_rows = 0
    raw_rating_dup_keys = 0
    if not ratings_raw.empty and {"teamId", "rating_date"}.issubset(ratings_raw.columns):
        ratings_raw["rating_date"] = pd.to_datetime(ratings_raw["rating_date"], errors="coerce")
        raw_rating_dup_mask = ratings_raw.duplicated(subset=["teamId", "rating_date"], keep=False)
        raw_rating_dup_rows = int(raw_rating_dup_mask.sum())
        raw_rating_dup_keys = int(
            ratings_raw.loc[raw_rating_dup_mask, ["teamId", "rating_date"]]
            .drop_duplicates()
            .shape[0]
        )
    ratings = load_efficiency_ratings(season, no_garbage=True)
    rating_dup_rows_after = 0
    if not ratings.empty and {"teamId", "rating_date"}.issubset(ratings.columns):
        rating_dup_rows_after = int(
            ratings.duplicated(subset=["teamId", "rating_date"], keep=False).sum()
        )
    click.echo(
        "  Efficiency ratings: "
        f"{len(ratings_raw)} raw rows, {raw_rating_dup_keys} duplicate team/date key(s), "
        f"{rating_dup_rows_after} duplicate row(s) after load"
    )

    # Build the actual live feature rows once and reuse them for prediction.
    features_df = build_features(
        season,
        game_date=game_date,
        extra_features=config.EXTRA_FEATURES,
        adjust_ff=config.ADJUST_FF,
        adjust_alpha=config.ADJUST_ALPHA,
        adjust_prior_weight=config.ADJUST_PRIOR,
        efficiency_source=config.EFFICIENCY_SOURCE,
    )
    click.echo(f"  Feature rows for {game_date}: {len(features_df)}")

    lines = load_lines(season)
    preferred_lines = select_preferred_lines(lines)
    missing_line_games = pd.DataFrame()
    if not slate_games.empty:
        merged_lines = slate_games.merge(
            preferred_lines[["gameId", "book_spread"]] if not preferred_lines.empty else pd.DataFrame(columns=["gameId", "book_spread"]),
            on="gameId",
            how="left",
        )
        missing_line_games = merged_lines[merged_lines["book_spread"].isna()].copy()
        click.echo(
            "  Preferred lines for today's slate: "
            f"{len(slate_games) - len(missing_line_games)}/{len(slate_games)} with spreads"
        )
        if not missing_line_games.empty:
            click.echo("  Missing preferred line rows:")
            for _, row in missing_line_games.sort_values("startDate").iterrows():
                click.echo(f"    - {_format_matchup(row)}")

    feature_missing = {}
    if not features_df.empty:
        for col in _CRITICAL_FEATURE_COLS:
            feature_missing[col] = int(features_df[col].isna().sum()) if col in features_df.columns else len(features_df)
    if feature_missing:
        click.echo("  Critical feature missingness:")
        for col in _CRITICAL_FEATURE_COLS:
            click.echo(f"    {col}: {feature_missing.get(col, len(features_df))}")

    audit_reports = []
    try:
        scaler = load_scaler()
    except Exception as exc:
        click.echo(f"  Live feature drift audit skipped: {exc}")
    else:
        audit_reports.append(
            audit_live_feature_drift(
                features_df,
                scaler,
                config.FEATURE_ORDER,
                _CRITICAL_FEATURE_COLS,
            )
        )
    audit_reports.append(audit_ratings_asof(slate_games, ratings))
    audit_reports.append(audit_hrb_lines(slate_games, lines, preferred_lines))

    for report in audit_reports:
        click.echo(f"  {report.label}:")
        for line in report.info:
            click.echo(f"    {line}")

    errors: list[str] = []
    warnings: list[str] = []

    if raw_game_conflicting_keys > 0:
        errors.append(f"fct_games has {raw_game_conflicting_keys} conflicting duplicate gameId key(s)")
    elif raw_game_dup_keys > 0:
        warnings.append(
            f"fct_games has {raw_game_dup_keys} duplicate gameId key(s), but the prediction-relevant fields are identical"
        )
    if boxscore_dup_keys > 0:
        errors.append(f"load_boxscores returned {boxscore_dup_keys} duplicate (gameid, teamid) row(s)")
    if rating_dup_rows_after > 0:
        errors.append(f"load_efficiency_ratings returned {rating_dup_rows_after} duplicate team/date row(s)")
    if not features_df.empty:
        for col in _CRITICAL_FEATURE_COLS[:4]:
            if feature_missing.get(col, 0) > 0:
                errors.append(f"critical efficiency feature {col} missing for {feature_missing[col]} row(s)")
        for col in _CRITICAL_FEATURE_COLS[4:]:
            if feature_missing.get(col, 0) > 0:
                warnings.append(f"turnover feature {col} missing for {feature_missing[col]} row(s)")
    if raw_rating_dup_keys > 0:
        warnings.append(
            f"raw gold ratings table still contains {raw_rating_dup_keys} duplicate team/date key(s); loader dedupe protected this run"
        )
    if not missing_line_games.empty:
        warnings.append(f"{len(missing_line_games)} scheduled game(s) have no preferred line")
    for report in audit_reports:
        errors.extend(report.errors)
        warnings.extend(report.warnings)

    if errors:
        click.echo("  Preflight errors:", err=True)
        for msg in errors:
            click.echo(f"    - {msg}", err=True)
        raise click.ClickException("preflight checks failed")

    if warnings:
        click.echo("  Preflight warnings:")
        for msg in warnings:
            click.echo(f"    - {msg}")

    click.echo("  Preflight passed.")
    return features_df, lines


def _today_et() -> str:
    """Today's date in US Eastern Time as YYYY-MM-DD."""
    return datetime.now(_ET).strftime("%Y-%m-%d")


@click.group()
def cli():
    """hoops-edge-predictor: College basketball game predictions."""
    pass


def _parse_seasons(seasons_str: str) -> list[int]:
    """Parse '2015-2025' or '2015,2016,2025' into a list of ints."""
    if "-" in seasons_str and "," not in seasons_str:
        start, end = seasons_str.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(s.strip()) for s in seasons_str.split(",")]


def _exclude_training_seasons(seasons: list[int]) -> list[int]:
    """Remove globally excluded seasons from training/tuning inputs."""
    return [season for season in seasons if season not in config.EXCLUDE_SEASONS]


def _production_regressor_hparams(
    efficiency_source: str,
    reg_epochs: int,
) -> dict[str, int | float]:
    """Keep the gold-backed sigma fit on a stable learning-rate/batch-size regime."""
    hp: dict[str, int | float] = {"epochs": reg_epochs}
    if efficiency_source == "gold":
        hp["lr"] = 1e-3
        hp["batch_size"] = 1024
    return hp


def _build_secondary_mu_features_if_needed(
    season: int,
    primary_df: pd.DataFrame,
    game_date: str | None = None,
) -> pd.DataFrame | None:
    """Build Torvik-side features only when the blend schedule needs them."""
    if config.EFFICIENCY_SOURCE != "gold":
        return None
    if not blend_enabled() or primary_df.empty:
        return None
    if not needs_secondary_mu_features(primary_df):
        return None
    secondary_df = build_features(
        season,
        game_date=game_date,
        no_garbage=True,
        extra_features=config.EXTRA_FEATURES,
        adjust_ff=config.ADJUST_FF,
        adjust_alpha=config.ADJUST_ALPHA,
        adjust_prior_weight=config.ADJUST_PRIOR,
        efficiency_source="torvik",
    )
    if secondary_df.empty:
        return None
    return secondary_df


# ── 1. build-features ──────────────────────────────────────────────


@cli.command()
@click.option("--season", required=True, type=int, help="Season year (e.g. 2026)")
@click.option("--upload-s3", is_flag=True, help="Also upload to S3 gold layer")
@click.option("--no-garbage", is_flag=True, help="Use no-garbage-time efficiency ratings")
@click.option("--adjusted/--no-adjusted", default=True,
              help="Use opponent-adjusted four-factors (default: True)")
@click.option("--adjust-ff-method", default=config.ADJUST_FF_METHOD,
              type=click.Choice(["multiplicative", "iterative"]),
              help="FF adjustment method (default: from config)")
@click.option("--efficiency-source", default=config.EFFICIENCY_SOURCE,
              type=click.Choice(["gold", "torvik"]),
              help="Efficiency rating source (default: from config)")
def build_features_cmd(season: int, upload_s3: bool, no_garbage: bool,
                       adjusted: bool, adjust_ff_method: str,
                       efficiency_source: str):
    """Build the 54-feature matrix for all games in a season."""
    variant = " (no-garbage)" if no_garbage else ""
    if efficiency_source == "torvik":
        variant += " (torvik)"
    if adjusted:
        variant += f" ({adjust_ff_method} adj a={config.ADJUST_ALPHA} p={config.ADJUST_PRIOR})"
    click.echo(f"Building features{variant} for season {season}...")

    df = build_features(
        season,
        no_garbage=no_garbage,
        extra_features=config.EXTRA_FEATURES if adjusted else None,
        adjust_ff=adjusted and config.ADJUST_FF,
        adjust_alpha=config.ADJUST_ALPHA,
        adjust_prior_weight=config.ADJUST_PRIOR,
        adjust_ff_method=adjust_ff_method,
        efficiency_source=efficiency_source,
    )
    if df.empty:
        click.echo("No games found. Check S3 data.")
        return

    # Report stats
    n_games = len(df)
    feat_matrix = get_feature_matrix(df)
    n_cols = feat_matrix.shape[1]
    null_per_col = feat_matrix.isnull().sum()
    n_nulls = null_per_col.sum()
    rows_zero_nulls = (feat_matrix.isnull().sum(axis=1) == 0).sum()
    pct_zero_nulls = 100.0 * rows_zero_nulls / n_games if n_games > 0 else 0.0
    click.echo(f"  Games (rows): {n_games}")
    click.echo(f"  Feature columns: {n_cols}")
    click.echo(f"  Total nulls: {n_nulls}")
    click.echo(f"  Rows with zero nulls: {rows_zero_nulls}/{n_games} ({pct_zero_nulls:.1f}%)")
    if n_nulls > 0:
        click.echo("  Null count per column:")
        for col in config.FEATURE_ORDER:
            cnt = null_per_col.get(col, 0)
            if cnt > 0:
                click.echo(f"    {col}: {cnt}")

    # Save locally
    suffix = "_no_garbage" if no_garbage else ""
    if efficiency_source == "torvik":
        suffix += "_torvik"
    if adjusted:
        if adjust_ff_method == "iterative":
            suffix += f"_adj_iter_p{config.ADJUST_PRIOR}"
        else:
            suffix += f"_adj_a{config.ADJUST_ALPHA}_p{config.ADJUST_PRIOR}"
    out_path = config.FEATURES_DIR / f"season_{season}{suffix}_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    click.echo(f"  Saved to: {out_path}")

    if upload_s3:
        from . import s3_reader
        import pyarrow as pa

        key = f"{config.GOLD_PREFIX}/game_predictions_37feat/season={season}/features.parquet"
        tbl = pa.Table.from_pandas(df)
        s3_reader.write_parquet_to_s3(tbl, key)
        click.echo(f"  Uploaded to s3://{config.S3_BUCKET}/{key}")


# ── 2. train ───────────────────────────────────────────────────────


@cli.command()
@click.option("--seasons", required=True, help="Training seasons (e.g. '2015-2025')")
@click.option("--reg-epochs", default=100, type=int, help="Regressor training epochs")
@click.option("--cls-epochs", default=100, type=int, help="Classifier training epochs")
@click.option("--no-garbage", is_flag=True, help="Use no-garbage-time features")
@click.option("--adj-suffix", default=None, type=str,
              help="Adjustment suffix (e.g. 'adj_a0.85_p10')")
@click.option("--min-date", default="12-01", type=str,
              help="Earliest MM-DD within each season to include (default: 12-01)")
@click.option("--efficiency-source", default=config.EFFICIENCY_SOURCE,
              type=click.Choice(["gold", "torvik"]),
              help="Efficiency rating source (default: from config)")
def train(seasons: str, reg_epochs: int, cls_epochs: int, no_garbage: bool,
          adj_suffix: str | None, min_date: str | None, efficiency_source: str):
    """Train MLPRegressor + MLPClassifier on historical features."""
    from .dataset import load_multi_season_features
    from .trainer import (
        fit_scaler,
        impute_column_means,
        save_checkpoint,
        save_tree_regressor,
        train_lightgbm_regressor,
        train_classifier,
        train_regressor,
    )

    requested_seasons = _parse_seasons(seasons)
    season_list = _exclude_training_seasons(requested_seasons)
    excluded = sorted(set(requested_seasons) - set(season_list))
    variant = " (no-garbage)" if no_garbage else ""
    if efficiency_source == "torvik":
        variant += " (torvik)"
    if adj_suffix:
        variant += f" ({adj_suffix})"
    click.echo(f"Loading features{variant} for seasons: {season_list}")
    if excluded:
        click.echo(f"  Excluding seasons per config: {excluded}")
    if min_date:
        click.echo(f"  Training date filter: games on or after MM-DD={min_date}")

    df = load_multi_season_features(season_list, no_garbage=no_garbage,
                                    adj_suffix=adj_suffix,
                                    min_month_day=min_date,
                                    efficiency_source=efficiency_source)

    # Drop games with missing scores (unplayed)
    df = df.dropna(subset=["homeScore", "awayScore"])
    n_before_zero = len(df)

    # Remove 0-0 games — data errors where scores are recorded as 0 instead of NULL
    df = df[(df["homeScore"] != 0) | (df["awayScore"] != 0)]
    n_removed_zero = n_before_zero - len(df)
    if n_removed_zero > 0:
        click.echo(f"  Removed {n_removed_zero} bogus 0-0 games")
    click.echo(f"  Training samples: {len(df)}")

    X = get_feature_matrix(df).values.astype(np.float32)
    targets = get_targets(df)
    y_spread = targets["spread_home"].values.astype(np.float32)
    y_win = targets["home_win"].values.astype(np.float32)

    # Impute NaN with column means (not zero — zero-fill distorts the scaler)
    n_nan = np.isnan(X).sum()
    X = impute_column_means(X)
    if n_nan > 0:
        click.echo(f"  Imputed {n_nan:,} NaN values with column means")

    # Scaler always saves to canonical path (artifacts/scaler.pkl) so infer.py
    # can find it regardless of no_garbage setting.
    ckpt_subdir = None

    # Fit scaler
    click.echo("Fitting StandardScaler...")
    scaler = fit_scaler(X, subdir=ckpt_subdir)
    X_scaled = scaler.transform(X)

    # Train regressor
    click.echo("Training MLPRegressor (Gaussian NLL)...")
    reg_hp = _production_regressor_hparams(efficiency_source, reg_epochs)
    regressor = train_regressor(X_scaled, y_spread, hparams=reg_hp)
    save_checkpoint(regressor, "regressor", hparams=reg_hp, subdir=ckpt_subdir)

    # Train production mu regressor on raw imputed features.
    click.echo("Training LightGBMRegressor (mu)...")
    tree_regressor = train_lightgbm_regressor(X, y_spread)
    tree_path = save_tree_regressor(tree_regressor, feature_order=config.FEATURE_ORDER)
    click.echo(f"  Tree regressor: {tree_path}")

    if blend_enabled() and efficiency_source == "gold":
        click.echo("Training Torvik LightGBMRegressor (mu blend side)...")
        torvik_df = load_multi_season_features(
            season_list,
            no_garbage=no_garbage,
            adj_suffix=adj_suffix,
            min_month_day=min_date,
            efficiency_source="torvik",
        )
        torvik_df = torvik_df.dropna(subset=["homeScore", "awayScore"])
        torvik_df = torvik_df[(torvik_df["homeScore"] != 0) | (torvik_df["awayScore"] != 0)]
        X_t = get_feature_matrix(torvik_df).values.astype(np.float32)
        y_t = get_targets(torvik_df)["spread_home"].values.astype(np.float32)
        X_t = impute_column_means(X_t)
        torvik_tree = train_lightgbm_regressor(X_t, y_t)
        torvik_tree_path = save_tree_regressor(
            torvik_tree,
            path=config.TORVIK_TREE_REGRESSOR_PATH,
            feature_order=config.FEATURE_ORDER,
        )
        click.echo(f"  Torvik tree regressor: {torvik_tree_path}")

    # Train classifier
    click.echo("Training MLPClassifier (BCE)...")
    cls_hp = {"epochs": cls_epochs}
    classifier = train_classifier(X_scaled, y_win, hparams=cls_hp)
    save_checkpoint(classifier, "classifier", hparams=cls_hp, subdir=ckpt_subdir)

    click.echo("Training complete.")


# ── 3. tune ────────────────────────────────────────────────────────


@cli.command()
@click.option("--seasons", required=True, help="Training seasons (e.g. '2015-2025')")
@click.option("--trials", default=50, type=int, help="Number of Optuna trials")
@click.option("--min-date", default="12-01", type=str,
              help="Earliest MM-DD within each season to include (default: 12-01)")
@click.option("--no-garbage", is_flag=True, default=True,
              help="Use no-garbage-time features (default: True)")
@click.option("--adj-suffix", default="adj_a0.85_p10", type=str,
              help="Adjustment suffix for feature files")
@click.option("--efficiency-source", default=config.EFFICIENCY_SOURCE,
              type=click.Choice(["gold", "torvik"]),
              help="Efficiency rating source (default: from config)")
def tune(seasons: str, trials: int, min_date: str | None,
         no_garbage: bool, adj_suffix: str | None, efficiency_source: str):
    """Optuna hyperparameter search for both models."""
    from .dataset import load_multi_season_features
    from .trainer import fit_scaler, impute_column_means
    from .tuner import tune_classifier, tune_regressor

    requested_seasons = _parse_seasons(seasons)
    season_list = _exclude_training_seasons(requested_seasons)
    excluded = sorted(set(requested_seasons) - set(season_list))
    variant = " (no-garbage)" if no_garbage else ""
    if efficiency_source == "torvik":
        variant += " (torvik)"
    if adj_suffix:
        variant += f" ({adj_suffix})"
    click.echo(f"Loading features{variant} for seasons: {season_list}")
    if excluded:
        click.echo(f"  Excluding seasons per config: {excluded}")
    if min_date:
        click.echo(f"  Tuning date filter: games on or after MM-DD={min_date}")

    df = load_multi_season_features(season_list, no_garbage=no_garbage,
                                    adj_suffix=adj_suffix,
                                    min_month_day=min_date,
                                    efficiency_source=efficiency_source)
    df = df.dropna(subset=["homeScore", "awayScore"])
    df = df[(df["homeScore"] != 0) | (df["awayScore"] != 0)]
    click.echo(f"  Tuning samples: {len(df)}")

    X = get_feature_matrix(df).values.astype(np.float32)
    targets = get_targets(df)
    y_spread = targets["spread_home"].values.astype(np.float32)
    y_win = targets["home_win"].values.astype(np.float32)

    X = impute_column_means(X)
    scaler = fit_scaler(X)
    X_scaled = scaler.transform(X)

    click.echo(f"Tuning regressor ({trials} trials)...")
    reg_params = tune_regressor(X_scaled, y_spread, n_trials=trials)

    click.echo(f"Tuning classifier ({trials} trials)...")
    cls_params = tune_classifier(X_scaled, y_win, n_trials=trials)

    # Save best params
    params_path = config.ARTIFACTS_DIR / "best_hparams.json"
    with open(params_path, "w") as f:
        json.dump({"regressor": reg_params, "classifier": cls_params}, f, indent=2)
    click.echo(f"  Saved best hyperparameters to {params_path}")


# ── 4. predict-today ───────────────────────────────────────────────


@cli.command("predict-today")
@click.option("--season", required=True, type=int)
@click.option("--date", "game_date", default=None, help="Date override (YYYY-MM-DD)")
def predict_today(season: int, game_date: str | None):
    """Predict today's games."""
    from .infer import predict, save_predictions

    if game_date is None:
        game_date = _today_et()

    click.echo(f"Building features for {game_date}...")
    df, lines = _run_prediction_preflight(season, game_date)
    if df.empty:
        click.echo(f"No games found for {game_date}.")
        return

    click.echo(f"  Games: {len(df)}")
    secondary_df = _build_secondary_mu_features_if_needed(season, df, game_date=game_date)
    preds = predict(df, lines_df=lines, secondary_mu_features_df=secondary_df)

    json_path, csv_path = save_predictions(preds, game_date=game_date)
    click.echo(f"  JSON: {json_path}")
    click.echo(f"  CSV:  {csv_path}")

    # Sort by |spread_diff| descending if available
    if "spread_diff" in preds.columns:
        preds = preds.copy()
        preds["_abs_diff"] = preds["spread_diff"].abs()
        preds = preds.sort_values("_abs_diff", ascending=False).drop(columns=["_abs_diff"])

    # Print summary with team names
    has_names = "homeTeam" in preds.columns and "awayTeam" in preds.columns
    has_book = "book_spread" in preds.columns

    for _, row in preds.iterrows():
        spread = row.get("predicted_spread", 0)
        sigma = row.get("spread_sigma", 0)
        prob = row.get("home_win_prob", 0.5)
        if has_names:
            away = str(row["awayTeam"])[:20]
            home = str(row["homeTeam"])[:20]
            matchup = f"{away:>20} @ {home:<20}"
        else:
            matchup = f"{int(row['awayTeamId']):>6} @ {int(row['homeTeamId']):<6}"

        line = f"  {matchup} | spread: {spread:+.1f} | sigma: {sigma:.1f} | P(home): {prob:.1%}"
        if has_book:
            book = row.get("book_spread")
            diff = row.get("spread_diff")
            if pd.notna(book) and pd.notna(diff):
                line += f" | book: {book:+.1f} | diff: {diff:+.1f}"
        click.echo(line)


# ── 5. predict-season ─────────────────────────────────────────────


@cli.command("predict-season")
@click.option("--season", required=True, type=int)
def predict_season(season: int):
    """Predict all games in a season."""
    from .infer import predict, save_predictions

    click.echo(f"Building features for full season {season}...")
    df = build_features(
        season,
        extra_features=config.EXTRA_FEATURES,
        adjust_ff=config.ADJUST_FF,
        adjust_alpha=config.ADJUST_ALPHA,
        adjust_prior_weight=config.ADJUST_PRIOR,
        efficiency_source=config.EFFICIENCY_SOURCE,
    )
    if df.empty:
        click.echo("No games found.")
        return

    click.echo(f"  Games: {len(df)}")

    lines = load_lines(season)
    secondary_df = _build_secondary_mu_features_if_needed(season, df)
    preds = predict(df, lines_df=lines, secondary_mu_features_df=secondary_df)

    json_path, csv_path = save_predictions(preds, game_date=f"season_{season}")
    click.echo(f"  JSON: {json_path}")
    click.echo(f"  CSV:  {csv_path}")
    click.echo(f"  Total predictions: {len(preds)}")


# ── 6. validate-features ──────────────────────────────────────────


@cli.command("validate-features")
@click.option("--season", required=True, type=int)
@click.option("--n-samples", default=10, type=int, help="Number of games to spot-check")
def validate_features(season: int, n_samples: int):
    """Spot-check features for random games in a season."""
    click.echo(f"Building features for season {season}...")
    df = build_features(
        season,
        extra_features=config.EXTRA_FEATURES,
        adjust_ff=config.ADJUST_FF,
        adjust_alpha=config.ADJUST_ALPHA,
        adjust_prior_weight=config.ADJUST_PRIOR,
        efficiency_source=config.EFFICIENCY_SOURCE,
    )
    if df.empty:
        click.echo("No games found.")
        return

    feat_matrix = get_feature_matrix(df)
    n_games = len(df)
    n_nulls_per_col = feat_matrix.isnull().sum()
    total_nulls = n_nulls_per_col.sum()

    click.echo(f"\n=== Season {season} Feature Summary ===")
    click.echo(f"  Total games: {n_games}")
    click.echo(f"  Feature columns: {feat_matrix.shape[1]}")
    click.echo(f"  Total null values: {total_nulls}")

    # Null breakdown
    if total_nulls > 0:
        click.echo("\n  Null counts by feature:")
        for col in config.FEATURE_ORDER:
            count = n_nulls_per_col.get(col, 0)
            if count > 0:
                click.echo(f"    {col}: {count}")

    # Feature stats
    click.echo("\n  Feature ranges:")
    for col in config.FEATURE_ORDER:
        vals = feat_matrix[col].dropna()
        if len(vals) > 0:
            click.echo(f"    {col}: [{vals.min():.4f}, {vals.max():.4f}] mean={vals.mean():.4f}")

    # Spot-check random games
    sample_idx = random.sample(range(n_games), min(n_samples, n_games))
    click.echo(f"\n  Spot-checking {len(sample_idx)} random games:")
    for idx in sample_idx:
        row = df.iloc[idx]
        click.echo(f"\n  Game {int(row['gameId'])} | {row.get('startDate', 'N/A')}")
        click.echo(f"    Away ({int(row['awayTeamId'])}) @ Home ({int(row['homeTeamId'])})")
        feats = feat_matrix.iloc[idx]
        nulls = feats.isnull().sum()
        if nulls > 0:
            click.echo(f"    WARNING: {nulls} null features!")
            null_cols = [c for c in config.FEATURE_ORDER if pd.isnull(feats[c])]
            click.echo(f"    Null columns: {null_cols}")
        else:
            click.echo(f"    All {len(config.FEATURE_ORDER)} features present.")

        # Flag outliers (values > 5 std from mean)
        for col in config.FEATURE_ORDER:
            val = feats[col]
            if pd.notna(val):
                col_mean = feat_matrix[col].mean()
                col_std = feat_matrix[col].std()
                if col_std > 0 and abs(val - col_mean) > 5 * col_std:
                    click.echo(f"    OUTLIER: {col} = {val:.4f} (mean={col_mean:.4f}, std={col_std:.4f})")


# ── 7. build-rankings ─────────────────────────────────────────────


@cli.command("build-rankings")
@click.option("--season", default=2026, type=int, help="Season year (e.g. 2026)")
def build_rankings(season: int):
    """Build power rankings JSON from S3 efficiency ratings."""
    script = config.PROJECT_ROOT / "scripts" / "build_rankings_json.py"
    subprocess.run(
        [sys.executable, str(script), "--season", str(season)],
        check=True,
        cwd=config.PROJECT_ROOT,
    )


# ── 9. backfill-season ────────────────────────────────────────────


@cli.command("backfill-season")
@click.option("--season", required=True, type=int, help="Season year (e.g. 2026)")
@click.option("--start-date", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end-date", default=None, help="End date (YYYY-MM-DD, default: yesterday)")
@click.option("--skip-existing/--no-skip-existing", default=True,
              help="Skip dates that already have predictions JSON")
def backfill_season(season: int, start_date: str, end_date: str | None,
                    skip_existing: bool):
    """Backfill predictions for a date range."""
    from .infer import predict, save_predictions

    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = (
        datetime.strptime(end_date, "%Y-%m-%d").date()
        if end_date
        else datetime.now(_ET).date() - timedelta(days=1)
    )

    click.echo(f"=== Backfill season {season}: {start} → {end} ===")

    lines = load_lines(season)
    script_dir = config.PROJECT_ROOT / "scripts"

    processed = 0
    current = start
    while current <= end:
        game_date = current.isoformat()
        current += timedelta(days=1)

        # Check if already exists
        if skip_existing:
            existing_json = config.SITE_DATA_DIR / f"predictions_{game_date}.json"
            if existing_json.exists():
                continue

        # Build features for this date
        df = build_features(
            season,
            game_date=game_date,
            extra_features=config.EXTRA_FEATURES,
            adjust_ff=config.ADJUST_FF,
            adjust_alpha=config.ADJUST_ALPHA,
            adjust_prior_weight=config.ADJUST_PRIOR,
            efficiency_source=config.EFFICIENCY_SOURCE,
        )
        if df.empty:
            continue

        # Predict and save (includes site JSON)
        preds = predict(df, lines_df=lines)
        save_predictions(preds, game_date=game_date)

        processed += 1
        click.echo(f"  {game_date}: {len(preds)} games")

    click.echo(f"\nProcessed {processed} dates.")

    # Run s3_finals_to_json once at end
    s3_finals = script_dir / "s3_finals_to_json.py"
    if s3_finals.exists():
        click.echo("Fetching final scores from S3...")
        subprocess.run(
            [sys.executable, str(s3_finals)],
            check=True,
            cwd=config.PROJECT_ROOT,
        )
        click.echo("Final scores complete.")


# ── 9. publish-site ───────────────────────────────────────────────


@cli.command("publish-site")
@click.option("--message", default=None, help="Custom commit message")
def publish_site(message: str | None):
    """Git commit and push site/public/data/ to deploy via Vercel."""
    import subprocess

    today_str = _today_et()
    msg = message or f"Update predictions {today_str}"

    click.echo("Staging site data files...")
    subprocess.run(
        ["git", "add", "site/public/data/"],
        check=True,
        cwd=config.PROJECT_ROOT,
    )

    # Check if there are staged changes
    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=config.PROJECT_ROOT,
    )
    if result.returncode == 0:
        click.echo("No changes to commit.")
        return

    click.echo(f"Committing: {msg}")
    subprocess.run(
        ["git", "commit", "-m", msg],
        check=True,
        cwd=config.PROJECT_ROOT,
    )

    click.echo("Pushing to origin main...")
    subprocess.run(
        ["git", "push", "origin", "main"],
        check=True,
        cwd=config.PROJECT_ROOT,
    )
    click.echo("Published.")


# ── 10. daily-update ─────────────────────────────────────────────


def _get_etl_root() -> Path:
    """Resolve the ETL repo root (sibling dir or CBBD_ETL_ROOT env var)."""
    etl = Path(os.environ.get(
        "CBBD_ETL_ROOT",
        str(config.PROJECT_ROOT.parent / "hoops_edge_database_etl"),
    ))
    if not etl.exists():
        click.echo(f"ETL repo not found at {etl}. Set CBBD_ETL_ROOT env var.", err=True)
        sys.exit(1)
    return etl


def _run(cmd: list[str], cwd: Path, label: str) -> None:
    """Run a subprocess, abort on failure.

    Strips VIRTUAL_ENV from env when cwd is outside the project root so that
    sibling repos (e.g. the ETL repo) use their own poetry virtualenv.
    Loads .env from the target cwd if present.
    """
    env = None
    if not str(cwd).startswith(str(config.PROJECT_ROOT)):
        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        # Load .env from the target repo (e.g. ETL's CBBD_API_KEY)
        dotenv_path = cwd / ".env"
        if dotenv_path.exists():
            for line in dotenv_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    env[key.strip()] = val.strip().strip("'\"")
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        click.echo(f"FAILED: {label} (exit {result.returncode})", err=True)
        sys.exit(result.returncode)


@cli.command("daily-update")
@click.option("--season", required=True, type=int, help="Season year (e.g. 2026)")
@click.option("--date", "game_date", default=None, help="Date override (YYYY-MM-DD)")
@click.option("--skip-etl", is_flag=True, help="Skip ETL ingest (step 1)")
@click.option("--skip-transforms", is_flag=True, help="Skip silver/gold transforms (steps 2-3)")
@click.option("--skip-predict", is_flag=True, help="Skip predictions + publish (steps 4-5)")
@click.option("--skip-deploy", is_flag=True, help="Skip git commit/push (step 6)")
@click.option("--no-lineups", is_flag=True,
              help="Exclude lineups + substitutions from ETL ingest")
def daily_update(season: int, game_date: str | None, skip_etl: bool,
                 skip_transforms: bool, skip_predict: bool, skip_deploy: bool,
                 no_lineups: bool):
    """Full end-to-end daily pipeline: ETL → silver → gold → predict → publish → deploy."""
    from .infer import predict, save_predictions

    if game_date is None:
        game_date = _today_et()

    etl_root = _get_etl_root()
    click.echo(f"=== Daily update for {game_date} (season {season}) ===")

    # ── Step 1: ETL ingest ─────────────────────────────────────────
    if not skip_etl:
        endpoints = "games,games_teams,lines,ratings_adjusted,plays_game"
        if not no_lineups:
            endpoints += ",lineups_game,substitutions_game"
        click.echo(f"\n[1/6] ETL incremental ingest...")
        if no_lineups:
            click.echo("  (skipping lineups + substitutions)")
        _run(
            ["poetry", "run", "python", "-m", "cbbd_etl", "incremental",
             "--season-start", str(season), "--season-end", str(season),
             "--only-endpoints", endpoints],
            cwd=etl_root,
            label="ETL incremental ingest",
        )

    # ── Steps 2-3: Silver + Gold transforms ────────────────────────
    if not skip_transforms:
        # Step 2: Silver — PBP pipeline: enriched → flat (both variants).
        # Incremental: only processes new dates (no --purge).
        click.echo("\n[2/6] Silver transforms (PBP enriched + flat tables)...")
        _run(
            ["poetry", "run", "python", "scripts/build_pbp_plays_enriched.py",
             "--season", str(season)],
            cwd=etl_root,
            label="build_pbp_plays_enriched",
        )
        _run(
            ["poetry", "run", "python", "scripts/build_pbp_game_teams_flat.py",
             "--season", str(season)],
            cwd=etl_root,
            label="build_pbp_game_teams_flat",
        )
        _run(
            ["poetry", "run", "python", "scripts/build_pbp_game_teams_flat.py",
             "--season", str(season), "--exclude-garbage-time",
             "--output-table", "fct_pbp_game_teams_flat_garbage_removed"],
            cwd=etl_root,
            label="build_pbp_game_teams_flat (no garbage)",
        )

        # Step 3: Gold — team_adjusted_efficiencies_no_garbage
        click.echo("\n[3/6] Gold transforms (team_adjusted_efficiencies_no_garbage)...")
        _run(
            ["poetry", "run", "python", "-m", "cbbd_etl.gold.runner",
             "--season", str(season),
             "--table", "team_adjusted_efficiencies_no_garbage"],
            cwd=etl_root,
            label="gold team_adjusted_efficiencies_no_garbage",
        )

        # Keep the preferred in-house ratings source current for rankings.
        repair_script = config.PROJECT_ROOT / "scripts" / f"repair_pbp_garbage_removed_{season}.py"
        promoted_gold_script = config.PROJECT_ROOT / "scripts" / "build_gold_softgarbage_priorreg_v1.py"
        if repair_script.exists():
            click.echo(f"  Refreshing repaired no-garbage gold tables for season {season}...")
            _run(
                [sys.executable, str(repair_script), "--season", str(season)],
                cwd=config.PROJECT_ROOT,
                label=f"repair_pbp_garbage_removed_{season}",
            )
        elif promoted_gold_script.exists():
            click.echo(f"  Refreshing {config.PRODUCTION_GOLD_RATINGS_TABLE}...")
            _run(
                [
                    sys.executable,
                    str(promoted_gold_script),
                    "--season-start",
                    str(season),
                    "--season-end",
                    str(season),
                ],
                cwd=config.PROJECT_ROOT,
                label="build_gold_softkeep25_priorreg_k5_v1",
            )

    # ── Freshness check: ensure gold data is current ────────────────
    click.echo("\nChecking gold data freshness...")
    try:
        from . import s3_reader
        gold_tbl = s3_reader.read_gold_table(config.PRODUCTION_GOLD_RATINGS_TABLE, season=season)
        gold_df = gold_tbl.to_pandas()
        import pandas as _pd
        gold_df["rating_date"] = _pd.to_datetime(gold_df["rating_date"], errors="coerce")
        max_date = gold_df["rating_date"].max()
        if _pd.notna(max_date):
            max_date_str = max_date.strftime("%Y-%m-%d")
            game_dt = _pd.Timestamp(game_date)
            days_stale = (game_dt - max_date).days
            click.echo(
                f"  Gold ratings through: {max_date_str} "
                f"({days_stale} day(s) before {game_date}) from {config.PRODUCTION_GOLD_RATINGS_TABLE}"
            )
            if days_stale > 2:
                click.echo(
                    f"  WARNING: Gold data is {days_stale} days stale! "
                    f"ETL PBP pipeline may have failed.", err=True
                )
                if not skip_etl:
                    click.echo("  ERROR: ETL ran but gold data is still stale. Aborting.", err=True)
                    sys.exit(1)
        else:
            click.echo("  WARNING: Could not determine gold data date.", err=True)
    except Exception as e:
        click.echo(f"  WARNING: Freshness check failed: {e}", err=True)

    # ── Steps 4-5: Predict + publish ──────────────────────────────
    if not skip_predict:
        # Step 4: Predict today's games
        click.echo(f"\n[4/6] Predictions for {game_date}...")
        df, lines = _run_prediction_preflight(season, game_date)
        if df.empty:
            click.echo(f"  No games found for {game_date}. Skipping predictions.")
        else:
            click.echo(f"  Games: {len(df)}")
            secondary_df = _build_secondary_mu_features_if_needed(season, df, game_date=game_date)
            preds = predict(df, lines_df=lines, secondary_mu_features_df=secondary_df)
            json_path, csv_path = save_predictions(preds, game_date=game_date)
            click.echo(f"  JSON: {json_path}")
            click.echo(f"  CSV:  {csv_path}")

        # Step 5: Publish pipeline — rankings → final scores
        # (Site JSON is now written directly by save_predictions())
        click.echo(f"\n[5/6] Publish pipeline...")
        script_dir = config.PROJECT_ROOT / "scripts"

        rankings_script = script_dir / "build_rankings_json.py"
        if rankings_script.exists():
            _run([sys.executable, str(rankings_script)], cwd=config.PROJECT_ROOT,
                 label="build_rankings_json")

        finals_script = script_dir / "s3_finals_to_json.py"
        if finals_script.exists():
            _run([sys.executable, str(finals_script), "--date", game_date],
                 cwd=config.PROJECT_ROOT, label="s3_finals_to_json")

        internal_bet_script = script_dir / "build_daily_internal_bet_filter_report.py"
        if internal_bet_script.exists():
            _run(
                [
                    sys.executable,
                    str(internal_bet_script),
                    "--season",
                    str(season),
                    "--date",
                    game_date,
                ],
                cwd=config.PROJECT_ROOT,
                label="build_daily_internal_bet_filter_report",
            )

        internal_tracking_script = script_dir / "build_internal_bet_filter_tracking_report.py"
        if internal_tracking_script.exists():
            _run(
                [
                    sys.executable,
                    str(internal_tracking_script),
                    "--season",
                    str(season),
                ],
                cwd=config.PROJECT_ROOT,
                label="build_internal_bet_filter_tracking_report",
            )

        internal_maintenance_script = script_dir / "build_internal_bet_filter_maintenance_report.py"
        if internal_maintenance_script.exists():
            _run(
                [
                    sys.executable,
                    str(internal_maintenance_script),
                    "--season",
                    str(season),
                ],
                cwd=config.PROJECT_ROOT,
                label="build_internal_bet_filter_maintenance_report",
            )
        click.echo("  Publish pipeline complete.")

    # ── Step 6: Deploy ────────────────────────────────────────────
    if not skip_deploy:
        click.echo(f"\n[6/6] Deploy (git commit + push)...")
        subprocess.run(
            ["git", "add", "site/public/data/", "predictions/"],
            cwd=config.PROJECT_ROOT,
        )
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=config.PROJECT_ROOT,
        )
        if result.returncode == 0:
            click.echo("  No changes to commit.")
        else:
            msg = f"daily-update {game_date}"
            _run(["git", "commit", "-m", msg], cwd=config.PROJECT_ROOT, label="git commit")
            _run(["git", "push", "origin", "main"], cwd=config.PROJECT_ROOT, label="git push")
            click.echo("  Deployed.")

    click.echo(f"\n=== Daily update complete for {game_date} ===")


if __name__ == "__main__":
    cli()
