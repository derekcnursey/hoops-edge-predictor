"""Click CLI for hoops-edge-predictor."""

from __future__ import annotations

import json
import random
from datetime import date

import click
import numpy as np
import pandas as pd

from . import config
from .features import build_features, get_feature_matrix, get_targets, load_lines


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


# ── 1. build-features ──────────────────────────────────────────────


@cli.command()
@click.option("--season", required=True, type=int, help="Season year (e.g. 2026)")
@click.option("--upload-s3", is_flag=True, help="Also upload to S3 gold layer")
@click.option("--no-garbage", is_flag=True, help="Use no-garbage-time efficiency ratings")
def build_features_cmd(season: int, upload_s3: bool, no_garbage: bool):
    """Build the 37-feature matrix for all games in a season."""
    variant = " (no-garbage)" if no_garbage else ""
    click.echo(f"Building features{variant} for season {season}...")

    df = build_features(season, no_garbage=no_garbage)
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
    out_path = config.FEATURES_DIR / f"season_{season}{suffix}_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    click.echo(f"  Saved to: {out_path}")

    if upload_s3:
        from . import s3_reader
        import pyarrow as pa
        import pyarrow.parquet as pq

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
def train(seasons: str, reg_epochs: int, cls_epochs: int, no_garbage: bool):
    """Train MLPRegressor + MLPClassifier on historical features."""
    from .dataset import load_multi_season_features
    from .trainer import (
        fit_scaler,
        save_checkpoint,
        train_classifier,
        train_regressor,
    )

    season_list = _parse_seasons(seasons)
    variant = " (no-garbage)" if no_garbage else ""
    click.echo(f"Loading features{variant} for seasons: {season_list}")

    df = load_multi_season_features(season_list, no_garbage=no_garbage)

    # Drop games with missing scores (unplayed)
    df = df.dropna(subset=["homeScore", "awayScore"])
    click.echo(f"  Training samples: {len(df)}")

    X = get_feature_matrix(df).values.astype(np.float32)
    targets = get_targets(df)
    y_spread = targets["spread_home"].values.astype(np.float32)
    y_win = targets["home_win"].values.astype(np.float32)

    # Fill NaN features with 0 before scaling
    X = np.nan_to_num(X, nan=0.0)

    # Subdirectory for no-garbage variant
    ckpt_subdir = "no_garbage" if no_garbage else None

    # Fit scaler
    click.echo("Fitting StandardScaler...")
    scaler = fit_scaler(X, subdir=ckpt_subdir)
    X_scaled = scaler.transform(X)

    # Train regressor
    click.echo("Training MLPRegressor (Gaussian NLL)...")
    reg_hp = {"epochs": reg_epochs}
    regressor = train_regressor(X_scaled, y_spread, hparams=reg_hp)
    save_checkpoint(regressor, "regressor", hparams=reg_hp, subdir=ckpt_subdir)

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
def tune(seasons: str, trials: int):
    """Optuna hyperparameter search for both models."""
    from .dataset import load_multi_season_features
    from .trainer import fit_scaler
    from .tuner import tune_classifier, tune_regressor

    season_list = _parse_seasons(seasons)
    click.echo(f"Loading features for seasons: {season_list}")

    df = load_multi_season_features(season_list)
    df = df.dropna(subset=["homeScore", "awayScore"])
    click.echo(f"  Tuning samples: {len(df)}")

    X = get_feature_matrix(df).values.astype(np.float32)
    targets = get_targets(df)
    y_spread = targets["spread_home"].values.astype(np.float32)
    y_win = targets["home_win"].values.astype(np.float32)

    X = np.nan_to_num(X, nan=0.0)
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
        game_date = date.today().isoformat()

    click.echo(f"Building features for {game_date}...")
    df = build_features(season, game_date=game_date)
    if df.empty:
        click.echo(f"No games found for {game_date}.")
        return

    click.echo(f"  Games: {len(df)}")

    lines = load_lines(season)
    preds = predict(df, lines_df=lines)

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
    df = build_features(season)
    if df.empty:
        click.echo("No games found.")
        return

    click.echo(f"  Games: {len(df)}")

    lines = load_lines(season)
    preds = predict(df, lines_df=lines)

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
    df = build_features(season)
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
            click.echo("    All 37 features present.")

        # Flag outliers (values > 5 std from mean)
        for col in config.FEATURE_ORDER:
            val = feats[col]
            if pd.notna(val):
                col_mean = feat_matrix[col].mean()
                col_std = feat_matrix[col].std()
                if col_std > 0 and abs(val - col_mean) > 5 * col_std:
                    click.echo(f"    OUTLIER: {col} = {val:.4f} (mean={col_mean:.4f}, std={col_std:.4f})")


if __name__ == "__main__":
    cli()
