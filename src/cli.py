"""Click CLI for hoops-edge-predictor."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

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
@click.option("--adjusted/--no-adjusted", default=True,
              help="Use opponent-adjusted four-factors (default: True)")
def build_features_cmd(season: int, upload_s3: bool, no_garbage: bool, adjusted: bool):
    """Build the 54-feature matrix for all games in a season."""
    variant = " (no-garbage)" if no_garbage else ""
    if adjusted:
        variant += f" (adj a={config.ADJUST_ALPHA} p={config.ADJUST_PRIOR})"
    click.echo(f"Building features{variant} for season {season}...")

    df = build_features(
        season,
        no_garbage=no_garbage,
        extra_features=config.EXTRA_FEATURES if adjusted else None,
        adjust_ff=adjusted and config.ADJUST_FF,
        adjust_alpha=config.ADJUST_ALPHA,
        adjust_prior_weight=config.ADJUST_PRIOR,
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
    if adjusted:
        suffix += f"_adj_a{config.ADJUST_ALPHA}_p{config.ADJUST_PRIOR}"
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
@click.option("--adj-suffix", default=None, type=str,
              help="Adjustment suffix (e.g. 'adj_a0.85_p10')")
@click.option("--min-date", default="12-01", type=str,
              help="Earliest MM-DD within each season to include (default: 12-01)")
def train(seasons: str, reg_epochs: int, cls_epochs: int, no_garbage: bool,
          adj_suffix: str | None, min_date: str | None):
    """Train MLPRegressor + MLPClassifier on historical features."""
    from .dataset import load_multi_season_features
    from .trainer import (
        fit_scaler,
        impute_column_means,
        save_checkpoint,
        train_classifier,
        train_regressor,
    )

    season_list = _parse_seasons(seasons)
    variant = " (no-garbage)" if no_garbage else ""
    if adj_suffix:
        variant += f" ({adj_suffix})"
    click.echo(f"Loading features{variant} for seasons: {season_list}")
    if min_date:
        click.echo(f"  Training date filter: games on or after MM-DD={min_date}")

    df = load_multi_season_features(season_list, no_garbage=no_garbage,
                                    adj_suffix=adj_suffix,
                                    min_month_day=min_date)

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
@click.option("--min-date", default="12-01", type=str,
              help="Earliest MM-DD within each season to include (default: 12-01)")
def tune(seasons: str, trials: int, min_date: str | None):
    """Optuna hyperparameter search for both models."""
    from .dataset import load_multi_season_features
    from .trainer import fit_scaler, impute_column_means
    from .tuner import tune_classifier, tune_regressor

    season_list = _parse_seasons(seasons)
    click.echo(f"Loading features for seasons: {season_list}")
    if min_date:
        click.echo(f"  Tuning date filter: games on or after MM-DD={min_date}")

    df = load_multi_season_features(season_list, min_month_day=min_date)
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
        game_date = date.today().isoformat()

    click.echo(f"Building features for {game_date}...")
    df = build_features(
        season,
        game_date=game_date,
        extra_features=config.EXTRA_FEATURES,
        adjust_ff=config.ADJUST_FF,
        adjust_alpha=config.ADJUST_ALPHA,
        adjust_prior_weight=config.ADJUST_PRIOR,
    )
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
    df = build_features(
        season,
        extra_features=config.EXTRA_FEATURES,
        adjust_ff=config.ADJUST_FF,
        adjust_alpha=config.ADJUST_ALPHA,
        adjust_prior_weight=config.ADJUST_PRIOR,
    )
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
    df = build_features(
        season,
        extra_features=config.EXTRA_FEATURES,
        adjust_ff=config.ADJUST_FF,
        adjust_alpha=config.ADJUST_ALPHA,
        adjust_prior_weight=config.ADJUST_PRIOR,
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


# ── 7. daily-run ──────────────────────────────────────────────────


@cli.command("daily-run")
@click.option("--season", required=True, type=int, help="Season year (e.g. 2026)")
@click.option("--date", "game_date", default=None, help="Date override (YYYY-MM-DD)")
def daily_run(season: int, game_date: str | None):
    """Full daily pipeline: build features → predict → CSV → JSON → final scores."""
    from .infer import predict, save_predictions

    if game_date is None:
        game_date = date.today().isoformat()

    click.echo(f"=== Daily run for {game_date} (season {season}) ===")

    # 1. Build features
    click.echo("Building features...")
    df = build_features(
        season,
        game_date=game_date,
        extra_features=config.EXTRA_FEATURES,
        adjust_ff=config.ADJUST_FF,
        adjust_alpha=config.ADJUST_ALPHA,
        adjust_prior_weight=config.ADJUST_PRIOR,
    )
    if df.empty:
        click.echo(f"No games found for {game_date}.")
        return

    click.echo(f"  Games: {len(df)}")

    # 2. Predict with edge calculations
    lines = load_lines(season)
    preds = predict(df, lines_df=lines)

    # 3. Save predictions
    json_path, csv_path = save_predictions(preds, game_date=game_date)
    click.echo(f"  JSON: {json_path}")
    click.echo(f"  CSV:  {csv_path}")

    # 4. Publish pipeline: csv_to_json → rankings → final scores
    script_dir = config.PROJECT_ROOT / "scripts"

    csv_to_json = script_dir / "csv_to_json.py"
    if csv_to_json.exists():
        click.echo("Converting CSV to site JSON...")
        csv_dir = config.PREDICTIONS_DIR / "csv"
        latest_csv = sorted(csv_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
        csv_arg = str(latest_csv[-1]) if latest_csv else str(csv_path)
        subprocess.run(
            [sys.executable, str(csv_to_json), csv_arg, game_date],
            check=True,
            cwd=config.PROJECT_ROOT,
        )

    rankings_script = script_dir / "build_rankings_json.py"
    if rankings_script.exists():
        click.echo("Building rankings...")
        subprocess.run(
            [sys.executable, str(rankings_script)],
            check=True,
            cwd=config.PROJECT_ROOT,
        )

    s3_finals = script_dir / "s3_finals_to_json.py"
    if s3_finals.exists():
        click.echo("Fetching final scores...")
        subprocess.run(
            [sys.executable, str(s3_finals)],
            check=True,
            cwd=config.PROJECT_ROOT,
        )

    click.echo("Publish pipeline complete.")

    # Print summary
    if "pick_side" in preds.columns and "pick_prob_edge" in preds.columns:
        preds = preds.copy()
        preds["_abs_edge"] = preds["pick_prob_edge"].abs()
        preds = preds.sort_values("_abs_edge", ascending=False).drop(columns=["_abs_edge"])
        click.echo(f"\n{'MATCHUP':>44} | {'PICK':>6} | {'EDGE':>7} | {'MODEL':>7} | {'BOOK':>7}")
        click.echo("-" * 85)
        for _, row in preds.iterrows():
            away = str(row.get("awayTeam", ""))[:16]
            home = str(row.get("homeTeam", ""))[:16]
            matchup = f"{away:>16} @ {home:<16}"
            pick = str(row.get("pick_side", ""))
            edge_pct = row.get("pick_prob_edge", 0)
            model = row.get("model_spread", 0)
            book = row.get("book_spread", 0)
            if pd.notna(edge_pct) and pd.notna(model) and pd.notna(book):
                click.echo(
                    f"  {matchup} | {pick:>6} | {edge_pct:+.1%} | {model:+.1f} | {book:+.1f}"
                )


# ── 8. build-rankings ─────────────────────────────────────────────


@cli.command("build-rankings")
@click.option("--season", default=2026, type=int, help="Season year (e.g. 2026)")
def build_rankings(season: int):
    """Build power rankings JSON from S3 efficiency ratings."""
    script = config.PROJECT_ROOT / "scripts" / "build_rankings_json.py"
    subprocess.run(
        [sys.executable, str(script)],
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
        else date.today() - timedelta(days=1)
    )

    click.echo(f"=== Backfill season {season}: {start} → {end} ===")

    lines = load_lines(season)
    script_dir = config.PROJECT_ROOT / "scripts"
    csv_to_json = script_dir / "csv_to_json.py"

    processed = 0
    current = start
    while current <= end:
        game_date = current.isoformat()
        current += timedelta(days=1)

        # Check if already exists
        if skip_existing:
            existing_json = (
                config.PROJECT_ROOT / "site" / "public" / "data"
                / f"predictions_{game_date}.json"
            )
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
        )
        if df.empty:
            continue

        # Predict
        preds = predict(df, lines_df=lines)

        # Save
        json_path, csv_path = save_predictions(preds, game_date=game_date)

        # Convert CSV to site JSON
        if csv_to_json.exists():
            csv_dir = config.PREDICTIONS_DIR / "csv"
            latest_csv = sorted(csv_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
            csv_arg = str(latest_csv[-1]) if latest_csv else str(csv_path)
            subprocess.run(
                [sys.executable, str(csv_to_json), csv_arg, game_date],
                check=True,
                cwd=config.PROJECT_ROOT,
            )

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

    today_str = date.today().isoformat()
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
    """Run a subprocess, abort on failure."""
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        click.echo(f"FAILED: {label} (exit {result.returncode})", err=True)
        sys.exit(result.returncode)


@cli.command("daily-update")
@click.option("--season", required=True, type=int, help="Season year (e.g. 2026)")
@click.option("--date", "game_date", default=None, help="Date override (YYYY-MM-DD)")
@click.option("--skip-etl", is_flag=True, help="Skip ETL ingest + transforms (steps 1-3)")
@click.option("--skip-predict", is_flag=True, help="Skip predictions + publish (steps 4-5)")
@click.option("--skip-deploy", is_flag=True, help="Skip git commit/push (step 6)")
def daily_update(season: int, game_date: str | None, skip_etl: bool,
                 skip_predict: bool, skip_deploy: bool):
    """Full end-to-end daily pipeline: ETL → silver → gold → predict → publish → deploy."""
    from .infer import predict, save_predictions

    if game_date is None:
        game_date = date.today().isoformat()

    etl_root = _get_etl_root()
    click.echo(f"=== Daily update for {game_date} (season {season}) ===")

    # ── Steps 1-3: ETL ingest + silver + gold ──────────────────────
    if not skip_etl:
        # Step 1: Minimal ETL ingest
        click.echo("\n[1/6] ETL ingest (games, games_teams, lines, ratings_adjusted)...")
        _run(
            ["poetry", "run", "python", "-m", "cbbd_etl", "incremental",
             "--only-endpoints", "games,games_teams,lines,ratings_adjusted"],
            cwd=etl_root,
            label="ETL incremental ingest",
        )

        # Step 2: Silver — fct_games/fct_lines/fct_ratings_adjusted built during ingest.
        # PBP pipeline: fct_plays → enriched → flat (both variants).
        click.echo("\n[2/6] Silver transforms (PBP enriched + flat tables)...")
        _run(
            ["poetry", "run", "python", "scripts/build_pbp_plays_enriched.py",
             "--season", str(season), "--purge"],
            cwd=etl_root,
            label="build_pbp_plays_enriched",
        )
        _run(
            ["poetry", "run", "python", "scripts/build_pbp_game_teams_flat.py",
             "--season", str(season), "--purge"],
            cwd=etl_root,
            label="build_pbp_game_teams_flat",
        )
        _run(
            ["poetry", "run", "python", "scripts/build_pbp_game_teams_flat.py",
             "--season", str(season), "--exclude-garbage-time",
             "--output-table", "fct_pbp_game_teams_flat_garbage_removed", "--purge"],
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

    # ── Steps 4-5: Predict + publish ──────────────────────────────
    if not skip_predict:
        # Step 4: Predict today's games
        click.echo(f"\n[4/6] Predictions for {game_date}...")
        df = build_features(
            season,
            game_date=game_date,
            extra_features=config.EXTRA_FEATURES,
            adjust_ff=config.ADJUST_FF,
            adjust_alpha=config.ADJUST_ALPHA,
            adjust_prior_weight=config.ADJUST_PRIOR,
        )
        if df.empty:
            click.echo(f"  No games found for {game_date}. Skipping predictions.")
        else:
            click.echo(f"  Games: {len(df)}")
            lines = load_lines(season)
            preds = predict(df, lines_df=lines)
            json_path, csv_path = save_predictions(preds, game_date=game_date)
            click.echo(f"  JSON: {json_path}")
            click.echo(f"  CSV:  {csv_path}")

        # Step 5: Publish pipeline — csv_to_json → rankings → final scores
        click.echo(f"\n[5/6] Publish pipeline...")
        script_dir = config.PROJECT_ROOT / "scripts"

        csv_to_json = script_dir / "csv_to_json.py"
        if csv_to_json.exists():
            csv_dir = config.PREDICTIONS_DIR / "csv"
            latest_csv = sorted(csv_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
            csv_arg = str(latest_csv[-1]) if latest_csv else ""
            if csv_arg:
                _run([sys.executable, str(csv_to_json), csv_arg, game_date],
                     cwd=config.PROJECT_ROOT, label="csv_to_json")

        for script_name in ["build_rankings_json.py", "s3_finals_to_json.py"]:
            script = script_dir / script_name
            if script.exists():
                _run([sys.executable, str(script)], cwd=config.PROJECT_ROOT,
                     label=script_name)
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
