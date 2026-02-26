"""Part A: Feature Importance, Error Analysis & Book Intelligence.

Produces reports/feature_importance_2025.md with:
  A1. Permutation importance for all 37 features
  A2. Error analysis (model vs book, pattern analysis)
  A3. Book intelligence (what does the book see that we don't?)
"""
from __future__ import annotations

import pickle
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.architecture import MLPRegressor
from src.dataset import load_season_features
from src.features import (
    get_feature_matrix,
    get_targets,
    load_efficiency_ratings,
    load_games,
)

HOLDOUT_SEASON = 2025
SEED = 42
N_SHUFFLES = 10

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = PROJECT_ROOT / "reports" / "feature_importance_2025.md"


# ── Shared data loading ──────────────────────────────────────────


def load_model_and_scaler():
    """Load the no-garbage regressor and scaler."""
    ckpt_path = config.CHECKPOINTS_DIR / "no_garbage" / "regressor.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hparams", {})
    feature_order = ckpt.get("feature_order", config.FEATURE_ORDER)
    model = MLPRegressor(
        input_dim=len(feature_order),
        hidden1=hp.get("hidden1", 256),
        hidden2=hp.get("hidden2", 128),
        dropout=hp.get("dropout", 0.3),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    scaler_path = config.ARTIFACTS_DIR / "no_garbage" / "scaler.pkl"
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler, feature_order


@torch.no_grad()
def predict_spread(model, X_scaled):
    """Get spread predictions from model."""
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    mu, _ = model(X_t)
    return mu.numpy()


def compute_mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


# ── A1: Permutation Importance ───────────────────────────────────


def permutation_importance(model, scaler, X_raw, y_true, feature_names):
    """Compute permutation importance for each feature.

    For each feature, shuffles the column N_SHUFFLES times and measures
    the increase in MAE compared to baseline.
    """
    X_scaled = scaler.transform(X_raw)
    baseline_mae = compute_mae(y_true, predict_spread(model, X_scaled))
    print(f"  Baseline MAE on holdout: {baseline_mae:.4f}")

    results = []
    rng = np.random.RandomState(SEED)

    for i, feat_name in enumerate(feature_names):
        mae_increases = []
        for _ in range(N_SHUFFLES):
            X_perm = X_raw.copy()
            X_perm[:, i] = rng.permutation(X_perm[:, i])
            X_perm_scaled = scaler.transform(X_perm)
            perm_mae = compute_mae(y_true, predict_spread(model, X_perm_scaled))
            mae_increases.append(perm_mae - baseline_mae)

        mean_inc = float(np.mean(mae_increases))
        std_inc = float(np.std(mae_increases))
        results.append({
            "feature": feat_name,
            "mean_mae_increase": mean_inc,
            "std_mae_increase": std_inc,
            "index": i,
        })

    return pd.DataFrame(results).sort_values("mean_mae_increase", ascending=False), baseline_mae


# ── A2: Error Analysis ──────────────────────────────────────────


def error_analysis(preds_df, eff_ratings):
    """Analyze patterns in model errors vs book errors."""
    df = preds_df.dropna(subset=["book_spread", "actual_margin"]).copy()

    df["model_error"] = df["predicted_spread"] - df["actual_margin"]
    df["book_error"] = (-df["book_spread"]) - df["actual_margin"]
    df["abs_model_error"] = df["model_error"].abs()
    df["abs_book_error"] = df["book_error"].abs()
    df["model_advantage"] = df["abs_book_error"] - df["abs_model_error"]

    # Parse month
    dates = pd.to_datetime(df["startDate"], errors="coerce", utc=True)
    df["month"] = dates.dt.tz_localize(None).dt.to_period("M").astype(str)

    # Team quality tiers from barthag
    team_barthag = {}
    if not eff_ratings.empty:
        latest = eff_ratings.sort_values("rating_date").groupby("teamId").last()
        if "barthag" in latest.columns:
            team_barthag = latest["barthag"].to_dict()

    def get_tier(team_id):
        b = team_barthag.get(team_id)
        if b is None:
            return "unknown"
        # Rank teams by barthag
        ranked = sorted(team_barthag.values(), reverse=True)
        rank = sorted(team_barthag.values(), reverse=True).index(b) + 1 if b in ranked else 999
        # Approximate: count how many are >= this barthag
        rank = sum(1 for v in team_barthag.values() if v >= b)
        if rank <= 50:
            return "top_50"
        elif rank <= 150:
            return "50_150"
        elif rank <= 250:
            return "150_250"
        else:
            return "250_plus"

    df["home_tier"] = df["homeTeamId"].map(get_tier)
    df["away_tier"] = df["awayTeamId"].map(get_tier)

    # Games where model is 5+ points worse than book
    model_much_worse = df[df["model_advantage"] < -5].copy()
    # Games where model beats book by 5+ points
    model_much_better = df[df["model_advantage"] > 5].copy()

    # Per-team MAE
    home_errors = df[["homeTeamId", "abs_model_error"]].rename(
        columns={"homeTeamId": "teamId"})
    away_errors = df[["awayTeamId", "abs_model_error"]].rename(
        columns={"awayTeamId": "teamId"})
    all_team_errors = pd.concat([home_errors, away_errors], ignore_index=True)
    team_mae = all_team_errors.groupby("teamId").agg(
        mae=("abs_model_error", "mean"),
        n=("abs_model_error", "count"),
    ).sort_values("mae", ascending=False)

    return {
        "df": df,
        "model_much_worse": model_much_worse,
        "model_much_better": model_much_better,
        "team_mae": team_mae,
    }


# ── A3: Book Intelligence ───────────────────────────────────────


def book_intelligence(preds_df, games, feature_names, scaler):
    """Analyze what the book sees that the model doesn't."""
    df = preds_df.dropna(subset=["book_spread"]).copy()
    df["disagreement"] = (-df["book_spread"]) - df["predicted_spread"]

    # Load holdout features to get the 37-feature matrix
    holdout = load_season_features(HOLDOUT_SEASON, no_garbage=True)
    holdout = holdout.dropna(subset=["homeScore", "awayScore"])

    # Match rows by gameId
    merged = df.merge(holdout[["gameId"] + feature_names], on="gameId", how="inner")
    X_feats = merged[feature_names].values.astype(np.float64)
    X_feats = np.nan_to_num(X_feats, nan=0.0)
    y_disagree = merged["disagreement"].values

    # Linear regression of disagreement on 37 features
    reg = LinearRegression()
    reg.fit(X_feats, y_disagree)
    r2_base = reg.score(X_feats, y_disagree)
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": reg.coef_,
        "abs_coef": np.abs(reg.coef_),
    }).sort_values("abs_coef", ascending=False)

    # Compute candidate factors not in current features
    games_season = games.copy()
    dates = pd.to_datetime(games_season["startDate"], errors="coerce")

    # Rest days
    rest_rows = []
    for _, g in games_season.iterrows():
        rest_rows.append({"gameId": int(g["gameId"]), "teamId": int(g["homeTeamId"]),
                          "date": dates[g.name], "side": "home"})
        rest_rows.append({"gameId": int(g["gameId"]), "teamId": int(g["awayTeamId"]),
                          "date": dates[g.name], "side": "away"})
    rest_df = pd.DataFrame(rest_rows).sort_values(["teamId", "date"]).reset_index(drop=True)
    rest_df["prev_date"] = rest_df.groupby("teamId")["date"].shift(1)
    rest_df["rest_days"] = (rest_df["date"] - rest_df["prev_date"]).dt.total_seconds() / 86400
    rest_df["rest_days"] = rest_df["rest_days"].fillna(5.0).clip(upper=30.0)

    home_rest = rest_df[rest_df["side"] == "home"][["gameId", "rest_days"]].rename(
        columns={"rest_days": "home_rest_days"})
    away_rest = rest_df[rest_df["side"] == "away"][["gameId", "rest_days"]].rename(
        columns={"rest_days": "away_rest_days"})
    merged2 = merged.merge(home_rest, on="gameId", how="left")
    merged2 = merged2.merge(away_rest, on="gameId", how="left")
    merged2["rest_advantage"] = merged2["home_rest_days"].fillna(5) - merged2["away_rest_days"].fillna(5)

    # Season progress: fraction of season (0=Nov, 1=Apr)
    game_dates = pd.to_datetime(merged2["startDate"], errors="coerce", utc=True).dt.tz_localize(None)
    season_start = pd.Timestamp(f"{HOLDOUT_SEASON - 1}-11-01")
    merged2["season_progress"] = (game_dates - season_start).dt.total_seconds() / (160 * 86400)
    merged2["season_progress"] = merged2["season_progress"].clip(0, 1)

    # Correlations of candidates with disagreement
    candidates = {
        "home_rest_days": merged2["home_rest_days"],
        "away_rest_days": merged2["away_rest_days"],
        "rest_advantage": merged2["rest_advantage"],
        "season_progress": merged2["season_progress"],
    }
    correlations = {}
    for name, series in candidates.items():
        valid = series.notna() & merged2["disagreement"].notna()
        if valid.sum() > 10:
            correlations[name] = float(series[valid].corr(merged2["disagreement"][valid]))
        else:
            correlations[name] = None

    # Regression with candidates added
    candidate_cols = ["home_rest_days", "away_rest_days", "rest_advantage", "season_progress"]
    X_expanded = np.column_stack([
        X_feats[:len(merged2)],
        merged2[candidate_cols].fillna(0).values,
    ])
    reg2 = LinearRegression()
    reg2.fit(X_expanded, merged2["disagreement"].values[:len(X_expanded)])
    r2_expanded = reg2.score(X_expanded, merged2["disagreement"].values[:len(X_expanded)])

    return {
        "r2_base": r2_base,
        "r2_expanded": r2_expanded,
        "coef_df": coef_df,
        "correlations": correlations,
        "n_games": len(merged),
    }


# ── Report generation ────────────────────────────────────────────


def generate_report(perm_results, baseline_mae, error_results, book_results):
    """Generate the markdown report."""
    lines = []
    lines.append("# Feature Importance & Error Analysis — Season 2025\n")

    # ── A1: Permutation Importance ──
    lines.append("## A1: Permutation Importance\n")
    lines.append(f"Baseline MAE on 2025 holdout: **{baseline_mae:.4f}**\n")
    lines.append("Each feature shuffled {0} times (seed={1}). "
                 "Ranked by mean MAE increase.\n".format(N_SHUFFLES, SEED))

    perm_df = perm_results

    # Group 1: Efficiency (indices 0-10)
    lines.append("### Group 1: Efficiency Metrics (features 0-10)\n")
    lines.append("| Rank | Feature | Mean MAE Increase | Std |")
    lines.append("|------|---------|-------------------|-----|")
    g1 = perm_df[perm_df["index"] <= 10].sort_values("mean_mae_increase", ascending=False)
    for rank, (_, row) in enumerate(g1.iterrows(), 1):
        flag = " *" if row["mean_mae_increase"] <= 0 else ""
        lines.append(f"| {rank} | {row['feature']} | {row['mean_mae_increase']:+.4f} | "
                     f"{row['std_mae_increase']:.4f} |{flag}")

    # Group 2: Rolling four-factors (indices 11-36)
    lines.append("\n### Group 2: Rolling Four-Factors (features 11-36)\n")
    lines.append("| Rank | Feature | Mean MAE Increase | Std |")
    lines.append("|------|---------|-------------------|-----|")
    g2 = perm_df[perm_df["index"] > 10].sort_values("mean_mae_increase", ascending=False)
    for rank, (_, row) in enumerate(g2.iterrows(), 1):
        flag = " *" if row["mean_mae_increase"] <= 0 else ""
        lines.append(f"| {rank} | {row['feature']} | {row['mean_mae_increase']:+.4f} | "
                     f"{row['std_mae_increase']:.4f} |{flag}")

    # Overall top 10
    lines.append("\n### Top 10 Overall\n")
    lines.append("| Rank | Feature | Mean MAE Increase |")
    lines.append("|------|---------|-------------------|")
    top10 = perm_df.head(10)
    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        lines.append(f"| {rank} | {row['feature']} | {row['mean_mae_increase']:+.4f} |")

    # Zero/negative importance
    zero_neg = perm_df[perm_df["mean_mae_increase"] <= 0]
    if len(zero_neg) > 0:
        lines.append(f"\n**Zero/negative importance features ({len(zero_neg)}):** "
                     + ", ".join(zero_neg["feature"].tolist()))

    # ── A2: Error Analysis ──
    lines.append("\n\n## A2: Error Analysis\n")

    df = error_results["df"]
    worse = error_results["model_much_worse"]
    better = error_results["model_much_better"]

    lines.append(f"Total games with book spread: {len(df)}\n")
    lines.append(f"- Model 5+ pts worse than book: **{len(worse)}** games")
    lines.append(f"- Model 5+ pts better than book: **{len(better)}** games\n")

    # Monthly pattern
    lines.append("### Monthly Pattern (Model 5+ pts worse)\n")
    if len(worse) > 0:
        lines.append("| Month | Count | % of Model-Worse Games |")
        lines.append("|-------|-------|------------------------|")
        month_counts = worse["month"].value_counts().sort_index()
        for month, count in month_counts.items():
            pct = 100 * count / len(worse)
            lines.append(f"| {month} | {count} | {pct:.1f}% |")

    # Tier pattern
    lines.append("\n### Team Quality Pattern (Model 5+ pts worse)\n")
    if len(worse) > 0:
        lines.append("| Home Tier | Count |")
        lines.append("|-----------|-------|")
        tier_counts = worse["home_tier"].value_counts()
        for tier, count in tier_counts.items():
            lines.append(f"| {tier} | {count} |")

    # Per-team MAE worst 20
    lines.append("\n### Worst 20 Teams by Model MAE\n")
    lines.append("| TeamId | MAE | Games |")
    lines.append("|--------|-----|-------|")
    team_mae = error_results["team_mae"]
    # Only teams with enough games
    team_mae_filtered = team_mae[team_mae["n"] >= 5].head(20)
    for tid, row in team_mae_filtered.iterrows():
        lines.append(f"| {int(tid)} | {row['mae']:.2f} | {int(row['n'])} |")

    # Overall error stats
    lines.append(f"\n### Error Distribution\n")
    lines.append(f"| Metric | Model | Book |")
    lines.append(f"|--------|-------|------|")
    lines.append(f"| MAE | {df['abs_model_error'].mean():.2f} | {df['abs_book_error'].mean():.2f} |")
    lines.append(f"| Median AE | {df['abs_model_error'].median():.2f} | {df['abs_book_error'].median():.2f} |")
    lines.append(f"| Std Error | {df['model_error'].std():.2f} | {df['book_error'].std():.2f} |")

    # ── A3: Book Intelligence ──
    lines.append("\n\n## A3: Book Intelligence\n")
    lines.append(f"Disagreement = book_spread_home - model_spread_home\n")
    lines.append(f"Games analyzed: {book_results['n_games']}\n")

    lines.append(f"### Linear Regression of Disagreement on 37 Features\n")
    lines.append(f"R-squared: **{book_results['r2_base']:.4f}**\n")
    lines.append("Top 10 coefficients (by absolute value):\n")
    lines.append("| Feature | Coefficient |")
    lines.append("|---------|-------------|")
    for _, row in book_results["coef_df"].head(10).iterrows():
        lines.append(f"| {row['feature']} | {row['coefficient']:+.4f} |")

    lines.append(f"\n### Candidate Features Not in Current Model\n")
    lines.append("Correlation with book-model disagreement:\n")
    lines.append("| Candidate | Correlation |")
    lines.append("|-----------|-------------|")
    for name, corr in book_results["correlations"].items():
        corr_str = f"{corr:+.4f}" if corr is not None else "N/A"
        lines.append(f"| {name} | {corr_str} |")

    lines.append(f"\n### R-squared with Candidates Added\n")
    lines.append(f"- Base (37 features): R2 = {book_results['r2_base']:.4f}")
    lines.append(f"- Expanded (37 + 4 candidates): R2 = {book_results['r2_expanded']:.4f}")
    r2_lift = book_results['r2_expanded'] - book_results['r2_base']
    lines.append(f"- Lift: {r2_lift:+.4f}")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("FEATURE ANALYSIS: Importance, Errors & Book Intelligence")
    print("=" * 70)

    # Load shared data
    print("\nLoading model, scaler, and holdout data...")
    model, scaler, feature_order = load_model_and_scaler()
    feature_names = list(feature_order)

    holdout = load_season_features(HOLDOUT_SEASON, no_garbage=True)
    holdout = holdout.dropna(subset=["homeScore", "awayScore"])
    print(f"  Holdout: {len(holdout)} games")

    X_raw = get_feature_matrix(holdout).values.astype(np.float32)
    X_raw = np.nan_to_num(X_raw, nan=0.0)
    targets = get_targets(holdout)
    y_spread = targets["spread_home"].values.astype(np.float32)

    # Load predictions for error/book analysis
    preds_path = config.PREDICTIONS_DIR / "backtest_2025_sos085.csv"
    preds_df = pd.read_csv(preds_path)
    print(f"  Predictions: {len(preds_df)} rows")
    print(f"  Games with book spread: {preds_df['book_spread'].notna().sum()}")

    # Load efficiency ratings for team quality tiers
    eff_ratings = load_efficiency_ratings(HOLDOUT_SEASON, no_garbage=True)

    # Load games for book intelligence
    games = load_games(HOLDOUT_SEASON)

    # ── A1: Permutation Importance ──
    print("\n--- A1: Permutation Importance ---")
    perm_results, baseline_mae = permutation_importance(
        model, scaler, X_raw, y_spread, feature_names,
    )
    print("\nTop 10 features by importance:")
    for _, row in perm_results.head(10).iterrows():
        print(f"  {row['feature']:>30s}: {row['mean_mae_increase']:+.4f}")

    zero_neg = perm_results[perm_results["mean_mae_increase"] <= 0]
    print(f"\nZero/negative importance: {len(zero_neg)} features")

    # ── A2: Error Analysis ──
    print("\n--- A2: Error Analysis ---")
    error_results = error_analysis(preds_df, eff_ratings)
    df_err = error_results["df"]
    print(f"  Model MAE: {df_err['abs_model_error'].mean():.2f}")
    print(f"  Book MAE: {df_err['abs_book_error'].mean():.2f}")
    print(f"  Model 5+ pts worse: {len(error_results['model_much_worse'])} games")
    print(f"  Model 5+ pts better: {len(error_results['model_much_better'])} games")

    # ── A3: Book Intelligence ──
    print("\n--- A3: Book Intelligence ---")
    book_results = book_intelligence(preds_df, games, feature_names, scaler)
    print(f"  R2 (37 features): {book_results['r2_base']:.4f}")
    print(f"  R2 (37 + candidates): {book_results['r2_expanded']:.4f}")
    print(f"  Correlations with disagreement:")
    for name, corr in book_results["correlations"].items():
        print(f"    {name}: {corr:+.4f}" if corr is not None else f"    {name}: N/A")

    # ── Generate report ──
    print("\nGenerating report...")
    report = generate_report(perm_results, baseline_mae, error_results, book_results)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
