"""Part B: Feature Expansion Evaluation.

Step 1: Build expanded features (v2) for all seasons with ALL extra features.
Step 2: Evaluate each feature group individually against baseline MAE.
Step 3: Combine positive-lift features and run full evaluation.
Step 4: If improved, save expanded artifacts and report.
"""
from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# Force unbuffered output so we can monitor progress
print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.architecture import MLPRegressor, MLPClassifier, gaussian_nll_loss
from src.dataset import load_season_features
from src.features import (
    EXTRA_FEATURE_NAMES,
    build_features,
    get_feature_matrix,
    get_targets,
    load_research_lines,
)
from src.trainer import train_regressor, train_classifier, save_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = PROJECT_ROOT / "reports" / "feature_expansion_2025.md"

TRAIN_SEASONS = list(range(2015, 2025))
HOLDOUT_SEASON = 2025
BASELINE_MAE = 9.62
ALL_EXTRA_GROUPS = list(EXTRA_FEATURE_NAMES.keys())


# ── Step 1: Build v2 feature parquets ────────────────────────────


def build_v2_features():
    """Build features for all seasons with ALL extra feature groups."""
    import time
    all_seasons = TRAIN_SEASONS + [HOLDOUT_SEASON]
    for season in all_seasons:
        out_path = config.FEATURES_DIR / f"season_{season}_no_garbage_v2_features.parquet"
        if out_path.exists():
            existing = pd.read_parquet(out_path)
            # Check if all extra feature columns exist
            all_extra_cols = []
            for group in ALL_EXTRA_GROUPS:
                all_extra_cols.extend(EXTRA_FEATURE_NAMES[group])
            missing_cols = [c for c in all_extra_cols if c not in existing.columns]
            if not missing_cols:
                print(f"  Season {season}: v2 features exist ({len(existing)} rows), skipping")
                continue
            print(f"  Season {season}: v2 features missing {len(missing_cols)} cols, rebuilding")

        t0 = time.time()
        print(f"  Building v2 features for season {season}...")
        df = build_features(
            season,
            no_garbage=True,
            extra_features=ALL_EXTRA_GROUPS,
        )
        elapsed = time.time() - t0
        if df.empty:
            print(f"    WARNING: No games for season {season} ({elapsed:.1f}s)")
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"    Saved {len(df)} rows -> {out_path.name} ({elapsed:.1f}s)")


def load_v2_features(season: int) -> pd.DataFrame:
    """Load v2 feature parquet for a season."""
    path = config.FEATURES_DIR / f"season_{season}_no_garbage_v2_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"v2 features not found: {path}")
    return pd.read_parquet(path)


def load_multi_v2(seasons: list[int]) -> pd.DataFrame:
    """Load and concatenate v2 features for multiple seasons."""
    dfs = []
    for s in seasons:
        try:
            dfs.append(load_v2_features(s))
        except FileNotFoundError:
            print(f"  Warning: No v2 features for season {s}, skipping.")
    if not dfs:
        raise FileNotFoundError(f"No v2 feature files found")
    return pd.concat(dfs, ignore_index=True)


# ── Evaluation helpers ───────────────────────────────────────────


@torch.no_grad()
def predict_all(reg_model, cls_model, X_test, scaler):
    """Generate spread predictions and home win probabilities."""
    reg_model.eval()
    cls_model.eval()
    X_scaled = scaler.transform(X_test)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    mu, log_sigma = reg_model(X_t)
    sigma = torch.nn.functional.softplus(log_sigma) + 1e-3
    sigma = sigma.clamp(min=0.5, max=30.0)
    logits = cls_model(X_t)
    home_win_prob = torch.sigmoid(logits).numpy()
    return mu.numpy(), sigma.numpy(), home_win_prob


def compute_roi(preds_df, threshold, sigma_filter=None):
    """Compute ATS ROI at a given threshold."""
    with_book = preds_df.dropna(subset=["book_spread"]).copy()
    if sigma_filter is not None:
        with_book = with_book[with_book["spread_sigma"] < sigma_filter]
    if len(with_book) == 0:
        return None, 0, 0, 0

    bets = with_book[with_book["spread_diff"].abs() > threshold]
    if len(bets) == 0:
        return None, 0, 0, 0

    wins = losses = 0
    for _, row in bets.iterrows():
        cover_margin = row["actual_margin"] + row["book_spread"]
        if row["spread_diff"] < 0:  # bet HOME
            wins += 1 if cover_margin > 0 else 0
            losses += 1 if cover_margin < 0 else 0
        else:  # bet AWAY
            wins += 1 if cover_margin < 0 else 0
            losses += 1 if cover_margin > 0 else 0

    n_bets = wins + losses
    if n_bets == 0:
        return None, 0, 0, 0
    roi = (wins * (100 / 110) - losses) / n_bets * 100
    return roi, n_bets, wins, losses


def train_and_evaluate(feature_cols, train_df, holdout_df, label=""):
    """Train regressor+classifier and evaluate on holdout.

    Returns dict with mae, book_mae, model on book-spread games, etc.
    """
    X_train = train_df[feature_cols].values.astype(np.float32)
    X_train = np.nan_to_num(X_train, nan=0.0)
    targets_train = get_targets(train_df)
    y_spread_train = targets_train["spread_home"].values.astype(np.float32)
    y_win_train = targets_train["home_win"].values.astype(np.float32)

    X_test = holdout_df[feature_cols].values.astype(np.float32)
    X_test = np.nan_to_num(X_test, nan=0.0)
    targets_test = get_targets(holdout_df)
    y_spread_test = targets_test["spread_home"].values.astype(np.float32)

    # Fit scaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)

    # Train regressor
    reg_model = train_regressor(X_train_scaled, y_spread_train, hparams={"epochs": 100})

    # Train classifier
    cls_model = train_classifier(X_train_scaled, y_win_train, hparams={"epochs": 100})

    # Predict
    mu, sigma, prob = predict_all(reg_model, cls_model, X_test, scaler)

    # Overall MAE
    mae = float(np.mean(np.abs(mu - y_spread_test)))

    # Build predictions DataFrame for ROI analysis
    preds = holdout_df[["gameId", "homeTeamId", "awayTeamId", "homeScore",
                         "awayScore", "startDate"]].copy()
    preds["predicted_spread"] = mu
    preds["spread_sigma"] = sigma
    preds["home_win_prob"] = prob
    preds["actual_margin"] = preds["homeScore"].astype(float) - preds["awayScore"].astype(float)

    # Attach book spreads
    try:
        lines = load_research_lines(HOLDOUT_SEASON)
        if lines is not None and not lines.empty:
            lines_dedup = lines.sort_values("provider").drop_duplicates(
                subset=["gameId"], keep="first")
            preds = preds.merge(
                lines_dedup[["gameId", "spread"]].rename(columns={"spread": "book_spread"}),
                on="gameId", how="left")
            preds["model_spread"] = -preds["predicted_spread"]
            preds["spread_diff"] = preds["model_spread"] - preds["book_spread"]
    except Exception:
        preds["book_spread"] = np.nan

    # Book-spread-game MAE
    with_book = preds.dropna(subset=["book_spread"])
    book_mae = None
    model_mae_book = None
    if len(with_book) > 0:
        book_mae = float(np.abs(-with_book["book_spread"] - with_book["actual_margin"]).mean())
        model_mae_book = float(np.abs(with_book["predicted_spread"] - with_book["actual_margin"]).mean())

    # Monthly MAE
    dates = pd.to_datetime(preds["startDate"], errors="coerce", utc=True)
    preds["month"] = dates.dt.tz_localize(None).dt.to_period("M")
    monthly = {}
    for month, group in preds.groupby("month"):
        m_mae = float(np.abs(group["predicted_spread"] - group["actual_margin"]).mean())
        monthly[str(month)] = (m_mae, len(group))

    return {
        "mae": mae,
        "model_mae_book": model_mae_book,
        "book_mae": book_mae,
        "n_book_games": len(with_book),
        "preds": preds,
        "monthly": monthly,
        "reg_model": reg_model,
        "cls_model": cls_model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "label": label,
    }


# ── Main ─────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("FEATURE EXPANSION EVALUATION")
    print("=" * 70)

    # Step 1: Build v2 features
    print("\n--- Step 1: Building v2 feature parquets ---")
    build_v2_features()

    # Load data
    print("\nLoading v2 training data (2015-2024)...")
    train_df = load_multi_v2(TRAIN_SEASONS)
    train_df = train_df.dropna(subset=["homeScore", "awayScore"])
    print(f"  Training samples: {len(train_df)}")

    print("Loading v2 holdout data (2025)...")
    holdout_df = load_v2_features(HOLDOUT_SEASON)
    holdout_df = holdout_df.dropna(subset=["homeScore", "awayScore"])
    print(f"  Holdout samples: {len(holdout_df)}")

    base_features = list(config.FEATURE_ORDER)

    # Step 2: Evaluate each group individually
    print("\n--- Step 2: Individual Feature Group Evaluation ---")
    print(f"Baseline MAE (book-spread games): {BASELINE_MAE}\n")

    group_results = {}
    for group_name in ALL_EXTRA_GROUPS:
        extra_cols = EXTRA_FEATURE_NAMES[group_name]
        feature_cols = base_features + extra_cols

        # Check columns exist
        missing = [c for c in feature_cols if c not in holdout_df.columns]
        if missing:
            print(f"  {group_name}: SKIP — missing columns: {missing}")
            continue

        print(f"  Evaluating: {group_name} ({len(extra_cols)} features: {extra_cols})")
        result = train_and_evaluate(feature_cols, train_df, holdout_df, label=group_name)

        lift = BASELINE_MAE - (result["model_mae_book"] or result["mae"])
        eval_mae = result["model_mae_book"] or result["mae"]
        print(f"    MAE (overall): {result['mae']:.4f}")
        if result["model_mae_book"] is not None:
            print(f"    MAE (book-spread games): {result['model_mae_book']:.4f}")
        print(f"    Lift vs baseline: {lift:+.4f}")
        print()

        group_results[group_name] = {
            **result,
            "lift": lift,
            "n_extra_features": len(extra_cols),
        }

    # Step 3: Combine positive-lift features
    print("\n--- Step 3: Combined Feature Evaluation ---")
    positive_groups = [g for g, r in group_results.items() if r["lift"] > 0]
    print(f"Positive-lift groups: {positive_groups}")

    combined_result = None
    if positive_groups:
        combined_extra = []
        for g in positive_groups:
            combined_extra.extend(EXTRA_FEATURE_NAMES[g])

        combined_features = base_features + combined_extra
        print(f"Combined feature count: {len(combined_features)} "
              f"(37 base + {len(combined_extra)} extra)")

        combined_result = train_and_evaluate(
            combined_features, train_df, holdout_df, label="combined")

        combined_mae = combined_result["model_mae_book"] or combined_result["mae"]
        combined_lift = BASELINE_MAE - combined_mae
        print(f"  Combined MAE (overall): {combined_result['mae']:.4f}")
        if combined_result["model_mae_book"] is not None:
            print(f"  Combined MAE (book-spread games): {combined_result['model_mae_book']:.4f}")
        print(f"  Combined lift vs baseline: {combined_lift:+.4f}")

        # ROI evaluation
        if combined_result["preds"] is not None:
            preds = combined_result["preds"]
            median_sigma = preds["spread_sigma"].median()
            print(f"\n  ATS ROI (combined model):")
            for t in [3, 5, 7]:
                roi, n, w, l = compute_roi(preds, t)
                if roi is not None:
                    print(f"    Threshold {t}: {roi:+.1f}% ({n} bets, {w}W/{l}L)")
                roi_s, ns, ws, ls = compute_roi(preds, t, sigma_filter=median_sigma)
                if roi_s is not None:
                    print(f"    Threshold {t} (sigma<{median_sigma:.1f}): "
                          f"{roi_s:+.1f}% ({ns} bets, {ws}W/{ls}L)")

    # Also test with ALL groups regardless of individual lift
    print("\n--- Step 3b: All Groups Combined ---")
    all_extra = []
    for g in ALL_EXTRA_GROUPS:
        all_extra.extend(EXTRA_FEATURE_NAMES[g])
    all_features = base_features + [c for c in all_extra if c in holdout_df.columns]
    all_result = train_and_evaluate(all_features, train_df, holdout_df, label="all_groups")
    all_mae = all_result["model_mae_book"] or all_result["mae"]
    all_lift = BASELINE_MAE - all_mae
    print(f"  All-groups MAE (overall): {all_result['mae']:.4f}")
    if all_result["model_mae_book"] is not None:
        print(f"  All-groups MAE (book-spread games): {all_result['model_mae_book']:.4f}")
    print(f"  All-groups lift vs baseline: {all_lift:+.4f}")

    # Pick best result between combined positive-lift and all-groups
    best_result = None
    best_label = None
    if combined_result is not None:
        c_mae = combined_result["model_mae_book"] or combined_result["mae"]
        a_mae = all_result["model_mae_book"] or all_result["mae"]
        if c_mae <= a_mae:
            best_result = combined_result
            best_label = "positive-lift combined"
        else:
            best_result = all_result
            best_label = "all groups"
    else:
        best_result = all_result
        best_label = "all groups"

    best_mae = best_result["model_mae_book"] or best_result["mae"]
    best_lift = BASELINE_MAE - best_mae
    print(f"\n  Best expansion: {best_label} (MAE={best_mae:.4f}, lift={best_lift:+.4f})")

    # Step 4: Save if improved
    print("\n--- Step 4: Save Results ---")
    if best_lift > 0:
        print(f"  Improvement detected ({best_lift:+.4f}). Saving expanded model...")

        # Backup original feature order
        backup_path = config.ARTIFACTS_DIR / "feature_order_v1.json"
        if not backup_path.exists():
            original = json.loads((config.ARTIFACTS_DIR / "feature_order.json").read_text())
            with open(backup_path, "w") as f:
                json.dump(original, f, indent=2)
            print(f"  Backed up original feature order to {backup_path}")

        # Save expanded feature order
        expanded_order = list(best_result["feature_cols"])
        with open(config.ARTIFACTS_DIR / "feature_order.json", "w") as f:
            json.dump(expanded_order, f, indent=2)
        print(f"  Saved expanded feature order ({len(expanded_order)} features)")

        # Save expanded model checkpoints with correct feature order
        save_checkpoint(best_result["reg_model"], "regressor",
                        hparams={"epochs": 100}, subdir="no_garbage",
                        feature_order=expanded_order)
        save_checkpoint(best_result["cls_model"], "classifier",
                        hparams={"epochs": 100}, subdir="no_garbage",
                        feature_order=expanded_order)

        # Save scaler
        import pickle
        scaler_path = config.ARTIFACTS_DIR / "no_garbage" / "scaler.pkl"
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scaler_path, "wb") as f:
            pickle.dump(best_result["scaler"], f)
        print(f"  Saved scaler to {scaler_path}")
    else:
        print(f"  No improvement (lift={best_lift:+.4f}). Keeping original model.")

    # ── Generate report ──────────────────────────────────────────
    lines = []
    lines.append("# Feature Expansion Evaluation — Season 2025\n")
    lines.append(f"Baseline MAE (book-spread games): **{BASELINE_MAE}**\n")

    lines.append("## Individual Feature Group Results\n")
    lines.append("| Group | Extra Features | MAE (overall) | MAE (book games) | Lift |")
    lines.append("|-------|---------------|---------------|------------------|------|")
    for group_name in ALL_EXTRA_GROUPS:
        if group_name in group_results:
            r = group_results[group_name]
            book_str = f"{r['model_mae_book']:.4f}" if r["model_mae_book"] else "N/A"
            lines.append(f"| {group_name} | {r['n_extra_features']} | "
                         f"{r['mae']:.4f} | {book_str} | {r['lift']:+.4f} |")
        else:
            lines.append(f"| {group_name} | - | SKIPPED | - | - |")

    if combined_result is not None:
        lines.append(f"\n## Combined Positive-Lift Groups: {positive_groups}\n")
        c_mae = combined_result["model_mae_book"] or combined_result["mae"]
        c_lift = BASELINE_MAE - c_mae
        lines.append(f"- Features: {len(combined_result['feature_cols'])} "
                      f"(37 base + {len(combined_result['feature_cols']) - 37} extra)")
        lines.append(f"- MAE (overall): {combined_result['mae']:.4f}")
        if combined_result["model_mae_book"] is not None:
            lines.append(f"- MAE (book-spread games): {combined_result['model_mae_book']:.4f}")
        lines.append(f"- Lift vs baseline: {c_lift:+.4f}")

    lines.append(f"\n## All Groups Combined\n")
    lines.append(f"- Features: {len(all_result['feature_cols'])}")
    lines.append(f"- MAE (overall): {all_result['mae']:.4f}")
    if all_result["model_mae_book"] is not None:
        lines.append(f"- MAE (book-spread games): {all_result['model_mae_book']:.4f}")
    lines.append(f"- Lift vs baseline: {all_lift:+.4f}")

    lines.append(f"\n## Best Result: {best_label}\n")
    lines.append(f"- MAE: {best_mae:.4f}")
    lines.append(f"- Lift: {best_lift:+.4f}")
    lines.append(f"- Features: {best_result['feature_cols']}")

    # Monthly breakdown for best result
    if best_result["monthly"]:
        lines.append(f"\n### Monthly MAE (Best Model)\n")
        lines.append("| Month | MAE | Games |")
        lines.append("|-------|-----|-------|")
        for month in sorted(best_result["monthly"].keys()):
            m_mae, n = best_result["monthly"][month]
            lines.append(f"| {month} | {m_mae:.2f} | {n} |")

    # ROI for best result
    if best_result["preds"] is not None and "book_spread" in best_result["preds"].columns:
        preds = best_result["preds"]
        median_sigma = preds["spread_sigma"].median()
        p25_sigma = preds["spread_sigma"].quantile(0.25)

        lines.append(f"\n### ATS ROI (Best Model)\n")
        lines.append(f"Sigma: median={median_sigma:.2f}, p25={p25_sigma:.2f}\n")

        for cut_label, cut_val in [("Unfiltered", None),
                                    (f"Sigma<{median_sigma:.1f}", median_sigma)]:
            lines.append(f"#### {cut_label}\n")
            lines.append("| Threshold | Bets | Wins | Losses | Win Rate | ROI |")
            lines.append("|-----------|------|------|--------|----------|-----|")
            for t in [3, 5, 7]:
                roi, n_bets, wins, losses = compute_roi(preds, t, sigma_filter=cut_val)
                if roi is not None:
                    wr = wins / n_bets if n_bets > 0 else 0
                    lines.append(f"| {t} | {n_bets} | {wins} | {losses} | "
                                 f"{wr:.1%} | {roi:+.1f}% |")
                else:
                    lines.append(f"| {t} | 0 | - | - | - | - |")
            lines.append("")

    # Save status
    if best_lift > 0:
        lines.append(f"\n## Action Taken\n")
        lines.append(f"- Backed up `feature_order_v1.json` (37 features)")
        lines.append(f"- Updated `feature_order.json` ({len(best_result['feature_cols'])} features)")
        lines.append(f"- Retrained models in `checkpoints/no_garbage/`")
    else:
        lines.append(f"\n## Action Taken\n")
        lines.append(f"- No improvement detected. Original model preserved.")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
