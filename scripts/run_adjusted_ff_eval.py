"""Evaluate opponent-adjusted four-factors: train, ablate, compare.

Runs on GPU (H100). Reads pre-built adjusted feature parquets from
features/ directory. Does NOT require AWS credentials.

Tasks 5-7 from the adjusted four-factors plan:
  Task 5: Evaluate 3 configurations + ablation + permutation importance
  Task 6: Tune adjustment parameters (prior_weight, alpha)
  Task 7: Master comparison table

Usage:
    python -u scripts/run_adjusted_ff_eval.py [--task 5|6|7] [--alpha 1.0] [--prior 5]
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# Force unbuffered output
print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.architecture import MLPClassifier, MLPRegressor, gaussian_nll_loss
from src.features import (
    EXTRA_FEATURE_NAMES,
    get_feature_matrix,
    get_targets,
)
from src.trainer import (
    train_regressor,
    train_classifier,
    save_checkpoint,
    fit_scaler,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = PROJECT_ROOT / "reports" / "adjusted_four_factors_2025.md"
STATE_DIR = PROJECT_ROOT / ".adj_ff_state"

TRAIN_SEASONS = list(range(2015, 2025))
HOLDOUT_SEASON = 2025

# Feature order files
V1_FEATURES: list[str] = json.loads(
    (config.ARTIFACTS_DIR / "feature_order_v1.json").read_text()
)
V2_FEATURES: list[str] = json.loads(
    (config.ARTIFACTS_DIR / "feature_order_v2.json").read_text()
)

FORCE_REMOVE = {"away_team_home"}

# Known baselines (from raw/unadjusted features)
BASELINES = {
    "37_raw": {"features": 37, "mae": 9.62},
    "54_raw": {"features": 54, "mae": 9.3787},
    "10_pruned_raw": {"features": 10, "mae": 9.48},
}

# Default hparams (no Optuna)
DEFAULT_HP = {
    "hidden1": 256,
    "hidden2": 128,
    "dropout": 0.3,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 100,
    "batch_size": 256,
}

# Faster hparams for ablation search
SEARCH_HP = {**DEFAULT_HP, "epochs": 50}


# ── Data Loading ─────────────────────────────────────────────────


def load_adj_features(season: int) -> pd.DataFrame:
    path = config.FEATURES_DIR / f"season_{season}_no_garbage_adj_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Adjusted features not found: {path}")
    return pd.read_parquet(path)


def load_raw_features(season: int) -> pd.DataFrame:
    path = config.FEATURES_DIR / f"season_{season}_no_garbage_v2_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Raw v2 features not found: {path}")
    return pd.read_parquet(path)


def load_all_adj_data():
    """Load adjusted training + holdout data."""
    print("Loading adjusted training data (2015-2024)...")
    dfs = [load_adj_features(s) for s in TRAIN_SEASONS]
    train_df = pd.concat(dfs, ignore_index=True)
    train_df = train_df.dropna(subset=["homeScore", "awayScore"])
    print(f"  Training samples: {len(train_df)}")

    print("Loading adjusted holdout data (2025)...")
    holdout_df = load_adj_features(HOLDOUT_SEASON)
    holdout_df = holdout_df.dropna(subset=["homeScore", "awayScore"])
    print(f"  Holdout samples: {len(holdout_df)}")

    return train_df, holdout_df


def load_all_raw_data():
    """Load raw v2 training + holdout data."""
    print("Loading raw v2 training data (2015-2024)...")
    dfs = [load_raw_features(s) for s in TRAIN_SEASONS]
    train_df = pd.concat(dfs, ignore_index=True)
    train_df = train_df.dropna(subset=["homeScore", "awayScore"])
    print(f"  Training samples: {len(train_df)}")

    print("Loading raw v2 holdout data (2025)...")
    holdout_df = load_raw_features(HOLDOUT_SEASON)
    holdout_df = holdout_df.dropna(subset=["homeScore", "awayScore"])
    print(f"  Holdout samples: {len(holdout_df)}")

    return train_df, holdout_df


# ── Training & Evaluation ────────────────────────────────────────


def load_holdout_lines() -> pd.DataFrame:
    """Load book spreads for holdout season from v2 parquet (no S3 needed)."""
    # Lines are attached to parquets during build — check if book_spread col exists
    # If not, try loading from a cached lines file
    lines_path = config.FEATURES_DIR / "lines_2025.parquet"
    if lines_path.exists():
        return pd.read_parquet(lines_path)
    return pd.DataFrame()


def train_and_predict(feature_cols, train_df, holdout_df, hparams=None):
    """Train regressor + classifier, predict on holdout."""
    hp = hparams or DEFAULT_HP

    X_train = get_feature_matrix(train_df, feature_order=feature_cols).values.astype(np.float32)
    targets = get_targets(train_df)
    y_spread = targets["spread_home"].values.astype(np.float32)
    y_win = targets["home_win"].values.astype(np.float32)

    X_holdout = get_feature_matrix(holdout_df, feature_order=feature_cols).values.astype(np.float32)
    targets_h = get_targets(holdout_df)
    y_test = targets_h["spread_home"].values.astype(np.float32)

    # Handle NaN
    nan_mask_train = np.isnan(X_train)
    if nan_mask_train.any():
        col_means = np.nanmean(X_train, axis=0)
        for j in range(X_train.shape[1]):
            X_train[nan_mask_train[:, j], j] = col_means[j]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    nan_mask_hold = np.isnan(X_holdout)
    if nan_mask_hold.any():
        col_means = scaler.mean_
        for j in range(X_holdout.shape[1]):
            X_holdout[nan_mask_hold[:, j], j] = col_means[j]

    X_holdout_scaled = scaler.transform(X_holdout)

    # Train
    reg = train_regressor(X_train_scaled, y_spread, hparams=hp)
    cls = train_classifier(X_train_scaled, y_win, hparams=hp)

    # Predict
    reg.eval()
    cls.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_holdout_scaled, dtype=torch.float32)
        mu_t, log_sigma_t = reg(X_t)
        sigma_t = torch.nn.functional.softplus(log_sigma_t) + 1e-3
        sigma_t = sigma_t.clamp(min=0.5, max=30.0)
        prob_t = torch.sigmoid(cls(X_t))

    mu = mu_t.numpy()
    sigma = sigma_t.numpy()
    prob = prob_t.numpy()

    preds = holdout_df[["gameId", "homeTeamId", "awayTeamId",
                         "homeScore", "awayScore", "startDate"]].copy()
    preds["predicted_spread"] = mu
    preds["spread_sigma"] = sigma
    preds["home_win_prob"] = prob
    preds["actual_margin"] = y_test

    return mu, sigma, y_test, preds, reg, cls, scaler


def attach_book_spreads(preds):
    """Attach book spreads from v2 holdout parquet."""
    try:
        from src.features import load_lines
        lines = load_lines(HOLDOUT_SEASON)
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
    return preds


def compute_book_mae(preds):
    """Compute MAE on book-spread games."""
    with_book = preds.dropna(subset=["book_spread"])
    if len(with_book) == 0:
        return None, 0
    mae = float(np.abs(with_book["predicted_spread"] - with_book["actual_margin"]).mean())
    return mae, len(with_book)


def evaluate_config(label, feature_cols, train_df, holdout_df, hparams=None):
    """Train, predict, and compute full metrics for a feature configuration."""
    print(f"\n  Evaluating: {label} ({len(feature_cols)} features)")
    mu, sigma, y_test, preds, reg, cls, scaler = train_and_predict(
        feature_cols, train_df, holdout_df, hparams=hparams)
    preds = attach_book_spreads(preds)

    mae_overall = float(np.mean(np.abs(mu - y_test)))
    mae_book, n_book = compute_book_mae(preds)

    print(f"    MAE (overall): {mae_overall:.4f}")
    print(f"    MAE (book, n={n_book}): {mae_book:.4f}" if mae_book else "    No book spreads")

    return {
        "label": label,
        "n_features": len(feature_cols),
        "features": feature_cols,
        "mae_overall": mae_overall,
        "mae_book": mae_book,
        "n_book": n_book,
        "preds": preds,
        "reg": reg,
        "cls": cls,
        "scaler": scaler,
        "sigma": sigma,
    }


def compute_roi(preds_df, threshold, sigma_filter=None):
    """Compute ATS ROI."""
    with_book = preds_df.dropna(subset=["book_spread"]).copy()
    if sigma_filter is not None:
        with_book = with_book[with_book["spread_sigma"] < sigma_filter]
    if len(with_book) == 0:
        return {"bets": 0, "wins": 0, "losses": 0, "win_rate": 0, "roi": 0}

    with_book["edge"] = with_book["model_spread"] - with_book["book_spread"]
    bets = with_book[with_book["edge"].abs() >= threshold]
    if len(bets) == 0:
        return {"bets": 0, "wins": 0, "losses": 0, "win_rate": 0, "roi": 0}

    bets = bets.copy()
    bets["bet_side"] = np.sign(bets["edge"])
    bets["cover"] = np.sign(bets["actual_margin"] + bets["book_spread"])
    bets["win"] = (bets["bet_side"] == bets["cover"]).astype(int)
    bets["push"] = (bets["cover"] == 0).astype(int)

    non_push = bets[bets["push"] == 0]
    wins = int(non_push["win"].sum())
    losses = len(non_push) - wins
    wr = wins / len(non_push) if len(non_push) > 0 else 0
    roi = (wins - losses * 1.1) / len(non_push) if len(non_push) > 0 else 0

    return {"bets": len(non_push), "wins": wins, "losses": losses,
            "win_rate": wr, "roi": roi}


def compute_calibration(preds_df):
    """Compute calibration (% within 1/2 sigma)."""
    valid = preds_df.dropna(subset=["predicted_spread"]).copy()
    if len(valid) == 0:
        return None, None

    residual = np.abs(valid["predicted_spread"].values - valid["actual_margin"].values)
    sigma = valid["spread_sigma"].values

    within_1 = float((residual <= sigma).mean())
    within_2 = float((residual <= 2 * sigma).mean())
    return within_1, within_2


def compute_monthly_mae(preds_df):
    """Compute MAE broken down by month."""
    valid = preds_df.dropna(subset=["book_spread"]).copy()
    valid["month"] = pd.to_datetime(valid["startDate"], errors="coerce").dt.to_period("M")
    return valid.groupby("month").apply(
        lambda g: pd.Series({
            "mae": np.abs(g["predicted_spread"] - g["actual_margin"]).mean(),
            "games": len(g),
        })
    ).reset_index()


def full_metrics(result):
    """Compute full evaluation metrics for a result dict."""
    preds = result["preds"]
    sigma_vals = result["sigma"]

    # Sigma percentile for filtering
    valid_sigma = preds.dropna(subset=["book_spread"])["spread_sigma"]
    sigma_p25 = float(valid_sigma.quantile(0.25)) if len(valid_sigma) > 0 else 11.5

    # Monthly MAE
    monthly = compute_monthly_mae(preds)

    # ATS ROI
    roi_unfiltered = {t: compute_roi(preds, t) for t in [3, 5, 7]}
    roi_filtered = {t: compute_roi(preds, t, sigma_filter=sigma_p25) for t in [3, 5, 7]}

    # Calibration
    cal_1, cal_2 = compute_calibration(preds)

    return {
        "monthly": monthly,
        "roi_unfiltered": roi_unfiltered,
        "roi_filtered": roi_filtered,
        "sigma_p25": sigma_p25,
        "cal_1sigma": cal_1,
        "cal_2sigma": cal_2,
    }


# ── Permutation Importance ───────────────────────────────────────


def permutation_importance(feature_cols, train_df, holdout_df, n_repeats=10):
    """Compute permutation importance for each feature."""
    print(f"\n--- Permutation Importance ({len(feature_cols)} features, {n_repeats} repeats) ---")

    mu, sigma, y_test, preds, reg, cls, scaler = train_and_predict(
        feature_cols, train_df, holdout_df)
    preds = attach_book_spreads(preds)
    baseline_mae, _ = compute_book_mae(preds)
    print(f"  Baseline MAE (book): {baseline_mae:.4f}")

    X_holdout = get_feature_matrix(holdout_df, feature_order=feature_cols).values.astype(np.float32)
    nan_mask = np.isnan(X_holdout)
    if nan_mask.any():
        col_means = scaler.mean_
        for j in range(X_holdout.shape[1]):
            X_holdout[nan_mask[:, j], j] = col_means[j]
    X_scaled = scaler.transform(X_holdout)

    with_book_mask = preds["book_spread"].notna().values
    actual = y_test[with_book_mask]

    importance = {}
    for j, feat in enumerate(feature_cols):
        deltas = []
        for rep in range(n_repeats):
            X_perm = X_scaled.copy()
            rng = np.random.RandomState(42 + rep)
            X_perm[:, j] = rng.permutation(X_perm[:, j])

            with torch.no_grad():
                X_t = torch.tensor(X_perm, dtype=torch.float32)
                mu_p, _ = reg(X_t)
            mu_perm = mu_p.numpy()
            mae_perm = float(np.abs(mu_perm[with_book_mask] - actual).mean())
            deltas.append(mae_perm - baseline_mae)

        importance[feat] = {"mean": np.mean(deltas), "std": np.std(deltas)}
        print(f"  {feat}: +{np.mean(deltas):.4f} (+/- {np.std(deltas):.4f})")

    # Sort by importance
    sorted_feats = sorted(importance.keys(), key=lambda f: importance[f]["mean"], reverse=True)
    return sorted_feats, importance, baseline_mae


# ── Ablation (Backward Elimination) ─────────────────────────────


def ablation_backward(feature_cols, train_df, holdout_df, importance_ranking):
    """Backward elimination: coarse (remove bottom 5) then fine (remove 1)."""
    print(f"\n--- Backward Elimination ---")

    current_features = [f for f in feature_cols if f not in FORCE_REMOVE]
    log = []

    # Initial MAE
    result = evaluate_config("start", current_features, train_df, holdout_df, hparams=SEARCH_HP)
    log.append({"step": "start", "n_features": len(current_features),
                "mae_book": result["mae_book"], "removed": [], "features": list(current_features)})

    # Coarse: remove bottom 5 by importance
    ranked = [f for f in importance_ranking if f in current_features]

    while len(current_features) > 13:
        bottom5 = ranked[-5:]
        candidate = [f for f in current_features if f not in bottom5]
        result = evaluate_config(f"coarse_{len(log)}", candidate, train_df, holdout_df, hparams=SEARCH_HP)
        log.append({"step": f"coarse_{len(log)}", "n_features": len(candidate),
                     "mae_book": result["mae_book"], "removed": bottom5, "features": list(candidate)})
        current_features = candidate
        ranked = [f for f in ranked if f not in bottom5]

    # Fine: remove 1 at a time
    while len(current_features) > 8:
        best_mae = None
        best_remove = None
        best_features = None

        for feat in list(current_features):
            candidate = [f for f in current_features if f != feat]
            result = evaluate_config(f"try_remove_{feat}", candidate, train_df, holdout_df, hparams=SEARCH_HP)
            if best_mae is None or result["mae_book"] < best_mae:
                best_mae = result["mae_book"]
                best_remove = feat
                best_features = candidate

        print(f"  Best removal: {best_remove} -> MAE={best_mae:.4f}")
        current_features = best_features
        log.append({"step": f"fine_remove_{best_remove}", "n_features": len(current_features),
                     "mae_book": best_mae, "removed": [best_remove], "features": list(current_features)})

    # Full validation at 100 epochs for the final set
    print(f"\n  Full validation (100 epochs) for {len(current_features)} features...")
    result = evaluate_config("backward_final", current_features, train_df, holdout_df, hparams=DEFAULT_HP)

    return current_features, result["mae_book"], log


# ── Task 5: Main Evaluation ─────────────────────────────────────


def task5(train_adj, holdout_adj, train_raw, holdout_raw, report_lines):
    """Evaluate adjusted four-factors: 3 configs + ablation + permutation importance."""
    print(f"\n{'='*70}")
    print("TASK 5: EVALUATE ADJUSTED FOUR-FACTORS")
    print(f"{'='*70}")

    results = {}

    # Config 1: 37 features with adjusted four-factors
    report_lines.append("\n## Config 1: 37 features (adjusted four-factors)\n")
    feat_37 = V1_FEATURES
    r1 = evaluate_config("37_adj", feat_37, train_adj, holdout_adj)
    results["37_adj"] = r1
    m1 = full_metrics(r1)
    report_lines.append(f"- MAE (book): **{r1['mae_book']:.4f}** (vs raw 37: {BASELINES['37_raw']['mae']})")
    report_lines.append(f"- Calibration: {m1['cal_1sigma']:.1%} within 1σ, {m1['cal_2sigma']:.1%} within 2σ")

    # Config 2: 54 features with adjusted four-factors
    report_lines.append("\n## Config 2: 54 features (adjusted four-factors)\n")
    feat_54 = [f for f in V2_FEATURES if f not in FORCE_REMOVE]
    r2 = evaluate_config("54_adj", feat_54, train_adj, holdout_adj)
    results["54_adj"] = r2
    m2 = full_metrics(r2)
    report_lines.append(f"- MAE (book): **{r2['mae_book']:.4f}** (vs raw 54: {BASELINES['54_raw']['mae']})")
    report_lines.append(f"- Calibration: {m2['cal_1sigma']:.1%} within 1σ, {m2['cal_2sigma']:.1%} within 2σ")

    # Permutation importance on Config 2
    report_lines.append("\n### Permutation Importance (Config 2: 54 adjusted)\n")
    sorted_feats, importance, pi_baseline = permutation_importance(
        feat_54, train_adj, holdout_adj, n_repeats=10)

    report_lines.append("| Rank | Feature | MAE Increase | Std |")
    report_lines.append("|------|---------|-------------|-----|")
    for rank, feat in enumerate(sorted_feats, 1):
        imp = importance[feat]
        report_lines.append(f"| {rank} | {feat} | +{imp['mean']:.4f} | {imp['std']:.4f} |")

    # Count how many Group 2 (rolling four-factor) features are in top 20
    rolling_feats_set = set()
    for fmap in [
        "away_eff_fg_pct", "away_ft_pct", "away_ft_rate", "away_3pt_rate",
        "away_3p_pct", "away_off_rebound_pct", "away_def_rebound_pct",
        "away_def_eff_fg_pct", "away_def_ft_rate", "away_def_3pt_rate",
        "away_def_3p_pct", "away_def_off_rebound_pct", "away_def_def_rebound_pct",
        "home_eff_fg_pct", "home_ft_pct", "home_ft_rate", "home_3pt_rate",
        "home_3p_pct", "home_off_rebound_pct", "home_def_rebound_pct",
        "home_def_eff_fg_pct", "home_opp_ft_rate", "home_def_3pt_rate",
        "home_def_3p_pct", "home_def_off_rebound_pct", "home_def_def_rebound_pct",
    ]:
        rolling_feats_set.add(fmap)

    top20_rolling = sum(1 for f in sorted_feats[:20] if f in rolling_feats_set)
    report_lines.append(f"\nGroup 2 (rolling four-factor) features in top 20: **{top20_rolling}/20**")

    # Config 3: Ablation on adjusted 54 features
    report_lines.append("\n## Config 3: Ablation on adjusted 54 features\n")
    ablated_features, ablated_mae, ablation_log = ablation_backward(
        feat_54, train_adj, holdout_adj, sorted_feats)

    report_lines.append("### Backward Elimination Log\n")
    report_lines.append("| Step | Features | MAE (book) | Removed |")
    report_lines.append("|------|----------|-----------|---------|")
    for entry in ablation_log:
        removed_str = ", ".join(entry["removed"]) if entry["removed"] else "-"
        report_lines.append(f"| {entry['step']} | {entry['n_features']} | "
                          f"{entry['mae_book']:.4f} | {removed_str} |")

    # Count surviving four-factor features
    surviving_ff = [f for f in ablated_features if f in rolling_feats_set]
    report_lines.append(f"\nSurviving four-factor features: **{len(surviving_ff)}/{len(rolling_feats_set)}**")
    report_lines.append(f"Features: {surviving_ff}")
    report_lines.append(f"Pruned MAE: **{ablated_mae:.4f}**")

    results["ablated_adj"] = {
        "n_features": len(ablated_features),
        "features": ablated_features,
        "mae_book": ablated_mae,
    }

    # Config 2 with and without conf_strength
    report_lines.append("\n## Conference Strength Redundancy Test\n")

    feat_no_conf = [f for f in feat_54 if f not in ("home_conf_strength", "away_conf_strength")]
    r_no_conf = evaluate_config("54_adj_no_conf", feat_no_conf, train_adj, holdout_adj)
    report_lines.append(f"- 54 adj WITH conf_strength: MAE={r2['mae_book']:.4f}")
    report_lines.append(f"- 54 adj WITHOUT conf_strength: MAE={r_no_conf['mae_book']:.4f}")
    report_lines.append(f"- Delta: {r2['mae_book'] - r_no_conf['mae_book']:+.4f}")

    # Full metrics tables for configs
    for label, result_obj in [("Config 1 (37 adj)", r1), ("Config 2 (54 adj)", r2)]:
        m = full_metrics(result_obj)
        report_lines.append(f"\n### {label} — Monthly MAE\n")
        report_lines.append("| Month | MAE | Games |")
        report_lines.append("|-------|-----|-------|")
        for _, row in m["monthly"].iterrows():
            report_lines.append(f"| {row['month']} | {row['mae']:.2f} | {int(row['games'])} |")

        report_lines.append(f"\n### {label} — ATS ROI (Unfiltered)\n")
        report_lines.append("| Threshold | Bets | Wins | Losses | Win Rate | ROI |")
        report_lines.append("|-----------|------|------|--------|----------|-----|")
        for t in [3, 5, 7]:
            r = m["roi_unfiltered"][t]
            report_lines.append(f"| {t} | {r['bets']} | {r['wins']} | {r['losses']} | "
                              f"{r['win_rate']:.1%} | {r['roi']:+.1%} |")

        report_lines.append(f"\n### {label} — ATS ROI (Sigma < {m['sigma_p25']:.1f})\n")
        report_lines.append("| Threshold | Bets | Wins | Losses | Win Rate | ROI |")
        report_lines.append("|-----------|------|------|--------|----------|-----|")
        for t in [3, 5, 7]:
            r = m["roi_filtered"][t]
            report_lines.append(f"| {t} | {r['bets']} | {r['wins']} | {r['losses']} | "
                              f"{r['win_rate']:.1%} | {r['roi']:+.1%} |")

    # Save state
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "preds" and kk != "reg" and kk != "cls" and kk != "scaler" and kk != "sigma"}
                    for k, v in results.items()},
        "importance": importance,
        "importance_ranking": sorted_feats,
        "ablated_features": ablated_features,
        "ablated_mae": ablated_mae,
    }
    with open(STATE_DIR / "task5.json", "w") as f:
        json.dump(state, f, indent=2, default=str)
    print("State saved to .adj_ff_state/task5.json")

    return results


# ── Task 6: Tune Adjustment Parameters ──────────────────────────


def task6(report_lines):
    """Tune prior_weight and alpha on the adjusted features."""
    print(f"\n{'='*70}")
    print("TASK 6: TUNE ADJUSTMENT PARAMETERS")
    print(f"{'='*70}")

    # Load task5 state to get best feature set
    state5_path = STATE_DIR / "task5.json"
    if state5_path.exists():
        with open(state5_path) as f:
            state5 = json.load(f)
        best_features = state5.get("ablated_features", V2_FEATURES)
    else:
        best_features = V2_FEATURES

    # We need to rebuild features with different alpha/prior_weight combos.
    # Since rebuilding is expensive (S3 reads), we check if parquets exist.
    # On GPU box, we test with pre-built parquets at alpha=1.0, prior=5.
    # For parameter tuning, we need to rebuild locally first.

    # For now, test with existing parquets and log what we have
    report_lines.append("\n## Task 6: Parameter Tuning\n")

    param_combos = [
        {"alpha": 1.0, "prior": 5},
        {"alpha": 0.85, "prior": 5},
        {"alpha": 0.7, "prior": 5},
        {"alpha": 0.5, "prior": 5},
        {"alpha": 1.0, "prior": 3},
        {"alpha": 1.0, "prior": 10},
        {"alpha": 1.0, "prior": 15},
        {"alpha": 0.85, "prior": 10},
    ]

    results = []

    for params in param_combos:
        alpha = params["alpha"]
        prior = params["prior"]
        suffix = f"a{alpha}_p{prior}"
        parquet_path = config.FEATURES_DIR / f"season_2025_no_garbage_adj_{suffix}_features.parquet"

        if not parquet_path.exists():
            # Check if default params match
            if alpha == 1.0 and prior == 5:
                parquet_path = config.FEATURES_DIR / "season_2025_no_garbage_adj_features.parquet"
            else:
                print(f"  Skipping alpha={alpha}, prior={prior} — parquet not found")
                continue

        if not parquet_path.exists():
            print(f"  Skipping alpha={alpha}, prior={prior} — parquet not found")
            continue

        print(f"\n  Testing alpha={alpha}, prior={prior}...")

        # Load adjusted data for this combo
        holdout_df = pd.read_parquet(parquet_path)
        holdout_df = holdout_df.dropna(subset=["homeScore", "awayScore"])

        # Load training data (same combo)
        train_dfs = []
        for s in TRAIN_SEASONS:
            if alpha == 1.0 and prior == 5:
                tp = config.FEATURES_DIR / f"season_{s}_no_garbage_adj_features.parquet"
            else:
                tp = config.FEATURES_DIR / f"season_{s}_no_garbage_adj_{suffix}_features.parquet"
            if tp.exists():
                train_dfs.append(pd.read_parquet(tp))
        if not train_dfs:
            print(f"  Skipping — no training data")
            continue

        train_df = pd.concat(train_dfs, ignore_index=True)
        train_df = train_df.dropna(subset=["homeScore", "awayScore"])

        r = evaluate_config(f"alpha={alpha},prior={prior}", best_features,
                           train_df, holdout_df, hparams=DEFAULT_HP)
        r = attach_book_spreads(r["preds"])  # This overwrites — need different approach

        # Re-evaluate properly
        result = evaluate_config(f"alpha={alpha},prior={prior}", best_features,
                                train_df, holdout_df, hparams=DEFAULT_HP)
        results.append({
            "alpha": alpha,
            "prior": prior,
            "mae_book": result["mae_book"],
            "mae_overall": result["mae_overall"],
        })

    report_lines.append("| Alpha | Prior Weight | MAE (book) | MAE (overall) |")
    report_lines.append("|-------|-------------|-----------|--------------|")
    for r in results:
        report_lines.append(f"| {r['alpha']} | {r['prior']} | {r['mae_book']:.4f} | {r['mae_overall']:.4f} |")

    if results:
        best = min(results, key=lambda x: x["mae_book"])
        report_lines.append(f"\n**Best**: alpha={best['alpha']}, prior={best['prior']}, MAE={best['mae_book']:.4f}")
    else:
        report_lines.append("\nNo parameter combos evaluated (need pre-built parquets).")
        best = {"alpha": 1.0, "prior": 5}

    # Save state
    with open(STATE_DIR / "task6.json", "w") as f:
        json.dump({"results": results, "best": best}, f, indent=2)

    return best


# ── Task 7: Master Comparison Table ──────────────────────────────


def task7(results5, best_params, report_lines):
    """Build the definitive comparison table."""
    print(f"\n{'='*70}")
    print("TASK 7: MASTER COMPARISON TABLE")
    print(f"{'='*70}")

    report_lines.append("\n## Task 7: Master Comparison Table\n")

    # Collect all known results
    rows = [
        ("Original (sos=1.0, 37 raw)", 37, 9.87, None, None),
        ("sos=0.85, 37 raw", 37, 9.62, None, None),
        ("sos=0.85, 54 raw", 54, 9.38, None, None),
        ("sos=0.85, 10 pruned raw", 10, 9.48, None, None),
    ]

    # Add adjusted results from task5
    if results5:
        for key, label in [
            ("37_adj", "sos=0.85, 37 adjusted"),
            ("54_adj", "sos=0.85, 54 adjusted"),
        ]:
            if key in results5:
                r = results5[key]
                m = full_metrics(r) if "preds" in r else None
                roi_5 = m["roi_filtered"][5]["roi"] if m else None
                roi_7 = m["roi_filtered"][7]["roi"] if m else None
                rows.append((label, r["n_features"], r["mae_book"], roi_5, roi_7))

        if "ablated_adj" in results5:
            ab = results5["ablated_adj"]
            rows.append((f"sos=0.85, {ab['n_features']} pruned adjusted",
                        ab["n_features"], ab["mae_book"], None, None))

    report_lines.append("| Config | Features | MAE | vs Book | Sigma<p25 @5 ROI | Sigma<p25 @7 ROI |")
    report_lines.append("|--------|----------|-----|---------|-------------------|-------------------|")
    book_mae = 8.76
    for label, n_feat, mae, roi5, roi7 in rows:
        vs_book = f"+{mae - book_mae:.2f}" if mae else "?"
        roi5_str = f"{roi5:+.1%}" if roi5 is not None else "?"
        roi7_str = f"{roi7:+.1%}" if roi7 is not None else "?"
        mae_str = f"{mae:.2f}" if mae else "?"
        report_lines.append(f"| {label} | {n_feat} | {mae_str} | {vs_book} | {roi5_str} | {roi7_str} |")


# ── Report Generation ────────────────────────────────────────────


def write_report(lines):
    """Write report to file."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines)
    with open(REPORT_PATH, "w") as f:
        f.write(content)
    print(f"\nReport saved to: {REPORT_PATH}")


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Evaluate adjusted four-factors")
    parser.add_argument("--task", type=int, choices=[5, 6, 7], help="Run specific task")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--prior", type=float, default=5.0)
    args = parser.parse_args()

    report_lines = ["# Opponent-Adjusted Four-Factors — Season 2025\n"]

    tasks = [args.task] if args.task else [5, 6, 7]

    results5 = None
    best_params = {"alpha": args.alpha, "prior": args.prior}

    if 5 in tasks:
        train_adj, holdout_adj = load_all_adj_data()
        train_raw, holdout_raw = load_all_raw_data()
        results5 = task5(train_adj, holdout_adj, train_raw, holdout_raw, report_lines)

    if 6 in tasks:
        best_params = task6(report_lines)

    if 7 in tasks:
        # Reload task5 results if not in memory
        if results5 is None:
            state5_path = STATE_DIR / "task5.json"
            if state5_path.exists():
                with open(state5_path) as f:
                    state5 = json.load(f)
                results5 = state5.get("results", {})
        task7(results5, best_params, report_lines)

    write_report(report_lines)
    print("\nDone!")


if __name__ == "__main__":
    main()
