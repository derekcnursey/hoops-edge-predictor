"""Feature Selection / Pruning — Tasks 1-4.

Task 1: Permutation importance on all 54 features
Task 2: Ablation study (backward elimination + forward selection)
Task 3: Overfitting diagnostics
Task 4: Final evaluation and comparison table

Usage:
    poetry run python scripts/feature_selection.py [--task N]
    # N=1,2,3,4 or omit for all tasks
"""
from __future__ import annotations

import argparse
import functools
import json
import pickle
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
from src.features import get_targets, load_lines
from src.trainer import train_regressor, train_classifier, save_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = PROJECT_ROOT / "reports" / "feature_selection_2025.md"

TRAIN_SEASONS = list(range(2015, 2025))
HOLDOUT_SEASON = 2025

# Load the 54-feature order from v2 backup
V2_FEATURES: list[str] = json.loads(
    (config.ARTIFACTS_DIR / "feature_order_v2.json").read_text()
)
assert len(V2_FEATURES) == 54, f"Expected 54 v2 features, got {len(V2_FEATURES)}"

# Known constant / useless features to remove unconditionally
FORCE_REMOVE = {"away_team_home"}

BASELINE_37_MAE = 9.62   # original 37-feature MAE on book-spread games
BASELINE_54_MAE = 9.3787  # 54-feature MAE on book-spread games


# ── Data loading ────────────────────────────────────────────────


def load_v2_features(season: int) -> pd.DataFrame:
    path = config.FEATURES_DIR / f"season_{season}_no_garbage_v2_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"v2 features not found: {path}")
    return pd.read_parquet(path)


def load_all_data():
    """Load training + holdout data from v2 parquets."""
    print("Loading v2 training data (2015-2024)...")
    dfs = []
    for s in TRAIN_SEASONS:
        dfs.append(load_v2_features(s))
    train_df = pd.concat(dfs, ignore_index=True)
    train_df = train_df.dropna(subset=["homeScore", "awayScore"])
    print(f"  Training samples: {len(train_df)}")

    print("Loading v2 holdout data (2025)...")
    holdout_df = load_v2_features(HOLDOUT_SEASON)
    holdout_df = holdout_df.dropna(subset=["homeScore", "awayScore"])
    print(f"  Holdout samples: {len(holdout_df)}")
    return train_df, holdout_df


# ── Training helpers ────────────────────────────────────────────


def train_and_predict(feature_cols, train_df, holdout_df, hparams=None):
    """Train regressor+classifier and return predictions on holdout.

    Returns (mu, sigma, y_spread_test, preds_df, reg_model, cls_model, scaler).
    """
    hp = hparams or {"epochs": 100}

    X_train = train_df[feature_cols].values.astype(np.float32)
    X_train = np.nan_to_num(X_train, nan=0.0)
    targets_train = get_targets(train_df)
    y_spread_train = targets_train["spread_home"].values.astype(np.float32)
    y_win_train = targets_train["home_win"].values.astype(np.float32)

    X_test = holdout_df[feature_cols].values.astype(np.float32)
    X_test = np.nan_to_num(X_test, nan=0.0)
    targets_test = get_targets(holdout_df)
    y_spread_test = targets_test["spread_home"].values.astype(np.float32)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)

    reg_model = train_regressor(X_train_s, y_spread_train, hparams=hp)
    cls_model = train_classifier(X_train_s, y_win_train, hparams=hp)

    reg_model.eval()
    cls_model.eval()
    X_test_s = scaler.transform(X_test)
    X_t = torch.tensor(X_test_s, dtype=torch.float32)

    with torch.no_grad():
        mu, log_sigma = reg_model(X_t)
        sigma = torch.nn.functional.softplus(log_sigma) + 1e-3
        sigma = sigma.clamp(min=0.5, max=30.0)
        logits = cls_model(X_t)
        prob = torch.sigmoid(logits).numpy()

    mu = mu.numpy()
    sigma = sigma.numpy()

    preds = holdout_df[["gameId", "homeTeamId", "awayTeamId",
                         "homeScore", "awayScore", "startDate"]].copy()
    preds["predicted_spread"] = mu
    preds["spread_sigma"] = sigma
    preds["home_win_prob"] = prob
    preds["actual_margin"] = preds["homeScore"].astype(float) - preds["awayScore"].astype(float)

    return mu, sigma, y_spread_test, preds, reg_model, cls_model, scaler


def attach_book_spreads(preds):
    """Attach book spreads to predictions DataFrame."""
    try:
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
    """Compute MAE on book-spread games only."""
    with_book = preds.dropna(subset=["book_spread"])
    if len(with_book) == 0:
        return None, 0
    mae = float(np.abs(with_book["predicted_spread"] - with_book["actual_margin"]).mean())
    return mae, len(with_book)


def quick_evaluate(feature_cols, train_df, holdout_df, label="", hparams=None):
    """Train and evaluate, returning dict with key metrics."""
    mu, sigma, y_test, preds, reg, cls, scaler = train_and_predict(
        feature_cols, train_df, holdout_df, hparams=hparams)
    preds = attach_book_spreads(preds)

    mae_overall = float(np.mean(np.abs(mu - y_test)))
    mae_book, n_book = compute_book_mae(preds)

    return {
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "mae_overall": mae_overall,
        "mae_book": mae_book,
        "n_book_games": n_book,
        "preds": preds,
        "reg_model": reg,
        "cls_model": cls,
        "scaler": scaler,
        "label": label,
    }


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


# ══════════════════════════════════════════════════════════════════
# TASK 1: Permutation Importance
# ══════════════════════════════════════════════════════════════════


def task1_permutation_importance(train_df, holdout_df, n_repeats=15):
    """Run permutation importance on all 54 features."""
    print("\n" + "=" * 70)
    print("TASK 1: PERMUTATION IMPORTANCE (54 features)")
    print("=" * 70)

    feature_cols = list(V2_FEATURES)
    print(f"\nTraining baseline model with {len(feature_cols)} features...")
    mu, sigma, y_test, preds, reg, cls, scaler = train_and_predict(
        feature_cols, train_df, holdout_df)

    baseline_mae = float(np.mean(np.abs(mu - y_test)))
    print(f"Baseline MAE (overall): {baseline_mae:.4f}")

    preds = attach_book_spreads(preds)
    baseline_book_mae, n_book = compute_book_mae(preds)
    print(f"Baseline MAE (book games): {baseline_book_mae:.4f} ({n_book} games)")

    # Permutation importance
    print(f"\nRunning permutation importance ({n_repeats} repeats per feature)...")
    X_test = holdout_df[feature_cols].values.astype(np.float32)
    X_test = np.nan_to_num(X_test, nan=0.0)
    X_test_scaled = scaler.transform(X_test)

    rng = np.random.RandomState(42)
    importance = {}

    reg.eval()
    for i, feat_name in enumerate(feature_cols):
        mae_increases = []
        for rep in range(n_repeats):
            X_shuffled = X_test_scaled.copy()
            X_shuffled[:, i] = rng.permutation(X_shuffled[:, i])
            X_t = torch.tensor(X_shuffled, dtype=torch.float32)
            with torch.no_grad():
                mu_perm, _ = reg(X_t)
            perm_mae = float(np.mean(np.abs(mu_perm.numpy() - y_test)))
            mae_increases.append(perm_mae - baseline_mae)

        mean_inc = np.mean(mae_increases)
        std_inc = np.std(mae_increases)
        importance[feat_name] = {
            "mean_mae_increase": mean_inc,
            "std_mae_increase": std_inc,
            "rank": 0,
        }
        sign = "+" if mean_inc > 0 else ""
        print(f"  [{i+1:2d}/54] {feat_name:35s} {sign}{mean_inc:.4f} (+/-{std_inc:.4f})")

    # Rank by importance (higher MAE increase = more important)
    ranked = sorted(importance.items(), key=lambda x: -x[1]["mean_mae_increase"])
    for rank, (name, info) in enumerate(ranked, 1):
        info["rank"] = rank

    print("\n--- Ranked Features ---")
    print(f"{'Rank':>4s}  {'Feature':35s}  {'MAE Increase':>12s}  {'Std':>8s}")
    zero_neg = []
    for rank, (name, info) in enumerate(ranked, 1):
        mi = info["mean_mae_increase"]
        print(f"{rank:4d}  {name:35s}  {mi:+12.4f}  {info['std_mae_increase']:8.4f}")
        if mi <= 0:
            zero_neg.append(name)

    print(f"\nZero/negative importance features ({len(zero_neg)}): {zero_neg}")
    print(f"Features forced to remove: {FORCE_REMOVE}")

    # Multicollinearity check
    print("\n--- Multicollinearity Check (|r| > 0.95) ---")
    X_all = holdout_df[feature_cols].values.astype(np.float32)
    X_all = np.nan_to_num(X_all, nan=0.0)
    corr_matrix = np.corrcoef(X_all.T)
    high_corr_pairs = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            r = abs(corr_matrix[i, j])
            if r > 0.95:
                high_corr_pairs.append((feature_cols[i], feature_cols[j], r))
    if high_corr_pairs:
        for f1, f2, r in sorted(high_corr_pairs, key=lambda x: -x[2]):
            print(f"  |r|={r:.4f}: {f1} <-> {f2}")
    else:
        print("  No pairs with |r| > 0.95")

    return {
        "importance": importance,
        "ranked": ranked,
        "zero_neg": zero_neg,
        "high_corr_pairs": high_corr_pairs,
        "baseline_mae": baseline_mae,
        "baseline_book_mae": baseline_book_mae,
        "n_book_games": n_book,
    }


# ══════════════════════════════════════════════════════════════════
# TASK 2: Ablation Study
# ══════════════════════════════════════════════════════════════════


def task2_ablation(train_df, holdout_df, task1_results):
    """Backward elimination + forward selection.

    Uses 50-epoch training for search steps (faster), then 100-epoch for
    final validation of the best sets.
    """
    print("\n" + "=" * 70)
    print("TASK 2: ABLATION STUDY")
    print("=" * 70)

    SEARCH_HP = {"epochs": 50}  # faster for search
    FULL_HP = {"epochs": 100}   # full training for final validation

    ranked = task1_results["ranked"]
    importance = task1_results["importance"]

    # Start: remove FORCE_REMOVE and zero/negative importance features
    current_features = [f for f in V2_FEATURES if f not in FORCE_REMOVE]
    print(f"\nStarting features: {len(current_features)} (removed {FORCE_REMOVE})")

    # ── Phase 1: Coarse backward elimination (remove bottom 5 at a time) ──
    print("\n--- Phase 1: Coarse Backward Elimination (remove bottom 5 at a time) ---")

    # Sort current_features by importance (least important last)
    def rank_key(f):
        return importance.get(f, {}).get("mean_mae_increase", 0)

    backward_log = []

    # Get initial MAE
    print(f"\nEvaluating starting set ({len(current_features)} features)...")
    result = quick_evaluate(current_features, train_df, holdout_df,
                            label=f"start_{len(current_features)}",
                            hparams=SEARCH_HP)
    current_mae = result["mae_book"]
    backward_log.append({
        "step": "start",
        "n_features": len(current_features),
        "mae_book": current_mae,
        "removed": [],
        "features": list(current_features),
    })
    print(f"  MAE (book): {current_mae:.4f}")

    step = 0
    while len(current_features) > 15:
        step += 1
        # Sort by importance, remove bottom 5
        sorted_feats = sorted(current_features, key=rank_key, reverse=True)
        to_remove = sorted_feats[-5:]
        candidate = [f for f in sorted_feats if f not in set(to_remove)]

        print(f"\n  Step {step}: removing {to_remove} -> {len(candidate)} features")
        result = quick_evaluate(candidate, train_df, holdout_df,
                                label=f"coarse_{len(candidate)}",
                                hparams=SEARCH_HP)
        new_mae = result["mae_book"]
        print(f"    MAE (book): {new_mae:.4f} (was {current_mae:.4f}, "
              f"delta={new_mae - current_mae:+.4f})")

        backward_log.append({
            "step": f"coarse_{step}",
            "n_features": len(candidate),
            "mae_book": new_mae,
            "removed": to_remove,
            "features": list(candidate),
        })

        current_features = candidate
        current_mae = new_mae

    # ── Phase 2: Fine backward elimination (remove 1 at a time) ──
    print("\n--- Phase 2: Fine Backward Elimination (remove 1 at a time) ---")
    print(f"Starting from {len(current_features)} features, MAE={current_mae:.4f}")

    best_mae = current_mae
    best_features = list(current_features)
    stale_count = 0

    while len(current_features) > 8 and stale_count < 3:
        best_candidate = None
        best_candidate_mae = best_mae + 999

        # Try removing each feature one at a time
        for feat in current_features:
            candidate = [f for f in current_features if f != feat]
            result = quick_evaluate(candidate, train_df, holdout_df,
                                    label=f"fine_{len(candidate)}",
                                    hparams=SEARCH_HP)
            mae = result["mae_book"]
            print(f"    Remove {feat:35s} -> MAE={mae:.4f} (delta={mae - current_mae:+.4f})")

            if mae < best_candidate_mae:
                best_candidate_mae = mae
                best_candidate = (feat, candidate)

        removed_feat, new_features = best_candidate
        print(f"\n  Best removal: {removed_feat} -> MAE={best_candidate_mae:.4f}")

        backward_log.append({
            "step": f"fine_remove_{removed_feat}",
            "n_features": len(new_features),
            "mae_book": best_candidate_mae,
            "removed": [removed_feat],
            "features": list(new_features),
        })

        # Accept if MAE didn't get worse by more than 0.08
        if best_candidate_mae <= current_mae + 0.08:
            current_features = new_features
            current_mae = best_candidate_mae
            if best_candidate_mae < best_mae:
                best_mae = best_candidate_mae
                best_features = list(new_features)
                stale_count = 0
            else:
                stale_count += 1
            print(f"  Accepted. Now at {len(current_features)} features, MAE={current_mae:.4f}")
        else:
            print(f"  Rejected (too much degradation). Stopping fine elimination.")
            break

    backward_result = {
        "features": best_features,
        "mae": best_mae,
        "n_features": len(best_features),
    }
    print(f"\nBackward elimination result: {len(best_features)} features, MAE={best_mae:.4f}")

    # ── Phase 3: Forward selection ──
    print("\n--- Phase 3: Forward Selection ---")

    # Start with top 10 features from permutation importance
    top_features = [name for name, _ in ranked[:10]]
    # Ensure we only use features that aren't in FORCE_REMOVE
    top_features = [f for f in top_features if f not in FORCE_REMOVE]

    # Only try adding features ranked 11-30 by permutation importance
    # (skip features below rank 30 for speed — they showed minimal importance)
    forward_candidates = [name for name, _ in ranked[10:30]
                          if name not in FORCE_REMOVE and name not in set(top_features)]
    remaining = list(forward_candidates)

    print(f"Starting with top {len(top_features)} features: {top_features}")
    print(f"Candidate pool for addition: {len(remaining)} features (ranks 11-30)")
    result = quick_evaluate(top_features, train_df, holdout_df,
                            label=f"forward_{len(top_features)}",
                            hparams=SEARCH_HP)
    current_mae_fwd = result["mae_book"]
    forward_log = [{
        "step": "start",
        "n_features": len(top_features),
        "mae_book": current_mae_fwd,
        "added": [],
        "features": list(top_features),
    }]
    print(f"  Start MAE (book): {current_mae_fwd:.4f}")

    stale_count = 0
    while remaining and stale_count < 3:
        best_add = None
        best_add_mae = current_mae_fwd + 999

        for feat in remaining:
            candidate = top_features + [feat]
            result = quick_evaluate(candidate, train_df, holdout_df,
                                    label=f"forward_{len(candidate)}",
                                    hparams=SEARCH_HP)
            mae = result["mae_book"]
            print(f"    Add {feat:35s} -> MAE={mae:.4f} (delta={mae - current_mae_fwd:+.4f})")

            if mae < best_add_mae:
                best_add_mae = mae
                best_add = feat

        print(f"\n  Best addition: {best_add} -> MAE={best_add_mae:.4f}")

        forward_log.append({
            "step": f"add_{best_add}",
            "n_features": len(top_features) + 1,
            "mae_book": best_add_mae,
            "added": [best_add],
            "features": list(top_features) + [best_add],
        })

        if best_add_mae < current_mae_fwd:
            top_features.append(best_add)
            remaining.remove(best_add)
            current_mae_fwd = best_add_mae
            stale_count = 0
            print(f"  Accepted. Now at {len(top_features)} features, MAE={current_mae_fwd:.4f}")
        else:
            stale_count += 1
            # Still add it to explore further, but track stale
            top_features.append(best_add)
            remaining.remove(best_add)
            current_mae_fwd = best_add_mae
            print(f"  No improvement (stale={stale_count}). Added anyway to explore.")

    # Track best forward result (before stale additions)
    best_fwd_mae = min(entry["mae_book"] for entry in forward_log if entry["mae_book"] is not None)
    best_fwd_entry = min(forward_log, key=lambda e: e["mae_book"] if e["mae_book"] is not None else 999)
    best_fwd_features = best_fwd_entry["features"]

    forward_result = {
        "features": list(best_fwd_features),
        "mae": best_fwd_mae,
        "n_features": len(best_fwd_features),
    }
    print(f"\nForward selection result: {len(best_fwd_features)} features, MAE={best_fwd_mae:.4f}")

    # ── Phase 4: Full validation of candidates ──
    print("\n--- Phase 4: Full Validation (100 epochs) ---")
    candidates = [
        ("backward", backward_result),
        ("forward", forward_result),
    ]

    for label, res in candidates:
        print(f"\n  Validating {label} ({res['n_features']} features)...")
        full_result = quick_evaluate(res["features"], train_df, holdout_df,
                                     label=f"{label}_full", hparams=FULL_HP)
        res["mae_full"] = full_result["mae_book"]
        print(f"    MAE (50ep): {res['mae']:.4f}, MAE (100ep): {res['mae_full']:.4f}")

    # ── Pick optimal feature set ──
    print("\n--- Choosing Optimal Feature Set ---")

    # Preference: parsimony. If MAEs within 0.05, prefer fewer features.
    for label, res in candidates:
        print(f"  {label}: {res['n_features']} features, "
              f"MAE={res['mae_full']:.4f}")

    # Sort by full-validated MAE
    candidates.sort(key=lambda x: (x[1]["mae_full"], x[1]["n_features"]))
    best_label, best_res = candidates[0]

    # Check if a smaller model is nearly as good
    for label, res in candidates:
        if res["n_features"] < best_res["n_features"] and res["mae_full"] < best_res["mae_full"] + 0.05:
            best_label = label
            best_res = res
            break

    best_res["mae"] = best_res["mae_full"]  # use full validation MAE
    print(f"\n  Selected: {best_label} ({best_res['n_features']} features, MAE={best_res['mae']:.4f})")

    return {
        "backward_log": backward_log,
        "forward_log": forward_log,
        "backward_result": backward_result,
        "forward_result": forward_result,
        "optimal_features": best_res["features"],
        "optimal_mae": best_res["mae"],
        "optimal_source": best_label,
    }


# ══════════════════════════════════════════════════════════════════
# TASK 3: Overfitting Diagnostics
# ══════════════════════════════════════════════════════════════════


def task3_overfitting(train_df, holdout_df, optimal_features):
    """Overfitting diagnostics with the optimal feature set."""
    print("\n" + "=" * 70)
    print("TASK 3: OVERFITTING DIAGNOSTICS")
    print("=" * 70)

    feature_cols = list(optimal_features)
    print(f"\nUsing {len(feature_cols)} features")

    # ── 3a: Train/val split ──
    print("\n--- 3a: Train/Validation Gap ---")
    np.random.seed(42)
    indices = np.random.permutation(len(train_df))
    split = int(0.8 * len(indices))
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_sub = train_df.iloc[train_idx].copy()
    val_sub = train_df.iloc[val_idx].copy()

    X_train_full = train_sub[feature_cols].values.astype(np.float32)
    X_train_full = np.nan_to_num(X_train_full, nan=0.0)
    targets_train = get_targets(train_sub)
    y_spread_train = targets_train["spread_home"].values.astype(np.float32)
    y_win_train = targets_train["home_win"].values.astype(np.float32)

    X_val = val_sub[feature_cols].values.astype(np.float32)
    X_val = np.nan_to_num(X_val, nan=0.0)
    targets_val = get_targets(val_sub)
    y_spread_val = targets_val["spread_home"].values.astype(np.float32)

    scaler = StandardScaler()
    scaler.fit(X_train_full)
    X_train_s = scaler.transform(X_train_full)
    X_val_s = scaler.transform(X_val)

    reg = train_regressor(X_train_s, y_spread_train, hparams={"epochs": 100})
    reg.eval()

    with torch.no_grad():
        mu_train, _ = reg(torch.tensor(X_train_s, dtype=torch.float32))
        mu_val, _ = reg(torch.tensor(X_val_s, dtype=torch.float32))

    train_mae = float(np.abs(mu_train.numpy() - y_spread_train).mean())
    val_mae = float(np.abs(mu_val.numpy() - y_spread_val).mean())
    gap = val_mae - train_mae

    print(f"  Train MAE: {train_mae:.4f}")
    print(f"  Val MAE: {val_mae:.4f}")
    print(f"  Gap: {gap:.4f}")

    # Also on holdout
    X_hold = holdout_df[feature_cols].values.astype(np.float32)
    X_hold = np.nan_to_num(X_hold, nan=0.0)
    X_hold_s = scaler.transform(X_hold)
    targets_hold = get_targets(holdout_df)
    y_spread_hold = targets_hold["spread_home"].values.astype(np.float32)

    with torch.no_grad():
        mu_hold, _ = reg(torch.tensor(X_hold_s, dtype=torch.float32))
    holdout_mae = float(np.abs(mu_hold.numpy() - y_spread_hold).mean())
    print(f"  Holdout MAE: {holdout_mae:.4f}")

    trainval_results = {
        "train_mae": train_mae,
        "val_mae": val_mae,
        "gap": gap,
        "holdout_mae": holdout_mae,
    }

    # ── 3b: Learning curves ──
    print("\n--- 3b: Learning Curves ---")
    fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    learning_curve = []

    for frac in fractions:
        n_samples = int(frac * len(X_train_full))
        X_lc = X_train_s[:n_samples]
        y_lc = y_spread_train[:n_samples]
        y_win_lc = y_win_train[:n_samples]

        reg_lc = train_regressor(X_lc, y_lc, hparams={"epochs": 100})
        reg_lc.eval()

        with torch.no_grad():
            mu_lc_train, _ = reg_lc(torch.tensor(X_lc, dtype=torch.float32))
            mu_lc_val, _ = reg_lc(torch.tensor(X_val_s, dtype=torch.float32))

        tr_mae = float(np.abs(mu_lc_train.numpy() - y_lc).mean())
        vl_mae = float(np.abs(mu_lc_val.numpy() - y_spread_val).mean())
        learning_curve.append({
            "fraction": frac,
            "n_samples": n_samples,
            "train_mae": tr_mae,
            "val_mae": vl_mae,
            "gap": vl_mae - tr_mae,
        })
        print(f"  {frac:.0%} ({n_samples:,} samples): train={tr_mae:.4f}, val={vl_mae:.4f}, "
              f"gap={vl_mae - tr_mae:.4f}")

    # ── 3c: Regularization experiments ──
    print("\n--- 3c: Regularization Experiments ---")
    reg_experiments = []

    configs = [
        {"dropout": 0.2, "weight_decay": 1e-4},
        {"dropout": 0.3, "weight_decay": 1e-4},  # default
        {"dropout": 0.4, "weight_decay": 1e-4},
        {"dropout": 0.5, "weight_decay": 1e-4},
        {"dropout": 0.3, "weight_decay": 1e-3},
        {"dropout": 0.3, "weight_decay": 1e-5},
        {"dropout": 0.4, "weight_decay": 1e-3},
    ]

    for cfg in configs:
        hp = {"epochs": 100, **cfg}
        result = quick_evaluate(feature_cols, train_df, holdout_df,
                                label=f"d{cfg['dropout']}_wd{cfg['weight_decay']}",
                                hparams=hp)
        reg_experiments.append({
            **cfg,
            "mae_overall": result["mae_overall"],
            "mae_book": result["mae_book"],
        })
        print(f"  dropout={cfg['dropout']}, wd={cfg['weight_decay']}: "
              f"MAE(book)={result['mae_book']:.4f}")

    # ── 3d: Test removing known-bad features ──
    print("\n--- 3d: Ablating Known-Weak Features ---")
    weak_features = ["away_team_home", "away_def_def_rebound_pct", "home_def_rebound_pct"]
    ablation_results = []

    for feat in weak_features:
        if feat not in feature_cols:
            print(f"  {feat}: already not in optimal set")
            ablation_results.append({"feature": feat, "in_set": False, "mae_book": None})
            continue
        candidate = [f for f in feature_cols if f != feat]
        result = quick_evaluate(candidate, train_df, holdout_df, label=f"without_{feat}")
        ablation_results.append({
            "feature": feat,
            "in_set": True,
            "mae_book": result["mae_book"],
        })
        print(f"  Without {feat:35s}: MAE(book)={result['mae_book']:.4f}")

    return {
        "trainval": trainval_results,
        "learning_curve": learning_curve,
        "reg_experiments": reg_experiments,
        "ablation_results": ablation_results,
    }


# ══════════════════════════════════════════════════════════════════
# TASK 4: Final Evaluation
# ══════════════════════════════════════════════════════════════════


def task4_final_evaluation(train_df, holdout_df, optimal_features, task1_results,
                           task2_results, task3_results):
    """Final evaluation with comparison table."""
    print("\n" + "=" * 70)
    print("TASK 4: FINAL EVALUATION")
    print("=" * 70)

    feature_cols = list(optimal_features)

    # Determine best regularization from task3
    best_reg = min(task3_results["reg_experiments"], key=lambda x: x["mae_book"] or 999)
    print(f"\nBest regularization: dropout={best_reg['dropout']}, wd={best_reg['weight_decay']}")
    print(f"  MAE(book): {best_reg['mae_book']:.4f}")

    # Final training with best regularization
    print(f"\nFinal training with {len(feature_cols)} features and best regularization...")
    hp = {
        "epochs": 100,
        "dropout": best_reg["dropout"],
        "weight_decay": best_reg["weight_decay"],
    }
    final_result = quick_evaluate(feature_cols, train_df, holdout_df,
                                  label="final", hparams=hp)
    preds = final_result["preds"]

    # Monthly MAE
    dates = pd.to_datetime(preds["startDate"], errors="coerce", utc=True)
    preds["month"] = dates.dt.tz_localize(None).dt.to_period("M")
    monthly = {}
    for month, group in preds.groupby("month"):
        m_mae = float(np.abs(group["predicted_spread"] - group["actual_margin"]).mean())
        monthly[str(month)] = (m_mae, len(group))

    # ROI at thresholds
    roi_results = {}
    for t in [3, 5, 7]:
        roi, n, w, l = compute_roi(preds, t)
        roi_results[f"t{t}"] = {"roi": roi, "n": n, "w": w, "l": l}

        median_sigma = preds["spread_sigma"].median()
        roi_s, ns, ws, ls = compute_roi(preds, t, sigma_filter=median_sigma)
        roi_results[f"t{t}_sigma"] = {"roi": roi_s, "n": ns, "w": ws, "l": ls,
                                       "sigma_cutoff": median_sigma}

    # Calibration: predicted sigma vs actual error
    with_book = preds.dropna(subset=["book_spread"])
    if len(with_book) > 0:
        errors = (with_book["predicted_spread"] - with_book["actual_margin"]).values
        sigmas = with_book["spread_sigma"].values
        # What fraction of errors fall within 1 sigma, 2 sigma?
        within_1 = float(np.mean(np.abs(errors) < sigmas))
        within_2 = float(np.mean(np.abs(errors) < 2 * sigmas))
        calibration = {"within_1sigma": within_1, "within_2sigma": within_2}
    else:
        calibration = {}

    print(f"\nFinal MAE (overall): {final_result['mae_overall']:.4f}")
    print(f"Final MAE (book): {final_result['mae_book']:.4f}")

    # ── Generate comparison table ──
    print("\n--- Comparison Table ---")
    print(f"{'Model':25s} {'Features':>8s} {'MAE(book)':>10s} {'vs 37-feat':>10s} {'vs 54-feat':>10s}")
    print("-" * 65)

    rows = [
        ("37-feature baseline", 37, BASELINE_37_MAE),
        ("54-feature expanded", 54, BASELINE_54_MAE),
        (f"Pruned ({task2_results['optimal_source']})",
         len(optimal_features), task2_results["optimal_mae"]),
        ("Final (tuned reg.)",
         len(optimal_features), final_result["mae_book"]),
    ]

    for name, n_feat, mae in rows:
        d37 = BASELINE_37_MAE - mae
        d54 = BASELINE_54_MAE - mae
        print(f"{name:25s} {n_feat:8d} {mae:10.4f} {d37:+10.4f} {d54:+10.4f}")

    # ── Save artifacts ──
    print("\n--- Saving Artifacts ---")

    # Save final feature order
    final_order = list(optimal_features)
    feature_order_path = config.ARTIFACTS_DIR / "feature_order.json"
    with open(feature_order_path, "w") as f:
        json.dump(final_order, f, indent=2)
    print(f"  Saved feature_order.json ({len(final_order)} features)")

    # Save model checkpoints
    save_checkpoint(final_result["reg_model"], "regressor",
                    hparams=hp, subdir="no_garbage",
                    feature_order=final_order)
    save_checkpoint(final_result["cls_model"], "classifier",
                    hparams=hp, subdir="no_garbage",
                    feature_order=final_order)

    # Save scaler
    scaler_path = config.ARTIFACTS_DIR / "no_garbage" / "scaler.pkl"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump(final_result["scaler"], f)
    print(f"  Saved scaler to {scaler_path}")

    return {
        "final_result": final_result,
        "monthly": monthly,
        "roi_results": roi_results,
        "calibration": calibration,
        "comparison_rows": rows,
        "best_hparams": hp,
        "feature_order": final_order,
    }


# ══════════════════════════════════════════════════════════════════
# Report Generation
# ══════════════════════════════════════════════════════════════════


def generate_report(task1_results, task2_results, task3_results, task4_results):
    """Generate markdown report."""
    lines = []
    lines.append("# Feature Selection / Pruning — Season 2025\n")

    # ── Task 1 ──
    lines.append("## Task 1: Permutation Importance (54 features)\n")
    lines.append(f"Baseline MAE (book-spread games): **{task1_results['baseline_book_mae']:.4f}** "
                 f"({task1_results['n_book_games']} games)\n")

    lines.append("| Rank | Feature | MAE Increase | Std |")
    lines.append("|------|---------|-------------|-----|")
    for rank, (name, info) in enumerate(task1_results["ranked"], 1):
        mi = info["mean_mae_increase"]
        lines.append(f"| {rank} | {name} | {mi:+.4f} | {info['std_mae_increase']:.4f} |")

    lines.append(f"\n**Zero/negative importance**: {task1_results['zero_neg']}\n")
    lines.append(f"**Force remove**: {list(FORCE_REMOVE)}\n")

    if task1_results["high_corr_pairs"]:
        lines.append("### Multicollinearity (|r| > 0.95)\n")
        lines.append("| Feature 1 | Feature 2 | |r| |")
        lines.append("|-----------|-----------|-----|")
        for f1, f2, r in task1_results["high_corr_pairs"]:
            lines.append(f"| {f1} | {f2} | {r:.4f} |")
        lines.append("")
    else:
        lines.append("### Multicollinearity: No pairs with |r| > 0.95\n")

    # ── Task 2 ──
    lines.append("## Task 2: Ablation Study\n")

    lines.append("### Backward Elimination Log\n")
    lines.append("| Step | Features | MAE (book) | Removed |")
    lines.append("|------|----------|-----------|---------|")
    for entry in task2_results["backward_log"]:
        removed_str = ", ".join(entry["removed"]) if entry["removed"] else "-"
        mae_str = f"{entry['mae_book']:.4f}" if entry["mae_book"] else "N/A"
        lines.append(f"| {entry['step']} | {entry['n_features']} | {mae_str} | {removed_str} |")

    lines.append(f"\nBackward result: **{task2_results['backward_result']['n_features']} features**, "
                 f"MAE={task2_results['backward_result']['mae']:.4f}\n")

    lines.append("### Forward Selection Log\n")
    lines.append("| Step | Features | MAE (book) | Added |")
    lines.append("|------|----------|-----------|-------|")
    for entry in task2_results["forward_log"]:
        added_str = ", ".join(entry["added"]) if entry["added"] else "-"
        mae_str = f"{entry['mae_book']:.4f}" if entry["mae_book"] else "N/A"
        lines.append(f"| {entry['step']} | {entry['n_features']} | {mae_str} | {added_str} |")

    lines.append(f"\nForward result: **{task2_results['forward_result']['n_features']} features**, "
                 f"MAE={task2_results['forward_result']['mae']:.4f}\n")

    lines.append(f"### Selected: {task2_results['optimal_source']}\n")
    lines.append(f"- Features: {task2_results['optimal_features']}")
    lines.append(f"- Count: {len(task2_results['optimal_features'])}")
    lines.append(f"- MAE: {task2_results['optimal_mae']:.4f}\n")

    # ── Task 3 ──
    lines.append("## Task 3: Overfitting Diagnostics\n")

    tv = task3_results["trainval"]
    lines.append("### Train/Val/Holdout Gap\n")
    lines.append(f"| Set | MAE |")
    lines.append(f"|-----|-----|")
    lines.append(f"| Train (80%) | {tv['train_mae']:.4f} |")
    lines.append(f"| Val (20%) | {tv['val_mae']:.4f} |")
    lines.append(f"| Holdout (2025) | {tv['holdout_mae']:.4f} |")
    lines.append(f"| Train-Val Gap | {tv['gap']:.4f} |\n")

    lines.append("### Learning Curves\n")
    lines.append("| Data % | Samples | Train MAE | Val MAE | Gap |")
    lines.append("|--------|---------|-----------|---------|-----|")
    for lc in task3_results["learning_curve"]:
        lines.append(f"| {lc['fraction']:.0%} | {lc['n_samples']:,} | "
                     f"{lc['train_mae']:.4f} | {lc['val_mae']:.4f} | {lc['gap']:.4f} |")

    lines.append("\n### Regularization Experiments\n")
    lines.append("| Dropout | Weight Decay | MAE (overall) | MAE (book) |")
    lines.append("|---------|-------------|--------------|-----------|")
    for exp in task3_results["reg_experiments"]:
        book_str = f"{exp['mae_book']:.4f}" if exp["mae_book"] else "N/A"
        lines.append(f"| {exp['dropout']} | {exp['weight_decay']} | "
                     f"{exp['mae_overall']:.4f} | {book_str} |")

    lines.append("\n### Known-Weak Feature Ablation\n")
    for ab in task3_results["ablation_results"]:
        if not ab["in_set"]:
            lines.append(f"- **{ab['feature']}**: already not in optimal set")
        else:
            lines.append(f"- Without **{ab['feature']}**: MAE(book)={ab['mae_book']:.4f}")
    lines.append("")

    # ── Task 4 ──
    lines.append("## Task 4: Final Evaluation\n")

    lines.append(f"Best hparams: dropout={task4_results['best_hparams']['dropout']}, "
                 f"weight_decay={task4_results['best_hparams']['weight_decay']}\n")

    lines.append("### Comparison Table\n")
    lines.append("| Model | Features | MAE (book) | vs 37-feat | vs 54-feat |")
    lines.append("|-------|----------|-----------|-----------|-----------|")
    for name, n_feat, mae in task4_results["comparison_rows"]:
        d37 = BASELINE_37_MAE - mae
        d54 = BASELINE_54_MAE - mae
        lines.append(f"| {name} | {n_feat} | {mae:.4f} | {d37:+.4f} | {d54:+.4f} |")

    lines.append("\n### Monthly MAE\n")
    lines.append("| Month | MAE | Games |")
    lines.append("|-------|-----|-------|")
    for month in sorted(task4_results["monthly"].keys()):
        m_mae, n = task4_results["monthly"][month]
        lines.append(f"| {month} | {m_mae:.2f} | {n} |")

    lines.append("\n### ATS ROI\n")
    lines.append("#### Unfiltered\n")
    lines.append("| Threshold | Bets | Wins | Losses | Win Rate | ROI |")
    lines.append("|-----------|------|------|--------|----------|-----|")
    for t in [3, 5, 7]:
        r = task4_results["roi_results"][f"t{t}"]
        if r["roi"] is not None:
            wr = r["w"] / r["n"] if r["n"] > 0 else 0
            lines.append(f"| {t} | {r['n']} | {r['w']} | {r['l']} | {wr:.1%} | {r['roi']:+.1f}% |")

    # Sigma-filtered
    for t in [3, 5, 7]:
        r = task4_results["roi_results"].get(f"t{t}_sigma", {})
        if r and r.get("roi") is not None:
            sigma_cut = r["sigma_cutoff"]
            break
    else:
        sigma_cut = None

    if sigma_cut:
        lines.append(f"\n#### Sigma < {sigma_cut:.1f}\n")
        lines.append("| Threshold | Bets | Wins | Losses | Win Rate | ROI |")
        lines.append("|-----------|------|------|--------|----------|-----|")
        for t in [3, 5, 7]:
            r = task4_results["roi_results"][f"t{t}_sigma"]
            if r["roi"] is not None:
                wr = r["w"] / r["n"] if r["n"] > 0 else 0
                lines.append(f"| {t} | {r['n']} | {r['w']} | {r['l']} | "
                              f"{wr:.1%} | {r['roi']:+.1f}% |")

    if task4_results["calibration"]:
        cal = task4_results["calibration"]
        lines.append(f"\n### Calibration\n")
        lines.append(f"- Within 1 sigma: {cal['within_1sigma']:.1%} (expected ~68%)")
        lines.append(f"- Within 2 sigma: {cal['within_2sigma']:.1%} (expected ~95%)")

    lines.append(f"\n## Final Feature Order ({len(task4_results['feature_order'])} features)\n")
    lines.append("```json")
    lines.append(json.dumps(task4_results["feature_order"], indent=2))
    lines.append("```\n")

    lines.append("## Action Taken\n")
    lines.append(f"- Backed up 54-feature order as `feature_order_v2.json`")
    lines.append(f"- Updated `feature_order.json` ({len(task4_results['feature_order'])} features)")
    lines.append(f"- Retrained models in `checkpoints/no_garbage/`")
    lines.append(f"- Saved scaler to `artifacts/no_garbage/scaler.pkl`")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))
    print(f"\nReport saved to: {REPORT_PATH}")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=0,
                        help="Run specific task (1-4) or 0 for all")
    args = parser.parse_args()

    train_df, holdout_df = load_all_data()

    # State files for incremental runs
    state_dir = PROJECT_ROOT / ".feature_selection_state"
    state_dir.mkdir(exist_ok=True)

    def save_state(name, data):
        with open(state_dir / f"{name}.json", "w") as f:
            json.dump(data, f, default=str)

    def load_state(name):
        path = state_dir / f"{name}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    task1_results = None
    task2_results = None
    task3_results = None
    task4_results = None

    if args.task in (0, 1):
        task1_results = task1_permutation_importance(train_df, holdout_df, n_repeats=15)
        # Save serializable parts of task1 results
        save_state("task1", {
            "importance": task1_results["importance"],
            "zero_neg": task1_results["zero_neg"],
            "high_corr_pairs": [(f1, f2, float(r)) for f1, f2, r in task1_results["high_corr_pairs"]],
            "baseline_mae": task1_results["baseline_mae"],
            "baseline_book_mae": task1_results["baseline_book_mae"],
            "n_book_games": task1_results["n_book_games"],
            "ranked_names": [name for name, _ in task1_results["ranked"]],
        })

    if args.task in (0, 2):
        if task1_results is None:
            # Load from state
            state = load_state("task1")
            if state is None:
                print("ERROR: Must run task 1 first")
                return
            # Reconstruct ranked list
            ranked = [(name, state["importance"][name]) for name in state["ranked_names"]]
            task1_results = {
                "ranked": ranked,
                "importance": state["importance"],
                "zero_neg": state["zero_neg"],
                "high_corr_pairs": [(f1, f2, r) for f1, f2, r in state["high_corr_pairs"]],
                "baseline_mae": state["baseline_mae"],
                "baseline_book_mae": state["baseline_book_mae"],
                "n_book_games": state["n_book_games"],
            }

        task2_results = task2_ablation(train_df, holdout_df, task1_results)
        save_state("task2", {
            "optimal_features": task2_results["optimal_features"],
            "optimal_mae": task2_results["optimal_mae"],
            "optimal_source": task2_results["optimal_source"],
            "backward_log": task2_results["backward_log"],
            "forward_log": task2_results["forward_log"],
            "backward_result": task2_results["backward_result"],
            "forward_result": task2_results["forward_result"],
        })

    if args.task in (0, 3):
        if task2_results is None:
            state = load_state("task2")
            if state is None:
                print("ERROR: Must run task 2 first")
                return
            task2_results = state

        optimal_features = task2_results["optimal_features"]
        task3_results = task3_overfitting(train_df, holdout_df, optimal_features)
        save_state("task3", task3_results)

    if args.task in (0, 4):
        if task1_results is None:
            state = load_state("task1")
            if state:
                ranked = [(name, state["importance"][name]) for name in state["ranked_names"]]
                task1_results = {
                    "ranked": ranked, "importance": state["importance"],
                    "zero_neg": state["zero_neg"],
                    "high_corr_pairs": [(f1, f2, r) for f1, f2, r in state["high_corr_pairs"]],
                    "baseline_mae": state["baseline_mae"],
                    "baseline_book_mae": state["baseline_book_mae"],
                    "n_book_games": state["n_book_games"],
                }
        if task2_results is None:
            task2_results = load_state("task2")
        if task3_results is None:
            task3_results = load_state("task3")

        if not all([task1_results, task2_results, task3_results]):
            print("ERROR: Must run tasks 1-3 first")
            return

        optimal_features = task2_results["optimal_features"]
        task4_results = task4_final_evaluation(
            train_df, holdout_df, optimal_features,
            task1_results, task2_results, task3_results)

        generate_report(task1_results, task2_results, task3_results, task4_results)

    print("\nDone!")


if __name__ == "__main__":
    main()
