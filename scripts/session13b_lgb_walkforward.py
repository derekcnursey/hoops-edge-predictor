#!/usr/bin/env python3
"""Session 13b: Pure LGB walk-forward with NN sigma for away-dog strategy.

For each test_year in [2019-2025]:
  - Train LGB with Optuna-winning hyperparams on prior years
  - Train C2-V2 NN on prior years (for sigma)
  - Predict test_year
  - Print BS-MAE, ROI@12% unfiltered, away-dog strategy ROI

Then print side-by-side vs C2-V2 walk-forward results.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.architecture import MLPRegressor, gaussian_nll_loss
from src.dataset import HoopsDataset, load_multi_season_features
from src.features import get_feature_matrix, get_targets, load_research_lines
from src.trainer import impute_column_means

ADJ_SUFFIX = f"adj_a{config.ADJUST_ALPHA}_p{config.ADJUST_PRIOR}"
REPORTS_DIR = config.PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# Optuna-winning LGB hyperparams from session 13b
LGB_PARAMS = {
    "objective": "regression_l1",
    "metric": "mae",
    "verbosity": -1,
    "n_jobs": -1,
    "seed": 42,
    "feature_pre_filter": False,
    "learning_rate": 0.02019353198222356,
    "num_leaves": 110,
    "max_depth": 9,
    "min_child_samples": 60,
    "subsample": 0.6469253206264255,
    "colsample_bytree": 0.4684413154048659,
    "reg_alpha": 1.561017503434852,
    "reg_lambda": 0.004257796528177626,
}

# C2-V2 NN hyperparams
NN_HP = {
    "hidden1": 384, "hidden2": 256, "dropout": 0.20,
    "lr": 3e-3, "batch_size": 4096, "weight_decay": 1e-4,
}

# C2-V2 walk-forward results from session 13 validation suite (Test 1)
C2V2_WF = {
    2019: {"bs_mae": 9.023, "roi_12_unfilt": 3.1},
    2020: {"bs_mae": 9.305, "roi_12_unfilt": 3.5},
    2021: {"bs_mae": 10.116, "roi_12_unfilt": -4.5},
    2022: {"bs_mae": 8.890, "roi_12_unfilt": 2.8},
    2023: {"bs_mae": 9.086, "roi_12_unfilt": -0.3},
    2024: {"bs_mae": 9.165, "roi_12_unfilt": 2.9},
    2025: {"bs_mae": 8.977, "roi_12_unfilt": None},  # will fill from log
}


def normal_cdf(x):
    return norm.cdf(x)


def train_nn(X_train_s, y_train, X_val_s, y_val):
    """Train C2-V2 architecture NN. Returns model."""
    hp = NN_HP
    model = MLPRegressor(
        input_dim=X_train_s.shape[1],
        hidden1=hp["hidden1"], hidden2=hp["hidden2"],
        dropout=hp["dropout"])

    optimizer = torch.optim.Adam(
        model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

    # Build datasets
    home_win_dummy = (y_train > 0).astype(np.float32)
    train_ds = HoopsDataset(X_train_s, y_train, home_win_dummy)
    loader = DataLoader(train_ds, batch_size=hp["batch_size"],
                        shuffle=True, drop_last=True)

    scaler = torch.amp.GradScaler()
    best_loss = float("inf")
    best_state = None
    patience = 0

    model.train()
    for ep in range(1, 201):
        for xb, yb, _ in loader:
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cpu"):
                mu, log_sigma = model(xb)
                nll, _ = gaussian_nll_loss(mu.squeeze(), log_sigma.squeeze(), yb)
                loss = nll.mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Validate
        model.eval()
        with torch.no_grad():
            X_vt = torch.tensor(X_val_s, dtype=torch.float32)
            mu_v, ls_v = model(X_vt)
            nll_v, _ = gaussian_nll_loss(
                mu_v.squeeze(), ls_v.squeeze(),
                torch.tensor(y_val, dtype=torch.float32))
            val_loss = nll_v.mean().item()
        model.train()

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 50:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def predict_nn(model, X_val_s):
    X_t = torch.tensor(X_val_s, dtype=torch.float32)
    with torch.no_grad():
        mu_t, ls_t = model(X_t)
    sigma_t = torch.exp(ls_t).clamp(min=0.5, max=30.0)
    return mu_t.numpy().ravel(), sigma_t.numpy().ravel()


def compute_roi_unfiltered(mu, sigma, book, actual, threshold):
    """ROI for prob_edge >= threshold, all games with book lines."""
    valid = ~np.isnan(book)
    if valid.sum() == 0:
        return {"bets": 0, "roi": 0.0, "units": 0.0, "wins": 0}

    edge_home = mu[valid] + book[valid]
    sigma_safe = np.clip(sigma[valid], 0.5, None)
    edge_z = edge_home / sigma_safe
    hcp = normal_cdf(edge_z)
    pick_home = edge_home >= 0
    pick_prob = np.where(pick_home, hcp, 1.0 - hcp)
    prob_edge = pick_prob - 0.5238

    bet_mask = prob_edge >= threshold
    n = int(bet_mask.sum())
    if n == 0:
        return {"bets": 0, "roi": 0.0, "units": 0.0, "wins": 0}

    actual_v = actual[valid]
    book_v = book[valid]
    home_covered = (actual_v + book_v) > 0
    pick_won = np.where(pick_home, home_covered, ~home_covered)
    w = int(pick_won[bet_mask].sum())
    profit_per_1 = 100.0 / 110.0
    units = w * profit_per_1 - (n - w)
    return {"bets": n, "roi": float(units / n), "units": float(units), "wins": w}


def compute_roi_away_dog(mu, sigma, book, actual, threshold=0.10, min_spread=10.0):
    """ROI for away dog strategy: prob_edge >= threshold, pick_side=away, |book| > min_spread."""
    valid = ~np.isnan(book)
    if valid.sum() == 0:
        return {"bets": 0, "roi": 0.0, "units": 0.0, "wins": 0}

    edge_home = mu[valid] + book[valid]
    sigma_safe = np.clip(sigma[valid], 0.5, None)
    edge_z = edge_home / sigma_safe
    hcp = normal_cdf(edge_z)
    pick_home = edge_home >= 0
    pick_prob = np.where(pick_home, hcp, 1.0 - hcp)
    prob_edge = pick_prob - 0.5238

    # Away pick = NOT pick_home
    pick_away = ~pick_home
    # |book_spread| > min_spread
    big_spread = np.abs(book[valid]) > min_spread

    bet_mask = (prob_edge >= threshold) & pick_away & big_spread
    n = int(bet_mask.sum())
    if n == 0:
        return {"bets": 0, "roi": 0.0, "units": 0.0, "wins": 0}

    actual_v = actual[valid]
    book_v = book[valid]
    home_covered = (actual_v + book_v) > 0
    pick_won = np.where(pick_home, home_covered, ~home_covered)
    w = int(pick_won[bet_mask].sum())
    profit_per_1 = 100.0 / 110.0
    units = w * profit_per_1 - (n - w)
    return {"bets": n, "roi": float(units / n), "units": float(units), "wins": w}


def main():
    print("=" * 70)
    print("  LGB PURE WALK-FORWARD + NN SIGMA")
    print("  LGB Optuna params | NN C2-V2 sigma | Away-dog strategy")
    print("=" * 70)

    test_years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
    results = []
    all_lgb_mu, all_nn_sigma, all_actual, all_book = [], [], [], []

    for ty in test_years:
        train_seasons = list(range(2015, ty))
        print(f"\n  --- {ty}: train on {train_seasons[0]}-{train_seasons[-1]} ---")
        t0 = time.time()

        # Load data
        df_tr = load_multi_season_features(
            train_seasons, adj_suffix=ADJ_SUFFIX, min_month_day="12-01")
        df_tr = df_tr.dropna(subset=["homeScore", "awayScore"])
        df_tr = df_tr[(df_tr["homeScore"] != 0) | (df_tr["awayScore"] != 0)]

        df_v = load_multi_season_features(
            [ty], adj_suffix=ADJ_SUFFIX, min_month_day="12-01")
        df_v = df_v.dropna(subset=["homeScore", "awayScore"])
        df_v = df_v[(df_v["homeScore"] != 0) | (df_v["awayScore"] != 0)]

        # Load book lines
        try:
            lines_df = load_research_lines(ty)
            if not lines_df.empty:
                ld = lines_df.sort_values("provider").drop_duplicates(
                    subset=["gameId"], keep="first")
                if "spread" in ld.columns:
                    df_v = df_v.merge(
                        ld[["gameId", "spread"]].rename(
                            columns={"spread": "bookSpread"}),
                        on="gameId", how="left")
        except Exception:
            pass

        # Features
        X_tr_raw = get_feature_matrix(df_tr).values.astype(np.float32)
        y_tr = get_targets(df_tr)["spread_home"].values.astype(np.float32)
        X_v_raw = get_feature_matrix(df_v).values.astype(np.float32)
        y_v = get_targets(df_v)["spread_home"].values.astype(np.float32)
        X_tr_raw = impute_column_means(X_tr_raw)
        X_v_raw = impute_column_means(X_v_raw)

        book = (df_v["bookSpread"].values.astype(np.float64)
                if "bookSpread" in df_v.columns
                else np.full(len(df_v), np.nan))
        has_book = ~np.isnan(book)

        # ── Train LGB ──
        lgb_tr = lgb.Dataset(X_tr_raw, y_tr,
                              params={"feature_pre_filter": False},
                              free_raw_data=False)
        lgb_vd = lgb.Dataset(X_v_raw, y_v, reference=lgb_tr,
                              free_raw_data=False)
        cb = [lgb.early_stopping(50), lgb.log_evaluation(0)]
        m_lgb = lgb.train(LGB_PARAMS, lgb_tr, num_boost_round=5000,
                          valid_sets=[lgb_vd], callbacks=cb)
        mu_lgb = m_lgb.predict(X_v_raw)

        # ── Train NN (for sigma) ──
        scaler = StandardScaler()
        scaler.fit(X_tr_raw)
        X_tr_s = scaler.transform(X_tr_raw).astype(np.float32)
        X_v_s = scaler.transform(X_v_raw).astype(np.float32)
        m_nn = train_nn(X_tr_s, y_tr, X_v_s, y_v)
        mu_nn, sigma_nn = predict_nn(m_nn, X_v_s)
        del m_nn
        torch.cuda.empty_cache()

        # ── Metrics ──
        bs_mae_lgb = float(np.mean(np.abs(y_v[has_book] - mu_lgb[has_book])))
        bs_mae_nn = float(np.mean(np.abs(y_v[has_book] - mu_nn[has_book])))

        # ROI: 12% unfiltered using LGB mu + NN sigma
        roi_12 = compute_roi_unfiltered(mu_lgb, sigma_nn, book, y_v, 0.12)

        # ROI: away dog strategy (10% edge, away pick, |book| > 10)
        roi_away = compute_roi_away_dog(mu_lgb, sigma_nn, book, y_v,
                                         threshold=0.10, min_spread=10.0)

        elapsed = time.time() - t0

        results.append({
            "year": ty,
            "bs_mae_lgb": bs_mae_lgb,
            "bs_mae_nn": bs_mae_nn,
            "roi_12": roi_12,
            "roi_away": roi_away,
            "games": int(has_book.sum()),
        })

        all_lgb_mu.append(mu_lgb[has_book])
        all_nn_sigma.append(sigma_nn[has_book])
        all_actual.append(y_v[has_book])
        all_book.append(book[has_book])

        print(f"    LGB BS-MAE={bs_mae_lgb:.3f}  NN BS-MAE={bs_mae_nn:.3f}")
        print(f"    ROI@12% unfilt: {roi_12['roi']*100:+.1f}% ({roi_12['bets']}b)")
        print(f"    Away dog 10%/>10: {roi_away['roi']*100:+.1f}% "
              f"({roi_away['bets']}b, {roi_away['wins']}W)")
        print(f"    [{elapsed:.0f}s]")

    # ── Pooled ──
    p_mu = np.concatenate(all_lgb_mu)
    p_sigma = np.concatenate(all_nn_sigma)
    p_actual = np.concatenate(all_actual)
    p_book = np.concatenate(all_book)
    pooled_lgb_mae = float(np.mean(np.abs(p_actual - p_mu)))
    pooled_roi_12 = compute_roi_unfiltered(
        p_mu, p_sigma, -p_book, p_actual, 0.12)  # note: book sign
    # Recompute pooled ROI properly with full arrays
    # Actually let's just sum the units and bets
    total_bets_12 = sum(r["roi_12"]["bets"] for r in results)
    total_units_12 = sum(r["roi_12"]["units"] for r in results)
    pooled_roi_12_val = total_units_12 / total_bets_12 if total_bets_12 > 0 else 0
    total_bets_away = sum(r["roi_away"]["bets"] for r in results)
    total_units_away = sum(r["roi_away"]["units"] for r in results)
    total_wins_away = sum(r["roi_away"]["wins"] for r in results)
    pooled_roi_away_val = total_units_away / total_bets_away if total_bets_away > 0 else 0

    # ══════════════════════════════════════════════════════════════════
    # SIDE-BY-SIDE TABLE
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  SIDE-BY-SIDE: C2-V2 (NN) vs LGB+NNσ WALK-FORWARD")
    print("=" * 70)

    print(f"\n  {'Year':>6} │ {'C2-V2 MAE':>9} {'C2-V2 ROI':>10} │ "
          f"{'LGB MAE':>8} {'LGB ROI':>9} │ {'ΔMAE':>6}")
    print(f"  {'─'*6} │ {'─'*9} {'─'*10} │ {'─'*8} {'─'*9} │ {'─'*6}")

    c2v2_pooled_mae = 9.202  # from session 13 validation
    for r in results:
        ty = r["year"]
        c2 = C2V2_WF.get(ty, {})
        c2_mae = c2.get("bs_mae", 0)
        c2_roi = c2.get("roi_12_unfilt")
        c2_roi_s = f"{c2_roi:+.1f}%" if c2_roi is not None else "  N/A"
        lgb_roi_s = f"{r['roi_12']['roi']*100:+.1f}%"
        delta = r["bs_mae_lgb"] - c2_mae
        print(f"  {ty:>6} │ {c2_mae:>9.3f} {c2_roi_s:>10} │ "
              f"{r['bs_mae_lgb']:>8.3f} {lgb_roi_s:>9} │ {delta:>+6.3f}")

    delta_pooled = pooled_lgb_mae - c2v2_pooled_mae
    print(f"  {'POOLED':>6} │ {c2v2_pooled_mae:>9.3f} {'':>10} │ "
          f"{pooled_lgb_mae:>8.3f} {pooled_roi_12_val*100:>+8.1f}% │ {delta_pooled:>+6.3f}")

    # ══════════════════════════════════════════════════════════════════
    # AWAY DOG STRATEGY TABLE
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  AWAY DOG STRATEGY: edge>=10%, away pick, |book|>10")
    print("  (using LGB mu + NN sigma)")
    print("=" * 70)

    print(f"\n  {'Year':>6} {'Bets':>6} {'W':>5} {'Win%':>6} {'ROI':>8} {'Units':>8}")
    print(f"  {'─'*6} {'─'*6} {'─'*5} {'─'*6} {'─'*8} {'─'*8}")
    pos_years = 0
    for r in results:
        ra = r["roi_away"]
        winp = ra["wins"] / ra["bets"] * 100 if ra["bets"] > 0 else 0
        print(f"  {r['year']:>6} {ra['bets']:>6} {ra['wins']:>5} "
              f"{winp:>5.1f}% {ra['roi']*100:>+7.1f}% {ra['units']:>+8.1f}")
        if ra["roi"] > 0:
            pos_years += 1

    winp_pool = total_wins_away / total_bets_away * 100 if total_bets_away > 0 else 0
    print(f"  {'POOLED':>6} {total_bets_away:>6} {total_wins_away:>5} "
          f"{winp_pool:>5.1f}% {pooled_roi_away_val*100:>+7.1f}% {total_units_away:>+8.1f}")
    print(f"  Positive years: {pos_years}/7")

    # ══════════════════════════════════════════════════════════════════
    # PRODUCTION DECISION
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  PRODUCTION DECISION")
    print("=" * 70)

    ship_lgb = pooled_lgb_mae < c2v2_pooled_mae

    print(f"\n  LGB pooled BS-MAE:  {pooled_lgb_mae:.3f}")
    print(f"  C2-V2 pooled BS-MAE: {c2v2_pooled_mae:.3f}")
    print(f"  Delta: {delta_pooled:+.3f}")

    if ship_lgb:
        print(f"\n  >>> DECISION: SHIP LGB + NN sigma hybrid <<<")
        print(f"  LGB wins walk-forward by {abs(delta_pooled):.3f} MAE")
        print(f"  Production setup:")
        print(f"    - LGB model for mu (point predictions)")
        print(f"    - C2-V2 NN model for sigma (uncertainty)")
        print(f"    - Both models required at inference time")

        # Save LGB params as production config
        prod_config = {
            "production_model": "lgb_hybrid",
            "lgb_params": {k: v for k, v in LGB_PARAMS.items()
                           if k not in ("objective", "metric", "verbosity",
                                        "n_jobs", "seed", "feature_pre_filter")},
            "nn_model": "C2-V2 (384→256, d=0.20, lr=3e-3)",
            "nn_role": "sigma only",
            "walkforward_pooled_mae": pooled_lgb_mae,
            "c2v2_pooled_mae": c2v2_pooled_mae,
            "delta": delta_pooled,
            "away_dog_pooled_roi": pooled_roi_away_val,
            "away_dog_positive_years": f"{pos_years}/7",
        }
        prod_path = config.PROJECT_ROOT / "artifacts" / "production_config.json"
        prod_path.write_text(json.dumps(prod_config, indent=2))
        print(f"  Saved production config to {prod_path}")

    else:
        print(f"\n  >>> DECISION: SHIP C2-V2 (unchanged) <<<")
        print(f"  C2-V2 wins walk-forward. LGB val-set advantage was in-sample.")
        print(f"  No changes to production artifacts needed.")

        prod_config = {
            "production_model": "c2v2_nn",
            "architecture": "384→256, d=0.20, lr=3e-3, Gaussian NLL",
            "walkforward_pooled_mae": c2v2_pooled_mae,
            "lgb_pooled_mae": pooled_lgb_mae,
            "lgb_val_mae": 9.080,
            "delta": delta_pooled,
            "note": "LGB beats on val (9.080 vs 9.129) but loses walk-forward",
        }
        prod_path = config.PROJECT_ROOT / "artifacts" / "production_config.json"
        prod_path.write_text(json.dumps(prod_config, indent=2))
        print(f"  Saved production config to {prod_path}")

    # Save full walk-forward report
    report_lines = [
        "# Session 13b: LGB Walk-Forward Results",
        f"\nDate: 2026-02-28",
        f"\n## Walk-Forward: LGB (Optuna-tuned) + NN sigma",
        f"\n| Year | C2-V2 MAE | LGB MAE | Delta | LGB ROI@12% | Away Dog ROI |",
        f"|------|-----------|---------|-------|-------------|-------------|",
    ]
    for r in results:
        ty = r["year"]
        c2_mae = C2V2_WF.get(ty, {}).get("bs_mae", 0)
        delta = r["bs_mae_lgb"] - c2_mae
        report_lines.append(
            f"| {ty} | {c2_mae:.3f} | {r['bs_mae_lgb']:.3f} | {delta:+.3f} | "
            f"{r['roi_12']['roi']*100:+.1f}% | {r['roi_away']['roi']*100:+.1f}% |")
    report_lines.append(
        f"| **Pooled** | **{c2v2_pooled_mae:.3f}** | **{pooled_lgb_mae:.3f}** | "
        f"**{delta_pooled:+.3f}** | **{pooled_roi_12_val*100:+.1f}%** | "
        f"**{pooled_roi_away_val*100:+.1f}%** |")
    report_lines.append(f"\n## Decision: {'LGB hybrid' if ship_lgb else 'C2-V2 unchanged'}")

    report_path = REPORTS_DIR / "session13b_lgb_walkforward.md"
    report_path.write_text("\n".join(report_lines))
    print(f"\n  Saved report to {report_path}")

    print(f"\n{'='*70}")
    print(f"  DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
