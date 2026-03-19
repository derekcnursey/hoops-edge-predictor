#!/usr/bin/env python3
"""Deep analysis of where the model's betting edge leaks.

Runs 53-feat no-garbage walk-forward (Config C), captures per-game predictions,
then analyzes edge picks across multiple dimensions:
  Part 1: Where do edge picks fail?
  Part 2: What does the market know that we don't?
  Part 3: Sigma calibration check
  Part 4: Filter-based betting strategies

Writes full report to /home/claude/edge_leak_analysis.md
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.architecture import MLPRegressor, gaussian_nll_loss
from src.dataset import HoopsDataset, load_multi_season_features
from src.features import get_feature_matrix, get_targets, load_research_lines
from src.trainer import impute_column_means
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# ── Config ────────────────────────────────────────────────────────────
ADJ_SUFFIX = f"adj_a{config.ADJUST_ALPHA}_p{config.ADJUST_PRIOR}"
HOLDOUT_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
TRAIN_START = 2015
MIN_DATE = "12-01"
MAX_EPOCHS = 500
PATIENCE = 50
EDGE_THRESHOLD = 0.05  # 5% minimum edge for picks

HP = {
    "hidden1": 384, "hidden2": 256, "dropout": 0.20,
    "lr": 3e-3, "batch_size": 4096, "weight_decay": 1e-4,
}

FEATURE_ORDER = config.FEATURE_ORDER  # 53 features

REPORT_PATH = Path(__file__).resolve().parent.parent / "analysis" / "edge_leak_analysis.md"


# ── Utilities (shared with hca_walkforward_comparison.py) ─────────────

def normal_cdf(z):
    z = np.asarray(z, dtype=float)
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(z / math.sqrt(2.0)))


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def _val_loss(model, X_val_t, y_val_t, device):
    model.eval()
    total, n = 0.0, 0
    for s in range(0, len(X_val_t), 4096):
        e = min(s + 4096, len(X_val_t))
        x = X_val_t[s:e].to(device)
        y = y_val_t[s:e].to(device)
        mu, ls = model(x)
        nll, _ = gaussian_nll_loss(mu, ls, y)
        total += nll.mean().item()
        n += 1
    model.train()
    return total / max(n, 1)


def train_model(X_train, y_train, X_val_s, y_val):
    device = get_device()
    use_amp = device.type == "cuda"
    model = MLPRegressor(
        input_dim=X_train.shape[1],
        hidden1=HP["hidden1"], hidden2=HP["hidden2"],
        dropout=HP["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=HP["lr"],
        weight_decay=HP.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)
    amp_scaler = GradScaler(device.type, enabled=use_amp)
    ds = HoopsDataset(X_train, spread=y_train, home_win=np.zeros(len(y_train)))
    loader = DataLoader(ds, batch_size=HP.get("batch_size", 4096),
                        shuffle=True, drop_last=True)
    X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0
    model.train()
    for epoch in range(MAX_EPOCHS):
        for batch in loader:
            x, spread, _ = [b.to(device) for b in batch]
            optimizer.zero_grad()
            with autocast(device.type, enabled=use_amp):
                mu, log_sigma = model(x)
                nll, _ = gaussian_nll_loss(mu, log_sigma, spread)
                loss = nll.mean()
            amp_scaler.scale(loss).backward()
            amp_scaler.step(optimizer)
            amp_scaler.update()
        scheduler.step()
        val_loss = _val_loss(model, X_val_t, y_val_t, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            break
    model.cpu()
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


@torch.no_grad()
def predict_model(model, X_scaled):
    model.eval()
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    mu_list, sigma_list = [], []
    for s in range(0, len(X_t), 4096):
        e = min(s + 4096, len(X_t))
        mu, log_sigma = model(X_t[s:e])
        sigma = torch.exp(log_sigma).clamp(0.5, 30.0)
        mu_list.append(mu.numpy())
        sigma_list.append(sigma.numpy())
    return np.concatenate(mu_list), np.concatenate(sigma_list)


# ── Walk-Forward with Game-Level Capture ──────────────────────────────

def run_walkforward_detailed():
    """Run 53-feat NG walk-forward, return per-game DataFrame with all metadata."""
    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD: 53-feat no_garbage (game-level capture)")
    print(f"{'='*70}")

    all_game_rows = []

    for holdout in HOLDOUT_YEARS:
        train_seasons = list(range(TRAIN_START, holdout))
        val_seasons = [holdout]
        print(f"\n  Holdout {holdout}: train {TRAIN_START}-{holdout-1}...", end=" ", flush=True)

        # Load features
        df_train = load_multi_season_features(
            train_seasons, no_garbage=True, adj_suffix=ADJ_SUFFIX, min_month_day=MIN_DATE)
        df_train = df_train.dropna(subset=["homeScore", "awayScore"])
        df_train = df_train[(df_train["homeScore"] != 0) | (df_train["awayScore"] != 0)]

        df_val = load_multi_season_features(
            val_seasons, no_garbage=True, adj_suffix=ADJ_SUFFIX, min_month_day=MIN_DATE)
        df_val = df_val.dropna(subset=["homeScore", "awayScore"])
        df_val = df_val[(df_val["homeScore"] != 0) | (df_val["awayScore"] != 0)]

        # Merge book spreads + conference info from lines
        try:
            lines_df = load_research_lines(holdout)
            if not lines_df.empty:
                ld = lines_df.sort_values("provider").drop_duplicates(
                    subset=["gameId"], keep="first")
                merge_cols = ["gameId"]
                line_cols = ["spread", "homeConference", "awayConference", "seasonType"]
                for c in line_cols:
                    if c in ld.columns:
                        merge_cols.append(c)
                merge_df = ld[merge_cols].copy()
                if "spread" in merge_df.columns:
                    merge_df = merge_df.rename(columns={"spread": "bookSpread"})
                df_val = df_val.merge(merge_df, on="gameId", how="left")
        except Exception:
            pass

        # Extract features
        X_train = get_feature_matrix(df_train, feature_order=FEATURE_ORDER).values.astype(np.float32)
        targets_train = get_targets(df_train)
        y_train = targets_train["spread_home"].values.astype(np.float32)

        X_val = get_feature_matrix(df_val, feature_order=FEATURE_ORDER).values.astype(np.float32)
        targets_val = get_targets(df_val)
        y_val = targets_val["spread_home"].values.astype(np.float32)

        # Impute and scale
        X_train = impute_column_means(X_train)
        X_val = impute_column_means(X_val)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_s = scaler.transform(X_train).astype(np.float32)
        X_val_s = scaler.transform(X_val).astype(np.float32)

        # Train
        model, best_epoch = train_model(X_train_s, y_train, X_val_s, y_val)
        print(f"ep={best_epoch}", end=" ", flush=True)

        # Predict
        mu, sigma = predict_model(model, X_val_s)
        mae = np.mean(np.abs(mu - y_val))
        print(f"MAE={mae:.2f} n={len(df_val)}")

        # Build per-game rows
        for i in range(len(df_val)):
            row = df_val.iloc[i]
            actual_spread = y_val[i]
            book = row.get("bookSpread")
            book_val = float(book) if pd.notna(book) else np.nan

            # Compute pick info if book spread available
            if not np.isnan(book_val):
                edge_home_pts = mu[i] + book_val
                prob_edge = normal_cdf(abs(edge_home_pts) / sigma[i]) - 0.5
                pick_home = edge_home_pts >= 0
                # Did the pick cover?
                home_covers = actual_spread + book_val > 0
                push = abs(actual_spread + book_val) < 0.5
                if push:
                    covered = np.nan  # push
                elif pick_home:
                    covered = 1.0 if home_covers else 0.0
                else:
                    covered = 1.0 if not home_covers else 0.0
            else:
                edge_home_pts = np.nan
                prob_edge = np.nan
                pick_home = np.nan
                covered = np.nan

            game_row = {
                "holdout_year": holdout,
                "gameId": row.get("gameId"),
                "startDate": row.get("startDate"),
                "homeTeam": row.get("homeTeam", ""),
                "awayTeam": row.get("awayTeam", ""),
                "homeTeamId": row.get("homeTeamId"),
                "awayTeamId": row.get("awayTeamId"),
                "neutral_site": row.get("neutral_site", 0),
                "homeConference": row.get("homeConference", ""),
                "awayConference": row.get("awayConference", ""),
                "seasonType": row.get("seasonType", ""),
                "mu": float(mu[i]),
                "sigma": float(sigma[i]),
                "actual_spread": float(actual_spread),
                "book_spread": book_val,
                "edge_home_pts": float(edge_home_pts) if not np.isnan(edge_home_pts) else np.nan,
                "prob_edge": float(prob_edge) if not isinstance(prob_edge, float) or not np.isnan(prob_edge) else np.nan,
                "pick_home": bool(pick_home) if not isinstance(pick_home, float) else np.nan,
                "covered": covered,
                "residual": float(mu[i]) - float(actual_spread),
                "best_epoch": best_epoch,
            }
            all_game_rows.append(game_row)

    df = pd.DataFrame(all_game_rows)
    df["startDate"] = pd.to_datetime(df["startDate"], errors="coerce")
    df["month"] = df["startDate"].dt.month
    df["abs_book_spread"] = df["book_spread"].abs()
    df["abs_edge"] = df["edge_home_pts"].abs()

    # Determine if conference game (home and away in same conference)
    df["is_conf_game"] = (
        (df["homeConference"] == df["awayConference"])
        & (df["homeConference"] != "")
        & (df["homeConference"].notna())
    )

    return df


# ── Analysis Functions ────────────────────────────────────────────────

def compute_roi(wins, losses, n_total):
    """Compute ROI at -110 juice."""
    if n_total == 0:
        return 0.0
    return (wins * 100 - losses * 110) / (n_total * 110)


def bucket_analysis(df_picks, col, bins, labels=None):
    """Analyze win rate and ROI by bucketed column."""
    df = df_picks.copy()
    df["_bucket"] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    results = []
    for bucket, group in df.groupby("_bucket", observed=False):
        n = len(group)
        valid = group["covered"].dropna()
        wins = int(valid.sum())
        losses = len(valid) - wins
        wr = wins / len(valid) if len(valid) > 0 else 0
        roi = compute_roi(wins, losses, len(valid))
        results.append({
            "bucket": str(bucket),
            "n": n,
            "n_resolved": len(valid),
            "wins": wins,
            "losses": losses,
            "win_rate": wr,
            "roi": roi,
            "mean_sigma": group["sigma"].mean(),
            "mean_edge": group["prob_edge"].mean(),
            "mean_abs_book": group["abs_book_spread"].mean(),
        })
    return pd.DataFrame(results)


def part1_edge_pick_analysis(df_all, report_lines):
    """Part 1: Where do edge picks fail?"""
    report_lines.append("\n# Part 1: Where Do Edge Picks Fail?\n")

    # Filter to games with book spreads and edge >= 5% (excl 2021)
    df = df_all[
        (df_all["book_spread"].notna())
        & (df_all["prob_edge"] >= EDGE_THRESHOLD)
        & (df_all["holdout_year"] != 2021)
    ].copy()

    report_lines.append(f"**Universe**: {len(df):,} picks with edge >= 5% across holdout years "
                        f"2019-2025 (excl 2021 COVID)\n")

    valid = df[df["covered"].notna()]
    wins = valid[valid["covered"] == 1]
    losses = valid[valid["covered"] == 0]
    pushes = len(df) - len(valid)

    total_wins = len(wins)
    total_losses = len(losses)
    overall_wr = total_wins / len(valid) if len(valid) > 0 else 0
    overall_roi = compute_roi(total_wins, total_losses, len(valid))

    report_lines.append(f"**Overall**: {total_wins}W-{total_losses}L-{pushes}P "
                        f"({overall_wr:.1%} win rate, {overall_roi:+.1%} ROI)\n")

    # 1a. Wins vs Losses comparison
    report_lines.append("## 1a. Wins vs Losses Profile\n")
    report_lines.append("| Metric | Wins | Losses | Diff |")
    report_lines.append("|--------|------|--------|------|")

    for label, col, fmt in [
        ("Avg sigma (uncertainty)", "sigma", ".2f"),
        ("Avg prob_edge", "prob_edge", ".3f"),
        ("Avg |edge_home_pts|", "abs_edge", ".2f"),
        ("Avg |book_spread|", "abs_book_spread", ".1f"),
    ]:
        w_val = wins[col].mean()
        l_val = losses[col].mean()
        diff = w_val - l_val
        report_lines.append(
            f"| {label} | {w_val:{fmt}} | {l_val:{fmt}} | {diff:+{fmt}} |"
        )

    # Home vs away breakdown
    home_picks = valid[valid["pick_home"] == True]
    away_picks = valid[valid["pick_home"] == False]
    h_wr = home_picks["covered"].mean() if len(home_picks) > 0 else 0
    a_wr = away_picks["covered"].mean() if len(away_picks) > 0 else 0
    h_wins = int(home_picks["covered"].sum())
    h_losses = len(home_picks[home_picks["covered"].notna()]) - h_wins
    a_wins = int(away_picks["covered"].sum())
    a_losses = len(away_picks[away_picks["covered"].notna()]) - a_wins
    h_roi = compute_roi(h_wins, h_losses, len(home_picks[home_picks["covered"].notna()]))
    a_roi = compute_roi(a_wins, a_losses, len(away_picks[away_picks["covered"].notna()]))

    report_lines.append(f"\n| Pick Side | Count | Win Rate | ROI |")
    report_lines.append(f"|-----------|-------|----------|-----|")
    report_lines.append(f"| Home | {len(home_picks):,} | {h_wr:.1%} | {h_roi:+.1%} |")
    report_lines.append(f"| Away | {len(away_picks):,} | {a_wr:.1%} | {a_roi:+.1%} |")

    # 1b. Win rate by edge bucket
    report_lines.append("\n## 1b. Win Rate by Edge Bucket\n")
    report_lines.append("If bigger edges don't win at higher rates, confidence is miscalibrated.\n")

    edge_bins = [0.05, 0.08, 0.11, 0.14, 0.17, 1.0]
    edge_labels = ["5-8%", "8-11%", "11-14%", "14-17%", "17%+"]
    edge_df = bucket_analysis(valid, "prob_edge", edge_bins, edge_labels)

    report_lines.append("| Edge Bucket | Count | Win Rate | ROI | Avg Sigma |")
    report_lines.append("|-------------|-------|----------|-----|-----------|")
    for _, r in edge_df.iterrows():
        report_lines.append(
            f"| {r['bucket']} | {r['n_resolved']:,} | {r['win_rate']:.1%} | "
            f"{r['roi']:+.1%} | {r['mean_sigma']:.1f} |"
        )

    # Check monotonicity
    wrs = edge_df["win_rate"].values
    is_monotonic = all(wrs[i] <= wrs[i+1] for i in range(len(wrs)-1) if edge_df.iloc[i+1]["n_resolved"] > 10)
    if is_monotonic:
        report_lines.append("\nEdge-to-win-rate curve is **monotonically increasing** — calibration direction is correct.")
    else:
        report_lines.append("\nEdge-to-win-rate curve is **NOT monotonic** — model confidence may be miscalibrated.")

    # 1c. Value destroyers
    report_lines.append("\n## 1c. Value Destroyers — Game Types That Lose Money\n")
    report_lines.append("| Segment | Count | Win Rate | ROI | Impact |")
    report_lines.append("|---------|-------|----------|-----|--------|")

    segments = []

    # Conference vs non-conference
    conf = valid[valid["is_conf_game"] == True]
    nonconf = valid[valid["is_conf_game"] == False]
    for label, sub in [("Conference", conf), ("Non-conference", nonconf)]:
        if len(sub) > 0:
            w = int(sub["covered"].sum())
            l = len(sub) - w
            wr = w / len(sub)
            roi = compute_roi(w, l, len(sub))
            segments.append((label, len(sub), wr, roi))

    # Book spread buckets
    blowout = valid[valid["abs_book_spread"] > 15]
    tossup = valid[valid["abs_book_spread"] < 3]
    mid = valid[(valid["abs_book_spread"] >= 3) & (valid["abs_book_spread"] <= 15)]
    for label, sub in [("Blowout (|spread|>15)", blowout), ("Toss-up (|spread|<3)", tossup),
                        ("Mid-range (3-15)", mid)]:
        if len(sub) > 0:
            w = int(sub["covered"].sum())
            l = len(sub) - w
            wr = w / len(sub)
            roi = compute_roi(w, l, len(sub))
            segments.append((label, len(sub), wr, roi))

    # Month buckets
    early = valid[valid["month"].isin([11, 12])]
    conf_play = valid[valid["month"].isin([1, 2])]
    march = valid[valid["month"].isin([3, 4])]
    for label, sub in [("Nov-Dec (early)", early), ("Jan-Feb (conf play)", conf_play),
                        ("Mar-Apr (tourney)", march)]:
        if len(sub) > 0:
            w = int(sub["covered"].sum())
            l = len(sub) - w
            wr = w / len(sub)
            roi = compute_roi(w, l, len(sub))
            segments.append((label, len(sub), wr, roi))

    # High sigma
    high_sig = valid[valid["sigma"] > 12]
    low_sig = valid[valid["sigma"] <= 10]
    mid_sig = valid[(valid["sigma"] > 10) & (valid["sigma"] <= 12)]
    for label, sub in [("Sigma > 12 (high unc)", high_sig),
                        ("Sigma 10-12", mid_sig),
                        ("Sigma <= 10 (low unc)", low_sig)]:
        if len(sub) > 0:
            w = int(sub["covered"].sum())
            l = len(sub) - w
            wr = w / len(sub)
            roi = compute_roi(w, l, len(sub))
            segments.append((label, len(sub), wr, roi))

    # Neutral site
    neutral = valid[valid["neutral_site"] == 1]
    non_neutral = valid[valid["neutral_site"] == 0]
    for label, sub in [("Neutral site", neutral), ("Non-neutral", non_neutral)]:
        if len(sub) > 0:
            w = int(sub["covered"].sum())
            l = len(sub) - w
            wr = w / len(sub)
            roi = compute_roi(w, l, len(sub))
            segments.append((label, len(sub), wr, roi))

    # Sort by ROI to find the worst segments
    segments.sort(key=lambda x: x[3])
    for label, n, wr, roi in segments:
        impact = "DRAG" if roi < -0.05 else ("BOOST" if roi > 0.02 else "NEUTRAL")
        report_lines.append(f"| {label} | {n:,} | {wr:.1%} | {roi:+.1%} | {impact} |")

    return df  # return filtered picks for later use


def part2_market_knowledge(df_all, report_lines):
    """Part 2: What does the market know that we don't?"""
    report_lines.append("\n\n# Part 2: What Does the Market Know That We Don't?\n")

    df = df_all[
        (df_all["book_spread"].notna())
        & (df_all["holdout_year"] != 2021)
    ].copy()

    # 2a. High-edge losses — games where model was very confident but wrong
    report_lines.append("## 2a. High-Edge Losses (edge >= 10%, pick lost)\n")

    high_edge_losses = df[
        (df["prob_edge"] >= 0.10)
        & (df["covered"] == 0)
    ].sort_values("prob_edge", ascending=False)

    report_lines.append(f"Total high-edge losses: {len(high_edge_losses):,}\n")

    # Sample up to 30
    sample = high_edge_losses.head(30)
    report_lines.append("| Date | Matchup | Book Spread | Model Spread | Edge | Sigma | Actual |")
    report_lines.append("|------|---------|-------------|--------------|------|-------|--------|")
    for _, r in sample.iterrows():
        date_str = r["startDate"].strftime("%Y-%m-%d") if pd.notna(r["startDate"]) else "?"
        matchup = f"{r['awayTeam']} @ {r['homeTeam']}"
        model_spread = -r["mu"]  # convert to book convention
        actual = r["actual_spread"]
        report_lines.append(
            f"| {date_str} | {matchup} | {r['book_spread']:+.1f} | "
            f"{model_spread:+.1f} | {r['prob_edge']:.1%} | {r['sigma']:.1f} | "
            f"{actual:+.1f} |"
        )

    # Check for repeat teams
    report_lines.append("\n### Repeat Teams in High-Edge Losses\n")
    home_counts = high_edge_losses["homeTeam"].value_counts()
    away_counts = high_edge_losses["awayTeam"].value_counts()
    all_team_counts = (home_counts.add(away_counts, fill_value=0)).sort_values(ascending=False)
    top_losers = all_team_counts[all_team_counts >= 3].head(15)
    if len(top_losers) > 0:
        report_lines.append("Teams appearing 3+ times in high-edge losses:\n")
        report_lines.append("| Team | Appearances |")
        report_lines.append("|------|-------------|")
        for team, count in top_losers.items():
            report_lines.append(f"| {team} | {int(count)} |")
    else:
        report_lines.append("No team appears 3+ times in the sample.\n")

    # 2b. Residual by team — most over/undervalued teams
    report_lines.append("\n## 2b. Most Over/Undervalued Teams\n")
    report_lines.append("Residual = predicted_spread - actual_spread. "
                        "Positive = model overvalues team (thinks they'll win by more than they do).\n")

    # Build per-team residuals from the home perspective
    team_residuals = defaultdict(list)
    for _, r in df.iterrows():
        if pd.notna(r["residual"]):
            # Home team: residual directly (positive = model overvalues home)
            team_residuals[r["homeTeam"]].append(r["residual"])
            # Away team: flip sign (positive = model overvalues away)
            team_residuals[r["awayTeam"]].append(-r["residual"])

    team_stats = []
    for team, resids in team_residuals.items():
        if len(resids) >= 15:  # minimum games for stability
            team_stats.append({
                "team": team,
                "n_games": len(resids),
                "mean_residual": np.mean(resids),
                "median_residual": np.median(resids),
                "std_residual": np.std(resids),
            })
    team_df = pd.DataFrame(team_stats).sort_values("mean_residual", ascending=False)

    report_lines.append("### Top 10 Most OVERVALUED Teams (model thinks they're better than they are)\n")
    report_lines.append("| Rank | Team | Games | Avg Residual | Median |")
    report_lines.append("|------|------|-------|-------------|--------|")
    for i, (_, r) in enumerate(team_df.head(10).iterrows()):
        report_lines.append(
            f"| {i+1} | {r['team']} | {r['n_games']} | {r['mean_residual']:+.2f} | "
            f"{r['median_residual']:+.2f} |"
        )

    report_lines.append("\n### Top 10 Most UNDERVALUED Teams (model underestimates them)\n")
    report_lines.append("| Rank | Team | Games | Avg Residual | Median |")
    report_lines.append("|------|------|-------|-------------|--------|")
    for i, (_, r) in enumerate(team_df.tail(10).iloc[::-1].iterrows()):
        report_lines.append(
            f"| {i+1} | {r['team']} | {r['n_games']} | {r['mean_residual']:+.2f} | "
            f"{r['median_residual']:+.2f} |"
        )

    # 2c. Monthly MAE — does accuracy deteriorate at certain points?
    report_lines.append("\n## 2c. Monthly MAE Across Seasons\n")
    report_lines.append("Does the model get worse at certain points in the season?\n")

    month_names = {11: "Nov", 12: "Dec", 1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr"}
    month_order = [11, 12, 1, 2, 3, 4]

    # Per-season monthly MAE
    report_lines.append("| Month | Overall MAE | n | MAE by Season... |")
    report_lines.append("|-------|------------|---|")

    # Build a table: rows=months, cols=seasons
    seasons = sorted([y for y in df["holdout_year"].unique() if y != 2021])
    header = "| Month |"
    for s in seasons:
        header += f" {s} |"
    header += " **All** |"
    report_lines[-2] = header
    report_lines[-1] = "|" + "------|" * (len(seasons) + 2)

    for m in month_order:
        m_df = df[df["month"] == m]
        if len(m_df) == 0:
            continue
        row = f"| {month_names.get(m, str(m))} |"
        for s in seasons:
            s_m = m_df[m_df["holdout_year"] == s]
            if len(s_m) > 5:
                mae = np.mean(np.abs(s_m["residual"]))
                row += f" {mae:.2f} ({len(s_m)}) |"
            else:
                row += " — |"
        overall_mae = np.mean(np.abs(m_df["residual"]))
        row += f" **{overall_mae:.2f}** ({len(m_df)}) |"
        report_lines.append(row)

    # Also look at picks ROI by month
    report_lines.append("\n### Monthly Pick Performance (edge >= 5%)\n")
    picks = df[(df["prob_edge"] >= EDGE_THRESHOLD) & (df["covered"].notna())]
    report_lines.append("| Month | Picks | Win Rate | ROI |")
    report_lines.append("|-------|-------|----------|-----|")
    for m in month_order:
        m_picks = picks[picks["month"] == m]
        if len(m_picks) < 10:
            continue
        w = int(m_picks["covered"].sum())
        l = len(m_picks) - w
        wr = w / len(m_picks)
        roi = compute_roi(w, l, len(m_picks))
        report_lines.append(f"| {month_names.get(m, str(m))} | {len(m_picks):,} | {wr:.1%} | {roi:+.1%} |")


def part3_sigma_calibration(df_all, report_lines):
    """Part 3: Sigma calibration check."""
    report_lines.append("\n\n# Part 3: Sigma Calibration Check\n")
    report_lines.append("Is the model's predicted uncertainty (sigma) well-calibrated?\n")

    df = df_all[df_all["holdout_year"] != 2021].copy()
    df["abs_error"] = np.abs(df["residual"])

    # Bin by sigma
    sigma_bins = [0, 7, 9, 11, 13, 30]
    sigma_labels = ["<7", "7-9", "9-11", "11-13", "13+"]
    df["sigma_bucket"] = pd.cut(df["sigma"], bins=sigma_bins, labels=sigma_labels, include_lowest=True)

    report_lines.append("| Sigma Bucket | Count | Avg Predicted σ | Actual Error Std | "
                        "Ratio (pred/actual) | Interpretation |")
    report_lines.append("|-------------|-------|----------------|-----------------|"
                        "--------------------|----------------|")

    for bucket in sigma_labels:
        sub = df[df["sigma_bucket"] == bucket]
        if len(sub) < 20:
            continue
        avg_sigma = sub["sigma"].mean()
        actual_std = sub["residual"].std()
        ratio = avg_sigma / actual_std if actual_std > 0 else 0
        if ratio > 1.1:
            interp = "Under-confident"
        elif ratio < 0.9:
            interp = "**OVER-CONFIDENT**"
        else:
            interp = "Well-calibrated"
        report_lines.append(
            f"| {bucket} | {len(sub):,} | {avg_sigma:.2f} | {actual_std:.2f} | "
            f"{ratio:.2f} | {interp} |"
        )

    # Also check: games where sigma is low — do picks there have better ROI?
    report_lines.append("\n## Sigma-Based Pick Performance\n")
    report_lines.append("If over-confident in some bins, those picks offer phantom edges.\n")

    picks = df[(df["prob_edge"] >= EDGE_THRESHOLD) & (df["covered"].notna()) & (df["book_spread"].notna())]

    report_lines.append("| Sigma Bucket | Picks | Win Rate | ROI | Avg Edge |")
    report_lines.append("|-------------|-------|----------|-----|----------|")
    for bucket in sigma_labels:
        sub = picks[picks["sigma_bucket"] == bucket]
        if len(sub) < 20:
            continue
        w = int(sub["covered"].sum())
        l = len(sub) - w
        wr = w / len(sub)
        roi = compute_roi(w, l, len(sub))
        avg_edge = sub["prob_edge"].mean()
        report_lines.append(f"| {bucket} | {len(sub):,} | {wr:.1%} | {roi:+.1%} | {avg_edge:.1%} |")

    # Calibration by holdout year
    report_lines.append("\n## Calibration Consistency Across Seasons\n")
    report_lines.append("| Season | Avg σ | Actual Error Std | Ratio | MAE |")
    report_lines.append("|--------|-------|-----------------|-------|-----|")
    for yr in sorted(df["holdout_year"].unique()):
        sub = df[df["holdout_year"] == yr]
        avg_sig = sub["sigma"].mean()
        actual_std = sub["residual"].std()
        ratio = avg_sig / actual_std if actual_std > 0 else 0
        mae = sub["abs_error"].mean()
        report_lines.append(f"| {yr} | {avg_sig:.2f} | {actual_std:.2f} | {ratio:.2f} | {mae:.2f} |")


def part4_filter_strategies(df_all, report_lines):
    """Part 4: Betting filter strategies."""
    report_lines.append("\n\n# Part 4: Betting Filter Strategies\n")
    report_lines.append("Testing filter combinations to find historically profitable subsets. "
                        "**Overfitting warning**: with many combinations, some will look good by chance. "
                        "Focus on filters consistent across multiple seasons.\n")

    df = df_all[
        (df_all["book_spread"].notna())
        & (df_all["covered"].notna())
        & (df_all["holdout_year"] != 2021)
    ].copy()

    # Define filter dimensions
    edge_mins = [0.05, 0.08, 0.10, 0.12, 0.15]
    sigma_maxes = [99, 12, 11, 10, 9]
    spread_ranges = [
        ("all", 0, 99),
        ("close (0-7)", 0, 7),
        ("mid (7-15)", 7, 15),
        ("wide (15+)", 15, 99),
        ("no blowout (<15)", 0, 15),
    ]
    month_groups = [
        ("all", [11, 12, 1, 2, 3, 4]),
        ("conf play (Jan-Mar)", [1, 2, 3]),
        ("early (Nov-Dec)", [11, 12]),
    ]

    # Test all combinations
    results = []
    for edge_min in edge_mins:
        for sigma_max in sigma_maxes:
            for spread_label, sp_lo, sp_hi in spread_ranges:
                for month_label, months in month_groups:
                    mask = (
                        (df["prob_edge"] >= edge_min)
                        & (df["sigma"] <= sigma_max)
                        & (df["abs_book_spread"] >= sp_lo)
                        & (df["abs_book_spread"] <= sp_hi)
                        & (df["month"].isin(months))
                    )
                    sub = df[mask]
                    if len(sub) < 100:  # skip tiny samples
                        continue

                    w = int(sub["covered"].sum())
                    l = len(sub) - w
                    wr = w / len(sub)
                    roi = compute_roi(w, l, len(sub))

                    # Check consistency: winning seasons / total seasons
                    season_rois = []
                    for yr in sorted(sub["holdout_year"].unique()):
                        yr_sub = sub[sub["holdout_year"] == yr]
                        if len(yr_sub) >= 10:
                            yr_w = int(yr_sub["covered"].sum())
                            yr_l = len(yr_sub) - yr_w
                            yr_roi = compute_roi(yr_w, yr_l, len(yr_sub))
                            season_rois.append(yr_roi)

                    profitable_seasons = sum(1 for r in season_rois if r > 0)
                    n_seasons = len(season_rois)

                    filter_desc = (f"edge>={edge_min:.0%}, sigma<={sigma_max}, "
                                   f"spread={spread_label}, months={month_label}")

                    results.append({
                        "filter": filter_desc,
                        "edge_min": edge_min,
                        "sigma_max": sigma_max,
                        "spread_range": spread_label,
                        "months": month_label,
                        "n": len(sub),
                        "win_rate": wr,
                        "roi": roi,
                        "profitable_seasons": profitable_seasons,
                        "n_seasons": n_seasons,
                        "consistency": profitable_seasons / n_seasons if n_seasons > 0 else 0,
                        "season_rois": season_rois,
                    })

    results_df = pd.DataFrame(results)

    # Top 10 by ROI with >= 200 picks
    report_lines.append("## Top 10 Filters by ROI (n >= 200)\n")
    big_enough = results_df[results_df["n"] >= 200].sort_values("roi", ascending=False).head(10)
    report_lines.append("| Rank | Filter | Picks | Win Rate | ROI | Seasons + | Consistency |")
    report_lines.append("|------|--------|-------|----------|-----|-----------|-------------|")
    for i, (_, r) in enumerate(big_enough.iterrows()):
        flag = "**" if r["n"] < 500 else ""
        report_lines.append(
            f"| {i+1} | {r['filter']} | {flag}{r['n']:,}{flag} | {r['win_rate']:.1%} | "
            f"{r['roi']:+.1%} | {r['profitable_seasons']}/{r['n_seasons']} | "
            f"{r['consistency']:.0%} |"
        )

    # Top 5 by consistency (profitable in most seasons) with n >= 300
    report_lines.append("\n## Top 5 Filters by Consistency (n >= 300, most seasons profitable)\n")
    consistent = (results_df[results_df["n"] >= 300]
                  .sort_values(["consistency", "roi"], ascending=[False, False])
                  .head(5))
    report_lines.append("| Rank | Filter | Picks | Win Rate | ROI | Seasons + | Per-Season ROIs |")
    report_lines.append("|------|--------|-------|----------|-----|-----------|-----------------|")
    for i, (_, r) in enumerate(consistent.iterrows()):
        rois_str = ", ".join(f"{x:+.1%}" for x in r["season_rois"])
        report_lines.append(
            f"| {i+1} | {r['filter']} | {r['n']:,} | {r['win_rate']:.1%} | "
            f"{r['roi']:+.1%} | {r['profitable_seasons']}/{r['n_seasons']} | {rois_str} |"
        )

    # Deep dive: best filter
    if len(big_enough) > 0:
        best = big_enough.iloc[0]
        report_lines.append(f"\n## Deep Dive: Best ROI Filter\n")
        report_lines.append(f"**{best['filter']}**\n")
        report_lines.append(f"- Picks: {best['n']:,}")
        report_lines.append(f"- Win Rate: {best['win_rate']:.1%}")
        report_lines.append(f"- ROI: {best['roi']:+.1%}")
        report_lines.append(f"- Consistency: {best['profitable_seasons']}/{best['n_seasons']} seasons profitable")
        report_lines.append(f"- Per-season ROIs: {', '.join(f'{x:+.1%}' for x in best['season_rois'])}\n")

        if best["n"] < 500:
            report_lines.append("**WARNING**: Sample size < 500. High overfitting risk.\n")

    # Baseline comparison
    report_lines.append("\n## Baseline Comparison\n")
    baseline = df[df["prob_edge"] >= 0.05]
    b_w = int(baseline["covered"].sum())
    b_l = len(baseline) - b_w
    b_wr = b_w / len(baseline)
    b_roi = compute_roi(b_w, b_l, len(baseline))
    report_lines.append(f"**Unfiltered baseline** (edge >= 5%, no other filters): "
                        f"{len(baseline):,} picks, {b_wr:.1%} WR, {b_roi:+.1%} ROI\n")

    # Also test: EXCLUDE the worst segments from Part 1
    report_lines.append("## Exclusion-Based Filters\n")
    report_lines.append("Instead of finding what to bet, find what NOT to bet.\n")

    exclusions = [
        ("Exclude blowout lines (>15)", baseline["abs_book_spread"] <= 15),
        ("Exclude high sigma (>12)", baseline["sigma"] <= 12),
        ("Exclude early season (Nov-Dec)", ~baseline["month"].isin([11, 12])),
        ("Exclude toss-ups (<3)", baseline["abs_book_spread"] >= 3),
        ("Exclude neutral site", baseline["neutral_site"] == 0),
        ("Combined: no blowouts, no high sigma, no early",
         (baseline["abs_book_spread"] <= 15) & (baseline["sigma"] <= 12) & (~baseline["month"].isin([11, 12]))),
    ]

    report_lines.append("| Exclusion | Picks | Win Rate | ROI | Δ ROI vs baseline |")
    report_lines.append("|-----------|-------|----------|-----|-------------------|")
    for label, mask in exclusions:
        sub = baseline.loc[mask]
        if len(sub) < 50:
            continue
        w = int(sub["covered"].sum())
        l = len(sub) - w
        wr = w / len(sub)
        roi = compute_roi(w, l, len(sub))
        delta = roi - b_roi
        report_lines.append(f"| {label} | {len(sub):,} | {wr:.1%} | {roi:+.1%} | {delta:+.1%} |")


def write_summary(df_all, report_lines):
    """Write executive summary at the top."""
    summary = []
    summary.append("# Edge Leak Analysis Report\n")
    summary.append(f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    summary.append(f"**Model**: 53-feat no_garbage, 384/256 MLP, walk-forward 2019-2025 (excl 2021)")
    summary.append(f"**Edge threshold**: {EDGE_THRESHOLD:.0%}\n")

    df = df_all[df_all["holdout_year"] != 2021]
    total = len(df)
    with_book = df["book_spread"].notna().sum()
    picks = df[(df["prob_edge"] >= EDGE_THRESHOLD) & (df["covered"].notna())]
    wins = int(picks["covered"].sum())
    losses = len(picks) - wins
    wr = wins / len(picks) if len(picks) > 0 else 0
    roi = compute_roi(wins, losses, len(picks))

    summary.append(f"## Quick Stats\n")
    summary.append(f"- Total games: {total:,}")
    summary.append(f"- Games with book spread: {with_book:,}")
    summary.append(f"- Qualified picks (edge >= 5%): {len(picks):,}")
    summary.append(f"- Record: {wins}W-{losses}L ({wr:.1%})")
    summary.append(f"- ROI: {roi:+.1%}")
    summary.append(f"- Break-even win rate at -110: 52.4%\n")

    gap = wr - 0.524
    summary.append(f"**The gap**: Model win rate is {gap:+.1%} vs break-even. "
                   f"Need to find ~{abs(gap):.1%} of edge to reach profitability.\n")

    summary.append("---\n")
    return summary


# ── Main ──────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    torch.manual_seed(42)
    np.random.seed(42)

    # Run walk-forward with game-level capture
    df = run_walkforward_detailed()
    print(f"\nCollected {len(df):,} game-level predictions")
    print(f"  With book spread: {df['book_spread'].notna().sum():,}")

    # Run all analyses
    report_lines = []

    print("\nRunning Part 1: Edge pick analysis...")
    picks_df = part1_edge_pick_analysis(df, report_lines)

    print("Running Part 2: Market knowledge analysis...")
    part2_market_knowledge(df, report_lines)

    print("Running Part 3: Sigma calibration...")
    part3_sigma_calibration(df, report_lines)

    print("Running Part 4: Filter strategies...")
    part4_filter_strategies(df, report_lines)

    # Build final report
    summary = write_summary(df, report_lines)
    full_report = "\n".join(summary + report_lines)

    # Write report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(full_report)
    print(f"\nReport written to {REPORT_PATH}")

    elapsed = time.time() - t0
    print(f"Total time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
