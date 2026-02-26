"""Generate head-to-head comparison data: Torvik model vs hoops-edge model.

Standalone script — reads from MySQL for Torvik features, loads Torvik checkpoint
directly, joins with hoops-edge backtest predictions and book spreads.

Tasks:
  1. Generate Torvik-model predictions for 2025 season
  2. Build team name mapping (gold/CBBD names ↔ Torvik names)
  3. Create joined DataFrame with both models' predictions
  4. Document Torvik training cutoff

Output: reports/head_to_head_data_2025.csv

Usage:
    poetry run python -u scripts/generate_h2h_data.py
"""
from __future__ import annotations

import functools
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sqlalchemy import create_engine

# Force unbuffered output
print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TORVIK_REPO = PROJECT_ROOT.parent / "college_basketball_local_mu_sig"

# Hoops-edge backtest
BACKTEST_PATH = PROJECT_ROOT / "predictions" / "backtest_2025_sos085.csv"
V2_FEATURES_PATH = config.FEATURES_DIR / "season_2025_no_garbage_v2_features.parquet"
LINES_PATH = config.FEATURES_DIR / "lines_2025.parquet"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "head_to_head_data_2025.csv"

# Torvik model artifacts
TORVIK_CKPT = TORVIK_REPO / "checkpoints" / "mlp_regressor.pth"
TORVIK_FEATURES = TORVIK_REPO / "artifacts" / "feature_order.json"
TORVIK_SCALER = TORVIK_REPO / "artifacts" / "scaler.pkl"
TORVIK_ENV = TORVIK_REPO / ".env"


# ── Torvik model architecture (inline to avoid import chain) ──────


class TorviKMLPRegressor(nn.Module):
    """Exact copy of college_basketball_local_mu_sig/bball/models/architecture.py."""

    def __init__(self, input_dim, hidden=256, hidden2=None, dropout=0.3):
        super().__init__()
        hidden2 = hidden2 or hidden // 2
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden2, 2)

    def forward(self, x):
        return self.head(self.features(x))


# ── Team name mapping (gold CBBD names → Torvik MySQL names) ─────


GOLD_TO_TORVIK: dict[str, str] = {
    "Alabama State": "Alabama St.",
    "Alcorn State": "Alcorn St.",
    "American University": "American",
    "App State": "Appalachian St.",
    "Arizona State": "Arizona St.",
    "Arkansas State": "Arkansas St.",
    "Arkansas-Pine Bluff": "Arkansas Pine Bluff",
    "Ball State": "Ball St.",
    "Bethune-Cookman": "Bethune Cookman",
    "Boise State": "Boise St.",
    "Cal State Bakersfield": "Cal St. Bakersfield",
    "Cal State Fullerton": "Cal St. Fullerton",
    "Cal State Northridge": "Cal St. Northridge",
    "California Baptist": "Cal Baptist",
    "Chicago State": "Chicago St.",
    "Cleveland State": "Cleveland St.",
    "Colorado State": "Colorado St.",
    "Coppin State": "Coppin St.",
    "Delaware State": "Delaware St.",
    "East Tennessee State": "East Tennessee St.",
    "East Texas A&M": "Texas A&M Commerce",
    "Florida International": "FIU",
    "Florida State": "Florida St.",
    "Fresno State": "Fresno St.",
    "Gardner-Webb": "Gardner Webb",
    "Georgia State": "Georgia St.",
    "Grambling": "Grambling St.",
    "Hawai'i": "Hawaii",
    "IU Indianapolis": "IU Indy",
    "Idaho State": "Idaho St.",
    "Illinois State": "Illinois St.",
    "Indiana State": "Indiana St.",
    "Iowa State": "Iowa St.",
    "Jackson State": "Jackson St.",
    "Jacksonville State": "Jacksonville St.",
    "Kansas City": "UMKC",
    "Kansas State": "Kansas St.",
    "Kennesaw State": "Kennesaw St.",
    "Kent State": "Kent St.",
    "Long Beach State": "Long Beach St.",
    "Long Island University": "LIU",
    "Loyola Maryland": "Loyola MD",
    "McNeese": "McNeese St.",
    "Miami": "Miami FL",
    "Miami (OH)": "Miami OH",
    "Michigan State": "Michigan St.",
    "Mississippi State": "Mississippi St.",
    "Mississippi Valley State": "Mississippi Valley St.",
    "Missouri State": "Missouri St.",
    "Montana State": "Montana St.",
    "Morehead State": "Morehead St.",
    "Morgan State": "Morgan St.",
    "Murray State": "Murray St.",
    "NC State": "N.C. State",
    "New Mexico State": "New Mexico St.",
    "Nicholls": "Nicholls St.",
    "Norfolk State": "Norfolk St.",
    "North Dakota State": "North Dakota St.",
    "Northwestern State": "Northwestern St.",
    "Ohio State": "Ohio St.",
    "Oklahoma State": "Oklahoma St.",
    "Ole Miss": "Mississippi",
    "Omaha": "Nebraska Omaha",
    "Oregon State": "Oregon St.",
    "Penn State": "Penn St.",
    "Pennsylvania": "Penn",
    "Portland State": "Portland St.",
    "Queens University": "Queens",
    "SE Louisiana": "Southeastern Louisiana",
    "Sacramento State": "Sacramento St.",
    "Sam Houston": "Sam Houston St.",
    "San Diego State": "San Diego St.",
    "San José State": "San Jose St.",
    "Seattle U": "Seattle",
    "South Carolina State": "South Carolina St.",
    "South Carolina Upstate": "USC Upstate",
    "South Dakota State": "South Dakota St.",
    "Southeast Missouri State": "Southeast Missouri St.",
    "St. Francis (PA)": "Saint Francis",
    "St. Thomas-Minnesota": "St. Thomas",
    "Tarleton State": "Tarleton St.",
    "Tennessee State": "Tennessee St.",
    "Texas A&M-Corpus Christi": "Texas A&M Corpus Chris",
    "Texas State": "Texas St.",
    "UAlbany": "Albany",
    "UConn": "Connecticut",
    "UIC": "Illinois Chicago",
    "UL Monroe": "Louisiana Monroe",
    "UT Martin": "Tennessee Martin",
    "Utah State": "Utah St.",
    "Washington State": "Washington St.",
    "Weber State": "Weber St.",
    "Wichita State": "Wichita St.",
    "Wright State": "Wright St.",
    "Youngstown State": "Youngstown St.",
}

# Reverse mapping: Torvik name → gold/CBBD name
TORVIK_TO_GOLD = {v: k for k, v in GOLD_TO_TORVIK.items()}


# ── MySQL connection ──────────────────────────────────────────────


def _mysql_engine():
    """Create MySQL engine using Torvik repo .env credentials."""
    import os
    from dotenv import load_dotenv

    load_dotenv(TORVIK_ENV)
    user = os.getenv("BBALL_DB_USER", "root")
    pwd = os.getenv("BBALL_DB_PASS", "")
    host = os.getenv("BBALL_DB_HOST", "localhost")
    db = os.getenv("BBALL_DB_NAME", "sports")
    return create_engine(f"mysql+pymysql://{user}:{pwd}@{host}/{db}")


# ── Task 5: Training cutoff ──────────────────────────────────────


def document_training_cutoff(engine):
    """Document the Torvik model's training data date range."""
    print(f"\n{'='*70}")
    print("TORVIK MODEL TRAINING CUTOFF")
    print(f"{'='*70}\n")

    stats = pd.read_sql(
        "SELECT MIN(date) as min_date, MAX(date) as max_date, COUNT(*) as cnt "
        "FROM sports.training_data",
        engine,
    )
    print(f"  sports.training_data: {stats['cnt'].iloc[0]:,} rows")
    print(f"  Date range: {stats['min_date'].iloc[0]} to {stats['max_date'].iloc[0]}")

    print("\n  Training method (from loaders.py load_training_dataframe()):")
    print("    SQL: SELECT * FROM sports.training_data")
    print("    → NO date filter — ALL historical games used for training")
    print("    → Targets: spread_home = -(away_pts - home_pts), home_win = binary")
    print("    → Drops: date, MOV, total_pts, team_names, scores")
    print("    → 80/20 train/val split (stratified on home_win)")

    # Check if 2025 season games are in the training data
    season_2025 = pd.read_sql(
        "SELECT COUNT(*) as cnt FROM sports.training_data "
        "WHERE date >= '2024-11-01' AND date < '2025-05-01'",
        engine,
    )
    print(f"\n  2024-25 season games in training_data: {season_2025['cnt'].iloc[0]:,}")
    print("  ⚠ IMPORTANT: The Torvik model trains on ALL data in the table,")
    print("    INCLUDING the 2024-25 season it predicts on.")
    print("    This is NOT a clean holdout — the model has seen these outcomes.")
    print("    However, the features (adj_oe, BARTHAG, rolling stats) are")
    print("    point-in-time from daily Torvik scrapes, so feature leakage")
    print("    depends on when the model was last retrained.")

    # Check checkpoint timestamp
    if TORVIK_CKPT.exists():
        import datetime

        mtime = TORVIK_CKPT.stat().st_mtime
        ts = datetime.datetime.fromtimestamp(mtime)
        print(f"\n  Checkpoint file last modified: {ts.strftime('%Y-%m-%d %H:%M')}")

        ckpt = torch.load(TORVIK_CKPT, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            fo = ckpt.get("feature_order", [])
            hp = ckpt.get("hparams", {})
            print(f"  Checkpoint features: {len(fo)}")
            print(f"  Checkpoint hparams: {hp}")


# ── Task 1-2: Generate Torvik predictions ─────────────────────────


def generate_torvik_predictions(engine) -> pd.DataFrame:
    """Generate Torvik model predictions for all 2025 season games."""
    print(f"\n{'='*70}")
    print("GENERATING TORVIK MODEL PREDICTIONS")
    print(f"{'='*70}\n")

    # Load season data from MySQL
    print("  Loading 2024-25 season from sports.training_data...")
    df = pd.read_sql(
        "SELECT * FROM sports.training_data "
        "WHERE date >= '2024-11-01' AND date < '2025-05-01'",
        engine,
    )
    print(f"  Loaded {len(df):,} games")

    if df.empty:
        print("  ERROR: No games found")
        return pd.DataFrame()

    # Compute targets (for ground truth)
    df["away_team_pts"] = pd.to_numeric(df["away_team_pts"], errors="coerce").fillna(0).astype("int64")
    df["home_team_pts"] = pd.to_numeric(df["home_team_pts"], errors="coerce").fillna(0).astype("int64")
    df["actual_margin"] = df["home_team_pts"] - df["away_team_pts"]
    df["home_team_home"] = df["neutral_site"].eq(0)
    df["away_team_home"] = False

    # Separate info and features
    info_cols = ["date", "away_team_name", "home_team_name", "neutral_site",
                 "away_team_pts", "home_team_pts", "actual_margin"]
    info_df = df[info_cols].copy()

    # Load Torvik feature order
    print(f"  Loading feature order from {TORVIK_FEATURES}...")
    feat_order = json.loads(TORVIK_FEATURES.read_text())
    print(f"  Feature order: {len(feat_order)} features")

    # Build feature matrix in correct order
    X_df = df.reindex(columns=feat_order)
    X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Load scaler
    print(f"  Loading scaler from {TORVIK_SCALER}...")
    scaler = joblib.load(TORVIK_SCALER)
    X_scaled = scaler.transform(X_df.values.astype(np.float32))

    # Load checkpoint and build model
    print(f"  Loading checkpoint from {TORVIK_CKPT}...")
    ckpt = torch.load(TORVIK_CKPT, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        hp = ckpt.get("hparams", {})
        state_dict = ckpt["state_dict"]
        ckpt_feat_order = ckpt.get("feature_order", [])
        if ckpt_feat_order:
            print(f"  Checkpoint feature_order: {len(ckpt_feat_order)} features")
            feat_order = ckpt_feat_order
            # Re-align features to checkpoint's order
            X_df = df.reindex(columns=feat_order)
            X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
            X_scaled = scaler.transform(X_df.values.astype(np.float32))
    else:
        hp = {}
        state_dict = ckpt

    input_dim = len(feat_order)
    print(f"  Architecture: input_dim={input_dim}, hidden={hp.get('hidden', 256)}, "
          f"hidden2={hp.get('hidden2', 128)}, dropout={hp.get('dropout', 0.3)}")

    model = TorviKMLPRegressor(
        input_dim=input_dim,
        hidden=hp.get("hidden", 256),
        hidden2=hp.get("hidden2", 128),
        dropout=hp.get("dropout", 0.3),
    )
    model.load_state_dict(state_dict)
    model.eval()

    # Run inference
    print("  Running inference...")
    with torch.no_grad():
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        output = model(X_t)  # (N, 2): [mu, raw_sigma]

    mu = output[:, 0].numpy()
    raw_sigma = output[:, 1].numpy()

    # Post-process sigma (same as Torvik infer.py)
    sigma = np.log1p(np.exp(raw_sigma)) + 1e-3  # softplus
    sigma = np.clip(sigma, 0.5, 30.0)

    # Compute home win probability from Normal(mu, sigma)
    z = mu / np.clip(sigma, 1e-6, None)
    erf_vec = np.vectorize(math.erf)
    home_win_prob = 0.5 * (1.0 + erf_vec(z / math.sqrt(2.0)))

    # Build output
    preds = info_df.copy()
    preds["torvik_pred_margin"] = mu
    preds["torvik_pred_sigma"] = sigma
    preds["torvik_home_win_prob"] = home_win_prob

    print(f"  Generated predictions for {len(preds):,} games")
    print(f"  Mean pred margin: {mu.mean():.2f}")
    print(f"  Mean sigma: {sigma.mean():.2f}")
    print(f"  MAE: {np.abs(mu - preds['actual_margin'].values).mean():.4f}")

    # Save standalone CSV
    torvik_path = PROJECT_ROOT / "predictions" / "torvik_preds_2025.csv"
    torvik_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(torvik_path, index=False)
    print(f"  Saved → {torvik_path}")

    return preds


# ── Task 3-4: Build joined head-to-head dataset ──────────────────


def build_h2h_dataset(torvik_preds: pd.DataFrame) -> pd.DataFrame:
    """Join Torvik and hoops-edge predictions by matching game identifiers."""
    print(f"\n{'='*70}")
    print("BUILDING HEAD-TO-HEAD DATASET")
    print(f"{'='*70}\n")

    # ── Load hoops-edge data ──

    # Bridge table: v2 features has gameId + team names
    print("  Loading bridge table (v2 features) for gameId ↔ team name mapping...")
    bridge = pd.read_parquet(V2_FEATURES_PATH)
    bridge = bridge[["gameId", "homeTeamId", "awayTeamId",
                      "homeTeam", "awayTeam", "startDate",
                      "homeScore", "awayScore"]].drop_duplicates(subset=["gameId"])
    print(f"  Bridge: {len(bridge)} games with gameId + team names")

    # Parse dates
    bridge["game_date"] = pd.to_datetime(bridge["startDate"], errors="coerce").dt.date

    # Map gold team names → Torvik names
    bridge["home_torvik"] = bridge["homeTeam"].map(lambda x: GOLD_TO_TORVIK.get(x, x))
    bridge["away_torvik"] = bridge["awayTeam"].map(lambda x: GOLD_TO_TORVIK.get(x, x))

    # ── Prep Torvik predictions for joining ──

    torvik = torvik_preds.copy()
    torvik["game_date"] = pd.to_datetime(torvik["date"], errors="coerce").dt.date

    # ── Multi-pass join to handle flipped teams and date ±1 (UTC offset) ──

    import datetime as _dt

    # Create order-agnostic team-pair key for both DataFrames
    def _pair_key(row, t1_col, t2_col):
        return tuple(sorted([row[t1_col], row[t2_col]]))

    bridge["_team_pair"] = bridge.apply(lambda r: _pair_key(r, "away_torvik", "home_torvik"), axis=1)
    torvik["_team_pair"] = torvik.apply(lambda r: _pair_key(r, "away_team_name", "home_team_name"), axis=1)

    # Pass 1: exact date + team pair
    print("  Pass 1: exact date + team pair...")
    merged = bridge.merge(
        torvik,
        left_on=["game_date", "_team_pair"],
        right_on=["game_date", "_team_pair"],
        how="inner",
        suffixes=("", "_torvik"),
    )
    matched_ids = set(merged["gameId"])
    print(f"    Matched: {len(merged)}")

    # Pass 2: date+1 (CBBD UTC is often +1 day vs Torvik local time)
    remaining_bridge = bridge[~bridge["gameId"].isin(matched_ids)].copy()
    remaining_bridge["game_date_m1"] = remaining_bridge["game_date"].apply(
        lambda d: d - _dt.timedelta(days=1) if d else d
    )
    pass2 = remaining_bridge.merge(
        torvik,
        left_on=["game_date_m1", "_team_pair"],
        right_on=["game_date", "_team_pair"],
        how="inner",
        suffixes=("", "_torvik"),
    )
    if len(pass2) > 0:
        # Use bridge's original game_date, drop the shifted column
        pass2 = pass2.drop(columns=["game_date_m1", "game_date_torvik"], errors="ignore")
        pass2.rename(columns={"game_date": "game_date_torvik"}, inplace=True, errors="ignore")
        matched_ids.update(pass2["gameId"])
    print(f"    Pass 2 (date-1): +{len(pass2)} matched")

    # Pass 3: date-1 (rare, but possible)
    remaining_bridge2 = bridge[~bridge["gameId"].isin(matched_ids)].copy()
    remaining_bridge2["game_date_p1"] = remaining_bridge2["game_date"].apply(
        lambda d: d + _dt.timedelta(days=1) if d else d
    )
    pass3 = remaining_bridge2.merge(
        torvik,
        left_on=["game_date_p1", "_team_pair"],
        right_on=["game_date", "_team_pair"],
        how="inner",
        suffixes=("", "_torvik"),
    )
    if len(pass3) > 0:
        pass3 = pass3.drop(columns=["game_date_p1", "game_date_torvik"], errors="ignore")
        pass3.rename(columns={"game_date": "game_date_torvik"}, inplace=True, errors="ignore")
        matched_ids.update(pass3["gameId"])
    print(f"    Pass 3 (date+1): +{len(pass3)} matched")

    # Combine all matches
    all_dfs = [df for df in [merged, pass2, pass3] if len(df) > 0]
    all_matched = pd.concat(all_dfs, ignore_index=True)
    # Deduplicate (prefer exact date match = first)
    all_matched = all_matched.drop_duplicates(subset=["gameId"], keep="first")
    print(f"  Total matched: {len(all_matched)} games")

    merged = all_matched
    # Ensure game_date column exists (might have been renamed in pass 2/3)
    if "game_date" not in merged.columns:
        # Reconstruct from startDate
        merged["game_date"] = pd.to_datetime(merged["startDate"], errors="coerce").dt.date

    # ── Load hoops-edge backtest predictions ──

    print("  Loading hoops-edge backtest predictions...")
    he_preds = pd.read_csv(BACKTEST_PATH)
    print(f"  Hoops-edge: {len(he_preds)} predictions")

    # ── Load book spreads from cached lines ──

    lines = None
    if LINES_PATH.exists():
        print("  Loading book spreads from cached lines...")
        lines_raw = pd.read_parquet(LINES_PATH)
        lines = lines_raw.sort_values("provider").drop_duplicates(
            subset=["gameId"], keep="first"
        )[["gameId", "spread"]].rename(columns={"spread": "book_spread_lines"})
        print(f"  Lines: {len(lines)} games with book spreads")

    # ── Join everything onto the bridge ──

    # Add hoops-edge predictions
    result = merged.merge(
        he_preds[["gameId", "predicted_spread", "spread_sigma", "home_win_prob"]].rename(
            columns={
                "predicted_spread": "he_pred_margin",
                "spread_sigma": "he_pred_sigma",
                "home_win_prob": "he_home_win_prob",
            }
        ),
        on="gameId",
        how="left",
    )

    # Add book spreads (from lines cache — more complete than backtest)
    if lines is not None:
        result = result.merge(lines, on="gameId", how="left")
        # Use lines book_spread as primary, fall back to backtest's
        if "book_spread" in he_preds.columns:
            he_book = he_preds[["gameId", "book_spread"]].rename(
                columns={"book_spread": "book_spread_backtest"}
            )
            result = result.merge(he_book, on="gameId", how="left")
            result["book_spread"] = result["book_spread_lines"].fillna(
                result["book_spread_backtest"]
            )
            result.drop(columns=["book_spread_lines", "book_spread_backtest"],
                        inplace=True, errors="ignore")
        else:
            result.rename(columns={"book_spread_lines": "book_spread"}, inplace=True)

    # ── Build clean output ──

    # actual_margin from bridge (gold standard)
    result["actual_margin"] = (
        pd.to_numeric(result["homeScore"], errors="coerce")
        - pd.to_numeric(result["awayScore"], errors="coerce")
    )

    output_cols = [
        "gameId", "game_date", "homeTeamId", "awayTeamId",
        "homeTeam", "awayTeam",
        "home_team_name",  # Torvik name (for debugging)
        "away_team_name",  # Torvik name
        "homeScore", "awayScore", "actual_margin",
        "he_pred_margin", "he_pred_sigma", "he_home_win_prob",
        "torvik_pred_margin", "torvik_pred_sigma", "torvik_home_win_prob",
        "book_spread",
    ]

    # Keep only columns that exist
    output_cols = [c for c in output_cols if c in result.columns]
    result = result[output_cols].copy()
    result = result.sort_values("game_date").reset_index(drop=True)

    # ── Summary stats ──

    print(f"\n  Final dataset: {len(result)} games")

    both_mask = result["he_pred_margin"].notna() & result["torvik_pred_margin"].notna()
    both = result[both_mask]
    print(f"  Both models have predictions: {len(both)} games")

    if len(both) > 0:
        he_mae = np.abs(both["he_pred_margin"] - both["actual_margin"]).mean()
        tv_mae = np.abs(both["torvik_pred_margin"] - both["actual_margin"]).mean()
        print(f"  Hoops-edge MAE: {he_mae:.4f}")
        print(f"  Torvik MAE:     {tv_mae:.4f}")

        book_mask = both["book_spread"].notna()
        if book_mask.sum() > 0:
            book_games = both[book_mask]
            he_book_mae = np.abs(book_games["he_pred_margin"] - book_games["actual_margin"]).mean()
            tv_book_mae = np.abs(book_games["torvik_pred_margin"] - book_games["actual_margin"]).mean()
            bk_mae = np.abs(-book_games["book_spread"] - book_games["actual_margin"]).mean()
            print(f"\n  On book-spread games ({len(book_games)}):")
            print(f"    Hoops-edge MAE: {he_book_mae:.4f}")
            print(f"    Torvik MAE:     {tv_book_mae:.4f}")
            print(f"    Book MAE:       {bk_mae:.4f}")

    # Report unmatched teams
    unmatched_bridge = set(bridge["home_torvik"]) | set(bridge["away_torvik"])
    unmatched_torvik = set(torvik["home_team_name"]) | set(torvik["away_team_name"])
    only_bridge = unmatched_bridge - unmatched_torvik
    only_torvik = unmatched_torvik - unmatched_bridge

    if only_bridge:
        print(f"\n  Teams in bridge (CBBD) but not in Torvik ({len(only_bridge)}):")
        for t in sorted(only_bridge)[:20]:
            print(f"    {t}")

    if only_torvik:
        print(f"\n  Teams in Torvik but not in bridge (CBBD) ({len(only_torvik)}):")
        for t in sorted(only_torvik)[:20]:
            print(f"    {t}")

    return result


# ── Save team mapping artifact ────────────────────────────────────


def save_team_mapping():
    """Save bidirectional team name mapping to artifacts."""
    mapping = {
        "gold_to_torvik": GOLD_TO_TORVIK,
        "torvik_to_gold": TORVIK_TO_GOLD,
    }
    out_path = config.ARTIFACTS_DIR / "team_name_mapping.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"  Team name mapping saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────


def main():
    engine = _mysql_engine()

    # Task 5: Document training cutoff
    document_training_cutoff(engine)

    # Task 1-2: Generate Torvik predictions
    torvik_preds = generate_torvik_predictions(engine)
    if torvik_preds.empty:
        print("No Torvik predictions generated. Exiting.")
        return

    # Task 3: Save team mapping
    print(f"\n{'='*70}")
    print("SAVING TEAM NAME MAPPING")
    print(f"{'='*70}\n")
    save_team_mapping()

    # Task 4: Build joined dataset
    result = build_h2h_dataset(torvik_preds)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Head-to-head data saved → {OUTPUT_PATH}")
    print(f"  Shape: {result.shape}")
    print("\nDone!")


if __name__ == "__main__":
    main()
