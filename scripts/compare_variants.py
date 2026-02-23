"""Compare All-Possessions vs No-Garbage-Time model variants side by side."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.architecture import MLPClassifier, MLPRegressor
from src.features import get_feature_matrix

SEASON = 2025


def load_model_and_scaler(subdir: str | None = None):
    """Load regressor, classifier, and scaler from a checkpoint directory."""
    ckpt_dir = config.CHECKPOINTS_DIR / subdir if subdir else config.CHECKPOINTS_DIR
    art_dir = config.ARTIFACTS_DIR / subdir if subdir else config.ARTIFACTS_DIR

    # Scaler
    with open(art_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Regressor
    ckpt = torch.load(ckpt_dir / "regressor.pt", map_location="cpu", weights_only=False)
    hp = ckpt.get("hparams", {})
    reg = MLPRegressor(input_dim=37, hidden1=hp.get("hidden1", 256),
                       hidden2=hp.get("hidden2", 128), dropout=hp.get("dropout", 0.3))
    reg.load_state_dict(ckpt["state_dict"])
    reg.eval()

    # Classifier
    ckpt = torch.load(ckpt_dir / "classifier.pt", map_location="cpu", weights_only=False)
    hp = ckpt.get("hparams", {})
    cls = MLPClassifier(input_dim=37, hidden1=hp.get("hidden1", 256),
                        dropout=hp.get("dropout", 0.3))
    cls.load_state_dict(ckpt["state_dict"])
    cls.eval()

    return reg, cls, scaler


@torch.no_grad()
def predict_variant(features_path: Path, subdir: str | None = None) -> pd.DataFrame:
    """Load features, run inference, return predictions with scores."""
    df = pd.read_parquet(features_path)
    df = df.dropna(subset=["homeScore", "awayScore"])

    reg, cls, scaler = load_model_and_scaler(subdir)

    X = get_feature_matrix(df).values.astype(np.float32)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        col_means = scaler.mean_
        for j in range(X.shape[1]):
            X[nan_mask[:, j], j] = col_means[j]

    X_scaled = scaler.transform(X)
    X_t = torch.tensor(X_scaled, dtype=torch.float32)

    mu, raw_sigma = reg(X_t)
    sigma = torch.nn.functional.softplus(raw_sigma) + 1e-3
    sigma = sigma.clamp(min=0.5, max=30.0)

    out = df[["gameId", "homeScore", "awayScore"]].copy()
    out["predicted_spread"] = mu.numpy()
    out["spread_sigma"] = sigma.numpy()
    out["actual_margin"] = out["homeScore"] - out["awayScore"]
    return out


def attach_lines(preds: pd.DataFrame) -> pd.DataFrame:
    """Merge book spreads from S3 fct_lines."""
    from src import s3_reader
    lines_tbl = s3_reader.read_silver_table("fct_lines", season=SEASON)
    if lines_tbl.num_rows == 0:
        preds["book_spread"] = np.nan
        return preds
    lines = lines_tbl.to_pandas()
    lines = lines.sort_values("provider").drop_duplicates(subset=["gameId"], keep="first")
    preds = preds.merge(lines[["gameId", "spread"]].rename(columns={"spread": "book_spread"}),
                        on="gameId", how="left")
    preds["model_spread"] = -preds["predicted_spread"]
    preds["spread_diff"] = preds["model_spread"] - preds["book_spread"]
    return preds


def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute MAE and ROI metrics for a predictions DataFrame."""
    with_book = df.dropna(subset=["book_spread"]).copy()

    # MAE
    mae = np.abs(df["predicted_spread"] - df["actual_margin"]).mean()

    # Sigma stats
    median_sigma = with_book["spread_sigma"].median()
    p25_sigma = with_book["spread_sigma"].quantile(0.25)

    results = {"MAE": mae, "n_games": len(df), "n_with_book": len(with_book)}

    # ROI at various thresholds and sigma filters
    for label, sigma_cut in [("unfilt", None), ("sig_med", median_sigma), ("sig_p25", p25_sigma)]:
        subset = with_book.copy()
        if sigma_cut is not None:
            subset = subset[subset["spread_sigma"] < sigma_cut]

        for thresh in [3, 5]:
            bets = subset[subset["spread_diff"].abs() > thresh]
            if len(bets) == 0:
                results[f"{label}_roi_{thresh}"] = None
                results[f"{label}_n_{thresh}"] = 0
                continue

            wins = 0
            losses = 0
            for _, row in bets.iterrows():
                cover = row["actual_margin"] + row["book_spread"]
                if row["spread_diff"] < 0:  # bet HOME
                    if cover > 0:
                        wins += 1
                    elif cover < 0:
                        losses += 1
                else:  # bet AWAY
                    if cover < 0:
                        wins += 1
                    elif cover > 0:
                        losses += 1

            n_bets = wins + losses
            roi = (wins * (100 / 110) - losses) / max(n_bets, 1) * 100 if n_bets > 0 else 0
            results[f"{label}_roi_{thresh}"] = roi
            results[f"{label}_n_{thresh}"] = n_bets

    return results


def main():
    print(f"=== Comparing All-Possessions vs No-Garbage-Time — Season {SEASON} ===\n")

    # Predict with both variants
    all_poss_path = config.FEATURES_DIR / f"season_{SEASON}_features.parquet"
    no_garb_path = config.FEATURES_DIR / f"season_{SEASON}_no_garbage_features.parquet"

    if not all_poss_path.exists():
        print(f"ERROR: {all_poss_path} not found. Run: build-features --season {SEASON}")
        return
    if not no_garb_path.exists():
        print(f"ERROR: {no_garb_path} not found. Run: build-features --season {SEASON} --no-garbage")
        return

    print("Running All-Possessions model...")
    preds_all = predict_variant(all_poss_path, subdir=None)
    preds_all = attach_lines(preds_all)
    m_all = compute_metrics(preds_all)

    print("Running No-Garbage model...")
    preds_ng = predict_variant(no_garb_path, subdir="no_garbage")
    preds_ng = attach_lines(preds_ng)
    m_ng = compute_metrics(preds_ng)

    # Print comparison table
    print(f"\n{'='*72}")
    print(f"  VARIANT COMPARISON — Season {SEASON}")
    print(f"{'='*72}")
    print(f"  Games: {m_all['n_games']} (all poss) / {m_ng['n_games']} (no garbage)")
    print(f"  Games with book spread: {m_all['n_with_book']} / {m_ng['n_with_book']}")
    print()

    header = f"  {'Metric':<22} {'All Possessions':>17} {'No Garbage':>17} {'Winner':>10}"
    print(header)
    print(f"  {'-'*22} {'-'*17} {'-'*17} {'-'*10}")

    rows = [
        ("MAE", m_all["MAE"], m_ng["MAE"], True),  # lower is better
        ("Unfilt ROI @3", m_all.get("unfilt_roi_3"), m_ng.get("unfilt_roi_3"), False),
        ("Unfilt ROI @5", m_all.get("unfilt_roi_5"), m_ng.get("unfilt_roi_5"), False),
        ("sigma<med ROI @3", m_all.get("sig_med_roi_3"), m_ng.get("sig_med_roi_3"), False),
        ("sigma<med ROI @5", m_all.get("sig_med_roi_5"), m_ng.get("sig_med_roi_5"), False),
        ("sigma<p25 ROI @3", m_all.get("sig_p25_roi_3"), m_ng.get("sig_p25_roi_3"), False),
        ("sigma<p25 ROI @5", m_all.get("sig_p25_roi_5"), m_ng.get("sig_p25_roi_5"), False),
    ]

    for label, v_all, v_ng, lower_better in rows:
        if v_all is None or v_ng is None:
            s_all = "N/A"
            s_ng = "N/A"
            winner = "—"
        elif label == "MAE":
            s_all = f"{v_all:.2f}"
            s_ng = f"{v_ng:.2f}"
            winner = "All Poss" if v_all < v_ng else ("No Garb" if v_ng < v_all else "TIE")
        else:
            # ROI — include bet count
            n_all = m_all.get(f"{label.split()[0].lower().replace('<', '_').replace('sigma', 'sig')}_n_{label.split()[-1].replace('@', '')}", "?")
            # Simpler: just use the key pattern
            key_prefix = "unfilt" if "Unfilt" in label else ("sig_med" if "med" in label else "sig_p25")
            thresh = label.split("@")[-1]
            n_a = m_all.get(f"{key_prefix}_n_{thresh}", "?")
            n_g = m_ng.get(f"{key_prefix}_n_{thresh}", "?")
            s_all = f"{v_all:+.1f}% ({n_a})"
            s_ng = f"{v_ng:+.1f}% ({n_g})"
            winner = "All Poss" if v_all > v_ng else ("No Garb" if v_ng > v_all else "TIE")

        print(f"  {label:<22} {s_all:>17} {s_ng:>17} {winner:>10}")

    # Summary
    print(f"\n{'='*72}")
    all_wins = sum(1 for _, va, vn, lb in rows if va is not None and vn is not None and
                   ((lb and va < vn) or (not lb and va > vn)))
    ng_wins = sum(1 for _, va, vn, lb in rows if va is not None and vn is not None and
                  ((lb and vn < va) or (not lb and vn > va)))
    print(f"  All Possessions wins: {all_wins}/7")
    print(f"  No Garbage wins:     {ng_wins}/7")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
