"""Load trained models and produce predictions."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch

from . import config
from .architecture import MLPClassifier, MLPRegressor, MLPRegressorSplit
from .trainer import load_scaler


# ── Betting math helpers ──────────────────────────────────────────────


def normal_cdf(z):
    """Standard normal CDF using math.erf."""
    z = np.asarray(z, dtype=float)
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(z / math.sqrt(2.0)))


def american_to_breakeven(odds):
    """Convert American odds to break-even probability. -110 → 0.5238."""
    o = np.asarray(odds, dtype=float)
    out = np.full_like(o, np.nan, dtype=float)
    neg = o < 0
    pos = o > 0
    out[neg] = (-o[neg]) / ((-o[neg]) + 100.0)
    out[pos] = 100.0 / (o[pos] + 100.0)
    return out


def american_profit_per_1(odds):
    """Profit per $1 staked if the bet wins. -110 → 0.9091."""
    o = np.asarray(odds, dtype=float)
    out = np.full_like(o, np.nan, dtype=float)
    neg = o < 0
    pos = o > 0
    out[neg] = 100.0 / (-o[neg])
    out[pos] = o[pos] / 100.0
    return out


def prob_to_american(p):
    """Convert probability to fair American odds (no vig)."""
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    out = np.full_like(p, np.nan, dtype=float)
    fav = p >= 0.5
    dog = ~fav
    out[fav] = -100.0 * (p[fav] / (1.0 - p[fav]))
    out[dog] = 100.0 * ((1.0 - p[dog]) / p[dog])
    return out


def load_regressor(path: Path | None = None) -> tuple[MLPRegressor | MLPRegressorSplit, dict, list[str], str]:
    """Load regressor from checkpoint, auto-detecting architecture type.

    Returns:
        (model, hparams, feature_order, sigma_param) where sigma_param is
        "exp" (new) or "softplus" (legacy).
    """
    if path is None:
        path = config.CHECKPOINTS_DIR / "regressor.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hparams", {})
    feature_order = ckpt.get("feature_order", config.FEATURE_ORDER)
    arch_type = ckpt.get("arch_type", "shared")
    sigma_param = ckpt.get("sigma_param", "softplus")

    ModelClass = MLPRegressorSplit if arch_type == "split" else MLPRegressor
    model = ModelClass(
        input_dim=len(feature_order),
        hidden1=hp.get("hidden1", 384),
        hidden2=hp.get("hidden2", 256),
        dropout=hp.get("dropout", 0.2),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, hp, feature_order, sigma_param


def load_classifier(path: Path | None = None) -> tuple[MLPClassifier, dict, list[str]]:
    """Load MLPClassifier from checkpoint.

    Returns:
        (model, hparams, feature_order) — feature_order from the checkpoint.
    """
    if path is None:
        path = config.CHECKPOINTS_DIR / "classifier.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hparams", {})
    feature_order = ckpt.get("feature_order", config.FEATURE_ORDER)
    model = MLPClassifier(
        input_dim=len(feature_order),
        hidden1=hp.get("hidden1", 384),
        dropout=hp.get("dropout", 0.2),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, hp, feature_order


@torch.no_grad()
def predict(
    features_df: pd.DataFrame,
    lines_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate predictions for a feature DataFrame.

    Args:
        features_df: DataFrame with feature columns (per FEATURE_ORDER) + metadata.
        lines_df: Optional lines data to attach spread/moneyline info.

    Returns:
        DataFrame with predictions: mu, sigma, home_win_prob, plus edge metrics.
    """
    scaler = load_scaler()
    regressor, _, reg_feature_order, sigma_param = load_regressor()
    classifier, _, cls_feature_order = load_classifier()

    # Validate classifier and regressor were trained on the same features
    assert cls_feature_order == reg_feature_order, (
        f"Feature order mismatch: regressor has {len(reg_feature_order)} features, "
        f"classifier has {len(cls_feature_order)}. Models must be retrained together."
    )

    # Use the feature order embedded in the checkpoint — ensures compatibility
    # even if config.FEATURE_ORDER has changed since the model was trained.
    feature_order = reg_feature_order
    X = features_df[feature_order].values.astype(np.float32)

    # Validate feature count matches model input dimension
    assert X.shape[1] == len(feature_order), (
        f"Expected {len(feature_order)} features, got {X.shape[1]}"
    )

    # Handle NaN: fill with column means (from scaler)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        col_means = scaler.mean_
        for j in range(X.shape[1]):
            X[nan_mask[:, j], j] = col_means[j]

    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Regressor: (mu, log_sigma)
    mu_raw, log_sigma_raw = regressor(X_tensor)
    if sigma_param == "exp":
        sigma = torch.exp(log_sigma_raw).clamp(min=0.5, max=30.0)
    else:
        # Legacy softplus parameterization
        sigma = torch.nn.functional.softplus(log_sigma_raw) + 1e-3
        sigma = sigma.clamp(min=0.5, max=30.0)

    mu = mu_raw.numpy()
    sigma = sigma.numpy()

    # Classifier: home_win_prob
    logits = classifier(X_tensor)
    home_win_prob = torch.sigmoid(logits).numpy()

    # Build output — include team names if available
    meta_cols = ["gameId", "homeTeamId", "awayTeamId"]
    if "homeTeam" in features_df.columns:
        meta_cols.append("homeTeam")
    if "awayTeam" in features_df.columns:
        meta_cols.append("awayTeam")
    meta_cols.append("startDate")
    if "neutralSite" in features_df.columns:
        meta_cols.append("neutralSite")
    out = features_df[meta_cols].copy()
    if "neutralSite" in out.columns:
        out = out.rename(columns={"neutralSite": "neutral_site"})
    out["predicted_spread"] = mu
    out["spread_sigma"] = sigma
    out["home_win_prob"] = home_win_prob
    out["away_win_prob"] = 1.0 - home_win_prob

    # Attach lines if available
    if lines_df is not None and not lines_df.empty:
        # Fix flipped spread signs using cross-provider majority vote.
        # ~1-2% of games have one provider with a flipped spread sign.
        # When multiple providers exist, use the majority sign as truth.
        lines_fixed = lines_df.copy()
        lines_fixed["spread"] = pd.to_numeric(lines_fixed["spread"], errors="coerce")
        lines_fixed["homeMoneyline"] = pd.to_numeric(
            lines_fixed["homeMoneyline"], errors="coerce"
        )

        # Compute majority spread sign per game (positive count vs negative)
        has_spread = lines_fixed["spread"].notna() & (lines_fixed["spread"] != 0)
        spread_sign = np.sign(lines_fixed.loc[has_spread, "spread"])
        majority_sign = (
            spread_sign.groupby(lines_fixed.loc[has_spread, "gameId"])
            .sum()  # net sign: positive if more providers say positive
            .rename("_majority_sign")
        )

        # Pick preferred provider: Draft Kings > ESPN BET > Bovada,
        # with completeness (has spread/total) as the top tiebreaker.
        _provider_rank = {"Draft Kings": 0, "ESPN BET": 1, "Bovada": 2}
        lines_dedup = (
            lines_fixed
            .assign(
                _has_spread=lines_fixed["spread"].notna().astype(int),
                _has_total=lines_fixed["overUnder"].notna().astype(int),
                _prov_rank=lines_fixed["provider"].map(_provider_rank).fillna(99),
            )
            .sort_values(
                ["_has_spread", "_has_total", "_prov_rank"],
                ascending=[False, False, True],
            )
            .drop_duplicates(subset=["gameId"], keep="first")
            .drop(columns=["_has_spread", "_has_total", "_prov_rank"])
            .copy()
        )

        # Merge majority sign and flip when the selected provider disagrees.
        # Only flip spreads >= 3 pts — smaller spreads can legitimately
        # differ in sign across providers for near pick'em games.
        lines_dedup = lines_dedup.merge(majority_sign, on="gameId", how="left")
        _sp = lines_dedup["spread"]
        _maj = lines_dedup["_majority_sign"]
        mask_majority_flip = (
            _sp.notna() & _maj.notna() & (_maj != 0)
            & (abs(_sp) >= 3)
            & (np.sign(_sp) != np.sign(_maj))
        )
        lines_dedup.loc[mask_majority_flip, "spread"] = -_sp[mask_majority_flip]

        # Fallback: for single-provider games, use moneyline as cross-check
        _sp2 = lines_dedup["spread"]
        _ml = lines_dedup["homeMoneyline"]
        mask_ml_fix = (
            _sp2.notna() & _ml.notna()
            & (~mask_majority_flip)  # not already fixed by majority
            & _maj.isna()  # single provider (no majority available) or no spread
            & (
                ((_sp2 > 3) & (_ml < -150))
                | ((_sp2 < -3) & (_ml > 150))
            )
        )
        lines_dedup.loc[mask_ml_fix, "spread"] = -_sp2[mask_ml_fix]
        lines_dedup = lines_dedup.drop(columns=["_majority_sign"])

        lines_dedup = lines_dedup.rename(columns={
            "spread": "book_spread",
            "overUnder": "book_total",
            "homeMoneyline": "home_moneyline",
            "awayMoneyline": "away_moneyline",
        })
        merge_cols = ["gameId", "book_spread", "book_total", "home_moneyline", "away_moneyline"]
        available = [c for c in merge_cols if c in lines_dedup.columns]
        out = out.merge(lines_dedup[available], on="gameId", how="left")

        # Layer 3: Model-based cross-check for phantom edges.
        # When book_spread and predicted_spread both say home wins (or both
        # say away wins) with combined edge > 9 pts, the spread sign is wrong.
        if "book_spread" in out.columns:
            _bs = out["book_spread"]
            _ps = out["predicted_spread"]
            _phantom = _bs + _ps
            mask_phantom = (
                _bs.notna() & _ps.notna()
                & (
                    ((_bs > 0) & (_ps > 0) & (_phantom >= 9))
                    | ((_bs < 0) & (_ps < 0) & (_phantom <= -9))
                )
            )
            out.loc[mask_phantom, "book_spread"] = -_bs[mask_phantom]

        # Edge metrics
        if "book_spread" in out.columns:
            # book_spread is from home perspective (negative = home favored)
            # predicted_spread is home_pts - away_pts (positive = home favored)
            # Convert predicted_spread to book convention: negate it
            out["model_spread"] = -out["predicted_spread"]
            out["spread_diff"] = out["model_spread"] - out["book_spread"]

            # Edge calculations (sign: predicted_spread positive=home, book_spread negative=home)
            out["edge_home_points"] = out["predicted_spread"] + out["book_spread"]

            sigma_safe = out["spread_sigma"].clip(lower=0.5)
            edge_z = out["edge_home_points"] / sigma_safe
            home_cover_prob = normal_cdf(edge_z)
            away_cover_prob = 1.0 - home_cover_prob

            out["pick_side"] = np.where(out["edge_home_points"] >= 0, "HOME", "AWAY")
            out["pick_cover_prob"] = np.where(
                out["edge_home_points"] >= 0, home_cover_prob, away_cover_prob
            )

            # Default -110 spread odds
            pick_spread_odds = -110
            pick_breakeven = float(american_to_breakeven(np.array([pick_spread_odds]))[0])
            pick_profit = float(american_profit_per_1(np.array([pick_spread_odds]))[0])

            out["pick_spread_odds"] = pick_spread_odds
            out["pick_prob_edge"] = out["pick_cover_prob"] - pick_breakeven
            out["pick_ev_per_1"] = out["pick_cover_prob"] * pick_profit - (1.0 - out["pick_cover_prob"])
            out["pick_fair_odds"] = prob_to_american(out["pick_cover_prob"].values)

    return out


def _slugify(text: str) -> str:
    """Lowercase and replace non-alphanum with underscores."""
    import re
    return re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")


def _to_native(v):
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v) if not np.isnan(v) else None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


# Column mapping from internal DataFrame names to site JSON field names
_SITE_FIELD_MAP = {
    "awayTeam": "away_team",
    "homeTeam": "home_team",
    "neutral_site": "neutral_site",
    "book_spread": "market_spread_home",
    "predicted_spread": "model_mu_home",
    "spread_sigma": "pred_sigma",
    "edge_home_points": "edge_home_points",
    "home_win_prob": "pred_home_win_prob",
    "pick_side": "pick_side",
    "pick_cover_prob": "pick_cover_prob",
    "pick_prob_edge": "pick_prob_edge",
    "pick_ev_per_1": "pick_ev_per_1",
    "pick_spread_odds": "pick_spread_odds",
    "pick_fair_odds": "pick_fair_odds",
    "startDate": "start_time",
}


def save_predictions(preds: pd.DataFrame, game_date: str | None = None) -> tuple[Path, Path]:
    """Save predictions as JSON, CSV, and site-compatible JSON.

    Saves to:
      - predictions/json/{game_date}.json (raw records)
      - predictions/preds_today.csv
      - predictions/csv/preds_YYYY_M_D_edge.csv (dated, bball convention)
      - site/public/data/predictions_{game_date}.json (site-compatible format)

    Returns:
        (json_path, csv_path)
    """
    if game_date is None:
        game_date = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    json_dir = config.PREDICTIONS_DIR / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = config.PREDICTIONS_DIR / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    json_path = json_dir / f"{game_date}.json"
    csv_path = config.PREDICTIONS_DIR / "preds_today.csv"

    # Dated CSV path: preds_YYYY_M_D_edge.csv
    try:
        parts = game_date.split("-")
        if len(parts) == 3:
            y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
            dated_csv_path = csv_dir / f"preds_{y}_{m}_{d}_edge.csv"
        else:
            dated_csv_path = csv_dir / f"preds_{game_date}_edge.csv"
    except (ValueError, IndexError):
        dated_csv_path = csv_dir / f"preds_{game_date}_edge.csv"

    # Raw JSON output
    records = preds.to_dict(orient="records")
    for rec in records:
        for k, v in list(rec.items()):
            rec[k] = _to_native(v)
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2, default=str)

    # CSV output
    preds.to_csv(csv_path, index=False)
    preds.to_csv(dated_csv_path, index=False)

    # Site-compatible JSON (replaces csv_to_json.py pipeline)
    config.SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    site_games = []
    for rec in records:
        game = {}
        for src_col, dst_col in _SITE_FIELD_MAP.items():
            if src_col in rec:
                game[dst_col] = rec[src_col]
        # Generate stable game_id from date + team names
        away = str(game.get("away_team") or "")
        home = str(game.get("home_team") or "")
        game["game_id"] = _slugify(f"{game_date}_{away}_{home}")
        site_games.append(game)

    site_payload = {
        "date": game_date,
        "generated_at": datetime.now(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z"),
        "games": site_games,
    }
    site_json_path = config.SITE_DATA_DIR / f"predictions_{game_date}.json"
    with open(site_json_path, "w") as f:
        json.dump(site_payload, f, indent=2, sort_keys=True)
        f.write("\n")

    return json_path, csv_path
