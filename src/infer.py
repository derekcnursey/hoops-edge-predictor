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
from .efficiency_blend import blend_enabled
from .line_selection import select_preferred_lines
from .mean_model_variants import (
    LEGACY_HOME_SLOT,
    TEAM_AB_ELITE_TAIL_ROUND64_V1,
    active_mean_model_variant,
    build_mean_model_feature_frame,
    build_team_ab_elite_tail_round64_contract,
    build_team_ab_source,
    legacy_variant_field,
    normalize_mean_model_variant,
    swap_team_ab_source,
    variant_prediction_field,
    variant_spec,
)
from .sigma_calibration import apply_sigma_transform
from .tournament_adjustments import (
    add_tournament_market_display_columns,
    tournament_primary_weights,
)
from .trainer import load_scaler, load_tree_regressor


_TORCH_INFERENCE_THREADS_CONFIGURED = False


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


def load_mu_regressor(
    variant: str | None = None,
    *,
    secondary: bool = False,
) -> tuple[object, list[str], str, dict]:
    """Load the preferred point regressor for mu.

    Prefers the production HistGradientBoosting artifact when present, falling
    back to the legacy MLP regressor checkpoint otherwise.
    """
    spec = variant_spec(variant or active_mean_model_variant())
    tree_path = spec.torvik_checkpoint_path if secondary else spec.checkpoint_path
    if tree_path is None:
        raise FileNotFoundError(
            f"No {'secondary ' if secondary else ''}mu artifact configured for variant {spec.name}"
        )
    if tree_path.exists():
        model, feature_order, meta = load_tree_regressor(tree_path)
        return model, feature_order, meta.get("model_type", "hist_gradient_boosting"), meta

    if spec.name != "legacy_home_slot" or secondary:
        raise FileNotFoundError(
            f"Missing {'secondary ' if secondary else ''}mu artifact for variant {spec.name}: {tree_path}"
        )

    regressor, _, feature_order, _ = load_regressor()
    return regressor, feature_order, "mlp", {}


def load_torvik_mu_regressor(
    variant: str | None = None,
) -> tuple[object, list[str], str, dict] | None:
    """Load the secondary Torvik-backed point regressor when available."""
    try:
        spec = variant_spec(variant or active_mean_model_variant())
    except ValueError:
        return None
    tree_path = spec.torvik_checkpoint_path
    if tree_path is None or not tree_path.exists():
        return None
    model, feature_order, meta = load_tree_regressor(tree_path)
    return model, feature_order, meta.get("model_type", "hist_gradient_boosting"), meta


def _swap_feature_frame(features_df: pd.DataFrame, feature_order: list[str]) -> pd.DataFrame:
    """Swap home/away feature slots for neutral-site symmetrization.

    The trained feature contract is not perfectly mirrored by name, so this
    helper handles:
    - generic ``home_*`` <-> ``away_*`` pairs
    - sign flip for ``rest_advantage``
    - slot-specific one-offs preserved as swap pairs
    - forcing ``home_team_hca`` to zero for neutral games
    """
    swapped = features_df[feature_order].copy()
    used: set[str] = set()
    explicit_pairs = [
        ("home_opp_ft_rate", "away_def_ft_rate"),
        ("home_team_efg_home_split", "away_team_efg_away_split"),
    ]
    for left, right in explicit_pairs:
        if left in swapped.columns and right in swapped.columns:
            tmp = swapped[left].copy()
            swapped[left] = swapped[right]
            swapped[right] = tmp
            used.add(left)
            used.add(right)

    for col in feature_order:
        if col in used:
            continue
        if col.startswith("home_"):
            other = "away_" + col[len("home_") :]
            if other in swapped.columns:
                tmp = swapped[col].copy()
                swapped[col] = swapped[other]
                swapped[other] = tmp
                used.add(col)
                used.add(other)

    if "rest_advantage" in swapped.columns:
        swapped["rest_advantage"] = -swapped["rest_advantage"]
    if "home_team_hca" in swapped.columns:
        swapped["home_team_hca"] = 0.0
    if "neutral_site" in swapped.columns:
        swapped["neutral_site"] = 1.0

    return swapped


def _fill_nan_with_scaler_means(X_df: pd.DataFrame, scaler) -> np.ndarray:
    """Fill missing values with the scaler's training means."""
    X = X_df.values.astype(np.float32)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        col_means = scaler.mean_
        for j in range(X.shape[1]):
            X[nan_mask[:, j], j] = col_means[j]
    return X


def _fill_nan_with_impute_means(X_df: pd.DataFrame, impute_means: np.ndarray | None) -> np.ndarray:
    """Fill missing values with persisted tree-training means when available."""
    X = X_df.values.astype(np.float32)
    if impute_means is None:
        return X
    means = np.asarray(impute_means, dtype=np.float32)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        for j in range(X.shape[1]):
            X[nan_mask[:, j], j] = means[j]
    return X


def _predict_mu_values(mu_regressor: object, mu_model_type: str, X_raw: np.ndarray, X_scaled: np.ndarray) -> np.ndarray:
    """Run the configured mean model on raw/scaled features."""
    if mu_model_type != "mlp":
        return mu_regressor.predict(X_raw).astype(np.float32)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    mu_raw, _ = mu_regressor(X_tensor)
    return mu_raw.numpy()


def _predict_mu_branch(
    features_df: pd.DataFrame,
    *,
    variant: str,
    scaler,
    secondary_mu_features_df: pd.DataFrame | None = None,
) -> np.ndarray:
    """Predict mu for a specific mean-model variant."""
    normalized = normalize_mean_model_variant(variant)
    spec = variant_spec(normalized)
    mu_regressor, mu_feature_order, mu_model_type, mu_meta = load_mu_regressor(normalized)

    if normalized == TEAM_AB_ELITE_TAIL_ROUND64_V1:
        source = build_team_ab_source(features_df)
        mu_frame = build_team_ab_elite_tail_round64_contract(source)
        X_df = mu_frame[mu_feature_order].copy()
        X_raw = _fill_nan_with_impute_means(X_df, mu_meta.get("impute_means"))
        mu = _predict_mu_values(mu_regressor, mu_model_type, X_raw, X_raw)
        neutral_mask = (
            pd.to_numeric(source["neutral_site"], errors="coerce").fillna(0.0).to_numpy()
            == 1.0
        )
        if neutral_mask.any():
            neutral_idx = np.flatnonzero(neutral_mask)
            swap_source = swap_team_ab_source(source.iloc[neutral_idx].reset_index(drop=True))
            swap_frame = build_team_ab_elite_tail_round64_contract(swap_source)
            swap_X_df = swap_frame[mu_feature_order].copy()
            swap_X_raw = _fill_nan_with_impute_means(swap_X_df, mu_meta.get("impute_means"))
            mu_swap = _predict_mu_values(mu_regressor, mu_model_type, swap_X_raw, swap_X_raw)
            mu = np.asarray(mu, dtype=np.float32)
            mu[neutral_idx] = (mu[neutral_idx] - mu_swap) / 2.0
    else:
        mu_frame = build_mean_model_feature_frame(features_df, normalized)
        X_df = mu_frame[mu_feature_order].copy()
        X_raw = _fill_nan_with_scaler_means(X_df, scaler)
        X_scaled = scaler.transform(X_raw)
        mu_primary_raw = _predict_mu_values(mu_regressor, mu_model_type, X_raw, X_scaled)
        mu = _symmetrize_neutral_mu(
            mu_primary_raw,
            features_df,
            mu_feature_order,
            scaler,
            mu_regressor,
            mu_model_type,
        ) if spec.use_legacy_full_sym else mu_primary_raw

        if spec.use_mu_blend and blend_enabled() and secondary_mu_features_df is not None:
            secondary_loaded = load_torvik_mu_regressor(normalized)
            if secondary_loaded is not None:
                sec_regressor, sec_feature_order, sec_model_type, _sec_meta = secondary_loaded
                sec_aligned = (
                    secondary_mu_features_df.set_index("gameId")
                    .reindex(features_df["gameId"])
                    .reset_index()
                )
                sec_df = sec_aligned[sec_feature_order].copy()
                sec_X = _fill_nan_with_scaler_means(sec_df, scaler)
                sec_X_scaled = scaler.transform(sec_X)
                mu_torvik = _predict_mu_values(sec_regressor, sec_model_type, sec_X, sec_X_scaled)
                if spec.use_legacy_full_sym:
                    mu_torvik = _symmetrize_neutral_mu(
                        mu_torvik,
                        sec_aligned,
                        sec_feature_order,
                        scaler,
                        sec_regressor,
                        sec_model_type,
                    )
                gold_w = tournament_primary_weights(features_df)
                mu = gold_w * mu + (1.0 - gold_w) * mu_torvik
    return np.asarray(mu, dtype=np.float32)


def _symmetrize_neutral_mu(
    mu: np.ndarray,
    features_df: pd.DataFrame,
    feature_order: list[str],
    scaler,
    mu_regressor: object,
    mu_model_type: str,
) -> np.ndarray:
    """Enforce anti-symmetry for neutral-site point predictions."""
    neutral_col = None
    if "neutral_site" in features_df.columns:
        neutral_col = "neutral_site"
    elif "neutralSite" in features_df.columns:
        neutral_col = "neutralSite"
    if neutral_col is None:
        return mu
    neutral_mask = features_df[neutral_col].fillna(0).astype(float).to_numpy() == 1.0
    if not neutral_mask.any():
        return mu

    X_df = features_df[feature_order].copy()
    neutral_idx = np.flatnonzero(neutral_mask)
    X_swap_df = _swap_feature_frame(X_df.iloc[neutral_idx], feature_order)
    X_swap = _fill_nan_with_scaler_means(X_swap_df, scaler)
    X_swap_scaled = scaler.transform(X_swap)
    mu_swap = _predict_mu_values(mu_regressor, mu_model_type, X_swap, X_swap_scaled)

    mu_out = mu.copy()
    mu_orig = mu_out[neutral_idx].copy()
    mu_out[neutral_idx] = (mu_orig - mu_swap) / 2.0
    return mu_out


def _postprocess_sigma(sigma: torch.Tensor) -> torch.Tensor:
    """Apply optional inference-time sigma sharpening while preserving positivity."""
    mode = config.SIGMA_CALIBRATION_MODE or ("cap" if config.SIGMA_CAP_MAX is not None else None)
    sigma_np = apply_sigma_transform(
        sigma.detach().cpu().numpy(),
        mode=mode,
        cap_max=config.SIGMA_CAP_MAX,
        scale=config.SIGMA_SCALE,
        affine_a=config.SIGMA_AFFINE_A,
        affine_b=config.SIGMA_AFFINE_B,
        shrink_alpha=config.SIGMA_SHRINK_ALPHA,
        shrink_target=config.SIGMA_SHRINK_TARGET,
    ).astype(np.float32)
    return torch.from_numpy(sigma_np).to(sigma.device)


def _configure_torch_inference_threads() -> None:
    """Cap CPU inference threading to avoid libomp stalls on small live slates."""
    global _TORCH_INFERENCE_THREADS_CONFIGURED
    if _TORCH_INFERENCE_THREADS_CONFIGURED:
        return
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Safe to ignore if PyTorch has already initialized the inter-op pool.
        pass
    _TORCH_INFERENCE_THREADS_CONFIGURED = True


@torch.no_grad()
def predict(
    features_df: pd.DataFrame,
    lines_df: pd.DataFrame | None = None,
    secondary_mu_features_df: pd.DataFrame | None = None,
    team_ab_features_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate predictions for a feature DataFrame.

    Args:
        features_df: DataFrame with feature columns (per FEATURE_ORDER) + metadata.
        lines_df: Optional lines data to attach spread/moneyline info.

    Returns:
        DataFrame with predictions: mu, sigma, home_win_prob, plus edge metrics.
    """
    _configure_torch_inference_threads()
    scaler = load_scaler()
    regressor, _, reg_feature_order, sigma_param = load_regressor()
    classifier, _, cls_feature_order = load_classifier()

    # Sigma and classifier remain on the legacy shared contract.
    assert cls_feature_order == reg_feature_order, (
        f"Feature order mismatch: sigma regressor={len(reg_feature_order)}, "
        f"classifier={len(cls_feature_order)}. "
        f"Legacy sigma/probability heads must be trained together."
    )

    feature_order = reg_feature_order
    X_df = features_df[feature_order].copy()

    # Validate feature count matches model input dimension
    assert X_df.shape[1] == len(feature_order), (
        f"Expected {len(feature_order)} features, got {X_df.shape[1]}"
    )

    X = _fill_nan_with_scaler_means(X_df, scaler)
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    legacy_mu = _predict_mu_branch(
        features_df,
        variant="legacy_home_slot",
        scaler=scaler,
        secondary_mu_features_df=secondary_mu_features_df,
    )
    active_variant = active_mean_model_variant()
    variant_predictions: dict[str, np.ndarray] = {legacy_variant_field(): legacy_mu}
    team_ab_field = variant_prediction_field(TEAM_AB_ELITE_TAIL_ROUND64_V1)
    try:
        team_ab_input_df = features_df
        if team_ab_features_df is not None:
            aligned = (
                team_ab_features_df.set_index("gameId")
                .reindex(features_df["gameId"])
                .reset_index()
            )
            payload_cols = [col for col in aligned.columns if col != "gameId"]
            missing_mask = aligned[payload_cols].isna().all(axis=1) if payload_cols else pd.Series(False, index=aligned.index)
            if bool(missing_mask.any()):
                missing = aligned.loc[missing_mask, "gameId"].tolist()
                raise ValueError(
                    "team_ab_features_df is missing rows for one or more gameIds "
                    f"from the primary feature frame: {missing}"
                )
            team_ab_input_df = aligned
        variant_predictions[team_ab_field] = _predict_mu_branch(
            team_ab_input_df,
            variant=TEAM_AB_ELITE_TAIL_ROUND64_V1,
            scaler=scaler,
            secondary_mu_features_df=secondary_mu_features_df,
        )
    except FileNotFoundError:
        if active_variant == TEAM_AB_ELITE_TAIL_ROUND64_V1:
            raise

    active_field = variant_prediction_field(active_variant)
    mu = variant_predictions.get(active_field, legacy_mu)

    # MLP regressor remains the uncertainty model for sigma.
    _, log_sigma_raw = regressor(X_tensor)
    if sigma_param == "exp":
        sigma = _postprocess_sigma(torch.exp(log_sigma_raw))
    else:
        # Legacy softplus parameterization
        sigma = _postprocess_sigma(torch.nn.functional.softplus(log_sigma_raw) + 1e-3)

    sigma = sigma.numpy()

    # Classifier: home_win_prob
    logits = classifier(X_tensor)
    home_win_prob = torch.sigmoid(logits).numpy()

    # Neutral-site safety: symmetrize slot orientation explicitly for sigma/probability.
    neutral_col = None
    if "neutral_site" in features_df.columns:
        neutral_col = "neutral_site"
    elif "neutralSite" in features_df.columns:
        neutral_col = "neutralSite"
    if neutral_col is not None:
        neutral_mask = features_df[neutral_col].fillna(0).astype(float).to_numpy() == 1.0
        if neutral_mask.any():
            neutral_idx = np.flatnonzero(neutral_mask)
            X_swap_df = _swap_feature_frame(X_df.iloc[neutral_idx], feature_order)
            X_swap = _fill_nan_with_scaler_means(X_swap_df, scaler)
            X_swap_scaled = scaler.transform(X_swap)
            X_swap_tensor = torch.tensor(X_swap_scaled, dtype=torch.float32)

            legacy_regressor, _legacy_feature_order, legacy_model_type, _legacy_meta = load_mu_regressor(LEGACY_HOME_SLOT)
            mu_swap = _predict_mu_values(legacy_regressor, legacy_model_type, X_swap, X_swap_scaled)

            _, log_sigma_swap_raw = regressor(X_swap_tensor)
            if sigma_param == "exp":
                sigma_swap = _postprocess_sigma(torch.exp(log_sigma_swap_raw))
            else:
                sigma_swap = _postprocess_sigma(
                    torch.nn.functional.softplus(log_sigma_swap_raw) + 1e-3
                )
            sigma_swap = sigma_swap.numpy()

            logits_swap = classifier(X_swap_tensor)
            home_win_prob_swap = torch.sigmoid(logits_swap).numpy()

            sigma_orig = sigma[neutral_idx].copy()
            p_orig = home_win_prob[neutral_idx].copy()

            home_win_prob[neutral_idx] = (p_orig + (1.0 - home_win_prob_swap)) / 2.0

            # Symmetric mixture of A-vs-B and mirrored B-vs-A distributions.
            mu_orig = legacy_mu[neutral_idx].copy()
            sigma_var = 0.5 * (sigma_orig ** 2 + sigma_swap ** 2) + ((mu_orig + mu_swap) ** 2) / 4.0
            sigma[neutral_idx] = np.sqrt(np.maximum(sigma_var, 0.25)).astype(np.float32)

    # Build output — include team names if available
    meta_cols = ["gameId", "homeTeamId", "awayTeamId"]
    if "homeTeam" in features_df.columns:
        meta_cols.append("homeTeam")
    if "awayTeam" in features_df.columns:
        meta_cols.append("awayTeam")
    meta_cols.append("startDate")
    if "neutralSite" in features_df.columns:
        meta_cols.append("neutralSite")
    for optional_col in ["gameType", "tournament", "conferenceGame"]:
        if optional_col in features_df.columns:
            meta_cols.append(optional_col)
    out = features_df[meta_cols].copy()
    if "neutralSite" in out.columns:
        out = out.rename(columns={"neutralSite": "neutral_site"})
    out[legacy_variant_field()] = legacy_mu
    if team_ab_field in variant_predictions:
        out[team_ab_field] = variant_predictions[team_ab_field]
    out["mean_model_variant_active"] = active_variant
    out["predicted_spread"] = mu
    out["spread_sigma"] = sigma
    out["home_win_prob"] = home_win_prob
    out["away_win_prob"] = 1.0 - home_win_prob

    # Attach lines if available
    if lines_df is not None and not lines_df.empty:
        lines_dedup = select_preferred_lines(lines_df)
        merge_cols = ["gameId", "book_spread", "book_total", "home_moneyline", "away_moneyline"]
        available = [c for c in merge_cols if c in lines_dedup.columns]
        out = out.merge(lines_dedup[available], on="gameId", how="left")

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

    return add_tournament_market_display_columns(out)


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

_SITE_DISPLAY_OVERRIDES = {
    "display_predicted_spread": "model_mu_home",
    "display_edge_home_points": "edge_home_points",
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

    site_field_map = dict(_SITE_FIELD_MAP)
    if config.NCAA_TOURNAMENT_DISPLAY_USE_POSTPROCESS:
        for src_col, dst_col in _SITE_DISPLAY_OVERRIDES.items():
            if src_col in preds.columns:
                for existing_src, mapped in list(site_field_map.items()):
                    if mapped == dst_col:
                        del site_field_map[existing_src]
                site_field_map[src_col] = dst_col

    # Site-compatible JSON (replaces csv_to_json.py pipeline)
    config.SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    site_games = []
    for rec in records:
        game = {}
        for src_col, dst_col in site_field_map.items():
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
        "provenance": {
            "mean_model_variant_active": (
                str(preds["mean_model_variant_active"].iloc[0])
                if "mean_model_variant_active" in preds.columns and not preds.empty
                else active_mean_model_variant()
            ),
            "primary_efficiency_source": config.EFFICIENCY_SOURCE,
            "team_ab_efficiency_source_active": config.TEAM_AB_EFFICIENCY_SOURCE,
            "live_lines_table": config.TABLE_FCT_LINES,
            "research_lines_table": config.RESEARCH_LINES_TABLE,
            "sigma_model": "legacy_mlp_regressor_pt",
            "stored_probability_model": "legacy_mlp_classifier_pt",
            "site_probability_surface": "mu_plus_sigma_meta_small_v1_neutral_beta_blend_v1",
        },
        "games": site_games,
    }
    site_json_path = config.SITE_DATA_DIR / f"predictions_{game_date}.json"
    with open(site_json_path, "w") as f:
        json.dump(site_payload, f, indent=2, sort_keys=True)
        f.write("\n")

    return json_path, csv_path
