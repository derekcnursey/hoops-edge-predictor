"""Inspect how the trained model checkpoints treat home-court-advantage features.

Analyzes:
1. Model state dict structure (all keys + shapes)
2. First-layer weight norms for `neutral_site` vs all features
3. Perturbation analysis: forward pass with neutral_site=0 vs 1
4. Both regressor and classifier checkpoints
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

# ── Project paths ────────────────────────────────────────────────
PROJECT = Path(__file__).resolve().parent.parent
CHECKPOINTS = PROJECT / "checkpoints"
ARTIFACTS = PROJECT / "artifacts"

# ── Load feature order from production checkpoint ─────────────────
def load_feature_order_from_checkpoint(ckpt_path):
    """Load feature order embedded in checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt.get("feature_order", None), ckpt


def load_scaler():
    """Load the fitted StandardScaler."""
    path = ARTIFACTS / "scaler.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def analyze_checkpoint(name, ckpt_path, feature_order):
    """Analyze a single checkpoint."""
    print_section(f"CHECKPOINT: {name} ({ckpt_path.name})")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Print metadata
    hp = ckpt.get("hparams", {})
    arch_type = ckpt.get("arch_type", "shared")
    sigma_param = ckpt.get("sigma_param", "softplus")
    ckpt_features = ckpt.get("feature_order", [])
    print(f"Architecture type: {arch_type}")
    print(f"Sigma param: {sigma_param}")
    print(f"Hyperparams: {hp}")
    print(f"Feature count in checkpoint: {len(ckpt_features)}")

    state_dict = ckpt["state_dict"]

    # Print all keys and shapes
    print(f"\nState dict keys ({len(state_dict)} total):")
    for key, tensor in state_dict.items():
        print(f"  {key:40s} {str(list(tensor.shape)):>20s}")

    # Find the first linear layer weight
    first_layer_key = None
    for key in state_dict:
        if "weight" in key and state_dict[key].dim() == 2:
            # Check if input dim matches feature count
            if state_dict[key].shape[1] == len(feature_order):
                first_layer_key = key
                break

    if first_layer_key is None:
        print("\nWARNING: Could not identify first linear layer matching feature count")
        return

    W = state_dict[first_layer_key].numpy()  # shape: (hidden_dim, input_dim)
    print(f"\nFirst layer: {first_layer_key}, shape: {W.shape}")
    print(f"  (output_dim={W.shape[0]}, input_dim={W.shape[1]})")

    # Find bias for first layer
    first_bias_key = first_layer_key.replace("weight", "bias")
    if first_bias_key in state_dict:
        b = state_dict[first_bias_key].numpy()
        print(f"  Bias shape: {b.shape}")

    # ── L1 norm analysis per feature ─────────────────────────────
    # L1 norm of weights connected to each input feature
    l1_norms = np.abs(W).sum(axis=0)  # shape: (input_dim,)
    l2_norms = np.sqrt((W ** 2).sum(axis=0))

    avg_l1 = l1_norms.mean()
    avg_l2 = l2_norms.mean()

    print(f"\n--- First-layer weight L1 norms (all {len(feature_order)} features) ---")
    print(f"  Average L1 norm across all features: {avg_l1:.4f}")
    print(f"  Average L2 norm across all features: {avg_l2:.4f}")

    # Sort by L1 norm (descending) and print ranking
    ranked_idx = np.argsort(-l1_norms)
    print(f"\n  Feature importance ranking (by L1 norm of first-layer weights):")
    print(f"  {'Rank':>4s}  {'Feature':40s}  {'L1':>8s}  {'L2':>8s}  {'Ratio':>8s}")
    print(f"  {'-'*4}  {'-'*40}  {'-'*8}  {'-'*8}  {'-'*8}")

    for rank, idx in enumerate(ranked_idx):
        feat_name = feature_order[idx]
        ratio = l1_norms[idx] / avg_l1
        marker = " <-- HCA" if feat_name in ("neutral_site", "home_team_home", "away_team_home", "is_home") else ""
        print(f"  {rank+1:>4d}  {feat_name:40s}  {l1_norms[idx]:8.4f}  {l2_norms[idx]:8.4f}  {ratio:8.3f}x{marker}")

    # Highlight HCA features specifically
    hca_features = ["neutral_site", "home_team_home", "away_team_home", "is_home"]
    print(f"\n--- HCA-related features specifically ---")
    for feat_name in hca_features:
        if feat_name in feature_order:
            idx = feature_order.index(feat_name)
            rank = list(ranked_idx).index(idx) + 1
            print(f"  {feat_name}:")
            print(f"    Index: {idx}")
            print(f"    L1 norm: {l1_norms[idx]:.4f} (rank {rank}/{len(feature_order)})")
            print(f"    L2 norm: {l2_norms[idx]:.4f}")
            print(f"    Ratio to avg: {l1_norms[idx]/avg_l1:.3f}x")
            print(f"    Raw weights (first 10): {W[:10, idx]}")

    return W, feature_order


def perturbation_analysis(feature_order):
    """Forward pass perturbation: compare output with neutral_site=0 vs 1.

    Uses a "typical" input vector (all features at their scaled mean = 0)
    and toggles the neutral_site feature to measure the model's learned HCA.
    """
    print_section("PERTURBATION ANALYSIS: Home Court Advantage in Points")

    scaler = load_scaler()

    # -- Load regressor --
    reg_path = CHECKPOINTS / "regressor.pt"
    ckpt = torch.load(reg_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hparams", {})
    arch_type = ckpt.get("arch_type", "shared")
    sigma_param = ckpt.get("sigma_param", "softplus")
    ckpt_features = ckpt.get("feature_order", feature_order)

    # Import architectures
    sys.path.insert(0, str(PROJECT / "src"))
    from architecture import MLPRegressor, MLPRegressorSplit, MLPClassifier

    ModelClass = MLPRegressorSplit if arch_type == "split" else MLPRegressor
    regressor = ModelClass(
        input_dim=len(ckpt_features),
        hidden1=hp.get("hidden1", 256),
        hidden2=hp.get("hidden2", 128),
        dropout=hp.get("dropout", 0.3),
    )
    regressor.load_state_dict(ckpt["state_dict"])
    regressor.eval()

    # -- Load classifier --
    cls_path = CHECKPOINTS / "classifier.pt"
    cls_ckpt = torch.load(cls_path, map_location="cpu", weights_only=False)
    cls_hp = cls_ckpt.get("hparams", {})
    classifier = MLPClassifier(
        input_dim=len(ckpt_features),
        hidden1=cls_hp.get("hidden1", 256),
        dropout=cls_hp.get("dropout", 0.3),
    )
    classifier.load_state_dict(cls_ckpt["state_dict"])
    classifier.eval()

    n_features = len(ckpt_features)

    # Find neutral_site index
    if "neutral_site" not in ckpt_features:
        print("neutral_site not found in feature order!")
        return
    ns_idx = ckpt_features.index("neutral_site")
    print(f"neutral_site is at feature index {ns_idx}")
    print(f"Scaler mean for neutral_site: {scaler.mean_[ns_idx]:.6f}")
    print(f"Scaler std  for neutral_site: {scaler.scale_[ns_idx]:.6f}")

    # The scaler was fit on training data.
    # neutral_site is binary: 0 (home game) or 1 (neutral site).
    # In the unscaled space: 0 = home game, 1 = neutral site.
    # We want to compare: is_home (neutral_site=0) vs neutral (neutral_site=1)

    # Create a "typical" game: use scaler means for all features (= 0 in scaled space)
    # Then toggle neutral_site in the UNSCALED space and rescale

    # Method 1: Toggle in scaled space
    print("\n--- Method 1: Perturbation at mean baseline (scaled space) ---")

    # Baseline: all features at scaled mean (= 0 vector)
    x_base = np.zeros((1, n_features), dtype=np.float32)

    # For neutral_site: compute scaled values of 0 and 1
    ns_scaled_0 = (0.0 - scaler.mean_[ns_idx]) / scaler.scale_[ns_idx]
    ns_scaled_1 = (1.0 - scaler.mean_[ns_idx]) / scaler.scale_[ns_idx]

    print(f"  neutral_site=0 (home game) in scaled space: {ns_scaled_0:.4f}")
    print(f"  neutral_site=1 (neutral)   in scaled space: {ns_scaled_1:.4f}")

    x_home = x_base.copy()
    x_home[0, ns_idx] = ns_scaled_0  # home game

    x_neutral = x_base.copy()
    x_neutral[0, ns_idx] = ns_scaled_1  # neutral site

    with torch.no_grad():
        # Regressor
        mu_home, ls_home = regressor(torch.tensor(x_home))
        mu_neutral, ls_neutral = regressor(torch.tensor(x_neutral))

        if sigma_param == "exp":
            sig_home = torch.exp(ls_home).clamp(0.5, 30.0)
            sig_neutral = torch.exp(ls_neutral).clamp(0.5, 30.0)
        else:
            sig_home = torch.nn.functional.softplus(ls_home) + 1e-3
            sig_neutral = torch.nn.functional.softplus(ls_neutral) + 1e-3

        # Classifier
        logit_home = classifier(torch.tensor(x_home))
        logit_neutral = classifier(torch.tensor(x_neutral))
        prob_home = torch.sigmoid(logit_home)
        prob_neutral = torch.sigmoid(logit_neutral)

    hca_points = float(mu_home - mu_neutral)
    print(f"\n  Regressor predictions (predicted_spread = home - away):")
    print(f"    Home game   (ns=0): mu = {mu_home.item():+.3f}, sigma = {sig_home.item():.3f}")
    print(f"    Neutral     (ns=1): mu = {mu_neutral.item():+.3f}, sigma = {sig_neutral.item():.3f}")
    print(f"    HCA effect: {hca_points:+.3f} points (home game boosts home team by this much)")

    print(f"\n  Classifier predictions (home win probability):")
    print(f"    Home game   (ns=0): P(home win) = {prob_home.item():.4f}")
    print(f"    Neutral     (ns=1): P(home win) = {prob_neutral.item():.4f}")
    print(f"    HCA effect: {prob_home.item() - prob_neutral.item():+.4f} probability points")

    # Method 2: Sweep across multiple "typical" game scenarios
    print("\n--- Method 2: Monte Carlo perturbation (1000 random inputs) ---")

    rng = np.random.RandomState(42)
    n_samples = 1000

    # Generate random inputs in scaled space (std normal, matching training distribution)
    X_random = rng.randn(n_samples, n_features).astype(np.float32)

    # Home game version: set neutral_site to scaled value of 0
    X_home = X_random.copy()
    X_home[:, ns_idx] = ns_scaled_0

    # Neutral site version: set neutral_site to scaled value of 1
    X_neutral = X_random.copy()
    X_neutral[:, ns_idx] = ns_scaled_1

    with torch.no_grad():
        mu_h, _ = regressor(torch.tensor(X_home))
        mu_n, _ = regressor(torch.tensor(X_neutral))
        logit_h = classifier(torch.tensor(X_home))
        logit_n = classifier(torch.tensor(X_neutral))
        prob_h = torch.sigmoid(logit_h)
        prob_n = torch.sigmoid(logit_n)

    diffs = (mu_h - mu_n).numpy()
    prob_diffs = (prob_h - prob_n).numpy()

    print(f"  Regressor HCA (home - neutral):")
    print(f"    Mean:   {diffs.mean():+.3f} points")
    print(f"    Median: {np.median(diffs):+.3f} points")
    print(f"    Std:    {diffs.std():.3f}")
    print(f"    Min:    {diffs.min():+.3f}")
    print(f"    Max:    {diffs.max():+.3f}")
    print(f"    (Positive = home court helps home team)")

    print(f"\n  Classifier HCA (home - neutral):")
    print(f"    Mean:   {prob_diffs.mean():+.4f} probability")
    print(f"    Median: {np.median(prob_diffs):+.4f}")
    print(f"    (Positive = home court increases home win probability)")

    # Method 3: Use actual training data distribution
    print("\n--- Method 3: Realistic perturbation using scaler means ---")
    # Create an "average team vs average team" scenario using actual feature means
    x_avg_unscaled = scaler.mean_.copy().reshape(1, -1).astype(np.float32)

    # Home game
    x_avg_home = x_avg_unscaled.copy()
    x_avg_home[0, ns_idx] = 0.0  # not neutral = home game
    x_avg_home_scaled = scaler.transform(x_avg_home).astype(np.float32)

    # Neutral site
    x_avg_neut = x_avg_unscaled.copy()
    x_avg_neut[0, ns_idx] = 1.0  # neutral site
    x_avg_neut_scaled = scaler.transform(x_avg_neut).astype(np.float32)

    with torch.no_grad():
        mu_h3, ls_h3 = regressor(torch.tensor(x_avg_home_scaled))
        mu_n3, ls_n3 = regressor(torch.tensor(x_avg_neut_scaled))

        logit_h3 = classifier(torch.tensor(x_avg_home_scaled))
        logit_n3 = classifier(torch.tensor(x_avg_neut_scaled))
        prob_h3 = torch.sigmoid(logit_h3)
        prob_n3 = torch.sigmoid(logit_n3)

    hca3 = float(mu_h3 - mu_n3)
    print(f"  Average team vs average team:")
    print(f"    Home game   (ns=0): mu = {mu_h3.item():+.3f}")
    print(f"    Neutral     (ns=1): mu = {mu_n3.item():+.3f}")
    print(f"    HCA effect: {hca3:+.3f} points")
    print(f"    P(home win) home: {prob_h3.item():.4f}, neutral: {prob_n3.item():.4f}")
    print(f"    HCA win prob effect: {prob_h3.item() - prob_n3.item():+.4f}")


def check_no_garbage_checkpoint():
    """Also analyze the no_garbage checkpoint if present."""
    ng_dir = CHECKPOINTS / "no_garbage"
    if not ng_dir.exists():
        return
    reg_path = ng_dir / "regressor.pt"
    cls_path = ng_dir / "classifier.pt"
    if reg_path.exists():
        fo, ckpt = load_feature_order_from_checkpoint(reg_path)
        if fo:
            analyze_checkpoint("no_garbage regressor", reg_path, fo)
    if cls_path.exists():
        fo, ckpt = load_feature_order_from_checkpoint(cls_path)
        if fo:
            analyze_checkpoint("no_garbage classifier", cls_path, fo)


def main():
    print_section("HOME COURT ADVANTAGE WEIGHT ANALYSIS")

    # Load feature order from production regressor
    reg_path = CHECKPOINTS / "regressor.pt"
    cls_path = CHECKPOINTS / "classifier.pt"

    feature_order, _ = load_feature_order_from_checkpoint(reg_path)
    if feature_order is None:
        # Fallback to artifact
        with open(ARTIFACTS / "feature_order.json") as f:
            feature_order = json.load(f)

    print(f"Feature order ({len(feature_order)} features):")
    for i, f in enumerate(feature_order):
        marker = " <-- HCA" if f in ("neutral_site", "home_team_home", "away_team_home", "is_home") else ""
        print(f"  [{i:2d}] {f}{marker}")

    # Check which HCA features exist
    hca_in_model = [f for f in ("neutral_site", "home_team_home", "away_team_home", "is_home")
                    if f in feature_order]
    print(f"\nHCA features in model: {hca_in_model}")

    # Analyze production checkpoints
    analyze_checkpoint("Production Regressor", reg_path, feature_order)
    analyze_checkpoint("Production Classifier", cls_path, feature_order)

    # Analyze no_garbage checkpoints
    check_no_garbage_checkpoint()

    # Perturbation analysis
    perturbation_analysis(feature_order)

    # Summary
    print_section("SUMMARY")
    print("The model encodes home court advantage through `neutral_site` (binary):")
    print("  - neutral_site = 0 -> home game (team has HCA)")
    print("  - neutral_site = 1 -> neutral site (no HCA)")
    print("")
    print("Note: `home_team_home` and `away_team_home` were in V1 (37 features)")
    print("but were PRUNED in V3/production (50 features). The model now relies")
    print("solely on `neutral_site` to capture HCA.")
    print("")
    print("The perturbation analysis above shows the model's learned HCA in points.")


if __name__ == "__main__":
    main()
