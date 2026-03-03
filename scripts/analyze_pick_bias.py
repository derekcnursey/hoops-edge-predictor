"""
Analyze home/away pick bias in historical CBB spread predictions.

Loads all predictions_*.json files and quantifies:
1. Pick-side distribution by edge threshold
2. Model's implied HCA vs market
3. Model accuracy breakdown by pick side
"""

import glob
import json
import statistics
from collections import defaultdict

DATA_DIR = "/Users/dereknursey/Desktop/ml_projects/hoops-edge-predictor/site/public/data"


def load_all_predictions():
    """Load all prediction JSON files and return flat list of games with book spreads."""
    pattern = f"{DATA_DIR}/predictions_*.json"
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} prediction files")

    all_games = []
    files_with_spreads = 0
    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        games = data.get("games", [])
        has_spread = False
        for g in games:
            if g.get("market_spread_home") is not None:
                all_games.append(g)
                has_spread = True
        if has_spread:
            files_with_spreads += 1

    print(f"Files with at least one book spread: {files_with_spreads}")
    print(f"Total games with book spread: {len(all_games)}")
    return all_games


def pick_bias_table(games):
    """Print pick-side counts at various edge thresholds."""
    thresholds = [0.0, 0.05, 0.10, 0.15]

    print("\n" + "=" * 78)
    print("PICK-SIDE DISTRIBUTION BY EDGE THRESHOLD")
    print("=" * 78)
    print(f"  Note: pick_prob_edge can be negative (model's best side still < 50% cover prob).")
    print(f"  'edge >= 0%' includes only games where model found a positive edge.")
    print(f"  'incl neg edge' includes ALL games with a book spread.\n")
    header = f"{'Threshold':>16} | {'Total':>7} | {'Home':>7} | {'Away':>7} | {'Home%':>7} | {'Away%':>7}"
    print(header)
    print("-" * 78)

    # First row: ALL games (including negative edge)
    total = len(games)
    home = sum(1 for g in games if g["pick_side"].upper() == "HOME")
    away = total - home
    home_pct = home / total * 100 if total else 0
    away_pct = away / total * 100 if total else 0
    print(
        f"{'incl neg edge':>16} | {total:>7,} | {home:>7,} | {away:>7,} | "
        f"{home_pct:>6.1f}% | {away_pct:>6.1f}%"
    )

    for thresh in thresholds:
        subset = [
            g for g in games
            if g.get("pick_prob_edge") is not None
            and g["pick_prob_edge"] >= thresh
        ]
        total = len(subset)
        home = sum(1 for g in subset if g["pick_side"].upper() == "HOME")
        away = sum(1 for g in subset if g["pick_side"].upper() == "AWAY")

        if total > 0:
            home_pct = home / total * 100
            away_pct = away / total * 100
        else:
            home_pct = away_pct = 0.0

        label = f"edge >= {thresh:.0%}"
        print(
            f"{label:>16} | {total:>7,} | {home:>7,} | {away:>7,} | "
            f"{home_pct:>6.1f}% | {away_pct:>6.1f}%"
        )

    # Also show neutral site breakdown
    print("\n--- Neutral site breakdown (all edges) ---")
    neutral = [g for g in games if g.get("neutral_site") is True]
    non_neutral = [g for g in games if not g.get("neutral_site")]
    for label, subset in [("Non-neutral", non_neutral), ("Neutral", neutral)]:
        total = len(subset)
        home = sum(1 for g in subset if g["pick_side"].upper() == "HOME")
        away = total - home
        if total > 0:
            home_pct = home / total * 100
            away_pct = away / total * 100
        else:
            home_pct = away_pct = 0.0
        print(
            f"{label:>12} | {total:>7,} | {home:>7,} | {away:>7,} | "
            f"{home_pct:>6.1f}% | {away_pct:>6.1f}%"
        )


def model_hca_analysis(games):
    """Compute model's implied HCA vs market."""
    print("\n" + "=" * 78)
    print("MODEL HOME-COURT ADVANTAGE (HCA) ANALYSIS")
    print("=" * 78)

    # model_mu_home: positive = home wins by that margin
    # market_spread_home: negative = home favored (book convention)
    # Book's implied home margin = -market_spread_home
    # HCA gap = model_mu_home - (-market_spread_home) = model_mu_home + market_spread_home

    model_mus = [g["model_mu_home"] for g in games]
    book_implied_margins = [-g["market_spread_home"] for g in games]
    hca_gaps = [g["model_mu_home"] - (-g["market_spread_home"]) for g in games]

    print(f"\nAll games with book spread (n={len(games):,}):")
    print(f"  Mean model_mu_home (raw model home advantage):  {statistics.mean(model_mus):+.3f}")
    print(f"  Mean book implied home margin (-spread):        {statistics.mean(book_implied_margins):+.3f}")
    print(f"  Mean HCA gap (model - book):                    {statistics.mean(hca_gaps):+.3f}")
    print(f"  Median HCA gap:                                 {statistics.median(hca_gaps):+.3f}")
    print(f"  Stdev HCA gap:                                  {statistics.stdev(hca_gaps):.3f}")

    # Break down by neutral vs non-neutral
    non_neutral = [g for g in games if not g.get("neutral_site")]
    neutral = [g for g in games if g.get("neutral_site") is True]

    if non_neutral:
        nn_mus = [g["model_mu_home"] for g in non_neutral]
        nn_book = [-g["market_spread_home"] for g in non_neutral]
        nn_gaps = [g["model_mu_home"] + g["market_spread_home"] for g in non_neutral]
        print(f"\nNon-neutral site games (n={len(non_neutral):,}):")
        print(f"  Mean model_mu_home:    {statistics.mean(nn_mus):+.3f}")
        print(f"  Mean book home margin: {statistics.mean(nn_book):+.3f}")
        print(f"  Mean HCA gap:          {statistics.mean(nn_gaps):+.3f}")

    if neutral:
        n_mus = [g["model_mu_home"] for g in neutral]
        n_book = [-g["market_spread_home"] for g in neutral]
        n_gaps = [g["model_mu_home"] + g["market_spread_home"] for g in neutral]
        print(f"\nNeutral site games (n={len(neutral):,}):")
        print(f"  Mean model_mu_home:    {statistics.mean(n_mus):+.3f}")
        print(f"  Mean book home margin: {statistics.mean(n_book):+.3f}")
        print(f"  Mean HCA gap:          {statistics.mean(n_gaps):+.3f}")

    # Distribution of HCA gap
    print(f"\nHCA gap distribution:")
    pcts = [10, 25, 50, 75, 90]
    sorted_gaps = sorted(hca_gaps)
    n = len(sorted_gaps)
    for p in pcts:
        idx = int(n * p / 100)
        idx = min(idx, n - 1)
        print(f"  P{p:02d}: {sorted_gaps[idx]:+.2f}")


def accuracy_by_side(games):
    """Break down model behavior by pick side."""
    print("\n" + "=" * 78)
    print("MODEL BEHAVIOR BY PICK SIDE")
    print("=" * 78)

    home_picks = [g for g in games if g["pick_side"].upper() == "HOME" and g.get("pick_prob_edge") is not None]
    away_picks = [g for g in games if g["pick_side"].upper() == "AWAY" and g.get("pick_prob_edge") is not None]

    print(f"\n{'Metric':<45} | {'Home Picks':>12} | {'Away Picks':>12}")
    print("-" * 78)

    # Count
    print(f"{'Count':<45} | {len(home_picks):>12,} | {len(away_picks):>12,}")

    # Mean edge (pick_prob_edge)
    if home_picks:
        mean_edge_home = statistics.mean([g["pick_prob_edge"] for g in home_picks])
    else:
        mean_edge_home = 0
    if away_picks:
        mean_edge_away = statistics.mean([g["pick_prob_edge"] for g in away_picks])
    else:
        mean_edge_away = 0
    print(f"{'Mean pick_prob_edge':<45} | {mean_edge_home:>11.4f} | {mean_edge_away:>11.4f}")

    # Median edge
    if home_picks:
        med_edge_home = statistics.median([g["pick_prob_edge"] for g in home_picks])
    else:
        med_edge_home = 0
    if away_picks:
        med_edge_away = statistics.median([g["pick_prob_edge"] for g in away_picks])
    else:
        med_edge_away = 0
    print(f"{'Median pick_prob_edge':<45} | {med_edge_home:>11.4f} | {med_edge_away:>11.4f}")

    # Mean absolute edge_home_points
    home_abs_edge = [abs(g["edge_home_points"]) for g in home_picks if g.get("edge_home_points") is not None]
    away_abs_edge = [abs(g["edge_home_points"]) for g in away_picks if g.get("edge_home_points") is not None]
    if home_abs_edge:
        mean_abs_home = statistics.mean(home_abs_edge)
    else:
        mean_abs_home = 0
    if away_abs_edge:
        mean_abs_away = statistics.mean(away_abs_edge)
    else:
        mean_abs_away = 0
    print(f"{'Mean |edge_home_points| (pts)':<45} | {mean_abs_home:>11.2f} | {mean_abs_away:>11.2f}")

    # Mean edge_home_points (signed)
    home_signed = [g["edge_home_points"] for g in home_picks if g.get("edge_home_points") is not None]
    away_signed = [g["edge_home_points"] for g in away_picks if g.get("edge_home_points") is not None]
    if home_signed:
        mean_signed_home = statistics.mean(home_signed)
    else:
        mean_signed_home = 0
    if away_signed:
        mean_signed_away = statistics.mean(away_signed)
    else:
        mean_signed_away = 0
    print(f"{'Mean edge_home_points (signed, pts)':<45} | {mean_signed_home:>+11.2f} | {mean_signed_away:>+11.2f}")

    # Mean model_mu_home
    home_mu = [g["model_mu_home"] for g in home_picks]
    away_mu = [g["model_mu_home"] for g in away_picks]
    print(f"{'Mean model_mu_home':<45} | {statistics.mean(home_mu):>+11.2f} | {statistics.mean(away_mu):>+11.2f}")

    # Mean market_spread_home
    home_spread = [g["market_spread_home"] for g in home_picks]
    away_spread = [g["market_spread_home"] for g in away_picks]
    print(f"{'Mean market_spread_home (book)':<45} | {statistics.mean(home_spread):>+11.2f} | {statistics.mean(away_spread):>+11.2f}")

    # Mean pick_cover_prob
    home_cp = [g["pick_cover_prob"] for g in home_picks if g.get("pick_cover_prob") is not None]
    away_cp = [g["pick_cover_prob"] for g in away_picks if g.get("pick_cover_prob") is not None]
    if home_cp and away_cp:
        print(f"{'Mean pick_cover_prob':<45} | {statistics.mean(home_cp):>11.4f} | {statistics.mean(away_cp):>11.4f}")


def season_breakdown(games):
    """Break down pick bias by season."""
    print("\n" + "=" * 78)
    print("PICK-SIDE DISTRIBUTION BY SEASON")
    print("=" * 78)

    # Extract season from game_id (format: YYYY_MM_DD_...)
    by_season = defaultdict(list)
    for g in games:
        gid = g.get("game_id", "")
        parts = gid.split("_")
        if len(parts) >= 3:
            year = int(parts[0])
            month = int(parts[1])
            # CBB season: Nov(11)-Apr(4). If month >= 11, season = year+1. Else season = year.
            season = year + 1 if month >= 11 else year
            by_season[season].append(g)

    print(f"\n{'Season':>8} | {'Total':>7} | {'Home':>7} | {'Away':>7} | {'Home%':>7} | {'Away%':>7} | {'Mean HCA Gap':>12}")
    print("-" * 78)

    for season in sorted(by_season.keys()):
        subset = by_season[season]
        total = len(subset)
        home = sum(1 for g in subset if g["pick_side"].upper() == "HOME")
        away = total - home
        home_pct = home / total * 100 if total else 0
        away_pct = away / total * 100 if total else 0
        hca_gaps = [g["model_mu_home"] + g["market_spread_home"] for g in subset]
        mean_gap = statistics.mean(hca_gaps) if hca_gaps else 0
        print(
            f"{season:>8} | {total:>7,} | {home:>7,} | {away:>7,} | "
            f"{home_pct:>6.1f}% | {away_pct:>6.1f}% | {mean_gap:>+11.2f}"
        )


def main():
    games = load_all_predictions()
    if not games:
        print("No games with book spreads found!")
        return

    pick_bias_table(games)
    model_hca_analysis(games)
    accuracy_by_side(games)
    season_breakdown(games)


if __name__ == "__main__":
    main()
