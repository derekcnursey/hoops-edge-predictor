"""
Away-side bias analysis: Does the model's tendency to pick away sides
actually help or hurt ATS performance?
"""
import json
import glob
import os
from collections import defaultdict
from datetime import datetime

DATA_DIR = "/Users/dereknursey/Desktop/ml_projects/hoops-edge-predictor/site/public/data"

def date_to_season(date_str: str) -> int:
    """CBB season: Nov 2024 -> Apr 2025 = '2025' season."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    if dt.month >= 8:  # Aug-Dec -> next year's season
        return dt.year + 1
    else:  # Jan-Apr
        return dt.year

def load_all_data():
    """Load all predictions and final scores, match by game_id."""
    # Load all predictions
    pred_files = sorted(glob.glob(os.path.join(DATA_DIR, "predictions_*.json")))
    score_files = sorted(glob.glob(os.path.join(DATA_DIR, "final_scores_*.json")))

    # Build score lookup: game_id -> {home_score, away_score}
    score_lookup = {}
    for sf in score_files:
        with open(sf) as f:
            data = json.load(f)
        for g in data.get("games", []):
            gid = g.get("game_id")
            hs = g.get("home_score")
            aws = g.get("away_score")
            if gid and hs is not None and aws is not None:
                score_lookup[gid] = {"home_score": hs, "away_score": aws}

    # Build matched records
    records = []
    for pf in pred_files:
        with open(pf) as f:
            data = json.load(f)
        date_str = data.get("date", "")
        for g in data.get("games", []):
            gid = g.get("game_id")
            pick_side = g.get("pick_side", "").upper()
            spread = g.get("market_spread_home")
            edge = g.get("pick_prob_edge")
            model_mu = g.get("model_mu_home")

            # Skip if no book spread or no pick
            if spread is None or pick_side not in ("HOME", "AWAY"):
                continue

            # Must have final score
            if gid not in score_lookup:
                continue

            sc = score_lookup[gid]
            season = date_to_season(date_str)

            records.append({
                "game_id": gid,
                "date": date_str,
                "season": season,
                "pick_side": pick_side,
                "market_spread_home": spread,
                "model_mu_home": model_mu,
                "pick_prob_edge": edge if edge is not None else 0.0,
                "home_score": sc["home_score"],
                "away_score": sc["away_score"],
                "home_team": g.get("home_team", ""),
                "away_team": g.get("away_team", ""),
            })

    return records


def compute_ats(records):
    """
    cover = home_score - away_score + market_spread_home
    If pick_side == HOME: win if cover > 0
    If pick_side == AWAY: win if cover < 0
    Push if cover == 0
    """
    results = []
    for r in records:
        cover = r["home_score"] - r["away_score"] + r["market_spread_home"]
        if r["pick_side"] == "HOME":
            if cover > 0:
                outcome = "W"
            elif cover < 0:
                outcome = "L"
            else:
                outcome = "P"
        else:  # AWAY
            if cover < 0:
                outcome = "W"
            elif cover > 0:
                outcome = "L"
            else:
                outcome = "P"
        results.append({**r, "cover": cover, "outcome": outcome})
    return results


def print_record(label, results):
    """Print W-L-P record and win rate."""
    w = sum(1 for r in results if r["outcome"] == "W")
    l = sum(1 for r in results if r["outcome"] == "L")
    p = sum(1 for r in results if r["outcome"] == "P")
    n = w + l  # exclude pushes for win rate
    pct = w / n * 100 if n > 0 else 0
    print(f"  {label:40s}  {w:4d}-{l:4d}-{p:3d}  ({n:5d} decided)  {pct:5.1f}%")
    return w, l, p


def main():
    print("Loading data...")
    records = load_all_data()
    print(f"Matched {len(records)} prediction+score pairs with book spreads.\n")

    results = compute_ats(records)

    # Count pick side distribution
    home_picks = [r for r in results if r["pick_side"] == "HOME"]
    away_picks = [r for r in results if r["pick_side"] == "AWAY"]
    total = len(results)
    print(f"Pick side distribution: HOME={len(home_picks)} ({len(home_picks)/total*100:.1f}%)  "
          f"AWAY={len(away_picks)} ({len(away_picks)/total*100:.1f}%)\n")

    # ── Section 1: ATS by pick side at different edge thresholds ──
    print("=" * 90)
    print("SECTION 1: ATS RECORD BY PICK SIDE AT DIFFERENT EDGE THRESHOLDS")
    print("=" * 90)
    for min_edge in [0.0, 0.05, 0.10]:
        pct_label = f"{min_edge*100:.0f}%"
        print(f"\n--- Edge >= {pct_label} ---")
        filtered = [r for r in results if r["pick_prob_edge"] >= min_edge]
        home_f = [r for r in filtered if r["pick_side"] == "HOME"]
        away_f = [r for r in filtered if r["pick_side"] == "AWAY"]
        print_record(f"ALL (edge >= {pct_label})", filtered)
        print_record(f"HOME picks (edge >= {pct_label})", home_f)
        print_record(f"AWAY picks (edge >= {pct_label})", away_f)

    # ── Section 2: ATS by season ──
    print("\n" + "=" * 90)
    print("SECTION 2: ATS RECORD BY SEASON")
    print("=" * 90)
    seasons = sorted(set(r["season"] for r in results))
    for season in seasons:
        print(f"\n--- Season {season} ---")
        s_results = [r for r in results if r["season"] == season]
        home_s = [r for r in s_results if r["pick_side"] == "HOME"]
        away_s = [r for r in s_results if r["pick_side"] == "AWAY"]
        print_record(f"ALL", s_results)
        print_record(f"HOME picks", home_s)
        print_record(f"AWAY picks", away_s)
        print(f"    Pick distribution: HOME={len(home_s)} AWAY={len(away_s)} "
              f"({len(away_s)/len(s_results)*100:.1f}% away)")

    # ── Section 3: ATS by season, edge >= 5% ──
    print("\n" + "=" * 90)
    print("SECTION 3: ATS RECORD BY SEASON (EDGE >= 5% ONLY)")
    print("=" * 90)
    for season in seasons:
        print(f"\n--- Season {season}, edge >= 5% ---")
        s_results = [r for r in results if r["season"] == season and r["pick_prob_edge"] >= 0.05]
        if not s_results:
            print("  (no games)")
            continue
        home_s = [r for r in s_results if r["pick_side"] == "HOME"]
        away_s = [r for r in s_results if r["pick_side"] == "AWAY"]
        print_record(f"ALL", s_results)
        print_record(f"HOME picks", home_s)
        print_record(f"AWAY picks", away_s)

    # ── Section 4: Average cover margin by pick side ──
    print("\n" + "=" * 90)
    print("SECTION 4: AVERAGE COVER MARGIN BY PICK SIDE")
    print("  (positive = in favor of the pick, negative = against the pick)")
    print("=" * 90)

    for min_edge in [0.0, 0.05, 0.10]:
        pct_label = f"{min_edge*100:.0f}%"
        print(f"\n--- Edge >= {pct_label} ---")

        home_f = [r for r in results if r["pick_side"] == "HOME" and r["pick_prob_edge"] >= min_edge]
        away_f = [r for r in results if r["pick_side"] == "AWAY" and r["pick_prob_edge"] >= min_edge]

        # For HOME picks: oriented cover = cover (positive means home covers)
        # For AWAY picks: oriented cover = -cover (positive means away covers)
        if home_f:
            home_margins = [r["cover"] for r in home_f]
            avg_home = sum(home_margins) / len(home_margins)
            wins_home = [r["cover"] for r in home_f if r["outcome"] == "W"]
            losses_home = [r["cover"] for r in home_f if r["outcome"] == "L"]
            avg_win_margin_h = sum(wins_home) / len(wins_home) if wins_home else 0
            avg_loss_margin_h = sum(losses_home) / len(losses_home) if losses_home else 0
            print(f"  HOME picks (n={len(home_f):4d}): avg cover margin = {avg_home:+.2f}")
            print(f"    When W (n={len(wins_home):4d}): avg cover = {avg_win_margin_h:+.2f}")
            print(f"    When L (n={len(losses_home):4d}): avg cover = {avg_loss_margin_h:+.2f}")
        else:
            print(f"  HOME picks: (none)")

        if away_f:
            away_margins = [-r["cover"] for r in away_f]  # flip sign for away
            avg_away = sum(away_margins) / len(away_margins)
            wins_away = [-r["cover"] for r in away_f if r["outcome"] == "W"]
            losses_away = [-r["cover"] for r in away_f if r["outcome"] == "L"]
            avg_win_margin_a = sum(wins_away) / len(wins_away) if wins_away else 0
            avg_loss_margin_a = sum(losses_away) / len(losses_away) if losses_away else 0
            print(f"  AWAY picks (n={len(away_f):4d}): avg cover margin = {avg_away:+.2f}")
            print(f"    When W (n={len(wins_away):4d}): avg cover = {avg_win_margin_a:+.2f}")
            print(f"    When L (n={len(losses_away):4d}): avg cover = {avg_loss_margin_a:+.2f}")
        else:
            print(f"  AWAY picks: (none)")

    # ── Section 5: What if we flipped? (Hypothetical: always pick home) ──
    print("\n" + "=" * 90)
    print("SECTION 5: HYPOTHETICAL -- WHAT IF THE MODEL ALWAYS PICKED THE SIDE WITH POSITIVE EDGE?")
    print("  (i.e., ignore pick_side, just check: does the model's predicted spread beat the book?)")
    print("=" * 90)
    # The model picks HOME when model_mu_home + market_spread_home > 0 (home edge)
    # and AWAY when < 0. So the pick_side IS the side with positive edge already.
    # Instead, let's ask: what if we ONLY picked home? Or ONLY away?
    print("\n--- What if we only bet HOME (model says home edge > 0)? ---")
    for min_edge in [0.0, 0.05, 0.10]:
        pct_label = f"{min_edge*100:.0f}%"
        home_only = [r for r in results if r["pick_side"] == "HOME" and r["pick_prob_edge"] >= min_edge]
        print_record(f"HOME-only (edge >= {pct_label})", home_only)

    print("\n--- What if we only bet AWAY (model says away edge > 0)? ---")
    for min_edge in [0.0, 0.05, 0.10]:
        pct_label = f"{min_edge*100:.0f}%"
        away_only = [r for r in results if r["pick_side"] == "AWAY" and r["pick_prob_edge"] >= min_edge]
        print_record(f"AWAY-only (edge >= {pct_label})", away_only)

    # ── Section 6: Deeper -- model_mu direction vs pick direction ──
    print("\n" + "=" * 90)
    print("SECTION 6: MODEL MU DIRECTION ANALYSIS")
    print("  Does the model predict away wins more often than home wins?")
    print("=" * 90)
    home_favored = [r for r in results if r["model_mu_home"] is not None and r["model_mu_home"] > 0]
    away_favored = [r for r in results if r["model_mu_home"] is not None and r["model_mu_home"] <= 0]
    print(f"\n  model_mu_home > 0 (model predicts home win): {len(home_favored)} games")
    print(f"  model_mu_home <= 0 (model predicts away win): {len(away_favored)} games")
    if results:
        avg_mu = sum(r["model_mu_home"] for r in results if r["model_mu_home"] is not None) / len(results)
        print(f"  Average model_mu_home: {avg_mu:+.2f}")

    # What fraction of away picks come from model predicting away win vs home win?
    away_and_model_away = [r for r in away_picks if r["model_mu_home"] is not None and r["model_mu_home"] <= 0]
    away_and_model_home = [r for r in away_picks if r["model_mu_home"] is not None and r["model_mu_home"] > 0]
    print(f"\n  Among AWAY picks:")
    print(f"    Model predicts away win (mu <= 0): {len(away_and_model_away)}")
    print(f"    Model predicts home win (mu > 0) but spread offers away value: {len(away_and_model_home)}")

    # ── Section 7: ROI calculation ──
    print("\n" + "=" * 90)
    print("SECTION 7: ROI BY PICK SIDE (assuming -110 on all bets, 1 unit per bet)")
    print("=" * 90)
    for min_edge in [0.0, 0.05, 0.10]:
        pct_label = f"{min_edge*100:.0f}%"
        print(f"\n--- Edge >= {pct_label} ---")
        for side_label, side_filter in [("HOME", "HOME"), ("AWAY", "AWAY"), ("ALL", None)]:
            if side_filter:
                subset = [r for r in results if r["pick_side"] == side_filter and r["pick_prob_edge"] >= min_edge]
            else:
                subset = [r for r in results if r["pick_prob_edge"] >= min_edge]
            w = sum(1 for r in subset if r["outcome"] == "W")
            l = sum(1 for r in subset if r["outcome"] == "L")
            n = w + l
            if n == 0:
                print(f"  {side_label:6s}: no bets")
                continue
            profit = w * (100/110) - l * 1.0  # win pays 100/110, lose costs 1
            roi = profit / n * 100
            print(f"  {side_label:6s}: {w:4d}W - {l:4d}L  units={profit:+.1f}  ROI={roi:+.1f}%  (n={n})")

    # ── Section 8: Summary ──
    print("\n" + "=" * 90)
    print("SUMMARY / KEY TAKEAWAY")
    print("=" * 90)
    # Compare at edge >= 5%
    home_5 = [r for r in results if r["pick_side"] == "HOME" and r["pick_prob_edge"] >= 0.05]
    away_5 = [r for r in results if r["pick_side"] == "AWAY" and r["pick_prob_edge"] >= 0.05]
    h_w = sum(1 for r in home_5 if r["outcome"] == "W")
    h_l = sum(1 for r in home_5 if r["outcome"] == "L")
    a_w = sum(1 for r in away_5 if r["outcome"] == "W")
    a_l = sum(1 for r in away_5 if r["outcome"] == "L")
    h_n = h_w + h_l
    a_n = a_w + a_l
    h_pct = h_w / h_n * 100 if h_n else 0
    a_pct = a_w / a_n * 100 if a_n else 0
    h_roi = (h_w * (100/110) - h_l) / h_n * 100 if h_n else 0
    a_roi = (a_w * (100/110) - a_l) / a_n * 100 if a_n else 0
    print(f"\n  At edge >= 5%:")
    print(f"    HOME picks: {h_w}-{h_l} ({h_pct:.1f}%), ROI={h_roi:+.1f}%")
    print(f"    AWAY picks: {a_w}-{a_l} ({a_pct:.1f}%), ROI={a_roi:+.1f}%")
    print(f"    Difference: away - home = {a_pct - h_pct:+.1f} pp win rate, {a_roi - h_roi:+.1f} pp ROI")
    if a_pct > h_pct:
        print(f"\n  >> AWAY picks outperform HOME picks by {a_pct - h_pct:.1f} pp.")
        print(f"  >> The away-side bias appears to be a FEATURE, not a bug.")
    elif h_pct > a_pct:
        print(f"\n  >> HOME picks outperform AWAY picks by {h_pct - a_pct:.1f} pp.")
        print(f"  >> The away-side bias may be HURTING performance.")
    else:
        print(f"\n  >> HOME and AWAY picks perform identically.")


if __name__ == "__main__":
    main()
