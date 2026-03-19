#!/usr/bin/env python3
"""Fix flipped spread signs in existing prediction CSVs.

Three layers of spread-sign correction:
1. Cross-provider majority vote: when the selected provider's sign disagrees
   with the majority of providers, flip it (>= 3 pts only).
2. Moneyline cross-check: for single-provider games, flip when moneyline
   strongly contradicts the spread sign.
3. Model-based cross-check: when book_spread and predicted_spread both say
   home wins with large magnitude (edge > 9 pts), flip the spread — this
   catches games where ALL providers have the wrong sign.
"""
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow importing project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import load_research_lines  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_DIR = PROJECT_ROOT / "predictions" / "csv"
CSV_TO_JSON = PROJECT_ROOT / "scripts" / "csv_to_json.py"

MIN_SPREAD_FOR_FLIP = 3  # only flip spreads with abs >= this
MIN_PHANTOM_EDGE = 9  # model-based fix: flip when edge exceeds this


def normal_cdf(z):
    z = np.asarray(z, dtype=float)
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(z / math.sqrt(2.0)))


def prob_to_american(p):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    out = np.full_like(p, np.nan, dtype=float)
    fav = p >= 0.5
    dog = ~fav
    out[fav] = -100.0 * (p[fav] / (1.0 - p[fav]))
    out[dog] = 100.0 * ((1.0 - p[dog]) / p[dog])
    return out


def build_correct_spreads() -> dict[int, float]:
    """Build gameId → correct spread from first-provider + majority vote fix.

    Returns the spread value that SHOULD be in the CSV for each game.
    """
    correct: dict[int, float] = {}

    for season in range(2016, 2027):
        lines_df = load_research_lines(season)
        if lines_df.empty:
            continue

        lines_df = lines_df.copy()
        lines_df["spread"] = pd.to_numeric(lines_df["spread"], errors="coerce")
        lines_df["homeMoneyline"] = pd.to_numeric(
            lines_df["homeMoneyline"], errors="coerce"
        )

        # Compute majority spread sign per game
        has_spread = lines_df["spread"].notna() & (lines_df["spread"] != 0)
        if not has_spread.any():
            continue

        spread_sign = np.sign(lines_df.loc[has_spread, "spread"])
        majority = (
            spread_sign.groupby(lines_df.loc[has_spread, "gameId"]).sum()
        )

        # Pick first provider alphabetically (original behavior)
        dedup = (
            lines_df.sort_values("provider")
            .drop_duplicates(subset=["gameId"], keep="first")
            .copy()
        )

        for _, row in dedup.iterrows():
            gid = int(row["gameId"])
            sp = row["spread"]
            ml = row["homeMoneyline"]

            if pd.isna(sp) or sp == 0:
                continue

            maj_sign = majority.get(gid, 0)

            # Majority vote fix (only for spreads >= threshold)
            if (
                maj_sign != 0
                and abs(sp) >= MIN_SPREAD_FOR_FLIP
                and np.sign(sp) != np.sign(maj_sign)
            ):
                correct[gid] = -sp
            # Moneyline fallback for single-provider games
            elif (
                maj_sign == 0  # only 1 provider had a spread
                and pd.notna(ml)
                and abs(sp) > 3
                and (
                    (sp > 3 and ml < -150)
                    or (sp < -3 and ml > 150)
                )
            ):
                correct[gid] = -sp
            else:
                correct[gid] = sp

    return correct


def recalc_edges(df: pd.DataFrame) -> None:
    """Recalculate all edge metrics in-place after spread fix."""
    df["model_spread"] = -df["predicted_spread"]
    df["spread_diff"] = df["model_spread"] - df["book_spread"]
    df["edge_home_points"] = df["predicted_spread"] + df["book_spread"]

    sigma_safe = df["spread_sigma"].clip(lower=0.5)
    edge_z = df["edge_home_points"] / sigma_safe
    home_cover_prob = normal_cdf(edge_z.values)
    away_cover_prob = 1.0 - home_cover_prob

    df["pick_side"] = np.where(df["edge_home_points"] >= 0, "HOME", "AWAY")
    df["pick_cover_prob"] = np.where(
        df["edge_home_points"] >= 0, home_cover_prob, away_cover_prob
    )

    pick_breakeven = 110 / 210
    pick_profit = 100 / 110

    df["pick_prob_edge"] = df["pick_cover_prob"] - pick_breakeven
    df["pick_ev_per_1"] = (
        df["pick_cover_prob"] * pick_profit - (1.0 - df["pick_cover_prob"])
    )
    df["pick_fair_odds"] = prob_to_american(df["pick_cover_prob"].values)


def main():
    print("Loading multi-provider lines from S3...")
    correct_spreads = build_correct_spreads()
    print(f"Built correct spreads for {len(correct_spreads)} games\n")

    csv_files = sorted(CSV_DIR.glob("preds_*_edge.csv"))
    total_fixed = 0
    files_changed = 0

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        if "book_spread" not in df.columns or "gameId" not in df.columns:
            continue

        sp = pd.to_numeric(df["book_spread"], errors="coerce")
        n_fix = 0

        for idx, row in df.iterrows():
            gid = int(row["gameId"]) if pd.notna(row["gameId"]) else None
            current = sp[idx]
            if gid is None or pd.isna(current):
                continue

            target = correct_spreads.get(gid)
            if target is not None and current != target:
                df.at[idx, "book_spread"] = target
                n_fix += 1

        # Layer 3: Model-based cross-check for phantom edges.
        # When book_spread and predicted_spread both say home wins (or both
        # say away wins) with combined edge > 9 pts, the spread sign is wrong.
        # This catches games where ALL providers agree on the wrong sign.
        n_model_fix = 0
        if "predicted_spread" in df.columns:
            bs = pd.to_numeric(df["book_spread"], errors="coerce")
            ps = pd.to_numeric(df["predicted_spread"], errors="coerce")
            phantom_edge = bs + ps
            mask_phantom = (
                bs.notna() & ps.notna()
                & (
                    ((bs > 0) & (ps > 0) & (phantom_edge >= MIN_PHANTOM_EDGE))
                    | ((bs < 0) & (ps < 0) & (phantom_edge <= -MIN_PHANTOM_EDGE))
                )
            )
            for idx in df.index[mask_phantom]:
                df.at[idx, "book_spread"] = -bs[idx]
                n_model_fix += 1
            n_fix += n_model_fix

        if n_fix == 0:
            continue

        recalc_edges(df)
        df.to_csv(csv_path, index=False)
        total_fixed += n_fix
        files_changed += 1

        # Regenerate site JSON
        parts = csv_path.stem.replace("preds_", "").replace("_edge", "").split("_")
        if len(parts) >= 3:
            date_str = f"{parts[0]}-{int(parts[1]):02d}-{int(parts[2]):02d}"
            subprocess.run(
                [sys.executable, str(CSV_TO_JSON), str(csv_path), date_str],
                check=True,
                cwd=PROJECT_ROOT,
                capture_output=True,
            )
            provider_lbl = f"{n_fix - n_model_fix} provider" if n_fix > n_model_fix else ""
            model_lbl = f"{n_model_fix} model" if n_model_fix else ""
            detail = " + ".join(filter(None, [provider_lbl, model_lbl]))
            print(f"  Fixed {n_fix} rows ({detail}) in {csv_path.name} → {date_str}")

    print(f"\nDone: {total_fixed} total rows fixed across {files_changed} files")


if __name__ == "__main__":
    main()
