"""Debug the single NaN mismatch between bulk and single-date builders."""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from src.features import build_features_v2, build_features_v2_bulk

SEASON = 2025
GAME_DATE = "2025-01-15"

# Load both
v2_single = build_features_v2(SEASON, game_date=GAME_DATE)
v2_bulk = build_features_v2_bulk(SEASON)

# Filter bulk to test date
v2_bulk["_date_str"] = (
    pd.to_datetime(v2_bulk["startDate"], errors="coerce", utc=True)
    .dt.tz_convert("America/New_York")
    .dt.strftime("%Y-%m-%d")
)
bulk_filtered = v2_bulk[v2_bulk["_date_str"] == GAME_DATE].copy()
bulk_filtered = bulk_filtered.drop(columns=["_date_str"])

# Sort both by gameId
s = v2_single.sort_values("gameId").reset_index(drop=True)
b = bulk_filtered.sort_values("gameId").reset_index(drop=True)

# Find the game(s) with NaN mismatch
test_col = "away_eff_fg_pct"
s_nan = s[test_col].isna()
b_nan = b[test_col].isna()
mismatch_mask = s_nan != b_nan

if mismatch_mask.any():
    idx = mismatch_mask[mismatch_mask].index[0]
    gid = s.loc[idx, "gameId"]
    print(f"NaN mismatch at game {gid}:")
    print(f"  Home: {s.loc[idx, 'homeTeam']} vs Away: {s.loc[idx, 'awayTeam']}")
    print(f"  Single-date NaN? {s_nan[idx]}  Bulk NaN? {b_nan[idx]}")
    print(f"  Single value: {s.loc[idx, test_col]}  Bulk value: {b.loc[idx, test_col]}")
    print()

    # Check multiple columns for this game
    for col in ["away_eff_fg_pct", "away_team_adj_oe", "away_form_delta",
                "away_rim_rate", "away_max_run_10poss", "away_efg_luck",
                "away_pace_variance", "away_scoring_variance", "away_tov_rate",
                "home_eff_fg_pct", "home_team_adj_oe"]:
        if col in s.columns and col in b.columns:
            sv = s.loc[idx, col]
            bv = b.loc[idx, col]
            print(f"  {col:45s}  single={sv}  bulk={bv}")
else:
    print("No NaN mismatch found for", test_col)
    # Check if it's a different column
    from src.config import FEATURE_ORDER_V3
    for col in FEATURE_ORDER_V3:
        if col in s.columns and col in b.columns:
            s_n = s[col].isna()
            b_n = b[col].isna()
            if (s_n != b_n).any():
                idxs = (s_n != b_n)
                for i in idxs[idxs].index:
                    print(f"  {col}: game {s.loc[i, 'gameId']} single_nan={s_n[i]} bulk_nan={b_n[i]}")
                break
