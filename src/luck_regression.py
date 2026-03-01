"""Luck / regression-to-the-mean features.

Computes how much a team's shooting deviates from league-average
expected shooting, given their shot distribution.  Positive values
indicate "lucky" (above-expectation) shooting.

All league averages are built causally: only games BEFORE the current
date contribute to the running league-wide shooting percentages.

Output columns:
  - efg_luck:       actual eFG% - expected eFG% (based on zone rates * league zone FG%)
  - three_pt_luck:  actual 3PT FG% - league-wide 3PT FG%
  - two_pt_luck:    actual 2PT FG% - league-wide 2PT FG%
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LUCK_FEATURE_COLS: list[str] = [
    "efg_luck",
    "three_pt_luck",
    "two_pt_luck",
]


def compute_luck_features(adv_stats: pd.DataFrame) -> pd.DataFrame:
    """Compute luck / regression features from raw (pre-adjusted) advanced stats.

    Uses a causal date-by-date loop so that league averages only include
    games played BEFORE the current date.

    Args:
        adv_stats: DataFrame from compute_advanced_stats() with columns:
            gameid, teamid, startdate,
            rim_fga, rim_fgm, mid_fga, mid_fgm, three_fga, three_fgm,
            rim_rate, mid_range_rate,
            rim_fg_pct, three_pt_fg_pct, two_pt_fg_pct.

    Returns:
        DataFrame with columns: gameid, teamid, startdate, + LUCK_FEATURE_COLS.
        First games of the season have NaN luck values (no league avg yet).
    """
    required = [
        "gameid", "teamid", "startdate",
        "rim_fga", "rim_fgm", "mid_fga", "mid_fgm",
        "three_fga", "three_fgm",
        "rim_rate", "mid_range_rate",
        "rim_fg_pct", "three_pt_fg_pct", "two_pt_fg_pct",
    ]
    missing = [c for c in required if c not in adv_stats.columns]
    if missing:
        # Return empty with expected schema
        return pd.DataFrame(columns=["gameid", "teamid", "startdate"] + LUCK_FEATURE_COLS)

    df = adv_stats.copy()
    df["_date"] = pd.to_datetime(df["startdate"], errors="coerce")
    df = df.sort_values(["_date", "gameid", "teamid"]).reset_index(drop=True)

    n = len(df)

    # Pre-allocate output arrays (NaN by default)
    efg_luck = np.full(n, np.nan)
    three_luck = np.full(n, np.nan)
    two_luck = np.full(n, np.nan)

    # Running league-wide zone sums (from all games BEFORE current date)
    lg_rim_fga = 0.0
    lg_rim_fgm = 0.0
    lg_mid_fga = 0.0
    lg_mid_fgm = 0.0
    lg_three_fga = 0.0
    lg_three_fgm = 0.0

    # Process date by date for causal ordering
    unique_dates = sorted(df["_date"].dropna().unique())
    dates_arr = df["_date"].values

    for date_val in unique_dates:
        mask = dates_arr == date_val
        indices = np.where(mask)[0]

        # Compute league averages from games BEFORE this date
        if lg_rim_fga + lg_mid_fga + lg_three_fga > 0:
            lg_rim_fg = lg_rim_fgm / lg_rim_fga if lg_rim_fga > 0 else 0.0
            lg_mid_fg = lg_mid_fgm / lg_mid_fga if lg_mid_fga > 0 else 0.0
            lg_3pt_fg = lg_three_fgm / lg_three_fga if lg_three_fga > 0 else 0.0

            # League-wide 2PT FG%
            lg_2pt_fga = lg_rim_fga + lg_mid_fga
            lg_2pt_fgm = lg_rim_fgm + lg_mid_fgm
            lg_2pt_fg = lg_2pt_fgm / lg_2pt_fga if lg_2pt_fga > 0 else 0.0

            # Compute luck for each game-team on this date
            for i in indices:
                rim_rate = df.iat[i, df.columns.get_loc("rim_rate")]
                mid_rate = df.iat[i, df.columns.get_loc("mid_range_rate")]
                rim_fg_pct = df.iat[i, df.columns.get_loc("rim_fg_pct")]
                three_fg_pct = df.iat[i, df.columns.get_loc("three_pt_fg_pct")]
                two_fg_pct = df.iat[i, df.columns.get_loc("two_pt_fg_pct")]

                if pd.isna(rim_rate) or pd.isna(mid_rate):
                    continue

                three_rate = max(1.0 - rim_rate - mid_rate, 0.0)

                # Expected eFG% based on shot distribution and league zone FG%
                # eFG% = (FGM + 0.5 * 3FGM) / FGA
                # expected_efg = rim_rate * lg_rim_fg + mid_rate * lg_mid_fg
                #              + three_rate * lg_3pt_fg * 1.5  (1.5 accounts for eFG 3pt bonus)
                expected_efg = (
                    rim_rate * lg_rim_fg
                    + mid_rate * lg_mid_fg
                    + three_rate * lg_3pt_fg * 1.5
                )

                # Actual eFG%: need to compute from zone data
                r_fga = df.iat[i, df.columns.get_loc("rim_fga")]
                r_fgm = df.iat[i, df.columns.get_loc("rim_fgm")]
                m_fga = df.iat[i, df.columns.get_loc("mid_fga")]
                m_fgm = df.iat[i, df.columns.get_loc("mid_fgm")]
                t_fga = df.iat[i, df.columns.get_loc("three_fga")]
                t_fgm = df.iat[i, df.columns.get_loc("three_fgm")]

                total_fga = r_fga + m_fga + t_fga
                if pd.isna(total_fga) or total_fga == 0:
                    continue
                total_fgm = r_fgm + m_fgm + t_fgm
                actual_efg = (total_fgm + 0.5 * t_fgm) / total_fga

                efg_luck[i] = actual_efg - expected_efg

                # Three-point luck
                if pd.notna(three_fg_pct):
                    three_luck[i] = three_fg_pct - lg_3pt_fg

                # Two-point luck
                if pd.notna(two_fg_pct):
                    two_luck[i] = two_fg_pct - lg_2pt_fg

        # After computing luck for this date, update running league sums
        for i in indices:
            r_fga = df.iat[i, df.columns.get_loc("rim_fga")]
            r_fgm = df.iat[i, df.columns.get_loc("rim_fgm")]
            m_fga = df.iat[i, df.columns.get_loc("mid_fga")]
            m_fgm = df.iat[i, df.columns.get_loc("mid_fgm")]
            t_fga = df.iat[i, df.columns.get_loc("three_fga")]
            t_fgm = df.iat[i, df.columns.get_loc("three_fgm")]

            if pd.notna(r_fga):
                lg_rim_fga += float(r_fga)
            if pd.notna(r_fgm):
                lg_rim_fgm += float(r_fgm)
            if pd.notna(m_fga):
                lg_mid_fga += float(m_fga)
            if pd.notna(m_fgm):
                lg_mid_fgm += float(m_fgm)
            if pd.notna(t_fga):
                lg_three_fga += float(t_fga)
            if pd.notna(t_fgm):
                lg_three_fgm += float(t_fgm)

    # Build output
    result = df[["gameid", "teamid", "startdate"]].copy()
    result["efg_luck"] = efg_luck
    result["three_pt_luck"] = three_luck
    result["two_pt_luck"] = two_luck

    return result
