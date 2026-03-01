"""Corrected efficiency features for V5 pipeline.

Instead of giving the model raw adj_oe + dozens of noisy correction factors
as separate inputs, this module computes corrected efficiencies directly —
fewer features, each maximally informative.

Shot-quality expected efficiency measures how many points per FGA a team
would score if every player shot at league-average rates from each zone.
This captures offensive system quality (shot creation) stripped of shooting
variance.  shot_quality_vs_actual captures the residual — how much a team
is over/under-performing their shot quality (i.e., shooting luck).

All league averages are built causally: only games BEFORE the current date
contribute to the running league-wide shooting percentages.

Output columns per team-game:
  - shot_quality_oe:        Expected pts/FGA from offensive shot distribution
                            × league-avg zone FG%
  - shot_quality_de:        Expected pts/FGA forced by defense (opponent shot
                            distribution × league-avg zone FG%)
  - shot_quality_vs_actual: Actual pts/FGA - shot_quality_oe (shooting luck
                            in pts/FGA terms)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CORRECTED_EFFICIENCY_COLS: list[str] = [
    "shot_quality_oe",
    "shot_quality_de",
    "shot_quality_vs_actual",
]


def compute_corrected_efficiencies(adv_stats: pd.DataFrame) -> pd.DataFrame:
    """Compute corrected efficiency features from raw (pre-adjusted) advanced stats.

    Uses a causal date-by-date loop so that league averages only include
    games played BEFORE the current date.

    Args:
        adv_stats: DataFrame from compute_advanced_stats() with columns:
            gameid, teamid, startdate,
            rim_fga, rim_fgm, mid_fga, mid_fgm, three_fga, three_fgm,
            rim_rate, mid_range_rate,
            def_rim_rate, def_mid_range_rate.

    Returns:
        DataFrame with columns: gameid, teamid, startdate, + CORRECTED_EFFICIENCY_COLS.
        First games of the season have NaN values (no league avg yet).
    """
    required = [
        "gameid", "teamid", "startdate",
        "rim_fga", "rim_fgm", "mid_fga", "mid_fgm",
        "three_fga", "three_fgm",
        "rim_rate", "mid_range_rate",
        "def_rim_rate", "def_mid_range_rate",
    ]
    missing = [c for c in required if c not in adv_stats.columns]
    if missing:
        return pd.DataFrame(
            columns=["gameid", "teamid", "startdate"] + CORRECTED_EFFICIENCY_COLS
        )

    df = adv_stats.copy()
    df["_date"] = pd.to_datetime(df["startdate"], errors="coerce", utc=True).dt.normalize()
    df["_date"] = df["_date"].dt.tz_localize(None)
    df = df.sort_values(["_date", "gameid", "teamid"]).reset_index(drop=True)

    n = len(df)

    # Pre-allocate output arrays (NaN by default)
    sq_oe = np.full(n, np.nan)
    sq_de = np.full(n, np.nan)
    sq_vs_actual = np.full(n, np.nan)

    # Running league-wide zone sums (from all games BEFORE current date)
    lg_rim_fga = 0.0
    lg_rim_fgm = 0.0
    lg_mid_fga = 0.0
    lg_mid_fgm = 0.0
    lg_three_fga = 0.0
    lg_three_fgm = 0.0

    # Column indices for fast .iat access
    ci = {c: df.columns.get_loc(c) for c in [
        "rim_rate", "mid_range_rate", "rim_fga", "rim_fgm",
        "mid_fga", "mid_fgm", "three_fga", "three_fgm",
        "def_rim_rate", "def_mid_range_rate",
    ]}

    unique_dates = sorted(df["_date"].dropna().unique())
    dates_arr = df["_date"].values

    for date_val in unique_dates:
        mask = dates_arr == date_val
        indices = np.where(mask)[0]

        # Compute league averages from games BEFORE this date
        total_lg_fga = lg_rim_fga + lg_mid_fga + lg_three_fga
        if total_lg_fga > 0:
            lg_rim_fg = lg_rim_fgm / lg_rim_fga if lg_rim_fga > 0 else 0.0
            lg_mid_fg = lg_mid_fgm / lg_mid_fga if lg_mid_fga > 0 else 0.0
            lg_3pt_fg = lg_three_fgm / lg_three_fga if lg_three_fga > 0 else 0.0

            for i in indices:
                # --- Shot Quality OE ---
                rim_rate = df.iat[i, ci["rim_rate"]]
                mid_rate = df.iat[i, ci["mid_range_rate"]]

                if pd.notna(rim_rate) and pd.notna(mid_rate):
                    three_rate = max(1.0 - rim_rate - mid_rate, 0.0)
                    # Expected pts per FGA if shooting at league averages
                    sq_oe[i] = (
                        rim_rate * lg_rim_fg * 2
                        + mid_rate * lg_mid_fg * 2
                        + three_rate * lg_3pt_fg * 3
                    )

                    # --- Actual pts per FGA ---
                    r_fga = df.iat[i, ci["rim_fga"]]
                    r_fgm = df.iat[i, ci["rim_fgm"]]
                    m_fga = df.iat[i, ci["mid_fga"]]
                    m_fgm = df.iat[i, ci["mid_fgm"]]
                    t_fga = df.iat[i, ci["three_fga"]]
                    t_fgm = df.iat[i, ci["three_fgm"]]

                    total_fga = r_fga + m_fga + t_fga
                    if pd.notna(total_fga) and total_fga > 0:
                        actual_fg_pts = r_fgm * 2 + m_fgm * 2 + t_fgm * 3
                        sq_vs_actual[i] = actual_fg_pts / total_fga - sq_oe[i]

                # --- Shot Quality DE ---
                def_rim_rate = df.iat[i, ci["def_rim_rate"]]
                def_mid_rate = df.iat[i, ci["def_mid_range_rate"]]

                if pd.notna(def_rim_rate) and pd.notna(def_mid_rate):
                    def_three_rate = max(1.0 - def_rim_rate - def_mid_rate, 0.0)
                    # Expected pts per opponent FGA at league averages
                    sq_de[i] = (
                        def_rim_rate * lg_rim_fg * 2
                        + def_mid_rate * lg_mid_fg * 2
                        + def_three_rate * lg_3pt_fg * 3
                    )

        # After computing for this date, update running league sums
        for i in indices:
            r_fga = df.iat[i, ci["rim_fga"]]
            r_fgm = df.iat[i, ci["rim_fgm"]]
            m_fga = df.iat[i, ci["mid_fga"]]
            m_fgm = df.iat[i, ci["mid_fgm"]]
            t_fga = df.iat[i, ci["three_fga"]]
            t_fgm = df.iat[i, ci["three_fgm"]]

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
    result["shot_quality_oe"] = sq_oe
    result["shot_quality_de"] = sq_de
    result["shot_quality_vs_actual"] = sq_vs_actual

    return result
