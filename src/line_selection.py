"""Shared betting-line selection helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

NUMERIC_LINE_COLS = [
    "spread",
    "overUnder",
    "homeMoneyline",
    "awayMoneyline",
]


def _provider_rank(provider: pd.Series) -> pd.Series:
    provider = provider.fillna("")
    rank = pd.Series(4, index=provider.index, dtype=int)
    rank.loc[provider == "Hard Rock Bet"] = 0
    rank.loc[provider == "consensus"] = 1
    rank.loc[provider == "Draft Kings"] = 2
    rank.loc[provider == "ESPN BET"] = 3
    rank.loc[provider == "Bovada"] = 5
    return rank


def _prepare_lines(lines_df: pd.DataFrame) -> pd.DataFrame:
    lines = lines_df.copy()
    for col in NUMERIC_LINE_COLS:
        if col in lines.columns:
            lines[col] = pd.to_numeric(lines[col], errors="coerce")
    return lines


def _fix_spread_signs(lines_df: pd.DataFrame) -> pd.DataFrame:
    lines = lines_df.copy()
    if "spread" not in lines.columns or "gameId" not in lines.columns:
        return lines

    has_spread = lines["spread"].notna() & (lines["spread"] != 0)
    spread_sign = np.sign(lines.loc[has_spread, "spread"])
    majority_sign = (
        spread_sign.groupby(lines.loc[has_spread, "gameId"])
        .sum()
        .rename("_majority_sign")
    )
    lines = lines.merge(majority_sign, on="gameId", how="left")

    spread = lines["spread"]
    majority = lines["_majority_sign"]
    no_majority = majority.isna() | majority.eq(0)
    mask_majority_flip = (
        spread.notna() & majority.notna() & (majority != 0)
        & (spread.abs() >= 3)
        & (np.sign(spread) != np.sign(majority))
    )
    lines.loc[mask_majority_flip, "spread"] = -spread[mask_majority_flip]

    if "homeMoneyline" in lines.columns:
        spread = lines["spread"]
        home_ml = lines["homeMoneyline"]
        mask_ml_fix = (
            spread.notna() & home_ml.notna()
            & (~mask_majority_flip)
            & no_majority
            & (
                ((spread > 3) & (home_ml < -150))
                | ((spread < -3) & (home_ml > 150))
            )
        )
        lines.loc[mask_ml_fix, "spread"] = -spread[mask_ml_fix]

    return lines.drop(columns=["_majority_sign"])


def _append_consensus_rows(lines_df: pd.DataFrame) -> pd.DataFrame:
    if lines_df.empty or "gameId" not in lines_df.columns or "provider" not in lines_df.columns:
        return lines_df

    existing_consensus_ids = set(
        lines_df.loc[lines_df["provider"] == "consensus", "gameId"].tolist()
    )
    consensus_rows: list[dict[str, object]] = []
    for game_id, game_df in lines_df.groupby("gameId", sort=False):
        if game_id in existing_consensus_ids:
            continue
        spread_rows = game_df[game_df["spread"].notna()] if "spread" in game_df.columns else pd.DataFrame()
        if spread_rows["provider"].nunique() < 2:
            continue

        row: dict[str, object] = {"gameId": game_id, "provider": "consensus"}
        for col in NUMERIC_LINE_COLS:
            if col not in game_df.columns:
                continue
            source = spread_rows[col] if col == "spread" else game_df[col]
            valid = pd.to_numeric(source, errors="coerce").dropna()
            if not valid.empty:
                row[col] = float(valid.median())
        consensus_rows.append(row)

    if not consensus_rows:
        return lines_df

    consensus_df = pd.DataFrame(consensus_rows)
    return pd.concat([lines_df, consensus_df], ignore_index=True, sort=False)


def _hrb_spread_outlier_mask(lines_df: pd.DataFrame) -> pd.Series:
    if (
        lines_df.empty
        or "gameId" not in lines_df.columns
        or "provider" not in lines_df.columns
        or "spread" not in lines_df.columns
    ):
        return pd.Series(False, index=lines_df.index, dtype=bool)

    outlier = pd.Series(False, index=lines_df.index, dtype=bool)
    hrb_mask = lines_df["provider"].fillna("").eq("Hard Rock Bet")
    if not hrb_mask.any():
        return outlier

    spreads = pd.to_numeric(lines_df["spread"], errors="coerce")
    for game_id, game_df in lines_df.groupby("gameId", sort=False):
        hrb_idx = game_df.index[game_df["provider"].fillna("").eq("Hard Rock Bet")]
        if len(hrb_idx) == 0:
            continue
        peer_mask = ~game_df["provider"].fillna("").isin(["Hard Rock Bet", "consensus"])
        peer_spreads = pd.to_numeric(game_df.loc[peer_mask, "spread"], errors="coerce").dropna()
        if peer_spreads.empty:
            continue
        peer_median = float(peer_spreads.median())
        hrb_spreads = spreads.loc[hrb_idx]
        bad_idx = hrb_spreads.index[
            hrb_spreads.notna()
            & ((hrb_spreads - peer_median).abs() >= config.HRB_SPREAD_DISLOCATION_WARN)
        ]
        if len(bad_idx) > 0:
            outlier.loc[bad_idx] = True
    return outlier


def select_preferred_lines(lines_df: pd.DataFrame) -> pd.DataFrame:
    """Build one preferred line row per game.

    Preference order:
      1. Hard Rock Bet
      2. Synthetic or upstream consensus, when at least two books have spreads.
      3. Draft Kings
      4. ESPN BET
      5. Any non-Bovada provider
      6. Bovada
    """
    if lines_df is None or lines_df.empty:
        return pd.DataFrame()

    lines = _prepare_lines(lines_df)
    lines = _fix_spread_signs(lines)
    lines = _append_consensus_rows(lines)
    hrb_outlier_mask = _hrb_spread_outlier_mask(lines)

    def _has_col(col: str) -> pd.Series:
        if col in lines.columns:
            return lines[col].notna().astype(int)
        return pd.Series(0, index=lines.index, dtype=int)

    provider_rank = (
        _provider_rank(lines["provider"]) if "provider" in lines.columns else pd.Series(99, index=lines.index, dtype=int)
    )
    if not hrb_outlier_mask.empty:
        provider_rank = provider_rank.mask(hrb_outlier_mask, 98)

    selected = (
        lines.assign(
            _has_spread=_has_col("spread"),
            _has_home_ml=_has_col("homeMoneyline"),
            _has_away_ml=_has_col("awayMoneyline"),
            _has_total=_has_col("overUnder"),
            _prov_rank=provider_rank,
        )
        .sort_values(
            ["_has_spread", "_prov_rank", "_has_home_ml", "_has_away_ml", "_has_total", "provider"],
            ascending=[False, True, False, False, False, True],
            kind="mergesort",
        )
        .drop_duplicates(subset=["gameId"], keep="first")
        .drop(columns=["_has_spread", "_has_home_ml", "_has_away_ml", "_has_total", "_prov_rank"])
        .copy()
    )

    selected = selected.rename(
        columns={
            "spread": "book_spread",
            "overUnder": "book_total",
            "homeMoneyline": "home_moneyline",
            "awayMoneyline": "away_moneyline",
        }
    )
    return selected.reset_index(drop=True)
