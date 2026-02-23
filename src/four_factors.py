"""Compute per-game four-factor stats from fct_pbp_game_teams_flat boxscores.

Each game row produces 13 raw stats for a team:
  OFFENSE (7): eff_fg_pct, ft_pct, ft_rate, 3pt_rate, 3p_pct, off_rebound_pct, def_rebound_pct
  DEFENSE (6): def_eff_fg_pct, def_ft_rate, def_3pt_rate, def_3p_pct, def_off_rebound_pct, def_def_rebound_pct

These are then rolled up into exponentially-decayed averages in rolling_averages.py.
"""

from __future__ import annotations

import pandas as pd


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Element-wise division returning NaN where denominator is 0 or null."""
    return num / den.replace(0, float("nan"))


def compute_game_four_factors(box: pd.DataFrame) -> pd.DataFrame:
    """Compute per-game four-factor stats from a boxscore DataFrame.

    Args:
        box: DataFrame with columns from fct_pbp_game_teams_flat:
             gameid, teamid, opponentid, ishometeam, startdate,
             team_fg_made, team_fg_att, team_3fg_made, team_3fg_att,
             team_ft_made, team_ft_att, team_reb_off, team_reb_def,
             opp_fg_made, opp_fg_att, opp_3fg_made, opp_3fg_att,
             opp_ft_made, opp_ft_att, opp_reb_off, opp_reb_def

    Returns:
        DataFrame with gameid, teamid, startdate, ishometeam, and 13 four-factor columns.
    """
    df = box.copy()

    # ── Offensive four factors ──────────────────────────────────────
    # eff_fg_pct = (FGM + 0.5 * 3PM) / FGA
    df["eff_fg_pct"] = _safe_div(
        df["team_fg_made"] + 0.5 * df["team_3fg_made"],
        df["team_fg_att"],
    )
    # ft_pct = FTM / FTA
    df["ft_pct"] = _safe_div(df["team_ft_made"], df["team_ft_att"])
    # ft_rate = FTA / FGA
    df["ft_rate"] = _safe_div(df["team_ft_att"], df["team_fg_att"])
    # 3pt_rate = 3PA / FGA
    df["three_pt_rate"] = _safe_div(df["team_3fg_att"], df["team_fg_att"])
    # 3p_pct = 3PM / 3PA
    df["three_p_pct"] = _safe_div(df["team_3fg_made"], df["team_3fg_att"])
    # off_rebound_pct = team_off_reb / (team_off_reb + opp_def_reb)
    df["off_rebound_pct"] = _safe_div(
        df["team_reb_off"],
        df["team_reb_off"] + df["opp_reb_def"],
    )
    # def_rebound_pct = team_def_reb / (team_def_reb + opp_off_reb)
    df["def_rebound_pct"] = _safe_div(
        df["team_reb_def"],
        df["team_reb_def"] + df["opp_reb_off"],
    )

    # ── Defensive four factors (computed from opponent's boxscore) ──
    # def_eff_fg_pct = (opp_FGM + 0.5 * opp_3PM) / opp_FGA
    df["def_eff_fg_pct"] = _safe_div(
        df["opp_fg_made"] + 0.5 * df["opp_3fg_made"],
        df["opp_fg_att"],
    )
    # def_ft_rate = opp_FTA / opp_FGA
    df["def_ft_rate"] = _safe_div(df["opp_ft_att"], df["opp_fg_att"])
    # def_3pt_rate = opp_3PA / opp_FGA
    df["def_3pt_rate"] = _safe_div(df["opp_3fg_att"], df["opp_fg_att"])
    # def_3p_pct = opp_3PM / opp_3PA
    df["def_3p_pct"] = _safe_div(df["opp_3fg_made"], df["opp_3fg_att"])
    # def_off_rebound_pct = opp_off_reb / (opp_off_reb + team_def_reb)
    df["def_off_rebound_pct"] = _safe_div(
        df["opp_reb_off"],
        df["opp_reb_off"] + df["team_reb_def"],
    )
    # def_def_rebound_pct = opp_def_reb / (opp_def_reb + team_off_reb)
    df["def_def_rebound_pct"] = _safe_div(
        df["opp_reb_def"],
        df["opp_reb_def"] + df["team_reb_off"],
    )

    keep_cols = [
        "gameid", "teamid", "startdate", "ishometeam",
        # Offense (7)
        "eff_fg_pct", "ft_pct", "ft_rate", "three_pt_rate", "three_p_pct",
        "off_rebound_pct", "def_rebound_pct",
        # Defense (6)
        "def_eff_fg_pct", "def_ft_rate", "def_3pt_rate", "def_3p_pct",
        "def_off_rebound_pct", "def_def_rebound_pct",
    ]
    return df[keep_cols].copy()


# Column names for the 13 four-factor stats (used by rolling_averages.py)
FOUR_FACTOR_COLS = [
    "eff_fg_pct", "ft_pct", "ft_rate", "three_pt_rate", "three_p_pct",
    "off_rebound_pct", "def_rebound_pct",
    "def_eff_fg_pct", "def_ft_rate", "def_3pt_rate", "def_3p_pct",
    "def_off_rebound_pct", "def_def_rebound_pct",
]
