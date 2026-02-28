"""Schedule/rest/fatigue features (Group L).

Per-game situational features — NOT rolling averages, NOT opponent-adjusted.
These describe the specific game context rather than team traits:
  - days_rest: days since team's last game
  - opponent_days_rest: days since opponent's last game
  - rest_differential: days_rest - opponent_days_rest
  - games_last_7_days: number of games played in the last 7 days
"""

from __future__ import annotations

import pandas as pd


def compute_schedule_features(games: pd.DataFrame) -> pd.DataFrame:
    """Compute schedule/rest features for each team in each game.

    Args:
        games: DataFrame with columns: gameId, homeTeamId, awayTeamId, startDate.

    Returns:
        DataFrame with columns: gameId, homeTeamId, awayTeamId,
        home_days_rest, away_days_rest, rest_differential,
        home_games_last_7, away_games_last_7.
    """
    df = games.copy()
    df["_date"] = pd.to_datetime(df["startDate"], errors="coerce")

    # Expand to per-team view
    rows = []
    for _, g in df.iterrows():
        dt = g["_date"]
        rows.append({"gameId": int(g["gameId"]), "teamId": int(g["homeTeamId"]), "date": dt})
        rows.append({"gameId": int(g["gameId"]), "teamId": int(g["awayTeamId"]), "date": dt})
    team_games = pd.DataFrame(rows)
    team_games = team_games.sort_values(["teamId", "date", "gameId"]).reset_index(drop=True)

    # Days since previous game
    team_games["prev_date"] = team_games.groupby("teamId")["date"].shift(1)
    team_games["days_rest"] = (team_games["date"] - team_games["prev_date"]).dt.total_seconds() / 86400
    team_games["days_rest"] = team_games["days_rest"].fillna(5.0).clip(upper=30.0)

    # Games in last 7 days
    team_games["games_last_7"] = _count_games_in_window(team_games, window_days=7)

    # Build per-game lookup
    rest_lookup = {}
    g7_lookup = {}
    for _, r in team_games.iterrows():
        rest_lookup[(int(r["gameId"]), int(r["teamId"]))] = float(r["days_rest"])
        g7_lookup[(int(r["gameId"]), int(r["teamId"]))] = int(r["games_last_7"])

    # Assemble output
    records = []
    for _, g in df.iterrows():
        gid = int(g["gameId"])
        home_tid = int(g["homeTeamId"])
        away_tid = int(g["awayTeamId"])

        home_rest = rest_lookup.get((gid, home_tid), 5.0)
        away_rest = rest_lookup.get((gid, away_tid), 5.0)

        records.append({
            "gameId": gid,
            "homeTeamId": home_tid,
            "awayTeamId": away_tid,
            "home_days_rest": home_rest,
            "away_days_rest": away_rest,
            "rest_differential": home_rest - away_rest,
            "home_games_last_7": g7_lookup.get((gid, home_tid), 1),
            "away_games_last_7": g7_lookup.get((gid, away_tid), 1),
        })

    return pd.DataFrame(records)


def _count_games_in_window(team_games: pd.DataFrame, window_days: int = 7) -> pd.Series:
    """Count number of games a team played in the last N days (excluding current game)."""
    counts = []
    for _, row in team_games.iterrows():
        tid = row["teamId"]
        dt = row["date"]
        if pd.isna(dt):
            counts.append(1)
            continue
        cutoff = dt - pd.Timedelta(days=window_days)
        team_mask = team_games["teamId"] == tid
        window_mask = (team_games["date"] >= cutoff) & (team_games["date"] < dt)
        counts.append(int((team_mask & window_mask).sum()))
    return pd.Series(counts, index=team_games.index)


# Feature names exported for integration
SCHEDULE_FEATURE_NAMES = [
    "home_days_rest",
    "away_days_rest",
    "rest_differential",
    "home_games_last_7",
    "away_games_last_7",
]
