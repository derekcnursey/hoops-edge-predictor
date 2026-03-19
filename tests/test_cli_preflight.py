import pandas as pd
import pytest
from click import ClickException

from src import cli, config, s3_reader


class _FakeTable:
    def __init__(self, df: pd.DataFrame):
        self._df = df
        self.num_rows = len(df)

    def to_pandas(self) -> pd.DataFrame:
        return self._df.copy()


def test_prediction_preflight_fails_on_duplicate_game_ids(monkeypatch):
    games_raw = pd.DataFrame(
        [
            {"gameId": 1, "homeTeamId": 10, "awayTeamId": 20, "startDate": "2026-03-12T18:00:00Z"},
            {"gameId": 1, "homeTeamId": 99, "awayTeamId": 20, "startDate": "2026-03-12T18:00:00Z"},
        ]
    )
    ratings_raw = pd.DataFrame(
        [
            {"teamId": 10, "rating_date": "2026-03-11", "adj_tempo": 65.0},
        ]
    )

    monkeypatch.setattr(
        s3_reader,
        "read_silver_table",
        lambda table_name, season=None: _FakeTable(games_raw if table_name == config.TABLE_FCT_GAMES else pd.DataFrame()),
    )
    monkeypatch.setattr(
        s3_reader,
        "read_gold_table",
        lambda table_name, season=None: _FakeTable(ratings_raw),
    )
    monkeypatch.setattr(
        cli,
        "load_games",
        lambda season: pd.DataFrame(
            [
                {
                    "gameId": 1,
                    "awayTeam": "BYU",
                    "homeTeam": "Houston",
                    "startDate": "2026-03-12T18:00:00Z",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        cli,
        "load_boxscores",
        lambda season: pd.DataFrame([{"gameid": 1, "teamid": 10}, {"gameid": 1, "teamid": 20}]),
    )
    monkeypatch.setattr(
        cli,
        "load_efficiency_ratings",
        lambda season, no_garbage=True: pd.DataFrame(
            [
                {
                    "teamId": 10,
                    "rating_date": pd.Timestamp("2026-03-11"),
                    "adj_oe": 120.0,
                    "adj_de": 95.0,
                    "adj_tempo": 65.0,
                    "barthag": 0.9,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        cli,
        "build_features",
        lambda *args, **kwargs: pd.DataFrame(
            [
                {
                    "gameId": 1,
                    "home_team_adj_oe": 120.0,
                    "away_team_adj_oe": 118.0,
                    "home_team_adj_de": 95.0,
                    "away_team_adj_de": 99.0,
                    "home_tov_rate": 0.15,
                    "away_tov_rate": 0.16,
                    "home_def_tov_rate": 0.18,
                    "away_def_tov_rate": 0.17,
                }
            ]
        ),
    )
    monkeypatch.setattr(cli, "load_lines", lambda season: pd.DataFrame())
    monkeypatch.setattr(cli, "select_preferred_lines", lambda lines: pd.DataFrame())

    with pytest.raises(ClickException, match="preflight checks failed"):
        cli._run_prediction_preflight(2026, "2026-03-12")


def test_prediction_preflight_warns_on_missing_lines_and_allows_run(monkeypatch, capsys):
    games_raw = pd.DataFrame([{"gameId": 1, "startDate": "2026-03-12T18:00:00Z"}])
    ratings_raw = pd.DataFrame(
        [
            {"teamId": 10, "rating_date": "2026-03-11", "adj_tempo": 65.0},
            {"teamId": 20, "rating_date": "2026-03-11", "adj_tempo": 66.0},
        ]
    )
    features_df = pd.DataFrame(
        [
            {
                "gameId": 1,
                "home_team_adj_oe": 120.0,
                "away_team_adj_oe": 118.0,
                "home_team_adj_de": 95.0,
                "away_team_adj_de": 99.0,
                "home_tov_rate": 0.15,
                "away_tov_rate": 0.16,
                "home_def_tov_rate": 0.18,
                "away_def_tov_rate": 0.17,
            }
        ]
    )
    slate_games = pd.DataFrame(
        [
            {
                "gameId": 1,
                "awayTeam": "BYU",
                "homeTeam": "Houston",
                "startDate": "2026-03-12T18:00:00Z",
            }
        ]
    )

    monkeypatch.setattr(
        s3_reader,
        "read_silver_table",
        lambda table_name, season=None: _FakeTable(games_raw if table_name == config.TABLE_FCT_GAMES else pd.DataFrame()),
    )
    monkeypatch.setattr(
        s3_reader,
        "read_gold_table",
        lambda table_name, season=None: _FakeTable(ratings_raw),
    )
    monkeypatch.setattr(cli, "load_games", lambda season: slate_games)
    monkeypatch.setattr(
        cli,
        "load_boxscores",
        lambda season: pd.DataFrame([{"gameid": 1, "teamid": 10}, {"gameid": 1, "teamid": 20}]),
    )
    monkeypatch.setattr(
        cli,
        "load_efficiency_ratings",
        lambda season, no_garbage=True: pd.DataFrame(
            [
                {
                    "teamId": 10,
                    "rating_date": pd.Timestamp("2026-03-11"),
                    "adj_oe": 120.0,
                    "adj_de": 95.0,
                    "adj_tempo": 65.0,
                    "barthag": 0.9,
                },
                {
                    "teamId": 20,
                    "rating_date": pd.Timestamp("2026-03-11"),
                    "adj_oe": 118.0,
                    "adj_de": 99.0,
                    "adj_tempo": 66.0,
                    "barthag": 0.8,
                },
            ]
        ),
    )
    monkeypatch.setattr(cli, "build_features", lambda *args, **kwargs: features_df)
    monkeypatch.setattr(cli, "load_lines", lambda season: pd.DataFrame())
    monkeypatch.setattr(cli, "select_preferred_lines", lambda lines: pd.DataFrame())

    out_df, out_lines = cli._run_prediction_preflight(2026, "2026-03-12")

    assert out_lines.empty
    pd.testing.assert_frame_equal(out_df.reset_index(drop=True), features_df.reset_index(drop=True))
    captured = capsys.readouterr()
    assert "Missing preferred line rows" in captured.out
    assert "Preflight passed." in captured.out
