import pyarrow as pa

from src import torvik_loader


def test_teamid_mapping_cache_is_scoped_per_season(monkeypatch):
    monkeypatch.setattr(torvik_loader, "_teamid_to_name_by_season", {})

    def fake_read_silver_table(table_name, season=None, latest_only=False):
        assert table_name == "fct_games"
        if season == 2025:
            return pa.table({
                "homeTeamId": [101],
                "homeTeam": ["Alpha"],
                "awayTeamId": [201],
                "awayTeam": ["Beta"],
            })
        if season == 2026:
            return pa.table({
                "homeTeamId": [101],
                "homeTeam": ["Gamma"],
                "awayTeamId": [202],
                "awayTeam": ["Delta"],
            })
        return pa.table({})

    monkeypatch.setattr(torvik_loader.s3_reader, "read_silver_table", fake_read_silver_table)

    torvik_loader._load_teamid_mapping(2025)
    torvik_loader._load_teamid_mapping(2026)

    assert torvik_loader._teamid_to_name_by_season[2025][101] == "Alpha"
    assert torvik_loader._teamid_to_name_by_season[2025][201] == "Beta"
    assert torvik_loader._teamid_to_name_by_season[2026][101] == "Gamma"
    assert torvik_loader._teamid_to_name_by_season[2026][202] == "Delta"
