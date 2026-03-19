import pandas as pd

from src.features import _dedupe_boxscores


def test_dedupe_boxscores_keeps_smallest_volume_row():
    df = pd.DataFrame(
        [
            {
                "gameid": 1,
                "teamid": 10,
                "team_fg_made": 25,
                "team_fg_att": 58,
                "opp_fg_made": 31,
                "opp_fg_att": 65,
                "team_poss": 63.8,
                "opp_poss": 63.8,
            },
            {
                "gameid": 1,
                "teamid": 10,
                "team_fg_made": 50,
                "team_fg_att": 116,
                "opp_fg_made": 62,
                "opp_fg_att": 130,
                "team_poss": 127.6,
                "opp_poss": 127.6,
            },
            {
                "gameid": 1,
                "teamid": 20,
                "team_fg_made": 31,
                "team_fg_att": 65,
                "opp_fg_made": 25,
                "opp_fg_att": 58,
                "team_poss": 63.8,
                "opp_poss": 63.8,
            },
        ]
    )

    out = _dedupe_boxscores(df)
    assert len(out) == 2

    kept = out[(out["gameid"] == 1) & (out["teamid"] == 10)].iloc[0]
    assert kept["team_fg_made"] == 25
    assert kept["team_fg_att"] == 58
    assert kept["team_poss"] == 63.8


def test_dedupe_boxscores_noop_when_unique():
    df = pd.DataFrame(
        [
            {"gameid": 1, "teamid": 10, "team_fg_att": 58},
            {"gameid": 2, "teamid": 10, "team_fg_att": 61},
        ]
    )
    out = _dedupe_boxscores(df)
    pd.testing.assert_frame_equal(out.reset_index(drop=True), df.reset_index(drop=True))
