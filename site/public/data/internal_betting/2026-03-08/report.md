# Internal Bet Filter Report: 2026-03-08

## Summary

- Season: `2026`
- Training seasons used: `[2019, 2020, 2022, 2023, 2024, 2025]`
- Slate games scored: `32`
- Promoted disagreement-aware threshold: `disagreement_logit >= 0.58`
- Raw baseline watchlist threshold: `pick_prob_edge >= 0.08`
- Decision-useful non-NCAA candidates: `7`
- NCAA caution / diagnostic-only rows: `0`
- Raw-edge-only watchlist rows: `8`
- Filter-passing rows flagged mainly by disagreement features: `6`
- Slice mix: `{'conference_tournaments': 23, 'march_only': 9}`

## Guardrails

- Use the promoted shortlist as an internal late-season / March / conference-tournament aid, not as a public model output.
- NCAA tournament rows remain caution / diagnostic-only even if they clear the internal threshold.
- `flagged_mainly_by_disagreement = true` means the game would not have cleared the raw edge watchlist on its own.
- `raw edge only / filtered out` means the market-disagreement layer did not support the bet strongly enough.

## Ranked Internal Shortlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,372269,2026-03-08,2026-03-08 22:00:00+00:00,conference_tournaments,Charleston,Towson,-3.5,0.605,AWAY,0.578,0.055,0.513,0.694,0.182,True,new/transient,4.304,4.304,0,0,True,disagreement-led new/transient conf-tourney,decision-useful internal candidate
2026,372275,2026-03-08,2026-03-09 00:30:00+00:00,conference_tournaments,Oregon State,San Francisco,3.75,-4.564,AWAY,0.522,-0.002,0.501,0.69,0.188,True,persistent,12.404,12.404,5,31,True,disagreement-led persistent,decision-useful internal candidate
2026,372268,2026-03-08,2026-03-08 20:00:00+00:00,conference_tournaments,East Tennessee State,Western Carolina,-3.25,4.654,HOME,0.552,0.028,0.512,0.686,0.174,True,persistent,11.958,11.958,5,27,True,disagreement-led persistent,decision-useful internal candidate
2026,372277,2026-03-08,2026-03-09 00:30:00+00:00,conference_tournaments,Siena,Fairfield,-1.75,2.013,HOME,0.509,-0.015,0.502,0.615,0.113,True,persistent,7.204,7.204,6,22,True,disagreement-led persistent,decision-useful internal candidate
2026,372279,2026-03-08,2026-03-09 02:20:00+00:00,conference_tournaments,Montana State,Idaho,1.5,1.34,HOME,0.592,0.068,0.515,0.604,0.09,True,persistent,4.503,4.503,5,25,True,disagreement-led persistent,decision-useful internal candidate
2026,359052,2026-03-08,2026-03-08 19:00:00+00:00,march_only,UAB,East Carolina,-10.25,11.374,HOME,0.538,0.015,0.491,0.77,0.279,True,persistent,19.676,19.676,5,28,False,disagreement-led persistent,decision-useful internal candidate
2026,372271,2026-03-08,2026-03-08 22:00:00+00:00,conference_tournaments,Merrimack,Marist,-3.25,-0.023,AWAY,0.621,0.098,0.519,0.582,0.062,False,persistent,3.172,3.172,5,22,True,raw edge + persistent disagreement support,decision-useful internal candidate
```

## NCAA Caution / Diagnostic Rows

_empty_

## Raw-Edge-Only Watchlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,372278,2026-03-08,2026-03-09 01:00:00+00:00,conference_tournaments,North Dakota State,North Dakota,-11.5,6.697,AWAY,0.691,0.168,0.531,0.306,-0.224,False,none,-11.073,11.073,0,0,True,raw edge only / filtered out,raw-edge watchlist only
2026,359049,2026-03-08,2026-03-08 18:00:00+00:00,march_only,South Florida,Charlotte,-17.25,11.984,AWAY,0.678,0.154,0.516,0.31,-0.207,False,none,-10.821,10.821,0,0,False,raw edge only / filtered out,raw-edge watchlist only
2026,215240,2026-03-08,2026-03-08 21:30:00+00:00,march_only,Michigan,Michigan State,-10.5,4.611,AWAY,0.668,0.144,0.52,0.467,-0.053,False,persistent,-3.659,3.659,5,29,False,raw edge only / filtered out,raw-edge watchlist only
2026,215241,2026-03-08,2026-03-08 19:00:00+00:00,march_only,Maryland,Illinois,14.25,-7.609,HOME,0.666,0.142,0.525,0.13,-0.395,False,none,-25.337,25.337,0,0,False,raw edge only / filtered out,raw-edge watchlist only
2026,372237,2026-03-08,2026-03-08 16:00:00+00:00,conference_tournaments,Lehigh,Colgate,2.25,1.291,HOME,0.625,0.101,0.526,0.432,-0.094,False,persistent,-4.398,4.398,5,26,False,raw edge only / filtered out,raw-edge watchlist only
2026,215239,2026-03-08,2026-03-08 21:00:00+00:00,march_only,Nebraska,Iowa,-5.75,2.296,AWAY,0.611,0.087,0.502,0.483,-0.019,False,persistent,-2.313,2.313,5,29,False,raw edge only / filtered out,raw-edge watchlist only
2026,372273,2026-03-08,2026-03-08 23:30:00+00:00,conference_tournaments,Portland State,Idaho State,-5.25,1.992,AWAY,0.607,0.083,0.518,0.456,-0.062,False,persistent,-4.119,4.119,5,27,True,raw edge only / filtered out,raw-edge watchlist only
2026,372267,2026-03-08,2026-03-08 19:00:00+00:00,conference_tournaments,Monmouth,Drexel,-4.5,0.342,AWAY,0.605,0.081,0.522,0.52,-0.002,False,persistent,-0.27,0.27,4,4,True,raw edge only / filtered out,raw-edge watchlist only
```
