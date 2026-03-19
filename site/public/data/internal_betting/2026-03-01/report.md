# Internal Bet Filter Report: 2026-03-01

## Summary

- Season: `2026`
- Training seasons used: `[2019, 2020, 2022, 2023, 2024, 2025]`
- Slate games scored: `23`
- Promoted disagreement-aware threshold: `disagreement_logit >= 0.58`
- Raw baseline watchlist threshold: `pick_prob_edge >= 0.08`
- Decision-useful non-NCAA candidates: `7`
- NCAA caution / diagnostic-only rows: `0`
- Raw-edge-only watchlist rows: `6`
- Filter-passing rows flagged mainly by disagreement features: `3`
- Slice mix: `{'march_only': 23}`

## Guardrails

- Use the promoted shortlist as an internal late-season / March / conference-tournament aid, not as a public model output.
- NCAA tournament rows remain caution / diagnostic-only even if they clear the internal threshold.
- `flagged_mainly_by_disagreement = true` means the game would not have cleared the raw edge watchlist on its own.
- `raw edge only / filtered out` means the market-disagreement layer did not support the bet strongly enough.

## Ranked Internal Shortlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,359005,2026-03-01,2026-03-01 17:00:00+00:00,march_only,South Florida,Tulane,-15.25,15.456,HOME,0.506,-0.018,0.479,0.72,0.241,True,persistent,16.041,16.041,5,24,False,disagreement-led persistent,decision-useful internal candidate
2026,211975,2026-03-01,2026-03-02 00:00:00+00:00,march_only,UNC Wilmington,Charleston,-5.5,7.592,HOME,0.572,0.048,0.498,0.629,0.131,True,persistent,8.608,8.608,6,25,False,disagreement-led persistent,decision-useful internal candidate
2026,359012,2026-03-01,2026-03-01 19:00:00+00:00,march_only,Florida Atlantic,Charlotte,-6.5,7.737,HOME,0.548,0.025,0.494,0.592,0.098,True,persistent,6.21,6.21,5,25,False,disagreement-led persistent,decision-useful internal candidate
2026,359006,2026-03-01,2026-03-01 17:00:00+00:00,march_only,UAB,North Texas,-4.5,7.747,HOME,0.608,0.085,0.507,0.642,0.135,False,persistent,9.36,9.36,5,26,False,raw edge + persistent disagreement support,decision-useful internal candidate
2026,215158,2026-03-01,2026-03-01 21:00:00+00:00,march_only,Marquette,DePaul,-4.5,-0.341,AWAY,0.672,0.148,0.513,0.636,0.123,False,persistent,8.123,8.123,4,9,False,raw edge + persistent disagreement support,decision-useful internal candidate
2026,359008,2026-03-01,2026-03-01 19:00:00+00:00,march_only,Bradley,Murray State,-4.25,0.966,AWAY,0.608,0.084,0.501,0.596,0.095,False,persistent,4.59,4.59,5,27,False,raw edge + persistent disagreement support,decision-useful internal candidate
2026,212029,2026-03-01,2026-03-01 19:00:00+00:00,march_only,Marist,Saint Peter's,-3.75,6.574,HOME,0.622,0.098,0.506,0.593,0.087,False,persistent,6.527,6.527,5,20,False,raw edge + persistent disagreement support,decision-useful internal candidate
```

## NCAA Caution / Diagnostic Rows

_empty_

## Raw-Edge-Only Watchlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,359009,2026-03-01,2026-03-01 22:00:00+00:00,march_only,Illinois State,Belmont,1.25,4.642,HOME,0.699,0.176,0.521,0.5,-0.021,False,persistent,-1.697,1.697,6,28,False,raw edge only / filtered out,raw-edge watchlist only
2026,359014,2026-03-01,2026-03-01 22:00:00+00:00,march_only,UTSA,Wichita State,15.75,-10.161,HOME,0.689,0.166,0.519,0.103,-0.415,False,persistent,-26.544,26.544,5,26,False,raw edge only / filtered out,raw-edge watchlist only
2026,215155,2026-03-01,2026-03-01 18:30:00+00:00,march_only,Ohio State,Purdue,6.25,-0.948,HOME,0.688,0.164,0.517,0.421,-0.096,False,persistent,-5.641,5.641,6,28,False,raw edge only / filtered out,raw-edge watchlist only
2026,359013,2026-03-01,2026-03-01 20:00:00+00:00,march_only,Drake,Northern Iowa,4.25,0.766,HOME,0.643,0.119,0.513,0.424,-0.089,False,persistent,-5.403,5.403,5,24,False,raw edge only / filtered out,raw-edge watchlist only
2026,359004,2026-03-01,2026-03-01 17:00:00+00:00,march_only,Davidson,La Salle,-10.75,6.265,AWAY,0.632,0.108,0.509,0.421,-0.088,False,persistent,-4.146,4.146,3,1,False,raw edge only / filtered out,raw-edge watchlist only
2026,372149,2026-03-01,2026-03-01 21:00:00+00:00,march_only,Evansville,Southern Illinois,8.75,-4.893,HOME,0.611,0.087,0.504,0.183,-0.32,False,persistent,-18.963,18.963,6,29,False,raw edge only / filtered out,raw-edge watchlist only
```
