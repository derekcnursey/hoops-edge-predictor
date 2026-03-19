# Internal Bet Filter Report: 2026-02-15

## Summary

- Season: `2026`
- Training seasons used: `[2019, 2020, 2022, 2023, 2024, 2025]`
- Slate games scored: `32`
- Promoted disagreement-aware threshold: `disagreement_logit >= 0.58`
- Raw baseline watchlist threshold: `pick_prob_edge >= 0.08`
- Decision-useful non-NCAA candidates: `4`
- NCAA caution / diagnostic-only rows: `0`
- Raw-edge-only watchlist rows: `6`
- Filter-passing rows flagged mainly by disagreement features: `3`
- Slice mix: `{'feb15_plus': 32}`

## Guardrails

- Use the promoted shortlist as an internal late-season / March / conference-tournament aid, not as a public model output.
- NCAA tournament rows remain caution / diagnostic-only even if they clear the internal threshold.
- `flagged_mainly_by_disagreement = true` means the game would not have cleared the raw edge watchlist on its own.
- `raw edge only / filtered out` means the market-disagreement layer did not support the bet strongly enough.

## Ranked Internal Shortlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,358886,2026-02-15,2026-02-15 17:00:00+00:00,feb15_plus,Charlotte,UTSA,-13.5,14.314,HOME,0.529,0.005,0.494,0.771,0.277,True,none,20.909,20.909,0,0,False,disagreement-led support,decision-useful internal candidate
2026,358890,2026-02-15,2026-02-15 19:00:00+00:00,feb15_plus,UAB,Tulane,-6.75,10.45,HOME,0.589,0.065,0.51,0.693,0.183,True,persistent,12.943,12.943,5,23,False,disagreement-led persistent,decision-useful internal candidate
2026,214949,2026-02-15,2026-02-15 18:00:00+00:00,feb15_plus,Illinois,Indiana,-8.75,8.76,HOME,0.5,-0.023,0.489,0.678,0.189,True,persistent,12.135,12.135,5,24,False,disagreement-led persistent,decision-useful internal candidate
2026,214951,2026-02-15,2026-02-15 23:00:00+00:00,feb15_plus,Butler,Seton Hall,4.25,5.487,HOME,0.75,0.226,0.555,0.659,0.104,False,persistent,7.018,7.018,5,24,False,raw edge + persistent disagreement support,decision-useful internal candidate
```

## NCAA Caution / Diagnostic Rows

_empty_

## Raw-Edge-Only Watchlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,358889,2026-02-15,2026-02-15 19:00:00+00:00,feb15_plus,Florida Atlantic,South Florida,4.75,3.876,HOME,0.792,0.268,0.546,0.556,0.009,False,persistent,1.269,1.269,5,21,False,raw edge only / filtered out,raw-edge watchlist only
2026,211639,2026-02-15,2026-02-15 21:00:00+00:00,feb15_plus,North Alabama,Eastern Kentucky,4.5,1.644,HOME,0.747,0.223,0.527,0.44,-0.087,False,persistent,-2.979,2.979,6,13,False,raw edge only / filtered out,raw-edge watchlist only
2026,358894,2026-02-15,2026-02-15 23:00:00+00:00,feb15_plus,Murray State,Belmont,1.5,4.665,HOME,0.735,0.212,0.527,0.432,-0.096,False,persistent,-4.879,4.879,6,24,False,raw edge only / filtered out,raw-edge watchlist only
2026,358888,2026-02-15,2026-02-15 19:00:00+00:00,feb15_plus,Temple,North Texas,-2.5,5.608,HOME,0.624,0.1,0.512,0.502,-0.01,False,none,1.2,1.2,0,0,False,raw edge only / filtered out,raw-edge watchlist only
2026,358891,2026-02-15,2026-02-15 20:00:00+00:00,feb15_plus,Northern Iowa,Drake,-9.25,5.306,AWAY,0.621,0.097,0.509,0.495,-0.015,False,persistent,-1.568,1.568,6,21,False,raw edge only / filtered out,raw-edge watchlist only
2026,211663,2026-02-15,2026-02-15 20:00:00+00:00,feb15_plus,Milwaukee,Green Bay,-2.5,4.716,HOME,0.608,0.084,0.507,0.541,0.035,False,none,3.921,3.921,0,0,False,raw edge only / filtered out,raw-edge watchlist only
```
