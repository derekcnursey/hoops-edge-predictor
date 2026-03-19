# Internal Bet Filter Report: 2026-03-10

## Summary

- Season: `2026`
- Training seasons used: `[2019, 2020, 2022, 2023, 2024, 2025]`
- Slate games scored: `24`
- Promoted disagreement-aware threshold: `disagreement_logit >= 0.58`
- Raw baseline watchlist threshold: `pick_prob_edge >= 0.08`
- Decision-useful non-NCAA candidates: `5`
- NCAA caution / diagnostic-only rows: `0`
- Raw-edge-only watchlist rows: `2`
- Filter-passing rows flagged mainly by disagreement features: `4`
- Slice mix: `{'conference_tournaments': 24}`

## Guardrails

- Use the promoted shortlist as an internal late-season / March / conference-tournament aid, not as a public model output.
- NCAA tournament rows remain caution / diagnostic-only even if they clear the internal threshold.
- `flagged_mainly_by_disagreement = true` means the game would not have cleared the raw edge watchlist on its own.
- `raw edge only / filtered out` means the market-disagreement layer did not support the bet strongly enough.

## Ranked Internal Shortlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,372331,2026-03-10,2026-03-10 23:30:00+00:00,conference_tournaments,Northwestern,Penn State,-4.75,4.828,HOME,0.502,-0.022,0.497,0.688,0.191,True,persistent,12.446,12.446,5,29,True,disagreement-led persistent,decision-useful internal candidate
2026,372289,2026-03-10,2026-03-10 23:00:00+00:00,conference_tournaments,Vermont,NJIT,-10.75,11.17,HOME,0.514,-0.01,0.507,0.677,0.169,True,persistent,12.997,12.997,5,5,False,disagreement-led persistent,decision-useful internal candidate
2026,372285,2026-03-10,2026-03-10 22:00:00+00:00,conference_tournaments,UMBC,UMass Lowell,-7.5,7.967,HOME,0.517,-0.007,0.51,0.663,0.154,True,persistent,11.811,11.811,2,2,False,disagreement-led persistent,decision-useful internal candidate
2026,372341,2026-03-10,2026-03-10 23:00:00+00:00,conference_tournaments,Wright State,Detroit Mercy,-5.0,5.162,HOME,0.505,-0.019,0.499,0.641,0.142,True,persistent,8.817,8.817,6,30,True,disagreement-led persistent,decision-useful internal candidate
2026,372332,2026-03-10,2026-03-11 01:00:00+00:00,conference_tournaments,Merrimack,Siena,-1.5,-2.025,AWAY,0.614,0.09,0.52,0.598,0.078,False,persistent,3.985,3.985,6,23,True,raw edge + persistent disagreement support,decision-useful internal candidate
```

## NCAA Caution / Diagnostic Rows

_empty_

## Raw-Edge-Only Watchlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,372283,2026-03-10,2026-03-10 19:00:00+00:00,conference_tournaments,Cincinnati,Utah,-12.5,6.866,AWAY,0.672,0.149,0.536,0.373,-0.162,False,none,-7.6,7.6,0,0,True,raw edge only / filtered out,raw-edge watchlist only
2026,372342,2026-03-10,2026-03-11 00:30:00+00:00,conference_tournaments,Jackson State,Grambling,6.75,-2.65,HOME,0.639,0.115,0.525,0.256,-0.269,False,persistent,-13.445,13.445,5,30,True,raw edge only / filtered out,raw-edge watchlist only
```
