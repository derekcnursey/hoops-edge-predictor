# Internal Bet Filter Report: 2026-02-24

## Summary

- Season: `2026`
- Training seasons used: `[2019, 2020, 2022, 2023, 2024, 2025]`
- Slate games scored: `38`
- Promoted disagreement-aware threshold: `disagreement_logit >= 0.58`
- Raw baseline watchlist threshold: `pick_prob_edge >= 0.08`
- Decision-useful non-NCAA candidates: `5`
- NCAA caution / diagnostic-only rows: `0`
- Raw-edge-only watchlist rows: `8`
- Filter-passing rows flagged mainly by disagreement features: `3`
- Slice mix: `{'feb15_plus': 38}`

## Guardrails

- Use the promoted shortlist as an internal late-season / March / conference-tournament aid, not as a public model output.
- NCAA tournament rows remain caution / diagnostic-only even if they clear the internal threshold.
- `flagged_mainly_by_disagreement = true` means the game would not have cleared the raw edge watchlist on its own.
- `raw edge only / filtered out` means the market-disagreement layer did not support the bet strongly enough.

## Ranked Internal Shortlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,211858,2026-02-24,2026-02-25 00:00:00+00:00,feb15_plus,Troy,Louisiana,-11.5,12.388,HOME,0.527,0.003,0.492,0.808,0.316,True,persistent,23.579,23.579,6,22,False,disagreement-led persistent,decision-useful internal candidate
2026,215070,2026-02-24,2026-02-24 23:30:00+00:00,feb15_plus,Rutgers,Washington,4.5,-4.793,AWAY,0.511,-0.013,0.491,0.635,0.144,True,persistent,9.483,9.483,5,19,False,disagreement-led persistent,decision-useful internal candidate
2026,358961,2026-02-24,2026-02-25 01:00:00+00:00,feb15_plus,UIC,Bradley,-1.75,3.345,HOME,0.558,0.034,0.5,0.589,0.089,True,persistent,6.334,6.334,6,13,False,disagreement-led persistent,decision-useful internal candidate
2026,215083,2026-02-24,2026-02-25 00:00:00+00:00,feb15_plus,Virginia,NC State,-5.75,9.188,HOME,0.638,0.114,0.514,0.671,0.157,False,persistent,11.432,11.432,6,23,False,raw edge + persistent disagreement support,decision-useful internal candidate
2026,215063,2026-02-24,2026-02-25 00:00:00+00:00,feb15_plus,Texas Tech,Cincinnati,-6.75,10.082,HOME,0.624,0.1,0.513,0.652,0.139,False,persistent,9.98,9.98,5,26,False,raw edge + persistent disagreement support,decision-useful internal candidate
```

## NCAA Caution / Diagnostic Rows

_empty_

## Raw-Edge-Only Watchlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,215062,2026-02-24,2026-02-25 02:00:00+00:00,feb15_plus,Utah,Iowa State,14.25,-4.676,HOME,0.797,0.273,0.554,0.214,-0.34,False,persistent,-18.36,18.36,6,8,False,raw edge only / filtered out,raw-edge watchlist only
2026,211865,2026-02-24,2026-02-25 00:00:00+00:00,feb15_plus,South Carolina,Kentucky,7.5,0.971,HOME,0.758,0.234,0.545,0.465,-0.08,False,persistent,-3.922,3.922,5,25,False,raw edge only / filtered out,raw-edge watchlist only
2026,215079,2026-02-24,2026-02-25 00:00:00+00:00,feb15_plus,Notre Dame,Duke,16.5,-7.689,HOME,0.736,0.212,0.548,0.109,-0.438,False,persistent,-29.233,29.233,5,26,False,raw edge only / filtered out,raw-edge watchlist only
2026,211855,2026-02-24,2026-02-25 00:00:00+00:00,feb15_plus,Ball State,Massachusetts,4.5,1.486,HOME,0.725,0.201,0.526,0.353,-0.173,False,persistent,-7.6,7.6,6,23,False,raw edge only / filtered out,raw-edge watchlist only
2026,215071,2026-02-24,2026-02-25 01:30:00+00:00,feb15_plus,Michigan,Minnesota,-21.75,15.027,AWAY,0.688,0.164,0.531,0.195,-0.336,False,persistent,-19.754,19.754,5,16,False,raw edge only / filtered out,raw-edge watchlist only
2026,358960,2026-02-24,2026-02-25 00:00:00+00:00,feb15_plus,Dayton,Saint Louis,3.75,0.198,HOME,0.639,0.115,0.511,0.298,-0.213,False,persistent,-12.102,12.102,5,22,False,raw edge only / filtered out,raw-edge watchlist only
2026,211853,2026-02-24,2026-02-25 00:00:00+00:00,feb15_plus,Toledo,Northern Illinois,-12.5,8.894,AWAY,0.628,0.104,0.508,0.275,-0.233,False,persistent,-12.004,12.004,5,23,False,raw edge only / filtered out,raw-edge watchlist only
2026,215073,2026-02-24,2026-02-25 02:00:00+00:00,feb15_plus,Boise State,Wyoming,-8.5,5.405,AWAY,0.627,0.103,0.506,0.502,-0.004,False,persistent,-0.901,0.901,5,23,False,raw edge only / filtered out,raw-edge watchlist only
```
