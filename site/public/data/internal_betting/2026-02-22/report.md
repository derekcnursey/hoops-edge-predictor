# Internal Bet Filter Report: 2026-02-22

## Summary

- Season: `2026`
- Training seasons used: `[2019, 2020, 2022, 2023, 2024, 2025]`
- Slate games scored: `21`
- Promoted disagreement-aware threshold: `disagreement_logit >= 0.58`
- Raw baseline watchlist threshold: `pick_prob_edge >= 0.08`
- Decision-useful non-NCAA candidates: `5`
- NCAA caution / diagnostic-only rows: `0`
- Raw-edge-only watchlist rows: `3`
- Filter-passing rows flagged mainly by disagreement features: `4`
- Slice mix: `{'feb15_plus': 21}`

## Guardrails

- Use the promoted shortlist as an internal late-season / March / conference-tournament aid, not as a public model output.
- NCAA tournament rows remain caution / diagnostic-only even if they clear the internal threshold.
- `flagged_mainly_by_disagreement = true` means the game would not have cleared the raw edge watchlist on its own.
- `raw edge only / filtered out` means the market-disagreement layer did not support the bet strongly enough.

## Ranked Internal Shortlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,358954,2026-02-22,2026-02-22 17:00:00+00:00,feb15_plus,Memphis,UAB,-6.0,2.556,AWAY,0.595,0.071,0.504,0.703,0.199,True,persistent,12.193,12.193,5,25,False,disagreement-led persistent,decision-useful internal candidate
2026,211842,2026-02-22,2026-02-22 19:00:00+00:00,feb15_plus,Siena,Saint Peter's,-5.75,6.438,HOME,0.531,0.007,0.496,0.629,0.133,True,persistent,8.844,8.844,6,18,False,disagreement-led persistent,decision-useful internal candidate
2026,215027,2026-02-22,2026-02-22 17:00:00+00:00,feb15_plus,Bucknell,Holy Cross,-2.5,1.452,AWAY,0.536,0.012,0.489,0.593,0.104,True,persistent,6.282,6.282,5,25,False,disagreement-led persistent,decision-useful internal candidate
2026,215056,2026-02-22,2026-02-22 21:00:00+00:00,feb15_plus,Wisconsin,Iowa,-3.5,2.289,AWAY,0.545,0.021,0.491,0.591,0.1,True,persistent,4.415,4.415,6,25,False,disagreement-led persistent,decision-useful internal candidate
2026,358957,2026-02-22,2026-02-22 21:00:00+00:00,feb15_plus,North Texas,Florida Atlantic,-2.5,-1.989,AWAY,0.665,0.141,0.515,0.694,0.179,False,persistent,11.66,11.66,5,23,False,raw edge + persistent disagreement support,decision-useful internal candidate
```

## NCAA Caution / Diagnostic Rows

_empty_

## Raw-Edge-Only Watchlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,358956,2026-02-22,2026-02-22 21:00:00+00:00,feb15_plus,Tulsa,UTSA,-21.75,15.048,AWAY,0.716,0.193,0.531,0.086,-0.445,False,persistent,-29.287,29.287,5,24,False,raw edge only / filtered out,raw-edge watchlist only
2026,211839,2026-02-22,2026-02-22 19:00:00+00:00,feb15_plus,Cleveland State,Purdue Fort Wayne,3.5,0.497,HOME,0.668,0.145,0.512,0.413,-0.099,False,persistent,-4.454,4.454,6,20,False,raw edge only / filtered out,raw-edge watchlist only
2026,211774,2026-02-22,2026-02-22 19:00:00+00:00,feb15_plus,Drexel,Towson,2.5,2.607,HOME,0.666,0.143,0.519,0.506,-0.013,False,new/transient,2.881,2.881,0,0,False,raw edge only / filtered out,raw-edge watchlist only
```
