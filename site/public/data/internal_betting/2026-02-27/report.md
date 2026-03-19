# Internal Bet Filter Report: 2026-02-27

## Summary

- Season: `2026`
- Training seasons used: `[2019, 2020, 2022, 2023, 2024, 2025]`
- Slate games scored: `21`
- Promoted disagreement-aware threshold: `disagreement_logit >= 0.58`
- Raw baseline watchlist threshold: `pick_prob_edge >= 0.08`
- Decision-useful non-NCAA candidates: `3`
- NCAA caution / diagnostic-only rows: `0`
- Raw-edge-only watchlist rows: `5`
- Filter-passing rows flagged mainly by disagreement features: `3`
- Slice mix: `{'feb15_plus': 21}`

## Guardrails

- Use the promoted shortlist as an internal late-season / March / conference-tournament aid, not as a public model output.
- NCAA tournament rows remain caution / diagnostic-only even if they clear the internal threshold.
- `flagged_mainly_by_disagreement = true` means the game would not have cleared the raw edge watchlist on its own.
- `raw edge only / filtered out` means the market-disagreement layer did not support the bet strongly enough.

## Ranked Internal Shortlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,211945,2026-02-27,2026-02-28 00:00:00+00:00,feb15_plus,Columbia,Brown,-5.25,8.536,HOME,0.593,0.07,0.509,0.641,0.132,True,persistent,9.564,9.564,4,19,False,disagreement-led persistent,decision-useful internal candidate
2026,211938,2026-02-27,2026-02-28 01:30:00+00:00,feb15_plus,South Alabama,Southern Miss,-4.25,6.009,HOME,0.541,0.017,0.493,0.627,0.134,True,persistent,8.728,8.728,7,23,False,disagreement-led persistent,decision-useful internal candidate
2026,211930,2026-02-27,2026-02-28 00:00:00+00:00,feb15_plus,Fairfield,Siena,0.25,-1.016,AWAY,0.524,0.0,0.491,0.626,0.134,True,persistent,8.919,8.919,5,19,False,disagreement-led persistent,decision-useful internal candidate
```

## NCAA Caution / Diagnostic Rows

_empty_

## Raw-Edge-Only Watchlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,215105,2026-02-27,2026-02-28 01:00:00+00:00,feb15_plus,Illinois,Michigan,2.0,5.117,HOME,0.742,0.218,0.535,0.575,0.041,False,persistent,2.397,2.397,5,27,False,raw edge only / filtered out,raw-edge watchlist only
2026,211944,2026-02-27,2026-02-27 23:00:00+00:00,feb15_plus,Cornell,Yale,4.25,0.634,HOME,0.69,0.167,0.518,0.271,-0.247,False,persistent,-13.115,13.115,4,8,False,raw edge only / filtered out,raw-edge watchlist only
2026,211935,2026-02-27,2026-02-28 00:00:00+00:00,feb15_plus,Princeton,Harvard,3.75,1.541,HOME,0.654,0.131,0.52,0.434,-0.086,False,none,-3.985,3.985,0,0,False,raw edge only / filtered out,raw-edge watchlist only
2026,211937,2026-02-27,2026-02-28 00:00:00+00:00,feb15_plus,Troy,UL Monroe,-18.25,15.108,AWAY,0.612,0.088,0.505,0.1,-0.405,False,persistent,-27.232,27.232,6,28,False,raw edge only / filtered out,raw-edge watchlist only
2026,211936,2026-02-27,2026-02-28 00:00:00+00:00,feb15_plus,Pennsylvania,Dartmouth,-6.25,2.841,AWAY,0.61,0.086,0.506,0.496,-0.009,False,none,-0.755,0.755,1,1,False,raw edge only / filtered out,raw-edge watchlist only
```
