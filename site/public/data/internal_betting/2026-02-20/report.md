# Internal Bet Filter Report: 2026-02-20

## Summary

- Season: `2026`
- Training seasons used: `[2019, 2020, 2022, 2023, 2024, 2025]`
- Slate games scored: `13`
- Promoted disagreement-aware threshold: `disagreement_logit >= 0.58`
- Raw baseline watchlist threshold: `pick_prob_edge >= 0.08`
- Decision-useful non-NCAA candidates: `2`
- NCAA caution / diagnostic-only rows: `0`
- Raw-edge-only watchlist rows: `3`
- Filter-passing rows flagged mainly by disagreement features: `2`
- Slice mix: `{'feb15_plus': 13}`

## Guardrails

- Use the promoted shortlist as an internal late-season / March / conference-tournament aid, not as a public model output.
- NCAA tournament rows remain caution / diagnostic-only even if they clear the internal threshold.
- `flagged_mainly_by_disagreement = true` means the game would not have cleared the raw edge watchlist on its own.
- `raw edge only / filtered out` means the market-disagreement layer did not support the bet strongly enough.

## Ranked Internal Shortlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,211751,2026-02-20,2026-02-21 00:00:00+00:00,feb15_plus,Manhattan,Marist,5.25,-5.636,AWAY,0.515,-0.009,0.492,0.772,0.28,True,persistent,20.056,20.056,7,18,False,disagreement-led persistent,decision-useful internal candidate
2026,211749,2026-02-20,2026-02-21 01:00:00+00:00,feb15_plus,Merrimack,Siena,-2.5,0.564,AWAY,0.568,0.044,0.496,0.604,0.108,True,persistent,5.693,5.693,6,17,False,disagreement-led persistent,decision-useful internal candidate
```

## NCAA Caution / Diagnostic Rows

_empty_

## Raw-Edge-Only Watchlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,211764,2026-02-20,2026-02-20 23:30:00+00:00,feb15_plus,Ball State,Akron,13.5,-9.382,HOME,0.652,0.129,0.512,0.118,-0.395,False,persistent,-25.394,25.394,6,22,False,raw edge only / filtered out,raw-edge watchlist only
2026,215008,2026-02-20,2026-02-21 01:00:00+00:00,feb15_plus,Purdue,Indiana,-12.0,8.649,AWAY,0.649,0.126,0.508,0.382,-0.125,False,persistent,-7.143,7.143,5,25,False,raw edge only / filtered out,raw-edge watchlist only
2026,211750,2026-02-20,2026-02-21 00:00:00+00:00,feb15_plus,Rider,Canisius,-1.25,-1.379,AWAY,0.627,0.103,0.503,0.562,0.059,False,persistent,4.961,4.961,6,25,False,raw edge only / filtered out,raw-edge watchlist only
```
