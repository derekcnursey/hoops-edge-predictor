# Internal Bet Filter Report: 2026-02-23

## Summary

- Season: `2026`
- Training seasons used: `[2019, 2020, 2022, 2023, 2024, 2025]`
- Slate games scored: `9`
- Promoted disagreement-aware threshold: `disagreement_logit >= 0.58`
- Raw baseline watchlist threshold: `pick_prob_edge >= 0.08`
- Decision-useful non-NCAA candidates: `1`
- NCAA caution / diagnostic-only rows: `0`
- Raw-edge-only watchlist rows: `3`
- Filter-passing rows flagged mainly by disagreement features: `1`
- Slice mix: `{'feb15_plus': 9}`

## Guardrails

- Use the promoted shortlist as an internal late-season / March / conference-tournament aid, not as a public model output.
- NCAA tournament rows remain caution / diagnostic-only even if they clear the internal threshold.
- `flagged_mainly_by_disagreement = true` means the game would not have cleared the raw edge watchlist on its own.
- `raw edge only / filtered out` means the market-disagreement layer did not support the bet strongly enough.

## Ranked Internal Shortlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,211849,2026-02-23,2026-02-24 00:30:00+00:00,feb15_plus,McNeese,UT Rio Grande Valley,-11.75,13.104,HOME,0.552,0.029,0.499,0.662,0.163,True,persistent,10.83,10.83,6,25,False,disagreement-led persistent,decision-useful internal candidate
```

## NCAA Caution / Diagnostic Rows

_empty_

## Raw-Edge-Only Watchlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,215060,2026-02-23,2026-02-24 00:00:00+00:00,feb15_plus,North Carolina,Louisville,3.5,3.24,HOME,0.719,0.195,0.532,0.564,0.032,False,persistent,1.655,1.655,6,26,False,raw edge only / filtered out,raw-edge watchlist only
2026,211850,2026-02-23,2026-02-24 00:30:00+00:00,feb15_plus,East Texas A&M,Houston Christian,-2.5,6.222,HOME,0.639,0.116,0.516,0.553,0.038,False,persistent,4.365,4.365,6,25,False,raw edge only / filtered out,raw-edge watchlist only
2026,215061,2026-02-23,2026-02-24 02:00:00+00:00,feb15_plus,Kansas,Houston,1.75,1.956,HOME,0.631,0.107,0.509,0.472,-0.037,False,persistent,-2.869,2.869,6,26,False,raw edge only / filtered out,raw-edge watchlist only
```
