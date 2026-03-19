# Internal Bet Filter Report: 2026-03-09

## Summary

- Season: `2026`
- Training seasons used: `[2019, 2020, 2022, 2023, 2024, 2025]`
- Slate games scored: `14`
- Promoted disagreement-aware threshold: `disagreement_logit >= 0.58`
- Raw baseline watchlist threshold: `pick_prob_edge >= 0.08`
- Decision-useful non-NCAA candidates: `2`
- NCAA caution / diagnostic-only rows: `0`
- Raw-edge-only watchlist rows: `2`
- Filter-passing rows flagged mainly by disagreement features: `2`
- Slice mix: `{'conference_tournaments': 14}`

## Guardrails

- Use the promoted shortlist as an internal late-season / March / conference-tournament aid, not as a public model output.
- NCAA tournament rows remain caution / diagnostic-only even if they clear the internal threshold.
- `flagged_mainly_by_disagreement = true` means the game would not have cleared the raw edge watchlist on its own.
- `raw edge only / filtered out` means the market-disagreement layer did not support the bet strongly enough.

## Ranked Internal Shortlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,372225,2026-03-09,2026-03-10 01:50:00+00:00,conference_tournaments,Robert Morris,Detroit Mercy,-4.5,4.626,HOME,0.504,-0.02,0.5,0.643,0.143,True,persistent,9.813,9.813,4,12,True,disagreement-led persistent,decision-useful internal candidate
2026,372322,2026-03-09,2026-03-09 22:00:00+00:00,conference_tournaments,Texas A&M-Corpus Christi,New Orleans,-2.0,3.78,HOME,0.551,0.027,0.511,0.598,0.088,True,persistent,6.189,6.189,5,18,True,disagreement-led persistent,decision-useful internal candidate
```

## NCAA Caution / Diagnostic Rows

_empty_

## Raw-Edge-Only Watchlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,372256,2026-03-09,2026-03-10 00:30:00+00:00,conference_tournaments,Grambling,Mississippi Valley State,-13.5,7.884,AWAY,0.685,0.161,0.536,0.139,-0.397,False,persistent,-22.203,22.203,7,30,True,raw edge only / filtered out,raw-edge watchlist only
2026,372328,2026-03-09,2026-03-10 01:00:00+00:00,conference_tournaments,Gonzaga,Oregon State,-20.5,15.319,AWAY,0.669,0.145,0.533,0.11,-0.423,False,persistent,-27.892,27.892,4,19,True,raw edge only / filtered out,raw-edge watchlist only
```
