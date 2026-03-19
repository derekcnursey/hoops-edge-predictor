# Internal Bet Filter Report: 2026-03-12

## Summary

- Season: `2026`
- Training seasons used: `[2019, 2020, 2022, 2023, 2024, 2025]`
- Slate games scored: `44`
- Promoted disagreement-aware threshold: `disagreement_logit >= 0.58`
- Raw baseline watchlist threshold: `pick_prob_edge >= 0.08`
- Decision-useful non-NCAA candidates: `8`
- NCAA caution / diagnostic-only rows: `0`
- Raw-edge-only watchlist rows: `2`
- Filter-passing rows flagged mainly by disagreement features: `7`
- Slice mix: `{'conference_tournaments': 44}`

## Guardrails

- Use the promoted shortlist as an internal late-season / March / conference-tournament aid, not as a public model output.
- NCAA tournament rows remain caution / diagnostic-only even if they clear the internal threshold.
- `flagged_mainly_by_disagreement = true` means the game would not have cleared the raw edge watchlist on its own.
- `raw edge only / filtered out` means the market-disagreement layer did not support the bet strongly enough.

## Ranked Internal Shortlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,372383,2026-03-12,2026-03-13 01:00:00+00:00,conference_tournaments,Cal State Northridge,UC San Diego,,-5.122,AWAY,,,0.486,0.726,0.239,True,persistent,14.293,14.293,6,30,True,disagreement-led persistent,decision-useful internal candidate
2026,372258,2026-03-12,2026-03-12 15:00:00+00:00,conference_tournaments,Miami (OH),Massachusetts,-7.75,7.793,HOME,0.501,-0.023,0.497,0.723,0.226,True,persistent,15.244,15.244,5,27,True,disagreement-led persistent,decision-useful internal candidate
2026,372310,2026-03-12,2026-03-12 15:30:00+00:00,conference_tournaments,Fordham,George Washington,5.75,-6.542,AWAY,0.526,0.002,0.506,0.664,0.159,True,persistent,10.5,10.5,4,28,True,disagreement-led persistent,decision-useful internal candidate
2026,372389,2026-03-12,2026-03-13 01:30:00+00:00,conference_tournaments,North Texas,Florida Atlantic,,-1.274,AWAY,,,0.49,0.649,0.159,True,persistent,8.452,8.452,5,28,True,disagreement-led persistent,decision-useful internal candidate
2026,372366,2026-03-12,2026-03-12 16:30:00+00:00,conference_tournaments,Texas Tech,Iowa State,,-1.117,AWAY,,,0.488,0.642,0.154,True,persistent,7.743,7.743,6,31,True,disagreement-led persistent,decision-useful internal candidate
2026,372370,2026-03-12,2026-03-12 18:30:00+00:00,conference_tournaments,Miami,Louisville,,-1.814,AWAY,,,0.489,0.584,0.095,True,persistent,3.781,3.781,6,31,True,disagreement-led persistent,decision-useful internal candidate
2026,372316,2026-03-12,2026-03-12 22:30:00+00:00,conference_tournaments,Louisiana Tech,Middle Tennessee,1.5,-3.101,AWAY,0.548,0.024,0.51,0.58,0.07,True,persistent,4.603,4.603,5,27,True,disagreement-led persistent,decision-useful internal candidate
2026,372312,2026-03-12,2026-03-12 18:30:00+00:00,conference_tournaments,Seton Hall,Creighton,-2.75,-1.074,AWAY,0.61,0.087,0.521,0.59,0.069,False,persistent,3.156,3.156,4,30,True,raw edge + persistent disagreement support,decision-useful internal candidate
```

## NCAA Caution / Diagnostic Rows

_empty_

## Raw-Edge-Only Watchlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,372313,2026-03-12,2026-03-12 20:00:00+00:00,conference_tournaments,Akron,Buffalo,-13.5,8.343,AWAY,0.679,0.156,0.533,0.214,-0.319,False,none,-17.342,17.342,0,0,True,raw edge only / filtered out,raw-edge watchlist only
2026,372261,2026-03-12,2026-03-13 01:10:00+00:00,conference_tournaments,North Carolina Central,Maryland Eastern Shore,-1.5,-1.309,AWAY,0.607,0.083,0.516,0.562,0.046,False,persistent,3.919,3.919,4,28,True,raw edge only / filtered out,raw-edge watchlist only
```
