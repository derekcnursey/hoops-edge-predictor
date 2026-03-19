# Internal Bet Filter Report: 2026-02-25

## Summary

- Season: `2026`
- Training seasons used: `[2019, 2020, 2022, 2023, 2024, 2025]`
- Slate games scored: `53`
- Promoted disagreement-aware threshold: `disagreement_logit >= 0.58`
- Raw baseline watchlist threshold: `pick_prob_edge >= 0.08`
- Decision-useful non-NCAA candidates: `10`
- NCAA caution / diagnostic-only rows: `0`
- Raw-edge-only watchlist rows: `15`
- Filter-passing rows flagged mainly by disagreement features: `8`
- Slice mix: `{'feb15_plus': 53}`

## Guardrails

- Use the promoted shortlist as an internal late-season / March / conference-tournament aid, not as a public model output.
- NCAA tournament rows remain caution / diagnostic-only even if they clear the internal threshold.
- `flagged_mainly_by_disagreement = true` means the game would not have cleared the raw edge watchlist on its own.
- `raw edge only / filtered out` means the market-disagreement layer did not support the bet strongly enough.

## Ranked Internal Shortlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,211862,2026-02-25,2026-02-26 02:00:00+00:00,feb15_plus,Alabama,Mississippi State,-12.75,15.139,HOME,0.572,0.049,0.503,0.777,0.274,True,persistent,20.0,20.0,6,26,False,disagreement-led persistent,decision-useful internal candidate
2026,215075,2026-02-25,2026-02-26 04:00:00+00:00,feb15_plus,San Diego State,Utah State,-0.25,-1.762,AWAY,0.58,0.056,0.498,0.716,0.218,True,persistent,13.349,13.349,6,25,False,disagreement-led persistent,decision-useful internal candidate
2026,215094,2026-02-25,2026-02-26 00:00:00+00:00,feb15_plus,Loyola Maryland,Navy,6.75,-6.84,AWAY,0.503,-0.021,0.485,0.703,0.218,True,none,15.526,15.526,0,0,False,disagreement-led support,decision-useful internal candidate
2026,211874,2026-02-25,2026-02-26 00:00:00+00:00,feb15_plus,Western Carolina,Mercer,0.25,-0.298,AWAY,0.502,-0.022,0.49,0.673,0.183,True,persistent,11.959,11.959,6,22,False,disagreement-led persistent,decision-useful internal candidate
2026,211871,2026-02-25,2026-02-26 00:00:00+00:00,feb15_plus,Robert Morris,Detroit Mercy,-7.75,8.209,HOME,0.52,-0.004,0.494,0.655,0.161,True,persistent,11.188,11.188,5,9,False,disagreement-led persistent,decision-useful internal candidate
2026,358980,2026-02-25,2026-02-26 04:00:00+00:00,feb15_plus,Saint Mary's,Santa Clara,-5.5,6.933,HOME,0.548,0.025,0.498,0.604,0.107,True,persistent,6.736,6.736,6,26,False,disagreement-led persistent,decision-useful internal candidate
2026,358967,2026-02-25,2026-02-26 00:00:00+00:00,feb15_plus,Saint Joseph's,George Mason,0.25,-0.511,AWAY,0.509,-0.015,0.489,0.602,0.114,True,persistent,7.057,7.057,5,25,False,disagreement-led persistent,decision-useful internal candidate
2026,211866,2026-02-25,2026-02-26 02:00:00+00:00,feb15_plus,Arkansas,Texas A&M,-7.75,7.99,HOME,0.509,-0.015,0.49,0.585,0.095,True,persistent,5.649,5.649,5,26,False,disagreement-led persistent,decision-useful internal candidate
2026,211876,2026-02-25,2026-02-26 00:00:00+00:00,feb15_plus,East Tennessee State,Wofford,-6.75,9.596,HOME,0.613,0.09,0.51,0.689,0.18,False,persistent,12.898,12.898,6,24,False,raw edge + persistent disagreement support,decision-useful internal candidate
2026,358963,2026-02-25,2026-02-26 00:00:00+00:00,feb15_plus,Northern Iowa,Illinois State,-4.75,0.913,AWAY,0.643,0.12,0.51,0.605,0.095,False,persistent,5.054,5.054,6,27,False,raw edge + persistent disagreement support,decision-useful internal candidate
```

## NCAA Caution / Diagnostic Rows

_empty_

## Raw-Edge-Only Watchlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,215081,2026-02-25,2026-02-26 03:00:00+00:00,feb15_plus,California,SMU,4.25,1.803,HOME,0.707,0.183,0.527,0.391,-0.136,False,persistent,-7.139,7.139,5,23,False,raw edge only / filtered out,raw-edge watchlist only
2026,215091,2026-02-25,2026-02-26 00:00:00+00:00,feb15_plus,Villanova,Butler,-10.5,4.671,AWAY,0.703,0.179,0.525,0.513,-0.012,False,persistent,-1.002,1.002,6,27,False,raw edge only / filtered out,raw-edge watchlist only
2026,211881,2026-02-25,2026-02-25 23:30:00+00:00,feb15_plus,IU Indianapolis,Oakland,5.5,0.692,HOME,0.693,0.17,0.527,0.274,-0.254,False,persistent,-13.195,13.195,5,27,False,raw edge only / filtered out,raw-edge watchlist only
2026,358976,2026-02-25,2026-02-26 02:00:00+00:00,feb15_plus,Gonzaga,Portland,-26.75,21.403,AWAY,0.67,0.146,0.521,0.084,-0.436,False,persistent,-31.001,31.001,5,26,False,raw edge only / filtered out,raw-edge watchlist only
2026,211879,2026-02-25,2026-02-26 00:00:00+00:00,feb15_plus,Texas,Florida,6.5,-1.037,HOME,0.666,0.142,0.521,0.384,-0.137,False,persistent,-7.595,7.595,4,25,False,raw edge only / filtered out,raw-edge watchlist only
2026,372147,2026-02-25,2026-02-25 23:00:00+00:00,feb15_plus,South Carolina State,Morgan State,1.5,2.249,HOME,0.662,0.139,0.51,0.49,-0.02,False,persistent,0.845,0.845,6,24,False,raw edge only / filtered out,raw-edge watchlist only
2026,211873,2026-02-25,2026-02-26 00:00:00+00:00,feb15_plus,Cleveland State,Northern Kentucky,8.5,-3.902,HOME,0.658,0.134,0.515,0.355,-0.161,False,persistent,-7.962,7.962,6,21,False,raw edge only / filtered out,raw-edge watchlist only
2026,358969,2026-02-25,2026-02-26 00:00:00+00:00,feb15_plus,Charlotte,North Texas,0.25,4.107,HOME,0.656,0.132,0.514,0.561,0.047,False,none,3.288,3.288,0,0,False,raw edge only / filtered out,raw-edge watchlist only
2026,211864,2026-02-25,2026-02-26 00:00:00+00:00,feb15_plus,Vanderbilt,Georgia,-7.5,3.621,AWAY,0.644,0.12,0.511,0.503,-0.008,False,persistent,-1.163,1.163,5,26,False,raw edge only / filtered out,raw-edge watchlist only
2026,215078,2026-02-25,2026-02-26 01:00:00+00:00,feb15_plus,Stanford,Pittsburgh,-8.5,4.712,AWAY,0.637,0.114,0.51,0.495,-0.015,False,none,-0.678,0.678,0,0,False,raw edge only / filtered out,raw-edge watchlist only
2026,358974,2026-02-25,2026-02-26 01:00:00+00:00,feb15_plus,Rice,South Florida,11.5,-7.663,HOME,0.632,0.108,0.51,0.212,-0.298,False,persistent,-16.899,16.899,5,20,False,raw edge only / filtered out,raw-edge watchlist only
2026,215088,2026-02-25,2026-02-26 00:00:00+00:00,feb15_plus,Nebraska,Maryland,-17.25,13.857,AWAY,0.63,0.106,0.507,0.16,-0.347,False,none,-21.465,21.465,0,0,False,raw edge only / filtered out,raw-edge watchlist only
2026,215087,2026-02-25,2026-02-26 04:00:00+00:00,feb15_plus,Oregon,Wisconsin,5.75,-2.363,HOME,0.627,0.103,0.507,0.231,-0.276,False,none,-15.44,15.44,1,1,False,raw edge only / filtered out,raw-edge watchlist only
2026,211880,2026-02-25,2026-02-26 00:00:00+00:00,feb15_plus,Eastern Kentucky,Queens University,2.5,0.874,HOME,0.624,0.1,0.507,0.441,-0.066,False,persistent,-3.651,3.651,6,25,False,raw edge only / filtered out,raw-edge watchlist only
2026,211870,2026-02-25,2026-02-26 01:00:00+00:00,feb15_plus,Milwaukee,Youngstown State,1.25,1.973,HOME,0.621,0.097,0.506,0.501,-0.005,False,none,-0.295,0.295,0,0,False,raw edge only / filtered out,raw-edge watchlist only
```
