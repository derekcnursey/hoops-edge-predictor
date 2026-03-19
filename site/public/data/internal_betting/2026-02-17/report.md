# Internal Bet Filter Report: 2026-02-17

## Summary

- Season: `2026`
- Training seasons used: `[2019, 2020, 2022, 2023, 2024, 2025]`
- Slate games scored: `30`
- Promoted disagreement-aware threshold: `disagreement_logit >= 0.58`
- Raw baseline watchlist threshold: `pick_prob_edge >= 0.08`
- Decision-useful non-NCAA candidates: `6`
- NCAA caution / diagnostic-only rows: `0`
- Raw-edge-only watchlist rows: `16`
- Filter-passing rows flagged mainly by disagreement features: `2`
- Slice mix: `{'feb15_plus': 30}`

## Guardrails

- Use the promoted shortlist as an internal late-season / March / conference-tournament aid, not as a public model output.
- NCAA tournament rows remain caution / diagnostic-only even if they clear the internal threshold.
- `flagged_mainly_by_disagreement = true` means the game would not have cleared the raw edge watchlist on its own.
- `raw edge only / filtered out` means the market-disagreement layer did not support the bet strongly enough.

## Ranked Internal Shortlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,211675,2026-02-17,2026-02-18 00:00:00+00:00,feb15_plus,Massachusetts,Miami (OH),3.25,-3.861,AWAY,0.515,-0.009,0.485,0.733,0.248,True,persistent,17.127,17.127,5,21,False,disagreement-led persistent,decision-useful internal candidate
2026,214971,2026-02-17,2026-02-18 01:30:00+00:00,feb15_plus,Michigan State,UCLA,-8.5,11.405,HOME,0.602,0.079,0.509,0.68,0.171,True,persistent,12.098,12.098,5,24,False,disagreement-led persistent,decision-useful internal candidate
2026,211677,2026-02-17,2026-02-17 23:30:00+00:00,feb15_plus,Eastern Michigan,Central Michigan,-4.5,10.012,HOME,0.674,0.15,0.528,0.706,0.178,False,persistent,14.217,14.217,6,10,False,raw edge + persistent disagreement support,decision-useful internal candidate
2026,214977,2026-02-17,2026-02-18 00:00:00+00:00,feb15_plus,NC State,North Carolina,-8.25,2.559,AWAY,0.697,0.173,0.524,0.646,0.123,False,persistent,7.755,7.755,5,24,False,raw edge + persistent disagreement support,decision-useful internal candidate
2026,211676,2026-02-17,2026-02-18 00:00:00+00:00,feb15_plus,Bowling Green,Kent State,-1.75,5.571,HOME,0.668,0.145,0.517,0.6,0.083,False,persistent,6.981,6.981,6,18,False,raw edge + persistent disagreement support,decision-useful internal candidate
2026,214969,2026-02-17,2026-02-18 03:45:00+00:00,feb15_plus,Oregon,Minnesota,-3.75,1.034,AWAY,0.605,0.082,0.503,0.583,0.081,False,persistent,4.474,4.474,5,14,False,raw edge + persistent disagreement support,decision-useful internal candidate
```

## NCAA Caution / Diagnostic Rows

_empty_

## Raw-Edge-Only Watchlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,214982,2026-02-17,2026-02-18 02:00:00+00:00,feb15_plus,New Mexico,Air Force,-27.75,18.704,AWAY,0.773,0.249,0.55,0.077,-0.473,False,persistent,-31.546,31.546,5,24,False,raw edge only / filtered out,raw-edge watchlist only
2026,214965,2026-02-17,2026-02-18 02:00:00+00:00,feb15_plus,Kansas State,Baylor,3.5,2.967,HOME,0.761,0.237,0.529,0.454,-0.075,False,none,-2.833,2.833,1,1,False,raw edge only / filtered out,raw-edge watchlist only
2026,214980,2026-02-17,2026-02-18 00:00:00+00:00,feb15_plus,SMU,Louisville,4.25,2.287,HOME,0.738,0.214,0.53,0.452,-0.078,False,persistent,-3.901,3.901,5,24,False,raw edge only / filtered out,raw-edge watchlist only
2026,214967,2026-02-17,2026-02-18 04:10:00+00:00,feb15_plus,Arizona State,Texas Tech,7.5,-1.21,HOME,0.707,0.183,0.528,0.319,-0.209,False,persistent,-11.251,11.251,5,24,False,raw edge only / filtered out,raw-edge watchlist only
2026,211679,2026-02-17,2026-02-18 00:00:00+00:00,feb15_plus,Western Michigan,Akron,14.25,-7.097,HOME,0.694,0.17,0.534,0.188,-0.346,False,persistent,-19.751,19.751,5,19,False,raw edge only / filtered out,raw-edge watchlist only
2026,211687,2026-02-17,2026-02-18 00:00:00+00:00,feb15_plus,Florida,South Carolina,-23.25,16.397,AWAY,0.676,0.152,0.531,0.34,-0.191,False,persistent,-10.512,10.512,5,23,False,raw edge only / filtered out,raw-edge watchlist only
2026,214981,2026-02-17,2026-02-17 23:00:00+00:00,feb15_plus,Florida State,Boston College,-11.5,6.506,AWAY,0.675,0.151,0.519,0.483,-0.035,False,none,-1.215,1.215,0,0,False,raw edge only / filtered out,raw-edge watchlist only
2026,214975,2026-02-17,2026-02-17 23:30:00+00:00,feb15_plus,Xavier,Villanova,4.5,-0.334,HOME,0.657,0.134,0.513,0.337,-0.176,False,persistent,-9.255,9.255,5,9,False,raw edge only / filtered out,raw-edge watchlist only
2026,214722,2026-02-17,2026-02-17 23:00:00+00:00,feb15_plus,Charleston Southern,Gardner-Webb,-17.5,11.735,AWAY,0.651,0.127,0.522,0.148,-0.375,False,persistent,-21.506,21.506,5,23,False,raw edge only / filtered out,raw-edge watchlist only
2026,214970,2026-02-17,2026-02-18 01:30:00+00:00,feb15_plus,Ohio State,Wisconsin,-1.25,4.913,HOME,0.641,0.117,0.515,0.491,-0.025,False,persistent,-0.498,0.498,5,25,False,raw edge only / filtered out,raw-edge watchlist only
2026,214974,2026-02-17,2026-02-18 03:00:00+00:00,feb15_plus,San Diego State,Grand Canyon,-8.25,4.478,AWAY,0.637,0.113,0.51,0.487,-0.023,False,persistent,-1.645,1.645,6,16,False,raw edge only / filtered out,raw-edge watchlist only
2026,214972,2026-02-17,2026-02-18 02:00:00+00:00,feb15_plus,Iowa,Nebraska,-1.25,4.505,HOME,0.619,0.095,0.512,0.416,-0.096,False,persistent,-4.442,4.442,6,24,False,raw edge only / filtered out,raw-edge watchlist only
2026,211678,2026-02-17,2026-02-18 00:00:00+00:00,feb15_plus,Ohio,Ball State,-9.75,7.158,AWAY,0.612,0.088,0.502,0.408,-0.095,False,persistent,-4.248,4.248,5,21,False,raw edge only / filtered out,raw-edge watchlist only
2026,358904,2026-02-17,2026-02-18 03:00:00+00:00,feb15_plus,San José State,Nevada,10.75,-7.628,HOME,0.61,0.086,0.505,0.192,-0.313,False,persistent,-18.488,18.488,5,23,False,raw edge only / filtered out,raw-edge watchlist only
2026,214966,2026-02-17,2026-02-18 00:00:00+00:00,feb15_plus,UCF,TCU,1.5,2.066,HOME,0.609,0.085,0.506,0.572,0.066,False,persistent,3.042,3.042,5,22,False,raw edge only / filtered out,raw-edge watchlist only
```
