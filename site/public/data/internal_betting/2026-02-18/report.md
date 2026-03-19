# Internal Bet Filter Report: 2026-02-18

## Summary

- Season: `2026`
- Training seasons used: `[2019, 2020, 2022, 2023, 2024, 2025]`
- Slate games scored: `59`
- Promoted disagreement-aware threshold: `disagreement_logit >= 0.58`
- Raw baseline watchlist threshold: `pick_prob_edge >= 0.08`
- Decision-useful non-NCAA candidates: `11`
- NCAA caution / diagnostic-only rows: `0`
- Raw-edge-only watchlist rows: `19`
- Filter-passing rows flagged mainly by disagreement features: `8`
- Slice mix: `{'feb15_plus': 59}`

## Guardrails

- Use the promoted shortlist as an internal late-season / March / conference-tournament aid, not as a public model output.
- NCAA tournament rows remain caution / diagnostic-only even if they clear the internal threshold.
- `flagged_mainly_by_disagreement = true` means the game would not have cleared the raw edge watchlist on its own.
- `raw edge only / filtered out` means the market-disagreement layer did not support the bet strongly enough.

## Ranked Internal Shortlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,211689,2026-02-18,2026-02-19 01:00:00+00:00,feb15_plus,Omaha,Oral Roberts,-7.0,8.641,HOME,0.559,0.036,0.5,0.677,0.177,True,persistent,12.769,12.769,6,19,False,disagreement-led persistent,decision-useful internal candidate
2026,358915,2026-02-18,2026-02-19 00:00:00+00:00,feb15_plus,Temple,UAB,-1.25,-1.157,AWAY,0.577,0.053,0.498,0.663,0.165,True,persistent,9.403,9.403,6,24,False,disagreement-led persistent,decision-useful internal candidate
2026,211695,2026-02-18,2026-02-18 23:30:00+00:00,feb15_plus,Furman,East Tennessee State,-1.5,-0.455,AWAY,0.565,0.041,0.495,0.658,0.163,True,persistent,9.219,9.219,6,22,False,disagreement-led persistent,decision-useful internal candidate
2026,214989,2026-02-18,2026-02-19 01:00:00+00:00,feb15_plus,Northwestern,Maryland,-7.5,8.61,HOME,0.544,0.02,0.498,0.643,0.146,True,persistent,9.612,9.612,6,24,False,disagreement-led persistent,decision-useful internal candidate
2026,358924,2026-02-18,2026-02-19 04:00:00+00:00,feb15_plus,UNLV,Colorado State,-2.5,1.393,AWAY,0.543,0.019,0.491,0.62,0.129,True,persistent,6.635,6.635,5,23,False,disagreement-led persistent,decision-useful internal candidate
2026,211680,2026-02-18,2026-02-19 00:00:00+00:00,feb15_plus,Texas A&M,Ole Miss,-9.5,10.927,HOME,0.555,0.031,0.499,0.617,0.117,True,persistent,8.074,8.074,5,18,False,disagreement-led persistent,decision-useful internal candidate
2026,211703,2026-02-18,2026-02-19 00:30:00+00:00,feb15_plus,Louisiana Tech,Jacksonville State,-2.25,0.063,AWAY,0.573,0.05,0.497,0.607,0.11,True,none,6.786,6.786,0,0,False,disagreement-led support,decision-useful internal candidate
2026,211690,2026-02-18,2026-02-19 01:00:00+00:00,feb15_plus,South Dakota State,North Dakota State,1.25,-1.418,AWAY,0.506,-0.018,0.488,0.589,0.101,True,persistent,6.398,6.398,4,21,False,disagreement-led persistent,decision-useful internal candidate
2026,214992,2026-02-18,2026-02-18 23:30:00+00:00,feb15_plus,Georgetown,Butler,-6.5,0.869,AWAY,0.688,0.165,0.523,0.714,0.19,False,persistent,12.649,12.649,6,25,False,raw edge + persistent disagreement support,decision-useful internal candidate
2026,358905,2026-02-18,2026-02-19 00:30:00+00:00,feb15_plus,Drake,Southern Illinois,1.5,4.231,HOME,0.674,0.15,0.523,0.594,0.071,False,persistent,3.904,3.904,7,22,False,raw edge + persistent disagreement support,decision-useful internal candidate
2026,358907,2026-02-18,2026-02-19 02:00:00+00:00,feb15_plus,Illinois State,Murray State,-2.25,6.527,HOME,0.649,0.126,0.519,0.586,0.067,False,persistent,5.249,5.249,6,25,False,raw edge + persistent disagreement support,decision-useful internal candidate
```

## NCAA Caution / Diagnostic Rows

_empty_

## Raw-Edge-Only Watchlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,358923,2026-02-18,2026-02-19 04:00:00+00:00,feb15_plus,San Francisco,Gonzaga,15.5,-3.214,HOME,0.837,0.314,0.577,0.283,-0.294,False,persistent,-15.23,15.23,6,27,False,raw edge only / filtered out,raw-edge watchlist only
2026,211696,2026-02-18,2026-02-19 00:30:00+00:00,feb15_plus,UL Monroe,Troy,17.0,-10.049,HOME,0.756,0.232,0.533,0.088,-0.445,False,persistent,-29.2,29.2,7,25,False,raw edge only / filtered out,raw-edge watchlist only
2026,211698,2026-02-18,2026-02-19 00:00:00+00:00,feb15_plus,North Alabama,Queens University,8.5,-2.384,HOME,0.753,0.229,0.527,0.38,-0.147,False,persistent,-6.207,6.207,7,14,False,raw edge only / filtered out,raw-edge watchlist only
2026,214994,2026-02-18,2026-02-19 00:00:00+00:00,feb15_plus,UConn,Creighton,-15.75,9.284,AWAY,0.739,0.215,0.53,0.393,-0.136,False,persistent,-6.94,6.94,5,25,False,raw edge only / filtered out,raw-edge watchlist only
2026,358919,2026-02-18,2026-02-19 01:00:00+00:00,feb15_plus,Tulsa,Charlotte,-12.5,6.53,AWAY,0.709,0.185,0.526,0.312,-0.214,False,none,-10.722,10.722,0,0,False,raw edge only / filtered out,raw-edge watchlist only
2026,214979,2026-02-18,2026-02-19 02:00:00+00:00,feb15_plus,Georgia Tech,Virginia,13.5,-7.378,HOME,0.667,0.143,0.526,0.195,-0.331,False,none,-19.131,19.131,0,0,False,raw edge only / filtered out,raw-edge watchlist only
2026,359076,2026-02-18,2026-02-19 01:00:00+00:00,feb15_plus,South Carolina State,North Carolina Central,2.75,1.166,HOME,0.664,0.14,0.511,0.446,-0.065,False,persistent,-1.432,1.432,4,22,False,raw edge only / filtered out,raw-edge watchlist only
2026,358920,2026-02-18,2026-02-19 02:00:00+00:00,feb15_plus,Seattle U,Saint Mary's,9.5,-3.111,HOME,0.66,0.136,0.527,0.242,-0.285,False,persistent,-16.483,16.483,5,24,False,raw edge only / filtered out,raw-edge watchlist only
2026,214976,2026-02-18,2026-02-19 00:00:00+00:00,feb15_plus,Wake Forest,Clemson,3.5,0.837,HOME,0.658,0.134,0.514,0.335,-0.179,False,persistent,-9.896,9.896,4,24,False,raw edge only / filtered out,raw-edge watchlist only
2026,211691,2026-02-18,2026-02-18 23:30:00+00:00,feb15_plus,Youngstown State,Cleveland State,-11.25,8.002,AWAY,0.656,0.132,0.507,0.36,-0.147,False,persistent,-7.181,7.181,6,19,False,raw edge only / filtered out,raw-edge watchlist only
2026,211701,2026-02-18,2026-02-19 00:00:00+00:00,feb15_plus,Delaware,Western Kentucky,4.5,0.043,HOME,0.652,0.128,0.515,0.379,-0.136,False,persistent,-6.195,6.195,6,18,False,raw edge only / filtered out,raw-edge watchlist only
2026,214984,2026-02-18,2026-02-19 02:00:00+00:00,feb15_plus,Oklahoma State,Kansas,6.5,-2.409,HOME,0.644,0.12,0.512,0.224,-0.288,False,persistent,-17.035,17.035,5,24,False,raw edge only / filtered out,raw-edge watchlist only
2026,211697,2026-02-18,2026-02-19 00:00:00+00:00,feb15_plus,Coastal Carolina,James Madison,0.25,3.398,HOME,0.639,0.115,0.509,0.491,-0.018,False,none,-0.602,0.602,0,0,False,raw edge only / filtered out,raw-edge watchlist only
2026,214983,2026-02-18,2026-02-19 01:30:00+00:00,feb15_plus,West Virginia,Utah,-10.25,7.285,AWAY,0.638,0.114,0.505,0.458,-0.047,False,persistent,-2.418,2.418,5,6,False,raw edge only / filtered out,raw-edge watchlist only
2026,214996,2026-02-18,2026-02-19 00:00:00+00:00,feb15_plus,Loyola Maryland,Army,-4.75,0.908,AWAY,0.633,0.109,0.51,0.569,0.059,False,persistent,4.257,4.257,6,23,False,raw edge only / filtered out,raw-edge watchlist only
```
