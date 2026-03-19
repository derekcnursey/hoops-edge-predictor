# Internal Bet Filter Report: 2026-02-26

## Summary

- Season: `2026`
- Training seasons used: `[2019, 2020, 2022, 2023, 2024, 2025]`
- Slate games scored: `56`
- Promoted disagreement-aware threshold: `disagreement_logit >= 0.58`
- Raw baseline watchlist threshold: `pick_prob_edge >= 0.08`
- Decision-useful non-NCAA candidates: `14`
- NCAA caution / diagnostic-only rows: `0`
- Raw-edge-only watchlist rows: `16`
- Filter-passing rows flagged mainly by disagreement features: `11`
- Slice mix: `{'feb15_plus': 56}`

## Guardrails

- Use the promoted shortlist as an internal late-season / March / conference-tournament aid, not as a public model output.
- NCAA tournament rows remain caution / diagnostic-only even if they clear the internal threshold.
- `flagged_mainly_by_disagreement = true` means the game would not have cleared the raw edge watchlist on its own.
- `raw edge only / filtered out` means the market-disagreement layer did not support the bet strongly enough.

## Ranked Internal Shortlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,211904,2026-02-26,2026-02-27 03:00:00+00:00,feb15_plus,UC San Diego,Cal State Bakersfield,-14.5,17.617,HOME,0.587,0.064,0.508,0.843,0.335,True,persistent,27.026,27.026,5,25,False,disagreement-led persistent,decision-useful internal candidate
2026,211906,2026-02-26,2026-02-27 02:00:00+00:00,feb15_plus,Montana,Sacramento State,-7.75,10.083,HOME,0.583,0.059,0.505,0.696,0.191,True,persistent,14.194,14.194,5,9,False,disagreement-led persistent,decision-useful internal candidate
2026,211926,2026-02-26,2026-02-27 00:30:00+00:00,feb15_plus,Middle Tennessee,UTEP,-8.5,10.038,HOME,0.544,0.02,0.495,0.655,0.16,True,persistent,10.746,10.746,5,23,False,disagreement-led persistent,decision-useful internal candidate
2026,211893,2026-02-26,2026-02-26 23:00:00+00:00,feb15_plus,UMBC,Bryant,-10.5,10.509,HOME,0.5,-0.024,0.485,0.642,0.157,True,persistent,10.666,10.666,5,21,False,disagreement-led persistent,decision-useful internal candidate
2026,211921,2026-02-26,2026-02-27 02:00:00+00:00,feb15_plus,Utah Tech,Abilene Christian,-2.25,4.502,HOME,0.574,0.051,0.503,0.633,0.13,True,persistent,9.415,9.415,5,13,False,disagreement-led persistent,decision-useful internal candidate
2026,211901,2026-02-26,2026-02-27 00:00:00+00:00,feb15_plus,Chattanooga,UNC Greensboro,-4.5,5.146,HOME,0.525,0.002,0.494,0.621,0.127,True,none,9.245,9.245,0,0,False,disagreement-led support,decision-useful internal candidate
2026,211922,2026-02-26,2026-02-27 00:00:00+00:00,feb15_plus,Jacksonville State,Delaware,-7.75,9.251,HOME,0.551,0.027,0.498,0.619,0.121,True,none,9.219,9.219,0,0,False,disagreement-led support,decision-useful internal candidate
2026,211925,2026-02-26,2026-02-27 02:00:00+00:00,feb15_plus,Sam Houston,Florida International,-6.25,7.575,HOME,0.549,0.026,0.498,0.61,0.112,True,persistent,7.515,7.515,4,23,False,disagreement-led persistent,decision-useful internal candidate
2026,211917,2026-02-26,2026-02-27 01:00:00+00:00,feb15_plus,Little Rock,Morehead State,-2.5,-0.026,AWAY,0.604,0.08,0.502,0.6,0.099,True,persistent,5.718,5.718,7,24,False,disagreement-led persistent,decision-useful internal candidate
2026,211924,2026-02-26,2026-02-27 00:30:00+00:00,feb15_plus,Louisiana Tech,Missouri State,-2.25,-0.061,AWAY,0.593,0.069,0.5,0.595,0.095,True,none,5.988,5.988,0,0,False,disagreement-led support,decision-useful internal candidate
2026,211916,2026-02-26,2026-02-27 01:30:00+00:00,feb15_plus,Southeast Missouri State,Tennessee State,-3.5,1.642,AWAY,0.571,0.047,0.496,0.587,0.091,True,none,5.144,5.144,1,1,False,disagreement-led support,decision-useful internal candidate
2026,358982,2026-02-26,2026-02-27 00:00:00+00:00,feb15_plus,Florida Atlantic,Temple,-4.0,8.35,HOME,0.638,0.114,0.519,0.656,0.137,False,persistent,10.186,10.186,4,24,False,raw edge + persistent disagreement support,decision-useful internal candidate
2026,211909,2026-02-26,2026-02-27 00:00:00+00:00,feb15_plus,Jacksonville,Stetson,-6.75,10.11,HOME,0.621,0.097,0.513,0.651,0.138,False,none,11.257,11.257,0,0,False,raw edge led,decision-useful internal candidate
2026,215104,2026-02-26,2026-02-27 02:30:00+00:00,feb15_plus,Prairie View A&M,Jackson State,-4.25,7.252,HOME,0.608,0.084,0.51,0.586,0.076,False,persistent,7.368,7.368,6,23,False,raw edge + persistent disagreement support,decision-useful internal candidate
```

## NCAA Caution / Diagnostic Rows

_empty_

## Raw-Edge-Only Watchlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,215099,2026-02-26,2026-02-27 00:00:00+00:00,feb15_plus,Charleston Southern,Winthrop,7.75,-0.593,HOME,0.758,0.234,0.535,0.394,-0.141,False,none,-6.147,6.147,0,0,False,raw edge only / filtered out,raw-edge watchlist only
2026,211894,2026-02-26,2026-02-26 16:00:00+00:00,feb15_plus,Hampton,Charleston,5.5,2.664,HOME,0.747,0.223,0.543,0.428,-0.115,False,persistent,-5.187,5.187,5,25,False,raw edge only / filtered out,raw-edge watchlist only
2026,211915,2026-02-26,2026-02-27 01:30:00+00:00,feb15_plus,SIU Edwardsville,Western Illinois,-14.0,6.334,AWAY,0.738,0.214,0.539,0.249,-0.29,False,persistent,-13.899,13.899,6,25,False,raw edge only / filtered out,raw-edge watchlist only
2026,211911,2026-02-26,2026-02-27 00:00:00+00:00,feb15_plus,North Florida,Florida Gulf Coast,6.75,0.125,HOME,0.727,0.203,0.533,0.396,-0.137,False,persistent,-5.891,5.891,6,21,False,raw edge only / filtered out,raw-edge watchlist only
2026,211900,2026-02-26,2026-02-27 01:00:00+00:00,feb15_plus,Samford,VMI,-16.75,11.758,AWAY,0.706,0.183,0.519,0.213,-0.306,False,persistent,-15.24,15.24,5,25,False,raw edge only / filtered out,raw-edge watchlist only
2026,211928,2026-02-26,2026-02-27 00:00:00+00:00,feb15_plus,Central Connecticut,Mercyhurst,-3.75,-1.515,AWAY,0.675,0.152,0.52,0.575,0.055,False,persistent,3.423,3.423,6,26,False,raw edge only / filtered out,raw-edge watchlist only
2026,211884,2026-02-26,2026-02-27 01:00:00+00:00,feb15_plus,Kansas City,South Dakota State,12.5,-8.065,HOME,0.665,0.142,0.515,0.195,-0.32,False,persistent,-17.484,17.484,4,25,False,raw edge only / filtered out,raw-edge watchlist only
2026,211891,2026-02-26,2026-02-27 00:00:00+00:00,feb15_plus,Vermont,UMass Lowell,-9.5,5.614,AWAY,0.657,0.133,0.511,0.389,-0.122,False,persistent,-5.792,5.792,5,24,False,raw edge only / filtered out,raw-edge watchlist only
2026,215101,2026-02-26,2026-02-27 00:00:00+00:00,feb15_plus,Gardner-Webb,UNC Asheville,12.5,-6.832,HOME,0.655,0.131,0.522,0.173,-0.349,False,persistent,-19.076,19.076,5,25,False,raw edge only / filtered out,raw-edge watchlist only
2026,211897,2026-02-26,2026-02-27 00:00:00+00:00,feb15_plus,UNC Wilmington,North Carolina A&T,-13.5,10.496,AWAY,0.633,0.109,0.505,0.295,-0.21,False,persistent,-11.596,11.596,5,23,False,raw edge only / filtered out,raw-edge watchlist only
2026,211914,2026-02-26,2026-02-27 01:30:00+00:00,feb15_plus,UT Martin,Tennessee Tech,-7.5,2.96,AWAY,0.615,0.091,0.512,0.43,-0.082,False,persistent,-4.093,4.093,6,15,False,raw edge only / filtered out,raw-edge watchlist only
2026,211920,2026-02-26,2026-02-27 01:00:00+00:00,feb15_plus,Utah Valley,Tarleton State,-17.25,14.167,AWAY,0.613,0.089,0.505,0.284,-0.221,False,none,-12.027,12.027,0,0,False,raw edge only / filtered out,raw-edge watchlist only
2026,211895,2026-02-26,2026-02-27 00:00:00+00:00,feb15_plus,Drexel,Campbell,-1.25,4.162,HOME,0.611,0.087,0.51,0.555,0.045,False,none,4.462,4.462,1,1,False,raw edge only / filtered out,raw-edge watchlist only
2026,215102,2026-02-26,2026-02-27 00:00:00+00:00,feb15_plus,Presbyterian,High Point,12.0,-8.673,HOME,0.609,0.086,0.505,0.172,-0.334,False,persistent,-20.637,20.637,5,25,False,raw edge only / filtered out,raw-edge watchlist only
2026,211882,2026-02-26,2026-02-27 01:00:00+00:00,feb15_plus,Oral Roberts,Denver,2.5,-0.039,HOME,0.608,0.084,0.502,0.34,-0.162,False,persistent,-7.67,7.67,5,25,False,raw edge only / filtered out,raw-edge watchlist only
```
