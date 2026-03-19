# Internal Bet Filter Report: 2026-02-21

## Summary

- Season: `2026`
- Training seasons used: `[2019, 2020, 2022, 2023, 2024, 2025]`
- Slate games scored: `148`
- Promoted disagreement-aware threshold: `disagreement_logit >= 0.58`
- Raw baseline watchlist threshold: `pick_prob_edge >= 0.08`
- Decision-useful non-NCAA candidates: `29`
- NCAA caution / diagnostic-only rows: `0`
- Raw-edge-only watchlist rows: `49`
- Filter-passing rows flagged mainly by disagreement features: `17`
- Slice mix: `{'feb15_plus': 148}`

## Guardrails

- Use the promoted shortlist as an internal late-season / March / conference-tournament aid, not as a public model output.
- NCAA tournament rows remain caution / diagnostic-only even if they clear the internal threshold.
- `flagged_mainly_by_disagreement = true` means the game would not have cleared the raw edge watchlist on its own.
- `raw edge only / filtered out` means the market-disagreement layer did not support the bet strongly enough.

## Ranked Internal Shortlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,358938,2026-02-21,2026-02-21 21:00:00+00:00,feb15_plus,Murray State,Evansville,-14.25,14.635,HOME,0.514,-0.01,0.492,0.778,0.286,True,persistent,20.265,20.265,7,26,False,disagreement-led persistent,decision-useful internal candidate
2026,211825,2026-02-21,2026-02-21 19:00:00+00:00,feb15_plus,Middle Tennessee,Delaware,-10.25,11.98,HOME,0.561,0.037,0.5,0.713,0.212,True,persistent,14.89,14.89,6,22,False,disagreement-led persistent,decision-useful internal candidate
2026,215012,2026-02-21,2026-02-21 19:30:00+00:00,feb15_plus,Texas Tech,Kansas State,-12.25,14.015,HOME,0.57,0.046,0.502,0.693,0.19,True,persistent,13.085,13.085,6,25,False,disagreement-led persistent,decision-useful internal candidate
2026,215049,2026-02-21,2026-02-21 19:00:00+00:00,feb15_plus,Colgate,Loyola Maryland,-8.5,10.266,HOME,0.565,0.041,0.501,0.67,0.169,True,none,12.805,12.805,0,0,False,disagreement-led support,decision-useful internal candidate
2026,358953,2026-02-21,2026-02-22 03:00:00+00:00,feb15_plus,Seattle U,Portland,-7.5,8.002,HOME,0.519,-0.005,0.493,0.67,0.177,True,persistent,11.647,11.647,5,25,False,disagreement-led persistent,decision-useful internal candidate
2026,211803,2026-02-21,2026-02-22 01:30:00+00:00,feb15_plus,Oklahoma,Texas A&M,0.0,-0.005,AWAY,0.5,-0.024,0.482,0.663,0.181,True,persistent,9.854,9.854,6,19,False,disagreement-led persistent,decision-useful internal candidate
2026,211828,2026-02-21,2026-02-21 20:30:00+00:00,feb15_plus,Sam Houston,Jacksonville State,-6.75,9.481,HOME,0.595,0.071,0.508,0.662,0.154,True,persistent,10.943,10.943,5,22,False,disagreement-led persistent,decision-useful internal candidate
2026,211778,2026-02-21,2026-02-22 00:00:00+00:00,feb15_plus,Elon,North Carolina A&T,-7.5,7.825,HOME,0.512,-0.012,0.491,0.646,0.155,True,persistent,10.145,10.145,5,21,False,disagreement-led persistent,decision-useful internal candidate
2026,211830,2026-02-21,2026-02-22 02:00:00+00:00,feb15_plus,New Mexico State,UTEP,-7.25,9.077,HOME,0.548,0.024,0.496,0.642,0.146,True,persistent,9.918,9.918,5,21,False,disagreement-led persistent,decision-useful internal candidate
2026,211809,2026-02-21,2026-02-22 01:38:00+00:00,feb15_plus,Auburn,Kentucky,-3.75,5.663,HOME,0.56,0.037,0.5,0.615,0.114,True,persistent,7.551,7.551,5,25,False,disagreement-led persistent,decision-useful internal candidate
2026,211784,2026-02-21,2026-02-21 19:00:00+00:00,feb15_plus,Mercer,Samford,-3.75,4.951,HOME,0.539,0.015,0.495,0.614,0.119,True,persistent,7.884,7.884,6,21,False,disagreement-led persistent,decision-useful internal candidate
2026,211835,2026-02-21,2026-02-21 22:00:00+00:00,feb15_plus,Portland State,Eastern Washington,-6.25,8.123,HOME,0.582,0.058,0.504,0.608,0.105,True,persistent,7.66,7.66,6,14,False,disagreement-led persistent,decision-useful internal candidate
2026,215051,2026-02-21,2026-02-22 00:00:00+00:00,feb15_plus,Fairleigh Dickinson,New Haven,-2.25,4.678,HOME,0.598,0.074,0.507,0.606,0.099,True,persistent,7.86,7.86,6,17,False,disagreement-led persistent,decision-useful internal candidate
2026,211787,2026-02-21,2026-02-22 03:00:00+00:00,feb15_plus,UC Irvine,UC San Diego,-4.5,2.005,AWAY,0.577,0.053,0.498,0.605,0.106,True,persistent,5.325,5.325,5,24,False,disagreement-led persistent,decision-useful internal candidate
2026,211829,2026-02-21,2026-02-21 22:00:00+00:00,feb15_plus,Kennesaw State,Louisiana Tech,-5.75,8.128,HOME,0.589,0.065,0.506,0.602,0.097,True,none,7.791,7.791,0,0,False,disagreement-led support,decision-useful internal candidate
```

## NCAA Caution / Diagnostic Rows

_empty_

## Raw-Edge-Only Watchlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,211807,2026-02-21,2026-02-21 23:00:00+00:00,feb15_plus,LSU,Alabama,7.5,1.496,HOME,0.792,0.268,0.549,0.298,-0.252,False,persistent,-12.994,12.994,5,25,False,raw edge only / filtered out,raw-edge watchlist only
2026,211793,2026-02-21,2026-02-21 20:00:00+00:00,feb15_plus,UL Monroe,Arkansas State,15.25,-7.84,HOME,0.788,0.264,0.536,0.145,-0.392,False,persistent,-22.074,22.074,6,26,False,raw edge only / filtered out,raw-edge watchlist only
2026,358925,2026-02-21,2026-02-22 00:30:00+00:00,feb15_plus,Mississippi Valley State,Texas Southern,12.25,-5.059,HOME,0.779,0.256,0.535,0.116,-0.419,False,persistent,-23.336,23.336,6,25,False,raw edge only / filtered out,raw-edge watchlist only
2026,211816,2026-02-21,2026-02-22 00:00:00+00:00,feb15_plus,Jacksonville,Austin Peay,6.5,1.286,HOME,0.772,0.248,0.54,0.462,-0.078,False,none,-2.693,2.693,0,0,False,raw edge only / filtered out,raw-edge watchlist only
2026,211781,2026-02-21,2026-02-21 18:00:00+00:00,feb15_plus,VMI,Western Carolina,9.0,-1.055,HOME,0.769,0.245,0.541,0.277,-0.264,False,persistent,-11.564,11.564,6,24,False,raw edge only / filtered out,raw-edge watchlist only
2026,211822,2026-02-21,2026-02-21 20:00:00+00:00,feb15_plus,UT Arlington,Utah Valley,6.25,0.737,HOME,0.734,0.21,0.534,0.446,-0.088,False,persistent,-4.253,4.253,6,16,False,raw edge only / filtered out,raw-edge watchlist only
2026,211776,2026-02-21,2026-02-21 19:00:00+00:00,feb15_plus,Northeastern,Hofstra,7.0,-0.984,HOME,0.734,0.21,0.526,0.244,-0.283,False,persistent,-14.166,14.166,2,0,False,raw edge only / filtered out,raw-edge watchlist only
2026,211757,2026-02-21,2026-02-21 20:00:00+00:00,feb15_plus,South Dakota State,North Dakota,-8.5,3.19,AWAY,0.732,0.208,0.521,0.46,-0.061,False,none,-2.193,2.193,0,0,False,raw edge only / filtered out,raw-edge watchlist only
2026,358951,2026-02-21,2026-02-22 03:00:00+00:00,feb15_plus,San Francisco,Santa Clara,6.5,0.403,HOME,0.723,0.199,0.533,0.427,-0.106,False,persistent,-5.669,5.669,6,28,False,raw edge only / filtered out,raw-edge watchlist only
2026,211806,2026-02-21,2026-02-21 17:00:00+00:00,feb15_plus,Ole Miss,Florida,13.5,-6.835,HOME,0.706,0.182,0.531,0.271,-0.26,False,persistent,-14.436,14.436,6,25,False,raw edge only / filtered out,raw-edge watchlist only
2026,215048,2026-02-21,2026-02-21 20:00:00+00:00,feb15_plus,Air Force,UNLV,16.5,-8.334,HOME,0.704,0.18,0.542,0.148,-0.394,False,persistent,-22.279,22.279,6,25,False,raw edge only / filtered out,raw-edge watchlist only
2026,215017,2026-02-21,2026-02-22 03:38:00+00:00,feb15_plus,BYU,Iowa State,2.5,3.075,HOME,0.701,0.178,0.523,0.342,-0.181,False,persistent,-9.816,9.816,6,25,False,raw edge only / filtered out,raw-edge watchlist only
2026,211786,2026-02-21,2026-02-21 22:00:00+00:00,feb15_plus,UC Riverside,UC Davis,2.75,2.626,HOME,0.691,0.167,0.521,0.46,-0.061,False,persistent,-2.646,2.646,6,10,False,raw edge only / filtered out,raw-edge watchlist only
2026,211823,2026-02-21,2026-02-21 22:00:00+00:00,feb15_plus,Tarleton State,Utah Tech,-1.5,5.174,HOME,0.684,0.16,0.516,0.383,-0.133,False,none,-4.724,4.724,0,0,False,raw edge only / filtered out,raw-edge watchlist only
2026,211768,2026-02-21,2026-02-21 19:00:00+00:00,feb15_plus,Mercyhurst,Long Island University,3.25,1.138,HOME,0.683,0.159,0.515,0.438,-0.077,False,persistent,-3.839,3.839,6,25,False,raw edge only / filtered out,raw-edge watchlist only
```
