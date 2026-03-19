# Internal Bet Filter Report: 2026-03-03

## Summary

- Season: `2026`
- Training seasons used: `[2019, 2020, 2022, 2023, 2024, 2025]`
- Slate games scored: `52`
- Promoted disagreement-aware threshold: `disagreement_logit >= 0.58`
- Raw baseline watchlist threshold: `pick_prob_edge >= 0.08`
- Decision-useful non-NCAA candidates: `9`
- NCAA caution / diagnostic-only rows: `0`
- Raw-edge-only watchlist rows: `17`
- Filter-passing rows flagged mainly by disagreement features: `9`
- Slice mix: `{'march_only': 52}`

## Guardrails

- Use the promoted shortlist as an internal late-season / March / conference-tournament aid, not as a public model output.
- NCAA tournament rows remain caution / diagnostic-only even if they clear the internal threshold.
- `flagged_mainly_by_disagreement = true` means the game would not have cleared the raw edge watchlist on its own.
- `raw edge only / filtered out` means the market-disagreement layer did not support the bet strongly enough.

## Ranked Internal Shortlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,212048,2026-03-03,2026-03-03 23:00:00+00:00,march_only,NJIT,UMBC,4.5,-4.659,AWAY,0.505,-0.019,0.483,0.706,0.223,True,new/transient,18.837,18.837,0,0,False,disagreement-led new/transient march,decision-useful internal candidate
2026,212041,2026-03-03,2026-03-04 00:00:00+00:00,march_only,Miami (OH),Toledo,-7.75,8.938,HOME,0.549,0.025,0.494,0.701,0.207,True,persistent,13.895,13.895,5,25,False,disagreement-led persistent,decision-useful internal candidate
2026,215171,2026-03-03,2026-03-04 04:00:00+00:00,march_only,UCLA,Nebraska,0.0,-2.15,AWAY,0.583,0.06,0.494,0.697,0.202,True,persistent,11.751,11.751,6,28,False,disagreement-led persistent,decision-useful internal candidate
2026,215181,2026-03-03,2026-03-04 02:00:00+00:00,march_only,Louisville,Syracuse,-13.25,15.588,HOME,0.584,0.06,0.501,0.687,0.187,True,persistent,12.663,12.663,5,28,False,disagreement-led persistent,decision-useful internal candidate
2026,212055,2026-03-03,2026-03-04 00:00:00+00:00,march_only,Hofstra,Drexel,-8.5,10.936,HOME,0.565,0.042,0.497,0.615,0.118,True,persistent,7.784,7.784,5,22,False,disagreement-led persistent,decision-useful internal candidate
2026,212057,2026-03-03,2026-03-04 00:00:00+00:00,march_only,Oklahoma,Missouri,-1.25,0.147,AWAY,0.542,0.018,0.487,0.61,0.124,True,persistent,5.714,5.714,6,28,False,disagreement-led persistent,decision-useful internal candidate
2026,215167,2026-03-03,2026-03-04 00:00:00+00:00,march_only,Cincinnati,BYU,-2.0,0.96,AWAY,0.537,0.013,0.485,0.594,0.109,True,persistent,4.67,4.67,6,28,False,disagreement-led persistent,decision-useful internal candidate
2026,372151,2026-03-03,2026-03-03 23:00:00+00:00,march_only,Louisiana,Georgia State,1.5,-4.117,AWAY,0.581,0.057,0.496,0.586,0.09,True,persistent,7.505,7.505,3,3,True,disagreement-led persistent,decision-useful internal candidate
2026,372154,2026-03-03,2026-03-04 00:00:00+00:00,march_only,Bucknell,Army,-0.25,-1.943,AWAY,0.574,0.05,0.493,0.583,0.09,True,persistent,5.325,5.325,6,27,False,disagreement-led persistent,decision-useful internal candidate
```

## NCAA Caution / Diagnostic Rows

_empty_

## Raw-Edge-Only Watchlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,212043,2026-03-03,2026-03-04 01:00:00+00:00,march_only,Northern Illinois,Kent State,10.0,-4.542,HOME,0.752,0.228,0.517,0.263,-0.254,False,persistent,-12.13,12.13,6,25,False,raw edge only / filtered out,raw-edge watchlist only
2026,215164,2026-03-03,2026-03-04 02:00:00+00:00,march_only,Utah,Colorado,1.25,4.412,HOME,0.73,0.206,0.519,0.529,0.01,False,persistent,1.011,1.011,6,10,False,raw edge only / filtered out,raw-edge watchlist only
2026,215169,2026-03-03,2026-03-04 02:00:00+00:00,march_only,Arizona State,Kansas,5.5,0.413,HOME,0.727,0.203,0.521,0.282,-0.24,False,persistent,-13.24,13.24,5,28,False,raw edge only / filtered out,raw-edge watchlist only
2026,359018,2026-03-03,2026-03-03 05:00:00+00:00,march_only,Fresno State,San José State,-7.5,1.87,AWAY,0.723,0.199,0.519,0.364,-0.155,False,persistent,-7.518,7.518,6,27,False,raw edge only / filtered out,raw-edge watchlist only
2026,212059,2026-03-03,2026-03-03 23:00:00+00:00,march_only,South Carolina,Tennessee,10.75,-4.344,HOME,0.716,0.192,0.525,0.299,-0.225,False,persistent,-12.433,12.433,5,27,False,raw edge only / filtered out,raw-edge watchlist only
2026,212056,2026-03-03,2026-03-04 00:00:00+00:00,march_only,Northeastern,Monmouth,3.75,3.376,HOME,0.701,0.177,0.53,0.414,-0.116,False,persistent,-5.018,5.018,3,3,False,raw edge only / filtered out,raw-edge watchlist only
2026,212060,2026-03-03,2026-03-03 23:30:00+00:00,march_only,Georgia,Alabama,1.5,3.961,HOME,0.699,0.175,0.518,0.484,-0.034,False,persistent,-2.358,2.358,6,28,False,raw edge only / filtered out,raw-edge watchlist only
2026,215184,2026-03-03,2026-03-03 05:00:00+00:00,march_only,Air Force,Grand Canyon,19.25,-13.088,HOME,0.685,0.161,0.522,0.105,-0.417,False,persistent,-26.751,26.751,6,28,False,raw edge only / filtered out,raw-edge watchlist only
2026,215174,2026-03-03,2026-03-04 00:00:00+00:00,march_only,Xavier,Seton Hall,1.75,3.449,HOME,0.683,0.159,0.516,0.55,0.034,False,persistent,1.984,1.984,5,13,False,raw edge only / filtered out,raw-edge watchlist only
2026,212050,2026-03-03,2026-03-04 00:00:00+00:00,march_only,Vermont,UAlbany,-9.5,5.693,AWAY,0.677,0.153,0.506,0.492,-0.014,False,persistent,-0.785,0.785,5,26,False,raw edge only / filtered out,raw-edge watchlist only
2026,212051,2026-03-03,2026-03-04 00:00:00+00:00,march_only,Elon,UNC Wilmington,4.75,-0.08,HOME,0.651,0.127,0.511,0.39,-0.121,False,none,-6.278,6.278,0,0,False,raw edge only / filtered out,raw-edge watchlist only
2026,215166,2026-03-03,2026-03-04 01:00:00+00:00,march_only,Kansas State,West Virginia,4.25,-0.294,HOME,0.646,0.122,0.507,0.439,-0.068,False,none,-3.473,3.473,0,0,False,raw edge only / filtered out,raw-edge watchlist only
2026,359019,2026-03-03,2026-03-04 00:00:00+00:00,march_only,VCU,George Mason,-10.25,6.518,AWAY,0.64,0.116,0.505,0.454,-0.051,False,persistent,-3.549,3.549,5,27,False,raw edge only / filtered out,raw-edge watchlist only
2026,215185,2026-03-03,2026-03-04 01:00:00+00:00,march_only,Alabama State,Southern,-0.25,3.373,HOME,0.628,0.104,0.507,0.458,-0.049,False,persistent,-0.791,0.791,5,18,False,raw edge only / filtered out,raw-edge watchlist only
2026,359022,2026-03-03,2026-03-04 04:00:00+00:00,march_only,UNLV,Utah State,7.5,-3.936,HOME,0.626,0.102,0.504,0.221,-0.282,False,none,-16.387,16.387,0,0,False,raw edge only / filtered out,raw-edge watchlist only
```
