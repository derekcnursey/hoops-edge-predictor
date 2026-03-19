# Internal Bet Filter Report: 2026-03-05

## Summary

- Season: `2026`
- Training seasons used: `[2019, 2020, 2022, 2023, 2024, 2025]`
- Slate games scored: `46`
- Promoted disagreement-aware threshold: `disagreement_logit >= 0.58`
- Raw baseline watchlist threshold: `pick_prob_edge >= 0.08`
- Decision-useful non-NCAA candidates: `9`
- NCAA caution / diagnostic-only rows: `0`
- Raw-edge-only watchlist rows: `18`
- Filter-passing rows flagged mainly by disagreement features: `9`
- Slice mix: `{'march_only': 29, 'conference_tournaments': 17}`

## Guardrails

- Use the promoted shortlist as an internal late-season / March / conference-tournament aid, not as a public model output.
- NCAA tournament rows remain caution / diagnostic-only even if they clear the internal threshold.
- `flagged_mainly_by_disagreement = true` means the game would not have cleared the raw edge watchlist on its own.
- `raw edge only / filtered out` means the market-disagreement layer did not support the bet strongly enough.

## Ranked Internal Shortlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,372171,2026-03-05,2026-03-06 00:00:00+00:00,conference_tournaments,Colgate,Loyola Maryland,-6.0,8.304,HOME,0.585,0.061,0.523,0.656,0.133,True,none,11.447,11.447,1,1,False,disagreement-led support,decision-useful internal candidate
2026,372184,2026-03-05,2026-03-06 01:30:00+00:00,conference_tournaments,Fairfield,Manhattan,-4.5,4.953,HOME,0.515,-0.009,0.503,0.636,0.134,True,persistent,9.518,9.518,5,23,True,disagreement-led persistent,decision-useful internal candidate
2026,359034,2026-03-05,2026-03-06 00:00:00+00:00,march_only,East Carolina,Tulsa,8.5,-8.76,AWAY,0.508,-0.015,0.484,0.79,0.306,True,persistent,22.055,22.055,4,23,False,disagreement-led persistent,decision-useful internal candidate
2026,212077,2026-03-05,2026-03-06 00:00:00+00:00,march_only,Delaware,Sam Houston,7.25,-7.692,AWAY,0.516,-0.008,0.487,0.786,0.299,True,persistent,21.315,21.315,5,25,False,disagreement-led persistent,decision-useful internal candidate
2026,212076,2026-03-05,2026-03-06 00:00:00+00:00,march_only,Liberty,Louisiana Tech,-8.75,11.522,HOME,0.583,0.059,0.502,0.685,0.183,True,persistent,12.553,12.553,5,26,False,disagreement-led persistent,decision-useful internal candidate
2026,212067,2026-03-05,2026-03-06 03:00:00+00:00,march_only,UC Irvine,Cal Poly,-9.5,10.098,HOME,0.523,-0.001,0.489,0.663,0.174,True,persistent,11.019,11.019,6,27,False,disagreement-led persistent,decision-useful internal candidate
2026,212070,2026-03-05,2026-03-06 01:00:00+00:00,march_only,Tarleton State,UT Arlington,-1.5,-0.636,AWAY,0.571,0.048,0.492,0.62,0.128,True,persistent,6.868,6.868,5,18,False,disagreement-led persistent,decision-useful internal candidate
2026,215203,2026-03-05,2026-03-06 00:00:00+00:00,march_only,Maryland Eastern Shore,Delaware State,-5.5,7.574,HOME,0.568,0.044,0.498,0.616,0.118,True,persistent,9.722,9.722,5,27,False,disagreement-led persistent,decision-useful internal candidate
2026,212074,2026-03-05,2026-03-06 02:00:00+00:00,march_only,UTEP,Kennesaw State,3.75,-4.699,AWAY,0.537,0.013,0.492,0.613,0.121,True,none,8.678,8.678,0,0,False,disagreement-led support,decision-useful internal candidate
```

## NCAA Caution / Diagnostic Rows

_empty_

## Raw-Edge-Only Watchlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,215200,2026-03-05,2026-03-06 01:00:00+00:00,march_only,Iowa,Michigan,8.75,1.162,HOME,0.835,0.311,0.552,0.383,-0.169,False,persistent,-8.291,8.291,5,28,False,raw edge only / filtered out,raw-edge watchlist only
2026,215201,2026-03-05,2026-03-06 02:30:00+00:00,march_only,Cal State Bakersfield,Cal State Northridge,8.5,-1.325,HOME,0.761,0.237,0.531,0.29,-0.24,False,persistent,-11.47,11.47,5,27,False,raw edge only / filtered out,raw-edge watchlist only
2026,359031,2026-03-05,2026-03-05 05:00:00+00:00,march_only,Norfolk State,Howard,6.0,1.143,HOME,0.757,0.233,0.53,0.364,-0.166,False,persistent,-7.273,7.273,5,14,False,raw edge only / filtered out,raw-edge watchlist only
2026,212073,2026-03-05,2026-03-06 01:00:00+00:00,march_only,Missouri State,Western Kentucky,1.5,2.978,HOME,0.691,0.167,0.511,0.478,-0.033,False,none,-1.216,1.216,0,0,False,raw edge only / filtered out,raw-edge watchlist only
2026,372203,2026-03-05,2026-03-06 00:00:00+00:00,conference_tournaments,Navy,Bucknell,-15.5,10.833,AWAY,0.684,0.16,0.534,0.2,-0.335,False,persistent,-17.823,17.823,6,27,False,raw edge only / filtered out,raw-edge watchlist only
2026,215205,2026-03-05,2026-03-06 01:30:00+00:00,march_only,Alabama A&M,Southern,1.25,2.875,HOME,0.673,0.15,0.508,0.44,-0.069,False,persistent,-3.097,3.097,6,26,False,raw edge only / filtered out,raw-edge watchlist only
2026,359036,2026-03-05,2026-03-06 02:00:00+00:00,march_only,Memphis,South Florida,7.5,-2.044,HOME,0.673,0.149,0.517,0.25,-0.267,False,none,-14.372,14.372,1,1,False,raw edge only / filtered out,raw-edge watchlist only
2026,215199,2026-03-05,2026-03-06 01:00:00+00:00,march_only,Michigan State,Rutgers,-20.0,14.262,AWAY,0.672,0.148,0.519,0.107,-0.412,False,persistent,-27.975,27.975,5,24,False,raw edge only / filtered out,raw-edge watchlist only
2026,372214,2026-03-05,2026-03-06 00:00:00+00:00,conference_tournaments,UT Martin,Eastern Illinois,-8.25,3.685,AWAY,0.663,0.139,0.529,0.434,-0.094,False,persistent,-4.456,4.456,6,28,True,raw edge only / filtered out,raw-edge watchlist only
2026,359035,2026-03-05,2026-03-06 00:30:00+00:00,march_only,North Carolina Central,South Carolina State,-7.5,2.889,AWAY,0.646,0.123,0.511,0.523,0.012,False,persistent,1.905,1.905,7,27,False,raw edge only / filtered out,raw-edge watchlist only
2026,215197,2026-03-05,2026-03-05 05:00:00+00:00,march_only,Florida A&M,Bethune-Cookman,2.5,1.076,HOME,0.638,0.114,0.504,0.479,-0.025,False,persistent,-1.374,1.374,6,25,False,raw edge only / filtered out,raw-edge watchlist only
2026,372191,2026-03-05,2026-03-06 02:30:00+00:00,conference_tournaments,Northern Iowa,Evansville,-14.5,9.999,AWAY,0.627,0.104,0.526,0.15,-0.377,False,persistent,-22.921,22.921,6,30,True,raw edge only / filtered out,raw-edge watchlist only
2026,372215,2026-03-05,2026-03-06 01:30:00+00:00,conference_tournaments,Arkansas State,Georgia Southern,-7.5,3.755,AWAY,0.622,0.098,0.522,0.365,-0.156,False,persistent,-8.7,8.7,6,30,True,raw edge only / filtered out,raw-edge watchlist only
2026,212066,2026-03-05,2026-03-06 03:00:00+00:00,march_only,Long Beach State,UC Davis,2.0,1.136,HOME,0.608,0.084,0.5,0.378,-0.122,False,persistent,-6.72,6.72,6,27,False,raw edge only / filtered out,raw-edge watchlist only
2026,372213,2026-03-05,2026-03-06 00:00:00+00:00,conference_tournaments,North Dakota State,Oral Roberts,-12.5,9.541,AWAY,0.607,0.084,0.517,0.169,-0.348,False,persistent,-21.078,21.078,7,28,True,raw edge only / filtered out,raw-edge watchlist only
```
