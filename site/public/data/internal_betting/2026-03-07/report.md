# Internal Bet Filter Report: 2026-03-07

## Summary

- Season: `2026`
- Training seasons used: `[2019, 2020, 2022, 2023, 2024, 2025]`
- Slate games scored: `93`
- Promoted disagreement-aware threshold: `disagreement_logit >= 0.58`
- Raw baseline watchlist threshold: `pick_prob_edge >= 0.08`
- Decision-useful non-NCAA candidates: `18`
- NCAA caution / diagnostic-only rows: `0`
- Raw-edge-only watchlist rows: `24`
- Filter-passing rows flagged mainly by disagreement features: `17`
- Slice mix: `{'march_only': 62, 'conference_tournaments': 31}`

## Guardrails

- Use the promoted shortlist as an internal late-season / March / conference-tournament aid, not as a public model output.
- NCAA tournament rows remain caution / diagnostic-only even if they clear the internal threshold.
- `flagged_mainly_by_disagreement = true` means the game would not have cleared the raw edge watchlist on its own.
- `raw edge only / filtered out` means the market-disagreement layer did not support the bet strongly enough.

## Ranked Internal Shortlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,372193,2026-03-07,2026-03-08 00:30:00+00:00,conference_tournaments,Idaho State,Northern Arizona,-4.5,7.075,HOME,0.591,0.067,0.52,0.705,0.185,True,persistent,13.367,13.367,5,26,True,disagreement-led persistent,decision-useful internal candidate
2026,372250,2026-03-07,2026-03-08 02:00:00+00:00,conference_tournaments,San Francisco,Portland,-7.25,7.804,HOME,0.518,-0.006,0.503,0.694,0.191,True,persistent,12.721,12.721,4,30,True,disagreement-led persistent,decision-useful internal candidate
2026,372182,2026-03-07,2026-03-08 01:30:00+00:00,conference_tournaments,Mercer,Western Carolina,-2.25,2.984,HOME,0.525,0.001,0.506,0.643,0.138,True,persistent,9.115,9.115,5,24,True,disagreement-led persistent,decision-useful internal candidate
2026,372194,2026-03-07,2026-03-08 03:00:00+00:00,conference_tournaments,Idaho,Sacramento State,-6.25,6.756,HOME,0.516,-0.008,0.502,0.643,0.141,True,none,10.554,10.554,0,0,True,disagreement-led support,decision-useful internal candidate
2026,372188,2026-03-07,2026-03-08 01:48:00+00:00,conference_tournaments,Siena,Mount St. Mary's,-2.75,4.284,HOME,0.547,0.023,0.51,0.622,0.112,True,persistent,7.69,7.69,5,21,True,disagreement-led persistent,decision-useful internal candidate
2026,372187,2026-03-07,2026-03-07 23:00:00+00:00,conference_tournaments,Quinnipiac,Marist,-1.5,-0.939,AWAY,0.588,0.064,0.513,0.592,0.079,True,persistent,4.025,4.025,4,21,True,disagreement-led persistent,decision-useful internal candidate
2026,372205,2026-03-07,2026-03-07 17:00:00+00:00,conference_tournaments,Stony Brook,Campbell,2.5,-3.324,AWAY,0.524,0.0,0.503,0.588,0.084,True,none,6.753,6.753,0,0,True,disagreement-led support,decision-useful internal candidate
2026,212093,2026-03-07,2026-03-07 20:30:00+00:00,march_only,Mississippi State,Georgia,5.25,-5.555,AWAY,0.509,-0.015,0.483,0.796,0.313,True,persistent,22.267,22.267,6,29,False,disagreement-led persistent,decision-useful internal candidate
2026,359041,2026-03-07,2026-03-07 19:00:00+00:00,march_only,Loyola Chicago,George Washington,9.75,-11.001,AWAY,0.54,0.016,0.491,0.762,0.271,True,persistent,19.155,19.155,4,27,False,disagreement-led persistent,decision-useful internal candidate
2026,212099,2026-03-07,2026-03-08 00:00:00+00:00,march_only,Abilene Christian,UT Arlington,-1.25,-1.159,AWAY,0.594,0.07,0.496,0.712,0.216,True,persistent,13.572,13.572,5,19,False,disagreement-led persistent,decision-useful internal candidate
2026,215228,2026-03-07,2026-03-07 17:30:00+00:00,march_only,Virginia,Virginia Tech,-11.5,11.904,HOME,0.514,-0.01,0.486,0.666,0.18,True,persistent,11.317,11.317,6,26,False,disagreement-led persistent,decision-useful internal candidate
2026,212088,2026-03-07,2026-03-08 02:00:00+00:00,march_only,UC Santa Barbara,UC San Diego,-1.75,1.36,AWAY,0.51,-0.013,0.475,0.658,0.183,True,persistent,9.549,9.549,5,28,False,disagreement-led persistent,decision-useful internal candidate
2026,212092,2026-03-07,2026-03-07 17:00:00+00:00,march_only,Missouri,Arkansas,-2.5,1.688,AWAY,0.531,0.007,0.485,0.656,0.171,True,persistent,8.802,8.802,6,29,False,disagreement-led persistent,decision-useful internal candidate
2026,215224,2026-03-07,2026-03-07 17:00:00+00:00,march_only,DePaul,Butler,-3.5,3.189,AWAY,0.51,-0.014,0.477,0.625,0.148,True,persistent,7.058,7.058,5,29,False,disagreement-led persistent,decision-useful internal candidate
2026,212100,2026-03-07,2026-03-07 20:00:00+00:00,march_only,Missouri State,Middle Tennessee,-2.5,-0.029,AWAY,0.586,0.062,0.496,0.608,0.112,True,persistent,5.517,5.517,6,26,False,disagreement-led persistent,decision-useful internal candidate
```

## NCAA Caution / Diagnostic Rows

_empty_

## Raw-Edge-Only Watchlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,215236,2026-03-07,2026-03-08 03:00:00+00:00,march_only,Nevada,Air Force,-25.5,15.018,AWAY,0.851,0.327,0.557,0.092,-0.465,False,persistent,-28.926,28.926,6,29,False,raw edge only / filtered out,raw-edge watchlist only
2026,215234,2026-03-07,2026-03-07 23:30:00+00:00,march_only,Duke,North Carolina,-18.5,7.42,AWAY,0.819,0.295,0.562,0.197,-0.365,False,persistent,-20.934,20.934,6,29,False,raw edge only / filtered out,raw-edge watchlist only
2026,215215,2026-03-07,2026-03-08 02:00:00+00:00,march_only,USC,UCLA,5.75,1.176,HOME,0.743,0.219,0.529,0.303,-0.226,False,none,-11.216,11.216,0,0,False,raw edge only / filtered out,raw-edge watchlist only
2026,215225,2026-03-07,2026-03-07 17:00:00+00:00,march_only,Villanova,Xavier,-13.25,7.003,AWAY,0.735,0.211,0.524,0.369,-0.155,False,persistent,-7.823,7.823,6,14,False,raw edge only / filtered out,raw-edge watchlist only
2026,215209,2026-03-07,2026-03-07 17:00:00+00:00,march_only,Oklahoma State,Houston,13.5,-6.251,HOME,0.726,0.202,0.531,0.214,-0.317,False,none,-17.604,17.604,0,0,False,raw edge only / filtered out,raw-edge watchlist only
2026,212101,2026-03-07,2026-03-07 19:00:00+00:00,march_only,Florida International,Western Kentucky,1.25,2.951,HOME,0.712,0.188,0.509,0.544,0.035,False,none,2.961,2.961,0,0,False,raw edge only / filtered out,raw-edge watchlist only
2026,215212,2026-03-07,2026-03-08 04:00:00+00:00,march_only,Colorado,Arizona,13.25,-6.514,HOME,0.685,0.161,0.526,0.14,-0.387,False,new/transient,-21.799,21.799,0,0,False,raw edge only / filtered out,raw-edge watchlist only
2026,212096,2026-03-07,2026-03-07 21:00:00+00:00,march_only,Kentucky,Florida,5.5,0.648,HOME,0.683,0.16,0.522,0.344,-0.178,False,persistent,-10.148,10.148,6,29,False,raw edge only / filtered out,raw-edge watchlist only
2026,372253,2026-03-07,2026-03-08 03:30:00+00:00,conference_tournaments,St. Thomas-Minnesota,North Dakota,-11.5,6.165,AWAY,0.681,0.157,0.534,0.29,-0.245,False,none,-12.405,12.405,0,0,True,raw edge only / filtered out,raw-edge watchlist only
2026,359045,2026-03-07,2026-03-07 21:00:00+00:00,march_only,George Mason,Saint Louis,7.25,-1.626,HOME,0.674,0.151,0.518,0.427,-0.092,False,persistent,-5.393,5.393,4,28,False,raw edge only / filtered out,raw-edge watchlist only
2026,372240,2026-03-07,2026-03-07 17:00:00+00:00,conference_tournaments,High Point,UNC Asheville,-14.0,8.449,AWAY,0.671,0.147,0.535,0.232,-0.303,False,persistent,-16.243,16.243,3,3,True,raw edge only / filtered out,raw-edge watchlist only
2026,212095,2026-03-07,2026-03-07 23:00:00+00:00,march_only,LSU,Texas A&M,2.5,2.241,HOME,0.67,0.146,0.512,0.351,-0.161,False,none,-8.282,8.282,0,0,False,raw edge only / filtered out,raw-edge watchlist only
2026,212094,2026-03-07,2026-03-07 18:00:00+00:00,march_only,Ole Miss,South Carolina,-7.25,3.471,AWAY,0.661,0.137,0.506,0.563,0.057,False,persistent,2.618,2.618,6,28,False,raw edge only / filtered out,raw-edge watchlist only
2026,359040,2026-03-07,2026-03-07 19:00:00+00:00,march_only,Dartmouth,Cornell,4.5,-0.291,HOME,0.658,0.134,0.509,0.399,-0.109,False,none,-5.39,5.39,0,0,False,raw edge only / filtered out,raw-edge watchlist only
2026,212091,2026-03-07,2026-03-07 19:00:00+00:00,march_only,Tennessee,Vanderbilt,-2.5,6.925,HOME,0.638,0.114,0.515,0.541,0.025,False,persistent,1.964,1.964,6,29,False,raw edge only / filtered out,raw-edge watchlist only
```
