# Internal Bet Filter Report: 2026-03-06

## Summary

- Season: `2026`
- Training seasons used: `[2019, 2020, 2022, 2023, 2024, 2025]`
- Slate games scored: `36`
- Promoted disagreement-aware threshold: `disagreement_logit >= 0.58`
- Raw baseline watchlist threshold: `pick_prob_edge >= 0.08`
- Decision-useful non-NCAA candidates: `5`
- NCAA caution / diagnostic-only rows: `0`
- Raw-edge-only watchlist rows: `10`
- Filter-passing rows flagged mainly by disagreement features: `3`
- Slice mix: `{'conference_tournaments': 25, 'march_only': 11}`

## Guardrails

- Use the promoted shortlist as an internal late-season / March / conference-tournament aid, not as a public model output.
- NCAA tournament rows remain caution / diagnostic-only even if they clear the internal threshold.
- `flagged_mainly_by_disagreement = true` means the game would not have cleared the raw edge watchlist on its own.
- `raw edge only / filtered out` means the market-disagreement layer did not support the bet strongly enough.

## Ranked Internal Shortlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,372230,2026-03-06,2026-03-07 01:00:00+00:00,conference_tournaments,Tennessee State,UT Martin,-1.75,3.99,HOME,0.585,0.061,0.518,0.675,0.156,True,persistent,12.199,12.199,3,3,True,disagreement-led persistent,decision-useful internal candidate
2026,372227,2026-03-06,2026-03-06 23:00:00+00:00,conference_tournaments,Texas State,Southern Miss,-1.5,-0.228,AWAY,0.55,0.027,0.505,0.596,0.092,True,none,5.613,5.613,0,0,True,disagreement-led support,decision-useful internal candidate
2026,212080,2026-03-06,2026-03-07 00:00:00+00:00,march_only,Ohio,Miami (OH),4.0,-4.49,AWAY,0.514,-0.01,0.483,0.792,0.309,True,persistent,22.027,22.027,6,26,False,disagreement-led persistent,decision-useful internal candidate
2026,372180,2026-03-06,2026-03-07 02:30:00+00:00,conference_tournaments,North Dakota,Denver,3.25,0.982,HOME,0.634,0.11,0.525,0.628,0.103,False,none,7.313,7.313,0,0,True,raw edge led,decision-useful internal candidate
2026,372234,2026-03-06,2026-03-07 03:25:00+00:00,conference_tournaments,Illinois State,Northern Iowa,1.75,2.789,HOME,0.631,0.107,0.527,0.617,0.091,False,persistent,4.729,4.729,5,29,True,raw edge + persistent disagreement support,decision-useful internal candidate
```

## NCAA Caution / Diagnostic Rows

_empty_

## Raw-Edge-Only Watchlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,372217,2026-03-06,2026-03-06 17:00:00+00:00,conference_tournaments,High Point,Gardner-Webb,-27.5,14.608,AWAY,0.848,0.324,0.595,0.081,-0.514,False,persistent,-32.931,32.931,6,28,True,raw edge only / filtered out,raw-edge watchlist only
2026,372220,2026-03-06,2026-03-06 22:00:00+00:00,conference_tournaments,Austin Peay,Stetson,-11.75,5.288,AWAY,0.698,0.174,0.542,0.367,-0.175,False,persistent,-8.549,8.549,6,29,True,raw edge only / filtered out,raw-edge watchlist only
2026,372226,2026-03-06,2026-03-06 18:00:00+00:00,conference_tournaments,Belmont,Drake,-12.5,6.462,AWAY,0.692,0.168,0.539,0.419,-0.121,False,persistent,-6.505,6.505,5,26,True,raw edge only / filtered out,raw-edge watchlist only
2026,212079,2026-03-06,2026-03-07 00:00:00+00:00,march_only,Akron,Northern Illinois,-23.25,17.251,AWAY,0.673,0.149,0.521,0.056,-0.465,False,persistent,-35.545,35.545,6,26,False,raw edge only / filtered out,raw-edge watchlist only
2026,212081,2026-03-06,2026-03-07 00:00:00+00:00,march_only,Ball State,Central Michigan,-1.75,7.198,HOME,0.669,0.145,0.523,0.502,-0.021,False,persistent,0.514,0.514,6,26,False,raw edge only / filtered out,raw-edge watchlist only
2026,212083,2026-03-06,2026-03-06 23:30:00+00:00,march_only,Eastern Michigan,Bowling Green,5.5,-0.502,HOME,0.651,0.127,0.513,0.468,-0.045,False,none,-2.339,2.339,0,0,False,raw edge only / filtered out,raw-edge watchlist only
2026,372221,2026-03-06,2026-03-07 00:30:00+00:00,conference_tournaments,Queens University,West Georgia,-8.5,3.914,AWAY,0.639,0.115,0.528,0.375,-0.153,False,persistent,-7.86,7.86,6,28,True,raw edge only / filtered out,raw-edge watchlist only
2026,372178,2026-03-06,2026-03-07 00:30:00+00:00,conference_tournaments,UNC Greensboro,VMI,-6.75,3.214,AWAY,0.637,0.114,0.522,0.302,-0.22,False,persistent,-10.15,10.15,5,27,True,raw edge only / filtered out,raw-edge watchlist only
2026,212078,2026-03-06,2026-03-07 00:00:00+00:00,march_only,Kent State,Western Michigan,-12.5,9.12,AWAY,0.621,0.097,0.502,0.421,-0.082,False,persistent,-4.606,4.606,6,24,False,raw edge only / filtered out,raw-edge watchlist only
2026,372192,2026-03-06,2026-03-06 20:30:00+00:00,conference_tournaments,Murray State,UIC,1.25,2.848,HOME,0.611,0.087,0.522,0.514,-0.008,False,persistent,-1.384,1.384,4,28,True,raw edge only / filtered out,raw-edge watchlist only
```
