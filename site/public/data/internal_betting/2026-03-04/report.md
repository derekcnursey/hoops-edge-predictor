# Internal Bet Filter Report: 2026-03-04

## Summary

- Season: `2026`
- Training seasons used: `[2019, 2020, 2022, 2023, 2024, 2025]`
- Slate games scored: `41`
- Promoted disagreement-aware threshold: `disagreement_logit >= 0.58`
- Raw baseline watchlist threshold: `pick_prob_edge >= 0.08`
- Decision-useful non-NCAA candidates: `6`
- NCAA caution / diagnostic-only rows: `0`
- Raw-edge-only watchlist rows: `14`
- Filter-passing rows flagged mainly by disagreement features: `5`
- Slice mix: `{'march_only': 23, 'conference_tournaments': 18}`

## Guardrails

- Use the promoted shortlist as an internal late-season / March / conference-tournament aid, not as a public model output.
- NCAA tournament rows remain caution / diagnostic-only even if they clear the internal threshold.
- `flagged_mainly_by_disagreement = true` means the game would not have cleared the raw edge watchlist on its own.
- `raw edge only / filtered out` means the market-disagreement layer did not support the bet strongly enough.

## Ranked Internal Shortlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,372155,2026-03-04,2026-03-04 17:00:00+00:00,conference_tournaments,Bellarmine,Jacksonville,-1.75,-1.529,AWAY,0.599,0.075,0.517,0.628,0.11,True,none,7.398,7.398,0,0,True,disagreement-led support,decision-useful internal candidate
2026,372164,2026-03-04,2026-03-05 00:00:00+00:00,conference_tournaments,Detroit Mercy,Milwaukee,-1.5,-0.485,AWAY,0.574,0.05,0.515,0.598,0.083,True,none,5.685,5.685,0,0,False,disagreement-led support,decision-useful internal candidate
2026,372165,2026-03-04,2026-03-05 00:00:00+00:00,conference_tournaments,Robert Morris,Youngstown State,-3.5,4.335,HOME,0.532,0.008,0.513,0.591,0.078,True,persistent,6.052,6.052,5,11,False,disagreement-led persistent,decision-useful internal candidate
2026,359024,2026-03-04,2026-03-05 00:00:00+00:00,march_only,George Washington,St. Bonaventure,-8.0,8.756,HOME,0.528,0.004,0.49,0.628,0.138,True,persistent,8.733,8.733,4,26,False,disagreement-led persistent,decision-useful internal candidate
2026,212064,2026-03-04,2026-03-05 00:00:00+00:00,march_only,Arkansas,Texas,-6.5,7.665,HOME,0.547,0.023,0.494,0.59,0.096,True,persistent,5.876,5.876,5,28,False,disagreement-led persistent,decision-useful internal candidate
2026,215193,2026-03-04,2026-03-05 00:00:00+00:00,march_only,Providence,Marquette,-4.0,8.292,HOME,0.634,0.11,0.514,0.585,0.07,False,persistent,6.53,6.53,2,2,False,raw edge + persistent disagreement support,decision-useful internal candidate
```

## NCAA Caution / Diagnostic Rows

_empty_

## Raw-Edge-Only Watchlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,215191,2026-03-04,2026-03-05 01:30:00+00:00,march_only,Northwestern,Purdue,11.5,-3.54,HOME,0.786,0.262,0.537,0.321,-0.216,False,persistent,-11.211,11.211,5,27,False,raw edge only / filtered out,raw-edge watchlist only
2026,212058,2026-03-04,2026-03-05 00:00:00+00:00,march_only,Ole Miss,Vanderbilt,7.25,-0.736,HOME,0.728,0.204,0.526,0.334,-0.192,False,persistent,-10.433,10.433,6,28,False,raw edge only / filtered out,raw-edge watchlist only
2026,372200,2026-03-04,2026-03-05 00:00:00+00:00,conference_tournaments,Wright State,Cleveland State,-14.5,8.761,AWAY,0.723,0.199,0.542,0.273,-0.269,False,persistent,-13.387,13.387,7,24,False,raw edge only / filtered out,raw-edge watchlist only
2026,215190,2026-03-04,2026-03-05 00:30:00+00:00,march_only,Penn State,Ohio State,6.75,-1.54,HOME,0.679,0.155,0.516,0.269,-0.247,False,none,-13.299,13.299,0,0,False,raw edge only / filtered out,raw-edge watchlist only
2026,215192,2026-03-04,2026-03-04 23:30:00+00:00,march_only,Indiana,Minnesota,-6.75,3.362,AWAY,0.678,0.154,0.503,0.41,-0.093,False,persistent,-5.045,5.045,5,18,False,raw edge only / filtered out,raw-edge watchlist only
2026,359023,2026-03-04,2026-03-04 23:30:00+00:00,march_only,La Salle,Fordham,3.5,0.927,HOME,0.667,0.143,0.51,0.517,0.006,False,persistent,1.163,1.163,4,2,False,raw edge only / filtered out,raw-edge watchlist only
2026,359028,2026-03-04,2026-03-05 01:00:00+00:00,march_only,Saint Louis,Loyola Chicago,-21.5,16.614,AWAY,0.642,0.118,0.512,0.074,-0.438,False,persistent,-32.06,32.06,4,27,False,raw edge only / filtered out,raw-edge watchlist only
2026,215179,2026-03-04,2026-03-05 02:00:00+00:00,march_only,Pittsburgh,Florida State,1.5,1.773,HOME,0.635,0.111,0.503,0.512,0.009,False,none,0.423,0.423,0,0,False,raw edge only / filtered out,raw-edge watchlist only
2026,215188,2026-03-04,2026-03-05 01:00:00+00:00,march_only,Wisconsin,Maryland,-14.5,11.281,AWAY,0.635,0.111,0.502,0.273,-0.23,False,none,-12.676,12.676,0,0,False,raw edge only / filtered out,raw-edge watchlist only
2026,215187,2026-03-04,2026-03-05 02:00:00+00:00,march_only,Houston,Baylor,-13.25,9.889,AWAY,0.633,0.109,0.503,0.243,-0.26,False,persistent,-15.599,15.599,5,27,False,raw edge only / filtered out,raw-edge watchlist only
2026,372156,2026-03-04,2026-03-04 19:30:00+00:00,conference_tournaments,Florida Gulf Coast,North Alabama,-7.5,3.435,AWAY,0.631,0.107,0.524,0.407,-0.117,False,persistent,-5.461,5.461,6,18,True,raw edge only / filtered out,raw-edge watchlist only
2026,372168,2026-03-04,2026-03-05 01:00:00+00:00,conference_tournaments,Oral Roberts,Kansas City,-8.5,4.812,AWAY,0.626,0.103,0.522,0.392,-0.13,False,persistent,-5.541,5.541,6,27,True,raw edge only / filtered out,raw-edge watchlist only
2026,372166,2026-03-04,2026-03-05 00:30:00+00:00,conference_tournaments,South Carolina Upstate,Gardner-Webb,-10.75,7.198,AWAY,0.613,0.09,0.52,0.264,-0.256,False,persistent,-12.571,12.571,6,27,True,raw edge only / filtered out,raw-edge watchlist only
2026,215194,2026-03-04,2026-03-05 01:00:00+00:00,march_only,DePaul,Villanova,1.75,1.217,HOME,0.612,0.088,0.5,0.414,-0.086,False,persistent,-4.989,4.989,5,10,False,raw edge only / filtered out,raw-edge watchlist only
```
