# Internal Bet Filter Report: 2026-03-11

## Summary

- Season: `2026`
- Training seasons used: `[2019, 2020, 2022, 2023, 2024, 2025]`
- Slate games scored: `38`
- Promoted disagreement-aware threshold: `disagreement_logit >= 0.58`
- Raw baseline watchlist threshold: `pick_prob_edge >= 0.08`
- Decision-useful non-NCAA candidates: `9`
- NCAA caution / diagnostic-only rows: `0`
- Raw-edge-only watchlist rows: `9`
- Filter-passing rows flagged mainly by disagreement features: `6`
- Slice mix: `{'conference_tournaments': 38}`

## Guardrails

- Use the promoted shortlist as an internal late-season / March / conference-tournament aid, not as a public model output.
- NCAA tournament rows remain caution / diagnostic-only even if they clear the internal threshold.
- `flagged_mainly_by_disagreement = true` means the game would not have cleared the raw edge watchlist on its own.
- `raw edge only / filtered out` means the market-disagreement layer did not support the bet strongly enough.

## Ranked Internal Shortlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,372357,2026-03-11,2026-03-12 00:30:00+00:00,conference_tournaments,Florida A&M,Jackson State,-5.75,5.823,HOME,0.503,-0.021,0.502,0.663,0.162,True,persistent,11.424,11.424,5,26,True,disagreement-led persistent,decision-useful internal candidate
2026,372353,2026-03-11,2026-03-11 21:00:00+00:00,conference_tournaments,Stephen F. Austin,McNeese,3.25,-3.374,AWAY,0.504,-0.019,0.502,0.636,0.133,True,persistent,8.214,8.214,5,29,True,disagreement-led persistent,decision-useful internal candidate
2026,372295,2026-03-11,2026-03-11 18:00:00+00:00,conference_tournaments,Richmond,Loyola Chicago,-4.25,4.322,HOME,0.502,-0.022,0.495,0.63,0.135,True,none,10.009,10.009,0,0,True,disagreement-led support,decision-useful internal candidate
2026,372337,2026-03-11,2026-03-12 01:40:00+00:00,conference_tournaments,Florida Atlantic,Temple,-1.5,2.487,HOME,0.527,0.003,0.504,0.597,0.094,True,persistent,5.93,5.93,5,27,True,disagreement-led persistent,decision-useful internal candidate
2026,372352,2026-03-11,2026-03-11 19:15:00+00:00,conference_tournaments,UCF,Cincinnati,3.5,-1.076,HOME,0.571,0.047,0.51,0.595,0.085,True,persistent,3.804,3.804,5,28,True,disagreement-led persistent,decision-useful internal candidate
2026,372360,2026-03-11,2026-03-12 01:38:00+00:00,conference_tournaments,Clemson,Wake Forest,-5.5,5.517,HOME,0.501,-0.023,0.498,0.585,0.087,True,persistent,5.052,5.052,5,30,True,disagreement-led persistent,decision-useful internal candidate
2026,372305,2026-03-11,2026-03-12 01:00:00+00:00,conference_tournaments,UC San Diego,Cal Poly,-4.5,9.414,HOME,0.655,0.131,0.537,0.748,0.212,False,persistent,16.167,16.167,5,29,True,raw edge + persistent disagreement support,decision-useful internal candidate
2026,372298,2026-03-11,2026-03-11 20:00:00+00:00,conference_tournaments,Butler,Providence,1.5,2.548,HOME,0.61,0.086,0.522,0.64,0.118,False,persistent,6.385,6.385,5,30,True,raw edge + persistent disagreement support,decision-useful internal candidate
2026,372300,2026-03-11,2026-03-11 22:30:00+00:00,conference_tournaments,Marquette,Xavier,-4.5,-2.707,AWAY,0.7,0.176,0.548,0.606,0.058,False,persistent,3.93,3.93,5,15,True,raw edge + persistent disagreement support,decision-useful internal candidate
```

## NCAA Caution / Diagnostic Rows

_empty_

## Raw-Edge-Only Watchlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,372307,2026-03-11,2026-03-12 01:40:00+00:00,conference_tournaments,Oklahoma,South Carolina,-8.5,-0.607,AWAY,0.749,0.225,0.563,0.56,-0.003,False,persistent,-0.194,0.194,5,29,True,raw edge only / filtered out,raw-edge watchlist only
2026,372299,2026-03-11,2026-03-11 21:30:00+00:00,conference_tournaments,Nevada,Air Force,-19.25,11.2,AWAY,0.733,0.209,0.555,0.083,-0.472,False,persistent,-30.899,30.899,5,30,True,raw edge only / filtered out,raw-edge watchlist only
2026,372308,2026-03-11,2026-03-12 03:30:00+00:00,conference_tournaments,Boise State,San José State,-14.5,10.456,AWAY,0.637,0.113,0.524,0.253,-0.271,False,persistent,-14.932,14.932,5,29,True,raw edge only / filtered out,raw-edge watchlist only
2026,372355,2026-03-11,2026-03-11 22:30:00+00:00,conference_tournaments,Indiana,Northwestern,-6.5,2.165,AWAY,0.627,0.103,0.525,0.571,0.046,False,persistent,1.594,1.594,6,30,True,raw edge only / filtered out,raw-edge watchlist only
2026,372301,2026-03-11,2026-03-11 23:00:00+00:00,conference_tournaments,Texas,Ole Miss,-6.25,2.019,AWAY,0.625,0.102,0.525,0.502,-0.023,False,persistent,-2.179,2.179,6,30,True,raw edge only / filtered out,raw-edge watchlist only
2026,372348,2026-03-11,2026-03-11 16:00:00+00:00,conference_tournaments,NC State,Pittsburgh,-8.5,4.492,AWAY,0.624,0.1,0.523,0.434,-0.089,False,none,-4.263,4.263,0,0,True,raw edge only / filtered out,raw-edge watchlist only
2026,372297,2026-03-11,2026-03-11 19:00:00+00:00,conference_tournaments,UNLV,Wyoming,-2.75,-1.423,AWAY,0.614,0.09,0.523,0.557,0.034,False,persistent,0.954,0.954,5,27,True,raw edge only / filtered out,raw-edge watchlist only
2026,372302,2026-03-11,2026-03-11 23:00:00+00:00,conference_tournaments,Florida State,California,-3.0,-1.085,AWAY,0.614,0.09,0.523,0.565,0.043,False,persistent,1.512,1.512,5,27,True,raw edge only / filtered out,raw-edge watchlist only
2026,372309,2026-03-11,2026-03-12 03:30:00+00:00,conference_tournaments,UC Davis,UC Santa Barbara,5.5,-1.921,HOME,0.607,0.084,0.52,0.461,-0.059,False,none,-2.831,2.831,0,0,True,raw edge only / filtered out,raw-edge watchlist only
```
