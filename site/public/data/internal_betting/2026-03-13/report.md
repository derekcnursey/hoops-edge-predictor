# Internal Bet Filter Report: 2026-03-13

## Summary

- Season: `2026`
- Training seasons used: `[2019, 2020, 2022, 2023, 2024, 2025]`
- Slate games scored: `34`
- Promoted disagreement-aware threshold: `disagreement_logit >= 0.58`
- Raw baseline watchlist threshold: `pick_prob_edge >= 0.08`
- Decision-useful non-NCAA candidates: `7`
- NCAA caution / diagnostic-only rows: `0`
- Raw-edge-only watchlist rows: `6`
- Filter-passing rows flagged mainly by disagreement features: `7`
- Slice mix: `{'conference_tournaments': 34}`

## Guardrails

- Use the promoted shortlist as an internal late-season / March / conference-tournament aid, not as a public model output.
- NCAA tournament rows remain caution / diagnostic-only even if they clear the internal threshold.
- `flagged_mainly_by_disagreement = true` means the game would not have cleared the raw edge watchlist on its own.
- `raw edge only / filtered out` means the market-disagreement layer did not support the bet strongly enough.

## Ranked Internal Shortlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,372406,2026-03-13,2026-03-13 21:30:00+00:00,conference_tournaments,St. John's,Seton Hall,-8.0,10.108,HOME,0.571,0.048,0.521,0.666,0.145,True,persistent,10.225,10.225,5,31,False,disagreement-led persistent,decision-useful internal candidate
2026,372410,2026-03-13,2026-03-13 23:00:00+00:00,conference_tournaments,Virginia,Miami,-3.0,4.346,HOME,0.535,0.011,0.505,0.665,0.16,True,persistent,10.527,10.527,5,27,True,disagreement-led persistent,decision-useful internal candidate
2026,372417,2026-03-13,2026-03-14 01:00:00+00:00,conference_tournaments,Michigan State,UCLA,-5.0,6.232,HOME,0.539,0.015,0.508,0.662,0.154,True,persistent,9.977,9.977,5,30,True,disagreement-led persistent,decision-useful internal candidate
2026,372426,2026-03-13,2026-03-14 03:59:00+00:00,conference_tournaments,San Diego State,New Mexico,-2.0,0.026,AWAY,0.553,0.029,0.505,0.623,0.118,True,persistent,5.772,5.772,5,29,True,disagreement-led persistent,decision-useful internal candidate
2026,372419,2026-03-13,2026-03-14 01:00:00+00:00,conference_tournaments,UC Irvine,Cal State Northridge,-3.5,4.804,HOME,0.537,0.013,0.507,0.618,0.112,True,persistent,7.168,7.168,5,29,True,disagreement-led persistent,decision-useful internal candidate
2026,372401,2026-03-13,2026-03-13 19:15:00+00:00,conference_tournaments,Sam Houston,Kennesaw State,-2.0,2.647,HOME,0.522,-0.002,0.505,0.609,0.104,True,persistent,6.419,6.419,6,28,True,disagreement-led persistent,decision-useful internal candidate
2026,372398,2026-03-13,2026-03-13 18:00:00+00:00,conference_tournaments,Alabama A&M,Prairie View A&M,-1.5,1.321,AWAY,0.506,-0.018,0.495,0.593,0.098,True,persistent,4.803,4.803,6,28,True,disagreement-led persistent,decision-useful internal candidate
```

## NCAA Caution / Diagnostic Rows

_empty_

## Raw-Edge-Only Watchlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,372407,2026-03-13,2026-03-13 22:00:00+00:00,conference_tournaments,Howard,South Carolina State,-15.0,8.612,AWAY,0.727,0.203,0.542,0.23,-0.312,False,persistent,-15.815,15.815,6,29,True,raw edge only / filtered out,raw-edge watchlist only
2026,372394,2026-03-13,2026-03-13 16:00:00+00:00,conference_tournaments,Michigan,Ohio State,-12.0,5.479,AWAY,0.681,0.157,0.542,0.4,-0.142,False,persistent,-7.992,7.992,5,31,True,raw edge only / filtered out,raw-edge watchlist only
2026,372411,2026-03-13,2026-03-13 23:00:00+00:00,conference_tournaments,Alabama,Ole Miss,-12.0,5.461,AWAY,0.655,0.131,0.54,0.336,-0.205,False,persistent,-11.639,11.639,6,31,True,raw edge only / filtered out,raw-edge watchlist only
2026,372408,2026-03-13,2026-03-13 22:30:00+00:00,conference_tournaments,Nebraska,Purdue,4.0,0.787,HOME,0.655,0.131,0.53,0.529,-0.001,False,persistent,-0.848,0.848,5,30,True,raw edge only / filtered out,raw-edge watchlist only
2026,372396,2026-03-13,2026-03-13 17:00:00+00:00,conference_tournaments,Florida,Kentucky,-10.0,4.312,AWAY,0.652,0.128,0.535,0.367,-0.168,False,persistent,-9.673,9.673,6,31,True,raw edge only / filtered out,raw-edge watchlist only
2026,372414,2026-03-13,2026-03-14 00:00:00+00:00,conference_tournaments,UConn,Georgetown,-14.0,10.177,AWAY,0.61,0.086,0.521,0.273,-0.248,False,none,-13.272,13.272,0,0,True,raw edge only / filtered out,raw-edge watchlist only
```
