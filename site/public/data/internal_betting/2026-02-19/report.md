# Internal Bet Filter Report: 2026-02-19

## Summary

- Season: `2026`
- Training seasons used: `[2019, 2020, 2022, 2023, 2024, 2025]`
- Slate games scored: `53`
- Promoted disagreement-aware threshold: `disagreement_logit >= 0.58`
- Raw baseline watchlist threshold: `pick_prob_edge >= 0.08`
- Decision-useful non-NCAA candidates: `10`
- NCAA caution / diagnostic-only rows: `0`
- Raw-edge-only watchlist rows: `14`
- Filter-passing rows flagged mainly by disagreement features: `7`
- Slice mix: `{'feb15_plus': 53}`

## Guardrails

- Use the promoted shortlist as an internal late-season / March / conference-tournament aid, not as a public model output.
- NCAA tournament rows remain caution / diagnostic-only even if they clear the internal threshold.
- `flagged_mainly_by_disagreement = true` means the game would not have cleared the raw edge watchlist on its own.
- `raw edge only / filtered out` means the market-disagreement layer did not support the bet strongly enough.

## Ranked Internal Shortlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,211725,2026-02-19,2026-02-20 01:00:00+00:00,feb15_plus,South Alabama,Texas State,-3.75,6.829,HOME,0.584,0.06,0.507,0.679,0.172,True,persistent,12.053,12.053,7,20,False,disagreement-led persistent,decision-useful internal candidate
2026,211710,2026-02-19,2026-02-20 00:00:00+00:00,feb15_plus,Chicago State,Fairleigh Dickinson,1.75,-4.524,AWAY,0.593,0.069,0.508,0.666,0.159,True,persistent,12.078,12.078,6,16,False,disagreement-led persistent,decision-useful internal candidate
2026,211722,2026-02-19,2026-02-20 04:59:00+00:00,feb15_plus,Hawai'i,Cal Poly,-11.0,12.009,HOME,0.544,0.02,0.498,0.657,0.159,True,persistent,11.234,11.234,5,11,False,disagreement-led persistent,decision-useful internal candidate
2026,211736,2026-02-19,2026-02-20 01:30:00+00:00,feb15_plus,Western Illinois,Southern Indiana,2.75,-5.139,AWAY,0.598,0.074,0.507,0.633,0.126,True,persistent,10.026,10.026,6,23,False,disagreement-led persistent,decision-useful internal candidate
2026,211715,2026-02-19,2026-02-20 00:00:00+00:00,feb15_plus,UNC Wilmington,Monmouth,-7.5,8.576,HOME,0.535,0.011,0.494,0.619,0.125,True,persistent,8.234,8.234,5,22,False,disagreement-led persistent,decision-useful internal candidate
2026,358926,2026-02-19,2026-02-20 00:00:00+00:00,feb15_plus,South Florida,Memphis,-9.5,9.595,HOME,0.503,-0.021,0.483,0.618,0.135,True,persistent,8.554,8.554,5,22,False,disagreement-led persistent,decision-useful internal candidate
2026,211728,2026-02-19,2026-02-19 23:30:00+00:00,feb15_plus,App State,Marshall,-3.5,2.066,AWAY,0.556,0.032,0.494,0.595,0.102,True,persistent,4.769,4.769,6,25,False,disagreement-led persistent,decision-useful internal candidate
2026,211743,2026-02-19,2026-02-20 01:00:00+00:00,feb15_plus,Tarleton State,Southern Utah,-4.75,9.32,HOME,0.679,0.155,0.522,0.622,0.1,False,none,9.031,9.031,0,0,False,raw edge led,decision-useful internal candidate
2026,211714,2026-02-19,2026-02-20 00:00:00+00:00,feb15_plus,NJIT,UAlbany,-3.5,-1.148,AWAY,0.666,0.143,0.516,0.607,0.091,False,persistent,6.372,6.372,5,22,False,raw edge + persistent disagreement support,decision-useful internal candidate
2026,211727,2026-02-19,2026-02-20 01:00:00+00:00,feb15_plus,Georgia State,Georgia Southern,1.5,5.064,HOME,0.723,0.199,0.53,0.603,0.073,False,none,5.762,5.762,0,0,False,raw edge led,decision-useful internal candidate
```

## NCAA Caution / Diagnostic Rows

_empty_

## Raw-Edge-Only Watchlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,215003,2026-02-19,2026-02-20 00:00:00+00:00,feb15_plus,Radford,Gardner-Webb,-18.5,11.168,AWAY,0.762,0.238,0.536,0.248,-0.288,False,persistent,-13.133,13.133,6,24,False,raw edge only / filtered out,raw-edge watchlist only
2026,211709,2026-02-19,2026-02-20 00:00:00+00:00,feb15_plus,St. Francis (PA),Long Island University,9.25,-2.381,HOME,0.742,0.219,0.533,0.211,-0.322,False,persistent,-16.444,16.444,6,23,False,raw edge only / filtered out,raw-edge watchlist only
2026,211735,2026-02-19,2026-02-20 00:00:00+00:00,feb15_plus,North Florida,Austin Peay,11.25,-4.914,HOME,0.709,0.185,0.529,0.287,-0.242,False,persistent,-11.869,11.869,6,19,False,raw edge only / filtered out,raw-edge watchlist only
2026,211720,2026-02-19,2026-02-20 00:00:00+00:00,feb15_plus,The Citadel,Samford,10.5,-5.02,HOME,0.666,0.142,0.521,0.201,-0.32,False,persistent,-16.464,16.464,6,23,False,raw edge only / filtered out,raw-edge watchlist only
2026,211719,2026-02-19,2026-02-20 00:00:00+00:00,feb15_plus,North Carolina A&T,Charleston,4.5,1.103,HOME,0.66,0.136,0.522,0.475,-0.046,False,persistent,-2.273,2.273,5,21,False,raw edge only / filtered out,raw-edge watchlist only
2026,211730,2026-02-19,2026-02-20 03:00:00+00:00,feb15_plus,Sacramento State,Eastern Washington,3.25,2.112,HOME,0.651,0.127,0.52,0.402,-0.118,False,persistent,-5.388,5.388,6,21,False,raw edge only / filtered out,raw-edge watchlist only
2026,211723,2026-02-19,2026-02-20 03:00:00+00:00,feb15_plus,Cal State Fullerton,UC Davis,-1.5,5.344,HOME,0.637,0.113,0.516,0.467,-0.049,False,none,-0.753,0.753,0,0,False,raw edge only / filtered out,raw-edge watchlist only
2026,211717,2026-02-19,2026-02-20 00:00:00+00:00,feb15_plus,Campbell,William & Mary,2.25,1.049,HOME,0.632,0.108,0.507,0.348,-0.159,False,none,-8.079,8.079,0,0,False,raw edge only / filtered out,raw-edge watchlist only
2026,211745,2026-02-19,2026-02-20 02:00:00+00:00,feb15_plus,Idaho State,Montana,1.25,2.811,HOME,0.627,0.104,0.511,0.523,0.013,False,persistent,-0.143,0.143,6,22,False,raw edge only / filtered out,raw-edge watchlist only
2026,211712,2026-02-19,2026-02-19 23:00:00+00:00,feb15_plus,UMass Lowell,New Hampshire,-3.5,6.841,HOME,0.608,0.085,0.512,0.526,0.014,False,persistent,2.988,2.988,5,22,False,raw edge only / filtered out,raw-edge watchlist only
2026,211738,2026-02-19,2026-02-20 01:20:00+00:00,feb15_plus,Little Rock,UT Martin,2.5,0.102,HOME,0.607,0.083,0.502,0.366,-0.136,False,persistent,-6.937,6.937,7,24,False,raw edge only / filtered out,raw-edge watchlist only
2026,211711,2026-02-19,2026-02-19 23:00:00+00:00,feb15_plus,UMBC,Vermont,1.5,1.27,HOME,0.606,0.083,0.503,0.498,-0.005,False,persistent,-0.548,0.548,5,19,False,raw edge only / filtered out,raw-edge watchlist only
2026,214999,2026-02-19,2026-02-20 01:50:00+00:00,feb15_plus,Mississippi Valley State,Prairie View A&M,9.25,-6.18,HOME,0.605,0.081,0.504,0.141,-0.363,False,persistent,-20.3,20.3,5,24,False,raw edge only / filtered out,raw-edge watchlist only
2026,211706,2026-02-19,2026-02-20 00:00:00+00:00,feb15_plus,Wright State,IU Indianapolis,-11.5,8.897,AWAY,0.605,0.081,0.502,0.244,-0.258,False,persistent,-14.537,14.537,5,26,False,raw edge only / filtered out,raw-edge watchlist only
```
