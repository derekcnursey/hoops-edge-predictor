# Internal Bet Filter Report: 2026-02-16

## Summary

- Season: `2026`
- Training seasons used: `[2019, 2020, 2022, 2023, 2024, 2025]`
- Slate games scored: `24`
- Promoted disagreement-aware threshold: `disagreement_logit >= 0.58`
- Raw baseline watchlist threshold: `pick_prob_edge >= 0.08`
- Decision-useful non-NCAA candidates: `5`
- NCAA caution / diagnostic-only rows: `0`
- Raw-edge-only watchlist rows: `7`
- Filter-passing rows flagged mainly by disagreement features: `4`
- Slice mix: `{'feb15_plus': 24}`

## Guardrails

- Use the promoted shortlist as an internal late-season / March / conference-tournament aid, not as a public model output.
- NCAA tournament rows remain caution / diagnostic-only even if they clear the internal threshold.
- `flagged_mainly_by_disagreement = true` means the game would not have cleared the raw edge watchlist on its own.
- `raw edge only / filtered out` means the market-disagreement layer did not support the bet strongly enough.

## Ranked Internal Shortlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,211096,2026-02-16,2026-02-16 22:00:00+00:00,feb15_plus,Old Dominion,Louisiana,-6.5,7.534,HOME,0.534,0.01,0.494,0.633,0.139,True,persistent,9.799,9.799,6,25,False,disagreement-led persistent,decision-useful internal candidate
2026,211669,2026-02-16,2026-02-17 01:00:00+00:00,feb15_plus,Houston Christian,Nicholls,-1.75,-0.123,AWAY,0.582,0.058,0.498,0.601,0.104,True,persistent,6.577,6.577,6,7,False,disagreement-led persistent,decision-useful internal candidate
2026,214962,2026-02-16,2026-02-16 21:45:00+00:00,feb15_plus,South Carolina State,Coppin State,-4.5,7.196,HOME,0.59,0.066,0.507,0.601,0.094,True,persistent,9.708,9.708,4,21,False,disagreement-led persistent,decision-useful internal candidate
2026,211667,2026-02-16,2026-02-17 01:00:00+00:00,feb15_plus,Stony Brook,Drexel,-3.25,2.718,AWAY,0.517,-0.007,0.484,0.599,0.115,True,none,6.327,6.327,0,0,False,disagreement-led support,decision-useful internal candidate
2026,359075,2026-02-16,2026-02-17 01:00:00+00:00,feb15_plus,Tarleton State,Abilene Christian,-0.25,4.927,HOME,0.68,0.157,0.523,0.583,0.06,False,none,6.329,6.329,0,0,False,raw edge led,decision-useful internal candidate
```

## NCAA Caution / Diagnostic Rows

_empty_

## Raw-Edge-Only Watchlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,214956,2026-02-16,2026-02-17 01:00:00+00:00,feb15_plus,Texas Southern,Southern,4.25,1.214,HOME,0.745,0.222,0.522,0.381,-0.141,False,persistent,-5.476,5.476,5,20,False,raw edge only / filtered out,raw-edge watchlist only
2026,358899,2026-02-16,2026-02-17 00:00:00+00:00,feb15_plus,Delaware State,Howard,10.75,-6.067,HOME,0.697,0.173,0.517,0.238,-0.279,False,persistent,-13.334,13.334,5,19,False,raw edge only / filtered out,raw-edge watchlist only
2026,359074,2026-02-16,2026-02-17 00:00:00+00:00,feb15_plus,Long Island University,Wagner,-10.5,5.721,AWAY,0.687,0.163,0.517,0.355,-0.163,False,persistent,-7.893,7.893,7,21,False,raw edge only / filtered out,raw-edge watchlist only
2026,211671,2026-02-16,2026-02-17 00:40:00+00:00,feb15_plus,Northwestern State,McNeese,14.75,-9.446,HOME,0.683,0.16,0.521,0.16,-0.361,False,persistent,-21.772,21.772,6,21,False,raw edge only / filtered out,raw-edge watchlist only
2026,214958,2026-02-16,2026-02-17 01:00:00+00:00,feb15_plus,Alabama A&M,Arkansas-Pine Bluff,-3.25,7.209,HOME,0.67,0.146,0.518,0.437,-0.081,False,persistent,-1.636,1.636,6,21,False,raw edge only / filtered out,raw-edge watchlist only
2026,211668,2026-02-16,2026-02-17 01:00:00+00:00,feb15_plus,Texas A&M-Corpus Christi,Stephen F. Austin,5.0,-0.246,HOME,0.652,0.128,0.516,0.41,-0.106,False,persistent,-5.728,5.728,6,13,False,raw edge only / filtered out,raw-edge watchlist only
2026,214961,2026-02-16,2026-02-17 00:00:00+00:00,feb15_plus,Maryland Eastern Shore,Norfolk State,2.5,0.188,HOME,0.607,0.084,0.503,0.428,-0.074,False,persistent,-3.284,3.284,5,23,False,raw edge only / filtered out,raw-edge watchlist only
```
