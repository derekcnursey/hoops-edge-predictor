# Internal Bet Filter Report: 2026-03-02

## Summary

- Season: `2026`
- Training seasons used: `[2019, 2020, 2022, 2023, 2024, 2025]`
- Slate games scored: `18`
- Promoted disagreement-aware threshold: `disagreement_logit >= 0.58`
- Raw baseline watchlist threshold: `pick_prob_edge >= 0.08`
- Decision-useful non-NCAA candidates: `2`
- NCAA caution / diagnostic-only rows: `0`
- Raw-edge-only watchlist rows: `4`
- Filter-passing rows flagged mainly by disagreement features: `1`
- Slice mix: `{'march_only': 18}`

## Guardrails

- Use the promoted shortlist as an internal late-season / March / conference-tournament aid, not as a public model output.
- NCAA tournament rows remain caution / diagnostic-only even if they clear the internal threshold.
- `flagged_mainly_by_disagreement = true` means the game would not have cleared the raw edge watchlist on its own.
- `raw edge only / filtered out` means the market-disagreement layer did not support the bet strongly enough.

## Ranked Internal Shortlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,212036,2026-03-02,2026-03-03 01:00:00+00:00,march_only,Northern Arizona,Montana State,5.75,-6.533,AWAY,0.533,0.009,0.492,0.742,0.25,True,persistent,17.316,17.316,5,24,False,disagreement-led persistent,decision-useful internal candidate
2026,212040,2026-03-02,2026-03-03 03:00:00+00:00,march_only,Sacramento State,Idaho State,1.25,-5.354,AWAY,0.65,0.126,0.514,0.716,0.202,False,persistent,14.916,14.916,5,25,False,raw edge + persistent disagreement support,decision-useful internal candidate
```

## NCAA Caution / Diagnostic Rows

_empty_

## Raw-Edge-Only Watchlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,215160,2026-03-02,2026-03-03 00:00:00+00:00,march_only,NC State,Duke,9.25,-2.961,HOME,0.69,0.166,0.524,0.126,-0.397,False,persistent,-26.184,26.184,5,19,False,raw edge only / filtered out,raw-edge watchlist only
2026,359016,2026-03-02,2026-03-02 21:00:00+00:00,march_only,Delaware State,South Carolina State,-3.5,-0.674,AWAY,0.672,0.148,0.509,0.542,0.033,False,persistent,4.065,4.065,6,26,False,raw edge only / filtered out,raw-edge watchlist only
2026,372150,2026-03-02,2026-03-03 00:00:00+00:00,march_only,Cleveland State,IU Indianapolis,1.75,2.956,HOME,0.644,0.121,0.511,0.511,-0.0,False,persistent,0.127,0.127,6,23,False,raw edge only / filtered out,raw-edge watchlist only
2026,215159,2026-03-02,2026-03-03 00:00:00+00:00,march_only,Howard,Coppin State,-19.25,15.097,AWAY,0.627,0.103,0.507,0.107,-0.4,False,persistent,-25.289,25.289,4,28,False,raw edge only / filtered out,raw-edge watchlist only
```
