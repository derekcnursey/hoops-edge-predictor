# Internal Bet Filter Report: 2026-02-28

## Summary

- Season: `2026`
- Training seasons used: `[2019, 2020, 2022, 2023, 2024, 2025]`
- Slate games scored: `143`
- Promoted disagreement-aware threshold: `disagreement_logit >= 0.58`
- Raw baseline watchlist threshold: `pick_prob_edge >= 0.08`
- Decision-useful non-NCAA candidates: `27`
- NCAA caution / diagnostic-only rows: `0`
- Raw-edge-only watchlist rows: `33`
- Filter-passing rows flagged mainly by disagreement features: `17`
- Slice mix: `{'feb15_plus': 143}`

## Guardrails

- Use the promoted shortlist as an internal late-season / March / conference-tournament aid, not as a public model output.
- NCAA tournament rows remain caution / diagnostic-only even if they clear the internal threshold.
- `flagged_mainly_by_disagreement = true` means the game would not have cleared the raw edge watchlist on its own.
- `raw edge only / filtered out` means the market-disagreement layer did not support the bet strongly enough.

## Ranked Internal Shortlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,359000,2026-02-28,2026-03-01 01:00:00+00:00,feb15_plus,Saint Louis,Duquesne,-14.0,14.664,HOME,0.526,0.002,0.494,0.786,0.292,True,persistent,21.088,21.088,5,26,False,disagreement-led persistent,decision-useful internal candidate
2026,211948,2026-02-28,2026-03-01 02:00:00+00:00,feb15_plus,St. Thomas-Minnesota,Omaha,-11.5,11.928,HOME,0.518,-0.006,0.494,0.729,0.235,True,persistent,16.167,16.167,5,23,False,disagreement-led persistent,decision-useful internal candidate
2026,211995,2026-02-28,2026-02-28 18:00:00+00:00,feb15_plus,Mississippi State,Missouri,1.25,-1.931,AWAY,0.519,-0.004,0.489,0.729,0.24,True,persistent,16.261,16.261,6,27,False,disagreement-led persistent,decision-useful internal candidate
2026,358987,2026-02-28,2026-02-28 17:30:00+00:00,feb15_plus,VCU,Fordham,-12.75,14.514,HOME,0.562,0.038,0.501,0.706,0.206,True,persistent,14.278,14.278,4,27,False,disagreement-led persistent,decision-useful internal candidate
2026,358993,2026-02-28,2026-02-28 21:00:00+00:00,feb15_plus,Loyola Chicago,Richmond,2.5,-2.857,AWAY,0.508,-0.016,0.481,0.694,0.214,True,none,15.198,15.198,0,0,False,disagreement-led support,decision-useful internal candidate
2026,212007,2026-02-28,2026-02-28 19:00:00+00:00,feb15_plus,UT Martin,Tennessee State,-2.5,-0.48,AWAY,0.6,0.077,0.503,0.688,0.185,True,persistent,12.436,12.436,2,2,False,disagreement-led persistent,decision-useful internal candidate
2026,212017,2026-02-28,2026-02-28 22:30:00+00:00,feb15_plus,Sam Houston,Missouri State,-6.75,9.274,HOME,0.59,0.066,0.506,0.676,0.169,True,persistent,11.908,11.908,5,24,False,disagreement-led persistent,decision-useful internal candidate
2026,212015,2026-02-28,2026-02-28 22:00:00+00:00,feb15_plus,Kennesaw State,Delaware,-11.75,11.801,HOME,0.502,-0.022,0.491,0.66,0.169,True,none,11.872,11.872,0,0,False,disagreement-led support,decision-useful internal candidate
2026,211967,2026-02-28,2026-02-28 19:00:00+00:00,feb15_plus,UAlbany,New Hampshire,-3.5,5.161,HOME,0.561,0.037,0.5,0.652,0.152,True,persistent,10.872,10.872,5,25,False,disagreement-led persistent,decision-useful internal candidate
2026,215112,2026-02-28,2026-03-01 01:00:00+00:00,feb15_plus,UCF,Baylor,-2.5,2.937,HOME,0.521,-0.003,0.496,0.644,0.149,True,persistent,9.571,9.571,5,25,False,disagreement-led persistent,decision-useful internal candidate
2026,359002,2026-02-28,2026-03-01 03:00:00+00:00,feb15_plus,UNLV,Nevada,0.25,-1.765,AWAY,0.56,0.036,0.5,0.634,0.134,True,persistent,8.895,8.895,5,27,False,disagreement-led persistent,decision-useful internal candidate
2026,215125,2026-02-28,2026-02-28 21:00:00+00:00,feb15_plus,Winthrop,Presbyterian,-7.5,9.962,HOME,0.581,0.057,0.505,0.631,0.126,True,persistent,9.237,9.237,5,12,False,disagreement-led persistent,decision-useful internal candidate
2026,215137,2026-02-28,2026-03-01 01:30:00+00:00,feb15_plus,North Carolina,Virginia Tech,-6.75,8.632,HOME,0.562,0.038,0.501,0.63,0.13,True,persistent,8.485,8.485,6,27,False,disagreement-led persistent,decision-useful internal candidate
2026,211953,2026-02-28,2026-02-28 20:00:00+00:00,feb15_plus,Oakland,Detroit Mercy,-8.5,8.641,HOME,0.505,-0.019,0.489,0.627,0.138,True,persistent,9.004,9.004,5,18,False,disagreement-led persistent,decision-useful internal candidate
2026,211985,2026-02-28,2026-02-28 23:30:00+00:00,feb15_plus,Texas A&M-Corpus Christi,Northwestern State,-5.5,5.555,HOME,0.502,-0.022,0.489,0.614,0.125,True,persistent,8.108,8.108,6,16,False,disagreement-led persistent,decision-useful internal candidate
```

## NCAA Caution / Diagnostic Rows

_empty_

## Raw-Edge-Only Watchlist

```csv
season,gameId,game_date,startDate,slice,homeTeam,awayTeam,book_spread,predicted_spread,model_pick_side,pick_cover_prob,pick_prob_edge,raw_logit_score,filter_score,score_lift_vs_raw_logit,flagged_mainly_by_disagreement,persistence_label,he_market_edge_for_pick,abs_he_vs_market_edge,pick_team_recent_same_sign_count_21d,pick_team_prior_same_sign_streak,neutral_site_flag,signal_driver,usage_label
2026,215139,2026-02-28,2026-02-28 17:00:00+00:00,feb15_plus,Georgia Tech,Florida State,6.0,0.332,HOME,0.736,0.212,0.529,0.454,-0.074,False,none,-2.745,2.745,0,0,False,raw edge only / filtered out,raw-edge watchlist only
2026,215122,2026-02-28,2026-02-28 19:00:00+00:00,feb15_plus,Minnesota,UCLA,1.5,3.393,HOME,0.718,0.195,0.518,0.479,-0.039,False,persistent,-1.756,1.756,5,17,False,raw edge only / filtered out,raw-edge watchlist only
2026,212023,2026-02-28,2026-02-28 23:00:00+00:00,feb15_plus,Columbia,Yale,6.25,0.533,HOME,0.715,0.191,0.532,0.304,-0.228,False,persistent,-12.171,12.171,5,20,False,raw edge only / filtered out,raw-edge watchlist only
2026,212006,2026-02-28,2026-02-28 19:00:00+00:00,feb15_plus,Bellarmine,Austin Peay,5.5,-0.063,HOME,0.697,0.173,0.522,0.37,-0.152,False,persistent,-7.354,7.354,6,26,False,raw edge only / filtered out,raw-edge watchlist only
2026,215141,2026-02-28,2026-02-28 19:00:00+00:00,feb15_plus,Clemson,Louisville,2.5,2.173,HOME,0.686,0.162,0.517,0.467,-0.05,False,persistent,-2.965,2.965,5,27,False,raw edge only / filtered out,raw-edge watchlist only
2026,358990,2026-02-28,2026-02-28 21:00:00+00:00,feb15_plus,Wyoming,Air Force,-21.5,15.901,AWAY,0.679,0.156,0.523,0.138,-0.385,False,persistent,-22.677,22.677,6,27,False,raw edge only / filtered out,raw-edge watchlist only
2026,215147,2026-02-28,2026-02-28 19:00:00+00:00,feb15_plus,South Carolina Upstate,Gardner-Webb,-12.75,8.474,AWAY,0.677,0.153,0.514,0.251,-0.263,False,persistent,-12.619,12.619,6,26,False,raw edge only / filtered out,raw-edge watchlist only
2026,215151,2026-02-28,2026-03-01 01:00:00+00:00,feb15_plus,UC Riverside,Cal State Northridge,5.25,-0.785,HOME,0.674,0.151,0.515,0.444,-0.071,False,persistent,-3.305,3.305,6,12,False,raw edge only / filtered out,raw-edge watchlist only
2026,211952,2026-02-28,2026-02-28 18:00:00+00:00,feb15_plus,Robert Morris,Cleveland State,-12.75,8.35,AWAY,0.674,0.15,0.515,0.257,-0.258,False,persistent,-13.597,13.597,6,22,False,raw edge only / filtered out,raw-edge watchlist only
2026,212010,2026-02-28,2026-02-28 21:30:00+00:00,feb15_plus,Western Illinois,Lindenwood,11.5,-6.863,HOME,0.671,0.147,0.516,0.208,-0.308,False,persistent,-16.086,16.086,6,26,False,raw edge only / filtered out,raw-edge watchlist only
2026,212001,2026-02-28,2026-02-28 19:00:00+00:00,feb15_plus,Central Arkansas,Queens University,-2.5,6.427,HOME,0.665,0.141,0.517,0.568,0.051,False,persistent,4.954,4.954,7,12,False,raw edge only / filtered out,raw-edge watchlist only
2026,212021,2026-02-28,2026-02-28 18:00:00+00:00,feb15_plus,Central Connecticut,St. Francis (PA),-8.5,4.611,AWAY,0.658,0.134,0.511,0.338,-0.173,False,persistent,-8.068,8.068,8,27,False,raw edge only / filtered out,raw-edge watchlist only
2026,211987,2026-02-28,2026-02-28 23:00:00+00:00,feb15_plus,New Orleans,McNeese,10.25,-4.859,HOME,0.658,0.134,0.52,0.235,-0.285,False,none,-15.788,15.788,0,0,False,raw edge only / filtered out,raw-edge watchlist only
2026,358992,2026-02-28,2026-02-28 21:00:00+00:00,feb15_plus,Morgan State,Howard,9.25,-4.528,HOME,0.655,0.131,0.516,0.165,-0.351,False,persistent,-19.652,19.652,5,24,False,raw edge only / filtered out,raw-edge watchlist only
2026,215131,2026-02-28,2026-02-28 22:30:00+00:00,feb15_plus,Creighton,Providence,-3.5,6.735,HOME,0.652,0.128,0.513,0.542,0.029,False,persistent,2.826,2.826,6,28,False,raw edge only / filtered out,raw-edge watchlist only
```
