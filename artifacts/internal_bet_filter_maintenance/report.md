# Internal Bet Filter Maintenance Report

## Recommendation

- Recommended action: **review**
- Default policy is to leave the filter alone unless review thresholds are met.
- In-season recalibration is not recommended; any real recalibration should wait for offseason unless there is a severe data-integrity issue.

## Operational Monitoring

- Update the live tracker after each slate settles.
- Watch `internal_filter`, `filter_only`, `raw_edge_baseline`, and `raw_only` side by side.
- Keep NCAA rows separate as `ncaa_caution` and treat them as diagnostic only.

## Season-To-Date Summary

```csv
window,strategy,slice,tracked_rows,bets,wins,losses,pushes,pending,ats_hit_rate,roi_per_1_at_minus_110,avg_pick_prob_edge,avg_filter_score
season_to_date,filter_only,conference_tournaments,46,44,25,19,1,1,0.5682,0.0847,0.0129,0.6388
season_to_date,internal_filter,conference_tournaments,55,53,30,23,1,1,0.566,0.0806,0.0309,0.6375
season_to_date,raw_edge_baseline,conference_tournaments,54,54,32,22,0,0,0.5926,0.1313,0.1279,0.3967
season_to_date,raw_only,conference_tournaments,45,45,27,18,0,0,0.6,0.1455,0.1305,0.35
season_to_date,filter_only,feb15_plus,170,89,47,42,1,80,0.5281,0.0082,0.0208,0.6514
season_to_date,internal_filter,feb15_plus,227,122,66,56,1,104,0.541,0.0328,0.0486,0.649
season_to_date,raw_edge_baseline,feb15_plus,376,193,111,82,0,183,0.5751,0.098,0.14,0.4011
season_to_date,raw_only,feb15_plus,319,160,92,68,0,159,0.575,0.0977,0.1419,0.3581
season_to_date,filter_only,full_live,170,89,47,42,1,80,0.5281,0.0082,0.0208,0.6514
season_to_date,internal_filter,full_live,227,122,66,56,1,104,0.541,0.0328,0.0486,0.649
season_to_date,raw_edge_baseline,full_live,376,193,111,82,0,183,0.5751,0.098,0.14,0.4011
season_to_date,raw_only,full_live,319,160,92,68,0,159,0.575,0.0977,0.1419,0.3581
season_to_date,filter_only,march,80,46,25,21,1,33,0.5435,0.0375,0.0154,0.6511
season_to_date,internal_filter,march,95,55,30,25,1,39,0.5455,0.0413,0.0313,0.6477
season_to_date,raw_edge_baseline,march,137,55,33,22,0,82,0.6,0.1455,0.1387,0.3835
season_to_date,raw_only,march,122,46,28,18,0,76,0.6087,0.1621,0.142,0.3532
season_to_date,filter_only,ncaa_tournament,0,0,0,0,0,0,,,,
season_to_date,internal_filter,ncaa_tournament,0,0,0,0,0,0,,,,
season_to_date,raw_edge_baseline,ncaa_tournament,0,0,0,0,0,0,,,,
season_to_date,raw_only,ncaa_tournament,0,0,0,0,0,0,,,,
```

## Trailing Window Summary

```csv
window,strategy,slice,tracked_rows,bets,wins,losses,pushes,pending,ats_hit_rate,roi_per_1_at_minus_110,avg_pick_prob_edge,avg_filter_score
trailing_14d,filter_only,conference_tournaments,46,44,25,19,1,1,0.5682,0.0847,0.0129,0.6388
trailing_14d,internal_filter,conference_tournaments,55,53,30,23,1,1,0.566,0.0806,0.0309,0.6375
trailing_14d,raw_edge_baseline,conference_tournaments,54,54,32,22,0,0,0.5926,0.1313,0.1279,0.3967
trailing_14d,filter_only,full_live,80,46,25,21,1,33,0.5435,0.0375,0.0154,0.6511
trailing_14d,internal_filter,full_live,95,55,30,25,1,39,0.5455,0.0413,0.0313,0.6477
trailing_14d,raw_edge_baseline,full_live,137,55,33,22,0,82,0.6,0.1455,0.1387,0.3835
trailing_14d,filter_only,march,80,46,25,21,1,33,0.5435,0.0375,0.0154,0.6511
trailing_14d,internal_filter,march,95,55,30,25,1,39,0.5455,0.0413,0.0313,0.6477
trailing_14d,raw_edge_baseline,march,137,55,33,22,0,82,0.6,0.1455,0.1387,0.3835
trailing_30d,filter_only,conference_tournaments,46,44,25,19,1,1,0.5682,0.0847,0.0129,0.6388
trailing_30d,internal_filter,conference_tournaments,55,53,30,23,1,1,0.566,0.0806,0.0309,0.6375
trailing_30d,raw_edge_baseline,conference_tournaments,54,54,32,22,0,0,0.5926,0.1313,0.1279,0.3967
trailing_30d,filter_only,full_live,170,89,47,42,1,80,0.5281,0.0082,0.0208,0.6514
trailing_30d,internal_filter,full_live,227,122,66,56,1,104,0.541,0.0328,0.0486,0.649
trailing_30d,raw_edge_baseline,full_live,376,193,111,82,0,183,0.5751,0.098,0.14,0.4011
trailing_30d,filter_only,march,80,46,25,21,1,33,0.5435,0.0375,0.0154,0.6511
trailing_30d,internal_filter,march,95,55,30,25,1,39,0.5455,0.0413,0.0313,0.6477
trailing_30d,raw_edge_baseline,march,137,55,33,22,0,82,0.6,0.1455,0.1387,0.3835
```

## Benchmark Anchor Comparison

```csv
strategy,slice,live_bets,live_ats_hit_rate,live_roi_per_1_at_minus_110,benchmark_bets,benchmark_ats_hit_rate,benchmark_roi_per_1_at_minus_110,roi_delta_vs_benchmark,ats_delta_vs_benchmark
internal_filter,full_live,122,0.541,0.0328,2884,0.5933,0.1326,-0.0998,-0.0523
internal_filter,feb15_plus,122,0.541,0.0328,421,0.5534,0.0566,-0.0238,-0.0125
internal_filter,march,55,0.5455,0.0413,242,0.5579,0.065,-0.0237,-0.0124
internal_filter,conference_tournaments,53,0.566,0.0806,105,0.619,0.1818,-0.1012,-0.053
internal_filter,ncaa_tournament,0,,,86,0.4884,-0.0677,,
raw_edge_baseline,full_live,193,0.5751,0.098,7639,0.5261,0.0044,0.0936,0.049
raw_edge_baseline,feb15_plus,193,0.5751,0.098,1540,0.489,-0.0665,0.1645,0.0862
raw_edge_baseline,march,55,0.6,0.1455,721,0.484,-0.0759,0.2214,0.116
raw_edge_baseline,conference_tournaments,54,0.5926,0.1313,296,0.5135,-0.0197,0.151,0.0791
raw_edge_baseline,ncaa_tournament,0,,,91,0.4176,-0.2028,,
```

## Alerts

```csv
level,category,window,strategy,slice,message
review,filter_only_lagging_raw_only,trailing_30d,filter_only,full_live,Filter-only ROI 0.008 trails raw-only ROI 0.098 by more than 0.080.
```

## Policy Guardrails

- Do not react to short runs smaller than `15` settled bets over `14` days.
- Do not treat March / conference-tournament underperformance as meaningful until at least `15` settled bets exist in that slice.
- Do not consider recalibration until there is a completed-season sample with at least `100` internal-filter bets and `50` filter-only bets.
- Do not change thresholds mid-season because of one bad week or one bad tournament pocket.
- Use review alerts to inspect grading, market-line integrity, and disagreement composition before touching model calibration.
