# Internal Bet Filter Live Tracking

## Scope

- Tracked report directory: `/Users/dereknursey/Desktop/ml_projects/codex_review/hoops-edge-predictor_codex/artifacts/daily_internal_bet_filter`
- Seasons included: `[2026]`
- Tracked slate dates: `['2026-02-15', '2026-02-16', '2026-02-17', '2026-02-18', '2026-02-19', '2026-02-20', '2026-02-21', '2026-02-22', '2026-02-23', '2026-02-24', '2026-02-25', '2026-02-26', '2026-02-27', '2026-02-28', '2026-03-01', '2026-03-02', '2026-03-03', '2026-03-04', '2026-03-05', '2026-03-06', '2026-03-07', '2026-03-08', '2026-03-09', '2026-03-10', '2026-03-11', '2026-03-12', '2026-03-13', '2026-03-14', '2026-03-15']`
- Ledger rows across strategy buckets: `1149`

## Headline Strategy Summary

```csv
strategy,slice,tracked_rows,bets,wins,losses,pushes,pending,ats_hit_rate,roi_per_1_at_minus_110,avg_pick_prob_edge,avg_filter_score
filter_only,conference_tournaments,46,44,25,19,1,1,0.5682,0.0847,0.0129,0.6388
internal_filter,conference_tournaments,55,53,30,23,1,1,0.566,0.0806,0.0309,0.6375
ncaa_caution,conference_tournaments,0,0,0,0,0,0,,,,
overlap,conference_tournaments,9,9,5,4,0,0,0.5556,0.0606,0.1148,0.6307
raw_edge_baseline,conference_tournaments,54,54,32,22,0,0,0.5926,0.1313,0.1279,0.3967
raw_only,conference_tournaments,45,45,27,18,0,0,0.6,0.1455,0.1305,0.35
filter_only,feb15_plus,170,89,47,42,1,80,0.5281,0.0082,0.0208,0.6514
internal_filter,feb15_plus,227,122,66,56,1,104,0.541,0.0328,0.0486,0.649
ncaa_caution,feb15_plus,0,0,0,0,0,0,,,,
overlap,feb15_plus,57,33,19,14,0,24,0.5758,0.0992,0.1296,0.6416
raw_edge_baseline,feb15_plus,376,193,111,82,0,183,0.5751,0.098,0.14,0.4011
raw_only,feb15_plus,319,160,92,68,0,159,0.575,0.0977,0.1419,0.3581
filter_only,full_live,170,89,47,42,1,80,0.5281,0.0082,0.0208,0.6514
internal_filter,full_live,227,122,66,56,1,104,0.541,0.0328,0.0486,0.649
ncaa_caution,full_live,0,0,0,0,0,0,,,,
overlap,full_live,57,33,19,14,0,24,0.5758,0.0992,0.1296,0.6416
raw_edge_baseline,full_live,376,193,111,82,0,183,0.5751,0.098,0.14,0.4011
raw_only,full_live,319,160,92,68,0,159,0.575,0.0977,0.1419,0.3581
filter_only,march,80,46,25,21,1,33,0.5435,0.0375,0.0154,0.6511
internal_filter,march,95,55,30,25,1,39,0.5455,0.0413,0.0313,0.6477
ncaa_caution,march,0,0,0,0,0,0,,,,
overlap,march,15,9,5,4,0,6,0.5556,0.0606,0.1123,0.6296
raw_edge_baseline,march,137,55,33,22,0,82,0.6,0.1455,0.1387,0.3835
raw_only,march,122,46,28,18,0,76,0.6087,0.1621,0.142,0.3532
filter_only,ncaa_tournament,0,0,0,0,0,0,,,,
internal_filter,ncaa_tournament,0,0,0,0,0,0,,,,
ncaa_caution,ncaa_tournament,0,0,0,0,0,0,,,,
overlap,ncaa_tournament,0,0,0,0,0,0,,,,
raw_edge_baseline,ncaa_tournament,0,0,0,0,0,0,,,,
raw_only,ncaa_tournament,0,0,0,0,0,0,,,,
```

## Recent By-Day Results

```csv
strategy,game_date,tracked_rows,bets,wins,losses,pushes,pending,ats_hit_rate,roi_per_1_at_minus_110,avg_pick_prob_edge,avg_filter_score
internal_filter,2026-03-10,5,5,3,2,0,0,0.6,0.1455,0.0064,0.6534
overlap,2026-03-10,1,1,1,0,0,0,1.0,0.9091,0.09,0.598
raw_edge_baseline,2026-03-10,3,3,3,0,0,0,1.0,0.9091,0.118,0.409
raw_only,2026-03-10,2,2,2,0,0,0,1.0,0.9091,0.132,0.3145
filter_only,2026-03-11,6,6,5,1,0,0,0.8333,0.5909,-0.0058,0.6177
internal_filter,2026-03-11,9,9,6,3,0,0,0.6667,0.2727,0.0398,0.6333
overlap,2026-03-11,3,3,1,2,0,0,0.3333,-0.3636,0.131,0.6647
raw_edge_baseline,2026-03-11,12,12,5,7,0,0,0.4167,-0.2045,0.1258,0.4983
raw_only,2026-03-11,9,9,4,5,0,0,0.4444,-0.1515,0.124,0.4429
filter_only,2026-03-12,7,7,0,7,0,0,0.0,-1.0,0.001,0.6526
internal_filter,2026-03-12,8,8,0,8,0,0,0.0,-1.0,0.0225,0.6447
overlap,2026-03-12,1,1,0,1,0,0,0.0,-1.0,0.087,0.59
raw_edge_baseline,2026-03-12,3,3,1,2,0,0,0.3333,-0.3636,0.1087,0.4553
raw_only,2026-03-12,2,2,1,1,0,0,0.5,-0.0455,0.1195,0.388
filter_only,2026-03-13,7,6,4,2,1,0,0.6667,0.2727,0.0137,0.6337
internal_filter,2026-03-13,7,6,4,2,1,0,0.6667,0.2727,0.0137,0.6337
raw_edge_baseline,2026-03-13,6,6,3,3,0,0,0.5,-0.0455,0.1393,0.3558
raw_only,2026-03-13,6,6,3,3,0,0,0.5,-0.0455,0.1393,0.3558
filter_only,2026-03-14,1,0,0,0,0,1,,,-0.019,0.608
internal_filter,2026-03-14,1,0,0,0,0,1,,,-0.019,0.608
```

## Notes

- `internal_filter` excludes NCAA threshold-pass rows; NCAA is tracked separately under `ncaa_caution`.
- ROI uses the same `-110` vig convention as the research benchmark.
- Pushes are tracked separately and excluded from ATS hit-rate denominators.
- Pending rows are live picks that do not have final scores attached yet.
