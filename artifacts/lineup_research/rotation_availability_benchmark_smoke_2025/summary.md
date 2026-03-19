# Rotation Availability Benchmark v1

- Player-game spine usable: `True` (usable)
- Primary evaluation slice: `dec15_plus` with cutoff `12-15`

## Pooled full-season metrics
```csv
variant,games,lined_games,mae_all,mae_lined
gold_priorreg_k5_v1,6292.0,5437.0,9.8397,9.2668
gold_priorreg_k5_v1_rotation_availability_v1,6292.0,5437.0,9.8266,9.2492
torvik,6292.0,5437.0,9.6341,9.1144
torvik_rotation_availability_v1,6292.0,5437.0,9.6357,9.1201
```

## Pooled dec15_plus metrics
```csv
variant,games,lined_games,mae_all,mae_lined
gold_priorreg_k5_v1,584.0,458.0,10.2854,9.3248
gold_priorreg_k5_v1_rotation_availability_v1,584.0,458.0,10.3216,9.372
torvik,584.0,458.0,10.414,9.4195
torvik_rotation_availability_v1,584.0,458.0,10.3843,9.3573
```

## By-season dec15_plus metrics
```csv
variant,holdout_season,games,lined_games,mae_all,mae_lined
gold_priorreg_k5_v1,2025,584.0,458.0,10.2854,9.3248
gold_priorreg_k5_v1_rotation_availability_v1,2025,584.0,458.0,10.3216,9.372
torvik,2025,584.0,458.0,10.414,9.4195
torvik_rotation_availability_v1,2025,584.0,458.0,10.3843,9.3573
```
