# Availability Shock Benchmark v1

- Player-game spine usable: `True` (usable)
- Primary evaluation slice: `dec15_plus` with cutoff `12-15`

## Pooled full-season metrics
```csv
variant,games,lined_games,mae_all,mae_lined
gold_priorreg_k5_v1,36557.0,32829.0,9.7457,9.2046
gold_priorreg_k5_v1_availability_shock_v1,36557.0,32829.0,9.7425,9.2068
torvik,36557.0,32829.0,9.6077,9.1027
torvik_availability_shock_v1,36557.0,32829.0,9.6096,9.103
```

## Pooled dec15_plus metrics
```csv
variant,games,lined_games,mae_all,mae_lined
gold_priorreg_k5_v1,3591.0,2981.0,10.2211,9.3697
gold_priorreg_k5_v1_availability_shock_v1,3591.0,2981.0,10.2627,9.397
torvik,3591.0,2981.0,10.2885,9.3905
torvik_availability_shock_v1,3591.0,2981.0,10.2789,9.3779
```

## By-season dec15_plus metrics
```csv
variant,holdout_season,games,lined_games,mae_all,mae_lined
gold_priorreg_k5_v1,2019,648.0,565.0,10.215,9.5718
gold_priorreg_k5_v1,2020,571.0,493.0,10.1943,9.2961
gold_priorreg_k5_v1,2022,455.0,360.0,10.5682,9.6836
gold_priorreg_k5_v1,2023,687.0,606.0,9.7228,9.0606
gold_priorreg_k5_v1,2024,646.0,499.0,10.4786,9.4036
gold_priorreg_k5_v1,2025,584.0,458.0,10.2854,9.3248
gold_priorreg_k5_v1_availability_shock_v1,2019,648.0,565.0,10.3546,9.6717
gold_priorreg_k5_v1_availability_shock_v1,2020,571.0,493.0,10.2009,9.2934
gold_priorreg_k5_v1_availability_shock_v1,2022,455.0,360.0,10.5404,9.6136
gold_priorreg_k5_v1_availability_shock_v1,2023,687.0,606.0,9.7798,9.1133
gold_priorreg_k5_v1_availability_shock_v1,2024,646.0,499.0,10.4318,9.3669
gold_priorreg_k5_v1_availability_shock_v1,2025,584.0,458.0,10.3858,9.4075
torvik,2019,648.0,565.0,10.2002,9.4763
torvik,2020,571.0,493.0,10.1917,9.2599
torvik,2022,455.0,360.0,10.8428,9.8426
torvik,2023,687.0,606.0,9.8682,9.1388
torvik,2024,646.0,499.0,10.4059,9.3752
torvik,2025,584.0,458.0,10.414,9.4195
torvik_availability_shock_v1,2019,648.0,565.0,10.2415,9.4823
torvik_availability_shock_v1,2020,571.0,493.0,10.1785,9.2367
torvik_availability_shock_v1,2022,455.0,360.0,10.7047,9.7503
torvik_availability_shock_v1,2023,687.0,606.0,9.8675,9.1315
torvik_availability_shock_v1,2024,646.0,499.0,10.4261,9.4006
torvik_availability_shock_v1,2025,584.0,458.0,10.4076,9.4095
```
