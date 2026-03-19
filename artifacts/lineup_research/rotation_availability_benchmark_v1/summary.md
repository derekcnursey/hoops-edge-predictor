# Rotation Availability Benchmark v1

- Player-game spine usable: `True` (usable)
- Primary evaluation slice: `dec15_plus` with cutoff `12-15`

## Pooled full-season metrics
```csv
variant,games,lined_games,mae_all,mae_lined
gold_priorreg_k5_v1,36557.0,32829.0,9.7457,9.2046
gold_priorreg_k5_v1_rotation_availability_v1,36557.0,32829.0,9.7518,9.2077
torvik,36557.0,32829.0,9.6077,9.1027
torvik_rotation_availability_v1,36557.0,32829.0,9.6142,9.1056
```

## Pooled dec15_plus metrics
```csv
variant,games,lined_games,mae_all,mae_lined
gold_priorreg_k5_v1,3591.0,2981.0,10.2211,9.3697
gold_priorreg_k5_v1_rotation_availability_v1,3591.0,2981.0,10.2512,9.3839
torvik,3591.0,2981.0,10.2885,9.3905
torvik_rotation_availability_v1,3591.0,2981.0,10.3067,9.3974
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
gold_priorreg_k5_v1_rotation_availability_v1,2019,648.0,565.0,10.342,9.6501
gold_priorreg_k5_v1_rotation_availability_v1,2020,571.0,493.0,10.1858,9.2471
gold_priorreg_k5_v1_rotation_availability_v1,2022,455.0,360.0,10.4831,9.6091
gold_priorreg_k5_v1_rotation_availability_v1,2023,687.0,606.0,9.7847,9.1032
gold_priorreg_k5_v1_rotation_availability_v1,2024,646.0,499.0,10.4872,9.4071
gold_priorreg_k5_v1_rotation_availability_v1,2025,584.0,458.0,10.3216,9.372
torvik,2019,648.0,565.0,10.2002,9.4763
torvik,2020,571.0,493.0,10.1917,9.2599
torvik,2022,455.0,360.0,10.8428,9.8426
torvik,2023,687.0,606.0,9.8682,9.1388
torvik,2024,646.0,499.0,10.4059,9.3752
torvik,2025,584.0,458.0,10.414,9.4195
torvik_rotation_availability_v1,2019,648.0,565.0,10.3086,9.5523
torvik_rotation_availability_v1,2020,571.0,493.0,10.1999,9.2426
torvik_rotation_availability_v1,2022,455.0,360.0,10.7685,9.8305
torvik_rotation_availability_v1,2023,687.0,606.0,9.8587,9.1209
torvik_rotation_availability_v1,2024,646.0,499.0,10.4803,9.4348
torvik_rotation_availability_v1,2025,584.0,458.0,10.3843,9.3573
```
