# Lined-Games Tie-Break: HistGradientBoosting vs LightGBM

This analysis uses lined games only from the completed canonical walk-forward benchmark.
- Bootstrap draws: 2000
- Bootstrap seed: 42
- Difference convention: `LightGBM MAE - HistGradientBoosting MAE`
- Negative difference favors LightGBM

## Pooled

| slice | n_games | hgbr_mae_lined | lgb_mae_lined | mae_diff_lgb_minus_hgbr | ci95_lo | ci95_hi | lgb_game_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| All lined games | 32851 | 9.1013 | 9.1016 | 0.0002 | -0.0167 | 0.0171 | 0.5005 |
| Nov-Dec | 12590 | 9.4856 | 9.4879 | 0.0023 | -0.0229 | 0.0308 | 0.5028 |
| Jan-Mar | 20217 | 8.8643 | 8.8635 | -0.0008 | -0.0204 | 0.0187 | 0.4993 |

## By Holdout Season

| holdout_season | n_games | hgbr_mae_lined | lgb_mae_lined | mae_diff_lgb_minus_hgbr | ci95_lo | ci95_hi | lgb_game_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2019.0000 | 5543.0000 | 9.0242 | 9.0290 | 0.0048 | -0.0337 | 0.0465 | 0.5003 |
| 2020.0000 | 5326.0000 | 9.0992 | 9.0978 | -0.0014 | -0.0409 | 0.0383 | 0.5002 |
| 2022.0000 | 5463.0000 | 8.9181 | 8.9205 | 0.0024 | -0.0363 | 0.0417 | 0.4924 |
| 2023.0000 | 5773.0000 | 9.1791 | 9.1444 | -0.0346 | -0.0738 | 0.0017 | 0.5108 |
| 2024.0000 | 5309.0000 | 9.2746 | 9.2499 | -0.0247 | -0.0664 | 0.0169 | 0.5054 |
| 2025.0000 | 5437.0000 | 9.1144 | 9.1707 | 0.0563 | 0.0125 | 0.0994 | 0.4937 |

## By Holdout Season And Phase

| holdout_season | phase | n_games | hgbr_mae_lined | lgb_mae_lined | mae_diff_lgb_minus_hgbr | ci95_lo | ci95_hi | lgb_game_win_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2019 | Jan-Mar | 3488 | 8.8083 | 8.8255 | 0.0172 | -0.0300 | 0.0648 | 0.4963 |
| 2019 | Nov-Dec | 2043 | 9.3986 | 9.3814 | -0.0173 | -0.0854 | 0.0522 | 0.5081 |
| 2020 | Jan-Mar | 3235 | 8.8299 | 8.8355 | 0.0056 | -0.0420 | 0.0559 | 0.4968 |
| 2020 | Nov-Dec | 2091 | 9.5157 | 9.5035 | -0.0122 | -0.0791 | 0.0573 | 0.5055 |
| 2022 | Jan-Mar | 3481 | 8.6375 | 8.6163 | -0.0212 | -0.0664 | 0.0239 | 0.5030 |
| 2022 | Nov-Dec | 1978 | 9.4127 | 9.4545 | 0.0418 | -0.0296 | 0.1106 | 0.4747 |
| 2023 | Jan-Mar | 3421 | 8.9173 | 8.8995 | -0.0178 | -0.0641 | 0.0279 | 0.5037 |
| 2023 | Nov-Dec | 2349 | 9.5649 | 9.5061 | -0.0587 | -0.1257 | 0.0066 | 0.5211 |
| 2024 | Jan-Mar | 3330 | 9.1480 | 9.1231 | -0.0249 | -0.0725 | 0.0243 | 0.5012 |
| 2024 | Nov-Dec | 1973 | 9.4911 | 9.4684 | -0.0227 | -0.0932 | 0.0497 | 0.5124 |
| 2025 | Jan-Mar | 3262 | 8.8550 | 8.8931 | 0.0381 | -0.0165 | 0.0923 | 0.4945 |
| 2025 | Nov-Dec | 2156 | 9.5142 | 9.6021 | 0.0878 | 0.0169 | 0.1575 | 0.4921 |
