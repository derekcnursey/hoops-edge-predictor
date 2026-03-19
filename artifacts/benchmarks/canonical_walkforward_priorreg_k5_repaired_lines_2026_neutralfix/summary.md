# Canonical Walk-Forward Benchmark v1

## Protocol

- Holdouts: 2019, 2020, 2022, 2023, 2024, 2025
- Excluded seasons: 2021
- Features: gold (team_adjusted_efficiencies_no_garbage_priorreg_k5_v1) + adjusted + 53 features (`adj_a0.85_p10`)
- Primary target: homeScore - awayScore
- External book benchmark: `PreferredBookSpread`
- Uses true closing timestamps: False
- Provider preference order: Draft Kings, ESPN BET, Bovada
- Line selection rule: Select one spread per game by taking the first non-null spread after sorting by preferred providers (Draft Kings, ESPN BET, Bovada), then by provider name.

## Pooled Metrics

| model | n_games | n_lined | MAE_all | RMSE_all | MedAE_all | WinAcc_all | MAE_lined | BookMAE_lined | DeltaVsBook_MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HomeMarginMean | 5528 | 4971 | 14.1363 | 18.9301 | 11.3969 | 0.6731 | 11.9651 | 9.0520 | 2.9131 |
| Ridge | 5528 | 4971 | 9.4156 | 12.1368 | 7.6322 | 0.7585 | 8.7435 | 9.0520 | -0.3085 |
| HistGradientBoosting | 5528 | 4971 | 9.3839 | 12.0327 | 7.7486 | 0.7489 | 9.1165 | 9.0520 | 0.0645 |
| LightGBM | 5528 | 4971 | 9.3601 | 11.9828 | 7.7036 | 0.7511 | 9.1121 | 9.0520 | 0.0601 |
| CurrentMLP | 5528 | 4971 | 8.8723 | 11.3041 | 7.3223 | 0.7612 | 8.8305 | 9.0520 | -0.2215 |
| PreferredBookSpread | 4971 | 4971 | NA | NA | NA | NA | 9.0520 | 9.0520 | 0.0000 |

## Fold Metrics

| model | holdout_season | n_games | n_lined | MAE_all | RMSE_all | MedAE_all | WinAcc_all | MAE_lined | BookMAE_lined | DeltaVsBook_MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CurrentMLP | 2026 | 5528 | 4971 | 8.8723 | 11.3041 | 7.3223 | 0.7612 | 8.8305 | 9.0520 | -0.2215 |
| HistGradientBoosting | 2026 | 5528 | 4971 | 9.3839 | 12.0327 | 7.7486 | 0.7489 | 9.1165 | 9.0520 | 0.0645 |
| HomeMarginMean | 2026 | 5528 | 4971 | 14.1363 | 18.9301 | 11.3969 | 0.6731 | 11.9651 | 9.0520 | 2.9131 |
| LightGBM | 2026 | 5528 | 4971 | 9.3601 | 11.9828 | 7.7036 | 0.7511 | 9.1121 | 9.0520 | 0.0601 |
| PreferredBookSpread | 2026 | 4971 | 4971 | NA | NA | NA | NA | 9.0520 | 9.0520 | 0.0000 |
| Ridge | 2026 | 5528 | 4971 | 9.4156 | 12.1368 | 7.6322 | 0.7585 | 8.7435 | 9.0520 | -0.3085 |
