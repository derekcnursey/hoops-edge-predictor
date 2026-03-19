# Canonical Walk-Forward Benchmark v1

## Protocol

- Holdouts: 2019, 2020, 2022, 2023, 2024, 2025
- Excluded seasons: 2021
- Features: torvik + adjusted + 53 features (`adj_a0.85_p10`)
- Primary target: homeScore - awayScore
- External book benchmark: `PreferredBookSpread`
- Uses true closing timestamps: False
- Provider preference order: Draft Kings, ESPN BET, Bovada
- Line selection rule: Select one spread per game by taking the first non-null spread after sorting by preferred providers (Draft Kings, ESPN BET, Bovada), then by provider name.

## Pooled Metrics

| model | n_games | n_lined | MAE_all | RMSE_all | MedAE_all | WinAcc_all | MAE_lined | BookMAE_lined | DeltaVsBook_MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HomeMarginMean | 5356 | 4799 | 14.2815 | 19.0967 | 11.3969 | 0.6744 | 12.0494 | 9.0718 | 2.9777 |
| Ridge | 5356 | 4799 | 11.5242 | 15.7029 | 8.7742 | 0.7407 | 9.7262 | 9.0718 | 0.6544 |
| HistGradientBoosting | 5356 | 4799 | 9.8856 | 12.7920 | 8.0389 | 0.7549 | 9.2156 | 9.0718 | 0.1438 |
| LightGBM | 5356 | 4799 | 9.9317 | 12.8116 | 8.0882 | 0.7511 | 9.2882 | 9.0718 | 0.2164 |
| CurrentMLP | 5356 | 4799 | 10.1486 | 13.3133 | 8.0991 | 0.7539 | 9.2579 | 9.0718 | 0.1861 |
| PreferredBookSpread | 4799 | 4799 | NA | NA | NA | NA | 9.0718 | 9.0718 | 0.0000 |

## Fold Metrics

| model | holdout_season | n_games | n_lined | MAE_all | RMSE_all | MedAE_all | WinAcc_all | MAE_lined | BookMAE_lined | DeltaVsBook_MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CurrentMLP | 2026 | 5356 | 4799 | 10.1486 | 13.3133 | 8.0991 | 0.7539 | 9.2579 | 9.0718 | 0.1861 |
| HistGradientBoosting | 2026 | 5356 | 4799 | 9.8856 | 12.7920 | 8.0389 | 0.7549 | 9.2156 | 9.0718 | 0.1438 |
| HomeMarginMean | 2026 | 5356 | 4799 | 14.2815 | 19.0967 | 11.3969 | 0.6744 | 12.0494 | 9.0718 | 2.9777 |
| LightGBM | 2026 | 5356 | 4799 | 9.9317 | 12.8116 | 8.0882 | 0.7511 | 9.2882 | 9.0718 | 0.2164 |
| PreferredBookSpread | 2026 | 4799 | 4799 | NA | NA | NA | NA | 9.0718 | 9.0718 | 0.0000 |
| Ridge | 2026 | 5356 | 4799 | 11.5242 | 15.7029 | 8.7742 | 0.7407 | 9.7262 | 9.0718 | 0.6544 |
