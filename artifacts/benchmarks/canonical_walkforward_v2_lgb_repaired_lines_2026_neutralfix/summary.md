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
| Ridge | 5356 | 4799 | 11.5283 | 15.7064 | 8.7702 | 0.7397 | 9.7269 | 9.0718 | 0.6551 |
| HistGradientBoosting | 5356 | 4799 | 9.8791 | 12.7793 | 8.0221 | 0.7545 | 9.2183 | 9.0718 | 0.1465 |
| LightGBM | 5356 | 4799 | 9.9158 | 12.7951 | 8.0676 | 0.7521 | 9.2812 | 9.0718 | 0.2094 |
| CurrentMLP | 5356 | 4799 | 10.1521 | 13.3183 | 8.1123 | 0.7534 | 9.2646 | 9.0718 | 0.1928 |
| PreferredBookSpread | 4799 | 4799 | NA | NA | NA | NA | 9.0718 | 9.0718 | 0.0000 |

## Fold Metrics

| model | holdout_season | n_games | n_lined | MAE_all | RMSE_all | MedAE_all | WinAcc_all | MAE_lined | BookMAE_lined | DeltaVsBook_MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CurrentMLP | 2026 | 5356 | 4799 | 10.1521 | 13.3183 | 8.1123 | 0.7534 | 9.2646 | 9.0718 | 0.1928 |
| HistGradientBoosting | 2026 | 5356 | 4799 | 9.8791 | 12.7793 | 8.0221 | 0.7545 | 9.2183 | 9.0718 | 0.1465 |
| HomeMarginMean | 2026 | 5356 | 4799 | 14.2815 | 19.0967 | 11.3969 | 0.6744 | 12.0494 | 9.0718 | 2.9777 |
| LightGBM | 2026 | 5356 | 4799 | 9.9158 | 12.7951 | 8.0676 | 0.7521 | 9.2812 | 9.0718 | 0.2094 |
| PreferredBookSpread | 2026 | 4799 | 4799 | NA | NA | NA | NA | 9.0718 | 9.0718 | 0.0000 |
| Ridge | 2026 | 5356 | 4799 | 11.5283 | 15.7064 | 8.7702 | 0.7397 | 9.7269 | 9.0718 | 0.6551 |
