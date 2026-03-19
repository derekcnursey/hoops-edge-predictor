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
| HomeMarginMean | 6292 | 5437 | 13.3450 | 17.7966 | 10.6995 | 0.6712 | 11.5445 | 8.7593 | 2.7852 |
| Ridge | 6292 | 5437 | 11.1038 | 14.9302 | 8.6690 | 0.7374 | 9.6329 | 8.7593 | 0.8735 |
| HistGradientBoosting | 6292 | 5437 | 9.6341 | 12.4389 | 7.9011 | 0.7414 | 9.1144 | 8.7593 | 0.3551 |
| HistGradientBoostingMatchup | 6292 | 5437 | 9.6626 | 12.4829 | 7.9100 | 0.7436 | 9.1280 | 8.7593 | 0.3686 |
| LightGBM | 6292 | 5437 | 9.6785 | 12.4997 | 7.9771 | 0.7405 | 9.1707 | 8.7593 | 0.4114 |
| CurrentMLP | 6292 | 5437 | 9.9355 | 12.9326 | 7.9975 | 0.7481 | 9.2343 | 8.7593 | 0.4750 |
| PreferredBookSpread | 5437 | 5437 | NA | NA | NA | NA | 8.7593 | 8.7593 | 0.0000 |

## Fold Metrics

| model | holdout_season | n_games | n_lined | MAE_all | RMSE_all | MedAE_all | WinAcc_all | MAE_lined | BookMAE_lined | DeltaVsBook_MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CurrentMLP | 2025 | 6292 | 5437 | 9.9355 | 12.9326 | 7.9975 | 0.7481 | 9.2343 | 8.7593 | 0.4750 |
| HistGradientBoosting | 2025 | 6292 | 5437 | 9.6341 | 12.4389 | 7.9011 | 0.7414 | 9.1144 | 8.7593 | 0.3551 |
| HistGradientBoostingMatchup | 2025 | 6292 | 5437 | 9.6626 | 12.4829 | 7.9100 | 0.7436 | 9.1280 | 8.7593 | 0.3686 |
| HomeMarginMean | 2025 | 6292 | 5437 | 13.3450 | 17.7966 | 10.6995 | 0.6712 | 11.5445 | 8.7593 | 2.7852 |
| LightGBM | 2025 | 6292 | 5437 | 9.6785 | 12.4997 | 7.9771 | 0.7405 | 9.1707 | 8.7593 | 0.4114 |
| PreferredBookSpread | 2025 | 5437 | 5437 | NA | NA | NA | NA | 8.7593 | 8.7593 | 0.0000 |
| Ridge | 2025 | 6292 | 5437 | 11.1038 | 14.9302 | 8.6690 | 0.7374 | 9.6329 | 8.7593 | 0.8735 |
