# HGBR Residual Sigma Study v1

- Benchmark dir: `artifacts/benchmarks/canonical_walkforward_v2_lgb`
- Residual dataset rows (lined): `32851`
- Evaluation seasons: `[2020, 2022, 2023, 2024, 2025]`
- Mu held fixed to canonical HistGradientBoosting holdout predictions
- Residual target: `abs(actual_margin - mu_hgbr_oof)`
- Residual model: `HistGradientBoostingRegressor`
- Scalar calibration objective: cover-event log loss on prior seasons only

## Pooled Metrics

             option     n_games  gaussian_nll  cover_logloss  cover_brier  mean_sigma  std_sigma  mean_abs_z     cov1     cov2  top100_roi  top200_roi  top500_roi  top100_winrate  top200_winrate  top500_winrate  top100_avg_prob  top200_avg_prob  top500_avg_prob
 best_posthoc_sigma 5466.699282      3.900752       0.705969     0.255623   13.935843   0.670800    0.658421 0.776586 0.978981    0.189359    0.108669    0.059535        0.624843        0.581629        0.555511         0.848479         0.796984         0.726758
  constant_14_sigma 5466.699282      3.900658       0.706027     0.255602   14.000000   0.000000    0.651213 0.782774 0.980738    0.176525    0.105781    0.061201        0.618384        0.580201        0.556374         0.852508         0.800586         0.728822
hgbr_residual_sigma 5466.699282      4.053060       0.698400     0.252461   19.127105   1.459382    0.479782 0.899517 0.997400    0.178388    0.100466    0.068772        0.619097        0.577343        0.560416         0.781707         0.733834         0.674223
      raw_mlp_sigma 5466.699282      3.937853       0.709385     0.256998   13.743372   2.683202    0.728944 0.734144 0.955947    0.153880    0.093693    0.057015        0.605571        0.573616        0.554221         0.853973         0.803429         0.736311

## Selected Scalars

 season  scalar_c         scalar_fit_mode            train_seasons
   2020       2.5 in_sample_single_season                     2019
   2022       2.5              season_oof                2019,2020
   2023       2.5              season_oof           2019,2020,2022
   2024       2.5              season_oof      2019,2020,2022,2023
   2025       2.5              season_oof 2019,2020,2022,2023,2024