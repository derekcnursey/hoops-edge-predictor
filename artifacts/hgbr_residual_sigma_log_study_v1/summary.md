# HGBR Residual Sigma Log Study v1

- Benchmark dir: `artifacts/benchmarks/canonical_walkforward_v2_lgb`
- Residual dataset rows (lined): `32851`
- Evaluation seasons: `[2020, 2022, 2023, 2024, 2025]`
- Log residual eps: `0.25`
- Mu held fixed to canonical HistGradientBoosting holdout predictions
- Residual model: `HistGradientBoostingRegressor`
- Compare scalar and affine calibration on prior seasons only

## Pooled Metrics

                  option     n_games  gaussian_nll  cover_logloss  cover_brier  mean_sigma  std_sigma  mean_abs_z     cov1     cov2  top100_roi  top200_roi  top500_roi  top100_winrate  top200_winrate  top500_winrate  top100_avg_prob  top200_avg_prob  top500_avg_prob
hgbr_residual_log_scalar 5466.699282      4.043649       0.698536     0.252523   18.834241   1.290769    0.486916 0.895012 0.996997    0.182513    0.118292    0.057093        0.621257        0.587040        0.554227         0.785875         0.737608         0.677114
hgbr_residual_log_affine 5466.699282      4.082145       0.698250     0.252383   19.897746   1.035042    0.469366 0.903472 0.996411    0.176555    0.116220    0.058682        0.618481        0.585940        0.555067         0.777011         0.730158         0.671387
      best_posthoc_sigma 5466.699282      3.900752       0.705969     0.255623   13.935843   0.670800    0.658421 0.776586 0.978981    0.189359    0.108669    0.059535        0.624843        0.581629        0.555511         0.848479         0.796984         0.726758
       constant_14_sigma 5466.699282      3.900658       0.706027     0.255602   14.000000   0.000000    0.651213 0.782774 0.980738    0.176525    0.105781    0.061201        0.618384        0.580201        0.556374         0.852508         0.800586         0.728822
           raw_mlp_sigma 5466.699282      3.937853       0.709385     0.256998   13.743372   2.683202    0.728944 0.734144 0.955947    0.153880    0.093693    0.057015        0.605571        0.573616        0.554221         0.853973         0.803429         0.736311

## Selected Calibrations

 season                fit_mode  scalar_c  affine_a  affine_b            train_seasons
   2020 in_sample_single_season       2.5       0.0       2.0                     2019
   2022              season_oof       2.5       6.0       2.0                2019,2020
   2023              season_oof       2.5       6.0       2.0           2019,2020,2022
   2024              season_oof       2.5       6.0       2.0      2019,2020,2022,2023
   2025              season_oof       2.5       6.0       2.0 2019,2020,2022,2023,2024