# Sigma Calibration Study

- Benchmark dir: `artifacts/benchmarks/canonical_walkforward_v2_lgb`
- Lined holdout rows: `32851`
- Evaluation seasons: `[2020, 2022, 2023, 2024, 2025]`
- Winner family: `shrink`
- Winner selected transform: `shrink_alpha0.25_to_14.00`

## Pooled Results

family  gaussian_nll  cover_logloss  cover_brier  mean_sigma  std_sigma  mean_abs_z     cov1     cov2  pick_prob_mean  pick_prob_edge_mean  pick_prob_85_plus_roi  edge_15_plus_roi  top100_roi  top200_roi  top500_roi  top100_winrate  top200_winrate  top500_winrate  top100_avg_prob  top200_avg_prob  top500_avg_prob  top200_overlap_vs_raw  top200_churn_vs_raw  season_wins_vs_raw_top200_roi
shrink      3.898568       0.706334     0.255780   13.793433   0.780861    0.666445 0.770621 0.977853        0.571645             0.047836               0.202282          0.065445    0.184000    0.106636    0.059164        0.622083        0.580574        0.555314         0.849862         0.798421         0.728320                  172.2                 55.6                              2
 const      3.900726       0.706097     0.255632   14.000000   0.000000    0.651220 0.782769 0.980734        0.570543             0.046733               0.245229          0.077194    0.174545    0.103818    0.060291        0.617374        0.579182        0.555902         0.852478         0.800555         0.728787                  160.8                 78.4                              2
   cap      3.901816       0.710269     0.257385   12.587452   1.608068    0.752425 0.720938 0.955024        0.579316             0.055506               0.266415          0.082193    0.176364    0.102818    0.052291        0.618250        0.578518        0.551706         0.865815         0.815105         0.745612                  188.0                 24.0                              2
affine      3.979973       0.702139     0.254084   16.646512   2.503382    0.581581 0.825628 0.986254        0.562224             0.038414               0.245502          0.101858    0.155273    0.094273    0.066073        0.606268        0.574076        0.559008         0.808368         0.759705         0.697645                  191.4                 17.2                              2
   raw      3.939401       0.709397     0.257001   13.794724   2.698370    0.727318 0.735210 0.956106        0.576334             0.052525               0.257903          0.089573    0.151455    0.092364    0.056145        0.604330        0.572932        0.553770         0.853637         0.802993         0.735876                  200.0                  0.0                              0
 scale      3.939401       0.709397     0.257001   13.794724   2.698370    0.727318 0.735210 0.956106        0.576334             0.052525               0.257903          0.089573    0.151455    0.092364    0.056145        0.604330        0.572932        0.553770         0.853637         0.802993         0.735876                  200.0                  0.0                              0

## Final Selected Transforms

family            selected_label  sigma_const  cap_max  scale  affine_a  affine_b  shrink_alpha  shrink_target
   raw                       raw          NaN      NaN    NaN       NaN       NaN           NaN            NaN
 const               const_14.00         14.0      NaN    NaN       NaN       NaN           NaN            NaN
   cap                 cap_17.00          NaN     17.0    NaN       NaN       NaN           NaN            NaN
 scale               scale_1.000          NaN      NaN    1.0       NaN       NaN           NaN            NaN
affine       affine_a3.00_b1.000          NaN      NaN    NaN       3.0       1.0           NaN            NaN
shrink shrink_alpha0.25_to_14.00          NaN      NaN    NaN       NaN       NaN          0.25           14.0