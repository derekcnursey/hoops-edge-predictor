# Sigma Calibration Study

- Benchmark dir: `artifacts/benchmarks/canonical_walkforward_v2_lgb_repaired_lines`
- Lined holdout rows: `32829`
- Evaluation seasons: `[2020, 2022, 2023, 2024, 2025]`
- Winner family: `cap`
- Winner selected transform: `cap_17.00`

## Pooled Results

family  gaussian_nll  cover_logloss  cover_brier  mean_sigma  std_sigma  mean_abs_z     cov1     cov2  pick_prob_mean  pick_prob_edge_mean  pick_prob_85_plus_roi  edge_15_plus_roi  top100_roi  top200_roi  top500_roi  top100_winrate  top200_winrate  top500_winrate  top100_avg_prob  top200_avg_prob  top500_avg_prob  top200_overlap_vs_raw  top200_churn_vs_raw  season_wins_vs_raw_top200_roi
   cap      3.901993       0.711593     0.257956   12.587120   1.608042    0.752602 0.720848 0.954982        0.578296             0.054486               0.197824          0.052480    0.078909    0.044455    0.023927        0.565959        0.547329        0.536486         0.847499         0.800486         0.737226                  185.8                 28.4                              2
 const      3.900822       0.707434     0.256201   14.000000   0.000000    0.651328 0.782750 0.980718        0.569526             0.045716               0.156610          0.045742    0.088545    0.042636    0.033855        0.571271        0.546496        0.541729         0.834699         0.785753         0.720189                  156.4                 87.2                              2
shrink      3.898265       0.707717     0.256372   13.769597   0.803292    0.667894 0.769699 0.977622        0.570745             0.046936               0.125790          0.036093    0.090364    0.041636    0.033091        0.571773        0.545983        0.541305         0.831216         0.783571         0.719960                  170.4                 59.2                              2
   raw      3.939575       0.710668     0.257559   13.793886   2.698239    0.727516 0.735088 0.956065        0.575375             0.051565               0.186269          0.048982    0.048364    0.025455    0.026255        0.549588        0.537214        0.537704         0.835181         0.789175         0.727921                  200.0                  0.0                              0
 scale      3.939575       0.710668     0.257559   13.793886   2.698239    0.727516 0.735088 0.956065        0.575375             0.051565               0.186269          0.048982    0.048364    0.025455    0.026255        0.549588        0.537214        0.537704         0.835181         0.789175         0.727921                  200.0                  0.0                              0
affine      3.980068       0.703377     0.254622   16.645829   2.503356    0.581728 0.825585 0.986241        0.561332             0.037523               0.151831          0.063663    0.063636    0.021636    0.037709        0.557588        0.535323        0.543762         0.789525         0.745790         0.689989                  190.4                 19.2                              2

## Final Selected Transforms

family            selected_label  sigma_const  cap_max  scale  affine_a  affine_b  shrink_alpha  shrink_target
   raw                       raw          NaN      NaN    NaN       NaN       NaN           NaN            NaN
 const               const_14.00         14.0      NaN    NaN       NaN       NaN           NaN            NaN
   cap                 cap_17.00          NaN     17.0    NaN       NaN       NaN           NaN            NaN
 scale               scale_1.000          NaN      NaN    1.0       NaN       NaN           NaN            NaN
affine       affine_a3.00_b1.000          NaN      NaN    NaN       3.0       1.0           NaN            NaN
shrink shrink_alpha0.25_to_14.00          NaN      NaN    NaN       NaN       NaN          0.25           14.0