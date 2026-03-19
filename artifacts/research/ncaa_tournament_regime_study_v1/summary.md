# NCAA Tournament Regime Study

## Promoted Production Path

- Historical/site benchmark bundle: `/Users/dereknursey/Desktop/ml_projects/codex_review/hoops-edge-predictor_codex/artifacts/benchmarks/canonical_walkforward_lgb_l2_blend_repaired_lines_neutralfix`
- Mean path in bundle: `LightGBMRegressionL2Blend` from `/Users/dereknursey/Desktop/ml_projects/codex_review/hoops-edge-predictor_codex/artifacts/research/objective_tail_compression_experiment_v1/LightGBMRegressionL2Blend_predictions.parquet`
- Sigma path in bundle: `CurrentMLP`
- The site history JSONs point at this bundle directly.
- In March, the live mean-path blend is effectively gold-only because `gold_weight_for_start_dates()` saturates to `1.0` well before the NCAA Tournament.

## Diagnosis

- NCAA Tournament lined sample: `328` games across 2019, 2022, 2023, 2024, 2025.
- Current promoted model is short versus the market on the market-favorite scale by `-2.38` points in NCAA games, versus `-0.69` in the regular season.
- NCAA favorite-slope vs market is `0.757` vs `0.940` in the regular season, which is direct compression toward zero.
- Round of 64 is the worst round: mean shortfall `-3.10` points.
- For `15+` market favorites, NCAA shortfall grows to `-5.80` points.
- Market margin MAE in NCAA is `9.367` vs the model's `9.996`.

## Regime Read

- Neutral-site non-conference games also compress, but less cleanly: NCAA shortfall `-2.38` vs neutral non-conf `-2.10`.
- Conference tournaments are materially less short than NCAA: `-1.08`.
- That points to an NCAA / elite mismatch regime more than a generic 'all neutral games' regime.

## Probability Read

- Current model straight-up accuracy in NCAA games: `0.701`.
- Market moneyline straight-up accuracy: `0.704`.
- Current model Brier / logloss: `0.1949` / `0.5729`.
- Market moneyline Brier / logloss: `0.1858` / `0.5482`.
- Interpretation: the model still ranks winners reasonably well, but margin calibration deteriorates more than winner-picking.

## Source-Side Evidence

- Promoted NCAA MAE / shortfall: `10.016` / `-2.38`.
- Torvik LGB NCAA MAE / shortfall: `9.761` / `-1.84`.
- Existing benchmark artifacts therefore already show that the March gold handoff is not the best NCAA source path.
- Separate conference-bridge gold research does not fix NCAA Tournament quality, so the issue is not solved by a small conference-strength bridge alone.

## Candidate Backtest

- Current unchanged (`A`) NCAA MAE: `10.016`.
- Best pooled NCAA candidate in this study: `G_market_blend_tournament_only` at `9.449` MAE.
- Tournament-only market blend (`G`) wins on pooled NCAA margin MAE at `9.449`.
- Separate NCAA model (`C`) improves some vs current (`9.796`) but is less stable and sample-starved.
- Affine tournament calibration (`B`) is smaller and safer, but only modestly improves current (`9.924`).
- Reweighted all-games and broader neutral/postseason retrains did not beat the simpler post-processing paths.

## Recommendation

- Do not ship a standalone NCAA-only model as the main production path yet. The sample is too small, and the simple tournament model only modestly helps.
- Preferred production shape: keep one year-round core model, but add a tournament-only post-processing layer.
- If the goal is the best displayed tournament spread, the evidence supports a tournament-only market-aware blend or market override.
- If the goal is a model-pure spread with minimal market dependence, the next best justified prototype is not a separate NCAA model; it is a tournament-only torvik-heavy mean override or a slower March handoff away from torvik.

## Caveats

- NCAA sample size is only 67 games per completed tournament season.
- Round-level results beyond Round of 64 / Round of 32 are noisy.
- Historical line work uses `fct_lines_repaired_v1`, a hindsight-rebuilt line archive rather than a true time-native close snapshot.

## Calibration Snapshot

```csv
holdout_season,prior_ncaa_rows,affine_slope,affine_intercept,market_blend_alpha
2019,0,1.0,0.0,1.0
2022,67,1.3675153487460598,-0.6611354396183406,0.45
2023,132,1.2384427876409532,-0.2554181661130901,0.23750000000000002
2024,199,1.0571311257331577,0.6067148943373706,0.2
2025,261,1.1714031188555671,0.46516566640092183,0.0
```

