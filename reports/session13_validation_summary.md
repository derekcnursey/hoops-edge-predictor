# Session 13: Validation Suite Summary

Date: 2026-02-27 | Model: C2-V2 (384→256, d=0.20, lr=3e-3, Gaussian)

## Verdict: Model is a good predictor. No profitable betting strategy survives walk-forward.

## Test Results

| # | Test | Result | Key Finding |
|---|------|--------|-------------|
| 1 | Walk-Forward (7 years) | **FAIL for betting** | Pooled ROI: -1.0% (strategy c), +0.8% (12% unfilt). 2021 wipes out gains. |
| 2 | Sigma Ablation | **PARTIAL** | Sigma adds +4.4% vs shuffled for σ-filtered strategy, but unfiltered is worse with real σ |
| 3 | Vig Sensitivity | PASS | σ12-16 strategy survives to -115 juice (+2.4%) — but only in-sample |
| 4 | Bootstrap CI | PASS (in-sample) | σ12-16: P(ROI>0)=96.2%, 90% CI [+0.9%, +18.1%] |
| 5 | Drawdown | PASS | Max DD=10.7 units, longest losing streak=6 |
| 6 | Bet Clustering | PASS | ICC=0.000, bets are independent, block bootstrap P(ROI>0)=98.7% |
| 7 | Feature Ablation | PASS | 48/50 features contribute. BARTHAG is #1 (+0.75 MAE). |
| 8 | HP Stability | PASS | BS-MAE range=0.050 across 8 perturbations |
| 9 | Calibration Slices | PASS | All quintile ratios within 0.70-1.30 across all slices |
| 10 | Baselines | PASS | Book baseline=0 bets (correct). Model beats regression-to-mean. |
| 11 | Line Staleness | INFO | Using Bovada/DK closing lines. Has spreadOpen column for future analysis. |
| 12 | Bet Profiling | INFO | Away dogs +20.4% ROI at |book|>15 — but in-sample only |

## Walk-Forward Detail (Test 1)

| Year | BS-MAE | c) σ12-16 ROI | b) 12% unfilt ROI |
|------|--------|---------------|-------------------|
| 2019 | 9.023 | +6.6% | +3.1% |
| 2020 | 9.305 | +15.9% | +3.5% |
| 2021 | 10.116 | **-9.3%** | -4.5% |
| 2022 | 8.890 | +0.9% | +2.8% |
| 2023 | 9.086 | +0.1% | -0.3% |
| 2024 | 9.165 | -1.0% | +2.9% |
| 2025 | 8.977 | +0.2% | +2.9% |
| **Pooled** | **9.202** | **-1.0%** | **+0.8%** |

## Targeted Walk-Forward (Away Dogs / Big Spreads)

| Strategy | Pooled ROI | Positive Yrs | Qualifies? |
|----------|-----------|-------------|-----------|
| 10% away |bk|>10 | +1.7% | 2/7 | No |
| 7% |bk|>15 | -3.4% | 2/7 | No |
| 10% |bk|>10 | -0.1% | 2/7 | No |
| 12% away |bk|>10 | -0.2% | 2/7 | No |
| 10% away dog (all) | +0.7% | 2/7 | No |
| 12% unfiltered | -2.0% | 3/7 | No |

Criteria: 5/7 positive years AND pooled ROI > 3%. **No strategy qualifies.**

## Conclusions

1. **The model predicts well**: BS-MAE 9.13 beats book baseline, stable across HP perturbations, well-calibrated sigma.
2. **The sigma is real**: Calibration passes all slice tests, features are meaningful, architecture is at a stable optimum.
3. **No betting edge survives out-of-sample**: Every strategy tested (8 variants × 7 years) fails walk-forward. The 2026 val set ROI of +9.8% was in-sample luck.
4. **2021 is the canary**: BS-MAE=10.12 (worst year), all strategies deeply negative. COVID-era data may be poisoning early training.
5. **The model's value is in prediction, not betting**: Use it for game analysis, bracket picks, or as one input to a larger system — not as a standalone ATS betting system.
