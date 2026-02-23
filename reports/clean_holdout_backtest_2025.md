# Clean Holdout Backtest — Season 2025

## Methodology

**Contaminated pipeline (old):**
- Rating params: half_life=30, margin_cap=15, HCA=4.1201 (tuned with 2025 in view)
- ML model trained on: seasons 2015-2025 (includes 2025 test data)
- Tested on: 2025

**Clean holdout pipeline:**
- Rating params: half_life=60, margin_cap=15, HCA=4.0266 (tuned on 2016-2023 ONLY)
- ML model trained on: seasons 2015-2024 (2025 completely unseen)
- Tested on: 2025 (true out-of-sample)

## Step A: Holdout-Tuned Parameters

Grid search over half_life=[15,20,30,45,60] and margin_cap=[10,15,20,None] using only seasons 2016-2023.

Best parameters by aggregate spread prediction MAE:
- half_life: 60 (was 30)
- margin_cap: 15 (unchanged)
- HCA OE: 4.0266 (was 4.1201)
- HCA DE: 4.0266 (was 4.1201)
- Holdout MAE: 9.1454

Top 5 combos:
1. hl=60 cap=15 MAE=9.1454
2. hl=45 cap=15 MAE=9.1943
3. hl=60 cap=20 MAE=9.2845
4. hl=60 cap=10 MAE=9.3104
5. hl=45 cap=10 MAE=9.3240

## Step D: Contaminated vs Clean Comparison

| Metric | Contaminated (old) | Clean Holdout | Difference |
|--------|-------------------|---------------|------------|
| Model MAE | 9.78 | 9.87 | +0.09 (worse) |
| Book MAE | 8.76 | 8.76 | 0.00 |
| Unfiltered ROI@3 | -1.6% | -4.5% | -2.9% |
| Unfiltered ROI@5 | -1.7% | -5.3% | -3.6% |
| Sigma<med ROI@3 | +3.4% | -2.6% | -6.0% |
| Sigma<med ROI@5 | +7.0% | +1.5% | -5.5% |
| Sigma<p25 ROI@3 | +9.9% | -6.5% | -16.4% |
| Sigma<p25 ROI@5 | +14.8% | +2.5% | -12.3% |
| Sigma<p25 ROI@7 | +26.4% (77 bets) | +15.3% (53 bets) | -11.1% |
| Sigma<p25 WR@5 | 60.1% | 53.7% | -6.4% |

## Clean Holdout Full ROI Table (2025)

### Unfiltered
| Threshold | Bets | Wins | Losses | Win Rate | ROI |
|-----------|------|------|--------|----------|-----|
| 3 | 2378 | 1190 | 1188 | 50.0% | -4.5% |
| 5 | 1334 | 662 | 672 | 49.6% | -5.3% |
| 7 | 869 | 436 | 433 | 50.2% | -4.2% |

### Sigma < median (11.9)
| Threshold | Bets | Wins | Losses | Win Rate | ROI |
|-----------|------|------|--------|----------|-----|
| 3 | 862 | 440 | 422 | 51.0% | -2.6% |
| 5 | 333 | 177 | 156 | 53.2% | +1.5% |
| 7 | 161 | 90 | 71 | 55.9% | +6.7% |

### Sigma < p25 (11.4)
| Threshold | Bets | Wins | Losses | Win Rate | ROI |
|-----------|------|------|--------|----------|-----|
| 3 | 396 | 194 | 202 | 49.0% | -6.5% |
| 5 | 136 | 73 | 63 | 53.7% | +2.5% |
| 7 | 53 | 32 | 21 | 60.4% | +15.3% |

## Calibration (Clean Holdout)

| Predicted Prob | Games | Actual Win Rate | Calibration |
|----------------|-------|-----------------|-------------|
| > 0.7 | 2749 | 83.1% | Good |
| 0.6 - 0.7 | 1374 | 69.4% | Good |
| 0.5 - 0.6 | 911 | 56.3% | Good |
| < 0.5 | 1264 | 37.3% | Good |

## Key Findings

1. **Contamination inflated ROI significantly.** The sigma<p25 @3 dropped from +9.9% to -6.5%, and @5 from +14.8% to +2.5%. This confirms the contaminated results overstated the model's ATS edge.

2. **The model still has some edge at high thresholds with sigma filtering.** Sigma<p25 @7 still shows +15.3% ROI (53 bets, 60.4% win rate), and sigma<med @7 shows +6.7% (161 bets, 55.9%). But sample sizes are small.

3. **MAE barely changed.** 9.78 → 9.87, only 0.09 worse. The ML model still beats the raw ratings baseline (10.85) by ~1 point. The model's spread predictions are genuinely better than raw ratings.

4. **Calibration actually improved.** The clean model shows "Good" calibration across all probability buckets, vs the contaminated model which was "Off" in two buckets.

5. **The high-threshold, low-sigma strategy needs more seasons of clean validation** before deploying with real money. The 53-bet sample at threshold 7 is too small to be conclusive.

6. **Rating parameter change (hl=30→60) matters.** The longer half_life weights the full season more evenly rather than heavily recency-weighting. This is a more conservative but robust approach.
