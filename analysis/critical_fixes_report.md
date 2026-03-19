# Critical Fixes Report — Post-Fix Walk-Forward
**Date**: 2026-03-03
## Bugs Fixed
1. **CRITICAL-1**: NaN imputation mismatch — `train_production.py` used `nan_to_num(0.0)`, now uses `impute_column_means()` matching inference.
2. **CRITICAL-3**: Rolling average fallback leaked future data — `_get_asof_rolling()` now returns `{}` when no prior data exists.
3. **HIGH-4**: `no_garbage` default inconsistency — unified to `config.NO_GARBAGE=True` across `features.py`, `dataset.py`, `cli.py`.

## Walk-Forward Results (53-feat, no_garbage, post-fix)
| Year | MAE | RMSE | σ | ATS@5% | WR | ROI | Home% |
|------|-----|------|---|--------|-----|-----|-------|
| 2019 | 9.53 | 12.49 | 11.8 | 1351W-1345L | 50.1% | -4.2% | 62% |
| 2020 | 9.60 | 12.53 | 11.5 | 1383W-1335L | 50.9% | -2.8% | 54% |
| 2021 * | 10.75 | 14.04 | 12.8 | 1281W-1315L | 49.3% | -5.7% | 65% |
| 2022 | 9.42 | 12.48 | 12.3 | 1416W-1228L | 53.6% | +2.2% | 52% |
| 2023 | 9.56 | 12.46 | 11.6 | 1547W-1401L | 52.5% | +0.2% | 46% |
| 2024 | 9.88 | 13.06 | 11.6 | 1361W-1272L | 51.7% | -1.3% | 51% |
| 2025 | 9.68 | 12.91 | 11.4 | 1393W-1323L | 51.3% | -2.1% | 45% |
| **AVG** | **9.61** | **12.65** | **11.7** | **8451W-7904L** | **51.7%** | **-1.3%** | **52%** |
*(excluding 2021)*

## Pre-Fix Baseline (previous session)
| Metric | Pre-Fix | Post-Fix | Change |
|--------|---------|----------|--------|
| MAE (avg excl 2021) | 9.62 | 9.61 | -0.01 |
| RMSE (avg excl 2021) | ~12.7 | 12.65 | — |
| ATS WR @5% | ~51.7% | 51.7% | -0.0pp |
| ATS ROI @5% | ~-1.3% | -1.3% | -0.0pp |

## Monthly ATS (edge >= 5%, excl 2021)
| Month | N | W | L | WR | ROI | Home% |
|-------|---|---|---|----|-----|-------|
| Dec | 4353 | 2265 | 2041 | 52.6% | +0.4% | 50% |
| Jan | 5290 | 2732 | 2494 | 52.3% | -0.2% | 56% |
| Feb | 4580 | 2272 | 2218 | 50.6% | -3.3% | 51% |
| Mar | 2323 | 1167 | 1135 | 50.7% | -3.2% | 45% |
| Apr | 31 | 15 | 16 | 48.4% | -7.6% | 61% |

## Key Question: Does December Edge Survive?
Pre-fix December: 53.2% WR, +1.6% ROI (edge leak analysis)

Post-fix December: 52.6% WR, +0.4% ROI

December edge appears **genuine** — survives the leakage fix.
