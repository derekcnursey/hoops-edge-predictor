# Torvik Hybrid Evaluation Report

Generated: 2026-03-05T14:46:55

## Configuration
- Efficiency source: Torvik daily_data from S3
- PBP features: Unchanged (from gold layer pipeline)
- Holdout years: 2019, 2020, 2022, 2023, 2024, 2025
- Excluded seasons: [2021]
- Training date filter: >= 12-01
- Edge threshold: 5%
- Architecture: MLPRegressor(384/256, d=0.2) + MLPClassifier(384, d=0.2)

## Comparison: A_baseline (gold layer) vs B_torvik (Torvik efficiencies)

| Year | A_MAE | B_MAE | ΔMAE | A_ROI | B_ROI | ΔROI | B_WR | B_W-L |
|------|-------|-------|------|-------|-------|------|------|-------|
| 2019 | 9.53 | 9.61 | +0.08 | -4.2% | -2.4% | +1.8% | 51.1% | 2067-1975 |
| 2020 | 9.60 | 9.63 | +0.03 | -2.8% | -2.2% | +0.6% | 51.2% | 1995-1898 |
| 2022 | 9.42 | 9.47 | +0.05 | 2.2% | -0.1% | -2.3% | 52.3% | 2041-1861 |
| 2023 | 9.56 | 9.69 | +0.13 | 0.2% | -2.7% | -2.9% | 51.0% | 2147-2064 |
| 2024 | 9.88 | 9.82 | -0.06 | -1.3% | -5.4% | -4.1% | 49.5% | 1915-1951 |
| 2025 | 9.68 | 9.69 | +0.01 | -2.1% | -3.0% | -0.9% | 50.8% | 2033-1967 |
| **AVG** | **9.61** | **9.65** | **+0.04** | **-1.3%** | **-2.6%** | **-1.3%** | | |

## Per-Year Details

### 2019
- Games: 6049 (5543 with book spread)
- MAE: 9.61, RMSE: 12.25, Sigma: 8.600000381469727
- ATS picks (edge >= 5%): 4175
- Record: 2067-1975 (51.1%)
- ROI: -2.4%, Units: -95.9
- Home pick %: 32%
- Monthly ATS breakdown:
  | Month | W-L | WR | ROI |
  |-------|-----|-----|-----|
  | Nov | 428-400 | 51.7% | -1.3% |
  | Dec | 399-335 | 54.4% | 3.8% |
  | Jan | 517-503 | 50.7% | -3.2% |
  | Feb | 448-446 | 50.1% | -4.3% |
  | Mar | 271-287 | 48.6% | -7.3% |
  | Apr | 4-4 | 50.0% | -4.5% |

### 2020
- Games: 5768 (5326 with book spread)
- MAE: 9.63, RMSE: 12.29, Sigma: 8.5
- ATS picks (edge >= 5%): 3971
- Record: 1995-1898 (51.2%)
- ROI: -2.2%, Units: -84.3
- Home pick %: 33%
- Monthly ATS breakdown:
  | Month | W-L | WR | ROI |
  |-------|-----|-----|-----|
  | Nov | 508-427 | 54.3% | 3.7% |
  | Dec | 362-367 | 49.7% | -5.2% |
  | Jan | 502-482 | 51.0% | -2.6% |
  | Feb | 480-489 | 49.5% | -5.4% |
  | Mar | 143-133 | 51.8% | -1.1% |

### 2022
- Games: 5977 (5463 with book spread)
- MAE: 9.47, RMSE: 12.09, Sigma: 9.199999809265137
- ATS picks (edge >= 5%): 4004
- Record: 2041-1861 (52.3%)
- ROI: -0.1%, Units: -5.5
- Home pick %: 35%
- Monthly ATS breakdown:
  | Month | W-L | WR | ROI |
  |-------|-----|-----|-----|
  | Nov | 474-403 | 54.0% | 3.2% |
  | Dec | 355-332 | 51.7% | -1.3% |
  | Jan | 513-446 | 53.5% | 2.1% |
  | Feb | 446-472 | 48.6% | -7.2% |
  | Mar | 252-207 | 54.9% | 4.8% |
  | Apr | 1-1 | 50.0% | -4.5% |

### 2023
- Games: 6228 (5773 with book spread)
- MAE: 9.69, RMSE: 12.38, Sigma: 9.100000381469727
- ATS picks (edge >= 5%): 4214
- Record: 2147-2064 (51.0%)
- ROI: -2.7%, Units: -112.2
- Home pick %: 33%
- Monthly ATS breakdown:
  | Month | W-L | WR | ROI |
  |-------|-----|-----|-----|
  | Nov | 522-450 | 53.7% | 2.5% |
  | Dec | 440-407 | 51.9% | -0.8% |
  | Jan | 501-505 | 49.8% | -4.9% |
  | Feb | 497-494 | 50.2% | -4.3% |
  | Mar | 186-206 | 47.4% | -9.4% |
  | Apr | 1-2 | 33.3% | -36.4% |

### 2024
- Games: 6243 (5309 with book spread)
- MAE: 9.82, RMSE: 12.32, Sigma: 9.199999809265137
- ATS picks (edge >= 5%): 3866
- Record: 1915-1951 (49.5%)
- ROI: -5.4%, Units: -210.1
- Home pick %: 31%
- Monthly ATS breakdown:
  | Month | W-L | WR | ROI |
  |-------|-----|-----|-----|
  | Nov | 413-410 | 50.2% | -4.2% |
  | Dec | 350-356 | 49.6% | -5.4% |
  | Jan | 462-482 | 48.9% | -6.6% |
  | Feb | 428-439 | 49.4% | -5.8% |
  | Mar | 260-262 | 49.8% | -4.9% |
  | Apr | 2-2 | 50.0% | -4.5% |

### 2025
- Games: 6292 (5437 with book spread)
- MAE: 9.69, RMSE: 12.41, Sigma: 9.300000190734863
- ATS picks (edge >= 5%): 4000
- Record: 2033-1967 (50.8%)
- ROI: -3.0%, Units: -118.8
- Home pick %: 27%
- Monthly ATS breakdown:
  | Month | W-L | WR | ROI |
  |-------|-----|-----|-----|
  | Nov | 502-515 | 49.4% | -5.8% |
  | Dec | 345-358 | 49.1% | -6.3% |
  | Jan | 509-464 | 52.3% | -0.1% |
  | Feb | 398-371 | 51.8% | -1.2% |
  | Mar | 271-252 | 51.8% | -1.1% |
  | Apr | 8-7 | 53.3% | 1.8% |

