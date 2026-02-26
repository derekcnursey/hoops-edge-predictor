# Feature Importance & Error Analysis — Season 2025

## A1: Permutation Importance

Baseline MAE on 2025 holdout: **10.1316**

Each feature shuffled 10 times (seed=42). Ranked by mean MAE increase.

### Group 1: Efficiency Metrics (features 0-10)

| Rank | Feature | Mean MAE Increase | Std |
|------|---------|-------------------|-----|
| 1 | away_team_adj_oe | +2.1960 | 0.0543 |
| 2 | home_team_adj_de | +0.9233 | 0.0349 |
| 3 | away_team_adj_de | +0.7740 | 0.0404 |
| 4 | home_team_BARTHAG | +0.7074 | 0.0480 |
| 5 | away_team_BARTHAG | +0.6913 | 0.0341 |
| 6 | home_team_adj_oe | +0.6913 | 0.0370 |
| 7 | away_team_adj_pace | +0.1915 | 0.0136 |
| 8 | home_team_adj_pace | +0.0936 | 0.0077 |
| 9 | neutral_site | +0.0732 | 0.0166 |
| 10 | home_team_home | +0.0409 | 0.0139 |
| 11 | away_team_home | +0.0000 | 0.0000 | *

### Group 2: Rolling Four-Factors (features 11-36)

| Rank | Feature | Mean MAE Increase | Std |
|------|---------|-------------------|-----|
| 1 | away_eff_fg_pct | +0.7027 | 0.0387 |
| 2 | away_def_3p_pct | +0.3416 | 0.0163 |
| 3 | away_def_eff_fg_pct | +0.3343 | 0.0218 |
| 4 | home_def_eff_fg_pct | +0.2180 | 0.0174 |
| 5 | home_3p_pct | +0.1982 | 0.0079 |
| 6 | away_off_rebound_pct | +0.1896 | 0.0232 |
| 7 | away_ft_pct | +0.1407 | 0.0113 |
| 8 | away_3p_pct | +0.1287 | 0.0274 |
| 9 | away_def_rebound_pct | +0.1233 | 0.0173 |
| 10 | away_3pt_rate | +0.0895 | 0.0115 |
| 11 | home_eff_fg_pct | +0.0552 | 0.0126 |
| 12 | home_off_rebound_pct | +0.0519 | 0.0104 |
| 13 | home_ft_rate | +0.0416 | 0.0084 |
| 14 | home_def_off_rebound_pct | +0.0408 | 0.0162 |
| 15 | home_ft_pct | +0.0353 | 0.0103 |
| 16 | home_3pt_rate | +0.0300 | 0.0121 |
| 17 | home_def_3p_pct | +0.0232 | 0.0063 |
| 18 | home_def_3pt_rate | +0.0192 | 0.0080 |
| 19 | away_def_ft_rate | +0.0180 | 0.0069 |
| 20 | home_def_def_rebound_pct | +0.0149 | 0.0074 |
| 21 | home_opp_ft_rate | +0.0093 | 0.0033 |
| 22 | away_def_3pt_rate | +0.0054 | 0.0096 |
| 23 | away_ft_rate | +0.0044 | 0.0060 |
| 24 | away_def_off_rebound_pct | +0.0041 | 0.0118 |
| 25 | away_def_def_rebound_pct | -0.0031 | 0.0034 | *
| 26 | home_def_rebound_pct | -0.0065 | 0.0068 | *

### Top 10 Overall

| Rank | Feature | Mean MAE Increase |
|------|---------|-------------------|
| 1 | away_team_adj_oe | +2.1960 |
| 2 | home_team_adj_de | +0.9233 |
| 3 | away_team_adj_de | +0.7740 |
| 4 | home_team_BARTHAG | +0.7074 |
| 5 | away_eff_fg_pct | +0.7027 |
| 6 | away_team_BARTHAG | +0.6913 |
| 7 | home_team_adj_oe | +0.6913 |
| 8 | away_def_3p_pct | +0.3416 |
| 9 | away_def_eff_fg_pct | +0.3343 |
| 10 | home_def_eff_fg_pct | +0.2180 |

**Zero/negative importance features (3):** away_team_home, away_def_def_rebound_pct, home_def_rebound_pct


## A2: Error Analysis

Total games with book spread: 5440

- Model 5+ pts worse than book: **682** games
- Model 5+ pts better than book: **337** games

### Monthly Pattern (Model 5+ pts worse)

| Month | Count | % of Model-Worse Games |
|-------|-------|------------------------|
| 2024-11 | 380 | 55.7% |
| 2024-12 | 155 | 22.7% |
| 2025-01 | 81 | 11.9% |
| 2025-02 | 35 | 5.1% |
| 2025-03 | 29 | 4.3% |
| 2025-04 | 2 | 0.3% |

### Team Quality Pattern (Model 5+ pts worse)

| Home Tier | Count |
|-----------|-------|
| 50_150 | 223 |
| top_50 | 164 |
| 250_plus | 156 |
| 150_250 | 139 |

### Worst 20 Teams by Model MAE

| TeamId | MAE | Games |
|--------|-----|-------|
| 118 | 15.54 | 34 |
| 175 | 14.56 | 28 |
| 41 | 14.39 | 27 |
| 192 | 13.93 | 26 |
| 60 | 13.56 | 25 |
| 202 | 13.46 | 27 |
| 203 | 13.42 | 28 |
| 351 | 13.11 | 30 |
| 247 | 13.01 | 28 |
| 62 | 12.92 | 24 |
| 310 | 12.83 | 31 |
| 261 | 12.71 | 33 |
| 317 | 12.69 | 26 |
| 141 | 12.67 | 26 |
| 72 | 12.64 | 37 |
| 18 | 12.61 | 35 |
| 188 | 12.41 | 33 |
| 5 | 12.25 | 36 |
| 228 | 12.24 | 31 |
| 319 | 12.16 | 25 |

### Error Distribution

| Metric | Model | Book |
|--------|-------|------|
| MAE | 9.62 | 8.76 |
| Median AE | 8.01 | 7.50 |
| Std Error | 12.30 | 11.16 |


## A3: Book Intelligence

Disagreement = book_spread_home - model_spread_home

Games analyzed: 5440

### Linear Regression of Disagreement on 37 Features

R-squared: **0.1358**

Top 10 coefficients (by absolute value):

| Feature | Coefficient |
|---------|-------------|
| away_def_rebound_pct | +12.7773 |
| home_off_rebound_pct | -12.3392 |
| home_def_off_rebound_pct | -11.6009 |
| away_off_rebound_pct | +10.2302 |
| away_eff_fg_pct | +8.9074 |
| away_3p_pct | -6.8631 |
| away_def_eff_fg_pct | -5.9304 |
| home_def_rebound_pct | -4.9069 |
| home_def_3pt_rate | +4.3919 |
| home_def_def_rebound_pct | -4.1685 |

### Candidate Features Not in Current Model

Correlation with book-model disagreement:

| Candidate | Correlation |
|-----------|-------------|
| home_rest_days | +0.0304 |
| away_rest_days | -0.0073 |
| rest_advantage | +0.0381 |
| season_progress | -0.0086 |

### R-squared with Candidates Added

- Base (37 features): R2 = 0.1358
- Expanded (37 + 4 candidates): R2 = 0.1427
- Lift: +0.0069