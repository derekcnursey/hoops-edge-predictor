# Feature Selection / Pruning — Season 2025

## Task 1: Permutation Importance (54 features)

Baseline MAE (book-spread games): **9.4054** (5440 games)

| Rank | Feature | MAE Increase | Std |
|------|---------|-------------|-----|
| 1 | away_team_adj_oe | +1.0413 | 0.0452 |
| 2 | away_sos_de | +0.9247 | 0.0458 |
| 3 | home_conf_strength | +0.6126 | 0.0322 |
| 4 | away_eff_fg_pct | +0.5799 | 0.0338 |
| 5 | home_team_adj_oe | +0.4947 | 0.0311 |
| 6 | away_team_BARTHAG | +0.4576 | 0.0314 |
| 7 | away_team_adj_de | +0.4475 | 0.0290 |
| 8 | home_team_BARTHAG | +0.4337 | 0.0268 |
| 9 | home_team_adj_de | +0.4164 | 0.0162 |
| 10 | away_conf_strength | +0.4005 | 0.0292 |
| 11 | away_def_tov_rate | +0.2872 | 0.0213 |
| 12 | away_def_eff_fg_pct | +0.2350 | 0.0202 |
| 13 | home_def_eff_fg_pct | +0.2011 | 0.0209 |
| 14 | away_off_rebound_pct | +0.1837 | 0.0221 |
| 15 | home_eff_fg_pct | +0.1706 | 0.0195 |
| 16 | away_3p_pct | +0.1687 | 0.0221 |
| 17 | home_sos_de | +0.1118 | 0.0203 |
| 18 | home_3p_pct | +0.1071 | 0.0221 |
| 19 | away_def_rebound_pct | +0.1047 | 0.0130 |
| 20 | away_margin_std | +0.1010 | 0.0084 |
| 21 | away_def_3p_pct | +0.0985 | 0.0173 |
| 22 | home_sos_oe | +0.0967 | 0.0120 |
| 23 | away_team_adj_pace | +0.0911 | 0.0155 |
| 24 | home_def_tov_rate | +0.0813 | 0.0139 |
| 25 | away_ft_pct | +0.0717 | 0.0121 |
| 26 | home_ft_pct | +0.0702 | 0.0130 |
| 27 | home_tov_rate | +0.0592 | 0.0109 |
| 28 | home_def_3p_pct | +0.0572 | 0.0154 |
| 29 | home_def_off_rebound_pct | +0.0525 | 0.0126 |
| 30 | home_off_rebound_pct | +0.0504 | 0.0130 |
| 31 | away_def_off_rebound_pct | +0.0489 | 0.0134 |
| 32 | home_team_adj_pace | +0.0466 | 0.0094 |
| 33 | neutral_site | +0.0463 | 0.0153 |
| 34 | home_margin_std | +0.0343 | 0.0098 |
| 35 | away_3pt_rate | +0.0332 | 0.0116 |
| 36 | home_def_3pt_rate | +0.0328 | 0.0124 |
| 37 | away_tov_rate | +0.0317 | 0.0135 |
| 38 | away_ft_rate | +0.0316 | 0.0089 |
| 39 | away_sos_oe | +0.0295 | 0.0128 |
| 40 | home_team_home | +0.0248 | 0.0078 |
| 41 | home_ft_rate | +0.0213 | 0.0083 |
| 42 | away_rest_days | +0.0188 | 0.0052 |
| 43 | home_3pt_rate | +0.0187 | 0.0136 |
| 44 | home_def_rebound_pct | +0.0171 | 0.0058 |
| 45 | rest_advantage | +0.0154 | 0.0050 |
| 46 | home_def_def_rebound_pct | +0.0116 | 0.0058 |
| 47 | home_rest_days | +0.0115 | 0.0054 |
| 48 | away_def_3pt_rate | +0.0106 | 0.0090 |
| 49 | away_def_def_rebound_pct | +0.0076 | 0.0059 |
| 50 | away_form_delta | +0.0049 | 0.0024 |
| 51 | home_opp_ft_rate | +0.0037 | 0.0060 |
| 52 | away_team_home | +0.0000 | 0.0000 |
| 53 | away_def_ft_rate | -0.0011 | 0.0040 |
| 54 | home_form_delta | -0.0029 | 0.0026 |

**Zero/negative importance**: ['away_team_home', 'away_def_ft_rate', 'home_form_delta']

**Force remove**: ['away_team_home']

### Multicollinearity (|r| > 0.95)

| Feature 1 | Feature 2 | |r| |
|-----------|-----------|-----|
| neutral_site | home_team_home | 1.0000 |
| away_team_adj_oe | away_team_adj_pace | 0.9618 |
| away_team_adj_oe | away_sos_oe | 0.9593 |
| away_team_adj_oe | away_sos_de | 0.9682 |
| away_team_adj_de | away_team_adj_pace | 0.9667 |
| away_team_adj_de | away_sos_oe | 0.9737 |
| away_team_adj_de | away_sos_de | 0.9641 |
| away_team_adj_pace | away_sos_oe | 0.9829 |
| away_team_adj_pace | away_sos_de | 0.9812 |
| home_team_adj_pace | home_sos_oe | 0.9606 |
| home_team_adj_pace | home_sos_de | 0.9584 |
| away_eff_fg_pct | away_3p_pct | 0.9612 |
| away_eff_fg_pct | away_def_rebound_pct | 0.9575 |
| away_def_eff_fg_pct | away_def_3p_pct | 0.9615 |
| away_def_eff_fg_pct | away_def_def_rebound_pct | 0.9505 |
| home_sos_oe | home_sos_de | 0.9567 |
| away_sos_oe | away_sos_de | 0.9819 |

## Task 2: Ablation Study

### Backward Elimination Log

| Step | Features | MAE (book) | Removed |
|------|----------|-----------|---------|
| start | 53 | 9.4275 | - |
| coarse_1 | 48 | 9.4218 | away_def_def_rebound_pct, away_form_delta, home_opp_ft_rate, away_def_ft_rate, home_form_delta |
| coarse_2 | 43 | 9.3939 | home_def_rebound_pct, rest_advantage, home_def_def_rebound_pct, home_rest_days, away_def_3pt_rate |
| coarse_3 | 38 | 9.4627 | away_sos_oe, home_team_home, home_ft_rate, away_rest_days, home_3pt_rate |
| coarse_4 | 33 | 9.4516 | home_margin_std, away_3pt_rate, home_def_3pt_rate, away_tov_rate, away_ft_rate |
| coarse_5 | 28 | 9.4855 | home_def_off_rebound_pct, home_off_rebound_pct, away_def_off_rebound_pct, home_team_adj_pace, neutral_site |
| coarse_6 | 23 | 9.5032 | home_def_tov_rate, away_ft_pct, home_ft_pct, home_tov_rate, home_def_3p_pct |
| coarse_7 | 18 | 9.5758 | away_def_rebound_pct, away_margin_std, away_def_3p_pct, home_sos_oe, away_team_adj_pace |
| coarse_8 | 13 | 9.5663 | away_off_rebound_pct, home_eff_fg_pct, away_3p_pct, home_sos_de, home_3p_pct |
| fine_remove_away_team_BARTHAG | 12 | 9.5060 | away_team_BARTHAG |
| fine_remove_home_team_BARTHAG | 11 | 9.5094 | home_team_BARTHAG |
| fine_remove_away_eff_fg_pct | 10 | 9.4856 | away_eff_fg_pct |
| fine_remove_home_def_eff_fg_pct | 9 | 9.4985 | home_def_eff_fg_pct |
| fine_remove_away_sos_de | 8 | 9.4946 | away_sos_de |

Backward result: **10 features**, MAE=9.4833

### Forward Selection Log

| Step | Features | MAE (book) | Added |
|------|----------|-----------|-------|
| start | 10 | 9.5621 | - |
| add_away_margin_std | 11 | 9.5206 | away_margin_std |
| add_home_3p_pct | 12 | 9.5073 | home_3p_pct |
| add_away_off_rebound_pct | 13 | 9.5229 | away_off_rebound_pct |
| add_away_def_3p_pct | 14 | 9.4932 | away_def_3p_pct |
| add_away_def_rebound_pct | 15 | 9.4978 | away_def_rebound_pct |
| add_home_tov_rate | 16 | 9.4900 | home_tov_rate |
| add_home_def_off_rebound_pct | 17 | 9.4975 | home_def_off_rebound_pct |
| add_home_def_tov_rate | 18 | 9.4864 | home_def_tov_rate |
| add_home_off_rebound_pct | 19 | 9.4848 | home_off_rebound_pct |
| add_home_def_3p_pct | 20 | 9.4828 | home_def_3p_pct |
| add_away_3p_pct | 21 | 9.4704 | away_3p_pct |
| add_home_def_eff_fg_pct | 22 | 9.4795 | home_def_eff_fg_pct |
| add_away_def_eff_fg_pct | 23 | 9.4826 | away_def_eff_fg_pct |
| add_away_def_tov_rate | 24 | 9.4622 | away_def_tov_rate |
| add_home_ft_pct | 25 | 9.4596 | home_ft_pct |
| add_home_eff_fg_pct | 26 | 9.4883 | home_eff_fg_pct |
| add_away_ft_pct | 27 | 9.4701 | away_ft_pct |
| add_away_team_adj_pace | 28 | 9.4657 | away_team_adj_pace |
| add_home_sos_de | 29 | 9.4474 | home_sos_de |
| add_home_sos_oe | 30 | 9.4678 | home_sos_oe |

Forward result: **29 features**, MAE=9.4474

### Selected: backward

- Features: ['away_team_adj_oe', 'away_sos_de', 'home_conf_strength', 'home_team_adj_oe', 'away_team_adj_de', 'home_team_adj_de', 'away_conf_strength', 'away_def_tov_rate', 'away_def_eff_fg_pct', 'home_def_eff_fg_pct']
- Count: 10
- MAE: 9.4833

## Task 3: Overfitting Diagnostics

### Train/Val/Holdout Gap

| Set | MAE |
|-----|-----|
| Train (80%) | 9.7236 |
| Val (20%) | 9.8209 |
| Holdout (2025) | 10.0348 |
| Train-Val Gap | 0.0973 |

### Learning Curves

| Data % | Samples | Train MAE | Val MAE | Gap |
|--------|---------|-----------|---------|-----|
| 10% | 4,800 | 10.4204 | 10.3765 | -0.0439 |
| 20% | 9,601 | 9.9258 | 9.9469 | 0.0211 |
| 40% | 19,203 | 9.7964 | 9.8662 | 0.0698 |
| 60% | 28,804 | 9.7889 | 9.8774 | 0.0885 |
| 80% | 38,406 | 9.7458 | 9.8271 | 0.0813 |
| 100% | 48,008 | 9.7146 | 9.8142 | 0.0996 |

### Regularization Experiments

| Dropout | Weight Decay | MAE (overall) | MAE (book) |
|---------|-------------|--------------|-----------|
| 0.2 | 0.0001 | 10.0708 | 9.5020 |
| 0.3 | 0.0001 | 10.0261 | 9.4994 |
| 0.4 | 0.0001 | 10.0876 | 9.4861 |
| 0.5 | 0.0001 | 10.0468 | 9.4872 |
| 0.3 | 0.001 | 10.3185 | 9.4744 |
| 0.3 | 1e-05 | 10.0470 | 9.4800 |
| 0.4 | 0.001 | 10.1753 | 9.5440 |

### Known-Weak Feature Ablation

- **away_team_home**: already not in optimal set
- **away_def_def_rebound_pct**: already not in optimal set
- **home_def_rebound_pct**: already not in optimal set

## Task 4: Final Evaluation

Best hparams: dropout=0.3, weight_decay=0.001

### Comparison Table

| Model | Features | MAE (book) | vs 37-feat | vs 54-feat |
|-------|----------|-----------|-----------|-----------|
| 37-feature baseline | 37 | 9.6200 | +0.0000 | -0.2413 |
| 54-feature expanded | 54 | 9.3787 | +0.2413 | +0.0000 |
| Pruned (backward) | 10 | 9.4833 | +0.1367 | -0.1046 |
| Final (tuned reg.) | 10 | 9.5492 | +0.0708 | -0.1705 |

### Monthly MAE

| Month | MAE | Games |
|-------|-----|-------|
| 2024-11 | 12.47 | 1512 |
| 2024-12 | 10.36 | 1162 |
| 2025-01 | 9.15 | 1451 |
| 2025-02 | 8.87 | 1371 |
| 2025-03 | 8.99 | 783 |
| 2025-04 | 8.41 | 19 |

### ATS ROI

#### Unfiltered

| Threshold | Bets | Wins | Losses | Win Rate | ROI |
|-----------|------|------|--------|----------|-----|
| 3 | 2278 | 1160 | 1118 | 50.9% | -2.8% |
| 5 | 1136 | 586 | 550 | 51.6% | -1.5% |
| 7 | 607 | 327 | 280 | 53.9% | +2.8% |

#### Sigma < 11.5

| Threshold | Bets | Wins | Losses | Win Rate | ROI |
|-----------|------|------|--------|----------|-----|
| 3 | 1015 | 507 | 508 | 50.0% | -4.6% |
| 5 | 371 | 192 | 179 | 51.8% | -1.2% |
| 7 | 133 | 76 | 57 | 57.1% | +9.1% |

### Calibration

- Within 1 sigma: 69.6% (expected ~68%)
- Within 2 sigma: 95.1% (expected ~95%)

## Final Feature Order (10 features)

```json
[
  "away_team_adj_oe",
  "away_sos_de",
  "home_conf_strength",
  "home_team_adj_oe",
  "away_team_adj_de",
  "home_team_adj_de",
  "away_conf_strength",
  "away_def_tov_rate",
  "away_def_eff_fg_pct",
  "home_def_eff_fg_pct"
]
```

## Action Taken

- Backed up 54-feature order as `feature_order_v2.json`
- Updated `feature_order.json` (10 features)
- Retrained models in `checkpoints/no_garbage/`
- Saved scaler to `artifacts/no_garbage/scaler.pkl`