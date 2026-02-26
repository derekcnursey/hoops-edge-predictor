# Feature Expansion Evaluation — Season 2025

Baseline MAE (book-spread games): **9.62**

## Individual Feature Group Results

| Group | Extra Features | MAE (overall) | MAE (book games) | Lift |
|-------|---------------|---------------|------------------|------|
| rest_days | 3 | 10.1101 | 9.6335 | -0.0135 |
| sos | 4 | 10.0937 | 9.6182 | +0.0018 |
| conf_strength | 2 | 9.9688 | 9.4498 | +0.1702 |
| form_delta | 2 | 10.1422 | 9.6068 | +0.0132 |
| tov_rate | 4 | 10.0896 | 9.5972 | +0.0228 |
| margin_std | 2 | 10.1054 | 9.5698 | +0.0502 |

## Combined Positive-Lift Groups: ['sos', 'conf_strength', 'form_delta', 'tov_rate', 'margin_std']

- Features: 51 (37 base + 14 extra)
- MAE (overall): 9.9118
- MAE (book-spread games): 9.4137
- Lift vs baseline: +0.2063

## All Groups Combined

- Features: 54
- MAE (overall): 9.9774
- MAE (book-spread games): 9.3787
- Lift vs baseline: +0.2413

## Best Result: all groups

- MAE: 9.3787
- Lift: +0.2413
- Features: ['neutral_site', 'away_team_adj_oe', 'away_team_BARTHAG', 'away_team_adj_de', 'away_team_adj_pace', 'home_team_adj_oe', 'home_team_adj_de', 'home_team_adj_pace', 'home_team_BARTHAG', 'home_team_home', 'away_team_home', 'away_eff_fg_pct', 'away_ft_pct', 'away_ft_rate', 'away_3pt_rate', 'away_3p_pct', 'away_off_rebound_pct', 'away_def_rebound_pct', 'away_def_eff_fg_pct', 'away_def_ft_rate', 'away_def_3pt_rate', 'away_def_3p_pct', 'away_def_off_rebound_pct', 'away_def_def_rebound_pct', 'home_eff_fg_pct', 'home_ft_pct', 'home_ft_rate', 'home_3pt_rate', 'home_3p_pct', 'home_off_rebound_pct', 'home_def_rebound_pct', 'home_def_eff_fg_pct', 'home_opp_ft_rate', 'home_def_3pt_rate', 'home_def_3p_pct', 'home_def_off_rebound_pct', 'home_def_def_rebound_pct', 'home_rest_days', 'away_rest_days', 'rest_advantage', 'home_sos_oe', 'home_sos_de', 'away_sos_oe', 'away_sos_de', 'home_conf_strength', 'away_conf_strength', 'home_form_delta', 'away_form_delta', 'home_tov_rate', 'home_def_tov_rate', 'away_tov_rate', 'away_def_tov_rate', 'home_margin_std', 'away_margin_std']

### Monthly MAE (Best Model)

| Month | MAE | Games |
|-------|-----|-------|
| 2024-11 | 12.49 | 1512 |
| 2024-12 | 9.92 | 1162 |
| 2025-01 | 9.05 | 1451 |
| 2025-02 | 8.85 | 1371 |
| 2025-03 | 8.93 | 783 |
| 2025-04 | 8.46 | 19 |

### ATS ROI (Best Model)

Sigma: median=10.97, p25=10.27

#### Unfiltered

| Threshold | Bets | Wins | Losses | Win Rate | ROI |
|-----------|------|------|--------|----------|-----|
| 3 | 2125 | 1075 | 1050 | 50.6% | -3.4% |
| 5 | 1009 | 501 | 508 | 49.7% | -5.2% |
| 7 | 490 | 265 | 225 | 54.1% | +3.2% |

#### Sigma<11.0

| Threshold | Bets | Wins | Losses | Win Rate | ROI |
|-----------|------|------|--------|----------|-----|
| 3 | 1001 | 509 | 492 | 50.8% | -2.9% |
| 5 | 385 | 190 | 195 | 49.4% | -5.8% |
| 7 | 144 | 78 | 66 | 54.2% | +3.4% |


## Action Taken

- Backed up `feature_order_v1.json` (37 features)
- Updated `feature_order.json` (54 features)
- Retrained models in `checkpoints/no_garbage/`