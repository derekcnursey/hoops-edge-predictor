# Solver Redesign: WLS + Possession Fix

## Changes Made

### Problem 1: Possession/Pace Fix (0.44 → 0.475)

Changed the FTA coefficient in possession formulas from 0.44 (NBA/Dean Oliver) to 0.475 (KenPom college-specific value) in:

| File | Lines | Change |
|------|-------|--------|
| `scripts/build_pbp_game_teams_flat.py` | 277-278 | Possession formula: 0.44 → 0.475 |
| `scripts/build_pbp_game_team_stats.py` | 271, 281 | Possession formula: 0.44 → 0.475 |
| `scripts/build_pbp_game_team_stats.py` | 273, 283 | Switched from `possessions_ends` to `possessions_formula` as primary |
| `src/cbbd_etl/gold/player_season_impact.py` | 131, 135 | Usage rate formula: 0.44 → 0.475 |
| `src/cbbd_etl/gold/adjusted_efficiencies.py` | 413, 456 | Updated comments |

**NOT changed** (correct at 0.44): True shooting percentage formula in `build_pbp_game_teams_flat.py`, `build_pbp_team_daily_rollup.py`, `player_season_impact.py`.

### Problem 2: WLS (Massey-style) Solver

Created `src/cbbd_etl/gold/least_squares_ratings.py` implementing a Weighted Least Squares solver as an alternative to the iterative fixed-point solver.

**Key design:**
- Additive Pomeroy model: `oe = league_avg + x_off[team] + x_def[opp] + HCA`
- Unknowns are deviations from league average (centered at 0)
- Ridge regression for regularization
- HCA estimated from data (2.57 pts/100poss for 2025 season)
- Possession-weighted equations (sqrt(poss) × recency)
- Optional margin capping
- Direct solve (no iterations needed)

**Wired into pipeline:**
- `config.yaml`: Added `solver: iterative` option (choices: `iterative`, `wls`)
- `adjusted_efficiencies.py`: Modified `_run_per_date_ratings()` to dispatch to either solver

## Validation Results (2025 Season)

### Iterative vs WLS Comparison

| Metric | Value |
|--------|-------|
| adj_oe correlation | r = 0.9842 |
| adj_de correlation | r = 0.9784 |
| adj_margin correlation | r = 0.9812 |
| MAE(margin) | 3.09 pts |
| WLS HCA estimate | 2.57 pts/100poss |

WLS produces ~5-8 pt wider margins for top teams (no sos_exponent=0.85 dampening).

### Parameter Sweep (64 configs)

- **Alpha (ridge)**: Minimal effect (0.001-0.1 → nearly identical ratings)
- **Margin cap**: 25pt cap reduces OE spread from 9.0→7.7, but increases DE spread 8.2→8.3
- **Half-life**: Recency weighting shifts league avg slightly (109.87→110.16)
- **HCA**: Consistent 2.18-2.69 across all configs

## Files Created/Modified

### ETL Repo (hoops_edge_database_etl)
- **CREATED**: `src/cbbd_etl/gold/least_squares_ratings.py`
- **MODIFIED**: `scripts/build_pbp_game_teams_flat.py` (0.475)
- **MODIFIED**: `scripts/build_pbp_game_team_stats.py` (0.475 + formula primary)
- **MODIFIED**: `scripts/build_pbp_team_daily_rollup.py` (unchanged, only has TS%)
- **MODIFIED**: `src/cbbd_etl/gold/player_season_impact.py` (0.475)
- **MODIFIED**: `src/cbbd_etl/gold/adjusted_efficiencies.py` (WLS support)
- **MODIFIED**: `config.yaml` (solver type + WLS params)

### Predictor Repo (hoops-edge-predictor)
- **CREATED**: `scripts/wls_solver_validation.py`
- **CREATED**: `scripts/wls_walkforward.py`

## Next Steps

1. Switch `config.yaml` to `solver: wls` and rebuild gold-layer ratings for all seasons
2. Rebuild silver layer with 0.475 possession formula
3. Run full walk-forward comparison (requires rebuilt features)
4. Spot-check WLS adj_tempo against KenPom AdjT values
