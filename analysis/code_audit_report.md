# Hoops Edge Predictor — Complete Code Audit Report

**Date**: 2026-03-03
**Scope**: Full codebase audit across src/, scripts/, artifacts/, checkpoints/, tests/, site/

---

## Executive Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 3     |
| HIGH     | 12    |
| MEDIUM   | 15    |
| LOW      | 18    |
| **Total**| **48**|

### Top 3 Most Urgent Findings

1. **CRITICAL-1**: NaN imputation mismatch between training and inference — training imputes NaN→0.0, inference imputes NaN→column mean. Affects 214,744 values across 66,308 training rows.
2. **CRITICAL-3**: `_get_asof_rolling()` fallback leaks future data — when no prior games exist, it uses ALL games (including future) and takes `iloc[-1]` (last game of season).
3. **HIGH-1**: `get_feature_matrix()` silently drops missing features with no warning or error — model could silently run on fewer features than expected.

---

## Category 1: Train/Inference Consistency

### CRITICAL-1: NaN Imputation Mismatch

- **Files**: `scripts/train_production.py:51`, `src/infer.py:136-140`, `src/cli.py:164`
- **Description**: Three different NaN imputation strategies exist across the codebase:
  - `train_production.py` line 51: `X = np.nan_to_num(X, nan=0.0)` — replaces NaN with **0.0**
  - `src/cli.py` line 164: `X = impute_column_means(X)` — replaces NaN with **per-column means**
  - `src/infer.py` lines 136-140: Uses `scaler.mean_` — replaces NaN with **scaler's stored means**
- **Evidence**: 214,744 NaN values across 66,308 training rows (3.2 NaN per row average). Features like `adj_oe` (mean ~100) get imputed as 0 during training but ~100 during inference — a ~100-point shift.
- **Impact**: The production model was trained on data where NaN→0.0, but inference fills NaN with means (~100 for efficiency features). This creates systematic bias in every prediction where a feature was NaN during training.
- **Fix**: Change `train_production.py` line 51 from `np.nan_to_num(X, nan=0.0)` to `impute_column_means(X)` (imported from `src.trainer`). Retrain the production model. Ensure all three paths use `impute_column_means()`.

### HIGH-2: Scaler Save Path Mismatch (CLI train --no-garbage)

- **Files**: `src/cli.py:169`, `src/infer.py:95-96`
- **Description**: When training with `--no-garbage`, the CLI saves the scaler to `artifacts/no_garbage/scaler.pkl`, but `infer.py` always loads from `artifacts/scaler.pkl`.
- **Evidence**: `cli.py` line 169: `scaler_path = ARTIFACTS / "no_garbage" / "scaler.pkl"` vs `infer.py` lines 95-96: `scaler = joblib.load(ARTIFACTS / "scaler.pkl")`
- **Impact**: If someone trains via `cli.py train --no-garbage`, the scaler won't be found by inference. Currently mitigated because production training uses `train_production.py` which saves to the default path.
- **Fix**: Either (a) make `infer.py` accept a scaler path parameter, or (b) always save scaler to the same canonical path regardless of no_garbage flag, or (c) remove the no_garbage subdirectory logic from CLI.

### HIGH-3: Architecture Default Mismatch

- **Files**: `src/architecture.py:26-28,115-116`, `src/infer.py:82-84`, `src/trainer.py:77-86`
- **Description**: Multiple places define architecture defaults that don't match production:
  - `architecture.py`: `input_dim=50` (should be 53)
  - `infer.py` fallback: `hidden1=256, hidden2=128, dropout=0.3`
  - `trainer.py` defaults: `hidden1=256, hidden2=128, dropout=0.3`
  - Production model: `hidden1=384, hidden2=256, dropout=0.2`
- **Evidence**: If checkpoint metadata is missing `hparams`, `infer.py` falls back to 256/128/0.3 — wrong architecture for the production 384/256/0.2 model.
- **Impact**: Currently mitigated because checkpoints include hparams. But if a checkpoint is saved without hparams, inference would construct the wrong architecture and silently produce garbage predictions.
- **Fix**: Update all defaults to match production (384/256/0.2/53). Better: fail loudly if hparams are missing from checkpoint instead of falling back to defaults.

### HIGH-4: `no_garbage` Default Inconsistency

- **Files**: `src/dataset.py:74`, `src/dataset.py:53`, `src/features.py:481`
- **Description**: `no_garbage` defaults differ across modules:
  - `features.py build_features()`: `no_garbage=True`
  - `dataset.py load_season_features()`: `no_garbage=False`
  - `dataset.py CBBDataset.__init__()`: `no_garbage=False`
- **Evidence**: Direct code inspection of default parameter values.
- **Impact**: Training via `CBBDataset` (without explicit `no_garbage=True`) uses the old unfiltered features, while inference uses no_garbage=True. This was the exact bug discovered in the previous session.
- **Fix**: Set `no_garbage=True` as default in `dataset.py` to match `features.py` and production config. Or better: remove the flag entirely and always use no_garbage behavior.

### MEDIUM-1: `predict-today` Doesn't Explicitly Pass `no_garbage`

- **File**: `src/cli.py:251-258`
- **Description**: The `predict-today` command calls `build_features()` without explicitly passing `no_garbage=True`. It relies on the default in `features.py`.
- **Impact**: If someone changes the default in `features.py`, predictions break silently. Explicit is better than implicit for production-critical paths.
- **Fix**: Add `no_garbage=True` explicitly to the `predict-today` call chain.

### MEDIUM-2: Feature Count Not Validated at Inference

- **File**: `src/infer.py:130-145`
- **Description**: `predict()` doesn't validate that the input feature count matches the model's expected input dimension. It relies on PyTorch to raise a shape error at runtime.
- **Fix**: Add an assertion: `assert X.shape[1] == model.input_dim, f"Expected {model.input_dim} features, got {X.shape[1]}"`

### HIGH-10: Classifier Feature Order Silently Ignored in `predict()`

- **File**: `src/infer.py:127-132`
- **Description**: The classifier's `feature_order` is loaded from its checkpoint but immediately discarded (assigned to `_` at line 128). Only the regressor's `feature_order` is used for both models. If the regressor and classifier were ever trained with different feature orders, the classifier would receive incorrectly ordered features and produce wrong probabilities with no error.
- **Evidence**: Line 128: `classifier, _, _ = load_classifier()` — third return value (feature_order) is discarded. Line 132: `feature_order = reg_feature_order` used for both models.
- **Fix**: Validate that classifier's feature_order matches regressor's: `assert cls_feature_order == reg_feature_order`

### HIGH-11: Temporal Data Leakage in Tuner (Random KFold on Time-Series)

- **File**: `src/tuner.py:120,155`
- **Description**: The tuner uses `KFold(n_splits=n_folds, shuffle=True, random_state=42)` for cross-validation. This randomly mixes games from different dates, allowing future games into training folds. Hyperparameters selected this way may overfit to temporal leakage. (Also moot since tuner crashes per CRITICAL-2, but would be a problem if the crash were fixed.)
- **Fix**: Use `sklearn.model_selection.TimeSeriesSplit` or custom date-based walk-forward splits.

### HIGH-12: Sklearn Version Mismatch for Scaler

- **File**: `artifacts/scaler.pkl`
- **Description**: The scaler was pickled with `sklearn 1.8.0` but the current environment has `sklearn 1.6.1`. Loading a newer-version pickle in an older sklearn raises `InconsistentVersionWarning` and could cause silent data corruption if the internal `StandardScaler` format changed between versions.
- **Fix**: Re-fit and re-save the scaler in the current environment, or upgrade sklearn to match.

---

## Category 2: Data Leakage

### CRITICAL-3: `_get_asof_rolling()` Fallback Leaks Future Data

- **File**: `src/features.py:212-216`
- **Description**: When no prior data exists for a team (e.g., first game of the season), the function falls back to using the **entire** `team_df` and takes `iloc[-1]` — which is the team's **last** game of the season. This leaks future data into early-season games.
- **Evidence**:
  ```python
  eligible = team_df[dates < cutoff]
  if eligible.empty:
      eligible = team_df  # BUG: uses ALL games including future
  row = eligible.iloc[-1]  # Takes the LAST row — from the future!
  ```
- **Impact**: Affects the first ~1-3 games per team per season where no prior rolling data exists. These games get end-of-season rolling averages, creating a subtle systematic bias in early-season predictions. During walk-forward training, this leaks holdout-year data into feature computation.
- **Fix**: Return an empty dict when no prior data exists:
  ```python
  if eligible.empty:
      return {}  # No prior data available — features will be NaN, handled by imputation
  ```

### LOW-1: Same-Date Game Ordering in Rolling Averages

- **File**: `src/rolling_averages.py`
- **Description**: Rolling averages use `shift(1)` after sorting by `(teamid, _date, gameid)`. If a team plays two games on the same date (extremely rare, e.g., rescheduled doubleheader), game B's rolling average would include game A's stats despite being "same day."
- **Impact**: Negligible — this scenario is virtually impossible in college basketball.
- **Fix**: No action needed. Document as known edge case.

### LOW-2: Dead Rolling Average Columns Still Computed

- **File**: `src/rolling_averages.py`
- **Description**: `away_def_def_rebound_pct` and `home_def_def_rebound_pct` are still in rolling maps but not in any FEATURE_ORDER.
- **Impact**: Wasted computation, no correctness issue.
- **Fix**: Remove from rolling maps if confirmed unused.

### VERIFIED OK: Efficiency Rating As-Of Lookup

- **File**: `src/features.py`
- **Description**: Efficiency ratings use strictly-before date semantics (`rating_date < game_date`). No leakage found.

### VERIFIED OK: Rolling Average Anti-Leakage

- **File**: `src/rolling_averages.py`
- **Description**: `shift(1)` correctly excludes the current game from rolling calculations. No leakage found.

### VERIFIED OK: HCA Features

- **File**: `src/features.py`
- **Description**: HCA features (home_team_hca, efg_home_split, efg_away_split) use strictly historical data. No leakage found.

---

## Category 3: Silent Failures and Edge Cases

### HIGH-5: `get_feature_matrix()` Silently Drops Missing Features

- **File**: `src/features.py:848-851`
- **Description**: `get_feature_matrix()` intersects requested features with available columns and silently returns only the overlap. If a feature is missing from the DataFrame, it's dropped without warning.
- **Evidence**: Code uses set intersection logic — no log, no warning, no error if a feature is absent.
- **Impact**: Model could silently run on fewer features than expected. With StandardScaler applied, the remaining features would have wrong indices, producing garbage predictions.
- **Fix**: Add validation: if any requested feature is missing, raise an error or at minimum log a warning with the missing feature names.

### HIGH-6: Two Competing JSON Pipelines

- **Files**: `src/infer.py:save_predictions()`, `scripts/csv_to_json.py`
- **Description**: Two independent code paths generate prediction JSON for the site:
  - `infer.py save_predictions()`: camelCase keys, flat array `[{...}, {...}]`
  - `csv_to_json.py`: snake_case keys, wrapped `{"games": [{...}]}`
- **Evidence**: Both write to `site/public/data/predictions_*.json` with different schemas.
- **Impact**: If the wrong pipeline runs, the frontend may fail to parse predictions. The site components must handle both schemas or one pipeline is dead code.
- **Fix**: Consolidate to one pipeline. Determine which schema the frontend actually consumes and remove the other.

### MEDIUM-3: `build-rankings` Ignores `--season` Parameter

- **File**: `src/cli.py:408-418`
- **Description**: The `build-rankings` CLI command accepts a `--season` option but internally ignores it and uses hardcoded logic or defaults.
- **Fix**: Wire the `--season` parameter through to the rankings builder.

### MEDIUM-4: Fragile "Latest CSV by mtime" Logic

- **Files**: `src/cli.py:482-489,703-705`
- **Description**: `backfill-season` and `daily-update` find the latest prediction CSV by sorting files by modification time. This breaks if files are touched out of order (e.g., by `ls`, `cp`, or editor autosave).
- **Fix**: Parse dates from filenames (`preds_YYYY_M_D_edge.csv`) instead of relying on mtime.

### MEDIUM-5: `tune` CLI Missing `no_garbage` Flag

- **File**: `src/cli.py:194-234`
- **Description**: The `tune` command doesn't accept `--no-garbage` or `--adj-suffix` options, so hyperparameter tuning always runs on unfiltered features.
- **Impact**: Tuning results won't match production training config.
- **Fix**: Add `--no-garbage` and `--adj-suffix` options to the `tune` command.

### MEDIUM-6: `model_error` Defaults to 0 on Frontend

- **File**: `site/src/pages/history.tsx`
- **Description**: When `model_error` is null/undefined, it defaults to 0 rather than being excluded from MAE calculations.
- **Impact**: Inflates apparent model accuracy on the history page for games without valid error data.
- **Fix**: Exclude null `model_error` from MAE aggregations.

### MEDIUM-7: `model_mu_home` Negation Pattern in Frontend

- **File**: `site/src/pages/history.tsx`
- **Description**: The frontend negates `model_mu_home` to display as a conventional spread. This is correct but undocumented and fragile — if the sign convention changes upstream, the display silently inverts.
- **Fix**: Add a comment explaining the sign convention. Consider sending `model_spread` (already negated) from the backend to avoid frontend sign manipulation.

### MEDIUM-8: No Null Guards on `.toFixed()` in Rankings

- **File**: `site/src/pages/rankings.tsx`
- **Description**: Several `.toFixed()` calls don't guard against null/undefined values, which would throw a runtime error.
- **Fix**: Add null checks: `value?.toFixed(1) ?? 'N/A'`

### MEDIUM-13: `bool("False")` Bug Risk for `neutralSite`

- **File**: `src/features.py:669`
- **Description**: `neutral = bool(game.get("neutralSite", False))` — if Parquet data stores `neutralSite` as a string `"False"`, then `bool("False") == True`, marking every game as neutral site. This would corrupt `neutral_site`, `home_team_home`, and `home_team_hca` features.
- **Fix**: Use explicit comparison: `neutral = game.get("neutralSite", False) in (True, 1)`

### MEDIUM-14: Timezone Handling Inconsistency

- **File**: `src/features.py`
- **Description**: The game date filter (line 521) parses dates with `utc=True` and converts to Eastern Time, but the efficiency lookup and rolling average dates parse without `utc=True`. If S3 dates contain timezone info, the date filtering could pick games on a different date than what the lookups use for anti-leakage cutoffs.
- **Fix**: Standardize timezone handling — always strip timezone or always convert to ET before date comparisons.

### MEDIUM-15: Stale `no_garbage/scaler.pkl` Has 10 Features (Production Has 53)

- **File**: `artifacts/no_garbage/scaler.pkl`
- **Description**: The no-garbage scaler has `n_features_in_=10`, while the production scaler has 53. This is a leftover from before the feature expansion. If anyone loads the `no_garbage` variant, it would fail with a dimension mismatch.
- **Fix**: Delete the stale `artifacts/no_garbage/` directory or regenerate with current features.

### LOW-3: `_safe_div` Returns NaN (Not 0) for Division by Zero

- **File**: `src/four_factors.py`
- **Description**: `_safe_div()` returns NaN when dividing by zero. This is correct behavior for downstream imputation but could cause issues if NaN propagation isn't handled.
- **Impact**: NaN is the right choice for statistical features — it's properly handled by the imputation pipeline.
- **Fix**: No action needed. This is correct.

### LOW-4: Empty DataFrame Not Checked in `build_features()`

- **File**: `src/features.py`
- **Description**: If the games DataFrame is empty (e.g., no games for a date), `build_features()` proceeds through all computation steps before returning an empty DataFrame.
- **Impact**: No crash, just wasted computation. Returns correct empty result.
- **Fix**: Optional early return if games DataFrame is empty.

---

## Category 4: Stale Code and Dead Paths

### CRITICAL-2: `tuner.py` Is Completely Broken

- **File**: `src/tuner.py:49-50,59`
- **Description**: `gaussian_nll_loss()` returns a tuple `(loss, nll_loss)` but tuner treats the return value as a scalar:
  - Line 49-50: `loss = gaussian_nll_loss(...)` then `loss.backward()` — calling `.backward()` on a tuple crashes
  - Line 59: `loss.item()` on a tuple also crashes
- **Evidence**: `gaussian_nll_loss` signature in `src/architecture.py` returns `Tuple[Tensor, Tensor]`.
- **Impact**: Any `tune` CLI call (`python -m src.cli tune`) crashes immediately with `AttributeError: 'tuple' object has no attribute 'backward'`.
- **Fix**: Unpack the tuple: `loss, nll_loss = gaussian_nll_loss(...)` then use `loss.backward()`.

### MEDIUM-9: Dead Code — `home_team_home` / `away_team_home` Still Computed

- **Files**: `src/features.py:690-691`, `src/rolling_averages.py`
- **Description**: `home_team_home` and `away_team_home` binary flags are still computed in `build_features()` but were pruned from FEATURE_ORDER in V4. They consume compute but are never used.
- **Fix**: Remove computation of these columns.

### MEDIUM-10: `build_features()` Parameter Defaults Don't Match Production

- **File**: `src/features.py`
- **Description**: `build_features()` has parameters like `adjust_ff`, `adjust_prior`, `decay_lambda` with defaults that may not match current production config.
- **Impact**: Calling `build_features()` without explicit parameters may produce different features than production.
- **Fix**: Centralize all defaults in `config.py` and reference them from `build_features()`.

### LOW-5: Stale Docstrings — Feature Count References

- **Files**: `src/infer.py:120` ("37 features"), `src/trainer.py:70,135` ("(N, 37)"), `src/cli.py:53` ("54-feature"), `src/features.py:503` ("base 37 features")
- **Description**: Multiple docstrings reference old feature counts (37, 50, 54) instead of current 53.
- **Fix**: Update all docstrings to reference current feature count, or better, reference `len(FEATURE_ORDER)` dynamically.

### LOW-6: Stale Default `input_dim=50` in Architecture

- **Files**: `src/architecture.py:26-28,115-116`
- **Description**: Both `MLPRegressor` and `MLPClassifier` default to `input_dim=50`, which is the old V1 feature count.
- **Impact**: Always overridden at construction, so no runtime effect.
- **Fix**: Update to 53 or remove default (require explicit input_dim).

### LOW-7: `form_delta` Mixes Offensive and Defensive Stats

- **File**: `src/features.py`
- **Description**: The `form_delta` feature computes the difference between recent and season-long performance but mixes offensive and defensive metrics in the calculation.
- **Impact**: The feature still captures "form change" signal, just not as cleanly as it could.
- **Fix**: Consider separating into `form_delta_off` and `form_delta_def` in a future feature revision.

### LOW-8: Dead Imports and Unused Variables

- **Files**: Various across `src/` and `scripts/`
- **Description**: Several files contain imports that are no longer used after refactoring.
- **Fix**: Run `ruff check --select F401` to identify and remove unused imports.

### LOW-9: Hardcoded Poetry Virtualenv Path in Scripts

- **Description**: Several scripts reference the full poetry virtualenv path. If the virtualenv is recreated, these break.
- **Fix**: Use `poetry run python` instead of the full path.

### LOW-10: Stale `analyze_pick_bias.py` and `away_bias_analysis.py`

- **Files**: `scripts/analyze_pick_bias.py`, `scripts/away_bias_analysis.py`
- **Description**: These analysis scripts reference old feature configurations and may not work with the current 53-feature setup.
- **Fix**: Update or remove if no longer needed.

---

## Category 5: Deployment Pipeline

### HIGH-7: No Tests for `predict()` Inference Pipeline

- **File**: `tests/`
- **Description**: No unit tests exercise the full `predict()` function from `src/infer.py`. The NaN imputation mismatch went undetected because there's no test that runs features through the full train→predict round-trip.
- **Fix**: Add integration test that: (1) builds features for a known game, (2) runs through scaler + model, (3) validates output shape and reasonable prediction range.

### HIGH-8: No Tests for `csv_to_json.py` Field Mapping

- **File**: `scripts/csv_to_json.py`, `tests/`
- **Description**: No tests verify that the CSV→JSON conversion preserves all fields correctly and matches the schema expected by the frontend.
- **Fix**: Add unit test with a sample CSV row, convert to JSON, validate all expected fields present with correct types.

### HIGH-9: `test_feature_order_columns` Doesn't Exercise HCA or Rolling Features

- **File**: `tests/test_features.py:44-98`
- **Description**: The main feature assembly test uses minimal mock data (5 boxscore rows) that doesn't generate meaningful rolling averages, HCA splits, or venue-split features. It only checks that the column count matches FEATURE_ORDER.
- **Impact**: Won't catch bugs where features are computed but contain all-NaN or all-zero values.
- **Fix**: Enhance mock data with enough games (20+) to produce non-trivial rolling averages and HCA values. Add assertions that key features are non-NaN.

### MEDIUM-11: Stale `inspect_hca_weights.py`

- **File**: `scripts/inspect_hca_weights.py`
- **Description**: This script may reference old checkpoint formats.
- **Fix**: Verify it works with current checkpoints or update.

### MEDIUM-12: Frontend Handles Both JSON Schemas

- **File**: `site/src/pages/index.tsx`, `site/src/pages/history.tsx`
- **Description**: The frontend may need to handle both the camelCase (from `infer.py`) and snake_case (from `csv_to_json.py`) schemas, depending on which pipeline produced the current day's data.
- **Fix**: Standardize on one JSON schema and one pipeline (see HIGH-6).

### LOW-11 through LOW-18: Minor Issues

| # | File | Issue |
|---|------|-------|
| LOW-11 | `src/config.py` | FEATURE_ORDER list could be auto-loaded from `artifacts/feature_order.json` to avoid drift |
| LOW-12 | `src/corrected_efficiencies.py` | Module imported but only used by V5 features |
| LOW-13 | `scripts/fix_spread_signs.py` | Hardcoded team name strings for spread sign correction |
| LOW-14 | `site/src/pages/metrics.tsx` | Edge range slider defaults (0-50%) may not match backend thresholds |
| LOW-15 | Various test files | Tests use hardcoded dates (2025-01-15) that may need updating for new seasons |
| LOW-16 | `src/features.py` | `load_lines` mock in tests doesn't return meaningful data |
| LOW-17 | `scripts/hca_walkforward_comparison.py` | Analysis script, not production — but references specific checkpoint paths |
| LOW-18 | `site/next.config.js` | No CSP headers configured for production deployment |

---

## Recommended Fix Priority

### Immediate (before next prediction run)
1. **CRITICAL-1**: Fix NaN imputation in `train_production.py` → use `impute_column_means()` → retrain model
2. **CRITICAL-3**: Fix `_get_asof_rolling()` fallback — return empty dict instead of using future data → retrain model
3. **HIGH-5**: Add validation to `get_feature_matrix()` — fail loudly on missing features
4. **HIGH-12**: Fix sklearn version mismatch for scaler — re-fit in current environment

### This week
5. **HIGH-2**: Consolidate scaler save/load paths
6. **HIGH-3**: Update architecture defaults to match production
7. **HIGH-4**: Unify `no_garbage` defaults to `True`
8. **HIGH-6**: Consolidate JSON pipelines to one schema
9. **HIGH-7/8/9**: Add inference pipeline and CSV→JSON tests
10. **HIGH-10**: Validate classifier feature_order matches regressor
11. **CRITICAL-2**: Fix `tuner.py` tuple unpacking (blocks any tuning work)
12. **HIGH-11**: Fix temporal leakage in tuner (TimeSeriesSplit)

### When convenient
13. **MEDIUM-1 through MEDIUM-15**: Parameter validation, CLI fixes, frontend guards, timezone handling
14. **LOW-1 through LOW-18**: Docstrings, dead code, minor cleanups

---

*Generated by code audit — 2026-03-03*
