# Internal Bet Filter Live Tracking

## Files Changed

- `scripts/build_daily_internal_bet_filter_report.py`
- `src/market_bet_tracking.py`
- `scripts/build_internal_bet_filter_tracking_report.py`
- `tests/test_market_bet_tracking.py`

## How The Workflow Runs

1. Build the daily internal slate report:
   - `poetry run python scripts/build_daily_internal_bet_filter_report.py --season 2026 --date 2026-03-14`
2. Rebuild the cumulative live tracker:
   - `poetry run python scripts/build_internal_bet_filter_tracking_report.py --season 2026`

The tracker reads saved daily internal artifacts from `artifacts/daily_internal_bet_filter/YYYY-MM-DD/`, joins final scores from `fct_games`, settles ATS results using the same `-110` vig convention as the research stack, and writes outputs under `artifacts/internal_bet_filter_tracking/`.

## What Is Tracked

- `internal_filter`
  - promoted disagreement-aware picks, excluding NCAA caution rows
- `raw_edge_baseline`
  - raw `pick_prob_edge` threshold picks
- `overlap`
  - picks that clear both filters
- `filter_only`
  - disagreement-aware picks that the raw baseline would miss
- `raw_only`
  - raw baseline picks that the disagreement-aware filter rejects
- `ncaa_caution`
  - NCAA threshold-pass rows tracked separately as diagnostic-only

Outputs:
- `tracked_pick_ledger.csv`
- `by_strategy_summary.csv`
- `by_day_summary.csv`
- `cumulative_by_day.csv`
- `report.md`
- `manifest.json`

## Validation

- Tests:
  - `poetry run pytest tests/test_market_bet_tracking.py tests/test_market_bet_filter.py tests/test_market_disagreement.py tests/test_market_ratings.py -q`
  - result: `14 passed`
- Completed-slate validation:
  - March 13, 2026 internal report generated and settled successfully
- Live/pending validation:
  - March 14, 2026 picks were included and remain pending until finals attach

Current live-tracking snapshot from the settled March 13 sample:
- `internal_filter`: `4-2-1`, `66.7%` ATS, `+27.27%` ROI
- `raw_edge_baseline`: `3-3`, `50.0%` ATS, `-4.55%` ROI

## Caveats

- This is internal-only and should stay separate from public prediction outputs.
- NCAA remains a diagnostic/supporting bucket, not a standalone decision rule.
- The current live sample is still tiny; use this as an honest tracking loop, not as proof that future edge is guaranteed.
- Older daily artifacts without explicit `gameId`/season columns are still supported through a team/date join fallback, but the richer current report format is preferred going forward.

## Ready Status

- `ready for routine live tracking`
