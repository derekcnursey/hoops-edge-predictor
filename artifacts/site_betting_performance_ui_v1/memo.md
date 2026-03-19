# Betting Performance Page Memo

## Files Changed

- `site/components/Layout.tsx`
- `site/pages/betting.tsx`
- `site/lib/betting-performance-data.ts`
- `site/pages/betting/performance.tsx`

## Route Added

- `/betting/performance`

## Data Source

- Reads the existing historical profitability artifacts from:
  - `artifacts/market_bet_profitability_v1/overall_profitability.csv`
  - `artifacts/market_bet_profitability_v1/by_season_profitability.csv`
  - `artifacts/market_bet_profitability_v1/signal_driver_profitability.csv`
  - `artifacts/market_bet_profitability_v1/robustness_summary.csv`
  - `artifacts/market_bet_profitability_v1/betting_tab_mapping_profitability.csv`
  - `artifacts/market_bet_profitability_v1/memo.md`

## What The Page Does

- Shows historical ATS / ROI validation for the internal betting workflow.
- Compares the promoted internal filter to the raw-edge baseline.
- Highlights key late-season slices:
  - full sample
  - Feb 15+
  - March only
  - conference tournaments
  - NCAA caution
- Includes practical subgroup views:
  - disagreement-led
  - raw-edge-led
  - persistent disagreement
  - new/transient disagreement
- Includes season-by-season and robustness context.

## Local Use

- Live workflow page:
  - `/betting`
- Historical validation page:
  - `/betting/performance`

## Validation

- `cd site && npm run build`
- Ran locally at:
  - `http://127.0.0.1:3011/betting/performance`
  - `http://127.0.0.1:3011/betting?date=2026-03-14`
- Confirmed:
  - the new historical page loads correctly
  - it reads the profitability artifacts correctly
  - the main Betting tab still works
  - Betting remains the active top-nav item for both routes
  - public prediction pages were untouched

## Intentionally Omitted

- No new research logic
- No public prediction changes
- No live betting score/threshold changes
- No analytics dashboard complexity beyond tables and summary cards

## Notes

- This page is meant as internal validation and explanation, not a public-facing tout page.
- The existing broad `server-data` pattern warning from Next build remains unrelated to this page addition.
