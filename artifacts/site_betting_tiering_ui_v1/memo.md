# Betting Tab Tiering UI Memo

## Files Changed

- `site/pages/betting.tsx`

## What Changed

- Added score tiers based on the existing audited score behavior:
  - `Strongest` for `score >= 0.62`
  - `Solid` for `0.58 <= score < 0.62`
  - `Monitor` for below-threshold review rows
  - `Caution only` for NCAA rows
- Added slate-rank and percentile context derived from the current slate's `filter_score`.
- Added `lift vs raw` display using the existing `score_lift_vs_raw_logit` field.
- Added compact regime-aware interpretation inside each row card.
- Added a tier guide and a `How To Read This` section directly on the page.
- Kept NCAA rows visually separated and caution-labeled.

## Route

- Betting tab/page: `/betting`

## Data Source

- Reads the existing internal daily betting artifacts from:
  - `site/public/data/internal_betting/YYYY-MM-DD/`
- Uses the current daily rows only; no score logic or threshold logic was changed.

## Local Use

- Build:
  - `cd site && npm run build`
- Run locally:
  - `cd site && npm run start -- --hostname 127.0.0.1 --port 3010`
- Example pages:
  - `http://127.0.0.1:3010/betting?date=2026-03-14`
  - `http://127.0.0.1:3010/betting?date=2026-03-15`

## Validation

- `npm run build`
- Confirmed `/betting?date=2026-03-14` renders:
  - tier cards
  - slate rank / percentile context
  - historical reading guide
  - the real shortlist row
- Confirmed `/betting?date=2026-03-15` renders:
  - empty shortlist state
  - tier guide
  - NCAA caution section still separate

## Intentionally Unchanged

- Public prediction logic
- Disagreement score math
- `0.58` promoted threshold
- NCAA caution handling
- Public site prediction pages

## Notes

- Tiering is based on the completed score audit, but the page still presents the score as a ranking tool rather than a literal probability.
- The existing broad `server-data` file pattern warning from Next build remains unrelated to this page change.
