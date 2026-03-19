# Site Internal Betting UI Integration

## Files Changed

- `site/components/Layout.tsx`
- `site/lib/internal-betting-data.ts`
- `site/pages/betting.tsx`
- `site/pages/rankings.tsx`

## Route / Tab Added

- New separate site route: `/betting`
- New nav tab label: `Betting`

This is intentionally separate from the public predictions page. The public `/` page was left alone.

## How The Betting Page Loads Data

- The page reads internal daily betting artifacts from:
  - `../artifacts/daily_internal_bet_filter/YYYY-MM-DD/`
- It loads on the server side via `site/lib/internal-betting-data.ts`
- Default behavior:
  - choose the latest available artifact date at or before today
- Supported files:
  - `manifest.json`
  - `slate_shortlist.csv`
  - `raw_edge_watchlist.csv`
  - `ncaa_caution.csv`
  - `slate_scores.csv`

## What The Page Shows

- internal candidate shortlist
- raw-edge watchlist
- NCAA caution section, visually separated
- disagreement-aware score
- raw edge
- market line
- persistence / new-transient context
- tournament / neutral context
- signal-driver / usage labeling
- report date / season / threshold summary

## BALLIN' Rename

User-facing rankings UI label changed from `DCN Index` to `BALLIN'` in:
- `site/pages/rankings.tsx`

Intentionally left unchanged:
- backend/internal key: `model_index`
- existing data payload fields like `model_index_label` inside the JSON files

That keeps the UI rename surgical without disturbing data contracts.

## Local Validation

- Site build:
  - `npm run build`
- Local runtime check:
  - `npm run dev`
  - open:
    - `http://localhost:3000/betting`
    - `http://localhost:3000/rankings`

Verified:
- `/betting` loads a real March 14 internal artifact set
- shortlist / raw watchlist / NCAA caution sections render
- `/rankings` shows `BALLIN'`
- public predictions page remains separate

## Notes

- No public prediction output or website-facing prediction values were altered.
- The internal betting page is only a site surface over existing internal artifacts; it does not rebuild or modify the betting workflow itself.
