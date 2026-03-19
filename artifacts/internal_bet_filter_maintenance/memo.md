# Internal Bet Filter Maintenance Policy

## Recommendation

- Recommended policy: keep the filter stable in season and use a conservative three-layer framework:
  - operational monitoring
  - diagnostic review alerts
  - offseason-only recalibration triggers

## Cadence

- Daily or near-daily:
  - refresh the live tracker after results settle
  - refresh the maintenance report
- In season:
  - do not change thresholds because of a short hot/cold run
  - only trigger a manual review when the alert thresholds are actually met
- Offseason:
  - consider recalibration only after a full completed live sample exists

## Recommended Guardrails

- Do not react to fewer than `15` settled bets over `14` days.
- Do not treat `30`-day underperformance as meaningful until there are at least `30` settled bets.
- Do not treat March / conference-tournament pockets as a recalibration signal until there are at least `15` settled bets in that slice.
- Do not consider true recalibration until there is a completed-season sample with at least:
  - `100` internal-filter bets
  - `50` filter-only bets

## Diagnostic Alerts vs Recalibration

- Review alert:
  - bad trailing ROI on a meaningful sample
  - late-season slice underperformance on a meaningful sample
  - filter-only picks materially lagging raw-only picks over a meaningful trailing window
- Recalibration trigger:
  - offseason only
  - completed-season sample
  - internal filter underperforms the raw baseline materially
  - filter-only ROI is also non-positive

## What To Monitor

- `internal_filter` vs `raw_edge_baseline`
- `filter_only` vs `raw_only`
- full live sample
- `Feb 15+`
- `March`
- conference tournaments
- NCAA tracked separately as `ncaa_caution`

## What Not To Do

- Do not recalibrate mid-season because of one bad week.
- Do not change the promoted `0.58` threshold off tiny live samples.
- Do not let NCAA-specific noise drive a general recalibration decision.
- Do not treat short-run ROI as enough evidence by itself; always compare sample size and composition first.

## Current Status

- Current recommendation from the live maintenance report: `no action`
- That is the intended default unless a real review threshold is crossed.
