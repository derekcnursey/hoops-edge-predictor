# Tournament Adjustment Benchmark

## Current Baseline
- NCAA sample: `335` games across `2019, 2022, 2023, 2024, 2025`.
- Current promoted NCAA margin MAE: `10.016`.
- Current promoted shortfall vs market favorite: `-2.38`.
- Current Round of 64 shortfall: `-3.10`.
- Current 15+ favorite shortfall: `-5.80`.

## Market-Aware Candidate
- Best pooled fixed NCAA market weight: `0.80`.
- Fixed market blend NCAA margin MAE: `9.416`.
- Fixed market blend shortfall vs market favorite: `-0.48`.
- Config-capped market blend (`0.55` market weight) NCAA margin MAE: `9.549`.
- Config-capped market blend shortfall vs market favorite: `-1.07`.
- Leave-one-season-out market blend NCAA margin MAE: `9.479`.
- Leave-one-season-out market weights by season: `2019=0.00, 2022=0.55, 2023=0.75, 2024=0.80, 2025=0.80`.
- Production use: display-only tournament spreads are appropriate because the candidate depends on the current market line.
- Betting research use: not appropriate as the main model signal because it intentionally shrinks toward market.

## Non-Market Fallback
- Best pooled fixed NCAA primary weight vs Torvik fallback: `0.35`.
- Fixed Torvik fallback NCAA margin MAE: `9.801`.
- Fixed Torvik fallback shortfall vs market favorite: `-2.03`.
- Config Torvik fallback (`0.35` primary weight) NCAA margin MAE: `9.801`.
- Config Torvik fallback shortfall vs market favorite: `-2.03`.
- Leave-one-season-out Torvik fallback NCAA margin MAE: `9.838`.
- Leave-one-season-out primary weights by season: `2019=1.00, 2022=0.35, 2023=0.35, 2024=0.35, 2025=0.35`.
- Production use: suitable as a core-path NCAA-only fallback because it does not consume market data.

## Recommendation
- Smallest production-safe improvement: add a tournament-only display blend for NCAA Tournament games only, with a capped configurable market weight.
- Core model path: keep the year-round model unchanged by default; if a non-market NCAA-only core adjustment is needed, use the Torvik fallback override instead of a separate tournament model.
- Do not use the market-aware blend as the primary betting-research signal.
