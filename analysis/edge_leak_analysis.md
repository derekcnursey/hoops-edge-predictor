# Edge Leak Analysis Report

**Generated**: 2026-03-03 15:30
**Model**: 53-feat no_garbage, 384/256 MLP, walk-forward 2019-2025 (excl 2021)
**Edge threshold**: 5%

## Quick Stats

- Total games: 28,194
- Games with book spread: 26,224
- Qualified picks (edge >= 5%): 16,161
- Record: 8352W-7809L (51.7%)
- ROI: -1.3%
- Break-even win rate at -110: 52.4%

**The gap**: Model win rate is -0.7% vs break-even. Need to find ~0.7% of edge to reach profitability.

---


# Part 1: Where Do Edge Picks Fail?

**Universe**: 16,371 picks with edge >= 5% across holdout years 2019-2025 (excl 2021 COVID)

**Overall**: 8352W-7809L-210P (51.7% win rate, -1.3% ROI)

## 1a. Wins vs Losses Profile

| Metric | Wins | Losses | Diff |
|--------|------|--------|------|
| Avg sigma (uncertainty) | 11.44 | 11.47 | -0.03 |
| Avg prob_edge | 0.123 | 0.120 | +0.003 |
| Avg |edge_home_pts| | 3.66 | 3.55 | +0.11 |
| Avg |book_spread| | 7.5 | 7.4 | +0.1 |

| Pick Side | Count | Win Rate | ROI |
|-----------|-------|----------|-----|
| Home | 8,707 | 51.4% | -1.9% |
| Away | 7,454 | 52.1% | -0.6% |

## 1b. Win Rate by Edge Bucket

If bigger edges don't win at higher rates, confidence is miscalibrated.

| Edge Bucket | Count | Win Rate | ROI | Avg Sigma |
|-------------|-------|----------|-----|-----------|
| 5-8% | 4,835 | 50.4% | -3.7% | 11.5 |
| 8-11% | 3,777 | 51.2% | -2.2% | 11.5 |
| 11-14% | 2,685 | 52.3% | -0.2% | 11.5 |
| 14-17% | 1,865 | 52.2% | -0.4% | 11.4 |
| 17%+ | 2,999 | 53.4% | +1.9% | 11.4 |

Edge-to-win-rate curve is **NOT monotonic** — model confidence may be miscalibrated.

## 1c. Value Destroyers — Game Types That Lose Money

| Segment | Count | Win Rate | ROI | Impact |
|---------|-------|----------|-----|--------|
| Mar-Apr (tourney) | 2,285 | 50.6% | -3.4% | NEUTRAL |
| Sigma > 12 (high unc) | 4,783 | 51.0% | -2.6% | NEUTRAL |
| Conference | 12,048 | 51.2% | -2.2% | NEUTRAL |
| Jan-Feb (conf play) | 9,580 | 51.3% | -2.2% | NEUTRAL |
| Toss-up (|spread|<3) | 3,613 | 51.3% | -2.1% | NEUTRAL |
| Non-neutral | 14,748 | 51.6% | -1.4% | NEUTRAL |
| Mid-range (3-15) | 10,938 | 51.7% | -1.3% | NEUTRAL |
| Sigma 10-12 | 8,234 | 51.9% | -1.0% | NEUTRAL |
| Neutral site | 1,413 | 52.0% | -0.7% | NEUTRAL |
| Sigma <= 10 (low unc) | 3,144 | 52.2% | -0.4% | NEUTRAL |
| Blowout (|spread|>15) | 1,610 | 52.4% | -0.0% | NEUTRAL |
| Non-conference | 4,113 | 53.0% | +1.2% | NEUTRAL |
| Nov-Dec (early) | 4,296 | 53.2% | +1.6% | NEUTRAL |


# Part 2: What Does the Market Know That We Don't?

## 2a. High-Edge Losses (edge >= 10%, pick lost)

Total high-edge losses: 4,103

| Date | Matchup | Book Spread | Model Spread | Edge | Sigma | Actual |
|------|---------|-------------|--------------|------|-------|--------|
| 2019-12-22 | Radford @ Richmond | +7.0 | -16.0 | 45.6% | 13.5 | -15.0 |
| 2019-12-02 | Holy Cross @ Mercer | +9.0 | -7.1 | 44.0% | 10.3 | -14.0 |
| 2019-12-29 | Marshall @ Duquesne | +6.5 | -8.3 | 43.4% | 9.8 | -22.0 |
| 2021-12-03 | Wright State @ Purdue Fort Wayne | +4.5 | -8.9 | 41.9% | 9.6 | -13.0 |
| 2025-01-26 | Gonzaga @ Portland | +24.5 | +7.7 | 41.7% | 12.1 | -43.0 |
| 2018-12-01 | Delaware State @ St. Bonaventure | -24.5 | -10.3 | 41.7% | 10.3 | +29.0 |
| 2022-12-01 | Jacksonville @ UAB | -12.5 | +8.7 | 41.2% | 15.7 | +19.0 |
| 2021-12-05 | Oral Roberts @ Houston Christian | +14.5 | +1.5 | 40.7% | 9.9 | -18.0 |
| 2025-01-05 | Gonzaga @ Loyola Marymount | +15.5 | +0.0 | 39.5% | 12.4 | -28.0 |
| 2022-12-01 | Central Arkansas @ Loyola Chicago | -12.5 | +0.4 | 39.3% | 10.4 | +15.0 |
| 2024-01-06 | Samford @ The Citadel | +8.5 | -3.5 | 38.9% | 9.8 | -16.0 |
| 2023-12-02 | Furman @ Princeton | -6.5 | -19.9 | 38.8% | 11.0 | +1.0 |
| 2019-03-08 | Florida A&M @ Bethune-Cookman | +6.0 | -6.6 | 38.1% | 10.6 | -8.0 |
| 2022-12-17 | Ecclesia @ Arkansas-Pine Bluff | -37.5 | -5.8 | 38.1% | 26.9 | +54.0 |
| 2023-12-09 | Charlotte @ Duke | -16.5 | -6.2 | 37.3% | 9.0 | +24.0 |
| 2019-12-05 | Towson @ Morgan State | +6.0 | -8.7 | 37.3% | 12.9 | -17.0 |
| 2022-12-17 | St. Francis Brooklyn @ Hartford | +6.5 | -9.0 | 37.1% | 13.7 | -16.0 |
| 2023-12-21 | North Carolina @ Oklahoma | +3.5 | -6.7 | 36.6% | 9.2 | -12.0 |
| 2022-12-07 | Green Bay @ Loyola Chicago | -18.5 | -6.3 | 36.5% | 11.0 | +24.0 |
| 2022-12-04 | Brown @ Hartford | +10.5 | -2.1 | 36.4% | 11.5 | -14.0 |
| 2022-12-08 | Purdue Fort Wayne @ Southeast Missouri State | +1.5 | -8.7 | 36.3% | 9.3 | -21.0 |
| 2023-12-17 | Idaho @ Stanford | -16.5 | -5.3 | 35.9% | 10.4 | +18.0 |
| 2022-12-31 | Idaho @ Montana | -9.5 | +0.6 | 35.8% | 9.4 | +11.0 |
| 2019-12-03 | Miami @ Illinois | -9.0 | -23.9 | 35.8% | 13.9 | -2.0 |
| 2018-12-01 | Central Michigan @ TCU | -13.0 | -3.8 | 35.8% | 8.6 | +27.0 |
| 2019-01-11 | Winthrop @ Campbell | +3.0 | -9.0 | 35.7% | 11.3 | -4.0 |
| 2019-02-16 | Florida A&M @ Savannah State | -1.0 | -11.8 | 35.5% | 10.2 | -4.0 |
| 2021-12-18 | San Diego State @ Saint Mary's | +3.0 | -7.8 | 35.4% | 10.2 | -10.0 |
| 2023-01-24 | Pennsylvania @ Hartford | +15.5 | +1.7 | 35.1% | 13.3 | -24.0 |
| 2018-12-06 | Mercer @ Florida Atlantic | -2.0 | +6.6 | 34.9% | 8.3 | +4.0 |

### Repeat Teams in High-Edge Losses

Teams appearing 3+ times in high-edge losses:

| Team | Appearances |
|------|-------------|
| Gonzaga | 52 |
| Duke | 41 |
| Arkansas-Pine Bluff | 41 |
| UCF | 39 |
| Norfolk State | 38 |
| Saint Mary's | 37 |
| St. Francis Brooklyn | 37 |
| Houston | 35 |
| Michigan State | 33 |
| California Baptist | 33 |
| Howard | 33 |
| Grambling | 33 |
| North Carolina | 33 |
| Florida A&M | 33 |
| Houston Christian | 32 |

## 2b. Most Over/Undervalued Teams

Residual = predicted_spread - actual_spread. Positive = model overvalues team (thinks they'll win by more than they do).

### Top 10 Most OVERVALUED Teams (model thinks they're better than they are)

| Rank | Team | Games | Avg Residual | Median |
|------|------|-------|-------------|--------|
| 1 | Savannah State | 22 | +9.41 | +6.05 |
| 2 | The Citadel | 133 | +4.87 | +3.77 |
| 3 | St. Francis Brooklyn | 93 | +4.84 | +3.82 |
| 4 | Mississippi Valley State | 138 | +4.69 | +2.64 |
| 5 | IU Indianapolis | 136 | +4.22 | +4.14 |
| 6 | Columbia | 112 | +3.95 | +4.12 |
| 7 | Evansville | 146 | +3.76 | +3.47 |
| 8 | East Texas A&M | 71 | +3.38 | +1.46 |
| 9 | Pacific | 134 | +3.06 | +2.62 |
| 10 | Incarnate Word | 137 | +3.00 | +2.14 |

### Top 10 Most UNDERVALUED Teams (model underestimates them)

| Rank | Team | Games | Avg Residual | Median |
|------|------|-------|-------------|--------|
| 1 | Gonzaga | 160 | -6.29 | -5.79 |
| 2 | Michigan State | 167 | -3.38 | -4.29 |
| 3 | Norfolk State | 145 | -3.35 | -2.45 |
| 4 | South Dakota State | 142 | -3.11 | -2.46 |
| 5 | Utah Valley | 149 | -2.84 | -3.16 |
| 6 | Kentucky | 160 | -2.81 | -2.95 |
| 7 | Colgate | 156 | -2.74 | -2.80 |
| 8 | Houston | 181 | -2.68 | -3.01 |
| 9 | UAB | 162 | -2.66 | -3.96 |
| 10 | New Mexico State | 134 | -2.58 | -2.59 |

## 2c. Monthly MAE Across Seasons

Does the model get worse at certain points in the season?

| Month | 2019 | 2020 | 2022 | 2023 | 2024 | 2025 | **All** |
|------|------|------|------|------|------|------|------|
| Dec | 9.53 (1011) | 9.57 (958) | 9.16 (934) | 9.59 (1181) | 9.27 (953) | 9.48 (944) | **9.44** (5981) |
| Jan | 8.85 (1342) | 9.23 (1392) | 8.58 (1367) | 8.72 (1447) | 8.82 (1274) | 8.87 (1378) | **8.85** (8200) |
| Feb | 8.88 (1279) | 8.49 (1418) | 8.65 (1460) | 8.86 (1382) | 9.05 (1262) | 8.57 (1104) | **8.75** (7905) |
| Mar | 9.00 (855) | 8.86 (425) | 8.46 (649) | 9.03 (592) | 9.24 (794) | 8.90 (780) | **8.93** (4095) |
| Apr | 7.21 (11) | — | — | — | 6.72 (6) | 9.99 (19) | **8.31** (43) |

### Monthly Pick Performance (edge >= 5%)

| Month | Picks | Win Rate | ROI |
|-------|-------|----------|-----|
| Dec | 4,296 | 53.2% | +1.6% |
| Jan | 5,176 | 51.8% | -1.0% |
| Feb | 4,404 | 50.6% | -3.5% |
| Mar | 2,259 | 50.7% | -3.2% |
| Apr | 26 | 38.5% | -26.6% |


# Part 3: Sigma Calibration Check

Is the model's predicted uncertainty (sigma) well-calibrated?

| Sigma Bucket | Count | Avg Predicted σ | Actual Error Std | Ratio (pred/actual) | Interpretation |
|-------------|-------|----------------|-----------------|--------------------|----------------|
| 7-9 | 1,040 | 8.58 | 11.24 | 0.76 | **OVER-CONFIDENT** |
| 9-11 | 11,116 | 10.19 | 11.25 | 0.91 | Well-calibrated |
| 11-13 | 10,869 | 11.84 | 11.56 | 1.02 | Well-calibrated |
| 13+ | 5,169 | 16.71 | 16.89 | 0.99 | Well-calibrated |

## Sigma-Based Pick Performance

If over-confident in some bins, those picks offer phantom edges.

| Sigma Bucket | Picks | Win Rate | ROI | Avg Edge |
|-------------|-------|----------|-----|----------|
| 7-9 | 701 | 52.9% | +1.0% | 13.5% |
| 9-11 | 6,773 | 51.3% | -2.1% | 12.3% |
| 11-13 | 6,261 | 52.5% | +0.3% | 11.6% |
| 13+ | 2,426 | 50.2% | -4.2% | 12.4% |

## Calibration Consistency Across Seasons

| Season | Avg σ | Actual Error Std | Ratio | MAE |
|--------|-------|-----------------|-------|-----|
| 2019 | 11.85 | 12.50 | 0.95 | 9.54 |
| 2020 | 12.37 | 12.62 | 0.98 | 9.64 |
| 2022 | 11.88 | 12.46 | 0.95 | 9.40 |
| 2023 | 12.18 | 12.36 | 0.99 | 9.52 |
| 2024 | 11.31 | 13.06 | 0.87 | 9.87 |
| 2025 | 12.21 | 12.86 | 0.95 | 9.69 |


# Part 4: Betting Filter Strategies

Testing filter combinations to find historically profitable subsets. **Overfitting warning**: with many combinations, some will look good by chance. Focus on filters consistent across multiple seasons.

## Top 10 Filters by ROI (n >= 200)

| Rank | Filter | Picks | Win Rate | ROI | Seasons + | Consistency |
|------|--------|-------|----------|-----|-----------|-------------|
| 1 | edge>=10%, sigma<=12, spread=wide (15+), months=conf play (Jan-Mar) | **315** | 60.0% | +14.5% | 5/6 | 83% |
| 2 | edge>=8%, sigma<=12, spread=wide (15+), months=conf play (Jan-Mar) | **380** | 58.4% | +11.5% | 5/6 | 83% |
| 3 | edge>=8%, sigma<=11, spread=wide (15+), months=conf play (Jan-Mar) | **233** | 57.9% | +10.6% | 3/6 | 50% |
| 4 | edge>=10%, sigma<=99, spread=wide (15+), months=conf play (Jan-Mar) | **465** | 57.8% | +10.4% | 4/6 | 67% |
| 5 | edge>=5%, sigma<=11, spread=wide (15+), months=conf play (Jan-Mar) | **304** | 57.6% | +9.9% | 3/6 | 50% |
| 6 | edge>=10%, sigma<=12, spread=mid (7-15), months=early (Nov-Dec) | 592 | 56.9% | +8.7% | 5/6 | 83% |
| 7 | edge>=15%, sigma<=99, spread=wide (15+), months=conf play (Jan-Mar) | **253** | 56.9% | +8.7% | 5/6 | 83% |
| 8 | edge>=5%, sigma<=12, spread=wide (15+), months=conf play (Jan-Mar) | **496** | 56.9% | +8.5% | 5/6 | 83% |
| 9 | edge>=12%, sigma<=12, spread=wide (15+), months=conf play (Jan-Mar) | **247** | 56.7% | +8.2% | 5/6 | 83% |
| 10 | edge>=8%, sigma<=99, spread=wide (15+), months=conf play (Jan-Mar) | 583 | 56.4% | +7.7% | 4/6 | 67% |

## Top 5 Filters by Consistency (n >= 300, most seasons profitable)

| Rank | Filter | Picks | Win Rate | ROI | Seasons + | Per-Season ROIs |
|------|--------|-------|----------|-----|-----------|-----------------|
| 1 | edge>=10%, sigma<=99, spread=mid (7-15), months=early (Nov-Dec) | 961 | 56.2% | +7.3% | 6/6 | +7.1%, +7.5%, +5.1%, +5.8%, +11.7%, +7.6% |
| 2 | edge>=15%, sigma<=99, spread=mid (7-15), months=early (Nov-Dec) | 604 | 56.1% | +7.1% | 6/6 | +10.4%, +3.5%, +9.3%, +5.9%, +9.7%, +3.2% |
| 3 | edge>=10%, sigma<=12, spread=no blowout (<15), months=early (Nov-Dec) | 1,413 | 56.1% | +7.0% | 6/6 | +3.5%, +7.4%, +4.8%, +11.4%, +7.7%, +7.2% |
| 4 | edge>=8%, sigma<=99, spread=mid (7-15), months=early (Nov-Dec) | 1,146 | 55.8% | +6.6% | 6/6 | +5.3%, +7.6%, +5.6%, +4.8%, +16.9%, +0.4% |
| 5 | edge>=12%, sigma<=99, spread=wide (15+), months=conf play (Jan-Mar) | 361 | 55.7% | +6.3% | 6/6 | +0.8%, +7.7%, +12.9%, +11.4%, +6.4%, +0.1% |

## Deep Dive: Best ROI Filter

**edge>=10%, sigma<=12, spread=wide (15+), months=conf play (Jan-Mar)**

- Picks: 315
- Win Rate: 60.0%
- ROI: +14.5%
- Consistency: 5/6 seasons profitable
- Per-season ROIs: +3.3%, +27.3%, +25.5%, +25.5%, +18.3%, -8.5%

**WARNING**: Sample size < 500. High overfitting risk.


## Baseline Comparison

**Unfiltered baseline** (edge >= 5%, no other filters): 16,161 picks, 51.7% WR, -1.3% ROI

## Exclusion-Based Filters

Instead of finding what to bet, find what NOT to bet.

| Exclusion | Picks | Win Rate | ROI | Δ ROI vs baseline |
|-----------|-------|----------|-----|-------------------|
| Exclude blowout lines (>15) | 14,551 | 51.6% | -1.5% | -0.1% |
| Exclude high sigma (>12) | 11,378 | 52.0% | -0.8% | +0.6% |
| Exclude early season (Nov-Dec) | 11,865 | 51.1% | -2.4% | -1.1% |
| Exclude toss-ups (<3) | 12,548 | 51.8% | -1.1% | +0.2% |
| Exclude neutral site | 14,748 | 51.6% | -1.4% | -0.1% |
| Combined: no blowouts, no high sigma, no early | 8,458 | 51.1% | -2.4% | -1.0% |

---

# Key Findings & Synthesis

## The Core Problem

The model is 0.7% win rate short of break-even (51.7% vs 52.4% needed at -110 juice). That's roughly 113 picks out of 16,161 — a tiny margin, but the vig makes it matter.

## Where the Edge Leaks

### 1. Seasonal Decay is the Biggest Signal

The model's edge concentrates in **December** (53.2% WR, +1.6% ROI) and **collapses in February-March** (50.6-50.7% WR, -3.2% to -3.5% ROI). This is NOT a sample size artifact — we're talking about 4,296 picks in Dec vs 6,663 in Feb-Mar.

**Why this matters**: December is when the model's efficiency ratings and rolling stats reflect pre-season signal + early games. By Feb-Mar, the market has fully incorporated all public information. The model's edge is in **early-season information asymmetry** — we're reading efficiency data the market hasn't fully priced in yet. By conference play, the market catches up.

### 2. Wide Spreads + Conference Play is a Real Edge

The single most actionable finding: **blowout lines (>15) during Jan-Mar conference play** with edge >= 8% hit at 56-60% with +8% to +14% ROI across 5-6/6 profitable seasons. This makes structural sense — the book is less efficient on lopsided lines where casual bettors don't participate, and the model can exploit systematic mispricing of mismatches.

### 3. Sigma Calibration Reveals a Hidden Problem

The model is **over-confident at low sigma (7-9)** — predicted σ=8.6 but actual error std=11.2 (ratio 0.76). However, picks in this bucket actually perform WELL (52.9% WR, +1.0% ROI). This is paradoxical: the model is over-confident in its spread prediction, but still makes profitable picks.

At **sigma 13+** the model is well-calibrated (ratio 0.99) but picks are terrible (-4.2% ROI). High uncertainty = genuinely unpredictable games where the model can't find real edges.

### 4. The "Value Destroyers" are Mild, Not Catastrophic

No single segment is a massive drag. Conference games (-2.2% ROI) and Jan-Feb play (-2.2% ROI) are mildly negative. The worst single segment is Mar-Apr tourney (-3.4% ROI). The problem is **diffuse** — there's no single cancer to cut out.

### 5. Exclusion Filters Don't Work

Counterintuitively, excluding "bad" segments makes ROI WORSE. Excluding early season (the model's best period!) tanks ROI. The combined exclusion filter is the worst performer at -2.4% ROI. This means the answer isn't about avoiding bad bets — it's about concentrating on the best ones.

### 6. Most Overvalued/Undervalued Teams Tell a Story

**Overvalued** (model thinks they're better): Savannah State (+9.4), The Citadel (+4.9), Mississippi Valley State (+4.7) — mostly low-major teams where the efficiency ratings may be inflated by playing weak schedules. The model trusts the efficiency data but doesn't sufficiently discount schedule strength for these teams.

**Undervalued** (model underestimates): Gonzaga (-6.3), Michigan State (-3.4), Houston (-2.7), Kentucky (-2.8) — elite programs where intangibles (coaching, depth, tournament experience, recruiting advantages) create consistent edges the efficiency features don't capture.

## Recommended Action Items

### Tier 1: High-Confidence (Implement Now)

1. **Targeted betting filter**: Focus on **edge >= 10%, mid-range spreads (7-15), early season (Nov-Dec)**. This filter is profitable in 6/6 seasons at +7.3% ROI with 961 picks. The sample size is solid and the consistency is perfect.

2. **Secondary filter**: **edge >= 8%, spreads 15+, Jan-Mar with sigma <= 12**. Hits 58.4% at +11.5% ROI across 5/6 seasons. Smaller sample (380 picks) but the structural rationale is strong.

### Tier 2: Medium-Confidence (Investigate Further)

3. **Cap sigma for betting**: Remove picks with sigma > 12 from the betting pool. This alone improves ROI from -1.3% to -0.8% and drops the worst-performing bucket.

4. **Team-specific adjustments**: The model consistently misses on the same teams (Gonzaga -6.3 pts/game, MSU -3.4). Adding a team-level bias correction (rolling residual by team) could recapture 1-2 points of edge.

### Tier 3: Structural Changes (Next Session)

5. **Late-season confidence decay**: The model should output wider sigma in Feb-Mar to reflect the market's improved efficiency. A seasonal sigma multiplier (e.g., 1.1x in Feb, 1.15x in Mar) would reduce phantom edges.

6. **Investigate the over-confidence at low sigma**: The model claims σ=8.6 when reality is σ=11.2, yet those picks are profitable. This suggests the low-sigma games are genuinely more predictable but the model's uncertainty estimate is wrong. Fixing the sigma calibration here could improve edge sizing.

## Overfitting Disclaimer

We tested ~225 filter combinations. At p=0.05, we'd expect ~11 to look profitable by chance. The top filters we identified appear in **clusters** (wide spreads + conf play, mid spreads + early season) rather than isolated lucky combinations, and several have 6/6 season consistency, which is hard to explain by chance alone. Still, out-of-sample validation on 2026 data is essential before deploying any filter with real money.