# Conference Tournament Analysis -- 2026

## Methodology

This analysis compares our model's conference tournament win probabilities against Vegas futures odds across all 31 Division I conferences. Model probabilities were derived from **50,000 Monte Carlo simulations** per conference tournament, using the retrained production model with **symmetric home/away augmentation** and neutral-site averaging to eliminate any residual home/away bias. Each simulation plays out the **real tournament bracket** round-by-round — including proper bye structures, stepladder formats, and reduced fields where applicable — sampling game outcomes from the model's predicted spread distributions.

**Odds Source:** Hard Rock Bet (Florida) where available. Other sportsbook odds used as reference where HRB lines were not found.

**Edge Calculation:**
- `Edge = Model Implied% - Vegas Implied%`
- Positive edge means the model assigns a higher probability than Vegas (potential value bet)
- Negative edge means Vegas is shorter than the model suggests (potential fade)

**Vegas Implied Probability Conversion:**
- Positive American odds: `Implied% = 100 / (odds + 100)`
- Negative American odds: `Implied% = |odds| / (|odds| + 100)`
- Note: Vegas implied probabilities include vig and will sum to >100%

**Flags:**
- **VALUE**: Edge > +3.0%
- **STRONG VALUE**: Edge > +5.0%
- **FADE**: Edge < -5.0%

---

## Conference-by-Conference Analysis

---

### ACC (15 of 18 teams qualify)

**Odds Source: Hard Rock Bet (Florida)**

| Seed | Team | Conf Record | Model% | Model Odds | HRB Odds | Vegas Implied% | Edge | Flag |
|------|------|-------------|--------|------------|----------|----------------|------|------|
| 1 | Duke | 17-1 | 39.4% | +154 | -325 | 76.5% | -37.1% | FADE |
| 2 | Virginia | 15-3 | 15.0% | +567 | +700 | 12.5% | +2.5% | |
| 3 | Miami | 13-5 | 10.6% | +842 | +2000 | 4.8% | +5.8% | STRONG VALUE |
| 4 | North Carolina | 12-6 | 10.9% | +817 | +2000 | 4.8% | +6.1% | STRONG VALUE |
| 5 | Clemson | 12-6 | 4.7% | +2028 | +3000 | 3.2% | +1.5% | |
| 6 | Louisville | 11-7 | 8.1% | +1142 | +1000 | 9.1% | -1.0% | |
| 7 | NC State | 10-8 | 4.0% | +2421 | +4000 | 2.4% | +1.6% | |
| 8 | Florida State | 10-8 | 1.3% | +7363 | +10000 | 1.0% | +0.3% | |
| 9 | California | 9-9 | 1.3% | +7592 | +15000 | 0.7% | +0.6% | |
| 10 | Stanford | 9-9 | 1.1% | +9246 | +15000 | 0.7% | +0.4% | |
| 11 | SMU | 8-10 | 1.5% | +6549 | +10000 | 1.0% | +0.5% | |
| 12 | Virginia Tech | 8-10 | 1.1% | +9334 | +15000 | 0.7% | +0.4% | |
| 13 | Wake Forest | 7-11 | 0.4% | +23823 | +20000 | 0.5% | -0.1% | |
| 14 | Syracuse | 6-12 | 0.4% | +22322 | +25000 | 0.4% | +0.0% | |
| 15 | Pittsburgh | 5-13 | 0.2% | +42273 | +25000 | 0.4% | -0.2% | |
| -- | Notre Dame | 4-14 | DNQ | --- | --- | --- | --- | |
| -- | Boston College | 4-14 | DNQ | --- | --- | --- | --- | |
| -- | Georgia Tech | 2-16 | DNQ | --- | --- | --- | --- | |

**Analysis:** Duke is the massive fade at -37.1% edge on HRB. The model gives Duke 39.4% in a 15-team bracket with real bye structure, while HRB prices them at -325 (76.5% implied). **North Carolina (+2000)** and **Miami (+2000)** are both STRONG VALUE at +6.1% and +5.8% edge respectively — HRB has them at very long odds while the model sees both as ~11% contenders. Louisville at +1000 is slightly overpriced by HRB relative to the model's 8.1%.

---

### Big Ten (18 teams)

**Odds Source: FanDuel Sportsbook**

| Seed | Team | Conf Record | Model% | Model Odds | FD Odds | Vegas Implied% | Edge | Flag |
|------|------|-------------|--------|------------|---------|----------------|------|------|
| 1 | Michigan | 19-1 | 30.8% | +225 | -105 | 51.2% | -20.4% | FADE |
| 2 | Illinois | 15-5 | 19.8% | +406 | +390 | 20.4% | -0.6% | |
| 3 | Michigan State | 15-5 | 15.0% | +567 | +600 | 14.3% | +0.7% | |
| 4 | Nebraska | 15-5 | 8.4% | +1097 | +1200 | 7.7% | +0.7% | |
| 5 | Wisconsin | 14-6 | 4.9% | +1957 | +2700 | 3.6% | +1.3% | |
| 6 | Purdue | 13-7 | 10.0% | +905 | +750 | 11.8% | -1.8% | |
| 7 | UCLA | 13-7 | 2.2% | +4352 | +5000 | 2.0% | +0.2% | |
| 8 | Ohio State | 12-8 | 2.7% | +3582 | +10000 | 1.0% | +1.7% | |
| 9 | Iowa | 10-10 | 2.5% | +3868 | +6500 | 1.5% | +1.0% | |
| 10 | Indiana | 9-11 | 1.5% | +6436 | +17500 | 0.6% | +0.9% | |
| 11 | Minnesota | 8-12 | 0.4% | +22221 | +30000 | 0.3% | +0.1% | |
| 12 | Washington | 7-13 | 0.9% | +10962 | +30000 | 0.3% | +0.6% | |
| 13 | USC | 7-13 | 0.5% | +18350 | +30000 | 0.3% | +0.2% | |
| 14 | Rutgers | 6-14 | 0.0% | +99999 | +50000 (est) | 0.2% | -0.2% | |
| 15 | Northwestern | 5-15 | 0.2% | +44148 | +30000 | 0.3% | -0.1% | |
| 16 | Oregon | 5-15 | 0.1% | +99999 | +30000 | 0.3% | -0.2% | |
| 17 | Maryland | 4-16 | 0.0% | +99999 | +75000 (est) | 0.1% | -0.1% | |
| 18 | Penn State | 3-17 | 0.0% | +99999 | +100000 (est) | 0.1% | -0.1% | |

**Analysis:** **Michigan (-105)** is a fade at -20.4% but more moderate than other top-seed fades — the Big Ten's 18-team bracket means only 4 wins to the title with double byes. FanDuel prices the Big Ten more efficiently than most conferences. No teams cross the VALUE or FADE thresholds beyond Michigan. Purdue at +750 (11.8% implied) is slightly overpriced vs the model's 10.0%. **Ohio State (+10000)** at +1.7% edge is the best model-over-Vegas gap but doesn't reach the VALUE threshold.

---

### Big 12 (16 teams)

**Odds Source: Hard Rock Bet (Florida) for top 5, other sportsbooks for remainder**

| Seed | Team | Conf Record | Model% | Model Odds | Vegas Odds | Vegas Implied% | Edge | Flag |
|------|------|-------------|--------|------------|------------|----------------|------|------|
| 1 | Arizona | 16-2 | 32.7% | +206 | +105 (HRB) | 48.8% | -16.1% | FADE |
| 2 | Houston | 14-4 | 21.0% | +375 | +185 (HRB) | 35.1% | -14.1% | FADE |
| 3 | Iowa State | 12-6 | 15.3% | +555 | +800 (HRB) | 11.1% | +4.2% | VALUE |
| 4 | Kansas | 12-6 | 8.9% | +1018 | +800 (HRB) | 11.1% | -2.2% | |
| 5 | Texas Tech | 12-6 | 8.1% | +1137 | +1200 (HRB) | 7.7% | +0.4% | |
| 6 | TCU | 11-7 | 2.9% | +3344 | +2500 | 3.8% | -0.9% | |
| 7 | BYU | 9-9 | 4.3% | +2243 | +2000 | 4.8% | -0.5% | |
| 8 | UCF | 9-9 | 1.5% | +6712 | +5000 | 2.0% | -0.5% | |
| 9 | Cincinnati | 9-9 | 2.1% | +4551 | +3000 | 3.2% | -1.1% | |
| 10 | West Virginia | 9-9 | 0.6% | +16511 | +5000 | 2.0% | -1.4% | |
| 11 | Colorado | 7-11 | 0.4% | +24172 | +10000 | 1.0% | -0.6% | |
| 12 | Arizona State | 7-11 | 0.6% | +16735 | +12000 | 0.8% | -0.2% | |
| 13 | Baylor | 6-12 | 1.0% | +9685 | +8000 | 1.2% | -0.2% | |
| 14 | Oklahoma State | 6-12 | 0.4% | +27833 | +15000 | 0.7% | -0.3% | |
| 15 | Kansas State | 3-15 | 0.1% | +81867 | +25000 | 0.4% | -0.3% | |
| 16 | Utah | 2-16 | 0.1% | +97939 | +50000 | 0.2% | -0.1% | |

**Analysis:** HRB has Arizona at +105 (48.8% implied) and Houston at +185 (35.1% implied) — both are massive fades. Combined they account for 83.9% of HRB implied probability but only 53.7% per the model. **Iowa State (+800 HRB)** is the clear value play at +4.2% edge — the model gives them 15.3% vs only 11.1% implied. The real bracket structure with double byes for top 4 seeds and single byes for 5-8 creates significant advantage for the favorites, but not enough to justify HRB's aggressive pricing.

---

### SEC (16 teams)

**Odds Source: FanDuel Sportsbook**

| Seed | Team | Conf Record | Model% | Model Odds | FD Odds | Vegas Implied% | Edge | Flag |
|------|------|-------------|--------|------------|---------|----------------|------|------|
| 1 | Florida | 16-2 | 30.5% | +228 | -180 | 64.3% | -33.8% | FADE |
| 2 | Alabama | 13-5 | 12.8% | +682 | +750 | 11.8% | +1.0% | |
| 3 | Arkansas | 13-5 | 14.9% | +570 | +600 | 14.3% | +0.6% | |
| 4 | Tennessee | 11-7 | 16.2% | +517 | +1700 | 5.6% | +10.6% | STRONG VALUE |
| 5 | Vanderbilt | 11-7 | 6.9% | +1357 | +1200 | 7.7% | -0.8% | |
| 6 | Texas A&M | 11-7 | 3.6% | +2641 | +5000 | 2.0% | +1.6% | |
| 7 | Georgia | 10-8 | 2.8% | +3426 | +3500 | 2.8% | +0.0% | |
| 8 | Kentucky | 10-8 | 3.8% | +2548 | +4500 | 2.2% | +1.6% | |
| 9 | Missouri | 10-8 | 1.1% | +9352 | +10000 | 1.0% | +0.1% | |
| 10 | Texas | 9-9 | 3.0% | +3233 | +10000 | 1.0% | +2.0% | |
| 11 | Auburn | 7-11 | 2.8% | +3439 | +15000 | 0.7% | +2.1% | |
| 12 | Oklahoma | 7-11 | 0.7% | +13414 | +12500 | 0.8% | -0.1% | |
| 13 | Mississippi State | 5-13 | 0.1% | +72364 | +30000 | 0.3% | -0.2% | |
| 14 | South Carolina | 4-14 | 0.1% | +99999 | +30000 (est) | 0.3% | -0.2% | |
| 15 | Ole Miss | 4-14 | 0.4% | +25281 | +30000 | 0.3% | +0.1% | |
| 16 | LSU | 3-15 | 0.2% | +41917 | +30000 | 0.3% | -0.1% | |

**Analysis:** **Tennessee (+1700)** is a massive STRONG VALUE at +10.6% edge — the model rates the Vols as the 2nd-best team (16.2%) while FanDuel prices them at just 5.6% implied. This is the single best value play in any major conference. **Florida (-180)** is conversely a massive fade at -33.8% — FanDuel prices them at 64.3% implied vs the model's 30.5%. The SEC odds shifted dramatically from preseason, with Auburn dropping from short odds to +15000 (reflecting their 7-11 conference record). Arkansas at +600 is now fairly priced per the model.

---

### Big East (11 teams)

**Odds Source: FanDuel / SportsBettingDime**

| Seed | Team | Conf Record | Model% | Model Odds | FD Odds | Vegas Implied% | Edge | Flag |
|------|------|-------------|--------|------------|---------|----------------|------|------|
| 1 | St. John's | 18-2 | 28.2% | +255 | +190 | 34.5% | -6.3% | FADE |
| 2 | UConn | 17-3 | 32.6% | +207 | -115 | 53.5% | -20.9% | FADE |
| 3 | Villanova | 15-5 | 12.8% | +682 | +700 | 12.5% | +0.3% | |
| 4 | Seton Hall | 10-10 | 9.2% | +986 | +1700 | 5.6% | +3.6% | VALUE |
| 5 | Creighton | 9-11 | 4.3% | +2234 | +5000 | 2.0% | +2.3% | |
| 6 | DePaul | 8-12 | 2.2% | +4458 | +10000 | 1.0% | +1.2% | |
| 7 | Butler | 7-13 | 2.8% | +3469 | +10000 | 1.0% | +1.8% | |
| 8 | Providence | 7-13 | 2.0% | +4925 | +6500 | 1.5% | +0.5% | |
| 9 | Marquette | 7-13 | 2.3% | +4317 | +6500 | 1.5% | +0.8% | |
| 10 | Xavier | 6-14 | 1.4% | +7074 | +20000 | 0.5% | +0.9% | |
| 11 | Georgetown | 6-14 | 2.3% | +4255 | +10000 | 1.0% | +1.3% | |

**Analysis:** **UConn (-115)** is a massive fade at -20.9% — FanDuel makes them the clear favorite at 53.5% implied despite being the 2-seed, while the model gives them 32.6%. The model rates UConn higher than St. John's but not nearly at the level FanDuel implies. **Seton Hall (+1700)** is the standout value play at +3.6% edge. The Big East lower seeds (Butler +10000, Georgetown +10000) all show mild model edges.

---

### WCC (12 teams — stepladder format, seeds 1-2 enter at semifinals)

**Odds Source: Other sportsbooks (HRB lines not found)**

| Seed | Team | Conf Record | Model% | Model Odds | Vegas Odds | Vegas Implied% | Edge | Flag |
|------|------|-------------|--------|------------|------------|----------------|------|------|
| 1 | Gonzaga | 17-2 | 49.9% | +100 | -200 | 66.7% | -16.8% | FADE |
| 2 | Santa Clara | 17-3 | 24.0% | +316 | +500 | 16.7% | +7.3% | STRONG VALUE |
| 3 | Saint Mary's | 16-3 | 19.1% | +422 | +350 | 22.2% | -3.1% | |
| 4 | Oregon State | 10-10 | 1.5% | +6410 | +3000 | 3.2% | -1.7% | |
| 5 | San Francisco | 9-11 | 1.9% | +5191 | +4000 | 2.4% | -0.5% | |
| 6 | Pacific | 9-11 | 1.2% | +8206 | +10000 | 1.0% | +0.2% | |
| 7 | Seattle U | 9-11 | 1.0% | +9704 | +5000 | 2.0% | -1.0% | |
| 8 | Portland | 8-13 | 0.2% | +40884 | +25000 | 0.4% | -0.2% | |
| 9 | Washington State | 7-12 | 0.8% | +12526 | +8000 | 1.2% | -0.4% | |
| 10 | Loyola Marymount | 6-13 | 0.2% | +59424 | +15000 | 0.7% | -0.5% | |
| 11 | San Diego | 6-14 | 0.0% | +99999 | +50000 | 0.2% | -0.2% | |
| 12 | Pepperdine | 4-15 | 0.0% | +99999 | +75000 | 0.1% | -0.1% | |

**Analysis:** With the real stepladder bracket (1-2 seeds enter at semis, only 2 wins needed), Gonzaga reaches 49.9% — still well below Vegas -200 (66.7%). **Santa Clara (+500)** is the standout STRONG VALUE at +7.3% edge — the model gives them 24.0% as the 2-seed entering at the semifinal round, while Vegas implies only 16.7%. Saint Mary's as the 3-seed must win 3 games, which drops them to 19.1%.

---

### Mountain West (12 teams)

**Odds Source: FanDuel Sportsbook (via MWC Connection)**

| Seed | Team | Conf Record | Model% | Model Odds | FD Odds | Vegas Implied% | Edge | Flag |
|------|------|-------------|--------|------------|---------|----------------|------|------|
| 1 | Utah State | 15-5 | 21.4% | +367 | +185 | 35.1% | -13.7% | FADE |
| 2 | San Diego State | 14-6 | 21.6% | +364 | +350 | 22.2% | -0.6% | |
| 3 | New Mexico | 13-7 | 15.5% | +545 | +500 | 16.7% | -1.2% | |
| 4 | Grand Canyon | 13-7 | 14.5% | +588 | +700 | 12.5% | +2.0% | |
| 5 | Boise State | 12-8 | 10.6% | +840 | +700 | 12.5% | -1.9% | |
| 6 | Nevada | 12-8 | 6.6% | +1406 | +1600 | 5.9% | +0.7% | |
| 7 | Colorado State | 11-9 | 3.1% | +3122 | +2000 | 4.8% | -1.7% | |
| 8 | UNLV | 11-9 | 2.6% | +3749 | +4000 | 2.4% | +0.2% | |
| 9 | Wyoming | 9-11 | 2.2% | +4344 | +5000 | 2.0% | +0.2% | |
| 10 | Fresno State | 7-13 | 1.4% | +6903 | +15000 | 0.7% | +0.7% | |
| 11 | San José State | 3-17 | 0.3% | +33233 | +25000 | 0.4% | -0.1% | |
| 12 | Air Force | 0-20 | 0.0% | +99999 | +25000 | 0.4% | -0.4% | |

**Analysis:** **Utah State (+185)** is a major fade at -13.7% edge — FanDuel makes them the heavy favorite at 35.1% implied while the model gives them only 21.4%. With updated FanDuel odds, SDSU and New Mexico are now fairly priced (previously overpriced with old sportsbook lines). No strong value plays in this conference. The Mountain West top tier is tightly bunched per the model (21.6% / 21.4% / 15.5% / 14.5%).

---

### A-10 (14 teams)

**Odds Source: Other sportsbooks (HRB lines not found)**

| Seed | Team | Conf Record | Model% | Model Odds | Vegas Odds | Vegas Implied% | Edge | Flag |
|------|------|-------------|--------|------------|------------|----------------|------|------|
| 1 | Saint Louis | 15-3 | 30.3% | +230 | +200 | 33.3% | -3.0% | |
| 2 | VCU | 15-3 | 21.7% | +361 | +300 | 25.0% | -3.3% | |
| 3 | Saint Joseph's | 13-5 | 9.0% | +1009 | +1500 | 6.3% | +2.7% | |
| 4 | Dayton | 12-6 | 15.3% | +553 | +500 | 16.7% | -1.4% | |
| 5 | George Mason | 11-7 | 2.4% | +4043 | +2000 | 4.8% | -2.4% | |
| 6 | Davidson | 10-8 | 6.2% | +1518 | +1000 | 9.1% | -2.9% | |
| 7 | Duquesne | 9-9 | 3.3% | +2910 | +3000 | 3.2% | +0.1% | |
| 8 | George Washington | 8-10 | 6.6% | +1415 | +2500 | 3.8% | +2.8% | |
| 9 | Fordham | 8-10 | 1.0% | +9920 | +5000 | 2.0% | -1.0% | |
| 10 | Rhode Island | 7-11 | 2.1% | +4708 | +8000 | 1.2% | +0.9% | |
| 11 | Richmond | 5-13 | 0.9% | +10584 | +10000 | 1.0% | -0.1% | |
| 12 | La Salle | 5-13 | 0.5% | +18557 | +15000 | 0.7% | -0.2% | |
| 13 | St. Bonaventure | 4-14 | 0.5% | +19741 | +20000 | 0.5% | +0.0% | |
| 14 | Loyola Chicago | 4-14 | 0.1% | +94240 | +30000 | 0.3% | -0.2% | |

**Analysis:** The A-10 is efficiently priced overall. No teams cross the VALUE or FADE thresholds. **George Washington (+2500)** shows the best model edge at +2.8%. The top two (Saint Louis and VCU) are slightly overpriced. **Saint Joseph's (+1500)** at +2.7% and **Dayton (+500)** fairly priced in the updated simulation.

---

### American (10 of 13 teams qualify — stepladder format)

**Odds Source: Other sportsbooks (HRB lines not found)**

| Seed | Team | Conf Record | Model% | Model Odds | Vegas Odds | Vegas Implied% | Edge | Flag |
|------|------|-------------|--------|------------|------------|----------------|------|------|
| 1 | South Florida | 15-3 | 41.2% | +143 | +175 | 36.4% | +4.8% | VALUE |
| 2 | Tulsa | 13-5 | 28.3% | +253 | +350 | 22.2% | +6.1% | STRONG VALUE |
| 3 | Wichita State | 13-5 | 14.7% | +579 | +400 | 20.0% | -5.3% | FADE |
| 4 | UAB | 11-7 | 5.7% | +1661 | +1500 | 6.3% | -0.6% | |
| 5 | Florida Atlantic | 9-9 | 4.2% | +2269 | +1000 | 9.1% | -4.9% | |
| 6 | Charlotte | 9-9 | 0.9% | +10770 | +3000 | 3.2% | -2.3% | |
| 7 | North Texas | 9-9 | 1.8% | +5468 | +2000 | 4.8% | -3.0% | |
| 8 | Temple | 8-10 | 0.6% | +17694 | +4000 | 2.4% | -1.8% | |
| 9 | Memphis | 8-10 | 2.2% | +4504 | +800 | 11.1% | -8.9% | FADE |
| 10 | Tulane | 8-10 | 0.4% | +24776 | +5000 | 2.0% | -1.6% | |
| -- | Rice | 7-11 | DNQ | --- | --- | --- | --- | |
| -- | East Carolina | 6-12 | DNQ | --- | --- | --- | --- | |
| -- | UTSA | 1-17 | DNQ | --- | --- | --- | --- | |

**Analysis:** The stepladder format massively benefits the top 2 seeds — **South Florida** at 41.2% and **Tulsa** at 28.3% combine for nearly 70% of model probability. **Tulsa (+350)** is the standout STRONG VALUE at +6.1% edge. **Memphis (+800)** remains the biggest fade at -8.9% — Vegas gives them 11.1% implied on brand name alone despite an 8-10 record and 9-seed. **Wichita State** is also a fade at -5.3%.

---

### Sun Belt (14 teams — 7-round stepladder)

**Odds Source: Estimated from available sources**

| Seed | Team | Conf Record | Model% | Model Odds | Vegas Odds | Vegas Implied% | Edge | Flag |
|------|------|-------------|--------|------------|------------|----------------|------|------|
| 1 | Troy | 14-6 | 30.5% | +228 | +200 (est) | 33.3% | -2.8% | |
| 2 | Georgia Southern | 13-11 | 16.3% | +513 | +800 (est) | 11.1% | +5.2% | STRONG VALUE |
| 3 | Southern Miss | 12-10 | 12.6% | +693 | +700 (est) | 12.5% | +0.1% | |
| 4 | Arkansas State | 11-8 | 21.3% | +369 | +600 (est) | 14.3% | +7.0% | STRONG VALUE |
| 5 | Marshall | 11-8 | 4.4% | +2182 | +1000 (est) | 9.1% | -4.7% | |
| 6 | South Alabama | 11-8 | 4.7% | +2044 | +1100 | 8.3% | -3.6% | |
| 7 | App State | 11-8 | 4.4% | +2198 | +500 | 16.7% | -12.3% | FADE |
| 8 | Coastal Carolina | 11-8 | 1.7% | +5803 | +1500 (est) | 6.3% | -4.6% | |
| 9 | Texas State | 11-8 | 1.6% | +5983 | +1800 (est) | 5.3% | -3.7% | |
| 10 | James Madison | 10-10 | 1.5% | +6648 | +2000 (est) | 4.8% | -3.3% | |
| 11 | Old Dominion | 8-12 | 0.8% | +12989 | +2500 (est) | 3.8% | -3.0% | |
| 12 | Louisiana | 8-12 | 0.1% | +72364 | +5000 (est) | 2.0% | -1.9% | |
| 13 | Georgia State | 7-12 | 0.1% | +99999 | +8000 (est) | 1.2% | -1.1% | |
| 14 | UL Monroe | 1-18 | 0.0% | +99999 | +25000 (est) | 0.4% | -0.4% | |

**Analysis:** The 7-round stepladder format dramatically benefits seeds 1-2. **Arkansas State (+600 est)** is a STRONG VALUE at +7.0% — the model rates them as the 2nd-best team (21.3%) despite the 4-seed. **Georgia Southern (+800 est)** also shows strong value at +5.2%. **App State (+500)** is a massive fade at -12.3% — Vegas loved them but the model strongly disagrees. Troy at 30.5% benefits from the stepladder bye structure.

---

### CAA (13 teams)

**Odds Source: Other sportsbooks (HRB lines not found)**

| Seed | Team | Conf Record | Model% | Model Odds | Vegas Odds | Vegas Implied% | Edge | Flag |
|------|------|-------------|--------|------------|------------|----------------|------|------|
| 1 | UNC Wilmington | 15-4 | 20.5% | +389 | +450 | 18.2% | +2.3% | |
| 2 | Charleston | 14-5 | 13.4% | +648 | +400 | 20.0% | -6.6% | FADE |
| 3 | Hofstra | 14-6 | 22.9% | +336 | +250 | 28.6% | -5.7% | FADE |
| 4 | Monmouth | 13-7 | 12.9% | +674 | +1000 | 9.1% | +3.8% | VALUE |
| 5 | William & Mary | 11-9 | 8.3% | +1103 | +800 | 11.1% | -2.8% | |
| 6 | Drexel | 11-9 | 5.1% | +1858 | +1500 | 6.3% | -1.2% | |
| 7 | Towson | 11-10 | 5.6% | +1678 | +2000 | 4.8% | +0.8% | |
| 8 | Campbell | 10-11 | 4.8% | +1996 | +2500 | 3.8% | +1.0% | |
| 9 | Stony Brook | 9-10 | 2.8% | +3479 | +4000 | 2.4% | +0.4% | |
| 10 | Hampton | 7-12 | 1.2% | +7939 | +10000 | 1.0% | +0.2% | |
| 11 | Elon | 6-13 | 1.6% | +6237 | +8000 | 1.2% | +0.4% | |
| 12 | North Carolina A&T | 4-15 | 0.4% | +25541 | +15000 | 0.7% | -0.3% | |
| 13 | Northeastern | 3-17 | 0.5% | +20476 | +20000 | 0.5% | +0.0% | |

**Analysis:** **Hofstra** and **Charleston** are both fades — Vegas rates them much higher than the model. The model actually rates Hofstra as the best team (22.9%) but still sees them as overpriced at +250. **Monmouth (+1000)** is the standout value at +3.8% edge.

---

### MAC (13 teams)

**Odds Source: Other sportsbooks (HRB lines not found)**

| Seed | Team | Conf Record | Model% | Model Odds | Vegas Odds | Vegas Implied% | Edge | Flag |
|------|------|-------------|--------|------------|------------|----------------|------|------|
| 1 | Miami (OH) | 18-0 | 22.3% | +349 | +200 | 33.3% | -11.0% | FADE |
| 2 | Akron | 17-1 | 35.1% | +185 | +150 | 40.0% | -4.9% | |
| 3 | Kent State | 14-4 | 10.6% | +845 | +600 | 14.3% | -3.7% | |
| 4 | Toledo | 11-7 | 9.7% | +933 | +800 | 11.1% | -1.4% | |
| 5 | Bowling Green | 9-9 | 8.1% | +1138 | +1200 | 7.7% | +0.4% | |
| 6 | Ohio | 9-9 | 2.3% | +4290 | +2500 | 3.8% | -1.5% | |
| 7 | Buffalo | 7-11 | 1.8% | +5456 | +3000 | 3.2% | -1.4% | |
| 8 | Massachusetts | 7-11 | 5.1% | +1865 | +4000 | 2.4% | +2.7% | |
| 9 | Ball State | 7-11 | 0.8% | +13163 | +8000 | 1.2% | -0.4% | |
| 10 | Central Michigan | 6-12 | 1.6% | +6310 | +10000 | 1.0% | +0.6% | |
| 11 | Eastern Michigan | 4-14 | 2.3% | +4313 | +15000 | 0.7% | +1.6% | |
| 12 | Western Michigan | 4-14 | 0.4% | +25541 | +20000 | 0.5% | -0.1% | |
| 13 | Northern Illinois | 4-14 | 0.1% | +67468 | +30000 | 0.3% | -0.2% | |

**Analysis:** **Miami (OH)** went 18-0 but the model gives them only 22.3% — less than Akron's 35.1%. The model rates Akron as the better team despite having 1 conference loss. **Miami (OH) at +200** is a significant fade at -11.0%. **Massachusetts (+4000)** shows mild value at +2.7% edge.

---

### MAAC (10 of 13 teams qualify)

**Odds Source: Estimated from available sources**

| Seed | Team | Conf Record | Model% | Model Odds | Vegas Odds | Vegas Implied% | Edge | Flag |
|------|------|-------------|--------|------------|------------|----------------|------|------|
| 1 | Merrimack | 19-3 | 17.8% | +462 | +150 (est) | 40.0% | -22.2% | FADE |
| 2 | Siena | 15-7 | 15.0% | +565 | +400 (est) | 20.0% | -5.0% | |
| 3 | Saint Peter's | 14-7 | 12.2% | +721 | +500 (est) | 16.7% | -4.5% | |
| 4 | Marist | 13-9 | 16.3% | +513 | +450 (est) | 18.2% | -1.9% | |
| 5 | Fairfield | 13-10 | 8.7% | +1055 | +1200 (est) | 7.7% | +1.0% | |
| 6 | Quinnipiac | 12-9 | 14.8% | +578 | +800 (est) | 11.1% | +3.7% | VALUE |
| 7 | Mount St. Mary's | 11-10 | 5.9% | +1596 | +2000 (est) | 4.8% | +1.1% | |
| 8 | Iona | 10-11 | 5.8% | +1617 | +2500 (est) | 3.8% | +2.0% | |
| 9 | Sacred Heart | 10-12 | 2.3% | +4203 | +5000 | 2.0% | +0.3% | |
| 10 | Manhattan | 8-13 | 1.2% | +8043 | +8000 (est) | 1.2% | +0.0% | |
| -- | Canisius | 5-15 | DNQ | --- | --- | --- | --- | |
| -- | Niagara | 5-15 | DNQ | --- | --- | --- | --- | |
| -- | Rider | 3-17 | DNQ | --- | --- | --- | --- | |

**Analysis:** **Merrimack** at +150 (est) is a massive fade at -22.2% — the model distributes probability much more evenly across the top 8. **Quinnipiac (+800 est)** is the standout value play at +3.7% edge — the model rates them at 14.8%, well above the 6-seed. The MAAC's reduced 10-team field concentrates opportunity for mid-tier teams.

---

### MVC (11 teams)

**Odds Source: Other sportsbooks (HRB lines not found)**

| Seed | Team | Conf Record | Model% | Model Odds | Vegas Odds | Vegas Implied% | Edge | Flag |
|------|------|-------------|--------|------------|------------|----------------|------|------|
| 1 | Belmont | 16-5 | 18.9% | +430 | +160 | 38.5% | -19.6% | FADE |
| 2 | Northern Iowa | 15-9 | 17.4% | +476 | +450 | 18.2% | -0.8% | |
| 3 | Bradley | 14-8 | 12.1% | +728 | +650 | 13.3% | -1.2% | |
| 4 | UIC | 14-9 | 12.5% | +699 | +800 | 11.1% | +1.4% | |
| 5 | Illinois State | 12-9 | 15.3% | +552 | +500 | 16.7% | -1.4% | |
| 6 | Murray State | 12-9 | 7.3% | +1272 | +1100 | 8.3% | -1.0% | |
| 7 | Valparaiso | 12-10 | 5.3% | +1786 | +3500 | 2.8% | +2.5% | |
| 8 | Southern Illinois | 10-11 | 7.5% | +1226 | +1600 | 5.9% | +1.6% | |
| 9 | Drake | 8-15 | 1.1% | +8781 | +20000 | 0.5% | +0.6% | |
| 10 | Indiana State | 4-17 | 2.1% | +4726 | +12500 | 0.8% | +1.3% | |
| 11 | Evansville | 3-18 | 0.5% | +19585 | +35000 | 0.3% | +0.2% | |

**Analysis:** **Belmont (+160)** is a massive fade at -19.6% — Vegas prices them at 38.5% while the model gives just 18.9%. Illinois State is interesting — the model rates them as the 2nd-best team (15.3%) despite the 5-seed. **Valparaiso (+3500)** and **Southern Illinois (+1600)** show mild value.

---

### Southland (8 of 12 teams qualify)

**Odds Source: Other sportsbooks (HRB lines not found)**

| Seed | Team | Conf Record | Model% | Model Odds | Vegas Odds | Vegas Implied% | Edge | Flag |
|------|------|-------------|--------|------------|------------|----------------|------|------|
| 1 | Stephen F. Austin | 20-2 | 36.3% | +175 | +125 | 44.4% | -8.1% | FADE |
| 2 | McNeese | 19-3 | 40.1% | +149 | +115 | 46.5% | -6.4% | FADE |
| 3 | UT Rio Grande Valley | 15-8 | 10.4% | +857 | +700 | 12.5% | -2.1% | |
| 4 | Texas A&M-Corpus Christi | 14-9 | 7.5% | +1233 | +1600 | 5.9% | +1.6% | |
| 5 | New Orleans | 13-11 | 3.3% | +2889 | +3000 | 3.2% | +0.1% | |
| 6 | Nicholls | 13-11 | 1.3% | +7329 | +12500 | 0.8% | +0.5% | |
| 7 | Northwestern State | 8-15 | 0.5% | +19057 | +20000 | 0.5% | +0.0% | |
| 8 | Houston Christian | 8-15 | 0.4% | +24776 | +25000 | 0.4% | +0.0% | |
| -- | Lamar | 7-15 | DNQ | --- | --- | --- | --- | |
| -- | Incarnate Word | 7-15 | DNQ | --- | --- | --- | --- | |
| -- | East Texas A&M | 6-16 | DNQ | --- | --- | --- | --- | |
| -- | SE Louisiana | 6-16 | DNQ | --- | --- | --- | --- | |

**Analysis:** The reduced 8-team bracket concentrates probability on the top 2 seeds. McNeese (40.1%) edges SFA (36.3%) as the model's pick despite being the 2-seed. Both top seeds are fades vs Vegas. No strong value plays in the remaining field.

---

### CUSA (10 of 12 teams qualify)

**Odds Source: Other sportsbooks (HRB lines not found)**

| Seed | Team | Conf Record | Model% | Model Odds | Vegas Odds | Vegas Implied% | Edge | Flag |
|------|------|-------------|--------|------------|------------|----------------|------|------|
| 1 | Liberty | 17-3 | 11.7% | +756 | +310 | 24.4% | -12.7% | FADE |
| 2 | Sam Houston | 13-7 | 29.1% | +244 | +200 | 33.3% | -4.2% | |
| 3 | Middle Tennessee | 11-9 | 10.7% | +834 | +950 | 9.5% | +1.2% | |
| 4 | Western Kentucky | 11-9 | 16.5% | +506 | +700 | 12.5% | +4.0% | VALUE |
| 5 | Louisiana Tech | 11-9 | 5.6% | +1695 | +1100 | 8.3% | -2.7% | |
| 6 | Kennesaw State | 10-10 | 11.5% | +766 | +850 | 10.5% | +1.0% | |
| 7 | Jacksonville State | 10-10 | 2.7% | +3566 | +2800 | 3.4% | -0.7% | |
| 8 | Florida International | 8-12 | 4.9% | +1936 | +3000 | 3.2% | +1.7% | |
| 9 | Missouri State | 8-12 | 4.3% | +2229 | +2500 | 3.8% | +0.5% | |
| 10 | New Mexico State | 8-12 | 3.0% | +3220 | +3500 | 2.8% | +0.2% | |
| -- | UTEP | 7-13 | DNQ | --- | --- | --- | --- | |
| -- | Delaware | 6-14 | DNQ | --- | --- | --- | --- | |

**Analysis:** **Liberty (+310)** is the biggest fade at -12.7% — despite going 17-3, the model gives them only 11.7% (3rd behind Sam Houston and Western Kentucky). **Western Kentucky (+700)** is the value play at +4.0% edge — the model rates them at 16.5%, the 2nd-best team despite the 4-seed.

---

### ASUN (12 teams)

**Odds Source: Other sportsbooks (HRB lines not found)**

| Seed | Team | Conf Record | Model% | Model Odds | Vegas Odds | Vegas Implied% | Edge | Flag |
|------|------|-------------|--------|------------|------------|----------------|------|------|
| 1 | Central Arkansas | 17-4 | 20.7% | +383 | +165 | 37.7% | -17.0% | FADE |
| 2 | Austin Peay | 16-4 | 31.5% | +218 | +240 | 29.4% | +2.1% | |
| 3 | Queens University | 16-5 | 14.2% | +606 | +440 | 18.5% | -4.3% | |
| 4 | Lipscomb | 12-7 | 12.7% | +685 | +500 | 16.7% | -4.0% | |
| 5 | Florida Gulf Coast | 10-11 | 5.7% | +1640 | +2200 | 4.3% | +1.4% | |
| 6 | West Georgia | 9-11 | 3.8% | +2511 | +5500 | 1.8% | +2.0% | |
| 7 | Bellarmine | 8-12 | 1.2% | +7900 | +5500 | 1.8% | -0.6% | |
| 8 | Stetson | 8-12 | 2.0% | +4783 | +15000 | 0.7% | +1.3% | |
| 9 | Jacksonville | 7-12 | 3.3% | +2966 | +5500 | 1.8% | +1.5% | |
| 10 | Eastern Kentucky | 7-12 | 2.7% | +3595 | +5000 | 2.0% | +0.7% | |
| 11 | North Florida | 5-14 | 1.2% | +8233 | +20000 | 0.5% | +0.7% | |
| 12 | North Alabama | 4-15 | 0.9% | +11111 | +25000 | 0.4% | +0.5% | |

**Analysis:** **Central Arkansas (+165)** is a massive fade at -17.0%. The model rates **Austin Peay** as the best team (31.5%) despite being the 2-seed. Queens and Lipscomb are also somewhat overpriced. The model distributes probability more evenly across the middle tier than Vegas does.

---

### SWAC (12 teams)

**Odds Source: Estimated from available sources**

| Seed | Team | Conf Record | Model% | Model Odds | Vegas Odds | Vegas Implied% | Edge | Flag |
|------|------|-------------|--------|------------|------------|----------------|------|------|
| 1 | Bethune-Cookman | 14-4 | 27.0% | +271 | -150 (est) | 60.0% | -33.0% | FADE |
| 2 | Southern | 11-7 | 20.5% | +387 | +500 (est) | 16.7% | +3.8% | VALUE |
| 3 | Florida A&M | 11-7 | 12.0% | +731 | +600 (est) | 14.3% | -2.3% | |
| 4 | Alabama A&M | 10-8 | 8.8% | +1042 | +1200 (est) | 7.7% | +1.1% | |
| 5 | Arkansas-Pine Bluff | 10-8 | 6.8% | +1360 | +1500 (est) | 6.3% | +0.5% | |
| 6 | Texas Southern | 10-8 | 5.2% | +1835 | +2000 (est) | 4.8% | +0.4% | |
| 7 | Jackson State | 10-8 | 2.6% | +3691 | +3000 (est) | 3.2% | -0.6% | |
| 8 | Prairie View A&M | 9-9 | 3.7% | +2572 | +4000 (est) | 2.4% | +1.3% | |
| 9 | Grambling | 8-11 | 5.6% | +1691 | +5000 (est) | 2.0% | +3.6% | VALUE |
| 10 | Alcorn State | 8-11 | 1.9% | +5028 | +8000 (est) | 1.2% | +0.7% | |
| 11 | Alabama State | 7-12 | 5.2% | +1828 | +6000 (est) | 1.6% | +3.6% | VALUE |
| 12 | Mississippi Valley State | 2-17 | 0.6% | +16456 | +25000 (est) | 0.4% | +0.2% | |

**Analysis:** Bethune-Cookman is dramatically overpriced at -33.0% edge. **Southern**, **Grambling**, and **Alabama State** all show value. Note: SWAC odds are largely estimated and should be treated as directional only.

---

### Big West (8 of 11 teams qualify)

**Odds Source: Other sportsbooks (HRB lines not found)**

| Seed | Team | Conf Record | Model% | Model Odds | Vegas Odds | Vegas Implied% | Edge | Flag |
|------|------|-------------|--------|------------|------------|----------------|------|------|
| 1 | UC Irvine | 15-5 | 33.0% | +203 | +205 | 32.8% | +0.2% | |
| 2 | Hawai'i | 14-6 | 31.5% | +218 | +300 (est) | 25.0% | +6.5% | STRONG VALUE |
| 3 | UC San Diego | 12-8 | 11.8% | +750 | +500 (est) | 16.7% | -4.9% | |
| 4 | Cal State Fullerton | 12-8 | 5.2% | +1817 | +1200 (est) | 7.7% | -2.5% | |
| 5 | Cal State Northridge | 12-8 | 7.2% | +1295 | +800 (est) | 11.1% | -3.9% | |
| 6 | UC Santa Barbara | 11-9 | 6.3% | +1494 | +225 | 30.8% | -24.5% | FADE |
| 7 | UC Davis | 11-9 | 3.4% | +2855 | +1500 (est) | 6.3% | -2.9% | |
| 8 | Cal Poly | 10-10 | 1.8% | +5569 | +3000 (est) | 3.2% | -1.4% | |
| -- | Long Beach State | 6-14 | DNQ | --- | --- | --- | --- | |
| -- | UC Riverside | 5-15 | DNQ | --- | --- | --- | --- | |
| -- | Cal State Bakersfield | 2-18 | DNQ | --- | --- | --- | --- | |

**Analysis:** **UC Santa Barbara (+225)** is one of the biggest fades in the entire analysis at -24.5% — Vegas has them essentially co-favorite despite the 6-seed, while the model gives them just 6.3%. **Hawai'i (+300 est)** is a STRONG VALUE at +6.5% — the model rates them nearly equal to UC Irvine (31.5% vs 33.0%). The reduced 8-team field benefits the top 2 seeds significantly.

---

### OVC (8 of 11 teams qualify)

**Odds Source: Other sportsbooks (HRB lines not found)**

| Seed | Team | Conf Record | Model% | Model Odds | Vegas Odds | Vegas Implied% | Edge | Flag |
|------|------|-------------|--------|------------|------------|----------------|------|------|
| 1 | Tennessee State | 17-5 | 28.7% | +248 | +190 | 34.5% | -5.8% | FADE |
| 2 | Morehead State | 16-6 | 17.6% | +467 | +425 | 19.0% | -1.4% | |
| 3 | Southeast Missouri State | 15-7 | 19.6% | +410 | +400 | 20.0% | -0.4% | |
| 4 | UT Martin | 14-8 | 14.1% | +608 | +360 | 21.7% | -7.6% | FADE |
| 5 | SIU Edwardsville | 12-9 | 7.0% | +1322 | +800 (est) | 11.1% | -4.1% | |
| 6 | Lindenwood | 12-10 | 8.9% | +1027 | +1000 (est) | 9.1% | -0.2% | |
| 7 | Little Rock | 9-12 | 2.0% | +5023 | +3000 (est) | 3.2% | -1.2% | |
| 8 | Eastern Illinois | 9-13 | 2.0% | +4792 | +4000 (est) | 2.4% | -0.4% | |
| -- | Tennessee Tech | 8-12 | DNQ | --- | --- | --- | --- | |
| -- | Southern Indiana | 4-16 | DNQ | --- | --- | --- | --- | |
| -- | Western Illinois | 1-19 | DNQ | --- | --- | --- | --- | |

**Analysis:** The model's top pick is **SEMO** at 19.6%, rated above the 1-seed Tennessee State. **UT Martin (+360)** is a notable fade at -7.6% — Vegas overprices them relative to the model. Tennessee State is also a fade. The 8-team reduced field creates a tight tournament.

---

### Horizon (11 teams)

**Odds Source: Estimated from available sources**

| Seed | Team | Conf Record | Model% | Model Odds | Vegas Odds | Vegas Implied% | Edge | Flag |
|------|------|-------------|--------|------------|------------|----------------|------|------|
| 1 | Wright State | 17-5 | 24.4% | +309 | +100 (est) | 50.0% | -25.6% | FADE |
| 2 | Robert Morris | 14-8 | 20.1% | +398 | +250 (est) | 28.6% | -8.5% | FADE |
| 3 | Detroit Mercy | 14-8 | 10.7% | +837 | +500 (est) | 16.7% | -6.0% | FADE |
| 4 | Green Bay | 13-9 | 9.2% | +992 | +800 (est) | 11.1% | -1.9% | |
| 5 | Oakland | 12-9 | 11.5% | +771 | +700 (est) | 12.5% | -1.0% | |
| 6 | Northern Kentucky | 12-11 | 12.6% | +691 | +1200 (est) | 7.7% | +4.9% | VALUE |
| 7 | Purdue Fort Wayne | 11-10 | 2.3% | +4188 | +2000 (est) | 4.8% | -2.5% | |
| 8 | Youngstown State | 8-13 | 5.9% | +1606 | +1600 | 5.9% | +0.0% | |
| 9 | Milwaukee | 8-13 | 1.9% | +5259 | +5000 (est) | 2.0% | -0.1% | |
| 10 | Cleveland State | 7-15 | 0.5% | +21086 | +10000 (est) | 1.0% | -0.5% | |
| 11 | IU Indianapolis | 3-18 | 1.0% | +9781 | +15000 (est) | 0.7% | +0.3% | |

**Analysis:** **Wright State** at -25.6% edge is a massive fade. **Robert Morris** and **Detroit Mercy** are also fades. **Northern Kentucky (+1200 est)** is the value play at +4.9% — the model rates them at 12.6% despite the 6-seed, nearly matching the top seed's probability.

---

### Big Sky (10 teams)

**Odds Source: Other sportsbooks (HRB lines not found)**

| Seed | Team | Conf Record | Model% | Model Odds | Vegas Odds | Vegas Implied% | Edge | Flag |
|------|------|-------------|--------|------------|------------|----------------|------|------|
| 1 | Portland State | 14-5 | 19.0% | +425 | +450 | 18.2% | +0.8% | |
| 2 | Montana State | 12-7 | 15.7% | +536 | +325 | 23.5% | -7.8% | FADE |
| 3 | Eastern Washington | 12-7 | 13.8% | +626 | +500 | 16.7% | -2.9% | |
| 4 | Montana | 11-8 | 10.8% | +829 | +700 | 12.5% | -1.7% | |
| 5 | Idaho | 11-9 | 14.5% | +588 | +800 (est) | 11.1% | +3.4% | VALUE |
| 6 | Northern Colorado | 10-9 | 16.1% | +521 | +350 | 22.2% | -6.1% | FADE |
| 7 | Weber State | 10-9 | 5.3% | +1778 | +1500 (est) | 6.3% | -1.0% | |
| 8 | Sacramento State | 6-13 | 2.2% | +4538 | +5000 (est) | 2.0% | +0.2% | |
| 9 | Idaho State | 6-14 | 1.9% | +5265 | +6000 (est) | 1.6% | +0.3% | |
| 10 | Northern Arizona | 4-15 | 0.7% | +14105 | +15000 (est) | 0.7% | +0.0% | |

**Analysis:** **Montana State (+325)** and **Northern Colorado (+350)** are both fades. **Idaho (+800 est)** is the value play at +3.4% — the model rates them at 14.5%, the 3rd-best team despite the 5-seed.

---

### SoCon (10 teams)

**Odds Source: Other sportsbooks (HRB lines not found)**

| Seed | Team | Conf Record | Model% | Model Odds | Vegas Odds | Vegas Implied% | Edge | Flag |
|------|------|-------------|--------|------------|------------|----------------|------|------|
| 1 | East Tennessee State | 15-6 | 24.2% | +313 | +190 | 34.5% | -10.3% | FADE |
| 2 | Furman | 13-8 | 18.8% | +432 | +450 | 18.2% | +0.6% | |
| 3 | Mercer | 11-8 | 15.8% | +534 | +475 | 17.4% | -1.6% | |
| 4 | Samford | 11-8 | 10.1% | +888 | +475 | 17.4% | -7.3% | FADE |
| 5 | Wofford | 11-8 | 10.8% | +823 | +725 | 12.1% | -1.3% | |
| 6 | Western Carolina | 11-9 | 12.9% | +676 | +1000 | 9.1% | +3.8% | VALUE |
| 7 | UNC Greensboro | 11-10 | 3.1% | +3164 | +3500 | 2.8% | +0.3% | |
| 8 | The Citadel | 8-12 | 1.1% | +9352 | +35000 | 0.3% | +0.8% | |
| 9 | Chattanooga | 7-12 | 2.7% | +3665 | +4000 | 2.4% | +0.3% | |
| 10 | VMI | 1-18 | 0.6% | +16792 | +50000 | 0.2% | +0.4% | |

**Analysis:** **ETSU (+190)** is a significant fade at -10.3%. **Samford (+475)** is also overpriced at -7.3%. **Western Carolina (+1000)** is the value play at +3.8% — the model rates them at 12.9%.

---

### Patriot (10 teams)

**Odds Source: Estimated from available sources**

| Seed | Team | Conf Record | Model% | Model Odds | Vegas Odds | Vegas Implied% | Edge | Flag |
|------|------|-------------|--------|------------|------------|----------------|------|------|
| 1 | Navy | 18-2 | 34.6% | +189 | -300 (est) | 75.0% | -40.4% | FADE |
| 2 | Lehigh | 13-7 | 10.2% | +884 | +800 (est) | 11.1% | -0.9% | |
| 3 | Colgate | 12-8 | 15.0% | +565 | +500 (est) | 16.7% | -1.7% | |
| 4 | Boston University | 12-8 | 8.8% | +1031 | +800 (est) | 11.1% | -2.3% | |
| 5 | American University | 9-10 | 13.6% | +636 | +600 (est) | 14.3% | -0.7% | |
| 6 | Lafayette | 8-11 | 7.0% | +1327 | +1500 (est) | 6.3% | +0.7% | |
| 7 | Loyola Maryland | 8-11 | 3.5% | +2757 | +3000 (est) | 3.2% | +0.3% | |
| 8 | Bucknell | 7-13 | 2.7% | +3539 | +4000 (est) | 2.4% | +0.3% | |
| 9 | Holy Cross | 6-14 | 1.6% | +6310 | +8000 (est) | 1.2% | +0.4% | |
| 10 | Army | 5-14 | 2.9% | +3294 | +5000 (est) | 2.0% | +0.9% | |

**Analysis:** **Navy (-300 est)** at -40.4% edge is one of the single biggest fades in the entire analysis. The model gives Navy 34.6% while Vegas implies 75%. No value plays exceed the +3% threshold. Most odds are estimated.

---

### NEC (10 teams)

**Odds Source: Other sportsbooks (HRB lines not found)**

| Seed | Team | Conf Record | Model% | Model Odds | Vegas Odds | Vegas Implied% | Edge | Flag |
|------|------|-------------|--------|------------|------------|----------------|------|------|
| 1 | Long Island University | 17-3 | 26.8% | +273 | -160 | 61.5% | -34.7% | FADE |
| 2 | Central Connecticut | 12-7 | 11.4% | +775 | +650 | 13.3% | -1.9% | |
| 3 | Mercyhurst | 12-8 | 13.7% | +629 | +440 | 18.5% | -4.8% | |
| 4 | Le Moyne | 10-9 | 13.7% | +628 | +700 | 12.5% | +1.2% | |
| 5 | New Haven | 9-9 | 7.9% | +1167 | +2200 (est) | 4.3% | +3.6% | VALUE |
| 6 | Wagner | 9-11 | 12.0% | +733 | +2200 | 4.3% | +7.7% | STRONG VALUE |
| 7 | Stonehill | 9-11 | 4.4% | +2155 | +5500 | 1.8% | +2.6% | |
| 8 | Fairleigh Dickinson | 8-11 | 3.7% | +2632 | +4500 | 2.2% | +1.5% | |
| 9 | St. Francis (PA) | 5-13 | 2.6% | +3685 | +8000 (est) | 1.2% | +1.4% | |
| 10 | Chicago State | 5-14 | 3.7% | +2623 | +10000 | 1.0% | +2.7% | |

**Analysis:** **LIU (-160)** is a massive fade at -34.7%. **Wagner (+2200)** is a STRONG VALUE at +7.7% edge — the model gives them 12.0% vs just 4.3% implied. **New Haven (+2200 est)** also shows value at +3.6%.

---

### Big South (9 teams)

**Odds Source: Other sportsbooks (HRB lines not found)**

| Seed | Team | Conf Record | Model% | Model Odds | Vegas Odds | Vegas Implied% | Edge | Flag |
|------|------|-------------|--------|------------|------------|----------------|------|------|
| 1 | High Point | 18-1 | 37.4% | +167 | -200 | 66.7% | -29.3% | FADE |
| 2 | Winthrop | 15-4 | 19.7% | +407 | +350 | 22.2% | -2.5% | |
| 3 | Radford | 9-8 | 9.8% | +923 | +800 | 11.1% | -1.3% | |
| 4 | UNC Asheville | 9-9 | 7.6% | +1211 | +1000 | 9.1% | -1.5% | |
| 5 | Longwood | 8-9 | 8.8% | +1034 | +1200 | 7.7% | +1.1% | |
| 6 | Presbyterian | 8-10 | 5.7% | +1649 | +2000 | 4.8% | +0.9% | |
| 7 | Charleston Southern | 6-11 | 7.6% | +1210 | +3000 | 3.2% | +4.4% | VALUE |
| 8 | South Carolina Upstate | 5-12 | 2.9% | +3320 | +5000 | 2.0% | +0.9% | |
| 9 | Gardner-Webb | 2-16 | 0.3% | +30203 | +15000 | 0.7% | -0.4% | |

**Analysis:** **High Point (-200)** is a massive fade at -29.3%. **Charleston Southern (+3000)** is the value play at +4.4% — the model rates them at 7.6%, more than double the 3.2% Vegas implies.

---

### Summit (9 teams)

**Odds Source: Other sportsbooks (HRB lines not found)**

| Seed | Team | Conf Record | Model% | Model Odds | Vegas Odds | Vegas Implied% | Edge | Flag |
|------|------|-------------|--------|------------|------------|----------------|------|------|
| 1 | North Dakota State | 17-2 | 26.5% | +277 | +150 | 40.0% | -13.5% | FADE |
| 2 | St. Thomas-Minnesota | 13-5 | 27.7% | +261 | +200 | 33.3% | -5.6% | FADE |
| 3 | North Dakota | 12-7 | 7.2% | +1287 | +600 | 14.3% | -7.1% | FADE |
| 4 | Omaha | 9-9 | 9.6% | +943 | +1000 | 9.1% | +0.5% | |
| 5 | Denver | 8-9 | 8.6% | +1063 | +1200 | 7.7% | +0.9% | |
| 6 | South Dakota | 8-9 | 7.0% | +1319 | +1500 | 6.3% | +0.7% | |
| 7 | South Dakota State | 7-10 | 9.6% | +939 | +800 | 11.1% | -1.5% | |
| 8 | Oral Roberts | 5-13 | 3.1% | +3140 | +3000 | 3.2% | -0.1% | |
| 9 | Kansas City | 1-16 | 0.6% | +16240 | +20000 | 0.5% | +0.1% | |

**Analysis:** Three fades in the Summit. **NDSU** at -13.5%, **North Dakota** at -7.1%, and **St. Thomas** at -5.6%. The model rates St. Thomas (27.7%) above NDSU (26.5%) despite being the 2-seed.

---

### Am. East (9 teams)

**Odds Source: Other sportsbooks (HRB lines not found)**

| Seed | Team | Conf Record | Model% | Model Odds | Vegas Odds | Vegas Implied% | Edge | Flag |
|------|------|-------------|--------|------------|------------|----------------|------|------|
| 1 | UMBC | 15-2 | 28.4% | +252 | -270 | 73.0% | -44.6% | FADE |
| 2 | Vermont | 13-4 | 18.0% | +455 | +300 (est) | 25.0% | -7.0% | FADE |
| 3 | NJIT | 11-6 | 14.2% | +604 | +800 (est) | 11.1% | +3.1% | VALUE |
| 4 | UMass Lowell | 10-7 | 12.1% | +729 | +2000 | 4.8% | +7.3% | STRONG VALUE |
| 5 | UAlbany | 7-10 | 10.2% | +885 | +4500 | 2.2% | +8.0% | STRONG VALUE |
| 6 | Maine | 6-11 | 5.7% | +1658 | +5000 (est) | 2.0% | +3.7% | VALUE |
| 7 | New Hampshire | 5-12 | 7.3% | +1264 | +6000 (est) | 1.6% | +5.7% | STRONG VALUE |
| 8 | Bryant | 5-12 | 2.7% | +3657 | +8000 (est) | 1.2% | +1.5% | |
| 9 | Binghamton | 4-12 | 1.4% | +6835 | +15000 (est) | 0.7% | +0.7% | |

**Analysis:** **UMBC (-270)** at -44.6% edge is the single biggest fade across all conferences. The Am. East produces the largest model-vs-Vegas gaps. **UAlbany (+4500)** at +8.0%, **UMass Lowell (+2000)** at +7.3%, and **New Hampshire (+6000 est)** at +5.7% are all STRONG VALUE plays. The model sees this as a wide-open conference despite UMBC's regular-season dominance.

---

### Ivy (4 of 8 teams qualify)

**Odds Source: Other sportsbooks (HRB lines not found)**

| Seed | Team | Conf Record | Model% | Model Odds | Vegas Odds | Vegas Implied% | Edge | Flag |
|------|------|-------------|--------|------------|------------|----------------|------|------|
| 1 | Yale | 11-3 | 38.1% | +163 | +100 | 50.0% | -11.9% | FADE |
| 2 | Harvard | 10-4 | 20.3% | +391 | +500 | 16.7% | +3.6% | VALUE |
| 3 | Pennsylvania | 9-5 | 24.1% | +315 | +300 | 25.0% | -0.9% | |
| 4 | Cornell | 8-6 | 17.5% | +472 | +450 | 18.2% | -0.7% | |
| -- | Columbia | 5-9 | DNQ | --- | --- | --- | --- | |
| -- | Princeton | 5-9 | DNQ | --- | --- | --- | --- | |
| -- | Dartmouth | 5-9 | DNQ | --- | --- | --- | --- | |
| -- | Brown | 3-11 | DNQ | --- | --- | --- | --- | |

**Analysis:** The Ivy's 4-team bracket (just 2 rounds) produces higher top-seed probabilities than larger brackets — Yale at 38.1% is the highest model% for any 1-seed outside Howard and Utah Valley. Still a fade at -11.9% vs Vegas. **Harvard (+500)** shows value at +3.6% — the model rates them at 20.3% while Vegas implies only 16.7%. Pennsylvania and Cornell are fairly priced.

---

### MEAC (8 teams)

**Odds Source: Other sportsbooks (HRB lines not found)**

| Seed | Team | Conf Record | Model% | Model Odds | Vegas Odds | Vegas Implied% | Edge | Flag |
|------|------|-------------|--------|------------|------------|----------------|------|------|
| 1 | Howard | 11-3 | 55.5% | -124 | -300 | 75.0% | -19.5% | FADE |
| 2 | Morgan State | 10-4 | 7.3% | +1267 | +600 | 14.3% | -7.0% | FADE |
| 3 | Norfolk State | 8-6 | 10.9% | +820 | +500 | 16.7% | -5.8% | FADE |
| 4 | North Carolina Central | 8-6 | 5.3% | +1773 | +1200 | 7.7% | -2.4% | |
| 5 | South Carolina State | 7-7 | 6.5% | +1440 | +2000 | 4.8% | +1.7% | |
| 6 | Maryland Eastern Shore | 5-9 | 6.9% | +1342 | +3000 | 3.2% | +3.7% | VALUE |
| 7 | Coppin State | 5-9 | 4.6% | +2063 | +4000 | 2.4% | +2.2% | |
| 8 | Delaware State | 2-12 | 3.0% | +3274 | +8000 | 1.2% | +1.8% | |

**Analysis:** Howard is the only team where the model itself makes them the favorite at negative odds (-124 / 55.5%). Even so, Vegas is dramatically shorter at -300 (75%). **Morgan State** and **Norfolk State** are also fades. **Maryland Eastern Shore (+3000)** is the value play at +3.7%.

---

### WAC (7 teams)

**Odds Source: Other sportsbooks (HRB lines not found)**

| Seed | Team | Conf Record | Model% | Model Odds | Vegas Odds | Vegas Implied% | Edge | Flag |
|------|------|-------------|--------|------------|------------|----------------|------|------|
| 1 | Utah Valley | 14-4 | 49.2% | +103 | -200 | 66.7% | -17.5% | FADE |
| 2 | California Baptist | 13-5 | 20.7% | +383 | +350 | 22.2% | -1.5% | |
| 3 | Utah Tech | 11-7 | 6.5% | +1444 | +1000 | 9.1% | -2.6% | |
| 4 | UT Arlington | 9-9 | 8.8% | +1032 | +800 | 11.1% | -2.3% | |
| 5 | Southern Utah | 6-12 | 3.2% | +3027 | +3000 | 3.2% | +0.0% | |
| 6 | Tarleton State | 5-13 | 7.0% | +1338 | +5000 | 2.0% | +5.0% | STRONG VALUE |
| 7 | Abilene Christian | 5-13 | 4.6% | +2058 | +6000 | 1.6% | +3.0% | VALUE |

**Analysis:** Smallest conference (7 teams, 3 rounds). **Utah Valley** is a fade at -17.5% but the model still makes them the favorite at 49.2%. **Tarleton State (+5000)** is a STRONG VALUE at +5.0% — the model sees them as significantly more dangerous than their 5-13 record suggests in a short bracket. **Abilene Christian (+6000)** also shows value at +3.0%.

---

## Summary

### Top 20 Value Bets Across All Conferences (Sorted by Edge)

| Rank | Team | Conference | Vegas Odds | Model% | Vegas Implied% | Edge | Flag |
|------|------|-----------|------------|--------|----------------|------|------|
| 1 | Tennessee | SEC | +1700 (FD) | 16.2% | 5.6% | +10.6% | STRONG VALUE |
| 2 | UAlbany | Am. East | +4500 | 10.2% | 2.2% | +8.0% | STRONG VALUE |
| 3 | Wagner | NEC | +2200 | 12.0% | 4.3% | +7.7% | STRONG VALUE |
| 4 | Santa Clara | WCC | +500 | 24.0% | 16.7% | +7.3% | STRONG VALUE |
| 5 | UMass Lowell | Am. East | +2000 | 12.1% | 4.8% | +7.3% | STRONG VALUE |
| 6 | Arkansas State | Sun Belt | +600 (est) | 21.3% | 14.3% | +7.0% | STRONG VALUE |
| 7 | Hawai'i | Big West | +300 (est) | 31.5% | 25.0% | +6.5% | STRONG VALUE |
| 8 | North Carolina | ACC | +2000 (HRB) | 10.9% | 4.8% | +6.1% | STRONG VALUE |
| 9 | Tulsa | American | +350 | 28.3% | 22.2% | +6.1% | STRONG VALUE |
| 10 | Miami | ACC | +2000 (HRB) | 10.6% | 4.8% | +5.8% | STRONG VALUE |
| 11 | New Hampshire | Am. East | +6000 (est) | 7.3% | 1.6% | +5.7% | STRONG VALUE |
| 12 | Georgia Southern | Sun Belt | +800 (est) | 16.3% | 11.1% | +5.2% | STRONG VALUE |
| 13 | Tarleton State | WAC | +5000 | 7.0% | 2.0% | +5.0% | STRONG VALUE |
| 14 | Northern Kentucky | Horizon | +1200 (est) | 12.6% | 7.7% | +4.9% | VALUE |
| 15 | South Florida | American | +175 | 41.2% | 36.4% | +4.8% | VALUE |
| 16 | Charleston Southern | Big South | +3000 | 7.6% | 3.2% | +4.4% | VALUE |
| 17 | Iowa State | Big 12 | +800 (HRB) | 15.3% | 11.1% | +4.2% | VALUE |
| 18 | Western Kentucky | CUSA | +700 | 16.5% | 12.5% | +4.0% | VALUE |
| 19 | Western Carolina | SoCon | +1000 | 12.9% | 9.1% | +3.8% | VALUE |
| 20 | Monmouth | CAA | +1000 | 12.9% | 9.1% | +3.8% | VALUE |

### Hard Rock Bet (Florida) Specific Value Bets

For bets actually available on Hard Rock Bet:

| Team | Conference | HRB Odds | Model% | HRB Implied% | Edge | Flag |
|------|-----------|----------|--------|--------------|------|------|
| North Carolina | ACC | +2000 | 10.9% | 4.8% | +6.1% | STRONG VALUE |
| Miami | ACC | +2000 | 10.6% | 4.8% | +5.8% | STRONG VALUE |
| Iowa State | Big 12 | +800 | 15.3% | 11.1% | +4.2% | VALUE |
| Virginia | ACC | +700 | 15.0% | 12.5% | +2.5% | |
| NC State | ACC | +4000 | 4.0% | 2.4% | +1.6% | |
| Clemson | ACC | +3000 | 4.7% | 3.2% | +1.5% | |
| California | ACC | +15000 | 1.3% | 0.7% | +0.6% | |
| SMU | ACC | +10000 | 1.5% | 1.0% | +0.5% | |
| Texas Tech | Big 12 | +1200 | 8.1% | 7.7% | +0.4% | |

**HRB Fades (avoid these):**

| Team | Conference | HRB Odds | Model% | HRB Implied% | Edge |
|------|-----------|----------|--------|--------------|------|
| Duke | ACC | -325 | 39.4% | 76.5% | -37.1% |
| Arizona | Big 12 | +105 | 32.7% | 48.8% | -16.1% |
| Houston | Big 12 | +185 | 21.0% | 35.1% | -14.1% |
| Kansas | Big 12 | +800 | 8.9% | 11.1% | -2.2% |
| Louisville | ACC | +1000 | 8.1% | 9.1% | -1.0% |

### Top 10 Fades (Sorted by Negative Edge)

| Rank | Team | Conference | Vegas Odds | Model% | Vegas Implied% | Edge |
|------|------|-----------|------------|--------|----------------|------|
| 1 | UMBC | Am. East | -270 | 28.4% | 73.0% | -44.6% |
| 2 | Navy | Patriot | -300 (est) | 34.6% | 75.0% | -40.4% |
| 3 | Duke | ACC | -325 (HRB) | 39.4% | 76.5% | -37.1% |
| 4 | LIU | NEC | -160 | 26.8% | 61.5% | -34.7% |
| 5 | Florida | SEC | -180 (FD) | 30.5% | 64.3% | -33.8% |
| 6 | Bethune-Cookman | SWAC | -150 (est) | 27.0% | 60.0% | -33.0% |
| 7 | High Point | Big South | -200 | 37.4% | 66.7% | -29.3% |
| 8 | Wright State | Horizon | +100 (est) | 24.4% | 50.0% | -25.6% |
| 9 | UC Santa Barbara | Big West | +225 | 6.3% | 30.8% | -24.5% |
| 10 | Merrimack | MAAC | +150 (est) | 17.8% | 40.0% | -22.2% |

### Overall Model Tendencies

1. **Systematic top-seed deflation:** The model's most consistent pattern is assigning substantially lower probability to the #1 seed than Vegas in every conference. This is a feature of 50,000 simulated bracket paths accounting for single-elimination variance, whereas Vegas futures tend to over-weight the "most likely winner."

2. **Real bracket structures matter:** The updated simulation uses actual tournament formats — stepladder brackets (WCC, Sun Belt, American), reduced fields (ACC 15/18, Ivy 4/8, Southland 8/12), and proper bye structures. This changes probabilities significantly vs generic power-of-2 brackets. Top seeds in stepladder formats (USF 41.2%, Gonzaga 49.9%, Troy 30.5%) benefit from needing fewer wins.

3. **Mid-tier inflation in smaller conferences:** The model systematically finds more value in seeds 4-8 in smaller conferences (Am. East, NEC, WAC, SWAC, MEAC) where Vegas pricing concentrates heavily on 1-2 teams.

4. **Brand-name bias exploitation:** Several of the largest fades (Auburn in SEC, Memphis in AAC, App State in Sun Belt, UCSB in Big West) involve teams with strong brands or preseason expectations that exceeded conference performance.

5. **Major conference efficiency:** Big conferences (Big 12, Big East, SEC) are generally more efficiently priced than mid-majors. The largest value plays are concentrated in smaller conferences where market liquidity is lower.

6. **HRB-specific opportunities:** On Hard Rock Bet specifically, the best value bets are **North Carolina +2000** (+6.1% edge), **Miami +2000** (+5.8% edge), and **Iowa State +800** (+4.2% edge). The biggest fades are **Duke -325** and **Arizona +105**. For conferences without HRB lines, **Tennessee +1700 (FanDuel)** is the single best value play across all major conferences at +10.6% edge — check HRB for equivalent pricing.

---

*Analysis generated 2026-03-10. Model: Hoops Edge Predictor v1 (MLPRegressor 384/256 + MLPClassifier, 50 features). Simulation: 50,000 Monte Carlo runs per conference with real tournament bracket structures, symmetric home/away augmentation, and neutral-site averaging. Vegas odds: Hard Rock Bet (Florida) for ACC and Big 12; FanDuel Sportsbook for SEC, Big Ten, Big East, and Mountain West; other sportsbooks and estimates for remaining conferences.*
