# Line Integrity Audit

## Flagged Scope

- Likely suspect evaluation rows: `157`
- High-confidence rows: `106`
- Medium-confidence rows: `51`

## Betting Impact

- Full top200 ROI: `0.1102`
- Excluding high-confidence rows: `0.0649`
- Excluding high+medium-confidence rows: `0.0449`
- Full top100 ROI: `0.1732`
- Excluding high-confidence rows: `0.0952`
- Excluding high+medium-confidence rows: `0.0712`

## Provider Scope

    provider  high  medium
 Draft Kings    38       0
    ESPN BET     9       7
   consensus   148      76
  numberfire     0       1
teamrankings     9       6

## Season Scope

 season  high  medium
   2016     2       2
   2017     3       5
   2018    30       5
   2020    40      24
   2021    16      20
   2022    66      27
   2026    47       7

## Example Flagged Rows

 season  gameId            homeTeam          awayTeam  provider  book_spread  homeMoneyline  awayMoneyline  pick_prob  edge_home_points confidence               reason_flag
   2022   19318    Central Michigan           Gonzaga consensus        -34.0         1613.0        -4586.0   0.999940        -53.617886       high strong_moneyline_conflict
   2020   31918              Kansas       Kansas City consensus         23.0        -7031.0         1780.0   0.997635         50.847208       high strong_moneyline_conflict
   2020   31331             Gonzaga     Southern Miss consensus         25.0      -250000.0         3060.0   0.996809         48.881281       high strong_moneyline_conflict
   2022   19337             Liberty   Bethune-Cookman consensus         18.0        -1694.0          945.0   0.996625         35.346184       high strong_moneyline_conflict
   2020   32152              Baylor         UT Martin consensus         22.0        -9878.0         1600.0   0.995866         47.536716       high strong_moneyline_conflict
   2020   30550          Texas Tech Houston Christian consensus         24.5       -12973.0         2075.0   0.994910         46.253445       high strong_moneyline_conflict
   2022   20772            Illinois          Missouri consensus         14.5        -1124.0          679.0   0.990724         32.394760       high strong_moneyline_conflict
   2020   31169            Colorado           Wyoming consensus         18.5        -4444.0         1282.0   0.984567         38.858168       high strong_moneyline_conflict
   2020   31060            Virginia     Massachusetts consensus         15.5        -2054.0          967.0   0.984303         31.466648       high strong_moneyline_conflict
   2022   20775       South Florida               BYU consensus        -14.5          691.0        -1190.0   0.981588        -27.367803       high strong_moneyline_conflict
   2022   19480      Saint Joseph's               USC consensus        -17.0          888.0        -1574.0   0.980519        -28.162162       high strong_moneyline_conflict
   2022   19297      Missouri State  Long Beach State consensus         12.5         -928.0          601.0   0.979374         25.592548       high strong_moneyline_conflict
   2022   19192       North Alabama              Iona consensus        -13.5          607.0         -932.0   0.974992        -26.615338       high strong_moneyline_conflict
   2022   20475              Butler            Purdue consensus        -15.5          834.0        -1482.0   0.974813        -27.354263       high strong_moneyline_conflict
   2022   18955 Central Connecticut          NC State consensus        -26.5         1388.0        -3535.0   0.973308        -24.916822       high strong_moneyline_conflict
   2022   19133     Bethune-Cookman            Bryant consensus        -17.5          892.0        -1815.0   0.968793        -25.296543       high strong_moneyline_conflict
   2022   19330        Murray State  Long Beach State consensus         11.0         -697.0          474.0   0.967020         23.588800       high strong_moneyline_conflict
   2020   31366       Southern Miss        Seton Hall consensus        -17.0         1120.0        -2570.0   0.966800        -30.063481       high strong_moneyline_conflict
   2020   30985          California              Duke consensus        -19.0         1242.0        -3529.0   0.965807        -28.077216       high strong_moneyline_conflict
   2020   30452       Bowling Green      Jacksonville consensus         10.0         -599.0          415.0   0.960659         25.979738       high strong_moneyline_conflict
   2020   30951              Baylor              Ohio consensus         14.0        -1336.0          737.0   0.960196         26.216564       high strong_moneyline_conflict
   2022   19425   George Washington    Missouri State consensus        -11.5          476.0         -677.0   0.957844        -22.862826       high strong_moneyline_conflict
   2020   31497             Hofstra        Holy Cross consensus         13.0        -1196.0          700.0   0.957332         26.048127       high strong_moneyline_conflict
   2020   31411             Alabama     Southern Miss consensus         12.0         -864.0          547.0   0.954989         25.424221       high strong_moneyline_conflict
   2022   19301          California           Florida consensus        -13.5          602.0         -931.0   0.948496        -22.226734       high strong_moneyline_conflict