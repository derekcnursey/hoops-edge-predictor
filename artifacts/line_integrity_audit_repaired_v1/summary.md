# Line Integrity Audit

## Flagged Scope

- Likely suspect evaluation rows: `167`
- High-confidence rows: `0`
- Medium-confidence rows: `0`

## Betting Impact

- Full top200 ROI: `0.0670`
- Excluding high-confidence rows: `0.0670`
- Excluding high+medium-confidence rows: `0.0670`
- Full top100 ROI: `0.0997`
- Excluding high-confidence rows: `0.0997`
- Excluding high+medium-confidence rows: `0.0997`

## Provider Scope

    provider  low
      Bovada    7
 Draft Kings    2
    ESPN BET   87
   consensus  170
  numberfire   43
teamrankings   18

## Season Scope

 season  low
   2016   25
   2017   24
   2018   41
   2019   28
   2020   39
   2021   26
   2022   48
   2023   15
   2024   16
   2025   21
   2026   44

## Example Flagged Rows

 season  gameId             homeTeam           awayTeam     provider  book_spread  homeMoneyline  awayMoneyline  pick_prob  edge_home_points confidence                     reason_flag
   2019   38084              Liberty      Alabama State   numberfire         17.0         1225.0        -2350.0   0.997963         35.352507        low extreme_rating_context_conflict
   2020   31349         Oregon State     San José State    consensus         20.0          162.0         1295.0   0.993007         38.149490        low extreme_rating_context_conflict
   2022   20772             Illinois           Missouri teamrankings         14.5          700.0        -1100.0   0.990724         32.394760        low extreme_rating_context_conflict
   2020   31169             Colorado            Wyoming   numberfire         19.0            NaN            NaN   0.985613         39.358168        low extreme_rating_context_conflict
   2022   19297       Missouri State   Long Beach State teamrankings         12.5          615.0         -850.0   0.979374         25.592548        low extreme_rating_context_conflict
   2020   32311             Arkansas         Valparaiso   numberfire         13.0          650.0         -975.0   0.976510         27.295839        low extreme_rating_context_conflict
   2022   18955  Central Connecticut           NC State teamrankings        -26.5        -3180.0         1277.0   0.973308        -24.916822        low extreme_rating_context_conflict
   2020   30452        Bowling Green       Jacksonville   numberfire         10.0          380.0         -520.0   0.960659         25.979738        low extreme_rating_context_conflict
   2020   31356     Long Beach State         Providence    consensus        -16.5          910.0         -132.0   0.950885        -29.762947        low extreme_rating_context_conflict
   2020   32269             Stanford          San Diego    consensus          9.0         -140.0          325.0   0.950213         22.771881        low extreme_rating_context_conflict
   2020   30945     Middle Tennessee          Villanova    consensus        -19.0         1218.0          129.0   0.949114        -29.453810        low extreme_rating_context_conflict
   2022   19301           California            Florida teamrankings        -13.5         -850.0          615.0   0.948496        -22.226734        low extreme_rating_context_conflict
   2020   31288              Wyoming                TCU teamrankings        -14.5        -1784.0          893.0   0.922854        -25.641649        low extreme_rating_context_conflict
   2020   31376        Chicago State Jacksonville State    consensus        -20.0         1263.0          177.0   0.921130        -25.428857        low extreme_rating_context_conflict
   2020   31358            Fairfield                USC    consensus        -14.5          753.0         -159.0   0.918532        -24.034493        low extreme_rating_context_conflict
   2022   19461           Pepperdine                TCU teamrankings        -11.0         -525.0          420.0   0.916854        -18.768286        low extreme_rating_context_conflict
   2022   19300             Missouri      Florida State teamrankings         -9.5         -525.0          420.0   0.912934        -17.831019        low extreme_rating_context_conflict
   2020   31741                  BYU               UNLV    consensus         10.0         -138.0          341.0   0.898541         19.223244        low extreme_rating_context_conflict
   2020   31838         Grand Canyon            Liberty    consensus         -8.5          284.0         -164.0   0.887308        -21.822041        low extreme_rating_context_conflict
   2022   19472     Central Arkansas                VMI teamrankings        -10.5         -479.0          349.0   0.883439        -15.758197        low extreme_rating_context_conflict
   2019   36520         South Dakota         High Point    consensus          7.5          272.0         -344.0   0.878357         21.002611        low extreme_rating_context_conflict
   2020   32221        Florida State      South Florida    consensus          9.5         -140.0          323.0   0.867930         17.792756        low extreme_rating_context_conflict
   2022   19366 East Tennessee State     Missouri State teamrankings         -7.0         -285.0          240.0   0.843731        -13.566587        low extreme_rating_context_conflict
   2022   19244                  SMU           Missouri teamrankings          7.0          245.0         -290.0   0.827415         12.905493        low extreme_rating_context_conflict
   2019   36702            UC Irvine               UTSA    consensus         10.5          473.0         -666.0   0.824413         16.781651        low extreme_rating_context_conflict