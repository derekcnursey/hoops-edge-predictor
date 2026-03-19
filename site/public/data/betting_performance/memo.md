# Historical Profitability Of Disagreement-Based Picks

## Scope

- Historical scored-bet source: `artifacts/market_bet_filter_v1/historical_scored_bets.parquet`
- Completed scored seasons available: `[2020, 2022, 2023, 2024, 2025]`
- Settlement/vig convention: ATS settlement at `-110`, using the existing internal research stack.
- Public prediction logic was left untouched. This is an internal historical profitability read on top of the existing disagreement-aware workflow.

## Main Answers

1. **Historically, the promoted disagreement-aware picks have been profitable.**
   - Full completed-season sample: `2798` bets, `59.6%` ATS, `13.88%` ROI.
   - Raw-edge baseline on the same broad historical frame: `7639` bets, `52.6%` ATS, `0.44%` ROI.

2. **Late season, the disagreement-aware filter still beats raw edge alone.**
   - Feb 15+: internal `8.85%` vs raw `-6.65%`.
   - March only: internal `13.81%` vs raw `-7.59%`.
   - Conference tournaments: internal `18.18%` vs raw `-1.97%`.

3. **NCAA remains caution-only.**
   - By design, the promoted internal shortlist excludes NCAA rows.
   - The historical NCAA caution bucket remains noisy/negative, which is why it stays diagnostic only.
   - Raw NCAA baseline: `-20.28%` on `91` bets.

4. **Disagreement-led picks are real, not just a side effect of raw edge.**
   - Full sample raw-edge-led is slightly better than disagreement-led, but the disagreement-led bucket is the real additive evidence because it is where the filter is disagreeing with raw edge alone.
   - March only disagreement-led: `86` bets, `17.65%` ROI.
   - March only raw-edge-led: `70` bets, `9.09%` ROI.

5. **New/transient disagreement is stronger than persistent disagreement in March.**
   - March only new/transient: `16` bets, `19.32%` ROI.
   - March only persistent: `70` bets, `3.64%` ROI.

6. **Extreme disagreement magnitude is not automatically better late season.**
   - March only `abs_he_vs_market_edge >= 6`: `17` bets, `1.07%` ROI.
   - March only `abs_he_vs_market_edge < 4`: `76` bets, `15.55%` ROI.
   - In this study, the best March/conference results came from low-to-mid disagreement buckets, not the most extreme outliers.

## Interpretation

- The promoted internal filter is historically profitable as a family and materially better than the raw-edge baseline in the late-season windows the workflow is designed for.
- The strongest disagreement subtype is **new/transient disagreement**, especially in March and conference-tournament contexts, but that bucket is still small and should be treated as directional evidence rather than a standalone rule.
- Persistent disagreement is still useful context, but it is not the best pure profitability pocket.
- NCAA tournament outputs should stay diagnostic/caution-only because the sample is small and still negative in this study.
- Low-to-mid disagreement magnitude looks healthier late season than the most extreme disagreement outliers.

## Robustness Read

- March-only promoted filter profitability is **not just one lucky season**:
  - profitable seasons with March bets: `4` of `5`
  - median March season ROI: `21.49%`
- Conference-tournament promoted filter profitability is also broad enough to matter:
  - profitable seasons with conference-tournament bets: `4` of `5`
  - median conference-tournament season ROI: `14.55%`

This is still a five-season completed-sample study, so any tiny bucket should be treated carefully. The reliable read is at the promoted-filter level and the larger disagreement-led/new-transient subfamilies, not the smallest NCAA-specific corners.

## How To Interpret The Betting Tab Historically

- `Internal Candidate Bets`: maps to `promoted_internal_filter`. This is the main historically profitable bucket.
- `Raw-Edge Watchlist`: maps to `raw_only`. Historically this has been much weaker than the promoted filter and is review context, not the main decision bucket.
- `NCAA Caution`: maps to `ncaa_caution`. Historically noisy and still not strong enough for standalone use.
- `Disagreement-led`: maps to promoted picks that did **not** also pass the raw-edge baseline. Historically important because this is where disagreement is adding decision value beyond raw edge alone.
- `Raw-edge-led`: maps to promoted picks that also passed the raw baseline. Still useful, but less diagnostic of disagreement-specific lift.

## Recommendation

- **The disagreement-aware internal workflow is justified as-is.**
- Keep the promoted threshold unchanged for now.
- Treat `new/transient disagreement`, especially in late-season / March / conference tournaments, as the strongest positive historical subtype.
- Keep NCAA caution rows diagnostic only.
- Do **not** treat small profitable pockets as a reason to launch a new threshold optimization pass. The current result is strong enough to support the existing internal workflow without further tuning.

## Key Tables

### Overall Profitability

```csv
group,slice,bets,wins,losses,pushes,win_rate,roi_per_1_at_minus_110,p_value_gt_breakeven
promoted_internal_filter,full,2798,1669,1129,0,0.5964974982130093,0.13876795113392681,6.255985584241529e-15
promoted_internal_filter,feb15_plus,335,191,144,0,0.5701492537313433,0.08846675712347353,0.04990983442224379
promoted_internal_filter,march_only,156,93,63,0,0.5961538461538461,0.1381118881118881,0.04150521173787988
promoted_internal_filter,conference_tournaments,105,65,40,0,0.6190476190476191,0.18181818181818185,0.03117151518054952
promoted_internal_filter,ncaa_tournament,0,0,0,0,,,
raw_edge_baseline,full,7639,4019,3620,0,0.5261159837675088,0.0044032417379714205,0.34752117402189253
raw_edge_baseline,feb15_plus,1540,753,787,0,0.488961038961039,-0.0665289256198347,0.9971278534771809
raw_edge_baseline,march_only,721,349,372,0,0.4840499306518724,-0.07590467784642543,0.9851341459845577
raw_edge_baseline,conference_tournaments,296,152,144,0,0.5135135135135135,-0.019656019656019652,0.6603949331208374
raw_edge_baseline,ncaa_tournament,91,38,53,0,0.4175824175824176,-0.20279720279720279,0.983617845682456
overlap,full,1609,969,640,0,0.6022374145431946,0.14972597321882594,1.4680725631230054e-10
overlap,feb15_plus,164,93,71,0,0.5670731707317073,0.08259423503325942,0.15120428708560024
overlap,march_only,70,40,30,0,0.5714285714285714,0.09090909090909091,0.24933816311533766
overlap,conference_tournaments,50,27,23,0,0.54,0.030909090909090907,0.46603281113807604
overlap,ncaa_tournament,0,0,0,0,,,
filter_only,full,1189,700,489,0,0.5887300252312868,0.12393913907791117,3.949026169694628e-06
filter_only,feb15_plus,171,98,73,0,0.5730994152046783,0.09409888357256775,0.11220924561234383
filter_only,march_only,86,53,33,0,0.6162790697674418,0.17653276955602532,0.05322819970879991
filter_only,conference_tournaments,55,38,17,0,0.6909090909090909,0.31900826446280994,0.008816021065145192
filter_only,ncaa_tournament,0,0,0,0,,,
raw_only,full,5994,3035,2959,0,0.5063396730063396,-0.03335153335153334,0.9967358683271673
raw_only,feb15_plus,1340,645,695,0,0.48134328358208955,-0.08107191316146539,0.9991489140580289
raw_only,march_only,615,294,321,0,0.47804878048780486,-0.08736141906873612,0.9895886779606514
raw_only,conference_tournaments,246,125,121,0,0.508130081300813,-0.029933481152993338,0.7111692852553793
raw_only,ncaa_tournament,55,23,32,0,0.41818181818181815,-0.20165289256198352,0.9558362742750753
ncaa_caution,full,86,42,44,0,0.4883720930232558,-0.06765327695560253,0.7782709504017253
ncaa_caution,feb15_plus,86,42,44,0,0.4883720930232558,-0.06765327695560253,0.7782709504017253
ncaa_caution,march_only,86,42,44,0,0.4883720930232558,-0.06765327695560253,0.7782709504017253
ncaa_caution,conference_tournaments,0,0,0,0,,,
ncaa_caution,ncaa_tournament,86,42,44,0,0.4883720930232558,-0.06765327695560253,0.7782709504017253
```

### Key Subgroup Profitability

```csv
group,slice,bets,win_rate,roi_per_1_at_minus_110,avg_abs_he_vs_market_edge,p_value_gt_breakeven
disagreement_led,full,1189,0.5887300252312868,0.12393913907791117,6.375628371756,3.949026169694628e-06
disagreement_led,feb15_plus,171,0.5730994152046783,0.09409888357256775,4.58760428446887,0.11220924561234383
disagreement_led,march_only,86,0.6162790697674418,0.17653276955602532,4.35133170119302,0.05322819970879991
disagreement_led,conference_tournaments,55,0.6909090909090909,0.31900826446280994,3.816662292773881,0.008816021065145192
disagreement_led,ncaa_tournament,0,,,,
raw_edge_led,full,1609,0.6022374145431946,0.14972597321882594,5.703127012246864,1.4680725631230054e-10
raw_edge_led,feb15_plus,164,0.5670731707317073,0.08259423503325942,4.1664577867852115,0.15120428708560024
raw_edge_led,march_only,70,0.5714285714285714,0.09090909090909091,3.588832056900188,0.24933816311533766
raw_edge_led,conference_tournaments,50,0.54,0.030909090909090907,3.468606678194719,0.46603281113807604
raw_edge_led,ncaa_tournament,0,,,,
persistent_disagreement,full,1178,0.5747028862478778,0.09716005556413028,6.416833879775489,0.0002542903289533169
persistent_disagreement,feb15_plus,146,0.5958904109589042,0.13760896637608963,5.0168412975187096,0.04794992409383104
persistent_disagreement,march_only,70,0.5428571428571428,0.036363636363636355,4.50893262182707,0.4217724088545652
persistent_disagreement,conference_tournaments,47,0.5319148936170213,0.015473887814313345,3.9902438078270746,0.5147745086165054
persistent_disagreement,ncaa_tournament,0,,,,
new_transient_disagreement,full,199,0.6984924623115578,0.33348560986751935,7.954152818738093,3.927178322733563e-07
new_transient_disagreement,feb15_plus,16,0.625,0.19318181818181815,1.451139035729759,0.2894334675775315
new_transient_disagreement,march_only,16,0.625,0.19318181818181815,1.451139035729759,0.2894334675775315
new_transient_disagreement,conference_tournaments,16,0.625,0.19318181818181815,1.451139035729759,0.2894334675775315
new_transient_disagreement,ncaa_tournament,0,,,,
high_disagreement_ge_6,full,1231,0.6108854589764419,0.16623587622775274,8.02279280495222,4.667568921569971e-10
high_disagreement_ge_6,feb15_plus,52,0.5576923076923077,0.06468531468531467,6.860506583889901,0.3639485420778456
high_disagreement_ge_6,march_only,17,0.5294117647058824,0.010695187165775399,6.958390157510978,0.5790090239051432
high_disagreement_ge_6,conference_tournaments,8,0.5,-0.04545454545454547,7.093867864361528,0.6874482733531904
high_disagreement_ge_6,ncaa_tournament,0,,,,
medium_disagreement_4_to_6,full,1126,0.5817051509769094,0.1105280155013725,5.044747979383091,5.425898333538268e-05
medium_disagreement_4_to_6,feb15_plus,147,0.6122448979591837,0.16883116883116883,4.8841193591135985,0.019107678317379806
medium_disagreement_4_to_6,march_only,63,0.6031746031746031,0.15151515151515146,4.815645056250229,0.12794132975795722
medium_disagreement_4_to_6,conference_tournaments,35,0.6,0.14545454545454542,4.74476157908319,0.2323457044123911
medium_disagreement_4_to_6,ncaa_tournament,0,,,,
low_disagreement_lt_4,full,441,0.5941043083900227,0.1341991341991342,2.7222433765114404,0.0017618552223603889
low_disagreement_lt_4,feb15_plus,136,0.5294117647058824,0.010695187165775404,2.890202364154246,0.4826215727670467
low_disagreement_lt_4,march_only,76,0.6052631578947368,0.15550239234449756,2.6809802508442337,0.09521863739859611
low_disagreement_lt_4,conference_tournaments,62,0.6451612903225806,0.23167155425219935,2.589180029507992,0.03624756877644819
low_disagreement_lt_4,ncaa_tournament,0,,,,
neutral_internal_filter,full,468,0.6645299145299145,0.2686480186480186,5.674691470983086,4.791613024600374e-10
neutral_internal_filter,feb15_plus,97,0.6288659793814433,0.2005623242736645,3.6976661579170713,0.0238626759431982
neutral_internal_filter,march_only,97,0.6288659793814433,0.2005623242736645,3.6976661579170713,0.0238626759431982
neutral_internal_filter,conference_tournaments,96,0.6354166666666666,0.2130681818181818,3.7075550062882066,0.017901350150679753
neutral_internal_filter,ncaa_tournament,0,,,,
non_neutral_internal_filter,full,2330,0.582832618025751,0.11268045259461566,6.05201626107425,5.967308892995601e-09
non_neutral_internal_filter,feb15_plus,238,0.5462184873949579,0.04278074866310158,4.660108371256283,0.2654780424856017
non_neutral_internal_filter,march_only,59,0.5423728813559322,0.03543913713405238,4.521341575722999,0.43923537887731373
non_neutral_internal_filter,conference_tournaments,9,0.4444444444444444,-0.1515151515151515,3.046831045403502,0.7906733264725863
non_neutral_internal_filter,ncaa_tournament,0,,,,
```

### By-Season Detail

```csv
season,group,slice,bets,win_rate,roi_per_1_at_minus_110
2020,promoted_internal_filter,full,519,0.581888246628131,0.11087756174461379
2020,promoted_internal_filter,feb15_plus,115,0.5565217391304348,0.0624505928853755
2020,promoted_internal_filter,march_only,36,0.6388888888888888,0.2196969696969697
2020,promoted_internal_filter,conference_tournaments,17,0.5882352941176471,0.1229946524064171
2020,promoted_internal_filter,ncaa_tournament,0,,
2022,promoted_internal_filter,full,672,0.6101190476190477,0.1647727272727273
2022,promoted_internal_filter,feb15_plus,98,0.6122448979591837,0.16883116883116883
2022,promoted_internal_filter,march_only,57,0.6491228070175439,0.23923444976076555
2022,promoted_internal_filter,conference_tournaments,43,0.6744186046511628,0.2875264270613107
2022,promoted_internal_filter,ncaa_tournament,0,,
2023,promoted_internal_filter,full,639,0.593114241001565,0.13230900554844216
2023,promoted_internal_filter,feb15_plus,53,0.5660377358490566,0.08061749571183532
2023,promoted_internal_filter,march_only,30,0.4666666666666667,-0.1090909090909091
2023,promoted_internal_filter,conference_tournaments,22,0.5,-0.04545454545454546
2023,promoted_internal_filter,ncaa_tournament,0,,
2024,promoted_internal_filter,full,452,0.5862831858407079,0.11926790024135156
2024,promoted_internal_filter,feb15_plus,28,0.5357142857142857,0.022727272727272728
2024,promoted_internal_filter,march_only,11,0.6363636363636364,0.21487603305785122
2024,promoted_internal_filter,conference_tournaments,10,0.6,0.14545454545454542
2024,promoted_internal_filter,ncaa_tournament,0,,
2025,promoted_internal_filter,full,516,0.6065891472868217,0.1580338266384778
2025,promoted_internal_filter,feb15_plus,41,0.5365853658536586,0.024390243902439022
2025,promoted_internal_filter,march_only,22,0.5454545454545454,0.04132231404958677
2025,promoted_internal_filter,conference_tournaments,13,0.6923076923076923,0.32167832167832167
2025,promoted_internal_filter,ncaa_tournament,0,,
2020,raw_edge_baseline,full,1443,0.5391545391545391,0.029295029295029307
2020,raw_edge_baseline,feb15_plus,236,0.4491525423728814,-0.14252696456086286
2020,raw_edge_baseline,march_only,78,0.3974358974358974,-0.2412587412587412
2020,raw_edge_baseline,conference_tournaments,32,0.40625,-0.22443181818181818
2020,raw_edge_baseline,ncaa_tournament,0,,
2022,raw_edge_baseline,full,1522,0.545335085413929,0.04109425397204635
2022,raw_edge_baseline,feb15_plus,256,0.51953125,-0.008167613636363612
2022,raw_edge_baseline,march_only,118,0.559322033898305,0.06779661016949151
2022,raw_edge_baseline,conference_tournaments,52,0.5384615384615384,0.027972027972027965
2022,raw_edge_baseline,ncaa_tournament,14,0.42857142857142855,-0.18181818181818185
2023,raw_edge_baseline,full,1522,0.545335085413929,0.04109425397204636
2023,raw_edge_baseline,feb15_plus,266,0.5150375939849624,-0.016746411483253582
2023,raw_edge_baseline,march_only,122,0.5327868852459017,0.017138599105812235
2023,raw_edge_baseline,conference_tournaments,61,0.5573770491803278,0.06408345752608047
2023,raw_edge_baseline,ncaa_tournament,23,0.5652173913043478,0.07905138339920947
2024,raw_edge_baseline,full,1424,0.511938202247191,-0.022663432073544428
2024,raw_edge_baseline,feb15_plus,333,0.46546546546546547,-0.11138411138411139
2024,raw_edge_baseline,march_only,180,0.43333333333333335,-0.1727272727272727
2024,raw_edge_baseline,conference_tournaments,74,0.47297297297297297,-0.09705159705159704
2024,raw_edge_baseline,ncaa_tournament,22,0.22727272727272727,-0.5661157024793388
2025,raw_edge_baseline,full,1728,0.4930555555555556,-0.0587121212121212
2025,raw_edge_baseline,feb15_plus,449,0.49443207126948774,-0.056084227576432465
2025,raw_edge_baseline,march_only,223,0.48878923766816146,-0.06685690990623724
2025,raw_edge_baseline,conference_tournaments,77,0.5454545454545454,0.041322314049586764
2025,raw_edge_baseline,ncaa_tournament,32,0.4375,-0.16477272727272727
2020,filter_only,full,249,0.5261044176706827,0.004381161007667035
2020,filter_only,feb15_plus,70,0.5714285714285714,0.0909090909090909
2020,filter_only,march_only,22,0.7272727272727273,0.3884297520661157
2020,filter_only,conference_tournaments,11,0.7272727272727273,0.3884297520661157
2020,filter_only,ncaa_tournament,0,,
2022,filter_only,full,233,0.6094420600858369,0.16348029652750684
2022,filter_only,feb15_plus,46,0.5869565217391305,0.12055335968379444
2022,filter_only,march_only,28,0.6071428571428571,0.15909090909090912
2022,filter_only,conference_tournaments,19,0.7368421052631579,0.4066985645933013
2022,filter_only,ncaa_tournament,0,,
2023,filter_only,full,283,0.5865724381625441,0.11982010921940249
2023,filter_only,feb15_plus,25,0.56,0.06909090909090908
2023,filter_only,march_only,17,0.5294117647058824,0.010695187165775366
2023,filter_only,conference_tournaments,13,0.6153846153846154,0.17482517482517482
2023,filter_only,ncaa_tournament,0,,
2024,filter_only,full,189,0.6190476190476191,0.1818181818181818
2024,filter_only,feb15_plus,6,0.5,-0.045454545454545435
2024,filter_only,march_only,2,0.5,-0.04545454545454547
2024,filter_only,conference_tournaments,2,0.5,-0.04545454545454547
2024,filter_only,ncaa_tournament,0,,
2025,filter_only,full,235,0.6127659574468085,0.169825918762089
2025,filter_only,feb15_plus,24,0.5833333333333334,0.1136363636363636
2025,filter_only,march_only,17,0.5882352941176471,0.12299465240641708
2025,filter_only,conference_tournaments,10,0.7,0.3363636363636363
2025,filter_only,ncaa_tournament,0,,
2020,raw_only,full,1173,0.5174765558397272,-0.012090211578702619
2020,raw_only,feb15_plus,191,0.4293193717277487,-0.18039029033793427
2020,raw_only,march_only,64,0.375,-0.28409090909090906
2020,raw_only,conference_tournaments,26,0.4230769230769231,-0.19230769230769226
2020,raw_only,ncaa_tournament,0,,
2022,raw_only,full,1069,0.520112254443405,-0.007058423335317624
2022,raw_only,feb15_plus,190,0.49473684210526314,-0.05550239234449762
2022,raw_only,march_only,75,0.5333333333333333,0.01818181818181817
2022,raw_only,conference_tournaments,28,0.4642857142857143,-0.11363636363636363
2022,raw_only,ncaa_tournament,0,,
2023,raw_only,full,1154,0.5294627383015598,0.010792500393886878
2023,raw_only,feb15_plus,226,0.5088495575221239,-0.028559935639581653
2023,raw_only,march_only,97,0.5567010309278351,0.06279287722586695
2023,raw_only,conference_tournaments,52,0.5961538461538461,0.13811188811188807
2023,raw_only,ncaa_tournament,11,0.6363636363636364,0.21487603305785122
2024,raw_only,full,1157,0.5012964563526361,-0.04297949241769467
2024,raw_only,feb15_plus,307,0.46254071661237783,-0.116967722830915
2024,raw_only,march_only,167,0.4251497005988024,-0.18835057158410454
2024,raw_only,conference_tournaments,66,0.45454545454545453,-0.1322314049586777
2024,raw_only,ncaa_tournament,18,0.2222222222222222,-0.5757575757575757
2025,raw_only,full,1441,0.472588480222068,-0.09778562866696107
2025,raw_only,feb15_plus,426,0.49765258215962443,-0.049935979513444306
2025,raw_only,march_only,212,0.49528301886792453,-0.054459691252144074
2025,raw_only,conference_tournaments,74,0.5405405405405406,0.03194103194103195
2025,raw_only,ncaa_tournament,26,0.46153846153846156,-0.11888111888111887
```

### Signal Driver Profitability

```csv
signal_driver,slice,bets,win_rate,roi_per_1_at_minus_110
disagreement-led new/transient,full,67,0.6865671641791045,0.3107191316146541
disagreement-led new/transient,feb15_plus,0,,
disagreement-led new/transient,march_only,0,,
disagreement-led new/transient,conference_tournaments,0,,
disagreement-led new/transient conf-tourney,full,10,0.6,0.14545454545454542
disagreement-led new/transient conf-tourney,feb15_plus,10,0.6,0.14545454545454542
disagreement-led new/transient conf-tourney,march_only,10,0.6,0.14545454545454542
disagreement-led new/transient conf-tourney,conference_tournaments,10,0.6,0.14545454545454542
disagreement-led persistent,full,524,0.5744274809160306,0.09663428174878556
disagreement-led persistent,feb15_plus,79,0.620253164556962,0.18411967779056387
disagreement-led persistent,march_only,43,0.5813953488372093,0.10993657505285409
disagreement-led persistent,conference_tournaments,25,0.64,0.2218181818181818
disagreement-led support,full,588,0.5901360544217688,0.1266233766233766
disagreement-led support,feb15_plus,82,0.524390243902439,0.0011086474501108745
disagreement-led support,march_only,33,0.6666666666666666,0.2727272727272727
disagreement-led support,conference_tournaments,20,0.8,0.5272727272727271
raw edge + new disagreement support,full,122,0.7131147540983607,0.3614008941877794
raw edge + new disagreement support,feb15_plus,6,0.6666666666666666,0.2727272727272727
raw edge + new disagreement support,march_only,6,0.6666666666666666,0.2727272727272727
raw edge + new disagreement support,conference_tournaments,6,0.6666666666666666,0.2727272727272727
raw edge + persistent disagreement support,full,654,0.5749235474006116,0.09758131776480401
raw edge + persistent disagreement support,feb15_plus,67,0.5671641791044776,0.08276797829036636
raw edge + persistent disagreement support,march_only,27,0.48148148148148145,-0.08080808080808081
raw edge + persistent disagreement support,conference_tournaments,22,0.4090909090909091,-0.21900826446280988
raw edge led,full,833,0.6074429771908764,0.15966386554621848
raw edge led,feb15_plus,91,0.5604395604395604,0.06993006993006994
raw edge led,march_only,37,0.6216216216216216,0.18673218673218672
raw edge led,conference_tournaments,22,0.6363636363636364,0.21487603305785122
```

### Robustness Summary

```csv
group,slice,season_count_with_bets,profitable_seasons,losing_seasons,median_season_roi,mean_season_roi,median_bets_per_season
disagreement_led,conference_tournaments,5,4,1,0.3363636363636363,0.2521725164787365,11.0
disagreement_led,feb15_plus,5,4,1,0.0909090909090909,0.06974703557312252,25.0
disagreement_led,full,5,5,0,0.16348029652750684,0.12786513346696943,235.0
disagreement_led,march_only,5,4,1,0.12299465240641708,0.12715119105493436,17.0
disagreement_led,ncaa_tournament,0,0,0,,,
new_transient_disagreement,conference_tournaments,4,3,1,0.4795454545454545,0.3125,4.5
new_transient_disagreement,feb15_plus,4,3,1,0.4795454545454545,0.3125,4.5
new_transient_disagreement,full,5,5,0,0.35371900826446284,0.28470856894301877,40.0
new_transient_disagreement,march_only,4,3,1,0.4795454545454545,0.3125,4.5
new_transient_disagreement,ncaa_tournament,0,0,0,,,
promoted_internal_filter,conference_tournaments,5,4,1,0.14545454545454542,0.16643988022920989,17.0
promoted_internal_filter,feb15_plus,5,5,0,0.0624505928853755,0.07180335481161829,53.0
promoted_internal_filter,full,5,5,0,0.13230900554844216,0.13705220428912251,519.0
promoted_internal_filter,march_only,5,4,1,0.21487603305785122,0.12120777149485282,30.0
promoted_internal_filter,ncaa_tournament,0,0,0,,,
raw_edge_baseline,conference_tournaments,5,3,2,0.027972027972027965,-0.03762112313714401,61.0
raw_edge_baseline,feb15_plus,5,0,5,-0.056084227576432465,-0.06698186572820478,266.0
raw_edge_baseline,full,5,3,2,0.029295029295029307,0.006021596790691278,1522.0
raw_edge_baseline,march_only,5,2,3,-0.06685690990623724,-0.07918154292338948,122.0
raw_edge_baseline,ncaa_tournament,4,1,3,-0.17329545454545456,-0.20841380704275964,22.5
```

### Betting Tab Mapping

```csv
page_label,group,slice,bets,win_rate,roi_per_1_at_minus_110,note
Internal Candidate Bets,promoted_internal_filter,full,2798,0.5964974982130093,0.13876795113392681,Main shortlist on the betting tab. Excludes NCAA caution rows.
Internal Candidate Bets,promoted_internal_filter,feb15_plus,335,0.5701492537313433,0.08846675712347353,Main shortlist on the betting tab. Excludes NCAA caution rows.
Internal Candidate Bets,promoted_internal_filter,march_only,156,0.5961538461538461,0.1381118881118881,Main shortlist on the betting tab. Excludes NCAA caution rows.
Internal Candidate Bets,promoted_internal_filter,conference_tournaments,105,0.6190476190476191,0.18181818181818185,Main shortlist on the betting tab. Excludes NCAA caution rows.
Internal Candidate Bets,promoted_internal_filter,ncaa_tournament,0,,,Main shortlist on the betting tab. Excludes NCAA caution rows.
Raw-Edge Watchlist,raw_only,full,5994,0.5063396730063396,-0.03335153335153334,Games the raw edge baseline liked but the promoted filter rejected.
Raw-Edge Watchlist,raw_only,feb15_plus,1340,0.48134328358208955,-0.08107191316146539,Games the raw edge baseline liked but the promoted filter rejected.
Raw-Edge Watchlist,raw_only,march_only,615,0.47804878048780486,-0.08736141906873612,Games the raw edge baseline liked but the promoted filter rejected.
Raw-Edge Watchlist,raw_only,conference_tournaments,246,0.508130081300813,-0.029933481152993338,Games the raw edge baseline liked but the promoted filter rejected.
Raw-Edge Watchlist,raw_only,ncaa_tournament,55,0.41818181818181815,-0.20165289256198352,Games the raw edge baseline liked but the promoted filter rejected.
NCAA Caution,ncaa_caution,full,86,0.4883720930232558,-0.06765327695560253,Rows that clear the internal score threshold but remain caution-only.
NCAA Caution,ncaa_caution,feb15_plus,86,0.4883720930232558,-0.06765327695560253,Rows that clear the internal score threshold but remain caution-only.
NCAA Caution,ncaa_caution,march_only,86,0.4883720930232558,-0.06765327695560253,Rows that clear the internal score threshold but remain caution-only.
NCAA Caution,ncaa_caution,conference_tournaments,0,,,Rows that clear the internal score threshold but remain caution-only.
NCAA Caution,ncaa_caution,ncaa_tournament,86,0.4883720930232558,-0.06765327695560253,Rows that clear the internal score threshold but remain caution-only.
Disagreement-Led,disagreement_led,full,1189,0.5887300252312868,0.12393913907791117,Promoted picks that were disagreement-led rather than raw-edge-led.
Disagreement-Led,disagreement_led,feb15_plus,171,0.5730994152046783,0.09409888357256775,Promoted picks that were disagreement-led rather than raw-edge-led.
Disagreement-Led,disagreement_led,march_only,86,0.6162790697674418,0.17653276955602532,Promoted picks that were disagreement-led rather than raw-edge-led.
Disagreement-Led,disagreement_led,conference_tournaments,55,0.6909090909090909,0.31900826446280994,Promoted picks that were disagreement-led rather than raw-edge-led.
Disagreement-Led,disagreement_led,ncaa_tournament,0,,,Promoted picks that were disagreement-led rather than raw-edge-led.
Raw-Edge-Led,raw_edge_led,full,1609,0.6022374145431946,0.14972597321882594,Promoted picks that also pass the raw edge baseline.
Raw-Edge-Led,raw_edge_led,feb15_plus,164,0.5670731707317073,0.08259423503325942,Promoted picks that also pass the raw edge baseline.
Raw-Edge-Led,raw_edge_led,march_only,70,0.5714285714285714,0.09090909090909091,Promoted picks that also pass the raw edge baseline.
Raw-Edge-Led,raw_edge_led,conference_tournaments,50,0.54,0.030909090909090907,Promoted picks that also pass the raw edge baseline.
Raw-Edge-Led,raw_edge_led,ncaa_tournament,0,,,Promoted picks that also pass the raw edge baseline.
```
