# Arizona vs Duke in `EDGE INDEX` (`2026`)

## Short answer

Yes, this `EDGE INDEX` is using **Hoops Edge internal efficiencies**, not Torvik.

Current preferred source:

- `team_adjusted_efficiencies_no_garbage_priorreg_k5_v1`

Fallback only if the preferred table is missing:

- `team_adjusted_efficiencies_no_garbage`

The `EDGE INDEX` itself is then computed from the current `HGBR` mean model as a **projected neutral-court spread vs an average D-I team**.

## Why Arizona is above Duke

For `2026`, the model-driven neutral-floor values are:

- Arizona: `15.109`
- Duke: `15.058`

So Arizona is only ahead by about:

- `0.051` points

This is a very small gap. It is not a large structural separation.

## What is actually driving the difference

For this specific pair, the gap is **not** coming from the obvious top-line efficiency fields:

- `adj_oe`
- `adj_de`
- `adj_tempo`
- `barthag`

Duke is better than Arizona on those headline fields:

| Team | adj_oe | adj_de | adj_tempo | barthag |
|---|---:|---:|---:|---:|
| Arizona | 123.296 | 91.811 | 62.934 | 0.967418 |
| Duke | 130.381 | 88.367 | 59.965 | 0.988716 |

But the `EDGE INDEX` build path also carries through the in-house schedule context fields from the gold ratings table:

| Team | sos_oe | sos_de |
|---|---:|---:|
| Arizona | 104.144 | 114.865 |
| Duke | 102.313 | 115.301 |

Counterfactual swaps on the exact ranking build logic show:

- swapping Arizona's `sos_oe` onto Duke moves Duke's index up to Arizona's number
- swapping Duke's `sos_oe` onto Arizona moves Arizona's index down to Duke's number
- swapping `adj_oe`, `adj_de`, `adj_tempo`, `barthag`, or `sos_de` does **not** change the ordering in this case

So the current Arizona-over-Duke ordering is being driven by the model's treatment of:

- `sos_oe`

more than the headline OE/DE/tempo fields.

## Interpretation

This means:

1. Arizona is not clearly "better than Duke" by a big model margin.
2. The current `EDGE INDEX` gap is tiny.
3. The ordering is being resolved by a secondary schedule-context signal, not by the obvious headline efficiency numbers.

If you want the ranking to behave more like a pure top-line team-strength display, the next thing to review would be whether `EDGE INDEX` should keep using those schedule-context fields in the synthetic average-team matchup.
