# Team A/B Bracket Default Rollout

## Change

Changed the default bracket-builder model variant from `legacy_synthetic` to `team_ab_elite_tail_round64_v1`.

This only changes the no-override default. Explicit variant overrides still work.

## Code Point

- [src/config.py](/Users/dereknursey/Desktop/ml_projects/codex_review/hoops-edge-predictor_codex/src/config.py#L23)

`BRACKET_MATCHUP_MODEL_VARIANT` now defaults to `team_ab_elite_tail_round64_v1`.

The bracket-builder CLI continues to honor:

- `HOOPS_BRACKET_MATCHUP_MODEL_VARIANT`
- `--bracket-model-variant legacy_synthetic`
- `--bracket-model-variant team_ab_elite_tail_round64_v1`

## Files Changed

- [src/config.py](/Users/dereknursey/Desktop/ml_projects/codex_review/hoops-edge-predictor_codex/src/config.py)
- [site/public/data/ncaa_bracket_builder_2026.json](/Users/dereknursey/Desktop/ml_projects/codex_review/hoops-edge-predictor_codex/site/public/data/ncaa_bracket_builder_2026.json)
- [site/public/data/ncaa_matchup_predictions_2026.json](/Users/dereknursey/Desktop/ml_projects/codex_review/hoops-edge-predictor_codex/site/public/data/ncaa_matchup_predictions_2026.json)

Implementation note:

- active advancement odds still run through active cached `win_prob_team1` with unchanged legacy sigma, per [site/lib/bracket/ncaaOdds.ts](/Users/dereknursey/Desktop/ml_projects/codex_review/hoops-edge-predictor_codex/site/lib/bracket/ncaaOdds.ts#L148) and [site/lib/bracket/ncaaOdds.ts](/Users/dereknursey/Desktop/ml_projects/codex_review/hoops-edge-predictor_codex/site/lib/bracket/ncaaOdds.ts#L434)

## Commands Run

```bash
python3 -m py_compile src/config.py scripts/build_ncaa_bracket_builder_data.py
poetry run python scripts/build_ncaa_bracket_builder_data.py --season 2026
poetry run python scripts/build_ncaa_bracket_builder_data.py --season 2026 --bracket-model-variant legacy_synthetic --field-output /tmp/ncaa_bracket_builder_2026_legacy_override_field.json --matchups-output /tmp/ncaa_matchup_predictions_2026_legacy_override.json
cd site && npx tsc --noEmit
```

## Validation Summary

- No-override bracket build now reports active variant `team_ab_elite_tail_round64_v1`.
- Explicit legacy override still reports active variant `legacy_synthetic`.
- Default active fields exactly match the Team A/B compare fields for all `2278` matchup cache entries:
  - active `mu_team1_minus_team2`
  - active `win_prob_team1`
- Legacy override active fields exactly match the legacy compare fields for all `2278` entries.
- Later-round hypothetical matchups still score successfully; full cache build completed with all `2278/2278` matchups.
- TypeScript validation passed with `npx tsc --noEmit`.

## Notable Behavior Changes Under The New Default

Examples from the real `site/public/data/ncaa_matchup_predictions_2026.json` default build versus explicit legacy override:

- Duke vs Siena:
  - default active `mu`: `21.821`
  - legacy active `mu`: `8.446`
  - default active `win_prob_team1`: `0.9703`
  - legacy active `win_prob_team1`: `0.7919`

- Arizona vs Lehigh:
  - default active `mu`: `25.464`
  - legacy active `mu`: `10.660`
  - default active `win_prob_team1`: `0.9836`
  - legacy active `win_prob_team1`: `0.8463`

- Georgia vs Ohio State:
  - default active `mu`: `0.498`
  - legacy active `mu`: `-0.273`
  - default active `win_prob_team1`: `0.5195`
  - legacy active `win_prob_team1`: `0.4888`

These changes flow through the site odds engine because bracket advancement probabilities consume active `win_prob_team1`, not a re-derived legacy matchup probability.

## Caveats

- `legacy_synthetic` remains fully available as an override for rollback/comparison.
- Sigma behavior is unchanged. This rollout does not redesign the uncertainty path or the separate synthetic tournament simulation path.
