# Team A/B Default Efficiency Source Rollout

## Change

Changed the Team A/B default efficiency source from internal gold softkeep25 to Torvik for:

- live Team A/B mean-model path
- bracket Team A/B matchup path

The legacy global efficiency default remains unchanged. This rollout only changes the Team A/B defaults when no explicit Team A/B override is set.

## Code

- [src/config.py](/Users/dereknursey/Desktop/ml_projects/codex_review/hoops-edge-predictor_codex/src/config.py#L65)

Key defaults now:

- `TEAM_AB_DEFAULT_EFFICIENCY_SOURCE = "torvik"`
- `HOOPS_TEAM_AB_EFFICIENCY_SOURCE` still overrides live Team A/B source
- `BRACKET_TEAM_AB_DEFAULT_EFFICIENCY_SOURCE = "torvik"`
- `HOOPS_BRACKET_TEAM_AB_EFFICIENCY_SOURCE` still overrides bracket Team A/B source

Gold override support is unchanged:

- `HOOPS_TEAM_AB_EFFICIENCY_SOURCE=gold`
- `HOOPS_TEAM_AB_GOLD_RATINGS_TABLE=team_adjusted_efficiencies_no_garbage_softkeep25_priorreg_k5_v1`
- `HOOPS_BRACKET_TEAM_AB_EFFICIENCY_SOURCE=gold`
- `HOOPS_BRACKET_TEAM_AB_GOLD_RATINGS_TABLE=team_adjusted_efficiencies_no_garbage_softkeep25_priorreg_k5_v1`

## Validation Commands

```bash
python3 -m py_compile src/config.py src/cli.py src/infer.py scripts/build_ncaa_bracket_builder_data.py

env HOOPS_MEAN_MODEL_VARIANT=team_ab_elite_tail_round64_v1 poetry run python - <<'PY'
from src import config
from src.cli import _run_prediction_preflight, _build_secondary_mu_features_if_needed, _build_team_ab_mu_features_if_needed
from src.infer import predict
season=2026
game_date='2026-03-17'
df, lines = _run_prediction_preflight(season, game_date)
secondary = _build_secondary_mu_features_if_needed(season, df, game_date=game_date)
team_ab = _build_team_ab_mu_features_if_needed(season, df, game_date=game_date)
preds = predict(df, lines_df=lines, secondary_mu_features_df=secondary, team_ab_features_df=team_ab)
print(config.TEAM_AB_EFFICIENCY_SOURCE)
print(preds[['gameId','predicted_spread','predicted_spread_legacy','predicted_spread_team_ab_elite_tail_round64_v1']].head(3).to_string(index=False))
PY

env HOOPS_MEAN_MODEL_VARIANT=team_ab_elite_tail_round64_v1 HOOPS_TEAM_AB_EFFICIENCY_SOURCE=gold HOOPS_TEAM_AB_GOLD_RATINGS_TABLE=team_adjusted_efficiencies_no_garbage_softkeep25_priorreg_k5_v1 poetry run python - <<'PY'
from src import config
from src.cli import _run_prediction_preflight, _build_secondary_mu_features_if_needed, _build_team_ab_mu_features_if_needed
from src.infer import predict
season=2026
game_date='2026-03-17'
df, lines = _run_prediction_preflight(season, game_date)
secondary = _build_secondary_mu_features_if_needed(season, df, game_date=game_date)
team_ab = _build_team_ab_mu_features_if_needed(season, df, game_date=game_date)
preds = predict(df, lines_df=lines, secondary_mu_features_df=secondary, team_ab_features_df=team_ab)
print(config.TEAM_AB_EFFICIENCY_SOURCE)
print(preds[['gameId','predicted_spread','predicted_spread_legacy','predicted_spread_team_ab_elite_tail_round64_v1']].head(3).to_string(index=False))
PY

env HOOPS_MEAN_MODEL_VARIANT=legacy_home_slot poetry run python - <<'PY'
from src import config
from src.cli import _run_prediction_preflight, _build_secondary_mu_features_if_needed, _build_team_ab_mu_features_if_needed
from src.infer import predict
season=2026
game_date='2026-03-17'
df, lines = _run_prediction_preflight(season, game_date)
secondary = _build_secondary_mu_features_if_needed(season, df, game_date=game_date)
team_ab = _build_team_ab_mu_features_if_needed(season, df, game_date=game_date)
preds = predict(df, lines_df=lines, secondary_mu_features_df=secondary, team_ab_features_df=team_ab)
print((preds['predicted_spread'].round(6) == preds['predicted_spread_legacy'].round(6)).all())
PY

poetry run python scripts/build_ncaa_bracket_builder_data.py --season 2026 --bracket-model-variant team_ab_elite_tail_round64_v1 --field-output /tmp/ncaa_bracket_builder_2026_teamab_default_field.json --matchups-output /tmp/ncaa_matchup_predictions_2026_teamab_default.json

env HOOPS_BRACKET_TEAM_AB_EFFICIENCY_SOURCE=gold HOOPS_BRACKET_TEAM_AB_GOLD_RATINGS_TABLE=team_adjusted_efficiencies_no_garbage_softkeep25_priorreg_k5_v1 poetry run python scripts/build_ncaa_bracket_builder_data.py --season 2026 --bracket-model-variant team_ab_elite_tail_round64_v1 --field-output /tmp/ncaa_bracket_builder_2026_teamab_gold_field.json --matchups-output /tmp/ncaa_matchup_predictions_2026_teamab_gold.json

poetry run python scripts/build_ncaa_bracket_builder_data.py --season 2026 --bracket-model-variant legacy_synthetic --field-output /tmp/ncaa_bracket_builder_2026_legacy_default_field.json --matchups-output /tmp/ncaa_matchup_predictions_2026_legacy_default.json

env HOOPS_MEAN_MODEL_VARIANT=team_ab_elite_tail_round64_v1 poetry run python -m src.cli predict-today --season 2026 --date 2026-03-17
```

## Validation Summary

- Team A/B live with no explicit override now reports `team_ab_efficiency_source = torvik` and scores successfully.
- Explicit gold override still works and reports `team_ab_efficiency_source = gold`.
- Legacy live path still matches `predicted_spread_legacy` exactly.
- Team A/B bracket build with no explicit override now reports `team_ab_efficiency_source_active = torvik`.
- Explicit gold bracket override still works.
- Legacy bracket build still works unchanged.
- Bracket probability plumbing remains active `mu` + legacy `sigma`; unchanged in this rollout.

## Daily Prediction Output

Ran the real daily prediction export for `2026-03-17` on the new Team A/B default path.

Outputs:

- [2026-03-17.json](/Users/dereknursey/Desktop/ml_projects/codex_review/hoops-edge-predictor_codex/predictions/json/2026-03-17.json)
- [preds_today.csv](/Users/dereknursey/Desktop/ml_projects/codex_review/hoops-edge-predictor_codex/predictions/preds_today.csv)

## Caveat

This does not change which model variant is active by default. It only changes the default efficiency source used when the Team A/B variant is selected.
