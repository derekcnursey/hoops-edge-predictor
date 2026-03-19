# NCAA Bracket Builder Field Workflow

The NCAA builder UI consumes generated files, not the hand-edited field input directly.

## Canonical input

Place the season field file at:

- `site/public/data/ncaa_field_input_<season>.json`

The canonical input format is:

```json
{
  "season": 2026,
  "source": "selection_sunday_manual",
  "note": "Optional operator note",
  "first_four_games": [
    {
      "id": "ff1",
      "region": "East",
      "seed": 11,
      "winner_to_slot": "east-11"
    }
  ],
  "entries": [
    {
      "team_id": 72,
      "team_name": "Duke",
      "seed": 1,
      "region": "East",
      "slot": "east-1",
      "is_first_four": false,
      "first_four_game_id": null,
      "feeder_slot": null
    },
    {
      "team_id": 257,
      "team_name": "San Diego State",
      "seed": 11,
      "region": "East",
      "slot": "ff1-team-1",
      "is_first_four": true,
      "first_four_game_id": "ff1",
      "feeder_slot": "east-11"
    }
  ]
}
```

Notes:

- Direct bracket teams use their main bracket slot, such as `east-1` or `midwest-12`.
- First Four participants use unique staging slots such as `ff1-team-1`.
- First Four participants must point at the game id they belong to and the main bracket slot that winner feeds.

## Regenerate builder data

1. Update `site/public/data/ncaa_field_input_<season>.json` with the official Selection Sunday bracket.
2. Ensure `site/public/data/rankings_<season>.json` has current metadata for every selected `team_id`.
3. Regenerate the builder payloads:

```bash
poetry run python scripts/build_ncaa_bracket_builder_data.py --season 2026
```

Optional dev helper:

```bash
poetry run python scripts/build_ncaa_bracket_builder_data.py --season 2026 --use-rankings-fallback
```

The fallback is for local development only. It does not run automatically if the canonical file is missing.

## Validation

The generator validates:

- exactly 68 team entries
- exactly four regions
- unique `team_id` values
- unique occupied slots
- complete region seed coverage
- valid First Four feeder structure
- valid canonical matchup cache output

If validation fails, the script exits before writing new site data.

## Generated files consumed by the site

- `site/public/data/ncaa_bracket_builder_<season>.json`
- `site/public/data/ncaa_matchup_predictions_<season>.json`

The `/brackets` page and `/api/predict-matchup` route continue to read the generated builder JSON and matchup cache, so no frontend changes are required when the official field is updated.
