# NCAA Results Update Workflow

The NCAA Builder reads actual tournament outcomes from:

- `site/public/data/ncaa_results_<season>.json`

Use the local helper script to validate and update that file safely.

## Results file shape

```json
{
  "version": 1,
  "season": 2026,
  "games": {
    "east-r64-1": {
      "winner_team_id": 72,
      "loser_team_id": 61,
      "status": "final"
    }
  }
}
```

Rules:

- `status` must be one of `pending`, `in_progress`, or `final`
- only `final` entries may include `winner_team_id` and `loser_team_id`
- game ids must match the NCAA bracket graph used by the builder
- winner and loser must be valid possible participants for that bracket game

## Validate the file

Validate and print the normalized JSON to stdout:

```bash
poetry run python scripts/manage_ncaa_results.py validate --season 2026
```

Validate and rewrite the file in canonical pretty-printed order:

```bash
poetry run python scripts/manage_ncaa_results.py validate --season 2026 --write
```

## Mark a game final

Write a final result directly into the season file and revalidate everything:

```bash
poetry run python scripts/manage_ncaa_results.py finalize \
  --season 2026 \
  --game-id east-r64-1 \
  --winner-team-id 72 \
  --loser-team-id 61
```

## Validation behavior

The helper reports clear failures for:

- bad game ids
- malformed statuses
- non-final games that incorrectly include winner/loser ids
- final games missing integer winner/loser ids
- impossible winner/loser combinations for the referenced game

Example impossible-combo message:

- `Game east-r64-1 cannot be 72 over 113. Expected one team from source A [...] and one from source B [...].`

If validation fails, the script exits non-zero and does not write partial broken data.
