#!/usr/bin/env bash
set -euo pipefail

CSV_PATH="${1:-}"
DATE_ARG="${2:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_DIR="$SCRIPT_DIR/../predictions/csv"

if [[ -z "$CSV_PATH" ]]; then
  if [[ ! -d "$DEFAULT_DIR" ]]; then
    echo "CSV directory not found: $DEFAULT_DIR" >&2
    exit 1
  fi
  latest_file=$(ls -t "$DEFAULT_DIR"/*.csv 2>/dev/null | head -n 1 || true)
  if [[ -z "$latest_file" ]]; then
    echo "No CSV files found in $DEFAULT_DIR" >&2
    exit 1
  fi
  CSV_PATH="$latest_file"
fi

echo "=== csv_to_json ==="
if [[ -n "$DATE_ARG" ]]; then
  poetry run python3 "$SCRIPT_DIR/csv_to_json.py" "$CSV_PATH" "$DATE_ARG"
else
  poetry run python3 "$SCRIPT_DIR/csv_to_json.py" "$CSV_PATH"
fi

echo "=== build_rankings_json ==="
poetry run python3 "$SCRIPT_DIR/build_rankings_json.py"

echo "=== s3_finals_to_json ==="
poetry run python3 "$SCRIPT_DIR/s3_finals_to_json.py"
