#!/usr/bin/env python3
"""Read game results from S3 fct_games and produce final_scores JSON.

Usage:
  s3_finals_to_json.py --date YYYY-MM-DD      # daily mode: yesterday + recent (default)
  s3_finals_to_json.py --all                   # match mode: process ALL prediction files
  s3_finals_to_json.py --backfill START END    # backfill mode: all games in date range

Daily mode (--date) fetches final scores for yesterday and any recent dates
(last 7 days) that are missing. This is the mode used by the daily pipeline.

Match mode (--all) scans predictions/json/ for predictions_YYYY-MM-DD.json files.
For each date before today, loads actual game results from S3 and writes
final_scores_{date}.json to both predictions/json/ and site/public/data/.

Backfill mode fetches all games from S3 for each date in [START, END]
and writes final_scores files directly, without requiring prediction files.
"""
import json
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone

from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")

# Allow running as standalone script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config
from src.s3_reader import get_column, read_silver_table


PREDICTIONS_RE = re.compile(r"^predictions_(\d{4}-\d{2}-\d{2})\.json$")
# Also match raw prediction files without the "predictions_" prefix
RAW_PRED_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.json$")


def _to_eastern_date(raw_date: str) -> str | None:
    """Convert an S3 startDate string to YYYY-MM-DD in US/Eastern.

    This matches the timezone logic in build_features() so that prediction
    dates and final-score dates are consistent.
    """
    if not raw_date:
        return None
    try:
        dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(_ET).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        # Fall back to regex extraction if not a valid ISO string
        match = re.search(r"(\d{4})-(\d{2})-(\d{2})", raw_date)
        return match.group(0) if match else None


def slugify(text: str) -> str:
    lowered = (text or "").lower()
    slug = re.sub(r"[^a-z0-9]+", "_", lowered)
    return slug.strip("_")


def list_prediction_dates() -> list[str]:
    """Scan predictions/json/ for prediction file dates."""
    json_dir = config.PREDICTIONS_DIR / "json"
    if not json_dir.exists():
        return []
    dates = set()
    for fname in sorted(json_dir.iterdir()):
        match = PREDICTIONS_RE.match(fname.name) or RAW_PRED_RE.match(fname.name)
        if match:
            dates.add(match.group(1))
    return sorted(dates)


def load_prediction_games(pred_date: str) -> list[dict]:
    """Load games from a predictions JSON file to get team names."""
    json_path = config.PREDICTIONS_DIR / "json" / f"predictions_{pred_date}.json"
    if not json_path.exists():
        return []
    with open(json_path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("games", data.get("predictions", []))
    return []


def get_season_for_date(d: str) -> int:
    """College basketball season: Nov-Apr. Nov 2025 → season 2026."""
    parts = d.split("-")
    year, month = int(parts[0]), int(parts[1])
    if month >= 11:
        return year + 1
    return year


def normalize_team(name: str) -> str:
    """Normalize team name for matching."""
    return slugify(name)


def fetch_scores_for_date(pred_date: str, season: int) -> dict[str, dict]:
    """Fetch game scores from S3 for a given date. Returns {normalized_key: score_info}."""
    tbl = read_silver_table("fct_games", season=season)
    if tbl.num_rows == 0:
        return {}

    game_ids = get_column(tbl, "gameId", "game_id", "id")
    start_dates = get_column(tbl, "startDate", "startTime", "date", "start_date")
    home_teams = get_column(tbl, "homeTeam", "home_team", "homeTeamName")
    away_teams = get_column(tbl, "awayTeam", "away_team", "awayTeamName")
    home_scores = get_column(tbl, "homeScore", "homePoints", "home_score", "home_points")
    away_scores = get_column(tbl, "awayScore", "awayPoints", "away_score", "away_points")

    scores = {}
    for i in range(tbl.num_rows):
        raw_date = str(start_dates[i] or "")
        game_date = _to_eastern_date(raw_date)
        if not game_date or game_date != pred_date:
            continue

        home = str(home_teams[i] or "")
        away = str(away_teams[i] or "")
        h_score = home_scores[i]
        a_score = away_scores[i]

        if h_score is None or a_score is None:
            continue
        # S3 has duplicate rows with 0-0 scores for unfinished games
        if int(h_score) == 0 and int(a_score) == 0:
            continue

        # Create lookup key
        key = f"{normalize_team(away)}__{normalize_team(home)}"
        scores[key] = {
            "game_id_s3": game_ids[i],
            "away_team": away,
            "home_team": home,
            "away_score": int(a_score) if a_score is not None else None,
            "home_score": int(h_score) if h_score is not None else None,
        }

    return scores


def build_final_scores(pred_date: str, pred_games: list[dict], s3_scores: dict) -> dict:
    """Match prediction games to S3 scores and build final scores payload."""
    games = []
    for pg in pred_games:
        away = str(
            pg.get("away_team")
            or pg.get("awayTeam")
            or pg.get("away_team_name")
            or ""
        )
        home = str(
            pg.get("home_team")
            or pg.get("homeTeam")
            or pg.get("home_team_name")
            or ""
        )
        game_id = pg.get("game_id", slugify(f"{pred_date}_{away}_{home}"))

        # Try matching by normalized team names
        key = f"{normalize_team(away)}__{normalize_team(home)}"
        match = s3_scores.get(key)

        if match:
            games.append({
                "game_id": game_id,
                "away_team": away,
                "home_team": home,
                "away_score": match["away_score"],
                "home_score": match["home_score"],
            })
        else:
            # No match found — game may not have been played yet
            pass

    return {
        "date": pred_date,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "games": games,
    }


def fetch_all_scores_for_date(target_date: str, season: int) -> list[dict]:
    """Fetch ALL game scores from S3 for a given date (backfill mode)."""
    tbl = read_silver_table("fct_games", season=season)
    if tbl.num_rows == 0:
        return []

    game_ids = get_column(tbl, "gameId", "game_id", "id")
    start_dates = get_column(tbl, "startDate", "startTime", "date", "start_date")
    home_teams = get_column(tbl, "homeTeam", "home_team", "homeTeamName")
    away_teams = get_column(tbl, "awayTeam", "away_team", "awayTeamName")
    home_scores = get_column(tbl, "homeScore", "homePoints", "home_score", "home_points")
    away_scores = get_column(tbl, "awayScore", "awayPoints", "away_score", "away_points")

    games = []
    for i in range(tbl.num_rows):
        raw_date = str(start_dates[i] or "")
        game_date = _to_eastern_date(raw_date)
        if not game_date or game_date != target_date:
            continue

        home = str(home_teams[i] or "")
        away = str(away_teams[i] or "")
        h_score = home_scores[i]
        a_score = away_scores[i]

        if h_score is None or a_score is None:
            continue
        if int(h_score) == 0 and int(a_score) == 0:
            continue

        game_id = slugify(f"{target_date}_{away}_{home}")
        games.append({
            "game_id": game_id,
            "away_team": away,
            "home_team": home,
            "away_score": int(a_score),
            "home_score": int(h_score),
        })

    return games


def write_final_scores(pred_date: str, payload: dict) -> None:
    """Write final scores JSON to predictions/json/ and site/public/data/."""
    repo_root = config.PROJECT_ROOT
    output_dirs = [
        repo_root / "predictions" / "json",
        repo_root / "site" / "public" / "data",
    ]

    filename = f"final_scores_{pred_date}.json"
    for out_dir in output_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"Wrote {out_path}")


def run_daily_mode(target_date: str) -> int:
    """Daily mode: fetch final scores for recent dates only (last 7 days)."""
    today_d = date.fromisoformat(target_date)
    pred_dates = list_prediction_dates()

    if not pred_dates:
        print("No prediction files found in predictions/json/", file=sys.stderr)
        return 1

    # Only look at the last 7 days (covers yesterday + any recent gaps)
    cutoff = (today_d - timedelta(days=7)).isoformat()
    recent_dates = [d for d in pred_dates if cutoff <= d < target_date]
    if not recent_dates:
        print(f"No recent prediction dates to process (window: {cutoff} to {target_date}).")
        return 0

    print(f"Processing {len(recent_dates)} recent dates for final scores...")

    for pred_date in recent_dates:
        existing = config.PROJECT_ROOT / "site" / "public" / "data" / f"final_scores_{pred_date}.json"
        if existing.exists():
            continue

        season = get_season_for_date(pred_date)
        pred_games = load_prediction_games(pred_date)
        if not pred_games:
            print(f"  {pred_date}: no prediction games found, skipping")
            continue

        s3_scores = fetch_scores_for_date(pred_date, season)
        if not s3_scores:
            print(f"  {pred_date}: no S3 scores found for season {season}, skipping")
            continue

        payload = build_final_scores(pred_date, pred_games, s3_scores)
        matched = len(payload["games"])
        total = len(pred_games)
        print(f"  {pred_date}: matched {matched}/{total} games")

        if matched > 0:
            write_final_scores(pred_date, payload)

    return 0


def run_match_mode() -> int:
    """Full match mode: process ALL prediction files (use --all flag)."""
    today = date.today().isoformat()
    pred_dates = list_prediction_dates()

    if not pred_dates:
        print("No prediction files found in predictions/json/", file=sys.stderr)
        return 1

    # Only process dates before today (games that should have final scores)
    past_dates = [d for d in pred_dates if d < today]
    if not past_dates:
        print("No past dates to process.")
        return 0

    print(f"Processing {len(past_dates)} dates for final scores...")

    for pred_date in past_dates:
        # Check if final scores already exist
        existing = config.PROJECT_ROOT / "site" / "public" / "data" / f"final_scores_{pred_date}.json"
        if existing.exists():
            continue

        season = get_season_for_date(pred_date)
        pred_games = load_prediction_games(pred_date)
        if not pred_games:
            print(f"  {pred_date}: no prediction games found, skipping")
            continue

        s3_scores = fetch_scores_for_date(pred_date, season)
        if not s3_scores:
            print(f"  {pred_date}: no S3 scores found for season {season}, skipping")
            continue

        payload = build_final_scores(pred_date, pred_games, s3_scores)
        matched = len(payload["games"])
        total = len(pred_games)
        print(f"  {pred_date}: matched {matched}/{total} games")

        if matched > 0:
            write_final_scores(pred_date, payload)

    return 0


def run_backfill(start_str: str, end_str: str) -> int:
    """Backfill mode: fetch all games from S3 for each date in [start, end]."""
    start = date.fromisoformat(start_str)
    end = date.fromisoformat(end_str)
    if start > end:
        print(f"Start date {start_str} is after end date {end_str}", file=sys.stderr)
        return 1

    total_days = (end - start).days + 1
    print(f"Backfilling final scores for {total_days} days: {start_str} → {end_str}")

    # Cache S3 tables by season to avoid re-reading
    season_cache: dict[int, object] = {}
    wrote = 0
    skipped_existing = 0
    skipped_no_games = 0

    current = start
    while current <= end:
        d = current.isoformat()
        current += timedelta(days=1)

        # Skip if already exists
        existing = config.PROJECT_ROOT / "site" / "public" / "data" / f"final_scores_{d}.json"
        if existing.exists():
            skipped_existing += 1
            continue

        season = get_season_for_date(d)

        # Cache the S3 table per season
        if season not in season_cache:
            print(f"  Loading S3 fct_games for season={season}...")
            tbl = read_silver_table("fct_games", season=season)
            if tbl.num_rows == 0:
                print(f"  Season {season}: no data in S3")
                season_cache[season] = None
            else:
                # Pre-extract columns once
                season_cache[season] = {
                    "game_ids": get_column(tbl, "gameId", "game_id", "id"),
                    "start_dates": get_column(tbl, "startDate", "startTime", "date", "start_date"),
                    "home_teams": get_column(tbl, "homeTeam", "home_team", "homeTeamName"),
                    "away_teams": get_column(tbl, "awayTeam", "away_team", "awayTeamName"),
                    "home_scores": get_column(tbl, "homeScore", "homePoints", "home_score", "home_points"),
                    "away_scores": get_column(tbl, "awayScore", "awayPoints", "away_score", "away_points"),
                    "num_rows": tbl.num_rows,
                }

        cached = season_cache.get(season)
        if cached is None:
            skipped_no_games += 1
            continue

        # Extract games for this date from cached data
        games = []
        for i in range(cached["num_rows"]):
            raw_date = str(cached["start_dates"][i] or "")
            game_date = _to_eastern_date(raw_date)
            if not game_date or game_date != d:
                continue

            home = str(cached["home_teams"][i] or "")
            away = str(cached["away_teams"][i] or "")
            h_score = cached["home_scores"][i]
            a_score = cached["away_scores"][i]

            if h_score is None or a_score is None:
                continue
            if int(h_score) == 0 and int(a_score) == 0:
                continue

            game_id = slugify(f"{d}_{away}_{home}")
            games.append({
                "game_id": game_id,
                "away_team": away,
                "home_team": home,
                "away_score": int(a_score),
                "home_score": int(h_score),
            })

        if not games:
            skipped_no_games += 1
            continue

        payload = {
            "date": d,
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "games": games,
        }
        write_final_scores(d, payload)
        wrote += 1
        print(f"  {d}: {len(games)} games")

    print(f"\nDone. Wrote {wrote} files, skipped {skipped_existing} existing, {skipped_no_games} with no games.")
    return 0


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "--backfill":
        if len(sys.argv) < 4:
            print("Usage: s3_finals_to_json.py --backfill START END", file=sys.stderr)
            print("  e.g.: s3_finals_to_json.py --backfill 2025-11-01 2026-02-25", file=sys.stderr)
            return 1
        return run_backfill(sys.argv[2], sys.argv[3])
    if len(sys.argv) >= 2 and sys.argv[1] == "--all":
        return run_match_mode()
    # Default: daily mode (--date DATE or today)
    if len(sys.argv) >= 2 and sys.argv[1] == "--date":
        target = sys.argv[2] if len(sys.argv) >= 3 else date.today().isoformat()
    else:
        target = date.today().isoformat()
    return run_daily_mode(target)


if __name__ == "__main__":
    raise SystemExit(main())
