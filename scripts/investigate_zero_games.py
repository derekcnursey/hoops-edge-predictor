"""Investigate WHY 1,646 games have scores of 0-0.

Check raw S3 silver data, PBP availability, and patterns to determine if this
is a pipeline bug (losing real scores) or genuine API data gaps.
"""

import sys
sys.path.insert(0, "/Users/dereknursey/Desktop/ml_projects/hoops-edge-predictor")

import pandas as pd
import numpy as np
from src import config, s3_reader

# ── Step 1: Load fct_games for all seasons and identify 0-0 games ──────────

print("=" * 80)
print("STEP 1: Load silver/fct_games and identify 0-0 games")
print("=" * 80)

all_games = []
for season in range(2015, 2026):
    tbl = s3_reader.read_silver_table(config.TABLE_FCT_GAMES, season=season)
    if tbl.num_rows == 0:
        print(f"  Season {season}: no data")
        continue
    df = tbl.to_pandas()
    df["_season"] = season
    all_games.append(df)
    print(f"  Season {season}: {len(df)} games, columns: {list(df.columns)[:10]}...")

games = pd.concat(all_games, ignore_index=True)
print(f"\nTotal games loaded: {len(games)}")
print(f"All columns: {list(games.columns)}")

# Check all score column candidates
score_cols = [c for c in games.columns if any(x in c.lower() for x in ["score", "point", "pts"])]
print(f"\nScore-related columns found: {score_cols}")

# Normalize score columns
for target, candidates in [
    ("homeScore", ["homeScore", "homePoints"]),
    ("awayScore", ["awayScore", "awayPoints"]),
    ("gameId", ["gameId"]),
    ("homeTeamId", ["homeTeamId"]),
    ("awayTeamId", ["awayTeamId"]),
    ("homeTeam", ["homeTeam"]),
    ("awayTeam", ["awayTeam"]),
    ("startDate", ["startDate", "startTime", "date"]),
]:
    for cand in candidates:
        if cand in games.columns and target not in games.columns:
            games = games.rename(columns={cand: target})
            break

# Find 0-0 games
zero_mask = (games["homeScore"] == 0) & (games["awayScore"] == 0)
zero_games = games[zero_mask].copy()
print(f"\n0-0 games found: {len(zero_games)}")

# ── Step 2: Examine 10 specific 0-0 games from the audit ───────────────────

print("\n" + "=" * 80)
print("STEP 2: Detailed look at 10 sample 0-0 games (raw silver columns)")
print("=" * 80)

sample_ids = [61428, 65234, 65395, 60503, 63630, 63629, 63675, 63933, 64582, 64607]
sample = games[games["gameId"].isin(sample_ids)]

print(f"\nFound {len(sample)} of {len(sample_ids)} sample games in silver data")
print("\nFull raw data for each sample game:")
for _, row in sample.iterrows():
    print(f"\n  gameId={row.get('gameId', 'N/A')}  Season={row.get('_season', 'N/A')}")
    # Print ALL columns for this game
    for col in sorted(games.columns):
        val = row.get(col)
        if pd.notna(val) and val != "" and str(val) != "nan":
            print(f"    {col}: {val}")
        else:
            print(f"    {col}: <NULL/NaN>")

# ── Step 3: Check if PBP/boxscore data exists for these games ──────────────

print("\n" + "=" * 80)
print("STEP 3: Check PBP/boxscore data for 0-0 games")
print("=" * 80)

# Load PBP for the relevant seasons
zero_seasons = zero_games["_season"].unique()
print(f"0-0 games found in seasons: {sorted(zero_seasons)}")

pbp_game_ids = set()
for season in sorted(zero_seasons):
    tbl = s3_reader.read_silver_table(config.TABLE_FCT_GAME_TEAMS, season=season)
    if tbl.num_rows == 0:
        print(f"  Season {season}: no PBP data")
        continue
    df_pbp = tbl.to_pandas()
    # Find game ID column
    gid_col = None
    for c in ["gameid", "gameId", "game_id"]:
        if c in df_pbp.columns:
            gid_col = c
            break
    if gid_col is None:
        print(f"  Season {season}: no game ID column in PBP! Columns: {list(df_pbp.columns)[:10]}")
        continue
    season_pbp_ids = set(df_pbp[gid_col].unique())
    pbp_game_ids.update(season_pbp_ids)

    # Check overlap with 0-0 games
    season_zero = zero_games[zero_games["_season"] == season]
    zero_ids_in_season = set(season_zero["gameId"])
    overlap = zero_ids_in_season & season_pbp_ids
    print(f"  Season {season}: {len(season_zero)} zero-games, "
          f"{len(overlap)} have PBP data ({100*len(overlap)/max(len(season_zero),1):.1f}%)")

# Check sample games specifically
print("\nSample 0-0 games — PBP data available?")
for gid in sample_ids:
    has_pbp = gid in pbp_game_ids
    row = sample[sample["gameId"] == gid]
    if len(row) > 0:
        row = row.iloc[0]
        home = row.get("homeTeam", row.get("homeTeamId", "?"))
        away = row.get("awayTeam", row.get("awayTeamId", "?"))
        dt = row.get("startDate", "?")
        print(f"  gameId={gid}: {away} @ {home} ({dt})  PBP={'YES' if has_pbp else 'NO'}")
    else:
        print(f"  gameId={gid}: NOT FOUND in silver data")

# ── Step 4: Check PBP score data for games that HAVE boxscores ─────────────

print("\n" + "=" * 80)
print("STEP 4: Can we recover scores from PBP/boxscore data?")
print("=" * 80)

# For 0-0 games that have PBP data, check if the boxscore has non-zero stats
for season in sorted(zero_seasons)[:3]:  # Check first 3 seasons
    tbl = s3_reader.read_silver_table(config.TABLE_FCT_GAME_TEAMS, season=season)
    if tbl.num_rows == 0:
        continue
    df_pbp = tbl.to_pandas()

    gid_col = None
    for c in ["gameid", "gameId", "game_id"]:
        if c in df_pbp.columns:
            gid_col = c
            break
    if gid_col is None:
        continue

    season_zero_ids = set(zero_games[zero_games["_season"] == season]["gameId"])
    pbp_zero = df_pbp[df_pbp[gid_col].isin(season_zero_ids)]

    if len(pbp_zero) > 0:
        print(f"\n  Season {season}: {len(pbp_zero)} PBP rows for {len(season_zero_ids)} zero-score games")
        # Check if FG data is non-zero
        fg_cols = [c for c in pbp_zero.columns if "fg" in c.lower() or "point" in c.lower() or "score" in c.lower()]
        print(f"  FG/score columns in PBP: {fg_cols}")
        if fg_cols:
            for col in fg_cols[:6]:
                nz = (pbp_zero[col] != 0).sum() if col in pbp_zero.columns else 0
                print(f"    {col}: {nz}/{len(pbp_zero)} non-zero")
    else:
        print(f"\n  Season {season}: NO PBP rows for zero-score games")

# ── Step 5: Pattern analysis — when do 0-0 games happen? ──────────────────

print("\n" + "=" * 80)
print("STEP 5: Pattern analysis of 0-0 games")
print("=" * 80)

# Season distribution (detailed)
print("\nSeason breakdown:")
season_counts = zero_games.groupby("_season").size()
total_counts = games.groupby("_season").size()
for season in sorted(zero_seasons):
    z = season_counts.get(season, 0)
    t = total_counts.get(season, 0)
    print(f"  {season}: {z:>5} / {t:>6} games ({100*z/max(t,1):.1f}%)")

# Date pattern
if "startDate" in zero_games.columns:
    zero_games["_date"] = pd.to_datetime(zero_games["startDate"], errors="coerce")
    zero_games["_month"] = zero_games["_date"].dt.month

    print("\nMonth distribution of 0-0 games:")
    month_counts = zero_games.groupby("_month").size()
    for m, c in month_counts.items():
        if pd.notna(m):
            print(f"  Month {int(m):>2}: {c:>5} games")

    # Check date range for COVID era (2021 season)
    covid = zero_games[zero_games["_season"] == 2021]
    if len(covid) > 0:
        print(f"\nCOVID 2021 season 0-0 games:")
        print(f"  Date range: {covid['_date'].min()} to {covid['_date'].max()}")
        print(f"  Total: {len(covid)}")
        # Sample some
        print(f"  Sample games:")
        for _, row in covid.head(10).iterrows():
            home = row.get("homeTeam", row.get("homeTeamId", "?"))
            away = row.get("awayTeam", row.get("awayTeamId", "?"))
            dt = row.get("startDate", "?")
            hs = row.get("homeScore", "?")
            aws = row.get("awayScore", "?")
            print(f"    {dt}: {away} @ {home} — score: {aws}-{hs}")

# ── Step 6: Check non-COVID 0-0 games (2023-2025) ─────────────────────────

print("\n" + "=" * 80)
print("STEP 6: Non-COVID 0-0 games (recent seasons)")
print("=" * 80)

recent_zero = zero_games[zero_games["_season"] >= 2023]
print(f"\nRecent 0-0 games (2023+): {len(recent_zero)}")
for _, row in recent_zero.iterrows():
    gid = row.get("gameId", "?")
    home = row.get("homeTeam", row.get("homeTeamId", "?"))
    away = row.get("awayTeam", row.get("awayTeamId", "?"))
    dt = row.get("startDate", "?")
    season = row.get("_season", "?")
    print(f"  gameId={gid}  {away} @ {home}  ({dt}, season={season})")

# ── Step 7: Check if other score columns exist in raw data ─────────────────

print("\n" + "=" * 80)
print("STEP 7: Check for alternative score fields in fct_games")
print("=" * 80)

# Check ALL columns that could hold scores
all_cols = list(games.columns)
print(f"All fct_games columns: {sorted(all_cols)}")

# For a sample 0-0 game, print every non-null column
sample_zero = zero_games.head(3)
for _, row in sample_zero.iterrows():
    gid = row.get("gameId", "?")
    print(f"\nAll non-null fields for gameId={gid}:")
    for col in sorted(all_cols):
        val = row.get(col)
        if pd.notna(val) and str(val) != "nan":
            print(f"  {col}: {val}")

# ── Step 8: Check if scores exist in a DIFFERENT asof partition ────────────

print("\n" + "=" * 80)
print("STEP 8: Check fct_games asof partitions (latest vs all)")
print("=" * 80)

# For season 2021 (most 0-0 games), check latest partition only
base_2021 = f"{config.SILVER_PREFIX}/{config.TABLE_FCT_GAMES}/season=2021/"
latest_prefix = s3_reader._get_latest_asof_prefix(base_2021)
print(f"Latest asof for 2021: {latest_prefix}")

# List all asof partitions
client = s3_reader._s3_client()
resp = client.list_objects_v2(
    Bucket=config.S3_BUCKET, Prefix=base_2021, Delimiter="/"
)
asof_partitions = [p["Prefix"] for p in resp.get("CommonPrefixes", [])]
print(f"All asof partitions for 2021: {len(asof_partitions)}")
for p in asof_partitions[:10]:
    print(f"  {p}")
if len(asof_partitions) > 10:
    print(f"  ... and {len(asof_partitions) - 10} more")

# Compare 0-0 count across partitions if multiple exist
if len(asof_partitions) > 1:
    print("\nComparing 0-0 counts across asof partitions (first 3):")
    for part in asof_partitions[:3]:
        keys = s3_reader.list_parquet_keys(part)
        if keys:
            tbl = s3_reader.read_parquet_table(keys)
            df_part = tbl.to_pandas()
            # Normalize columns
            for target, candidates in [
                ("homeScore", ["homeScore", "homePoints"]),
                ("awayScore", ["awayScore", "awayPoints"]),
            ]:
                for cand in candidates:
                    if cand in df_part.columns and target not in df_part.columns:
                        df_part = df_part.rename(columns={cand: target})
                        break
            if "homeScore" in df_part.columns and "awayScore" in df_part.columns:
                zz = ((df_part["homeScore"] == 0) & (df_part["awayScore"] == 0)).sum()
                print(f"  {part}: {len(df_part)} games, {zz} with 0-0 score")

# ── Step 9: Check the ETL sibling repo if accessible ───────────────────────

print("\n" + "=" * 80)
print("STEP 9: Summary — Root cause analysis")
print("=" * 80)

total_zero = len(zero_games)
zero_with_pbp = len(set(zero_games["gameId"]) & pbp_game_ids)
zero_without_pbp = total_zero - zero_with_pbp

print(f"""
FINDINGS:
  Total 0-0 games: {total_zero}
  Have PBP/boxscore data: {zero_with_pbp} ({100*zero_with_pbp/max(total_zero,1):.1f}%)
  Missing PBP data: {zero_without_pbp} ({100*zero_without_pbp/max(total_zero,1):.1f}%)

HYPOTHESIS:
  If 0-0 games LACK PBP data → These are games the API knows about (scheduled)
    but never got scores/boxscores populated (cancelled, postponed, or data gap).
  If 0-0 games HAVE PBP data → Pipeline bug losing scores during ETL transforms.
""")
