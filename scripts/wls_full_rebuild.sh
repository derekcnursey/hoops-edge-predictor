#!/usr/bin/env bash
# WLS Solver Full Pipeline Rebuild
# Rebuilds silver → gold → features for seasons 2015-2026 with WLS solver + 0.475 possessions.
#
# Prerequisites:
#   1. config.yaml already set to solver: wls
#   2. 0.475 possession constant in ETL build scripts
#
# Usage:
#   bash scripts/wls_full_rebuild.sh [--silver] [--gold] [--features] [--all]
#   bash scripts/wls_full_rebuild.sh --all          # full rebuild
#   bash scripts/wls_full_rebuild.sh --gold --features  # skip silver

set -euo pipefail

ETL_DIR="$HOME/Desktop/ml_projects/hoops_edge_database_etl"
PRED_DIR="$HOME/Desktop/ml_projects/hoops-edge-predictor"
SEASONS=(2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026)

DO_SILVER=false
DO_GOLD=false
DO_FEATURES=false

for arg in "$@"; do
  case "$arg" in
    --silver)   DO_SILVER=true ;;
    --gold)     DO_GOLD=true ;;
    --features) DO_FEATURES=true ;;
    --all)      DO_SILVER=true; DO_GOLD=true; DO_FEATURES=true ;;
    *)          echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

if ! $DO_SILVER && ! $DO_GOLD && ! $DO_FEATURES; then
  echo "Usage: $0 [--silver] [--gold] [--features] [--all]"
  exit 1
fi

LOG_DIR="$PRED_DIR/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/wls_rebuild_${TIMESTAMP}.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "WLS Full Pipeline Rebuild started"
log "Seasons: ${SEASONS[*]}"

# ── Step 1: Silver layer ──────────────────────────────────────────
if $DO_SILVER; then
  log "=== SILVER LAYER REBUILD ==="
  cd "$ETL_DIR"

  for S in "${SEASONS[@]}"; do
    log "Silver: season $S (regular)..."
    poetry run python scripts/build_pbp_game_teams_flat.py --season "$S" --purge 2>&1 | tail -3 | tee -a "$LOG"

    log "Silver: season $S (no-garbage)..."
    poetry run python scripts/build_pbp_game_teams_flat.py --season "$S" \
      --exclude-garbage-time \
      --output-table fct_pbp_game_teams_flat_garbage_removed \
      --purge 2>&1 | tail -3 | tee -a "$LOG"
  done

  log "Silver rebuild complete."

  # Spot-check: print avg pace for 2025 (should be ~67-68, was ~62)
  log "--- Silver spot-check: verifying 0.475 possession formula ---"
  poetry run python -c "
import pyarrow.parquet as pq
from cbbd_etl.s3_io import S3IO
from cbbd_etl.config import load_config
cfg = load_config()
s3 = S3IO(cfg['s3']['bucket'], cfg['s3']['region'])
prefix = f\"{cfg['s3']['silver_prefix']}/fct_pbp_game_teams_flat/season=2025\"
keys = s3.list_keys(prefix)
import io, numpy as np
paces = []
for k in keys[:5]:
    buf = s3.read_bytes(k)
    tbl = pq.read_table(io.BytesIO(buf))
    df = tbl.to_pandas()
    if 'pace' in df.columns:
        paces.extend(df['pace'].dropna().tolist())
if paces:
    avg = np.mean(paces)
    print(f'Avg pace (5 partitions, 2025): {avg:.1f}  (expect ~67-68)')
else:
    print('WARNING: No pace column found in silver data')
" 2>&1 | tee -a "$LOG"
fi

# ── Step 2: Gold layer ────────────────────────────────────────────
if $DO_GOLD; then
  log "=== GOLD LAYER REBUILD ==="
  cd "$ETL_DIR"

  for S in "${SEASONS[@]}"; do
    log "Gold: season $S (team_adjusted_efficiencies_no_garbage)..."
    poetry run python -m cbbd_etl.gold.runner --season "$S" \
      --table team_adjusted_efficiencies_no_garbage 2>&1 | tail -5 | tee -a "$LOG"
  done

  log "Gold rebuild complete."

  # Spot-check: top-10 teams by adj_margin for 2025
  log "--- Gold spot-check: 2025 top-10, adj_pace, HCA ---"
  poetry run python -c "
import pyarrow.parquet as pq
from cbbd_etl.s3_io import S3IO
from cbbd_etl.config import load_config
import io, pandas as pd, numpy as np
cfg = load_config()
s3 = S3IO(cfg['s3']['bucket'], cfg['s3']['region'])
prefix = f\"{cfg['s3']['gold_prefix']}/team_adjusted_efficiencies_no_garbage/season=2025\"
keys = s3.list_keys(prefix)
frames = []
for k in keys:
    buf = s3.read_bytes(k)
    tbl = pq.read_table(io.BytesIO(buf))
    frames.append(tbl.to_pandas())
if frames:
    df = pd.concat(frames, ignore_index=True)
    # Latest snapshot per team
    if 'as_of_date' in df.columns:
        df['as_of_date'] = pd.to_datetime(df['as_of_date'])
        df = df.sort_values('as_of_date').drop_duplicates('team_id', keep='last')
    if 'adj_oe' in df.columns and 'adj_de' in df.columns:
        df['adj_margin'] = df['adj_oe'] - df['adj_de']
        top10 = df.nlargest(10, 'adj_margin')[['team_id','adj_oe','adj_de','adj_margin']].reset_index(drop=True)
        print('Top-10 by adj_margin (2025):')
        print(top10.to_string())
    if 'adj_pace' in df.columns:
        print(f\"\\nAvg adj_pace: {df['adj_pace'].mean():.1f} (expect ~67-68)\")
    if 'hca_pts_per_100' in df.columns:
        print(f\"HCA pts/100poss: {df['hca_pts_per_100'].iloc[-1]:.2f} (expect ~2.5)\")
else:
    print('WARNING: No gold data found for 2025')
" 2>&1 | tee -a "$LOG"
fi

# ── Step 3: Features ──────────────────────────────────────────────
if $DO_FEATURES; then
  log "=== FEATURE REBUILD ==="
  cd "$PRED_DIR"

  for S in "${SEASONS[@]}"; do
    log "Features: season $S..."
    poetry run python -m src.cli build-features --season "$S" --no-garbage --adjusted 2>&1 | tail -5 | tee -a "$LOG"
  done

  log "Feature rebuild complete."

  # Spot-check: verify feature count and adj_pace
  log "--- Feature spot-check ---"
  poetry run python -c "
import pandas as pd, numpy as np
from pathlib import Path
feat_dir = Path('features')
for s in [2020, 2025]:
    f = feat_dir / f'season_{s}_no_garbage_adj_a0.85_p10_features.parquet'
    if f.exists():
        df = pd.read_parquet(f)
        print(f'Season {s}: {len(df)} games, {df.shape[1]} columns')
        for col in ['home_adj_pace', 'away_adj_pace', 'adj_pace']:
            if col in df.columns:
                print(f'  {col}: mean={df[col].mean():.1f}, std={df[col].std():.1f}')
        nan_cols = df.isnull().sum()
        bad = nan_cols[nan_cols > len(df)*0.1]
        if len(bad) > 0:
            print(f'  WARNING: >10% NaN in: {dict(bad)}')
        else:
            print(f'  NaN check: OK (no col >10% NaN)')
    else:
        print(f'Season {s}: FILE NOT FOUND at {f}')
" 2>&1 | tee -a "$LOG"
fi

log "=== ALL DONE ==="
log "Log saved to: $LOG"
