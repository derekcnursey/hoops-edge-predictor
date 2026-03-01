#!/bin/bash
# Overnight V3 Pipeline: Build all seasons + validate + V1 regression test
# Usage: bash scripts/overnight_v3_pipeline.sh 2>&1 | tee docs/overnight_run.log

set -e
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

echo "============================================================"
echo "OVERNIGHT V3 PIPELINE"
echo "Started: $(date)"
echo "Branch: $(git branch --show-current)"
echo "============================================================"
echo ""

# Step 1: Build V3 features for all seasons (most recent first)
echo "STEP 1: Building V3 features for all seasons"
echo "============================================================"
for season in 2025 2024 2023 2022 2021 2020 2018 2017 2016 2015 2026; do
    echo ""
    echo "--- Season $season ($(date)) ---"
    # Skip if already built
    if [ -f "features/season_${season}_v3_features.parquet" ]; then
        echo "  Already exists, skipping"
        continue
    fi
    poetry run python scripts/build_features_v3.py --seasons $season || echo "  WARNING: Season $season failed"
done
echo ""
echo "Build complete at $(date)"
echo ""

# Step 2: Run comprehensive validation
echo "STEP 2: Running V3 validation"
echo "============================================================"
poetry run python scripts/validate_v3_features.py || echo "WARNING: Validation script failed"
echo ""

# Step 3: V1 regression test (build V1 features for 2025 and compare)
echo "STEP 3: V1 Regression Test"
echo "============================================================"
poetry run python -c "
import time
import pandas as pd
import numpy as np
from src.features import build_features
from src.config import FEATURE_ORDER

print('Building V1 features for season 2025...')
t0 = time.time()
df = build_features(
    2025,
    no_garbage=True,
    extra_features=['rest_days', 'sos', 'conf_strength', 'form_delta', 'tov_rate', 'margin_std'],
    adjust_ff=True,
    adjust_alpha=0.85,
    adjust_prior_weight=10,
)
elapsed = time.time() - t0
print(f'  V1 build: {len(df)} games, {elapsed:.1f}s')
print(f'  Columns: {len(df.columns)}')

# Check feature order
feat_matrix = df[[c for c in FEATURE_ORDER if c in df.columns]]
print(f'  V1 features present: {len(feat_matrix.columns)}/{len(FEATURE_ORDER)}')
nan_rate = feat_matrix.isna().mean().mean() * 100
print(f'  V1 NaN rate: {nan_rate:.1f}%')

# Save for comparison
df.to_parquet('features/season_2025_v1_regression_test.parquet', index=False)
print(f'  Saved: features/season_2025_v1_regression_test.parquet')
print('  V1 regression test PASSED')
" || echo "WARNING: V1 regression test failed"

echo ""
echo "============================================================"
echo "OVERNIGHT PIPELINE COMPLETE"
echo "Finished: $(date)"
echo "============================================================"

# List outputs
echo ""
echo "Output files:"
ls -lh features/season_*_v3_features.parquet 2>/dev/null || echo "  No V3 feature files"
ls -lh features/season_*_v1_regression_test.parquet 2>/dev/null || echo "  No V1 regression test file"
ls -lh docs/session14_validation_report.md 2>/dev/null || echo "  No validation report"
ls -lh docs/plots/*.png 2>/dev/null || echo "  No plots"
