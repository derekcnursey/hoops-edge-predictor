#!/usr/bin/env python3
"""Run remaining tests (7, 8, 9, 11, 12) that didn't complete in the first run."""

from __future__ import annotations
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import everything from the main suite
from scripts.session13_validation_suite import (
    load_winner_model, load_val_data,
    test_7_feature_ablation, test_8_hp_stability,
    test_9_calibration_slices, test_11_line_staleness,
    test_12_bet_profiling,
)


def main():
    print("=" * 70)
    print("  REMAINING TESTS (7, 8, 9, 11, 12)")
    print("=" * 70)
    t_start = time.time()

    model = load_winner_model()
    X_val_s, y_val, book, df_val = load_val_data()

    # TEST 12: Bet profiling (fixed)
    test_12_bet_profiling(model, X_val_s, y_val, book, df_val)

    # TEST 7: Feature ablation (GPU)
    test_7_feature_ablation(model, X_val_s, y_val, book, df_val)

    # TEST 8: HP stability (GPU - trains 8 models)
    test_8_hp_stability()

    # TEST 9: Calibration slices
    test_9_calibration_slices(model, X_val_s, y_val, book, df_val)

    # TEST 11: Line staleness
    test_11_line_staleness(df_val)

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  REMAINING TESTS COMPLETE — {elapsed/60:.1f} minutes")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
