from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _load_module():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "benchmark_tiebreak_lined.py"
    spec = importlib.util.spec_from_file_location("benchmark_tiebreak_lined", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


mod = _load_module()


def test_phase_label_maps_calendar_months():
    dates = pd.Series([
        "2024-11-15T00:00:00.000Z",
        "2024-12-01T00:00:00.000Z",
        "2025-01-10T00:00:00.000Z",
        "2025-03-05T00:00:00.000Z",
        "2025-04-01T00:00:00.000Z",
    ])
    got = mod._phase_label(dates).tolist()
    assert got[:4] == ["Nov-Dec", "Nov-Dec", "Jan-Mar", "Jan-Mar"]
    assert pd.isna(got[4])


def test_bootstrap_ci_handles_constant_values():
    lo, hi = mod._bootstrap_ci(np.array([0.1, 0.1, 0.1]))
    assert lo == pytest.approx(0.1)
    assert hi == pytest.approx(0.1)


def test_summarize_slice_uses_lgb_minus_hgbr_convention():
    df = pd.DataFrame({
        "hgbr_abs_error": [1.0, 2.0, 3.0],
        "lgb_abs_error": [0.5, 2.5, 2.0],
        "mae_diff_lgb_minus_hgbr": [-0.5, 0.5, -1.0],
    })
    row = mod._summarize_slice(df, "x")
    assert row["hgbr_mae_lined"] == 2.0
    assert row["lgb_mae_lined"] == 5.0 / 3.0
    assert row["mae_diff_lgb_minus_hgbr"] == -1.0 / 3.0
    assert row["lgb_game_win_rate"] == 2.0 / 3.0
