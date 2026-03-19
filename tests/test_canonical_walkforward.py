"""Focused tests for canonical walk-forward benchmark line benchmark behavior."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _load_module():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "canonical_walkforward.py"
    spec = importlib.util.spec_from_file_location("canonical_walkforward", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cw = _load_module()


def test_book_benchmark_metadata_is_honest_and_stable():
    meta = cw._line_selection_metadata()

    assert meta["benchmark_label"] == "PreferredBookSpread"
    assert meta["uses_true_closing_timestamps"] is False
    assert meta["provider_preference_order"] == ["Draft Kings", "ESPN BET", "Bovada"]
    assert "first non-null spread" in meta["selection_rule"]


def test_dedupe_lines_matches_documented_provider_preference_behavior():
    lines = pd.DataFrame([
        {"gameId": 1, "provider": "Bovada", "spread": -4.5},
        {"gameId": 1, "provider": "Draft Kings", "spread": -3.5},
        {"gameId": 1, "provider": "ESPN BET", "spread": -4.0},
        {"gameId": 2, "provider": "Draft Kings", "spread": None},
        {"gameId": 2, "provider": "ESPN BET", "spread": 2.5},
        {"gameId": 2, "provider": "Bovada", "spread": 1.5},
        {"gameId": 3, "provider": "ZooBook", "spread": -1.5},
        {"gameId": 3, "provider": "AlphaBook", "spread": -2.0},
    ])

    deduped = cw._dedupe_lines(lines).sort_values("gameId").reset_index(drop=True)

    assert deduped.loc[0, "book_spread"] == -3.5
    assert deduped.loc[1, "book_spread"] == 2.5
    assert deduped.loc[2, "book_spread"] == -2.0


def test_book_benchmark_frame_uses_negative_book_spread_sign_convention():
    holdout = pd.DataFrame([
        {
            "gameId": 10,
            "startDate": "2025-01-15",
            "homeTeamId": 1,
            "awayTeamId": 2,
            "homeTeam": "Home",
            "awayTeam": "Away",
            "homeScore": 75,
            "awayScore": 70,
            "book_spread": -3.5,
        }
    ])

    out = cw._build_book_spread_benchmark_frame(2025, holdout)

    assert out.loc[0, "model"] == "PreferredBookSpread"
    assert out.loc[0, "actual_margin"] == 5.0
    assert out.loc[0, "pred_margin"] == 3.5
    assert out.loc[0, "pred_home_win"] == 1


def test_summary_documents_honest_book_benchmark_rule(tmp_path):
    fold_metrics = pd.DataFrame([
        {
            "model": "PreferredBookSpread",
            "holdout_season": 2025,
            "n_games": 1,
            "n_lined": 1,
            "MAE_all": float("nan"),
            "RMSE_all": float("nan"),
            "MedAE_all": float("nan"),
            "WinAcc_all": float("nan"),
            "MAE_lined": 1.0,
            "BookMAE_lined": 1.0,
            "DeltaVsBook_MAE": 0.0,
        }
    ])
    pooled_metrics = pd.DataFrame([
        {
            "model": "PreferredBookSpread",
            "n_games": 1,
            "n_lined": 1,
            "MAE_all": float("nan"),
            "RMSE_all": float("nan"),
            "MedAE_all": float("nan"),
            "WinAcc_all": float("nan"),
            "MAE_lined": 1.0,
            "BookMAE_lined": 1.0,
            "DeltaVsBook_MAE": 0.0,
            "GaussianNLL": float("nan"),
            "MeanAbsZ": float("nan"),
            "Coverage_1sigma": float("nan"),
            "Coverage_2sigma": float("nan"),
        }
    ])

    cw._write_summary(tmp_path, fold_metrics, pooled_metrics)
    summary = (tmp_path / "summary.md").read_text()

    assert "PreferredBookSpread" in summary
    assert "Uses true closing timestamps: False" in summary
    assert "Provider preference order: Draft Kings, ESPN BET, Bovada" in summary
    assert "first non-null spread" in summary


def test_lightgbm_predict_path_runs_on_small_synthetic_fold():
    rows = []
    for i in range(20):
        row = {
            "gameId": 1000 + i,
            "startDate": f"2024-01-{(i % 28) + 1:02d}",
            "homeScore": float(70 + (i % 11)),
            "awayScore": float(65 + (i % 9)),
        }
        for j, feature in enumerate(cw.config.FEATURE_ORDER):
            row[feature] = float((i + 1) * (j + 1) % 17) / 10.0
        rows.append(row)
    train_df = pd.DataFrame(rows)

    test_rows = []
    for i in range(5):
        row = {
            "gameId": 2000 + i,
            "startDate": f"2024-02-{(i % 28) + 1:02d}",
            "homeScore": float(72 + i),
            "awayScore": float(66 + i),
        }
        for j, feature in enumerate(cw.config.FEATURE_ORDER):
            row[feature] = float((i + 3) * (j + 2) % 19) / 10.0
        test_rows.append(row)
    test_df = pd.DataFrame(test_rows)

    pred, best_iteration = cw._predict_lightgbm(train_df, test_df)

    assert pred.shape == (len(test_df),)
    assert np.isfinite(pred).all()
    assert best_iteration > 0


def _make_synthetic_fold(n_train: int = 24) -> pd.DataFrame:
    rows = []
    for i in range(n_train):
        row = {
            "gameId": 5000 + i,
            "startDate": f"2024-01-{(i % 28) + 1:02d}T19:00:00Z",
            "homeScore": float(68 + (i % 13)),
            "awayScore": float(63 + (i % 11)),
            "neutralSite": 0,
        }
        for j, feature in enumerate(cw.config.FEATURE_ORDER):
            row[feature] = float(((i + 1) * (j + 3)) % 23) / 10.0
        if "neutral_site" in cw.config.FEATURE_ORDER:
            row["neutral_site"] = 0.0
        if "home_team_hca" in cw.config.FEATURE_ORDER:
            row["home_team_hca"] = 3.2
        rows.append(row)
    return pd.DataFrame(rows)


def _make_neutral_test_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    row = {
        "gameId": 9001,
        "startDate": "2025-03-21T20:00:00Z",
        "homeScore": 75.0,
        "awayScore": 70.0,
        "neutralSite": 1,
    }
    for j, feature in enumerate(cw.config.FEATURE_ORDER):
        row[feature] = float((j * 7 + 5) % 29) / 10.0
    if "neutral_site" in cw.config.FEATURE_ORDER:
        row["neutral_site"] = 1.0
    if "home_team_hca" in cw.config.FEATURE_ORDER:
        row["home_team_hca"] = 0.0
    test_df = pd.DataFrame([row])

    feature_df = cw.get_feature_matrix(test_df).copy()
    swapped_feature_df = cw._swap_feature_frame(feature_df, list(feature_df.columns))
    swapped_df = test_df.copy()
    for col in swapped_feature_df.columns:
        swapped_df[col] = swapped_feature_df[col].values
    return test_df, swapped_df


def test_hgbr_neutral_predictions_are_antisymmetric_under_slot_swap():
    train_df = _make_synthetic_fold()
    test_df, swapped_df = _make_neutral_test_pair()

    pred = cw._predict_hgbr(train_df, test_df)
    pred_swap = cw._predict_hgbr(train_df, swapped_df)

    assert pred.shape == (1,)
    assert pred_swap.shape == (1,)
    assert np.isclose(pred[0], -pred_swap[0], atol=1e-6)


def test_mlp_neutral_predictions_are_antisymmetric_with_matching_sigma():
    train_df = _make_synthetic_fold(28)
    test_df, swapped_df = _make_neutral_test_pair()

    old_epochs = cw.MLP_HP["epochs"]
    old_batch = cw.MLP_HP["batch_size"]
    cw.MLP_HP["epochs"] = 2
    cw.MLP_HP["batch_size"] = 16
    try:
        pred, sigma, _ = cw._predict_mlp(train_df, test_df)
        pred_swap, sigma_swap, _ = cw._predict_mlp(train_df, swapped_df)
    finally:
        cw.MLP_HP["epochs"] = old_epochs
        cw.MLP_HP["batch_size"] = old_batch

    assert pred.shape == (1,)
    assert pred_swap.shape == (1,)
    assert sigma.shape == (1,)
    assert sigma_swap.shape == (1,)
    assert np.isclose(pred[0], -pred_swap[0], atol=1e-6)
    assert np.isclose(sigma[0], sigma_swap[0], atol=1e-6)
