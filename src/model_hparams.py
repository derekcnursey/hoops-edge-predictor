"""Helpers for loading persisted model hyperparameters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import config


def best_hparams_path() -> Path:
    return config.ARTIFACTS_DIR / "best_hparams.json"


def load_best_hparams() -> dict[str, Any]:
    path = best_hparams_path()
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_best_hparams(payload: dict[str, Any]) -> None:
    path = best_hparams_path()
    path.write_text(json.dumps(payload, indent=2) + "\n")


def production_mu_hparams() -> dict[str, Any]:
    best = load_best_hparams()
    tuned = best.get("mu_regressor", {})
    return {**config.LGBM_REG_L2_PARAMS, **tuned}
