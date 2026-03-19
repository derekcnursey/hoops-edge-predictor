"""Source-aware fallback margins for synthetic NCAA neutral matchups."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DirectMarginFit:
    intercept: float
    coef_simple_eff: float


ROUND64_OPENING_LABELS = {"First Four", "Round of 64"}
GOLD_DIRECT_FIT_NCAA = DirectMarginFit(
    intercept=0.953555,
    coef_simple_eff=1.082141,
)
GOLD_DIRECT_FIT_ROUND64 = DirectMarginFit(
    intercept=1.517813,
    coef_simple_eff=1.137470,
)


def simple_efficiency_margin(team_a: pd.Series, team_b: pd.Series) -> float:
    """Map dated team ratings into a neutral-site signed spread proxy."""
    team_a_net = float(team_a["adj_oe"]) - float(team_a["adj_de"])
    team_b_net = float(team_b["adj_oe"]) - float(team_b["adj_de"])
    avg_tempo = 0.5 * (float(team_a["adj_tempo"]) + float(team_b["adj_tempo"]))
    return float((team_a_net - team_b_net) * (avg_tempo / 100.0))


def _normalize_source_label(ratings_source: str | None) -> str:
    source = (ratings_source or "").strip().lower()
    if source == "torvik":
        return "torvik"
    return "gold"


def direct_margin_fit_for_round(
    ratings_source: str | None,
    round_label: str | None,
) -> tuple[DirectMarginFit | None, str]:
    """Return the direct margin mapping for the requested source/round."""
    source = _normalize_source_label(ratings_source)
    if source == "torvik":
        return None, "torvik_simple_identity_v1"
    if round_label in ROUND64_OPENING_LABELS:
        return GOLD_DIRECT_FIT_ROUND64, "gold_direct_fitted_round64_v1"
    return GOLD_DIRECT_FIT_NCAA, "gold_direct_fitted_ncaa_v1"


def synthetic_fallback_margin(
    team_a: pd.Series,
    team_b: pd.Series,
    *,
    ratings_source: str | None,
    round_label: str | None,
) -> tuple[float, float, str]:
    """Return the synthetic fallback margin plus provenance metadata.

    The fallback is source-aware:
    - Torvik-backed synthetic matchups use the direct simple-efficiency identity.
    - Gold/internal synthetic matchups use the fitted NCAA mappings from the
      neutral-fix design study, with a stronger opening-round mapping for
      First Four / Round of 64 matchups.
    """
    simple_margin = simple_efficiency_margin(team_a, team_b)
    fit, label = direct_margin_fit_for_round(ratings_source, round_label)
    if fit is None:
        return simple_margin, simple_margin, label
    mapped_margin = fit.intercept + fit.coef_simple_eff * simple_margin
    return float(mapped_margin), simple_margin, label
