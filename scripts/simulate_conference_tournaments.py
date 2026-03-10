#!/usr/bin/env python3
"""Simulate all conference tournaments via Monte Carlo using the production model.

Uses REAL 2025-26 tournament formats: correct number of qualifiers, proper bye
structures, and stepladder brackets where applicable. Team seedings are derived
from conference records with adj_margin tiebreaker.

Extracts each team's latest rolling stats from the feature parquet, constructs
matchup feature vectors identical to the daily prediction pipeline, and runs
them through the regressor to get mu/sigma. P(A wins) = Φ(mu / sigma).

Usage:
    poetry run python scripts/simulate_conference_tournaments.py
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.infer import load_regressor
from src.trainer import load_scaler

NUM_SIMS = 50_000


# ══════════════════════════════════════════════════════════════════
# Real 2025-26 conference tournament bracket definitions.
#
# Each bracket is a nested tuple where:
#   - An integer = a seed number (team enters here)
#   - A tuple (left, right) = a game (winner of left vs winner of right)
#
# The simulate_tournament() function recurses through the tree.
# ══════════════════════════════════════════════════════════════════

# --- ACC (15 of 18 teams) ---
# Bottom 3 by conf record do NOT qualify.
# Top 4 double bye → QF. Seeds 5-8 single bye → 2nd round.
# Seeds 9-15 play first round (seed 9 gets first-round bye).
# First round: 10v15, 11v14, 12v13.  2nd round: 5v(12/13), 6v(11/14),
# 7v(10/15), 8v9.  QF: 1v(8/9 side), 4v(5 side), 2v(7 side), 3v(6 side).
ACC_BRACKET = (
    ((1, (8, 9)), (4, (5, (12, 13)))),
    ((2, (7, (10, 15))), (3, (6, (11, 14)))),
)

# --- Big Ten (all 18 teams) ---
# Play-in: 15v18, 16v17.  Then standard 16-team bracket.
# Top 4 double bye → QF.  Seeds 5-8 single bye → 2nd round.
# Seeds 9-14 + 2 play-in winners fill first round.
BIG_TEN_BRACKET = (
    ((1, (8, (9, (16, 17)))), (4, (5, (12, 13)))),
    ((2, (7, (10, (15, 18)))), (3, (6, (11, 14)))),
)

# --- Big 12 (all 16 teams) ---
# Top 4 double bye → QF.  Seeds 5-8 single bye → 2nd round.
# Seeds 9-16 play first round.
BIG_12_BRACKET = (
    ((1, (8, (9, 16))), (4, (5, (12, 13)))),
    ((2, (7, (10, 15))), (3, (6, (11, 14)))),
)

# --- SEC (all 16 teams) ---
# Same structure as Big 12.
SEC_BRACKET = BIG_12_BRACKET

# --- Big East (all 11 teams) ---
# Top 5 bye → QF.  Seeds 6-11 play first round (6v11, 7v10, 8v9).
# QF: 1v(8/9), 2v(7/10), 3v(6/11), 4v5.
BIG_EAST_BRACKET = (
    ((1, (8, 9)), (4, 5)),
    ((2, (7, 10)), (3, (6, 11))),
)

# --- WCC (all 12 teams) — STEPLADDER ---
# Seeds 1-2 → semis (2 wins needed).  Seeds 3-4 → QF (3 wins).
# Seeds 5-6 enter round before QF (4 wins).  Seeds 7-8 earlier (5 wins).
# Seeds 9-12 play first round (6 wins needed to win tourney).
WCC_BRACKET = (
    (1, (4, (5, (8, (9, 12))))),
    (2, (3, (6, (7, (10, 11))))),
)

# --- Mountain West (all 12 teams) ---
# Top 4 bye → QF.  Seeds 5-12 play first round.
MW_BRACKET = (
    ((1, (8, 9)), (4, (5, 12))),
    ((2, (7, 10)), (3, (6, 11))),
)

# --- A-10 (all 14 teams) ---
# Top 4 double bye → QF.  Seeds 5-10 single bye → 2nd round.
# Seeds 11-14 play first round (11v14, 12v13).
A10_BRACKET = (
    ((1, (8, 9)), (4, (5, (12, 13)))),
    ((2, (7, 10)), (3, (6, (11, 14)))),
)

# --- American (top 10 of 13) — STEPLADDER ---
# Seeds 1-2 → semis.  Seeds 3-4 → QF.
# Seeds 5-10 fill earlier rounds.
AMERICAN_BRACKET = (
    (1, (4, (5, (8, 9)))),
    (2, (3, (6, (7, 10)))),
)

# --- CAA (all 13 teams) ---
# Top 4 double bye → QF.  Seeds 5-11 single bye → 2nd round.
# Seeds 12-13 play-in, winner faces seed 5 in 2nd round.
CAA_BRACKET = (
    ((1, (8, 9)), (4, (5, (12, 13)))),
    ((2, (7, 10)), (3, (6, 11))),
)

# --- MAAC (top 10 of 13) ---
# Top 6 bye → QF.  Seeds 7-10 play first round (7v10, 8v9).
MAAC_BRACKET = (
    ((1, (8, 9)), (4, 5)),
    ((2, (7, 10)), (3, 6)),
)

# --- MVC (all 11 teams) ---
# Top 5 bye → QF.  Seeds 6-11 play first round (6v11, 7v10, 8v9).
MVC_BRACKET = (
    ((1, (8, 9)), (4, 5)),
    ((2, (7, 10)), (3, (6, 11))),
)

# --- Sun Belt (all 14 teams) — STEPLADDER ---
# 7-round ladder.  Seeds 1-2 → SF (2 wins).  Seeds 3-4 → QF (3 wins).
# Seeds 5-6 → 4th round (4 wins).  Seeds 7-8 → 3rd round (5 wins).
# Seeds 9-10 → 2nd round (6 wins).  Seeds 11-14 → 1st round (7 wins).
SUN_BELT_BRACKET = (
    (1, (4, (5, (8, (9, (12, 13)))))),
    (2, (3, (6, (7, (10, (11, 14)))))),
)

# --- MAC (all 13 teams) ---
# Top 5 bye → QF.  Seeds 6-13 play first round (6v13, 7v12, 8v11, 9v10).
# QF: 1v(8/11 winner), 2v(7/12 winner), 3v(6/13 winner), 4v(9/10 winner),
#      5 faces the remaining slot.
# Simplified: use standard bracket approach for 13 with top 5 byes.
MAC_BRACKET = (
    ((1, (8, 9)), (4, (5, (12, 13)))),
    ((2, (7, 10)), (3, (6, 11))),
)

# --- Southland (top 8 of 12) ---
# Top 2 bye → SF.  Seeds 3-8 play QF (3v6, 4v5, 7v... wait, 6 teams for QF).
# Standard 8-team bracket: 3v6, 4v5, then SF with 1-2.
SOUTHLAND_BRACKET = (
    (1, (4, (5, 8))),
    (2, (3, (6, 7))),
)

# --- CUSA (top 10 of 12) ---
# Top 6 bye → QF.  Seeds 7-10 play first round (7v10, 8v9).
CUSA_BRACKET = (
    ((1, (8, 9)), (4, 5)),
    ((2, (7, 10)), (3, 6)),
)

# --- ASUN (all 12 teams) ---
# Top 4 bye → QF.  Seeds 5-12 play first round (5v12, 6v11, 7v10, 8v9).
ASUN_BRACKET = (
    ((1, (8, 9)), (4, (5, 12))),
    ((2, (7, 10)), (3, (6, 11))),
)

# --- SWAC (all 12 teams) ---
# Top 4 bye → QF.  Seeds 5-12 play first round.
SWAC_BRACKET = ASUN_BRACKET

# --- OVC (top 8 of 11) — STEPLADDER ---
# Top 2 bye → SF.  Seeds 3-4 enter at QF.  Seeds 5-8 play first round.
OVC_BRACKET = (
    (1, (4, (5, 8))),
    (2, (3, (6, 7))),
)

# --- Horizon League (all 11 teams) ---
# 10v11 play-in, then 10-team bracket.  Top 2 get 2nd-round bye.
# Seeds 3-6 in first round, 7-10 (after play-in) in first round.
# Simplified as standard with byes.
HORIZON_BRACKET = (
    ((1, (8, 9)), (4, 5)),
    ((2, (7, (10, 11))), (3, 6)),
)

# --- Big Sky (all 10 teams) ---
# Seeds 7-10 play R1 (7v10, 8v9).  Seeds 1-6 + 2 winners fill QF.
# QF: 1v(8/9), 2v(7/10), 3v6, 4v5.  Then SF, Final.
BIG_SKY_BRACKET = (
    ((1, (8, 9)), (4, 5)),
    ((2, (7, 10)), (3, 6)),
)

# --- SoCon (all 10 teams) ---
# Same structure as Big Sky.
SOCON_BRACKET = BIG_SKY_BRACKET

# --- Patriot League (all 10 teams) ---
# Same structure as Big Sky.
PATRIOT_BRACKET = BIG_SKY_BRACKET

# --- NEC (all 10 teams) ---
# Same structure.
NEC_BRACKET = BIG_SKY_BRACKET

# --- Big South (all 9 teams) ---
# 8v9 play-in, then standard 8-team bracket.
# QF: 1v(8/9), 2v7, 3v6, 4v5.  SF, Final.
BIG_SOUTH_BRACKET = (
    ((1, (8, 9)), (4, 5)),
    ((2, 7), (3, 6)),
)

# --- Summit League (all 9 teams) ---
SUMMIT_BRACKET = BIG_SOUTH_BRACKET

# --- America East (all 9 teams) ---
AM_EAST_BRACKET = BIG_SOUTH_BRACKET

# --- Ivy League (top 4 of 8) ---
# Only top 4 by conf record qualify.  Standard semis + final.
IVY_BRACKET = ((1, 4), (2, 3))

# --- MEAC (all 8 teams) ---
# Standard 8-team single elimination, no byes.
MEAC_BRACKET = (
    ((1, 8), (4, 5)),
    ((2, 7), (3, 6)),
)

# --- WAC (all 7 teams) ---
# Seed 1 bye.  2v7, 3v6, 4v5 in QF.  1 enters at SF.
WAC_BRACKET = (
    (1, (4, 5)),
    ((2, 7), (3, 6)),
)

# --- Big West (top 8 of 11) ---
# Top 2 double bye → SF.  Seeds 3-4 single bye → QF.
# Seeds 5-8 play first round (5v8, 6v7).
BIG_WEST_BRACKET = (
    (1, (4, (5, 8))),
    (2, (3, (6, 7))),
)


# ══════════════════════════════════════════════════════════════════
# Master config: conference name → (n_qualifying_teams, bracket)
# ══════════════════════════════════════════════════════════════════

CONFERENCE_FORMATS: dict[str, tuple[int, tuple]] = {
    "ACC":            (15, ACC_BRACKET),
    "Big Ten":        (18, BIG_TEN_BRACKET),
    "Big 12":         (16, BIG_12_BRACKET),
    "SEC":            (16, SEC_BRACKET),
    "Big East":       (11, BIG_EAST_BRACKET),
    "WCC":            (12, WCC_BRACKET),
    "Mountain West":  (12, MW_BRACKET),
    "A-10":           (14, A10_BRACKET),
    "American":       (10, AMERICAN_BRACKET),
    "CAA":            (13, CAA_BRACKET),
    "MAAC":           (10, MAAC_BRACKET),
    "MVC":            (11, MVC_BRACKET),
    "Sun Belt":       (14, SUN_BELT_BRACKET),
    "MAC":            (13, MAC_BRACKET),
    "Southland":      (8,  SOUTHLAND_BRACKET),
    "CUSA":           (10, CUSA_BRACKET),
    "ASUN":           (12, ASUN_BRACKET),
    "SWAC":           (12, SWAC_BRACKET),
    "OVC":            (8,  OVC_BRACKET),
    "Horizon":        (11, HORIZON_BRACKET),
    "Big Sky":        (10, BIG_SKY_BRACKET),
    "SoCon":          (10, SOCON_BRACKET),
    "Patriot":        (10, PATRIOT_BRACKET),
    "NEC":            (10, NEC_BRACKET),
    "Big South":      (9,  BIG_SOUTH_BRACKET),
    "Summit":         (9,  SUMMIT_BRACKET),
    "Am. East":       (9,  AM_EAST_BRACKET),
    "Ivy":            (4,  IVY_BRACKET),
    "MEAC":           (8,  MEAC_BRACKET),
    "WAC":            (7,  WAC_BRACKET),
    "Big West":       (8,  BIG_WEST_BRACKET),
}


# ── Feature column mappings ──────────────────────────────────────
# Map home_* and away_* feature columns to neutral stat names so we can
# extract a team's profile regardless of whether they last played home or away.

# home feature col → neutral stat name
HOME_TO_STAT = {
    "home_team_adj_oe": "adj_oe", "home_team_adj_de": "adj_de",
    "home_team_adj_pace": "adj_pace", "home_team_BARTHAG": "BARTHAG",
    "home_eff_fg_pct": "eff_fg_pct", "home_ft_pct": "ft_pct",
    "home_ft_rate": "ft_rate", "home_3pt_rate": "3pt_rate",
    "home_3p_pct": "3p_pct", "home_off_rebound_pct": "off_rebound_pct",
    "home_def_rebound_pct": "def_rebound_pct",
    "home_def_eff_fg_pct": "def_eff_fg_pct",
    "home_opp_ft_rate": "def_ft_rate",
    "home_def_3pt_rate": "def_3pt_rate", "home_def_3p_pct": "def_3p_pct",
    "home_def_off_rebound_pct": "def_off_rebound_pct",
    "home_sos_oe": "sos_oe", "home_sos_de": "sos_de",
    "home_conf_strength": "conf_strength", "home_form_delta": "form_delta",
    "home_tov_rate": "tov_rate", "home_def_tov_rate": "def_tov_rate",
    "home_margin_std": "margin_std",
}

# away feature col → neutral stat name
AWAY_TO_STAT = {
    "away_team_adj_oe": "adj_oe", "away_team_adj_de": "adj_de",
    "away_team_adj_pace": "adj_pace", "away_team_BARTHAG": "BARTHAG",
    "away_eff_fg_pct": "eff_fg_pct", "away_ft_pct": "ft_pct",
    "away_ft_rate": "ft_rate", "away_3pt_rate": "3pt_rate",
    "away_3p_pct": "3p_pct", "away_off_rebound_pct": "off_rebound_pct",
    "away_def_rebound_pct": "def_rebound_pct",
    "away_def_eff_fg_pct": "def_eff_fg_pct",
    "away_def_ft_rate": "def_ft_rate",
    "away_def_3pt_rate": "def_3pt_rate", "away_def_3p_pct": "def_3p_pct",
    "away_def_off_rebound_pct": "def_off_rebound_pct",
    "away_sos_oe": "sos_oe", "away_sos_de": "sos_de",
    "away_conf_strength": "conf_strength", "away_form_delta": "form_delta",
    "away_tov_rate": "tov_rate", "away_def_tov_rate": "def_tov_rate",
    "away_margin_std": "margin_std",
}

# neutral stat name → home feature col (for building matchup vectors)
STAT_TO_HOME = {v: k for k, v in HOME_TO_STAT.items()}

# neutral stat name → away feature col
STAT_TO_AWAY = {v: k for k, v in AWAY_TO_STAT.items()}


# ── Team profile extraction ──────────────────────────────────────

def extract_team_profiles(df: pd.DataFrame) -> dict[int, dict[str, float]]:
    """Extract each team's latest rolling stats from the feature parquet."""
    df = df.copy()
    df["_dt"] = pd.to_datetime(df["startDate"], errors="coerce", utc=True)
    df = df.sort_values("_dt")

    profiles: dict[int, dict[str, float]] = {}

    for _, row in df.iterrows():
        home_id = int(row["homeTeamId"])
        away_id = int(row["awayTeamId"])

        h: dict[str, float] = {}
        for feat, stat in HOME_TO_STAT.items():
            v = row.get(feat)
            if pd.notna(v):
                h[stat] = float(v)
        if h:
            profiles[home_id] = {**profiles.get(home_id, {}), **h}

        a: dict[str, float] = {}
        for feat, stat in AWAY_TO_STAT.items():
            v = row.get(feat)
            if pd.notna(v):
                a[stat] = float(v)
        if a:
            profiles[away_id] = {**profiles.get(away_id, {}), **a}

    return profiles


# ── Matchup prediction ───────────────────────────────────────────

def build_matchup_features(a_stats: dict, b_stats: dict) -> dict[str, float | None]:
    """Build 53-feature dict for team_a (home slot) vs team_b (away slot), neutral site."""
    feat: dict[str, float | None] = {"neutral_site": 1}

    for stat, col in STAT_TO_HOME.items():
        feat[col] = a_stats.get(stat)

    for stat, col in STAT_TO_AWAY.items():
        feat[col] = b_stats.get(stat)

    # Neutral site → zero out venue-specific features
    feat["home_team_hca"] = 0.0
    feat["home_team_efg_home_split"] = 0.0
    feat["away_team_efg_away_split"] = 0.0

    # Tournament rest: 1 day, equal for both
    feat["home_rest_days"] = 1.0
    feat["away_rest_days"] = 1.0
    feat["rest_advantage"] = 0.0

    return feat


def normal_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2))


class TournamentPredictor:
    """Predicts win probabilities using regressor mu/sigma.

    For neutral-site games, averages predictions from both home/away slot
    assignments to cancel out the model's home-slot bias.
    """

    def __init__(self, scaler, regressor, sigma_param: str,
                 feature_order: list[str], profiles: dict[int, dict]):
        self.scaler = scaler
        self.reg = regressor
        self.reg.eval()
        self.sigma_param = sigma_param
        self.feature_order = feature_order
        self.profiles = profiles
        self._cache: dict[tuple[int, int], float] = {}

    def _predict_one(self, home_stats: dict, away_stats: dict) -> float:
        """Run one prediction with given home/away assignment. Returns P(home wins)."""
        feat = build_matchup_features(home_stats, away_stats)
        X = np.array(
            [[feat.get(f) for f in self.feature_order]],
            dtype=np.float32,
        )

        nan_mask = np.isnan(X)
        if nan_mask.any():
            for j in range(X.shape[1]):
                if nan_mask[0, j]:
                    X[0, j] = self.scaler.mean_[j]

        X_scaled = self.scaler.transform(X)
        with torch.no_grad():
            t = torch.tensor(X_scaled, dtype=torch.float32)
            mu_raw, log_sigma_raw = self.reg(t)
            mu = mu_raw.item()
            if self.sigma_param == "exp":
                sigma = torch.exp(log_sigma_raw).clamp(min=0.5, max=30.0).item()
            else:
                sigma = (torch.nn.functional.softplus(log_sigma_raw) + 1e-3).clamp(min=0.5, max=30.0).item()

        return normal_cdf(mu / sigma)

    def win_prob(self, a_id: int, b_id: int) -> float:
        key = (a_id, b_id)
        if key in self._cache:
            return self._cache[key]

        a_stats = self.profiles.get(a_id, {})
        b_stats = self.profiles.get(b_id, {})

        # Average both slot assignments to cancel home-slot bias
        p_a_home = self._predict_one(a_stats, b_stats)    # A in home slot
        p_a_away = 1.0 - self._predict_one(b_stats, a_stats)  # A in away slot
        p = (p_a_home + p_a_away) / 2.0

        self._cache[key] = p
        self._cache[(b_id, a_id)] = 1.0 - p
        return p


# ── Simulation ────────────────────────────────────────────────────

def simulate_tournament(bracket, seeds, pred, rng):
    if isinstance(bracket, int):
        return seeds[bracket]
    left = simulate_tournament(bracket[0], seeds, pred, rng)
    right = simulate_tournament(bracket[1], seeds, pred, rng)
    p = pred.win_prob(left, right)
    return left if rng.random() < p else right


def prob_to_american(p: float) -> str:
    if p <= 0.001:
        return "+99999"
    if p >= 0.999:
        return "-99999"
    if p >= 0.5:
        return f"{-100 * p / (1 - p):+.0f}"
    return f"+{100 * (1 - p) / p:.0f}"


def parse_record(rec: str) -> tuple[int, int]:
    parts = rec.split("-")
    return int(parts[0]), int(parts[1])


# ── Fallback bracket builder for unknown conferences ─────────────

def _seed_order(n: int) -> list[int]:
    if n == 1:
        return [1]
    prev = _seed_order(n // 2)
    out: list[int] = []
    for s in prev:
        out.append(s)
        out.append(n + 1 - s)
    return out


def _build_tree(seeds: list[int]):
    if len(seeds) == 1:
        return seeds[0]
    if len(seeds) == 2:
        return (seeds[0], seeds[1])
    mid = len(seeds) // 2
    return (_build_tree(seeds[:mid]), _build_tree(seeds[mid:]))


def build_bracket(n_teams: int):
    """Fallback generic bracket for conferences without explicit format."""
    p = 1
    while p < n_teams:
        p *= 2
    full = _build_tree(_seed_order(p))

    def collapse(node):
        if isinstance(node, int):
            return node if node <= n_teams else None
        left = collapse(node[0])
        right = collapse(node[1])
        if left is None:
            return right
        if right is None:
            return left
        return (left, right)

    return collapse(full)


# ── Main ──────────────────────────────────────────────────────────

def main():
    print("Loading model and data...")
    scaler = load_scaler()
    regressor, _, feature_order, sigma_param = load_regressor()

    # Load rankings for team/conference mapping
    with open(config.SITE_DATA_DIR / "rankings_2026.json") as f:
        rankings = json.load(f)

    team_by_id: dict[int, dict] = {}
    for t in rankings["teams"]:
        team_by_id[t["team_id"]] = t

    # Group by conference, sort by conf record → seeds
    conf_teams: dict[str, list[dict]] = {}
    for t in rankings["teams"]:
        conf = t["conference"]
        if conf not in conf_teams:
            conf_teams[conf] = []
        conf_teams[conf].append(t)

    for conf in conf_teams:
        conf_teams[conf].sort(
            key=lambda t: (
                parse_record(t.get("conf_record", "0-0"))[0],
                -parse_record(t.get("conf_record", "0-0"))[1],
                t["adj_margin"],
            ),
            reverse=True,
        )

    # Load feature parquet and extract team profiles
    feat_path = (
        config.FEATURES_DIR
        / "season_2026_no_garbage_torvik_adj_a0.85_p10_features.parquet"
    )
    print(f"Loading features from {feat_path.name}...")
    df = pd.read_parquet(feat_path)
    profiles = extract_team_profiles(df)
    print(f"  {len(profiles)} team profiles extracted")

    pred = TournamentPredictor(scaler, regressor, sigma_param, feature_order, profiles)
    rng = random.Random(42)

    # Sort conferences by tournament size (largest first)
    sorted_confs = sorted(conf_teams.items(), key=lambda x: -len(x[1]))

    for conf_name, teams_list in sorted_confs:
        # Look up real tournament format; fall back to generic bracket
        fmt = CONFERENCE_FORMATS.get(conf_name)
        if fmt is not None:
            n_qualify, bracket = fmt
            # Limit to qualifying teams (top N by conf record)
            n_qualify = min(n_qualify, len(teams_list))
            qualified = teams_list[:n_qualify]
        else:
            # Unknown conference — use all teams with generic bracket
            qualified = teams_list
            n_qualify = len(qualified)
            bracket = build_bracket(n_qualify)

        if n_qualify < 2:
            continue

        seed_map = {i + 1: t["team_id"] for i, t in enumerate(qualified)}
        excluded = teams_list[n_qualify:]

        wins: dict[int, int] = {t["team_id"]: 0 for t in qualified}
        for _ in range(NUM_SIMS):
            winner = simulate_tournament(bracket, seed_map, pred, rng)
            wins[winner] += 1

        print(f"\n{'=' * 65}")
        print(f"  {conf_name} Conference Tournament  ({n_qualify} teams, {NUM_SIMS:,} sims)")
        print(f"{'=' * 65}")
        print(f"{'Seed':>4}  {'Team':<25} {'Conf':<8} {'Win%':>7}  {'Odds':>8}")
        print(f"{'─'*4:>4}  {'─'*25:<25} {'─'*8:<8} {'─'*7:>7}  {'─'*8:>8}")

        for i, t in enumerate(qualified):
            tid = t["team_id"]
            pct = wins[tid] / NUM_SIMS * 100
            odds = prob_to_american(wins[tid] / NUM_SIMS)
            name = t["team"]
            seed = i + 1
            cr = t.get("conf_record", "")
            print(f"  {seed:>2}   {name:<25} {cr:<8} {pct:>6.1f}%  {odds:>8}")

        if excluded:
            print(f"  {'─'*60}")
            for t in excluded:
                cr = t.get("conf_record", "")
                print(f"  DNQ  {t['team']:<25} {cr:<8}   ---      ---")

    print(f"\n{'=' * 65}")
    print("  Done!")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
