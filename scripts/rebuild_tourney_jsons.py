#!/usr/bin/env python3
"""Rebuild conference tournament JSONs from the current production model.

Treats the existing bracket JSON as the authoritative bracket structure for each
conference, then recomputes exact advancement/champion probabilities using the
current promoted production model on neutral synthetic matchups.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config
from src.infer import _fill_nan_with_scaler_means
from src.ml_odds import fair_american_odds, site_home_win_prob_from_mu_sigma
from src.trainer import load_scaler, load_tree_regressor
from src.infer import load_regressor

from build_rankings_json import _load_latest_ratings


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "site" / "public" / "data"
BRACKETS_TEMPLATE_PATH = DATA_DIR / "brackets_2026_template.json"
BRACKETS_PATH = DATA_DIR / "brackets_2026.json"
TOURNEYS_PATH = DATA_DIR / "tourneys_2026.json"
RANKINGS_PATH = DATA_DIR / "rankings_2026.json"

CURRENT_SEASON = 2026

LEAF_RE = re.compile(r"^\((\s*\d+)\)\s+(.*?)(?:[─┐┘]|$)")
NODE_RE = re.compile(r"(\s*)([A-Za-z0-9&'()./\- ]+?)\s+(\d+)%")
CHAMPION_RE = re.compile(r"^(.*?[├└]──\s*)(.*?)(\s+★ CHAMPION.*)$")


@dataclass
class BracketNode:
    row: int
    ci: int
    co: int
    label: str | None = None
    seed: int | None = None
    team: str | None = None
    left: "BracketNode | None" = None
    right: "BracketNode | None" = None
    probs: dict[str, float] | None = None
    game_probs: dict[str, float] | None = None

    @property
    def is_leaf(self) -> bool:
        return self.team is not None

    @property
    def display_label(self) -> str:
        return self.team if self.team is not None else (self.label or "")


def _read_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def american_to_prob(odds: str | int | float | None) -> float | None:
    if odds is None:
        return None
    if isinstance(odds, str):
        value = odds.strip().replace(",", "")
        if not value:
            return None
        odds_num = float(value)
    else:
        odds_num = float(odds)
    if odds_num < 0:
        return (-odds_num) / ((-odds_num) + 100.0)
    if odds_num > 0:
        return 100.0 / (odds_num + 100.0)
    return None


def _parse_bracket_lines(lines: list[str]) -> tuple[BracketNode, list[BracketNode]]:
    leaves: list[BracketNode] = []
    internals: list[BracketNode] = []

    for row, line in enumerate(lines):
        leaf_m = LEAF_RE.match(line)
        if leaf_m:
            seed = int(leaf_m.group(1))
            team = leaf_m.group(2).rstrip("─").strip()
            co = max(line.rfind("┐"), line.rfind("┘"))
            leaves.append(BracketNode(row=row, ci=-1, co=co, seed=seed, team=team))

        node_m = None if "CHAMPION" in line else NODE_RE.search(line)
        if node_m:
            left_connector_idx = min(
                (idx for idx in (line.find("├"), line.find("└"), line.find("┌")) if idx != -1),
                default=-1,
            )
            ci = left_connector_idx if left_connector_idx != -1 else node_m.start()
            co = max(line.rfind("┐"), line.rfind("┘"))
            internals.append(BracketNode(row=row, ci=ci, co=co, label=node_m.group(1).strip()))

    internals.sort(key=lambda n: (n.ci, n.row))

    all_nodes: list[BracketNode] = leaves[:]
    for node in internals:
        children = [n for n in all_nodes if n.co == node.ci]
        above = [n for n in children if n.row < node.row]
        below = [n for n in children if n.row > node.row]
        if not above or not below:
            raise ValueError(f"Could not attach children for node at row {node.row}")
        node.left = max(above, key=lambda n: n.row)
        node.right = min(below, key=lambda n: n.row)
        all_nodes.append(node)

    root_candidates = [n for n in internals if n.co == -1]
    if not root_candidates:
        # top-most internal node(s) with no parent; some bracket templates omit the
        # final root and connect the two semifinal winners directly to the champion line.
        child_ids = {id(n.left) for n in internals if n.left} | {id(n.right) for n in internals if n.right}
        root_candidates = [n for n in internals if id(n) not in child_ids]
    if len(root_candidates) == 1:
        return root_candidates[0], internals
    if len(root_candidates) == 2:
        top, bottom = sorted(root_candidates, key=lambda n: n.row)
        virtual_root = BracketNode(
            row=(top.row + bottom.row) // 2,
            ci=-1,
            co=-1,
            label="FINAL",
            left=top,
            right=bottom,
        )
        return virtual_root, internals
    raise ValueError(f"Expected one root, found {len(root_candidates)}")


def _iter_leaves(node: BracketNode) -> list[BracketNode]:
    if node.is_leaf:
        return [node]
    out: list[BracketNode] = []
    if node.left is not None:
        out.extend(_iter_leaves(node.left))
    if node.right is not None:
        out.extend(_iter_leaves(node.right))
    return out


def _apply_conference_seed_mapping(root: BracketNode, conf: dict[str, Any]) -> None:
    seed_to_team = {int(team["seed"]): str(team["team"]) for team in conf["teams"]}
    for leaf in _iter_leaves(root):
        if leaf.seed is None:
            raise ValueError(f"Bracket leaf missing seed for conference {conf['name']}")
        mapped = seed_to_team.get(int(leaf.seed))
        if mapped is None:
            raise ValueError(
                f"No team mapping found for conference {conf['name']} seed {leaf.seed}"
            )
        leaf.team = mapped


def _load_team_inputs() -> pd.DataFrame:
    rankings = pd.DataFrame(_read_json(RANKINGS_PATH)["teams"])
    ratings, _ = _load_latest_ratings(CURRENT_SEASON)
    merged = rankings.copy()
    ratings_by_team = ratings.set_index("teamId")
    metric_cols = [
        "adj_oe",
        "adj_de",
        "adj_tempo",
        "barthag",
        "sos_oe",
        "sos_de",
    ]
    merged["teamId"] = merged["team_id"]
    for col in metric_cols:
        if col in ratings_by_team.columns:
            merged[col] = merged["team_id"].map(ratings_by_team[col])
    if merged["adj_oe"].isna().any():
        missing = merged.loc[merged["adj_oe"].isna(), "team"].tolist()
        raise ValueError(f"Missing ratings rows for rankings teams: {missing[:10]}")
    return merged


def _build_synthetic_rows(
    team_a: pd.Series,
    team_b: pd.Series,
    feature_order: list[str],
    scaler,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = {name: float(scaler.mean_[i]) for i, name in enumerate(feature_order)}
    avg_rest = float(np.mean([base.get("home_rest_days", 5.0), base.get("away_rest_days", 5.0)]))
    for vec in (base,):
        vec["neutral_site"] = 1.0
        vec["home_team_hca"] = 0.0
        vec["home_rest_days"] = avg_rest
        vec["away_rest_days"] = avg_rest
        vec["rest_advantage"] = 0.0

    def apply_home(vec: dict[str, float], row: pd.Series) -> None:
        vec["home_team_adj_oe"] = float(row["adj_oe"])
        vec["home_team_adj_de"] = float(row["adj_de"])
        vec["home_team_adj_pace"] = float(row["adj_tempo"])
        vec["home_team_BARTHAG"] = float(row["barthag"])
        if "home_sos_oe" in vec and "sos_oe" in row.index and pd.notna(row.get("sos_oe")):
            vec["home_sos_oe"] = float(row["sos_oe"])
        if "home_sos_de" in vec and "sos_de" in row.index and pd.notna(row.get("sos_de")):
            vec["home_sos_de"] = float(row["sos_de"])

    def apply_away(vec: dict[str, float], row: pd.Series) -> None:
        vec["away_team_adj_oe"] = float(row["adj_oe"])
        vec["away_team_adj_de"] = float(row["adj_de"])
        vec["away_team_adj_pace"] = float(row["adj_tempo"])
        vec["away_team_BARTHAG"] = float(row["barthag"])
        if "away_sos_oe" in vec and "sos_oe" in row.index and pd.notna(row.get("sos_oe")):
            vec["away_sos_oe"] = float(row["sos_oe"])
        if "away_sos_de" in vec and "sos_de" in row.index and pd.notna(row.get("sos_de")):
            vec["away_sos_de"] = float(row["sos_de"])

    def neutralize_context(vec: dict[str, float]) -> None:
        for name in (
            "home_sos_oe",
            "away_sos_oe",
            "home_sos_de",
            "away_sos_de",
            "home_conf_strength",
            "away_conf_strength",
        ):
            if name in vec:
                vec[name] = float(scaler.mean_[feature_order.index(name)])

    home_vec = dict(base)
    apply_home(home_vec, team_a)
    apply_away(home_vec, team_b)
    neutralize_context(home_vec)

    away_vec = dict(base)
    apply_home(away_vec, team_b)
    apply_away(away_vec, team_a)
    neutralize_context(away_vec)

    return (
        pd.DataFrame([home_vec], columns=feature_order),
        pd.DataFrame([away_vec], columns=feature_order),
    )


def _predict_pairwise_probability(
    team_a: pd.Series,
    team_b: pd.Series,
    feature_order: list[str],
    scaler,
    tree_model,
    sigma_model,
    sigma_param: str,
    month: int,
    day: int,
) -> tuple[float, float]:
    row_ab, row_ba = _build_synthetic_rows(team_a, team_b, feature_order, scaler)
    X_ab = _fill_nan_with_scaler_means(row_ab, scaler)
    X_ba = _fill_nan_with_scaler_means(row_ba, scaler)
    mu_ab = float(tree_model.predict(X_ab.astype(np.float32))[0])
    mu_ba = float(tree_model.predict(X_ba.astype(np.float32))[0])
    mu = (mu_ab - mu_ba) / 2.0

    X_ab_scaled = scaler.transform(X_ab)
    X_ba_scaled = scaler.transform(X_ba)
    X_ab_tensor = torch.tensor(X_ab_scaled, dtype=torch.float32)
    X_ba_tensor = torch.tensor(X_ba_scaled, dtype=torch.float32)
    with torch.no_grad():
        _, log_sigma_ab = sigma_model(X_ab_tensor)
        _, log_sigma_ba = sigma_model(X_ba_tensor)
        if sigma_param == "exp":
            sigma_ab = np.exp(log_sigma_ab.numpy())[0]
            sigma_ba = np.exp(log_sigma_ba.numpy())[0]
        else:
            sigma_ab = (torch.nn.functional.softplus(log_sigma_ab) + 1e-3).numpy()[0]
            sigma_ba = (torch.nn.functional.softplus(log_sigma_ba) + 1e-3).numpy()[0]
    sigma_var = 0.5 * (sigma_ab ** 2 + sigma_ba ** 2) + ((mu_ab + mu_ba) ** 2) / 4.0
    sigma = float(max(math.sqrt(max(sigma_var, 0.25)), 0.5))
    prob_home = float(
        site_home_win_prob_from_mu_sigma(
            mu,
            sigma,
            start_month=month,
            start_day=day,
            neutral_site=True,
            odds_mode="meta_small_v1",
        )
    )
    return mu, prob_home


def _compute_team_distributions(
    node: BracketNode,
    team_lookup: dict[str, pd.Series],
    pair_cache: dict[tuple[str, str], float],
    feature_order: list[str],
    scaler,
    tree_model,
    sigma_model,
    sigma_param: str,
    month: int,
    day: int,
) -> dict[str, float]:
    if node.probs is not None:
        return node.probs
    if node.is_leaf:
        node.probs = {node.team: 1.0}
        node.game_probs = {node.team: 1.0}
        return node.probs
    assert node.left is not None and node.right is not None
    left_probs = _compute_team_distributions(
        node.left, team_lookup, pair_cache, feature_order, scaler, tree_model, sigma_model, sigma_param, month, day
    )
    right_probs = _compute_team_distributions(
        node.right, team_lookup, pair_cache, feature_order, scaler, tree_model, sigma_model, sigma_param, month, day
    )
    out: dict[str, float] = {}
    game_out: dict[str, float] = {}
    for team_l, p_l in left_probs.items():
        cond_win_l = 0.0
        for team_r, p_r in right_probs.items():
            key = (team_l, team_r)
            if key not in pair_cache:
                _, prob_l = _predict_pairwise_probability(
                    team_lookup[team_l],
                    team_lookup[team_r],
                    feature_order,
                    scaler,
                    tree_model,
                    sigma_model,
                    sigma_param,
                    month,
                    day,
                )
                pair_cache[(team_l, team_r)] = prob_l
                pair_cache[(team_r, team_l)] = 1.0 - prob_l
            cond_win_l += p_r * pair_cache[key]
            matchup_prob = p_l * p_r
            out[team_l] = out.get(team_l, 0.0) + matchup_prob * pair_cache[key]
            out[team_r] = out.get(team_r, 0.0) + matchup_prob * pair_cache[(team_r, team_l)]
        game_out[team_l] = cond_win_l
    for team_r, p_r in right_probs.items():
        cond_win_r = 0.0
        for team_l, p_l in left_probs.items():
            cond_win_r += p_l * pair_cache[(team_r, team_l)]
        game_out[team_r] = cond_win_r
    node.probs = out
    node.game_probs = game_out
    return out


def _format_pct(prob: float) -> int:
    return int(round(prob * 100.0))


def _fit_team_and_pct(label: str, prob: float, width: int) -> str:
    pct = f"{_format_pct(prob)}%"
    text = f"{label} {pct}"
    if width <= 0:
        return text
    if len(text) > width:
        min_label_width = width - len(pct) - 1
        if min_label_width <= 0:
            return pct.rjust(width)
        if len(label) > min_label_width:
            if min_label_width <= 3:
                label = label[:min_label_width]
            else:
                label = label[: min_label_width - 3] + "..."
        text = f"{label} {pct}"
    return text.ljust(width)


def _update_bracket_lines(lines: list[str], internals: list[BracketNode]) -> list[str]:
    updated = list(lines)
    by_row = {node.row: node for node in internals}
    for row_idx, line in enumerate(lines):
        node = by_row.get(row_idx)
        if node is None or not node.probs or not node.game_probs:
            continue
        winner, prob = max(node.game_probs.items(), key=lambda kv: kv[1])
        match = NODE_RE.search(line)
        if not match:
            continue
        start, end = match.span()
        leading = match.group(1)
        span_len = end - start - len(leading)
        replacement = leading + _fit_team_and_pct(winner, node.game_probs[winner], span_len)
        updated[row_idx] = line[:start] + replacement + line[end:]
    return updated


def _update_champion_line(lines: list[str], winner: str, prob: float) -> list[str]:
    updated = list(lines)
    for row_idx, line in enumerate(lines):
        match = CHAMPION_RE.match(line)
        if not match:
            continue
        prefix, middle, suffix = match.groups()
        replacement = _fit_team_and_pct(winner, prob, len(middle))
        updated[row_idx] = prefix + replacement + suffix
        break
    return updated


def _team_table_for_conference(
    conf: dict[str, Any],
    champion_probs: dict[str, float],
) -> list[dict[str, Any]]:
    out = []
    for team in conf["teams"]:
        model_pct = round(champion_probs.get(team["team"], 0.0) * 100.0, 1)
        model_prob = champion_probs.get(team["team"], 0.0)
        model_odds = fair_american_odds(model_prob) if model_prob > 0 else np.nan
        model_odds_str = None if not np.isfinite(model_odds) else (f"+{int(round(model_odds))}" if model_odds > 0 else f"{int(round(model_odds))}")
        vegas_prob = american_to_prob(team.get("hrb_odds"))
        vegas_pct = None if vegas_prob is None else round(vegas_prob * 100.0, 1)
        edge = None if vegas_pct is None else round(model_pct - vegas_pct, 1)
        flag = None
        if edge is not None:
            if edge > 5:
                flag = "STRONG VALUE"
            elif edge > 3:
                flag = "VALUE"
            elif edge < -5:
                flag = "FADE"
        out.append(
            {
                **team,
                "model_pct": model_pct,
                "model_odds": model_odds_str,
                "vegas_implied_pct": vegas_pct,
                "edge": edge,
                "flag": flag,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=CURRENT_SEASON)
    parser.add_argument("--month", type=int, default=3)
    parser.add_argument("--day", type=int, default=13)
    args = parser.parse_args()

    brackets_source = BRACKETS_TEMPLATE_PATH if BRACKETS_TEMPLATE_PATH.exists() else BRACKETS_PATH
    brackets = _read_json(brackets_source)
    tourneys = _read_json(TOURNEYS_PATH)
    team_inputs = _load_team_inputs()
    team_lookup = {row["team"]: row for _, row in team_inputs.iterrows()}

    scaler = load_scaler()
    tree_model, feature_order, _ = load_tree_regressor()
    sigma_model, _, sigma_feature_order, sigma_param = load_regressor()
    if sigma_feature_order != feature_order:
        raise ValueError("Tree and sigma feature orders do not match")

    pair_cache: dict[tuple[str, str], float] = {}
    conf_probs: dict[str, dict[str, float]] = {}

    for conf in brackets["conferences"]:
        root, internals = _parse_bracket_lines(conf["bracket_lines"])
        source_conf = next(c for c in tourneys["conferences"] if c["name"] == conf["name"])
        _apply_conference_seed_mapping(root, source_conf)
        probs = _compute_team_distributions(
            root, team_lookup, pair_cache, feature_order, scaler, tree_model, sigma_model, sigma_param, args.month, args.day
        )
        conf_probs[conf["name"]] = probs
        winner, prob = max(probs.items(), key=lambda kv: kv[1])
        champion_game_prob = prob
        if root.game_probs and winner in root.game_probs:
            champion_game_prob = root.game_probs[winner]
        seed_lookup = {team["team"]: team["seed"] for team in source_conf["teams"]}
        conf["champion"] = winner
        conf["champion_seed"] = seed_lookup.get(winner)
        conf["bracket_lines"] = _update_bracket_lines(conf["bracket_lines"], internals)
        conf["bracket_lines"] = _update_champion_line(conf["bracket_lines"], winner, champion_game_prob)

    # Rebuild tourneys JSON
    new_confs: list[dict[str, Any]] = []
    for conf in tourneys["conferences"]:
        teams = _team_table_for_conference(conf, conf_probs.get(conf["name"], {}))
        new_confs.append({**conf, "teams": teams})
    tourneys["conferences"] = new_confs
    value_bets = []
    fades = []
    for conf in new_confs:
        for team in conf["teams"]:
            edge = team.get("edge")
            flag = team.get("flag")
            if edge is None or flag is None or not conf.get("has_hrb_odds"):
                continue
            row = {
                "conf": conf["name"],
                "team": team["team"],
                "model_pct": team["model_pct"],
                "hrb_odds": team["hrb_odds"],
                "edge": edge,
                "flag": flag,
            }
            if flag in ("VALUE", "STRONG VALUE"):
                value_bets.append(row)
            elif flag == "FADE":
                fades.append(row)
    value_bets.sort(key=lambda r: r["edge"], reverse=True)
    fades.sort(key=lambda r: r["edge"])
    tourneys["value_bets"] = value_bets
    tourneys["fades"] = fades
    tourneys["generated_at"] = datetime.utcnow().isoformat() + "Z"
    tourneys["methodology"] = {
        "simulations": 0,
        "odds_source": tourneys.get("methodology", {}).get("odds_source", "Hard Rock Bet (Florida)"),
        "note": (
            "Exact bracket probabilities from the current production model using the "
            "current conference bracket structure. Neutral-site synthetic matchup scoring "
            "with site ML probability correction."
        ),
    }
    brackets["generated_at"] = datetime.utcnow().isoformat() + "Z"

    with open(BRACKETS_PATH, "w") as f:
        json.dump(brackets, f, indent=2)
        f.write("\n")
    with open(TOURNEYS_PATH, "w") as f:
        json.dump(tourneys, f, indent=2)
        f.write("\n")

    print(f"Rebuilt {len(brackets['conferences'])} conference brackets and tournament odds.")


if __name__ == "__main__":
    main()
