#!/usr/bin/env python3
"""Generate visual ASCII bracket predictions for all conference tournaments.

Plays out each bracket deterministically — always picking the model's
predicted winner — and renders visual brackets with win probabilities.

Usage:
    poetry run python scripts/generate_bracket_predictions.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from io import StringIO

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.infer import load_regressor
from src.trainer import load_scaler

from scripts.simulate_conference_tournaments import (
    CONFERENCE_FORMATS,
    build_bracket,
    extract_team_profiles,
    TournamentPredictor,
    parse_record,
)

MAX_NAME = 14  # truncate team names for compact display


def get_bracket_winner(node, seed_map, pred):
    """Recursively determine the winner from a bracket node."""
    if isinstance(node, int):
        return seed_map[node], node
    l_id, l_s = get_bracket_winner(node[0], seed_map, pred)
    r_id, r_s = get_bracket_winner(node[1], seed_map, pred)
    p = pred.win_prob(l_id, r_id)
    return (l_id, l_s) if p >= 0.5 else (r_id, r_s)


def trunc(name: str, n: int = MAX_NAME) -> str:
    return name if len(name) <= n else name[:n - 1] + "."


def render_bracket(node, seed_map, pred, names):
    """Render bracket as ASCII art. Returns (list_of_lines, midpoint_index)."""
    if isinstance(node, int):
        tid = seed_map[node]
        label = f"({node:>2}) {trunc(names.get(tid, '???'))}"
        return [label], 0

    l_lines, l_mid = render_bracket(node[0], seed_map, pred, names)
    r_lines, r_mid = render_bracket(node[1], seed_map, pred, names)

    # Determine game winner
    l_wid, l_ws = get_bracket_winner(node[0], seed_map, pred)
    r_wid, r_ws = get_bracket_winner(node[1], seed_map, pred)
    p = pred.win_prob(l_wid, r_wid)
    if p >= 0.5:
        w_name, w_prob = trunc(names.get(l_wid, "?")), p
    else:
        w_name, w_prob = trunc(names.get(r_wid, "?")), 1 - p

    winner_tag = f"{w_name} {w_prob * 100:.0f}%"

    max_w = max(len(l) for l in l_lines + r_lines)
    cc = max_w + 2  # connector column (where ┐│├┘ go)

    out = []

    # ── Left child ──
    for i, line in enumerate(l_lines):
        gap = cc - len(line)
        if i < l_mid:
            out.append(line + " " * (gap + 1))
        elif i == l_mid:
            out.append(line + "─" * max(gap, 1) + "┐")
        else:
            out.append(line + " " * gap + "│")

    # ── Winner line ──
    out.append(" " * cc + "├── " + winner_tag)

    # ── Right child ──
    for i, line in enumerate(r_lines):
        gap = cc - len(line)
        if i < r_mid:
            out.append(line + " " * gap + "│")
        elif i == r_mid:
            out.append(line + "─" * max(gap, 1) + "┘")
        else:
            out.append(line + " " * (gap + 1))

    return out, len(l_lines)


def main():
    print("Loading model and data...")
    scaler = load_scaler()
    regressor, _, feature_order, sigma_param = load_regressor()

    with open(config.SITE_DATA_DIR / "rankings_2026.json") as f:
        rankings = json.load(f)

    names = {t["team_id"]: t["team"] for t in rankings["teams"]}

    # Group by conference, sort by conf record
    conf_teams: dict[str, list[dict]] = {}
    for t in rankings["teams"]:
        conf_teams.setdefault(t["conference"], []).append(t)

    for conf in conf_teams:
        conf_teams[conf].sort(
            key=lambda t: (
                parse_record(t.get("conf_record", "0-0"))[0],
                -parse_record(t.get("conf_record", "0-0"))[1],
                t["adj_margin"],
            ),
            reverse=True,
        )

    # Load features and build predictor
    feat_path = (
        config.FEATURES_DIR
        / "season_2026_no_garbage_torvik_adj_a0.85_p10_features.parquet"
    )
    print(f"Loading features from {feat_path.name}...")
    df = pd.read_parquet(feat_path)
    profiles = extract_team_profiles(df)
    print(f"  {len(profiles)} team profiles extracted")

    pred = TournamentPredictor(scaler, regressor, sigma_param, feature_order, profiles)

    sorted_confs = sorted(conf_teams.items(), key=lambda x: -len(x[1]))

    out = StringIO()
    out.write("# Conference Tournament Bracket Predictions — 2026\n\n")
    out.write("Model picks the predicted winner of every game. Win % = model confidence.\n\n")

    json_confs = []

    for conf_name, teams_list in sorted_confs:
        fmt = CONFERENCE_FORMATS.get(conf_name)
        if fmt is not None:
            n_qualify, bracket = fmt
            n_qualify = min(n_qualify, len(teams_list))
            qualified = teams_list[:n_qualify]
        else:
            qualified = teams_list
            n_qualify = len(qualified)
            bracket = build_bracket(n_qualify)

        if n_qualify < 2:
            continue

        seed_map = {i + 1: t["team_id"] for i, t in enumerate(qualified)}
        champ_id, champ_seed = get_bracket_winner(bracket, seed_map, pred)
        champ_name = names.get(champ_id, "???")

        lines, _ = render_bracket(bracket, seed_map, pred, names)

        # Mark champion on the outermost winner line
        for i, line in enumerate(lines):
            if "├── " in line and line.strip().startswith("├──"):
                # Check if this is the outermost ├── (championship game)
                # It's the one at the shallowest indentation with no ┐┘│ after it
                if "┐" not in line[line.index("├──"):] and "┘" not in line[line.index("├──"):]:
                    lines[i] = line + "  ★ CHAMPION"
                    break

        excluded = teams_list[n_qualify:]

        out.write(f"---\n\n")
        out.write(f"### {conf_name} ({n_qualify} teams) — Champion: ({champ_seed}) {champ_name}\n\n")
        out.write("```\n")
        for line in lines:
            out.write(line.rstrip() + "\n")
        out.write("```\n\n")

        dnq_list = [t["team"] for t in excluded]
        if dnq_list:
            dnq_str = ", ".join(dnq_list)
            out.write(f"*DNQ: {dnq_str}*\n\n")

        json_confs.append({
            "name": conf_name,
            "team_count": n_qualify,
            "champion": champ_name,
            "champion_seed": champ_seed,
            "bracket_lines": [line.rstrip() for line in lines],
            "dnq": dnq_list,
        })

        print(f"  {conf_name}: ({champ_seed}) {champ_name}")

    out_path = PROJECT_ROOT / "reports" / "conference_tournament_brackets.md"
    out_path.write_text(out.getvalue())
    print(f"\nBrackets written to {out_path}")

    # Also write JSON for the website
    from datetime import datetime

    json_data = {
        "generated_at": datetime.now().isoformat(),
        "season": 2026,
        "conferences": json_confs,
    }
    json_path = config.SITE_DATA_DIR / "brackets_2026.json"
    json_path.write_text(json.dumps(json_data, indent=2))
    print(f"JSON written to {json_path}")


if __name__ == "__main__":
    main()
