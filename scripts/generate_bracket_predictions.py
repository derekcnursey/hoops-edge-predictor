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
from datetime import datetime
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

# ---------------------------------------------------------------------------
# Official 2025-26 tournament seedings (from conference announcements)
# Team names must exactly match rankings_2026.json
# ---------------------------------------------------------------------------
TOURNAMENT_SEEDINGS_2026: dict[str, list[str]] = {
    "ACC": [
        "Duke", "Virginia", "Miami", "North Carolina", "Clemson",
        "Louisville", "NC State", "Florida State", "California", "Stanford",
        "SMU", "Virginia Tech", "Wake Forest", "Syracuse", "Pittsburgh",
    ],
    "Big Ten": [
        "Michigan", "Nebraska", "Michigan State", "Illinois", "Wisconsin",
        "UCLA", "Purdue", "Ohio State", "Iowa", "Indiana",
        "Minnesota", "Washington", "USC", "Rutgers", "Northwestern",
        "Oregon", "Maryland", "Penn State",
    ],
    "Big 12": [
        "Arizona", "Houston", "Kansas", "Texas Tech", "Iowa State",
        "TCU", "West Virginia", "UCF", "Cincinnati", "BYU",
        "Colorado", "Arizona State", "Baylor", "Oklahoma State",
        "Kansas State", "Utah",
    ],
    "SEC": [
        "Florida", "Alabama", "Arkansas", "Vanderbilt", "Tennessee",
        "Texas A&M", "Georgia", "Missouri", "Kentucky", "Texas",
        "Oklahoma", "Auburn", "Mississippi State", "South Carolina",
        "Ole Miss", "LSU",
    ],
    "Big East": [
        "St. John's", "UConn", "Villanova", "Seton Hall", "Creighton",
        "DePaul", "Marquette", "Butler", "Providence", "Xavier",
        "Georgetown",
    ],
    "A-10": [
        "Saint Louis", "VCU", "Saint Joseph's", "Dayton", "George Mason",
        "Davidson", "Duquesne", "Fordham", "George Washington",
        "Rhode Island", "Richmond", "La Salle", "St. Bonaventure",
        "Loyola Chicago",
    ],
    "Mountain West": [
        "Utah State", "San Diego State", "New Mexico", "Grand Canyon",
        "Nevada", "Boise State", "Colorado State", "UNLV", "Wyoming",
        "Fresno State", "San José State", "Air Force",
    ],
    "WCC": [
        "Gonzaga", "Saint Mary's", "Santa Clara", "Oregon State",
        "San Francisco", "Pacific", "Seattle U", "Washington State",
        "Portland", "Loyola Marymount", "San Diego", "Pepperdine",
    ],
    "American": [
        "South Florida", "Wichita State", "Tulsa", "UAB", "Charlotte",
        "North Texas", "Florida Atlantic", "Memphis", "Tulane", "Temple",
    ],
    "MVC": [
        "Belmont", "Bradley", "Illinois State", "Murray State", "UIC",
        "Northern Iowa", "Valparaiso", "Southern Illinois", "Drake",
        "Indiana State", "Evansville",
    ],
    "Sun Belt": [
        "Troy", "Marshall", "Coastal Carolina", "App State", "Texas State",
        "South Alabama", "Arkansas State", "Southern Miss", "James Madison",
        "Georgia Southern", "Old Dominion", "Louisiana", "Georgia State",
        "UL Monroe",
    ],
    "MAC": [
        "Miami (OH)", "Akron", "Kent State", "Toledo", "Bowling Green",
        "Ohio", "Buffalo", "Massachusetts",
    ],
    "CAA": [
        "UNC Wilmington", "Charleston", "Hofstra", "Monmouth", "Drexel",
        "William & Mary", "Towson", "Stony Brook", "Campbell", "Hampton",
        "Elon", "North Carolina A&T", "Northeastern",
    ],
    "MAAC": [
        "Merrimack", "Saint Peter's", "Siena", "Quinnipiac", "Marist",
        "Mount St. Mary's", "Fairfield", "Iona", "Sacred Heart", "Manhattan",
    ],
    "Southland": [
        "Stephen F. Austin", "McNeese", "UT Rio Grande Valley",
        "Texas A&M-Corpus Christi", "New Orleans", "Nicholls",
        "Northwestern State", "Houston Christian",
    ],
    "CUSA": [
        "Liberty", "Sam Houston", "Western Kentucky", "Louisiana Tech",
        "Middle Tennessee", "Kennesaw State", "Jacksonville State",
        "Florida International", "Missouri State", "New Mexico State",
    ],
    "ASUN": [
        "Central Arkansas", "Austin Peay", "Queens University", "Lipscomb",
        "Florida Gulf Coast", "West Georgia", "Eastern Kentucky",
        "Bellarmine", "Jacksonville", "Stetson", "North Florida",
        "North Alabama",
    ],
    "SWAC": [
        "Bethune-Cookman", "Florida A&M", "Southern", "Texas Southern",
        "Alabama A&M", "Arkansas-Pine Bluff", "Jackson State",
        "Prairie View A&M", "Grambling", "Alabama State", "Alcorn State",
        "Mississippi Valley State",
    ],
    "Big West": [
        "UC Irvine", "Hawai'i", "Cal State Fullerton", "Cal State Northridge",
        "UC San Diego", "UC Davis", "UC Santa Barbara", "Cal Poly",
    ],
    "OVC": [
        "Tennessee State", "Morehead State", "Southeast Missouri State",
        "UT Martin", "SIU Edwardsville", "Lindenwood", "Little Rock",
        "Eastern Illinois",
    ],
    "Horizon": [
        "Wright State", "Robert Morris", "Detroit Mercy", "Oakland",
        "Green Bay", "Purdue Fort Wayne", "Northern Kentucky", "Milwaukee",
        "Youngstown State", "Cleveland State", "IU Indianapolis",
    ],
    "Big Sky": [
        "Portland State", "Montana State", "Eastern Washington", "Montana",
        "Northern Colorado", "Weber State", "Idaho", "Sacramento State",
        "Idaho State", "Northern Arizona",
    ],
    "SoCon": [
        "East Tennessee State", "Wofford", "Samford", "Mercer",
        "Western Carolina", "Furman", "UNC Greensboro", "Chattanooga",
        "The Citadel", "VMI",
    ],
    "Patriot": [
        "Navy", "Lehigh", "Colgate", "Boston University",
        "American University", "Loyola Maryland", "Lafayette", "Bucknell",
        "Army", "Holy Cross",
    ],
    "NEC": [
        "Long Island University", "Central Connecticut", "Mercyhurst",
        "Le Moyne", "Stonehill", "Fairleigh Dickinson", "Wagner",
        "Chicago State",
    ],
    "Big South": [
        "High Point", "Winthrop", "Radford", "UNC Asheville", "Longwood",
        "Presbyterian", "Charleston Southern", "South Carolina Upstate",
        "Gardner-Webb",
    ],
    "Summit": [
        "North Dakota State", "St. Thomas-Minnesota", "North Dakota",
        "South Dakota", "Omaha", "Denver", "South Dakota State",
        "Oral Roberts", "Kansas City",
    ],
    "Am. East": [
        "UMBC", "Vermont", "NJIT", "UMass Lowell", "UAlbany", "Maine",
        "Bryant", "New Hampshire",
    ],
    "Ivy": [
        "Yale", "Harvard", "Pennsylvania", "Cornell",
    ],
    "MEAC": [
        "Howard", "Morgan State", "North Carolina Central", "Norfolk State",
        "South Carolina State", "Maryland Eastern Shore", "Delaware State",
    ],
    "WAC": [
        "Utah Valley", "California Baptist", "Utah Tech", "UT Arlington",
        "Southern Utah", "Abilene Christian", "Tarleton State",
    ],
}


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
    cc = max_w + 2  # connector column (where ┐│├┘ do)

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
    name_to_id = {t["team"]: t["team_id"] for t in rankings["teams"]}

    # Group by conference
    conf_teams: dict[str, list[dict]] = {}
    for t in rankings["teams"]:
        conf_teams.setdefault(t["conference"], []).append(t)

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

    # Build ordered conference list (power 5 first, then by size)
    power5 = ["ACC", "Big Ten", "Big 12", "SEC", "Big East"]
    other_confs = sorted(
        [c for c in conf_teams if c not in power5],
        key=lambda c: -len(conf_teams[c]),
    )
    conf_order = power5 + other_confs

    out = StringIO()
    out.write("# Conference Tournament Bracket Predictions — 2026\n\n")
    out.write("Model picks the predicted winner of every game. Win % = model confidence.\n\n")

    json_confs = []

    for conf_name in conf_order:
        teams_list = conf_teams.get(conf_name, [])
        if not teams_list:
            continue

        # Use official seedings if available, else fall back to conf record sort
        seedings = TOURNAMENT_SEEDINGS_2026.get(conf_name)
        if seedings:
            # Build qualified list from official seedings
            qualified = []
            for tname in seedings:
                tid = name_to_id.get(tname)
                if tid is None:
                    print(f"  WARNING: {tname} not found in rankings for {conf_name}")
                    continue
                # Find the team dict
                tdict = next((t for t in teams_list if t["team_id"] == tid), None)
                if tdict:
                    qualified.append(tdict)
            n_qualify = len(qualified)
        else:
            # Fallback: sort by conf record
            teams_list.sort(
                key=lambda t: (
                    parse_record(t.get("conf_record", "0-0"))[0],
                    -parse_record(t.get("conf_record", "0-0"))[1],
                    t["adj_margin"],
                ),
                reverse=True,
            )
            fmt = CONFERENCE_FORMATS.get(conf_name)
            if fmt is not None:
                n_qualify = min(fmt[0], len(teams_list))
            else:
                n_qualify = len(teams_list)
            qualified = teams_list[:n_qualify]

        if n_qualify < 2:
            continue

        # Get bracket structure — use format only if team count matches
        fmt = CONFERENCE_FORMATS.get(conf_name)
        if fmt is not None and fmt[0] == n_qualify:
            _, bracket = fmt
        else:
            bracket = build_bracket(n_qualify)

        seed_map = {i + 1: t["team_id"] for i, t in enumerate(qualified)}
        champ_id, champ_seed = get_bracket_winner(bracket, seed_map, pred)
        champ_name = names.get(champ_id, "???")

        lines, _ = render_bracket(bracket, seed_map, pred, names)

        # Mark champion on the outermost winner line
        for i, line in enumerate(lines):
            if "├── " in line and line.strip().startswith("├──"):
                if "┐" not in line[line.index("├──"):] and "┘" not in line[line.index("├──"):]:
                    lines[i] = line + "  ★ CHAMPION"
                    break

        # DNQ = teams in conf but not in qualified list
        qualified_ids = {t["team_id"] for t in qualified}
        excluded = [t for t in teams_list if t["team_id"] not in qualified_ids]

        out.write("---\n\n")
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
