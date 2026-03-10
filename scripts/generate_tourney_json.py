#!/usr/bin/env python3
"""Parse conference_tournament_analysis.md and emit tourneys_2026.json."""

from __future__ import annotations
import json
import re
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MD_PATH = PROJECT_ROOT / "reports" / "conference_tournament_analysis.md"
OUT_PATH = PROJECT_ROOT / "site" / "public" / "data" / "tourneys_2026.json"


def parse_team_row(cols: list[str], has_hrb: bool) -> dict | None:
    """Parse a single markdown table row into a team dict."""
    seed_raw = cols[0].strip()
    team = cols[1].strip()
    conf_record = cols[2].strip()

    if seed_raw == "--" or team == "":
        # DNQ row
        if team:
            return {"team": team, "dnq": True}
        return None

    try:
        seed = int(seed_raw)
    except ValueError:
        return None

    model_pct_str = cols[3].strip().replace("%", "")
    model_odds = cols[4].strip()

    row: dict = {
        "seed": seed,
        "team": team,
        "conf_record": conf_record,
        "model_pct": float(model_pct_str) if model_pct_str not in ("DNQ", "---") else None,
        "model_odds": model_odds if model_odds not in ("---",) else None,
    }

    if has_hrb:
        hrb_odds = cols[5].strip() if len(cols) > 5 else "--"
        vegas_pct_str = cols[6].strip().replace("%", "") if len(cols) > 6 else "--"
        edge_str = cols[7].strip().replace("%", "") if len(cols) > 7 else "--"
        flag = cols[8].strip() if len(cols) > 8 else ""

        row["hrb_odds"] = hrb_odds if hrb_odds != "--" else None
        row["vegas_implied_pct"] = float(vegas_pct_str) if vegas_pct_str not in ("--", "---") else None
        row["edge"] = float(edge_str) if edge_str not in ("--", "---") else None
        row["flag"] = flag if flag else None

    return row


def parse_summary_row(cols: list[str]) -> dict:
    """Parse a value-bet or fade summary row."""
    conf = cols[0].strip()
    team = cols[1].strip()
    model_pct = float(cols[2].strip().replace("%", ""))
    hrb_odds = cols[3].strip()
    edge = float(cols[4].strip().replace("%", "").replace("+", ""))
    flag = cols[5].strip()
    return {
        "conf": conf,
        "team": team,
        "model_pct": model_pct,
        "hrb_odds": hrb_odds,
        "edge": edge,
        "flag": flag,
    }


def main():
    text = MD_PATH.read_text()
    lines = text.splitlines()

    conferences: list[dict] = []
    value_bets: list[dict] = []
    fades: list[dict] = []

    i = 0
    in_value_bets = False
    in_fades = False

    while i < len(lines):
        line = lines[i]

        # Detect summary sections
        if line.startswith("## Top Value Bets"):
            in_value_bets = True
            in_fades = False
            i += 1
            continue
        if line.startswith("## Top Fades"):
            in_fades = True
            in_value_bets = False
            i += 1
            continue

        # Summary table rows
        if (in_value_bets or in_fades) and line.startswith("|") and not line.startswith("|---") and not line.startswith("| Conf"):
            cols = [c for c in line.split("|")[1:-1]]
            if len(cols) >= 6:
                row = parse_summary_row(cols)
                if in_value_bets:
                    value_bets.append(row)
                else:
                    fades.append(row)
            i += 1
            continue

        # Conference header: ### ACC (15 of 18 qualify) or ### Big Ten (18 teams)
        m = re.match(r"^### (.+?)(?:\s*\((.+?)\))?\s*$", line)
        if m and not in_value_bets and not in_fades:
            conf_name = m.group(1).strip()
            qualifier_text = m.group(2).strip() if m.group(2) else ""
            i += 1

            # Check for HRB odds source
            has_hrb = False
            while i < len(lines):
                if lines[i].startswith("**Odds Source:"):
                    has_hrb = True
                    i += 1
                    break
                elif lines[i].startswith("*HRB odds not available"):
                    has_hrb = False
                    i += 1
                    break
                elif lines[i].startswith("|"):
                    break
                i += 1

            # Skip to table header
            while i < len(lines) and not lines[i].startswith("| Seed"):
                i += 1

            if i >= len(lines):
                break

            # Skip header + separator
            i += 2  # header + |---|---|...

            teams: list[dict] = []
            dnq: list[str] = []

            while i < len(lines) and lines[i].startswith("|"):
                cols = [c for c in lines[i].split("|")[1:-1]]
                row = parse_team_row(cols, has_hrb)
                if row:
                    if row.get("dnq"):
                        dnq.append(row["team"])
                    else:
                        teams.append(row)
                i += 1

            conferences.append({
                "name": conf_name,
                "qualifier_text": qualifier_text,
                "has_hrb_odds": has_hrb,
                "teams": teams,
                "dnq": dnq,
            })
            continue

        i += 1

    data = {
        "generated_at": datetime.now().isoformat(),
        "season": 2026,
        "methodology": {
            "simulations": 50000,
            "odds_source": "Hard Rock Bet (Florida)",
            "note": "Neutral-site symmetric averaging. Official conference tournament seedings.",
        },
        "conferences": conferences,
        "value_bets": value_bets,
        "fades": fades,
    }

    OUT_PATH.write_text(json.dumps(data, indent=2))
    print(f"Wrote {len(conferences)} conferences, {len(value_bets)} value bets, {len(fades)} fades")
    print(f"  -> {OUT_PATH}")


if __name__ == "__main__":
    main()
