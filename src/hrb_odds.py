"""Live Hard Rock Bet NCAAB odds fetch + game matching."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib import error, request
from zoneinfo import ZoneInfo

import pandas as pd

from . import config

_ET = ZoneInfo("America/New_York")
_HRB_PROVIDER = "Hard Rock Bet"
_HRB_SEGMENT_URL = "https://api.hardrocksportsbook.com/sportsbook/v1/api/whatsMySegment"
_HRB_ROOT_LADDER_URL = "https://api.hardrocksportsbook.com/sportsbook/v1/api/getRootLadder"
_HRB_GRAPHQL_URL = "https://api.hardrocksportsbook.com/java-graphql/graphql?type=event_tree"
_HRB_EVENT_TREE_URL = (
    "https://api.hardrocksportsbook.com/sportsbook/api/public/events/tree"
)
_HRB_NCAAB_COMPETITION_NAME = "NCAAB"
_HRB_GRAPHQL_QUERY = """
query betSync(
  $filters: [Filter]
  $segment: String
  $region: String
  $language: String
  $search: String
  $slice: Interval
  $timeInterval: Interval
  $channel: String
  $marketTypes: [String]
  $sports: [String]
) {
  betSync(
    cmsSegment: $segment
    region: $region
    language: $language
    channel: $channel
    eventParams: {
      marketTypes: $marketTypes
      sportList: $sports
    }
  ) {
    events(
      filters: $filters
      slice: $slice
      timeInterval: $timeInterval
      search: $search
      sports: $sports
    ) {
      data {
        id
        compId
        compName
        eventTime
        name
        participants {
          name
          position
        }
        markets(keyMarkets: true, marketTypes: $marketTypes) {
          id
          name
          type
          subtype
          line
          displayed
          suspended
          state
          selection {
            id
            name
            type
            displayed
            suspended
            rootIdx
          }
        }
      }
      count
    }
  }
}
""".strip()
_HRB_MARKET_TYPES = [
    "BASKETBALL:FTOT:SPRD",
    "BASKETBALL:FTOT:ML",
    "BASKETBALL:FTOT:OU",
]
_TEAM_ALIASES = {
    "floridaam": "floridaam",
    "miamifl": "miami",
    "miamioh": "miamioh",
    "ncstate": "northcarolinastate",
    "olemiss": "mississippi",
    "southernuniversity": "southern",
    "saintjohns": "stjohns",
    "saintjosephs": "stjosephs",
    "stjohns": "stjohns",
    "stjosephs": "stjosephs",
    "texasarlington": "utarlington",
    "uconn": "connecticut",
    "umass": "massachusetts",
    "usf": "southflorida",
}
_TEAM_VARIANT_GROUPS = [
    {"southflorida", "usf"},
    {"stlouis", "saintlouis"},
    {"stjohns", "saintjohns"},
    {"stjosephs", "saintjosephs"},
    {"utarlington", "texasarlington"},
]


@dataclass(frozen=True)
class _HrbContext:
    channel: str
    language: str
    region: str
    segment: str


@dataclass(frozen=True)
class _MatchedGame:
    game: pd.Series
    slot_to_side: dict[str, str]


def current_cbb_season(now: datetime | None = None) -> int:
    now = now or datetime.now(_ET)
    return now.year + 1 if now.month >= 11 else now.year


def live_overlay_enabled_for_season(season: int) -> bool:
    if not config.HRB_LIVE_ODDS_ENABLED:
        return False
    return season == current_cbb_season()


def fetch_hrb_lines_for_games(games_df: pd.DataFrame) -> pd.DataFrame:
    if games_df.empty:
        return pd.DataFrame()

    schedule = _prepare_games_for_matching(games_df)
    if schedule.empty:
        return pd.DataFrame()

    context = _fetch_context()
    ladder = _fetch_root_ladder()
    competition_id = _fetch_ncaab_competition_id(context)
    events = _fetch_event_rows(context, competition_id)
    if not events:
        return pd.DataFrame()

    matched_rows: list[dict[str, Any]] = []
    for event in events:
        game = _match_event_to_game(event, schedule)
        if game is None:
            continue
        line_row = _build_line_row(event, game, ladder)
        if line_row is not None:
            matched_rows.append(line_row)

    if not matched_rows:
        return pd.DataFrame()

    out = pd.DataFrame(matched_rows)
    for col in ["spread", "overUnder", "homeMoneyline", "awayMoneyline"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.drop_duplicates(subset=["gameId"], keep="last").reset_index(drop=True)


def _prepare_games_for_matching(games_df: pd.DataFrame) -> pd.DataFrame:
    games = games_df.copy()
    if games.empty:
        return games

    required = {"gameId", "awayTeam", "homeTeam", "startDate"}
    if not required.issubset(games.columns):
        return pd.DataFrame()

    games["startDate"] = pd.to_datetime(games["startDate"], utc=True, errors="coerce")
    games = games.dropna(subset=["gameId", "awayTeam", "homeTeam", "startDate"]).copy()
    if games.empty:
        return games

    now = pd.Timestamp.now(tz="UTC")
    start_floor = now - pd.Timedelta(hours=config.HRB_MATCH_LOOKBACK_HOURS)
    start_cap = now + pd.Timedelta(days=config.HRB_MATCH_LOOKAHEAD_DAYS)
    games = games[
        (games["startDate"] >= start_floor) & (games["startDate"] <= start_cap)
    ].copy()
    if games.empty:
        return games

    games["away_keys"] = games["awayTeam"].map(_team_key_variants)
    games["home_keys"] = games["homeTeam"].map(_team_key_variants)
    return games


def _fetch_context() -> _HrbContext:
    payload = _request_json(_HRB_SEGMENT_URL)
    raw = payload.get("segmentCodeResponse") or payload.get("Result") or payload
    segment = raw.get("segmentCode") or raw.get("forLayoutSegmentCode") or raw.get("cmsCode")
    channel = raw.get("channelCode")
    region = (raw.get("regionCode") or "US").lower()
    language = str(raw.get("defaultLocale") or "en-us").replace("-", "")
    if not segment or not channel:
        raise ValueError("Hard Rock Bet segment bootstrap did not return segment/channel")
    return _HrbContext(
        channel=str(channel),
        language=language,
        region=region,
        segment=str(segment),
    )


def _fetch_root_ladder() -> dict[int, dict[str, Any]]:
    payload = _request_json(_HRB_ROOT_LADDER_URL)
    ladder_rows = (
        payload.get("PriceAdjustmentDetailsResponse", {}).get("rootLadder")
        or payload.get("rootLadder")
        or []
    )
    ladder: dict[int, dict[str, Any]] = {}
    for row in ladder_rows:
        try:
            root_index = int(row["rootIndex"])
        except (KeyError, TypeError, ValueError):
            continue
        ladder[root_index] = {
            "decimal": float(row["decimal"]),
            "moneyline": int(float(row["moneyline"])),
        }
    if not ladder:
        raise ValueError("Hard Rock Bet root ladder was empty")
    return ladder


def _fetch_ncaab_competition_id(context: _HrbContext) -> str:
    url = f"{_HRB_EVENT_TREE_URL}?channel={context.channel}&segment={context.segment}"
    payload = _request_json(url)
    competition_id = _walk_for_competition_id(payload, _HRB_NCAAB_COMPETITION_NAME)
    if competition_id is None:
        raise ValueError("Unable to locate NCAAB competition in Hard Rock Bet event tree")
    return competition_id


def _walk_for_competition_id(node: Any, target_name: str) -> str | None:
    if isinstance(node, dict):
        name = node.get("name")
        if name == target_name and "id" in node:
            return str(node["id"])
        for value in node.values():
            found = _walk_for_competition_id(value, target_name)
            if found is not None:
                return found
        return None
    if isinstance(node, list):
        for item in node:
            found = _walk_for_competition_id(item, target_name)
            if found is not None:
                return found
    return None


def _fetch_event_rows(context: _HrbContext, competition_id: str) -> list[dict[str, Any]]:
    variables = {
        "channel": context.channel,
        "segment": context.segment,
        "region": context.region,
        "language": context.language,
        "filters": [
            {"field": "compId", "values": [competition_id]},
            {"field": "displayed", "value": "true"},
            {"field": "outright", "value": "false"},
        ],
        "marketTypes": _HRB_MARKET_TYPES,
        "slice": {"from": 0, "to": 500},
        "sports": ["BASKETBALL"],
    }
    payload = _request_json(
        _HRB_GRAPHQL_URL,
        data={
            "operationName": "betSync",
            "query": _HRB_GRAPHQL_QUERY,
            "variables": variables,
        },
        content_type="application/json",
    )
    errors = payload.get("errors")
    if errors:
        raise ValueError(f"Hard Rock Bet GraphQL returned errors: {errors}")
    events = payload.get("data", {}).get("betSync", {}).get("events", {}).get("data", [])
    return [event for event in events if isinstance(event, dict)]


def _match_event_to_game(event: dict[str, Any], schedule: pd.DataFrame) -> _MatchedGame | None:
    participants = sorted(
        event.get("participants") or [],
        key=lambda row: int(row.get("position", 0)),
    )
    if len(participants) < 2:
        return None

    event_keys = {
        "A": _team_key_variants(participants[0].get("name")),
        "B": _team_key_variants(participants[1].get("name")),
    }
    if not event_keys["A"] or not event_keys["B"]:
        return None

    event_time_raw = event.get("eventTime")
    if event_time_raw is None:
        return None
    event_time = pd.to_datetime(event_time_raw, unit="ms", utc=True, errors="coerce")
    if pd.isna(event_time):
        return None

    best: tuple[int, pd.Timedelta, int, pd.Series, dict[str, str]] | None = None
    max_diff = pd.Timedelta(hours=config.HRB_MATCH_TIME_TOLERANCE_HOURS)
    for _, game in schedule.iterrows():
        time_diff = abs(game["startDate"] - event_time)
        if time_diff > max_diff:
            continue

        away_keys = set(game["away_keys"])
        home_keys = set(game["home_keys"])
        score_ab = int(bool(event_keys["A"] & away_keys)) + int(bool(event_keys["B"] & home_keys))
        score_ba = int(bool(event_keys["A"] & home_keys)) + int(bool(event_keys["B"] & away_keys))
        if score_ab < 2 and score_ba < 2:
            continue

        if score_ba > score_ab:
            slot_to_side = {"A": "home", "B": "away"}
            name_score = score_ba
        else:
            slot_to_side = {"A": "away", "B": "home"}
            name_score = score_ab

        candidate = (name_score, time_diff, int(game["gameId"]), game, slot_to_side)
        if best is None or (name_score > best[0]) or (
            name_score == best[0] and (time_diff, int(game["gameId"])) < (best[1], best[2])
        ):
            best = candidate

    if best is None:
        return None
    return _MatchedGame(game=best[3], slot_to_side=best[4])


def _build_line_row(
    event: dict[str, Any],
    matched_game: _MatchedGame,
    ladder: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    game = matched_game.game
    participants = sorted(
        event.get("participants") or [],
        key=lambda row: int(row.get("position", 0)),
    )
    if len(participants) < 2:
        return None

    moneyline = _extract_moneyline(event.get("markets") or [], matched_game.slot_to_side, ladder)
    spread = _extract_spread(event.get("markets") or [], matched_game.slot_to_side, ladder)
    total = _extract_total(event.get("markets") or [], ladder)

    if moneyline[0] is None and moneyline[1] is None and spread is None and total is None:
        return None

    return {
        "gameId": int(game["gameId"]),
        "provider": _HRB_PROVIDER,
        "awayTeam": game["awayTeam"],
        "homeTeam": game["homeTeam"],
        "startDate": game["startDate"].isoformat().replace("+00:00", "Z"),
        "season": int(current_cbb_season()),
        "spread": spread,
        "overUnder": total,
        "homeMoneyline": moneyline[0],
        "awayMoneyline": moneyline[1],
        "hrbEventId": str(event.get("id") or ""),
    }


def _extract_moneyline(
    markets: list[dict[str, Any]],
    slot_to_side: dict[str, str],
    ladder: dict[int, dict[str, Any]],
) -> tuple[int | None, int | None]:
    best_score: float | None = None
    best: tuple[int | None, int | None] = (None, None)
    for market in markets:
        if market.get("type") != "BASKETBALL:FTOT:ML":
            continue
        selections = market.get("selection") or []
        if len(selections) < 2:
            continue

        by_side: dict[str, int] = {}
        decimals: list[float] = []
        for selection in selections:
            team_slot = _selection_team_slot(str(selection.get("type") or ""))
            if team_slot is None:
                continue
            side = slot_to_side.get(team_slot)
            odds = _odds_from_root_idx(selection.get("rootIdx"), ladder)
            if side is None or odds is None:
                continue
            by_side[side] = odds["moneyline"]
            decimals.append(odds["decimal"])

        if "home" not in by_side or "away" not in by_side or len(decimals) < 2:
            continue
        score = abs(decimals[0] - decimals[1])
        if best_score is None or score < best_score:
            best_score = score
            best = (by_side["home"], by_side["away"])
    return best


def _extract_spread(
    markets: list[dict[str, Any]],
    slot_to_side: dict[str, str],
    ladder: dict[int, dict[str, Any]],
) -> float | None:
    best_score: float | None = None
    best_line: float | None = None
    for market in markets:
        if market.get("type") != "BASKETBALL:FTOT:SPRD":
            continue
        selections = market.get("selection") or []
        if len(selections) < 2:
            continue

        decimals: list[float] = []
        line_by_side: dict[str, float] = {}
        for selection in selections:
            team_slot = _selection_team_slot(str(selection.get("type") or ""))
            if team_slot is None:
                continue
            side = slot_to_side.get(team_slot)
            odds = _odds_from_root_idx(selection.get("rootIdx"), ladder)
            spread_line = _parse_signed_line(selection.get("name"))
            if side is None or odds is None or spread_line is None:
                continue
            line_by_side[side] = spread_line
            decimals.append(odds["decimal"])

        if "home" not in line_by_side or len(decimals) < 2:
            continue
        score = abs(decimals[0] - decimals[1])
        if best_score is None or score < best_score:
            best_score = score
            best_line = line_by_side["home"]
    return best_line


def _extract_total(markets: list[dict[str, Any]], ladder: dict[int, dict[str, Any]]) -> float | None:
    best_score: float | None = None
    best_line: float | None = None
    for market in markets:
        if market.get("type") != "BASKETBALL:FTOT:OU":
            continue
        selections = market.get("selection") or []
        if len(selections) < 2:
            continue

        decimals: list[float] = []
        for selection in selections:
            odds = _odds_from_root_idx(selection.get("rootIdx"), ladder)
            if odds is not None:
                decimals.append(odds["decimal"])
        total_line = _parse_market_points(market.get("subtype"))
        if total_line is None or len(decimals) < 2:
            continue
        score = abs(decimals[0] - decimals[1])
        if best_score is None or score < best_score:
            best_score = score
            best_line = total_line
    return best_line


def _selection_team_slot(selection_type: str) -> str | None:
    selection_type = (selection_type or "").upper()
    if selection_type.startswith("A"):
        return "A"
    if selection_type.startswith("B"):
        return "B"
    return None


def _parse_signed_line(text: Any) -> float | None:
    if not isinstance(text, str):
        return None
    match = re.search(r"([+-]\d+(?:\.\d+)?)\s*$", text)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _parse_market_points(text: Any) -> float | None:
    if not isinstance(text, str):
        return None
    match = re.search(r"#(-?\d+(?:\.\d+)?)$", text)
    if match is None:
        return None
    try:
        return abs(float(match.group(1)))
    except ValueError:
        return None


def _odds_from_root_idx(root_idx: Any, ladder: dict[int, dict[str, Any]]) -> dict[str, Any] | None:
    try:
        return ladder[int(root_idx)]
    except (KeyError, TypeError, ValueError):
        return None


def _normalize_team_name(name: Any) -> str:
    if not isinstance(name, str):
        return ""
    text = name.strip().lower().replace("’", "'")
    text = re.sub(r"^\(?\d+\)?\s+", "", text)
    text = text.replace("&", " and ")
    text = text.replace("(oh)", " oh ")
    text = text.replace("(fl)", " fl ")
    text = text.replace("saint", "st")
    text = text.replace("st.", "st")
    text = re.sub(r"[^a-z0-9]+", "", text)
    return _TEAM_ALIASES.get(text, text)


def _team_key_variants(name: Any) -> frozenset[str]:
    base = _normalize_team_name(name)
    if not base:
        return frozenset()

    variants = {base}
    alias_target = _TEAM_ALIASES.get(base)
    if alias_target:
        variants.add(alias_target)
    for group in _TEAM_VARIANT_GROUPS:
        if base in group:
            variants.update(group)
    return frozenset(variants)


def _request_json(
    url: str,
    data: dict[str, Any] | None = None,
    content_type: str | None = None,
) -> dict[str, Any]:
    headers = {
        "accept": "application/json, text/plain, */*",
        "origin": "https://app.hardrock.bet",
        "referer": "https://app.hardrock.bet/",
        "user-agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
        ),
    }
    payload = None
    if data is not None:
        payload = json.dumps(data).encode("utf-8")
        headers["content-type"] = content_type or "application/json"

    req = request.Request(url, data=payload, headers=headers, method="POST" if payload else "GET")
    try:
        with request.urlopen(req, timeout=config.HRB_HTTP_TIMEOUT_SECS) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise ValueError(f"Hard Rock Bet request failed [{exc.code}] {url}: {detail[:500]}") from exc
