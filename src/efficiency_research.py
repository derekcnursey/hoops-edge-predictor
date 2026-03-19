"""Research helpers for adjusted-efficiency experiments."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SoftGarbageConfig:
    """Settings for soft garbage-time attenuation."""

    garbage_keep_fraction: float = 0.25


@dataclass(frozen=True)
class ConferenceBridgeConfig:
    """Settings for conference-bridge prior adaptation."""

    bridge_weight: float = 0.6
    min_peer_teams: int = 3
    coverage_exponent: float = 0.5


def build_conference_bridge_prior(
    current_result: dict[int, dict[str, float | int]],
    preseason_prior: dict[int, tuple[float, float]] | None,
    team_conference: dict[int, str | None],
    config: ConferenceBridgeConfig = ConferenceBridgeConfig(),
) -> dict[int, tuple[float, float]]:
    """Shift team priors by peer conference drift while excluding the team itself.

    The intent is to preserve the stabilizing carryover prior while allowing the
    prior to move with conference-wide evidence as the season unfolds.
    """
    if not preseason_prior:
        return {}

    conf_members: dict[str, list[dict[str, float | int]]] = {}
    conf_sizes: dict[str, int] = {}

    for team_id, (prior_oe, prior_de) in preseason_prior.items():
        conf = team_conference.get(team_id)
        if not conf:
            continue
        conf_sizes[conf] = conf_sizes.get(conf, 0) + 1
        vals = current_result.get(team_id)
        if not vals:
            continue
        games_played = int(vals.get("games_played", 0) or 0)
        if games_played <= 0:
            continue
        weight = float(games_played)
        conf_members.setdefault(conf, []).append(
            {
                "team_id": team_id,
                "weight": weight,
                "adj_oe": float(vals["adj_oe"]),
                "adj_de": float(vals["adj_de"]),
                "prior_oe": float(prior_oe),
                "prior_de": float(prior_de),
            }
        )

    shifted: dict[int, tuple[float, float]] = {}
    for team_id, (prior_oe, prior_de) in preseason_prior.items():
        conf = team_conference.get(team_id)
        if not conf:
            shifted[team_id] = (prior_oe, prior_de)
            continue

        peers = conf_members.get(conf, [])
        peer_rows = [row for row in peers if int(row["team_id"]) != team_id]
        if len(peer_rows) < config.min_peer_teams:
            shifted[team_id] = (prior_oe, prior_de)
            continue

        total_w = sum(float(row["weight"]) for row in peer_rows)
        if total_w <= 0:
            shifted[team_id] = (prior_oe, prior_de)
            continue

        current_oe = sum(float(row["adj_oe"]) * float(row["weight"]) for row in peer_rows) / total_w
        current_de = sum(float(row["adj_de"]) * float(row["weight"]) for row in peer_rows) / total_w
        prior_peer_oe = sum(float(row["prior_oe"]) * float(row["weight"]) for row in peer_rows) / total_w
        prior_peer_de = sum(float(row["prior_de"]) * float(row["weight"]) for row in peer_rows) / total_w

        conf_size = max(conf_sizes.get(conf, len(peer_rows) + 1), 1)
        coverage = min(1.0, len(peer_rows) / max(conf_size - 1, 1))
        coverage_scale = coverage ** config.coverage_exponent
        oe_shift = config.bridge_weight * coverage_scale * (current_oe - prior_peer_oe)
        de_shift = config.bridge_weight * coverage_scale * (current_de - prior_peer_de)
        shifted[team_id] = (prior_oe + oe_shift, prior_de + de_shift)

    return shifted


def blend_soft_garbage_team_games(
    raw_frame: pd.DataFrame,
    no_garbage_frame: pd.DataFrame,
    config: SoftGarbageConfig = SoftGarbageConfig(),
) -> pd.DataFrame:
    """Keep a fraction of the possessions/points removed as garbage time.

    The candidate branch is intentionally simple: binary no-garbage removal may
    throw away too much low-signal but still directionally useful information.
    This function blends the raw and no-garbage team-game rows so that:

    - `0.0` = current no-garbage baseline
    - `1.0` = full raw game totals
    """
    keep = float(config.garbage_keep_fraction)
    if keep < 0.0 or keep > 1.0:
        raise ValueError("garbage_keep_fraction must be between 0 and 1")

    keys = ["gameid", "teamid"]
    merged = raw_frame.merge(
        no_garbage_frame,
        on=keys,
        how="inner",
        suffixes=("_raw", "_ng"),
    ).copy()
    if merged.empty:
        return merged

    # Preserve the baseline no-garbage row shape/metadata and only soften the
    # points/possession totals that feed the adjusted-efficiency solver.
    out = pd.DataFrame(
        {
            "gameid": merged["gameid"],
            "teamid": merged["teamid"],
            "opponentid": merged["opponentid_ng"].fillna(merged["opponentid_raw"]),
            "startdate": merged["startdate_ng"].fillna(merged["startdate_raw"]),
            "ishometeam": merged["ishometeam_ng"].fillna(merged["ishometeam_raw"]),
            "garbage_time_minutes": merged["garbage_time_minutes_ng"].fillna(
                merged["garbage_time_minutes_raw"]
            ),
            "game_minutes": merged["game_minutes_ng"].fillna(merged["game_minutes_raw"]),
            "max_period": merged["max_period_ng"].fillna(merged["max_period_raw"]),
        }
    )

    blend_cols = [
        "team_points_total",
        "opp_points_total",
        "team_possessions",
        "opp_possessions",
        "team_possessions_formula",
        "opp_possessions_formula",
    ]
    for col in blend_cols:
        raw_col = f"{col}_raw"
        ng_col = f"{col}_ng"
        if raw_col not in merged.columns and ng_col not in merged.columns:
            continue
        raw_vals = pd.to_numeric(merged.get(raw_col), errors="coerce")
        ng_vals = pd.to_numeric(merged.get(ng_col), errors="coerce")
        blend = ng_vals + keep * (raw_vals - ng_vals)
        out[col] = blend.where(raw_vals.notna() & ng_vals.notna(), ng_vals.where(ng_vals.notna(), raw_vals))

    out["soft_removed_possessions"] = (
        pd.to_numeric(merged.get("team_possessions_formula_raw"), errors="coerce")
        - pd.to_numeric(merged.get("team_possessions_formula_ng"), errors="coerce")
    )
    out["garbage_keep_fraction"] = keep
    return out


def select_min_possession_team_game_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate team-game rows by keeping the lowest-possession copy.

    The repaired 2026 PBP tables exposed a real production failure mode where
    the same team-game appeared multiple times at different scale factors.
    For the soft-garbage branch, the lowest-possession copy is the canonical
    one; higher copies are inflated duplicates.
    """
    if frame.empty:
        return frame

    keys = ["gameid", "teamid"]
    for key in keys:
        if key not in frame.columns:
            return frame

    poss_col = None
    for candidate in ("team_possessions_formula", "team_possessions"):
        if candidate in frame.columns:
            poss_col = candidate
            break

    if poss_col is None:
        return frame.drop_duplicates(keys, keep="first").reset_index(drop=True)

    working = frame.copy()
    working["_poss_metric"] = pd.to_numeric(working[poss_col], errors="coerce").fillna(float("inf"))
    working = working.sort_values(keys + ["_poss_metric"], kind="mergesort")
    working = working.drop_duplicates(keys, keep="first")
    return working.drop(columns="_poss_metric").reset_index(drop=True)


def classify_efficiency_slice(
    frame: pd.DataFrame,
    *,
    date_col: str = "startDate",
    season_col: str = "holdout_season",
    season_type_col: str = "seasonType",
    game_type_col: str = "gameType",
    tournament_col: str = "tournament",
    conference_game_col: str = "conferenceGame",
) -> pd.Series:
    """Attach March-focused evaluation slice labels for research reporting."""
    out = pd.Series("full", index=frame.index, dtype="object")
    dt = pd.to_datetime(frame[date_col], errors="coerce", utc=True).dt.tz_convert("America/New_York")

    holdout = frame[season_col].astype("Int64")
    cutoff_dec15 = pd.to_datetime((holdout - 1).astype(str) + "-12-15", errors="coerce")
    cutoff_feb15 = pd.to_datetime(holdout.astype(str) + "-02-15", errors="coerce")

    season_type = frame.get(season_type_col, pd.Series(index=frame.index, dtype="object")).astype(str)
    game_type = frame.get(game_type_col, pd.Series(index=frame.index, dtype="object")).astype(str)
    tournament = frame.get(tournament_col, pd.Series(index=frame.index, dtype="object"))
    conference_game = (
        frame.get(conference_game_col, pd.Series(False, index=frame.index))
        .fillna(False)
        .astype(bool)
    )

    is_dec15_plus = dt.dt.tz_localize(None) >= cutoff_dec15
    is_feb15_plus = dt.dt.tz_localize(None) >= cutoff_feb15
    is_march = dt.dt.month.eq(3)
    is_ncaa = tournament.astype(str).eq("NCAA")
    is_conf_tourney = (
        game_type.eq("TRNMNT")
        & tournament.isna()
        & conference_game
        & is_march
        & dt.dt.day.between(4, 17, inclusive="both")
    )
    is_postseason = season_type.eq("postseason")

    out.loc[is_dec15_plus.fillna(False)] = "dec15_plus"
    out.loc[is_feb15_plus.fillna(False)] = "feb15_plus"
    out.loc[is_march.fillna(False)] = "march_only"
    out.loc[is_conf_tourney.fillna(False)] = "conference_tournaments"
    out.loc[is_ncaa.fillna(False)] = "ncaa_tournament"
    out.loc[(is_postseason & ~is_ncaa).fillna(False)] = "ignore_other_postseason"
    return out
