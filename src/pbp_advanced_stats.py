"""Single-pass PBP advanced stats extraction.

Reads enriched play-by-play data and computes per-game per-team stats for:
  - Group A: Shot quality & distribution (rim_rate, mid_range_rate, etc.)
  - Group C: Turnover decomposition (live_ball_tov_rate, dead_ball_tov_rate, etc.)
  - Group D: Possession-level tempo (avg_possession_length, early/late clock rates)
  - Group G: Putback / second-chance features
  - Group H: Clutch performance splits
  - Group I: Scoring drought frequency
  - Group J: Half / period splits
  - Group M: Rotation depth / scoring concentration
  - Group N: Free throw shooting under pressure

All stats exclude garbage_time plays by default.
Output: one row per team per game with all computed stats.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ── Shot classification helpers ──────────────────────────────────

def _classify_shot_zone(shot_range: Optional[str]) -> Optional[str]:
    """Map shot_range values to zones: rim, mid_range, three_pointer, free_throw."""
    if shot_range is None or pd.isna(shot_range):
        return None
    sr = str(shot_range).lower().strip()
    if sr == "rim":
        return "rim"
    elif sr == "jumper":
        return "mid_range"
    elif sr == "three_pointer":
        return "three_pointer"
    elif sr == "free_throw":
        return "free_throw"
    return None


def _is_corner_three(
    shot_range: Optional[str],
    shot_loc_x: Optional[float],
    shot_loc_y: Optional[float],
    basket_x: float = 827.0,
) -> bool:
    """Detect corner three-pointers from pixel coordinates.

    Corner 3s: three-pointers with large horizontal distance from basket
    and near the baseline (low y-value in pixel coords).

    Pixel coordinate system (from data exploration):
      - Basket at approximately (827, 250)
      - Court spans x: ~50-920, y: ~0-500
      - Corner 3s have high |x - basket_x| and low y (< ~85)
        OR high |x - basket_x| and high y (> ~415)
    """
    if shot_range != "three_pointer":
        return False
    if shot_loc_x is None or shot_loc_y is None:
        return False
    if pd.isna(shot_loc_x) or pd.isna(shot_loc_y):
        return False

    dist_x = abs(float(shot_loc_x) - basket_x)
    y = float(shot_loc_y)

    # Corner threes: far from center horizontally AND near baseline
    return dist_x > 200 and (y < 90 or y > 410)


# ── Main extraction function ─────────────────────────────────────

def compute_advanced_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """Compute all PBP-derived advanced stats in a single pass per game.

    Args:
        pbp: Enriched PBP DataFrame with columns:
            gameId, teamId, opponentId, period, secondsRemaining,
            playType, scoringPlay, shootingPlay, scoreValue,
            possession_id, offense_team_id, defense_team_id, possession_end,
            garbage_time, shot_range, shot_made, shot_assisted,
            shot_loc_x, shot_loc_y, shot_shooter_id,
            homeScore, awayScore, isHomeTeam, gameStartDate

    Returns:
        DataFrame with one row per team per game containing all stats.
        Columns: gameid, teamid, opponentid, startdate, + stat columns.
    """
    # Filter out garbage time
    df = pbp[~pbp["garbage_time"]].copy()

    if df.empty:
        return pd.DataFrame()

    # Ensure types
    df["scoreValue"] = pd.to_numeric(df["scoreValue"], errors="coerce").fillna(0)
    df["shot_made_bool"] = df["shot_made"].astype(str).str.lower() == "true"
    df["shot_assisted_bool"] = df["shot_assisted"].astype(str).str.lower() == "true"

    # Classify shot zones
    df["shot_zone"] = df["shot_range"].apply(_classify_shot_zone)

    # Detect corner threes
    df["is_corner_three"] = df.apply(
        lambda r: _is_corner_three(r["shot_range"], r.get("shot_loc_x"), r.get("shot_loc_y")),
        axis=1,
    )

    results = []

    # Group by game, then iterate over both teams using offense/defense team IDs
    # This is necessary because teamId tracks the acting team, while
    # offense_team_id/defense_team_id track possession assignment.
    for game_id, game_df in df.groupby("gameId"):
        # Identify both teams from offense_team_id
        teams_in_game = game_df["offense_team_id"].dropna().unique()
        start_date = game_df["gameStartDate"].iloc[0] if "gameStartDate" in game_df.columns else None

        for team_id in teams_in_game:
            team_id = int(team_id)

            # Get opponent ID
            other_teams = [t for t in teams_in_game if int(t) != team_id]
            opp_id = int(other_teams[0]) if other_teams else None

            row = {
                "gameid": int(game_id),
                "teamid": team_id,
                "opponentid": opp_id,
                "startdate": start_date,
            }

            # Split ALL game plays into this team's offense and defense
            # using the possession-level offense/defense team assignment
            off_plays = game_df[game_df["offense_team_id"] == team_id]
            def_plays = game_df[game_df["defense_team_id"] == team_id]

            # ── Group A: Shot quality & distribution ──────────────────
            row.update(_compute_shot_quality(off_plays, def_plays))

            # ── Group C: Turnover decomposition ───────────────────────
            row.update(_compute_turnover_decomp(off_plays, def_plays, df, game_id, team_id))

            # ── Group D: Possession-level tempo ───────────────────────
            row.update(_compute_tempo_features(off_plays))

            # ── Group G: Putback / second-chance ──────────────────────
            row.update(_compute_putback_features(off_plays))

            # ── Group H: Clutch performance splits ────────────────────
            row.update(_compute_clutch_stats(game_df, team_id))

            # ── Group I: Scoring drought frequency ────────────────────
            row.update(_compute_drought_features(off_plays, def_plays, df, game_id, team_id))

            # ── Group J: Half / period splits ─────────────────────────
            row.update(_compute_half_splits(game_df, team_id))

            # ── Group M: Rotation depth / scoring concentration ───────
            row.update(_compute_rotation_depth(off_plays))

            # ── Group N: Pressure free throws ─────────────────────────
            row.update(_compute_pressure_ft(game_df, team_id))

            # ── Composite features ──────────────────────────────────
            row.update(_compute_composites(row))

            results.append(row)

    return pd.DataFrame(results)


# ── Group A: Shot quality & distribution ─────────────────────────

def _compute_shot_quality(off_plays: pd.DataFrame, def_plays: pd.DataFrame) -> dict:
    """Compute shot quality and distribution stats."""
    stats = {}

    # Offensive shooting plays (excluding free throws)
    off_shots = off_plays[
        (off_plays["shootingPlay"] == True) & (off_plays["shot_zone"] != "free_throw")
    ]
    total_fga = len(off_shots)

    if total_fga > 0:
        # Shot distribution rates
        rim_shots = off_shots[off_shots["shot_zone"] == "rim"]
        mid_shots = off_shots[off_shots["shot_zone"] == "mid_range"]
        three_shots = off_shots[off_shots["shot_zone"] == "three_pointer"]

        stats["rim_rate"] = len(rim_shots) / total_fga
        stats["mid_range_rate"] = len(mid_shots) / total_fga

        # Corner 3 vs above-break 3
        total_3pa = len(three_shots)
        if total_3pa > 0:
            corner_3s = off_shots[off_shots["is_corner_three"] == True]
            stats["three_pt_rate_corner"] = len(corner_3s) / total_3pa
            stats["three_pt_rate_above_break"] = 1.0 - stats["three_pt_rate_corner"]
        else:
            stats["three_pt_rate_corner"] = np.nan
            stats["three_pt_rate_above_break"] = np.nan

        # Shot efficiency by zone
        rim_made = rim_shots[rim_shots["shot_made_bool"] == True]
        stats["rim_fg_pct"] = len(rim_made) / len(rim_shots) if len(rim_shots) > 0 else np.nan

        mid_made = mid_shots[mid_shots["shot_made_bool"] == True]
        stats["mid_range_fg_pct"] = len(mid_made) / len(mid_shots) if len(mid_shots) > 0 else np.nan

        # Assisted FG%
        made_fgs = off_shots[off_shots["shot_made_bool"] == True]
        if len(made_fgs) > 0:
            assisted = made_fgs[made_fgs["shot_assisted_bool"] == True]
            stats["assisted_fg_pct"] = len(assisted) / len(made_fgs)
            stats["unassisted_fg_pct"] = 1.0 - stats["assisted_fg_pct"]
        else:
            stats["assisted_fg_pct"] = np.nan
            stats["unassisted_fg_pct"] = np.nan

        # Zone counts (intermediate data for luck/composite features)
        stats["rim_fga"] = len(rim_shots)
        stats["rim_fgm"] = len(rim_made)
        stats["mid_fga"] = len(mid_shots)
        stats["mid_fgm"] = len(mid_made)
        stats["three_fga"] = total_3pa
        three_made = three_shots[three_shots["shot_made_bool"] == True]
        stats["three_fgm"] = len(three_made)

        # Raw FG% by shot type
        stats["three_pt_fg_pct"] = len(three_made) / total_3pa if total_3pa > 0 else np.nan
        two_pt_shots = off_shots[off_shots["shot_zone"].isin(["rim", "mid_range"])]
        two_pt_made = two_pt_shots[two_pt_shots["shot_made_bool"] == True]
        stats["two_pt_fg_pct"] = len(two_pt_made) / len(two_pt_shots) if len(two_pt_shots) > 0 else np.nan
    else:
        for col in ["rim_rate", "mid_range_rate", "three_pt_rate_corner",
                     "three_pt_rate_above_break", "rim_fg_pct", "mid_range_fg_pct",
                     "assisted_fg_pct", "unassisted_fg_pct"]:
            stats[col] = np.nan
        for col in ZONE_COUNT_COLS:
            stats[col] = np.nan

    # Defensive shot quality (opponent shooting against this team)
    def_shots = def_plays[
        (def_plays["shootingPlay"] == True) & (def_plays["shot_zone"] != "free_throw")
    ]
    def_total_fga = len(def_shots)

    if def_total_fga > 0:
        def_rim = def_shots[def_shots["shot_zone"] == "rim"]
        def_mid = def_shots[def_shots["shot_zone"] == "mid_range"]

        stats["def_rim_rate"] = len(def_rim) / def_total_fga
        stats["def_mid_range_rate"] = len(def_mid) / def_total_fga

        def_rim_made = def_rim[def_rim["shot_made_bool"] == True]
        stats["def_rim_fg_pct"] = len(def_rim_made) / len(def_rim) if len(def_rim) > 0 else np.nan

        def_mid_made = def_mid[def_mid["shot_made_bool"] == True]
        stats["def_mid_range_fg_pct"] = len(def_mid_made) / len(def_mid) if len(def_mid) > 0 else np.nan
    else:
        for col in ["def_rim_rate", "def_mid_range_rate", "def_rim_fg_pct", "def_mid_range_fg_pct"]:
            stats[col] = np.nan

    return stats


# ── Group C: Turnover decomposition ──────────────────────────────

def _compute_turnover_decomp(
    off_plays: pd.DataFrame,
    def_plays: pd.DataFrame,
    all_game_plays: pd.DataFrame,
    game_id: int,
    team_id: int,
) -> dict:
    """Compute turnover decomposition stats."""
    stats = {}

    # Count offensive possessions
    off_possessions = off_plays["possession_id"].nunique()

    if off_possessions > 0:
        # Live-ball turnovers: possessions where opponent got a steal
        # Steals are recorded under the stealing team's teamId, in the
        # same possession where offense_team_id is the team losing the ball.
        off_poss_ids = set(off_plays["possession_id"].unique())

        # Find steals in this game's possessions where our team was on offense
        game_plays = all_game_plays[all_game_plays["gameId"] == game_id]
        steals_in_our_poss = game_plays[
            (game_plays["playType"] == "Steal") &
            (game_plays["possession_id"].isin(off_poss_ids))
        ]
        live_tov_poss = set(steals_in_our_poss["possession_id"].unique())
        stats["live_ball_tov_rate"] = len(live_tov_poss) / off_possessions

        # Dead-ball turnovers: turnovers without a steal in that possession
        tov_plays = game_plays[
            (game_plays["playType"].str.contains("Turnover|Lost Ball", case=False, na=False)) &
            (game_plays["possession_id"].isin(off_poss_ids))
        ]
        tov_poss = set(tov_plays["possession_id"].unique())
        dead_tov_poss = tov_poss - live_tov_poss
        stats["dead_ball_tov_rate"] = len(dead_tov_poss) / off_possessions
    else:
        stats["live_ball_tov_rate"] = np.nan
        stats["dead_ball_tov_rate"] = np.nan

    # Defensive stats
    def_possessions = def_plays["possession_id"].nunique()
    if def_possessions > 0:
        # Steals forced by our team (steals in possessions where we were defending)
        game_plays = all_game_plays[all_game_plays["gameId"] == game_id]
        def_poss_ids = set(def_plays["possession_id"].unique())
        our_steals = game_plays[
            (game_plays["playType"] == "Steal") &
            (game_plays["possession_id"].isin(def_poss_ids))
        ]
        stats["steal_rate_defense"] = len(our_steals) / def_possessions
    else:
        stats["steal_rate_defense"] = np.nan

    # Transition rate: possessions with very short duration (< 7 seconds)
    if off_possessions > 0:
        poss_durations = off_plays.groupby("possession_id").agg(
            max_sec=("secondsRemaining", "max"),
            min_sec=("secondsRemaining", "min"),
        )
        poss_durations["duration"] = poss_durations["max_sec"] - poss_durations["min_sec"]
        transition = poss_durations[poss_durations["duration"] < 7]
        stats["transition_rate"] = len(transition) / off_possessions

        # Transition scoring efficiency: pts scored on transition possessions / count
        n_transition = len(transition)
        if n_transition > 0:
            trans_poss_ids = set(transition.index)
            trans_plays = off_plays[off_plays["possession_id"].isin(trans_poss_ids)]
            trans_pts = trans_plays[trans_plays["scoringPlay"] == True]["scoreValue"].sum()
            stats["transition_scoring_efficiency"] = trans_pts / n_transition
        else:
            stats["transition_scoring_efficiency"] = np.nan

        # Halfcourt (non-transition) scoring efficiency
        n_halfcourt = off_possessions - n_transition
        if n_halfcourt > 0:
            halfcourt_poss_ids = set(poss_durations[poss_durations["duration"] >= 7].index)
            hc_plays = off_plays[off_plays["possession_id"].isin(halfcourt_poss_ids)]
            hc_pts = hc_plays[hc_plays["scoringPlay"] == True]["scoreValue"].sum()
            stats["halfcourt_scoring_efficiency"] = hc_pts / n_halfcourt
        else:
            stats["halfcourt_scoring_efficiency"] = np.nan
    else:
        stats["transition_rate"] = np.nan
        stats["transition_scoring_efficiency"] = np.nan
        stats["halfcourt_scoring_efficiency"] = np.nan

    # Defensive halfcourt/transition split (opponent's possessions against us)
    def_possessions_count = def_plays["possession_id"].nunique()
    if def_possessions_count > 0:
        def_poss_durations = def_plays.groupby("possession_id").agg(
            max_sec=("secondsRemaining", "max"),
            min_sec=("secondsRemaining", "min"),
        )
        def_poss_durations["duration"] = (
            def_poss_durations["max_sec"] - def_poss_durations["min_sec"]
        )
        def_transition = def_poss_durations[def_poss_durations["duration"] < 7]
        n_def_trans = len(def_transition)

        if n_def_trans > 0:
            dt_poss_ids = set(def_transition.index)
            dt_plays = def_plays[def_plays["possession_id"].isin(dt_poss_ids)]
            dt_pts = dt_plays[dt_plays["scoringPlay"] == True]["scoreValue"].sum()
            stats["def_transition_scoring_efficiency"] = dt_pts / n_def_trans
        else:
            stats["def_transition_scoring_efficiency"] = np.nan

        n_def_hc = def_possessions_count - n_def_trans
        if n_def_hc > 0:
            dhc_poss_ids = set(
                def_poss_durations[def_poss_durations["duration"] >= 7].index
            )
            dhc_plays = def_plays[def_plays["possession_id"].isin(dhc_poss_ids)]
            dhc_pts = dhc_plays[dhc_plays["scoringPlay"] == True]["scoreValue"].sum()
            stats["def_halfcourt_scoring_efficiency"] = dhc_pts / n_def_hc
        else:
            stats["def_halfcourt_scoring_efficiency"] = np.nan
    else:
        stats["def_transition_scoring_efficiency"] = np.nan
        stats["def_halfcourt_scoring_efficiency"] = np.nan

    return stats


# ── Group D: Possession-level tempo ──────────────────────────────

def _compute_tempo_features(off_plays: pd.DataFrame) -> dict:
    """Compute possession-level tempo features."""
    stats = {}

    poss_data = off_plays.groupby("possession_id").agg(
        max_sec=("secondsRemaining", "max"),
        min_sec=("secondsRemaining", "min"),
        has_shot=("shootingPlay", "any"),
    )
    poss_data["duration"] = poss_data["max_sec"] - poss_data["min_sec"]
    n_poss = len(poss_data)

    if n_poss > 0:
        stats["avg_possession_length"] = poss_data["duration"].mean()

        # Early clock shots: shot in first 10 seconds (duration < 10)
        shot_poss = poss_data[poss_data["has_shot"]]
        if len(shot_poss) > 0:
            early = shot_poss[shot_poss["duration"] < 10]
            stats["early_clock_shot_rate"] = len(early) / len(shot_poss)

            # Shot clock pressure: shot comes late (duration > 25 seconds)
            late = shot_poss[shot_poss["duration"] > 25]
            stats["shot_clock_pressure_rate"] = len(late) / len(shot_poss)
        else:
            stats["early_clock_shot_rate"] = np.nan
            stats["shot_clock_pressure_rate"] = np.nan
    else:
        stats["avg_possession_length"] = np.nan
        stats["early_clock_shot_rate"] = np.nan
        stats["shot_clock_pressure_rate"] = np.nan

    return stats


# ── Group G: Putback / second-chance ─────────────────────────────

def _compute_putback_features(off_plays: pd.DataFrame) -> dict:
    """Compute putback and second-chance features."""
    stats = {}

    # Find offensive rebounds
    off_rebs = off_plays[off_plays["playType"] == "Offensive Rebound"]
    total_oreb = len(off_rebs)

    if total_oreb > 0:
        # For each OREB, check if a scoring play follows in the same possession
        oreb_poss = off_rebs["possession_id"].unique()
        putback_count = 0
        second_chance_pts = 0

        for poss_id in oreb_poss:
            poss_plays = off_plays[off_plays["possession_id"] == poss_id]
            oreb_plays = poss_plays[poss_plays["playType"] == "Offensive Rebound"]
            if oreb_plays.empty:
                continue

            # Find scoring plays after the offensive rebound in this possession
            scoring_after = poss_plays[poss_plays["scoringPlay"] == True]
            if len(scoring_after) > 0:
                putback_count += 1
                second_chance_pts += scoring_after["scoreValue"].sum()

        stats["putback_rate"] = putback_count / total_oreb
        stats["second_chance_pts_per_oreb"] = second_chance_pts / total_oreb
    else:
        stats["putback_rate"] = np.nan
        stats["second_chance_pts_per_oreb"] = np.nan

    return stats


# ── Group H: Clutch performance splits ───────────────────────────

def _compute_clutch_stats(team_df: pd.DataFrame, team_id: int) -> dict:
    """Compute clutch performance stats.

    Clutch: margin <= 8, last 5 minutes of 2nd half or any OT period.
    """
    stats = {}

    margin = (team_df["homeScore"] - team_df["awayScore"]).abs()

    # Clutch filter: period >= 2, secondsRemaining <= 300 (5 min), margin <= 8
    # For OT (period >= 3): include all plays with margin <= 8
    clutch_mask = (
        ((team_df["period"] == 2) & (team_df["secondsRemaining"] <= 300) & (margin <= 8)) |
        ((team_df["period"] >= 3) & (margin <= 8))
    )
    clutch = team_df[clutch_mask]
    non_clutch = team_df[~clutch_mask]

    # Clutch offensive plays
    clutch_off = clutch[clutch["offense_team_id"] == team_id]
    clutch_def = clutch[clutch["defense_team_id"] == team_id]

    clutch_off_poss = clutch_off["possession_id"].nunique()
    clutch_def_poss = clutch_def["possession_id"].nunique()

    if clutch_off_poss >= 3:  # Need minimum possessions for meaningful stats
        clutch_pts = clutch_off[clutch_off["scoringPlay"] == True]["scoreValue"].sum()
        stats["clutch_off_efficiency"] = clutch_pts / clutch_off_poss

        # Clutch eFG%
        clutch_shots = clutch_off[
            (clutch_off["shootingPlay"] == True) & (clutch_off["shot_zone"] != "free_throw")
        ]
        if len(clutch_shots) > 0:
            made = clutch_shots[clutch_shots["shot_made_bool"] == True]
            three_made = made[made["shot_zone"] == "three_pointer"]
            stats["clutch_eff_fg_pct"] = (len(made) + 0.5 * len(three_made)) / len(clutch_shots)
        else:
            stats["clutch_eff_fg_pct"] = np.nan

        # Clutch TOV rate
        clutch_tovs = clutch_off[clutch_off["playType"].str.contains("Turnover|Lost Ball", case=False, na=False)]
        stats["clutch_tov_rate"] = len(clutch_tovs) / clutch_off_poss

        # Clutch FT%
        clutch_fts = clutch_off[clutch_off["shot_zone"] == "free_throw"]
        if len(clutch_fts) > 0:
            clutch_ft_made = clutch_fts[clutch_fts["scoringPlay"] == True]
            stats["clutch_ft_pct"] = len(clutch_ft_made) / len(clutch_fts)
        else:
            stats["clutch_ft_pct"] = np.nan
    else:
        stats["clutch_off_efficiency"] = np.nan
        stats["clutch_eff_fg_pct"] = np.nan
        stats["clutch_tov_rate"] = np.nan
        stats["clutch_ft_pct"] = np.nan

    if clutch_def_poss >= 3:
        clutch_def_pts = clutch_def[clutch_def["scoringPlay"] == True]["scoreValue"].sum()
        stats["clutch_def_efficiency"] = clutch_def_pts / clutch_def_poss
    else:
        stats["clutch_def_efficiency"] = np.nan

    # Non-clutch to clutch delta
    non_clutch_off = non_clutch[non_clutch["offense_team_id"] == team_id]
    non_clutch_off_poss = non_clutch_off["possession_id"].nunique()
    if clutch_off_poss >= 3 and non_clutch_off_poss >= 5:
        nc_pts = non_clutch_off[non_clutch_off["scoringPlay"] == True]["scoreValue"].sum()
        nc_eff = nc_pts / non_clutch_off_poss
        stats["non_clutch_to_clutch_delta"] = stats["clutch_off_efficiency"] - nc_eff
    else:
        stats["non_clutch_to_clutch_delta"] = np.nan

    return stats


# ── Group I: Scoring drought frequency ───────────────────────────

def _compute_drought_features(
    off_plays: pd.DataFrame,
    def_plays: pd.DataFrame,
    all_game_plays: pd.DataFrame,
    game_id: int,
    team_id: int,
) -> dict:
    """Compute scoring drought features for both offense and defense."""
    stats = {}

    # Offensive droughts: consecutive possessions where team doesn't score
    off_drought = _compute_droughts_for_side(off_plays)
    stats["off_avg_drought_length"] = off_drought["avg_length"]
    stats["off_max_drought_length"] = off_drought["max_length"]
    stats["off_drought_frequency"] = off_drought["frequency"]

    # Defensive droughts (opponent fails to score = good for us)
    # Get opponent's offensive plays from this game
    game_plays = all_game_plays[all_game_plays["gameId"] == game_id]
    opp_off_plays = game_plays[
        (game_plays["offense_team_id"] != team_id) &
        (game_plays["offense_team_id"].notna())
    ]
    def_drought = _compute_droughts_for_side(opp_off_plays)
    stats["def_avg_drought_length"] = def_drought["avg_length"]
    stats["def_max_drought_length"] = def_drought["max_length"]
    stats["def_drought_frequency"] = def_drought["frequency"]

    return stats


def _compute_droughts_for_side(plays: pd.DataFrame) -> dict:
    """Compute drought stats for a set of offensive plays."""
    if plays.empty:
        return {"avg_length": np.nan, "max_length": np.nan, "frequency": np.nan}

    # Get unique possessions sorted by ID
    poss_data = plays.groupby("possession_id").agg(
        scored=("scoringPlay", "any"),
    ).sort_index()

    if poss_data.empty:
        return {"avg_length": np.nan, "max_length": np.nan, "frequency": np.nan}

    # Walk through possessions tracking droughts
    droughts = []
    current_drought = 0
    for scored in poss_data["scored"]:
        if not scored:
            current_drought += 1
        else:
            if current_drought > 0:
                droughts.append(current_drought)
            current_drought = 0
    if current_drought > 0:
        droughts.append(current_drought)

    n_poss = len(poss_data)
    if droughts:
        return {
            "avg_length": np.mean(droughts),
            "max_length": max(droughts),
            "frequency": sum(1 for d in droughts if d >= 4) / max(n_poss / 10, 1),  # per ~10 possessions
        }
    return {"avg_length": 0.0, "max_length": 0.0, "frequency": 0.0}


# ── Group J: Half / period splits ────────────────────────────────

def _compute_half_splits(team_df: pd.DataFrame, team_id: int) -> dict:
    """Compute first-half vs second-half efficiency splits."""
    stats = {}

    for period, label in [(1, "first_half"), (2, "second_half")]:
        period_plays = team_df[team_df["period"] == period]
        off_plays = period_plays[period_plays["offense_team_id"] == team_id]
        def_plays = period_plays[period_plays["defense_team_id"] == team_id]

        off_poss = off_plays["possession_id"].nunique()
        def_poss = def_plays["possession_id"].nunique()

        if off_poss > 0:
            pts = off_plays[off_plays["scoringPlay"] == True]["scoreValue"].sum()
            stats[f"{label}_off_efficiency"] = pts / off_poss
        else:
            stats[f"{label}_off_efficiency"] = np.nan

        if def_poss > 0:
            def_pts = def_plays[def_plays["scoringPlay"] == True]["scoreValue"].sum()
            stats[f"{label}_def_efficiency"] = def_pts / def_poss
        else:
            stats[f"{label}_def_efficiency"] = np.nan

    # Deltas (second half - first half)
    if pd.notna(stats.get("first_half_off_efficiency")) and pd.notna(stats.get("second_half_off_efficiency")):
        stats["half_adjustment_delta"] = (
            stats["second_half_off_efficiency"] - stats["first_half_off_efficiency"]
        )
    else:
        stats["half_adjustment_delta"] = np.nan

    if pd.notna(stats.get("first_half_def_efficiency")) and pd.notna(stats.get("second_half_def_efficiency")):
        stats["second_half_def_delta"] = (
            stats["second_half_def_efficiency"] - stats["first_half_def_efficiency"]
        )
    else:
        stats["second_half_def_delta"] = np.nan

    return stats


# ── Group M: Rotation depth / scoring concentration ──────────────

def _compute_rotation_depth(off_plays: pd.DataFrame) -> dict:
    """Compute scoring concentration and rotation depth stats."""
    stats = {}

    # Get scoring plays with valid shooter IDs
    scoring = off_plays[
        (off_plays["scoringPlay"] == True) &
        (off_plays["shot_shooter_id"].notna())
    ]

    if scoring.empty:
        stats["unique_scorers"] = 0
        stats["scoring_hhi"] = np.nan
        stats["top2_scorer_pct"] = np.nan
        return stats

    # Points per player
    player_pts = scoring.groupby("shot_shooter_id")["scoreValue"].sum()
    total_pts = player_pts.sum()

    stats["unique_scorers"] = len(player_pts)

    if total_pts > 0:
        # HHI: sum of squared shares
        shares = player_pts / total_pts
        stats["scoring_hhi"] = (shares ** 2).sum()

        # Top 2 scorer percentage
        top2 = player_pts.nlargest(2).sum()
        stats["top2_scorer_pct"] = top2 / total_pts
    else:
        stats["scoring_hhi"] = np.nan
        stats["top2_scorer_pct"] = np.nan

    return stats


# ── Group N: Pressure free throws ────────────────────────────────

def _compute_pressure_ft(team_df: pd.DataFrame, team_id: int) -> dict:
    """Compute free throw shooting under pressure.

    Pressure: margin <= 5, period >= 2, secondsRemaining <= 300 (last 5 min 2H + OT).
    """
    stats = {}

    # All FT attempts by this team
    ft_plays = team_df[
        (team_df["offense_team_id"] == team_id) &
        (team_df["shot_zone"] == "free_throw")
    ]

    if ft_plays.empty:
        stats["pressure_ft_pct"] = np.nan
        stats["non_pressure_ft_pct"] = np.nan
        stats["ft_pressure_delta"] = np.nan
        return stats

    margin = (team_df["homeScore"] - team_df["awayScore"]).abs()
    # Build pressure mask for ft_plays indices
    ft_margin = margin.loc[ft_plays.index]
    ft_period = ft_plays["period"]
    ft_seconds = ft_plays["secondsRemaining"]

    pressure_mask = (
        ((ft_period == 2) & (ft_seconds <= 300) & (ft_margin <= 5)) |
        ((ft_period >= 3) & (ft_margin <= 5))
    )

    pressure_fts = ft_plays[pressure_mask]
    non_pressure_fts = ft_plays[~pressure_mask]

    if len(pressure_fts) >= 2:
        made = pressure_fts[pressure_fts["scoringPlay"] == True]
        stats["pressure_ft_pct"] = len(made) / len(pressure_fts)
    else:
        stats["pressure_ft_pct"] = np.nan

    if len(non_pressure_fts) >= 2:
        made = non_pressure_fts[non_pressure_fts["scoringPlay"] == True]
        stats["non_pressure_ft_pct"] = len(made) / len(non_pressure_fts)
    else:
        stats["non_pressure_ft_pct"] = np.nan

    if pd.notna(stats["pressure_ft_pct"]) and pd.notna(stats["non_pressure_ft_pct"]):
        stats["ft_pressure_delta"] = stats["pressure_ft_pct"] - stats["non_pressure_ft_pct"]
    else:
        stats["ft_pressure_delta"] = np.nan

    return stats


# ── Composite features ────────────────────────────────────────────

def _compute_composites(row: dict) -> dict:
    """Compute composite features derived from other per-game stats.

    Args:
        row: Dict containing all previously computed per-game stats.

    Returns:
        Dict with composite feature values.
    """
    stats = {}

    # expected_pts_per_shot = rim_rate * rim_fg_pct * 2 + mid_rate * mid_fg_pct * 2
    #                        + three_rate * three_pt_fg_pct * 3
    rim_rate = row.get("rim_rate")
    rim_fg = row.get("rim_fg_pct")
    mid_rate = row.get("mid_range_rate")
    mid_fg = row.get("mid_range_fg_pct")
    three_fg = row.get("three_pt_fg_pct")
    # three_rate is 1 - rim_rate - mid_rate (share of FGA that are 3s)
    if all(pd.notna(v) for v in [rim_rate, rim_fg, mid_rate, mid_fg, three_fg]):
        three_rate = max(1.0 - rim_rate - mid_rate, 0.0)
        stats["expected_pts_per_shot"] = (
            rim_rate * rim_fg * 2
            + mid_rate * mid_fg * 2
            + three_rate * three_fg * 3
        )
    else:
        stats["expected_pts_per_shot"] = np.nan

    # transition_value = steal_rate_defense * transition_scoring_efficiency
    steal_rate = row.get("steal_rate_defense")
    trans_eff = row.get("transition_scoring_efficiency")
    if pd.notna(steal_rate) and pd.notna(trans_eff):
        stats["transition_value"] = steal_rate * trans_eff
    else:
        stats["transition_value"] = np.nan

    return stats


# ── Column names exported for use by other modules ───────────────

# All stat columns produced by this module (per-team)
SHOT_QUALITY_COLS = [
    "rim_rate", "mid_range_rate", "three_pt_rate_corner", "three_pt_rate_above_break",
    "rim_fg_pct", "mid_range_fg_pct", "assisted_fg_pct", "unassisted_fg_pct",
    # Defensive versions
    "def_rim_rate", "def_mid_range_rate", "def_rim_fg_pct", "def_mid_range_fg_pct",
]

TURNOVER_DECOMP_COLS = [
    "live_ball_tov_rate", "dead_ball_tov_rate", "steal_rate_defense", "transition_rate",
    "transition_scoring_efficiency", "halfcourt_scoring_efficiency",
    "def_transition_scoring_efficiency", "def_halfcourt_scoring_efficiency",
]

TEMPO_COLS = [
    "avg_possession_length", "early_clock_shot_rate", "shot_clock_pressure_rate",
]

PUTBACK_COLS = [
    "putback_rate", "second_chance_pts_per_oreb",
]

CLUTCH_COLS = [
    "clutch_off_efficiency", "clutch_def_efficiency", "clutch_eff_fg_pct",
    "clutch_tov_rate", "clutch_ft_pct", "non_clutch_to_clutch_delta",
]

DROUGHT_COLS = [
    "off_avg_drought_length", "off_max_drought_length", "off_drought_frequency",
    "def_avg_drought_length", "def_max_drought_length", "def_drought_frequency",
]

HALF_SPLIT_COLS = [
    "first_half_off_efficiency", "second_half_off_efficiency",
    "first_half_def_efficiency", "second_half_def_efficiency",
    "half_adjustment_delta", "second_half_def_delta",
]

ROTATION_DEPTH_COLS = [
    "unique_scorers", "scoring_hhi", "top2_scorer_pct",
]

PRESSURE_FT_COLS = [
    "pressure_ft_pct", "non_pressure_ft_pct", "ft_pressure_delta",
]

# Zone counts: intermediate data for luck/composite features (NOT for rolling)
ZONE_COUNT_COLS = [
    "rim_fga", "rim_fgm", "mid_fga", "mid_fgm",
    "three_fga", "three_fgm", "three_pt_fg_pct", "two_pt_fg_pct",
]

COMPOSITE_COLS = [
    "transition_scoring_efficiency", "expected_pts_per_shot", "transition_value",
]

ALL_ADVANCED_STAT_COLS = (
    SHOT_QUALITY_COLS + TURNOVER_DECOMP_COLS + TEMPO_COLS +
    PUTBACK_COLS + CLUTCH_COLS + DROUGHT_COLS + HALF_SPLIT_COLS +
    ROTATION_DEPTH_COLS + PRESSURE_FT_COLS + ZONE_COUNT_COLS + COMPOSITE_COLS
)
