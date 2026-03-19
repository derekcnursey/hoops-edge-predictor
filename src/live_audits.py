"""Operational audits for live slates and daily prediction runs."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from . import config

_ET = "America/New_York"
_HRB_PROVIDER = "Hard Rock Bet"


@dataclass
class AuditReport:
    label: str
    info: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _format_matchup(row: pd.Series) -> str:
    away = row.get("awayTeam") or row.get("awayTeamId") or "?"
    home = row.get("homeTeam") or row.get("homeTeamId") or "?"
    return f"{away} @ {home}"


def audit_live_feature_drift(
    features_df: pd.DataFrame,
    scaler,
    feature_order: Iterable[str],
    critical_features: Iterable[str],
) -> AuditReport:
    report = AuditReport(label="Live feature drift")
    if features_df.empty:
        report.info.append("no live feature rows")
        return report

    feature_names = [col for col in feature_order if col in features_df.columns]
    if not feature_names:
        report.info.append("no modeled feature columns present on live slate")
        return report

    feature_frame = features_df[feature_names].apply(pd.to_numeric, errors="coerce")
    n_rows = len(feature_frame)
    training_means = np.asarray(getattr(scaler, "mean_", np.zeros(len(feature_names))), dtype=float)
    training_scales = np.asarray(getattr(scaler, "scale_", np.ones(len(feature_names))), dtype=float)
    safe_scales = np.where(np.isfinite(training_scales) & (np.abs(training_scales) > 1e-6), training_scales, 1.0)

    drift_rows: list[tuple[str, float, float]] = []
    fill_rows: list[tuple[str, float]] = []
    critical_fill_rows: list[tuple[str, float]] = []
    for idx, col in enumerate(feature_names):
        series = feature_frame[col]
        fill_rate = float(series.isna().mean())
        if fill_rate >= config.LIVE_FEATURE_FILL_WARN_RATE:
            fill_rows.append((col, fill_rate))
        if col in critical_features and fill_rate > 0.0:
            critical_fill_rows.append((col, fill_rate))
        nonnull = series.dropna()
        if nonnull.empty:
            continue
        live_mean = float(nonnull.mean())
        z_drift = abs(live_mean - float(training_means[idx])) / float(safe_scales[idx])
        drift_rows.append((col, z_drift, fill_rate))

    drift_rows.sort(key=lambda row: (-row[1], row[0]))
    top_drift = drift_rows[: config.LIVE_FEATURE_DRIFT_TOP_N]
    if top_drift:
        report.info.append(
            "worst mean-z drift: "
            + ", ".join(f"{col}={z:.1f}" for col, z, _fill in top_drift)
        )

    severe_drift = [
        (col, z)
        for col, z, fill_rate in drift_rows
        if z >= config.LIVE_FEATURE_DRIFT_WARN_Z and fill_rate < 1.0
    ]
    if severe_drift:
        report.warnings.append(
            f"{len(severe_drift)} feature(s) exceed mean-z drift {config.LIVE_FEATURE_DRIFT_WARN_Z:.1f}: "
            + ", ".join(f"{col}={z:.1f}" for col, z in severe_drift[: config.LIVE_FEATURE_DRIFT_TOP_N])
        )

    if fill_rows:
        report.warnings.append(
            f"{len(fill_rows)} feature(s) require scaler-mean fill at >= {config.LIVE_FEATURE_FILL_WARN_RATE:.0%}: "
            + ", ".join(f"{col}={rate:.0%}" for col, rate in fill_rows[: config.LIVE_FEATURE_DRIFT_TOP_N])
        )
    if critical_fill_rows:
        report.warnings.append(
            "critical feature fill present: "
            + ", ".join(f"{col}={rate:.0%}" for col, rate in critical_fill_rows)
        )

    report.info.insert(0, f"{n_rows} live row(s), {len(feature_names)} modeled feature columns")
    return report


def audit_ratings_asof(
    games_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
) -> AuditReport:
    report = AuditReport(label="Ratings as-of")
    if games_df.empty:
        report.info.append("no slate games to audit")
        return report
    if ratings_df.empty:
        report.errors.append("no gold ratings rows available for live as-of lookup")
        return report
    needed_cols = {"homeTeamId", "awayTeamId", "startDate"}
    if not needed_cols.issubset(games_df.columns):
        missing = sorted(needed_cols - set(games_df.columns))
        report.warnings.append(
            f"ratings as-of audit skipped missing slate columns: {', '.join(missing)}"
        )
        return report

    ratings = ratings_df.copy()
    ratings["rating_date"] = pd.to_datetime(ratings["rating_date"], errors="coerce")
    ratings = ratings.dropna(subset=["teamId", "rating_date"]).sort_values(["teamId", "rating_date"])
    lookup = {int(team_id): group for team_id, group in ratings.groupby("teamId", sort=False)}

    stale_rows: list[str] = []
    missing_rows: list[str] = []
    ages: list[int] = []
    for _, game in games_df.iterrows():
        start = pd.to_datetime(game.get("startDate"), utc=True, errors="coerce")
        if pd.isna(start):
            continue
        cutoff = start.tz_convert(_ET).tz_localize(None).normalize() - pd.Timedelta(days=1)
        for side in ["home", "away"]:
            team_id = game.get(f"{side}TeamId")
            if pd.isna(team_id):
                missing_rows.append(f"{_format_matchup(game)} missing {side}TeamId")
                continue
            team_ratings = lookup.get(int(team_id))
            if team_ratings is None:
                missing_rows.append(f"{_format_matchup(game)} missing {side} rating history")
                continue
            eligible = team_ratings[team_ratings["rating_date"] <= cutoff]
            if eligible.empty:
                missing_rows.append(
                    f"{_format_matchup(game)} has no {side} rating on/before {cutoff.date().isoformat()}"
                )
                continue
            rating_date = pd.Timestamp(eligible.iloc[-1]["rating_date"]).normalize()
            age_days = int((cutoff - rating_date).days)
            ages.append(age_days)
            if age_days > config.RATINGS_ASOF_STALE_WARN_DAYS:
                team_name = game.get(f"{side}Team") or f"{side}TeamId={int(team_id)}"
                stale_rows.append(f"{team_name} age={age_days}d ({_format_matchup(game)})")

    if missing_rows:
        report.errors.extend(missing_rows[:10])
        if len(missing_rows) > 10:
            report.errors.append(f"... plus {len(missing_rows) - 10} more missing as-of rating issue(s)")
    if stale_rows:
        report.warnings.append(
            f"{len(stale_rows)} team lookup(s) used stale ratings older than {config.RATINGS_ASOF_STALE_WARN_DAYS} day(s): "
            + "; ".join(stale_rows[:5])
        )
    if ages:
        report.info.append(
            f"{len(ages)} team lookups resolved, median age={int(np.median(ages))}d, max age={max(ages)}d"
        )
    else:
        report.info.append("no valid team rating lookups resolved")
    return report


def audit_hrb_lines(
    slate_games: pd.DataFrame,
    all_lines: pd.DataFrame,
    preferred_lines: pd.DataFrame,
) -> AuditReport:
    report = AuditReport(label="Hard Rock Bet line sanity")
    if slate_games.empty:
        report.info.append("no slate games to audit")
        return report
    if all_lines.empty:
        report.warnings.append("no line rows available for Hard Rock Bet audit")
        return report

    slate_ids = set(pd.to_numeric(slate_games["gameId"], errors="coerce").dropna().astype(int).tolist())
    lines = all_lines.copy()
    lines = lines[lines["gameId"].isin(slate_ids)].copy()
    if lines.empty:
        report.warnings.append("no line rows matched the live slate")
        return report

    hrb = (
        lines[lines["provider"].fillna("") == _HRB_PROVIDER].copy()
        if "provider" in lines.columns
        else pd.DataFrame()
    )
    selected = (
        preferred_lines[preferred_lines["gameId"].isin(slate_ids)].copy()
        if not preferred_lines.empty
        else pd.DataFrame()
    )
    selected_hrb = (
        selected[selected["provider"].fillna("") == _HRB_PROVIDER].copy()
        if ("provider" in selected.columns and not selected.empty)
        else pd.DataFrame()
    )

    if hrb.empty:
        report.warnings.append("no Hard Rock Bet rows matched the live slate")
        return report

    dup_count = int(hrb.duplicated(subset=["gameId"], keep=False).sum())
    if dup_count > 0:
        report.errors.append(f"{dup_count} duplicate Hard Rock Bet row(s) detected on live slate")

    report.info.append(
        f"{len(hrb)}/{len(slate_games)} slate games have Hard Rock Bet rows; "
        f"{len(selected_hrb)}/{len(slate_games)} are selected as preferred"
    )

    sign_conflicts: list[str] = []
    spread_dislocations: list[str] = []
    total_range_rows: list[str] = []
    for _, row in hrb.iterrows():
        spread = pd.to_numeric(row.get("spread"), errors="coerce")
        home_ml = pd.to_numeric(row.get("homeMoneyline"), errors="coerce")
        away_ml = pd.to_numeric(row.get("awayMoneyline"), errors="coerce")
        total = pd.to_numeric(row.get("overUnder"), errors="coerce")
        if pd.notna(spread) and pd.notna(home_ml) and pd.notna(away_ml):
            if home_ml < away_ml and spread > 0:
                sign_conflicts.append(f"{_format_matchup(row)} spread {spread:+.1f} vs ML {int(home_ml)}/{int(away_ml)}")
            elif away_ml < home_ml and spread < 0:
                sign_conflicts.append(f"{_format_matchup(row)} spread {spread:+.1f} vs ML {int(home_ml)}/{int(away_ml)}")
        if pd.notna(total) and not (90.0 <= float(total) <= 220.0):
            total_range_rows.append(f"{_format_matchup(row)} total={float(total):.1f}")

        if pd.notna(spread):
            peers = lines[
                (lines["gameId"] == row["gameId"])
                & (lines["provider"].fillna("") != _HRB_PROVIDER)
            ].copy()
            if "spread" in peers.columns:
                peer_spreads = pd.to_numeric(peers["spread"], errors="coerce").dropna()
                if not peer_spreads.empty:
                    peer_median = float(peer_spreads.median())
                    if abs(float(spread) - peer_median) >= config.HRB_SPREAD_DISLOCATION_WARN:
                        spread_dislocations.append(
                            f"{_format_matchup(row)} HRB {float(spread):+.1f} vs peer median {peer_median:+.1f}"
                        )

    if sign_conflicts:
        report.errors.extend(sign_conflicts[:10])
        if len(sign_conflicts) > 10:
            report.errors.append(f"... plus {len(sign_conflicts) - 10} more Hard Rock Bet sign conflict(s)")
    if spread_dislocations:
        report.warnings.append(
            f"{len(spread_dislocations)} Hard Rock Bet spread(s) are >= {config.HRB_SPREAD_DISLOCATION_WARN:.1f} away from peer median: "
            + "; ".join(spread_dislocations[:5])
        )
    if total_range_rows:
        report.warnings.append(
            "Hard Rock Bet totals outside sanity band [90, 220]: "
            + "; ".join(total_range_rows[:5])
        )
    return report
