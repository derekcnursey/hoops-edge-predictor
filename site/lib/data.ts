export type PredictionRow = Record<string, unknown>;

export type DataFile = {
  date: string;
  filename: string;
};

export function normalizeRows(payload: unknown): PredictionRow[] {
  if (!payload) {
    return [];
  }
  if (Array.isArray(payload)) {
    return payload as PredictionRow[];
  }
  if (typeof payload === "object") {
    const record = payload as Record<string, unknown>;
    const candidates = [
      record.predictions,
      record.games,
      record.rows,
      record.data
    ];
    for (const candidate of candidates) {
      if (Array.isArray(candidate)) {
        return candidate as PredictionRow[];
      }
    }
  }
  return [];
}

export function pickColumns(rows: PredictionRow[]): string[] {
  const preferred = [
    "away_team",
    "home_team",
    "pick_prob_edge",
    "edge_home_points",
    "model_mu_home",
    "market_spread_home",
    "pred_sigma",
    "neutral_site"
  ];
  return preferred;
}

export function getEdgeValue(row: PredictionRow): number {
  const candidates = [
    row.edge_home_points,
    row.edge_points,
    row.edge_home,
    row.edge,
    row.pred_edge,
    row.predicted_edge,
    row.edgeHome
  ];
  for (const value of candidates) {
    if (typeof value === "number") {
      return value;
    }
    if (typeof value === "string" && value.trim() !== "") {
      const parsed = Number(value);
      if (!Number.isNaN(parsed)) {
        return parsed;
      }
    }
  }
  return 0;
}

export function getPickProbEdge(row: PredictionRow): number {
  const candidates = [row.pick_prob_edge, row.pickProbEdge];
  for (const value of candidates) {
    if (typeof value === "number") {
      return value;
    }
    if (typeof value === "string" && value.trim() !== "") {
      const parsed = Number(value);
      if (!Number.isNaN(parsed)) {
        return parsed;
      }
    }
  }
  return 0;
}

export function getTeams(row: PredictionRow): { home?: string; away?: string } {
  const home =
    (row.home_team as string | undefined) ||
    (row.home_team_name as string | undefined) ||
    (row.team_home as string | undefined) ||
    (row.home as string | undefined) ||
    (row.teamHome as string | undefined);
  const away =
    (row.away_team as string | undefined) ||
    (row.away_team_name as string | undefined) ||
    (row.team_away as string | undefined) ||
    (row.away as string | undefined) ||
    (row.teamAway as string | undefined);
  return { home, away };
}

export function normalizeTeam(name?: string): string {
  return (name || "").trim().toLowerCase();
}

/** Display-name transform applied to all team names on the site. */
export function displayTeam(name: string): string {
  return name.replace(/\bDuke\b/g, "dook 💩").replace(/\bduke\b/g, "dook 💩");
}

export function getActualMargin(row: PredictionRow): number | null {
  const candidates = [
    row.margin_home,
    row.actual_margin,
    row.result_home,
    row.marginHome
  ];
  for (const value of candidates) {
    if (typeof value === "number") {
      return value;
    }
    if (typeof value === "string" && value.trim() !== "") {
      const parsed = Number(value);
      if (!Number.isNaN(parsed)) {
        return parsed;
      }
    }
  }

  const homeScore =
    row.score_home ?? row.home_score ?? row.scoreHome ?? row.homeScore;
  const awayScore =
    row.score_away ?? row.away_score ?? row.scoreAway ?? row.awayScore;
  if (typeof homeScore === "number" && typeof awayScore === "number") {
    return homeScore - awayScore;
  }
  if (typeof homeScore === "string" && typeof awayScore === "string") {
    const home = Number(homeScore);
    const away = Number(awayScore);
    if (!Number.isNaN(home) && !Number.isNaN(away)) {
      return home - away;
    }
  }
  return null;
}

export function buildResultLookup(rows: PredictionRow[]): Map<string, PredictionRow> {
  const map = new Map<string, PredictionRow>();
  for (const row of rows) {
    const teams = getTeams(row);
    if (teams.home && teams.away) {
      const key = `${normalizeTeam(teams.home)}__${normalizeTeam(teams.away)}`;
      map.set(key, row);
    }
  }
  return map;
}
