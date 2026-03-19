export type PredictionRow = Record<string, unknown>;

export type DataFile = {
  date: string;
  filename: string;
};

const SITE_ML_SIGMA_CAP = 14;
type MlOddsMode = "cap14_mu_sigma" | "meta_small_v1";
const SITE_ML_ODDS_MODE: MlOddsMode = "meta_small_v1";
const META_SMALL_V1 = {
  intercept: 0.020175630994879585,
  coefficients: {
    mu: 0.15059080225978677,
    sigma_cap14: -0.008192640820978753,
    z14: 0.08477442454897068,
    post_dec15: 0.0740213341463982,
    abs_mu: 0.004037300255673246,
  },
} as const;
const NEUTRAL_BETA_BLEND_V1 = {
  alpha: 0.6,
  intercept: -0.3232338741070838,
  coefficients: {
    log_p: 1.0790706225223776,
    log1m_p: -1.6575437915942683,
  },
} as const;

function toFiniteNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    return Number.isNaN(parsed) ? null : parsed;
  }
  return null;
}

function erfApprox(x: number): number {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y =
    1 -
    (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t) *
      Math.exp(-ax * ax);
  return sign * y;
}

function normalCdf(x: number): number {
  return 0.5 * (1 + erfApprox(x / Math.sqrt(2)));
}

function logistic(x: number): number {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}

function toBoolean(value: unknown): boolean {
  if (typeof value === "boolean") return value;
  if (typeof value === "number") return value !== 0;
  if (typeof value === "string") {
    const trimmed = value.trim().toLowerCase();
    return trimmed === "true" || trimmed === "1";
  }
  return false;
}

function isPostDec15(value: unknown): boolean {
  if (typeof value !== "string" || !value) return false;
  const dt = new Date(value);
  if (Number.isNaN(dt.getTime())) return false;
  const et = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York",
    year: "numeric",
    month: "numeric",
    day: "numeric",
  }).formatToParts(dt);
  const month = Number(et.find((p) => p.type === "month")?.value ?? "0");
  const day = Number(et.find((p) => p.type === "day")?.value ?? "0");
  if (month === 11) return false;
  if (month === 12) return day >= 15;
  return month >= 1 && month <= 3;
}

export function getModelMuHome(row: PredictionRow): number | null {
  return toFiniteNumber(row.model_mu_home);
}

export function getPredSigma(row: PredictionRow): number | null {
  return toFiniteNumber(row.pred_sigma);
}

export function getCap14MuSigmaHomeWinProbFromValues(
  mu: number | null,
  sigma: number | null,
): number | null {
  if (mu === null || sigma === null) return null;
  const sigmaSafe = Math.max(Math.min(sigma, SITE_ML_SIGMA_CAP), 0.5);
  return Math.min(Math.max(normalCdf(mu / sigmaSafe), 1e-6), 1 - 1e-6);
}

export function getMuSigmaHomeWinProb(row: PredictionRow): number | null {
  return getCap14MuSigmaHomeWinProbFromValues(getModelMuHome(row), getPredSigma(row));
}

export function getSiteHomeWinProbFromValues(
  mu: number | null,
  sigma: number | null,
  startTime: string | null,
  neutralSite?: boolean | null,
): number | null {
  const baseline = getCap14MuSigmaHomeWinProbFromValues(mu, sigma);
  if (baseline === null || SITE_ML_ODDS_MODE === "cap14_mu_sigma") return baseline;

  const sigmaCap14 = Math.max(Math.min(sigma as number, SITE_ML_SIGMA_CAP), 0.5);
  const absMu = Math.abs(mu as number);
  const z14 = (mu as number) / sigmaCap14;
  const postDec15 = isPostDec15(startTime) ? 1 : 0;
  const score =
    META_SMALL_V1.intercept +
    META_SMALL_V1.coefficients.mu * (mu as number) +
    META_SMALL_V1.coefficients.sigma_cap14 * sigmaCap14 +
    META_SMALL_V1.coefficients.z14 * z14 +
    META_SMALL_V1.coefficients.post_dec15 * postDec15 +
    META_SMALL_V1.coefficients.abs_mu * absMu;
  const currentProb = Math.min(Math.max(logistic(score), 1e-6), 1 - 1e-6);
  if (!neutralSite) return currentProb;
  const betaScore =
    NEUTRAL_BETA_BLEND_V1.intercept +
    NEUTRAL_BETA_BLEND_V1.coefficients.log_p * Math.log(currentProb) +
    NEUTRAL_BETA_BLEND_V1.coefficients.log1m_p * Math.log1p(-currentProb);
  const betaProb = Math.min(Math.max(logistic(betaScore), 1e-6), 1 - 1e-6);
  return (
    (1 - NEUTRAL_BETA_BLEND_V1.alpha) * currentProb +
    NEUTRAL_BETA_BLEND_V1.alpha * betaProb
  );
}

export function getSiteHomeWinProb(row: PredictionRow): number | null {
  const startTime =
    typeof row.start_time === "string"
      ? row.start_time
      : typeof row.startDate === "string"
        ? row.startDate
        : null;
  const neutralSite = toBoolean(row.neutral_site ?? row.neutralSite);
  return getSiteHomeWinProbFromValues(getModelMuHome(row), getPredSigma(row), startTime, neutralSite);
}

export function formatAmericanOddsFromProb(prob: number | null): string | null {
  if (prob === null || prob <= 0 || prob >= 1) return null;
  const odds =
    prob >= 0.5 ? -100 * prob / (1 - prob) : 100 * (1 - prob) / prob;
  const rounded = Math.round(odds);
  return rounded > 0 ? `+${rounded}` : `${rounded}`;
}

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
  return name
    .replace(/\bDuke\b/g, "dook💩")
    .replace(/\bduke\b/g, "dook💩")
    .replace(/\bGeorgia Tech\b/g, "Georgia Tech👑")
    .replace(/\bUCF\b/g, "UCF👸")
    .replace(/\bUSF\b/g, "USF🃏")
    .replace(/\bSouth Florida\b/g, "South Florida🃏");
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
