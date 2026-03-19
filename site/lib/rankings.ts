type RankingsTeamLike = {
  rank?: unknown;
  team?: unknown;
  adj_margin?: unknown;
};

function toNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function toRank(value: unknown): number {
  const parsed = toNumber(value);
  return parsed !== null ? parsed : Number.MAX_SAFE_INTEGER;
}

function toTeam(value: unknown): string {
  return typeof value === "string" ? value : String(value ?? "");
}

export function normalizeRankingsTeams<T extends RankingsTeamLike>(teams: T[]): T[] {
  return [...teams]
    .sort((a, b) => {
      const marginDiff = (toNumber(b.adj_margin) ?? Number.NEGATIVE_INFINITY) - (toNumber(a.adj_margin) ?? Number.NEGATIVE_INFINITY);
      if (marginDiff !== 0) return marginDiff;

      const rankDiff = toRank(a.rank) - toRank(b.rank);
      if (rankDiff !== 0) return rankDiff;

      return toTeam(a.team).localeCompare(toTeam(b.team));
    })
    .map((team, index) => ({
      ...team,
      rank: index + 1,
    }));
}
