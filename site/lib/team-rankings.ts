import { readJsonFile } from "./server-data";
import { normalizeRankingsTeams } from "./rankings";

type RankedTeam = {
  rank?: unknown;
  team?: unknown;
};

type RankingsPayload = {
  teams?: RankedTeam[];
};

function teamKey(name: string): string {
  return name
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/&/g, " and ")
    .replace(/[^a-z0-9]+/g, " ")
    .trim()
    .replace(/\s+/g, " ");
}

export function seasonFromDate(date: string): number {
  const year = Number(date.slice(0, 4));
  const month = Number(date.slice(5, 7));
  return month >= 11 ? year + 1 : year;
}

export function getTeamRankMapForSeason(season: number): Map<string, number> {
  const payload = readJsonFile(`rankings_${season}.json`) as RankingsPayload | null;
  const teams = Array.isArray(payload?.teams) ? normalizeRankingsTeams(payload.teams) : [];
  const teamRanks = new Map<string, number>();

  for (const team of teams) {
    if (typeof team.team !== "string") {
      continue;
    }
    const rank =
      typeof team.rank === "number"
        ? team.rank
        : typeof team.rank === "string" && team.rank.trim() !== ""
          ? Number(team.rank)
          : NaN;
    if (!Number.isFinite(rank)) {
      continue;
    }
    teamRanks.set(teamKey(team.team), rank);
  }

  return teamRanks;
}

export function getTeamRankMapForDate(date: string): Map<string, number> {
  return getTeamRankMapForSeason(seasonFromDate(date));
}

export function getTeamRank(teamName: string, teamRanks: Map<string, number>): number | null {
  return teamRanks.get(teamKey(teamName)) ?? null;
}
