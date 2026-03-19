import {
  formatAmericanOddsFromProb,
} from "../data";
import { buildNcaaBracketGames, getBracketTeams } from "./ncaaBracket";
import { buildFinalResultsMap } from "./results";
import {
  BracketRoundId,
  BracketSource,
  MatchupPredictionCache,
  NcaaBracketField,
  NcaaTournamentResults,
} from "./types";

const DEFAULT_TOURNAMENT_START = "2026-03-20T00:00:00.000Z";

export type NcaaOddsRoundKey =
  | "round-of-64"
  | "round-of-32"
  | "sweet-16"
  | "elite-8"
  | "final-four"
  | "national-championship"
  | "champion";

export type NcaaOddsRow = {
  teamId: number;
  team: string;
  seed: number;
  region: string | null;
  conference: string;
  record: string;
  confRecord: string;
  rank: number;
  roundProbabilities: Record<NcaaOddsRoundKey, number>;
};

export type NcaaOddsSummary = {
  titleFavorite: NcaaOddsRow | null;
  finalFourLocks: NcaaOddsRow[];
};

export type NcaaOddsData = {
  generatedAt: string;
  season: number;
  methodology: {
    type: "exact_bracket";
    note: string;
  };
  rows: NcaaOddsRow[];
  summary: NcaaOddsSummary;
  optimalBracket: NcaaOptimalBracketPlan;
};

export type NcaaOddsProbabilityVariant =
  | "active"
  | "display_average"
  | "legacy_synthetic"
  | "team_ab_elite_tail_round64_v1"
  | "team_ab_internal";

export type NcaaExpectedBracketPick = {
  gameId: string;
  roundId: BracketRoundId;
  roundLabel: string;
  roundOrder: number;
  region: string | null;
  matchupOrder: number;
  points: number;
  teamId: number;
  team: string;
  seed: number;
  winProbability: number;
  expectedPoints: number;
};

export type NcaaExpectedScoreRound = {
  roundId: BracketRoundId;
  roundLabel: string;
  pointsPerGame: number;
  maxPoints: number;
  expectedPoints: number;
  picks: NcaaExpectedBracketPick[];
};

export type NcaaOptimalBracketPlan = {
  totalExpectedPoints: number;
  totalPossiblePoints: number;
  rounds: NcaaExpectedScoreRound[];
};

type ProbabilityMap = Map<number, number>;
type RowMap = Map<number, NcaaOddsRow>;
type PickOption = {
  score: number;
  picks: NcaaExpectedBracketPick[];
};

const SCORE_BY_ROUND: Record<BracketRoundId, number> = {
  "first-four": 0,
  "round-of-64": 1,
  "round-of-32": 2,
  "sweet-16": 4,
  "elite-8": 8,
  "final-four": 16,
  "national-championship": 32,
};
const ROUND_ORDER: Record<BracketRoundId, number> = {
  "first-four": 0,
  "round-of-64": 1,
  "round-of-32": 2,
  "sweet-16": 3,
  "elite-8": 4,
  "final-four": 5,
  "national-championship": 6,
};

function roundKeyFromGame(
  roundId: BracketRoundId,
): Exclude<NcaaOddsRoundKey, "champion"> | null {
  if (roundId === "first-four") return null;
  return roundId;
}

function emptyRoundProbabilities(): Record<NcaaOddsRoundKey, number> {
  return {
    "round-of-64": 0,
    "round-of-32": 0,
    "sweet-16": 0,
    "elite-8": 0,
    "final-four": 0,
    "national-championship": 0,
    champion: 0,
  };
}

function addProbability(
  target: ProbabilityMap,
  teamId: number,
  probability: number,
): void {
  target.set(teamId, (target.get(teamId) ?? 0) + probability);
}

function probabilityForMatchup(
  cache: MatchupPredictionCache,
  teamAId: number,
  teamBId: number,
  variant: NcaaOddsProbabilityVariant,
): number {
  const canonicalA = Math.min(teamAId, teamBId);
  const canonicalB = Math.max(teamAId, teamBId);
  const key = `${canonicalA}::${canonicalB}`;
  const entry = cache.predictions[key];
  if (!entry) {
    throw new Error(`Missing NCAA matchup cache entry for ${key}`);
  }

  const directOrder = entry.team1_id === teamAId && entry.team2_id === teamBId;
  const activeProbability = directOrder ? entry.win_prob_team1 : 1 - entry.win_prob_team1;
  const internalProbability = directOrder
    ? entry.win_prob_team1_team_ab_internal
    : entry.win_prob_team1_team_ab_internal == null
      ? null
      : 1 - entry.win_prob_team1_team_ab_internal;
  const tailProbability = directOrder
    ? entry.win_prob_team1_team_ab_elite_tail_round64_v1
    : entry.win_prob_team1_team_ab_elite_tail_round64_v1 == null
      ? null
      : 1 - entry.win_prob_team1_team_ab_elite_tail_round64_v1;

  const variantProbability = (() => {
    if (variant === "display_average") {
      if (
        entry.matchup_model_variant_active === "team_ab_elite_tail_round64_v1" &&
        internalProbability != null
      ) {
        return (activeProbability + internalProbability) / 2;
      }
      if (
        entry.matchup_model_variant_active === "legacy_synthetic" &&
        tailProbability != null
      ) {
        return (activeProbability + tailProbability) / 2;
      }
      return activeProbability;
    }
    if (variant === "team_ab_internal") {
      return internalProbability;
    }
    if (variant === "legacy_synthetic") {
      return directOrder
        ? entry.win_prob_team1_legacy_synthetic
        : entry.win_prob_team1_legacy_synthetic == null
          ? null
          : 1 - entry.win_prob_team1_legacy_synthetic;
    }
    if (variant === "team_ab_elite_tail_round64_v1") {
      return tailProbability;
    }
    return activeProbability;
  })();
  const probability = variantProbability;
  if (probability != null && Number.isFinite(probability)) {
    return probability;
  }

  const legacyProbability = directOrder
    ? entry.win_prob_team1_legacy_synthetic
    : entry.win_prob_team1_legacy_synthetic == null
      ? null
      : 1 - entry.win_prob_team1_legacy_synthetic;
  if (legacyProbability != null && Number.isFinite(legacyProbability)) {
    return legacyProbability;
  }

  throw new Error(`Missing NCAA matchup probability for ${key}`);
}

function sourceDistribution(
  source: BracketSource,
  winnerByGame: Map<string, ProbabilityMap>,
): ProbabilityMap {
  if (source.type === "team") {
    return new Map([[source.teamId, 1]]);
  }
  const distribution = winnerByGame.get(source.gameId);
  if (!distribution) {
    throw new Error(`Missing feeder winner distribution for ${source.gameId}`);
  }
  return distribution;
}

function sortedRows(rows: NcaaOddsRow[]): NcaaOddsRow[] {
  return [...rows].sort((a, b) => {
    const champDiff =
      b.roundProbabilities.champion - a.roundProbabilities.champion;
    if (champDiff !== 0) return champDiff;
    const titleDiff =
      b.roundProbabilities["national-championship"] -
      a.roundProbabilities["national-championship"];
    if (titleDiff !== 0) return titleDiff;
    return a.seed - b.seed || a.team.localeCompare(b.team);
  });
}

function scoreForRound(roundId: BracketRoundId): number {
  return SCORE_BY_ROUND[roundId];
}

function totalPossiblePoints(games: ReturnType<typeof buildNcaaBracketGames>): number {
  return games.reduce((sum, game) => sum + scoreForRound(game.roundId), 0);
}

function chooseBetterOption(
  current: PickOption | undefined,
  candidate: PickOption,
): PickOption {
  if (!current) return candidate;
  if (candidate.score > current.score + 1e-9) return candidate;
  if (current.score > candidate.score + 1e-9) return current;
  const currentKey = current.picks.map((pick) => `${pick.gameId}:${pick.team}`).join("|");
  const candidateKey = candidate.picks.map((pick) => `${pick.gameId}:${pick.team}`).join("|");
  return candidateKey.localeCompare(currentKey) < 0 ? candidate : current;
}

function pickForGame(
  game: ReturnType<typeof buildNcaaBracketGames>[number],
  teamsById: Map<number, ReturnType<typeof getBracketTeams>[number]>,
  teamId: number,
  probability: number,
): NcaaExpectedBracketPick {
  const team = teamsById.get(teamId);
  if (!team) {
    throw new Error(`Missing bracket team ${teamId} for ${game.id}`);
  }
  const points = scoreForRound(game.roundId);
  return {
    gameId: game.id,
    roundId: game.roundId,
    roundLabel: game.roundLabel,
    roundOrder: game.roundOrder,
    region: game.region ?? null,
    matchupOrder: game.matchupOrder,
    points,
    teamId,
    team: team.name,
    seed: team.seed,
    winProbability: probability,
    expectedPoints: probability * points,
  };
}

function sourcePickOptions(
  source: BracketSource,
  optimalByGame: Map<string, Map<number, PickOption>>,
): Map<number, PickOption> {
  if (source.type === "team") {
    return new Map([[source.teamId, { score: 0, picks: [] }]]);
  }
  const options = optimalByGame.get(source.gameId);
  if (!options) {
    throw new Error(`Missing optimal options for feeder game ${source.gameId}`);
  }
  return options;
}

function buildTournamentState(
  field: NcaaBracketField,
  cache: MatchupPredictionCache,
  variant: NcaaOddsProbabilityVariant,
  results: NcaaTournamentResults | null = null,
): {
  rowsByTeamId: RowMap;
  games: ReturnType<typeof buildNcaaBracketGames>;
  winnerByGame: Map<string, ProbabilityMap>;
  teamsById: Map<number, ReturnType<typeof getBracketTeams>[number]>;
} {
  const teams = getBracketTeams(field);
  const teamsById = new Map(teams.map((team) => [team.id, team]));
  const games = buildNcaaBracketGames(field).sort(
    (a, b) => a.roundOrder - b.roundOrder || a.matchupOrder - b.matchupOrder,
  );
  const rowsByTeamId: RowMap = new Map(
    teams.map((team) => [
      team.id,
      {
        teamId: team.id,
        team: team.name,
        seed: team.seed,
        region: team.region ?? null,
        conference: team.conference,
        record: team.record,
        confRecord: team.confRecord,
        rank: team.rank,
        roundProbabilities: emptyRoundProbabilities(),
      },
    ]),
  );
  const winnerByGame = new Map<string, ProbabilityMap>();
  const finalResults = buildFinalResultsMap(results);

  for (const game of games) {
    const distA = sourceDistribution(game.sourceA, winnerByGame);
    const distB = sourceDistribution(game.sourceB, winnerByGame);
    const reachRound = roundKeyFromGame(game.roundId);
    if (reachRound) {
      for (const [teamId, probability] of distA.entries()) {
        const row = rowsByTeamId.get(teamId);
        if (row) row.roundProbabilities[reachRound] += probability;
      }
      for (const [teamId, probability] of distB.entries()) {
        const row = rowsByTeamId.get(teamId);
        if (row) row.roundProbabilities[reachRound] += probability;
      }
    }

    const finalResult = finalResults[game.id];
    const winners = new Map<number, number>();
    if (finalResult?.winner_team_id != null) {
      winners.set(finalResult.winner_team_id, 1);
    } else {
      for (const [teamAId, probAReach] of distA.entries()) {
        for (const [teamBId, probBReach] of distB.entries()) {
          const meetingProb = probAReach * probBReach;
          const probAWin = probabilityForMatchup(cache, teamAId, teamBId, variant);
          addProbability(winners, teamAId, meetingProb * probAWin);
          addProbability(winners, teamBId, meetingProb * (1 - probAWin));
        }
      }
    }
    winnerByGame.set(game.id, winners);
  }

  const championDistribution =
    winnerByGame.get("national-championship") ?? new Map<number, number>();
  for (const [teamId, probability] of championDistribution.entries()) {
    const row = rowsByTeamId.get(teamId);
    if (row) row.roundProbabilities.champion = probability;
  }

  return { rowsByTeamId, games, winnerByGame, teamsById };
}

function buildOptimalBracketPlan(
  games: ReturnType<typeof buildNcaaBracketGames>,
  winnerByGame: Map<string, ProbabilityMap>,
  teamsById: Map<number, ReturnType<typeof getBracketTeams>[number]>,
): NcaaOptimalBracketPlan {
  const optimalByGame = new Map<string, Map<number, PickOption>>();
  const bestOverallByGame = new Map<string, PickOption>();

  for (const game of games) {
    const dist = winnerByGame.get(game.id);
    if (!dist) {
      throw new Error(`Missing winner distribution for ${game.id}`);
    }
    const leftOptions = sourcePickOptions(game.sourceA, optimalByGame);
    const rightOptions = sourcePickOptions(game.sourceB, optimalByGame);

    let bestLeftOverall: PickOption | null = null;
    for (const option of leftOptions.values()) {
      bestLeftOverall = chooseBetterOption(bestLeftOverall ?? undefined, option);
    }
    let bestRightOverall: PickOption | null = null;
    for (const option of rightOptions.values()) {
      bestRightOverall = chooseBetterOption(bestRightOverall ?? undefined, option);
    }
    if (!bestLeftOverall || !bestRightOverall) {
      throw new Error(`Missing feeder options for ${game.id}`);
    }

    const bestByWinner = new Map<number, PickOption>();
    for (const [teamId, probability] of dist.entries()) {
      const gamePick = pickForGame(game, teamsById, teamId, probability);
      const leftWinner = leftOptions.get(teamId);
      if (leftWinner) {
        bestByWinner.set(
          teamId,
          chooseBetterOption(bestByWinner.get(teamId), {
            score: leftWinner.score + bestRightOverall.score + gamePick.expectedPoints,
            picks: [...leftWinner.picks, ...bestRightOverall.picks, gamePick],
          }),
        );
      }
      const rightWinner = rightOptions.get(teamId);
      if (rightWinner) {
        bestByWinner.set(
          teamId,
          chooseBetterOption(bestByWinner.get(teamId), {
            score: bestLeftOverall.score + rightWinner.score + gamePick.expectedPoints,
            picks: [...bestLeftOverall.picks, ...rightWinner.picks, gamePick],
          }),
        );
      }
    }

    let bestOverall: PickOption | undefined;
    for (const option of bestByWinner.values()) {
      bestOverall = chooseBetterOption(bestOverall, option);
    }
    if (!bestOverall) {
      throw new Error(`Unable to build optimal bracket options for ${game.id}`);
    }
    optimalByGame.set(game.id, bestByWinner);
    bestOverallByGame.set(game.id, bestOverall);
  }

  const finalPlan = bestOverallByGame.get("national-championship");
  if (!finalPlan) {
    throw new Error("Missing national championship optimal plan");
  }

  const roundMap = new Map<BracketRoundId, NcaaExpectedScoreRound>();
  for (const game of games) {
    const pointsPerGame = scoreForRound(game.roundId);
    if (pointsPerGame === 0) continue;
    const existing = roundMap.get(game.roundId);
    if (existing) {
      existing.maxPoints += pointsPerGame;
    } else {
      roundMap.set(game.roundId, {
        roundId: game.roundId,
        roundLabel: game.roundLabel,
        pointsPerGame,
        maxPoints: pointsPerGame,
        expectedPoints: 0,
        picks: [],
      });
    }
  }

  for (const pick of finalPlan.picks) {
    if (pick.points === 0) continue;
    const round = roundMap.get(pick.roundId);
    if (!round) continue;
    round.expectedPoints += pick.expectedPoints;
    round.picks.push(pick);
  }

  const rounds = Array.from(roundMap.values())
    .map((round) => ({
      ...round,
      expectedPoints: Number(round.expectedPoints.toFixed(3)),
      picks: [...round.picks].sort((a, b) => a.matchupOrder - b.matchupOrder),
    }))
    .sort((a, b) => ROUND_ORDER[a.roundId] - ROUND_ORDER[b.roundId]);

  return {
    totalExpectedPoints: Number(finalPlan.score.toFixed(3)),
    totalPossiblePoints: totalPossiblePoints(games),
    rounds,
  };
}

export function buildNcaaOddsData(
  field: NcaaBracketField,
  cache: MatchupPredictionCache,
  variant: NcaaOddsProbabilityVariant = "active",
  results: NcaaTournamentResults | null = null,
): NcaaOddsData {
  const { rowsByTeamId, games, winnerByGame, teamsById } = buildTournamentState(
    field,
    cache,
    variant,
    results,
  );
  const rows = sortedRows(Array.from(rowsByTeamId.values()));
  const titleFavorite = rows[0] ?? null;
  const finalFourLocks = rows
    .filter((row) => row.roundProbabilities["final-four"] >= 0.5)
    .slice(0, 8);
  const optimalBracket = buildOptimalBracketPlan(games, winnerByGame, teamsById);

  return {
    generatedAt: cache.generated_at,
    season: field.season,
    methodology: {
      type: "exact_bracket",
      note:
        variant === "display_average"
          ? "Exact NCAA bracket advancement probabilities using the same shown-average matchup win probabilities as the bracket cards. When a comparison baseline exists, each matchup probability is the midpoint between the active model and the comparison model."
          : variant === "team_ab_internal"
            ? "Exact NCAA bracket advancement probabilities using the Team A/B internal-efficiency comparison matchup probabilities from the cache. Sigma remains on the cached legacy path."
            : "Exact NCAA bracket advancement probabilities using the active cached matchup win probabilities for the selected bracket-model variant. Those matchup probabilities keep the cached sigma path but follow the active bracket mean-model variant.",
    },
    rows,
    summary: {
      titleFavorite,
      finalFourLocks,
    },
    optimalBracket,
  };
}

export function formatRoundOdds(probability: number): string | null {
  return formatAmericanOddsFromProb(probability);
}
