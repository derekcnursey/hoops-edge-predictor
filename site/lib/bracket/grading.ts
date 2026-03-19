import {
  BracketGameDefinition,
  BracketGradingSummary,
  BracketRoundId,
  BracketTeam,
  GradedGameResult,
  GradedRoundSummary,
  NcaaTournamentResults,
} from "./types";
import { buildFinalResultsMap } from "./results";

export const ROUND_POINTS: Record<BracketRoundId, number> = {
  "first-four": 0,
  "round-of-64": 1,
  "round-of-32": 2,
  "sweet-16": 4,
  "elite-8": 8,
  "final-four": 16,
  "national-championship": 32,
};

function createRoundSummary(game: BracketGameDefinition): GradedRoundSummary {
  return {
    roundId: game.roundId,
    roundLabel: game.roundLabel,
    correct: 0,
    incorrect: 0,
    pending: 0,
    score: 0,
    possibleScore: 0,
  };
}

export function gradeBracketPicks(
  games: BracketGameDefinition[],
  picks: Record<string, number>,
  results: NcaaTournamentResults | null,
  teamById: Record<number, BracketTeam>,
): BracketGradingSummary {
  const finalResults = buildFinalResultsMap(results);
  const rounds = new Map<BracketRoundId, GradedRoundSummary>();
  const byGame: Record<string, GradedGameResult> = {};

  let correct = 0;
  let incorrect = 0;
  let pending = 0;
  let totalGraded = 0;
  let score = 0;
  let possibleScore = 0;

  for (const game of games) {
    if (!rounds.has(game.roundId)) {
      rounds.set(game.roundId, createRoundSummary(game));
    }
    const round = rounds.get(game.roundId)!;
    const result = finalResults[game.id];
    const actualWinnerId =
      result && Number.isInteger(result.winner_team_id) ? Number(result.winner_team_id) : undefined;
    const actualLoserId =
      result && Number.isInteger(result.loser_team_id) ? Number(result.loser_team_id) : undefined;
    const isFinal = actualWinnerId != null && actualLoserId != null;
    const isScored = ROUND_POINTS[game.roundId] > 0;
    const pickId = picks[game.id];

    let status: GradedGameResult["status"] = "pending";
    if (isFinal && Number.isInteger(pickId)) {
      totalGraded += 1;
      if (pickId === actualWinnerId) {
        status = "correct";
        correct += 1;
        round.correct += 1;
      } else {
        status = "incorrect";
        incorrect += 1;
        round.incorrect += 1;
      }
    } else {
      pending += 1;
      round.pending += 1;
    }

    const pointsPossible = isFinal && isScored ? ROUND_POINTS[game.roundId] : 0;
    const pointsEarned = status === "correct" ? pointsPossible : 0;
    possibleScore += pointsPossible;
    score += pointsEarned;
    round.possibleScore += pointsPossible;
    round.score += pointsEarned;

    byGame[game.id] = {
      gameId: game.id,
      roundId: game.roundId,
      roundLabel: game.roundLabel,
      actualWinnerId,
      actualLoserId,
      actualWinnerName: actualWinnerId ? teamById[actualWinnerId]?.name : undefined,
      actualLoserName: actualLoserId ? teamById[actualLoserId]?.name : undefined,
      status,
      isFinal,
      isScored,
      pointsEarned,
      pointsPossible,
    };
  }

  return {
    correct,
    incorrect,
    pending,
    totalGraded,
    score,
    possibleScore,
    rounds: Array.from(rounds.values()),
    byGame,
  };
}
