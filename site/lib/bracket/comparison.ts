import { BracketTeam, MatchupPrediction, ResolvedBracketGame } from "./types";

export const MAJOR_UPSET_SEED_GAP = 5;

export type ComparisonConfidence = "strong" | "moderate" | "toss-up";

export type GameComparison = {
  selectedWinnerId?: number;
  selectedWinnerName?: string;
  modelWinnerId?: number;
  modelWinnerName?: string;
  agreesWithModel?: boolean;
  isUpset: boolean;
  isMajorUpset: boolean;
  confidence: ComparisonConfidence;
  confidenceLabel: string;
};

export type BracketComparisonSummary = {
  picksMade: number;
  modelAgreements: number;
  modelFades: number;
  upsetPicks: number;
  majorUpsets: number;
};

function getSelectedWinner(game: ResolvedBracketGame): BracketTeam | null {
  if (!game.selectedWinnerId) return null;
  if (game.teamA?.id === game.selectedWinnerId) return game.teamA;
  if (game.teamB?.id === game.selectedWinnerId) return game.teamB;
  return null;
}

function getOpponent(game: ResolvedBracketGame, selectedWinnerId?: number): BracketTeam | null {
  if (!selectedWinnerId) return null;
  if (game.teamA?.id === selectedWinnerId) return game.teamB;
  if (game.teamB?.id === selectedWinnerId) return game.teamA;
  return null;
}

export function isUpsetPick(winner: BracketTeam | null, opponent: BracketTeam | null): boolean {
  return Boolean(winner && opponent && winner.seed > opponent.seed);
}

export function isMajorUpsetPick(winner: BracketTeam | null, opponent: BracketTeam | null): boolean {
  return Boolean(winner && opponent && winner.seed - opponent.seed >= MAJOR_UPSET_SEED_GAP);
}

export function getConfidenceBucket(prediction: MatchupPrediction | undefined): ComparisonConfidence {
  if (!prediction) return "toss-up";
  const favoriteWinProb = Math.max(
    prediction.displayWinProbA ?? prediction.winProbA,
    prediction.displayWinProbB ?? prediction.winProbB,
  );
  if (favoriteWinProb >= 0.7) return "strong";
  if (favoriteWinProb >= 0.6) return "moderate";
  return "toss-up";
}

export function getConfidenceLabel(prediction: MatchupPrediction | undefined): string {
  const bucket = getConfidenceBucket(prediction);
  if (bucket === "strong") return "Strong model lean";
  if (bucket === "moderate") return "Moderate lean";
  return "Toss-up";
}

export function buildGameComparison(
  game: ResolvedBracketGame,
  prediction?: MatchupPrediction,
): GameComparison {
  const selectedWinner = getSelectedWinner(game);
  const opponent = getOpponent(game, selectedWinner?.id);

  return {
    selectedWinnerId: selectedWinner?.id,
    selectedWinnerName: selectedWinner?.name,
    modelWinnerId: prediction?.modelWinnerId,
    modelWinnerName: prediction?.modelWinnerName,
    agreesWithModel:
      selectedWinner?.id != null && prediction?.modelWinnerId != null
        ? selectedWinner.id === prediction.modelWinnerId
        : undefined,
    isUpset: isUpsetPick(selectedWinner, opponent),
    isMajorUpset: isMajorUpsetPick(selectedWinner, opponent),
    confidence: getConfidenceBucket(prediction),
    confidenceLabel: getConfidenceLabel(prediction),
  };
}

export function summarizeBracketComparisons(
  games: ResolvedBracketGame[],
  predictionsByGame: Record<string, MatchupPrediction | undefined>,
): BracketComparisonSummary {
  const summary: BracketComparisonSummary = {
    picksMade: 0,
    modelAgreements: 0,
    modelFades: 0,
    upsetPicks: 0,
    majorUpsets: 0,
  };

  for (const game of games) {
    if (!game.selectedWinnerId) continue;
    summary.picksMade += 1;

    const comparison = buildGameComparison(game, predictionsByGame[game.id]);
    if (comparison.isUpset) summary.upsetPicks += 1;
    if (comparison.isMajorUpset) summary.majorUpsets += 1;
    if (comparison.agreesWithModel === true) summary.modelAgreements += 1;
    if (comparison.agreesWithModel === false) summary.modelFades += 1;
  }

  return summary;
}
