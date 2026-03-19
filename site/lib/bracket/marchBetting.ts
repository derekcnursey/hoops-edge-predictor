import { buildNcaaBracketGames, getBracketTeams } from "./ncaaBracket";
import { buildFinalResultsMap } from "./results";
import {
  buildPredictionFromCacheEntry,
  canonicalMatchupKey,
  canonicalizePrediction,
  getDisplayFavoriteSummary,
} from "./predictions";
import {
  BracketSource,
  BracketTeam,
  MarchBettingGame,
  MatchupPrediction,
  MatchupPredictionCache,
  NcaaBracketField,
  NcaaTournamentResults,
} from "./types";

function resolveSource(
  source: BracketSource,
  teamById: Record<number, BracketTeam>,
  finalResults: Record<string, { winner_team_id?: number | null }>,
): BracketTeam | null {
  if (source.type === "team") {
    return teamById[source.teamId] ?? null;
  }
  const winnerId = finalResults[source.gameId]?.winner_team_id;
  if (winnerId == null) return null;
  return teamById[winnerId] ?? null;
}

export function buildScheduledNcaaMarchData(
  field: NcaaBracketField,
  cache: MatchupPredictionCache,
  results: NcaaTournamentResults | null = null,
): {
  initialPredictionCache: Record<string, MatchupPrediction>;
  marchGames: MarchBettingGame[];
} {
  const initialPredictionCache: Record<string, MatchupPrediction> = {};
  const marchGames: MarchBettingGame[] = [];
  const teamById = Object.fromEntries(
    getBracketTeams(field).map((team) => [team.id, team]),
  ) as Record<number, BracketTeam>;
  const finalResults = buildFinalResultsMap(results);

  const scheduledGames = buildNcaaBracketGames(field)
    .filter((game) => game.roundId === "first-four" || game.roundId === "round-of-64")
    .sort((a, b) => a.roundOrder - b.roundOrder || a.matchupOrder - b.matchupOrder);

  for (const game of scheduledGames) {
    const gameResult = finalResults[game.id];
    if (gameResult?.winner_team_id != null) continue;

    const teamA = resolveSource(game.sourceA, teamById, finalResults);
    const teamB = resolveSource(game.sourceB, teamById, finalResults);
    if (!teamA || !teamB) continue;

    const matchupKey = canonicalMatchupKey(teamA.id, teamB.id);
    const entry = cache.predictions[matchupKey];
    if (!entry) continue;

    const prediction = buildPredictionFromCacheEntry(entry, teamA.id, teamB.id);
    initialPredictionCache[matchupKey] = canonicalizePrediction(prediction);

    const displaySummary = getDisplayFavoriteSummary(prediction);
    const displayFavoredTeamId = displaySummary.favoriteTeamId;
    const displayFavoredTeamName = displaySummary.favoriteTeamName;
    const favoriteWinProb = displaySummary.favoriteWinProb;
    const rawDisplaySpreadHome =
      prediction.modelSpreadHome == null ? null : -prediction.modelSpreadHome;
    const displayDisplaySpreadHome =
      prediction.displayModelSpreadHome == null ? null : -prediction.displayModelSpreadHome;
    const rawDiffAbs =
      rawDisplaySpreadHome != null && prediction.marketSpreadHome != null
        ? Math.abs(rawDisplaySpreadHome - prediction.marketSpreadHome)
        : null;
    const displayDiffAbs =
      displayDisplaySpreadHome != null && prediction.marketSpreadHome != null
        ? Math.abs(displayDisplaySpreadHome - prediction.marketSpreadHome)
        : null;
    const spreadDiffAbs =
      prediction.marketProjectedSpread != null
        ? Math.abs(displaySummary.spread - prediction.marketProjectedSpread)
        : null;

    marchGames.push({
      gameId: game.id,
      roundId: game.roundId,
      roundLabel: game.roundLabel,
      region: game.region ?? null,
      matchupOrder: game.matchupOrder,
      startTime: prediction.scheduledStartTime ?? null,
      teamAId: teamA.id,
      teamAName: teamA.name,
      teamASeed: teamA.seed,
      teamBId: teamB.id,
      teamBName: teamB.name,
      teamBSeed: teamB.seed,
      homeTeamId: prediction.scheduledHomeTeamId ?? null,
      homeTeamName: prediction.scheduledHomeTeamName ?? null,
      awayTeamId: prediction.scheduledAwayTeamId ?? null,
      awayTeamName: prediction.scheduledAwayTeamName ?? null,
      favoriteTeamId: prediction.modelWinnerId,
      favoriteTeamName: prediction.modelWinnerName,
      favoriteWinProb,
      rawProjectedSpread: prediction.rawProjectedSpread ?? prediction.projectedSpread,
      displayProjectedSpread: displaySummary.spread,
      modelSpreadHome: prediction.modelSpreadHome ?? null,
      displayModelSpreadHome: prediction.displayModelSpreadHome ?? null,
      predSigma: prediction.predSigma ?? null,
      edgeHomePoints: prediction.edgeHomePoints ?? null,
      displayEdgeHomePoints: prediction.displayEdgeHomePoints ?? null,
      pickSide: prediction.pickSide ?? null,
      pickCoverProb: prediction.pickCoverProb ?? null,
      pickProbEdge: prediction.pickProbEdge ?? null,
      displayFavoredTeamId,
      displayFavoredTeamName,
      marketSpreadHome: prediction.marketSpreadHome ?? null,
      marketProjectedSpread: prediction.marketProjectedSpread ?? null,
      marketFavoredTeamId: prediction.marketFavoredTeamId ?? null,
      marketFavoredTeamName: prediction.marketFavoredTeamName ?? null,
      marketLineSource: prediction.marketLineSource ?? null,
      rawDiffAbs,
      displayDiffAbs,
      spreadDiffAbs,
    });
  }

  return { initialPredictionCache, marchGames };
}
