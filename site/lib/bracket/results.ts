import {
  BracketGameDefinition,
  BracketSource,
  NcaaTournamentResultGame,
  NcaaTournamentResults,
} from "./types";

export const NCAA_RESULTS_VERSION = 1;

export type PossibleGameTeams = {
  sourceA: Set<number>;
  sourceB: Set<number>;
  all: Set<number>;
};

function collectPossibleSourceTeamIds(
  source: BracketSource,
  gamesById: Map<string, BracketGameDefinition>,
  gameCache: Map<string, Set<number>>,
): Set<number> {
  if (source.type === "team") return new Set([source.teamId]);

  const cached = gameCache.get(source.gameId);
  if (cached) return new Set(cached);

  const feeder = gamesById.get(source.gameId);
  if (!feeder) return new Set();

  const ids = new Set<number>([
    ...collectPossibleSourceTeamIds(feeder.sourceA, gamesById, gameCache),
    ...collectPossibleSourceTeamIds(feeder.sourceB, gamesById, gameCache),
  ]);
  gameCache.set(source.gameId, ids);
  return new Set(ids);
}

export function buildPossibleTeamsByGame(games: BracketGameDefinition[]): Record<string, PossibleGameTeams> {
  const gamesById = new Map(games.map((game) => [game.id, game]));
  const gameCache = new Map<string, Set<number>>();

  return Object.fromEntries(
    games.map((game) => {
      const sourceA = collectPossibleSourceTeamIds(game.sourceA, gamesById, gameCache);
      const sourceB = collectPossibleSourceTeamIds(game.sourceB, gamesById, gameCache);
      return [
        game.id,
        {
          sourceA,
          sourceB,
          all: new Set([...sourceA, ...sourceB]),
        },
      ];
    }),
  );
}

export function isFinalTournamentResult(game: NcaaTournamentResultGame | undefined): boolean {
  return game?.status === "final";
}

export function buildFinalResultsMap(results: NcaaTournamentResults | null): Record<string, NcaaTournamentResultGame> {
  if (!results || !results.games || typeof results.games !== "object" || Array.isArray(results.games)) return {};
  return Object.fromEntries(
    Object.entries(results.games).filter(([, game]) => isFinalTournamentResult(game)),
  );
}
