import {
  BracketGameDefinition,
  MatchupPredictionCache,
  NcaaBracketField,
  NcaaTournamentResults,
  NcaaValidationResult,
} from "./types";
import { buildPossibleTeamsByGame, NCAA_RESULTS_VERSION } from "./results";

function result(errors: string[]): NcaaValidationResult {
  return { valid: errors.length === 0, errors };
}

export function validateNcaaField(field: NcaaBracketField | null): NcaaValidationResult {
  if (!field) return result(["Missing NCAA bracket field payload"]);

  const errors: string[] = [];
  const regionNames = field.regions.map((region) => region.name);
  if (field.regions.length !== 4) {
    errors.push(`Expected 4 regions, found ${field.regions.length}`);
  }
  if (new Set(regionNames).size !== regionNames.length) {
    errors.push("Region names must be unique");
  }

  const firstFourIds = new Set(field.first_four.map((game) => game.id));
  if (field.first_four.length !== 4) {
    errors.push(`Expected 4 First Four games, found ${field.first_four.length}`);
  }
  if (firstFourIds.size !== field.first_four.length) {
    errors.push("First Four game ids must be unique");
  }

  const teamIds = new Set<number>();
  const teamNames = new Set<string>();
  const regionPlayInRefs = new Map<string, { region: string; seed: number }>();
  let totalTeams = 0;

  for (const region of field.regions) {
    const seeds = region.entries.map((entry) => entry.seed).sort((a, b) => a - b);
    const expectedSeeds = Array.from({ length: 16 }, (_, index) => index + 1);
    if (seeds.length !== 16 || seeds.some((seed, index) => seed !== expectedSeeds[index])) {
      errors.push(`Region ${region.name} must contain seeds 1-16 exactly once`);
    }

    for (const entry of region.entries) {
      if (entry.source === "team") {
        totalTeams += 1;
        if (teamIds.has(entry.team_id)) errors.push(`Duplicate team id ${entry.team_id}`);
        if (teamNames.has(entry.team)) errors.push(`Duplicate team ${entry.team}`);
        teamIds.add(entry.team_id);
        teamNames.add(entry.team);
      } else {
        if (!firstFourIds.has(entry.play_in_game_id)) {
          errors.push(`Region ${region.name} references unknown play-in game ${entry.play_in_game_id}`);
          continue;
        }
        if (regionPlayInRefs.has(entry.play_in_game_id)) {
          errors.push(`Play-in game ${entry.play_in_game_id} is assigned to multiple region slots`);
        } else {
          regionPlayInRefs.set(entry.play_in_game_id, { region: region.name, seed: entry.seed });
        }
      }
    }
  }

  for (const game of field.first_four) {
    if (game.teams.length !== 2) {
      errors.push(`Play-in game ${game.id} must contain exactly 2 teams`);
      continue;
    }
    for (const team of game.teams) {
      totalTeams += 1;
      if (teamIds.has(team.team_id)) errors.push(`Duplicate team id ${team.team_id}`);
      if (teamNames.has(team.team)) errors.push(`Duplicate team ${team.team}`);
      teamIds.add(team.team_id);
      teamNames.add(team.team);
    }
  }

  for (const game of field.first_four) {
    const regionSlot = regionPlayInRefs.get(game.id);
    if (!regionSlot) {
      errors.push(`Play-in game ${game.id} is not wired into any region seed slot`);
      continue;
    }
    if (regionSlot.region !== game.region || regionSlot.seed !== game.seed) {
      errors.push(
        `Play-in game ${game.id} metadata (${game.region} ${game.seed}) does not match region slot `
          + `(${regionSlot.region} ${regionSlot.seed})`,
      );
    }
  }

  if (totalTeams !== 68) {
    errors.push(`Expected 68 unique NCAA teams, found ${totalTeams}`);
  }

  return result(errors);
}

export function validateBracketGraph(games: BracketGameDefinition[]): NcaaValidationResult {
  const errors: string[] = [];
  const ids = new Set<string>();
  const rounds = new Map<string, number>();

  for (const game of games) {
    if (ids.has(game.id)) errors.push(`Duplicate game id ${game.id}`);
    ids.add(game.id);
    rounds.set(game.id, game.roundOrder);
  }

  for (const game of games) {
    for (const source of [game.sourceA, game.sourceB]) {
      if (source.type !== "winner") continue;
      const feederRound = rounds.get(source.gameId);
      if (feederRound == null) {
        errors.push(`Game ${game.id} references missing feeder ${source.gameId}`);
        continue;
      }
      if (feederRound >= game.roundOrder) {
        errors.push(`Game ${game.id} references feeder ${source.gameId} from same or later round`);
      }
    }
  }

  return result(errors);
}

export function validateMatchupCache(
  cache: MatchupPredictionCache | null,
  field: NcaaBracketField | null,
): NcaaValidationResult {
  if (!cache) return result(["Missing NCAA matchup prediction cache"]);
  const fieldValidation = validateNcaaField(field);
  const errors = [...fieldValidation.errors];
  if (!field) return result(errors);

  const teamIds = new Set<number>();
  for (const region of field.regions) {
    for (const entry of region.entries) {
      if (entry.source === "team") teamIds.add(entry.team_id);
    }
  }
  for (const game of field.first_four) {
    for (const team of game.teams) teamIds.add(team.team_id);
  }

  const expectedCount = (teamIds.size * (teamIds.size - 1)) / 2;
  const keys = Object.keys(cache.predictions);
  if (keys.length !== expectedCount) {
    errors.push(`Expected ${expectedCount} matchup cache rows, found ${keys.length}`);
  }

  for (const key of keys) {
    const entry = cache.predictions[key];
    const [rawA, rawB] = key.split("::").map(Number);
    if (!Number.isInteger(rawA) || !Number.isInteger(rawB)) {
      errors.push(`Invalid matchup key ${key}`);
      continue;
    }
    if (rawA >= rawB) errors.push(`Matchup key ${key} must be canonical low::high`);
    if (entry.team1_id !== rawA || entry.team2_id !== rawB) {
      errors.push(`Matchup key ${key} does not align with cached team ids`);
    }
    if (!teamIds.has(entry.team1_id) || !teamIds.has(entry.team2_id)) {
      errors.push(`Matchup key ${key} references a team missing from the NCAA field`);
    }
    if (entry.win_prob_team1 < 0 || entry.win_prob_team1 > 1) {
      errors.push(`Matchup key ${key} has invalid win probability ${entry.win_prob_team1}`);
    }
    for (const compareProb of [
      entry.win_prob_team1_legacy_synthetic,
      entry.win_prob_team1_team_ab_elite_tail_round64_v1,
      entry.win_prob_team1_team_ab_internal,
    ]) {
      if (compareProb == null) continue;
      if (compareProb < 0 || compareProb > 1) {
        errors.push(`Matchup key ${key} has invalid compare win probability ${compareProb}`);
      }
    }
  }

  return result(errors);
}

export function validateNcaaResults(
  results: NcaaTournamentResults | null,
  field: NcaaBracketField | null,
  games: BracketGameDefinition[],
): NcaaValidationResult {
  if (!results) return result([]);
  if (!field) return result(["Cannot validate NCAA results without NCAA field data"]);

  const errors: string[] = [];
  if (!Number.isInteger(results.version) || results.version !== NCAA_RESULTS_VERSION) {
    errors.push(`Expected NCAA results version ${NCAA_RESULTS_VERSION}, found ${results.version}`);
  }
  if (!Number.isInteger(results.season) || results.season !== field.season) {
    errors.push(`NCAA results season ${results.season} does not match field season ${field.season}`);
  }
  if (!results.games || typeof results.games !== "object" || Array.isArray(results.games)) {
    errors.push("NCAA results payload must include a games object");
    return result(errors);
  }

  const gamesById = new Map(games.map((game) => [game.id, game]));
  const possibleTeamsByGame = buildPossibleTeamsByGame(games);

  for (const [gameId, gameResult] of Object.entries(results.games)) {
    if (!gamesById.has(gameId)) {
      errors.push(`Results reference unknown game id ${gameId}`);
      continue;
    }
    if (!["pending", "in_progress", "final"].includes(gameResult.status)) {
      errors.push(`Results for ${gameId} have invalid status ${gameResult.status}`);
      continue;
    }

    if (gameResult.status !== "final") {
      if (gameResult.winner_team_id != null || gameResult.loser_team_id != null) {
        errors.push(`Non-final result ${gameId} must not include winner_team_id or loser_team_id`);
      }
      continue;
    }

    if (!Number.isInteger(gameResult.winner_team_id) || !Number.isInteger(gameResult.loser_team_id)) {
      errors.push(`Final result ${gameId} must include integer winner_team_id and loser_team_id`);
      continue;
    }
    if (gameResult.winner_team_id === gameResult.loser_team_id) {
      errors.push(`Final result ${gameId} winner and loser must be different teams`);
      continue;
    }

    const possible = possibleTeamsByGame[gameId];
    if (!possible) {
      errors.push(`Final result ${gameId} could not be matched to a bracket game`);
      continue;
    }

    const winner = Number(gameResult.winner_team_id);
    const loser = Number(gameResult.loser_team_id);
    const validOrientation =
      (possible.sourceA.has(winner) && possible.sourceB.has(loser))
      || (possible.sourceA.has(loser) && possible.sourceB.has(winner));
    if (!validOrientation) {
      errors.push(`Final result ${gameId} teams do not match the possible bracket participants`);
    }
  }

  return result(errors);
}
