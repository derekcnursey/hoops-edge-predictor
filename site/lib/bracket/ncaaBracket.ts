import {
  BracketGameDefinition,
  BracketRoundId,
  BracketSource,
  BracketTeam,
  NcaaBracketField,
  NcaaRegionEntry,
} from "./types";

const REGION_PAIRINGS = [
  { seedA: 1, seedB: 16, slot: 1 },
  { seedA: 8, seedB: 9, slot: 2 },
  { seedA: 5, seedB: 12, slot: 3 },
  { seedA: 4, seedB: 13, slot: 4 },
  { seedA: 6, seedB: 11, slot: 5 },
  { seedA: 3, seedB: 14, slot: 6 },
  { seedA: 7, seedB: 10, slot: 7 },
  { seedA: 2, seedB: 15, slot: 8 },
] as const;

const ROUND_LABELS: Record<BracketRoundId, string> = {
  "first-four": "First Four",
  "round-of-64": "Round of 64",
  "round-of-32": "Round of 32",
  "sweet-16": "Sweet 16",
  "elite-8": "Elite 8",
  "final-four": "Final Four",
  "national-championship": "National Championship",
};

function entryToSource(entry: NcaaRegionEntry, region: string): BracketSource {
  if (entry.source === "team") {
    return {
      type: "team",
      teamId: entry.team_id,
      label: `(${entry.seed}) ${entry.team}`,
    };
  }
  return {
    type: "winner",
    gameId: entry.play_in_game_id,
    label: `${region} ${entry.seed} winner`,
  };
}

export function getBracketTeams(field: NcaaBracketField): BracketTeam[] {
  const byId = new Map<number, BracketTeam>();

  for (const region of field.regions) {
    for (const entry of region.entries) {
      if (entry.source !== "team") continue;
      byId.set(entry.team_id, {
        id: entry.team_id,
        name: entry.team,
        seed: entry.seed,
        region: region.name,
        rank: entry.rank,
        conference: entry.conference,
        record: entry.record,
        confRecord: entry.conf_record,
        adjOe: entry.adj_oe,
        adjDe: entry.adj_de,
        adjNet: entry.adj_margin,
        adjTempo: entry.adj_tempo,
        modelIndex: entry.model_index,
        adjOeRank: entry.adj_oe_rank ?? null,
        adjDeRank: entry.adj_de_rank ?? null,
        adjNetRank: entry.adj_margin_rank ?? null,
        adjTempoRank: entry.adj_tempo_rank ?? null,
        ftPct: entry.ft_pct ?? null,
        threePPct: entry.three_p_pct ?? null,
        defThreePPct: entry.def_3p_pct ?? null,
        ftPctRank: entry.ft_pct_rank ?? null,
        threePPctRank: entry.three_p_pct_rank ?? null,
        defThreePPctRank: entry.def_3p_pct_rank ?? null,
        modelIndexRank: entry.model_index_rank ?? null,
      });
    }
  }

  for (const game of field.first_four) {
    for (const team of game.teams) {
      byId.set(team.team_id, {
        id: team.team_id,
        name: team.team,
        seed: game.seed,
        region: game.region,
        rank: team.rank,
        conference: team.conference,
        record: team.record,
        confRecord: team.conf_record,
        adjOe: team.adj_oe,
        adjDe: team.adj_de,
        adjNet: team.adj_margin,
        adjTempo: team.adj_tempo,
        modelIndex: team.model_index,
        adjOeRank: team.adj_oe_rank ?? null,
        adjDeRank: team.adj_de_rank ?? null,
        adjNetRank: team.adj_margin_rank ?? null,
        adjTempoRank: team.adj_tempo_rank ?? null,
        ftPct: team.ft_pct ?? null,
        threePPct: team.three_p_pct ?? null,
        defThreePPct: team.def_3p_pct ?? null,
        ftPctRank: team.ft_pct_rank ?? null,
        threePPctRank: team.three_p_pct_rank ?? null,
        defThreePPctRank: team.def_3p_pct_rank ?? null,
        modelIndexRank: team.model_index_rank ?? null,
      });
    }
  }

  return Array.from(byId.values());
}

export function buildNcaaBracketGames(field: NcaaBracketField): BracketGameDefinition[] {
  const games: BracketGameDefinition[] = [];

  for (const [index, game] of field.first_four.entries()) {
    games.push({
      id: game.id,
      roundId: "first-four",
      roundLabel: ROUND_LABELS["first-four"],
      roundOrder: 0,
      title: `${game.region} ${game.seed} seed play-in`,
      region: game.region,
      matchupOrder: index + 1,
      sourceA: {
        type: "team",
        teamId: game.teams[0].team_id,
        label: `(${game.seed}) ${game.teams[0].team}`,
      },
      sourceB: {
        type: "team",
        teamId: game.teams[1].team_id,
        label: `(${game.seed}) ${game.teams[1].team}`,
      },
    });
  }

  for (const [regionIndex, region] of field.regions.entries()) {
    const entryBySeed = new Map(region.entries.map((entry) => [entry.seed, entry]));
    const regionSlug = region.name.toLowerCase().replace(/\s+/g, "-");

    for (const pairing of REGION_PAIRINGS) {
      const entryA = entryBySeed.get(pairing.seedA);
      const entryB = entryBySeed.get(pairing.seedB);
      if (!entryA || !entryB) {
        throw new Error(`Missing seed assignment in ${region.name}`);
      }
      games.push({
        id: `${regionSlug}-r64-${pairing.slot}`,
        roundId: "round-of-64",
        roundLabel: ROUND_LABELS["round-of-64"],
        roundOrder: 1,
        title: `${region.name} game ${pairing.slot}`,
        region: region.name,
        matchupOrder: regionIndex * 8 + pairing.slot,
        sourceA: entryToSource(entryA, region.name),
        sourceB: entryToSource(entryB, region.name),
      });
    }

    for (let slot = 1; slot <= 4; slot += 1) {
      games.push({
        id: `${regionSlug}-r32-${slot}`,
        roundId: "round-of-32",
        roundLabel: ROUND_LABELS["round-of-32"],
        roundOrder: 2,
        title: `${region.name} round of 32`,
        region: region.name,
        matchupOrder: regionIndex * 4 + slot,
        sourceA: {
          type: "winner",
          gameId: `${regionSlug}-r64-${slot * 2 - 1}`,
          label: `${region.name} game ${slot * 2 - 1} winner`,
        },
        sourceB: {
          type: "winner",
          gameId: `${regionSlug}-r64-${slot * 2}`,
          label: `${region.name} game ${slot * 2} winner`,
        },
      });
    }

    for (let slot = 1; slot <= 2; slot += 1) {
      games.push({
        id: `${regionSlug}-s16-${slot}`,
        roundId: "sweet-16",
        roundLabel: ROUND_LABELS["sweet-16"],
        roundOrder: 3,
        title: `${region.name} Sweet 16`,
        region: region.name,
        matchupOrder: regionIndex * 2 + slot,
        sourceA: {
          type: "winner",
          gameId: `${regionSlug}-r32-${slot * 2 - 1}`,
          label: `${region.name} round of 32 winner`,
        },
        sourceB: {
          type: "winner",
          gameId: `${regionSlug}-r32-${slot * 2}`,
          label: `${region.name} round of 32 winner`,
        },
      });
    }

    games.push({
      id: `${regionSlug}-e8`,
      roundId: "elite-8",
      roundLabel: ROUND_LABELS["elite-8"],
      roundOrder: 4,
      title: `${region.name} Elite 8`,
      region: region.name,
      matchupOrder: regionIndex + 1,
      sourceA: {
        type: "winner",
        gameId: `${regionSlug}-s16-1`,
        label: `${region.name} Sweet 16 winner`,
      },
      sourceB: {
        type: "winner",
        gameId: `${regionSlug}-s16-2`,
        label: `${region.name} Sweet 16 winner`,
      },
    });
  }

  const finalFourMatchups = [
    { id: "final-four-1", left: "east-e8", right: "south-e8", order: 1, title: "East vs South" },
    { id: "final-four-2", left: "west-e8", right: "midwest-e8", order: 2, title: "West vs Midwest" },
  ] as const;

  for (const matchup of finalFourMatchups) {
    games.push({
      id: matchup.id,
      roundId: "final-four",
      roundLabel: ROUND_LABELS["final-four"],
      roundOrder: 5,
      title: matchup.title,
      matchupOrder: matchup.order,
      sourceA: {
        type: "winner",
        gameId: matchup.left,
        label: `${matchup.title.split(" vs ")[0]} region winner`,
      },
      sourceB: {
        type: "winner",
        gameId: matchup.right,
        label: `${matchup.title.split(" vs ")[1]} region winner`,
      },
    });
  }

  games.push({
    id: "national-championship",
    roundId: "national-championship",
    roundLabel: ROUND_LABELS["national-championship"],
    roundOrder: 6,
    title: "National Championship",
    matchupOrder: 1,
    sourceA: {
      type: "winner",
      gameId: "final-four-1",
      label: "Final Four winner",
    },
    sourceB: {
      type: "winner",
      gameId: "final-four-2",
      label: "Final Four winner",
    },
  });

  return games;
}

export function getRoundOrder(): BracketRoundId[] {
  return [
    "first-four",
    "round-of-64",
    "round-of-32",
    "sweet-16",
    "elite-8",
    "final-four",
    "national-championship",
  ];
}
