export const SHARE_QUERY_PARAM = "ncaa";
const STATE_VERSION = 1;

type ParseResult = {
  picks: Record<string, number> | null;
  error?: string;
};

type ExportPayload = {
  version: number;
  season: number;
  picks: Record<string, number>;
};

function orderedPicks(gameIds: string[], picks: Record<string, number>): Record<string, number> {
  const next: Record<string, number> = {};
  for (const gameId of gameIds) {
    const winnerId = picks[gameId];
    if (Number.isInteger(winnerId)) next[gameId] = winnerId;
  }
  return next;
}

export function encodeBracketState(
  season: number,
  gameIds: string[],
  picks: Record<string, number>,
): string {
  const tokens: string[] = [];
  for (const [index, gameId] of gameIds.entries()) {
    const winnerId = picks[gameId];
    if (!Number.isInteger(winnerId)) continue;
    tokens.push(`${index.toString(36)}-${winnerId.toString(36)}`);
  }
  return `${STATE_VERSION.toString(36)}~${season.toString(36)}~${tokens.join(".")}`;
}

export function decodeBracketState(
  encoded: string | null | undefined,
  season: number,
  gameIds: string[],
): ParseResult {
  if (!encoded) return { picks: null };

  const [rawVersion, rawSeason, rawPairs] = encoded.split("~");
  const version = Number.parseInt(rawVersion ?? "", 36);
  const encodedSeason = Number.parseInt(rawSeason ?? "", 36);
  if (version !== STATE_VERSION) {
    return { picks: null, error: "Unsupported shared bracket version" };
  }
  if (encodedSeason !== season) {
    return { picks: null, error: `Shared bracket is for season ${encodedSeason}, not ${season}` };
  }
  if (!rawPairs) return { picks: {} };

  const picks: Record<string, number> = {};
  const seenIndexes = new Set<number>();
  for (const pair of rawPairs.split(".")) {
    if (!pair) continue;
    const [rawIndex, rawWinnerId] = pair.split("-");
    const gameIndex = Number.parseInt(rawIndex ?? "", 36);
    const winnerId = Number.parseInt(rawWinnerId ?? "", 36);
    if (!Number.isInteger(gameIndex) || !Number.isInteger(winnerId)) {
      return { picks: null, error: "Shared bracket includes an invalid pick token" };
    }
    if (gameIndex < 0 || gameIndex >= gameIds.length) {
      return { picks: null, error: `Shared bracket references unknown game index ${rawIndex}` };
    }
    if (seenIndexes.has(gameIndex)) {
      return { picks: null, error: `Shared bracket repeats game index ${rawIndex}` };
    }
    seenIndexes.add(gameIndex);
    picks[gameIds[gameIndex]] = winnerId;
  }

  return { picks };
}

export function buildShareUrl(
  currentUrl: string,
  season: number,
  gameIds: string[],
  picks: Record<string, number>,
): string {
  const url = new URL(currentUrl);
  url.searchParams.set(SHARE_QUERY_PARAM, encodeBracketState(season, gameIds, picks));
  return url.toString();
}

export function clearShareStateFromUrl(currentUrl: string): string {
  const url = new URL(currentUrl);
  url.searchParams.delete(SHARE_QUERY_PARAM);
  return `${url.pathname}${url.search}${url.hash}`;
}

export function readShareStateFromUrl(
  currentUrl: string,
  season: number,
  gameIds: string[],
): ParseResult {
  const url = new URL(currentUrl);
  return decodeBracketState(url.searchParams.get(SHARE_QUERY_PARAM), season, gameIds);
}

export function serializeBracketExport(
  season: number,
  gameIds: string[],
  picks: Record<string, number>,
): string {
  const payload: ExportPayload = {
    version: STATE_VERSION,
    season,
    picks: orderedPicks(gameIds, picks),
  };
  return JSON.stringify(payload, null, 2);
}

export function deserializeBracketImport(
  jsonText: string,
  season: number,
  gameIds: string[],
): ParseResult {
  let parsed: unknown;
  try {
    parsed = JSON.parse(jsonText);
  } catch {
    return { picks: null, error: "Imported bracket is not valid JSON" };
  }

  if (!parsed || typeof parsed !== "object") {
    return { picks: null, error: "Imported bracket must be a JSON object" };
  }

  const payload = parsed as Partial<ExportPayload>;
  if (payload.version !== STATE_VERSION) {
    return { picks: null, error: "Imported bracket uses an unsupported version" };
  }
  if (payload.season !== season) {
    return { picks: null, error: `Imported bracket is for season ${payload.season}, not ${season}` };
  }
  if (!payload.picks || typeof payload.picks !== "object" || Array.isArray(payload.picks)) {
    return { picks: null, error: "Imported bracket must include a picks object" };
  }

  const knownGameIds = new Set(gameIds);
  const picks: Record<string, number> = {};
  for (const [gameId, rawWinnerId] of Object.entries(payload.picks)) {
    if (!knownGameIds.has(gameId)) {
      return { picks: null, error: `Imported bracket references unknown game ${gameId}` };
    }
    if (!Number.isInteger(rawWinnerId)) {
      return { picks: null, error: `Imported bracket winner for ${gameId} must be an integer team id` };
    }
    picks[gameId] = rawWinnerId;
  }

  return { picks: orderedPicks(gameIds, picks) };
}
