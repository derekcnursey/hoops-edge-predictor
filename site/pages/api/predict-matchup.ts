import type { NextApiRequest, NextApiResponse } from "next";
import { readJsonFile } from "../../lib/server-data";
import {
  NcaaBracketField,
  MatchupPredictionCache,
} from "../../lib/bracket/types";
import { buildPredictionFromCacheEntry, canonicalMatchupKey } from "../../lib/bracket/predictions";
import { validateMatchupCache } from "../../lib/bracket/validation";

let cache: MatchupPredictionCache | null | undefined;
let field: NcaaBracketField | null | undefined;
let cacheValidationErrors: string[] | null | undefined;

function getPredictionCache(): MatchupPredictionCache | null {
  if (cache !== undefined) return cache;
  const rawCache = readJsonFile("ncaa_matchup_predictions_2026.json");
  const rawField = readJsonFile("ncaa_bracket_builder_2026.json");
  cache = rawCache as MatchupPredictionCache | null;
  field = rawField as NcaaBracketField | null;
  const validation = validateMatchupCache(cache, field);
  cacheValidationErrors = validation.valid ? null : validation.errors;
  return cache;
}

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== "GET") {
    res.setHeader("Allow", "GET");
    return res.status(405).json({ error: "Method not allowed" });
  }

  const teamAId = Number(req.query.teamAId);
  const teamBId = Number(req.query.teamBId);
  if (!Number.isInteger(teamAId) || !Number.isInteger(teamBId) || teamAId === teamBId) {
    return res.status(400).json({ error: "Valid teamAId and teamBId are required" });
  }

  const predictionCache = getPredictionCache();
  if (!predictionCache) {
    return res.status(500).json({ error: "Prediction cache unavailable" });
  }
  if (cacheValidationErrors?.length) {
    return res.status(500).json({ error: `Prediction cache invalid: ${cacheValidationErrors[0]}` });
  }

  const entry = predictionCache.predictions[canonicalMatchupKey(teamAId, teamBId)];
  if (!entry) {
    return res.status(404).json({ error: "Matchup prediction not found" });
  }

  const payload = buildPredictionFromCacheEntry(entry, teamAId, teamBId);
  res.setHeader("Cache-Control", "public, max-age=300, stale-while-revalidate=3600");
  return res.status(200).json(payload);
}
