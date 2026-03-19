import { buildNcaaBracketGames } from "./ncaaBracket";
import { buildNcaaOddsData, NcaaOddsData } from "./ncaaOdds";
import {
  fetchHardRockComparisonData,
  HardRockComparisonData,
} from "./hardRockComparison";
import { buildScheduledNcaaMarchData } from "./marchBetting";
import {
  MatchupPrediction,
  MatchupPredictionCache,
  MarchBettingGame,
  NcaaBracketField,
  NcaaTournamentResults,
} from "./types";
import {
  validateBracketGraph,
  validateMatchupCache,
  validateNcaaField,
  validateNcaaResults,
} from "./validation";
import { readJsonFile } from "../server-data";
import { ConferenceBracketsData } from "../../components/bracket/ConferenceBrackets";

export type MarchPageProps = {
  conferenceData: ConferenceBracketsData | null;
  ncaaField: NcaaBracketField | null;
  ncaaErrors: string[];
  ncaaResults: NcaaTournamentResults | null;
  ncaaResultsErrors: string[];
  initialPredictionCache: Record<string, MatchupPrediction>;
  marchGames: MarchBettingGame[];
  ncaaData: NcaaOddsData | null;
  ncaaTorvikData: NcaaOddsData | null;
  ncaaInternalData: NcaaOddsData | null;
  hardRockReport: HardRockComparisonData | null;
};

export async function loadMarchPageProps(): Promise<MarchPageProps> {
  const conferenceRaw = readJsonFile("brackets_2026.json");
  const ncaaRaw = readJsonFile("ncaa_bracket_builder_2026.json");
  const matchupRaw = readJsonFile("ncaa_matchup_predictions_2026.json");
  const ncaaResultsRaw = readJsonFile("ncaa_results_2026.json");

  const ncaaField = ncaaRaw as NcaaBracketField | null;
  const matchupCache = matchupRaw as MatchupPredictionCache | null;
  const bracketGames = ncaaField ? buildNcaaBracketGames(ncaaField) : [];
  const ncaaResultsPayload = ncaaResultsRaw as NcaaTournamentResults | null;

  const errors = ncaaField
    ? [
        ...validateNcaaField(ncaaField).errors,
        ...validateBracketGraph(bracketGames).errors,
      ]
    : ["NCAA bracket-builder data unavailable"];

  const resultsErrors = ncaaField
    ? validateNcaaResults(ncaaResultsPayload, ncaaField, bracketGames).errors
    : [];

  const matchupValidation =
    ncaaField && matchupCache ? validateMatchupCache(matchupCache, ncaaField) : null;

  const { initialPredictionCache, marchGames } =
    ncaaField && matchupCache && matchupValidation?.valid
      ? buildScheduledNcaaMarchData(ncaaField, matchupCache, ncaaResultsPayload)
      : { initialPredictionCache: {}, marchGames: [] };

  let ncaaData: NcaaOddsData | null = null;
  let ncaaTorvikData: NcaaOddsData | null = null;
  let ncaaInternalData: NcaaOddsData | null = null;
  if (ncaaField && matchupCache && matchupValidation?.valid) {
    try {
      ncaaData = buildNcaaOddsData(
        ncaaField,
        matchupCache,
        "display_average",
        ncaaResultsPayload,
      );
      ncaaTorvikData = buildNcaaOddsData(
        ncaaField,
        matchupCache,
        "active",
        ncaaResultsPayload,
      );
      ncaaInternalData = buildNcaaOddsData(
        ncaaField,
        matchupCache,
        "team_ab_internal",
        ncaaResultsPayload,
      );
    } catch (error) {
      console.error("Failed to build NCAA odds data", error);
    }
  }

  const hardRockReport = await fetchHardRockComparisonData(ncaaData);

  return {
    conferenceData: conferenceRaw as ConferenceBracketsData | null,
    ncaaField,
    ncaaErrors: errors,
    ncaaResults: resultsErrors.length === 0 ? ncaaResultsPayload : null,
    ncaaResultsErrors: resultsErrors,
    initialPredictionCache,
    marchGames,
    ncaaData,
    ncaaTorvikData,
    ncaaInternalData,
    hardRockReport,
  };
}
