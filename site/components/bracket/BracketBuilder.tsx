import { ChangeEvent, useEffect, useMemo, useRef, useState } from "react";
import { displayTeam } from "../../lib/data";
import { buildNcaaBracketGames, getBracketTeams, getRoundOrder } from "../../lib/bracket/ncaaBracket";
import {
  BracketGradingSummary,
  BracketSource,
  BracketRoundId,
  BracketTeam,
  MatchupPrediction,
  NcaaBracketField,
  NcaaTournamentResults,
  NcaaValidationResult,
  ResolvedBracketGame,
} from "../../lib/bracket/types";
import {
  MAJOR_UPSET_SEED_GAP,
  buildGameComparison,
} from "../../lib/bracket/comparison";
import { canonicalMatchupKey, canonicalizePrediction, orientPrediction } from "../../lib/bracket/predictions";
import { gradeBracketPicks } from "../../lib/bracket/grading";
import { buildFinalResultsMap } from "../../lib/bracket/results";
import {
  SHARE_QUERY_PARAM,
  buildShareUrl,
  clearShareStateFromUrl,
  deserializeBracketImport,
  readShareStateFromUrl,
  serializeBracketExport,
} from "../../lib/bracket/state";
import { validateBracketGraph, validateNcaaField } from "../../lib/bracket/validation";
import BracketGame from "./BracketGame";
import BracketRound from "./BracketRound";

type BracketRoundSection = {
  label: string;
  games: ResolvedBracketGame[];
};

type BoardRoundKey = "round-of-64" | "round-of-32" | "sweet-16" | "elite-8";

type BoardMetrics = {
  boardHeight: number;
  positions: Record<BoardRoundKey, number[]>;
};

const REGION_CARD_HEIGHT = 154;
const REGION_BASE_GAP = 12;
const REGION_CONNECTOR_WIDTH = 22;
const FIRST_FOUR_RAIL_WIDTH = 320;
const REGION_LEFT_WIDTHS = [368, 344, 320, 292] as const;
const REGION_RIGHT_WIDTHS = [292, 320, 344, 368] as const;
const REGION_MAIN_BOARD_WIDTH =
  REGION_LEFT_WIDTHS[0] +
  REGION_CONNECTOR_WIDTH +
  REGION_LEFT_WIDTHS[1] +
  REGION_CONNECTOR_WIDTH +
  REGION_LEFT_WIDTHS[2] +
  REGION_CONNECTOR_WIDTH +
  REGION_LEFT_WIDTHS[3];
const REGION_LANE_WIDTH =
  FIRST_FOUR_RAIL_WIDTH +
  10 +
  REGION_MAIN_BOARD_WIDTH;
const CENTER_CARD_HEIGHT = 156;
const CENTER_GAP = 118;
const CENTER_CONNECTOR_WIDTH = 24;
const CENTER_SEMIFINAL_WIDTH = 380;
const CENTER_CHAMPIONSHIP_WIDTH = 400;
const CENTER_BOARD_WIDTH =
  CENTER_SEMIFINAL_WIDTH + CENTER_CONNECTOR_WIDTH + CENTER_CHAMPIONSHIP_WIDTH;
const CENTER_SECTION_WIDTH = CENTER_BOARD_WIDTH + 24;
const DESKTOP_BRACKET_GRID_GAP = 10;
const DESKTOP_BRACKET_WIDTH =
  REGION_LANE_WIDTH +
  CENTER_SECTION_WIDTH +
  REGION_LANE_WIDTH +
  DESKTOP_BRACKET_GRID_GAP * 2;
const FEEDER_TOP_ROW_OFFSET = -12;
const FEEDER_BOTTOM_ROW_OFFSET = 28;
const MATCHUP_API_VERSION = "2026-03-16-display-v2";
const DEREK_BRACKET_JSON = `{
  "version": 1,
  "season": 2026,
  "picks": {
    "ff1": 167,
    "ff2": 295,
    "ff3": 232,
    "ff4": 114,
    "east-r64-1": 72,
    "east-r64-2": 216,
    "east-r64-3": 279,
    "east-r64-4": 131,
    "east-r64-5": 271,
    "east-r64-6": 169,
    "east-r64-7": 312,
    "east-r64-8": 314,
    "west-r64-1": 11,
    "west-r64-2": 338,
    "west-r64-3": 355,
    "west-r64-4": 12,
    "west-r64-5": 295,
    "west-r64-6": 102,
    "west-r64-7": 168,
    "west-r64-8": 236,
    "south-r64-1": 87,
    "south-r64-2": 124,
    "south-r64-3": 336,
    "south-r64-4": 188,
    "south-r64-5": 333,
    "south-r64-6": 118,
    "south-r64-7": 253,
    "south-r64-8": 113,
    "midwest-r64-1": 170,
    "midwest-r64-2": 98,
    "midwest-r64-3": 3,
    "midwest-r64-4": 5,
    "midwest-r64-5": 292,
    "midwest-r64-6": 339,
    "midwest-r64-7": 261,
    "midwest-r64-8": 125,
    "east-r32-1": 72,
    "east-r32-2": 279,
    "east-r32-3": 169,
    "east-r32-4": 312,
    "west-r32-1": 11,
    "west-r32-2": 12,
    "west-r32-3": 102,
    "west-r32-4": 236,
    "south-r32-1": 87,
    "south-r32-2": 336,
    "south-r32-3": 118,
    "south-r32-4": 113,
    "midwest-r32-1": 170,
    "midwest-r32-2": 5,
    "midwest-r32-3": 339,
    "midwest-r32-4": 125,
    "east-s16-1": 72,
    "east-s16-2": 169,
    "west-s16-1": 11,
    "west-s16-2": 102,
    "south-s16-1": 87,
    "south-s16-2": 113,
    "midwest-s16-1": 170,
    "midwest-s16-2": 125,
    "east-e8": 169,
    "west-e8": 11,
    "south-e8": 113,
    "midwest-e8": 125,
    "final-four-1": 113,
    "final-four-2": 11,
    "national-championship": 11
  }
}`;

function sortGames(list: ResolvedBracketGame[]): ResolvedBracketGame[] {
  return [...list].sort((a, b) => a.matchupOrder - b.matchupOrder || a.roundOrder - b.roundOrder);
}

function collapsePositions(previous: number[]): number[] {
  const next: number[] = [];
  for (let index = 0; index < previous.length; index += 2) {
    const a = previous[index];
    const b = previous[index + 1];
    if (typeof a !== "number" || typeof b !== "number") break;
    next.push((a + b) / 2);
  }
  return next;
}

function buildRegionBoardMetrics(): BoardMetrics {
  const roundOf64 = Array.from({ length: 8 }, (_, index) => index * (REGION_CARD_HEIGHT + REGION_BASE_GAP));
  const roundOf32 = collapsePositions(roundOf64);
  const sweet16 = collapsePositions(roundOf32);
  const elite8 = collapsePositions(sweet16);
  return {
    boardHeight: roundOf64[roundOf64.length - 1] + REGION_CARD_HEIGHT,
    positions: {
      "round-of-64": roundOf64,
      "round-of-32": roundOf32,
      "sweet-16": sweet16,
      "elite-8": elite8,
    },
  };
}

const REGION_BOARD = buildRegionBoardMetrics();

function gradingBreakdown(summary: BracketGradingSummary): string {
  return summary.rounds
    .filter((round) => round.possibleScore > 0 || round.correct > 0 || round.incorrect > 0)
    .map((round) => `${round.roundLabel}: ${round.correct}-${round.incorrect}-${round.pending}`)
    .join(" | ");
}

function normalizeRegionName(name: string): string {
  return name.trim().toLowerCase().replace(/\s+/g, "-");
}

function canonicalRegionName(field: NcaaBracketField, target: string, fallbackIndex: number): string {
  return (
    field.regions.find((region) => normalizeRegionName(region.name) === target)?.name ??
    field.regions[fallbackIndex]?.name ??
    target
  );
}

export default function BracketBuilder({
  field,
  results,
  resultsErrors = [],
  initialPredictionCache = {},
}: {
  field: NcaaBracketField;
  results?: NcaaTournamentResults | null;
  resultsErrors?: string[];
  initialPredictionCache?: Record<string, MatchupPrediction>;
}) {
  const games = useMemo(() => buildNcaaBracketGames(field), [field]);
  const teams = useMemo(() => getBracketTeams(field), [field]);
  const orderedGames = useMemo(
    () => [...games].sort((a, b) => a.roundOrder - b.roundOrder || a.matchupOrder - b.matchupOrder),
    [games],
  );
  const teamById = useMemo(
    () => Object.fromEntries(teams.map((team) => [team.id, team])) as Record<number, BracketTeam>,
    [teams],
  );
  const storageKey = `hoops-edge-ncaa-bracket-${field.season}`;
  const gameIds = useMemo(() => orderedGames.map((game) => game.id), [orderedGames]);
  const [validation, setValidation] = useState<NcaaValidationResult | null>(null);

  const [selectedWinners, setSelectedWinners] = useState<Record<string, number>>({});
  const [predictionCache, setPredictionCache] = useState<Record<string, MatchupPrediction>>(initialPredictionCache);
  const [loadingMatchups, setLoadingMatchups] = useState<Record<string, boolean>>({});
  const [errorMatchups, setErrorMatchups] = useState<Record<string, string>>({});
  const [autoFillMode, setAutoFillMode] = useState<"seed" | "model" | null>(null);
  const [isCompactLayout, setIsCompactLayout] = useState(false);
  const [shareStatus, setShareStatus] = useState<{ tone: "success" | "error" | "info"; text: string } | null>(null);
  const [modelBracketPicks, setModelBracketPicks] = useState<Record<string, number> | null>(null);
  const [modelBracketLoading, setModelBracketLoading] = useState(false);

  const inFlight = useRef(new Set<string>());
  const hasHydrated = useRef(false);
  const hasUrlState = useRef(false);
  const importInputRef = useRef<HTMLInputElement | null>(null);

  function resolveSource(source: BracketSource, winners: Record<string, number>): BracketTeam | null {
    if (source.type === "team") return teamById[source.teamId] ?? null;
    const winnerId = winners[source.gameId];
    return winnerId ? teamById[winnerId] ?? null : null;
  }

  function sanitizeWinnerMap(winners: Record<string, number>): Record<string, number> {
    const next = { ...winners };
    // Re-walk the bracket from earliest to latest so any upstream change drops
    // invalid downstream picks automatically.
    for (const game of orderedGames) {
      const teamA = resolveSource(game.sourceA, next);
      const teamB = resolveSource(game.sourceB, next);
      const selected = next[game.id];
      if (!teamA || !teamB || (selected !== teamA.id && selected !== teamB.id)) {
        delete next[game.id];
      }
    }
    return next;
  }

  const resolvedGames = useMemo(() => {
    return [...games]
      .sort((a, b) => a.roundOrder - b.roundOrder || a.matchupOrder - b.matchupOrder)
      .map((game) => ({
        ...game,
        teamA: resolveSource(game.sourceA, selectedWinners),
        teamB: resolveSource(game.sourceB, selectedWinners),
        selectedWinnerId: selectedWinners[game.id],
      }));
  }, [games, selectedWinners, teamById]);

  const championshipGame = resolvedGames.find((game) => game.id === "national-championship");
  const champion = championshipGame?.selectedWinnerId
    ? teamById[championshipGame.selectedWinnerId] ?? null
    : null;
  const picksMade = Object.keys(selectedWinners).length;
  const totalPickableGames = games.length;
  const predictionsByGame = useMemo(() => {
    return Object.fromEntries(
      resolvedGames.map((game) => {
        const key = game.teamA && game.teamB ? canonicalMatchupKey(game.teamA.id, game.teamB.id) : "";
        const cached = key ? predictionCache[key] : undefined;
        const oriented =
          cached && game.teamA && game.teamB
            ? orientPrediction(cached, game.teamA.id, game.teamB.id)
            : undefined;
        return [game.id, oriented];
      }),
    ) as Record<string, MatchupPrediction | undefined>;
  }, [predictionCache, resolvedGames]);
  const comparisonsByGame = useMemo(() => {
    return Object.fromEntries(
      resolvedGames.map((game) => [game.id, buildGameComparison(game, predictionsByGame[game.id])]),
    );
  }, [predictionsByGame, resolvedGames]);
  const finalResultsByGame = useMemo(() => buildFinalResultsMap(results ?? null), [results]);
  const gradingActive = Object.keys(finalResultsByGame).length > 0;
  const userGrade = useMemo(
    () => (gradingActive ? gradeBracketPicks(games, selectedWinners, results ?? null, teamById) : null),
    [games, gradingActive, results, selectedWinners, teamById],
  );
  const modelGrade = useMemo(
    () => (gradingActive && modelBracketPicks ? gradeBracketPicks(games, modelBracketPicks, results ?? null, teamById) : null),
    [games, gradingActive, modelBracketPicks, results, teamById],
  );

  function applyValidatedPicks(next: Record<string, number>): boolean {
    const sanitized = sanitizeWinnerMap(next);
    if (Object.keys(sanitized).length !== Object.keys(next).length) {
      return false;
    }
    setSelectedWinners(sanitized);
    return true;
  }

  function clearUrlShareState() {
    if (typeof window === "undefined") return;
    const url = new URL(window.location.href);
    if (!url.searchParams.has(SHARE_QUERY_PARAM)) return;
    const nextUrl = clearShareStateFromUrl(window.location.href);
    window.history.replaceState({}, "", nextUrl);
    hasUrlState.current = false;
  }

  function disconnectFromSharedUrlIfNeeded() {
    if (!hasUrlState.current) return;
    clearUrlShareState();
    setShareStatus({
      tone: "info",
      text: "Detached from shared link. Current picks are now local to this browser until you share again.",
    });
  }

  useEffect(() => {
    if (typeof window === "undefined") return;
    const shared = readShareStateFromUrl(window.location.href, field.season, gameIds);
    if (shared.picks) {
      if (applyValidatedPicks(shared.picks)) {
        hasUrlState.current = true;
        hasHydrated.current = true;
        setShareStatus({
          tone: "info",
          text: "Loaded bracket from share link. URL picks override saved local picks for this session.",
        });
        window.localStorage.setItem(storageKey, JSON.stringify(shared.picks));
        return;
      }
      setShareStatus({
        tone: "error",
        text: "Shared bracket link was invalid for this field. Restored local picks if available.",
      });
    } else if (shared.error) {
      setShareStatus({
        tone: "error",
        text: shared.error,
      });
    }

    const raw = window.localStorage.getItem(storageKey);
    if (raw) {
      try {
        const parsed = JSON.parse(raw) as Record<string, number>;
        if (!applyValidatedPicks(parsed)) {
          window.localStorage.removeItem(storageKey);
        }
      } catch {
        window.localStorage.removeItem(storageKey);
      }
    }
    hasHydrated.current = true;
  }, [field.season, gameIds, storageKey]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (!hasHydrated.current) return;
    window.localStorage.setItem(storageKey, JSON.stringify(selectedWinners));
  }, [selectedWinners, storageKey]);

  useEffect(() => {
    setModelBracketPicks(null);
    setModelBracketLoading(false);
  }, [field.season, gradingActive]);

  useEffect(() => {
    setPredictionCache(initialPredictionCache);
  }, [field.season]);

  useEffect(() => {
    const validationResults = [validateNcaaField(field), validateBracketGraph(games)];
    const errors = validationResults.flatMap((item) => item.errors);
    setValidation({ valid: errors.length === 0, errors });
  }, [field, games]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const mediaQuery = window.matchMedia("(max-width: 960px)");
    const update = () => setIsCompactLayout(mediaQuery.matches);
    update();
    mediaQuery.addEventListener("change", update);
    return () => mediaQuery.removeEventListener("change", update);
  }, []);

  async function fetchPrediction(teamAId: number, teamBId: number): Promise<MatchupPrediction> {
    const key = canonicalMatchupKey(teamAId, teamBId);
    // Cache canonically by ids, then orient to the slot order currently shown.
    if (predictionCache[key]) return orientPrediction(predictionCache[key], teamAId, teamBId);

    const response = await fetch(
      `/api/predict-matchup?teamAId=${teamAId}&teamBId=${teamBId}&v=${MATCHUP_API_VERSION}`,
      { cache: "no-store" },
    );
    if (!response.ok) {
      const payload = (await response.json().catch(() => null)) as { error?: string } | null;
      throw new Error(payload?.error || "Prediction lookup failed");
    }

    const prediction = (await response.json()) as MatchupPrediction;
    setPredictionCache((current) => ({ ...current, [key]: canonicalizePrediction(prediction) }));
    setErrorMatchups((current) => {
      const next = { ...current };
      delete next[key];
      return next;
    });
    return orientPrediction(canonicalizePrediction(prediction), teamAId, teamBId);
  }

  useEffect(() => {
    let cancelled = false;
    for (const game of resolvedGames) {
      if (!game.teamA || !game.teamB) continue;
      const key = canonicalMatchupKey(game.teamA.id, game.teamB.id);
      if (predictionCache[key] || inFlight.current.has(key)) continue;

      inFlight.current.add(key);
      setLoadingMatchups((current) => ({ ...current, [key]: true }));

      fetchPrediction(game.teamA.id, game.teamB.id)
        .catch((error: Error) => {
          if (cancelled) return;
          setErrorMatchups((current) => ({ ...current, [key]: error.message }));
        })
        .finally(() => {
          inFlight.current.delete(key);
          if (cancelled) return;
          setLoadingMatchups((current) => ({ ...current, [key]: false }));
        });
    }

    return () => {
      cancelled = true;
    };
  }, [predictionCache, resolvedGames]);

  useEffect(() => {
    if (!gradingActive || modelBracketPicks || modelBracketLoading) return;

    let cancelled = false;
    setModelBracketLoading(true);

    const buildModelBracket = async () => {
      let next: Record<string, number> = {};
      for (const game of orderedGames) {
        const teamA = resolveSource(game.sourceA, next);
        const teamB = resolveSource(game.sourceB, next);
        if (!teamA || !teamB) continue;
        const prediction = await fetchPrediction(teamA.id, teamB.id);
        next = { ...next, [game.id]: prediction.modelWinnerId };
      }
      if (!cancelled) {
        setModelBracketPicks(next);
      }
    };

    buildModelBracket()
      .catch(() => {
        if (!cancelled) {
          setModelBracketPicks({});
        }
      })
      .finally(() => {
        if (!cancelled) {
          setModelBracketLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [gradingActive, modelBracketLoading, modelBracketPicks, orderedGames]);

  function handleSelectWinner(gameId: string, teamId: number) {
    disconnectFromSharedUrlIfNeeded();
    setShareStatus(null);
    setSelectedWinners((current) => sanitizeWinnerMap({ ...current, [gameId]: teamId }));
  }

  function handleReset() {
    clearUrlShareState();
    setSelectedWinners({});
    setShareStatus({ tone: "info", text: "Bracket reset. Saved local picks cleared." });
    if (typeof window !== "undefined") {
      window.localStorage.removeItem(storageKey);
    }
  }

  function pickBetterSeed(teamA: BracketTeam, teamB: BracketTeam, prediction?: MatchupPrediction): number {
    if (teamA.seed !== teamB.seed) {
      return teamA.seed < teamB.seed ? teamA.id : teamB.id;
    }
    return prediction?.modelWinnerId ?? teamA.id;
  }

  async function handleAutofill(mode: "seed" | "model") {
    setAutoFillMode(mode);
    try {
      disconnectFromSharedUrlIfNeeded();
      let next = { ...selectedWinners };
      for (const game of orderedGames) {
        const teamA = resolveSource(game.sourceA, next);
        const teamB = resolveSource(game.sourceB, next);
        if (!teamA || !teamB) continue;
        const prediction = await fetchPrediction(teamA.id, teamB.id);
        const winnerId = mode === "model" ? prediction.modelWinnerId : pickBetterSeed(teamA, teamB, prediction);
        next = sanitizeWinnerMap({ ...next, [game.id]: winnerId });
      }

      setSelectedWinners(next);
      setShareStatus({
        tone: "success",
        text: mode === "model" ? "Bracket filled with model picks." : "Bracket filled with better-seed picks.",
      });
      if (typeof window !== "undefined") {
        window.localStorage.setItem(storageKey, JSON.stringify(next));
      }
    } finally {
      setAutoFillMode(null);
    }
  }

  async function handleCopyShareLink() {
    if (typeof window === "undefined") return;
    const shareUrl = buildShareUrl(window.location.href, field.season, gameIds, selectedWinners);
    try {
      await navigator.clipboard.writeText(shareUrl);
      setShareStatus({ tone: "success", text: "Share link copied to clipboard." });
    } catch {
      window.prompt("Copy this bracket link", shareUrl);
      setShareStatus({ tone: "info", text: "Share link ready to copy." });
    }
  }

  function handleExportJson() {
    if (typeof window === "undefined") return;
    const json = serializeBracketExport(field.season, gameIds, selectedWinners);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `hoops-edge-ncaa-bracket-${field.season}.json`;
    link.click();
    URL.revokeObjectURL(url);
    setShareStatus({ tone: "success", text: "Bracket JSON exported." });
  }

  async function handleImportJson(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      const text = await file.text();
      const parsed = deserializeBracketImport(text, field.season, gameIds);
      if (!parsed.picks) {
        setShareStatus({ tone: "error", text: parsed.error ?? "Imported bracket is invalid." });
        return;
      }

      disconnectFromSharedUrlIfNeeded();
      if (!applyValidatedPicks(parsed.picks)) {
        setShareStatus({
          tone: "error",
          text: "Imported bracket has invalid downstream winners for the current field.",
        });
        return;
      }

      setShareStatus({ tone: "success", text: `Imported ${Object.keys(parsed.picks).length} bracket picks.` });
    } catch {
      setShareStatus({ tone: "error", text: "Failed to import bracket file." });
    } finally {
      event.target.value = "";
    }
  }

  function handleLoadDerekBracket() {
    const parsed = deserializeBracketImport(DEREK_BRACKET_JSON, field.season, gameIds);
    if (!parsed.picks) {
      setShareStatus({ tone: "error", text: parsed.error ?? "Derek's bracket preset is invalid." });
      return;
    }

    disconnectFromSharedUrlIfNeeded();
    if (!applyValidatedPicks(parsed.picks)) {
      setShareStatus({
        tone: "error",
        text: "Derek's bracket preset has invalid downstream winners for the current field.",
      });
      return;
    }

    setShareStatus({ tone: "success", text: "Loaded Derek's bracket." });
    if (typeof window !== "undefined") {
      window.localStorage.setItem(storageKey, JSON.stringify(parsed.picks));
    }
  }

  const roundGames = useMemo(() => {
    const grouped = new Map<string, ResolvedBracketGame[]>();
    for (const roundId of getRoundOrder()) grouped.set(roundId, []);
    for (const game of resolvedGames) {
      const current = grouped.get(game.roundId) ?? [];
      current.push(game);
      grouped.set(game.roundId, current);
    }
    return grouped;
  }, [resolvedGames]);
  const regionRoundGames = useMemo(() => {
    const grouped = new Map<string, Map<BracketRoundId, ResolvedBracketGame[]>>();
    for (const region of field.regions) {
      const roundMap = new Map<BracketRoundId, ResolvedBracketGame[]>();
      for (const roundId of getRoundOrder()) {
        roundMap.set(roundId, []);
      }
      grouped.set(region.name, roundMap);
    }

    for (const game of resolvedGames) {
      if (!game.region) continue;
      const roundMap = grouped.get(game.region);
      if (!roundMap) continue;
      roundMap.set(game.roundId, [...(roundMap.get(game.roundId) ?? []), game]);
    }

    return grouped;
  }, [field.regions, resolvedGames]);
  const laneRegions = useMemo(
    () => ({
      topLeft: canonicalRegionName(field, "east", 0),
      bottomLeft: canonicalRegionName(field, "west", 1),
      topRight: canonicalRegionName(field, "south", 2),
      bottomRight: canonicalRegionName(field, "midwest", 3),
    }),
    [field],
  );

  function buildRoundSectionState(roundList: ResolvedBracketGame[]) {
    const predictions = Object.fromEntries(roundList.map((game) => [game.id, predictionsByGame[game.id]])) as Record<
      string,
      MatchupPrediction | undefined
    >;
    const comparisons = Object.fromEntries(roundList.map((game) => [game.id, comparisonsByGame[game.id]]));
    const grading = Object.fromEntries(roundList.map((game) => [game.id, userGrade?.byGame[game.id]]));
    const loadingGames = Object.fromEntries(
      roundList.map((game) => {
        const key = game.teamA && game.teamB ? canonicalMatchupKey(game.teamA.id, game.teamB.id) : "";
        return [game.id, key ? loadingMatchups[key] : false];
      }),
    ) as Record<string, boolean | undefined>;
    const errorGames = Object.fromEntries(
      roundList.map((game) => {
        const key = game.teamA && game.teamB ? canonicalMatchupKey(game.teamA.id, game.teamB.id) : "";
        return [game.id, key ? errorMatchups[key] : undefined];
      }),
    ) as Record<string, string | undefined>;

    return { predictions, comparisons, grading, loadingGames, errorGames };
  }

  function renderRoundSection(section: BracketRoundSection, minWidth: number) {
    if (!section.games.length) return null;
    const roundState = buildRoundSectionState(section.games);
    return (
      <BracketRound
        key={`${section.label}-${section.games[0]?.id ?? "empty"}`}
        label={section.label}
        games={section.games}
        predictions={roundState.predictions}
        comparisons={roundState.comparisons}
        grading={roundState.grading}
        loadingGames={roundState.loadingGames}
        errorGames={roundState.errorGames}
        onSelectWinner={handleSelectWinner}
        compact={isCompactLayout}
        stickyTitle={false}
        dense
        minWidth={minWidth}
      />
    );
  }

  function renderConnectorGutter(
    outerPositions: number[],
    innerPositions: number[],
    boardHeight: number,
    outerOn: "left" | "right",
    keyPrefix: string,
  ) {
    const spineX = outerOn === "left" ? Math.round(REGION_CONNECTOR_WIDTH * 0.58) : Math.round(REGION_CONNECTOR_WIDTH * 0.42);
    return (
      <div style={{ position: "relative", height: boardHeight }}>
        {innerPositions.map((innerTop, index) => {
          const outerTopA = outerPositions[index * 2];
          const outerTopB = outerPositions[index * 2 + 1];
          if (typeof outerTopA !== "number" || typeof outerTopB !== "number") return null;

          const outerCenterA = outerTopA + REGION_CARD_HEIGHT / 2;
          const outerCenterB = outerTopB + REGION_CARD_HEIGHT / 2;
          const innerCenter = innerTop + REGION_CARD_HEIGHT / 2;
          const outerSpanStart = outerOn === "left" ? 0 : spineX;
          const outerSpanWidth = outerOn === "left" ? spineX : REGION_CONNECTOR_WIDTH - spineX;
          const innerSpanStart = outerOn === "left" ? spineX : 0;
          const innerSpanWidth = outerOn === "left" ? REGION_CONNECTOR_WIDTH - spineX : spineX;

          return (
            <div key={`${keyPrefix}-${index}`}>
              <div
                style={{
                  position: "absolute",
                  top: outerCenterA - 1,
                  left: outerSpanStart,
                  width: outerSpanWidth,
                  height: 2,
                  borderRadius: 999,
                  background: "#cbd5e1",
                }}
              />
              <div
                style={{
                  position: "absolute",
                  top: outerCenterB - 1,
                  left: outerSpanStart,
                  width: outerSpanWidth,
                  height: 2,
                  borderRadius: 999,
                  background: "#cbd5e1",
                }}
              />
              <div
                style={{
                  position: "absolute",
                  top: Math.min(outerCenterA, outerCenterB),
                  left: spineX - 1,
                  width: 2,
                  height: Math.abs(outerCenterB - outerCenterA),
                  borderRadius: 999,
                  background: "#cbd5e1",
                }}
              />
              <div
                style={{
                  position: "absolute",
                  top: innerCenter - 1,
                  left: innerSpanStart,
                  width: innerSpanWidth,
                  height: 2,
                  borderRadius: 999,
                  background: "#cbd5e1",
                }}
              />
            </div>
          );
        })}
      </div>
    );
  }

  function renderBoardGameColumn(
    gamesForColumn: ResolvedBracketGame[],
    positions: number[],
    boardHeight: number,
    width: number,
    keyPrefix: string,
  ) {
    const roundState = buildRoundSectionState(gamesForColumn);
    return (
      <div style={{ position: "relative", height: boardHeight, width }}>
        {gamesForColumn.map((game, index) => (
          <div
            key={`${keyPrefix}-${game.id}`}
            style={{
              position: "absolute",
              top: positions[index] ?? 0,
              left: 0,
              right: 0,
            }}
          >
            <BracketGame
              game={game}
              prediction={roundState.predictions[game.id]}
              comparison={roundState.comparisons[game.id]}
              grading={roundState.grading[game.id]}
              predictionLoading={roundState.loadingGames[game.id]}
              predictionError={roundState.errorGames[game.id]}
              onSelectWinner={handleSelectWinner}
              compact
              fixedHeight={REGION_CARD_HEIGHT}
            />
          </div>
        ))}
      </div>
    );
  }

  function renderFirstFourRail(
    gamesForRail: ResolvedBracketGame[],
    roundOf64Games: ResolvedBracketGame[],
    side: "left" | "right",
    boardHeight: number,
  ) {
    if (!gamesForRail.length) return null;

    const alignedEntries = gamesForRail.map((game, index) => {
      const targetIndex = roundOf64Games.findIndex(
        (targetGame) =>
          (targetGame.sourceA.type === "winner" && targetGame.sourceA.gameId === game.id) ||
          (targetGame.sourceB.type === "winner" && targetGame.sourceB.gameId === game.id),
      );
      const targetGame = targetIndex >= 0 ? roundOf64Games[targetIndex] : null;
      const feedOffset =
        targetGame?.sourceA.type === "winner" && targetGame.sourceA.gameId === game.id
          ? FEEDER_TOP_ROW_OFFSET
          : FEEDER_BOTTOM_ROW_OFFSET;
      return {
        game,
        top:
          targetIndex >= 0
            ? (REGION_BOARD.positions["round-of-64"][targetIndex] ?? 0) + feedOffset
            : index * (REGION_CARD_HEIGHT + 12),
      };
    });

    return (
      <div
        style={{
          width: FIRST_FOUR_RAIL_WIDTH,
          position: "relative",
          height: boardHeight,
          alignSelf: "start",
        }}
      >
        <div
          style={{
            padding: "4px 6px",
            borderRadius: 8,
            background: "#eef2ff",
            color: "#475569",
            fontFamily: "'IBM Plex Mono', monospace",
            fontSize: 10,
            textAlign: side === "left" ? "right" : "left",
          }}
        >
          First Four
        </div>
        {alignedEntries.map(({ game, top }) => {
          const state = buildRoundSectionState([game]);
          return (
            <div
              key={game.id}
              style={{
                position: "absolute",
                top,
                left: 0,
                right: 0,
              }}
            >
              <BracketGame
                game={game}
                prediction={state.predictions[game.id]}
                comparison={state.comparisons[game.id]}
                grading={state.grading[game.id]}
                predictionLoading={state.loadingGames[game.id]}
                predictionError={state.errorGames[game.id]}
                onSelectWinner={handleSelectWinner}
                compact
                fixedHeight={REGION_CARD_HEIGHT}
              />
            </div>
          );
        })}
      </div>
    );
  }

  function renderRegionLane(regionName: string, side: "left" | "right") {
    const roundMap = regionRoundGames.get(regionName);
    if (!roundMap) return null;

    const firstFourGames = sortGames(roundMap.get("first-four") ?? []);
    const roundOf64Games = sortGames(roundMap.get("round-of-64") ?? []);
    const roundOf32Games = sortGames(roundMap.get("round-of-32") ?? []);
    const sweet16Games = sortGames(roundMap.get("sweet-16") ?? []);
    const elite8Games = sortGames(roundMap.get("elite-8") ?? []);

    if (!isCompactLayout) {
      const boardHeight = REGION_BOARD.boardHeight;
      const columns =
        side === "left"
          ? [
              renderBoardGameColumn(
                roundOf64Games,
                REGION_BOARD.positions["round-of-64"],
                boardHeight,
                REGION_LEFT_WIDTHS[0],
                `${regionName}-r64`,
              ),
              renderConnectorGutter(
                REGION_BOARD.positions["round-of-64"],
                REGION_BOARD.positions["round-of-32"],
                boardHeight,
                "left",
                `${regionName}-c1`,
              ),
              renderBoardGameColumn(
                roundOf32Games,
                REGION_BOARD.positions["round-of-32"],
                boardHeight,
                REGION_LEFT_WIDTHS[1],
                `${regionName}-r32`,
              ),
              renderConnectorGutter(
                REGION_BOARD.positions["round-of-32"],
                REGION_BOARD.positions["sweet-16"],
                boardHeight,
                "left",
                `${regionName}-c2`,
              ),
              renderBoardGameColumn(
                sweet16Games,
                REGION_BOARD.positions["sweet-16"],
                boardHeight,
                REGION_LEFT_WIDTHS[2],
                `${regionName}-s16`,
              ),
              renderConnectorGutter(
                REGION_BOARD.positions["sweet-16"],
                REGION_BOARD.positions["elite-8"],
                boardHeight,
                "left",
                `${regionName}-c3`,
              ),
              renderBoardGameColumn(
                elite8Games,
                REGION_BOARD.positions["elite-8"],
                boardHeight,
                REGION_LEFT_WIDTHS[3],
                `${regionName}-e8`,
              ),
            ]
          : [
              renderBoardGameColumn(
                elite8Games,
                REGION_BOARD.positions["elite-8"],
                boardHeight,
                REGION_RIGHT_WIDTHS[0],
                `${regionName}-e8`,
              ),
              renderConnectorGutter(
                REGION_BOARD.positions["sweet-16"],
                REGION_BOARD.positions["elite-8"],
                boardHeight,
                "right",
                `${regionName}-c3`,
              ),
              renderBoardGameColumn(
                sweet16Games,
                REGION_BOARD.positions["sweet-16"],
                boardHeight,
                REGION_RIGHT_WIDTHS[1],
                `${regionName}-s16`,
              ),
              renderConnectorGutter(
                REGION_BOARD.positions["round-of-32"],
                REGION_BOARD.positions["sweet-16"],
                boardHeight,
                "right",
                `${regionName}-c2`,
              ),
              renderBoardGameColumn(
                roundOf32Games,
                REGION_BOARD.positions["round-of-32"],
                boardHeight,
                REGION_RIGHT_WIDTHS[2],
                `${regionName}-r32`,
              ),
              renderConnectorGutter(
                REGION_BOARD.positions["round-of-64"],
                REGION_BOARD.positions["round-of-32"],
                boardHeight,
                "right",
                `${regionName}-c1`,
              ),
              renderBoardGameColumn(
                roundOf64Games,
                REGION_BOARD.positions["round-of-64"],
                boardHeight,
                REGION_RIGHT_WIDTHS[3],
                `${regionName}-r64`,
              ),
            ];

      return (
        <section
          style={{
            position: "relative",
            borderRadius: 14,
            border: "1px solid #e2e8f0",
            background: "#f8fafc",
            padding: 10,
            boxShadow: "inset 0 1px 0 rgba(255,255,255,0.7)",
            width: REGION_MAIN_BOARD_WIDTH,
            justifySelf: side === "left" ? "end" : "start",
            overflow: "visible",
          }}
        >
          <div
            style={{
              marginBottom: 8,
              display: "flex",
              justifyContent: "space-between",
              gap: 8,
              alignItems: "baseline",
            }}
          >
            <h2
              style={{
                fontSize: 13,
                fontWeight: 700,
                letterSpacing: "-0.02em",
                margin: 0,
                color: "#0f172a",
              }}
            >
              {regionName} Region
            </h2>
            <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 10, color: "#94a3b8" }}>
              {side === "left" ? "Rounds move inward →" : "← Rounds move inward"}
            </span>
          </div>

          <div style={{ position: "relative", width: REGION_MAIN_BOARD_WIDTH, height: boardHeight }}>
            {firstFourGames.length ? (
              <div
                style={{
                  position: "absolute",
                  top: 0,
                  [side === "left" ? "left" : "right"]: -(FIRST_FOUR_RAIL_WIDTH + 10),
                }}
              >
                {renderFirstFourRail(firstFourGames, roundOf64Games, side, boardHeight)}
              </div>
            ) : null}
            <div
              style={{
                display: "grid",
                gridTemplateColumns:
                  side === "left"
                    ? `${REGION_LEFT_WIDTHS[0]}px ${REGION_CONNECTOR_WIDTH}px ${REGION_LEFT_WIDTHS[1]}px ${REGION_CONNECTOR_WIDTH}px ${REGION_LEFT_WIDTHS[2]}px ${REGION_CONNECTOR_WIDTH}px ${REGION_LEFT_WIDTHS[3]}px`
                    : `${REGION_RIGHT_WIDTHS[0]}px ${REGION_CONNECTOR_WIDTH}px ${REGION_RIGHT_WIDTHS[1]}px ${REGION_CONNECTOR_WIDTH}px ${REGION_RIGHT_WIDTHS[2]}px ${REGION_CONNECTOR_WIDTH}px ${REGION_RIGHT_WIDTHS[3]}px`,
                alignItems: "start",
                justifyContent: side === "left" ? "start" : "end",
              }}
            >
              {columns.map((column, index) => (
                <div key={`${regionName}-${side}-desktop-${index}`}>{column}</div>
              ))}
            </div>
          </div>
        </section>
      );
    }

    const sectionsByColumn: BracketRoundSection[][] =
      side === "left"
        ? [
            [
              { label: "First Four", games: firstFourGames },
              { label: "Round of 64", games: roundOf64Games },
            ],
            [{ label: "Round of 32", games: roundOf32Games }],
            [{ label: "Sweet 16", games: sweet16Games }],
            [{ label: "Elite 8", games: elite8Games }],
          ]
        : [
            [{ label: "Elite 8", games: elite8Games }],
            [{ label: "Sweet 16", games: sweet16Games }],
            [{ label: "Round of 32", games: roundOf32Games }],
            [
              { label: "Round of 64", games: roundOf64Games },
              { label: "First Four", games: firstFourGames },
            ],
          ];

    return (
      <section
        style={{
          borderRadius: 12,
          border: "1px solid #e2e8f0",
          background: "#f8fafc",
          padding: 10,
        }}
      >
        <div
          style={{
            marginBottom: 8,
            display: "flex",
            justifyContent: "space-between",
            gap: 8,
            alignItems: "baseline",
          }}
        >
          <h2
            style={{
              fontSize: 14,
              fontWeight: 700,
              letterSpacing: "-0.02em",
              margin: 0,
              color: "#0f172a",
            }}
          >
            {regionName} Region
          </h2>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: isCompactLayout
              ? "repeat(2, minmax(0, 1fr))"
              : side === "left"
                ? "minmax(138px,1.18fr) repeat(2, minmax(128px,1fr)) minmax(134px,0.94fr)"
                : "minmax(134px,0.94fr) repeat(2, minmax(128px,1fr)) minmax(138px,1.18fr)",
            gap: 8,
            alignItems: "start",
          }}
        >
          {sectionsByColumn.map((sections, index) => (
            <div key={`${regionName}-${side}-${index}`} style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {sections.map((section) => renderRoundSection(section, 124))}
            </div>
          ))}
        </div>
      </section>
    );
  }

  function renderCenterBoard() {
    const finalFourGames = sortGames(roundGames.get("final-four") ?? []);
    const championshipGames = sortGames(roundGames.get("national-championship") ?? []);
    const semifinalState = buildRoundSectionState(finalFourGames);
    const championshipState = buildRoundSectionState(championshipGames);
    const semifinalPositions = [0, CENTER_CARD_HEIGHT + CENTER_GAP];
    const championshipPositions = collapsePositions(semifinalPositions);
    const boardHeight = semifinalPositions[semifinalPositions.length - 1] + CENTER_CARD_HEIGHT;
    const spineX = Math.round(CENTER_CONNECTOR_WIDTH * 0.58);

    return (
      <section
        style={{
          gridColumn: 2,
          gridRow: "1 / span 2",
          alignSelf: "stretch",
          borderRadius: 14,
          border: "1px solid #e2e8f0",
          background: "#f8fafc",
          padding: 12,
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          gap: 10,
        }}
      >
        <div>
          <h2
            style={{
              fontSize: 14,
              fontWeight: 700,
              letterSpacing: "-0.02em",
              margin: "0 0 4px",
              color: "#0f172a",
              textAlign: "center",
            }}
          >
            Final Four
          </h2>
          <div
            style={{
              fontFamily: "'IBM Plex Mono', monospace",
              fontSize: 10,
              color: "#94a3b8",
              textAlign: "center",
            }}
          >
            Semifinals feed the title game in the center
          </div>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: `${CENTER_SEMIFINAL_WIDTH}px ${CENTER_CONNECTOR_WIDTH}px ${CENTER_CHAMPIONSHIP_WIDTH}px`,
            justifyContent: "center",
            alignItems: "start",
          }}
        >
          <div style={{ position: "relative", height: boardHeight, width: CENTER_SEMIFINAL_WIDTH }}>
            {finalFourGames.map((game, index) => (
              <div
                key={`final-four-${game.id}`}
                style={{
                  position: "absolute",
                  top: semifinalPositions[index] ?? 0,
                  left: 0,
                  right: 0,
                }}
              >
                <BracketGame
                  game={game}
                  prediction={semifinalState.predictions[game.id]}
                  comparison={semifinalState.comparisons[game.id]}
                  grading={semifinalState.grading[game.id]}
                  predictionLoading={semifinalState.loadingGames[game.id]}
                  predictionError={semifinalState.errorGames[game.id]}
                  onSelectWinner={handleSelectWinner}
                  compact
                  fixedHeight={CENTER_CARD_HEIGHT}
                />
              </div>
            ))}
          </div>

          <div style={{ position: "relative", height: boardHeight }}>
            {championshipPositions.map((championshipTop, index) => {
              const semifinalTopA = semifinalPositions[index * 2];
              const semifinalTopB = semifinalPositions[index * 2 + 1];
              if (typeof semifinalTopA !== "number" || typeof semifinalTopB !== "number") return null;
              const semifinalCenterA = semifinalTopA + CENTER_CARD_HEIGHT / 2;
              const semifinalCenterB = semifinalTopB + CENTER_CARD_HEIGHT / 2;
              const championshipCenter = championshipTop + CENTER_CARD_HEIGHT / 2;

              return (
                <div key={`center-connector-${index}`}>
                  <div
                    style={{
                      position: "absolute",
                      top: semifinalCenterA - 1,
                      left: 0,
                      width: spineX,
                      height: 2,
                      borderRadius: 999,
                      background: "#cbd5e1",
                    }}
                  />
                  <div
                    style={{
                      position: "absolute",
                      top: semifinalCenterB - 1,
                      left: 0,
                      width: spineX,
                      height: 2,
                      borderRadius: 999,
                      background: "#cbd5e1",
                    }}
                  />
                  <div
                    style={{
                      position: "absolute",
                      top: Math.min(semifinalCenterA, semifinalCenterB),
                      left: spineX - 1,
                      width: 2,
                      height: Math.abs(semifinalCenterB - semifinalCenterA),
                      borderRadius: 999,
                      background: "#cbd5e1",
                    }}
                  />
                  <div
                    style={{
                      position: "absolute",
                      top: championshipCenter - 1,
                      left: spineX,
                      width: CENTER_CONNECTOR_WIDTH - spineX,
                      height: 2,
                      borderRadius: 999,
                      background: "#cbd5e1",
                    }}
                  />
                </div>
              );
            })}
          </div>

          <div style={{ position: "relative", height: boardHeight, width: CENTER_CHAMPIONSHIP_WIDTH }}>
            {championshipGames.map((game, index) => (
              <div
                key={`national-championship-${game.id}`}
                style={{
                  position: "absolute",
                  top: championshipPositions[index] ?? 0,
                  left: 0,
                  right: 0,
                }}
              >
                <BracketGame
                  game={game}
                  prediction={championshipState.predictions[game.id]}
                  comparison={championshipState.comparisons[game.id]}
                  grading={championshipState.grading[game.id]}
                  predictionLoading={championshipState.loadingGames[game.id]}
                  predictionError={championshipState.errorGames[game.id]}
                  onSelectWinner={handleSelectWinner}
                  compact
                  fixedHeight={CENTER_CARD_HEIGHT}
                />
              </div>
            ))}
          </div>
        </div>
      </section>
    );
  }

  if (validation && !validation.valid) {
    return (
      <section
        style={{
          padding: 20,
          borderRadius: 12,
          border: "1px solid #fecaca",
          background: "#fef2f2",
          color: "#991b1b",
        }}
      >
        <div style={{ fontWeight: 700, marginBottom: 8 }}>NCAA builder data failed validation</div>
        <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 12, lineHeight: 1.6 }}>
          {validation.errors.slice(0, 8).join(" | ")}
        </div>
      </section>
    );
  }

  return (
    <section
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 18,
      }}
    >
      <div
        style={{
          background: "#ffffff",
          border: "1px solid #e2e8f0",
          borderRadius: 12,
          padding: 14,
          boxShadow: "0 1px 3px rgba(0, 0, 0, 0.04)",
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            gap: 12,
            flexWrap: "wrap",
            marginBottom: 10,
          }}
        >
          <div>
            <h1
              style={{
                fontSize: 22,
                fontWeight: 700,
                letterSpacing: "-0.02em",
                margin: "0 0 4px",
                color: "#0f172a",
              }}
            >
              NCAA Tournament Bracket Builder
            </h1>
            <div
              style={{
                fontFamily: "'IBM Plex Mono', monospace",
                fontSize: 11,
                color: "#64748b",
              }}
            >
              Manual picks with live Hoops Edge matchup projections on every resolved game
            </div>
          </div>

          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            <button
              type="button"
              onClick={handleCopyShareLink}
              style={{
                fontFamily: "'IBM Plex Mono', monospace",
                padding: "8px 11px",
                borderRadius: 8,
                border: "1px solid #cbd5e1",
                background: "#ffffff",
                color: "#0f172a",
                cursor: "pointer",
              }}
            >
              Copy share link
            </button>
            <button
              type="button"
              onClick={handleExportJson}
              style={{
                fontFamily: "'IBM Plex Mono', monospace",
                padding: "8px 11px",
                borderRadius: 8,
                border: "1px solid #cbd5e1",
                background: "#ffffff",
                color: "#0f172a",
                cursor: "pointer",
              }}
            >
              Export JSON
            </button>
            <button
              type="button"
              onClick={() => importInputRef.current?.click()}
              style={{
                fontFamily: "'IBM Plex Mono', monospace",
                padding: "8px 11px",
                borderRadius: 8,
                border: "1px solid #cbd5e1",
                background: "#ffffff",
                color: "#0f172a",
                cursor: "pointer",
              }}
            >
              Import JSON
            </button>
            <button
              type="button"
              onClick={handleLoadDerekBracket}
              style={{
                fontFamily: "'IBM Plex Mono', monospace",
                padding: "8px 11px",
                borderRadius: 8,
                border: "1px solid #cbd5e1",
                background: "#ffffff",
                color: "#0f172a",
                cursor: "pointer",
              }}
            >
              Derek&apos;s bracket
            </button>
            <button
              type="button"
              onClick={() => handleAutofill("model")}
              disabled={autoFillMode !== null}
              style={{
                fontFamily: "'IBM Plex Mono', monospace",
                padding: "8px 11px",
                borderRadius: 8,
                border: "1px solid #0f172a",
                background: "#0f172a",
                color: "#ffffff",
                cursor: autoFillMode ? "wait" : "pointer",
              }}
            >
              {autoFillMode === "model" ? "Auto-filling..." : "Auto-fill model picks"}
            </button>
            <button
              type="button"
              onClick={() => handleAutofill("seed")}
              disabled={autoFillMode !== null}
              style={{
                fontFamily: "'IBM Plex Mono', monospace",
                padding: "8px 11px",
                borderRadius: 8,
                border: "1px solid #cbd5e1",
                background: "#ffffff",
                color: "#0f172a",
                cursor: autoFillMode ? "wait" : "pointer",
              }}
            >
              {autoFillMode === "seed" ? "Auto-filling..." : "Auto-fill better seeds"}
            </button>
            <button
              type="button"
              onClick={handleReset}
              style={{
                fontFamily: "'IBM Plex Mono', monospace",
                padding: "8px 11px",
                borderRadius: 8,
                border: "1px solid #cbd5e1",
                background: "#ffffff",
                color: "#0f172a",
                cursor: "pointer",
              }}
            >
              Reset + clear saved
            </button>
            <input
              ref={importInputRef}
              type="file"
              accept="application/json,.json"
              onChange={handleImportJson}
              style={{ display: "none" }}
            />
          </div>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
            gap: 10,
            marginBottom: 10,
          }}
        >
          <div
            style={{
              borderRadius: 9,
              border: "1px solid #e2e8f0",
              background: "#f8fafc",
              padding: 10,
            }}
          >
            <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "#64748b", marginBottom: 4 }}>
              Champion
            </div>
            <div style={{ fontSize: 15, fontWeight: 700, color: "#0f172a" }}>
              {champion ? displayTeam(champion.name) : "Awaiting picks"}
            </div>
            {champion ? (
              <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "#475569", marginTop: 4 }}>
                ({champion.seed}) {champion.region} | Rank {champion.rank}
              </div>
            ) : null}
          </div>

          <div
            style={{
              borderRadius: 9,
              border: "1px solid #e2e8f0",
              background: "#f8fafc",
              padding: 10,
            }}
          >
            <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "#64748b", marginBottom: 4 }}>
              Progress
            </div>
            <div style={{ fontSize: 14, fontWeight: 600, color: "#0f172a" }}>
              {picksMade}/{totalPickableGames} picks made
            </div>
            <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "#475569", marginTop: 4 }}>
              Picks save locally until reset
            </div>
          </div>

          <div
            style={{
              borderRadius: 9,
              border: "1px solid #e2e8f0",
              background: "#f8fafc",
              padding: 10,
            }}
          >
            <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "#64748b", marginBottom: 4 }}>
              Details
            </div>
            <div style={{ fontSize: 14, fontWeight: 600, color: "#0f172a" }}>
              Spread and ML% live in each team tile
            </div>
            <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "#475569", marginTop: 4 }}>
              Adj Pace / OE / DE / Net live in the info modal
            </div>
          </div>

          {gradingActive && userGrade ? (
            <div
              style={{
                borderRadius: 9,
                border: "1px solid #dcfce7",
                background: "#f0fdf4",
                padding: 10,
              }}
            >
              <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "#166534", marginBottom: 4 }}>
                Live Grading
              </div>
              <div style={{ fontSize: 14, fontWeight: 600, color: "#0f172a" }}>
                {userGrade.correct} correct • {userGrade.incorrect} missed • {userGrade.pending} pending
              </div>
              <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "#166534", marginTop: 4 }}>
                Score {userGrade.score}/{userGrade.possibleScore}
              </div>
              <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "#166534", marginTop: 4 }}>
                {gradingBreakdown(userGrade) || "Awaiting scored rounds"}
              </div>
            </div>
          ) : null}

          {gradingActive ? (
            <div
              style={{
                borderRadius: 9,
                border: "1px solid #e2e8f0",
                background: "#f8fafc",
                padding: 10,
              }}
            >
              <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "#64748b", marginBottom: 4 }}>
                Vs Model
              </div>
              {modelGrade ? (
                <>
                  <div style={{ fontSize: 14, fontWeight: 600, color: "#0f172a" }}>
                    You {userGrade?.score ?? 0} pts • Model {modelGrade.score} pts
                  </div>
                  <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "#475569", marginTop: 4 }}>
                    You {userGrade?.correct ?? 0} correct • Model {modelGrade.correct} correct
                  </div>
                </>
              ) : (
                <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "#475569" }}>
                  {modelBracketLoading ? "Building model bracket..." : "Model grading unavailable"}
                </div>
              )}
            </div>
          ) : null}

          <div
            style={{
              borderRadius: 9,
              border: "1px solid #e2e8f0",
              background: "#f8fafc",
              padding: 10,
            }}
          >
            <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "#64748b", marginBottom: 4 }}>
              Legend
            </div>
            <div style={{ fontSize: 14, fontWeight: 600, color: "#0f172a" }}>
              Dark = your pick, Blue = favorite, Amber = upset/fade
            </div>
            <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: "#475569", marginTop: 4 }}>
              Major upset = pick seeded {MAJOR_UPSET_SEED_GAP}+ lines worse
            </div>
          </div>
        </div>

        <div
          style={{
            fontFamily: "'IBM Plex Mono', monospace",
            fontSize: 11,
            lineHeight: 1.6,
            color: "#64748b",
            marginTop: -2,
          }}
        >
          {field.note}
        </div>
        {resultsErrors.length ? (
          <div
            style={{
              marginTop: 10,
              padding: "8px 10px",
              borderRadius: 8,
              border: "1px solid #fde68a",
              background: "#fffbeb",
              color: "#92400e",
              fontFamily: "'IBM Plex Mono', monospace",
              fontSize: 11,
              lineHeight: 1.5,
            }}
          >
            Results grading unavailable: {resultsErrors[0]}
          </div>
        ) : null}
        {shareStatus ? (
          <div
            style={{
              marginTop: 10,
              padding: "8px 10px",
              borderRadius: 8,
              border: `1px solid ${
                shareStatus.tone === "error" ? "#fecaca" : shareStatus.tone === "success" ? "#bbf7d0" : "#cbd5e1"
              }`,
              background:
                shareStatus.tone === "error" ? "#fef2f2" : shareStatus.tone === "success" ? "#f0fdf4" : "#f8fafc",
              color: shareStatus.tone === "error" ? "#991b1b" : shareStatus.tone === "success" ? "#166534" : "#475569",
              fontFamily: "'IBM Plex Mono', monospace",
              fontSize: 11,
              lineHeight: 1.5,
            }}
          >
            {shareStatus.text}
          </div>
        ) : null}
      </div>

      <div
        style={{
          paddingBottom: 8,
        }}
      >
        {isCompactLayout ? (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 14,
            }}
          >
            {renderRegionLane(laneRegions.topLeft, "left")}
            {renderRegionLane(laneRegions.bottomLeft, "left")}
            <section
              style={{
                borderRadius: 12,
                border: "1px solid #e2e8f0",
                background: "#f8fafc",
                padding: 12,
                display: "flex",
                flexDirection: "column",
                gap: 10,
              }}
            >
              {renderRoundSection({ label: "Final Four", games: roundGames.get("final-four") ?? [] }, 150)}
              {renderRoundSection({ label: "National Championship", games: roundGames.get("national-championship") ?? [] }, 150)}
            </section>
            {renderRegionLane(laneRegions.topRight, "right")}
            {renderRegionLane(laneRegions.bottomRight, "right")}
          </div>
        ) : (
          <div
            style={{
              display: "grid",
              gridTemplateColumns: `${REGION_LANE_WIDTH}px ${CENTER_SECTION_WIDTH}px ${REGION_LANE_WIDTH}px`,
              gridTemplateRows: "auto auto",
              gap: DESKTOP_BRACKET_GRID_GAP,
              alignItems: "start",
              minWidth: DESKTOP_BRACKET_WIDTH,
              width: "max-content",
            }}
          >
            <div style={{ gridColumn: 1, gridRow: 1 }}>{renderRegionLane(laneRegions.topLeft, "left")}</div>
            <div style={{ gridColumn: 1, gridRow: 2 }}>{renderRegionLane(laneRegions.bottomLeft, "left")}</div>
            {renderCenterBoard()}

            <div style={{ gridColumn: 3, gridRow: 1 }}>{renderRegionLane(laneRegions.topRight, "right")}</div>
            <div style={{ gridColumn: 3, gridRow: 2 }}>{renderRegionLane(laneRegions.bottomRight, "right")}</div>
          </div>
        )}
      </div>
    </section>
  );
}
