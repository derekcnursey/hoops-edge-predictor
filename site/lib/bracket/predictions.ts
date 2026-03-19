import { getSiteHomeWinProbFromValues } from "../data";
import { MatchupPrediction, MatchupPredictionCacheEntry } from "./types";

export function canonicalMatchupKey(teamAId: number, teamBId: number): string {
  return teamAId < teamBId ? `${teamAId}::${teamBId}` : `${teamBId}::${teamAId}`;
}

const MARGIN_EPSILON = 1e-6;

function favoredFromMargin(
  marginForA: number | null | undefined,
  teamAId: number,
  teamAName: string,
  teamBId: number,
  teamBName: string,
): {
  favoriteId: number | null;
  favoriteName: string | null;
  spread: number | null;
} {
  if (marginForA == null || Number.isNaN(marginForA)) {
    return { favoriteId: null, favoriteName: null, spread: null };
  }
  if (marginForA >= 0) {
    return { favoriteId: teamAId, favoriteName: teamAName, spread: Math.abs(marginForA) };
  }
  return { favoriteId: teamBId, favoriteName: teamBName, spread: Math.abs(marginForA) };
}

export function getDisplayFavoriteSummary(prediction: MatchupPrediction): {
  favoriteTeamId: number;
  favoriteTeamName: string;
  spread: number;
  favoriteWinProb: number;
} {
  const displaySpread =
    prediction.displayProjectedSpread ??
    prediction.rawProjectedSpread ??
    prediction.projectedSpread ??
    0;
  const probA = prediction.displayWinProbA ?? prediction.winProbA;
  const probB = prediction.displayWinProbB ?? prediction.winProbB;
  const explicitFavoriteId =
    prediction.displayFavoredTeamId ??
    prediction.favoredTeamId ??
    (probA >= probB ? prediction.teamAId : prediction.teamBId);
  const explicitFavoriteName =
    prediction.displayFavoredTeamName ??
    prediction.favoredTeamName ??
    (explicitFavoriteId === prediction.teamAId ? prediction.teamAName : prediction.teamBName);
  const marginA =
    prediction.displayMarginA ??
    prediction.rawMarginA ??
    (explicitFavoriteId === prediction.teamAId ? displaySpread : -displaySpread);

  if (marginA > MARGIN_EPSILON) {
    return {
      favoriteTeamId: prediction.teamAId,
      favoriteTeamName: prediction.teamAName,
      spread: displaySpread || Math.abs(marginA),
      favoriteWinProb: probA,
    };
  }
  if (marginA < -MARGIN_EPSILON) {
    return {
      favoriteTeamId: prediction.teamBId,
      favoriteTeamName: prediction.teamBName,
      spread: displaySpread || Math.abs(marginA),
      favoriteWinProb: probB,
    };
  }

  if (displaySpread > MARGIN_EPSILON) {
    return {
      favoriteTeamId: explicitFavoriteId,
      favoriteTeamName: explicitFavoriteName,
      spread: displaySpread,
      favoriteWinProb: explicitFavoriteId === prediction.teamAId ? probA : probB,
    };
  }

  return {
    favoriteTeamId: explicitFavoriteId,
    favoriteTeamName: explicitFavoriteName,
    spread: 0,
    favoriteWinProb: explicitFavoriteId === prediction.teamAId ? probA : probB,
  };
}

function buildComparisonRecord(base: {
  variant: string;
  label: string;
  teamAId: number;
  teamAName: string;
  teamBId: number;
  teamBName: string;
  winProbA: number;
  rawMarginA: number;
}): NonNullable<MatchupPrediction["comparisonModel"]> {
  const favorite = favoredFromMargin(
    base.rawMarginA,
    base.teamAId,
    base.teamAName,
    base.teamBId,
    base.teamBName,
  );
  return {
    variant: base.variant,
    label: base.label,
    favoredTeamId: favorite.favoriteId,
    favoredTeamName: favorite.favoriteName,
    winProbA: base.winProbA,
    winProbB: 1 - base.winProbA,
    projectedSpread: favorite.spread,
    rawMarginA: base.rawMarginA,
  };
}

function buildPredictionRecord(base: {
  teamAId: number;
  teamAName: string;
  teamBId: number;
  teamBName: string;
  activeModelVariant?: string | null;
  winProbA: number;
  rawMarginA: number;
  displayMarginA: number;
  displayWinProbAOverride?: number | null;
  marketMarginA?: number | null;
  marketLineSource?: string | null;
  predSigma?: number | null;
  scheduledStartTime?: string | null;
  scheduledHomeTeamId?: number | null;
  scheduledAwayTeamId?: number | null;
  scheduledHomeTeamName?: string | null;
  scheduledAwayTeamName?: string | null;
  marketSpreadHome?: number | null;
  modelSpreadHome?: number | null;
  displayModelSpreadHome?: number | null;
  edgeHomePoints?: number | null;
  displayEdgeHomePoints?: number | null;
  pickSide?: MatchupPrediction["pickSide"];
  pickCoverProb?: number | null;
  pickProbEdge?: number | null;
  pickFairOdds?: number | null;
  scheduledGameId?: number | null;
  scheduledRoundId?: MatchupPrediction["scheduledRoundId"];
  scheduledRoundLabel?: string | null;
  comparisonModel?: MatchupPrediction["comparisonModel"];
}): MatchupPrediction {
  const rawFavorite = favoredFromMargin(
    base.rawMarginA,
    base.teamAId,
    base.teamAName,
    base.teamBId,
    base.teamBName,
  );
  const displayFavorite = favoredFromMargin(
    base.displayMarginA,
    base.teamAId,
    base.teamAName,
    base.teamBId,
    base.teamBName,
  );
  const marketFavorite = favoredFromMargin(
    base.marketMarginA,
    base.teamAId,
    base.teamAName,
    base.teamBId,
    base.teamBName,
  );
  const modelWinnerId = rawFavorite.favoriteId ?? base.teamAId;
  const modelWinnerName = rawFavorite.favoriteName ?? base.teamAName;
  const hasDisplayAdjustment = Math.abs(base.displayMarginA - base.rawMarginA) > 1e-9;
  const recalculatedDisplayWinProbA =
    base.displayWinProbAOverride == null && hasDisplayAdjustment
      ? getSiteHomeWinProbFromValues(
          base.displayMarginA,
          base.predSigma ?? null,
          base.scheduledStartTime ?? null,
          true,
        )
      : null;
  const displayWinProbA =
    base.displayWinProbAOverride ?? recalculatedDisplayWinProbA ?? base.winProbA;

  return {
    teamAId: base.teamAId,
    teamAName: base.teamAName,
    teamBId: base.teamBId,
    teamBName: base.teamBName,
    activeModelVariant: base.activeModelVariant ?? null,
    favoredTeamId: modelWinnerId,
    favoredTeamName: modelWinnerName,
    underdogTeamId: modelWinnerId === base.teamAId ? base.teamBId : base.teamAId,
    underdogTeamName: modelWinnerId === base.teamAId ? base.teamBName : base.teamAName,
    winProbA: base.winProbA,
    winProbB: 1 - base.winProbA,
    displayWinProbA,
    displayWinProbB: 1 - displayWinProbA,
    projectedSpread: rawFavorite.spread ?? 0,
    rawProjectedSpread: rawFavorite.spread,
    displayProjectedSpread: displayFavorite.spread,
    rawMarginA: base.rawMarginA,
    displayMarginA: base.displayMarginA,
    displayFavoredTeamId: displayFavorite.favoriteId,
    displayFavoredTeamName: displayFavorite.favoriteName,
    marketMarginA: base.marketMarginA ?? null,
    marketProjectedSpread: marketFavorite.spread,
    marketFavoredTeamId: marketFavorite.favoriteId,
    marketFavoredTeamName: marketFavorite.favoriteName,
    marketLineSource: base.marketLineSource ?? null,
    predSigma: base.predSigma ?? null,
    scheduledStartTime: base.scheduledStartTime ?? null,
    scheduledHomeTeamId: base.scheduledHomeTeamId ?? null,
    scheduledAwayTeamId: base.scheduledAwayTeamId ?? null,
    scheduledHomeTeamName: base.scheduledHomeTeamName ?? null,
    scheduledAwayTeamName: base.scheduledAwayTeamName ?? null,
    marketSpreadHome: base.marketSpreadHome ?? null,
    modelSpreadHome: base.modelSpreadHome ?? null,
    displayModelSpreadHome: base.displayModelSpreadHome ?? null,
    edgeHomePoints: base.edgeHomePoints ?? null,
    displayEdgeHomePoints: base.displayEdgeHomePoints ?? null,
    pickSide: base.pickSide ?? null,
    pickCoverProb: base.pickCoverProb ?? null,
    pickProbEdge: base.pickProbEdge ?? null,
    pickFairOdds: base.pickFairOdds ?? null,
    scheduledGameId: base.scheduledGameId ?? null,
    scheduledRoundId: base.scheduledRoundId ?? null,
    scheduledRoundLabel: base.scheduledRoundLabel ?? null,
    modelWinnerId,
    modelWinnerName,
    projectedScoreA: null,
    projectedScoreB: null,
    comparisonModel: base.comparisonModel ?? null,
  };
}

export function flipPrediction(prediction: MatchupPrediction): MatchupPrediction {
  return buildPredictionRecord({
    teamAId: prediction.teamBId,
    teamAName: prediction.teamBName,
    teamBId: prediction.teamAId,
    teamBName: prediction.teamAName,
    activeModelVariant: prediction.activeModelVariant ?? null,
    winProbA: prediction.winProbB,
    rawMarginA: -(prediction.rawMarginA ?? 0),
    displayMarginA: -(prediction.displayMarginA ?? prediction.rawMarginA ?? 0),
    marketMarginA:
      prediction.marketMarginA == null || Number.isNaN(prediction.marketMarginA)
        ? null
        : -prediction.marketMarginA,
    marketLineSource: prediction.marketLineSource ?? null,
    predSigma: prediction.predSigma ?? null,
    scheduledStartTime: prediction.scheduledStartTime ?? null,
    scheduledHomeTeamId: prediction.scheduledHomeTeamId ?? null,
    scheduledAwayTeamId: prediction.scheduledAwayTeamId ?? null,
    scheduledHomeTeamName: prediction.scheduledHomeTeamName ?? null,
    scheduledAwayTeamName: prediction.scheduledAwayTeamName ?? null,
    marketSpreadHome: prediction.marketSpreadHome ?? null,
    modelSpreadHome: prediction.modelSpreadHome ?? null,
    displayModelSpreadHome: prediction.displayModelSpreadHome ?? null,
    edgeHomePoints: prediction.edgeHomePoints ?? null,
    displayEdgeHomePoints: prediction.displayEdgeHomePoints ?? null,
    pickSide: prediction.pickSide ?? null,
    pickCoverProb: prediction.pickCoverProb ?? null,
    pickProbEdge: prediction.pickProbEdge ?? null,
    pickFairOdds: prediction.pickFairOdds ?? null,
    scheduledGameId: prediction.scheduledGameId ?? null,
    scheduledRoundId: prediction.scheduledRoundId ?? null,
    scheduledRoundLabel: prediction.scheduledRoundLabel ?? null,
    comparisonModel: prediction.comparisonModel
      ? buildComparisonRecord({
          variant: prediction.comparisonModel.variant,
          label: prediction.comparisonModel.label,
          teamAId: prediction.teamBId,
          teamAName: prediction.teamBName,
          teamBId: prediction.teamAId,
          teamBName: prediction.teamAName,
          winProbA: prediction.comparisonModel.winProbB,
          rawMarginA: -(prediction.comparisonModel.rawMarginA ?? 0),
        })
      : null,
  });
}

export function orientPrediction(
  prediction: MatchupPrediction,
  teamAId: number,
  teamBId: number,
): MatchupPrediction {
  if (prediction.teamAId === teamAId && prediction.teamBId === teamBId) return prediction;
  if (prediction.teamAId === teamBId && prediction.teamBId === teamAId) {
    return flipPrediction(prediction);
  }
  throw new Error("Cached prediction does not match requested matchup");
}

export function canonicalizePrediction(prediction: MatchupPrediction): MatchupPrediction {
  if (prediction.teamAId < prediction.teamBId) return prediction;
  return flipPrediction(prediction);
}

export function buildPredictionFromCacheEntry(
  entry: MatchupPredictionCacheEntry,
  teamAId: number,
  teamBId: number,
): MatchupPrediction {
  const directOrder = entry.team1_id === teamAId && entry.team2_id === teamBId;
  const activeVariant = entry.matchup_model_variant_active ?? null;
  const rawMarginA = directOrder ? entry.mu_team1_minus_team2 : -entry.mu_team1_minus_team2;
  const displayMu = entry.display_mu_team1_minus_team2 ?? entry.mu_team1_minus_team2;
  const displayMarginA = directOrder ? displayMu : -displayMu;
  const marketMu = entry.market_mu_team1_minus_team2;
  const marketMarginA =
    marketMu == null || Number.isNaN(marketMu) ? null : directOrder ? marketMu : -marketMu;
  const winProbA = directOrder ? entry.win_prob_team1 : 1 - entry.win_prob_team1;
  const teamAName = directOrder ? entry.team1_name : entry.team2_name;
  const teamBName = directOrder ? entry.team2_name : entry.team1_name;
  const comparisonModel =
    activeVariant === "team_ab_elite_tail_round64_v1" &&
    entry.mu_team1_minus_team2_team_ab_internal != null &&
    entry.win_prob_team1_team_ab_internal != null
      ? buildComparisonRecord({
          variant: "team_ab_internal",
          label: "Internal",
          teamAId,
          teamAName,
          teamBId,
          teamBName,
          winProbA: directOrder
            ? entry.win_prob_team1_team_ab_internal
            : 1 - entry.win_prob_team1_team_ab_internal,
          rawMarginA: directOrder
            ? entry.mu_team1_minus_team2_team_ab_internal
            : -entry.mu_team1_minus_team2_team_ab_internal,
        })
      : activeVariant === "legacy_synthetic" &&
          entry.mu_team1_minus_team2_team_ab_elite_tail_round64_v1 != null &&
          entry.win_prob_team1_team_ab_elite_tail_round64_v1 != null
        ? buildComparisonRecord({
            variant: "team_ab_elite_tail_round64_v1",
            label: "Team A/B",
            teamAId,
            teamAName,
            teamBId,
            teamBName,
            winProbA: directOrder
              ? entry.win_prob_team1_team_ab_elite_tail_round64_v1
              : 1 - entry.win_prob_team1_team_ab_elite_tail_round64_v1,
            rawMarginA: directOrder
              ? entry.mu_team1_minus_team2_team_ab_elite_tail_round64_v1
              : -entry.mu_team1_minus_team2_team_ab_elite_tail_round64_v1,
          })
        : null;
  const averagedDisplayMarginA =
    comparisonModel?.rawMarginA == null
      ? displayMarginA
      : (rawMarginA + comparisonModel.rawMarginA) / 2;
  const averagedDisplayWinProbA =
    comparisonModel == null ? winProbA : (winProbA + comparisonModel.winProbA) / 2;

  return buildPredictionRecord({
    teamAId,
    teamAName,
    teamBId,
    teamBName,
    activeModelVariant: activeVariant,
    winProbA,
    rawMarginA,
    displayMarginA: averagedDisplayMarginA,
    displayWinProbAOverride: averagedDisplayWinProbA,
    marketMarginA,
    marketLineSource: entry.market_line_source ?? null,
    predSigma: entry.pred_sigma ?? null,
    scheduledStartTime: entry.start_time ?? null,
    scheduledHomeTeamId: entry.home_team_id ?? null,
    scheduledAwayTeamId: entry.away_team_id ?? null,
    scheduledHomeTeamName: entry.home_team_name ?? null,
    scheduledAwayTeamName: entry.away_team_name ?? null,
    marketSpreadHome: entry.market_spread_home ?? null,
    modelSpreadHome: entry.model_mu_home ?? null,
    displayModelSpreadHome: entry.display_model_mu_home ?? null,
    edgeHomePoints: entry.edge_home_points ?? null,
    displayEdgeHomePoints: entry.display_edge_home_points ?? null,
    pickSide: entry.pick_side ?? null,
    pickCoverProb: entry.pick_cover_prob ?? null,
    pickProbEdge: entry.pick_prob_edge ?? null,
    pickFairOdds: entry.pick_fair_odds ?? null,
    scheduledGameId: entry.scheduled_game_id ?? null,
    scheduledRoundId: entry.scheduled_round_id ?? null,
    scheduledRoundLabel: entry.scheduled_round_label ?? null,
    comparisonModel,
  });
}
