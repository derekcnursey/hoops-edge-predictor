export type NcaaFieldTeam = {
  team_id: number;
  team: string;
  rank: number;
  conference: string;
  record: string;
  conf_record: string;
  adj_oe: number;
  adj_de: number;
  adj_margin: number;
  adj_tempo: number;
  model_index: number | null;
  adj_oe_rank?: number | null;
  adj_de_rank?: number | null;
  adj_margin_rank?: number | null;
  adj_tempo_rank?: number | null;
  ft_pct?: number | null;
  three_p_pct?: number | null;
  def_3p_pct?: number | null;
  ft_pct_rank?: number | null;
  three_p_pct_rank?: number | null;
  def_3p_pct_rank?: number | null;
  model_index_rank?: number | null;
};

export type NcaaRegionEntry =
  | ({
      seed: number;
      source: "team";
    } & NcaaFieldTeam)
  | {
      seed: number;
      source: "play_in";
      play_in_game_id: string;
    };

export type NcaaRegion = {
  name: string;
  entries: NcaaRegionEntry[];
};

export type NcaaFirstFourGame = {
  id: string;
  label: string;
  region: string;
  seed: number;
  teams: NcaaFieldTeam[];
};

export type NcaaBracketField = {
  generated_at: string;
  season: number;
  source: string;
  note: string;
  regions: NcaaRegion[];
  first_four: NcaaFirstFourGame[];
};

export type MatchupPredictionCacheEntry = {
  team1_id: number;
  team1_name: string;
  team2_id: number;
  team2_name: string;
  matchup_model_variant_active?: string | null;
  mu_team1_minus_team2: number;
  mu_team1_minus_team2_legacy_synthetic?: number | null;
  mu_team1_minus_team2_team_ab_elite_tail_round64_v1?: number | null;
  mu_team1_minus_team2_team_ab_internal?: number | null;
  display_mu_team1_minus_team2?: number | null;
  win_prob_team1: number;
  win_prob_team1_legacy_synthetic?: number | null;
  win_prob_team1_team_ab_elite_tail_round64_v1?: number | null;
  win_prob_team1_team_ab_internal?: number | null;
  pred_sigma?: number | null;
  scheduled_game_id?: number | null;
  scheduled_round_id?: BracketRoundId | null;
  scheduled_round_label?: string | null;
  start_time?: string | null;
  home_team_id?: number | null;
  away_team_id?: number | null;
  home_team_name?: string | null;
  away_team_name?: string | null;
  model_mu_home?: number | null;
  model_mu_home_legacy_synthetic?: number | null;
  model_mu_home_team_ab_elite_tail_round64_v1?: number | null;
  model_mu_home_team_ab_internal?: number | null;
  display_model_mu_home?: number | null;
  edge_home_points?: number | null;
  display_edge_home_points?: number | null;
  pick_side?: "HOME" | "AWAY" | null;
  pick_cover_prob?: number | null;
  pick_prob_edge?: number | null;
  pick_fair_odds?: number | null;
  market_mu_team1_minus_team2?: number | null;
  market_spread_home?: number | null;
  market_home_team_id?: number | null;
  market_away_team_id?: number | null;
  market_home_moneyline?: number | null;
  market_away_moneyline?: number | null;
  market_line_source?: string | null;
};

export type MatchupPredictionCache = {
  generated_at: string;
  season: number;
  neutral_site: boolean;
  source: string;
  note: string;
  matchup_model_variant_active?: string | null;
  matchup_model_variants_available?: string[] | null;
  predictions: Record<string, MatchupPredictionCacheEntry>;
};

export type NcaaResultsStatus = "pending" | "in_progress" | "final";

export type NcaaTournamentResultGame = {
  winner_team_id?: number | null;
  loser_team_id?: number | null;
  status: NcaaResultsStatus;
};

export type NcaaTournamentResults = {
  version: number;
  season: number;
  games: Record<string, NcaaTournamentResultGame>;
};

export type BracketRoundId =
  | "first-four"
  | "round-of-64"
  | "round-of-32"
  | "sweet-16"
  | "elite-8"
  | "final-four"
  | "national-championship";

export type BracketTeam = {
  id: number;
  name: string;
  seed: number;
  region?: string;
  rank: number;
  conference: string;
  record: string;
  confRecord: string;
  adjOe: number;
  adjDe: number;
  adjNet: number;
  adjTempo: number;
  modelIndex: number | null;
  adjOeRank?: number | null;
  adjDeRank?: number | null;
  adjNetRank?: number | null;
  adjTempoRank?: number | null;
  ftPct?: number | null;
  threePPct?: number | null;
  defThreePPct?: number | null;
  ftPctRank?: number | null;
  threePPctRank?: number | null;
  defThreePPctRank?: number | null;
  modelIndexRank?: number | null;
};

export type BracketSource =
  | {
      type: "team";
      teamId: number;
      label: string;
    }
  | {
      type: "winner";
      gameId: string;
      label: string;
    };

export type BracketGameDefinition = {
  id: string;
  roundId: BracketRoundId;
  roundLabel: string;
  roundOrder: number;
  title: string;
  region?: string;
  matchupOrder: number;
  sourceA: BracketSource;
  sourceB: BracketSource;
};

export type ResolvedBracketGame = BracketGameDefinition & {
  teamA: BracketTeam | null;
  teamB: BracketTeam | null;
  selectedWinnerId?: number;
};

export type MatchupPrediction = {
  teamAId: number;
  teamAName: string;
  teamBId: number;
  teamBName: string;
  activeModelVariant?: string | null;
  favoredTeamId: number;
  favoredTeamName: string;
  underdogTeamId: number;
  underdogTeamName: string;
  winProbA: number;
  winProbB: number;
  displayWinProbA?: number | null;
  displayWinProbB?: number | null;
  projectedSpread: number;
  rawProjectedSpread?: number | null;
  displayProjectedSpread?: number | null;
  rawMarginA?: number | null;
  displayMarginA?: number | null;
  displayFavoredTeamId?: number | null;
  displayFavoredTeamName?: string | null;
  marketMarginA?: number | null;
  marketProjectedSpread?: number | null;
  marketFavoredTeamId?: number | null;
  marketFavoredTeamName?: string | null;
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
  pickSide?: "HOME" | "AWAY" | null;
  pickCoverProb?: number | null;
  pickProbEdge?: number | null;
  pickFairOdds?: number | null;
  scheduledGameId?: number | null;
  scheduledRoundId?: BracketRoundId | null;
  scheduledRoundLabel?: string | null;
  modelWinnerId: number;
  modelWinnerName: string;
  projectedScoreA?: number | null;
  projectedScoreB?: number | null;
  comparisonModel?: {
    variant: string;
    label: string;
    favoredTeamId: number | null;
    favoredTeamName: string | null;
    winProbA: number;
    winProbB: number;
    projectedSpread: number | null;
    rawMarginA: number | null;
  } | null;
};

export type MarchBettingGame = {
  gameId: string;
  roundId: BracketRoundId;
  roundLabel: string;
  region?: string | null;
  matchupOrder: number;
  startTime?: string | null;
  teamAId: number;
  teamAName: string;
  teamASeed: number;
  teamBId: number;
  teamBName: string;
  teamBSeed: number;
  homeTeamId: number | null;
  homeTeamName: string | null;
  awayTeamId: number | null;
  awayTeamName: string | null;
  favoriteTeamId: number;
  favoriteTeamName: string;
  favoriteWinProb: number;
  rawProjectedSpread: number | null;
  displayProjectedSpread: number | null;
  modelSpreadHome: number | null;
  displayModelSpreadHome: number | null;
  predSigma: number | null;
  edgeHomePoints: number | null;
  displayEdgeHomePoints: number | null;
  pickSide: "HOME" | "AWAY" | null;
  pickCoverProb: number | null;
  pickProbEdge: number | null;
  displayFavoredTeamId: number | null;
  displayFavoredTeamName: string | null;
  marketSpreadHome: number | null;
  marketProjectedSpread: number | null;
  marketFavoredTeamId: number | null;
  marketFavoredTeamName: string | null;
  marketLineSource: string | null;
  rawDiffAbs: number | null;
  displayDiffAbs: number | null;
  spreadDiffAbs: number | null;
};

export type NcaaValidationResult = {
  valid: boolean;
  errors: string[];
};

export type GradedGameStatus = "pending" | "correct" | "incorrect";

export type GradedGameResult = {
  gameId: string;
  roundId: BracketRoundId;
  roundLabel: string;
  actualWinnerId?: number;
  actualLoserId?: number;
  actualWinnerName?: string;
  actualLoserName?: string;
  status: GradedGameStatus;
  isFinal: boolean;
  isScored: boolean;
  pointsEarned: number;
  pointsPossible: number;
};

export type GradedRoundSummary = {
  roundId: BracketRoundId;
  roundLabel: string;
  correct: number;
  incorrect: number;
  pending: number;
  score: number;
  possibleScore: number;
};

export type BracketGradingSummary = {
  correct: number;
  incorrect: number;
  pending: number;
  totalGraded: number;
  score: number;
  possibleScore: number;
  rounds: GradedRoundSummary[];
  byGame: Record<string, GradedGameResult>;
};
