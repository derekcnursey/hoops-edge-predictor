import fs from "fs";
import path from "path";

export type ProfitabilityRow = {
  group: string;
  family?: string | null;
  slice: string;
  bets: number;
  wins: number;
  losses: number;
  pushes: number;
  win_rate: number | null;
  roi_per_1_at_minus_110: number | null;
  p_value_gt_breakeven?: number | null;
  avg_pick_prob_edge?: number | null;
  avg_filter_score?: number | null;
  avg_abs_he_vs_market_edge?: number | null;
  new_transient_rate?: number | null;
  persistent_rate?: number | null;
  neutral_rate?: number | null;
};

export type BySeasonProfitabilityRow = ProfitabilityRow & {
  season: number;
};

export type RobustnessRow = {
  group: string;
  slice: string;
  season_count_with_bets: number;
  profitable_seasons: number;
  losing_seasons: number;
  breakeven_or_better_seasons: number;
  median_season_roi: number | null;
  mean_season_roi: number | null;
  season_roi_std: number | null;
  min_season_roi: number | null;
  max_season_roi: number | null;
  median_season_hit_rate: number | null;
  median_bets_per_season: number | null;
};

export type SignalDriverRow = {
  signal_driver: string;
  slice: string;
  bets: number;
  wins: number;
  losses: number;
  pushes: number;
  win_rate: number | null;
  roi_per_1_at_minus_110: number | null;
};

export type BettingTabMappingRow = {
  page_label: string;
  group: string;
  slice: string;
  note: string;
  bets: number;
  wins: number;
  losses: number;
  pushes: number;
  win_rate: number | null;
  roi_per_1_at_minus_110: number | null;
};

export type BettingPerformancePayload = {
  overall: ProfitabilityRow[];
  bySeason: BySeasonProfitabilityRow[];
  signalDrivers: SignalDriverRow[];
  robustness: RobustnessRow[];
  tabMapping: BettingTabMappingRow[];
  pickTables: Record<string, HistoricalPickRow[]>;
  memo: string | null;
};

export type HistoricalPickRow = {
  season: number;
  game_date: string;
  slice: string;
  gameId: number;
  game: string;
  final_score: string | null;
  pick_team: string;
  model_pick_side: string;
  hoops_edge_line_for_pick: number | null;
  market_line_for_pick: number | null;
  pick_prob_edge: number | null;
  disagreement_logit_score: number | null;
  he_market_edge_for_pick: number | null;
  abs_he_vs_market_edge: number | null;
  disagreement_context: string | null;
  pick_team_recent_same_sign_count_21d: number | null;
  pick_team_prior_same_sign_streak: number | null;
  neutral_site_flag: boolean | null;
  is_conference_tournament: boolean | null;
  is_ncaa_tournament: boolean | null;
  signal_driver: string | null;
  bet_result: string | null;
  roi_per_1_at_minus_110: number | null;
};

function parseCsvLine(line: string): string[] {
  const out: string[] = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (ch === '"') {
      const next = line[i + 1];
      if (inQuotes && next === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }
    if (ch === "," && !inQuotes) {
      out.push(current);
      current = "";
      continue;
    }
    current += ch;
  }
  out.push(current);
  return out;
}

function coerceCell(raw: string): string | number | boolean | null {
  if (raw === "") return null;
  if (raw === "True") return true;
  if (raw === "False") return false;
  if (/^-?\d+(\.\d+)?([eE]-?\d+)?$/.test(raw)) return Number(raw);
  return raw;
}

function readCsvRows<T extends Record<string, unknown>>(filePath: string): T[] {
  if (!fs.existsSync(filePath)) return [];
  const raw = fs.readFileSync(filePath, "utf-8").trim();
  if (!raw) return [];
  const lines = raw.split(/\r?\n/);
  if (lines.length < 2) return [];
  const headers = parseCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const cells = parseCsvLine(line);
    const row: Record<string, unknown> = {};
    headers.forEach((header, idx) => {
      row[header] = coerceCell(cells[idx] ?? "");
    });
    return row as T;
  });
}

function getBettingPerformanceRoot(): string {
  const candidates = [
    path.join(process.cwd(), "site", "public", "data", "betting_performance"),
    path.join(process.cwd(), "public", "data", "betting_performance"),
    path.join(process.cwd(), "..", "site", "public", "data", "betting_performance"),
    path.join(__dirname, "..", "public", "data", "betting_performance"),
    path.join(__dirname, "..", "..", "public", "data", "betting_performance"),
    path.join(process.cwd(), "artifacts", "market_bet_profitability_v1"),
    path.join(process.cwd(), "..", "artifacts", "market_bet_profitability_v1"),
    path.join(process.cwd(), "site", "..", "artifacts", "market_bet_profitability_v1"),
    path.join(__dirname, "..", "..", "artifacts", "market_bet_profitability_v1"),
    path.join(__dirname, "..", "..", "..", "artifacts", "market_bet_profitability_v1"),
  ];
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) return candidate;
  }
  return candidates[0];
}

export function readBettingPerformancePayload(): BettingPerformancePayload | null {
  const root = getBettingPerformanceRoot();
  if (!fs.existsSync(root)) return null;
  const overall = readCsvRows<ProfitabilityRow>(path.join(root, "overall_profitability.csv"));
  const bySeason = readCsvRows<BySeasonProfitabilityRow>(path.join(root, "by_season_profitability.csv"));
  const signalDrivers = readCsvRows<SignalDriverRow>(path.join(root, "signal_driver_profitability.csv"));
  const robustness = readCsvRows<RobustnessRow>(path.join(root, "robustness_summary.csv"));
  const tabMapping = readCsvRows<BettingTabMappingRow>(path.join(root, "betting_tab_mapping_profitability.csv"));
  const pickTables: Record<string, HistoricalPickRow[]> = {
    promoted_internal_filter: readCsvRows<HistoricalPickRow>(path.join(root, "promoted_internal_filter.csv")),
    raw_edge_baseline: readCsvRows<HistoricalPickRow>(path.join(root, "raw_edge_baseline.csv")),
    overlap: readCsvRows<HistoricalPickRow>(path.join(root, "overlap.csv")),
    filter_only: readCsvRows<HistoricalPickRow>(path.join(root, "filter_only.csv")),
    raw_only: readCsvRows<HistoricalPickRow>(path.join(root, "raw_only.csv")),
    ncaa_caution: readCsvRows<HistoricalPickRow>(path.join(root, "ncaa_caution.csv")),
  };
  const memoPath = path.join(root, "memo.md");
  const memo = fs.existsSync(memoPath) ? fs.readFileSync(memoPath, "utf-8") : null;
  if (!overall.length) return null;
  return {
    overall,
    bySeason,
    signalDrivers,
    robustness,
    tabMapping,
    pickTables,
    memo,
  };
}
