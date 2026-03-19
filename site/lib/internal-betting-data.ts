import fs from "fs";
import path from "path";
import { todayET } from "./server-data";

export type InternalBettingManifest = {
  season?: number;
  date?: string;
  training_seasons?: number[];
  slate_games?: number;
  filter_threshold?: number;
  raw_edge_threshold?: number;
  filter_pass_games?: number;
  non_ncaa_shortlist_games?: number;
  ncaa_caution_games?: number;
  raw_watchlist_games?: number;
  flagged_mainly_by_disagreement?: number;
  slice_mix?: Record<string, number>;
};

export type InternalBettingRow = {
  season: number | null;
  gameId: number | null;
  game_date: string | null;
  startDate: string | null;
  slice: string | null;
  homeTeam: string | null;
  awayTeam: string | null;
  book_spread: number | null;
  predicted_spread: number | null;
  model_pick_side: string | null;
  pick_cover_prob: number | null;
  pick_prob_edge: number | null;
  raw_logit_score: number | null;
  filter_score: number | null;
  score_lift_vs_raw_logit: number | null;
  flagged_mainly_by_disagreement: boolean | null;
  persistence_label: string | null;
  he_market_edge_for_pick: number | null;
  abs_he_vs_market_edge: number | null;
  pick_team_recent_same_sign_count_21d: number | null;
  pick_team_prior_same_sign_streak: number | null;
  neutral_site_flag: boolean | null;
  signal_driver: string | null;
  usage_label: string | null;
  filter_pass?: boolean | null;
  raw_edge_pass?: boolean | null;
};

export type InternalBettingPayload = {
  availableDates: string[];
  currentDate: string;
  manifest: InternalBettingManifest | null;
  shortlist: InternalBettingRow[];
  rawWatchlist: InternalBettingRow[];
  ncaaCaution: InternalBettingRow[];
  slateScores: InternalBettingRow[];
};

function getInternalBettingRoot(): string {
  const candidates = [
    path.join(process.cwd(), "site", "public", "data", "internal_betting"),
    path.join(process.cwd(), "public", "data", "internal_betting"),
    path.join(process.cwd(), "..", "site", "public", "data", "internal_betting"),
    path.join(__dirname, "..", "public", "data", "internal_betting"),
    path.join(__dirname, "..", "..", "public", "data", "internal_betting"),
    path.join(process.cwd(), "artifacts", "daily_internal_bet_filter"),
    path.join(process.cwd(), "..", "artifacts", "daily_internal_bet_filter"),
    path.join(process.cwd(), "site", "..", "artifacts", "daily_internal_bet_filter"),
    path.join(__dirname, "..", "..", "artifacts", "daily_internal_bet_filter"),
    path.join(__dirname, "..", "..", "..", "artifacts", "daily_internal_bet_filter"),
  ];
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) return candidate;
  }
  return candidates[0];
}

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
  if (/^-?\d+(\.\d+)?$/.test(raw)) return Number(raw);
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

export function listInternalBettingDates(): string[] {
  const root = getInternalBettingRoot();
  if (!fs.existsSync(root)) return [];
  const today = todayET();
  return fs
    .readdirSync(root, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && /^\d{4}-\d{2}-\d{2}$/.test(entry.name))
    .map((entry) => entry.name)
    .filter((date) => date <= today)
    .sort((a, b) => (a < b ? 1 : -1));
}

export function getLatestInternalBettingDate(): string | null {
  const dates = listInternalBettingDates();
  return dates.length ? dates[0] : null;
}

export function readInternalBettingPayload(date?: string | null): InternalBettingPayload | null {
  const availableDates = listInternalBettingDates();
  const latest = availableDates[0] ?? null;
  const currentDate = date && availableDates.includes(date) ? date : latest;
  if (!currentDate) return null;

  const root = getInternalBettingRoot();
  const dir = path.join(root, currentDate);
  const manifestPath = path.join(dir, "manifest.json");
  const manifest = fs.existsSync(manifestPath)
    ? (JSON.parse(fs.readFileSync(manifestPath, "utf-8")) as InternalBettingManifest)
    : null;

  return {
    availableDates,
    currentDate,
    manifest,
    shortlist: readCsvRows<InternalBettingRow>(path.join(dir, "slate_shortlist.csv")),
    rawWatchlist: readCsvRows<InternalBettingRow>(path.join(dir, "raw_edge_watchlist.csv")),
    ncaaCaution: readCsvRows<InternalBettingRow>(path.join(dir, "ncaa_caution.csv")),
    slateScores: readCsvRows<InternalBettingRow>(path.join(dir, "slate_scores.csv")),
  };
}
