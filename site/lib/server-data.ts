import fs from "fs";
import path from "path";
import { DataFile, PredictionRow, normalizeRows } from "./data";

const PREDICTIONS_RE = /^predictions_(\d{4}-\d{2}-\d{2})\.json$/;
const RESULTS_RE = /^results_(\d{4}-\d{2}-\d{2})\.json$/;
const FINAL_SCORES_RE = /^final_scores_(\d{4}-\d{2}-\d{2})\.json$/;
const RANKINGS_RE = /^rankings_(\d{4})\.json$/;
const TRUE_WALKFORWARD_MANIFEST = "true_walkforward_manifest.json";

type TrueWalkforwardManifest = {
  holdout_seasons?: number[];
  walkforward_dates?: string[];
};

/** Today's date in US Eastern Time as YYYY-MM-DD. */
export function todayET(): string {
  return new Date().toLocaleDateString("en-CA", { timeZone: "America/New_York" });
}

export function getDataDir(): string {
  const primary = path.join(process.cwd(), "public", "data");
  if (fs.existsSync(primary)) {
    return primary;
  }
  const fallback = path.join(process.cwd(), "site", "public", "data");
  if (fs.existsSync(fallback)) {
    return fallback;
  }
  return primary;
}

export function listPredictionFiles(): DataFile[] {
  const dir = getDataDir();
  if (!fs.existsSync(dir)) {
    return [];
  }
  return fs
    .readdirSync(dir)
    .map((filename) => {
      const match = filename.match(PREDICTIONS_RE);
      if (!match) {
        return null;
      }
      return { date: match[1], filename };
    })
    .filter((entry): entry is DataFile => Boolean(entry))
    .sort((a, b) => (a.date < b.date ? -1 : 1));
}

export function listResultFiles(): DataFile[] {
  const dir = getDataDir();
  if (!fs.existsSync(dir)) {
    return [];
  }
  return fs
    .readdirSync(dir)
    .map((filename) => {
      const match = filename.match(RESULTS_RE);
      if (!match) {
        return null;
      }
      return { date: match[1], filename };
    })
    .filter((entry): entry is DataFile => Boolean(entry))
    .sort((a, b) => (a.date < b.date ? -1 : 1));
}

export function listFinalScoreFiles(): DataFile[] {
  const dir = getDataDir();
  if (!fs.existsSync(dir)) {
    return [];
  }
  return fs
    .readdirSync(dir)
    .map((filename) => {
      const match = filename.match(FINAL_SCORES_RE);
      if (!match) {
        return null;
      }
      return { date: match[1], filename };
    })
    .filter((entry): entry is DataFile => Boolean(entry))
    .sort((a, b) => (a.date < b.date ? -1 : 1));
}

export function listRankingsSeasons(): number[] {
  const dir = getDataDir();
  if (!fs.existsSync(dir)) {
    return [];
  }
  return fs
    .readdirSync(dir)
    .map((filename) => {
      const match = filename.match(RANKINGS_RE);
      if (!match) return null;
      return Number(match[1]);
    })
    .filter((n): n is number => n !== null)
    .sort((a, b) => b - a);
}

export function readJsonFile(filename: string): unknown {
  const filepath = path.join(getDataDir(), filename);
  if (!fs.existsSync(filepath)) {
    return null;
  }
  const raw = fs.readFileSync(filepath, "utf-8");
  return JSON.parse(raw);
}

export function getLatestPredictionFile(): DataFile | null {
  const today = todayET();
  const files = listPredictionFiles().filter((f) => f.date <= today);
  if (!files.length) {
    return null;
  }
  return files[files.length - 1];
}

export function getPredictionRowsByDate(date: string): PredictionRow[] {
  const filename = `predictions_${date}.json`;
  return normalizeRows(readJsonFile(filename));
}

export function getPredictionRowsByFilename(filename: string): PredictionRow[] {
  return normalizeRows(readJsonFile(filename));
}

export function getResultRowsByDate(date: string): PredictionRow[] {
  const filename = `results_${date}.json`;
  return normalizeRows(readJsonFile(filename));
}

function seasonFromDate(date: string): number {
  const year = Number(date.slice(0, 4));
  const month = Number(date.slice(5, 7));
  return month >= 11 ? year + 1 : year;
}

export function readTrueWalkforwardManifest(): TrueWalkforwardManifest | null {
  const payload = readJsonFile(TRUE_WALKFORWARD_MANIFEST);
  if (!payload || typeof payload !== "object") {
    return null;
  }
  return payload as TrueWalkforwardManifest;
}

export function listWalkforwardPredictionFiles(): DataFile[] {
  const manifest = readTrueWalkforwardManifest();
  const files = listPredictionFiles();
  const dates = new Set(manifest?.walkforward_dates ?? []);
  if (dates.size) {
    return files.filter((f) => dates.has(f.date));
  }
  const seasons = new Set(manifest?.holdout_seasons ?? []);
  if (seasons.size) {
    return files.filter((f) => seasons.has(seasonFromDate(f.date)));
  }
  return files;
}

export function listPerformancePredictionFiles(): DataFile[] {
  const walkforward = listWalkforwardPredictionFiles();
  const walkforwardDates = new Set(walkforward.map((f) => f.date));
  const currentSeason = seasonFromDate(todayET());
  const currentSeasonFiles = listPredictionFiles().filter(
    (f) => seasonFromDate(f.date) === currentSeason && !walkforwardDates.has(f.date)
  );
  return [...walkforward, ...currentSeasonFiles].sort((a, b) =>
    a.date < b.date ? -1 : 1
  );
}
