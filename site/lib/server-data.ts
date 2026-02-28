import fs from "fs";
import path from "path";
import { DataFile, PredictionRow, normalizeRows } from "./data";

const PREDICTIONS_RE = /^predictions_(\d{4}-\d{2}-\d{2})\.json$/;
const RESULTS_RE = /^results_(\d{4}-\d{2}-\d{2})\.json$/;
const FINAL_SCORES_RE = /^final_scores_(\d{4}-\d{2}-\d{2})\.json$/;

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
