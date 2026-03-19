import { GetServerSideProps } from "next";
import dynamic from "next/dynamic";
import { CSSProperties, useMemo, useState } from "react";
import Layout from "../components/Layout";
import { normalizeRows } from "../lib/data";
import {
  listFinalScoreFiles,
  listPerformancePredictionFiles,
  readJsonFile
} from "../lib/server-data";

const CumulativeChart = dynamic(
  () => import("../components/CumulativeChart"),
  { ssr: false }
);

/* -- types -- */

type GameResult = {
  date: string;
  week: string;
  month: string;
  monthSort: number;
  edge: number;
  result: "win" | "loss" | "push";
  model_error: number;
  sigma_z: number | null;
  diff: number | null;
};

type PerformanceProps = {
  games: GameResult[];
  seasonLabel: string;
};

/* -- server -- */

export const getServerSideProps: GetServerSideProps<
  PerformanceProps
> = async () => {
  const predFiles = listPerformancePredictionFiles();
  const finalFiles = listFinalScoreFiles();
  const finalByDate = new Map(
    finalFiles.map((f) => [f.date, f.filename])
  );

  const games: GameResult[] = [];

  for (const pf of predFiles) {
    const ff = finalByDate.get(pf.date);
    if (!ff) continue;

    const preds = normalizeRows(readJsonFile(pf.filename));
    const finals = normalizeRows(readJsonFile(ff));

    const fLookup = new Map<string, Record<string, unknown>>();
    for (const r of finals) {
      const gid = r.game_id;
      if (typeof gid === "string") fLookup.set(gid, r);
    }

    for (const pred of preds) {
      const gid = typeof pred.game_id === "string" ? pred.game_id : "";
      const fin = fLookup.get(gid);
      if (!fin) continue;

      const msh = pn(pred.market_spread_home);
      if (msh === null) continue;

      const homeScore = pn(fin.home_score);
      const awayScore = pn(fin.away_score);
      if (homeScore === null || awayScore === null) continue;

      const pickSide = ss(pred.pick_side).toUpperCase();
      if (pickSide !== "HOME" && pickSide !== "AWAY") continue;

      const ppe = pn(pred.pick_prob_edge) ?? 0;
      const mmh = pn(pred.model_mu_home);
      const sig = pn(pred.pred_sigma);

      const actualMargin = homeScore - awayScore;
      const cover = actualMargin + msh;

      let result: "win" | "loss" | "push";
      if (cover === 0) {
        result = "push";
      } else if (pickSide === "HOME") {
        result = cover > 0 ? "win" : "loss";
      } else {
        result = cover < 0 ? "win" : "loss";
      }

      const model_error =
        mmh !== null ? Math.abs(actualMargin - mmh) : 0;
      const sigma_z = sig !== null && sig > 0 ? model_error / sig : null;
      const diff = mmh !== null ? Math.abs(-mmh - msh) : null;

      games.push({
        date: pf.date,
        week: weekLabel(pf.date),
        month: monthName(pf.date),
        monthSort: monthSortKey(pf.date),
        edge: ppe * 100,
        result,
        model_error,
        sigma_z,
        diff
      });
    }
  }

  games.sort((a, b) => (a.date < b.date ? -1 : 1));

  const dates = games.map((g) => g.date);
  const minYear = dates.length
    ? Number(dates[0].slice(0, 4))
    : new Date().getFullYear();
  const maxYear = dates.length
    ? Number(dates[dates.length - 1].slice(0, 4))
    : minYear;
  const seasonLabel =
    minYear === maxYear
      ? `${minYear} Season`
      : `${minYear}\u2013${String(maxYear).slice(2)} Season`;

  return { props: { games, seasonLabel } };
};

function ss(v: unknown): string {
  return typeof v === "string" ? v : String(v ?? "");
}

function pn(v: unknown): number | null {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string" && v.trim()) {
    const n = Number(v);
    return Number.isNaN(n) ? null : n;
  }
  return null;
}

function weekLabel(dateStr: string): string {
  const d = new Date(dateStr + "T12:00:00");
  const day = d.getDay();
  const diff = day === 0 ? -6 : 1 - day;
  const monday = new Date(d);
  monday.setDate(d.getDate() + diff);
  const months = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec"
  ];
  return `${months[monday.getMonth()]} ${monday.getDate()}`;
}

function monthName(dateStr: string): string {
  const yr = dateStr.slice(0, 4);
  const mo = Number(dateStr.slice(5, 7));
  const name = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
  ][mo - 1];
  return `${name} ${yr}`;
}

function monthSortKey(dateStr: string): number {
  const yr = Number(dateStr.slice(0, 4));
  const mo = Number(dateStr.slice(5, 7));
  return yr * 12 + mo;
}

/* -- helpers -- */

function getSeasonFromDate(dateStr: string): number {
  const month = Number(dateStr.slice(5, 7));
  const year = Number(dateStr.slice(0, 4));
  return month >= 11 ? year + 1 : year;
}

/** Map calendar month to season-relative order: Nov=0, Dec=1, Jan=2, ..., Oct=11 */
function seasonMonthOrd(mo: number): number {
  return mo >= 11 ? mo - 11 : mo + 1;
}

const SEASON_MONTHS = [
  { value: 11, label: "Nov" },
  { value: 12, label: "Dec" },
  { value: 1, label: "Jan" },
  { value: 2, label: "Feb" },
  { value: 3, label: "Mar" },
  { value: 4, label: "Apr" },
];

const mono: CSSProperties = {
  fontFamily: "'IBM Plex Mono', monospace"
};

/* -- component -- */

export default function Performance({
  games,
  seasonLabel: _serverSeasonLabel
}: PerformanceProps) {
  const [edgeMin, setEdgeMin] = useState(0);
  const [edgeMax, setEdgeMax] = useState(50);
  const [diffMin, setDiffMin] = useState(0);
  const [startMonth, setStartMonth] = useState<number | "">("");

  const availableSeasons = useMemo(() => {
    const set = new Set<number>();
    for (const g of games) set.add(getSeasonFromDate(g.date));
    return Array.from(set).sort((a, b) => b - a);
  }, [games]);

  const [seasonFilter, setSeasonFilter] = useState<number | "all">(() =>
    availableSeasons.length > 0 ? availableSeasons[0] : "all"
  );

  const seasonLabel = useMemo(() => {
    if (seasonFilter === "all") return "All Seasons";
    return `${seasonFilter - 1}\u2013${String(seasonFilter).slice(2)} Season`;
  }, [seasonFilter]);

  const minEdge = useMemo(() => {
    if (!games.length) return -10;
    return Math.floor(Math.min(...games.map((g) => g.edge)));
  }, [games]);

  const maxEdge = useMemo(() => {
    if (!games.length) return 30;
    return Math.ceil(Math.max(...games.map((g) => g.edge)));
  }, [games]);

  const maxDiff = useMemo(() => {
    if (!games.length) return 20;
    const diffs = games.filter((g) => g.diff !== null).map((g) => g.diff!);
    return diffs.length > 0 ? Math.ceil(Math.max(...diffs)) : 20;
  }, [games]);

  /* filtered games */
  const filtered = useMemo(
    () =>
      games
        .filter((g) => seasonFilter === "all" || getSeasonFromDate(g.date) === seasonFilter)
        .filter((g) => startMonth === "" || seasonMonthOrd(Number(g.date.slice(5, 7))) >= seasonMonthOrd(startMonth as number))
        .filter((g) => g.edge >= edgeMin && g.edge <= edgeMax)
        .filter((g) => diffMin === 0 || (g.diff !== null && g.diff >= diffMin)),
    [games, seasonFilter, startMonth, edgeMin, edgeMax, diffMin]
  );

  /* stats */
  const stats = useMemo(() => {
    const wins = filtered.filter((g) => g.result === "win").length;
    const losses = filtered.filter((g) => g.result === "loss").length;
    const bets = wins + losses;
    const units = wins * 0.91 - losses;
    const roi = bets > 0 ? (units / bets) * 100 : 0;
    const mae =
      filtered.length > 0
        ? filtered.reduce((s, g) => s + g.model_error, 0) /
          filtered.length
        : null;
    const szGames = filtered.filter((g) => g.sigma_z !== null);
    const sigmaCal =
      szGames.length > 0
        ? szGames.reduce((s, g) => s + g.sigma_z!, 0) / szGames.length
        : null;

    return {
      record: bets > 0 ? `${wins}-${losses}` : "\u2014",
      winPct:
        bets > 0 ? `${((wins / bets) * 100).toFixed(1)}%` : "\u2014",
      winPctNum: bets > 0 ? (wins / bets) * 100 : 50,
      units:
        bets > 0
          ? `${units >= 0 ? "+" : ""}${units.toFixed(1)}u`
          : "\u2014",
      unitsNum: units,
      roi:
        bets > 0
          ? `${roi >= 0 ? "+" : ""}${roi.toFixed(1)}%`
          : "\u2014",
      roiNum: roi,
      mae: mae !== null ? mae.toFixed(2) : "\u2014",
      sigmaCal: sigmaCal !== null ? sigmaCal.toFixed(2) : "\u2014"
    };
  }, [filtered]);

  /* chart data: cumulative units by week */
  const chartData = useMemo(() => {
    const allWeeks: string[] = [];
    const seen = new Set<string>();
    for (const g of filtered) {
      if (!seen.has(g.week)) {
        seen.add(g.week);
        allWeeks.push(g.week);
      }
    }

    let cum = 0;
    return allWeeks.map((w) => {
      const weekGames = filtered.filter((g) => g.week === w);
      for (const g of weekGames) {
        if (g.result === "win") cum += 0.91;
        else if (g.result === "loss") cum -= 1;
      }
      return { week: w, units: Math.round(cum * 10) / 10 };
    });
  }, [filtered]);

  /* monthly breakdown */
  const months = useMemo(() => {
    const map = new Map<
      string,
      {
        month: string;
        sort: number;
        wins: number;
        losses: number;
        pushes: number;
        errors: number[];
        sigmaZs: number[];
        diffs: number[];
      }
    >();

    for (const g of filtered) {
      let entry = map.get(g.month);
      if (!entry) {
        entry = {
          month: g.month,
          sort: g.monthSort,
          wins: 0,
          losses: 0,
          pushes: 0,
          errors: [],
          sigmaZs: [],
          diffs: []
        };
        map.set(g.month, entry);
      }
      if (g.result === "win") entry.wins++;
      else if (g.result === "loss") entry.losses++;
      else entry.pushes++;
      entry.errors.push(g.model_error);
      if (g.sigma_z !== null) entry.sigmaZs.push(g.sigma_z);
      if (g.diff !== null) entry.diffs.push(g.diff);
    }

    return Array.from(map.values()).sort((a, b) => a.sort - b.sort);
  }, [filtered]);

  /* total row */
  const total = useMemo(() => {
    const wins = filtered.filter((g) => g.result === "win").length;
    const losses = filtered.filter((g) => g.result === "loss").length;
    const bets = wins + losses;
    const units = wins * 0.91 - losses;
    const roi = bets > 0 ? (units / bets) * 100 : 0;
    const winPct = bets > 0 ? (wins / bets) * 100 : null;
    const mae =
      filtered.length > 0
        ? filtered.reduce((s, g) => s + g.model_error, 0) /
          filtered.length
        : null;
    const szGames = filtered.filter((g) => g.sigma_z !== null);
    const sigmaCal =
      szGames.length > 0
        ? szGames.reduce((s, g) => s + g.sigma_z!, 0) / szGames.length
        : null;
    const diffGames = filtered.filter((g) => g.diff !== null);
    const meanDiff =
      diffGames.length > 0
        ? diffGames.reduce((s, g) => s + g.diff!, 0) / diffGames.length
        : null;
    const sdDiff =
      diffGames.length > 1 && meanDiff !== null
        ? Math.sqrt(
            diffGames.reduce((s, g) => s + (g.diff! - meanDiff) ** 2, 0) /
              diffGames.length
          )
        : null;
    return { wins, losses, bets, units, roi, winPct, mae, sigmaCal, meanDiff, sdDiff };
  }, [filtered]);

  return (
    <Layout>
      <div>
        {/* -- Title + Edge Slider -- */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: 24,
            flexWrap: "wrap",
            gap: 12
          }}
        >
          <div
            style={{ display: "flex", alignItems: "center", gap: 12 }}
          >
            <h1
              style={{
                fontSize: 24,
                fontWeight: 700,
                letterSpacing: "-0.02em",
                margin: 0,
                color: "#0f172a"
              }}
            >
              Performance
            </h1>
            {availableSeasons.length > 1 && (
              <select
                value={seasonFilter}
                onChange={(e) => {
                  const v = e.target.value;
                  setSeasonFilter(v === "all" ? "all" : Number(v));
                }}
                style={{
                  ...mono,
                  padding: "4px 8px",
                  border: "1px solid #e2e8f0",
                  borderRadius: 6,
                  fontSize: 13,
                  background: "#fff",
                  color: "#334155",
                }}
              >
                {availableSeasons.map((s) => (
                  <option key={s} value={s}>
                    {s - 1}&ndash;{String(s).slice(2)}
                  </option>
                ))}
                <option value="all">All Seasons</option>
              </select>
            )}
            <span style={{ ...mono, fontSize: 13, color: "#64748b" }}>
              {seasonLabel}
            </span>
            <div
              style={{ display: "flex", alignItems: "center", gap: 6 }}
            >
              <span
                style={{
                  ...mono,
                  fontSize: 11,
                  fontWeight: 600,
                  color: "#64748b"
                }}
              >
                FROM
              </span>
              <select
                value={startMonth}
                onChange={(e) => {
                  const v = e.target.value;
                  setStartMonth(v === "" ? "" : Number(v));
                }}
                style={{
                  ...mono,
                  padding: "4px 8px",
                  border: "1px solid #e2e8f0",
                  borderRadius: 6,
                  fontSize: 13,
                  background: "#fff",
                  color: "#334155",
                }}
              >
                <option value="">All</option>
                {SEASON_MONTHS.map((m) => (
                  <option key={m.value} value={m.value}>
                    {m.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div
            style={{ display: "flex", alignItems: "center", gap: 16 }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span
                style={{
                  ...mono,
                  fontSize: 11,
                  fontWeight: 600,
                  color: "#64748b"
                }}
              >
                EDGE
              </span>
              <span
                style={{
                  ...mono,
                  fontSize: 13,
                  fontWeight: 700,
                  color: "#0f172a",
                  minWidth: 28,
                  textAlign: "right"
                }}
              >
                {edgeMin}%
              </span>
              <input
                type="range"
                min={minEdge}
                max={maxEdge}
                step={1}
                value={edgeMin}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  setEdgeMin(Math.min(v, edgeMax));
                }}
                style={{ width: 100, accentColor: "#0f172a" }}
              />
              <span
                style={{
                  ...mono,
                  fontSize: 11,
                  color: "#94a3b8"
                }}
              >
                &ndash;
              </span>
              <input
                type="range"
                min={minEdge}
                max={maxEdge}
                step={1}
                value={edgeMax}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  setEdgeMax(Math.max(v, edgeMin));
                }}
                style={{ width: 100, accentColor: "#0f172a" }}
              />
              <span
                style={{
                  ...mono,
                  fontSize: 13,
                  fontWeight: 700,
                  color: "#0f172a",
                  minWidth: 28
                }}
              >
                {edgeMax}%
              </span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ ...mono, fontSize: 10, color: "#94a3b8", fontWeight: 500 }}>DIFF</span>
              <input type="range" min={0} max={maxDiff} step={1} value={diffMin} onChange={(e) => setDiffMin(Number(e.target.value))} style={{ width: 100, accentColor: "#0f172a" }} />
              <span style={{ ...mono, fontSize: 12, fontWeight: 700, color: "#0f172a", minWidth: 30 }}>{diffMin}</span>
            </div>
          </div>
        </div>

        {/* -- Stats Grid 3x2 -- */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr 1fr",
            gap: 1,
            background: "#e2e8f0",
            borderRadius: 8,
            overflow: "hidden",
            marginBottom: 20
          }}
        >
          {[
            {
              label: "RECORD",
              value: stats.record,
              color: "#0f172a"
            },
            {
              label: "WIN %",
              value: stats.winPct,
              color: stats.winPctNum >= 55 ? "#16a34a" : stats.winPctNum >= 50 ? "#0f172a" : "#dc2626"
            },
            {
              label: "UNITS",
              value: stats.units,
              color:
                stats.unitsNum >= 0 ? "#16a34a" : "#dc2626"
            },
            {
              label: "ROI",
              value: stats.roi,
              color: stats.roiNum >= 0 ? "#16a34a" : "#dc2626"
            },
            {
              label: "MODEL MAE",
              value: stats.mae,
              color: "#0f172a"
            },
            {
              label: "\u03c3 CALIBRATION",
              value: stats.sigmaCal,
              color: "#0f172a"
            }
          ].map((c) => (
            <div
              key={c.label}
              style={{
                background: "#fff",
                padding: "18px 14px",
                textAlign: "center"
              }}
            >
              <div
                style={{
                  ...mono,
                  fontSize: 10,
                  fontWeight: 500,
                  letterSpacing: "0.1em",
                  color: "#64748b",
                  marginBottom: 6
                }}
              >
                {c.label}
              </div>
              <div
                style={{
                  ...mono,
                  fontSize: 22,
                  fontWeight: 700,
                  lineHeight: 1,
                  color: c.color
                }}
              >
                {c.value}
              </div>
            </div>
          ))}
        </div>

        {/* -- Cumulative Units Chart -- */}
        <div style={{ marginBottom: 24 }}>
          <div
            style={{
              ...mono,
              fontSize: 11,
              fontWeight: 500,
              letterSpacing: "0.08em",
              color: "#64748b",
              marginBottom: 10
            }}
          >
            CUMULATIVE UNITS
          </div>
          <div
            style={{
              background: "#fff",
              border: "1px solid #e2e8f0",
              borderRadius: 10,
              padding: "20px 16px 12px",
              boxShadow: "0 1px 3px rgba(0,0,0,0.04)"
            }}
          >
            {chartData.length > 0 ? (
              <CumulativeChart data={chartData} />
            ) : (
              <div
                style={{
                  height: 220,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  color: "#94a3b8",
                  ...mono,
                  fontSize: 13
                }}
              >
                No data at this threshold
              </div>
            )}
          </div>
        </div>

        {/* -- Monthly Breakdown Table -- */}
        <div>
          <div
            style={{
              ...mono,
              fontSize: 11,
              fontWeight: 500,
              letterSpacing: "0.08em",
              color: "#64748b",
              marginBottom: 10
            }}
          >
            MONTHLY BREAKDOWN
          </div>
          <div
            style={{
              background: "#fff",
              border: "1px solid #e2e8f0",
              borderRadius: 10,
              overflow: "hidden",
              boxShadow: "0 1px 3px rgba(0,0,0,0.04)"
            }}
          >
            <div style={{ overflowX: "auto" }}>
              <table
                style={{
                  width: "100%",
                  borderCollapse: "collapse",
                  fontVariantNumeric: "tabular-nums"
                }}
              >
                <thead>
                  <tr>
                    {[
                      { label: "MONTH", align: "left" as const },
                      { label: "RECORD", align: "center" as const },
                      { label: "BETS", align: "center" as const },
                      { label: "WIN %", align: "center" as const },
                      { label: "UNITS", align: "center" as const },
                      { label: "ROI", align: "center" as const },
                      { label: "DIFF", align: "center" as const },
                      { label: "DIFF SD", align: "center" as const },
                      { label: "MAE", align: "center" as const },
                      { label: "\u03c3 CAL", align: "center" as const }
                    ].map((h) => (
                      <th
                        key={h.label}
                        style={{
                          ...mono,
                          fontSize: 10,
                          fontWeight: 600,
                          letterSpacing: "0.08em",
                          padding: "10px 14px",
                          textAlign: h.align,
                          background: "#fafbfc",
                          color: "#64748b",
                          borderBottom: "1px solid #e2e8f0",
                          whiteSpace: "nowrap"
                        }}
                      >
                        {h.label}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {months.length === 0 ? (
                    <tr>
                      <td
                        colSpan={10}
                        style={{
                          padding: 24,
                          textAlign: "center",
                          color: "#94a3b8",
                          borderBottom: "none"
                        }}
                      >
                        No data at this threshold
                      </td>
                    </tr>
                  ) : (
                    <>
                      {months.map((m) => {
                        const bets = m.wins + m.losses;
                        const units = m.wins * 0.91 - m.losses;
                        const roi =
                          bets > 0 ? (units / bets) * 100 : 0;
                        const winPct =
                          bets > 0 ? (m.wins / bets) * 100 : null;
                        const meanDiff =
                          m.diffs.length > 0
                            ? m.diffs.reduce((s, e) => s + e, 0) /
                              m.diffs.length
                            : null;
                        const sdDiff =
                          m.diffs.length > 1 && meanDiff !== null
                            ? Math.sqrt(
                                m.diffs.reduce((s, e) => s + (e - meanDiff) ** 2, 0) /
                                  m.diffs.length
                              )
                            : null;
                        const mae =
                          m.errors.length > 0
                            ? m.errors.reduce((s, e) => s + e, 0) /
                              m.errors.length
                            : null;
                        const sc =
                          m.sigmaZs.length > 0
                            ? m.sigmaZs.reduce((s, e) => s + e, 0) /
                              m.sigmaZs.length
                            : null;
                        const bd = "1px solid #f1f5f9";

                        return (
                          <tr key={m.month}>
                            <td
                              style={{
                                padding: "10px 14px",
                                textAlign: "left",
                                fontSize: 14,
                                fontWeight: 600,
                                color: "#0f172a",
                                borderBottom: bd
                              }}
                            >
                              {m.month}
                            </td>
                            <td
                              style={{
                                ...mono,
                                padding: "10px 14px",
                                textAlign: "center",
                                fontSize: 14,
                                fontWeight: 700,
                                color: "#0f172a",
                                borderBottom: bd
                              }}
                            >
                              {bets > 0
                                ? `${m.wins}-${m.losses}`
                                : "\u2014"}
                            </td>
                            <td
                              style={{
                                ...mono,
                                padding: "10px 14px",
                                textAlign: "center",
                                fontSize: 13,
                                color: "#64748b",
                                borderBottom: bd
                              }}
                            >
                              {bets || "\u2014"}
                            </td>
                            <td
                              style={{
                                ...mono,
                                padding: "10px 14px",
                                textAlign: "center",
                                fontSize: 13,
                                fontWeight: 600,
                                color: winPctColor(winPct),
                                borderBottom: bd
                              }}
                            >
                              {winPct !== null
                                ? `${winPct.toFixed(1)}%`
                                : "\u2014"}
                            </td>
                            <td
                              style={{
                                ...mono,
                                padding: "10px 14px",
                                textAlign: "center",
                                fontSize: 13,
                                fontWeight: 700,
                                color:
                                  units >= 0 ? "#16a34a" : "#dc2626",
                                borderBottom: bd
                              }}
                            >
                              {bets > 0
                                ? `${units >= 0 ? "+" : ""}${units.toFixed(1)}u`
                                : "\u2014"}
                            </td>
                            <td
                              style={{
                                ...mono,
                                padding: "10px 14px",
                                textAlign: "center",
                                fontSize: 13,
                                fontWeight: 700,
                                color:
                                  roi >= 0 ? "#16a34a" : "#dc2626",
                                borderBottom: bd
                              }}
                            >
                              {bets > 0
                                ? `${roi >= 0 ? "+" : ""}${roi.toFixed(1)}%`
                                : "\u2014"}
                            </td>
                            <td
                              style={{
                                ...mono,
                                padding: "10px 14px",
                                textAlign: "center",
                                fontSize: 13,
                                fontWeight: 600,
                                color: "#334155",
                                borderBottom: bd
                              }}
                            >
                              {meanDiff !== null ? meanDiff.toFixed(1) : "\u2014"}
                            </td>
                            <td
                              style={{
                                ...mono,
                                padding: "10px 14px",
                                textAlign: "center",
                                fontSize: 13,
                                fontWeight: 600,
                                color: "#64748b",
                                borderBottom: bd
                              }}
                            >
                              {sdDiff !== null ? sdDiff.toFixed(1) : "\u2014"}
                            </td>
                            <td
                              style={{
                                ...mono,
                                padding: "10px 14px",
                                textAlign: "center",
                                fontSize: 13,
                                fontWeight: 600,
                                color: "#0f172a",
                                borderBottom: bd
                              }}
                            >
                              {mae !== null ? mae.toFixed(2) : "\u2014"}
                            </td>
                            <td
                              style={{
                                ...mono,
                                padding: "10px 14px",
                                textAlign: "center",
                                fontSize: 13,
                                fontWeight: 600,
                                color: "#0f172a",
                                borderBottom: bd
                              }}
                            >
                              {sc !== null ? sc.toFixed(2) : "\u2014"}
                            </td>
                          </tr>
                        );
                      })}

                      {/* Total row */}
                      <tr>
                        <td
                          style={{
                            padding: "10px 14px",
                            textAlign: "left",
                            fontSize: 14,
                            fontWeight: 800,
                            color: "#0f172a",
                            borderTop: "2px solid #e2e8f0",
                            background: "#fafbfc"
                          }}
                        >
                          Total
                        </td>
                        <td
                          style={{
                            ...mono,
                            padding: "10px 14px",
                            textAlign: "center",
                            fontSize: 14,
                            fontWeight: 800,
                            color: "#0f172a",
                            borderTop: "2px solid #e2e8f0",
                            background: "#fafbfc"
                          }}
                        >
                          {total.bets > 0
                            ? `${total.wins}-${total.losses}`
                            : "\u2014"}
                        </td>
                        <td
                          style={{
                            ...mono,
                            padding: "10px 14px",
                            textAlign: "center",
                            fontSize: 13,
                            fontWeight: 800,
                            color: "#0f172a",
                            borderTop: "2px solid #e2e8f0",
                            background: "#fafbfc"
                          }}
                        >
                          {total.bets || "\u2014"}
                        </td>
                        <td
                          style={{
                            ...mono,
                            padding: "10px 14px",
                            textAlign: "center",
                            fontSize: 13,
                            fontWeight: 800,
                            color: winPctColor(total.winPct),
                            borderTop: "2px solid #e2e8f0",
                            background: "#fafbfc"
                          }}
                        >
                          {total.winPct !== null
                            ? `${total.winPct.toFixed(1)}%`
                            : "\u2014"}
                        </td>
                        <td
                          style={{
                            ...mono,
                            padding: "10px 14px",
                            textAlign: "center",
                            fontSize: 13,
                            fontWeight: 800,
                            color:
                              total.units >= 0
                                ? "#16a34a"
                                : "#dc2626",
                            borderTop: "2px solid #e2e8f0",
                            background: "#fafbfc"
                          }}
                        >
                          {total.bets > 0
                            ? `${total.units >= 0 ? "+" : ""}${total.units.toFixed(1)}u`
                            : "\u2014"}
                        </td>
                        <td
                          style={{
                            ...mono,
                            padding: "10px 14px",
                            textAlign: "center",
                            fontSize: 13,
                            fontWeight: 800,
                            color:
                              total.roi >= 0
                                ? "#16a34a"
                                : "#dc2626",
                            borderTop: "2px solid #e2e8f0",
                            background: "#fafbfc"
                          }}
                        >
                          {total.bets > 0
                            ? `${total.roi >= 0 ? "+" : ""}${total.roi.toFixed(1)}%`
                            : "\u2014"}
                        </td>
                        <td
                          style={{
                            ...mono,
                            padding: "10px 14px",
                            textAlign: "center",
                            fontSize: 13,
                            fontWeight: 800,
                            color: "#334155",
                            borderTop: "2px solid #e2e8f0",
                            background: "#fafbfc"
                          }}
                        >
                          {total.meanDiff !== null
                            ? total.meanDiff.toFixed(1)
                            : "\u2014"}
                        </td>
                        <td
                          style={{
                            ...mono,
                            padding: "10px 14px",
                            textAlign: "center",
                            fontSize: 13,
                            fontWeight: 800,
                            color: "#64748b",
                            borderTop: "2px solid #e2e8f0",
                            background: "#fafbfc"
                          }}
                        >
                          {total.sdDiff !== null
                            ? total.sdDiff.toFixed(1)
                            : "\u2014"}
                        </td>
                        <td
                          style={{
                            ...mono,
                            padding: "10px 14px",
                            textAlign: "center",
                            fontSize: 13,
                            fontWeight: 800,
                            color: "#0f172a",
                            borderTop: "2px solid #e2e8f0",
                            background: "#fafbfc"
                          }}
                        >
                          {total.mae !== null
                            ? total.mae.toFixed(2)
                            : "\u2014"}
                        </td>
                        <td
                          style={{
                            ...mono,
                            padding: "10px 14px",
                            textAlign: "center",
                            fontSize: 13,
                            fontWeight: 800,
                            color: "#0f172a",
                            borderTop: "2px solid #e2e8f0",
                            background: "#fafbfc"
                          }}
                        >
                          {total.sigmaCal !== null
                            ? total.sigmaCal.toFixed(2)
                            : "\u2014"}
                        </td>
                      </tr>
                    </>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}

function winPctColor(pct: number | null): string {
  if (pct === null) return "#94a3b8";
  if (pct >= 55) return "#16a34a";
  if (pct >= 50) return "#0f172a";
  return "#dc2626";
}
