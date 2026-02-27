import { GetServerSideProps } from "next";
import Link from "next/link";
import { CSSProperties, useMemo, useState } from "react";
import Layout from "../components/Layout";
import { normalizeRows } from "../lib/data";
import { listFinalScoreFiles, readJsonFile } from "../lib/server-data";

/* -- types -- */

type HistoryGame = {
  away_team: string;
  home_team: string;
  away_score: number | null;
  home_score: number | null;
  pick_side: string;
  pick_team: string;
  market_spread_home: number | null;
  model_mu_home: number | null;
  pick_prob_edge: number;
  ats_result: "win" | "loss" | "push" | null;
  has_book: boolean;
};

type HistoryProps = {
  date: string | null;
  games: HistoryGame[];
  prevDate: string | null;
  nextDate: string | null;
};

type SortKey =
  | "matchup"
  | "score"
  | "pick"
  | "book"
  | "model"
  | "ats"
  | "edge";

type SortState = { key: SortKey; dir: "asc" | "desc" };

/* -- server -- */

export const getServerSideProps: GetServerSideProps<HistoryProps> = async (
  context
) => {
  const queryDate =
    typeof context.query.date === "string" ? context.query.date : null;

  const finalFiles = listFinalScoreFiles();
  const today = new Date().toISOString().slice(0, 10);
  const availableDates = finalFiles.map((f) => f.date).filter((d) => d < today).sort();

  if (!availableDates.length) {
    return {
      props: { date: null, games: [], prevDate: null, nextDate: null }
    };
  }

  const date =
    queryDate && availableDates.includes(queryDate)
      ? queryDate
      : availableDates[availableDates.length - 1];

  const idx = availableDates.indexOf(date);
  const prevDate = idx > 0 ? availableDates[idx - 1] : null;
  const nextDate =
    idx < availableDates.length - 1 ? availableDates[idx + 1] : null;

  const predRows = normalizeRows(
    readJsonFile(`predictions_${date}.json`)
  );
  const finalRows = normalizeRows(
    readJsonFile(`final_scores_${date}.json`)
  );

  const finalLookup = new Map<string, Record<string, unknown>>();
  for (const r of finalRows) {
    const gid = r.game_id;
    if (typeof gid === "string") finalLookup.set(gid, r);
  }

  const games: HistoryGame[] = predRows.map((pred) => {
    const gid = typeof pred.game_id === "string" ? pred.game_id : "";
    const fin = finalLookup.get(gid);

    const away_team = s(pred.away_team);
    const home_team = s(pred.home_team);
    const pick_side = s(pred.pick_side).toUpperCase();
    const pick_team = pick_side === "HOME" ? home_team : away_team;

    const market_spread_home = pn(pred.market_spread_home);
    const model_mu_home = pn(pred.model_mu_home);
    const pick_prob_edge = pn(pred.pick_prob_edge) ?? 0;
    const has_book = market_spread_home !== null;

    const away_score = fin ? pn(fin.away_score) : null;
    const home_score = fin ? pn(fin.home_score) : null;

    let ats_result: HistoryGame["ats_result"] = null;
    if (
      away_score !== null &&
      home_score !== null &&
      market_spread_home !== null &&
      pick_side
    ) {
      const cover = home_score - away_score + market_spread_home;
      if (cover === 0) {
        ats_result = "push";
      } else if (pick_side === "HOME") {
        ats_result = cover > 0 ? "win" : "loss";
      } else {
        ats_result = cover < 0 ? "win" : "loss";
      }
    }

    return {
      away_team,
      home_team,
      away_score,
      home_score,
      pick_side,
      pick_team,
      market_spread_home,
      model_mu_home,
      pick_prob_edge,
      ats_result,
      has_book
    };
  });

  return { props: { date, games, prevDate, nextDate } };
};

function s(v: unknown): string {
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

/* -- helpers -- */

const mono: CSSProperties = {
  fontFamily: "'IBM Plex Mono', monospace"
};

function sp(v: number): string {
  return v > 0 ? `+${v.toFixed(1)}` : v.toFixed(1);
}

function fmtDate(d: string): string {
  const [yr, mo, dy] = d.split("-");
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
  return `${months[Number(mo) - 1]} ${Number(dy)}, ${yr}`;
}

/* sort helpers */

function atsOrd(r: "win" | "loss" | "push" | null): number {
  if (r === "win") return 2;
  if (r === "push") return 1;
  if (r === "loss") return 0;
  return -1;
}

function sortVal(g: HistoryGame, key: SortKey): string | number {
  switch (key) {
    case "matchup":
      return `${g.away_team} @ ${g.home_team}`;
    case "score":
      return g.home_score !== null && g.away_score !== null
        ? g.home_score - g.away_score
        : -Infinity;
    case "pick":
      return g.pick_team;
    case "book":
      return g.market_spread_home ?? -Infinity;
    case "model":
      return g.model_mu_home ?? -Infinity;
    case "ats":
      return atsOrd(g.ats_result);
    case "edge":
      return g.pick_prob_edge;
  }
}

/* -- column definitions -- */

const columns: { key: SortKey; label: string; align: "left" | "center" }[] = [
  { key: "matchup", label: "MATCHUP", align: "left" },
  { key: "score", label: "SCORE", align: "center" },
  { key: "pick", label: "PICK", align: "center" },
  { key: "book", label: "BOOK LINE", align: "center" },
  { key: "model", label: "MODEL", align: "center" },
  { key: "ats", label: "ATS", align: "center" },
  { key: "edge", label: "EDGE", align: "center" }
];

/* -- component -- */

export default function History({
  date,
  games,
  prevDate,
  nextDate
}: HistoryProps) {
  const [search, setSearch] = useState("");
  const [edgeMin, setEdgeMin] = useState(0);
  const [resultFilter, setResultFilter] = useState<
    "all" | "wins" | "losses"
  >("all");
  const [sort, setSort] = useState<SortState>({
    key: "edge",
    dir: "desc"
  });

  const maxEdge = useMemo(() => {
    if (!games.length) return 30;
    return Math.ceil(
      Math.max(...games.map((g) => g.pick_prob_edge * 100))
    );
  }, [games]);

  /* compute daily stats from games above threshold */
  const stats = useMemo(() => {
    const above = games.filter(
      (g) =>
        g.has_book &&
        g.pick_prob_edge * 100 >= edgeMin &&
        g.ats_result !== null
    );
    const wins = above.filter((g) => g.ats_result === "win").length;
    const losses = above.filter((g) => g.ats_result === "loss").length;
    const bets = wins + losses;
    const units = wins * 0.91 - losses;
    const roi = bets > 0 ? (units / bets) * 100 : 0;
    const winRate = bets > 0 ? (wins / bets) * 100 : 0;
    return {
      record: bets > 0 ? `${wins}-${losses}` : "\u2014",
      winRate: bets > 0 ? `${winRate.toFixed(1)}%` : "\u2014",
      units: bets > 0 ? `${units >= 0 ? "+" : ""}${units.toFixed(1)}u` : "\u2014",
      roi: bets > 0 ? `${roi >= 0 ? "+" : ""}${roi.toFixed(1)}%` : "\u2014",
      bets: String(bets),
      unitsNum: units,
      roiNum: roi
    };
  }, [games, edgeMin]);

  /* filtered + sorted rows */
  const tableRows = useMemo(() => {
    let list = [...games];

    if (search.trim()) {
      const q = search.trim().toLowerCase();
      list = list.filter(
        (g) =>
          g.away_team.toLowerCase().includes(q) ||
          g.home_team.toLowerCase().includes(q) ||
          g.pick_team.toLowerCase().includes(q)
      );
    }

    if (resultFilter === "wins") {
      list = list.filter((g) => g.ats_result === "win");
    } else if (resultFilter === "losses") {
      list = list.filter((g) => g.ats_result === "loss");
    }

    list.sort((a, b) => {
      const aHas = a.has_book;
      const bHas = b.has_book;
      if (aHas !== bHas) return aHas ? -1 : 1;

      const av = sortVal(a, sort.key);
      const bv = sortVal(b, sort.key);
      if (typeof av === "number" && typeof bv === "number") {
        return sort.dir === "asc" ? av - bv : bv - av;
      }
      const cmp = String(av).localeCompare(String(bv));
      return sort.dir === "asc" ? cmp : -cmp;
    });

    return list;
  }, [games, search, resultFilter, sort]);

  function handleSort(key: SortKey) {
    setSort((prev) =>
      prev.key === key
        ? { key, dir: prev.dir === "desc" ? "asc" : "desc" }
        : { key, dir: "desc" }
    );
  }

  if (!date) {
    return (
      <Layout>
        <div style={{ padding: 24, color: "#94a3b8", textAlign: "center" }}>
          No historical data available.
        </div>
      </Layout>
    );
  }

  const edgeThreshold = edgeMin / 100;

  return (
    <Layout>
      <div>
        {/* -- Title + Date Nav -- */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: 24
          }}
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
            History
          </h1>

          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            {prevDate ? (
              <Link
                href={`/history?date=${prevDate}`}
                style={{
                  width: 32,
                  height: 32,
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  background: "#fff",
                  border: "1px solid #e2e8f0",
                  borderRadius: 6,
                  color: "#334155",
                  fontSize: 16,
                  textDecoration: "none"
                }}
              >
                \u2190
              </Link>
            ) : (
              <span
                style={{
                  width: 32,
                  height: 32,
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  background: "#fff",
                  border: "1px solid #e2e8f0",
                  borderRadius: 6,
                  color: "#e2e8f0",
                  fontSize: 16
                }}
              >
                \u2190
              </span>
            )}

            <span
              style={{
                ...mono,
                fontSize: 14,
                fontWeight: 600,
                background: "#fff",
                border: "1px solid #e2e8f0",
                borderRadius: 6,
                padding: "6px 14px",
                color: "#0f172a"
              }}
            >
              {fmtDate(date)}
            </span>

            {nextDate ? (
              <Link
                href={`/history?date=${nextDate}`}
                style={{
                  width: 32,
                  height: 32,
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  background: "#fff",
                  border: "1px solid #e2e8f0",
                  borderRadius: 6,
                  color: "#334155",
                  fontSize: 16,
                  textDecoration: "none"
                }}
              >
                \u2192
              </Link>
            ) : (
              <span
                style={{
                  width: 32,
                  height: 32,
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  background: "#fff",
                  border: "1px solid #e2e8f0",
                  borderRadius: 6,
                  color: "#e2e8f0",
                  fontSize: 16
                }}
              >
                \u2192
              </span>
            )}
          </div>
        </div>

        {/* -- Daily Stats Strip -- */}
        <div
          style={{
            display: "flex",
            gap: 1,
            borderRadius: 8,
            overflow: "hidden",
            background: "#e2e8f0",
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
              label: "WIN RATE",
              value: stats.winRate,
              color: "#0f172a"
            },
            {
              label: "UNITS",
              value: stats.units,
              color: stats.unitsNum >= 0 ? "#16a34a" : "#dc2626"
            },
            {
              label: "ROI",
              value: stats.roi,
              color: stats.roiNum >= 0 ? "#16a34a" : "#dc2626"
            },
            {
              label: "BETS",
              value: stats.bets,
              color: "#0f172a"
            }
          ].map((c) => (
            <div
              key={c.label}
              style={{
                flex: 1,
                background: "#fff",
                padding: "12px 8px",
                textAlign: "center"
              }}
            >
              <div
                style={{
                  ...mono,
                  fontSize: 9,
                  fontWeight: 500,
                  letterSpacing: "0.1em",
                  color: "#64748b",
                  marginBottom: 4
                }}
              >
                {c.label}
              </div>
              <div
                style={{
                  ...mono,
                  fontSize: 16,
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

        {/* -- Controls Row -- */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: 10,
            flexWrap: "wrap",
            gap: 8
          }}
        >
          {/* Left: count + search */}
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontSize: 13, color: "#64748b" }}>
              {games.length} games
            </span>
            <input
              type="text"
              placeholder="Search team..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              style={{
                ...mono,
                width: 140,
                padding: "6px 10px",
                border: "1px solid #e2e8f0",
                borderRadius: 6,
                fontSize: 12,
                outline: "none",
                background: "#fff",
                color: "#334155"
              }}
            />
          </div>

          {/* Center: edge slider */}
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span
              style={{
                ...mono,
                fontSize: 10,
                color: "#94a3b8",
                fontWeight: 500
              }}
            >
              EDGE
            </span>
            <input
              type="range"
              min={0}
              max={maxEdge}
              step={1}
              value={edgeMin}
              onChange={(e) => setEdgeMin(Number(e.target.value))}
              style={{ width: 140, accentColor: "#0f172a" }}
            />
            <span
              style={{
                ...mono,
                fontSize: 12,
                fontWeight: 700,
                color: "#0f172a",
                minWidth: 30
              }}
            >
              {edgeMin}%
            </span>
          </div>

          {/* Right: filter buttons */}
          <div style={{ display: "flex", gap: 6 }}>
            {(["all", "wins", "losses"] as const).map((f) => (
              <button
                key={f}
                type="button"
                onClick={() => setResultFilter(f)}
                style={{
                  ...mono,
                  fontSize: 12,
                  fontWeight: 500,
                  padding: "5px 12px",
                  borderRadius: 6,
                  border: `1px solid ${
                    resultFilter === f ? "#0f172a" : "#e2e8f0"
                  }`,
                  background: resultFilter === f ? "#0f172a" : "#fff",
                  color: resultFilter === f ? "#fff" : "#64748b",
                  cursor: "pointer",
                  textTransform: "capitalize"
                }}
              >
                {f === "all" ? "All" : f === "wins" ? "Wins" : "Losses"}
              </button>
            ))}
          </div>
        </div>

        {/* -- Table -- */}
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
                  {columns.map((col) => {
                    const active = sort.key === col.key;
                    return (
                      <th
                        key={col.key}
                        onClick={() => handleSort(col.key)}
                        style={{
                          ...mono,
                          fontSize: 10,
                          fontWeight: 600,
                          letterSpacing: "0.08em",
                          padding: "10px 14px",
                          textAlign: col.align,
                          background: "#fafbfc",
                          color: active ? "#0f172a" : "#64748b",
                          borderBottom: "1px solid #e2e8f0",
                          cursor: "pointer",
                          userSelect: "none",
                          whiteSpace: "nowrap",
                          ...(col.key === "matchup" ? { width: "1%" } : {})
                        }}
                      >
                        {col.label}
                        {active && (
                          <span style={{ marginLeft: 4 }}>
                            {sort.dir === "desc" ? "\u2193" : "\u2191"}
                          </span>
                        )}
                      </th>
                    );
                  })}
                </tr>
              </thead>
              <tbody>
                {tableRows.length === 0 ? (
                  <tr>
                    <td
                      colSpan={columns.length}
                      style={{
                        padding: 24,
                        textAlign: "center",
                        color: "#94a3b8",
                        borderBottom: "none"
                      }}
                    >
                      No games found
                    </td>
                  </tr>
                ) : (
                  tableRows.map((g, i) => {
                    const aboveThreshold =
                      g.has_book && g.pick_prob_edge >= edgeThreshold;
                    const dimmed = !aboveThreshold;
                    const bd = "1px solid #f1f5f9";

                    return (
                      <tr
                        key={`${g.away_team}-${g.home_team}-${i}`}
                        style={{
                          animation: `fadeIn 0.3s ease ${i * 0.02}s both`
                        }}
                      >
                        {/* MATCHUP */}
                        <td
                          style={{
                            padding: "10px 14px",
                            textAlign: "left",
                            fontSize: 14,
                            color: "#334155",
                            whiteSpace: "nowrap",
                            width: "1%",
                            borderBottom: bd,
                            opacity: dimmed ? 0.4 : 1
                          }}
                        >
                          {g.away_team} @ {g.home_team}
                        </td>

                        {/* SCORE */}
                        <td
                          style={{
                            ...mono,
                            padding: "10px 14px",
                            textAlign: "center",
                            fontSize: 13,
                            color: "#334155",
                            borderBottom: bd,
                            opacity: dimmed ? 0.4 : 1
                          }}
                        >
                          {g.away_score !== null && g.home_score !== null
                            ? `${g.away_score}-${g.home_score}`
                            : "\u2014"}
                        </td>

                        {/* PICK */}
                        <td
                          style={{
                            ...mono,
                            padding: "10px 14px",
                            textAlign: "center",
                            fontSize: 13,
                            fontWeight: 700,
                            color: "#0f172a",
                            borderBottom: bd,
                            opacity: dimmed ? 0.4 : 1
                          }}
                        >
                          {g.pick_team}
                        </td>

                        {/* BOOK LINE */}
                        <td
                          style={{
                            ...mono,
                            padding: "10px 14px",
                            textAlign: "center",
                            fontSize: 13,
                            color: "#334155",
                            borderBottom: bd,
                            opacity: dimmed ? 0.4 : 1
                          }}
                        >
                          {g.has_book && g.market_spread_home !== null
                            ? sp(g.market_spread_home)
                            : "\u2014"}
                        </td>

                        {/* MODEL */}
                        <td
                          style={{
                            ...mono,
                            padding: "10px 14px",
                            textAlign: "center",
                            fontSize: 14,
                            fontWeight: 700,
                            color: "#0f172a",
                            borderBottom: bd,
                            opacity: dimmed ? 0.4 : 1
                          }}
                        >
                          {g.model_mu_home !== null
                            ? sp(g.model_mu_home)
                            : "\u2014"}
                        </td>

                        {/* ATS -- never dimmed */}
                        <td
                          style={{
                            padding: "10px 14px",
                            textAlign: "center",
                            borderBottom: bd
                          }}
                        >
                          {!g.has_book ? (
                            <span
                              style={{
                                ...mono,
                                fontSize: 11,
                                fontWeight: 600,
                                color: "#94a3b8"
                              }}
                            >
                              \u2014
                            </span>
                          ) : !aboveThreshold ? (
                            <span
                              style={{
                                ...mono,
                                display: "inline-block",
                                fontSize: 11,
                                fontWeight: 600,
                                color: "#94a3b8",
                                background: "#f1f5f9",
                                padding: "3px 8px",
                                borderRadius: 5
                              }}
                            >
                              NO BET
                            </span>
                          ) : (
                            <AtsBadge result={g.ats_result} />
                          )}
                        </td>

                        {/* EDGE */}
                        <td
                          style={{
                            ...mono,
                            padding: "10px 14px",
                            textAlign: "center",
                            fontSize: 13,
                            fontWeight: 700,
                            color: g.has_book
                              ? g.pick_prob_edge >= 0
                                ? "#16a34a"
                                : "#dc2626"
                              : "#94a3b8",
                            borderBottom: bd,
                            opacity: dimmed ? 0.4 : 1
                          }}
                        >
                          {g.has_book
                            ? `${g.pick_prob_edge >= 0 ? "+" : ""}${(
                                g.pick_prob_edge * 100
                              ).toFixed(1)}%`
                            : "\u2014"}
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </Layout>
  );
}

/* -- ATS badge sub-component -- */

function AtsBadge({
  result
}: {
  result: "win" | "loss" | "push" | null;
}) {
  if (!result) {
    return (
      <span
        style={{
          ...mono,
          fontSize: 11,
          fontWeight: 600,
          color: "#94a3b8"
        }}
      >
        \u2014
      </span>
    );
  }

  const config = {
    win: {
      label: "\u2713 WIN",
      color: "#16a34a",
      bg: "#16a34a0d",
      border: "#16a34a20"
    },
    loss: {
      label: "\u2717 LOSS",
      color: "#dc2626",
      bg: "#dc26260d",
      border: "#dc262620"
    },
    push: {
      label: "PUSH",
      color: "#64748b",
      bg: "#f1f5f9",
      border: "#e2e8f0"
    }
  }[result];

  return (
    <span
      style={{
        ...mono,
        display: "inline-block",
        fontSize: 11,
        fontWeight: 600,
        color: config.color,
        background: config.bg,
        border: `1px solid ${config.border}`,
        padding: "3px 8px",
        borderRadius: 5
      }}
    >
      {config.label}
    </span>
  );
}
