import { GetServerSideProps } from "next";
import { CSSProperties, useMemo, useState } from "react";
import Layout from "../components/Layout";
import { PredictionRow, displayTeam } from "../lib/data";
import {
  getLatestPredictionFile,
  getPredictionRowsByFilename,
} from "../lib/server-data";

type HomeProps = {
  date: string | null;
  rows: PredictionRow[];
};

export const getServerSideProps: GetServerSideProps<HomeProps> = async () => {
  const latest = getLatestPredictionFile();
  if (!latest) {
    return { props: { date: null, rows: [] } };
  }
  const rows = getPredictionRowsByFilename(latest.filename);
  return { props: { date: latest.date, rows } };
};

/* -- helpers -- */

const mono: CSSProperties = {
  fontFamily: "'IBM Plex Mono', monospace"
};

function formatSpread(v: number): string {
  return v > 0 ? `+${v.toFixed(1)}` : v.toFixed(1);
}

function formatDateDisplay(dateStr: string): string {
  const [year, month, day] = dateStr.split("-");
  const months = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
  ];
  return `${months[Number(month) - 1]} ${Number(day)}, ${year}`;
}

function str(v: unknown): string {
  return typeof v === "string" ? v : String(v ?? "");
}

function num(v: unknown): number | null {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string" && v.trim() !== "") {
    const n = Number(v);
    return Number.isNaN(n) ? null : n;
  }
  return null;
}

function getPickTeam(row: PredictionRow): string {
  const side = str(row.pick_side).toUpperCase();
  return displayTeam(side === "HOME" ? str(row.home_team) : str(row.away_team));
}

function formatGameTime(row: PredictionRow): string | null {
  const raw = row.start_time ?? row.startDate;
  if (!raw || typeof raw !== "string") return null;
  try {
    const d = new Date(raw);
    if (isNaN(d.getTime())) return null;
    return d.toLocaleTimeString("en-US", {
      timeZone: "America/New_York",
      hour: "numeric",
      minute: "2-digit",
    });
  } catch {
    return null;
  }
}

function hasBook(row: PredictionRow): boolean {
  return row.market_spread_home !== null && row.market_spread_home !== undefined && row.market_spread_home !== "";
}

function bookSpread(row: PredictionRow): number | null {
  return num(row.market_spread_home);
}

function modelSpread(row: PredictionRow): number | null {
  const v = num(row.model_mu_home);
  return v !== null ? -v : null; // Negate: model_mu_home is home-away, display as book convention
}

function sigma(row: PredictionRow): number | null {
  return num(row.pred_sigma);
}

function edge(row: PredictionRow): number {
  return num(row.pick_prob_edge) ?? 0;
}

function diff(row: PredictionRow): number | null {
  const m = modelSpread(row);
  const b = bookSpread(row);
  if (m === null || b === null) return null;
  return Math.abs(m - b);
}

function pickSpread(row: PredictionRow): number | null {
  const b = bookSpread(row);
  if (b === null) return null;
  return str(row.pick_side).toUpperCase() === "HOME" ? b : -b;
}

/* -- sort -- */

type SortKey = "matchup" | "book" | "model" | "sigma" | "diff" | "edge";

type SortState = { key: SortKey; dir: "asc" | "desc" };

function sortVal(row: PredictionRow, key: SortKey): string | number {
  switch (key) {
    case "matchup":
      return `${displayTeam(str(row.away_team))} @ ${displayTeam(str(row.home_team))}`;
    case "book":
      return bookSpread(row) ?? -Infinity;
    case "model":
      return modelSpread(row) ?? -Infinity;
    case "sigma":
      return sigma(row) ?? -Infinity;
    case "diff":
      return diff(row) ?? -Infinity;
    case "edge":
      return edge(row);
  }
}

/* -- column defs -- */

const columns: { key: SortKey; label: string; align: "left" | "center" }[] = [
  { key: "matchup", label: "MATCHUP", align: "left" },
  { key: "book", label: "HOME SPREAD", align: "center" },
  { key: "model", label: "MODEL", align: "center" },
  { key: "sigma", label: "SIGMA", align: "center" },
  { key: "diff", label: "DIFF", align: "center" },
  { key: "edge", label: "EDGE", align: "center" }
];

/* -- component -- */

export default function Home({ date, rows }: HomeProps) {
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<"all" | "edge10">("all");
  const [diffMin, setDiffMin] = useState(0);
  const [sort, setSort] = useState<SortState>({ key: "edge", dir: "desc" });

  const maxDiff = useMemo(() => {
    if (!rows.length) return 20;
    const diffs = rows.map((r) => diff(r)).filter((d): d is number => d !== null);
    return diffs.length > 0 ? Math.ceil(Math.max(...diffs)) : 20;
  }, [rows]);

  const tableRows = useMemo(() => {
    let list = [...rows];

    if (search.trim()) {
      const q = search.trim().toLowerCase();
      list = list.filter((r) => {
        const a = str(r.away_team).toLowerCase();
        const h = str(r.home_team).toLowerCase();
        const p = getPickTeam(r).toLowerCase();
        return a.includes(q) || h.includes(q) || p.includes(q);
      });
    }

    if (filter === "edge10") {
      list = list.filter((r) => hasBook(r) && edge(r) >= 0.10);
    }

    if (diffMin > 0) {
      list = list.filter((r) => {
        const d = diff(r);
        return d === null || d >= diffMin;
      });
    }

    list.sort((a, b) => {
      const aHas = hasBook(a);
      const bHas = hasBook(b);
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
  }, [rows, search, filter, diffMin, sort]);

  function handleSort(key: SortKey) {
    setSort((prev) =>
      prev.key === key
        ? { key, dir: prev.dir === "desc" ? "asc" : "desc" }
        : { key, dir: "desc" }
    );
  }

  if (!rows.length) {
    return (
      <Layout>
        <div style={{ padding: 24, color: "#94a3b8", textAlign: "center" }}>
          No games found for today.
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      {/* single wrapper so .content gap doesn't add extra spacing */}
      <div>
        {/* -- Title Row -- */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "baseline",
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
            Today&apos;s Picks
          </h1>
          <span style={{ ...mono, fontSize: 13, color: "#64748b" }}>
            {date ? formatDateDisplay(date) : ""} · {rows.length} games
          </span>
        </div>

        {/* -- All Games Table -- */}
        <div>
          {/* Controls row */}
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: 10
            }}
          >
            <span style={{ fontSize: 13, fontWeight: 500, color: "#64748b" }}>
              All Games
            </span>

            <input
              type="text"
              placeholder="Search team..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              style={{
                ...mono,
                width: 180,
                padding: "6px 10px",
                border: "1px solid #e2e8f0",
                borderRadius: 6,
                fontSize: 13,
                outline: "none",
                background: "#fff",
                color: "#334155"
              }}
            />

            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ ...mono, fontSize: 10, color: "#94a3b8", fontWeight: 500 }}>DIFF</span>
              <input type="range" min={0} max={maxDiff} step={1} value={diffMin} onChange={(e) => setDiffMin(Number(e.target.value))} style={{ width: 100, accentColor: "#0f172a" }} />
              <span style={{ ...mono, fontSize: 12, fontWeight: 700, color: "#0f172a", minWidth: 30 }}>{diffMin}</span>
            </div>

            <div style={{ display: "flex", gap: 6 }}>
              {(["all", "edge10"] as const).map((f) => (
                <button
                  key={f}
                  type="button"
                  onClick={() => setFilter(f)}
                  style={{
                    ...mono,
                    fontSize: 12,
                    fontWeight: 500,
                    padding: "5px 12px",
                    borderRadius: 6,
                    border: `1px solid ${filter === f ? "#0f172a" : "#e2e8f0"}`,
                    background: filter === f ? "#0f172a" : "#fff",
                    color: filter === f ? "#fff" : "#64748b",
                    cursor: "pointer"
                  }}
                >
                  {f === "all" ? "All" : "Edge \u2265 10%"}
                </button>
              ))}
            </div>
          </div>

          {/* Table container */}
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
                    tableRows.map((row, i) => {
                      const bk = bookSpread(row);
                      const md = modelSpread(row);
                      const sg = sigma(row);
                      const df = diff(row);
                      const eg = edge(row);
                      const hb = hasBook(row);

                      return (
                        <tr
                          key={`${str(row.away_team)}-${str(row.home_team)}-${i}`}
                          style={{
                            borderBottom: "1px solid #f1f5f9",
                            animation: `fadeIn 0.3s ease ${i * 0.02}s both`
                          }}
                        >
                          {/* MATCHUP — picked side is bold */}
                          <td
                            style={{
                              padding: "10px 14px",
                              textAlign: "left",
                              fontSize: 14,
                              color: "#334155",
                              whiteSpace: "nowrap",
                              width: "1%",
                              borderBottom: "1px solid #f1f5f9"
                            }}
                          >
                            <span style={{ fontWeight: str(row.pick_side).toUpperCase() === "AWAY" ? 700 : 400 }}>
                              {displayTeam(str(row.away_team))}
                            </span>
                            {" @ "}
                            <span style={{ fontWeight: str(row.pick_side).toUpperCase() === "HOME" ? 700 : 400 }}>
                              {displayTeam(str(row.home_team))}
                            </span>
                            {formatGameTime(row) && (
                              <span style={{ ...mono, marginLeft: 6, fontSize: 11, color: "#94a3b8" }}>
                                {formatGameTime(row)}
                              </span>
                            )}
                          </td>

                          {/* HOME SPREAD */}
                          <td
                            style={{
                              ...mono,
                              padding: "10px 14px",
                              textAlign: "center",
                              fontSize: 13,
                              color: "#334155",
                              borderBottom: "1px solid #f1f5f9"
                            }}
                          >
                            {hb && bk !== null ? formatSpread(bk) : "\u2014"}
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
                              borderBottom: "1px solid #f1f5f9"
                            }}
                          >
                            {md !== null ? formatSpread(md) : "\u2014"}
                          </td>

                          {/* SIGMA */}
                          <td
                            style={{
                              ...mono,
                              padding: "10px 14px",
                              textAlign: "center",
                              fontSize: 13,
                              color: "#64748b",
                              borderBottom: "1px solid #f1f5f9"
                            }}
                          >
                            {sg !== null ? sg.toFixed(1) : "\u2014"}
                          </td>

                          {/* DIFF */}
                          <td
                            style={{
                              ...mono,
                              padding: "10px 14px",
                              textAlign: "center",
                              fontSize: 13,
                              fontWeight: 600,
                              color: "#334155",
                              borderBottom: "1px solid #f1f5f9"
                            }}
                          >
                            {hb && df !== null ? df.toFixed(1) : "\u2014"}
                          </td>

                          {/* EDGE */}
                          <td
                            style={{
                              ...mono,
                              padding: "10px 14px",
                              textAlign: "center",
                              fontSize: 13,
                              fontWeight: 700,
                              color: hb
                                ? eg >= 0
                                  ? "#16a34a"
                                  : "#dc2626"
                                : "#94a3b8",
                              borderBottom: "1px solid #f1f5f9"
                            }}
                          >
                            {hb
                              ? `${eg >= 0 ? "+" : ""}${(eg * 100).toFixed(1)}%`
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
      </div>
    </Layout>
  );
}
