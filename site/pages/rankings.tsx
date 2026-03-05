import { GetServerSideProps } from "next";
import { useRouter } from "next/router";
import { CSSProperties, useMemo, useState } from "react";
import Layout from "../components/Layout";
import { displayTeam } from "../lib/data";
import { listRankingsSeasons, readJsonFile } from "../lib/server-data";

/* -- types -- */

type RankedTeam = {
  rank: number;
  team: string;
  team_id: number;
  conference: string;
  record: string;
  conf_record: string;
  adj_oe: number;
  adj_de: number;
  adj_margin: number;
  adj_tempo: number;
  edge_index: number | null;
};

type RankingsData = {
  generated_at: string;
  as_of_date: string;
  season: number;
  teams: RankedTeam[];
};

type Props = {
  data: RankingsData | null;
  availableSeasons: number[];
  currentSeason: number;
};

/* -- server-side -- */

export const getServerSideProps: GetServerSideProps<Props> = async (context) => {
  const availableSeasons = listRankingsSeasons();
  const latestSeason = availableSeasons.length > 0 ? availableSeasons[0] : 2026;

  const qSeason = context.query.season;
  const currentSeason =
    typeof qSeason === "string" && /^\d{4}$/.test(qSeason)
      ? Number(qSeason)
      : latestSeason;

  const filename = `rankings_${currentSeason}.json`;
  let raw = readJsonFile(filename);
  // Fall back to generic rankings.json if season-specific file doesn't exist
  if (!raw) raw = readJsonFile("rankings.json");
  const data = raw as RankingsData | null;

  return { props: { data, availableSeasons, currentSeason } };
};

/* -- helpers -- */

const mono: CSSProperties = {
  fontFamily: "'IBM Plex Mono', monospace",
};

type SortKey =
  | "rank"
  | "team"
  | "conference"
  | "record"
  | "adj_oe"
  | "adj_de"
  | "adj_margin"
  | "adj_tempo"
  | "edge_index";

type SortState = { key: SortKey; dir: "asc" | "desc" };

function sortVal(t: RankedTeam, key: SortKey): string | number {
  switch (key) {
    case "rank":
      return t.rank;
    case "team":
      return t.team;
    case "conference":
      return t.conference;
    case "record": {
      const [w] = t.record.split("-").map(Number);
      return w ?? 0;
    }
    case "adj_oe":
      return t.adj_oe;
    case "adj_de":
      return t.adj_de;
    case "adj_margin":
      return t.adj_margin;
    case "adj_tempo":
      return t.adj_tempo;
    case "edge_index":
      return t.edge_index ?? 0;
  }
}

function formatDateDisplay(dateStr: string): string {
  const [year, month, day] = dateStr.split("-");
  const months = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
  ];
  return `${months[Number(month) - 1]} ${Number(day)}, ${year}`;
}

function netColor(val: number): string {
  if (val >= 20) return "rgba(22, 163, 74, 0.18)";
  if (val >= 10) return "rgba(22, 163, 74, 0.10)";
  if (val > 0) return "rgba(22, 163, 74, 0.05)";
  if (val <= -20) return "rgba(220, 38, 38, 0.18)";
  if (val <= -10) return "rgba(220, 38, 38, 0.10)";
  if (val < 0) return "rgba(220, 38, 38, 0.05)";
  return "transparent";
}

function netTextColor(val: number): string {
  if (val > 0) return "#0f7b50";
  if (val < 0) return "#b42318";
  return "#334155";
}

/* -- column defs -- */

const columns: {
  key: SortKey;
  label: string;
  align: "left" | "center";
  defaultDir: "asc" | "desc";
}[] = [
  { key: "rank", label: "#", align: "center", defaultDir: "asc" },
  { key: "team", label: "TEAM", align: "left", defaultDir: "asc" },
  { key: "conference", label: "CONF", align: "center", defaultDir: "asc" },
  { key: "record", label: "RECORD", align: "center", defaultDir: "desc" },
  { key: "adj_oe", label: "ADJ O", align: "center", defaultDir: "desc" },
  { key: "adj_de", label: "ADJ D", align: "center", defaultDir: "asc" },
  { key: "adj_margin", label: "NET RATING", align: "center", defaultDir: "desc" },
  { key: "adj_tempo", label: "TEMPO", align: "center", defaultDir: "desc" },
  { key: "edge_index", label: "EDGE INDEX", align: "center", defaultDir: "desc" },
];

/* -- component -- */

function seasonLabel(season: number): string {
  return `${season - 1}\u2013${String(season).slice(2)} Season`;
}

export default function Rankings({ data, availableSeasons, currentSeason }: Props) {
  const router = useRouter();
  const [search, setSearch] = useState("");
  const [confFilter, setConfFilter] = useState("all");
  const [sort, setSort] = useState<SortState>({ key: "rank", dir: "asc" });

  const conferences = useMemo(() => {
    if (!data) return [];
    const set = new Set<string>();
    for (const t of data.teams) {
      if (t.conference) set.add(t.conference);
    }
    return Array.from(set).sort();
  }, [data]);

  const tableRows = useMemo(() => {
    if (!data) return [];
    let list = [...data.teams];

    if (search.trim()) {
      const q = search.trim().toLowerCase();
      list = list.filter((t) => t.team.toLowerCase().includes(q));
    }

    if (confFilter !== "all") {
      list = list.filter((t) => t.conference === confFilter);
    }

    list.sort((a, b) => {
      const av = sortVal(a, sort.key);
      const bv = sortVal(b, sort.key);
      if (typeof av === "number" && typeof bv === "number") {
        return sort.dir === "asc" ? av - bv : bv - av;
      }
      const cmp = String(av).localeCompare(String(bv));
      return sort.dir === "asc" ? cmp : -cmp;
    });

    return list;
  }, [data, search, confFilter, sort]);

  function handleSort(key: SortKey) {
    setSort((prev) => {
      if (prev.key === key) {
        return { key, dir: prev.dir === "desc" ? "asc" : "desc" };
      }
      const col = columns.find((c) => c.key === key);
      return { key, dir: col?.defaultDir ?? "desc" };
    });
  }

  if (!data || !data.teams.length) {
    return (
      <Layout>
        <div style={{ padding: 24, color: "#94a3b8", textAlign: "center" }}>
          No rankings data available.
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div>
        {/* -- Title Row -- */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "baseline",
            marginBottom: 24,
          }}
        >
          <h1
            style={{
              fontSize: 24,
              fontWeight: 700,
              letterSpacing: "-0.02em",
              margin: 0,
              color: "#0f172a",
            }}
          >
            Power Rankings
          </h1>
          <span style={{ ...mono, fontSize: 13, color: "#64748b" }}>
            {seasonLabel(currentSeason)} · {data.as_of_date ? formatDateDisplay(data.as_of_date) : ""} ·{" "}
            {data.teams.length} teams
          </span>
        </div>

        {/* -- Controls -- */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: 10,
            gap: 10,
            flexWrap: "wrap",
          }}
        >
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            {availableSeasons.length > 1 && (
              <select
                value={currentSeason}
                onChange={(e) => router.push(`/rankings?season=${e.target.value}`)}
                style={{
                  ...mono,
                  padding: "6px 10px",
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
              </select>
            )}
            <select
              value={confFilter}
              onChange={(e) => setConfFilter(e.target.value)}
              style={{
                ...mono,
                padding: "6px 10px",
                border: "1px solid #e2e8f0",
                borderRadius: 6,
                fontSize: 13,
                background: "#fff",
                color: "#334155",
              }}
            >
              <option value="all">All Conferences</option>
              {conferences.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          </div>

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
              color: "#334155",
            }}
          />
        </div>

        {/* -- Table -- */}
        <div
          style={{
            background: "#fff",
            border: "1px solid #e2e8f0",
            borderRadius: 10,
            overflow: "hidden",
            boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
          }}
        >
          <div style={{ overflowX: "auto" }}>
            <table
              style={{
                width: "100%",
                borderCollapse: "collapse",
                fontVariantNumeric: "tabular-nums",
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
                          ...(col.key === "team" ? { width: "1%" } : {}),
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
                        borderBottom: "none",
                      }}
                    >
                      No teams found
                    </td>
                  </tr>
                ) : (
                  tableRows.map((t, i) => (
                    <tr
                      key={t.team_id}
                      style={{
                        borderBottom: "1px solid #f1f5f9",
                        animation: `fadeIn 0.3s ease ${i * 0.02}s both`,
                      }}
                    >
                      {/* RANK */}
                      <td
                        style={{
                          ...mono,
                          padding: "10px 14px",
                          textAlign: "center",
                          fontSize: 13,
                          fontWeight: 700,
                          color: "#0f172a",
                          borderBottom: "1px solid #f1f5f9",
                        }}
                      >
                        {t.rank}
                      </td>

                      {/* TEAM */}
                      <td
                        style={{
                          padding: "10px 14px",
                          textAlign: "left",
                          fontSize: 14,
                          fontWeight: 600,
                          color: "#334155",
                          whiteSpace: "nowrap",
                          width: "1%",
                          borderBottom: "1px solid #f1f5f9",
                        }}
                      >
                        {displayTeam(t.team)}
                      </td>

                      {/* CONFERENCE */}
                      <td
                        style={{
                          ...mono,
                          padding: "10px 14px",
                          textAlign: "center",
                          fontSize: 12,
                          color: "#64748b",
                          borderBottom: "1px solid #f1f5f9",
                        }}
                      >
                        {t.conference || "\u2014"}
                      </td>

                      {/* RECORD */}
                      <td
                        style={{
                          ...mono,
                          padding: "10px 14px",
                          textAlign: "center",
                          fontSize: 13,
                          color: "#334155",
                          borderBottom: "1px solid #f1f5f9",
                        }}
                      >
                        <span>{t.record}</span>
                        {t.conf_record && t.conf_record !== "0-0" && (
                          <span
                            style={{
                              marginLeft: 6,
                              fontSize: 11,
                              color: "#94a3b8",
                            }}
                          >
                            ({t.conf_record})
                          </span>
                        )}
                      </td>

                      {/* ADJ O */}
                      <td
                        style={{
                          ...mono,
                          padding: "10px 14px",
                          textAlign: "center",
                          fontSize: 13,
                          color: "#334155",
                          borderBottom: "1px solid #f1f5f9",
                        }}
                      >
                        {t.adj_oe.toFixed(1)}
                      </td>

                      {/* ADJ D */}
                      <td
                        style={{
                          ...mono,
                          padding: "10px 14px",
                          textAlign: "center",
                          fontSize: 13,
                          color: "#334155",
                          borderBottom: "1px solid #f1f5f9",
                        }}
                      >
                        {t.adj_de.toFixed(1)}
                      </td>

                      {/* NET RATING */}
                      <td
                        style={{
                          ...mono,
                          padding: "10px 14px",
                          textAlign: "center",
                          fontSize: 13,
                          fontWeight: 700,
                          color: netTextColor(t.adj_margin),
                          background: netColor(t.adj_margin),
                          borderBottom: "1px solid #f1f5f9",
                        }}
                      >
                        {t.adj_margin > 0 ? "+" : ""}
                        {t.adj_margin.toFixed(1)}
                      </td>

                      {/* TEMPO */}
                      <td
                        style={{
                          ...mono,
                          padding: "10px 14px",
                          textAlign: "center",
                          fontSize: 13,
                          color: "#64748b",
                          borderBottom: "1px solid #f1f5f9",
                        }}
                      >
                        {t.adj_tempo.toFixed(1)}
                      </td>

                      {/* EDGE INDEX */}
                      <td
                        style={{
                          ...mono,
                          padding: "10px 14px",
                          textAlign: "center",
                          fontSize: 13,
                          fontWeight: 600,
                          color: "#0f172a",
                          borderBottom: "1px solid #f1f5f9",
                        }}
                      >
                        {t.edge_index !== null
                          ? `${(t.edge_index * 100).toFixed(1)}%`
                          : "\u2014"}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </Layout>
  );
}
