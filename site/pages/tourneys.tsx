import { GetServerSideProps } from "next";
import { CSSProperties, useMemo, useState } from "react";
import Layout from "../components/Layout";
import {
  buildNcaaOddsData,
  formatRoundOdds,
  NcaaOddsData,
  NcaaOddsRoundKey,
  NcaaOddsRow,
} from "../lib/bracket/ncaaOdds";
import {
  MatchupPredictionCache,
  NcaaBracketField,
  NcaaTournamentResults,
} from "../lib/bracket/types";
import { readJsonFile } from "../lib/server-data";

/* -- types -- */

type Team = {
  seed: number;
  team: string;
  conf_record: string;
  model_pct: number | null;
  model_odds: string | null;
  hrb_odds?: string | null;
  vegas_implied_pct?: number | null;
  edge?: number | null;
  flag?: string | null;
};

type Conference = {
  name: string;
  qualifier_text: string;
  has_hrb_odds: boolean;
  teams: Team[];
  dnq: string[];
};

type SummaryRow = {
  conf: string;
  team: string;
  model_pct: number;
  hrb_odds: string;
  edge: number;
  flag: string;
};

type TourneysData = {
  generated_at: string;
  season: number;
  methodology: { simulations: number; odds_source: string; note: string };
  conferences: Conference[];
  value_bets: SummaryRow[];
  fades: SummaryRow[];
};

type Props = {
  data: TourneysData | null;
  ncaaData: NcaaOddsData | null;
  ncaaInternalData: NcaaOddsData | null;
};

/* -- server-side -- */

export const getServerSideProps: GetServerSideProps<Props> = async () => {
  const raw = readJsonFile("tourneys_2026.json");
  const data = raw as TourneysData | null;
  const rawNcaaField = readJsonFile("ncaa_bracket_builder_2026.json");
  const rawNcaaCache = readJsonFile("ncaa_matchup_predictions_2026.json");
  const rawNcaaResults = readJsonFile("ncaa_results_2026.json");

  let ncaaData: NcaaOddsData | null = null;
  let ncaaInternalData: NcaaOddsData | null = null;
  if (rawNcaaField && rawNcaaCache) {
    try {
      ncaaData = buildNcaaOddsData(
        rawNcaaField as NcaaBracketField,
        rawNcaaCache as MatchupPredictionCache,
        "active",
        rawNcaaResults as NcaaTournamentResults | null,
      );
      ncaaInternalData = buildNcaaOddsData(
        rawNcaaField as NcaaBracketField,
        rawNcaaCache as MatchupPredictionCache,
        "team_ab_internal",
        rawNcaaResults as NcaaTournamentResults | null,
      );
    } catch (error) {
      console.error("Failed to build NCAA odds data", error);
    }
  }

  return { props: { data, ncaaData, ncaaInternalData } };
};

/* -- helpers -- */

const mono: CSSProperties = {
  fontFamily: "'IBM Plex Mono', monospace",
};

const POWER_CONFERENCES = ["ACC", "Big Ten", "Big 12", "SEC", "Big East"];
const NCAA_ROUND_COLUMNS: Array<{ key: NcaaOddsRoundKey; label: string }> = [
  { key: "round-of-64", label: "R64" },
  { key: "round-of-32", label: "R32" },
  { key: "sweet-16", label: "S16" },
  { key: "elite-8", label: "E8" },
  { key: "final-four", label: "F4" },
  { key: "national-championship", label: "Title" },
  { key: "champion", label: "Champ" },
];

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function oddsDeltaColor(deltaPctPoints: number): string {
  if (deltaPctPoints >= 1) return "#15803d";
  if (deltaPctPoints > 0) return "#16a34a";
  if (deltaPctPoints <= -1) return "#b91c1c";
  if (deltaPctPoints < 0) return "#dc2626";
  return "#64748b";
}

function flagStyle(flag: string | null | undefined): CSSProperties {
  if (!flag) return {};
  if (flag === "STRONG VALUE")
    return {
      ...mono,
      fontSize: 10,
      fontWeight: 600,
      padding: "2px 6px",
      borderRadius: 4,
      background: "rgba(22, 163, 74, 0.12)",
      color: "#15803d",
      letterSpacing: "0.04em",
    };
  if (flag === "VALUE")
    return {
      ...mono,
      fontSize: 10,
      fontWeight: 600,
      padding: "2px 6px",
      borderRadius: 4,
      background: "rgba(22, 163, 74, 0.08)",
      color: "#16a34a",
      letterSpacing: "0.04em",
    };
  if (flag === "FADE")
    return {
      ...mono,
      fontSize: 10,
      fontWeight: 600,
      padding: "2px 6px",
      borderRadius: 4,
      background: "rgba(220, 38, 38, 0.10)",
      color: "#dc2626",
      letterSpacing: "0.04em",
    };
  return {};
}

function edgeColor(edge: number | null | undefined): string {
  if (edge == null) return "#94a3b8";
  if (edge >= 5) return "#15803d";
  if (edge >= 3) return "#16a34a";
  if (edge <= -5) return "#dc2626";
  return "#334155";
}

/* -- component -- */

export default function Tourneys({ data, ncaaData, ncaaInternalData }: Props) {
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<"all" | "power" | "mid" | "hrb">("all");
  const [scope, setScope] = useState<"ncaa" | "conference">(
    ncaaData ? "ncaa" : "conference",
  );
  const [tab, setTab] = useState<"conferences" | "value" | "fades">(
    "conferences",
  );

  const conferences = useMemo(() => {
    if (!data) return [];
    let list = data.conferences;

    if (filter === "power") {
      list = list.filter((c) => POWER_CONFERENCES.includes(c.name));
    } else if (filter === "mid") {
      list = list.filter((c) => !POWER_CONFERENCES.includes(c.name));
    } else if (filter === "hrb") {
      list = list.filter((c) => c.has_hrb_odds);
    }

    if (search.trim()) {
      const q = search.trim().toLowerCase();
      list = list.filter(
        (c) =>
          c.name.toLowerCase().includes(q) ||
          c.teams.some((t) => t.team.toLowerCase().includes(q)),
      );
    }

    return list;
  }, [data, search, filter]);

  if ((!data || !data.conferences.length) && !ncaaData) {
    return (
      <Layout>
        <div style={{ padding: 24, color: "#94a3b8", textAlign: "center" }}>
          No tournament analysis data available.
        </div>
      </Layout>
    );
  }

  const hrbCount = data?.conferences.filter((c) => c.has_hrb_odds).length ?? 0;
  const methodologyLabel =
    data?.methodology.simulations && data.methodology.simulations > 0
      ? `${Math.round(data.methodology.simulations / 1000)}K sims`
      : "exact bracket";

  return (
    <Layout wide={scope === "ncaa"}>
      <div>
        {/* -- Title Row -- */}
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "baseline",
            marginBottom: 16,
            flexWrap: "wrap",
            gap: 12,
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
            Tournament Odds
          </h1>
          <span style={{ ...mono, fontSize: 13, color: "#64748b" }}>
            {scope === "ncaa" && ncaaData
              ? `68 teams · exact bracket · display-odds ML calc`
              : `${data?.conferences.length ?? 0} conferences · ${hrbCount} with HRB odds · ${methodologyLabel}`}
          </span>
        </div>

        <div
          style={{
            display: "flex",
            gap: 0,
            marginBottom: 16,
            borderBottom: "1px solid #e2e8f0",
          }}
        >
          {ncaaData && (
            <button
              onClick={() => setScope("ncaa")}
              style={{
                ...mono,
                padding: "10px 20px",
                border: "none",
                borderBottom: `2px solid ${scope === "ncaa" ? "#0f172a" : "transparent"}`,
                fontSize: 13,
                fontWeight: scope === "ncaa" ? 600 : 400,
                background: "transparent",
                color: scope === "ncaa" ? "#0f172a" : "#64748b",
                cursor: "pointer",
                transition: "all 0.15s",
              }}
            >
              NCAA Tournament
            </button>
          )}
          {data && (
            <button
              onClick={() => setScope("conference")}
              style={{
                ...mono,
                padding: "10px 20px",
                border: "none",
                borderBottom: `2px solid ${scope === "conference" ? "#0f172a" : "transparent"}`,
                fontSize: 13,
                fontWeight: scope === "conference" ? 600 : 400,
                background: "transparent",
                color: scope === "conference" ? "#0f172a" : "#64748b",
                cursor: "pointer",
                transition: "all 0.15s",
              }}
            >
              Conference Tournaments
            </button>
          )}
        </div>

        {scope === "ncaa" && ncaaData ? (
          <NcaaOddsSection data={ncaaData} comparisonData={ncaaInternalData} />
        ) : (
          <>
            {/* -- Methodology note -- */}
            <div
              style={{
                ...mono,
                fontSize: 11,
                color: "#94a3b8",
                marginBottom: 20,
                lineHeight: 1.6,
              }}
            >
              Edge = Model% - Vegas Implied%. VALUE: edge &gt; +3% | STRONG
              VALUE: edge &gt; +5% | FADE: edge &lt; -5%
            </div>

            {/* -- Tabs -- */}
            <div
              style={{
                display: "flex",
                gap: 0,
                marginBottom: 16,
                borderBottom: "1px solid #e2e8f0",
              }}
            >
              {(
                [
                  ["conferences", "All Conferences"],
                  ["value", `Value Bets (${data?.value_bets.length ?? 0})`],
                  ["fades", `Fades (${data?.fades.length ?? 0})`],
                ] as const
              ).map(([key, label]) => (
                <button
                  key={key}
                  onClick={() => setTab(key)}
                  style={{
                    ...mono,
                    padding: "10px 20px",
                    border: "none",
                    borderBottom: `2px solid ${tab === key ? "#0f172a" : "transparent"}`,
                    fontSize: 13,
                    fontWeight: tab === key ? 600 : 400,
                    background: "transparent",
                    color: tab === key ? "#0f172a" : "#64748b",
                    cursor: "pointer",
                    transition: "all 0.15s",
                  }}
                >
                  {label}
                </button>
              ))}
            </div>

            {/* -- Conferences Tab -- */}
            {tab === "conferences" && data && (
              <>
                {/* -- Controls -- */}
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    marginBottom: 16,
                    gap: 10,
                    flexWrap: "wrap",
                  }}
                >
                  <div style={{ display: "flex", gap: 6 }}>
                    {(["all", "power", "mid", "hrb"] as const).map((f) => (
                      <button
                        key={f}
                        onClick={() => setFilter(f)}
                        style={{
                          ...mono,
                          padding: "6px 14px",
                          border: `1px solid ${filter === f ? "#0f172a" : "#e2e8f0"}`,
                          borderRadius: 6,
                          fontSize: 12,
                          fontWeight: filter === f ? 600 : 400,
                          background: filter === f ? "#0f172a" : "#fff",
                          color: filter === f ? "#fff" : "#64748b",
                          cursor: "pointer",
                          transition: "all 0.15s",
                        }}
                      >
                        {f === "all"
                          ? "All"
                          : f === "power"
                            ? "Power 5"
                            : f === "mid"
                              ? "Mid-Major"
                              : "HRB Odds"}
                      </button>
                    ))}
                  </div>
                  <input
                    type="text"
                    placeholder="Search conference or team..."
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    style={{
                      ...mono,
                      width: 260,
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

                {/* -- Conference Cards -- */}
                <div
                  style={{ display: "flex", flexDirection: "column", gap: 20 }}
                >
                  {conferences.map((conf) => (
                    <ConferenceCard key={conf.name} conf={conf} />
                  ))}
                </div>

                {conferences.length === 0 && (
                  <div
                    style={{
                      padding: 40,
                      textAlign: "center",
                      color: "#94a3b8",
                      fontSize: 14,
                    }}
                  >
                    No conferences match your search.
                  </div>
                )}
              </>
            )}

            {/* -- Value Bets Tab -- */}
            {tab === "value" && data && (
              <SummaryTable rows={data.value_bets} title="Top Value Bets" />
            )}

            {/* -- Fades Tab -- */}
            {tab === "fades" && data && (
              <SummaryTable rows={data.fades} title="Top Fades" />
            )}
          </>
        )}
      </div>
    </Layout>
  );
}

/* -- NCAA Odds -- */

function NcaaOddsSection({
  data,
  comparisonData,
}: {
  data: NcaaOddsData;
  comparisonData: NcaaOddsData | null;
}) {
  const [search, setSearch] = useState("");
  const [region, setRegion] = useState<string>("all");
  const [sortKey, setSortKey] = useState<NcaaOddsRoundKey>("champion");
  const comparisonRowsByTeamId = useMemo(
    () => new Map((comparisonData?.rows ?? []).map((row) => [row.teamId, row])),
    [comparisonData],
  );

  const regions = useMemo(
    () =>
      Array.from(
        new Set(data.rows.map((row) => row.region).filter(Boolean)),
      ).sort() as string[],
    [data.rows],
  );

  const rows = useMemo(() => {
    const query = search.trim().toLowerCase();
    return [...data.rows]
      .filter((row) => region === "all" || row.region === region)
      .filter((row) => {
        if (!query) return true;
        return (
          row.team.toLowerCase().includes(query) ||
          row.conference.toLowerCase().includes(query) ||
          (row.region ?? "").toLowerCase().includes(query)
        );
      })
      .sort((a, b) => {
        const diff =
          b.roundProbabilities[sortKey] - a.roundProbabilities[sortKey];
        if (diff !== 0) return diff;
        return a.seed - b.seed || a.team.localeCompare(b.team);
      });
  }, [data.rows, region, search, sortKey]);

  return (
    <>
      <div
        style={{
          ...mono,
          fontSize: 11,
          color: "#94a3b8",
          marginBottom: 20,
          lineHeight: 1.6,
        }}
      >
        Exact bracket advancement odds using the NCAA display-adjusted spread
        path for moneyline conversion. Active columns use Team A/B with Torvik.
        Internal columns use the same Team A/B model on Hoops Edge efficiencies.
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
          gap: 12,
          marginBottom: 18,
        }}
      >
        <NcaaSummaryCard
          label="Title Favorite"
          primary={
            data.summary.titleFavorite
              ? `${data.summary.titleFavorite.team} (${data.summary.titleFavorite.seed})`
              : "--"
          }
          secondary={
            data.summary.titleFavorite
              ? `${(data.summary.titleFavorite.roundProbabilities.champion * 100).toFixed(1)}% · ${formatRoundOdds(data.summary.titleFavorite.roundProbabilities.champion) ?? "--"}`
              : "No data"
          }
        />
        <NcaaSummaryCard
          label="Final Four Locks"
          primary={`${data.summary.finalFourLocks.length}`}
          secondary="Teams at 50%+ to make the Final Four"
        />
        <NcaaSummaryCard
          label="Method"
          primary="Display ML"
          secondary={data.methodology.note}
        />
      </div>

      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: 10,
          marginBottom: 16,
          flexWrap: "wrap",
        }}
      >
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {["all", ...regions].map((value) => (
            <button
              key={value}
              onClick={() => setRegion(value)}
              style={{
                ...mono,
                padding: "6px 14px",
                border: `1px solid ${region === value ? "#0f172a" : "#e2e8f0"}`,
                borderRadius: 6,
                fontSize: 12,
                fontWeight: region === value ? 600 : 400,
                background: region === value ? "#0f172a" : "#fff",
                color: region === value ? "#fff" : "#64748b",
                cursor: "pointer",
              }}
            >
              {value === "all" ? "All Regions" : value}
            </button>
          ))}
        </div>
        <input
          type="text"
          placeholder="Search team, region, conference..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{
            ...mono,
            width: 280,
            maxWidth: "100%",
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
              fontSize: 13,
              fontVariantNumeric: "tabular-nums",
            }}
          >
            <thead>
              <tr
                style={{
                  background: "#fafbfc",
                  borderBottom: "1px solid #e2e8f0",
                }}
              >
                <th style={thStyle}>Seed</th>
                <th style={{ ...thStyle, textAlign: "left" }}>Team</th>
                <th style={thStyle}>Region</th>
                <th style={thStyle}>
                  Active
                  <div style={{ fontSize: 10, fontWeight: 500, textTransform: "none", letterSpacing: 0 }}>
                    Team A/B Torvik
                  </div>
                </th>
                <th style={thStyle}>
                  Internal
                  <div style={{ fontSize: 10, fontWeight: 500, textTransform: "none", letterSpacing: 0 }}>
                    Same model, HE effs
                  </div>
                </th>
                <th style={thStyle}>
                  Δ
                  <div style={{ fontSize: 10, fontWeight: 500, textTransform: "none", letterSpacing: 0 }}>
                    Act - Int
                  </div>
                </th>
                {NCAA_ROUND_COLUMNS.map((column) => (
                  <th key={column.key} style={thStyle}>
                    <button
                      onClick={() => setSortKey(column.key)}
                      style={{
                        ...mono,
                        fontSize: 11,
                        fontWeight: sortKey === column.key ? 700 : 600,
                        color: sortKey === column.key ? "#0f172a" : "#64748b",
                        background: "transparent",
                        border: "none",
                        cursor: "pointer",
                        textTransform: "uppercase",
                        letterSpacing: "0.05em",
                      }}
                    >
                      {column.label}
                    </button>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <NcaaOddsRowView
                  key={row.teamId}
                  row={row}
                  comparisonRow={comparisonRowsByTeamId.get(row.teamId) ?? null}
                  sortKey={sortKey}
                />
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {rows.length === 0 && (
        <div
          style={{
            padding: 40,
            textAlign: "center",
            color: "#94a3b8",
            fontSize: 14,
          }}
        >
          No NCAA teams match your search.
        </div>
      )}
    </>
  );
}

function NcaaSummaryCard({
  label,
  primary,
  secondary,
}: {
  label: string;
  primary: string;
  secondary: string;
}) {
  return (
    <div
      style={{
        background: "#fff",
        border: "1px solid #e2e8f0",
        borderRadius: 10,
        padding: "14px 16px",
        boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
      }}
    >
      <div style={{ ...mono, fontSize: 11, color: "#94a3b8", marginBottom: 8 }}>
        {label}
      </div>
      <div
        style={{
          fontSize: 18,
          fontWeight: 700,
          color: "#0f172a",
          marginBottom: 6,
        }}
      >
        {primary}
      </div>
      <div style={{ ...mono, fontSize: 11, color: "#64748b", lineHeight: 1.5 }}>
        {secondary}
      </div>
    </div>
  );
}

function NcaaOddsRowView({
  row,
  comparisonRow,
  sortKey,
}: {
  row: NcaaOddsRow;
  comparisonRow: NcaaOddsRow | null;
  sortKey: NcaaOddsRoundKey;
}) {
  const activeProbability = row.roundProbabilities[sortKey];
  const internalProbability = comparisonRow?.roundProbabilities[sortKey] ?? null;
  const deltaPctPoints =
    internalProbability == null ? null : (activeProbability - internalProbability) * 100;
  return (
    <tr style={{ borderBottom: "1px solid #f1f5f9" }}>
      <td style={{ ...tdStyle, ...mono, fontWeight: 600, color: "#64748b" }}>
        {row.seed}
      </td>
      <td style={{ ...tdStyle, textAlign: "left" }}>
        <div style={{ fontWeight: 600, color: "#0f172a" }}>{row.team}</div>
        <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>
          {row.conference}
        </div>
      </td>
      <td style={{ ...tdStyle, ...mono, color: "#64748b" }}>
        {row.region ?? "--"}
      </td>
      <td style={tdStyle}>
        <div style={{ ...mono, fontWeight: 700, color: "#0f172a" }}>
          {formatPercent(activeProbability)}
        </div>
        <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>
          {formatRoundOdds(activeProbability) ?? "--"}
        </div>
      </td>
      <td style={tdStyle}>
        <div style={{ ...mono, fontWeight: 700, color: "#0f172a" }}>
          {internalProbability == null ? "--" : formatPercent(internalProbability)}
        </div>
        <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>
          {internalProbability == null ? "--" : formatRoundOdds(internalProbability) ?? "--"}
        </div>
      </td>
      <td style={tdStyle}>
        <div
          style={{
            ...mono,
            fontWeight: 700,
            color: deltaPctPoints == null ? "#94a3b8" : oddsDeltaColor(deltaPctPoints),
          }}
        >
          {deltaPctPoints == null ? "--" : `${deltaPctPoints > 0 ? "+" : ""}${deltaPctPoints.toFixed(1)} pp`}
        </div>
        <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>
          {comparisonRow ? "Act - Int" : "--"}
        </div>
      </td>
      {NCAA_ROUND_COLUMNS.map((column) => {
        const probability = row.roundProbabilities[column.key];
        const internalRoundProbability = comparisonRow?.roundProbabilities[column.key] ?? null;
        return (
          <td key={column.key} style={tdStyle}>
            <div style={{ ...mono, fontWeight: 700, color: "#0f172a" }}>
              T {formatPercent(probability)}
            </div>
            <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>
              {formatRoundOdds(probability) ?? "--"}
            </div>
            <div style={{ ...mono, fontWeight: 700, color: "#334155", marginTop: 6 }}>
              HE {internalRoundProbability == null ? "--" : formatPercent(internalRoundProbability)}
            </div>
            <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>
              {internalRoundProbability == null ? "--" : formatRoundOdds(internalRoundProbability) ?? "--"}
            </div>
          </td>
        );
      })}
    </tr>
  );
}

/* -- Conference Card -- */

function ConferenceCard({ conf }: { conf: Conference }) {
  const favorite = conf.teams.reduce(
    (best, t) => ((t.model_pct ?? 0) > (best.model_pct ?? 0) ? t : best),
    conf.teams[0],
  );
  const hasHrb = conf.has_hrb_odds;

  return (
    <div
      style={{
        background: "#fff",
        border: "1px solid #e2e8f0",
        borderRadius: 10,
        overflow: "hidden",
        boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
      }}
    >
      {/* Card header */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          padding: "14px 18px",
          borderBottom: "1px solid #f1f5f9",
          flexWrap: "wrap",
          gap: 8,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 16, fontWeight: 700, color: "#0f172a" }}>
            {conf.name}
          </span>
          <span style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>
            {conf.qualifier_text || `${conf.teams.length} teams`}
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          {hasHrb && (
            <span
              style={{
                ...mono,
                fontSize: 10,
                fontWeight: 600,
                padding: "2px 8px",
                borderRadius: 4,
                background: "rgba(15, 23, 42, 0.06)",
                color: "#475569",
                letterSpacing: "0.05em",
              }}
            >
              HRB
            </span>
          )}
          {favorite && (
            <span
              style={{
                ...mono,
                fontSize: 13,
                fontWeight: 600,
                color: "#0f172a",
              }}
            >
              Fav: ({favorite.seed}) {favorite.team}{" "}
              {favorite.model_pct?.toFixed(1)}%
            </span>
          )}
        </div>
      </div>

      {/* Table */}
      <div style={{ overflowX: "auto", padding: "0 0 4px" }}>
        <table
          style={{
            width: "100%",
            borderCollapse: "collapse",
            fontSize: 13,
            fontVariantNumeric: "tabular-nums",
          }}
        >
          <thead>
            <tr
              style={{
                background: "#fafbfc",
                borderBottom: "1px solid #e2e8f0",
              }}
            >
              <th style={thStyle}>#</th>
              <th style={{ ...thStyle, textAlign: "left" }}>Team</th>
              <th style={thStyle}>Record</th>
              <th style={thStyle}>Model %</th>
              <th style={thStyle}>Model Odds</th>
              {hasHrb && (
                <>
                  <th style={thStyle}>HRB Odds</th>
                  <th style={thStyle}>Vegas %</th>
                  <th style={thStyle}>Edge</th>
                  <th style={thStyle}>Flag</th>
                </>
              )}
            </tr>
          </thead>
          <tbody>
            {conf.teams.map((t) => (
              <tr key={t.team} style={{ borderBottom: "1px solid #f1f5f9" }}>
                <td
                  style={{
                    ...tdStyle,
                    ...mono,
                    fontWeight: 600,
                    color: "#64748b",
                  }}
                >
                  {t.seed}
                </td>
                <td
                  style={{
                    ...tdStyle,
                    textAlign: "left",
                    fontWeight: 500,
                    color: "#0f172a",
                  }}
                >
                  {t.team}
                </td>
                <td style={{ ...tdStyle, ...mono, color: "#64748b" }}>
                  {t.conf_record}
                </td>
                <td
                  style={{
                    ...tdStyle,
                    ...mono,
                    fontWeight: 600,
                    color:
                      t.model_pct != null && t.model_pct >= 20
                        ? "#0f172a"
                        : "#334155",
                  }}
                >
                  {t.model_pct != null ? `${t.model_pct.toFixed(1)}%` : "--"}
                </td>
                <td style={{ ...tdStyle, ...mono, color: "#64748b" }}>
                  {t.model_odds ?? "--"}
                </td>
                {hasHrb && (
                  <>
                    <td
                      style={{
                        ...tdStyle,
                        ...mono,
                        fontWeight: 600,
                        color: t.hrb_odds
                          ? t.hrb_odds.startsWith("-")
                            ? "#0f172a"
                            : "#334155"
                          : "#94a3b8",
                      }}
                    >
                      {t.hrb_odds ?? "--"}
                    </td>
                    <td style={{ ...tdStyle, ...mono, color: "#64748b" }}>
                      {t.vegas_implied_pct != null
                        ? `${t.vegas_implied_pct.toFixed(1)}%`
                        : "--"}
                    </td>
                    <td
                      style={{
                        ...tdStyle,
                        ...mono,
                        fontWeight: 600,
                        color: edgeColor(t.edge),
                      }}
                    >
                      {t.edge != null
                        ? `${t.edge > 0 ? "+" : ""}${t.edge.toFixed(1)}%`
                        : "--"}
                    </td>
                    <td style={tdStyle}>
                      {t.flag && (
                        <span style={flagStyle(t.flag)}>{t.flag}</span>
                      )}
                    </td>
                  </>
                )}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* DNQ footer */}
      {conf.dnq.length > 0 && (
        <div
          style={{
            padding: "8px 18px 12px",
            borderTop: "1px solid #f1f5f9",
          }}
        >
          <span style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>
            DNQ: {conf.dnq.join(", ")}
          </span>
        </div>
      )}
    </div>
  );
}

/* -- Summary Table -- */

function SummaryTable({ rows, title }: { rows: SummaryRow[]; title: string }) {
  return (
    <div
      style={{
        background: "#fff",
        border: "1px solid #e2e8f0",
        borderRadius: 10,
        overflow: "hidden",
        boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
      }}
    >
      <div
        style={{
          padding: "14px 18px",
          borderBottom: "1px solid #f1f5f9",
        }}
      >
        <span style={{ fontSize: 16, fontWeight: 700, color: "#0f172a" }}>
          {title}
        </span>
        <span
          style={{ ...mono, fontSize: 11, color: "#94a3b8", marginLeft: 12 }}
        >
          {rows.length} picks
        </span>
      </div>
      <div style={{ overflowX: "auto" }}>
        <table
          style={{
            width: "100%",
            borderCollapse: "collapse",
            fontSize: 13,
            fontVariantNumeric: "tabular-nums",
          }}
        >
          <thead>
            <tr
              style={{
                background: "#fafbfc",
                borderBottom: "1px solid #e2e8f0",
              }}
            >
              <th style={{ ...thStyle, textAlign: "left" }}>Conference</th>
              <th style={{ ...thStyle, textAlign: "left" }}>Team</th>
              <th style={thStyle}>Model %</th>
              <th style={thStyle}>HRB Odds</th>
              <th style={thStyle}>Edge</th>
              <th style={thStyle}>Flag</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} style={{ borderBottom: "1px solid #f1f5f9" }}>
                <td style={{ ...tdStyle, textAlign: "left", color: "#64748b" }}>
                  {r.conf}
                </td>
                <td
                  style={{
                    ...tdStyle,
                    textAlign: "left",
                    fontWeight: 500,
                    color: "#0f172a",
                  }}
                >
                  {r.team}
                </td>
                <td style={{ ...tdStyle, ...mono, fontWeight: 600 }}>
                  {r.model_pct.toFixed(1)}%
                </td>
                <td
                  style={{
                    ...tdStyle,
                    ...mono,
                    fontWeight: 600,
                    color: r.hrb_odds.startsWith("-") ? "#0f172a" : "#334155",
                  }}
                >
                  {r.hrb_odds}
                </td>
                <td
                  style={{
                    ...tdStyle,
                    ...mono,
                    fontWeight: 600,
                    color: edgeColor(r.edge),
                  }}
                >
                  {r.edge > 0 ? "+" : ""}
                  {r.edge.toFixed(1)}%
                </td>
                <td style={tdStyle}>
                  <span style={flagStyle(r.flag)}>{r.flag}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* -- shared cell styles -- */

const thStyle: CSSProperties = {
  ...mono,
  padding: "8px 12px",
  fontSize: 11,
  fontWeight: 600,
  color: "#64748b",
  textAlign: "center",
  textTransform: "uppercase",
  letterSpacing: "0.05em",
  whiteSpace: "nowrap",
};

const tdStyle: CSSProperties = {
  padding: "8px 12px",
  textAlign: "center",
  whiteSpace: "nowrap",
};
