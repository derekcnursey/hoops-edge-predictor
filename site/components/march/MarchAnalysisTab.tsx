import { CSSProperties, useMemo, useState } from "react";
import {
  HardRockComparisonData,
  HardRockMarketKey,
  HardRockMarketReport,
  HardRockMarketRow,
} from "../../lib/bracket/hardRockComparison";
import {
  formatRoundOdds,
  NcaaOddsData,
  NcaaOddsRoundKey,
} from "../../lib/bracket/ncaaOdds";

const mono: CSSProperties = {
  fontFamily: "'IBM Plex Mono', monospace",
};

const ROUND_COLUMNS: Array<{ key: NcaaOddsRoundKey; label: string }> = [
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

function deltaColor(delta: number): string {
  if (delta >= 1) return "#15803d";
  if (delta > 0) return "#16a34a";
  if (delta <= -1) return "#b91c1c";
  if (delta < 0) return "#dc2626";
  return "#475569";
}

function ComparisonSummaryCard({
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
        borderRadius: 12,
        padding: "14px 16px",
        boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
      }}
    >
      <div style={{ ...mono, fontSize: 11, color: "#94a3b8", marginBottom: 8 }}>{label}</div>
      <div style={{ fontSize: 18, fontWeight: 700, color: "#0f172a", marginBottom: 6 }}>
        {primary}
      </div>
      <div style={{ ...mono, fontSize: 11, color: "#64748b", lineHeight: 1.5 }}>{secondary}</div>
    </div>
  );
}

function roundExpectedPoints(round: number): string {
  return round.toFixed(2).replace(/\.00$/, "");
}

function reportLabelForRow(row: HardRockMarketRow): string {
  return row.marketLabel === "Region Winner" ? "Model F4" : `Model ${row.marketLabel}`;
}

function MarketTopList({
  title,
  rows,
}: {
  title: string;
  rows: HardRockMarketRow[];
}) {
  return (
    <div
      style={{
        background: "#fff",
        border: "1px solid #e2e8f0",
        borderRadius: 12,
        overflow: "hidden",
        boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
      }}
    >
      <div
        style={{
          padding: "14px 16px",
          borderBottom: "1px solid #f1f5f9",
          fontWeight: 700,
          color: "#0f172a",
        }}
      >
        {title}
      </div>
      <div style={{ display: "flex", flexDirection: "column" }}>
        {rows.map((row) => (
          <div
            key={`${title}-${row.marketKey}-${row.group}-${row.team}`}
            style={{
              display: "grid",
              gridTemplateColumns: "minmax(0, 1fr) auto",
              gap: 12,
              padding: "12px 16px",
              borderBottom: "1px solid #f8fafc",
            }}
          >
            <div>
              <div style={{ fontWeight: 600, color: "#0f172a" }}>
                {row.marketKey === "champion" ? "" : `${row.group} · `}({row.seed}) {row.team}
              </div>
              <div style={{ ...mono, fontSize: 11, color: "#64748b" }}>
                HRB {row.hrbOdds} fair {formatPercent(row.hrbFairProb)} · {reportLabelForRow(row)}{" "}
                {row.modelOdds ?? "--"}
              </div>
            </div>
            <div
              style={{
                ...mono,
                fontWeight: 700,
                color: deltaColor(row.deltaPctPoints),
                whiteSpace: "nowrap",
              }}
            >
              {row.deltaPctPoints > 0 ? "+" : ""}
              {row.deltaPctPoints.toFixed(2)} pp
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function MarketComparisonTable({
  report,
  rows,
}: {
  report: HardRockMarketReport;
  rows: HardRockMarketRow[];
}) {
  return (
    <div
      style={{
        background: "#fff",
        border: "1px solid #e2e8f0",
        borderRadius: 12,
        overflow: "hidden",
        boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
      }}
    >
      <div
        style={{
          padding: "14px 16px",
          borderBottom: "1px solid #f1f5f9",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: 12,
          flexWrap: "wrap",
        }}
      >
        <div>
          <div style={{ fontSize: 16, fontWeight: 700, color: "#0f172a" }}>
            Hard Rock {report.label} Comparison
          </div>
          <div style={{ ...mono, fontSize: 11, color: "#64748b", marginTop: 4 }}>
            {report.snapshotLabel}
          </div>
          <div style={{ ...mono, fontSize: 11, color: "#94a3b8", marginTop: 4 }}>
            {report.note}
          </div>
        </div>
        <div style={{ ...mono, fontSize: 11, color: "#64748b", textAlign: "right" }}>
          {Object.entries(report.groupHoldPct).map(([groupName, holdPct], index, entries) => (
            <span key={`${report.key}-${groupName}`}>
              {groupName} hold {holdPct.toFixed(2)}%
              {index < entries.length - 1 ? " · " : ""}
            </span>
          ))}
        </div>
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
            <tr style={{ background: "#fafbfc", borderBottom: "1px solid #e2e8f0" }}>
              {report.scope === "region" && <th style={thStyle}>Region</th>}
              <th style={thStyle}>Seed</th>
              <th style={{ ...thStyle, textAlign: "left" }}>Team</th>
              <th style={thStyle}>HRB</th>
              <th style={thStyle}>HRB Fair %</th>
              <th style={thStyle}>Model</th>
              <th style={thStyle}>Delta</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={`${report.key}-${row.group}-${row.team}`} style={{ borderBottom: "1px solid #f8fafc" }}>
                {report.scope === "region" && (
                  <td style={{ ...tdStyle, ...mono, color: "#64748b" }}>{row.group}</td>
                )}
                <td style={{ ...tdStyle, ...mono, color: "#64748b", fontWeight: 600 }}>{row.seed}</td>
                <td style={{ ...tdStyle, textAlign: "left" }}>
                  <div style={{ fontWeight: 600, color: "#0f172a" }}>{row.team}</div>
                  <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>
                    {row.marketKey === "champion" ? row.region ?? "--" : row.hrbTeamName}
                  </div>
                </td>
                <td style={tdStyle}>
                  <div style={{ ...mono, fontWeight: 700, color: "#0f172a" }}>{row.hrbOdds}</div>
                  <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>
                    {formatPercent(row.hrbProb)}
                  </div>
                </td>
                <td style={tdStyle}>
                  <div style={{ ...mono, fontWeight: 700, color: "#0f172a" }}>
                    {formatPercent(row.hrbFairProb)}
                  </div>
                </td>
                <td style={tdStyle}>
                  <div style={{ ...mono, fontWeight: 700, color: "#0f172a" }}>
                    {row.modelOdds ?? "--"}
                  </div>
                  <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>
                    {formatPercent(row.modelProb)}
                  </div>
                </td>
                <td style={tdStyle}>
                  <div
                    style={{
                      ...mono,
                      fontWeight: 700,
                      color: deltaColor(row.deltaPctPoints),
                    }}
                  >
                    {row.deltaPctPoints > 0 ? "+" : ""}
                    {row.deltaPctPoints.toFixed(2)} pp
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function filterMarketRows(
  report: HardRockMarketReport,
  search: string,
  region: string,
): HardRockMarketRow[] {
  const query = search.trim().toLowerCase();
  return report.rows.filter((row) => {
    if (region !== "all" && row.region !== region) return false;
    if (!query) return true;
    return (
      row.team.toLowerCase().includes(query) ||
      row.hrbTeamName.toLowerCase().includes(query) ||
      row.group.toLowerCase().includes(query) ||
      (row.region ?? "").toLowerCase().includes(query)
    );
  });
}

function hardRockMarketKeyForRound(
  roundKey: NcaaOddsRoundKey,
): HardRockMarketKey | null {
  if (roundKey === "sweet-16") return "sweet-16";
  if (roundKey === "elite-8") return "elite-8";
  if (roundKey === "final-four") return "final-four";
  if (roundKey === "champion") return "champion";
  return null;
}

function marketLabelForRound(roundKey: NcaaOddsRoundKey): string {
  if (roundKey === "final-four") return "HR F4";
  if (roundKey === "champion") return "HR Champ";
  return "HR";
}

export default function MarchAnalysisTab({
  ncaaData,
  ncaaTorvikData,
  ncaaInternalData,
  hardRockReport,
}: {
  ncaaData: NcaaOddsData | null;
  ncaaTorvikData: NcaaOddsData | null;
  ncaaInternalData: NcaaOddsData | null;
  hardRockReport: HardRockComparisonData | null;
}) {
  const [search, setSearch] = useState("");
  const [region, setRegion] = useState<string>("all");
  const [sortKey, setSortKey] = useState<NcaaOddsRoundKey>("champion");

  const regions = useMemo(
    () =>
      Array.from(new Set((ncaaData?.rows ?? []).map((row) => row.region).filter(Boolean))).sort() as string[],
    [ncaaData?.rows],
  );

  const filteredOddsRows = useMemo(() => {
    if (!ncaaData) return [];
    const query = search.trim().toLowerCase();
    return [...ncaaData.rows]
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
        const diff = b.roundProbabilities[sortKey] - a.roundProbabilities[sortKey];
        if (diff !== 0) return diff;
        return a.seed - b.seed || a.team.localeCompare(b.team);
      });
  }, [ncaaData, region, search, sortKey]);

  const comparisonRowsByTeamId = useMemo(
    () => new Map((ncaaInternalData?.rows ?? []).map((row) => [row.teamId, row])),
    [ncaaInternalData],
  );
  const torvikRowsByTeamId = useMemo(
    () => new Map((ncaaTorvikData?.rows ?? []).map((row) => [row.teamId, row])),
    [ncaaTorvikData],
  );
  const hardRockRowsByMarketAndTeam = useMemo(() => {
    const byMarket = new Map<HardRockMarketKey, Map<number, HardRockMarketRow>>();
    for (const report of hardRockReport?.reports ?? []) {
      const marketRows = new Map<number, HardRockMarketRow>();
      for (const row of report.rows) {
        const matchedTeam = ncaaData?.rows.find((teamRow) => teamRow.team === row.team);
        if (matchedTeam) {
          marketRows.set(matchedTeam.teamId, row);
        }
      }
      byMarket.set(report.key, marketRows);
    }
    return byMarket;
  }, [hardRockReport?.reports, ncaaData?.rows]);

  const filteredReports = useMemo(
    () =>
      (hardRockReport?.reports ?? []).map((report) => ({
        report,
        rows: filterMarketRows(report, search, region),
      })),
    [hardRockReport?.reports, search, region],
  );

  if (!ncaaData) {
    return (
      <div
        style={{
          padding: 24,
          borderRadius: 12,
          border: "1px solid #e2e8f0",
          background: "#fff",
          color: "#475569",
        }}
      >
        NCAA tournament odds data is unavailable.
      </div>
    );
  }

  const sweet16Overlay = hardRockReport?.sweet16Report?.topOverlays[0] ?? null;
  const elite8Overlay = hardRockReport?.elite8Report?.topOverlays[0] ?? null;
  const finalFourOverlay = hardRockReport?.regionWinnerReport?.topOverlays[0] ?? null;
  const championshipOverlay = hardRockReport?.championshipReport?.topOverlays[0] ?? null;
  const optimalBracket = ncaaData.optimalBracket;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
          gap: 12,
        }}
      >
        <ComparisonSummaryCard
          label="Method"
          primary="Shown Avg ML"
          secondary="Shown Avg uses the matchup-card midpoint. The table below now also exposes raw Torvik, internal HE, and Hard Rock for post-R32 rounds."
        />
        <ComparisonSummaryCard
          label="Hard Rock Feed"
          primary={hardRockReport?.status === "manual_snapshot" ? "Manual snapshot" : "Unavailable"}
          secondary={hardRockReport?.note ?? "No Hard Rock comparison loaded."}
        />
        <ComparisonSummaryCard
          label="Best Sweet 16 Overlay"
          primary={sweet16Overlay ? `${sweet16Overlay.team} (${sweet16Overlay.group})` : "--"}
          secondary={
            sweet16Overlay
              ? `Model ${sweet16Overlay.modelOdds ?? "--"} vs HRB ${sweet16Overlay.hrbOdds}`
              : "No Sweet 16 data"
          }
        />
        <ComparisonSummaryCard
          label="Best Elite 8 Overlay"
          primary={elite8Overlay ? `${elite8Overlay.team} (${elite8Overlay.group})` : "--"}
          secondary={
            elite8Overlay
              ? `Model ${elite8Overlay.modelOdds ?? "--"} vs HRB ${elite8Overlay.hrbOdds}`
              : "No Elite 8 data"
          }
        />
        <ComparisonSummaryCard
          label="Best Region Overlay"
          primary={finalFourOverlay ? `${finalFourOverlay.team} (${finalFourOverlay.group})` : "--"}
          secondary={
            finalFourOverlay
              ? `Model ${finalFourOverlay.modelOdds ?? "--"} vs HRB ${finalFourOverlay.hrbOdds}`
              : "No region data"
          }
        />
        <ComparisonSummaryCard
          label="Best Title Overlay"
          primary={championshipOverlay ? championshipOverlay.team : "--"}
          secondary={
            championshipOverlay
              ? `Model ${championshipOverlay.modelOdds ?? "--"} vs HRB ${championshipOverlay.hrbOdds}`
              : "No title data"
          }
        />
        <ComparisonSummaryCard
          label="Max Expected Score"
          primary={`${roundExpectedPoints(optimalBracket.totalExpectedPoints)} / ${optimalBracket.totalPossiblePoints}`}
          secondary="Exact expected bracket points using display win probabilities and 1-2-4-8-16-32 scoring."
        />
        <ComparisonSummaryCard
          label="Best Round"
          primary={
            optimalBracket.rounds
              .slice()
              .sort((a, b) => b.expectedPoints - a.expectedPoints)[0]?.roundLabel ?? "--"
          }
          secondary={
            optimalBracket.rounds.length
              ? `${roundExpectedPoints(
                  optimalBracket.rounds
                    .slice()
                    .sort((a, b) => b.expectedPoints - a.expectedPoints)[0].expectedPoints,
                )} expected points`
              : "No round data"
          }
        />
      </div>

      <div
        style={{
          background: "#fff",
          border: "1px solid #e2e8f0",
          borderRadius: 12,
          overflow: "hidden",
          boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
        }}
      >
        <div
          style={{
            padding: "14px 16px",
            borderBottom: "1px solid #f1f5f9",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            gap: 12,
            flexWrap: "wrap",
          }}
        >
          <div>
            <div style={{ fontSize: 16, fontWeight: 700, color: "#0f172a" }}>
              Optimal Expected-Score Bracket
            </div>
            <div style={{ ...mono, fontSize: 11, color: "#64748b", marginTop: 4 }}>
              Uses the exact bracket tree and display-based moneyline probabilities. Picks maximize expected total points, not title probability.
            </div>
          </div>
          <div style={{ ...mono, fontSize: 12, color: "#0f172a", fontWeight: 700 }}>
            {roundExpectedPoints(optimalBracket.totalExpectedPoints)} / {optimalBracket.totalPossiblePoints}
          </div>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
            gap: 12,
            padding: "14px 16px",
            borderBottom: "1px solid #f8fafc",
            background: "#fafbfc",
          }}
        >
          {optimalBracket.rounds.map((round) => (
            <div
              key={round.roundId}
              style={{
                borderRadius: 10,
                border: "1px solid #e2e8f0",
                background: "#fff",
                padding: "10px 12px",
              }}
            >
              <div style={{ fontWeight: 700, color: "#0f172a", marginBottom: 6 }}>
                {round.roundLabel}
              </div>
              <div style={{ ...mono, fontSize: 11, color: "#64748b" }}>
                {roundExpectedPoints(round.expectedPoints)} expected / {round.maxPoints} max
              </div>
              <div style={{ ...mono, fontSize: 11, color: "#94a3b8", marginTop: 4 }}>
                {round.pointsPerGame} pts per game
              </div>
            </div>
          ))}
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
            gap: 12,
            padding: "14px 16px",
          }}
        >
          {optimalBracket.rounds.map((round) => (
            <div
              key={`picks-${round.roundId}`}
              style={{
                borderRadius: 12,
                border: "1px solid #e2e8f0",
                background: "#fff",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  padding: "12px 14px",
                  borderBottom: "1px solid #f1f5f9",
                  display: "flex",
                  justifyContent: "space-between",
                  gap: 10,
                  alignItems: "baseline",
                }}
              >
                <div style={{ fontWeight: 700, color: "#0f172a" }}>{round.roundLabel}</div>
                <div style={{ ...mono, fontSize: 11, color: "#64748b" }}>
                  {roundExpectedPoints(round.expectedPoints)} pts
                </div>
              </div>
              <div style={{ display: "flex", flexDirection: "column" }}>
                {round.picks.map((pick) => (
                  <div
                    key={pick.gameId}
                    style={{
                      display: "grid",
                      gridTemplateColumns: "minmax(0, 1fr) auto",
                      gap: 12,
                      padding: "10px 14px",
                      borderBottom: "1px solid #f8fafc",
                    }}
                  >
                    <div>
                      <div style={{ fontWeight: 600, color: "#0f172a" }}>
                        ({pick.seed}) {pick.team}
                      </div>
                      <div style={{ ...mono, fontSize: 11, color: "#64748b" }}>
                        {pick.region ?? "National"} · {formatPercent(pick.winProbability)} win
                      </div>
                    </div>
                    <div style={{ textAlign: "right" }}>
                      <div style={{ ...mono, fontWeight: 700, color: "#0f172a" }}>
                        +{roundExpectedPoints(pick.expectedPoints)}
                      </div>
                      <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>
                        {pick.points} pts
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: 10,
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
          onChange={(event) => setSearch(event.target.value)}
          style={{
            ...mono,
            width: 300,
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

      {hardRockReport &&
        filteredReports.map(({ report, rows }) => (
          <div key={report.key} style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
                gap: 12,
              }}
            >
              <MarketTopList title={`Top ${report.label} Overlays`} rows={report.topOverlays} />
              <MarketTopList title={`Top ${report.label} Underlays`} rows={report.topUnderlays} />
            </div>
            <MarketComparisonTable report={report} rows={rows} />
          </div>
        ))}

      <div
        style={{
          background: "#fff",
          border: "1px solid #e2e8f0",
          borderRadius: 12,
          overflow: "hidden",
          boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
        }}
      >
        <div
          style={{
            padding: "14px 16px",
            borderBottom: "1px solid #f1f5f9",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            gap: 12,
            flexWrap: "wrap",
          }}
        >
          <div>
            <div style={{ fontSize: 16, fontWeight: 700, color: "#0f172a" }}>
              Hoops Edge Round Advancement Odds
            </div>
            <div style={{ ...mono, fontSize: 11, color: "#64748b", marginTop: 4 }}>
              Shown Avg = bracket-card midpoint. Torvik = raw active Team A/B Torvik path. Internal = pure internal Team A/B HE path. Hard Rock rows appear for Sweet 16, Elite 8, Final Four, and Champion.
            </div>
          </div>
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
              <tr style={{ background: "#fafbfc", borderBottom: "1px solid #e2e8f0" }}>
                <th style={thStyle}>Seed</th>
                <th style={{ ...thStyle, textAlign: "left" }}>Team</th>
                <th style={thStyle}>Region</th>
                <th style={thStyle}>
                  Shown Avg
                  <div style={{ fontSize: 10, fontWeight: 500, textTransform: "none", letterSpacing: 0 }}>
                    Modal midpoint
                  </div>
                </th>
                <th style={thStyle}>
                  Torvik
                  <div style={{ fontSize: 10, fontWeight: 500, textTransform: "none", letterSpacing: 0 }}>
                    Raw active
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
                    Shown - Int
                  </div>
                </th>
                {ROUND_COLUMNS.map((column) => (
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
              {filteredOddsRows.map((row) => (
                <tr key={row.teamId} style={{ borderBottom: "1px solid #f8fafc" }}>
                  <td style={{ ...tdStyle, ...mono, fontWeight: 600, color: "#64748b" }}>
                    {row.seed}
                  </td>
                  <td style={{ ...tdStyle, textAlign: "left" }}>
                    <div style={{ fontWeight: 600, color: "#0f172a" }}>{row.team}</div>
                    <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>{row.conference}</div>
                  </td>
                  <td style={{ ...tdStyle, ...mono, color: "#64748b" }}>{row.region ?? "--"}</td>
                  <td style={tdStyle}>
                    <div style={{ ...mono, fontWeight: 700, color: "#0f172a" }}>
                      {formatPercent(row.roundProbabilities[sortKey])}
                    </div>
                    <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>
                      {formatRoundOdds(row.roundProbabilities[sortKey]) ?? "--"}
                    </div>
                  </td>
                  <td style={tdStyle}>
                    <div style={{ ...mono, fontWeight: 700, color: "#0f172a" }}>
                      {torvikRowsByTeamId.get(row.teamId)
                        ? formatPercent(torvikRowsByTeamId.get(row.teamId)!.roundProbabilities[sortKey])
                        : "--"}
                    </div>
                    <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>
                      {torvikRowsByTeamId.get(row.teamId)
                        ? formatRoundOdds(torvikRowsByTeamId.get(row.teamId)!.roundProbabilities[sortKey]) ?? "--"
                        : "--"}
                    </div>
                  </td>
                  <td style={tdStyle}>
                    <div style={{ ...mono, fontWeight: 700, color: "#0f172a" }}>
                      {comparisonRowsByTeamId.get(row.teamId)
                        ? formatPercent(comparisonRowsByTeamId.get(row.teamId)!.roundProbabilities[sortKey])
                        : "--"}
                    </div>
                    <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>
                      {comparisonRowsByTeamId.get(row.teamId)
                        ? formatRoundOdds(comparisonRowsByTeamId.get(row.teamId)!.roundProbabilities[sortKey]) ?? "--"
                        : "--"}
                    </div>
                  </td>
                  <td style={tdStyle}>
                    {(() => {
                      const internalRow = comparisonRowsByTeamId.get(row.teamId);
                      const delta =
                        internalRow == null
                          ? null
                          : (row.roundProbabilities[sortKey] - internalRow.roundProbabilities[sortKey]) * 100;
                      return (
                        <>
                          <div
                            style={{
                              ...mono,
                              fontWeight: 700,
                              color: delta == null ? "#94a3b8" : deltaColor(delta),
                            }}
                          >
                            {delta == null ? "--" : `${delta > 0 ? "+" : ""}${delta.toFixed(1)} pp`}
                          </div>
                          <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>
                            {internalRow ? "Shown - Int" : "--"}
                          </div>
                        </>
                      );
                    })()}
                  </td>
                  {ROUND_COLUMNS.map((column) => {
                    const probability = row.roundProbabilities[column.key];
                    const torvikProbability =
                      torvikRowsByTeamId.get(row.teamId)?.roundProbabilities[column.key] ?? null;
                    const internalRoundProbability =
                      comparisonRowsByTeamId.get(row.teamId)?.roundProbabilities[column.key] ?? null;
                    const hardRockRow = (() => {
                      const marketKey = hardRockMarketKeyForRound(column.key);
                      if (!marketKey) return null;
                      return hardRockRowsByMarketAndTeam.get(marketKey)?.get(row.teamId) ?? null;
                    })();
                    return (
                      <td key={column.key} style={tdStyle}>
                        <div style={{ ...mono, fontWeight: 700, color: "#0f172a" }}>
                          Avg {formatPercent(probability)}
                        </div>
                        <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>
                          {formatRoundOdds(probability) ?? "--"}
                        </div>
                        <div style={{ ...mono, fontWeight: 700, color: "#334155", marginTop: 6 }}>
                          T {torvikProbability == null ? "--" : formatPercent(torvikProbability)}
                        </div>
                        <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>
                          {torvikProbability == null ? "--" : formatRoundOdds(torvikProbability) ?? "--"}
                        </div>
                        <div style={{ ...mono, fontWeight: 700, color: "#334155", marginTop: 6 }}>
                          HE {internalRoundProbability == null ? "--" : formatPercent(internalRoundProbability)}
                        </div>
                        <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>
                          {internalRoundProbability == null ? "--" : formatRoundOdds(internalRoundProbability) ?? "--"}
                        </div>
                        <div style={{ ...mono, fontWeight: 700, color: "#334155", marginTop: 6 }}>
                          {marketLabelForRound(column.key)} {hardRockRow == null ? "--" : formatPercent(hardRockRow.hrbFairProb)}
                        </div>
                        <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>
                          {hardRockRow == null ? "--" : hardRockRow.hrbOdds}
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

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
  padding: "10px 12px",
  textAlign: "center",
  whiteSpace: "nowrap",
};
