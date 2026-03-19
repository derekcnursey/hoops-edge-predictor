import { GetServerSideProps } from "next";
import Link from "next/link";
import { useRouter } from "next/router";
import { CSSProperties } from "react";
import Layout from "../../components/Layout";
import {
  BettingPerformancePayload,
  BySeasonProfitabilityRow,
  HistoricalPickRow,
  ProfitabilityRow,
  readBettingPerformancePayload,
  RobustnessRow,
  SignalDriverRow,
} from "../../lib/betting-performance-data";

type Props = {
  payload: BettingPerformancePayload | null;
  initialFocus: PerformanceFocusKey;
  initialMonth: MonthFilterKey;
  initialSeason: string;
};

type PerformanceFocusKey = "promoted" | "raw" | "conf" | "disagreement" | "ncaa";
type MonthFilterKey = "all" | "nov" | "dec" | "jan" | "feb" | "mar";

const mono: CSSProperties = {
  fontFamily: "'IBM Plex Mono', monospace",
};

export const getServerSideProps: GetServerSideProps<Props> = async (context) => {
  const qFocus = typeof context.query.focus === "string" ? context.query.focus : null;
  const qMonth = typeof context.query.month === "string" ? context.query.month : null;
  const qSeason = typeof context.query.season === "string" ? context.query.season : null;
  const validFocus: PerformanceFocusKey[] = ["promoted", "raw", "conf", "disagreement", "ncaa"];
  const validMonth: MonthFilterKey[] = ["all", "nov", "dec", "jan", "feb", "mar"];
  return {
    props: {
      payload: readBettingPerformancePayload(),
      initialFocus: validFocus.includes((qFocus ?? "") as PerformanceFocusKey)
        ? (qFocus as PerformanceFocusKey)
        : "promoted",
      initialMonth: validMonth.includes((qMonth ?? "") as MonthFilterKey)
        ? (qMonth as MonthFilterKey)
        : "all",
      initialSeason: qSeason ?? "all",
    },
  };
};

function fmtPct(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

function fmtNum(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${v}`;
}

function formatSliceLabel(slice: string): string {
  return slice.replace(/_/g, " ").replace(/\b\w/g, (ch) => ch.toUpperCase());
}

function getOverallRow(rows: ProfitabilityRow[], group: string, slice: string): ProfitabilityRow | null {
  return rows.find((row) => row.group === group && row.slice === slice) ?? null;
}

function getRobustnessRow(rows: RobustnessRow[], group: string, slice: string): RobustnessRow | null {
  return rows.find((row) => row.group === group && row.slice === slice) ?? null;
}

function metricTone(v: number | null | undefined): CSSProperties {
  if (v === null || v === undefined || Number.isNaN(v)) return { color: "#64748b" };
  if (v > 0) return { color: "#166534" };
  if (v < 0) return { color: "#b91c1c" };
  return { color: "#334155" };
}

function cardStyle(highlight = false): CSSProperties {
  return {
    background: highlight ? "linear-gradient(180deg, rgba(22, 163, 74, 0.10), #fff)" : "#fff",
    border: highlight ? "1px solid rgba(22, 163, 74, 0.20)" : "1px solid #e2e8f0",
    borderRadius: 12,
    padding: 16,
    boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
  };
}

function SummaryCards({
  payload,
  initialFocus,
  initialMonth,
  initialSeason,
}: {
  payload: BettingPerformancePayload;
  initialFocus: PerformanceFocusKey;
  initialMonth: MonthFilterKey;
  initialSeason: string;
}) {
  const promotedFull = getOverallRow(payload.overall, "promoted_internal_filter", "full");
  const rawFull = getOverallRow(payload.overall, "raw_edge_baseline", "full");
  const promotedConf = getOverallRow(payload.overall, "promoted_internal_filter", "conference_tournaments");
  const disagreementMarch = getOverallRow(payload.overall, "disagreement_led", "march_only");
  const ncaaCaution = getOverallRow(payload.overall, "ncaa_caution", "ncaa_tournament");

  const cards = [
    {
      label: "Promoted Filter",
      subtitle: "Full completed sample",
      value: promotedFull ? fmtPct(promotedFull.roi_per_1_at_minus_110) : "—",
      detail: promotedFull ? `${fmtPct(promotedFull.win_rate)} ATS · ${fmtNum(promotedFull.bets)} bets` : "—",
      highlight: true,
      focus: "promoted" as const,
    },
    {
      label: "Raw Baseline",
      subtitle: "Full completed sample",
      value: rawFull ? fmtPct(rawFull.roi_per_1_at_minus_110) : "—",
      detail: rawFull ? `${fmtPct(rawFull.win_rate)} ATS · ${fmtNum(rawFull.bets)} bets` : "—",
      highlight: false,
      focus: "raw" as const,
    },
    {
      label: "Conference Tourneys",
      subtitle: "Promoted filter",
      value: promotedConf ? fmtPct(promotedConf.roi_per_1_at_minus_110) : "—",
      detail: promotedConf ? `${fmtPct(promotedConf.win_rate)} ATS · ${fmtNum(promotedConf.bets)} bets` : "—",
      highlight: true,
      focus: "conf" as const,
    },
    {
      label: "Disagreement-Led March",
      subtitle: "Historically additive",
      value: disagreementMarch ? fmtPct(disagreementMarch.roi_per_1_at_minus_110) : "—",
      detail: disagreementMarch ? `${fmtPct(disagreementMarch.win_rate)} ATS · ${fmtNum(disagreementMarch.bets)} bets` : "—",
      highlight: true,
      focus: "disagreement" as const,
    },
    {
      label: "NCAA Caution",
      subtitle: "Diagnostic only",
      value: ncaaCaution ? fmtPct(ncaaCaution.roi_per_1_at_minus_110) : "—",
      detail: ncaaCaution ? `${fmtPct(ncaaCaution.win_rate)} ATS · ${fmtNum(ncaaCaution.bets)} bets` : "—",
      highlight: false,
      focus: "ncaa" as const,
    },
  ];

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
        gap: 12,
      }}
    >
      {cards.map((card) => (
        <a
          key={card.label}
          href={`/betting/performance?focus=${card.focus}&month=${initialMonth}&season=${initialSeason}#historical-picks`}
          style={{
            ...cardStyle(card.highlight),
            textDecoration: "none",
            border: card.focus === initialFocus ? "1px solid #0f172a" : cardStyle(card.highlight).border,
            cursor: "pointer",
            display: "block",
          }}
          title="Open matching historical pick table"
        >
          <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>{card.label.toUpperCase()}</div>
          <div style={{ marginTop: 8, fontSize: 24, fontWeight: 800, letterSpacing: "-0.03em", ...metricTone(card.value === "—" ? null : Number.NaN) }}>
            <span style={metricTone(card.value === "—" ? null : Number(card.value.replace("%", "")) / 100)}>{card.value}</span>
          </div>
          <div style={{ marginTop: 6, fontSize: 13, color: "#334155" }}>{card.subtitle}</div>
          <div style={{ ...mono, marginTop: 6, fontSize: 12, color: "#64748b" }}>{card.detail}</div>
        </a>
      ))}
    </div>
  );
}

function fmtLine(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return v > 0 ? `+${v.toFixed(1)}` : v.toFixed(1);
}

function monthFloor(key: MonthFilterKey): number {
  const order: Record<MonthFilterKey, number> = {
    all: 0,
    nov: 1,
    dec: 2,
    jan: 3,
    feb: 4,
    mar: 5,
  };
  return order[key];
}

function monthFilterLabel(key: MonthFilterKey): string {
  switch (key) {
    case "nov":
      return "Nov forward";
    case "dec":
      return "Dec forward";
    case "jan":
      return "Jan forward";
    case "feb":
      return "Feb forward";
    case "mar":
      return "Mar forward";
    default:
      return "All months";
  }
}

function resolveFocusedPicks(
  payload: BettingPerformancePayload,
  focus: PerformanceFocusKey,
): { title: string; subtitle: string; rows: HistoricalPickRow[] } {
  switch (focus) {
    case "raw":
      return {
        title: "Raw Baseline Pick Table",
        subtitle: "All historical picks from the raw-edge baseline bucket.",
        rows: payload.pickTables.raw_edge_baseline,
      };
    case "conf":
      return {
        title: "Conference Tournament Pick Table",
        subtitle: "Promoted internal-filter picks from conference-tournament games only.",
        rows: payload.pickTables.promoted_internal_filter.filter((row) => Boolean(row.is_conference_tournament)),
      };
    case "disagreement":
      return {
        title: "Disagreement-Led March Pick Table",
        subtitle: "Historically additive filter-only picks in March and conference tournaments.",
        rows: payload.pickTables.filter_only.filter((row) =>
          ["march_only", "conference_tournaments"].includes(row.slice)
        ),
      };
    case "ncaa":
      return {
        title: "NCAA Caution Pick Table",
        subtitle: "Diagnostic-only NCAA rows that should not be treated like the main decision bucket.",
        rows: payload.pickTables.ncaa_caution,
      };
    default:
      return {
        title: "Promoted Filter Pick Table",
        subtitle: "All historical picks from the promoted internal disagreement-aware filter.",
        rows: payload.pickTables.promoted_internal_filter,
      };
  }
}

function HistoricalPickTable({
  payload,
  initialFocus,
  initialMonth,
  initialSeason,
}: {
  payload: BettingPerformancePayload;
  initialFocus: PerformanceFocusKey;
  initialMonth: MonthFilterKey;
  initialSeason: string;
}) {
  const router = useRouter();
  const focused = resolveFocusedPicks(payload, initialFocus);
  const filterEnabled = initialFocus === "promoted";
  const floor = monthFloor(initialMonth);
  const monthFilteredRows = filterEnabled && floor
    ? focused.rows.filter((row) => {
        const month = Number(String(row.game_date).slice(5, 7));
        const seasonOrder =
          month === 11 ? 1 :
          month === 12 ? 2 :
          month === 1 ? 3 :
          month === 2 ? 4 :
          month === 3 ? 5 :
          month === 4 ? 6 : 0;
        return seasonOrder >= floor;
      })
    : focused.rows;
  const availableSeasons = Array.from(new Set(focused.rows.map((row) => String(row.season)))).sort();
  const filteredRows = filterEnabled && initialSeason !== "all"
    ? monthFilteredRows.filter((row) => String(row.season) === initialSeason)
    : monthFilteredRows;
  return (
    <section id="historical-picks" style={{ display: "grid", gap: 14 }}>
      <div>
        <h2 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: "#0f172a" }}>{focused.title}</h2>
        <div style={{ ...mono, fontSize: 12, color: "#64748b", marginTop: 4 }}>
          {focused.subtitle}{filterEnabled ? ` · ${monthFilterLabel(initialMonth)}` : ""}
        </div>
      </div>
      {filterEnabled ? (
        <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
          <span style={{ ...mono, fontSize: 12, color: "#64748b" }}>Month forward:</span>
          {(["all", "nov", "dec", "jan", "feb", "mar"] as MonthFilterKey[]).map((month) => (
            <Link
              key={month}
              href={`/betting/performance?focus=${initialFocus}&month=${month}&season=${initialSeason}#historical-picks`}
              style={{
                ...mono,
                fontSize: 12,
                padding: "6px 10px",
                borderRadius: 999,
                border: month === initialMonth ? "1px solid #0f172a" : "1px solid #e2e8f0",
                background: "#fff",
                color: "#334155",
              }}
            >
              {monthFilterLabel(month)}
            </Link>
          ))}
          <span style={{ ...mono, fontSize: 12, color: "#64748b", marginLeft: 8 }}>Season:</span>
          <select
            value={initialSeason}
            onChange={(e) => router.push(`/betting/performance?focus=${initialFocus}&month=${initialMonth}&season=${e.target.value}#historical-picks`)}
            style={{
              ...mono,
              padding: "6px 10px",
              border: "1px solid #e2e8f0",
              borderRadius: 8,
              background: "#fff",
              color: "#334155",
            }}
          >
            <option value="all">All seasons</option>
            {availableSeasons.map((season) => (
              <option key={season} value={season}>
                {season}
              </option>
            ))}
          </select>
        </div>
      ) : null}
      <div style={{ overflowX: "auto", background: "#fff", border: "1px solid #e2e8f0", borderRadius: 12 }}>
        <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 1280 }}>
          <thead>
            <tr style={{ background: "#f8fafc", textAlign: "left" }}>
              {["Date", "Game", "Final", "Pick", "HE Line", "Market Line", "Raw Edge", "Score", "Disagreement", "Context", "Driver", "Result"].map((label) => (
                <th key={label} style={{ ...mono, fontSize: 11, color: "#64748b", padding: "12px 14px", borderBottom: "1px solid #e2e8f0" }}>
                  {label.toUpperCase()}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filteredRows.slice(0, 500).map((row) => (
              <tr key={`${row.gameId}-${row.pick_team}`}>
                <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9" }}>{row.game_date}</td>
                <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9", color: "#0f172a", fontWeight: 600 }}>{row.game}</td>
                <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9" }}>{row.final_score ?? "—"}</td>
                <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9" }}>{row.pick_team}</td>
                <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9" }}>{fmtLine(row.hoops_edge_line_for_pick)}</td>
                <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9" }}>{fmtLine(row.market_line_for_pick)}</td>
                <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9" }}>{fmtPct(row.pick_prob_edge)}</td>
                <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9" }}>{fmtPct(row.disagreement_logit_score)}</td>
                <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9" }}>{fmtNum(row.he_market_edge_for_pick)}</td>
                <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9" }}>
                  {[row.disagreement_context, row.slice.replace(/_/g, " ")].filter(Boolean).join(" · ")}
                </td>
                <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9" }}>{row.signal_driver ?? "—"}</td>
                <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9", ...metricTone(row.roi_per_1_at_minus_110) }}>{row.bet_result ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {filteredRows.length > 500 ? (
        <div style={{ ...mono, fontSize: 12, color: "#64748b" }}>
          Showing first 500 rows of {filteredRows.length}. The full pick list remains in the historical artifact CSVs.
        </div>
      ) : null}
    </section>
  );
}

function ComparisonTable({ payload }: { payload: BettingPerformancePayload }) {
  const rows = [
    ["Full", "promoted_internal_filter", "raw_edge_baseline", "full"],
    ["Feb 15+", "promoted_internal_filter", "raw_edge_baseline", "feb15_plus"],
    ["March Only", "promoted_internal_filter", "raw_edge_baseline", "march_only"],
    ["Conference Tournaments", "promoted_internal_filter", "raw_edge_baseline", "conference_tournaments"],
    ["NCAA Caution", "ncaa_caution", "raw_edge_baseline", "ncaa_tournament"],
  ] as const;

  return (
    <section style={{ display: "grid", gap: 14 }}>
      <div>
        <h2 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: "#0f172a" }}>Promoted Filter vs Raw Baseline</h2>
        <div style={{ ...mono, fontSize: 12, color: "#64748b", marginTop: 4 }}>
          The main historical comparison backing the internal betting workflow.
        </div>
      </div>
      <div style={{ overflowX: "auto", background: "#fff", border: "1px solid #e2e8f0", borderRadius: 12 }}>
        <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 760 }}>
          <thead>
            <tr style={{ background: "#f8fafc", textAlign: "left" }}>
              {["Slice", "Promoted ATS", "Promoted ROI", "Promoted Bets", "Raw ATS", "Raw ROI", "Raw Bets"].map((label) => (
                <th key={label} style={{ ...mono, fontSize: 11, color: "#64748b", padding: "12px 14px", borderBottom: "1px solid #e2e8f0" }}>
                  {label.toUpperCase()}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map(([label, leftGroup, rightGroup, slice]) => {
              const left = getOverallRow(payload.overall, leftGroup, slice);
              const right = getOverallRow(payload.overall, rightGroup, slice);
              return (
                <tr key={label}>
                  <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9", color: "#0f172a", fontWeight: 600 }}>{label}</td>
                  <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9" }}>{fmtPct(left?.win_rate)}</td>
                  <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9", ...metricTone(left?.roi_per_1_at_minus_110) }}>
                    {fmtPct(left?.roi_per_1_at_minus_110)}
                  </td>
                  <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9" }}>{fmtNum(left?.bets)}</td>
                  <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9" }}>{fmtPct(right?.win_rate)}</td>
                  <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9", ...metricTone(right?.roi_per_1_at_minus_110) }}>
                    {fmtPct(right?.roi_per_1_at_minus_110)}
                  </td>
                  <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9" }}>{fmtNum(right?.bets)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function SubgroupTable({ payload }: { payload: BettingPerformancePayload }) {
  const rows = payload.overall.filter((row) =>
    ["disagreement_led", "raw_edge_led", "persistent_disagreement", "new_transient_disagreement"].includes(row.group) &&
    ["full", "march_only", "conference_tournaments"].includes(row.slice)
  );
  return (
    <section style={{ display: "grid", gap: 14 }}>
      <div>
        <h2 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: "#0f172a" }}>Useful Subgroups</h2>
        <div style={{ ...mono, fontSize: 12, color: "#64748b", marginTop: 4 }}>
          Disagreement-led and late-season subgroups that mattered most in the research.
        </div>
      </div>
      <div style={{ overflowX: "auto", background: "#fff", border: "1px solid #e2e8f0", borderRadius: 12 }}>
        <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 700 }}>
          <thead>
            <tr style={{ background: "#f8fafc", textAlign: "left" }}>
              {["Group", "Slice", "ATS", "ROI", "Bets", "Avg Score"].map((label) => (
                <th key={label} style={{ ...mono, fontSize: 11, color: "#64748b", padding: "12px 14px", borderBottom: "1px solid #e2e8f0" }}>
                  {label.toUpperCase()}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={`${row.group}-${row.slice}`}>
                <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9", color: "#0f172a", fontWeight: 600 }}>
                  {row.group.replace(/_/g, " ")}
                </td>
                <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9", color: "#475569" }}>{formatSliceLabel(row.slice)}</td>
                <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9" }}>{fmtPct(row.win_rate)}</td>
                <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9", ...metricTone(row.roi_per_1_at_minus_110) }}>
                  {fmtPct(row.roi_per_1_at_minus_110)}
                </td>
                <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9" }}>{fmtNum(row.bets)}</td>
                <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9" }}>{fmtPct(row.avg_filter_score)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function BySeasonTable({ payload }: { payload: BettingPerformancePayload }) {
  const seasons = Array.from(new Set(payload.bySeason.map((row) => row.season))).sort((a, b) => a - b);
  const getRow = (group: string, season: number): BySeasonProfitabilityRow | null =>
    payload.bySeason.find((row) => row.group === group && row.slice === "full" && row.season === season) ?? null;

  return (
    <section style={{ display: "grid", gap: 14 }}>
      <div>
        <h2 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: "#0f172a" }}>By Season</h2>
        <div style={{ ...mono, fontSize: 12, color: "#64748b", marginTop: 4 }}>
          Full-sample season-by-season comparison for the promoted filter and the raw baseline.
        </div>
      </div>
      <div style={{ overflowX: "auto", background: "#fff", border: "1px solid #e2e8f0", borderRadius: 12 }}>
        <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 720 }}>
          <thead>
            <tr style={{ background: "#f8fafc", textAlign: "left" }}>
              {["Season", "Promoted ATS", "Promoted ROI", "Promoted Bets", "Raw ATS", "Raw ROI", "Raw Bets"].map((label) => (
                <th key={label} style={{ ...mono, fontSize: 11, color: "#64748b", padding: "12px 14px", borderBottom: "1px solid #e2e8f0" }}>
                  {label.toUpperCase()}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {seasons.map((season) => {
              const promoted = getRow("promoted_internal_filter", season);
              const raw = getRow("raw_edge_baseline", season);
              return (
                <tr key={season}>
                  <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9", color: "#0f172a", fontWeight: 600 }}>{season}</td>
                  <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9" }}>{fmtPct(promoted?.win_rate)}</td>
                  <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9", ...metricTone(promoted?.roi_per_1_at_minus_110) }}>
                    {fmtPct(promoted?.roi_per_1_at_minus_110)}
                  </td>
                  <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9" }}>{fmtNum(promoted?.bets)}</td>
                  <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9" }}>{fmtPct(raw?.win_rate)}</td>
                  <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9", ...metricTone(raw?.roi_per_1_at_minus_110) }}>
                    {fmtPct(raw?.roi_per_1_at_minus_110)}
                  </td>
                  <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9" }}>{fmtNum(raw?.bets)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function RobustnessCards({ payload }: { payload: BettingPerformancePayload }) {
  const promotedFull = getRobustnessRow(payload.robustness, "promoted_internal_filter", "full");
  const promotedMarch = getRobustnessRow(payload.robustness, "promoted_internal_filter", "march_only");
  const disagreementMarch = getRobustnessRow(payload.robustness, "disagreement_led", "march_only");

  const cards = [
    {
      title: "Promoted Filter",
      detail: promotedFull
        ? `${promotedFull.profitable_seasons}/${promotedFull.season_count_with_bets} full seasons profitable`
        : "—",
      sub: promotedFull ? `Median season ROI ${fmtPct(promotedFull.median_season_roi)}` : "—",
    },
    {
      title: "March Robustness",
      detail: promotedMarch
        ? `${promotedMarch.profitable_seasons}/${promotedMarch.season_count_with_bets} March samples profitable`
        : "—",
      sub: promotedMarch ? `Median March ROI ${fmtPct(promotedMarch.median_season_roi)}` : "—",
    },
    {
      title: "Disagreement-Led March",
      detail: disagreementMarch
        ? `${disagreementMarch.profitable_seasons}/${disagreementMarch.season_count_with_bets} seasons profitable`
        : "—",
      sub: disagreementMarch ? `Median March ROI ${fmtPct(disagreementMarch.median_season_roi)}` : "—",
    },
  ];

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
        gap: 12,
      }}
    >
      {cards.map((card) => (
        <div key={card.title} style={cardStyle(true)}>
          <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>{card.title.toUpperCase()}</div>
          <div style={{ marginTop: 8, fontSize: 18, fontWeight: 700, color: "#0f172a" }}>{card.detail}</div>
          <div style={{ marginTop: 6, color: "#475569", fontSize: 13 }}>{card.sub}</div>
        </div>
      ))}
    </div>
  );
}

function SignalDriverTable({ payload }: { payload: BettingPerformancePayload }) {
  const rows = payload.signalDrivers.filter((row) =>
    ["full", "march_only", "conference_tournaments"].includes(row.slice) &&
    [
      "disagreement-led persistent",
      "raw edge led",
      "raw edge + new disagreement support",
      "disagreement-led new/transient conf-tourney",
    ].includes(row.signal_driver)
  );
  return (
    <section style={{ display: "grid", gap: 14 }}>
      <div>
        <h2 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: "#0f172a" }}>Signal Driver Cuts</h2>
        <div style={{ ...mono, fontSize: 12, color: "#64748b", marginTop: 4 }}>
          Compact examples of how specific live-page signal labels behaved historically.
        </div>
      </div>
      <div style={{ overflowX: "auto", background: "#fff", border: "1px solid #e2e8f0", borderRadius: 12 }}>
        <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 720 }}>
          <thead>
            <tr style={{ background: "#f8fafc", textAlign: "left" }}>
              {["Signal driver", "Slice", "ATS", "ROI", "Bets"].map((label) => (
                <th key={label} style={{ ...mono, fontSize: 11, color: "#64748b", padding: "12px 14px", borderBottom: "1px solid #e2e8f0" }}>
                  {label.toUpperCase()}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={`${row.signal_driver}-${row.slice}`}>
                <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9", color: "#0f172a", fontWeight: 600 }}>{row.signal_driver}</td>
                <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9", color: "#475569" }}>{formatSliceLabel(row.slice)}</td>
                <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9" }}>{fmtPct(row.win_rate)}</td>
                <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9", ...metricTone(row.roi_per_1_at_minus_110) }}>
                  {fmtPct(row.roi_per_1_at_minus_110)}
                </td>
                <td style={{ padding: "12px 14px", borderBottom: "1px solid #f1f5f9" }}>{fmtNum(row.bets)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default function BettingPerformancePage({ payload, initialFocus, initialMonth, initialSeason }: Props) {
  if (!payload) {
    return (
      <Layout>
        <div style={{ padding: 24, color: "#64748b", textAlign: "center" }}>
          No historical betting performance artifacts found.
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div style={{ display: "grid", gap: 24 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 16, flexWrap: "wrap" }}>
          <div style={{ display: "grid", gap: 10 }}>
            <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
              <span style={{ ...mono, fontSize: 11, padding: "4px 8px", borderRadius: 999, background: "rgba(22, 163, 74, 0.10)", color: "#166534", border: "1px solid rgba(22, 163, 74, 0.2)" }}>
                Internal betting validation
              </span>
              <span style={{ ...mono, fontSize: 11, padding: "4px 8px", borderRadius: 999, background: "rgba(245, 158, 11, 0.12)", color: "#92400e", border: "1px solid rgba(245, 158, 11, 0.25)" }}>
                Separate from public predictions
              </span>
            </div>
            <div>
              <h1 style={{ margin: 0, fontSize: 28, fontWeight: 800, letterSpacing: "-0.03em", color: "#0f172a" }}>
                Betting Performance
              </h1>
              <div style={{ ...mono, fontSize: 13, color: "#64748b", marginTop: 6 }}>
                Historical ATS / ROI view for the internal disagreement-aware betting workflow
              </div>
            </div>
          </div>

          <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
            <Link
              href="/betting"
              style={{
                ...mono,
                fontSize: 12,
                padding: "8px 10px",
                borderRadius: 8,
                border: "1px solid #e2e8f0",
                background: "#fff",
                color: "#334155",
              }}
            >
              Live betting workflow
            </Link>
            <Link
              href="/"
              style={{
                ...mono,
                fontSize: 12,
                padding: "8px 10px",
                borderRadius: 8,
                border: "1px solid #e2e8f0",
                background: "#fff",
                color: "#334155",
              }}
            >
              Public predictions
            </Link>
          </div>
        </div>

        <SummaryCards
          payload={payload}
          initialFocus={initialFocus}
          initialMonth={initialMonth}
          initialSeason={initialSeason}
        />

        <div
          style={{
            background: "rgba(15, 23, 42, 0.03)",
            border: "1px solid #e2e8f0",
            borderRadius: 12,
            padding: 16,
            color: "#334155",
            fontSize: 14,
            lineHeight: 1.6,
          }}
        >
          This page explains the historical record behind the internal betting workflow. It is not a public tout page and it does not change the public Hoops Edge prediction product. The important read is that the promoted internal filter has historically been profitable, the raw-edge-only baseline has been much weaker, disagreement-led picks have been additive, and NCAA stays caution-only.
        </div>

        <div
          style={{
            background: "#fff",
            border: "1px solid #e2e8f0",
            borderRadius: 12,
            padding: 16,
            display: "grid",
            gap: 10,
          }}
        >
          <h2 style={{ margin: 0, fontSize: 18, fontWeight: 700, color: "#0f172a" }}>How To Interpret This Page</h2>
          <div style={{ display: "grid", gap: 8, color: "#334155", fontSize: 14, lineHeight: 1.6 }}>
            <div><strong>Promoted internal filter</strong>: the main historical validation bucket for the live betting tab.</div>
            <div><strong>Raw-edge baseline</strong>: the old simple benchmark. Historically much weaker, especially late season.</div>
            <div><strong>Disagreement-led</strong>: historically additive signal and the best evidence that the disagreement layer matters.</div>
            <div><strong>Conference tournaments</strong>: strongest late-season use case.</div>
            <div><strong>NCAA caution</strong>: tracked for honesty, but not strong enough to treat as a trusted decision bucket.</div>
          </div>
        </div>

        <HistoricalPickTable
          payload={payload}
          initialFocus={initialFocus}
          initialMonth={initialMonth}
          initialSeason={initialSeason}
        />
        <ComparisonTable payload={payload} />
        <RobustnessCards payload={payload} />
        <SubgroupTable payload={payload} />
        <BySeasonTable payload={payload} />
        <SignalDriverTable payload={payload} />
      </div>
    </Layout>
  );
}
