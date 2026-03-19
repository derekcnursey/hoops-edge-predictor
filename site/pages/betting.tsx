import { GetServerSideProps } from "next";
import Link from "next/link";
import { useRouter } from "next/router";
import { CSSProperties } from "react";
import Layout from "../components/Layout";
import {
  InternalBettingPayload,
  InternalBettingRow,
  readInternalBettingPayload,
} from "../lib/internal-betting-data";

type Props = {
  payload: InternalBettingPayload | null;
};

const mono: CSSProperties = {
  fontFamily: "'IBM Plex Mono', monospace",
};

type TierKey = "strongest" | "solid" | "monitor" | "caution";

type RankMeta = {
  rank: number;
  total: number;
  percentile: number;
  percentileLabel: string;
};

type DecoratedRow = InternalBettingRow & {
  scoreTier: TierKey;
  tierLabel: string;
  tierNote: string;
  rankMeta: RankMeta | null;
  disagreementLiftLabel: string;
  regimeNote: string;
};

export const getServerSideProps: GetServerSideProps<Props> = async (context) => {
  const qDate = typeof context.query.date === "string" ? context.query.date : null;
  return {
    props: {
      payload: readInternalBettingPayload(qDate),
    },
  };
};

function fmtSpread(v: number | null): string {
  if (v === null || Number.isNaN(v)) return "—";
  return v > 0 ? `+${v.toFixed(1)}` : v.toFixed(1);
}

function displayModelSpread(v: number | null): number | null {
  if (v === null || Number.isNaN(v)) return null;
  // Internal report stores home margin; site displays book-style home spread.
  return -v;
}

function fmtPct(v: number | null): string {
  if (v === null || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

function fmtScore(v: number | null): string {
  if (v === null || Number.isNaN(v)) return "—";
  return v.toFixed(3);
}

function formatDateDisplay(dateStr: string): string {
  const [year, month, day] = dateStr.split("-");
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  return `${months[Number(month) - 1]} ${Number(day)}, ${year}`;
}

function formatSliceLabel(slice: string | null): string {
  if (!slice) return "regular";
  return slice
    .replace(/_/g, " ")
    .replace(/\b\w/g, (ch) => ch.toUpperCase());
}

function num(v: number | null | undefined): number {
  return typeof v === "number" && Number.isFinite(v) ? v : Number.NaN;
}

function badgeStyle(kind: "internal" | "caution" | "raw" | "neutral"): CSSProperties {
  switch (kind) {
    case "internal":
      return { background: "rgba(22, 163, 74, 0.10)", color: "#166534", border: "1px solid rgba(22, 163, 74, 0.2)" };
    case "caution":
      return { background: "rgba(245, 158, 11, 0.12)", color: "#92400e", border: "1px solid rgba(245, 158, 11, 0.25)" };
    case "raw":
      return { background: "rgba(37, 99, 235, 0.10)", color: "#1d4ed8", border: "1px solid rgba(37, 99, 235, 0.2)" };
    default:
      return { background: "#f8fafc", color: "#475569", border: "1px solid #e2e8f0" };
  }
}

function tierStyle(tier: TierKey): CSSProperties {
  switch (tier) {
    case "strongest":
      return { background: "#0f172a", color: "#f8fafc", border: "1px solid #0f172a" };
    case "solid":
      return { background: "rgba(22, 163, 74, 0.10)", color: "#166534", border: "1px solid rgba(22, 163, 74, 0.2)" };
    case "monitor":
      return { background: "rgba(37, 99, 235, 0.10)", color: "#1d4ed8", border: "1px solid rgba(37, 99, 235, 0.2)" };
    default:
      return { background: "rgba(245, 158, 11, 0.12)", color: "#92400e", border: "1px solid rgba(245, 158, 11, 0.25)" };
  }
}

function rankKey(row: InternalBettingRow): string {
  return row.gameId !== null && row.gameId !== undefined
    ? `gid:${row.gameId}`
    : `${row.awayTeam ?? "away"}@${row.homeTeam ?? "home"}:${row.startDate ?? row.game_date ?? "na"}`;
}

function buildRankMap(rows: InternalBettingRow[]): Map<string, RankMeta> {
  const sorted = [...rows].sort((a, b) => {
    const scoreDiff = num(b.filter_score) - num(a.filter_score);
    if (!Number.isNaN(scoreDiff) && scoreDiff !== 0) return scoreDiff;
    const edgeDiff = num(b.pick_prob_edge) - num(a.pick_prob_edge);
    if (!Number.isNaN(edgeDiff) && edgeDiff !== 0) return edgeDiff;
    return (a.gameId ?? 0) - (b.gameId ?? 0);
  });
  const total = sorted.length;
  return new Map(
    sorted.map((row, idx) => {
      const percentile = total > 1 ? ((total - idx - 1) / (total - 1)) * 100 : 100;
      let percentileLabel = "lower slate band";
      if (percentile >= 95) percentileLabel = "top 5% of slate";
      else if (percentile >= 80) percentileLabel = "top 20% of slate";
      else if (percentile >= 50) percentileLabel = "upper half of slate";
      return [
        rankKey(row),
        {
          rank: idx + 1,
          total,
          percentile,
          percentileLabel,
        },
      ];
    })
  );
}

function deriveTier(row: InternalBettingRow, mode: "internal" | "raw" | "caution"): {
  key: TierKey;
  label: string;
  note: string;
} {
  if (mode === "caution" || row.slice === "ncaa_tournament") {
    return {
      key: "caution",
      label: "Caution only",
      note: "NCAA rows stay diagnostic only, even when the score clears the threshold.",
    };
  }
  const score = num(row.filter_score);
  if (score >= 0.62) {
    return {
      key: "strongest",
      label: "Strongest",
      note: "Top-of-board score band. Historically best overall ROI, but on a smaller sample.",
    };
  }
  if (score >= 0.58) {
    return {
      key: "solid",
      label: "Solid",
      note: "Promoted operating band. This is the main actionable shortlist zone.",
    };
  }
  return {
    key: "monitor",
    label: "Monitor",
    note: "Below the promoted threshold. Review context, not a main internal play.",
  };
}

function buildRegimeNote(row: InternalBettingRow): string {
  if (row.slice === "ncaa_tournament") {
    return "NCAA remains caution-only. Treat this as context, not a trusted main pick.";
  }
  if (row.slice === "conference_tournaments") {
    return "Conference tournaments are the strongest historical use case for this workflow.";
  }
  if (row.slice === "march_only") {
    return "March is noisier than the full sample. Use rank and tier more than the exact decimal score.";
  }
  if (row.flagged_mainly_by_disagreement) {
    return "Disagreement-led rows have been historically additive beyond raw edge alone.";
  }
  return "Raw-edge-led rows are still useful, but they are less diagnostic of disagreement-specific value.";
}

function decorateRows(rows: InternalBettingRow[], allRows: InternalBettingRow[], mode: "internal" | "raw" | "caution"): DecoratedRow[] {
  const rankMap = buildRankMap(allRows);
  const sortRows = [...rows].sort((a, b) => {
    const scoreDiff = num(b.filter_score) - num(a.filter_score);
    if (!Number.isNaN(scoreDiff) && scoreDiff !== 0) return scoreDiff;
    const rawDiff = num(b.pick_prob_edge) - num(a.pick_prob_edge);
    if (!Number.isNaN(rawDiff) && rawDiff !== 0) return rawDiff;
    return (a.gameId ?? 0) - (b.gameId ?? 0);
  });
  return sortRows.map((row) => {
    const tier = deriveTier(row, mode);
    const lift = num(row.score_lift_vs_raw_logit);
    const disagreementLiftLabel = Number.isNaN(lift)
      ? "—"
      : lift >= 0
        ? `+${lift.toFixed(3)} vs raw`
        : `${lift.toFixed(3)} vs raw`;
    return {
      ...row,
      scoreTier: tier.key,
      tierLabel: tier.label,
      tierNote: tier.note,
      rankMeta: rankMap.get(rankKey(row)) ?? null,
      disagreementLiftLabel,
      regimeNote: buildRegimeNote(row),
    };
  });
}

function contextLabel(row: InternalBettingRow): string {
  const parts: string[] = [];
  if (row.persistence_label) parts.push(row.persistence_label);
  if (row.neutral_site_flag) parts.push("neutral");
  if (row.slice) parts.push(formatSliceLabel(row.slice));
  return parts.join(" · ");
}

function RowCard({ row, mode }: { row: DecoratedRow; mode: "internal" | "raw" | "caution" }) {
  const disagreementLed = Boolean(row.flagged_mainly_by_disagreement);
  return (
    <div
      style={{
        background: "#fff",
        border: mode === "caution" ? "1px solid rgba(245, 158, 11, 0.25)" : "1px solid #e2e8f0",
        borderRadius: 12,
        padding: 16,
        boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
        display: "grid",
        gap: 12,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12, flexWrap: "wrap" }}>
        <div>
          <div style={{ fontSize: 17, fontWeight: 700, color: "#0f172a" }}>
            {row.awayTeam} at {row.homeTeam}
          </div>
          <div style={{ ...mono, fontSize: 12, color: "#64748b", marginTop: 4 }}>
            {contextLabel(row)}
          </div>
        </div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", justifyContent: "flex-end" }}>
          <span style={{ ...mono, fontSize: 11, padding: "4px 8px", borderRadius: 999, ...badgeStyle(mode) }}>
            {mode === "internal" ? "Internal candidate" : mode === "raw" ? "Raw-edge watchlist" : "NCAA caution"}
          </span>
          <span style={{ ...mono, fontSize: 11, padding: "4px 8px", borderRadius: 999, ...tierStyle(row.scoreTier) }}>
            {row.tierLabel}
          </span>
          <span style={{ ...mono, fontSize: 11, padding: "4px 8px", borderRadius: 999, ...badgeStyle(disagreementLed ? "internal" : "raw") }}>
            {disagreementLed ? "Disagreement-led" : "Raw-edge-led"}
          </span>
          {row.rankMeta ? (
            <span style={{ ...mono, fontSize: 11, padding: "4px 8px", borderRadius: 999, ...badgeStyle("neutral") }}>
              Slate rank #{row.rankMeta.rank}/{row.rankMeta.total}
            </span>
          ) : null}
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))",
          gap: 10,
        }}
      >
        <div>
          <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>MARKET LINE</div>
          <div style={{ fontSize: 15, fontWeight: 700 }}>{fmtSpread(row.book_spread)}</div>
        </div>
        <div>
          <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>HOOPS EDGE</div>
          <div style={{ fontSize: 15, fontWeight: 700 }}>{fmtSpread(displayModelSpread(row.predicted_spread))}</div>
        </div>
        <div>
          <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>RAW EDGE</div>
          <div style={{ fontSize: 15, fontWeight: 700 }}>{fmtPct(row.pick_prob_edge)}</div>
        </div>
        <div>
          <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>BET SCORE</div>
          <div style={{ fontSize: 15, fontWeight: 700 }}>{fmtScore(row.filter_score)}</div>
        </div>
        <div>
          <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>PICK SIDE</div>
          <div style={{ fontSize: 15, fontWeight: 700 }}>{row.model_pick_side ?? "—"}</div>
        </div>
        <div>
          <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>DISAGREEMENT EDGE</div>
          <div style={{ fontSize: 15, fontWeight: 700 }}>{fmtScore(row.he_market_edge_for_pick)}</div>
        </div>
        <div>
          <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>LIFT VS RAW</div>
          <div style={{ fontSize: 15, fontWeight: 700 }}>{row.disagreementLiftLabel}</div>
        </div>
        <div>
          <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>PERCENTILE</div>
          <div style={{ fontSize: 15, fontWeight: 700 }}>{row.rankMeta ? row.rankMeta.percentileLabel : "—"}</div>
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
          gap: 10,
          color: "#334155",
          fontSize: 13,
        }}
      >
        <div>
          <strong>Signal driver:</strong> {row.signal_driver ?? "—"}
        </div>
        <div>
          <strong>Tier:</strong> {row.tierLabel} · {row.tierNote}
        </div>
        <div>
          <strong>Regime read:</strong> {row.regimeNote}
        </div>
        <div>
          <strong>Usage:</strong> {row.usage_label ?? "—"}
        </div>
        <div>
          <strong>Same-sign 21d:</strong> {row.pick_team_recent_same_sign_count_21d ?? "—"}
        </div>
        <div>
          <strong>Prior streak:</strong> {row.pick_team_prior_same_sign_streak ?? "—"}
        </div>
      </div>
    </div>
  );
}

function Section({
  title,
  subtitle,
  rows,
  mode,
  emptyText,
}: {
  title: string;
  subtitle: string;
  rows: DecoratedRow[];
  mode: "internal" | "raw" | "caution";
  emptyText: string;
}) {
  return (
    <section style={{ display: "grid", gap: 14 }}>
      <div>
        <h2 style={{ margin: 0, fontSize: 20, fontWeight: 700, color: "#0f172a" }}>{title}</h2>
        <div style={{ ...mono, fontSize: 12, color: "#64748b", marginTop: 4 }}>{subtitle}</div>
      </div>
      {rows.length ? (
        <div style={{ display: "grid", gap: 12 }}>
          {rows.map((row) => (
            <RowCard
              key={`${row.gameId ?? `${row.homeTeam}-${row.awayTeam}`}-${row.signal_driver ?? "row"}`}
              row={row}
              mode={mode}
            />
          ))}
        </div>
      ) : (
        <div style={{ background: "#fff", border: "1px solid #e2e8f0", borderRadius: 12, padding: 18, color: "#64748b" }}>
          {emptyText}
        </div>
      )}
    </section>
  );
}

export default function BettingPage({ payload }: Props) {
  const router = useRouter();

  if (!payload) {
    return (
      <Layout>
        <div style={{ padding: 24, color: "#64748b", textAlign: "center" }}>
          No internal betting artifacts found.
        </div>
      </Layout>
    );
  }

  const manifest = payload.manifest;
  const shortlist = decorateRows(payload.shortlist, payload.slateScores, "internal");
  const rawWatchlist = decorateRows(payload.rawWatchlist, payload.slateScores, "raw");
  const ncaaCaution = decorateRows(payload.ncaaCaution, payload.slateScores, "caution");
  const allDecorated = decorateRows(payload.slateScores, payload.slateScores, "raw");
  const strongestCount = allDecorated.filter((row) => row.scoreTier === "strongest" && row.filter_pass).length;
  const solidBandCount = allDecorated.filter((row) => row.scoreTier === "solid" && row.filter_pass).length;
  return (
    <Layout>
      <div style={{ display: "grid", gap: 24 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 16, flexWrap: "wrap" }}>
          <div style={{ display: "grid", gap: 10 }}>
            <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
              <span style={{ ...mono, fontSize: 11, padding: "4px 8px", borderRadius: 999, ...badgeStyle("internal") }}>
                Internal betting aid
              </span>
              <span style={{ ...mono, fontSize: 11, padding: "4px 8px", borderRadius: 999, ...badgeStyle("caution") }}>
                Separate from public predictions
              </span>
            </div>
            <div>
              <h1 style={{ margin: 0, fontSize: 28, fontWeight: 800, letterSpacing: "-0.03em", color: "#0f172a" }}>
                Betting Workflow
              </h1>
              <div style={{ ...mono, fontSize: 13, color: "#64748b", marginTop: 6 }}>
                {formatDateDisplay(payload.currentDate)} · Season {manifest?.season ?? "—"} · {manifest?.slate_games ?? payload.slateScores.length} scored games
              </div>
            </div>
          </div>

          <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
            <select
              value={payload.currentDate}
              onChange={(e) => router.push(`/betting?date=${e.target.value}`)}
              style={{
                ...mono,
                padding: "8px 10px",
                border: "1px solid #e2e8f0",
                borderRadius: 8,
                background: "#fff",
                color: "#334155",
              }}
            >
              {payload.availableDates.map((date) => (
                <option key={date} value={date}>
                  {date}
                </option>
              ))}
            </select>
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
            <Link
              href="/betting/performance"
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
              Historical performance
            </Link>
          </div>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
            gap: 12,
          }}
        >
          {[
            ["Threshold", manifest?.filter_threshold !== undefined ? `disagreement_logit ≥ ${manifest.filter_threshold}` : "—"],
            ["Raw baseline", manifest?.raw_edge_threshold !== undefined ? `pick_prob_edge ≥ ${manifest.raw_edge_threshold}` : "—"],
            ["Shortlist", String(manifest?.non_ncaa_shortlist_games ?? shortlist.length)],
            ["Strongest tier", String(strongestCount)],
            ["Solid tier", String(solidBandCount)],
            ["Raw watchlist", String(manifest?.raw_watchlist_games ?? rawWatchlist.length)],
            ["NCAA caution", String(manifest?.ncaa_caution_games ?? ncaaCaution.length)],
            ["Disagreement-led", String(manifest?.flagged_mainly_by_disagreement ?? shortlist.filter((r) => r.flagged_mainly_by_disagreement).length)],
          ].map(([label, value]) => (
            <div
              key={label}
              style={{
                background: "#fff",
                border: "1px solid #e2e8f0",
                borderRadius: 12,
                padding: 14,
                boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
                textAlign: "left",
                display: "grid",
                gap: 6,
              }}
            >
              <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>{label.toUpperCase()}</div>
              <div style={{ fontSize: 18, fontWeight: 700, color: "#0f172a" }}>{value}</div>
            </div>
          ))}
        </div>

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
          This tab is the internal betting workflow. It uses the disagreement-aware filter on top of the public Hoops Edge slate, but it does not change the public model spread or win probability product. The score is best used as a ranking signal, not a literal ATS probability. NCAA caution rows are separated intentionally and should not be treated like the main shortlist.
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
            gap: 12,
          }}
        >
          {[
            ["Strongest", "Score ≥ 0.62", "Top-of-board band. Historically best full-sample ROI, but on a smaller sample."],
            ["Solid", "0.58 to 0.619", "Main actionable zone. This is the promoted shortlist band."],
            ["Monitor", "0.54 to 0.579", "Useful review context, but below the promoted rule."],
            ["Caution", "NCAA / below-threshold", "Diagnostic only. Keep separate from the main shortlist."],
          ].map(([title, band, note]) => (
            <div
              key={title}
              style={{
                background: "#fff",
                border: "1px solid #e2e8f0",
                borderRadius: 12,
                padding: 14,
                boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
                display: "grid",
                gap: 8,
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8 }}>
                <strong style={{ color: "#0f172a" }}>{title}</strong>
                <span style={{ ...mono, fontSize: 11, color: "#64748b" }}>{band}</span>
              </div>
              <div style={{ fontSize: 13, lineHeight: 1.5, color: "#475569" }}>{note}</div>
            </div>
          ))}
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
          <h2 style={{ margin: 0, fontSize: 18, fontWeight: 700, color: "#0f172a" }}>How To Read This</h2>
          <div style={{ display: "grid", gap: 8, color: "#334155", fontSize: 14, lineHeight: 1.6 }}>
            <div><strong>Internal Candidate Bets</strong>: historically strongest live bucket. This is the promoted disagreement-aware shortlist.</div>
            <div><strong>Disagreement-led</strong>: historically additive beyond raw edge alone. These deserve extra attention when they clear the threshold.</div>
            <div><strong>Raw-Edge Watchlist</strong>: weaker than the promoted filter historically. Keep as review context, not the main decision bucket.</div>
            <div><strong>NCAA Caution</strong>: diagnostic only. The historical sample is too noisy to treat NCAA rows like the main shortlist.</div>
            <div><strong>Slate rank / percentile</strong>: use these more than the exact decimal score, especially in March. The score is a ranking signal first.</div>
            <div><strong>Conference tournaments</strong>: strongest historical use case. <strong>March</strong> is still useful, but noisier. <strong>NCAA</strong> stays caution-only.</div>
          </div>
        </div>

        <Section
          title="Internal Candidate Bets"
          subtitle="Promoted disagreement-aware shortlist for daily use, with tier and slate-rank context."
          rows={shortlist}
          mode="internal"
          emptyText="No non-NCAA internal candidates on this slate."
        />

        <Section
          title="Raw Edge Watchlist"
          subtitle="Games that clear the raw edge baseline but not the promoted disagreement-aware rule."
          rows={rawWatchlist}
          mode="raw"
          emptyText="No raw-edge-only watchlist rows on this slate."
        />

        <Section
          title="NCAA Caution"
          subtitle="Tracked separately as caution / diagnostic context only."
          rows={ncaaCaution}
          mode="caution"
          emptyText="No NCAA caution rows on this slate."
        />
      </div>
    </Layout>
  );
}
