import { CSSProperties, useMemo, useState } from "react";
import { displayTeam } from "../../lib/data";
import { MarchBettingGame } from "../../lib/bracket/types";

const mono: CSSProperties = {
  fontFamily: "'IBM Plex Mono', monospace",
};

type SortKey = "matchup" | "time" | "book" | "model" | "sigma" | "diff" | "edge";
type SortState = { key: SortKey; dir: "asc" | "desc" };

function num(value: number | null | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function formatSpread(value: number): string {
  if (Math.abs(value) < 0.05) return "PK";
  return value > 0 ? `+${value.toFixed(1)}` : value.toFixed(1);
}

function displayModelSpread(value: number | null | undefined): number | null {
  const v = num(value);
  return v == null ? null : -v;
}

function formatPct(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function formatGameTime(value: string | null | undefined): string | null {
  if (!value) return null;
  const dt = new Date(value);
  if (Number.isNaN(dt.getTime())) return null;
  return dt.toLocaleTimeString("en-US", {
    timeZone: "America/New_York",
    hour: "numeric",
    minute: "2-digit",
  });
}

function matchupLabel(game: MarchBettingGame): string {
  const away = displayTeam(game.awayTeamName ?? game.teamAName);
  const home = displayTeam(game.homeTeamName ?? game.teamBName);
  return `${away} @ ${home}`;
}

function renderTeamName(game: MarchBettingGame, side: "away" | "home") {
  const isAway = side === "away";
  const name = isAway ? game.awayTeamName ?? game.teamAName : game.homeTeamName ?? game.teamBName;
  const seed =
    isAway
      ? game.awayTeamId === game.teamAId
        ? game.teamASeed
        : game.teamBSeed
      : game.homeTeamId === game.teamAId
        ? game.teamASeed
        : game.teamBSeed;
  return (
    <>
      <span style={{ ...mono, fontSize: 11, color: "#64748b", marginRight: 2 }}>{seed}</span>
      {displayTeam(name)}
    </>
  );
}

function rowSortValue(game: MarchBettingGame, key: SortKey): string | number {
  switch (key) {
    case "matchup":
      return matchupLabel(game);
    case "time": {
      if (!game.startTime) return Number.POSITIVE_INFINITY;
      const ms = new Date(game.startTime).getTime();
      return Number.isNaN(ms) ? Number.POSITIVE_INFINITY : ms;
    }
    case "book":
      return game.marketSpreadHome ?? -Infinity;
    case "model":
      return displayModelSpread(game.displayModelSpreadHome) ?? -Infinity;
    case "sigma":
      return game.predSigma ?? -Infinity;
    case "diff":
      return game.displayDiffAbs ?? -Infinity;
    case "edge":
      return game.pickProbEdge ?? -Infinity;
  }
}

function insightTone(game: MarchBettingGame): {
  label: string;
  style: CSSProperties;
} {
  if (game.marketSpreadHome == null) {
    return {
      label: "No line",
      style: { background: "#f8fafc", color: "#475569", border: "1px solid #e2e8f0" },
    };
  }
  if (
    game.displayFavoredTeamId != null &&
    game.marketFavoredTeamId != null &&
    game.displayFavoredTeamId !== game.marketFavoredTeamId
  ) {
    return {
      label: "Side disagreement",
      style: { background: "#fef2f2", color: "#b91c1c", border: "1px solid #fecaca" },
    };
  }
  if ((game.displayDiffAbs ?? 0) >= 5) {
    return {
      label: "Big display gap",
      style: { background: "#fffbeb", color: "#92400e", border: "1px solid #fde68a" },
    };
  }
  if ((game.pickProbEdge ?? 0) >= 0.08) {
    return {
      label: "Edge 8%+",
      style: { background: "#eff6ff", color: "#1d4ed8", border: "1px solid #bfdbfe" },
    };
  }
  return {
    label: "Near market",
    style: { background: "#f0fdf4", color: "#166534", border: "1px solid #bbf7d0" },
  };
}

export default function MarchBettingTab({ games }: { games: MarchBettingGame[] }) {
  const [search, setSearch] = useState("");
  const [roundFilter, setRoundFilter] = useState<"all" | "first-four" | "round-of-64">("all");
  const [sort, setSort] = useState<SortState>({ key: "edge", dir: "desc" });

  const summary = useMemo(() => {
    const lined = games.filter((game) => game.marketSpreadHome != null);
    const avgDisplayDiff =
      lined.length > 0
        ? lined.reduce((sum, game) => sum + (game.displayDiffAbs ?? 0), 0) / lined.length
        : null;
    const avgRawEdge =
      lined.length > 0
        ? lined.reduce((sum, game) => sum + (game.pickProbEdge ?? 0), 0) / lined.length
        : null;
    const topEdge = [...lined].sort((a, b) => (b.pickProbEdge ?? -1) - (a.pickProbEdge ?? -1))[0] ?? null;
    return { linedCount: lined.length, avgDisplayDiff, avgRawEdge, topEdge };
  }, [games]);

  const tableRows = useMemo(() => {
    let list = [...games];

    if (roundFilter !== "all") {
      list = list.filter((game) => game.roundId === roundFilter);
    }
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      list = list.filter((game) =>
        [
          game.awayTeamName,
          game.homeTeamName,
          game.teamAName,
          game.teamBName,
          game.displayFavoredTeamName,
          game.marketFavoredTeamName,
        ]
          .filter(Boolean)
          .some((value) => String(value).toLowerCase().includes(q)),
      );
    }

    list.sort((a, b) => {
      const av = rowSortValue(a, sort.key);
      const bv = rowSortValue(b, sort.key);
      if (typeof av === "number" && typeof bv === "number") {
        return sort.dir === "asc" ? av - bv : bv - av;
      }
      const cmp = String(av).localeCompare(String(bv));
      return sort.dir === "asc" ? cmp : -cmp;
    });
    return list;
  }, [games, roundFilter, search, sort]);

  function handleSort(key: SortKey) {
    setSort((prev) =>
      prev.key === key
        ? { key, dir: prev.dir === "desc" ? "asc" : "desc" }
        : { key, dir: key === "matchup" || key === "time" ? "asc" : "desc" },
    );
  }

  if (!games.length) {
    return (
      <div
        style={{
          padding: 24,
          borderRadius: 12,
          border: "1px solid #e2e8f0",
          background: "#ffffff",
          color: "#64748b",
        }}
      >
        March Betting opening-round data is unavailable.
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))",
          gap: 10,
        }}
      >
        <div style={{ border: "1px solid #e2e8f0", borderRadius: 12, background: "#ffffff", padding: 14 }}>
          <div style={{ ...mono, fontSize: 10, color: "#64748b", marginBottom: 6 }}>OPENING BOARD</div>
          <div style={{ fontSize: 22, fontWeight: 700, color: "#0f172a" }}>{games.length}</div>
          <div style={{ fontSize: 13, color: "#64748b" }}>First Four + Round of 64 games</div>
        </div>
        <div style={{ border: "1px solid #e2e8f0", borderRadius: 12, background: "#ffffff", padding: 14 }}>
          <div style={{ ...mono, fontSize: 10, color: "#64748b", marginBottom: 6 }}>LINE COVERAGE</div>
          <div style={{ fontSize: 22, fontWeight: 700, color: "#0f172a" }}>{summary.linedCount}</div>
          <div style={{ fontSize: 13, color: "#64748b" }}>Games with market spread context</div>
        </div>
        <div style={{ border: "1px solid #e2e8f0", borderRadius: 12, background: "#ffffff", padding: 14 }}>
          <div style={{ ...mono, fontSize: 10, color: "#64748b", marginBottom: 6 }}>AVG DISPLAY DIFF</div>
          <div style={{ fontSize: 22, fontWeight: 700, color: "#0f172a" }}>
            {summary.avgDisplayDiff == null ? "—" : summary.avgDisplayDiff.toFixed(1)}
          </div>
          <div style={{ fontSize: 13, color: "#64748b" }}>Adjusted display spread vs market line</div>
        </div>
        <div style={{ border: "1px solid #e2e8f0", borderRadius: 12, background: "#ffffff", padding: 14 }}>
          <div style={{ ...mono, fontSize: 10, color: "#64748b", marginBottom: 6 }}>AVG RAW EDGE</div>
          <div style={{ fontSize: 22, fontWeight: 700, color: "#0f172a" }}>
            {summary.avgRawEdge == null ? "—" : `${summary.avgRawEdge >= 0 ? "+" : ""}${(summary.avgRawEdge * 100).toFixed(1)}%`}
          </div>
          <div style={{ fontSize: 13, color: "#64748b" }}>Raw model pick_prob_edge</div>
        </div>
      </div>

      <div
        style={{
          borderRadius: 12,
          border: "1px solid #e2e8f0",
          background: "#ffffff",
          padding: 14,
          display: "grid",
          gap: 12,
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
          <div>
            <div style={{ ...mono, fontSize: 12, color: "#64748b", marginBottom: 4 }}>March Betting</div>
            <div style={{ fontSize: 20, fontWeight: 700, color: "#0f172a" }}>Today-style opening-round board</div>
            <div style={{ fontSize: 13, color: "#64748b", marginTop: 4, maxWidth: 760 }}>
              `MODEL` uses the NCAA display spread in the same book-style sign convention as the Today page. `SIGMA` and `EDGE` are the raw model metrics from the same
              scheduled-game inference pass, not reconstructed estimates.
            </div>
          </div>
          <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
            <input
              type="text"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search team..."
              style={{
                ...mono,
                width: 180,
                padding: "8px 10px",
                borderRadius: 8,
                border: "1px solid #e2e8f0",
                background: "#ffffff",
                color: "#334155",
                fontSize: 13,
              }}
            />
            {(["all", "first-four", "round-of-64"] as const).map((value) => (
              <button
                key={value}
                type="button"
                onClick={() => setRoundFilter(value)}
                style={{
                  ...mono,
                  fontSize: 12,
                  padding: "8px 10px",
                  borderRadius: 8,
                  border: `1px solid ${roundFilter === value ? "#0f172a" : "#e2e8f0"}`,
                  background: roundFilter === value ? "#0f172a" : "#ffffff",
                  color: roundFilter === value ? "#ffffff" : "#475569",
                  cursor: "pointer",
                }}
              >
                {value === "all" ? "All" : value === "first-four" ? "First Four" : "Round of 64"}
              </button>
            ))}
          </div>
        </div>

        {summary.topEdge ? (
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: 8,
              alignItems: "center",
              padding: "10px 12px",
              borderRadius: 10,
              border: "1px solid #dbeafe",
              background: "#eff6ff",
            }}
          >
            <span style={{ ...mono, fontSize: 11, color: "#1d4ed8" }}>Top raw edge</span>
            <span style={{ fontSize: 14, fontWeight: 700, color: "#0f172a" }}>{matchupLabel(summary.topEdge)}</span>
            <span style={{ ...mono, fontSize: 12, color: "#1e3a8a" }}>
              {summary.topEdge.pickProbEdge == null
                ? "—"
                : `${summary.topEdge.pickProbEdge >= 0 ? "+" : ""}${(summary.topEdge.pickProbEdge * 100).toFixed(1)}%`}
              {" · "}
              {summary.topEdge.pickSide ?? "—"}
            </span>
          </div>
        ) : null}

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
            <table style={{ width: "100%", borderCollapse: "collapse", fontVariantNumeric: "tabular-nums" }}>
              <thead>
                <tr>
                  {[
                    { key: "matchup", label: "MATCHUP", align: "left" },
                    { key: "time", label: "TIME", align: "center" },
                    { key: "book", label: "HOME SPREAD", align: "center" },
                    { key: "model", label: "MODEL", align: "center" },
                    { key: "sigma", label: "SIGMA", align: "center" },
                    { key: "diff", label: "DIFF", align: "center" },
                    { key: "edge", label: "EDGE", align: "center" },
                  ].map((column) => {
                    const active = sort.key === column.key;
                    return (
                      <th
                        key={column.key}
                        onClick={() => handleSort(column.key as SortKey)}
                        style={{
                          ...mono,
                          fontSize: 10,
                          fontWeight: 600,
                          letterSpacing: "0.08em",
                          padding: "10px 14px",
                          textAlign: column.align as CSSProperties["textAlign"],
                          background: "#fafbfc",
                          color: active ? "#0f172a" : "#64748b",
                          borderBottom: "1px solid #e2e8f0",
                          cursor: "pointer",
                          whiteSpace: "nowrap",
                        }}
                      >
                        {column.label}
                        {active ? <span style={{ marginLeft: 4 }}>{sort.dir === "desc" ? "↓" : "↑"}</span> : null}
                      </th>
                    );
                  })}
                  <th
                    style={{
                      ...mono,
                      fontSize: 10,
                      fontWeight: 600,
                      letterSpacing: "0.08em",
                      padding: "10px 14px",
                      textAlign: "center",
                      background: "#fafbfc",
                      color: "#64748b",
                      borderBottom: "1px solid #e2e8f0",
                      whiteSpace: "nowrap",
                    }}
                  >
                    INSIGHT
                  </th>
                </tr>
              </thead>
              <tbody>
                {tableRows.length === 0 ? (
                  <tr>
                    <td colSpan={8} style={{ padding: 24, textAlign: "center", color: "#94a3b8" }}>
                      No opening-round NCAA games match the current filters.
                    </td>
                  </tr>
                ) : (
                  tableRows.map((game, index) => {
                    const tone = insightTone(game);
                    return (
                      <tr
                        key={game.gameId}
                        style={{
                          borderBottom: "1px solid #f1f5f9",
                          animation: `fadeIn 0.3s ease ${index * 0.02}s both`,
                        }}
                      >
                        <td
                          style={{
                            padding: "10px 14px",
                            textAlign: "left",
                            fontSize: 14,
                            color: "#334155",
                            whiteSpace: "nowrap",
                            minWidth: 260,
                          }}
                        >
                          <span style={{ fontWeight: game.pickSide === "AWAY" ? 700 : 400 }}>
                            {renderTeamName(game, "away")}
                          </span>
                          <span style={{ fontSize: 11, color: "#64748b", margin: "0 4px" }}>at</span>
                          <span style={{ fontWeight: game.pickSide === "HOME" ? 700 : 400 }}>
                            {renderTeamName(game, "home")}
                          </span>
                          <div style={{ ...mono, fontSize: 10, color: "#94a3b8", marginTop: 4 }}>
                            {game.roundLabel}
                            {game.region ? ` · ${game.region}` : ""}
                            {game.favoriteWinProb ? ` · ${formatPct(game.favoriteWinProb)} fav win prob` : ""}
                          </div>
                        </td>
                        <td style={{ ...mono, padding: "10px 14px", textAlign: "center", fontSize: 12, color: "#64748b" }}>
                          {formatGameTime(game.startTime) ?? "—"}
                        </td>
                        <td style={{ ...mono, padding: "10px 14px", textAlign: "center", fontSize: 13, color: "#334155" }}>
                          {game.marketSpreadHome != null ? formatSpread(game.marketSpreadHome) : "—"}
                        </td>
                        <td
                          style={{
                            ...mono,
                            padding: "10px 14px",
                            textAlign: "center",
                            fontSize: 13,
                            fontWeight: 700,
                            color: "#0f172a",
                            minWidth: 150,
                          }}
                        >
                          <div>
                            {displayModelSpread(game.displayModelSpreadHome) != null
                              ? formatSpread(displayModelSpread(game.displayModelSpreadHome) as number)
                              : "—"}
                          </div>
                          <div style={{ fontSize: 10, color: "#64748b", fontWeight: 500, marginTop: 4 }}>
                            raw {displayModelSpread(game.modelSpreadHome) != null
                              ? formatSpread(displayModelSpread(game.modelSpreadHome) as number)
                              : "—"}
                          </div>
                        </td>
                        <td style={{ ...mono, padding: "10px 14px", textAlign: "center", fontSize: 13, color: "#64748b" }}>
                          {game.predSigma != null ? game.predSigma.toFixed(1) : "—"}
                        </td>
                        <td style={{ ...mono, padding: "10px 14px", textAlign: "center", fontSize: 13, fontWeight: 600, color: "#334155" }}>
                          {game.displayDiffAbs != null ? game.displayDiffAbs.toFixed(1) : "—"}
                        </td>
                        <td
                          style={{
                            ...mono,
                            padding: "10px 14px",
                            textAlign: "center",
                            fontSize: 13,
                            fontWeight: 700,
                            color: (game.pickProbEdge ?? 0) >= 0 ? "#0f766e" : "#b91c1c",
                          }}
                        >
                          {game.pickProbEdge != null
                            ? `${game.pickProbEdge >= 0 ? "+" : ""}${(game.pickProbEdge * 100).toFixed(1)}%`
                            : "—"}
                        </td>
                        <td style={{ padding: "10px 14px", textAlign: "center", minWidth: 180 }}>
                          <div
                            style={{
                              ...mono,
                              fontSize: 11,
                              padding: "4px 8px",
                              borderRadius: 999,
                              display: "inline-flex",
                              alignItems: "center",
                              gap: 6,
                              ...tone.style,
                            }}
                          >
                            <span>{tone.label}</span>
                          </div>
                          <div style={{ ...mono, fontSize: 10, color: "#94a3b8", marginTop: 6 }}>
                            {game.marketLineSource ?? "—"}
                          </div>
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
  );
}
