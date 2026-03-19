import { CSSProperties, useMemo, useState } from "react";

export type ConferenceBracket = {
  name: string;
  team_count: number;
  champion: string;
  champion_seed: number;
  bracket_lines: string[];
  dnq: string[];
};

export type ConferenceBracketsData = {
  generated_at: string;
  season: number;
  conferences: ConferenceBracket[];
};

const mono: CSSProperties = {
  fontFamily: "'IBM Plex Mono', monospace",
};

const POWER_CONFERENCES = ["ACC", "Big Ten", "Big 12", "SEC", "Big East"];

function isUpset(conf: ConferenceBracket): boolean {
  return conf.champion_seed > 1;
}

function bracketPreview(lines: string[]): string[] {
  return lines
    .map((line) => line.trimEnd())
    .filter((line) => line.trim().length > 0)
    .slice(0, 4);
}

export default function ConferenceBrackets({
  data,
}: {
  data: ConferenceBracketsData | null;
}) {
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<"all" | "power" | "mid">("all");
  const [openConference, setOpenConference] = useState<string | null>(null);

  const conferences = useMemo(() => {
    if (!data) return [];
    let list = data.conferences;

    if (filter === "power") {
      list = list.filter((c) => POWER_CONFERENCES.includes(c.name));
    } else if (filter === "mid") {
      list = list.filter((c) => !POWER_CONFERENCES.includes(c.name));
    }

    if (search.trim()) {
      const q = search.trim().toLowerCase();
      list = list.filter(
        (c) => c.name.toLowerCase().includes(q) || c.champion.toLowerCase().includes(q),
      );
    }

    return [...list].sort((a, b) => a.name.localeCompare(b.name));
  }, [data, search, filter]);

  if (!data || !data.conferences.length) {
    return (
      <div style={{ padding: 24, color: "#94a3b8", textAlign: "center" }}>
        No bracket data available.
      </div>
    );
  }

  const upsetCount = data.conferences.filter(isUpset).length;
  const activeConference = conferences.find((conf) => conf.name === openConference) ?? null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-end",
          marginBottom: 4,
          flexWrap: "wrap",
          gap: 12,
        }}
      >
        <div>
          <h1
            style={{
              fontSize: 24,
              fontWeight: 700,
              letterSpacing: "-0.02em",
              margin: "0 0 4px",
              color: "#0f172a",
            }}
          >
            Conference Tournament Brackets
          </h1>
          <div style={{ ...mono, fontSize: 12, color: "#64748b" }}>
            Browse by conference card. Open a bracket only when you need it.
          </div>
        </div>
        <span style={{ ...mono, fontSize: 12, color: "#64748b" }}>
          {data.conferences.length} conferences • {upsetCount} upset champs
        </span>
      </div>

      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: 10,
          flexWrap: "wrap",
          padding: 12,
          borderRadius: 12,
          border: "1px solid #e2e8f0",
          background: "#ffffff",
          boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
        }}
      >
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
          {(["all", "power", "mid"] as const).map((f) => (
            <button
              key={f}
              type="button"
              onClick={() => setFilter(f)}
              style={{
                ...mono,
                padding: "7px 12px",
                border: `1px solid ${filter === f ? "#0f172a" : "#e2e8f0"}`,
                borderRadius: 999,
                fontSize: 12,
                fontWeight: filter === f ? 600 : 500,
                background: filter === f ? "#0f172a" : "#fff",
                color: filter === f ? "#fff" : "#64748b",
                cursor: "pointer",
              }}
            >
              {f === "all" ? "All conferences" : f === "power" ? "Power conferences" : "Others"}
            </button>
          ))}
        </div>
        <input
          type="text"
          placeholder="Search conference or champion"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          style={{
            ...mono,
            width: 280,
            maxWidth: "100%",
            padding: "8px 10px",
            border: "1px solid #e2e8f0",
            borderRadius: 8,
            fontSize: 13,
            outline: "none",
            background: "#fff",
            color: "#334155",
          }}
        />
      </div>

      {conferences.length === 0 ? (
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
      ) : (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
            gap: 14,
          }}
        >
          {conferences.map((conf) => (
            <article
              key={conf.name}
              style={{
                background: "#fff",
                border: "1px solid #e2e8f0",
                borderRadius: 12,
                padding: 14,
                boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
                display: "flex",
                flexDirection: "column",
                gap: 12,
                minHeight: 214,
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "flex-start" }}>
                <div>
                  <div style={{ fontSize: 17, fontWeight: 700, color: "#0f172a" }}>{conf.name}</div>
                  <div style={{ ...mono, fontSize: 11, color: "#64748b", marginTop: 3 }}>
                    {conf.team_count} teams
                  </div>
                </div>
                {isUpset(conf) ? (
                  <span
                    style={{
                      ...mono,
                      fontSize: 10,
                      fontWeight: 700,
                      padding: "3px 7px",
                      borderRadius: 999,
                      background: "#fffbeb",
                      color: "#b45309",
                    }}
                  >
                    UPSET
                  </span>
                ) : null}
              </div>

              <div
                style={{
                  borderRadius: 10,
                  border: "1px solid #e2e8f0",
                  background: "#f8fafc",
                  padding: 10,
                }}
              >
                <div style={{ ...mono, fontSize: 10, color: "#64748b", marginBottom: 4 }}>
                  Champion
                </div>
                <div style={{ fontSize: 14, fontWeight: 700, color: "#0f172a" }}>
                  ({conf.champion_seed}) {conf.champion}
                </div>
              </div>

              <div
                style={{
                  borderRadius: 10,
                  border: "1px solid #e2e8f0",
                  background: "#fbfdff",
                  padding: 10,
                  minHeight: 86,
                }}
              >
                <div style={{ ...mono, fontSize: 10, color: "#64748b", marginBottom: 6 }}>
                  Bracket preview
                </div>
                <div
                  style={{
                    fontFamily: "'IBM Plex Mono', 'JetBrains Mono', monospace",
                    fontSize: 10,
                    lineHeight: 1.45,
                    color: "#475569",
                    display: "flex",
                    flexDirection: "column",
                    gap: 2,
                    overflow: "hidden",
                  }}
                >
                  {bracketPreview(conf.bracket_lines).map((line, index) => (
                    <div key={`${conf.name}-preview-${index}`}>{line || " "}</div>
                  ))}
                </div>
              </div>

              <div style={{ marginTop: "auto", display: "flex", justifyContent: "space-between", gap: 8, alignItems: "center" }}>
                <div style={{ ...mono, fontSize: 11, color: "#94a3b8" }}>
                  {conf.dnq.length ? `DNQ ${conf.dnq.length}` : "No DNQ list"}
                </div>
                <button
                  type="button"
                  onClick={() => setOpenConference(conf.name)}
                  style={{
                    ...mono,
                    padding: "8px 11px",
                    borderRadius: 8,
                    border: "1px solid #0f172a",
                    background: "#ffffff",
                    color: "#0f172a",
                    cursor: "pointer",
                    fontSize: 12,
                  }}
                >
                  View bracket
                </button>
              </div>
            </article>
          ))}
        </div>
      )}

      {activeConference ? (
        <div
          onClick={() => setOpenConference(null)}
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(15, 23, 42, 0.48)",
            zIndex: 70,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            padding: 20,
          }}
        >
          <div
            onClick={(event) => event.stopPropagation()}
            style={{
              width: "min(1100px, 100%)",
              maxHeight: "min(88vh, 980px)",
              overflow: "hidden",
              borderRadius: 16,
              background: "#ffffff",
              boxShadow: "0 20px 50px rgba(15,23,42,0.28)",
              display: "flex",
              flexDirection: "column",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                gap: 10,
                alignItems: "flex-start",
                padding: "16px 18px 14px",
                borderBottom: "1px solid #e2e8f0",
              }}
            >
              <div>
                <div style={{ fontSize: 20, fontWeight: 700, color: "#0f172a" }}>{activeConference.name}</div>
                <div style={{ ...mono, fontSize: 12, color: "#64748b", marginTop: 4 }}>
                  Champion ({activeConference.champion_seed}) {activeConference.champion} • {activeConference.team_count} teams
                </div>
              </div>
              <button
                type="button"
                onClick={() => setOpenConference(null)}
                style={{
                  ...mono,
                  width: 30,
                  height: 30,
                  borderRadius: 999,
                  border: "1px solid #cbd5e1",
                  background: "#ffffff",
                  color: "#475569",
                  cursor: "pointer",
                  flexShrink: 0,
                }}
              >
                x
              </button>
            </div>

            <div style={{ padding: "12px 18px", borderBottom: "1px solid #f1f5f9", display: "flex", gap: 8, flexWrap: "wrap" }}>
              {isUpset(activeConference) ? (
                <span
                  style={{
                    ...mono,
                    fontSize: 10,
                    fontWeight: 700,
                    padding: "3px 7px",
                    borderRadius: 999,
                    background: "#fffbeb",
                    color: "#b45309",
                  }}
                >
                  UPSET CHAMPION
                </span>
              ) : null}
              {activeConference.dnq.length ? (
                <span style={{ ...mono, fontSize: 11, color: "#64748b" }}>
                  DNQ: {activeConference.dnq.join(", ")}
                </span>
              ) : null}
            </div>

            <div style={{ padding: 18, overflow: "auto" }}>
              <pre
                style={{
                  fontFamily: "'IBM Plex Mono', 'JetBrains Mono', monospace",
                  fontSize: 12,
                  lineHeight: 1.5,
                  margin: 0,
                  color: "#334155",
                  whiteSpace: "pre",
                  minWidth: "fit-content",
                }}
              >
                {activeConference.bracket_lines.join("\n")}
              </pre>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
