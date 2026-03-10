import { GetServerSideProps } from "next";
import { CSSProperties, useMemo, useState } from "react";
import Layout from "../components/Layout";
import { readJsonFile } from "../lib/server-data";

/* -- types -- */

type Conference = {
  name: string;
  team_count: number;
  champion: string;
  champion_seed: number;
  bracket_lines: string[];
  dnq: string[];
};

type BracketsData = {
  generated_at: string;
  season: number;
  conferences: Conference[];
};

type Props = {
  data: BracketsData | null;
};

/* -- server-side -- */

export const getServerSideProps: GetServerSideProps<Props> = async () => {
  const raw = readJsonFile("brackets_2026.json");
  const data = raw as BracketsData | null;
  return { props: { data } };
};

/* -- helpers -- */

const mono: CSSProperties = {
  fontFamily: "'IBM Plex Mono', monospace",
};

const POWER_CONFERENCES = ["ACC", "Big Ten", "Big 12", "SEC", "Big East"];

function isUpset(conf: Conference): boolean {
  return conf.champion_seed > 1;
}

/* -- component -- */

export default function Brackets({ data }: Props) {
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<"all" | "power" | "mid">("all");

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
        (c) =>
          c.name.toLowerCase().includes(q) ||
          c.champion.toLowerCase().includes(q)
      );
    }

    return list;
  }, [data, search, filter]);

  if (!data || !data.conferences.length) {
    return (
      <Layout>
        <div style={{ padding: 24, color: "#94a3b8", textAlign: "center" }}>
          No bracket data available.
        </div>
      </Layout>
    );
  }

  const upsetCount = data.conferences.filter(isUpset).length;

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
            Conference Tournament Brackets
          </h1>
          <span style={{ ...mono, fontSize: 13, color: "#64748b" }}>
            {data.conferences.length} conferences · {upsetCount} upset picks
          </span>
        </div>

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
            {(["all", "power", "mid"] as const).map((f) => (
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
                {f === "all" ? "All" : f === "power" ? "Power 5" : "Mid-Major"}
              </button>
            ))}
          </div>
          <input
            type="text"
            placeholder="Search conference or champion..."
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

        {/* -- Bracket Cards -- */}
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          {conferences.map((conf) => (
            <div
              key={conf.name}
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
                  <span
                    style={{
                      fontSize: 16,
                      fontWeight: 700,
                      color: "#0f172a",
                    }}
                  >
                    {conf.name}
                  </span>
                  <span
                    style={{
                      ...mono,
                      fontSize: 11,
                      color: "#94a3b8",
                    }}
                  >
                    {conf.team_count} teams
                  </span>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  {isUpset(conf) && (
                    <span
                      style={{
                        ...mono,
                        fontSize: 10,
                        fontWeight: 600,
                        padding: "2px 8px",
                        borderRadius: 4,
                        background: "rgba(234, 179, 8, 0.12)",
                        color: "#a16207",
                        letterSpacing: "0.05em",
                      }}
                    >
                      UPSET
                    </span>
                  )}
                  <span
                    style={{
                      ...mono,
                      fontSize: 13,
                      fontWeight: 600,
                      color: "#0f172a",
                    }}
                  >
                    ({conf.champion_seed}) {conf.champion}
                  </span>
                </div>
              </div>

              {/* Bracket display */}
              <div
                style={{
                  padding: "12px 18px 16px",
                  overflowX: "auto",
                }}
              >
                <pre
                  style={{
                    fontFamily: "'IBM Plex Mono', 'JetBrains Mono', monospace",
                    fontSize: 12,
                    lineHeight: 1.5,
                    margin: 0,
                    color: "#334155",
                    whiteSpace: "pre",
                  }}
                >
                  {conf.bracket_lines.join("\n")}
                </pre>
              </div>

              {/* DNQ footer */}
              {conf.dnq.length > 0 && (
                <div
                  style={{
                    padding: "8px 18px 12px",
                    borderTop: "1px solid #f1f5f9",
                  }}
                >
                  <span
                    style={{
                      ...mono,
                      fontSize: 11,
                      color: "#94a3b8",
                    }}
                  >
                    DNQ: {conf.dnq.join(", ")}
                  </span>
                </div>
              )}
            </div>
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
      </div>
    </Layout>
  );
}
