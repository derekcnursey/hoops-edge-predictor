import { CSSProperties, useState } from "react";
import BracketBuilder from "../bracket/BracketBuilder";
import ConferenceBrackets from "../bracket/ConferenceBrackets";
import MarchBettingTab from "../bracket/MarchBettingTab";
import Layout from "../Layout";
import { MarchPageProps } from "../../lib/bracket/marchPageData";
import MarchAnalysisTab from "./MarchAnalysisTab";

const mono: CSSProperties = {
  fontFamily: "'IBM Plex Mono', monospace",
};

export default function MarchHub({
  conferenceData,
  ncaaField,
  ncaaErrors,
  ncaaResults,
  ncaaResultsErrors,
  initialPredictionCache,
  marchGames,
  ncaaData,
  ncaaTorvikData,
  ncaaInternalData,
  hardRockReport,
}: MarchPageProps) {
  const [tab, setTab] = useState<"builder" | "betting" | "analysis" | "conference">("builder");

  return (
    <Layout wide>
      <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            gap: 12,
            flexWrap: "wrap",
          }}
        >
          <div>
            <div style={{ ...mono, fontSize: 12, color: "#64748b", marginBottom: 6 }}>
              March
            </div>
            <div
              style={{
                fontSize: 28,
                fontWeight: 700,
                letterSpacing: "-0.03em",
                color: "#0f172a",
              }}
            >
              NCAA tournament hub
            </div>
          </div>

          <div
            style={{
              display: "flex",
              gap: 8,
              padding: 4,
              borderRadius: 10,
              border: "1px solid #e2e8f0",
              background: "#ffffff",
              flexWrap: "wrap",
            }}
          >
            <button
              type="button"
              onClick={() => setTab("builder")}
              style={tabButtonStyle(tab === "builder")}
            >
              NCAA Builder
            </button>
            <button
              type="button"
              onClick={() => setTab("betting")}
              style={tabButtonStyle(tab === "betting")}
            >
              March Betting
            </button>
            <button
              type="button"
              onClick={() => setTab("analysis")}
              style={tabButtonStyle(tab === "analysis")}
            >
              Odds Report
            </button>
            <button
              type="button"
              onClick={() => setTab("conference")}
              style={tabButtonStyle(tab === "conference")}
            >
              Conference Brackets
            </button>
          </div>
        </div>

        {tab === "builder" ? (
          ncaaField && ncaaErrors.length === 0 ? (
            <BracketBuilder
              field={ncaaField}
              results={ncaaResults}
              resultsErrors={ncaaResultsErrors}
              initialPredictionCache={initialPredictionCache}
            />
          ) : (
            <div
              style={{
                padding: 24,
                borderRadius: 12,
                border: "1px solid #fecaca",
                background: "#fef2f2",
                color: "#991b1b",
              }}
            >
              <div style={{ fontWeight: 700, marginBottom: 8 }}>
                NCAA bracket-builder data unavailable
              </div>
              <div style={{ ...mono, fontSize: 12 }}>
                {ncaaErrors[0] ?? "Unknown validation error"}
              </div>
            </div>
          )
        ) : tab === "betting" ? (
          <MarchBettingTab games={marchGames} />
        ) : tab === "analysis" ? (
          <MarchAnalysisTab
            ncaaData={ncaaData}
            ncaaTorvikData={ncaaTorvikData}
            ncaaInternalData={ncaaInternalData}
            hardRockReport={hardRockReport}
          />
        ) : (
          <ConferenceBrackets data={conferenceData} />
        )}
      </div>
    </Layout>
  );
}

function tabButtonStyle(active: boolean): CSSProperties {
  return {
    ...mono,
    padding: "10px 14px",
    borderRadius: 8,
    border: "none",
    background: active ? "#0f172a" : "transparent",
    color: active ? "#ffffff" : "#475569",
    cursor: "pointer",
  };
}
