import { GradedGameResult, ResolvedBracketGame, MatchupPrediction } from "../../lib/bracket/types";
import { GameComparison } from "../../lib/bracket/comparison";
import BracketGame from "./BracketGame";

export default function BracketRound({
  label,
  games,
  predictions,
  comparisons,
  grading,
  loadingGames,
  errorGames,
  onSelectWinner,
  compact,
  stickyTitle = false,
  minWidth,
  dense = false,
}: {
  label: string;
  games: ResolvedBracketGame[];
  predictions: Record<string, MatchupPrediction | undefined>;
  comparisons: Record<string, GameComparison | undefined>;
  grading: Record<string, GradedGameResult | undefined>;
  loadingGames: Record<string, boolean | undefined>;
  errorGames: Record<string, string | undefined>;
  onSelectWinner: (gameId: string, teamId: number) => void;
  compact?: boolean;
  stickyTitle?: boolean;
  minWidth?: number;
  dense?: boolean;
}) {
  return (
    <section
      style={{
        minWidth: compact ? "auto" : minWidth ?? 300,
        display: "flex",
        flexDirection: "column",
        gap: dense ? 8 : 12,
      }}
    >
      <div
        style={{
          position: stickyTitle ? "sticky" : "static",
          top: stickyTitle ? 0 : undefined,
          zIndex: stickyTitle ? 1 : undefined,
          background: stickyTitle ? "#f8fafc" : "transparent",
          paddingBottom: 4,
        }}
      >
        <h2
          style={{
            fontSize: dense ? 12 : 15,
            fontWeight: 700,
            letterSpacing: "-0.02em",
            margin: 0,
            color: dense ? "#475569" : "#0f172a",
            textTransform: dense ? "uppercase" : "none",
          }}
        >
          {label}
        </h2>
      </div>

      {games.map((game) => (
        <BracketGame
          key={game.id}
          game={game}
          prediction={predictions[game.id]}
          comparison={comparisons[game.id]}
          grading={grading[game.id]}
          predictionLoading={loadingGames[game.id]}
          predictionError={errorGames[game.id]}
          onSelectWinner={onSelectWinner}
        />
      ))}
    </section>
  );
}
