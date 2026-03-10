import Link from "next/link";
import { useRouter } from "next/router";
import { ReactNode } from "react";

export default function Layout({ children }: { children: ReactNode }) {
  const router = useRouter();
  return (
    <div className="page">
      <header className="site-header">
        <div className="brand">
          <svg width={28} height={28} viewBox="0 0 40 40" fill="none">
            <rect x="2" y="2" width="36" height="36" rx="8" fill="#0f172a" />
            <circle
              cx="20"
              cy="20"
              r="11"
              stroke="#e2e8f0"
              strokeWidth="1.5"
              fill="none"
            />
            <line
              x1="20"
              y1="9"
              x2="20"
              y2="31"
              stroke="#e2e8f0"
              strokeWidth="1"
            />
            <path
              d="M9 20 Q14 16, 20 16 Q26 16, 31 20"
              stroke="#e2e8f0"
              strokeWidth="1"
              fill="none"
            />
            <path
              d="M9 20 Q14 24, 20 24 Q26 24, 31 20"
              stroke="#e2e8f0"
              strokeWidth="1"
              fill="none"
            />
            <path
              d="M6 28 L14 22 L22 25 L30 12 L34 14"
              stroke="#16a34a"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              fill="none"
            />
          </svg>
          <span className="brand-name">HoopsEdge</span>
        </div>
        <nav className="nav">
          <Link href="/" className={router.pathname === "/" ? "active" : ""}>
            Today
          </Link>
          <Link
            href="/history"
            className={router.pathname === "/history" ? "active" : ""}
          >
            History
          </Link>
          <Link
            href="/rankings"
            className={router.pathname === "/rankings" ? "active" : ""}
          >
            Rankings
          </Link>
          <Link
            href="/brackets"
            className={router.pathname === "/brackets" ? "active" : ""}
          >
            Brackets
          </Link>
          <Link
            href="/tourneys"
            className={router.pathname === "/tourneys" ? "active" : ""}
          >
            Tourneys
          </Link>
          <Link
            href="/metrics"
            className={router.pathname === "/metrics" ? "active" : ""}
          >
            Performance
          </Link>
        </nav>
      </header>
      <main className="content">{children}</main>
      <footer className="site-footer">
        <span>Model predictions · Not financial advice</span>
        <span>HoopsEdge © 2026</span>
      </footer>
    </div>
  );
}
