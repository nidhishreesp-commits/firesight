import { NavLink, Route, Routes } from "react-router-dom";
import { VideoSessionProvider } from "./VideoSession";
import LiveView from "./pages/LiveView";
import TestBench from "./pages/TestBench";
import BenchReview from "./pages/BenchReview";
import BenchRReview from "./pages/BenchRReview";
import BenchRResults from "./pages/BenchRResults";
import Benchmark from "./pages/Benchmark";
import VideoTest from "./pages/VideoTest";
import ReviewQueue from "./pages/ReviewQueue";
import ModelComparison from "./pages/ModelComparison";

const links = [
  { to: "/", label: "Live view" },
  { to: "/testbench", label: "Test bench" },
  { to: "/bench-review", label: "Bench review" },
  { to: "/bench-R-review", label: "Reference bench" },
  { to: "/bench-R-results", label: "Bench results" },
  { to: "/benchmark", label: "Run benchmark" },
  { to: "/video", label: "Video test" },
  { to: "/review", label: "Review queue" },
  { to: "/models", label: "Model comparison" }
];

export default function App() {
  return (
    <VideoSessionProvider>
    <div className="min-h-screen flex">
      <aside className="w-56 shrink-0 border-r border-slate-800 bg-slate-900/60 p-4">
        <div className="mb-6">
          <div className="text-xl font-bold text-ember-500">FireSight</div>
          <div className="text-xs text-slate-400">fire + smoke detection</div>
        </div>
        <nav className="space-y-1">
          {links.map((l) => (
            <NavLink
              key={l.to}
              to={l.to}
              end={l.to === "/"}
              className={({ isActive }) =>
                `block rounded px-3 py-2 text-sm ${isActive ? "bg-ember-600 text-white" : "text-slate-300 hover:bg-slate-800"}`
              }
            >
              {l.label}
            </NavLink>
          ))}
        </nav>
        <div className="mt-8 text-[11px] leading-relaxed text-slate-500">
          Supplements, never replaces, certified fire alarm systems. Human review required.
        </div>
      </aside>
      <main className="flex-1 p-6 overflow-auto">
        <Routes>
          <Route path="/" element={<LiveView />} />
          <Route path="/testbench" element={<TestBench />} />
          <Route path="/bench-review" element={<BenchReview />} />
          <Route path="/bench-R-review" element={<BenchRReview />} />
          <Route path="/bench-R-results" element={<BenchRResults />} />
          <Route path="/benchmark" element={<Benchmark />} />
          <Route path="/video" element={<VideoTest />} />
          <Route path="/review" element={<ReviewQueue />} />
          <Route path="/models" element={<ModelComparison />} />
        </Routes>
      </main>
    </div>
    </VideoSessionProvider>
  );
}
