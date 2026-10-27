import { useEffect, useMemo, useState } from "react";
import { api } from "../api";

type Split = "dev" | "sealed" | "all";

const FLAME_KEYS = ["uncontrolled", "controlled", "spark", "screen", "unknown", "smoke_only"];

const pct = (v?: number | null) => (typeof v === "number" ? `${(v * 100).toFixed(2)}%` : "-");

function Table({ head, rows, highlight }: { head: string[]; rows: (string | number)[][]; highlight?: number }) {
  return (
    <div className="overflow-auto rounded border border-slate-800">
      <table className="w-full text-left text-xs">
        <thead className="bg-slate-800/60 text-slate-300">
          <tr>
            {head.map((h) => (
              <th key={h} className="whitespace-nowrap px-2 py-1 font-medium">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={i}
              className={`border-t border-slate-800 ${highlight === i ? "bg-ember-600/20 text-white" : "text-slate-300"}`}
            >
              {row.map((v, j) => (
                <td key={j} className="whitespace-nowrap px-2 py-1">
                  {v}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function BenchRResults() {
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState("");
  const [split, setSplit] = useState<Split>("all");

  useEffect(() => {
    api
      .benchRResults()
      .then((d) => {
        setData(d);
        setError("");
      })
      .catch((e) => setError(String(e.message ?? e)));
  }, []);

  const models: any[] = useMemo(() => data?.models ?? [], [data]);

  const bestIdx = useMemo(() => {
    let best = -1;
    let acc = -1;
    models.forEach((m, i) => {
      const a = m?.splits?.[split]?.combined?.accuracy;
      if (typeof a === "number" && a > acc) {
        acc = a;
        best = i;
      }
    });
    return best;
  }, [models, split]);

  const mainRows = useMemo(
    () =>
      models.map((m) => {
        const s = m?.splits?.[split] ?? {};
        const c = s.combined ?? {};
        const ap = s.box_ap ?? {};
        const pol = s.policy_uncontrolled_plus_smoke ?? {};
        const strict = s.policy_strict ?? {};
        return [
          m.model,
          String(s.n ?? "-"),
          pct(c.accuracy),
          pct(c.recall),
          pct(c.specificity),
          pct(c.fpr),
          pct(c.precision),
          pct(c.f1),
          pct(ap.fire),
          pct(ap.smoke),
          pct(strict.recall),
          pct(pol.recall),
        ];
      }),
    [models, split],
  );

  const flameKeys = useMemo(() => {
    const present = new Set<string>(FLAME_KEYS);
    models.forEach((m) => Object.keys(m?.splits?.[split]?.by_flame_type ?? {}).forEach((k) => present.add(k)));
    return FLAME_KEYS.filter((k) => present.has(k));
  }, [models, split]);

  const flameRows = useMemo(
    () =>
      models.map((m) => {
        const by = m?.splits?.[split]?.by_flame_type ?? {};
        return [m.model, ...flameKeys.map((k) => (typeof by[k]?.recall === "number" ? `${pct(by[k].recall)} (n=${by[k].n})` : "-"))];
      }),
    [models, split, flameKeys],
  );

  return (
    <div className="space-y-4">
      <div className="flex items-baseline justify-between gap-4">
        <h1 className="text-2xl font-semibold">Reference bench results</h1>
        <span className="font-mono text-[11px] text-slate-500">bench_R manifest {String(data?.bench_locked_sha256 ?? "").slice(0, 12)}</span>
      </div>

      {error && <div className="rounded border border-red-800 bg-red-950/40 px-3 py-2 text-sm text-red-300">{error}</div>}

      <div className="flex flex-wrap items-center gap-2 text-sm">
        <select
          className="rounded bg-slate-800 px-2 py-1 text-sm"
          value={split}
          onChange={(e) => setSplit(e.target.value as Split)}
        >
          <option value="dev">dev (300, tunable)</option>
          <option value="sealed">sealed (700, one-shot)</option>
          <option value="all">all (1000)</option>
        </select>
        <span className="text-slate-400">
          Thresholds calibrated on {data?.calibrated_on ?? "val only"} — never tuned on bench_R.
        </span>
      </div>

      {split !== "dev" && (
        <div className="rounded border border-amber-800/60 bg-amber-950/30 px-3 py-2 text-xs text-amber-300">
          This split has already been scored. Iterate on <b>dev</b> only; treat every non-dev number as final.
        </div>
      )}

      <div>
        <div className="mb-2 text-sm text-slate-300">
          Image-level accuracy &amp; detection quality — best on {split}:{" "}
          <b className="text-ember-400">{bestIdx >= 0 ? models[bestIdx]?.model : "-"}</b>
        </div>
        <Table
          head={[
            "model", "n", "accuracy", "recall", "specificity", "fpr", "precision", "f1",
            "fire AP50", "smoke AP50", "strict policy", "legacy policy",
          ]}
          rows={mainRows}
          highlight={bestIdx}
        />
      </div>

      <div>
        <div className="mb-2 text-sm text-slate-300">
          Recall by flame type (uncontrolled / controlled / spark / screen / unknown / smoke-only) — split: {split}
        </div>
        <Table head={["model", ...flameKeys]} rows={flameRows} />
      </div>

      {!!models.length && (
        <div className="rounded border border-slate-800 px-3 py-2 text-[11px] leading-relaxed text-slate-500">
          <b className="text-slate-400">strict policy</b> = alarm iff the ground-truth flame type is
          <i> uncontrolled</i>, for fire <b>and</b> smoke (controlled / spark / screen / unknown are suppressed and leave the
          denominator). <b className="text-slate-400">legacy policy</b> = alarm unless controlled/spark/screen, so
          <i> unknown</i> still alarms. Policy rows are not comparable to plain accuracy because suppressed positives are
          excluded from n. &nbsp;·&nbsp; bench_R is negative-heavy (1000 images, 716 negative / 284 positive), so plain
          accuracy flatters weak models; recall is the metric that matters. Box AP rests on 527 fire boxes across 178 fire
          images. &nbsp;·&nbsp;
          {!Object.keys(data?.not_run ?? {}).length ? "" : `Not run: ${Object.keys(data.not_run).join(", ")}.`}
        </div>
      )}
    </div>
  );
}
