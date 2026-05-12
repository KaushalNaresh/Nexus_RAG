"use client";

import { useState } from "react";
import { Progress } from "@/components/ui/progress";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  CheckCircle2, XCircle, MinusCircle, ChevronDown, ChevronUp,
  BookOpen, TrendingUp, Info
} from "lucide-react";
import evalData from "@/public/eval_data.json";

type MetricKey = "faithfulness" | "answer_relevancy" | "context_precision" | "context_recall";

const METRIC_KEYS: MetricKey[] = [
  "faithfulness", "answer_relevancy", "context_precision", "context_recall",
];

const METRIC_COLORS: Record<MetricKey, string> = {
  faithfulness: "bg-violet-500",
  answer_relevancy: "bg-blue-500",
  context_precision: "bg-emerald-500",
  context_recall: "bg-amber-500",
};

const METRIC_LABELS: Record<MetricKey, string> = {
  faithfulness: "Faithfulness",
  answer_relevancy: "Answer Relevancy",
  context_precision: "Context Precision",
  context_recall: "Context Recall",
};

function scoreStatus(score: number | null, threshold: number) {
  if (score === null) return "unknown";
  if (score >= threshold) return "pass";
  if (score >= threshold * 0.7) return "warn";
  return "fail";
}

function StatusIcon({ status }: { status: string }) {
  if (status === "pass") return <CheckCircle2 className="h-4 w-4 text-green-400" />;
  if (status === "warn") return <MinusCircle className="h-4 w-4 text-yellow-400" />;
  if (status === "fail") return <XCircle className="h-4 w-4 text-red-400" />;
  return <MinusCircle className="h-4 w-4 text-slate-500" />;
}

function MetricCard({ metricKey }: { metricKey: MetricKey }) {
  const [expanded, setExpanded] = useState(false);
  const info = evalData.metrics_info[metricKey];
  const score = evalData.latest[metricKey];
  const status = scoreStatus(score, info.good_threshold);
  const pct = score !== null ? Math.round(score * 100) : 0;

  const statusColor =
    status === "pass" ? "text-green-400" : status === "warn" ? "text-yellow-400" : "text-red-400";
  const borderColor =
    status === "pass" ? "border-green-500/20" : status === "warn" ? "border-yellow-500/20" : "border-red-500/20";
  const bgColor =
    status === "pass" ? "bg-green-500/5" : status === "warn" ? "bg-yellow-500/5" : "bg-red-500/5";

  return (
    <div className={`rounded-xl border ${borderColor} ${bgColor} p-4 space-y-3`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <StatusIcon status={status} />
          <span className="text-sm font-semibold text-white">{info.name}</span>
        </div>
        <span className={`text-xl font-bold font-mono ${statusColor}`}>
          {score !== null ? score.toFixed(4) : "N/A"}
        </span>
      </div>

      <Progress
        value={pct}
        indicatorClassName={METRIC_COLORS[metricKey]}
        className="h-2"
      />

      <p className="text-xs text-slate-400 leading-relaxed">{info.description}</p>

      <button
        onClick={() => setExpanded((e) => !e)}
        className="flex items-center gap-1 text-xs text-slate-600 hover:text-slate-400 transition-colors"
      >
        {expanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
        How it&apos;s calculated
      </button>

      {expanded && (
        <div className="rounded-lg bg-black/20 border border-white/8 px-3 py-2">
          <code className="text-xs text-brand-300">{info.formula}</code>
        </div>
      )}
    </div>
  );
}

function ProgressionChart() {
  const maxScore = 1.0;

  return (
    <div className="space-y-4">
      {evalData.runs.map((run) => (
        <div key={run.run} className="space-y-2">
          <div className="flex items-center justify-between">
            <div>
              <span className="text-xs font-semibold text-white">
                Run {run.run} — {run.label}
              </span>
              <span className="ml-2 text-xs text-slate-600">
                ({run.docs_ingested} doc{run.docs_ingested !== 1 ? "s" : ""})
              </span>
            </div>
          </div>

          {/* Mini bar chart per metric */}
          <div className="grid grid-cols-4 gap-2">
            {METRIC_KEYS.map((key) => {
              const score = run[key] as number | null;
              const pct = score !== null ? Math.round((score / maxScore) * 100) : 0;
              return (
                <div key={key} className="space-y-1">
                  <div className="text-xs text-slate-600 truncate">{METRIC_LABELS[key].split(" ")[0]}</div>
                  <div className="h-1.5 rounded-full bg-white/8 overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-700 ${METRIC_COLORS[key]}`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <div className="text-xs font-mono text-slate-400">
                    {score !== null ? score.toFixed(2) : "N/A"}
                  </div>
                </div>
              );
            })}
          </div>

          {run.note && (
            <p className="text-xs text-slate-600 italic">{run.note}</p>
          )}

          {run.run < evalData.runs.length && (
            <div className="border-b border-white/6 mt-3" />
          )}
        </div>
      ))}
    </div>
  );
}

export default function EvalReport() {
  return (
    <div className="space-y-8">
      {/* Latest scores */}
      <section className="space-y-4">
        <div className="flex items-center gap-2">
          <BookOpen className="h-5 w-5 text-brand-400" />
          <h2 className="text-lg font-bold text-white">Current Scores</h2>
          <span className="text-xs text-slate-500">
            {evalData.latest.num_samples} samples · {evalData.latest.docs_ingested} docs ingested
          </span>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {METRIC_KEYS.map((key) => (
            <MetricCard key={key} metricKey={key} />
          ))}
        </div>
      </section>

      {/* Score interpretation */}
      <section>
        <div className="rounded-xl border border-white/8 bg-white/3 p-4 space-y-3">
          <div className="flex items-center gap-2">
            <Info className="h-4 w-4 text-brand-400" />
            <h3 className="text-sm font-semibold text-white">Why are scores moderate?</h3>
          </div>
          <div className="text-xs text-slate-400 space-y-2 leading-relaxed">
            <p>
              These scores reflect a knowledge base built from general Wikipedia and documentation URLs
              rather than the specific technical corpus the golden QA pairs were written for.
              In a production deployment with a domain-specific knowledge base, faithfulness and recall
              would increase substantially.
            </p>
            <p>
              Key improvements already implemented:{" "}
              <span className="text-slate-300">hybrid BM25 + dense retrieval</span> boosts precision by
              matching rare technical terms that dense search misses.{" "}
              <span className="text-slate-300">Cross-encoder reranking</span> moves the most relevant
              chunks to the top before the LLM sees them.
            </p>
          </div>
        </div>
      </section>

      {/* Progression chart */}
      <section className="space-y-4">
        <div className="flex items-center gap-2">
          <TrendingUp className="h-5 w-5 text-brand-400" />
          <h2 className="text-lg font-bold text-white">Improvement Trajectory</h2>
          <span className="text-xs text-slate-500">4 evaluation runs</span>
        </div>
        <Card>
          <CardContent className="pt-5">
            <ProgressionChart />
          </CardContent>
        </Card>
      </section>

      {/* How we measured */}
      <section>
        <Card>
          <CardHeader>
            <CardTitle className="text-sm text-white">How We Measured</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xs text-slate-400 space-y-2 leading-relaxed">
              <p>
                Evaluation was run using{" "}
                <code className="text-brand-300 bg-brand-600/10 px-1 rounded">Ragas 0.4.x</code>
                {" "}with 8 golden QA pairs covering RAG, BM25, cross-encoders, and prompt injection.
              </p>
              <p>
                LLM judge:{" "}
                <code className="text-brand-300 bg-brand-600/10 px-1 rounded">GPT-4o-mini</code>
                {" "}wrapped in <code className="text-brand-300 bg-brand-600/10 px-1 rounded">LangchainLLMWrapper</code>.
                Embeddings:{" "}
                <code className="text-brand-300 bg-brand-600/10 px-1 rounded">text-embedding-3-small</code>.
              </p>
              <p>
                Results saved to{" "}
                <code className="text-brand-300 bg-brand-600/10 px-1 rounded">eval_results/ragas_*.json</code>.
                Run with:{" "}
                <code className="text-brand-300 bg-brand-600/10 px-1 rounded">python scripts/run_evaluation.py</code>
              </p>
            </div>
          </CardContent>
        </Card>
      </section>
    </div>
  );
}
