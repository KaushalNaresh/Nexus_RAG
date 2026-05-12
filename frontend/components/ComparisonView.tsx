"use client";

import { useState } from "react";
import {
  Loader2, Send, ShieldOff, Shield, Shuffle, Cpu, Clock,
  Database, ArrowRight, ChevronDown, ChevronUp
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import SourceCard from "@/components/SourceCard";
import { GuardrailMiniTag } from "@/components/GuardrailBadge";
import { compareRag } from "@/lib/api";
import type { QueryResponse } from "@/lib/api";
import { formatLatency } from "@/lib/utils";

const EXAMPLE_QUERIES = [
  "What are the limitations of Retrieval-Augmented Generation?",
  "How does BM25 score documents for retrieval?",
  "What is prompt injection and how can it be prevented?",
  "Explain the difference between precision and recall in evaluation.",
];

interface ColumnProps {
  label: string;
  tag: string;
  tagColor: string;
  features: string[];
  result: QueryResponse | null;
  loading: boolean;
  variant: "naive" | "production";
}

function ResultColumn({ label, tag, tagColor, features, result, loading, variant }: ColumnProps) {
  const [showSources, setShowSources] = useState(true);

  const formatAnswer = (text: string) =>
    text
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\n\n/g, "</p><p>")
      .replace(/\n- /g, "<br/>• ")
      .replace(/^- /g, "• ");

  return (
    <div className="flex flex-col gap-3 min-w-0">
      {/* Column header */}
      <div className="rounded-xl border border-white/10 bg-white/3 p-4">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-bold text-white">{label}</span>
          <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${tagColor}`}>
            {tag}
          </span>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {features.map((f) => (
            <span
              key={f}
              className="text-xs text-slate-500 bg-white/5 border border-white/8 rounded px-2 py-0.5"
            >
              {f}
            </span>
          ))}
        </div>
      </div>

      {/* Result area */}
      <div className="flex-1 rounded-xl border border-white/10 bg-white/3 p-4 min-h-[320px]">
        {loading && (
          <div className="flex flex-col items-center justify-center h-full gap-3 text-slate-500">
            <Loader2 className="h-6 w-6 animate-spin" />
            <span className="text-xs">Running pipeline…</span>
          </div>
        )}

        {!loading && !result && (
          <div className="flex flex-col items-center justify-center h-full gap-2 text-slate-600">
            <Database className="h-8 w-8 opacity-30" />
            <span className="text-xs">Results will appear here</span>
          </div>
        )}

        {!loading && result && (
          <div className="space-y-3">
            {/* Meta row */}
            <div className="flex flex-wrap items-center gap-2">
              <GuardrailMiniTag triggered={result.guardrail_triggered} />
              <span className="flex items-center gap-1 text-xs text-slate-500">
                <Clock className="h-3 w-3" />
                {formatLatency(result.latency_ms)}
              </span>
              <span className="flex items-center gap-1 text-xs text-slate-500">
                <Database className="h-3 w-3" />
                {result.sources.length} chunks
              </span>
            </div>

            {/* Answer */}
            <div
              className="answer-text text-sm text-slate-200 leading-relaxed"
              dangerouslySetInnerHTML={{
                __html: `<p>${formatAnswer(result.answer)}</p>`,
              }}
            />

            {/* Sources toggle */}
            {result.sources.length > 0 && (
              <div>
                <button
                  onClick={() => setShowSources((s) => !s)}
                  className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-300 mb-2"
                >
                  {showSources ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                  {result.sources.length} sources
                </button>
                {showSources && (
                  <div className="space-y-2">
                    {result.sources.map((doc, i) => (
                      <SourceCard key={i} doc={doc} index={i} variant={variant} />
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default function ComparisonView() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [naive, setNaive] = useState<QueryResponse | null>(null);
  const [production, setProduction] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = async (q?: string) => {
    const text = (q ?? query).trim();
    if (!text || loading) return;
    setLoading(true);
    setError(null);
    setNaive(null);
    setProduction(null);
    try {
      const res = await compareRag(text);
      setNaive(res.naive);
      setProduction(res.production);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Compare failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-5">
      {/* Example queries */}
      {!naive && !loading && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {EXAMPLE_QUERIES.map((q) => (
            <button
              key={q}
              onClick={() => { setQuery(q); run(q); }}
              className="text-left rounded-lg border border-white/8 bg-white/3 p-3 text-xs text-slate-400 hover:bg-white/6 hover:text-white hover:border-brand-500/30 transition-all"
            >
              {q}
            </button>
          ))}
        </div>
      )}

      {/* Query input */}
      <div className="flex gap-3 items-end">
        <Textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); run(); } }}
          placeholder="Type a question to compare naive vs. production RAG…"
          className="flex-1 min-h-[56px]"
          disabled={loading}
        />
        <Button onClick={() => run()} disabled={loading || !query.trim()} className="h-14 px-5">
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          <span className="ml-1 hidden sm:inline">{loading ? "Running…" : "Compare"}</span>
        </Button>
      </div>

      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-400">
          {error}
        </div>
      )}

      {/* Side-by-side */}
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto_1fr] gap-3 items-start">
        <ResultColumn
          label="Naive RAG"
          tag="Baseline"
          tagColor="text-yellow-400 bg-yellow-500/10 border border-yellow-500/25"
          features={["Dense-only", "top_k=5", "No reranking", "No guardrails"]}
          result={naive}
          loading={loading}
          variant="naive"
        />

        {/* Divider */}
        <div className="hidden lg:flex flex-col items-center justify-center gap-2 self-stretch py-8">
          <div className="h-full w-px bg-white/8" />
          <div className="shrink-0 rounded-full border border-white/10 bg-white/5 p-1.5">
            <ArrowRight className="h-4 w-4 text-brand-400 rotate-90 lg:rotate-0" />
          </div>
          <div className="h-full w-px bg-white/8" />
        </div>

        <ResultColumn
          label="Production RAG"
          tag="Nexus"
          tagColor="text-brand-400 bg-brand-600/10 border border-brand-600/25"
          features={["Hybrid (α=0.5)", "BM25 + Dense", "Cross-encoder rerank", "Dual guardrails"]}
          result={production}
          loading={loading}
          variant="production"
        />
      </div>

      {/* Difference callout */}
      {naive && production && (
        <div className="rounded-xl border border-white/8 bg-white/3 p-4 text-xs text-slate-400 space-y-1.5">
          <p className="font-semibold text-white text-sm">What changed?</p>
          <ul className="space-y-1 list-disc list-inside">
            <li>
              <strong className="text-slate-300">Retrieval score range</strong>: Naive uses cosine similarity (0–1). Production uses cross-encoder logits (higher = more precise).
            </li>
            <li>
              <strong className="text-slate-300">Latency delta</strong>:{" "}
              {naive.latency_ms && production.latency_ms
                ? `+${formatLatency(production.latency_ms - naive.latency_ms)} for reranking + guardrails`
                : "unavailable"}
            </li>
            <li>
              <strong className="text-slate-300">Guardrails</strong>: Production validates input (NeMo) and output (PII · faithfulness · toxicity).
            </li>
          </ul>
        </div>
      )}
    </div>
  );
}
