"use client";

import { useState, useRef, useEffect } from "react";
import { Send, Loader2, Clock, Database, Cpu, ChevronDown, ChevronUp, AlertTriangle, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import GuardrailBadge from "@/components/GuardrailBadge";
import SourceCard from "@/components/SourceCard";
import { queryRag } from "@/lib/api";
import type { QueryResponse } from "@/lib/api";
import { formatLatency } from "@/lib/utils";

const EXAMPLE_QUERIES = [
  "What is Retrieval-Augmented Generation and how does it work?",
  "How does a cross-encoder differ from a bi-encoder for reranking?",
  "What metrics does Ragas use to evaluate RAG systems?",
  "How does BM25 handle term frequency and inverse document frequency?",
];

export default function ChatInterface() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isColdStart, setIsColdStart] = useState(false);
  const [showSources, setShowSources] = useState(true);
  const answerRef = useRef<HTMLDivElement>(null);
  const coldStartTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastQuery = useRef<string>("");

  const submit = async (q?: string) => {
    const text = (q ?? query).trim();
    if (!text) return;
    lastQuery.current = text;
    setLoading(true);
    setError(null);
    setResponse(null);
    setIsColdStart(false);

    // Show cold-start warning after 8s still loading
    coldStartTimer.current = setTimeout(() => setIsColdStart(true), 8000);

    try {
      const res = await queryRag(text);
      setResponse(res);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Query failed";
      const isFetchError = msg.toLowerCase().includes("fetch") || msg.toLowerCase().includes("network");
      setError(isFetchError
        ? "Could not reach the backend. If this is the first request, the server may be waking up (free tier). Wait 30s and try again."
        : msg);
    } finally {
      setLoading(false);
      setIsColdStart(false);
      if (coldStartTimer.current) clearTimeout(coldStartTimer.current);
    }
  };

  useEffect(() => {
    if (response && answerRef.current) {
      answerRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [response]);

  const formatAnswer = (text: string) => {
    // Basic markdown-lite: **bold** → <strong>, bullet points
    return text
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\n\n/g, "</p><p>")
      .replace(/\n- /g, "<br/>• ")
      .replace(/^- /g, "• ");
  };

  return (
    <div className="flex flex-col gap-4 h-full">
      {/* Example queries */}
      {!response && !loading && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {EXAMPLE_QUERIES.map((q) => (
            <button
              key={q}
              onClick={() => { setQuery(q); submit(q); }}
              className="text-left rounded-lg border border-white/8 bg-white/3 p-3 text-xs text-slate-400 hover:bg-white/6 hover:text-white hover:border-brand-500/30 transition-all"
            >
              {q}
            </button>
          ))}
        </div>
      )}

      {/* Loading skeleton */}
      {loading && (
        <div className="space-y-3">
          {isColdStart ? (
            <div className="flex items-start gap-3 rounded-lg border border-yellow-500/30 bg-yellow-500/8 p-4">
              <Loader2 className="h-4 w-4 text-yellow-400 animate-spin mt-0.5 shrink-0" />
              <div>
                <p className="text-sm font-medium text-yellow-400">Backend is waking up…</p>
                <p className="text-xs text-yellow-300/70 mt-1">
                  The Render free tier spins down after inactivity. First request takes ~30s.
                  Hang tight — this only happens once per session.
                </p>
              </div>
            </div>
          ) : (
            <div className="space-y-3 animate-pulse">
              <div className="h-4 rounded bg-white/8 w-3/4" />
              <div className="h-4 rounded bg-white/8 w-full" />
              <div className="h-4 rounded bg-white/8 w-5/6" />
              <div className="h-4 rounded bg-white/8 w-2/3" />
            </div>
          )}
        </div>
      )}

      {/* Answer */}
      {response && !loading && (
        <div ref={answerRef} className="space-y-3">
          {/* Guard + meta row */}
          <div className="flex flex-wrap items-center gap-2">
            <GuardrailBadge
              triggered={response.guardrail_triggered}
              message={response.guardrail_message ?? undefined}
              latencyMs={response.latency_ms}
            />
            <div className="flex items-center gap-1.5 text-xs text-slate-500 bg-white/4 rounded-lg px-2.5 py-1.5 border border-white/8">
              <Clock className="h-3 w-3" />
              {formatLatency(response.latency_ms)}
            </div>
            <div className="flex items-center gap-1.5 text-xs text-slate-500 bg-white/4 rounded-lg px-2.5 py-1.5 border border-white/8">
              <Database className="h-3 w-3" />
              {response.sources.length} sources
            </div>
          </div>

          {/* Answer text */}
          {!response.guardrail_triggered && (
            <div className="rounded-xl border border-brand-600/20 bg-brand-600/5 p-4">
              <div className="flex items-center gap-2 mb-3">
                <Cpu className="h-4 w-4 text-brand-400" />
                <span className="text-xs font-semibold text-brand-400 uppercase tracking-wide">
                  GPT-4o-mini · Hybrid Search · Reranked
                </span>
              </div>
              <div
                className="answer-text text-sm text-slate-200 leading-relaxed"
                dangerouslySetInnerHTML={{
                  __html: `<p>${formatAnswer(response.answer)}</p>`,
                }}
              />
            </div>
          )}

          {/* Sources */}
          {response.sources.length > 0 && (
            <div>
              <button
                onClick={() => setShowSources((s) => !s)}
                className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-300 transition-colors mb-2"
              >
                {showSources ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                Retrieved Sources ({response.sources.length})
              </button>
              {showSources && (
                <div className="space-y-2">
                  {response.sources.map((doc, i) => (
                    <SourceCard key={i} doc={doc} index={i} />
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-4 space-y-2">
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 text-red-400 shrink-0" />
            <p className="text-sm text-red-400">{error}</p>
          </div>
          {error.includes("waking up") || error.includes("fetch") ? (
            <button
              onClick={() => submit(lastQuery.current)}
              className="flex items-center gap-1.5 text-xs text-red-300 hover:text-white transition-colors"
            >
              <RefreshCw className="h-3 w-3" /> Retry
            </button>
          ) : null}
        </div>
      )}

      {/* Input area */}
      <div className="mt-auto flex flex-col gap-2">
        <Textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); }
          }}
          placeholder="Ask anything about your knowledge base… (Enter to send, Shift+Enter for newline)"
          className="min-h-[72px] resize-none"
          disabled={loading}
        />
        <div className="flex items-center justify-between">
          {response && (
            <button
              onClick={() => { setResponse(null); setQuery(""); }}
              className="text-xs text-slate-600 hover:text-slate-400 transition-colors"
            >
              Clear
            </button>
          )}
          <Button
            onClick={() => submit()}
            disabled={loading || !query.trim()}
            className="ml-auto gap-2"
          >
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            {loading ? "Thinking…" : "Ask Nexus"}
          </Button>
        </div>
      </div>
    </div>
  );
}
