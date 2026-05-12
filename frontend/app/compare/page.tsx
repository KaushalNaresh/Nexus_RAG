import ComparisonView from "@/components/ComparisonView";
import { GitCompare, Info } from "lucide-react";

export const metadata = {
  title: "Compare — Nexus RAG + Evals",
  description: "Side-by-side comparison of naive vs. Nexus RAG architecture",
};

export default function ComparePage() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <GitCompare className="h-5 w-5 text-brand-400" />
            <h1 className="text-2xl font-bold text-white tracking-tight">
              Naive RAG vs. Nexus RAG
            </h1>
          </div>
          <p className="text-sm text-slate-400">
            One query, two pipelines. See what hybrid search, reranking, and guardrails actually change.
          </p>
        </div>

        {/* Architecture badges */}
        <div className="flex flex-col gap-2 shrink-0">
          <div className="flex items-center gap-2 rounded-lg border border-yellow-500/20 bg-yellow-500/5 px-3 py-2">
            <div className="h-2 w-2 rounded-full bg-yellow-400" />
            <span className="text-xs text-yellow-300 font-medium">Naive</span>
            <span className="text-xs text-slate-500">dense · no rerank · no guard</span>
          </div>
          <div className="flex items-center gap-2 rounded-lg border border-brand-500/20 bg-brand-500/5 px-3 py-2">
            <div className="h-2 w-2 rounded-full bg-brand-400" />
            <span className="text-xs text-brand-300 font-medium">Nexus RAG</span>
            <span className="text-xs text-slate-500">hybrid · reranked · guarded</span>
          </div>
        </div>
      </div>

      {/* Info banner */}
      <div className="flex items-start gap-2 rounded-lg border border-white/8 bg-white/3 p-3">
        <Info className="h-4 w-4 text-slate-500 shrink-0 mt-0.5" />
        <p className="text-xs text-slate-500">
          Both pipelines use the same LLM (GPT-4o-mini) and the same Pinecone index.
          The difference is purely in retrieval strategy and safety layers.
          The compare endpoint runs both in a single atomic backend call.
        </p>
      </div>

      <ComparisonView />
    </div>
  );
}
