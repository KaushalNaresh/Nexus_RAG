import EvalReport from "@/components/EvalReport";
import { BarChart3 } from "lucide-react";

export const metadata = {
  title: "Evals — Nexus RAG + Evals",
  description: "Ragas evaluation results: faithfulness, answer relevancy, context precision, and recall",
};

export default function EvalsPage() {
  return (
    <div className="space-y-6 max-w-3xl mx-auto">
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 mb-1">
          <BarChart3 className="h-5 w-5 text-brand-400" />
          <h1 className="text-2xl font-bold text-white tracking-tight">
            Ragas Evaluation
          </h1>
        </div>
        <p className="text-sm text-slate-400">
          Objective metrics measuring retrieval quality and answer groundedness across 8 golden QA pairs.
          Static data — no API calls needed.
        </p>
      </div>

      <EvalReport />
    </div>
  );
}
