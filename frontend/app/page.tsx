import DocumentUploader from "@/components/DocumentUploader";
import ChatInterface from "@/components/ChatInterface";
import { Card, CardContent } from "@/components/ui/card";
import { Layers, Shield, Zap, Search } from "lucide-react";

const features = [
  { icon: Search, label: "Hybrid Search", desc: "Dense + BM25 sparse" },
  { icon: Zap, label: "Cross-Encoder", desc: "MS-MARCO reranker" },
  { icon: Shield, label: "Dual Guardrails", desc: "NeMo + Guardrails AI" },
  { icon: Layers, label: "Ragas Evals", desc: "Faithfulness · Recall" },
];

export default function ChatPage() {
  return (
    <div className="space-y-6">
      {/* Hero */}
      <div className="text-center space-y-2 pb-2">
        <h1 className="text-2xl sm:text-3xl font-bold text-white tracking-tight">
          Nexus RAG{" "}
          <span className="text-brand-400">+ Evals</span>
        </h1>
        <p className="text-sm text-slate-400 max-w-xl mx-auto">
          Production-grade Retrieval-Augmented Generation with hybrid search, cross-encoder
          reranking, dual-layer guardrails, and Ragas evaluation.
        </p>
        {/* Feature chips */}
        <div className="flex flex-wrap justify-center gap-2 pt-1">
          {features.map(({ icon: Icon, label, desc }) => (
            <div
              key={label}
              className="flex items-center gap-1.5 rounded-full border border-white/10 bg-white/4 px-3 py-1 text-xs text-slate-400"
            >
              <Icon className="h-3 w-3 text-brand-400" />
              <span className="font-medium text-white">{label}</span>
              <span className="text-slate-500">· {desc}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Two-panel layout */}
      <div className="grid grid-cols-1 lg:grid-cols-[280px_1fr] gap-4">
        {/* Left — uploader */}
        <Card>
          <CardContent className="pt-5">
            <DocumentUploader />
          </CardContent>
        </Card>

        {/* Right — chat */}
        <Card className="min-h-[520px]">
          <CardContent className="pt-5 flex flex-col h-full">
            <ChatInterface />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
