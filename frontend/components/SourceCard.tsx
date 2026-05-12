import { ExternalLink, FileText } from "lucide-react";
import { truncate, scoreBarWidth } from "@/lib/utils";
import type { SourceDocument } from "@/lib/api";

interface Props {
  doc: SourceDocument;
  index: number;
  variant?: "default" | "naive" | "production";
}

export default function SourceCard({ doc, index, variant = "default" }: Props) {
  const score = doc.score ?? 0;
  const barPct = scoreBarWidth(score);
  const isNaive = variant === "naive";

  // For naive, scores are cosine similarities (0–1); for production they are cross-encoder logits
  const scoreLabel = isNaive
    ? score.toFixed(3)
    : score > 0
    ? `+${score.toFixed(2)}`
    : score.toFixed(2);

  const hostname = (() => {
    try {
      return doc.source ? new URL(doc.source).hostname : null;
    } catch {
      return null;
    }
  })();

  return (
    <div className="rounded-lg border border-white/8 bg-white/3 p-3 hover:bg-white/5 transition-colors">
      <div className="flex items-start justify-between gap-2 mb-2">
        <div className="flex items-center gap-1.5 min-w-0">
          <FileText className="h-3.5 w-3.5 text-slate-500 shrink-0" />
          <span className="text-xs font-medium text-brand-400">
            [{index + 1}]
          </span>
          {hostname && (
            <span className="text-xs text-slate-500 truncate">{hostname}</span>
          )}
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          <span
            className={`text-xs font-mono font-semibold ${
              isNaive ? "text-yellow-400" : score > 3 ? "text-green-400" : score > 0 ? "text-yellow-400" : "text-red-400"
            }`}
          >
            {scoreLabel}
          </span>
          {doc.source && (
            <a
              href={doc.source}
              target="_blank"
              rel="noreferrer"
              className="text-slate-600 hover:text-brand-400 transition-colors"
            >
              <ExternalLink className="h-3 w-3" />
            </a>
          )}
        </div>
      </div>

      {/* Score bar */}
      <div className="mb-2 h-1 w-full rounded-full bg-white/6 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ${
            isNaive ? "bg-yellow-500" : score > 3 ? "bg-green-500" : score > 0 ? "bg-yellow-500" : "bg-red-500"
          }`}
          style={{ width: `${barPct}%` }}
        />
      </div>

      <p className="text-xs text-slate-400 leading-relaxed">
        {truncate(doc.content, 180)}
      </p>
    </div>
  );
}
