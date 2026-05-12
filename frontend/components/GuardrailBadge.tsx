import { ShieldCheck, ShieldX, ShieldAlert } from "lucide-react";

interface Props {
  triggered: boolean;
  message?: string;
  latencyMs?: number;
}

export default function GuardrailBadge({ triggered, message, latencyMs }: Props) {
  if (triggered) {
    return (
      <div className="flex items-start gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2">
        <ShieldX className="h-4 w-4 text-red-400 mt-0.5 shrink-0" />
        <div>
          <p className="text-xs font-semibold text-red-400 uppercase tracking-wide">
            Guardrail Blocked
          </p>
          {message && (
            <p className="text-xs text-red-300/80 mt-0.5">{message}</p>
          )}
          {latencyMs !== undefined && latencyMs < 5 && (
            <p className="text-xs text-red-300/60 mt-0.5">
              Regex pre-check in {latencyMs.toFixed(1)}ms
            </p>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2 rounded-lg border border-green-500/25 bg-green-500/8 px-3 py-2">
      <ShieldCheck className="h-4 w-4 text-green-400 shrink-0" />
      <div>
        <p className="text-xs font-semibold text-green-400 uppercase tracking-wide">
          Guardrails Passed
        </p>
        <p className="text-xs text-green-300/60">
          Input + Output validated
        </p>
      </div>
    </div>
  );
}

export function GuardrailMiniTag({ triggered }: { triggered: boolean }) {
  if (triggered) {
    return (
      <span className="inline-flex items-center gap-1 text-xs font-medium text-red-400 bg-red-500/10 border border-red-500/25 rounded px-1.5 py-0.5">
        <ShieldX className="h-3 w-3" /> Blocked
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 text-xs font-medium text-green-400 bg-green-500/10 border border-green-500/25 rounded px-1.5 py-0.5">
      <ShieldAlert className="h-3 w-3" /> Passed
    </span>
  );
}
