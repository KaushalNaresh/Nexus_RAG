"use client";

import { useEffect, useState } from "react";
import { checkHealth } from "@/lib/api";
import { Loader2, Wifi, WifiOff } from "lucide-react";

type Status = "checking" | "online" | "offline";

export default function BackendPinger() {
  const [status, setStatus] = useState<Status>("checking");

  useEffect(() => {
    let cancelled = false;

    const ping = async () => {
      try {
        await checkHealth();
        if (!cancelled) setStatus("online");
      } catch {
        if (!cancelled) setStatus("offline");
      }
    };

    ping();
    return () => { cancelled = true; };
  }, []);

  if (status === "checking") {
    return (
      <div className="flex items-center gap-1.5 text-xs text-slate-500 bg-white/3 border border-white/8 rounded-lg px-3 py-1.5 mb-4">
        <Loader2 className="h-3 w-3 animate-spin" />
        Waking up backend…
      </div>
    );
  }

  if (status === "offline") {
    return (
      <div className="flex items-center gap-1.5 text-xs text-yellow-500 bg-yellow-500/8 border border-yellow-500/20 rounded-lg px-3 py-1.5 mb-4">
        <WifiOff className="h-3 w-3" />
        Backend is starting up (free tier cold start — ~30s). First query may be slow.
      </div>
    );
  }

  return (
    <div className="flex items-center gap-1.5 text-xs text-green-500 bg-green-500/8 border border-green-500/20 rounded-lg px-3 py-1.5 mb-4">
      <Wifi className="h-3 w-3" />
      Backend online
    </div>
  );
}
