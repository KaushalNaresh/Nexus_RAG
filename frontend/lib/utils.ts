import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatLatency(ms?: number): string {
  if (ms === undefined || ms === null) return "—";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

export function truncate(str: string, n: number): string {
  return str.length > n ? str.slice(0, n) + "…" : str;
}

export function scoreColor(score: number): string {
  if (score > 5) return "text-green-400";
  if (score > 0) return "text-yellow-400";
  return "text-red-400";
}

export function scoreBarWidth(score: number): number {
  // Cross-encoder scores roughly range from -12 to +12
  // Normalize to 0-100 for display
  const normalized = (score + 12) / 24;
  return Math.min(100, Math.max(2, Math.round(normalized * 100)));
}
