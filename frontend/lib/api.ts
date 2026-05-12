// Typed API client — mirrors FastAPI response models exactly

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface SourceDocument {
  content: string;
  source?: string;
  score?: number;
  metadata: Record<string, unknown>;
}

export interface QueryResponse {
  answer: string;
  sources: SourceDocument[];
  session_id?: string;
  latency_ms?: number;
  guardrail_triggered: boolean;
  guardrail_message?: string;
  timestamp: string;
}

export interface IngestResponse {
  success: boolean;
  chunks_indexed: number;
  source: string;
  message: string;
  timestamp: string;
}

export interface CompareResponse {
  naive: QueryResponse;
  production: QueryResponse;
}

async function request<T>(
  path: string,
  init: RequestInit,
  timeoutMs = 90_000   // 90s — generous for cold start + model load + LLM call
): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(`${API_BASE}${path}`, {
      ...init,
      signal: controller.signal,
      headers: { Accept: "application/json", ...(init.headers ?? {}) },
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`API ${res.status}: ${text}`);
    }
    return res.json() as Promise<T>;
  } catch (e: unknown) {
    if (e instanceof DOMException && e.name === "AbortError") {
      throw new Error("Request timed out. The backend may still be waking up — please retry in 10s.");
    }
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

export async function ingestFile(file: File): Promise<IngestResponse> {
  const form = new FormData();
  form.append("file", file);
  return request<IngestResponse>("/api/v1/ingest/file", {
    method: "POST",
    body: form,
  });
}

export async function ingestUrl(url: string): Promise<IngestResponse> {
  return request<IngestResponse>("/api/v1/ingest/url", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
}

export async function queryRag(
  query: string,
  options?: { top_k?: number; alpha?: number }
): Promise<QueryResponse> {
  return request<QueryResponse>("/api/v1/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, ...options }),
  });
}

export async function compareRag(query: string): Promise<CompareResponse> {
  return request<CompareResponse>("/api/v1/query/compare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
}

export async function checkHealth(): Promise<{ status: string }> {
  return request<{ status: string }>("/health", { method: "GET" });
}
