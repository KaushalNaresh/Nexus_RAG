"use client";

import { useState, useRef, useCallback } from "react";
import { Upload, Link2, CheckCircle2, Loader2, X, FileText } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ingestFile, ingestUrl } from "@/lib/api";

interface UploadResult {
  chunks: number;
  source: string;
}

export default function DocumentUploader() {
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<UploadResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [url, setUrl] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  const reset = () => {
    setResult(null);
    setError(null);
  };

  const handleFile = useCallback(async (file: File) => {
    if (!file.name.endsWith(".pdf") && !file.name.endsWith(".txt") && !file.name.endsWith(".md")) {
      setError("Only PDF, TXT, and Markdown files are supported.");
      return;
    }
    reset();
    setLoading(true);
    try {
      const res = await ingestFile(file);
      setResult({ chunks: res.chunks_indexed, source: res.source });
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setLoading(false);
    }
  }, []);

  const handleUrl = async () => {
    if (!url.trim()) return;
    reset();
    setLoading(true);
    try {
      const res = await ingestUrl(url.trim());
      setResult({ chunks: res.chunks_indexed, source: res.source });
      setUrl("");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Ingestion failed");
    } finally {
      setLoading(false);
    }
  };

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  return (
    <div className="space-y-3">
      <h2 className="text-xs font-semibold uppercase tracking-widest text-slate-500 px-1">
        Knowledge Base
      </h2>

      {/* Drop zone */}
      <div
        onClick={() => fileRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={onDrop}
        className={`
          relative flex flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed
          p-5 cursor-pointer transition-all duration-200
          ${isDragging
            ? "border-brand-500 bg-brand-600/10"
            : "border-white/15 bg-white/3 hover:border-brand-500/50 hover:bg-white/5"
          }
        `}
      >
        <input
          ref={fileRef}
          type="file"
          accept=".pdf,.txt,.md"
          className="hidden"
          onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
        />
        {loading ? (
          <Loader2 className="h-6 w-6 text-brand-400 animate-spin" />
        ) : (
          <Upload className="h-6 w-6 text-slate-500" />
        )}
        <div className="text-center">
          <p className="text-sm text-slate-400">
            {loading ? "Indexing document…" : "Drop a PDF or click to upload"}
          </p>
          <p className="text-xs text-slate-600 mt-0.5">.pdf · .txt · .md</p>
        </div>
      </div>

      {/* URL input */}
      <div className="flex gap-2">
        <Input
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleUrl()}
          placeholder="https://example.com/article"
          className="flex-1 text-xs"
          disabled={loading}
        />
        <Button
          size="sm"
          variant="secondary"
          onClick={handleUrl}
          disabled={loading || !url.trim()}
          className="shrink-0"
        >
          {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Link2 className="h-3.5 w-3.5" />}
        </Button>
      </div>

      {/* Result / Error */}
      {result && (
        <div className="flex items-start gap-2 rounded-lg border border-green-500/25 bg-green-500/8 p-3">
          <CheckCircle2 className="h-4 w-4 text-green-400 mt-0.5 shrink-0" />
          <div className="flex-1 min-w-0">
            <p className="text-xs font-medium text-green-400">
              {result.chunks} chunks indexed
            </p>
            <p className="text-xs text-slate-500 truncate mt-0.5">{result.source}</p>
          </div>
          <button onClick={reset} className="text-slate-600 hover:text-slate-400">
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      )}
      {error && (
        <div className="flex items-start gap-2 rounded-lg border border-red-500/25 bg-red-500/8 p-3">
          <X className="h-4 w-4 text-red-400 mt-0.5 shrink-0" />
          <div className="flex-1 min-w-0">
            <p className="text-xs font-medium text-red-400">Error</p>
            <p className="text-xs text-red-300/70 mt-0.5">{error}</p>
          </div>
          <button onClick={reset} className="text-slate-600 hover:text-slate-400">
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      )}

      {/* Divider + tip */}
      <div className="border-t border-white/8 pt-3">
        <p className="text-xs text-slate-600 flex items-center gap-1.5">
          <FileText className="h-3 w-3" />
          Pre-loaded: RAG Wikipedia, BM25, Cross-Encoders, Ragas docs
        </p>
      </div>
    </div>
  );
}
