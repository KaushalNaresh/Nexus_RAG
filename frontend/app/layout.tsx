import type { Metadata } from "next";
import "./globals.css";
import Navbar from "@/components/Navbar";

export const metadata: Metadata = {
  title: "Nexus RAG + Evals",
  description:
    "Production-grade RAG system with hybrid search, cross-encoder reranking, dual-layer guardrails, and Ragas evaluation.",
  openGraph: {
    title: "Nexus RAG + Evals",
    description: "Production-grade RAG with hybrid search, reranking & guardrails",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased">
        <Navbar />
        <main className="mx-auto max-w-7xl px-4 py-6 sm:px-6">
          {children}
        </main>
        <footer className="mt-16 border-t border-white/8 py-6 text-center text-xs text-slate-600">
          Nexus RAG · FastAPI + LangChain + Pinecone + GPT-4o-mini · Built for portfolio
        </footer>
      </body>
    </html>
  );
}
