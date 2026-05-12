"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { Layers, GitCompare, BarChart3, Zap } from "lucide-react";

const links = [
  { href: "/", label: "Chat", icon: Zap },
  { href: "/compare", label: "Compare", icon: GitCompare },
  { href: "/evals", label: "Evals", icon: BarChart3 },
];

export default function Navbar() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 w-full border-b border-white/8 bg-[#0a0a12]/90 backdrop-blur-md">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4 sm:px-6">
        {/* Brand */}
        <Link href="/" className="flex items-center gap-2.5 group">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-brand-600 shadow-lg shadow-brand-900/50 group-hover:bg-brand-500 transition-colors">
            <Layers className="h-4 w-4 text-white" />
          </div>
          <div className="leading-tight">
            <span className="text-sm font-bold text-white tracking-tight">
              Nexus RAG
            </span>
            <span className="ml-1.5 text-xs text-brand-400 font-medium hidden sm:inline">
              + Evals
            </span>
          </div>
        </Link>

        {/* Nav links */}
        <nav className="flex items-center gap-1">
          {links.map(({ href, label, icon: Icon }) => {
            const active =
              href === "/" ? pathname === "/" : pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                className={cn(
                  "flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium transition-all",
                  active
                    ? "bg-brand-600/20 text-brand-400 border border-brand-600/30"
                    : "text-slate-400 hover:text-white hover:bg-white/5"
                )}
              >
                <Icon className="h-3.5 w-3.5" />
                {label}
              </Link>
            );
          })}
        </nav>

        {/* Backend status pill */}
        <div className="hidden sm:flex items-center gap-1.5 text-xs text-slate-500">
          <span className="h-1.5 w-1.5 rounded-full bg-green-400 animate-pulse" />
          FastAPI + GPT-4o-mini
        </div>
      </div>
    </header>
  );
}
