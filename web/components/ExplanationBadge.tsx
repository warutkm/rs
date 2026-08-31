'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Sparkles, Info, BrainCircuit } from 'lucide-react';

interface ExplanationBadgeProps {
  explanation?: string | null;
  source?: string;
}

export default function ExplanationBadge({ explanation, source }: ExplanationBadgeProps) {
  const [isOpen, setIsOpen] = useState(false);
  const badgeRef = useRef<HTMLDivElement>(null);

  // Close popup on outside click
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (badgeRef.current && !badgeRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen]);

  // If no explanation is provided yet, derive a clean heuristic reason
  const effectiveExplanation = explanation && explanation.trim().length > 0
    ? explanation
    : source === 'content_cold_start'
    ? 'Top candidate matching your category preference and catalog popularity.'
    : source === 'popular_baseline'
    ? 'Trending item with high average rating and helpful review velocity.'
    : 'Top candidate ranked by LambdaMART from collaborative and content signals.';

  const isAsyncFallback = !explanation;

  return (
    <div className="relative w-full max-w-full" ref={badgeRef}>
      {/* Clickable Pill (Matches Score button behavior) */}
      <button
        type="button"
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setIsOpen(!isOpen);
        }}
        className={`w-full max-w-full flex items-center justify-between gap-1.5 px-2.5 py-1 rounded-lg text-[10px] font-medium transition-all shadow-sm overflow-hidden text-left border ${
          isOpen
            ? 'bg-sky-950/80 border-sky-400 text-sky-200 ring-1 ring-sky-500/40'
            : 'bg-slate-900/90 hover:bg-slate-800 border-sky-500/30 hover:border-sky-400 text-sky-300'
        }`}
        title="Click to toggle AI recommendation reasoning popup"
      >
        <div className="flex items-center gap-1.5 min-w-0 flex-1 overflow-hidden">
          <Sparkles className="w-3 h-3 text-sky-400 shrink-0" />
          <span className="truncate flex-1 min-w-0 font-medium">
            {effectiveExplanation}
          </span>
        </div>
        <Info className="w-3 h-3 text-sky-400/80 shrink-0 ml-1" />
      </button>

      {/* Floating Popup on Card (Identical pattern to Score Feature Signals popup) */}
      {isOpen && (
        <div
          onClick={(e) => e.stopPropagation()}
          className="absolute left-0 bottom-full mb-2 w-64 p-3.5 rounded-xl bg-[#070d19] border border-sky-500/60 shadow-2xl shadow-black z-50 animate-fadeIn text-slate-200"
        >
          <div className="flex items-center justify-between pb-1.5 mb-2 border-b border-slate-800">
            <div className="flex items-center gap-1.5 text-xs font-bold text-sky-300">
              <BrainCircuit className="w-3.5 h-3.5 text-sky-400" />
              <span>LLM "Why This" Reason</span>
            </div>
            <span className="text-[9px] font-mono text-emerald-400 font-semibold">
              {isAsyncFallback ? 'Fast Attribution' : 'Cached in Redis'}
            </span>
          </div>

          <p className="text-[11px] text-slate-200 leading-relaxed font-normal">
            "{effectiveExplanation}"
          </p>

          <div className="mt-2.5 pt-1.5 border-t border-slate-800 flex items-center justify-between text-[9px] text-slate-400">
            <span>Model: Gemini 3.5 Flash-Lite</span>
            <span className="text-sky-400 font-medium cursor-pointer" onClick={() => setIsOpen(false)}>
              Click to close
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
