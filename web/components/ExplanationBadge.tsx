'use client';

import React, { useState } from 'react';
import { Sparkles, Info, BrainCircuit } from 'lucide-react';

interface ExplanationBadgeProps {
  explanation?: string | null;
  source?: string;
}

export default function ExplanationBadge({ explanation, source }: ExplanationBadgeProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!explanation) {
    return (
      <div 
        className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium bg-slate-800/80 text-slate-400 border border-slate-700/60"
        title="Explanation is generating asynchronously in the background and will be cached in Redis."
      >
        <Sparkles className="w-3 h-3 text-slate-500 animate-pulse" />
        <span>AI Reasoning Async</span>
      </div>
    );
  }

  return (
    <div className="relative inline-block">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-gradient-to-r from-sky-500/15 via-indigo-500/15 to-purple-500/15 text-sky-300 border border-sky-500/30 hover:border-sky-400 hover:bg-sky-500/25 transition-all shadow-sm group text-left"
        title="Click to toggle LLM explanation rationale"
      >
        <Sparkles className="w-3 h-3 text-sky-400 group-hover:scale-110 transition-transform" />
        <span className="truncate max-w-[200px] sm:max-w-[260px]">
          {explanation}
        </span>
        <Info className="w-2.5 h-2.5 text-sky-400/70" />
      </button>

      {/* Expanded Tooltip / Flyout Modal */}
      {isExpanded && (
        <div className="absolute left-0 bottom-full mb-2 w-72 p-3 rounded-xl bg-slate-900/95 border border-sky-500/40 shadow-2xl backdrop-blur-md z-40 animate-fadeIn">
          <div className="flex items-center gap-1.5 text-xs font-bold text-sky-300 mb-1">
            <BrainCircuit className="w-3.5 h-3.5 text-sky-400" />
            <span>LLM "Why This" Explanation</span>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            "{explanation}"
          </p>
          <div className="mt-2 pt-2 border-t border-slate-800 flex items-center justify-between text-[10px] text-slate-400">
            <span>Model: Gemini 3.5 Flash-Lite</span>
            <span className="text-sky-400 font-mono">Cached in Redis</span>
          </div>
        </div>
      )}
    </div>
  );
}
