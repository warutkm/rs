'use client';

import React from 'react';
import { Layers, Activity, Zap } from 'lucide-react';

interface ScoreBreakdownProps {
  score?: number;
  embScore?: number;
  bm25Score?: number;
  source?: string;
}

export default function ScoreBreakdown({
  score,
  embScore,
  bm25Score,
  source,
}: ScoreBreakdownProps) {
  // If we have hybrid search score breakdown
  if (embScore !== undefined && bm25Score !== undefined) {
    const embPct = Math.min(100, Math.max(0, embScore * 100));
    const bm25Pct = Math.min(100, Math.max(0, bm25Score * 100));
    const hybridPct = Math.min(100, Math.max(0, (score || 0) * 100));

    return (
      <div className="space-y-1.5 p-2.5 rounded-lg bg-slate-900/80 border border-slate-800 text-[11px]">
        <div className="flex items-center justify-between text-slate-300 font-medium">
          <span className="flex items-center gap-1">
            <Zap className="w-3 h-3 text-amber-400" />
            Hybrid Score:
          </span>
          <span className="font-mono text-sky-400 font-bold">{(score || 0).toFixed(4)}</span>
        </div>
        
        <div className="space-y-1 pt-1">
          <div>
            <div className="flex justify-between text-[10px] text-slate-400">
              <span>e5 Semantic Vector:</span>
              <span className="font-mono">{embScore.toFixed(3)}</span>
            </div>
            <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-sky-500 to-indigo-500 rounded-full"
                style={{ width: `${embPct}%` }}
              />
            </div>
          </div>

          <div>
            <div className="flex justify-between text-[10px] text-slate-400">
              <span>BM25 Lexical Keyword:</span>
              <span className="font-mono">{bm25Score.toFixed(3)}</span>
            </div>
            <div className="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-amber-500 to-rose-500 rounded-full"
                style={{ width: `${bm25Pct}%` }}
              />
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Single ranker score representation
  if (score !== undefined) {
    const formattedScore = score.toFixed(4);
    return (
      <div className="flex items-center justify-between text-[11px] px-2 py-1 rounded bg-slate-900/60 border border-slate-800">
        <span className="text-slate-400 flex items-center gap-1">
          <Activity className="w-3 h-3 text-sky-400" />
          {source ? source.replace(/_/g, ' ') : 'Ranker Score'}:
        </span>
        <span className="font-mono font-bold text-sky-300">{formattedScore}</span>
      </div>
    );
  }

  return null;
}
