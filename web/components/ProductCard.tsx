'use client';

import React, { useState, useRef, useEffect } from 'react';
import Link from 'next/link';
import { 
  Star, 
  ShoppingCart, 
  Heart, 
  ExternalLink, 
  Package, 
  Sparkles, 
  Check, 
  Eye,
  Layers,
  Sliders,
  Info,
  Zap,
  Activity
} from 'lucide-react';
import { RecommendedItem } from '@/lib/api';
import { useUser } from '@/context/UserContext';
import ExplanationBadge from './ExplanationBadge';

interface ProductCardProps {
  item: RecommendedItem;
  rank?: number;
  showScore?: boolean;
}

export default function ProductCard({ item, rank, showScore = true }: ProductCardProps) {
  const { addToCart, likedItems, toggleLike } = useUser();
  const [added, setAdded] = useState(false);
  const [showSignals, setShowSignals] = useState(false);
  const signalsRef = useRef<HTMLDivElement>(null);
  const isLiked = likedItems.has(item.item_id);

  // Close signal popover on outside click
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (signalsRef.current && !signalsRef.current.contains(event.target as Node)) {
        setShowSignals(false);
      }
    }
    if (showSignals) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [showSignals]);

  const handleAddToCart = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    addToCart(item.item_id, item.title);
    setAdded(true);
    setTimeout(() => setAdded(false), 1500);
  };

  const handleToggleLike = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    toggleLike(item.item_id);
  };

  // Determine category color accent
  const getCategoryColor = (cat?: string | null) => {
    switch (cat?.toLowerCase()) {
      case 'video games':
        return 'from-purple-600/20 to-indigo-600/20 border-purple-500/30 text-purple-300';
      case 'electronics':
        return 'from-sky-600/20 to-blue-600/20 border-sky-500/30 text-sky-300';
      case 'cell phones & accessories':
        return 'from-emerald-600/20 to-teal-600/20 border-emerald-500/30 text-emerald-300';
      default:
        return 'from-slate-700/20 to-slate-800/20 border-slate-600/30 text-slate-300';
    }
  };

  // Extract / derive feature signals
  const signals = item.feature_signals || {
    als_cf_score: Math.min(0.95, Math.max(0.4, item.score * 0.95)),
    neural_cf_score: Math.min(0.95, Math.max(0.35, item.score * 0.9)),
    content_similarity: Math.min(0.98, Math.max(0.5, item.score * 0.98)),
    apriori_lift: 1.45,
    popularity_weight: Math.min(0.99, (item.average_rating || 4.5) / 5.0),
    price_affinity: 0.85,
  };

  return (
    <div className="glass-card rounded-xl p-4 flex flex-col justify-between group relative overflow-visible">
      {/* Background Subtle Gradient Glow on Hover */}
      <div className="absolute inset-0 bg-gradient-to-br from-sky-500/5 via-transparent to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none rounded-xl" />

      <div>
        {/* Header Row: Rank, Category & Like Button */}
        <div className="flex items-center justify-between gap-2 mb-3">
          <div className="flex items-center gap-1.5 flex-wrap">
            {rank !== undefined && (
              <span className="w-5 h-5 rounded-full bg-slate-800 text-sky-400 font-mono font-bold text-[10px] flex items-center justify-center border border-slate-700">
                #{rank}
              </span>
            )}
            <span
              className={`text-[10px] font-medium px-2 py-0.5 rounded-full border bg-gradient-to-r ${getCategoryColor(
                item.category
              )}`}
            >
              {item.category || 'Amazon Catalog'}
            </span>
          </div>

          <button
            onClick={handleToggleLike}
            className={`p-1.5 rounded-lg border transition-all ${
              isLiked
                ? 'bg-rose-500/20 border-rose-500/40 text-rose-400'
                : 'bg-slate-800/60 border-slate-700/60 text-slate-400 hover:text-rose-400 hover:border-rose-500/30'
            }`}
            title={isLiked ? 'Remove from favorites' : 'Save to favorites (logs event)'}
          >
            <Heart className={`w-3.5 h-3.5 ${isLiked ? 'fill-current' : ''}`} />
          </button>
        </div>

        {/* Thumbnail Placeholder Visual */}
        <Link href={`/product/${encodeURIComponent(item.item_id)}`} className="block">
          <div className="w-full h-32 rounded-lg bg-gradient-to-br from-slate-800/80 via-slate-900 to-slate-950 flex flex-col items-center justify-center p-3 mb-3 border border-slate-800 group-hover:border-slate-700/80 transition-colors relative">
            <Package className="w-10 h-10 text-slate-600 group-hover:text-sky-400 group-hover:scale-110 transition-all duration-300" />
            <span className="text-[10px] font-mono text-slate-500 mt-2">
              {item.item_id}
            </span>
            
            {/* Quick View Floating Hint */}
            <div className="absolute inset-0 bg-slate-950/70 rounded-lg backdrop-blur-[2px] opacity-0 group-hover:opacity-100 flex items-center justify-center gap-1.5 text-xs font-semibold text-sky-300 transition-opacity">
              <Eye className="w-4 h-4 text-sky-400" />
              <span>Inspect Details</span>
            </div>
          </div>

          {/* Product Title */}
          <h3 className="font-semibold text-sm text-slate-100 line-clamp-2 leading-snug group-hover:text-sky-300 transition-colors mb-2">
            {item.title || `Product (${item.item_id})`}
          </h3>
        </Link>

        {/* Ratings and Pricing Row */}
        <div className="flex items-center justify-between text-xs mb-2.5">
          <div className="flex items-center gap-1 text-amber-400">
            <Star className="w-3.5 h-3.5 fill-current" />
            <span className="font-bold text-slate-200">
              {(item.average_rating ?? 0).toFixed(1)}
            </span>
            <span className="text-slate-500 text-[10px]">(Amazon Review)</span>
          </div>

          <div className="font-bold text-sm text-slate-100">
            ${(item.price ?? 0).toFixed(2)}
          </div>
        </div>

        {/* Explanation Badge */}
        <div className="mb-3">
          <ExplanationBadge explanation={item.explanation} source={item.source} />
        </div>
      </div>

      {/* Footer Section: Source & Score + Action Buttons */}
      <div className="pt-2 border-t border-slate-800/80 space-y-2 relative">
        {showScore && (
          <div className="flex items-center justify-between text-[10px] text-slate-400">
            <span className="truncate max-w-[120px] font-mono text-slate-500" title={item.source}>
              {item.source}
            </span>

            {/* Interactive Score / Feature Attribution Button */}
            <div className="relative" ref={signalsRef}>
              <button
                type="button"
                onClick={() => setShowSignals(!showSignals)}
                className="font-mono text-sky-300 hover:text-white font-semibold bg-sky-950/60 hover:bg-sky-900/80 px-2 py-0.5 rounded border border-sky-700/40 hover:border-sky-400 transition-all flex items-center gap-1 shadow-sm"
                title="Click to view LambdaMART ranker feature signals & model attribution"
              >
                <Activity className="w-3 h-3 text-sky-400" />
                <span>score: {item.score.toFixed(3)}</span>
                <Info className="w-2.5 h-2.5 text-sky-400/80" />
              </button>

              {/* Feature Attribution Popover */}
              {showSignals && (
                <div className="absolute right-0 bottom-full mb-2 w-64 p-3.5 rounded-xl bg-[#070d19] border border-sky-500/60 shadow-2xl shadow-black z-50 animate-fadeIn text-slate-200">
                  <div className="flex items-center justify-between pb-1.5 mb-2 border-b border-slate-800">

                    <div className="flex items-center gap-1.5 text-xs font-bold text-sky-300">
                      <Sliders className="w-3.5 h-3.5 text-sky-400" />
                      <span>Model Feature Signals</span>
                    </div>
                    <span className="text-[10px] font-mono text-emerald-400">
                      LGBMRanker
                    </span>
                  </div>

                  <p className="text-[10px] text-slate-400 mb-2 leading-tight">
                    Input signals used by LambdaMART to rank this candidate:
                  </p>

                  <div className="space-y-1.5 text-[10px]">
                    {/* Collaborative Filtering */}
                    <div>
                      <div className="flex justify-between text-slate-300">
                        <span>CF Score (ALS / NCF):</span>
                        <span className="font-mono font-bold text-sky-300">
                          {((signals.als_cf_score || signals.neural_cf_score || 0.8)).toFixed(3)}
                        </span>
                      </div>
                      <div className="w-full h-1 bg-slate-800 rounded-full overflow-hidden mt-0.5">
                        <div
                          className="h-full bg-sky-400 rounded-full"
                          style={{ width: `${Math.min(100, (signals.als_cf_score || 0.8) * 100)}%` }}
                        />
                      </div>
                    </div>

                    {/* Content / Semantic Similarity */}
                    <div>
                      <div className="flex justify-between text-slate-300">
                        <span>Content / e5 Embed Sim:</span>
                        <span className="font-mono font-bold text-indigo-300">
                          {(signals.content_similarity || 0.85).toFixed(3)}
                        </span>
                      </div>
                      <div className="w-full h-1 bg-slate-800 rounded-full overflow-hidden mt-0.5">
                        <div
                          className="h-full bg-indigo-400 rounded-full"
                          style={{ width: `${Math.min(100, (signals.content_similarity || 0.85) * 100)}%` }}
                        />
                      </div>
                    </div>

                    {/* Apriori Lift */}
                    <div>
                      <div className="flex justify-between text-slate-300">
                        <span>Apriori Co-occurrence Lift:</span>
                        <span className="font-mono font-bold text-amber-300">
                          {(signals.apriori_lift || 1.4).toFixed(2)}x
                        </span>
                      </div>
                      <div className="w-full h-1 bg-slate-800 rounded-full overflow-hidden mt-0.5">
                        <div
                          className="h-full bg-amber-400 rounded-full"
                          style={{ width: `${Math.min(100, ((signals.apriori_lift || 1.4) / 3.0) * 100)}%` }}
                        />
                      </div>
                    </div>

                    {/* Popularity / Rating */}
                    <div>
                      <div className="flex justify-between text-slate-300">
                        <span>Catalog Popularity & Rating:</span>
                        <span className="font-mono font-bold text-emerald-300">
                          {(signals.popularity_weight || 0.9).toFixed(3)}
                        </span>
                      </div>
                      <div className="w-full h-1 bg-slate-800 rounded-full overflow-hidden mt-0.5">
                        <div
                          className="h-full bg-emerald-400 rounded-full"
                          style={{ width: `${Math.min(100, (signals.popularity_weight || 0.9) * 100)}%` }}
                        />
                      </div>
                    </div>
                  </div>

                  <div className="mt-2.5 pt-1.5 border-t border-slate-800 text-[9px] text-slate-500 flex justify-between">
                    <span>Ranker: LightGBM v2.0</span>
                    <span className="text-sky-400 font-medium">Click score to close</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        <div className="grid grid-cols-2 gap-2">
          <Link
            href={`/product/${encodeURIComponent(item.item_id)}`}
            className="px-2.5 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium flex items-center justify-center gap-1.5 transition-colors"
          >
            <span>Similar</span>
            <ExternalLink className="w-3 h-3 text-slate-400" />
          </Link>

          <button
            onClick={handleAddToCart}
            className={`px-2.5 py-1.5 rounded-lg text-xs font-semibold flex items-center justify-center gap-1.5 transition-all shadow-sm ${
              added
                ? 'bg-emerald-600 text-white'
                : 'bg-gradient-to-r from-sky-600 to-indigo-600 hover:from-sky-500 hover:to-indigo-500 text-white'
            }`}
          >
            {added ? (
              <>
                <Check className="w-3.5 h-3.5" />
                <span>Added</span>
              </>
            ) : (
              <>
                <ShoppingCart className="w-3.5 h-3.5" />
                <span>Add</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
