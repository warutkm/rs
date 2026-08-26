'use client';

import React, { useState, useEffect, Suspense } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { 
  Search, 
  Sparkles, 
  Filter, 
  DollarSign, 
  Tag, 
  Compass, 
  SlidersHorizontal, 
  ExternalLink, 
  Package, 
  Star,
  Activity,
  Layers
} from 'lucide-react';
import { RecSysAPI, SearchResponse, SearchResult } from '@/lib/api';
import ProductCard from '@/components/ProductCard';
import ScoreBreakdown from '@/components/ScoreBreakdown';
import Link from 'next/link';

const EXAMPLE_QUERIES = [
  'noise cancelling wireless headphones for travel',
  'budget mechanical gaming keyboard with rgb',
  'ergonomic vertical mouse under 50',
  'fast charging 100w gan usb-c power bank',
  '4k streaming webcam with ring light',
];

function SearchContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const initialQuery = searchParams.get('q') || '';

  const [query, setQuery] = useState(initialQuery);
  const [activeQuery, setActiveQuery] = useState(initialQuery);
  const [searchResponse, setSearchResponse] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const executeSearch = async (searchTerm: string) => {
    if (!searchTerm.trim()) return;
    setLoading(true);
    setActiveQuery(searchTerm);
    try {
      const res = await RecSysAPI.search(searchTerm);
      setSearchResponse(res);
    } catch (err) {
      console.error('Search failed:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (initialQuery) {
      setQuery(initialQuery);
      executeSearch(initialQuery);
    } else {
      // Default query for immediate visual satisfaction
      setQuery('wireless gaming headphones');
      executeSearch('wireless gaming headphones');
    }
  }, [initialQuery]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      router.push(`/search?q=${encodeURIComponent(query.trim())}`);
      executeSearch(query.trim());
    }
  };

  const handleChipClick = (chip: string) => {
    setQuery(chip);
    router.push(`/search?q=${encodeURIComponent(chip)}`);
    executeSearch(chip);
  };

  return (
    <div className="space-y-8 animate-fade-in">
      
      {/* Search Header & Input */}
      <section className="glass-panel p-6 sm:p-8 rounded-2xl border border-slate-800 space-y-4 shadow-xl">
        <div className="max-w-2xl">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-sky-500/10 text-sky-300 text-xs font-semibold border border-sky-500/20 mb-2">
            <Sparkles className="w-3.5 h-3.5 text-sky-400" />
            <span>LLM Query Understanding + Hybrid e5 / BM25 Search</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white">
            Semantic Catalog Search
          </h1>
          <p className="text-xs sm:text-sm text-slate-400 mt-1">
            Free-text queries are rewritten by Gemini 3.5 Flash-Lite into structured filters and semantic embeddings before running hybrid vector & BM25 retrieval.
          </p>
        </div>

        {/* Search Bar */}
        <form onSubmit={handleSubmit} className="relative">
          <div className="flex gap-2">
            <div className="relative flex-1">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Describe what you are looking for (e.g. 'cheap wireless headphones for gym')..."
                className="w-full pl-11 pr-4 py-3 bg-slate-900/90 text-slate-100 placeholder-slate-500 rounded-xl border border-slate-700/80 focus:border-sky-500 focus:ring-1 focus:ring-sky-500 focus:outline-none text-sm transition-all shadow-inner"
              />
              <Search className="w-5 h-5 text-slate-400 absolute left-3.5 top-3.5" />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="px-6 py-3 rounded-xl bg-gradient-to-r from-sky-600 via-indigo-600 to-purple-600 hover:from-sky-500 hover:to-purple-500 text-white text-sm font-bold transition-all shadow-lg flex items-center gap-2 shrink-0"
            >
              <Sparkles className="w-4 h-4" />
              <span>{loading ? 'Analyzing...' : 'Search'}</span>
            </button>
          </div>
        </form>

        {/* Suggestion Chips */}
        <div className="flex items-center gap-2 flex-wrap text-xs">
          <span className="text-slate-400 font-medium">Try searching:</span>
          {EXAMPLE_QUERIES.map((example, i) => (
            <button
              key={i}
              onClick={() => handleChipClick(example)}
              className="px-2.5 py-1 rounded-lg bg-slate-900/80 hover:bg-slate-800 text-slate-300 border border-slate-800 transition-colors text-[11px]"
            >
              {example}
            </button>
          ))}
        </div>
      </section>

      {/* Query Understanding Breakdown Card (if response available) */}
      {searchResponse && (
        <section className="glass-panel p-5 rounded-xl border border-sky-500/30 bg-gradient-to-r from-slate-900/90 via-indigo-950/20 to-slate-900/90 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs font-bold text-sky-300">
              <Sparkles className="w-4 h-4 text-sky-400" />
              <span>LLM Query Breakdown & Filter Extraction</span>
            </div>
            <span className="text-[10px] font-mono text-slate-400">
              Engine: Gemini 3.5 Flash-Lite (Fast Structured Output)
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3 text-xs">
            {/* Original Query */}
            <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <span className="text-[10px] uppercase font-semibold text-slate-400 block mb-1">
                Raw User Query
              </span>
              <span className="font-medium text-slate-200 break-words">
                "{searchResponse.query}"
              </span>
            </div>

            {/* Rewritten Query */}
            <div className="p-3 rounded-lg bg-slate-950/60 border border-sky-500/30">
              <span className="text-[10px] uppercase font-semibold text-sky-400 block mb-1">
                Semantic Rewritten Query
              </span>
              <span className="font-semibold text-sky-300 break-words">
                "{searchResponse.rewritten_query || searchResponse.query}"
              </span>
            </div>

            {/* Inferred Category */}
            <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <span className="text-[10px] uppercase font-semibold text-slate-400 block mb-1">
                Extracted Category
              </span>
              <span className="font-medium text-emerald-400">
                {searchResponse.category_filter || 'All Categories'}
              </span>
            </div>

            {/* Inferred Max Price / Intent */}
            <div className="p-3 rounded-lg bg-slate-950/60 border border-slate-800">
              <span className="text-[10px] uppercase font-semibold text-slate-400 block mb-1">
                Price Cap & Intent
              </span>
              <div className="flex items-center gap-2">
                <span className="text-amber-300 font-bold">
                  {searchResponse.price_max ? `< $${searchResponse.price_max}` : 'Unconstrained'}
                </span>
                {searchResponse.intent && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                    {searchResponse.intent}
                  </span>
                )}
              </div>
            </div>
          </div>
        </section>
      )}

      {/* Search Results Grid */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
            Search Results
            {searchResponse && (
              <span className="text-xs font-normal text-slate-400">
                ({searchResponse.results.length} hybrid retrieved items)
              </span>
            )}
          </h2>
        </div>

        {loading ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
            {[1, 2, 3, 4, 5, 6].map((i) => (
              <div key={i} className="glass-card rounded-xl p-4 h-72 animate-pulse bg-slate-900/60" />
            ))}
          </div>
        ) : searchResponse && searchResponse.results.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {searchResponse.results.map((item, idx) => (
              <div
                key={`${item.item_id}_${idx}`}
                className="glass-card rounded-xl p-4 flex flex-col justify-between space-y-3"
              >
                <div>
                  {/* Category & Rank */}
                  <div className="flex items-center justify-between gap-2 mb-2">
                    <span className="text-[10px] font-medium px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-slate-700">
                      {item.category || 'Product'}
                    </span>
                    <span className="font-mono text-[10px] text-slate-500">
                      Rank #{idx + 1}
                    </span>
                  </div>

                  <Link href={`/product/${encodeURIComponent(item.item_id)}`} className="block group">
                    <h3 className="font-semibold text-sm text-slate-100 line-clamp-2 group-hover:text-sky-300 transition-colors">
                      {item.title || `Item ${item.item_id}`}
                    </h3>
                  </Link>

                  <div className="flex items-center justify-between mt-2 text-xs">
                    <div className="flex items-center gap-1 text-amber-400">
                      <Star className="w-3.5 h-3.5 fill-current" />
                      <span className="font-bold text-slate-200">
                        {(item.average_rating || 4.5).toFixed(1)}
                      </span>
                    </div>
                    <span className="font-bold text-slate-100">
                      ${(item.price || 29.99).toFixed(2)}
                    </span>
                  </div>
                </div>

                {/* Score Breakdown Bars */}
                <ScoreBreakdown
                  score={item.hybrid_score}
                  embScore={item.emb_score}
                  bm25Score={item.bm25_score}
                />

                <Link
                  href={`/product/${encodeURIComponent(item.item_id)}`}
                  className="w-full py-1.5 px-3 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium flex items-center justify-center gap-1.5 transition-colors"
                >
                  <span>View Details & Similar</span>
                  <ExternalLink className="w-3 h-3 text-slate-400" />
                </Link>
              </div>
            ))}
          </div>
        ) : (
          <div className="glass-panel p-8 text-center rounded-xl border border-slate-800">
            <p className="text-slate-400 text-sm">No items found matching your query.</p>
          </div>
        )}
      </section>

    </div>
  );
}

export default function SearchPage() {
  return (
    <Suspense fallback={<div className="p-8 text-center text-slate-400">Loading search...</div>}>
      <SearchContent />
    </Suspense>
  );
}
