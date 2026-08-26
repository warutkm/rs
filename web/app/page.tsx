'use client';

import React, { useState, useEffect } from 'react';
import { useUser } from '@/context/UserContext';
import { RecSysAPI, RecommendResponse, RecommendedItem } from '@/lib/api';
import ProductCard from '@/components/ProductCard';
import { 
  Sparkles, 
  Flame, 
  RefreshCw, 
  SlidersHorizontal, 
  Zap, 
  Layers, 
  Cpu, 
  ShieldCheck,
  UserCheck,
  Compass,
  ArrowRight
} from 'lucide-react';
import Link from 'next/link';

const CATEGORIES = [
  'All Categories',
  'Video Games',
  'Electronics',
  'Cell Phones & Accessories',
  'Camera & Photo',
  'Smart Home',
];

export default function HomePage() {
  const { currentUser, allUsers, setCurrentUser } = useUser();
  const [selectedCategory, setSelectedCategory] = useState('All Categories');
  
  // Recommendations state
  const [personalizedRecs, setPersonalizedRecs] = useState<RecommendResponse | null>(null);
  const [trendingRecs, setTrendingRecs] = useState<RecommendResponse | null>(null);
  const [loadingPersonalized, setLoadingPersonalized] = useState(true);
  const [loadingTrending, setLoadingTrending] = useState(true);

  // Fetch recommendations whenever user or category changes
  const fetchRecommendations = async () => {
    setLoadingPersonalized(true);
    try {
      const catFilter = selectedCategory === 'All Categories' ? undefined : selectedCategory;
      const res = await RecSysAPI.getRecommendations({
        userId: currentUser.id,
        topK: 8,
        categoryFilter: catFilter,
      });
      setPersonalizedRecs(res);
    } catch (err) {
      console.error('Failed to load personalized recommendations:', err);
    } finally {
      setLoadingPersonalized(false);
    }
  };

  // Fetch trending items once on load
  const fetchTrending = async () => {
    setLoadingTrending(true);
    try {
      const res = await RecSysAPI.getRecommendations({
        userId: 'guest_cold_start',
        topK: 4,
      });
      setTrendingRecs(res);
    } catch (err) {
      console.error('Failed to load trending items:', err);
    } finally {
      setLoadingTrending(false);
    }
  };

  useEffect(() => {
    fetchRecommendations();
  }, [currentUser.id, selectedCategory]);

  useEffect(() => {
    fetchTrending();
  }, []);

  const isColdStart = currentUser.id === 'guest_cold_start';

  return (
    <div className="space-y-10 animate-fade-in">
      
      {/* Hero / Architecture Banner */}
      <section className="relative rounded-2xl overflow-hidden glass-panel p-6 sm:p-8 border border-sky-500/20 shadow-2xl">
        <div className="absolute -right-20 -top-20 w-80 h-80 bg-sky-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute -left-20 -bottom-20 w-80 h-80 bg-purple-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10 max-w-3xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-gradient-to-r from-sky-500/20 to-indigo-500/20 text-sky-300 text-xs font-semibold border border-sky-500/30 mb-4">
            <Sparkles className="w-3.5 h-3.5 text-sky-400" />
            <span>Multi-Stage Two-Tower Retrieval + LambdaMART Ranker</span>
          </div>
          
          <h1 className="text-2xl sm:text-4xl font-extrabold text-white tracking-tight leading-tight">
            Amazon RecSys <span className="text-gradient">Production Platform</span>
          </h1>
          
          <p className="mt-3 text-sm sm:text-base text-slate-300 leading-relaxed">
            Real-time personalized ranking combining <span className="text-sky-300 font-medium">Qdrant Vector ANN</span>, <span className="text-indigo-300 font-medium">Implicit ALS / Neural CF</span> candidate generators, a learned <span className="text-purple-300 font-medium">LightGBM Ranker</span>, and cached <span className="text-amber-300 font-medium">Gemini 3.5 Flash-Lite</span> reasoning.
          </p>

          {/* Architecture Pipeline Pills */}
          <div className="mt-5 flex flex-wrap items-center gap-2 text-xs text-slate-300 font-mono">
            <span className="px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-400" />
              Stage 1: Retrieval (200 cands)
            </span>
            <span className="text-slate-600 font-sans">→</span>
            <span className="px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-sky-400" />
              Stage 2: LambdaMART (Top 10)
            </span>
            <span className="text-slate-600 font-sans">→</span>
            <span className="px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-indigo-400" />
              Stage 3: LLM Explanation (Redis)
            </span>
          </div>
        </div>
      </section>

      {/* Active Persona Banner & Fast Switcher */}
      <section className="glass-panel rounded-xl p-4 border border-slate-800/90 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="flex items-center gap-3.5">
          <div className="text-3xl p-2 rounded-xl bg-slate-900 border border-slate-800 select-none shadow-inner">
            {currentUser.avatar}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Active User Persona
              </span>
              <span
                className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${currentUser.badgeColor}`}
              >
                {isColdStart ? 'Cold-Start Mode' : 'Personalized Mode'}
              </span>
            </div>
            <h2 className="text-base font-bold text-white">
              {currentUser.name}{' '}
              <span className="text-xs font-normal text-slate-400">
                ({currentUser.persona})
              </span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5 line-clamp-1">
              {currentUser.description}
            </p>
          </div>
        </div>

        {/* Quick Switch Persona Pills */}
        <div className="flex items-center gap-1.5 flex-wrap w-full md:w-auto">
          <span className="text-xs text-slate-400 font-medium mr-1 hidden sm:inline">
            Quick Switch:
          </span>
          {allUsers.slice(0, 4).map((user) => (
            <button
              key={user.id}
              onClick={() => setCurrentUser(user)}
              className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-all ${
                user.id === currentUser.id
                  ? 'bg-sky-600 text-white shadow-md'
                  : 'bg-slate-900/80 hover:bg-slate-800 text-slate-300 border border-slate-800'
              }`}
            >
              <span className="mr-1">{user.avatar}</span>
              {user.name.split(' ')[0]}
            </button>
          ))}
          <button
            onClick={() => setCurrentUser(allUsers[allUsers.length - 1])}
            className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-all ${
              isColdStart
                ? 'bg-amber-600 text-white shadow-md'
                : 'bg-slate-900/80 hover:bg-slate-800 text-amber-300 border border-slate-800'
            }`}
          >
            👤 Guest (Cold Start)
          </button>
        </div>
      </section>

      {/* Category Filter Pills & Refresh */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 max-w-full">
          {CATEGORIES.map((cat) => (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-all ${
                selectedCategory === cat
                  ? 'bg-slate-200 text-slate-950 font-bold shadow'
                  : 'bg-slate-900/80 text-slate-300 hover:text-white hover:bg-slate-800 border border-slate-800'
              }`}
            >
              {cat}
            </button>
          ))}
        </div>

        <button
          onClick={fetchRecommendations}
          disabled={loadingPersonalized}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-300 hover:text-white bg-slate-900 hover:bg-slate-800 rounded-lg border border-slate-800 transition-colors shrink-0"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loadingPersonalized ? 'animate-spin text-sky-400' : ''}`} />
          <span>Refresh Recommendations</span>
        </button>
      </div>

      {/* Personalized Recommendation Rail */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-sky-500/10 border border-sky-500/20 flex items-center justify-center">
              <Sparkles className="w-4 h-4 text-sky-400" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
                Recommended For You
                {personalizedRecs && (
                  <span className="text-xs font-mono font-normal px-2 py-0.5 rounded bg-slate-900 text-sky-400 border border-slate-800">
                    Source: {personalizedRecs.source}
                  </span>
                )}
              </h2>
              <p className="text-xs text-slate-400">
                {isColdStart 
                  ? 'Cold-start mode: Showing popularity & content-based baselines since user history is empty.'
                  : `Personalized for ${currentUser.name} using learned LambdaMART ranker & candidate embeddings.`}
              </p>
            </div>
          </div>
        </div>

        {/* Product Cards Grid */}
        {loadingPersonalized ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {[1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
              <div key={i} className="glass-card rounded-xl p-4 h-80 animate-pulse bg-slate-900/60 flex flex-col justify-between">
                <div className="space-y-3">
                  <div className="w-full h-32 bg-slate-800/60 rounded-lg" />
                  <div className="w-3/4 h-4 bg-slate-800/60 rounded" />
                  <div className="w-1/2 h-3 bg-slate-800/40 rounded" />
                </div>
                <div className="w-full h-8 bg-slate-800/60 rounded" />
              </div>
            ))}
          </div>
        ) : personalizedRecs && personalizedRecs.results.length > 0 ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {personalizedRecs.results.map((item, idx) => (
              <ProductCard
                key={`${item.item_id}_${idx}`}
                item={item}
                rank={idx + 1}
                showScore={true}
              />
            ))}
          </div>
        ) : (
          <div className="glass-panel p-8 text-center rounded-xl border border-slate-800">
            <p className="text-slate-400 text-sm">No recommendations returned for this filter criteria.</p>
          </div>
        )}
      </section>

      {/* Trending / Best Sellers Rail */}
      <section className="space-y-4 pt-4 border-t border-slate-900">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-amber-500/10 border border-amber-500/20 flex items-center justify-center">
              <Flame className="w-4 h-4 text-amber-400" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white tracking-tight">
                Trending & Popular Across Catalog
              </h2>
              <p className="text-xs text-slate-400">
                Top rated items by global interaction volume and helpful review velocity.
              </p>
            </div>
          </div>
        </div>

        {loadingTrending ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="glass-card rounded-xl p-4 h-80 animate-pulse bg-slate-900/60" />
            ))}
          </div>
        ) : trendingRecs && trendingRecs.results.length > 0 ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
            {trendingRecs.results.map((item, idx) => (
              <ProductCard
                key={`trending_${item.item_id}_${idx}`}
                item={item}
                rank={idx + 1}
                showScore={false}
              />
            ))}
          </div>
        ) : null}
      </section>

    </div>
  );
}
