'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname, useRouter } from 'next/navigation';
import { 
  Sparkles, 
  Search, 
  ShoppingCart, 
  Activity, 
  Layers, 
  ShieldCheck, 
  Flame, 
  Compass,
  Cpu
} from 'lucide-react';
import { useUser } from '@/context/UserContext';
import { RecSysAPI, HealthResponse } from '@/lib/api';
import UserSwitcher from './UserSwitcher';

export default function Navbar() {
  const pathname = usePathname();
  const router = useRouter();
  const { cartCount } = useUser();
  const [searchQuery, setSearchQuery] = useState('');
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    RecSysAPI.getHealth()
      .then((h) => setHealth(h))
      .catch(() => setHealth(null));
    
    const interval = setInterval(() => {
      RecSysAPI.getHealth()
        .then((h) => setHealth(h))
        .catch(() => setHealth(null));
    }, 15000);
    return () => clearInterval(interval);
  }, []);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      router.push(`/search?q=${encodeURIComponent(searchQuery.trim())}`);
    }
  };

  const isHealthy = health?.status === 'ok' || health?.status === 'healthy';

  return (
    <header className="sticky top-0 z-50 glass-panel border-b border-slate-800/80 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 gap-4">
          
          {/* Logo & Brand */}
          <Link href="/" className="flex items-center gap-2.5 group shrink-0">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-sky-500 via-indigo-500 to-purple-600 p-0.5 shadow-lg shadow-sky-500/20 group-hover:scale-105 transition-transform duration-200">
              <div className="w-full h-full bg-slate-950 rounded-[10px] flex items-center justify-center">
                <Sparkles className="w-5 h-5 text-sky-400 group-hover:rotate-12 transition-transform" />
              </div>
            </div>
            <div className="flex flex-col">
              <div className="flex items-center gap-1.5">
                <span className="font-bold text-lg tracking-tight text-white group-hover:text-sky-300 transition-colors">
                  Amazon RecSys
                </span>
                <span className="text-[10px] font-semibold uppercase tracking-wider px-1.5 py-0.5 rounded bg-sky-500/10 text-sky-400 border border-sky-500/20">
                  v2.0
                </span>
              </div>
              <span className="text-[11px] text-slate-400 hidden sm:inline">
                Two-Stage Retrieval + LLM Ranking
              </span>
            </div>
          </Link>

          {/* Search Bar */}
          <form onSubmit={handleSearchSubmit} className="flex-1 max-w-lg hidden md:block">
            <div className="relative group">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search headphones, keyboards, cameras (LLM Query Rewrite)..."
                className="w-full pl-10 pr-24 py-2 text-sm bg-slate-900/90 text-slate-200 placeholder-slate-500 rounded-lg border border-slate-700/60 focus:border-sky-500 focus:ring-1 focus:ring-sky-500 focus:outline-none transition-all shadow-inner"
              />
              <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-3 group-focus-within:text-sky-400 transition-colors" />
              <button
                type="submit"
                className="absolute right-1.5 top-1.5 px-3 py-1 text-xs font-medium bg-gradient-to-r from-sky-600 to-indigo-600 hover:from-sky-500 hover:to-indigo-500 text-white rounded-md transition-all shadow-sm flex items-center gap-1"
              >
                <span>AI Search</span>
              </button>
            </div>
          </form>

          {/* Navigation Items & User Switcher */}
          <div className="flex items-center gap-3">
            {/* Demo User Switcher */}
            <UserSwitcher />

            {/* Nav Links */}
            <nav className="flex items-center gap-1">
              <Link
                href="/search"
                className={`p-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-1.5 ${
                  pathname === '/search'
                    ? 'bg-slate-800 text-sky-400 border border-slate-700'
                    : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
                }`}
                title="Free-text Search"
              >
                <Search className="w-4 h-4 md:hidden" />
                <span className="hidden lg:inline">Search</span>
              </Link>

              <Link
                href="/admin"
                className={`p-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-1.5 ${
                  pathname === '/admin'
                    ? 'bg-slate-800 text-amber-400 border border-slate-700'
                    : 'text-slate-300 hover:text-white hover:bg-slate-800/60'
                }`}
                title="Observability & Metrics Telemetry"
              >
                <Activity className="w-4 h-4 text-amber-400" />
                <span className="hidden lg:inline">Admin / Metrics</span>
              </Link>
            </nav>

            {/* Health & Cart Indicators */}
            <div className="flex items-center gap-2 pl-2 border-l border-slate-800">
              {/* System Health Badge */}
              <div 
                className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-slate-900 border border-slate-800 text-[11px] font-medium"
                title={`Backend API Status: ${health?.status || 'Connecting...'}`}
              >
                <span className={`w-2 h-2 rounded-full ${isHealthy ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`} />
                <span className="text-slate-400 hidden xl:inline">API</span>
                <span className={isHealthy ? 'text-emerald-400 hidden xl:inline' : 'text-amber-400 hidden xl:inline'}>
                  {isHealthy ? 'Live' : 'Standby'}
                </span>
              </div>

              {/* Cart Counter */}
              <div 
                className="relative p-2 text-slate-300 hover:text-white rounded-lg hover:bg-slate-800/60 transition-colors"
                title="Shopping Bag"
              >
                <ShoppingCart className="w-4 h-4" />
                {cartCount > 0 && (
                  <span className="absolute -top-1 -right-1 bg-sky-500 text-slate-950 font-bold text-[10px] w-4 h-4 rounded-full flex items-center justify-center animate-pulse">
                    {cartCount}
                  </span>
                )}
              </div>
            </div>

          </div>

        </div>
      </div>
    </header>
  );
}
