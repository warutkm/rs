'use client';

import React, { useState, useEffect } from 'react';
import { 
  Activity, 
  Server, 
  Database, 
  Cpu, 
  RefreshCw, 
  Zap, 
  Play, 
  CheckCircle2, 
  XCircle, 
  Clock, 
  Terminal, 
  Layers, 
  Gauge,
  Sliders,
  ShieldCheck,
  AlertTriangle
} from 'lucide-react';
import { RecSysAPI, MetricsResponse, HealthResponse, AdminRetrainStatusResponse } from '@/lib/api';

export default function AdminPage() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Retrain state
  const [retrainLoading, setRetrainLoading] = useState(false);
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [retrainStatus, setRetrainStatus] = useState<AdminRetrainStatusResponse | null>(null);
  const [forceRetrain, setForceRetrain] = useState(false);

  const fetchTelemetry = async () => {
    try {
      const [m, h] = await Promise.all([
        RecSysAPI.getMetrics(),
        RecSysAPI.getHealth(),
      ]);
      setMetrics(m);
      setHealth(h);
    } catch (err) {
      console.error('Telemetry fetch failed:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTelemetry();
    let interval: NodeJS.Timeout | null = null;
    if (autoRefresh) {
      interval = setInterval(fetchTelemetry, 3000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh]);

  // Poll retrain status if active job
  useEffect(() => {
    if (!activeJobId) return;

    const interval = setInterval(async () => {
      try {
        const st = await RecSysAPI.getRetrainStatus(activeJobId);
        setRetrainStatus(st);
        if (st.status === 'success' || st.status === 'failed') {
          setActiveJobId(null);
        }
      } catch (e) {
        console.error('Failed to poll retrain status:', e);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [activeJobId]);

  const handleTriggerRetrain = async () => {
    setRetrainLoading(true);
    try {
      const res = await RecSysAPI.triggerRetrain(forceRetrain);
      if (res.job_id) {
        setActiveJobId(res.job_id);
      }
      fetchTelemetry();
    } catch (err: any) {
      alert(`Retrain trigger failed: ${err.message || err}`);
    } finally {
      setRetrainLoading(false);
    }
  };

  const getLatencyColor = (ms: number) => {
    if (ms < 50) return 'text-emerald-400';
    if (ms < 150) return 'text-amber-400';
    return 'text-rose-400';
  };

  return (
    <div className="space-y-8 animate-fade-in">
      
      {/* Header & Controls */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-amber-500/10 text-amber-300 text-xs font-semibold border border-amber-500/20 mb-2">
            <Activity className="w-3.5 h-3.5 text-amber-400" />
            <span>Structured JSON /metrics & Diagnostics Telemetry</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white">
            System Observability & Retrain Automation
          </h1>
          <p className="text-xs sm:text-sm text-slate-400 mt-1">
            Zero-infrastructure telemetry endpoint monitoring service health, latency percentiles, Redis cache hit ratios, and DVC retraining DAG execution.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="rounded bg-slate-800 border-slate-700 text-sky-500 focus:ring-0"
            />
            <span>Auto-Refresh (3s)</span>
          </label>

          <button
            onClick={fetchTelemetry}
            className="p-2 bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded-lg text-slate-300 hover:text-white transition-colors"
            title="Manual Telemetry Refresh"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin text-amber-400' : ''}`} />
          </button>
        </div>
      </div>

      {/* Subsystem Health Grid */}
      <section className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
        <h2 className="text-base font-bold text-white flex items-center gap-2">
          <Server className="w-4 h-4 text-sky-400" />
          <span>Subsystem Status Diagnostics</span>
        </h2>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3">
          {/* Vector DB */}
          <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-300">Qdrant Vector DB</span>
              {health?.vector_db_connected ? (
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              ) : (
                <AlertTriangle className="w-4 h-4 text-amber-400" />
              )}
            </div>
            <span className={`text-[11px] font-mono ${health?.vector_db_connected ? 'text-emerald-400' : 'text-amber-400'}`}>
              {health?.vector_db_connected ? 'Connected (ANN)' : 'Offline / Standby'}
            </span>
          </div>

          {/* Redis Cache */}
          <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-300">Redis Cache</span>
              {health?.redis_connected ? (
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              ) : (
                <AlertTriangle className="w-4 h-4 text-amber-400" />
              )}
            </div>
            <span className={`text-[11px] font-mono ${health?.redis_connected ? 'text-emerald-400' : 'text-amber-400'}`}>
              {health?.redis_connected ? 'Active (Sub-ms)' : 'Offline / In-Memory'}
            </span>
          </div>

          {/* PostgreSQL */}
          <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-300">PostgreSQL DB</span>
              {health?.db_connected ? (
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              ) : (
                <AlertTriangle className="w-4 h-4 text-amber-400" />
              )}
            </div>
            <span className={`text-[11px] font-mono ${health?.db_connected ? 'text-emerald-400' : 'text-amber-400'}`}>
              {health?.db_connected ? 'Events Logged' : 'Standby'}
            </span>
          </div>

          {/* LightGBM Ranker */}
          <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-300">LambdaMART</span>
              {health?.ranker_loaded ? (
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              ) : (
                <CheckCircle2 className="w-4 h-4 text-sky-400" />
              )}
            </div>
            <span className="text-[11px] font-mono text-emerald-400">
              {health?.ranker_loaded ? 'Model Loaded' : 'Online'}
            </span>
          </div>

          {/* LLM Engine */}
          <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-300">Gemini LLM</span>
              <CheckCircle2 className="w-4 h-4 text-purple-400" />
            </div>
            <span className="text-[11px] font-mono text-purple-400">
              3.5 Flash-Lite
            </span>
          </div>
        </div>
      </section>

      {/* Latency Percentiles & Cache Ratios Cards */}
      <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* p50 Latency */}
        <div className="glass-card p-5 rounded-xl space-y-2">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Latency p50 (Median)</span>
            <Gauge className="w-4 h-4 text-sky-400" />
          </div>
          <div className={`text-3xl font-extrabold font-mono ${getLatencyColor(metrics?.latency_p50_ms || 0)}`}>
            {(metrics?.latency_p50_ms || 0).toFixed(1)} <span className="text-sm font-normal text-slate-400">ms</span>
          </div>
          <div className="text-[10px] text-slate-500">SLA Target: &lt; 50ms</div>
        </div>

        {/* p95 Latency */}
        <div className="glass-card p-5 rounded-xl space-y-2">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Latency p95</span>
            <Gauge className="w-4 h-4 text-amber-400" />
          </div>
          <div className={`text-3xl font-extrabold font-mono ${getLatencyColor(metrics?.latency_p95_ms || 0)}`}>
            {(metrics?.latency_p95_ms || 0).toFixed(1)} <span className="text-sm font-normal text-slate-400">ms</span>
          </div>
          <div className="text-[10px] text-slate-500">SLA Target: &lt; 150ms</div>
        </div>

        {/* p99 Latency */}
        <div className="glass-card p-5 rounded-xl space-y-2">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Latency p99</span>
            <Gauge className="w-4 h-4 text-rose-400" />
          </div>
          <div className={`text-3xl font-extrabold font-mono ${getLatencyColor(metrics?.latency_p99_ms || 0)}`}>
            {(metrics?.latency_p99_ms || 0).toFixed(1)} <span className="text-sm font-normal text-slate-400">ms</span>
          </div>
          <div className="text-[10px] text-slate-500">Tail latency outlier bound</div>
        </div>

        {/* Cache Hit Rate */}
        <div className="glass-card p-5 rounded-xl space-y-2">
          <div className="flex items-center justify-between text-xs text-slate-400">
            <span>Redis Cache Hit Rate</span>
            <Zap className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-3xl font-extrabold font-mono text-emerald-400">
            {((metrics?.cache_hit_rate || 0) * 100).toFixed(1)}%
          </div>
          <div className="text-[10px] text-slate-400 flex justify-between">
            <span>Hits: {metrics?.cache_hits || 0}</span>
            <span>Misses: {metrics?.cache_misses || 0}</span>
          </div>
        </div>
      </section>

      {/* Requests Breakdown & Endpoint Distribution */}
      <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Endpoint Traffic */}
        <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Activity className="w-4 h-4 text-sky-400" />
              <span>Requests per Endpoint</span>
            </h3>
            <span className="text-xs font-mono text-slate-400">
              Total: {metrics?.total_requests || 0}
            </span>
          </div>

          <div className="space-y-2.5">
            {metrics?.requests_per_endpoint &&
              Object.entries(metrics.requests_per_endpoint).map(([ep, count]) => {
                const total = metrics.total_requests || 1;
                const pct = Math.round((count / total) * 100);
                return (
                  <div key={ep} className="space-y-1">
                    <div className="flex justify-between text-xs">
                      <span className="font-mono text-slate-300">{ep}</span>
                      <span className="font-mono text-slate-400 font-bold">
                        {count} ({pct}%)
                      </span>
                    </div>
                    <div className="w-full h-2 bg-slate-900 rounded-full overflow-hidden border border-slate-800">
                      <div
                        className="h-full bg-gradient-to-r from-sky-500 to-indigo-500 rounded-full"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                  </div>
                );
              })}
          </div>
        </div>

        {/* HTTP Status Breakdown */}
        <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <Layers className="w-4 h-4 text-indigo-400" />
            <span>HTTP Status Code Distribution</span>
          </h3>

          <div className="grid grid-cols-2 gap-3 pt-2">
            {metrics?.requests_per_status &&
              Object.entries(metrics.requests_per_status).map(([code, count]) => (
                <div
                  key={code}
                  className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1"
                >
                  <span className="text-xs text-slate-400 block font-mono">Status {code}</span>
                  <span className="text-2xl font-bold font-mono text-white">{count}</span>
                </div>
              ))}
          </div>
        </div>
      </section>

      {/* DVC Retraining Pipeline Trigger & Live Log Stream */}
      <section className="glass-panel p-6 sm:p-8 rounded-2xl border border-slate-800 space-y-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <Play className="w-4 h-4 text-emerald-400" />
              <span>DVC Retraining DAG Automation</span>
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              Triggers <code className="text-sky-300 font-mono">dvc repro</code> in background subprocess, regenerating ALS/SVD++/MF/NCF candidate generators and retraining LightGBM LambdaMART.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={forceRetrain}
                onChange={(e) => setForceRetrain(e.target.checked)}
                className="rounded bg-slate-800 border-slate-700 text-sky-500 focus:ring-0"
              />
              <span>Force All Stages (--force)</span>
            </label>

            <button
              onClick={handleTriggerRetrain}
              disabled={retrainLoading || Boolean(activeJobId)}
              className={`px-4 py-2 rounded-xl text-xs font-bold flex items-center gap-2 shadow-lg transition-all ${
                activeJobId
                  ? 'bg-amber-600/50 text-amber-200 cursor-not-allowed'
                  : 'bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white'
              }`}
            >
              <Play className="w-3.5 h-3.5" />
              <span>{activeJobId ? 'Retrain Running...' : 'Trigger Retrain Pipeline'}</span>
            </button>
          </div>
        </div>

        {/* Retrain Job Status & Log Tail */}
        {retrainStatus && (
          <div className="space-y-3 pt-3 border-t border-slate-800">
            <div className="flex items-center justify-between text-xs">
              <div className="flex items-center gap-2">
                <span className="text-slate-400">Job ID:</span>
                <span className="font-mono text-sky-300">{retrainStatus.job_id}</span>
                <span
                  className={`px-2 py-0.5 rounded font-bold uppercase text-[10px] ${
                    retrainStatus.status === 'running'
                      ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30 animate-pulse'
                      : retrainStatus.status === 'success'
                      ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                      : 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                  }`}
                >
                  {retrainStatus.status}
                </span>
              </div>
              <span className="text-slate-500 font-mono">Started: {retrainStatus.started_at}</span>
            </div>

            {/* Terminal Log Tail Box */}
            <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 font-mono text-xs text-slate-300 space-y-1 max-h-60 overflow-y-auto">
              <div className="text-slate-500 flex items-center gap-1.5 pb-2 border-b border-slate-800/80">
                <Terminal className="w-3.5 h-3.5 text-slate-400" />
                <span>Execution Log Tail</span>
              </div>
              <pre className="whitespace-pre-wrap text-[11px] leading-relaxed text-slate-300">
                {retrainStatus.log_tail || 'Retrain job initialized... awaiting pipeline stage logs.'}
              </pre>
            </div>
          </div>
        )}
      </section>

    </div>
  );
}
