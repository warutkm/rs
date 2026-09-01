#!/usr/bin/env python3
"""
scripts/verify_tier1_connectivity.py
Phase 11 — Tier 1 Free Cloud Deployment Pre-Flight Connectivity Verification Tool

Performs automated connectivity, authentication, schema, and latency diagnostics against:
  1. Neon PostgreSQL (Serverless event store / DATABASE_URL)
  2. Upstash Redis (Serverless cache / REDIS_URL)
  3. Qdrant Cloud Cluster (Managed vector store / QDRANT_URL + QDRANT_API_KEY)
  4. Google Gemini API (LLM layer / GEMINI_API_KEY)
  5. FastAPI Live Backend Endpoint (Health & Metrics diagnostics)

Usage:
  python scripts/verify_tier1_connectivity.py --all
  python scripts/verify_tier1_connectivity.py --db --redis --qdrant --llm
  python scripts/verify_tier1_connectivity.py --api-url https://amazon-recsys-api.onrender.com
  python scripts/verify_tier1_connectivity.py --json
"""

import os
import sys
import time
import json
import asyncio
import argparse
from typing import Dict, Any, Optional

# Add project root to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
from api.db import _parse_and_clean_dsn, CREATE_TABLE_SQL


# Color formatting helpers for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_banner():
    print(f"\n{BOLD}{BLUE}==================================================================={RESET}")
    print(f"{BOLD}{BLUE}  Amazon RecSys v2 — Tier 1 Cloud Connectivity Pre-Flight Check   {RESET}")
    print(f"{BOLD}{BLUE}==================================================================={RESET}\n")


async def check_postgres(dsn: Optional[str] = None) -> Dict[str, Any]:
    """
    Validates PostgreSQL (Neon / Supabase / Local) connection, SSL handshake,
    executes 'SELECT 1', and verifies the 'events' table schema.
    """
    target_dsn = dsn or config.POSTGRES_URL
    if not target_dsn:
        target_dsn = (
            f"postgresql://{config.POSTGRES_USER}:{config.POSTGRES_PASSWORD}@"
            f"{config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DB}"
        )

    clean_dsn, ssl_param = _parse_and_clean_dsn(target_dsn)
    from urllib.parse import urlparse

    parsed = urlparse(clean_dsn)
    host_display = f"{parsed.hostname}:{parsed.port or 5432}/{parsed.path.lstrip('/')}"

    start_t = time.perf_counter()
    try:
        import asyncpg
    except ImportError:
        return {
            "service": "PostgreSQL",
            "host": host_display,
            "status": "WARN",
            "latency_ms": 0.0,
            "message": "asyncpg library not installed. In-memory event fallback will be active.",
        }

    try:
        pool_kwargs = {
            "dsn": clean_dsn,
            "min_size": 1,
            "max_size": 2,
            "command_timeout": 8,
            "timeout": 8,
        }
        if ssl_param is not None:
            pool_kwargs["ssl"] = ssl_param

        pool = await asyncpg.create_pool(**pool_kwargs)
        async with pool.acquire() as conn:
            val = await conn.fetchval("SELECT 1;")
            await conn.execute(CREATE_TABLE_SQL)
            table_check = await conn.fetchval(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'events');"
            )
        await pool.close()
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        return {
            "service": "PostgreSQL",
            "host": host_display,
            "status": "PASS" if (val == 1 and table_check) else "WARN",
            "latency_ms": round(elapsed_ms, 2),
            "message": f"Connected successfully (SSL={ssl_param}). 'events' table schema verified.",
        }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        return {
            "service": "PostgreSQL",
            "host": host_display,
            "status": "FAIL",
            "latency_ms": round(elapsed_ms, 2),
            "message": f"Connection failed: {str(exc)}",
            "remediation": "Verify DATABASE_URL in .env, check Neon project active state, and ensure ?sslmode=require is set.",
        }


async def check_redis(
    url: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    password: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validates Redis (Upstash / Local) connection, executes PING,
    and performs a probe SET/GET/DEL cycle.
    """
    target_url = url or config.REDIS_URL
    target_host = host or config.REDIS_HOST
    target_port = port or config.REDIS_PORT
    target_password = password or config.REDIS_PASSWORD

    from urllib.parse import urlparse

    if target_url:
        parsed = urlparse(target_url)
        host_display = f"{parsed.hostname}:{parsed.port or 6379} ({parsed.scheme})"
    else:
        host_display = f"{target_host}:{target_port}"

    start_t = time.perf_counter()
    try:
        import redis.asyncio as aioredis
    except ImportError:
        return {
            "service": "Redis Cache",
            "host": host_display,
            "status": "WARN",
            "latency_ms": 0.0,
            "message": "redis.asyncio not installed. In-memory dictionary cache will be active.",
        }

    try:
        if target_url:
            client = aioredis.from_url(
                target_url,
                password=target_password,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
            )
        else:
            client = aioredis.Redis(
                host=target_host,
                port=target_port,
                password=target_password,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5,
            )

        # Probe test
        probe_key = f"preflight_probe_{int(time.time())}"
        await client.ping()
        await client.set(probe_key, "ok", ex=10)
        val = await client.get(probe_key)
        await client.delete(probe_key)
        await client.aclose()
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        if val == "ok":
            return {
                "service": "Redis Cache",
                "host": host_display,
                "status": "PASS",
                "latency_ms": round(elapsed_ms, 2),
                "message": "Connected & probe SET/GET/DEL verified with 10s TTL.",
            }
        else:
            return {
                "service": "Redis Cache",
                "host": host_display,
                "status": "WARN",
                "latency_ms": round(elapsed_ms, 2),
                "message": f"Ping succeeded but probe write/read mismatch (got {val}).",
            }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        return {
            "service": "Redis Cache",
            "host": host_display,
            "status": "FAIL",
            "latency_ms": round(elapsed_ms, 2),
            "message": f"Connection failed: {str(exc)}",
            "remediation": "Verify REDIS_URL format (rediss:// for Upstash) or check if local Redis container is up.",
        }


def check_qdrant(
    url: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    api_key: Optional[str] = None,
    collection_name: str = config.QDRANT_COLLECTION_NAME,
) -> Dict[str, Any]:
    """
    Validates Qdrant Cloud / Local vector store connectivity,
    checks cluster collection list, and inspects vector points count.
    """
    target_url = url or config.QDRANT_URL
    target_host = host or config.QDRANT_HOST
    target_port = port or config.QDRANT_PORT
    target_api_key = api_key or config.QDRANT_API_KEY

    host_display = target_url or f"{target_host}:{target_port}"

    start_t = time.perf_counter()
    try:
        from pipeline.sync_embeddings import get_qdrant_client

        client = get_qdrant_client(
            url=target_url,
            host=target_host,
            port=target_port,
            api_key=target_api_key,
        )
        collections_resp = client.get_collections()
        collection_names = [c.name for c in collections_resp.collections]
        has_collection = collection_name in collection_names
        points_count = 0
        if has_collection:
            info = client.get_collection(collection_name=collection_name)
            points_count = info.points_count or 0

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        if has_collection and points_count > 0:
            return {
                "service": "Qdrant Vector DB",
                "host": host_display,
                "status": "PASS",
                "latency_ms": round(elapsed_ms, 2),
                "message": f"Cluster healthy. Collection '{collection_name}' active with {points_count:,} indexed vector points.",
            }
        elif has_collection:
            return {
                "service": "Qdrant Vector DB",
                "host": host_display,
                "status": "WARN",
                "latency_ms": round(elapsed_ms, 2),
                "message": f"Collection '{collection_name}' exists but contains 0 points. Run 'python pipeline/sync_embeddings.py' to populate.",
            }
        else:
            return {
                "service": "Qdrant Vector DB",
                "host": host_display,
                "status": "WARN",
                "latency_ms": round(elapsed_ms, 2),
                "message": f"Cluster connected, but collection '{collection_name}' not found. Run 'python pipeline/sync_embeddings.py'.",
            }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        return {
            "service": "Qdrant Vector DB",
            "host": host_display,
            "status": "FAIL",
            "latency_ms": round(elapsed_ms, 2),
            "message": f"Connection failed: {str(exc)}",
            "remediation": "Verify QDRANT_URL and QDRANT_API_KEY. On Qdrant Cloud, check if the cluster suspended from 7-day inactivity.",
        }


def check_llm(
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Validates Google Gemini API key and tests query rewriting with Gemini 3.5 Flash-Lite.
    """
    target_key = api_key or config.GEMINI_API_KEY
    target_model = model_name or config.LLM_MODEL

    if not target_key:
        return {
            "service": "Gemini LLM Layer",
            "host": f"Google AI API ({target_model})",
            "status": "WARN",
            "latency_ms": 0.0,
            "message": "GEMINI_API_KEY is not set in environment. System will use heuristic query parser fallback.",
        }

    start_t = time.perf_counter()
    try:
        import importlib

        llm_mod = importlib.import_module("src.14_llm_layer") if "src.14_llm_layer" in sys.modules or os.path.exists(os.path.join(BASE_DIR, "src", "14_llm_layer.py")) else importlib.import_module("14_llm_layer")
        layer = llm_mod.LLMLayer(api_key=target_key, model_name=target_model)
        rewrite_res = layer.rewrite_query("wireless noise cancelling headphones under 50 dollars")
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        if rewrite_res and "rewritten_query" in rewrite_res:
            return {
                "service": "Gemini LLM Layer",
                "host": f"Google AI API ({target_model})",
                "status": "PASS",
                "latency_ms": round(elapsed_ms, 2),
                "message": f"Prompt & query rewrite operational: '{rewrite_res.get('rewritten_query')}'.",
            }
        else:
            return {
                "service": "Gemini LLM Layer",
                "host": f"Google AI API ({target_model})",
                "status": "WARN",
                "latency_ms": round(elapsed_ms, 2),
                "message": "API call succeeded but returned empty structured rewrite.",
            }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        return {
            "service": "Gemini LLM Layer",
            "host": f"Google AI API ({target_model})",
            "status": "WARN",
            "latency_ms": round(elapsed_ms, 2),
            "message": f"LLM test encountered error: {str(exc)}",
            "remediation": "Check GEMINI_API_KEY validity at https://aistudio.google.com/.",
        }


async def check_api_endpoint(base_url: str) -> Dict[str, Any]:
    """
    Hits /v2/health and /metrics on the deployed or local API endpoint.
    """
    clean_url = base_url.rstrip("/")
    start_t = time.perf_counter()
    try:
        import urllib.request

        health_url = f"{clean_url}/v2/health"
        req = urllib.request.Request(
            health_url,
            headers={"User-Agent": "RecSys-PreFlight/2.0"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            status_code = resp.status

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        if status_code == 200 and data.get("status") == "healthy":
            return {
                "service": "FastAPI Service",
                "host": health_url,
                "status": "PASS",
                "latency_ms": round(elapsed_ms, 2),
                "message": f"Service healthy (version={data.get('version')}, ranker={data.get('ranker_loaded')}, items={data.get('n_items')}).",
            }
        else:
            return {
                "service": "FastAPI Service",
                "host": health_url,
                "status": "WARN",
                "latency_ms": round(elapsed_ms, 2),
                "message": f"Endpoint responded with HTTP {status_code}: {data}",
            }
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0
        return {
            "service": "FastAPI Service",
            "host": f"{clean_url}/v2/health",
            "status": "FAIL",
            "latency_ms": round(elapsed_ms, 2),
            "message": f"Health check failed: {str(exc)}",
            "remediation": "If hosted on Render free-tier, note that cold-start can take 30-60 seconds. Retry shortly.",
        }


async def main_async(args: argparse.Namespace) -> int:
    run_all = args.all or not (args.db or args.redis or args.qdrant or args.llm or args.api_url)

    results = []

    if run_all or args.db:
        res = await check_postgres(dsn=args.db_url)
        results.append(res)

    if run_all or args.redis:
        res = await check_redis(url=args.redis_url)
        results.append(res)

    if run_all or args.qdrant:
        res = check_qdrant(url=args.qdrant_url, api_key=args.qdrant_api_key)
        results.append(res)

    if run_all or args.llm:
        res = check_llm(api_key=args.gemini_key)
        results.append(res)

    if args.api_url:
        res = await check_api_endpoint(base_url=args.api_url)
        results.append(res)

    if args.json:
        print(json.dumps(results, indent=2))
        failures = sum(1 for r in results if r["status"] == "FAIL")
        return 1 if failures > 0 else 0

    print_banner()
    has_failure = False

    for r in results:
        status = r["status"]
        if status == "PASS":
            tag = f"{GREEN}[PASS]{RESET}"
        elif status == "WARN":
            tag = f"{YELLOW}[WARN]{RESET}"
        else:
            tag = f"{RED}[FAIL]{RESET}"
            has_failure = True

        print(f"{tag} {BOLD}{r['service']}{RESET} ({r['host']})")
        print(f"      Latency: {r['latency_ms']} ms")
        print(f"      Details: {r['message']}")
        if "remediation" in r:
            print(f"      {YELLOW}Advice:  {r['remediation']}{RESET}")
        print()

    print(f"{BOLD}Summary:{RESET} {len(results)} services checked. Status: {'ALL GREEN' if not has_failure else 'ACTION REQUIRED'}\n")
    return 1 if has_failure else 0


def main():
    parser = argparse.ArgumentParser(
        description="Verify Tier 1 Free Cloud Deployment Connectivity (Neon, Upstash, Qdrant Cloud, Gemini, FastAPI)"
    )
    parser.add_argument("--all", action="store_true", help="Check all Tier 1 cloud services (default)")
    parser.add_argument("--db", action="store_true", help="Check PostgreSQL / Neon database")
    parser.add_argument("--redis", action="store_true", help="Check Redis / Upstash cache")
    parser.add_argument("--qdrant", action="store_true", help="Check Qdrant Cloud vector database")
    parser.add_argument("--llm", action="store_true", help="Check Google Gemini API")
    parser.add_argument("--api-url", type=str, default=None, help="Check live FastAPI deployment URL (e.g. https://amazon-recsys-api.onrender.com)")
    parser.add_argument("--db-url", type=str, default=None, help="Override DATABASE_URL")
    parser.add_argument("--redis-url", type=str, default=None, help="Override REDIS_URL")
    parser.add_argument("--qdrant-url", type=str, default=None, help="Override QDRANT_URL")
    parser.add_argument("--qdrant-api-key", type=str, default=None, help="Override QDRANT_API_KEY")
    parser.add_argument("--gemini-key", type=str, default=None, help="Override GEMINI_API_KEY")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()
    exit_code = asyncio.run(main_async(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
