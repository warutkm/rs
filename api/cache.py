"""
Phase 7 — FastAPI v2 Service
File: api/cache.py

Asynchronous Redis caching layer using redis.asyncio.
Handles:
  1. Response caching for recommendations and search queries.
  2. LLM explanation caching keyed by (user_id, item_id, model_version).
  3. Real-time cache hit/miss accounting and telemetry.
  4. In-memory dictionary fallback with TTL when Redis is offline.
"""

import os
import sys
import json
import time
import logging
from typing import Dict, Optional, Any

# Setup paths
API_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(API_DIR, ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config

logger = logging.getLogger("api.cache")

# Async Redis client instance
_redis_client = None

# Telemetry counters
_cache_hits = 0
_cache_misses = 0

# In-memory fallback dictionary: key -> (value, expire_timestamp)
_memory_cache: Dict[str, tuple[Any, float]] = {}


def _cleanup_memory_cache():
    """Removes expired items from memory cache."""
    now = time.time()
    expired = [k for k, (_, exp) in _memory_cache.items() if exp < now]
    for k in expired:
        _memory_cache.pop(k, None)


async def init_redis_pool() -> bool:
    """Initializes async Redis connection pool."""
    global _redis_client
    try:
        import redis.asyncio as aioredis
    except ImportError:
        logger.warning("[Cache] redis.asyncio not available; using in-memory cache fallback.")
        return False

    url = config.REDIS_URL
    try:
        if url:
            _redis_client = aioredis.from_url(
                url,
                password=config.REDIS_PASSWORD,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=3,
                socket_connect_timeout=3,
            )
        else:
            _redis_client = aioredis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                password=config.REDIS_PASSWORD,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=3,
                socket_connect_timeout=3,
            )
        # Test connection
        await _redis_client.ping()
        logger.info(f"[Cache] Connected to Redis at {config.REDIS_HOST}:{config.REDIS_PORT}.")
        return True
    except Exception as e:
        logger.warning(f"[Cache] Redis unavailable ({e}); running with in-memory cache fallback.")
        _redis_client = None
        return False


async def close_redis_pool():
    """Closes Redis connection pool on app shutdown."""
    global _redis_client
    if _redis_client is not None:
        try:
            await _redis_client.aclose()
            logger.info("[Cache] Redis connection pool closed.")
        except Exception as e:
            logger.warning(f"[Cache] Error closing Redis connection: {e}")
        finally:
            _redis_client = None


def get_redis():
    """Returns active async Redis client or None."""
    return _redis_client


async def check_redis_health() -> bool:
    """Checks if Redis server is alive."""
    if _redis_client is None:
        return False
    try:
        return bool(await _redis_client.ping())
    except Exception:
        return False


def record_hit():
    """Increment cache hit counter."""
    global _cache_hits
    _cache_hits += 1


def record_miss():
    """Increment cache miss counter."""
    global _cache_misses
    _cache_misses += 1


def get_cache_stats() -> Dict[str, Any]:
    """Returns cache hits, misses, and hit rate percentage."""
    total = _cache_hits + _cache_misses
    hit_rate = float(_cache_hits / total) if total > 0 else 0.0
    return {
        "cache_hits": _cache_hits,
        "cache_misses": _cache_misses,
        "total_lookups": total,
        "cache_hit_rate": round(hit_rate, 4),
    }


def format_explanation_key(user_id: str, item_id: str, model_version: str = config.MODEL_VERSION) -> str:
    """Generate cache key for item explanation."""
    return f"explanation:{user_id}:{item_id}:{model_version}"


async def get_cached_explanation(
    user_id: str,
    item_id: str,
    model_version: str = config.MODEL_VERSION,
) -> Optional[Dict[str, Any]]:
    """Retrieves cached LLM explanation for (user_id, item_id, model_version)."""
    key = format_explanation_key(user_id, item_id, model_version)

    # 1. Try Redis
    if _redis_client is not None:
        try:
            val = await _redis_client.get(key)
            if val is not None:
                record_hit()
                return json.loads(val)
            record_miss()
            return None
        except Exception as e:
            logger.warning(f"[Cache] Redis read error ({e}); falling back to memory.")

    # 2. In-memory fallback
    _cleanup_memory_cache()
    if key in _memory_cache:
        data, exp = _memory_cache[key]
        if exp > time.time():
            record_hit()
            return data
        else:
            _memory_cache.pop(key, None)

    record_miss()
    return None


async def set_cached_explanation(
    user_id: str,
    item_id: str,
    data: Dict[str, Any],
    model_version: str = config.MODEL_VERSION,
    ttl: int = config.EXPLANATION_CACHE_TTL,
):
    """Caches LLM explanation for (user_id, item_id, model_version)."""
    key = format_explanation_key(user_id, item_id, model_version)
    val_json = json.dumps(data)

    if _redis_client is not None:
        try:
            await _redis_client.set(key, val_json, ex=ttl)
            return
        except Exception as e:
            logger.warning(f"[Cache] Redis set error ({e}); falling back to memory.")

    # In-memory fallback
    _memory_cache[key] = (data, time.time() + ttl)


async def get_cached_response(cache_key: str) -> Optional[Any]:
    """Retrieves generic cached API response."""
    if _redis_client is not None:
        try:
            val = await _redis_client.get(cache_key)
            if val is not None:
                record_hit()
                return json.loads(val)
            record_miss()
            return None
        except Exception as e:
            logger.warning(f"[Cache] Redis error: {e}")

    _cleanup_memory_cache()
    if cache_key in _memory_cache:
        data, exp = _memory_cache[cache_key]
        if exp > time.time():
            record_hit()
            return data
        else:
            _memory_cache.pop(cache_key, None)

    record_miss()
    return None


async def set_cached_response(cache_key: str, data: Any, ttl: int = 60):
    """Caches API response payload for given TTL (seconds)."""
    val_json = json.dumps(data)
    if _redis_client is not None:
        try:
            await _redis_client.set(cache_key, val_json, ex=ttl)
            return
        except Exception as e:
            logger.warning(f"[Cache] Redis error: {e}")

    _memory_cache[cache_key] = (data, time.time() + ttl)
