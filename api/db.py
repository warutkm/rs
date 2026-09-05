"""
Phase 7 — FastAPI v2 Service
File: api/db.py

Asynchronous PostgreSQL client and connection pool using asyncpg.
Handles user events, click/purchase logging, and interaction tracking.
Provides in-memory fallback if PostgreSQL is offline (e.g. during local tests).
"""

import os
import sys
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

# Setup paths
API_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(API_DIR, ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config

logger = logging.getLogger("api.db")

# Global pool reference and in-memory fallback storage
_pool = None
_in_memory_events: List[Dict[str, Any]] = []


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS events (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(128) NOT NULL,
    item_id VARCHAR(64) NOT NULL,
    event_type VARCHAR(32) NOT NULL,
    rating DOUBLE PRECISION,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_events_user_id ON events(user_id);
CREATE INDEX IF NOT EXISTS idx_events_item_id ON events(item_id);
CREATE INDEX IF NOT EXISTS idx_events_created_at ON events(created_at);
"""


from urllib.parse import urlparse, parse_qs, urlencode, urlunparse


def _parse_and_clean_dsn(raw_dsn: str) -> tuple[str, Optional[Any]]:
    """
    Normalizes PostgreSQL DSN for asyncpg compatibility.
    Strips query parameters like 'sslmode' that cause asyncpg errors
    and maps them to the appropriate asyncpg ssl keyword argument.
    """
    if raw_dsn.startswith("postgres://"):
        raw_dsn = "postgresql://" + raw_dsn[len("postgres://") :]

    parsed = urlparse(raw_dsn)
    qs = parse_qs(parsed.query)

    ssl_setting = None
    if "sslmode" in qs:
        sslmode_val = qs.pop("sslmode", [""])[0].lower()
        if sslmode_val in ("require", "verify-ca", "verify-full"):
            ssl_setting = "require"
        elif sslmode_val in ("disable", "allow", "prefer"):
            ssl_setting = False
    elif "ssl" in qs:
        ssl_val = qs.pop("ssl", [""])[0].lower()
        if ssl_val in ("true", "1", "require"):
            ssl_setting = "require"
        elif ssl_val in ("false", "0", "disable"):
            ssl_setting = False

    # Strip libpq-only parameters that asyncpg does not accept in query strings
    qs.pop("channel_binding", None)
    qs.pop("gssencmode", None)
    qs.pop("target_session_attrs", None)

    # Auto-detect cloud providers that enforce SSL (Neon, Supabase, AWS RDS, Render)
    if ssl_setting is None and parsed.hostname:
        host_lower = parsed.hostname.lower()
        if any(h in host_lower for h in (".neon.tech", ".supabase.co", ".render.com", ".aivencloud.com")):
            ssl_setting = "require"

    clean_query = urlencode(qs, doseq=True)
    clean_dsn = urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            clean_query,
            parsed.fragment,
        )
    )
    return clean_dsn, ssl_setting


async def init_db_pool() -> bool:
    """
    Initializes asyncpg connection pool and verifies table schemas.
    Returns True if connection succeeded, False otherwise.
    """
    global _pool
    try:
        import asyncpg
    except ImportError:
        logger.warning("[DB] asyncpg not installed; using in-memory event storage fallback.")
        return False

    dsn = config.POSTGRES_URL
    if not dsn:
        dsn = (
            f"postgresql://{config.POSTGRES_USER}:{config.POSTGRES_PASSWORD}@"
            f"{config.POSTGRES_HOST}:{config.POSTGRES_PORT}/{config.POSTGRES_DB}"
        )

    try:
        clean_dsn, ssl_param = _parse_and_clean_dsn(dsn)
        logger.info(f"[DB] Connecting to PostgreSQL (host={config.POSTGRES_HOST}, ssl={ssl_param}) ...")
        pool_kwargs = {
            "dsn": clean_dsn,
            "min_size": 1,
            "max_size": 10,
            "command_timeout": 10,
            "timeout": 5,
        }
        if ssl_param is not None:
            pool_kwargs["ssl"] = ssl_param

        _pool = await asyncpg.create_pool(**pool_kwargs)
        # Verify schema
        async with _pool.acquire() as conn:
            await conn.execute(CREATE_TABLE_SQL)
        logger.info("[DB] PostgreSQL pool initialized and 'events' schema verified.")
        return True
    except Exception as e:
        logger.warning(f"[DB] PostgreSQL unavailable ({e}); running with in-memory event fallback.")
        _pool = None
        return False


async def close_db_pool():
    """Closes the asyncpg connection pool on app shutdown."""
    global _pool
    if _pool is not None:
        try:
            await _pool.close()
            logger.info("[DB] PostgreSQL connection pool closed.")
        except Exception as e:
            logger.warning(f"[DB] Error closing PostgreSQL pool: {e}")
        finally:
            _pool = None


def get_db_pool():
    """Returns active asyncpg connection pool or None."""
    return _pool


async def check_db_health() -> bool:
    """Checks if PostgreSQL database connection is alive."""
    if _pool is None:
        return False
    try:
        async with _pool.acquire() as conn:
            val = await conn.fetchval("SELECT 1")
            return val == 1
    except Exception:
        return False


async def log_event(
    user_id: str,
    item_id: str,
    event_type: str,
    rating: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Logs an interaction event (click, purchase, rating, view) to PostgreSQL or in-memory fallback.
    `item_id` MUST be parent_asin per GEMINI.md hard rules.
    """
    meta_json = json.dumps(metadata or {})
    created_at = datetime.now(timezone.utc)

    if _pool is not None:
        try:
            query = """
            INSERT INTO events (user_id, item_id, event_type, rating, metadata, created_at)
            VALUES ($1, $2, $3, $4, $5::jsonb, $6)
            RETURNING id;
            """
            async with _pool.acquire() as conn:
                event_id = await conn.fetchval(query, user_id, str(item_id), event_type, rating, meta_json, created_at)
            return {
                "status": "ok",
                "event_id": event_id,
                "storage": "postgres",
            }
        except Exception as e:
            logger.error(f"[DB] Failed to insert event into PostgreSQL ({e}); falling back to memory.")

    # In-memory storage fallback
    event_entry = {
        "id": len(_in_memory_events) + 1,
        "user_id": user_id,
        "item_id": str(item_id),
        "event_type": event_type,
        "rating": rating,
        "metadata": metadata or {},
        "created_at": created_at.isoformat(),
    }
    _in_memory_events.append(event_entry)
    if len(_in_memory_events) > 5000:
        _in_memory_events.pop(0)

    return {
        "status": "ok",
        "event_id": event_entry["id"],
        "storage": "memory",
    }


async def get_user_events(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves the most recent interaction events for a user."""
    if _pool is not None:
        try:
            query = """
            SELECT id, user_id, item_id, event_type, rating, metadata, created_at
            FROM events
            WHERE user_id = $1
            ORDER BY created_at DESC
            LIMIT $2;
            """
            async with _pool.acquire() as conn:
                rows = await conn.fetch(query, user_id, limit)
            return [
                {
                    "id": r["id"],
                    "user_id": r["user_id"],
                    "item_id": r["item_id"],
                    "event_type": r["event_type"],
                    "rating": r["rating"],
                    "metadata": json.loads(r["metadata"]) if isinstance(r["metadata"], str) else (r["metadata"] or {}),
                    "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                }
                for r in rows
            ]
        except Exception as e:
            logger.warning(f"[DB] Error fetching user events from PostgreSQL: {e}")

    # Fallback from in-memory store
    matches = [e for e in reversed(_in_memory_events) if e["user_id"] == user_id]
    return matches[:limit]
