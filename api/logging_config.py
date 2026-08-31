"""
Phase 7 / Phase 10 — Observability & Telemetry
File: api/logging_config.py

Structured JSON logging formatter and API metrics collector.
Provides real-time p50/p95/p99 latency tracking, requests-per-endpoint counters,
and cache hit-rate telemetry for the GET /metrics endpoint.
"""

import sys
import json
import logging
from datetime import datetime, timezone
from collections import deque
from typing import Dict, Any
import numpy as np

from api.cache import get_cache_stats


# =============================================================================
# STRUCTURED JSON LOGGING FORMATTER
# =============================================================================


class JSONFormatter(logging.Formatter):
    """
    Formatter that outputs structured JSON log entries.
    Standardized fields: timestamp, level, name, message, request_id, client_ip, method, path, status_code, latency_ms.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Inject extra request/response attributes if present
        for field in ("request_id", "client_ip", "method", "path", "status_code", "latency_ms"):
            val = getattr(record, field, None)
            if val is not None:
                log_obj[field] = val

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj)


def configure_logging(level: int = logging.INFO):
    """Configures structured JSON logging on the root and API loggers."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())

    root_logger = logging.getLogger()
    # Avoid duplicate handlers on reload
    root_logger.handlers = [handler]
    root_logger.setLevel(level)

    # Silence overly verbose third-party loggers
    for noisy in ("uvicorn.access", "httpx", "urllib3", "qdrant_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# =============================================================================
# METRICS COLLECTOR (JSON /metrics ENDPOINT)
# =============================================================================


class MetricsCollector:
    """
    In-process telemetry and metrics tracker.
    Computes rolling latency percentiles (p50, p95, p99) and request counters.
    """

    def __init__(self, window_size: int = 5000):
        self.window_size = window_size
        self.latencies: deque = deque(maxlen=window_size)
        self.endpoint_counts: Dict[str, int] = {}
        self.status_counts: Dict[str, int] = {}
        self.total_requests: int = 0

    def record_request(self, endpoint: str, latency_ms: float, status_code: int):
        """Record an incoming request observation."""
        self.total_requests += 1
        self.latencies.append(latency_ms)

        # Normalize endpoint for metrics grouping
        ep_key = endpoint.split("?")[0]
        self.endpoint_counts[ep_key] = self.endpoint_counts.get(ep_key, 0) + 1

        sc_key = str(status_code)
        self.status_counts[sc_key] = self.status_counts.get(sc_key, 0) + 1

    def get_metrics(self) -> Dict[str, Any]:
        """Calculates current telemetry counters and latency percentiles."""
        if self.latencies:
            arr = np.array(self.latencies, dtype=np.float64)
            p50 = float(np.percentile(arr, 50))
            p95 = float(np.percentile(arr, 95))
            p99 = float(np.percentile(arr, 99))
        else:
            p50, p95, p99 = 0.0, 0.0, 0.0

        cache_stats = get_cache_stats()

        return {
            "status": "healthy",
            "total_requests": self.total_requests,
            "requests_per_endpoint": dict(self.endpoint_counts),
            "requests_per_status": dict(self.status_counts),
            "latency_p50_ms": round(p50, 2),
            "latency_p95_ms": round(p95, 2),
            "latency_p99_ms": round(p99, 2),
            "cache_hits": cache_stats["cache_hits"],
            "cache_misses": cache_stats["cache_misses"],
            "cache_hit_rate": cache_stats["cache_hit_rate"],
        }

    def reset(self):
        """Reset all recorded latencies and request counters."""
        self.latencies.clear()
        self.endpoint_counts.clear()
        self.status_counts.clear()
        self.total_requests = 0


# Global metrics collector singleton
metrics_collector = MetricsCollector()

