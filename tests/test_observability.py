"""
tests/test_observability.py
===========================
Phase 10 — Observability & Tier 0 Local Docker Stack Test Suite

Tests:
  1. Structured JSON Logging Formatter (JSONFormatter)
  2. Telemetry & MetricsCollector (p50/p95/p99 latency calculation, counters, cache stats)
  3. GET /metrics Endpoint (schema validation, real-time counter updates)
  4. Multi-Service Health Endpoints (GET /v2/health and alias GET /health)
  5. Docker Compose Configuration Integrity (all Tier 0 services, healthchecks, networks, volumes)
"""

import os
import sys
import json
import logging
import yaml
import pytest
from fastapi.testclient import TestClient

# Setup paths
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from api.main import app
from api.logging_config import JSONFormatter, MetricsCollector
from api.cache import record_hit, record_miss, reset_cache_stats


@pytest.fixture(scope="module")
def client():
    """Module-scoped TestClient managing lifespan startup and shutdown."""
    with TestClient(app) as c:
        yield c


# =============================================================================
# 1. STRUCTURED JSON LOGGING TESTS
# =============================================================================


def test_json_formatter_standard_record():
    """Verify JSONFormatter outputs valid JSON with standardized schema."""
    formatter = JSONFormatter()
    logger = logging.getLogger("test.logger")

    record = logger.makeRecord(
        name="test.logger",
        level=logging.INFO,
        fn="test_observability.py",
        lno=42,
        msg="Service request completed",
        args=(),
        exc_info=None,
        extra={
            "request_id": "req-uuid-test-1234",
            "client_ip": "127.0.0.1",
            "method": "POST",
            "path": "/v2/recommend",
            "status_code": 200,
            "latency_ms": 14.52,
        },
    )

    output = formatter.format(record)
    log_data = json.loads(output)

    assert log_data["level"] == "INFO"
    assert log_data["logger"] == "test.logger"
    assert log_data["message"] == "Service request completed"
    assert log_data["request_id"] == "req-uuid-test-1234"
    assert log_data["client_ip"] == "127.0.0.1"
    assert log_data["method"] == "POST"
    assert log_data["path"] == "/v2/recommend"
    assert log_data["status_code"] == 200
    assert log_data["latency_ms"] == 14.52
    assert "timestamp" in log_data


def test_json_formatter_exception_handling():
    """Verify JSONFormatter includes serialized traceback on exceptions."""
    formatter = JSONFormatter()
    logger = logging.getLogger("test.error.logger")

    try:
        raise ValueError("Simulated connection timeout")
    except ValueError:
        exc_info = sys.exc_info()

    record = logger.makeRecord(
        name="test.error.logger",
        level=logging.ERROR,
        fn="test_observability.py",
        lno=80,
        msg="Database query failure",
        args=(),
        exc_info=exc_info,
    )

    output = formatter.format(record)
    log_data = json.loads(output)

    assert log_data["level"] == "ERROR"
    assert "exception" in log_data
    assert "ValueError: Simulated connection timeout" in log_data["exception"]


# =============================================================================
# 2. METRICS COLLECTOR UNIT TESTS
# =============================================================================


def test_metrics_collector_latency_percentiles():
    """Verify MetricsCollector calculates accurate p50, p95, and p99 percentiles."""
    collector = MetricsCollector(window_size=1000)
    collector.reset()
    reset_cache_stats()

    # Record 100 uniform samples from 1.0ms to 100.0ms
    for ms in range(1, 101):
        collector.record_request("/v2/recommend", float(ms), 200)

    m = collector.get_metrics()
    assert m["total_requests"] == 100
    assert m["requests_per_endpoint"]["/v2/recommend"] == 100
    assert m["requests_per_status"]["200"] == 100

    # Percentiles for 1..100: p50 ≈ 50.5, p95 ≈ 95.05, p99 ≈ 99.01
    assert 49.0 <= m["latency_p50_ms"] <= 52.0
    assert 94.0 <= m["latency_p95_ms"] <= 96.0
    assert 98.0 <= m["latency_p99_ms"] <= 100.0


def test_metrics_collector_empty_state():
    """Verify MetricsCollector gracefully returns zeros when no requests recorded."""
    collector = MetricsCollector()
    collector.reset()
    reset_cache_stats()

    m = collector.get_metrics()
    assert m["total_requests"] == 0
    assert m["latency_p50_ms"] == 0.0
    assert m["latency_p95_ms"] == 0.0
    assert m["latency_p99_ms"] == 0.0
    assert m["requests_per_endpoint"] == {}
    assert m["requests_per_status"] == {}


def test_metrics_collector_cache_telemetry_integration():
    """Verify MetricsCollector incorporates cache hit/miss rates."""
    collector = MetricsCollector()
    reset_cache_stats()

    # Simulate 3 hits and 1 miss (75% hit rate)
    record_hit()
    record_hit()
    record_hit()
    record_miss()

    m = collector.get_metrics()
    assert m["cache_hits"] == 3
    assert m["cache_misses"] == 1
    assert m["cache_hit_rate"] == 0.75


# =============================================================================
# 3. GET /metrics & /health INTEGRATION TESTS
# =============================================================================


def test_metrics_endpoint_response_schema(client):
    """Test GET /metrics returns HTTP 200 with full telemetry structure."""
    # Issue requests to populate endpoint counters
    client.get("/v2/health")
    client.get("/health")

    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()

    assert data["status"] == "healthy"
    assert isinstance(data["total_requests"], int)
    assert data["total_requests"] >= 2
    assert "/v2/health" in data["requests_per_endpoint"] or "/health" in data["requests_per_endpoint"]
    assert "200" in data["requests_per_status"]
    assert "latency_p50_ms" in data
    assert "latency_p95_ms" in data
    assert "latency_p99_ms" in data
    assert "cache_hit_rate" in data


def test_health_and_alias_endpoints(client):
    """Test both GET /v2/health and GET /health alias succeed with matching schema."""
    r_v2 = client.get("/v2/health")
    assert r_v2.status_code == 200
    data_v2 = r_v2.json()

    r_alias = client.get("/health")
    assert r_alias.status_code == 200
    data_alias = r_alias.json()

    assert data_v2["status"] == "ok"
    assert data_alias["status"] == "ok"
    assert data_v2["version"] == data_alias["version"] == "2.0.0"
    for key in ("model_loaded", "ranker_loaded", "vector_db_connected", "redis_connected", "db_connected"):
        assert key in data_v2
        assert key in data_alias


# =============================================================================
# 4. DOCKER COMPOSE CONFIGURATION & TIER 0 HARDENING TESTS
# =============================================================================


def test_docker_compose_tier0_stack_definition():
    """Verify docker-compose.yml defines complete Tier 0 service stack."""
    compose_path = os.path.join(BASE_DIR, "docker-compose.yml")
    assert os.path.exists(compose_path), "docker-compose.yml must exist at repo root"

    with open(compose_path, "r", encoding="utf-8") as f:
        compose = yaml.safe_load(f)

    services = compose.get("services", {})

    # 1. Tier 0 Required Services
    required_services = ["api", "postgres", "redis", "qdrant", "web"]
    for svc in required_services:
        assert svc in services, f"Service '{svc}' missing from docker-compose.yml"

    # 2. No Prometheus / Grafana Containers (hard rule)
    forbidden_services = ["prometheus", "grafana"]
    for forbidden in forbidden_services:
        assert forbidden not in services, f"Forbidden monitoring service '{forbidden}' found in docker-compose.yml"

    # 3. Service Configurations & Healthchecks
    for svc_name in ["api", "postgres", "redis", "qdrant"]:
        svc_cfg = services[svc_name]
        assert "healthcheck" in svc_cfg, f"Service '{svc_name}' must have a healthcheck"
        assert "restart" in svc_cfg, f"Service '{svc_name}' must have a restart policy"
        assert "networks" in svc_cfg, f"Service '{svc_name}' must attach to network"

    # 4. API Service Invariants
    api_cfg = services["api"]
    api_ports = api_cfg.get("ports", [])
    assert any("8000" in str(p) for p in api_ports), "API service must expose port 8000"
    assert "depends_on" in api_cfg
    assert "postgres" in api_cfg["depends_on"]
    assert "redis" in api_cfg["depends_on"]
    assert "qdrant" in api_cfg["depends_on"]

    # 5. Web Frontend Service Invariants
    web_cfg = services["web"]
    web_ports = web_cfg.get("ports", [])
    assert any("3000" in str(p) for p in web_ports), "Web service must expose port 3000"
    assert "depends_on" in web_cfg
    assert "api" in web_cfg["depends_on"]

    # 6. Volumes & Networks
    volumes = compose.get("volumes", {})
    assert "postgres_data" in volumes
    assert "redis_data" in volumes
    assert "qdrant_data" in volumes

    networks = compose.get("networks", {})
    assert "recsys_net" in networks
