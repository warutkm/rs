"""
tests/test_api.py
=================
Phase 7 — FastAPI v2 Serving Layer Integration & Unit Test Suite

Tests:
  - GET  /v2/health & GET /health
  - GET  /metrics (p50/p95/p99 latency, request counts, cache hit rate)
  - POST /v2/recommend (warm user personalization & cold-start)
  - GET  /v2/similar/{item_id} (ANN vector similarity)
  - GET  /v2/search (query understanding & hybrid search)
  - POST /v2/events (interaction logging to DB / in-memory buffer)
  - POST /admin/retrain & GET /admin/retrain/status (auth & execution)
  - Tracing (X-Request-ID propagation)
"""

import os
import sys
import json
import pytest
from fastapi.testclient import TestClient

# Setup paths
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
from api.main import app, state


@pytest.fixture(scope="module")
def client():
    """Module-scoped TestClient managing lifespan startup and shutdown."""
    with TestClient(app) as c:
        yield c


def test_health_endpoints(client):
    """Test GET /v2/health status."""
    r = client.get("/v2/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "ranker_loaded" in data
    assert "version" in data
    assert data["version"] == "2.0.0"


def test_metrics_endpoint(client):
    """Test GET /metrics telemetry counters and latency percentiles."""
    # Hit health endpoint first to populate metrics
    client.get("/v2/health")

    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert "total_requests" in data
    assert data["total_requests"] > 0
    assert "requests_per_endpoint" in data
    assert "latency_p50_ms" in data
    assert "latency_p95_ms" in data
    assert "latency_p99_ms" in data
    assert "cache_hit_rate" in data
    assert isinstance(data["cache_hit_rate"], float)


def test_x_request_id_tracing(client):
    """Test X-Request-ID header generation and propagation."""
    # 1. Without header (auto-generated)
    r1 = client.get("/v2/health")
    assert "X-Request-ID" in r1.headers
    assert len(r1.headers["X-Request-ID"]) > 0

    # 2. With client-provided header
    custom_id = "test-custom-trace-uuid-12345"
    r2 = client.get("/v2/health", headers={"X-Request-ID": custom_id})
    assert r2.headers.get("X-Request-ID") == custom_id


def test_recommend_warm_user(client):
    """Test POST /v2/recommend for a known user with personalization."""
    # Pick a user from user_map if loaded, else use fallback
    user_id = list(state.get("user_map", {}).keys())[0] if state.get("user_map") else "AHPI18EE22YZMH5TQ4YNLBAFZJA"
    item_id = list(state.get("item_meta", {}).keys())[0] if state.get("item_meta") else "B08N5WRWNW"

    payload = {
        "user_id": user_id,
        "item_id": item_id,
        "top_k": 5,
    }
    r = client.post("/v2/recommend", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["user_id"] == user_id
    assert "results" in data
    assert isinstance(data["results"], list)
    assert len(data["results"]) <= 5

    if data["results"]:
        first = data["results"][0]
        assert "item_id" in first
        assert "score" in first
        assert "title" in first
        assert "source" in first


def test_recommend_cold_start_user(client):
    """Test POST /v2/recommend for a brand-new unknown user."""
    payload = {
        "user_id": "__UNKNOWN_TEST_USER_99999__",
        "top_k": 3,
    }
    r = client.post("/v2/recommend", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["cold_start"] is True
    assert len(data["results"]) <= 3


def test_recommend_validation_error(client):
    """Test POST /v2/recommend validation error on missing user_id."""
    r = client.post("/v2/recommend", json={"top_k": 5})
    assert r.status_code == 422


def test_similar_items(client):
    """Test GET /v2/similar/{item_id} vector similarity lookup."""
    item_id = list(state.get("product_vecs", {}).keys())[0] if state.get("product_vecs") else "B08N5WRWNW"

    r = client.get(f"/v2/similar/{item_id}?top_k=4")
    assert r.status_code == 200
    data = r.json()
    assert data["item_id"] == item_id
    assert "results" in data
    assert len(data["results"]) <= 4

    if data["results"]:
        assert "item_id" in data["results"][0]
        assert "score" in data["results"][0]


def test_similar_items_404(client):
    """Test GET /v2/similar/{item_id} returns 404 for completely non-existent item."""
    r = client.get("/v2/similar/__TOTALLY_NON_EXISTENT_ITEM_XYZ__")
    assert r.status_code == 404


def test_search_hybrid(client):
    """Test GET /v2/search with query rewriting and hybrid scoring."""
    r = client.get("/v2/search", params={"q": "wireless gaming headphones", "top_k": 5})
    assert r.status_code == 200
    data = r.json()
    assert data["query"] == "wireless gaming headphones"
    assert "rewritten_query" in data
    assert "results" in data
    assert isinstance(data["results"], list)

    if data["results"]:
        first = data["results"][0]
        assert "item_id" in first
        assert "hybrid_score" in first
        assert "emb_score" in first
        assert "bm25_score" in first


def test_events_logging(client):
    """Test POST /v2/events user feedback logging."""
    payload = {
        "user_id": "test_user_pytest_001",
        "item_id": "B08N5WRWNW",
        "event_type": "click",
        "rating": 5.0,
        "metadata": {"source": "unit_test", "page": "home"},
    }
    r = client.post("/v2/events", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "event_id" in data


def test_admin_retrain_auth(client):
    """Test POST /admin/retrain authentication gates."""
    # 1. Without API key -> 401
    r1 = client.post("/admin/retrain", json={"force": False})
    assert r1.status_code == 401

    # 2. With incorrect API key -> 401
    r2 = client.post("/admin/retrain", json={"force": False}, headers={"X-Admin-API-Key": "wrong_key"})
    assert r2.status_code == 401

    # 3. With correct API key -> 200 or 409
    r3 = client.post(
        "/admin/retrain",
        json={"force": False},
        headers={"X-Admin-API-Key": config.ADMIN_API_KEY},
    )
    assert r3.status_code in (200, 409)

    # 4. Status endpoint with correct key
    r4 = client.get(
        "/admin/retrain/status",
        headers={"X-Admin-API-Key": config.ADMIN_API_KEY},
    )
    assert r4.status_code == 200
    assert "status" in r4.json()


def test_category_filter_normalization(client):
    """Test category filter with spaces vs underscores (Video Games vs Video_Games)."""
    r = client.post("/v2/recommend", json={
        "user_id": "guest_cold_start",
        "category_filter": "Video Games",
        "top_k": 5,
    })
    assert r.status_code == 200
    data = r.json()
    assert len(data["results"]) > 0
    for item in data["results"]:
        assert "video" in item["category"].lower()


def test_product_type_satisfaction_ranking(client):
    """Test product_type search with review satisfaction ranking strategy."""
    r = client.post("/v2/recommend", json={
        "user_id": "guest_cold_start",
        "product_type": "keyboard",
        "sort_by": "satisfaction",
        "top_k": 5,
    })
    assert r.status_code == 200
    data = r.json()
    assert len(data["results"]) > 0
    assert data["source"] == "satisfaction_ranker"
    scores = [it["score"] for it in data["results"]]
    assert scores == sorted(scores, reverse=True)


def test_similar_includes_target_item(client):
    """Test GET /v2/similar/{item_id} returns target_item with metadata."""
    sample_id = "B07MFMFW34"
    r = client.get(f"/v2/similar/{sample_id}?top_k=3")
    assert r.status_code == 200
    data = r.json()
    assert "target_item" in data
    assert data["target_item"] is not None
    assert data["target_item"]["item_id"] == sample_id
    assert data["target_item"]["price"] is not None
    assert data["target_item"]["average_rating"] is not None

