"""
Phase 7 — End-to-End API Integration Test Script
File: api/test_api.py

Run against a live service:
    # Local uvicorn
    uvicorn api.main:app --host 0.0.0.0 --port 8000
    python api/test_api.py

Tests:
  1. GET  /v2/health & GET /health
  2. GET  /metrics
  3. POST /v2/recommend — warm user
  4. POST /v2/recommend — cold-start user
  5. POST /v2/recommend — validation error
  6. GET  /v2/similar/{item_id}
  7. GET  /v2/search?q=
  8. POST /v2/events
  9. GET  /admin/retrain/status
"""

import sys
import json
import httpx

BASE_URL = "http://localhost:8000"

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

results = []


def check(name: str, passed: bool, detail: str = ""):
    status = PASS if passed else FAIL
    print(f"  {status}  {name}" + (f"  →  {detail}" if detail else ""))
    results.append((name, passed))


def run_tests():
    print(f"\n{'='*60}")
    print("  DS11 Phase 7 — FastAPI v2 Integration Tests")
    print(f"  Target: {BASE_URL}")
    print(f"{'='*60}\n")

    client = httpx.Client(base_url=BASE_URL, timeout=60.0)

    # 1. GET /v2/health
    print("TEST 1: GET /v2/health")
    try:
        r = client.get("/v2/health")
        check("status code 200", r.status_code == 200, str(r.status_code))
        body = r.json()
        check("status == 'ok'", body.get("status") == "ok", str(body))
        check("model_loaded present", "model_loaded" in body)
        check("ranker_loaded present", "ranker_loaded" in body)
    except Exception as e:
        check("GET /v2/health reachable", False, str(e))

    # 2. GET /metrics
    print("\nTEST 2: GET /metrics")
    try:
        r = client.get("/metrics")
        check("status code 200", r.status_code == 200, str(r.status_code))
        body = r.json()
        check("total_requests in metrics", "total_requests" in body)
        check("latency_p50_ms in metrics", "latency_p50_ms" in body)
        check("cache_hit_rate in metrics", "cache_hit_rate" in body)
    except Exception as e:
        check("GET /metrics reachable", False, str(e))

    # 3. POST /v2/recommend (warm path)
    print("\nTEST 3: POST /v2/recommend — warm user")
    try:
        payload = {
            "item_id": "B08N5WRWNW",
            "user_id": "AHPI18EE22YZMH5TQ4YNLBAFZJA",
            "top_k": 5,
        }
        r = client.post("/v2/recommend", json=payload)
        check("status code 200", r.status_code == 200, str(r.status_code))
        if r.status_code == 200:
            body = r.json()
            check("results list present", "results" in body)
            check("cold_start field", "cold_start" in body)
            results_list = body.get("results", [])
            if results_list:
                first = results_list[0]
                check("result has item_id", "item_id" in first, str(first.get("item_id")))
                check("result has score", "score" in first)
                check("result has source", "source" in first)
                check("result has title", "title" in first)
    except Exception as e:
        check("POST /v2/recommend warm", False, str(e))

    # 4. POST /v2/recommend (cold start)
    print("\nTEST 4: POST /v2/recommend — cold-start user")
    try:
        payload = {
            "user_id": "__BRAND_NEW_USER_COLD_START__",
            "top_k": 5,
        }
        r = client.post("/v2/recommend", json=payload)
        check("status code 200", r.status_code == 200, str(r.status_code))
        if r.status_code == 200:
            body = r.json()
            check("cold_start == True", body.get("cold_start") is True, str(body.get("cold_start")))
            check("results list present", isinstance(body.get("results"), list))
    except Exception as e:
        check("POST /v2/recommend cold-start", False, str(e))

    # 5. POST /v2/recommend (422 validation)
    print("\nTEST 5: POST /v2/recommend — missing required field (422)")
    try:
        r = client.post("/v2/recommend", json={"top_k": 5})
        check("status code 422", r.status_code == 422, str(r.status_code))
    except Exception as e:
        check("422 validation check", False, str(e))

    # 6. GET /v2/similar/{item_id}
    print("\nTEST 6: GET /v2/similar/{item_id}")
    try:
        r = client.get("/v2/similar/B08N5WRWNW?top_k=5")
        check("status code 200 or 404", r.status_code in (200, 404), str(r.status_code))
        if r.status_code == 200:
            body = r.json()
            check("results list present", "results" in body)
            check("correct top_k count", len(body.get("results", [])) <= 5)
    except Exception as e:
        check("GET /v2/similar", False, str(e))

    # 7. GET /v2/search
    print("\nTEST 7: GET /v2/search?q=wireless+headphones")
    try:
        r = client.get("/v2/search", params={"q": "wireless headphones", "top_k": 5})
        check("status code 200", r.status_code == 200, str(r.status_code))
        if r.status_code == 200:
            body = r.json()
            check("query echoed back", body.get("query") == "wireless headphones")
            check("results list present", isinstance(body.get("results"), list))
            if body.get("results"):
                first = body["results"][0]
                check("result has hybrid_score", "hybrid_score" in first)
                check("result has item_id", "item_id" in first)
    except Exception as e:
        check("GET /v2/search", False, str(e))

    # 8. POST /v2/events
    print("\nTEST 8: POST /v2/events")
    try:
        payload = {
            "user_id": "test_script_user",
            "item_id": "B08N5WRWNW",
            "event_type": "click",
            "rating": 5.0,
            "metadata": {"source": "test_script"},
        }
        r = client.post("/v2/events", json=payload)
        check("status code 200", r.status_code == 200, str(r.status_code))
        if r.status_code == 200:
            body = r.json()
            check("status == 'ok'", body.get("status") == "ok", str(body))
    except Exception as e:
        check("POST /v2/events", False, str(e))

    # 9. Summary
    total = len(results)
    passed = sum(1 for _, ok in results if ok)
    failed = total - passed

    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{total} passed   {failed} failed")
    print(f"{'='*60}\n")

    client.close()
    return failed == 0


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)
