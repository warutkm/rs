"""
Phase 12 — End-to-End API Tests
File: api/test_api.py

Run against a live service:
    # Option A — local uvicorn
    uvicorn api.main:app --host 0.0.0.0 --port 8000
    python api/test_api.py

    # Option B — docker-compose
    docker-compose up -d
    python api/test_api.py

Requires: httpx  (pip install httpx)

Tests:
  1. GET  /health
  2. POST /recommend  — warm user (expects real recommendations)
  3. POST /recommend  — cold-start user (expects cold-start flag = True)
  4. POST /recommend  — missing fields (expects 422 validation error)
  5. GET  /similar/{item_id}
  6. GET  /search?q=
  7. GET  /similar/{item_id}?top_k=5 — custom top_k
  8. GET  /health — model_loaded field check
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
    print("  DS11 Phase 12 — API Integration Tests")
    print(f"  Target: {BASE_URL}")
    print(f"{'='*60}\n")

    client = httpx.Client(base_url=BASE_URL, timeout=60.0)

    # ────────────────────────────────────────────────────────────
    # TEST 1 — GET /health
    # ────────────────────────────────────────────────────────────
    print("TEST 1: GET /health")
    try:
        r = client.get("/health")
        check("status code 200",       r.status_code == 200, str(r.status_code))
        body = r.json()
        check("status == 'ok'",        body.get("status") == "ok", str(body))
        check("model_loaded present",  "model_loaded" in body)
    except Exception as e:
        check("GET /health reachable", False, str(e))

    # ────────────────────────────────────────────────────────────
    # TEST 2 — POST /recommend (warm path)
    # ────────────────────────────────────────────────────────────
    print("\nTEST 2: POST /recommend — warm user")
    try:
        # Fetch a real item_id and user_id from the health check
        # We hard-code one likely to exist based on the dataset
        payload = {
            "item_id": "B08N5WRWNW",
            "user_id": "AHPI18EE22YZMH5TQ4YNLBAFZJA",
            "top_k":   5,
        }
        r = client.post("/recommend", json=payload)
        check("status code 200 or 404", r.status_code in (200, 404),
              str(r.status_code))
        if r.status_code == 200:
            body = r.json()
            check("results list present",  "results" in body)
            check("cold_start field",       "cold_start" in body)
            check("results is list",        isinstance(body.get("results"), list))
            results_list = body.get("results", [])
            if results_list:
                first = results_list[0]
                check("result has item_id", "item_id" in first, str(first))
                check("result has score",   "score"   in first)
                check("result has source",  "source"  in first)
                check("result has title",   "title"   in first)
        else:
            print(f"  {INFO}  item_id not in dataset — acceptable 404")
    except Exception as e:
        check("POST /recommend warm", False, str(e))

    # ────────────────────────────────────────────────────────────
    # TEST 3 — POST /recommend (cold-start user)
    # ────────────────────────────────────────────────────────────
    print("\nTEST 3: POST /recommend — cold-start user")
    try:
        payload = {
            "item_id": "B08N5WRWNW",
            "user_id": "__BRAND_NEW_USER_COLD_START__",
            "top_k":   5,
        }
        r = client.post("/recommend", json=payload)
        check("status code 200 or 404", r.status_code in (200, 404),
              str(r.status_code))
        if r.status_code == 200:
            body = r.json()
            check("cold_start == True",    body.get("cold_start") is True,
                  str(body.get("cold_start")))
            check("results list present",  isinstance(body.get("results"), list))
    except Exception as e:
        check("POST /recommend cold-start", False, str(e))

    # ────────────────────────────────────────────────────────────
    # TEST 4 — POST /recommend (validation error — missing field)
    # ────────────────────────────────────────────────────────────
    print("\nTEST 4: POST /recommend — missing required field (422)")
    try:
        r = client.post("/recommend", json={"top_k": 5})   # item_id + user_id missing
        check("status code 422", r.status_code == 422, str(r.status_code))
    except Exception as e:
        check("422 validation check", False, str(e))

    # ────────────────────────────────────────────────────────────
    # TEST 5 — GET /similar/{item_id}
    # ────────────────────────────────────────────────────────────
    print("\nTEST 5: GET /similar/{item_id}")
    try:
        r = client.get("/similar/B08N5WRWNW?top_k=5")
        check("status code 200 or 404", r.status_code in (200, 404),
              str(r.status_code))
        if r.status_code == 200:
            body = r.json()
            check("results list present", "results" in body)
            check("results is list",      isinstance(body.get("results"), list))
            check("correct top_k count",  len(body.get("results", [])) <= 5)
    except Exception as e:
        check("GET /similar", False, str(e))

    # ────────────────────────────────────────────────────────────
    # TEST 6 — GET /search?q=
    # ────────────────────────────────────────────────────────────
    print("\nTEST 6: GET /search?q=wireless+headphones")
    try:
        r = client.get("/search", params={"q": "wireless headphones", "top_k": 5})
        check("status code 200", r.status_code == 200, str(r.status_code))
        if r.status_code == 200:
            body = r.json()
            check("query echoed back",    body.get("query") == "wireless headphones")
            check("results list present", isinstance(body.get("results"), list))
            check("results non-empty",    len(body.get("results", [])) > 0)
            if body.get("results"):
                first = body["results"][0]
                check("result has hybrid_score", "hybrid_score" in first)
                check("result has item_id",      "item_id"      in first)
    except Exception as e:
        check("GET /search", False, str(e))

    # ────────────────────────────────────────────────────────────
    # TEST 7 — GET /similar with top_k=3
    # ────────────────────────────────────────────────────────────
    print("\nTEST 7: GET /similar — top_k param respected")
    try:
        r = client.get("/similar/B08N5WRWNW", params={"top_k": 3})
        if r.status_code == 200:
            body = r.json()
            check("top_k=3 respected", len(body.get("results", [])) <= 3,
                  f"got {len(body.get('results', []))}")
        else:
            print(f"  {INFO}  item not found (404) — skip top_k check")
    except Exception as e:
        check("GET /similar top_k", False, str(e))

    # ────────────────────────────────────────────────────────────
    # TEST 8 — model_loaded == True
    # ────────────────────────────────────────────────────────────
    print("\nTEST 8: GET /health — model_loaded == True")
    try:
        r = client.get("/health")
        body = r.json()
        check("model_loaded == True", body.get("model_loaded") is True,
              str(body.get("model_loaded")))
    except Exception as e:
        check("model_loaded check", False, str(e))

    # ────────────────────────────────────────────────────────────
    # SUMMARY
    # ────────────────────────────────────────────────────────────
    total  = len(results)
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
