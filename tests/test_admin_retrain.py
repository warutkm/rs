"""
tests/test_admin_retrain.py
Unit and API integration tests for RetrainManager and /admin/retrain endpoints.
"""

import os
import sys
import time
import pytest
from fastapi.testclient import TestClient

from api.main import app, retrain_manager
from api.retrain_manager import RetrainManager
from config import ADMIN_API_KEY


@pytest.fixture
def client():
    """Create FastAPI test client."""
    return TestClient(app)


# =============================================================================
# RetrainManager UNIT TESTS
# =============================================================================


def test_retrain_manager_lifecycle(tmp_path):
    """Test RetrainManager execution lifecycle with a dummy command."""
    log_file = os.path.join(tmp_path, "test_retrain.log")
    mgr = RetrainManager(log_path=log_file, base_dir=str(tmp_path))

    assert mgr.status == "idle"
    assert not mgr.is_running()

    # Trigger short sleep command
    cmd = [sys.executable, "-c", "import time; print('start'); time.sleep(0.4); print('done')"]
    res = mgr.trigger(custom_cmd=cmd)

    assert res["success"] is True
    assert res["status"] == "triggered"
    assert res["job_id"] is not None
    assert mgr.is_running() is True

    # Attempt concurrent trigger while running -> should reject
    res_conflict = mgr.trigger(custom_cmd=cmd)
    assert res_conflict["success"] is False
    assert res_conflict["status"] == "in_progress"

    # Wait for completion
    time.sleep(0.8)

    assert mgr.is_running() is False
    assert mgr.status == "completed"
    assert mgr.return_code == 0

    status = mgr.get_status()
    assert status["status"] == "completed"
    assert status["return_code"] == 0
    assert "start" in status["log_tail"]
    assert "done" in status["log_tail"]


def test_retrain_manager_failure_handling(tmp_path):
    """Test RetrainManager captures failure exit codes cleanly."""
    log_file = os.path.join(tmp_path, "test_failure.log")
    mgr = RetrainManager(log_path=log_file, base_dir=str(tmp_path))

    cmd = [sys.executable, "-c", "import sys; print('failing now'); sys.exit(42)"]
    res = mgr.trigger(custom_cmd=cmd)

    assert res["success"] is True
    time.sleep(0.5)

    assert mgr.status == "failed"
    assert mgr.return_code == 42
    status = mgr.get_status()
    assert status["status"] == "failed"
    assert status["return_code"] == 42
    assert "failing now" in status["log_tail"]


# =============================================================================
# /admin/retrain API ENDPOINT TESTS
# =============================================================================


def test_admin_retrain_unauthorized(client):
    """POST /admin/retrain without API key must return 401."""
    r = client.post("/admin/retrain")
    assert r.status_code == 401
    assert "Invalid or missing admin API key" in r.json()["detail"]


def test_admin_retrain_invalid_key(client):
    """POST /admin/retrain with incorrect API key must return 401."""
    headers = {"X-Admin-API-Key": "completely_wrong_key"}
    r = client.post("/admin/retrain", headers=headers)
    assert r.status_code == 401


def test_admin_status_unauthorized(client):
    """GET /admin/retrain/status without API key must return 401."""
    r = client.get("/admin/retrain/status")
    assert r.status_code == 401


def test_admin_retrain_authorized_headers(client, monkeypatch, tmp_path):
    """Test /admin/retrain with X-Admin-API-Key, X-API-Key, and Bearer token."""
    # Point retrain_manager to temp log for test
    test_log = os.path.join(tmp_path, "api_retrain.log")
    monkeypatch.setattr(retrain_manager, "log_path", test_log)

    # 1. Header: X-Admin-API-Key
    headers = {"X-Admin-API-Key": ADMIN_API_KEY}
    r = client.get("/admin/retrain/status", headers=headers)
    assert r.status_code == 200
    assert "status" in r.json()

    # 2. Header: X-API-Key
    headers_alt = {"X-API-Key": ADMIN_API_KEY}
    r = client.get("/admin/retrain/status", headers=headers_alt)
    assert r.status_code == 200

    # 3. Header: Authorization: Bearer <key>
    headers_bearer = {"Authorization": f"Bearer {ADMIN_API_KEY}"}
    r = client.get("/admin/retrain/status", headers=headers_bearer)
    assert r.status_code == 200


def test_admin_retrain_trigger_and_conflict(client, monkeypatch, tmp_path):
    """Test triggering a retrain via API and verifying 409 Conflict on overlap."""
    test_log = os.path.join(tmp_path, "api_conflict.log")
    monkeypatch.setattr(retrain_manager, "log_path", test_log)

    # Mock trigger with a short running command
    cmd = [sys.executable, "-c", "import time; time.sleep(0.5)"]
    headers = {"X-Admin-API-Key": ADMIN_API_KEY}

    original_trigger = retrain_manager.trigger

    # Custom trigger helper
    def mock_trigger(force=False, targets=None, custom_cmd=None):
        return original_trigger(force=force, targets=targets, custom_cmd=cmd)

    monkeypatch.setattr(retrain_manager, "trigger", mock_trigger)

    # 1st request -> 200 Triggered
    r1 = client.post("/admin/retrain", json={"force": True}, headers=headers)
    assert r1.status_code == 200
    body1 = r1.json()
    assert body1["status"] == "triggered"
    assert body1["job_id"] is not None

    # 2nd request immediately -> 409 Conflict
    r2 = client.post("/admin/retrain", json={"force": False}, headers=headers)
    assert r2.status_code == 409
    assert "already in progress" in r2.json()["detail"]

    # Wait for process to finish
    time.sleep(0.8)

    # Status check
    r_status = client.get("/admin/retrain/status", headers=headers)
    assert r_status.status_code == 200
    assert r_status.json()["status"] == "completed"
