"""
tests/test_tier1_deployment.py
Phase 11 — Tier 1 Free Cloud Deployment Configuration & Manifest Verification Tests
"""

import os
import sys
import json
import pytest
import yaml
from unittest.mock import patch, MagicMock, AsyncMock

# Add project root to sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from api.db import _parse_and_clean_dsn
import scripts.verify_tier1_connectivity as preflight


def test_render_yaml_structure_and_validity():
    """Verify render.yaml exists, parses as valid YAML, and meets Render Blueprint specification."""
    render_yaml_path = os.path.join(BASE_DIR, "render.yaml")
    assert os.path.exists(render_yaml_path), "render.yaml blueprint manifest is missing"

    with open(render_yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert "services" in config, "render.yaml must define a 'services' list"
    assert len(config["services"]) > 0, "render.yaml must have at least one service"

    api_svc = config["services"][0]
    assert api_svc.get("type") == "web"
    assert api_svc.get("runtime") == "docker"
    assert api_svc.get("dockerfilePath") == "api/Dockerfile"
    assert api_svc.get("dockerContext") == "."
    assert api_svc.get("plan") == "free"
    assert api_svc.get("healthCheckPath") == "/v2/health"

    # Verify essential environment variable keys
    env_keys = [item["key"] for item in api_svc.get("envVars", [])]
    required_keys = ["DATABASE_URL", "REDIS_URL", "QDRANT_URL", "QDRANT_API_KEY", "GEMINI_API_KEY", "LLM_MODEL", "PORT"]
    for k in required_keys:
        assert k in env_keys, f"render.yaml missing essential envVar key: {k}"


def test_vercel_json_structure_and_validity():
    """Verify root vercel.json and web/vercel.json exist and are valid JSON configs."""
    root_vercel = os.path.join(BASE_DIR, "vercel.json")
    web_vercel = os.path.join(BASE_DIR, "web", "vercel.json")

    assert os.path.exists(root_vercel), "root vercel.json is missing"
    assert os.path.exists(web_vercel), "web/vercel.json is missing"

    with open(root_vercel, "r", encoding="utf-8") as f:
        root_cfg = json.load(f)
    assert root_cfg.get("framework") == "nextjs"
    assert "buildCommand" in root_cfg

    with open(web_vercel, "r", encoding="utf-8") as f:
        web_cfg = json.load(f)
    assert web_cfg.get("framework") == "nextjs"
    assert web_cfg.get("buildCommand") == "npm run build"


def test_environment_templates_completeness():
    """Verify all environment example templates exist and contain required keys."""
    env_files = [
        os.path.join(BASE_DIR, ".env.example"),
        os.path.join(BASE_DIR, ".env.tier1.example"),
        os.path.join(BASE_DIR, ".env.tier0.example"),
        os.path.join(BASE_DIR, "web", ".env.example"),
    ]

    for p in env_files:
        assert os.path.exists(p), f"Environment template missing: {p}"
        with open(p, "r", encoding="utf-8") as f:
            content = f.read()
            assert len(content.strip()) > 0, f"Template is empty: {p}"

    # Verify Tier 1 template has all target cloud configurations
    with open(os.path.join(BASE_DIR, ".env.tier1.example"), "r", encoding="utf-8") as f:
        t1_content = f.read()
        assert "DATABASE_URL" in t1_content
        assert "REDIS_URL" in t1_content
        assert "QDRANT_URL" in t1_content
        assert "QDRANT_API_KEY" in t1_content
        assert "GEMINI_API_KEY" in t1_content
        assert "neon.tech" in t1_content
        assert "upstash.io" in t1_content


def test_postgres_dsn_ssl_and_normalization():
    """Verify _parse_and_clean_dsn handles Neon/Supabase SSL params and normalizes schemes."""
    # Test 1: postgres:// scheme with sslmode=require
    dsn1 = "postgres://user:pass@ep-xyz.us-east-2.aws.neon.tech/neondb?sslmode=require"
    clean1, ssl1 = _parse_and_clean_dsn(dsn1)
    assert clean1.startswith("postgresql://")
    assert "sslmode" not in clean1
    assert ssl1 == "require"

    # Test 2: postgresql:// with auto-detected neon.tech hostname
    dsn2 = "postgresql://user:pass@ep-123.region.neon.tech/mydb"
    clean2, ssl2 = _parse_and_clean_dsn(dsn2)
    assert clean2.startswith("postgresql://")
    assert ssl2 == "require"

    # Test 3: localhost without ssl
    dsn3 = "postgresql://postgres:postgres@localhost:5432/recsys"
    clean3, ssl3 = _parse_and_clean_dsn(dsn3)
    assert clean3 == dsn3
    assert ssl3 is None

    # Test 4: sslmode=disable
    dsn4 = "postgresql://user:pass@host/db?sslmode=disable"
    clean4, ssl4 = _parse_and_clean_dsn(dsn4)
    assert ssl4 is False


def test_api_dockerfile_dynamic_port():
    """Verify api/Dockerfile includes dynamic $PORT substitution for cloud hosting."""
    dockerfile_path = os.path.join(BASE_DIR, "api", "Dockerfile")
    with open(dockerfile_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "PORT:-8000" in content, "api/Dockerfile must support ${PORT:-8000} dynamic port binding"


@pytest.mark.asyncio
async def test_preflight_check_postgres_mock():
    """Verify preflight check_postgres handles successful connection and error states."""
    with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_pool_create:
        mock_pool = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(side_effect=[1, True])
        mock_conn.execute = AsyncMock()
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        mock_pool.close = AsyncMock()
        mock_pool_create.return_value = mock_pool

        res = await preflight.check_postgres("postgresql://u:p@localhost:5432/db")
        assert res["status"] == "PASS"
        assert res["service"] == "PostgreSQL"


@pytest.mark.asyncio
async def test_preflight_check_redis_mock():
    """Verify preflight check_redis handles probe write/read cycles."""
    with patch("redis.asyncio.from_url") as mock_from_url:
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=True)
        mock_client.set = AsyncMock(return_value=True)
        mock_client.get = AsyncMock(return_value="ok")
        mock_client.delete = AsyncMock(return_value=1)
        mock_client.aclose = AsyncMock()
        mock_from_url.return_value = mock_client

        res = await preflight.check_redis(url="rediss://default:p@xyz.upstash.io:6379")
        assert res["status"] == "PASS"
        assert res["service"] == "Redis Cache"


def test_preflight_check_qdrant_mock():
    """Verify preflight check_qdrant handles cluster inspection."""
    with patch("pipeline.sync_embeddings.get_qdrant_client") as mock_get_client:
        mock_client = MagicMock()
        collection_obj = MagicMock()
        collection_obj.name = "products"
        mock_client.get_collections.return_value = MagicMock(collections=[collection_obj])
        info_mock = MagicMock()
        info_mock.points_count = 3000
        mock_client.get_collection.return_value = info_mock
        mock_get_client.return_value = mock_client

        res = preflight.check_qdrant(url="https://xyz.cloud.qdrant.io:6333", api_key="secret")
        assert res["status"] == "PASS"
        assert "3,000" in res["message"]


def test_tier1_documentation_exists():
    """Verify comprehensive Tier 1 deployment guide exists in docs/DEPLOYMENT_TIER1.md."""
    doc_path = os.path.join(BASE_DIR, "docs", "DEPLOYMENT_TIER1.md")
    assert os.path.exists(doc_path), "docs/DEPLOYMENT_TIER1.md is missing"
    with open(doc_path, "r", encoding="utf-8") as f:
        doc = f.read()
    assert "Render" in doc
    assert "Vercel" in doc
    assert "Neon" in doc
    assert "Upstash" in doc
    assert "Qdrant Cloud" in doc
    assert "UptimeRobot" in doc
