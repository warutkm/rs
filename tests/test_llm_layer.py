"""
tests/test_llm_layer.py
=======================
Unit and integration tests for Phase 6 (LLM Explanations & Query Understanding Layer):
  - Pydantic schema validation (FeatureExplanationInput, ItemExplanation, ParsedQuery)
  - Deterministic rule-based explanation generation & dominant factor extraction
  - Search query parsing, semantic rewriting, price extraction, category mapping
  - ExplanationCache (Redis + in-memory fallback, async get/set/batch, telemetry)
  - Gemini LLM generation with structured JSON output and prompt prefix caching
  - Resilient fallbacks when LLM raises errors or returns malformed data
  - Async batch execution with concurrency throttling & background caching
"""

import json
import pytest
from unittest.mock import MagicMock

import config
import importlib

_mod = importlib.import_module("src.14_llm_layer")
FeatureExplanationInput = _mod.FeatureExplanationInput
ItemExplanation = _mod.ItemExplanation
ParsedQuery = _mod.ParsedQuery
ExplanationCache = _mod.ExplanationCache
LLMLayer = _mod.LLMLayer
generate_rule_based_explanation = _mod.generate_rule_based_explanation
parse_query_rule_based = _mod.parse_query_rule_based
EXPLANATION_SYSTEM_INSTRUCTION = _mod.EXPLANATION_SYSTEM_INSTRUCTION
QUERY_PARSER_SYSTEM_INSTRUCTION = _mod.QUERY_PARSER_SYSTEM_INSTRUCTION


# =====================================================================
# 1. STRUCTURED SCHEMA TESTS
# =====================================================================


def test_feature_explanation_input_schema():
    """Verify FeatureExplanationInput structure and item_id=parent_asin invariant."""
    inp = FeatureExplanationInput(
        user_id="U1001",
        item_id="B08N5WRWNW",
        title="Sample Guitar Tuner",
        category="Musical_Instruments",
        price=19.99,
        rating_mean=4.7,
        features={
            "als_score": 0.85,
            "svdpp_score": 0.80,
            "mf_score": 0.82,
            "ncf_score": 0.88,
            "content_score": 0.60,
            "apriori_lift": 0.5,
            "price_score": 0.9,
            "recency": 0.7,
            "popularity": 0.8,
            "helpful_votes": 0.4,
        },
        user_recent_items=["B0002E1O2C"],
        user_top_categories=["Musical_Instruments"],
    )
    assert inp.user_id == "U1001"
    assert inp.item_id == "B08N5WRWNW"
    assert len(inp.features) == 10
    assert inp.features["als_score"] == 0.85


def test_item_explanation_serialization():
    """Verify ItemExplanation JSON serialization and default fields."""
    expl = ItemExplanation(
        user_id="U1001",
        item_id="B08N5WRWNW",
        explanation="Recommended because you like guitar accessories.",
        confidence=0.92,
        source="llm",
        dominant_factor="collaborative_filtering",
        cached=False,
    )
    dumped = expl.model_dump_json()
    data = json.loads(dumped)
    assert data["user_id"] == "U1001"
    assert data["item_id"] == "B08N5WRWNW"
    assert data["dominant_factor"] == "collaborative_filtering"
    assert data["model_version"] == config.MODEL_VERSION


def test_parsed_query_schema():
    """Verify ParsedQuery fields, types, and defaults."""
    pq = ParsedQuery(
        raw_query="cheap ps5 games under 30",
        rewritten_query="ps5 games",
        category="Video_Games",
        price_max=30.0,
        intent="product_search",
        extracted_attributes=["budget-friendly"],
    )
    assert pq.raw_query == "cheap ps5 games under 30"
    assert pq.rewritten_query == "ps5 games"
    assert pq.price_max == 30.0
    assert pq.category == "Video_Games"
    assert "budget-friendly" in pq.extracted_attributes


# =====================================================================
# 2. DETERMINISTIC RULE-BASED EXPLANATION TESTS
# =====================================================================


def test_rule_based_explanation_apriori_lift():
    """High apriori_lift should produce frequently_bought_together dominant factor."""
    inp = FeatureExplanationInput(
        user_id="U1",
        item_id="B001",
        features={"apriori_lift": 3.5, "als_score": 0.1, "content_score": 0.1},
    )
    expl = generate_rule_based_explanation(inp)
    assert expl.dominant_factor == "frequently_bought_together"
    assert "bought together" in expl.explanation.lower()
    assert expl.source == "rule_fallback"


def test_rule_based_explanation_collaborative_filtering():
    """High CF score should produce collaborative_filtering dominant factor."""
    inp = FeatureExplanationInput(
        user_id="U2",
        item_id="B002",
        features={"ncf_score": 0.95, "als_score": 0.90, "apriori_lift": 0.0},
    )
    expl = generate_rule_based_explanation(inp)
    assert expl.dominant_factor == "collaborative_filtering"
    assert "similar taste" in expl.explanation.lower()


def test_rule_based_explanation_content_similarity():
    """High content_score should produce content_similarity dominant factor."""
    inp = FeatureExplanationInput(
        user_id="U3",
        item_id="B003",
        features={"content_score": 0.90, "als_score": 0.1, "apriori_lift": 0.1},
    )
    expl = generate_rule_based_explanation(inp)
    assert expl.dominant_factor == "content_similarity"
    assert "features" in expl.explanation.lower() or "style" in expl.explanation.lower()


def test_rule_based_explanation_cold_start():
    """Zero/empty features should produce cold_start dominant factor gracefully."""
    inp = FeatureExplanationInput(
        user_id="U_NEW",
        item_id="B_NEW",
        category="Software",
        features={},
    )
    expl = generate_rule_based_explanation(inp)
    assert expl.dominant_factor == "cold_start"
    assert "Software" in expl.explanation


# =====================================================================
# 3. QUERY UNDERSTANDING & REWRITING TESTS
# =====================================================================


def test_parse_query_price_under():
    """Test extraction of price ceiling (e.g. under $50)."""
    pq = parse_query_rule_based("wireless gaming mouse under $45")
    assert pq.price_max == 45.0
    assert pq.category == "Video_Games"
    assert "wireless" in pq.extracted_attributes
    assert "45" not in pq.rewritten_query


def test_parse_query_price_range():
    """Test extraction of price range (e.g. between $30 and $80)."""
    pq = parse_query_rule_based("electric guitar pedal between $30 and $80")
    assert pq.price_min == 30.0
    assert pq.price_max == 80.0
    assert pq.category == "Musical_Instruments"
    assert "electric" in pq.extracted_attributes


def test_parse_query_category_detection():
    """Test category mapping for Video_Games, Musical_Instruments, Software."""
    assert parse_query_rule_based("best nintendo switch games").category == "Video_Games"
    assert parse_query_rule_based("acoustic piano keyboard").category == "Musical_Instruments"
    assert parse_query_rule_based("antivirus security software").category == "Software"


def test_parse_query_empty():
    """Empty queries should return empty ParsedQuery without crashing."""
    llm = LLMLayer(use_mock=True)
    pq = llm.rewrite_query("")
    assert pq.raw_query == ""
    assert pq.rewritten_query == ""


# =====================================================================
# 4. REDIS & IN-MEMORY CACHE TESTS
# =====================================================================


def test_cache_key_format():
    """Verify cache key strictly follows 'explanation:{user_id}:{item_id}:{model_version}'."""
    key = ExplanationCache.make_key("U100", "B000XYZ", "v2.0")
    assert key == "explanation:U100:B000XYZ:v2.0"


def test_cache_set_get_and_metrics():
    """Test set, get, hit count, miss count, and hit rate calculation."""
    cache = ExplanationCache()
    cache.clear()

    expl = ItemExplanation(
        user_id="U_CACHE_TEST",
        item_id="B_CACHE_ITEM",
        explanation="Test cached explanation.",
        dominant_factor="popularity",
    )

    # 1. Miss initially
    assert cache.get("U_CACHE_TEST", "B_CACHE_ITEM") is None

    # 2. Set explanation
    assert cache.set(expl, ttl=300) is True

    # 3. Hit on second get
    cached_val = cache.get("U_CACHE_TEST", "B_CACHE_ITEM")
    assert cached_val is not None
    assert cached_val.explanation == "Test cached explanation."
    assert cached_val.cached is True

    # 4. Telemetry metrics
    metrics = cache.get_metrics()
    assert metrics["cache_hits"] >= 1
    assert metrics["cache_misses"] >= 1
    assert 0.0 < metrics["cache_hit_rate"] <= 1.0


@pytest.mark.asyncio
async def test_async_cache_operations():
    """Test asynchronous get_async, set_async, and get_batch_async."""
    cache = ExplanationCache()
    cache.clear()

    expl1 = ItemExplanation(
        user_id="U_ASYNC_1",
        item_id="B_ASYNC_1",
        explanation="Async explanation 1",
    )
    expl2 = ItemExplanation(
        user_id="U_ASYNC_2",
        item_id="B_ASYNC_2",
        explanation="Async explanation 2",
    )

    await cache.set_async(expl1, ttl=60)
    await cache.set_async(expl2, ttl=60)

    res1 = await cache.get_async("U_ASYNC_1", "B_ASYNC_1")
    assert res1 is not None
    assert res1.explanation == "Async explanation 1"

    batch_res = await cache.get_batch_async(
        [
            ("U_ASYNC_1", "B_ASYNC_1"),
            ("U_ASYNC_2", "B_ASYNC_2"),
            ("U_ASYNC_MISS", "B_ASYNC_MISS"),
        ]
    )
    assert batch_res[("U_ASYNC_1", "B_ASYNC_1")] is not None
    assert batch_res[("U_ASYNC_2", "B_ASYNC_2")] is not None
    assert batch_res[("U_ASYNC_MISS", "B_ASYNC_MISS")] is None


# =====================================================================
# 5. MOCK LLM GENERATION & PREFIX CACHING TESTS
# =====================================================================


def test_mock_llm_explanation_generation():
    """Test LLMLayer explanation generation with mocked Gemini response."""
    cache = ExplanationCache()
    cache.clear()

    llm = LLMLayer(cache=cache)
    llm._is_configured = True
    llm.use_mock = False
    llm.api_key = "mock_key"

    mock_model = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(
        {
            "explanation": "Because you recently purchased headphones, this amplifier offers the ideal power match.",
            "dominant_factor": "frequently_bought_together",
            "confidence": 0.95,
        }
    )
    mock_model.generate_content.return_value = mock_resp

    mock_genai = MagicMock()
    mock_genai.GenerativeModel.return_value = mock_model
    llm._genai_client = mock_genai

    inp = FeatureExplanationInput(
        user_id="U_MOCK",
        item_id="B_MOCK",
        title="Headphone Amp",
        features={"apriori_lift": 2.5},
    )

    res = llm.explain(inp, use_cache=False)
    assert res.source == "llm"
    assert "amplifier" in res.explanation
    assert res.dominant_factor == "frequently_bought_together"
    assert res.confidence == 0.95

    # Verify Gemini system instruction prompt caching pattern was used
    mock_genai.GenerativeModel.assert_called_with(
        model_name=llm.model_name,
        system_instruction=EXPLANATION_SYSTEM_INSTRUCTION,
        generation_config={"response_mime_type": "application/json", "temperature": 0.2},
    )


def test_mock_llm_query_rewriting():
    """Test LLMLayer query rewriting with mocked Gemini response."""
    llm = LLMLayer(use_mock=False, api_key="mock_key")
    llm._is_configured = True

    mock_model = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(
        {
            "raw_query": "budget overdrive for electric guitar < $50",
            "rewritten_query": "overdrive pedal electric guitar",
            "category": "Musical_Instruments",
            "price_min": None,
            "price_max": 50.0,
            "brand": None,
            "intent": "product_search",
            "extracted_attributes": ["overdrive", "budget"],
        }
    )
    mock_model.generate_content.return_value = mock_resp

    mock_genai = MagicMock()
    mock_genai.GenerativeModel.return_value = mock_model
    llm._genai_client = mock_genai

    parsed = llm.rewrite_query("budget overdrive for electric guitar < $50")
    assert parsed.source == "llm"
    assert parsed.rewritten_query == "overdrive pedal electric guitar"
    assert parsed.category == "Musical_Instruments"
    assert parsed.price_max == 50.0

    mock_genai.GenerativeModel.assert_called_with(
        model_name=llm.model_name,
        system_instruction=QUERY_PARSER_SYSTEM_INSTRUCTION,
        generation_config={"response_mime_type": "application/json", "temperature": 0.1},
    )


def test_llm_exception_fallback():
    """When LLM API fails or raises an error, it should cleanly fall back to rule engine."""
    llm = LLMLayer(use_mock=False, api_key="mock_key")
    llm._is_configured = True

    mock_model = MagicMock()
    mock_model.generate_content.side_effect = RuntimeError("API rate limit exceeded")

    mock_genai = MagicMock()
    mock_genai.GenerativeModel.return_value = mock_model
    llm._genai_client = mock_genai

    inp = FeatureExplanationInput(
        user_id="U_ERR",
        item_id="B_ERR",
        features={"content_score": 0.85},
    )

    # Should not raise RuntimeError, must return rule fallback
    res = llm.explain(inp, use_cache=False)
    assert res.source == "rule_fallback"
    assert res.dominant_factor == "content_similarity"


# =====================================================================
# 6. ASYNC BATCH & BACKGROUND CACHING TESTS
# =====================================================================


@pytest.mark.asyncio
async def test_async_batch_explanation_concurrency():
    """Test explain_batch_async with multiple candidate inputs."""
    cache = ExplanationCache()
    cache.clear()
    llm = LLMLayer(cache=cache, use_mock=True)

    inputs = [
        FeatureExplanationInput(user_id=f"U_{i}", item_id=f"B_{i}", features={"popularity": 0.8}) for i in range(5)
    ]

    results = await llm.explain_batch_async(inputs, use_cache=True, concurrency=3)
    assert len(results) == 5
    for expl in results:
        assert expl.dominant_factor == "popularity"
        assert expl.source == "rule_fallback"


@pytest.mark.asyncio
async def test_background_cache_explanations():
    """Test background_cache_explanations asynchronously populates cache."""
    cache = ExplanationCache()
    cache.clear()
    llm = LLMLayer(cache=cache, use_mock=True)

    inputs = [
        FeatureExplanationInput(user_id="U_BG_1", item_id="B_BG_1", features={"apriori_lift": 2.0}),
        FeatureExplanationInput(user_id="U_BG_2", item_id="B_BG_2", features={"content_score": 0.9}),
    ]

    # Precondition: cache is empty
    assert await cache.get_async("U_BG_1", "B_BG_1") is None

    # Execute background caching
    await llm.background_cache_explanations(inputs)

    # Verify both items are now in cache
    val1 = await cache.get_async("U_BG_1", "B_BG_1")
    val2 = await cache.get_async("U_BG_2", "B_BG_2")
    assert val1 is not None and val1.dominant_factor == "frequently_bought_together"
    assert val2 is not None and val2.dominant_factor == "content_similarity"
