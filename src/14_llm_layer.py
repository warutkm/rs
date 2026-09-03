"""
src/14_llm_layer.py
===================
Phase 6 — LLM Explanations & Query Understanding Layer

Components:
  1. Structured Schemas (Pydantic):
       - FeatureExplanationInput: ranker feature vector + item metadata
       - ItemExplanation: 1-sentence customer-facing explanation with dominant factor & confidence
       - ParsedQuery: rewritten semantic search query + structured filters (price, category, brand, attributes)
  2. Prompt Prefix Caching Architecture:
       - Static system instruction + fixed JSON schema prefix (Gemini 3.5 Flash-Lite `gemini-3.5-flash-lite`)
       - Volatile per-request data suffix
  3. Redis Caching Integration:
       - Keyed by `explanation:{user_id}:{item_id}:{model_version}`
       - Asynchronous & synchronous cache get/set/batch with TTL
       - In-memory fallback if Redis is unavailable
       - Cache hit/miss observability telemetry
  4. Deterministic Feature-Grounded Fallbacks:
       - Rule-based explanation generator based on ranker feature signals
       - Heuristic/regex query understanding for price bounds and catalog categories
  5. Unified LLMLayer:
       - Synchronous and asynchronous explanation & query rewriting APIs
       - Concurrency-throttled batch generation
       - Non-blocking background caching integration
"""

import os
import sys
import json
import re
import time
import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field

# Setup paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR) if os.path.basename(SRC_DIR) == "src" else SRC_DIR
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config

logger = logging.getLogger("recsys.llm_layer")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# =====================================================================
# 1. STRUCTURED SCHEMAS
# =====================================================================


class FeatureExplanationInput(BaseModel):
    """
    Input payload for generating recommendation explanation.
    `item_id` MUST be parent_asin per GEMINI.md hard rule.
    """

    user_id: str
    item_id: str  # parent_asin
    title: Optional[str] = ""
    category: Optional[str] = ""
    price: Optional[float] = None
    rating_mean: Optional[float] = None
    features: Dict[str, float] = Field(default_factory=dict)
    user_recent_items: Optional[List[str]] = None
    user_top_categories: Optional[List[str]] = None


class ItemExplanation(BaseModel):
    """
    Structured output explanation for a recommended item.
    """

    user_id: str
    item_id: str  # parent_asin
    explanation: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = "llm"  # "llm", "cache", "rule_fallback", "cold_start"
    dominant_factor: Optional[str] = (
        None  # e.g., "collaborative_filtering", "frequently_bought_together",
        # "content_similarity", "popularity", "price_match"
    )
    cached: bool = False
    model_version: str = config.MODEL_VERSION


class ParsedQuery(BaseModel):
    """
    Structured output for free-text search query understanding & rewriting.
    """

    raw_query: str
    rewritten_query: str
    category: Optional[str] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    brand: Optional[str] = None
    intent: str = (
        "product_search"  # "product_search", "category_browse", "brand_search", "comparison", "feature_search"
    )
    extracted_attributes: List[str] = Field(default_factory=list)
    source: str = "llm"  # "llm", "rule_fallback"


# =====================================================================
# 2. SYSTEM INSTRUCTIONS & PROMPT PREFIX CACHING
# =====================================================================

EXPLANATION_SYSTEM_INSTRUCTION = """You are an expert e-commerce recommendation explainer for an Amazon product catalog.
Given a user ID, item ID, product metadata, and underlying ranker feature signals, produce a single concise,
truthful, customer-facing sentence (max 25 words) explaining why this item is recommended.

Signal definitions:
- Collaborative filtering (als_score, svdpp_score, mf_score, ncf_score): High value means shoppers with
  tastes similar to the user loved or bought this item.
- Content similarity (content_score): High value means the item closely matches the style, category, or features
  of items the user interacted with.
- Association lift (apriori_lift): High value means this item is frequently bought together with past items.
- Popularity & helpful votes (popularity, helpful_votes): Category best-seller with trusted community feedback.
- Price score (price_score): High value indicates the item falls directly in the user's preferred budget tier.

Rules:
1. Ground the explanation strictly in the dominant feature signals.
2. Never mention internal technical variable names (e.g. do not say 'als_score', 'NCF', or 'LambdaMART').
3. Keep the tone warm, concise, and customer-centric.
4. Output MUST be valid JSON with keys: 'explanation' (string), 'dominant_factor' (string), 'confidence' (float).
"""

QUERY_PARSER_SYSTEM_INSTRUCTION = """You are an e-commerce search query understanding and semantic rewriting engine
for an Amazon product catalog.
Catalog categories: Video_Games, Musical_Instruments, Software.

Given a user's raw search query:
1. 'rewritten_query': Clean, descriptive semantic search string optimized for embedding retrieval (e5 model).
   Remove conversational noise, price constraints, and filler words.
2. 'category': Map to one of ["Video_Games", "Musical_Instruments", "Software"] if mentioned or implied; else null.
3. 'price_min': Extract minimum price floor as float, or null.
4. 'price_max': Extract maximum price ceiling as float, or null.
5. 'brand': Extract brand/publisher if explicitly named, or null.
6. 'intent': One of ["product_search", "category_browse", "brand_search", "comparison", "feature_search"].
7. 'extracted_attributes': List of key product attributes or specifications (e.g. ["wireless", "mechanical"]).

Output MUST be valid JSON matching the schema.
"""


# =====================================================================
# 3. REDIS CACHING INTEGRATION LAYER
# =====================================================================


class ExplanationCache:
    """
    Redis explanation cache with in-memory fallback.
    Keys formatted as: explanation:{user_id}:{item_id}:{model_version}
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        url: Optional[str] = None,
        password: Optional[str] = None,
        default_ttl: int = config.EXPLANATION_CACHE_TTL,
    ):
        self.host = host or config.REDIS_HOST
        self.port = port or config.REDIS_PORT
        self.url = url or config.REDIS_URL
        self.password = password or config.REDIS_PASSWORD
        self.default_ttl = default_ttl

        self._redis_client = None
        self._async_redis_client = None
        self._in_memory_cache: Dict[str, Tuple[str, float]] = {}  # key -> (json_str, expire_at)
        self._use_redis = False

        # Observability metrics
        self.hits = 0
        self.misses = 0
        self.errors = 0

        self._init_redis()

    def _init_redis(self):
        try:
            import redis

            if self.url:
                self._redis_client = redis.from_url(self.url, decode_responses=True)
            else:
                self._redis_client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    password=self.password,
                    decode_responses=True,
                    socket_connect_timeout=1.0,
                    socket_timeout=1.0,
                )
            self._redis_client.ping()
            self._use_redis = True
            logger.info("ExplanationCache: Connected to Redis successfully.")
        except Exception as e:
            self._use_redis = False
            logger.info(f"ExplanationCache: Redis unavailable ({e}). Using in-memory fallback cache.")

    async def _get_async_client(self):
        if not self._use_redis:
            return None
        if self._async_redis_client is None:
            try:
                import redis.asyncio as aioredis

                if self.url:
                    self._async_redis_client = aioredis.from_url(self.url, decode_responses=True)
                else:
                    self._async_redis_client = aioredis.Redis(
                        host=self.host,
                        port=self.port,
                        password=self.password,
                        decode_responses=True,
                        socket_connect_timeout=1.0,
                        socket_timeout=1.0,
                    )
                await self._async_redis_client.ping()
            except Exception as e:
                self._use_redis = False
                logger.warning(f"Async Redis connection failed: {e}. Reverting to in-memory cache.")
                self._async_redis_client = None
        return self._async_redis_client

    @staticmethod
    def make_key(user_id: str, item_id: str, model_version: str = config.MODEL_VERSION) -> str:
        return f"explanation:{user_id}:{item_id}:{model_version}"

    def get(self, user_id: str, item_id: str, model_version: str = config.MODEL_VERSION) -> Optional[ItemExplanation]:
        """Synchronous cache retrieval."""
        key = self.make_key(user_id, item_id, model_version)
        if self._use_redis and self._redis_client:
            try:
                val = self._redis_client.get(key)
                if val:
                    self.hits += 1
                    data = json.loads(val)
                    data["cached"] = True
                    return ItemExplanation(**data)
                self.misses += 1
                return None
            except Exception as e:
                self.errors += 1
                logger.debug(f"Redis get error for {key}: {e}")

        # In-memory fallback
        now = time.time()
        if key in self._in_memory_cache:
            val_str, expire_at = self._in_memory_cache[key]
            if expire_at > now:
                self.hits += 1
                data = json.loads(val_str)
                data["cached"] = True
                return ItemExplanation(**data)
            else:
                del self._in_memory_cache[key]
        self.misses += 1
        return None

    def set(self, explanation: ItemExplanation, ttl: Optional[int] = None) -> bool:
        """Synchronous cache storage."""
        ttl = ttl if ttl is not None else self.default_ttl
        key = self.make_key(explanation.user_id, explanation.item_id, explanation.model_version)
        payload = explanation.model_dump_json()

        if self._use_redis and self._redis_client:
            try:
                self._redis_client.setex(key, ttl, payload)
                return True
            except Exception as e:
                self.errors += 1
                logger.debug(f"Redis set error for {key}: {e}")

        # In-memory fallback
        self._in_memory_cache[key] = (payload, time.time() + ttl)
        return True

    async def get_async(
        self, user_id: str, item_id: str, model_version: str = config.MODEL_VERSION
    ) -> Optional[ItemExplanation]:
        """Asynchronous cache retrieval."""
        key = self.make_key(user_id, item_id, model_version)
        client = await self._get_async_client()
        if client:
            try:
                val = await client.get(key)
                if val:
                    self.hits += 1
                    data = json.loads(val)
                    data["cached"] = True
                    return ItemExplanation(**data)
                self.misses += 1
                return None
            except Exception as e:
                self.errors += 1
                logger.debug(f"Async Redis get error for {key}: {e}")

        # In-memory fallback
        now = time.time()
        if key in self._in_memory_cache:
            val_str, expire_at = self._in_memory_cache[key]
            if expire_at > now:
                self.hits += 1
                data = json.loads(val_str)
                data["cached"] = True
                return ItemExplanation(**data)
            else:
                del self._in_memory_cache[key]
        self.misses += 1
        return None

    async def set_async(self, explanation: ItemExplanation, ttl: Optional[int] = None) -> bool:
        """Asynchronous cache storage."""
        ttl = ttl if ttl is not None else self.default_ttl
        key = self.make_key(explanation.user_id, explanation.item_id, explanation.model_version)
        payload = explanation.model_dump_json()

        client = await self._get_async_client()
        if client:
            try:
                await client.setex(key, ttl, payload)
                return True
            except Exception as e:
                self.errors += 1
                logger.debug(f"Async Redis set error for {key}: {e}")

        # In-memory fallback
        self._in_memory_cache[key] = (payload, time.time() + ttl)
        return True

    def get_batch(
        self, items: List[Tuple[str, str]], model_version: str = config.MODEL_VERSION
    ) -> Dict[Tuple[str, str], Optional[ItemExplanation]]:
        """Batch synchronous lookup."""
        result = {}
        for uid, iid in items:
            result[(uid, iid)] = self.get(uid, iid, model_version)
        return result

    async def get_batch_async(
        self, items: List[Tuple[str, str]], model_version: str = config.MODEL_VERSION
    ) -> Dict[Tuple[str, str], Optional[ItemExplanation]]:
        """Batch asynchronous lookup."""
        result = {}
        keys = [self.make_key(uid, iid, model_version) for uid, iid in items]
        client = await self._get_async_client()

        if client and keys:
            try:
                values = await client.mget(keys)
                for (uid, iid), val in zip(items, values):
                    if val:
                        self.hits += 1
                        data = json.loads(val)
                        data["cached"] = True
                        result[(uid, iid)] = ItemExplanation(**data)
                    else:
                        self.misses += 1
                        result[(uid, iid)] = None
                return result
            except Exception as e:
                self.errors += 1
                logger.debug(f"Async Redis mget error: {e}")

        # In-memory fallback
        now = time.time()
        for uid, iid in items:
            key = self.make_key(uid, iid, model_version)
            if key in self._in_memory_cache:
                val_str, expire_at = self._in_memory_cache[key]
                if expire_at > now:
                    self.hits += 1
                    data = json.loads(val_str)
                    data["cached"] = True
                    result[(uid, iid)] = ItemExplanation(**data)
                    continue
                else:
                    del self._in_memory_cache[key]
            self.misses += 1
            result[(uid, iid)] = None

        return result

    def get_metrics(self) -> Dict[str, Union[int, float]]:
        """Return cache hit rate and counts for observability metrics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total) if total > 0 else 0.0
        return {
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "cache_errors": self.errors,
            "cache_hit_rate": round(hit_rate, 4),
            "backend": "redis" if self._use_redis else "in_memory",
        }

    def clear(self):
        """Clear cache (for testing)."""
        if self._use_redis and self._redis_client:
            try:
                self._redis_client.flushdb()
            except Exception:
                pass
        self._in_memory_cache.clear()
        self.hits = 0
        self.misses = 0
        self.errors = 0


# =====================================================================
# 4. DETERMINISTIC FEATURE-GROUNDED RULE FALLBACKS
# =====================================================================


def generate_rule_based_explanation(inp: FeatureExplanationInput) -> ItemExplanation:
    """
    Deterministic rule-based explanation generation directly grounded in ranker features.
    Provides a high-quality fallback when LLM is unavailable or offline.
    """
    f = inp.features
    apriori_lift = f.get("apriori_lift", 0.0)
    cf_score = max(
        f.get("als_score", 0.0),
        f.get("svdpp_score", 0.0),
        f.get("mf_score", 0.0),
        f.get("ncf_score", 0.0),
    )
    content_score = f.get("content_score", 0.0)
    popularity = f.get("popularity", 0.0)
    price_score = f.get("price_score", 0.0)

    # Determine dominant factor by highest weighted normalized signal
    signals = [
        (
            "frequently_bought_together",
            apriori_lift * 1.5,
            f"Frequently bought together with products in your recent shopping activity.",
        ),
        (
            "collaborative_filtering",
            cf_score * 1.3,
            f"Highly rated by shoppers who have similar taste and preferences to you.",
        ),
        (
            "content_similarity",
            content_score * 1.2,
            f"Matches the features, specifications, and style of items you previously browsed.",
        ),
        (
            "popularity",
            popularity * 1.0,
            f"A top-rated favorite and best-seller in its category with high customer satisfaction.",
        ),
        (
            "price_match",
            price_score * 0.9,
            f"Carefully chosen to fit right within your preferred price and budget range.",
        ),
    ]

    # If item metadata is available, enrich the sentence
    cat_str = f" in {inp.category}" if inp.category else ""

    # Sort signals descending by score
    signals.sort(key=lambda x: x[1], reverse=True)
    best_factor, best_score, default_template = signals[0]

    if best_score <= 0.1:
        dominant_factor = "cold_start"
        explanation = f"Popular and highly rated product{cat_str} recommended for your exploration."
        confidence = 0.7
    else:
        dominant_factor = best_factor
        confidence = min(0.95, max(0.65, float(best_score)))
        explanation = default_template

    return ItemExplanation(
        user_id=inp.user_id,
        item_id=inp.item_id,
        explanation=explanation,
        confidence=round(confidence, 2),
        source="rule_fallback",
        dominant_factor=dominant_factor,
        cached=False,
        model_version=config.MODEL_VERSION,
    )


def parse_query_rule_based(raw_query: str) -> ParsedQuery:
    """
    Regex and heuristic search query parser and semantic rewriter.
    Extracts price bounds, category constraints, and cleaned search tokens.
    """
    query = raw_query.strip()
    clean_query = query.lower()

    price_min: Optional[float] = None
    price_max: Optional[float] = None
    category: Optional[str] = None
    attributes: List[str] = []
    brand: Optional[str] = None
    intent = "product_search"

    # 1. Price extraction
    # Pattern: between $X and $Y / $X - $Y / X to Y dollars
    range_match = re.search(
        r"(?:between\s+)?\$?(\d+(?:\.\d+)?)\s*(?:-|to|and)\s*\$?(\d+(?:\.\d+)?)\s*(?:dollars|\$)?", clean_query
    )
    if range_match:
        try:
            p1, p2 = float(range_match.group(1)), float(range_match.group(2))
            price_min, price_max = min(p1, p2), max(p1, p2)
        except ValueError:
            pass

    # Pattern: under/below/less than $X / < $X / max $X
    if price_max is None:
        under_match = re.search(r"(?:under|below|less\s+than|<|max(?:imum)?)\s*\$?(\d+(?:\.\d+)?)", clean_query)
        if under_match:
            try:
                price_max = float(under_match.group(1))
            except ValueError:
                pass

    # Pattern: above/over/more than $X / > $X / min $X
    if price_min is None:
        over_match = re.search(r"(?:above|over|more\s+than|>|min(?:imum)?)\s*\$?(\d+(?:\.\d+)?)", clean_query)
        if over_match:
            try:
                price_min = float(over_match.group(1))
            except ValueError:
                pass

    # "cheap" / "budget" heuristic if no explicit price
    if price_max is None and re.search(r"\b(cheap|budget|affordable|low cost)\b", clean_query):
        attributes.append("budget-friendly")

    # 2. Category mapping
    video_games_kws = [
        "game",
        "games",
        "gaming",
        "ps5",
        "ps4",
        "xbox",
        "nintendo",
        "switch",
        "controller",
        "rpg",
        "fps",
    ]
    instruments_kws = [
        "guitar",
        "piano",
        "keyboard",
        "drum",
        "mic",
        "microphone",
        "pedal",
        "violin",
        "synth",
        "amplifier",
        "amp",
        "tuner",
    ]
    software_kws = [
        "software",
        "antivirus",
        "operating system",
        "editor",
        "photoshop",
        "cad",
        "tax",
        "suite",
        "utility",
        "driver",
    ]

    if any(re.search(rf"\b{kw}\b", clean_query) for kw in video_games_kws):
        category = "Video_Games"
    elif any(re.search(rf"\b{kw}\b", clean_query) for kw in instruments_kws):
        category = "Musical_Instruments"
    elif any(re.search(rf"\b{kw}\b", clean_query) for kw in software_kws):
        category = "Software"

    # 3. Attribute detection
    attr_kws = [
        "wireless",
        "bluetooth",
        "noise-canceling",
        "noise cancelling",
        "mechanical",
        "rgb",
        "usb-c",
        "portable",
        "vintage",
        "pro",
        "acoustic",
        "electric",
        "multiplayer",
        "vr",
        "waterproof",
    ]
    for attr in attr_kws:
        if re.search(rf"\b{re.escape(attr)}\b", clean_query):
            attributes.append(attr)

    # 4. Brand detection
    brand_kws = ["sony", "microsoft", "nintendo", "yamaha", "fender", "gibson", "shure", "adobe", "logitech", "razer"]
    for b in brand_kws:
        if re.search(rf"\b{b}\b", clean_query):
            brand = b.capitalize()
            break

    # 5. Intent detection
    if re.search(r"\b(compare|vs|versus|difference)\b", clean_query):
        intent = "comparison"
    elif re.search(r"\b(browse|all|top|best|popular)\b", clean_query):
        intent = "category_browse"
    elif brand and len(clean_query.split()) <= 2:
        intent = "brand_search"
    else:
        intent = "product_search"

    # 6. Rewritten query generation (strip filler words and price clauses)
    rewritten = query
    rewritten = re.sub(
        r"(?i)(?:under|below|less than|above|over|between)\s*\$?\d+(?:\.\d+)?(?:\s*(?:and|to|-)\s*\$?\d+(?:\.\d+)?)?",
        "",
        rewritten,
    )
    rewritten = re.sub(
        r"(?i)\b(cheap|budget|affordable|show me|find me|looking for|recommend me|best|top)\b", "", rewritten
    )
    rewritten = re.sub(r"\s+", " ", rewritten).strip()

    if not rewritten:
        rewritten = query

    return ParsedQuery(
        raw_query=raw_query,
        rewritten_query=rewritten,
        category=category,
        price_min=price_min,
        price_max=price_max,
        brand=brand,
        intent=intent,
        extracted_attributes=list(set(attributes)),
        source="rule_fallback",
    )


# =====================================================================
# 5. GEMINI LLM CLIENT & ENGINE (GEMINI 3.5 FLASH-LITE)
# =====================================================================


class LLMLayer:
    """
    Unified LLM Layer for Explanation Generation and Query Understanding.
    Default Model: Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        cache: Optional[ExplanationCache] = None,
        use_mock: bool = False,
    ):
        self.model_name = model_name or config.LLM_MODEL
        self.api_key = api_key if api_key is not None else config.GEMINI_API_KEY
        self.cache = cache or ExplanationCache()
        self.use_mock = use_mock
        self._genai_client = None
        self._is_configured = False

        if not self.use_mock and self.api_key:
            self._init_gemini()

    def _init_gemini(self):
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self._genai_client = genai
            self._is_configured = True
            logger.info(f"LLMLayer: Initialized Google Generative AI with model '{self.model_name}'.")
        except Exception as e:
            logger.warning(f"LLMLayer: Failed to initialize Google GenAI ({e}). Operating in fallback mode.")
            self._is_configured = False

    # -----------------------------------------------------------------
    # EXPLANATION GENERATION
    # -----------------------------------------------------------------

    def _build_explanation_payload(self, inp: FeatureExplanationInput) -> str:
        """Formats the volatile suffix for prompt prefix caching."""
        payload = {
            "user_id": inp.user_id,
            "item_id": inp.item_id,
            "title": inp.title,
            "category": inp.category,
            "price": inp.price,
            "rating_mean": inp.rating_mean,
            "ranker_features": {k: round(v, 4) for k, v in inp.features.items()},
        }
        if inp.user_recent_items:
            payload["user_recent_items"] = inp.user_recent_items[:5]
        if inp.user_top_categories:
            payload["user_top_categories"] = inp.user_top_categories[:3]
        return json.dumps(payload, ensure_ascii=False)

    def explain(self, inp: FeatureExplanationInput, use_cache: bool = True) -> ItemExplanation:
        """
        Generate explanation for a single item (Synchronous).
        Checks Redis cache first. If missing, calls Gemini LLM (or rule fallback) and caches result.
        """
        if use_cache:
            cached = self.cache.get(inp.user_id, inp.item_id)
            if cached is not None:
                return cached

        if not self._is_configured or self.use_mock or not self.api_key:
            res = generate_rule_based_explanation(inp)
            if use_cache:
                self.cache.set(res)
            return res

        try:
            model = self._genai_client.GenerativeModel(
                model_name=self.model_name,
                system_instruction=EXPLANATION_SYSTEM_INSTRUCTION,
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.2,
                },
            )
            prompt = self._build_explanation_payload(inp)
            response = model.generate_content(prompt)
            data = json.loads(response.text)

            explanation_text = data.get("explanation", "").strip()
            dominant_factor = data.get("dominant_factor", "collaborative_filtering")
            confidence = float(data.get("confidence", 0.9))

            if not explanation_text:
                raise ValueError("Empty explanation returned from LLM")

            result = ItemExplanation(
                user_id=inp.user_id,
                item_id=inp.item_id,
                explanation=explanation_text,
                confidence=min(1.0, max(0.0, confidence)),
                source="llm",
                dominant_factor=dominant_factor,
                cached=False,
                model_version=config.MODEL_VERSION,
            )
            if use_cache:
                self.cache.set(result)
            return result

        except Exception as e:
            logger.warning(f"LLM explanation generation failed ({e}). Falling back to rule-based explanation.")
            res = generate_rule_based_explanation(inp)
            if use_cache:
                self.cache.set(res)
            return res

    async def explain_async(self, inp: FeatureExplanationInput, use_cache: bool = True) -> ItemExplanation:
        """
        Generate explanation for a single item (Asynchronous).
        """
        if use_cache:
            cached = await self.cache.get_async(inp.user_id, inp.item_id)
            if cached is not None:
                return cached

        if not self._is_configured or self.use_mock or not self.api_key:
            res = generate_rule_based_explanation(inp)
            if use_cache:
                await self.cache.set_async(res)
            return res

        try:
            model = self._genai_client.GenerativeModel(
                model_name=self.model_name,
                system_instruction=EXPLANATION_SYSTEM_INSTRUCTION,
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.2,
                },
            )
            prompt = self._build_explanation_payload(inp)
            response = await model.generate_content_async(prompt)
            data = json.loads(response.text)

            explanation_text = data.get("explanation", "").strip()
            dominant_factor = data.get("dominant_factor", "collaborative_filtering")
            confidence = float(data.get("confidence", 0.9))

            if not explanation_text:
                raise ValueError("Empty explanation returned from LLM")

            result = ItemExplanation(
                user_id=inp.user_id,
                item_id=inp.item_id,
                explanation=explanation_text,
                confidence=min(1.0, max(0.0, confidence)),
                source="llm",
                dominant_factor=dominant_factor,
                cached=False,
                model_version=config.MODEL_VERSION,
            )
            if use_cache:
                await self.cache.set_async(result)
            return result

        except Exception as e:
            logger.warning(f"Async LLM explanation generation failed ({e}). Falling back to rule-based explanation.")
            res = generate_rule_based_explanation(inp)
            if use_cache:
                await self.cache.set_async(res)
            return res

    async def explain_batch_async(
        self,
        items: List[FeatureExplanationInput],
        use_cache: bool = True,
        concurrency: int = 10,
    ) -> List[ItemExplanation]:
        """
        Generate explanations for a batch of top-N ranked items with concurrency control.
        """
        if not items:
            return []

        sem = asyncio.Semaphore(concurrency)

        async def _sem_explain(item_inp: FeatureExplanationInput):
            async with sem:
                return await self.explain_async(item_inp, use_cache=use_cache)

        tasks = [_sem_explain(item) for item in items]
        return await asyncio.gather(*tasks)

    async def background_cache_explanations(self, items: List[FeatureExplanationInput]) -> None:
        """
        Background task helper: fires asynchronous explanation generation and populates Redis cache.
        Does not block recommendation serving.
        """
        try:
            uncached = []
            for item in items:
                cached = await self.cache.get_async(item.user_id, item.item_id)
                if cached is None:
                    uncached.append(item)

            if uncached:
                logger.info(f"Background worker generating explanations for {len(uncached)} uncached items...")
                await self.explain_batch_async(uncached, use_cache=True)
                logger.info(f"Background explanation generation completed.")
        except Exception as e:
            logger.error(f"Error in background_cache_explanations: {e}")

    # -----------------------------------------------------------------
    # QUERY UNDERSTANDING & SEMANTIC REWRITING
    # -----------------------------------------------------------------

    def rewrite_query(self, raw_query: str) -> ParsedQuery:
        """
        Parse and rewrite user search query (Synchronous).
        """
        if not raw_query or not raw_query.strip():
            return ParsedQuery(raw_query="", rewritten_query="")

        if not self._is_configured or self.use_mock or not self.api_key:
            return parse_query_rule_based(raw_query)

        try:
            model = self._genai_client.GenerativeModel(
                model_name=self.model_name,
                system_instruction=QUERY_PARSER_SYSTEM_INSTRUCTION,
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.1,
                },
            )
            prompt = json.dumps({"raw_query": raw_query})
            response = model.generate_content(prompt)
            data = json.loads(response.text)

            return ParsedQuery(
                raw_query=raw_query,
                rewritten_query=data.get("rewritten_query", raw_query),
                category=data.get("category"),
                price_min=data.get("price_min"),
                price_max=data.get("price_max"),
                brand=data.get("brand"),
                intent=data.get("intent", "product_search"),
                extracted_attributes=data.get("extracted_attributes", []),
                source="llm",
            )
        except Exception as e:
            logger.warning(f"LLM query rewriting failed ({e}). Falling back to heuristic parser.")
            return parse_query_rule_based(raw_query)

    async def rewrite_query_async(self, raw_query: str) -> ParsedQuery:
        """
        Parse and rewrite user search query (Asynchronous).
        """
        if not raw_query or not raw_query.strip():
            return ParsedQuery(raw_query="", rewritten_query="")

        if not self._is_configured or self.use_mock or not self.api_key:
            return parse_query_rule_based(raw_query)

        try:
            model = self._genai_client.GenerativeModel(
                model_name=self.model_name,
                system_instruction=QUERY_PARSER_SYSTEM_INSTRUCTION,
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.1,
                },
            )
            prompt = json.dumps({"raw_query": raw_query})
            response = await model.generate_content_async(prompt)
            data = json.loads(response.text)

            return ParsedQuery(
                raw_query=raw_query,
                rewritten_query=data.get("rewritten_query", raw_query),
                category=data.get("category"),
                price_min=data.get("price_min"),
                price_max=data.get("price_max"),
                brand=data.get("brand"),
                intent=data.get("intent", "product_search"),
                extracted_attributes=data.get("extracted_attributes", []),
                source="llm",
            )
        except Exception as e:
            logger.warning(f"Async LLM query rewriting failed ({e}). Falling back to heuristic parser.")
            return parse_query_rule_based(raw_query)

    # Aliases for query parsing
    parse_query_async = rewrite_query_async
    parse_query = rewrite_query


# =====================================================================
# 6. MAIN EXECUTION DEMO / VALIDATION
# =====================================================================


def main():
    """
    Demonstrate and validate LLM layer features:
      1. Deterministic rule-based explanations for diverse ranker feature inputs.
      2. Search query understanding and semantic rewriting.
      3. Redis cache storage, retrieval, and hit-rate telemetry.
    """
    print("================================================================")
    print(" Amazon RecSys v2 — LLM Explanations & Query Rewriter Demo")
    print(f" LLM Model Default: {config.LLM_MODEL}")
    print(f" Cache Backend: Redis ({config.REDIS_HOST}:{config.REDIS_PORT}) / In-Memory Fallback")
    print("================================================================\n")

    cache = ExplanationCache()
    llm = LLMLayer(cache=cache)

    # 1. Test Feature Explanation Inputs
    test_inputs = [
        FeatureExplanationInput(
            user_id="U_DEMO_01",
            item_id="B08N5WRWNW",  # parent_asin
            title="Wireless Noise Cancelling Gaming Headset",
            category="Video_Games",
            price=79.99,
            rating_mean=4.6,
            features={
                "als_score": 0.82,
                "svdpp_score": 0.79,
                "mf_score": 0.85,
                "ncf_score": 0.88,
                "content_score": 0.65,
                "apriori_lift": 0.4,
                "price_score": 0.7,
                "recency": 0.9,
                "popularity": 0.8,
                "helpful_votes": 0.75,
            },
        ),
        FeatureExplanationInput(
            user_id="U_DEMO_02",
            item_id="B0002E1O2C",  # parent_asin
            title="Classic Analog Distortion Guitar Pedal",
            category="Musical_Instruments",
            price=49.50,
            rating_mean=4.8,
            features={
                "als_score": 0.35,
                "svdpp_score": 0.30,
                "mf_score": 0.40,
                "ncf_score": 0.38,
                "content_score": 0.45,
                "apriori_lift": 2.8,  # High Apriori Lift
                "price_score": 0.8,
                "recency": 0.5,
                "popularity": 0.6,
                "helpful_votes": 0.5,
            },
        ),
    ]

    print("--- 1. Generating Explanations ---")
    for inp in test_inputs:
        expl = llm.explain(inp)
        print(f"User: {expl.user_id} | Item: {expl.item_id}")
        print(f"Dominant Factor: {expl.dominant_factor} (Confidence: {expl.confidence})")
        print(f'Explanation: "{expl.explanation}"')
        print(f"Source: {expl.source} | Cached: {expl.cached}\n")

    # 2. Test Cache Hit on Second Fetch
    print("--- 2. Validating Cache Retrieval ---")
    cached_expl = llm.explain(test_inputs[0])
    print(f"Second fetch item {cached_expl.item_id}: Cached={cached_expl.cached}, Source={cached_expl.source}")
    print(f"Telemetry: {cache.get_metrics()}\n")

    # 3. Test Query Rewriting & Understanding
    print("--- 3. Testing Query Understanding & Rewriting ---")
    queries = [
        "cheap wireless bluetooth headphones for ps5 gaming under $60",
        "vintage overdrive pedal between $30 and $80",
        "best photo editing software with lifetime license",
    ]
    for q in queries:
        parsed = llm.rewrite_query(q)
        print(f"Raw Query:      '{parsed.raw_query}'")
        print(f"Rewritten Query: '{parsed.rewritten_query}'")
        print(
            f"Filters:         Category={parsed.category}, "
            f"Price=[{parsed.price_min}, {parsed.price_max}], Brand={parsed.brand}"
        )
        print(f"Attributes:      {parsed.extracted_attributes} | Intent={parsed.intent}\n")

    print("✅ LLM layer validation completed successfully.")


if __name__ == "__main__":
    main()
