"""
Phase 7 — FastAPI v2 Service
File: api/schemas.py

Pydantic request/response models for all endpoints:
  - Recommendation (/v2/recommend)
  - Similar items (/v2/similar/{item_id})
  - Search (/v2/search)
  - Event feedback logging (/v2/events)
  - Observability & Health (/v2/health, /metrics)
  - Retrain automation (/admin/retrain)
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


# =============================================================================
# SHARED & RECOMMENDATION SCHEMAS
# =============================================================================

class RecommendedItem(BaseModel):
    """Single recommended item representation."""
    item_id: str        = Field(..., description="Item ID (parent_asin)")
    title: str          = Field(..., description="Product title")
    score: float        = Field(..., description="Relevance / similarity score")
    source: str         = Field(..., description="Source engine (personalized_ranker, content_cold_start, etc.)")
    category: Optional[str]       = Field(None, description="Catalog category")
    price: Optional[float]        = Field(None, description="Product price")
    average_rating: Optional[float] = Field(None, description="Average review rating")
    explanation: Optional[str]    = Field(None, description="Customer-facing 1-sentence explanation")
    feature_signals: Optional[Dict[str, float]] = Field(None, description="Model input signals & feature scores")


class RecommendRequest(BaseModel):
    """
    Payload for POST /v2/recommend.
    Supports user personalization with optional seed item, category filtering, and product-type ranking.
    """
    user_id: str                  = Field(..., description="User ID — pass any string for cold-start")
    item_id: Optional[str]        = Field(None, description="Optional seed product ID (parent_asin)")
    top_k: int                    = Field(10, ge=1, le=100, description="Number of results to return")
    category_filter: Optional[str]= Field(None, description="Optional category filter (e.g. Video_Games)")
    product_type: Optional[str]   = Field(None, description="Optional product type / keyword filter (e.g. keyboard, mouse)")
    sort_by: Optional[str]        = Field("ranker", description="Ranking strategy: 'ranker' (LambdaMART) or 'satisfaction' (review-based satisfaction score)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "AHPI18EE22YZMH5TQ4YNLBAFZJA",
                "item_id": "B08N5WRWNW",
                "top_k": 10,
                "category_filter": None,
                "product_type": "keyboard",
                "sort_by": "satisfaction",
            }
        }
    }


class RecommendResponse(BaseModel):
    """Response returned by POST /v2/recommend."""
    user_id: str
    item_id: Optional[str]       = None
    cold_start: bool             = Field(False, description="Whether cold-start fallback was triggered")
    source: str                  = Field("personalized_ranker", description="Primary recommendation strategy used")
    results: List[RecommendedItem]
    model_version: str           = "v2.0"


# =============================================================================
# SIMILAR ITEMS SCHEMAS
# =============================================================================

class SimilarResponse(BaseModel):
    """Response returned by GET /v2/similar/{item_id}."""
    item_id: str
    target_item: Optional[RecommendedItem] = Field(None, description="Queried item's own metadata")
    results: List[RecommendedItem]


# =============================================================================
# SEARCH SCHEMAS
# =============================================================================

class SearchResult(BaseModel):
    """Item result returned by hybrid search."""
    item_id: str
    title: Optional[str]          = None
    category: Optional[str]       = None
    price: Optional[float]        = None
    average_rating: Optional[float] = None
    hybrid_score: float
    emb_score: float
    bm25_score: float


class SearchResponse(BaseModel):
    """Response returned by GET /v2/search."""
    query: str
    rewritten_query: Optional[str] = None
    category_filter: Optional[str] = None
    price_max: Optional[float]     = None
    intent: Optional[str]          = None
    results: List[SearchResult]


# =============================================================================
# EVENT LOGGING SCHEMAS (POST /v2/events)
# =============================================================================

class EventCreateRequest(BaseModel):
    """Interaction or feedback event payload to log into PostgreSQL."""
    user_id: str                  = Field(..., description="User ID performing the action")
    item_id: str                  = Field(..., description="Target item ID (parent_asin)")
    event_type: str               = Field(..., description="Event type: click, view, purchase, rating, cart")
    rating: Optional[float]       = Field(None, ge=1.0, le=5.0, description="Optional rating value (1-5)")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context metadata")

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "AHPI18EE22YZMH5TQ4YNLBAFZJA",
                "item_id": "B08N5WRWNW",
                "event_type": "click",
                "rating": 5.0,
                "metadata": {"source": "recommendation_rail", "position": 1}
            }
        }
    }


class EventResponse(BaseModel):
    """Acknowledgment of logged interaction event."""
    status: str                   = "ok"
    event_id: Optional[Union[int, str]] = None
    message: Optional[str]        = None


# =============================================================================
# HEALTH & OBSERVABILITY SCHEMAS
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response covering all subsystem connections."""
    status: str                   = "ok"
    model_loaded: bool            = False
    ranker_loaded: bool           = False
    vector_db_connected: bool     = False
    redis_connected: bool         = False
    db_connected: bool            = False
    n_items: Optional[int]        = None
    version: str                  = "2.0.0"


class MetricsResponse(BaseModel):
    """Structured JSON telemetry metrics (/metrics)."""
    total_requests: int
    requests_per_endpoint: Dict[str, int]
    requests_per_status: Dict[str, int]
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    status: str                   = "healthy"


# =============================================================================
# ADMIN RETRAIN SCHEMAS
# =============================================================================

class AdminRetrainRequest(BaseModel):
    force: bool                   = Field(False, description="Force re-execution of all DVC stages (--force)")
    targets: Optional[List[str]]  = Field(None, description="Optional list of specific DVC stage targets to reproduce")


class AdminRetrainResponse(BaseModel):
    status: str
    message: str
    job_id: Optional[str]         = None
    started_at: Optional[str]     = None


class AdminRetrainStatusResponse(BaseModel):
    status: str
    job_id: Optional[str]         = None
    started_at: Optional[str]     = None
    finished_at: Optional[str]    = None
    return_code: Optional[int]    = None
    log_tail: Optional[str]       = None
