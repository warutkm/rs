"""
Phase 12 — FastAPI Service
File: api/schemas.py

Pydantic request/response models for all endpoints.
Validates inputs and shapes outputs consistently.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# =============================================================================
# SHARED
# =============================================================================

class RecommendedItem(BaseModel):
    """Single recommendation result — shared across all endpoints."""
    item_id: str
    title:   str
    score:   float
    source:  str


# =============================================================================
# POST /recommend
# =============================================================================

class RecommendRequest(BaseModel):
    item_id: str  = Field(...,  description="Seed product (item_id / parent_asin)")
    user_id: str  = Field(...,  description="User ID — pass any string for cold-start")
    top_k:   int  = Field(10,   ge=1, le=100, description="Number of results to return")

    model_config = {"json_schema_extra": {
        "example": {
            "item_id": "B08N5WRWNW",
            "user_id": "AHPI18EE22YZMH5TQ4YNLBAFZJA",
            "top_k":   10,
        }
    }}


class RecommendResponse(BaseModel):
    item_id:      str
    user_id:      str
    cold_start:   bool
    results:      List[RecommendedItem]


# =============================================================================
# GET /similar/{item_id}
# =============================================================================

class SimilarResponse(BaseModel):
    item_id: str
    results: List[RecommendedItem]


# =============================================================================
# GET /search?q=
# =============================================================================

class SearchResult(BaseModel):
    item_id:      str
    hybrid_score: float
    emb_score:    float
    bm25_score:   float


class SearchResponse(BaseModel):
    query:   str
    results: List[SearchResult]


# =============================================================================
# GET /health
# =============================================================================

class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    n_items:      Optional[int] = None
