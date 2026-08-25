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


# =============================================================================
# POST /admin/retrain & GET /admin/retrain/status
# =============================================================================

class AdminRetrainRequest(BaseModel):
    force:   bool                  = Field(False, description="Force re-execution of all DVC stages (--force)")
    targets: Optional[List[str]]   = Field(None,  description="Optional list of specific DVC stage targets to reproduce")


class AdminRetrainResponse(BaseModel):
    status:     str
    message:    str
    job_id:     Optional[str] = None
    started_at: Optional[str] = None


class AdminRetrainStatusResponse(BaseModel):
    status:      str
    job_id:      Optional[str] = None
    started_at:  Optional[str] = None
    finished_at: Optional[str] = None
    return_code: Optional[int] = None
    log_tail:    Optional[str] = None

