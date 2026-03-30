"""
Phase 12 — FastAPI Service
File: api/main.py

Endpoints:
  POST /recommend   {item_id: str, user_id: str, top_k: int=10}
  GET  /similar/{item_id}?top_k=10
  GET  /search?q=wireless+headphones&top_k=10
  GET  /health -> {status: ok, model_loaded: bool}

Cold-start detection in /recommend (exact workflow spec):
  if user_id not in cf_recommender.asin_to_item_idx:
      return content_only_path(item_id, top_k)   # new user
  if item_reviews[item_id] < 3:
      return embedding_similarity_path(item_id, top_k)  # new item
"""

import os
import sys
import json
import logging
from contextlib import asynccontextmanager
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException, Query

# ---------------------------------------------------------------------------
# Path setup — works whether uvicorn is run from E:\rs or E:\rs\api
# ---------------------------------------------------------------------------
API_DIR  = os.path.dirname(os.path.abspath(__file__))   # .../api/
BASE_DIR = os.path.abspath(os.path.join(API_DIR, "..")) # .../rs/
SRC_DIR  = os.path.join(BASE_DIR, "src")                # .../rs/src/

# schemas.py lives next to main.py in api/ — must be on path first
for _p in (API_DIR, SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# schemas is now importable regardless of working directory
from schemas import (
    RecommendRequest, RecommendResponse, RecommendedItem,
    SimilarResponse,
    SearchResponse, SearchResult,
    HealthResponse,
)

# ---------------------------------------------------------------------------
# Artefact paths  (absolute inside container: /app/...)
# ---------------------------------------------------------------------------
MODELS_DIR       = os.path.join(BASE_DIR, "models")
EMBED_DIR        = os.path.join(BASE_DIR, "embeddings")

HYBRID_PKL       = os.path.join(MODELS_DIR, "hybrid_recommender.pkl")
ALS_NPZ          = os.path.join(MODELS_DIR, "als_model.npz")
CF_PKL           = os.path.join(MODELS_DIR, "cf_recommender.pkl")
PRODUCT_PKL      = os.path.join(MODELS_DIR, "product_recommender.pkl")
PRODUCT_VECS_NPZ = os.path.join(EMBED_DIR,  "product_vecs.npz")
BM25_CORPUS_JSON = os.path.join(EMBED_DIR,  "bm25_corpus.json")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

# ---------------------------------------------------------------------------
# Application state — populated once at startup lifespan
# ---------------------------------------------------------------------------
state: dict = {
    "hybrid":         None,   # HybridRecommender
    "cf_recommender": None,   # CollaborativeFilteringRecommender
    "product_rec":    None,   # ProductRecommender
    "product_vecs":   None,   # dict[item_id -> np.ndarray]
    "item_reviews":   {},     # dict[item_id -> int]  (review counts)
    "bm25_model":     None,   # BM25Okapi | None
    "bm25_ids":       [],     # list[str]
    "als_model":      None,   # implicit ALS | None
    "model_loaded":   False,
    "embedder":       None,
}


# =============================================================================
# LIFESPAN — load all models on startup (workflow 12.1)
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=== Startup: loading models ===")

    # 1. HybridRecommender -------------------------------------------------
    try:
        import dill
        with open(HYBRID_PKL, "rb") as f:
            hybrid = dill.load(f)
        state["hybrid"]       = hybrid
        # _review_counts is built inside Phase 8 HybridRecommender.__init__
        state["item_reviews"] = hybrid._review_counts
        log.info(f"[OK] HybridRecommender  ({len(hybrid._review_counts):,} items tracked)")
    except Exception as e:
        log.error(f"[FAIL] HybridRecommender: {e}")

    # 2. cf_recommender  (for asin_to_item_idx cold-start gate) -----------
    # Pull from hybrid.cf_rec first (avoids double-loading)
    try:
        if state["hybrid"] is not None and hasattr(state["hybrid"], "cf_rec"):
            state["cf_recommender"] = state["hybrid"].cf_rec
            log.info("[OK] cf_recommender  ← hybrid.cf_rec")
        else:
            import dill
            with open(CF_PKL, "rb") as f:
                state["cf_recommender"] = dill.load(f)
            log.info(f"[OK] cf_recommender  ← {CF_PKL}")
    except Exception as e:
        log.warning(f"[WARN] cf_recommender not loaded: {e}")

    # 3. ProductRecommender (content_only_path) ----------------------------
    try:
        if state["hybrid"] is not None and hasattr(state["hybrid"], "product_rec"):
            state["product_rec"] = state["hybrid"].product_rec
            log.info("[OK] product_rec  ← hybrid.product_rec")
        else:
            import dill
            with open(PRODUCT_PKL, "rb") as f:
                state["product_rec"] = dill.load(f)
            log.info(f"[OK] product_rec  ← {PRODUCT_PKL}")
    except Exception as e:
        log.warning(f"[WARN] product_rec not loaded: {e}")

    # 4. product_vecs  (/similar + embedding_similarity_path) -------------
    try:
        data = np.load(PRODUCT_VECS_NPZ, allow_pickle=True)
        ids  = [str(i) for i in data["keys"].tolist()]
        vecs = data["vecs"].astype(np.float32)
        state["product_vecs"] = {iid: vecs[i] for i, iid in enumerate(ids)}
        log.info(f"[OK] product_vecs: {len(ids):,} items")
    except Exception as e:
        log.warning(f"[WARN] product_vecs not loaded: {e}")

    # 5. ALS model  (workflow 12.1 — loaded on startup) -------------------
    try:
        import implicit
        als  = implicit.als.AlternatingLeastSquares()
        data = np.load(ALS_NPZ, allow_pickle=True)
        als.user_factors = data["user_factors"]
        als.item_factors = data["item_factors"]
        state["als_model"] = als
        log.info(f"[OK] ALS model  ← {ALS_NPZ}")
    except Exception as e:
        log.warning(f"[WARN] ALS model not loaded: {e}")

    # 6. BM25 index
    try:
        from rank_bm25 import BM25Okapi
        with open(BM25_CORPUS_JSON, "r") as f:
            bm25_data = json.load(f)
        state["bm25_ids"]   = bm25_data["item_ids"]
        state["bm25_model"] = BM25Okapi(bm25_data["corpus"])
        log.info(f"[OK] BM25: {len(state['bm25_ids']):,} docs")
    except Exception as e:
        log.warning(f"[WARN] BM25 not loaded: {e}")

    # ✅ Step 2: LOAD EMBEDDING MODEL AT STARTUP
    try:
        from sentence_transformers import SentenceTransformer
        log.info("Loading SentenceTransformer at startup…")
        state["embedder"] = SentenceTransformer("intfloat/e5-base-v2")
        log.info("[OK] SentenceTransformer loaded")
    except Exception as e:
        log.warning(f"[WARN] SentenceTransformer not loaded: {e}")

    state["model_loaded"] = state["hybrid"] is not None
    log.info("=== Startup complete ===")

    yield

    log.info("=== Shutdown ===")


# =============================================================================
# APP
# =============================================================================
app = FastAPI(
    title       = "Amazon Hybrid Recommender API",
    description = "Phase 12 — DS11 Project",
    version     = "1.0.0",
    lifespan    = lifespan,
)

@app.get("/")
def root():
    return {"message": "Hybrid Recommender API is running"}

# =============================================================================
# HELPERS
# =============================================================================

def _get_title(item_id: str) -> str:
    hybrid = state.get("hybrid")
    if hybrid is None:
        return item_id
    return hybrid._item_meta.get(item_id, {}).get("title_meta", item_id)


# =============================================================================
# COLD-START PATHS  (exact names and logic from workflow spec)
# =============================================================================

def content_only_path(item_id: str, top_k: int) -> List[RecommendedItem]:
    """
    New-user cold-start.
    Triggered when: user_id not in cf_recommender.asin_to_item_idx
    Uses ProductRecommender (content) + embedding similarity to fill top_k.
    """
    results: List[RecommendedItem] = []

    product_rec = state.get("product_rec")
    if product_rec is not None:
        try:
            for r in product_rec.get_recommendations(item_id, top_n=top_k):
                results.append(RecommendedItem(
                    item_id = str(r["item_id"]),
                    title   = _get_title(str(r["item_id"])),
                    score   = round(float(r["score"]), 6),
                    source  = "content_cold_start",
                ))
        except Exception as e:
            log.warning(f"content_only_path: {e}")

    # Top up with semantic similarity if content came up short
    if len(results) < top_k:
        seen = {r.item_id for r in results}
        for r in embedding_similarity_path(item_id, top_k):
            if r.item_id not in seen:
                r.source = "semantic_cold_start"
                results.append(r)
                if len(results) == top_k:
                    break

    return results[:top_k]


def embedding_similarity_path(item_id: str, top_k: int) -> List[RecommendedItem]:
    """
    New-item cold-start.
    Triggered when: item_reviews[item_id] < 3
    Pure cosine similarity over product_vecs.
    """
    product_vecs = state.get("product_vecs") or {}
    if item_id not in product_vecs:
        return []

    q   = product_vecs[item_id]
    ids = list(product_vecs.keys())
    mat = np.array(list(product_vecs.values()), dtype=np.float32)

    qn   = q   / (np.linalg.norm(q)   + 1e-9)
    mn   = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    sims = mn @ qn

    top_i = np.argpartition(-sims, min(top_k + 1, len(sims) - 1))[: top_k + 1]
    top_i = top_i[np.argsort(-sims[top_i])]

    out: List[RecommendedItem] = []
    for i in top_i:
        iid = ids[i]
        if iid == item_id:
            continue
        out.append(RecommendedItem(
            item_id = iid,
            title   = _get_title(iid),
            score   = round(float(sims[i]), 6),
            source  = "embedding_similarity",
        ))
        if len(out) == top_k:
            break
    return out


# =============================================================================
# POST /recommend   {item_id: str, user_id: str, top_k: int=10}
# =============================================================================

@app.post("/recommend", response_model=RecommendResponse, tags=["recommend"])
def recommend(req: RecommendRequest):
    """
    Hybrid recommendations with explicit cold-start detection.

    Workflow spec (exact):
      if user_id not in cf_recommender.asin_to_item_idx:
          return content_only_path(item_id, top_k)   # new user
      if item_reviews[item_id] < 3:
          return embedding_similarity_path(item_id, top_k)  # new item
    """
    hybrid = state.get("hybrid")
    if hybrid is None:
        raise HTTPException(503, "HybridRecommender not loaded")

    item_id = req.item_id
    user_id = req.user_id
    top_k   = req.top_k

    # ── Gate 1: new user ────────────────────────────────────────────────────
    # Workflow: if user_id not in cf_recommender.asin_to_item_idx
    cf_rec = state.get("cf_recommender")
    if cf_rec is not None:
        # CollaborativeFilteringRecommender uses item_map for items
        # and user_map for users — asin_to_item_idx maps to item_map
        asin_to_item_idx = getattr(cf_rec, "asin_to_item_idx",
                           getattr(cf_rec, "item_map", {}))
        user_map         = getattr(cf_rec, "user_map", {})
        is_new_user      = str(user_id) not in user_map
    else:
        # Fallback: use hybrid's internal gate
        is_new_user = hybrid._is_new_user(str(user_id))

    if is_new_user:
        return RecommendResponse(
            item_id    = item_id,
            user_id    = user_id,
            cold_start = True,
            results    = content_only_path(item_id, top_k),
        )

    # ── Gate 2: new item ────────────────────────────────────────────────────
    # Workflow: if item_reviews[item_id] < 3
    item_reviews = state.get("item_reviews", {})
    if item_reviews.get(str(item_id), 0) < 3:
        return RecommendResponse(
            item_id    = item_id,
            user_id    = user_id,
            cold_start = True,
            results    = embedding_similarity_path(item_id, top_k),
        )

    # ── Warm path — full hybrid ──────────────────────────────────────────────
    try:
        raw = hybrid.final_recommendation(item_id, user_id, top_n=top_k)
    except Exception as e:
        log.error(f"/recommend warm path error: {e}")
        raise HTTPException(500, f"Recommendation failed: {e}")

    results = [
        RecommendedItem(
            item_id = r["item_id"],
            title   = r.get("title") or _get_title(r["item_id"]),
            score   = round(float(r["score"]), 6),
            source  = r.get("source", "hybrid"),
        )
        for r in raw
    ]

    return RecommendResponse(
        item_id    = item_id,
        user_id    = user_id,
        cold_start = False,
        results    = results,
    )


# =============================================================================
# GET /similar/{item_id}?top_k=10
# =============================================================================

@app.get("/similar/{item_id}", response_model=SimilarResponse, tags=["similar"])
def similar(item_id: str, top_k: int = Query(10, ge=1, le=100)):
    """
    Top-K similar items by product_vecs cosine similarity.
    Endpoint: GET /similar/{item_id}?top_k=10
    """
    if state.get("product_vecs") is None:
        raise HTTPException(503, "product_vecs not loaded")
    if item_id not in state["product_vecs"]:
        raise HTTPException(404, f"item_id '{item_id}' not found in product_vecs")

    results = embedding_similarity_path(item_id, top_k)
    return SimilarResponse(item_id=item_id, results=results)


# =============================================================================
# GET /search?q=wireless+headphones&top_k=10
# =============================================================================

@app.get("/search", response_model=SearchResponse, tags=["search"])
def search(
    q:     str = Query(..., min_length=1),
    top_k: int = Query(10, ge=1, le=100),
):
    product_vecs = state.get("product_vecs")
    if product_vecs is None:
        raise HTTPException(503, "product_vecs not loaded")

    HYBRID_EMB_W  = 0.55
    HYBRID_BM25_W = 0.45
    HYBRID_THRESH = 0.20
    HYBRID_MIN    = 20

    def _minmax(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-9)

    item_ids = list(product_vecs.keys())
    mat      = np.array(list(product_vecs.values()), dtype=np.float32)

    # ✅ Step 3: USE PRELOADED MODEL
    embedder = state.get("embedder")
    if embedder is None:
        log.warning("/search embedding unavailable: embedder not loaded")
        emb_norm = np.zeros(len(item_ids))
    else:
        q_emb = embedder.encode(
            [q], normalize_embeddings=True
        )[0].astype(np.float32)
        emb_norm = _minmax((mat @ q_emb).astype(np.float64))

    # BM25 (UNCHANGED)
    bm25_model = state.get("bm25_model")
    bm25_ids   = state.get("bm25_ids", [])
    if bm25_model is not None:
        bm25_raw   = np.array(bm25_model.get_scores(q.lower().split()))
        bm25_norm  = _minmax(bm25_raw)
        bm25_map   = dict(zip(bm25_ids, bm25_norm))
        bm25_align = np.array([bm25_map.get(iid, 0.0) for iid in item_ids])
    else:
        bm25_align = np.zeros(len(item_ids))

    # Fuse (UNCHANGED)
    hybrid_scores = HYBRID_EMB_W * emb_norm + HYBRID_BM25_W * bm25_align
    order  = np.argsort(-hybrid_scores)
    above  = [i for i in order if hybrid_scores[i] >= HYBRID_THRESH]
    picks  = above[:top_k] if len(above) >= HYBRID_MIN else list(order[:max(HYBRID_MIN, top_k)])
    picks  = picks[:top_k]

    results = [
        SearchResult(
            item_id      = item_ids[i],
            hybrid_score = round(float(hybrid_scores[i]), 6),
            emb_score    = round(float(emb_norm[i]),      6),
            bm25_score   = round(float(bm25_align[i]),    6),
        )
        for i in picks
    ]

    return SearchResponse(query=q, results=results)


# =============================================================================
# GET /health  ->  {status: ok, model_loaded: bool}
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["health"])
def health():
    """
    Liveness check.
    Returns: {status: ok, model_loaded: bool}
    """
    pv = state.get("product_vecs") or {}
    return HealthResponse(
        status       = "ok",
        model_loaded = state["model_loaded"],
        n_items      = len(pv) if pv else None,
    )
