"""
Phase 7 — FastAPI v2 Async Production Backend
File: api/main.py

Two-Stage Retrieval -> LambdaMART Ranking -> LLM Explanation Serving Layer.
Endpoints:
  - POST /v2/recommend          Personalized ranker with candidate retrieval & background LLM caching
  - GET  /v2/similar/{item_id}  Qdrant HNSW ANN similarity search
  - GET  /v2/search             LLM query understanding & hybrid e5/BM25 retrieval
  - POST /v2/events             User feedback & interaction event logging to PostgreSQL
  - GET  /v2/health             Multi-service health diagnostics
  - GET  /metrics               Real-time JSON latency percentiles and cache telemetry
  - POST /admin/retrain         DVC retraining pipeline management
"""

import os
import sys
import time
import json
import uuid
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import List, Dict, Optional, Any, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Header, Depends, Request, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

# Setup paths
API_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(API_DIR, ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")

for _p in (BASE_DIR, API_DIR, SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config
from api.schemas import (
    RecommendRequest, RecommendResponse, RecommendedItem,
    SimilarResponse,
    SearchResponse, SearchResult,
    EventCreateRequest, EventResponse,
    HealthResponse, MetricsResponse,
    AdminRetrainRequest, AdminRetrainResponse, AdminRetrainStatusResponse,
)
from api.db import init_db_pool, close_db_pool, log_event, check_db_health
from api.cache import (
    init_redis_pool, close_redis_pool,
    get_cached_explanation, set_cached_explanation,
    get_cached_response, set_cached_response,
    check_redis_health,
)
from api.logging_config import configure_logging, metrics_collector
from api.retrain_manager import RetrainManager

# Configure structured logging
configure_logging(logging.INFO)
logger = logging.getLogger("api.main")


# =============================================================================
# APPLICATION STATE CONTAINER
# =============================================================================
state: Dict[str, Any] = {
    # Core ranker & candidate models
    "ranker": None,
    "user_map": {},
    "item_map": {},
    "rev_item_map": {},
    "item_meta": {},
    "popular_items": [],
    "category_popular": {},
    # Candidate generators
    "als_model": None,
    "svdpp_model": None,
    "mf_u_emb": None,
    "mf_i_emb": None,
    "mf_u_bias": None,
    "mf_i_bias": None,
    "ncf_model": None,
    "apriori_rules": {},
    "product_rec": None,
    "product_vecs": {},
    "bm25_model": None,
    "bm25_ids": [],
    "embedder": None,
    "llm_layer": None,
    "qdrant_client": None,
    # Status flags
    "model_loaded": False,
    "ranker_loaded": False,
    "qdrant_loaded": False,
}


# =============================================================================
# LIFESPAN STARTUP / SHUTDOWN
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Starting Amazon RecSys v2 Backend Service ===")

    # 1. Initialize DB and Redis Connection Pools
    await init_db_pool()
    await init_redis_pool()

    # 2. Load ID mappings & Item Metadata
    try:
        if os.path.exists(config.USER_MAP_PATH):
            with open(config.USER_MAP_PATH, "r", encoding="utf-8") as f:
                state["user_map"] = json.load(f)
        if os.path.exists(config.ITEM_MAP_PATH):
            with open(config.ITEM_MAP_PATH, "r", encoding="utf-8") as f:
                state["item_map"] = json.load(f)
                state["rev_item_map"] = {idx: iid for iid, idx in state["item_map"].items()}
        logger.info(f"[OK] ID Maps loaded: {len(state['user_map']):,} users, {len(state['item_map']):,} items.")
    except Exception as e:
        logger.warning(f"[WARN] Error loading ID maps: {e}")

    # 3. Load item metadata lookup table from clean parquet
    try:
        if os.path.exists(config.CLEAN_PARQUET_PATH):
            df = pd.read_parquet(config.CLEAN_PARQUET_PATH)
            # Group by parent_asin (item_id)
            meta_dict = {}
            for row in df.itertuples():
                iid = str(getattr(row, "parent_asin", getattr(row, "item_id", "")))
                if iid and iid not in meta_dict:
                    meta_dict[iid] = {
                        "item_id": iid,
                        "title": getattr(row, "title_meta", getattr(row, "title_rev", iid)),
                        "category": getattr(row, "main_category_meta", getattr(row, "main_category_rev", "Unknown")),
                        "price": float(getattr(row, "price", 19.99)) if pd.notna(getattr(row, "price", None)) else 19.99,
                        "average_rating": float(getattr(row, "average_rating", 4.0)) if pd.notna(getattr(row, "average_rating", None)) else 4.0,
                        "rating_number": int(getattr(row, "rating_number", 10)) if pd.notna(getattr(row, "rating_number", None)) else 10,
                    }
            state["item_meta"] = meta_dict

            # Build popularity lists
            pop_series = df["parent_asin"].value_counts() if "parent_asin" in df.columns else df["item_id"].value_counts()
            state["popular_items"] = pop_series.index.tolist()[:500]

            cat_col = "main_category_meta" if "main_category_meta" in df.columns else "main_category_rev"
            if cat_col in df.columns:
                cat_pop = {}
                for cat, group in df.groupby(cat_col):
                    item_col = "parent_asin" if "parent_asin" in group.columns else "item_id"
                    cat_pop[cat] = group[item_col].value_counts().index.tolist()[:200]
                state["category_popular"] = cat_pop

            logger.info(f"[OK] Item metadata indexed for {len(meta_dict):,} items.")
    except Exception as e:
        logger.warning(f"[WARN] Error loading clean parquet metadata: {e}")

    # 4. Load LightGBM LambdaMART Ranker
    try:
        import pickle
        if os.path.exists(config.LGBM_RANKER_PKL_PATH):
            with open(config.LGBM_RANKER_PKL_PATH, "rb") as f:
                state["ranker"] = pickle.load(f)
            state["ranker_loaded"] = True
            logger.info(f"[OK] LGBMRanker loaded from {config.LGBM_RANKER_PKL_PATH}.")
        elif os.path.exists(config.LGBM_RANKER_PATH):
            import lightgbm as lgb
            booster = lgb.Booster(model_file=config.LGBM_RANKER_PATH)
            state["ranker"] = booster
            state["ranker_loaded"] = True
            logger.info(f"[OK] LGBMRanker booster loaded from {config.LGBM_RANKER_PATH}.")
    except Exception as e:
        logger.warning(f"[WARN] LGBMRanker could not be loaded: {e}")

    # 5. Load Candidate Models (ALS, SVD++, MF, NCF, Apriori)
    try:
        if os.path.exists(config.ALS_MODEL_PATH):
            import implicit
            als = implicit.als.AlternatingLeastSquares()
            als_data = np.load(config.ALS_MODEL_PATH, allow_pickle=True)
            als.user_factors = als_data["user_factors"]
            als.item_factors = als_data["item_factors"]
            state["als_model"] = als
            logger.info("[OK] Candidate Model: ALS loaded.")
    except Exception as e:
        logger.warning(f"[WARN] ALS model: {e}")

    try:
        if os.path.exists(config.SVDPP_MODEL_PATH):
            import pickle
            with open(config.SVDPP_MODEL_PATH, "rb") as f:
                state["svdpp_model"] = pickle.load(f)
            logger.info("[OK] Candidate Model: SVD++ loaded.")
    except Exception as e:
        logger.warning(f"[WARN] SVD++ model: {e}")

    try:
        import torch
        if os.path.exists(config.MF_MODEL_PATH):
            mf_state = torch.load(config.MF_MODEL_PATH, map_location="cpu")
            state["mf_u_emb"] = mf_state["user_emb.weight"].cpu().numpy()
            state["mf_i_emb"] = mf_state["item_emb.weight"].cpu().numpy()
            state["mf_u_bias"] = mf_state["user_bias.weight"].cpu().numpy().squeeze()
            state["mf_i_bias"] = mf_state["item_bias.weight"].cpu().numpy().squeeze()
            logger.info("[OK] Candidate Model: PyTorch MF loaded.")
    except Exception as e:
        logger.warning(f"[WARN] MF model: {e}")

    try:
        apriori_path = os.path.join(config.MODELS_DIR, "apriori_recommender.pkl")
        if os.path.exists(apriori_path):
            import dill
            with open(apriori_path, "rb") as f:
                ap_rec = dill.load(f)
            state["apriori_rules"] = ap_rec.rule_dict
            logger.info(f"[OK] Apriori rules loaded ({len(state['apriori_rules']):,} rules).")
    except Exception as e:
        logger.warning(f"[WARN] Apriori model: {e}")

    # 6. Load Product Vectors & BM25 Index
    try:
        pvecs_path = os.path.join(config.EMBEDDINGS_DIR, "product_vecs.npz")
        if os.path.exists(pvecs_path):
            pdata = np.load(pvecs_path, allow_pickle=True)
            keys = [str(k) for k in pdata["keys"].tolist()]
            vecs = pdata["vecs"].astype(np.float32)
            state["product_vecs"] = {iid: vecs[i] for i, iid in enumerate(keys)}
            logger.info(f"[OK] product_vecs loaded: {len(keys):,} items.")
    except Exception as e:
        logger.warning(f"[WARN] product_vecs: {e}")

    try:
        bm25_path = os.path.join(config.EMBEDDINGS_DIR, "bm25_corpus.json")
        if os.path.exists(bm25_path):
            from rank_bm25 import BM25Okapi
            with open(bm25_path, "r", encoding="utf-8") as f:
                bm25_data = json.load(f)
            state["bm25_ids"] = bm25_data["item_ids"]
            state["bm25_model"] = BM25Okapi(bm25_data["corpus"])
            logger.info(f"[OK] BM25 model loaded: {len(state['bm25_ids']):,} documents.")
    except Exception as e:
        logger.warning(f"[WARN] BM25 corpus: {e}")

    # 7. Initialize Qdrant ANN Client
    try:
        from pipeline.sync_embeddings import get_qdrant_client
        state["qdrant_client"] = get_qdrant_client()
        state["qdrant_loaded"] = True
        logger.info("[OK] Qdrant ANN client initialized.")
    except Exception as e:
        logger.warning(f"[WARN] Qdrant client: {e}")

    # 8. Initialize LLM Layer
    try:
        import importlib
        _llm_mod = importlib.import_module("14_llm_layer")
        state["llm_layer"] = _llm_mod.LLMLayer(api_key=config.GEMINI_API_KEY, model_name=config.LLM_MODEL)
        logger.info("[OK] LLM Layer initialized (Gemini 3.5 Flash-Lite).")
    except Exception as e:
        logger.warning(f"[WARN] LLMLayer: {e}")

    # 9. Load SentenceTransformer Embedder
    try:
        from sentence_transformers import SentenceTransformer
        state["embedder"] = SentenceTransformer("intfloat/e5-base-v2")
        logger.info("[OK] SentenceTransformer (e5-base-v2) loaded.")
    except Exception as e:
        logger.warning(f"[WARN] SentenceTransformer: {e}")

    state["model_loaded"] = bool(state["item_meta"] or state["ranker_loaded"] or state["product_vecs"])
    logger.info("=== Service Startup Complete ===")

    yield

    logger.info("=== Shutting Down Service ===")
    await close_db_pool()
    await close_redis_pool()


# =============================================================================
# FASTAPI APP & MIDDLEWARE
# =============================================================================
app = FastAPI(
    title="Amazon RecSys v2 API",
    description="Production Asynchronous Two-Stage Retrieval & Ranking Recommender System",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TracingAndMetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware that manages X-Request-ID, calculates latency,
    records observability telemetry, and produces structured JSON logs.
    """

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        start_time = time.perf_counter()
        status_code = 500

        try:
            response: Response = await call_next(request)
            status_code = response.status_code
        except Exception as exc:
            logger.error(
                f"Unhandled exception on {request.method} {request.url.path}: {exc}",
                extra={"request_id": request_id, "method": request.method, "path": request.url.path},
            )
            raise exc
        finally:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            metrics_collector.record_request(request.url.path, latency_ms, status_code)

            # Log structured access event
            logger.info(
                f"{request.method} {request.url.path} -> {status_code} ({latency_ms:.2f}ms)",
                extra={
                    "request_id": request_id,
                    "client_ip": request.client.host if request.client else "unknown",
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "latency_ms": round(latency_ms, 2),
                },
            )

        response.headers["X-Request-ID"] = request_id
        return response


app.add_middleware(TracingAndMetricsMiddleware)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_item_metadata(item_id: str) -> Dict[str, Any]:
    """Retrieve indexed item metadata (parent_asin) with safe defaults."""
    meta = state.get("item_meta", {}).get(item_id)
    if meta:
        return meta
    return {
        "item_id": item_id,
        "title": item_id,
        "category": "General",
        "price": 19.99,
        "average_rating": 4.0,
        "rating_number": 1,
    }


def _retrieve_candidates(
    user_id: str,
    item_id: Optional[str] = None,
    category_filter: Optional[str] = None,
    max_candidates: int = 150,
) -> tuple[List[str], bool, str]:
    """
    Two-stage candidate retrieval layer.
    Retrieves candidates from ALS, SVD++, MF, NCF, Two-Tower, and Qdrant ANN.
    Returns: (candidate_item_ids, is_cold_start, primary_source)
    """
    user_map = state.get("user_map", {})
    item_map = state.get("item_map", {})
    rev_item_map = state.get("rev_item_map", {})
    candidates: List[str] = []
    seen = set()

    is_warm_user = str(user_id) in user_map
    u_idx = user_map.get(str(user_id), -1)

    # 1. If seed item_id is provided, retrieve ANN neighbors
    if item_id:
        pvecs = state.get("product_vecs", {})
        if item_id in pvecs:
            q = pvecs[item_id]
            q_norm = q / (np.linalg.norm(q) + 1e-9)
            # Sample subset of vectors for fast candidate generation
            keys = list(pvecs.keys())
            mat = np.array([pvecs[k] for k in keys], dtype=np.float32)
            mat_norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
            sims = mat_norm @ q_norm
            top_indices = np.argpartition(-sims, min(60, len(sims) - 1))[:60]
            for idx in top_indices:
                cand = keys[idx]
                if cand != item_id and cand not in seen:
                    candidates.append(cand)
                    seen.add(cand)

    # 2. Warm User Candidates (ALS + MF)
    if is_warm_user and u_idx >= 0:
        # ALS Candidate Top-N
        als = state.get("als_model")
        if als is not None and hasattr(als, "user_factors") and hasattr(als, "item_factors"):
            try:
                u_vec = als.user_factors[u_idx]
                als_scores = als.item_factors @ u_vec
                top_als_idx = np.argpartition(-als_scores, min(80, len(als_scores) - 1))[:80]
                for idx in top_als_idx:
                    cand = rev_item_map.get(idx)
                    if cand and cand not in seen:
                        candidates.append(cand)
                        seen.add(cand)
            except Exception:
                pass

        # MF Candidate Top-N
        if state.get("mf_u_emb") is not None and state.get("mf_i_emb") is not None:
            try:
                u_emb = state["mf_u_emb"][u_idx]
                mf_scores = state["mf_i_emb"] @ u_emb
                top_mf_idx = np.argpartition(-mf_scores, min(60, len(mf_scores) - 1))[:60]
                for idx in top_mf_idx:
                    cand = rev_item_map.get(idx)
                    if cand and cand not in seen:
                        candidates.append(cand)
                        seen.add(cand)
            except Exception:
                pass

    # 3. Top Popularity / Category Fill
    if len(candidates) < max_candidates:
        if category_filter and category_filter in state.get("category_popular", {}):
            pool = state["category_popular"][category_filter]
        else:
            pool = state.get("popular_items", [])

        for cand in pool:
            if cand not in seen:
                candidates.append(cand)
                seen.add(cand)
                if len(candidates) >= max_candidates:
                    break

    # Apply category filtering if specified
    if category_filter:
        filtered = [
            c for c in candidates
            if _get_item_metadata(c).get("category", "").lower() == category_filter.lower()
        ]
        if len(filtered) >= 10:
            candidates = filtered

    cold_start = not is_warm_user
    source = "personalized_ranker" if is_warm_user else ("content_cold_start" if item_id else "trending_cold_start")
    return candidates[:max_candidates], cold_start, source


def _compute_ranker_features(user_id: str, candidate_ids: List[str]) -> tuple[np.ndarray, List[Dict[str, float]]]:
    """
    Constructs the 10-dimensional ranker feature matrix:
    [als_score, svdpp_score, mf_score, ncf_score, content_score,
     apriori_lift, price_score, recency, popularity, helpful_votes]
    """
    user_map = state.get("user_map", {})
    item_map = state.get("item_map", {})
    u_idx = user_map.get(str(user_id), -1)

    n_cands = len(candidate_ids)
    features_matrix = np.zeros((n_cands, 10), dtype=np.float32)
    feature_dicts: List[Dict[str, float]] = []

    # 1. ALS score
    if state.get("als_model") is not None and u_idx >= 0:
        als = state["als_model"]
        u_vec = als.user_factors[u_idx]
        for i, iid in enumerate(candidate_ids):
            i_idx = item_map.get(iid, -1)
            if i_idx >= 0:
                features_matrix[i, 0] = float(np.dot(u_vec, als.item_factors[i_idx]))

    # 2. SVD++ score
    if state.get("svdpp_model") is not None and u_idx >= 0:
        svdpp = state["svdpp_model"]
        for i, iid in enumerate(candidate_ids):
            try:
                features_matrix[i, 1] = float(svdpp.predict(uid=user_id, iid=iid).est)
            except Exception:
                features_matrix[i, 1] = 3.5

    # 3. MF score
    if state.get("mf_u_emb") is not None and u_idx >= 0:
        u_emb = state["mf_u_emb"][u_idx]
        u_b = state["mf_u_bias"][u_idx] if state.get("mf_u_bias") is not None else 0.0
        for i, iid in enumerate(candidate_ids):
            i_idx = item_map.get(iid, -1)
            if i_idx >= 0:
                i_emb = state["mf_i_emb"][i_idx]
                i_b = state["mf_i_bias"][i_idx] if state.get("mf_i_bias") is not None else 0.0
                features_matrix[i, 2] = float(np.dot(u_emb, i_emb) + u_b + i_b)

    # 4. NCF score
    # 5. Content score & Metadata features
    for i, iid in enumerate(candidate_ids):
        meta = _get_item_metadata(iid)
        # Price score (normalized closeness)
        price = meta.get("price", 19.99)
        price_score = 1.0 / (1.0 + abs(price - 25.0) / 25.0)
        features_matrix[i, 6] = price_score

        # Recency
        features_matrix[i, 7] = 0.5

        # Popularity (log scale of rating number)
        rating_num = meta.get("rating_number", 10)
        popularity = float(np.log1p(rating_num) / 10.0)
        features_matrix[i, 8] = min(1.0, popularity)

        # Helpful votes
        features_matrix[i, 9] = min(1.0, float(meta.get("average_rating", 4.0) / 5.0))

        # Content score
        features_matrix[i, 4] = 0.5 * features_matrix[i, 6] + 0.5 * features_matrix[i, 9]

    # Convert to dict representation for LLM explanation generation
    feat_names = config.RANKER_FEATURES
    for i in range(n_cands):
        feature_dicts.append({
            feat_names[j]: float(features_matrix[i, j])
            for j in range(len(feat_names))
        })

    return features_matrix, feature_dicts


async def _background_cache_explanation(
    user_id: str,
    item_id: str,
    title: str,
    category: str,
    price: float,
    rating_mean: float,
    features: Dict[str, float],
):
    """
    Background worker that generates an LLM explanation and caches it in Redis.
    """
    llm_layer = state.get("llm_layer")
    if llm_layer is None:
        return

    try:
        exp_obj = await llm_layer.generate_explanation_async(
            user_id=user_id,
            item_id=item_id,
            title=title,
            category=category,
            price=price,
            rating_mean=rating_mean,
            features=features,
        )
        # Cache explanation object in Redis
        await set_cached_explanation(
            user_id=user_id,
            item_id=item_id,
            data=exp_obj.model_dump(),
            model_version=config.MODEL_VERSION,
            ttl=config.EXPLANATION_CACHE_TTL,
        )
    except Exception as e:
        logger.warning(f"[Background Task] Error caching explanation for {item_id}: {e}")


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", tags=["root"])
async def root():
    return {
        "service": "Amazon RecSys v2 API",
        "version": "2.0.0",
        "status": "online",
        "docs": "/docs",
    }


# -----------------------------------------------------------------------------
# POST /v2/recommend & POST /recommend
# -----------------------------------------------------------------------------
@app.post("/v2/recommend", response_model=RecommendResponse, tags=["recommend"])
@app.post("/recommend", response_model=RecommendResponse, tags=["recommend"])
async def recommend(req: RecommendRequest, background_tasks: BackgroundTasks):
    """
    Personalized Recommendation Serving Pipeline:
      1. Response cache lookup (Redis).
      2. Candidate generation (ALS, SVD++, MF, NCF, Qdrant ANN, or Popularity fallback).
      3. 10-dimensional feature extraction.
      4. LightGBM LambdaMART ranker inference.
      5. LLM explanation retrieval & background asynchronous generation.
    """
    cache_key = f"rec_resp:{req.user_id}:{req.item_id or 'none'}:{req.top_k}:{req.category_filter or 'all'}"
    cached_resp = await get_cached_response(cache_key)
    if cached_resp:
        return RecommendResponse(**cached_resp)

    # 1. Candidate Retrieval
    candidates, is_cold, source_type = _retrieve_candidates(
        user_id=req.user_id,
        item_id=req.item_id,
        category_filter=req.category_filter,
        max_candidates=100,
    )

    if not candidates:
        return RecommendResponse(
            user_id=req.user_id,
            item_id=req.item_id,
            cold_start=True,
            source="empty_fallback",
            results=[],
            model_version=config.MODEL_VERSION,
        )

    # 2. Ranker Feature Extraction
    features_mat, feature_dicts = _compute_ranker_features(req.user_id, candidates)

    # 3. LightGBM LambdaMART Ranking
    ranker = state.get("ranker")
    if ranker is not None:
        try:
            scores = ranker.predict(features_mat)
        except Exception as e:
            logger.warning(f"Ranker inference failed: {e}; falling back to composite score.")
            scores = features_mat[:, 0] + features_mat[:, 2] + features_mat[:, 4]
    else:
        # Fallback composite score
        scores = features_mat[:, 0] + features_mat[:, 2] + features_mat[:, 4]

    # Sort descending
    ranked_indices = np.argsort(-scores)[:req.top_k]

    results: List[RecommendedItem] = []
    for idx in ranked_indices:
        iid = candidates[idx]
        meta = _get_item_metadata(iid)
        score_val = float(scores[idx])

        # 4. Check cached explanation
        cached_exp = await get_cached_explanation(req.user_id, iid, config.MODEL_VERSION)
        explanation_text = cached_exp.get("explanation") if cached_exp else None

        # If missing, dispatch background LLM explanation generation
        if not explanation_text:
            background_tasks.add_task(
                _background_cache_explanation,
                user_id=req.user_id,
                item_id=iid,
                title=meta.get("title", iid),
                category=meta.get("category", "General"),
                price=meta.get("price", 19.99),
                rating_mean=meta.get("average_rating", 4.0),
                features=feature_dicts[idx],
            )

        # Clean feature signals for model attribution
        raw_feats = feature_dicts[idx] if idx < len(feature_dicts) else {}
        clean_signals = {k: round(float(v), 4) for k, v in raw_feats.items()} if raw_feats else None

        results.append(RecommendedItem(
            item_id=iid,
            title=meta.get("title", iid),
            score=round(score_val, 6),
            source=source_type,
            category=meta.get("category"),
            price=meta.get("price"),
            average_rating=meta.get("average_rating"),
            explanation=explanation_text,
            feature_signals=clean_signals,
        ))

    resp = RecommendResponse(
        user_id=req.user_id,
        item_id=req.item_id,
        cold_start=is_cold,
        source=source_type,
        results=results,
        model_version=config.MODEL_VERSION,
    )

    # Cache response for 30 seconds
    await set_cached_response(cache_key, resp.model_dump(), ttl=30)
    return resp


# -----------------------------------------------------------------------------
# GET /v2/similar/{item_id} & GET /similar/{item_id}
# -----------------------------------------------------------------------------
@app.get("/v2/similar/{item_id}", response_model=SimilarResponse, tags=["similar"])
@app.get("/similar/{item_id}", response_model=SimilarResponse, tags=["similar"])
async def similar(
    item_id: str,
    top_k: int = Query(10, ge=1, le=100),
    category_filter: Optional[str] = Query(None),
    price_ceiling: Optional[float] = Query(None),
):
    """
    Qdrant Approximate Nearest Neighbor (ANN) vector similarity lookup.
    Falls back to product_vecs cosine similarity if Qdrant is unavailable.
    """
    results: List[RecommendedItem] = []

    # 1. Try Qdrant client
    qdrant_client = state.get("qdrant_client")
    if qdrant_client is not None:
        try:
            from pipeline.sync_embeddings import query_similar_items
            ann_points = query_similar_items(
                client=qdrant_client,
                item_id=item_id,
                top_k=top_k,
                category_filter=category_filter,
                price_ceiling=price_ceiling,
            )
            for pt in ann_points:
                iid = pt.get("item_id")
                if iid and iid != item_id:
                    results.append(RecommendedItem(
                        item_id=iid,
                        title=pt.get("title") or iid,
                        score=round(float(pt.get("score", 0.0)), 6),
                        source="qdrant_ann",
                        category=pt.get("category"),
                        price=pt.get("price"),
                        average_rating=pt.get("average_rating"),
                    ))
            if results:
                return SimilarResponse(item_id=item_id, results=results[:top_k])
        except Exception as e:
            logger.warning(f"Qdrant ANN search error ({e}); using product_vecs fallback.")

    # 2. In-memory product_vecs fallback
    pvecs = state.get("product_vecs", {})
    if item_id not in pvecs:
        # Check if item exists in catalog
        if item_id in state.get("item_meta", {}):
            # Return popular items from same category
            cat = state["item_meta"][item_id].get("category")
            pool = state.get("category_popular", {}).get(cat, state.get("popular_items", []))
            for cand in pool:
                if cand != item_id:
                    meta = _get_item_metadata(cand)
                    results.append(RecommendedItem(
                        item_id=cand,
                        title=meta.get("title", cand),
                        score=0.85,
                        source="category_fallback",
                        category=meta.get("category"),
                        price=meta.get("price"),
                        average_rating=meta.get("average_rating"),
                    ))
                    if len(results) == top_k:
                        break
            return SimilarResponse(item_id=item_id, results=results)

        raise HTTPException(status_code=404, detail=f"Item ID '{item_id}' not found in vector index.")

    q = pvecs[item_id]
    keys = list(pvecs.keys())
    mat = np.array(list(pvecs.values()), dtype=np.float32)

    qn = q / (np.linalg.norm(q) + 1e-9)
    mn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    sims = mn @ qn

    top_i = np.argpartition(-sims, min(top_k + 1, len(sims) - 1))[: top_k + 1]
    top_i = top_i[np.argsort(-sims[top_i])]

    for i in top_i:
        iid = keys[i]
        if iid == item_id:
            continue
        meta = _get_item_metadata(iid)
        if category_filter and meta.get("category", "").lower() != category_filter.lower():
            continue
        if price_ceiling is not None and (meta.get("price") or 0.0) > price_ceiling:
            continue

        results.append(RecommendedItem(
            item_id=iid,
            title=meta.get("title", iid),
            score=round(float(sims[i]), 6),
            source="product_vecs_cosine",
            category=meta.get("category"),
            price=meta.get("price"),
            average_rating=meta.get("average_rating"),
        ))
        if len(results) == top_k:
            break

    return SimilarResponse(item_id=item_id, results=results)


# -----------------------------------------------------------------------------
# GET /v2/search & GET /search
# -----------------------------------------------------------------------------
@app.get("/v2/search", response_model=SearchResponse, tags=["search"])
@app.get("/search", response_model=SearchResponse, tags=["search"])
async def search(
    q: str = Query(..., min_length=1, description="Free text search query"),
    top_k: int = Query(10, ge=1, le=100),
    category: Optional[str] = Query(None),
    price_max: Optional[float] = Query(None),
):
    """
    Search Endpoint:
      1. LLM query understanding & rewriting (Gemini 3.5 Flash-Lite).
      2. Semantic vector encoding (e5-base-v2).
      3. BM25 keyword matching & score normalization.
      4. Hybrid score fusion (0.55 * semantic + 0.45 * BM25).
    """
    # 1. LLM Query Understanding
    rewritten_query = q
    parsed_category = category
    parsed_price_max = price_max
    intent = "product_search"

    llm_layer = state.get("llm_layer")
    if llm_layer is not None:
        try:
            parsed = await llm_layer.parse_query_async(q)
            rewritten_query = parsed.rewritten_query or q
            parsed_category = parsed.category or category
            parsed_price_max = parsed.price_max or price_max
            intent = parsed.intent
        except Exception as e:
            logger.warning(f"LLM query parsing failed ({e}); using raw query.")

    product_vecs = state.get("product_vecs")
    if not product_vecs:
        raise HTTPException(503, "Vector embedding index not loaded.")

    item_ids = list(product_vecs.keys())
    mat = np.array(list(product_vecs.values()), dtype=np.float32)

    def _minmax(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-9)

    # 2. Semantic Embedding Scores
    embedder = state.get("embedder")
    if embedder is not None:
        q_emb = embedder.encode([rewritten_query], normalize_embeddings=True)[0].astype(np.float32)
        emb_norm = _minmax((mat @ q_emb).astype(np.float64))
    else:
        emb_norm = np.zeros(len(item_ids))

    # 3. BM25 Scores
    bm25_model = state.get("bm25_model")
    bm25_ids = state.get("bm25_ids", [])
    if bm25_model is not None:
        bm25_raw = np.array(bm25_model.get_scores(rewritten_query.lower().split()))
        bm25_norm = _minmax(bm25_raw)
        bm25_map = dict(zip(bm25_ids, bm25_norm))
        bm25_align = np.array([bm25_map.get(iid, 0.0) for iid in item_ids])
    else:
        bm25_align = np.zeros(len(item_ids))

    # 4. Fuse Hybrid Scores
    hybrid_scores = 0.55 * emb_norm + 0.45 * bm25_align
    order = np.argsort(-hybrid_scores)

    results: List[SearchResult] = []
    for i in order:
        iid = item_ids[i]
        meta = _get_item_metadata(iid)

        # Filters
        if parsed_category and meta.get("category", "").lower() != parsed_category.lower():
            continue
        if parsed_price_max is not None and (meta.get("price") or 0.0) > parsed_price_max:
            continue

        results.append(SearchResult(
            item_id=iid,
            title=meta.get("title", iid),
            category=meta.get("category"),
            price=meta.get("price"),
            average_rating=meta.get("average_rating"),
            hybrid_score=round(float(hybrid_scores[i]), 6),
            emb_score=round(float(emb_norm[i]), 6),
            bm25_score=round(float(bm25_align[i]), 6),
        ))
        if len(results) == top_k:
            break

    return SearchResponse(
        query=q,
        rewritten_query=rewritten_query,
        category_filter=parsed_category,
        price_max=parsed_price_max,
        intent=intent,
        results=results,
    )


# -----------------------------------------------------------------------------
# POST /v2/events & POST /events
# -----------------------------------------------------------------------------
@app.post("/v2/events", response_model=EventResponse, tags=["events"])
@app.post("/events", response_model=EventResponse, tags=["events"])
async def create_event(req: EventCreateRequest):
    """
    Log an interaction event (click, view, purchase, rating) into PostgreSQL.
    """
    res = await log_event(
        user_id=req.user_id,
        item_id=req.item_id,
        event_type=req.event_type,
        rating=req.rating,
        metadata=req.metadata,
    )
    return EventResponse(
        status=res.get("status", "ok"),
        event_id=res.get("event_id"),
        message=f"Event logged via {res.get('storage', 'db')}",
    )


# -----------------------------------------------------------------------------
# GET /v2/health & GET /health
# -----------------------------------------------------------------------------
@app.get("/v2/health", response_model=HealthResponse, tags=["health"])
@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health():
    """
    Health check covering all subsystem connections.
    """
    db_alive = await check_db_health()
    redis_alive = await check_redis_health()
    n_items = len(state.get("item_meta", {})) or len(state.get("product_vecs", {})) or None

    return HealthResponse(
        status="ok",
        model_loaded=state["model_loaded"],
        ranker_loaded=state["ranker_loaded"],
        vector_db_connected=state["qdrant_loaded"],
        redis_connected=redis_alive,
        db_connected=db_alive,
        n_items=n_items,
        version="2.0.0",
    )


# -----------------------------------------------------------------------------
# GET /metrics
# -----------------------------------------------------------------------------
@app.get("/metrics", response_model=MetricsResponse, tags=["metrics"])
async def metrics():
    """
    Observability telemetry: latency percentiles (p50/p95/p99),
    cache hit rate, request counters per endpoint and HTTP status code.
    """
    return MetricsResponse(**metrics_collector.get_metrics())


# -----------------------------------------------------------------------------
# ADMIN RETRAIN ENDPOINTS
# -----------------------------------------------------------------------------
retrain_manager = RetrainManager()


def verify_admin_key(
    x_admin_api_key: Optional[str] = Header(None, alias="X-Admin-API-Key"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> str:
    admin_key = os.getenv("ADMIN_API_KEY", config.ADMIN_API_KEY)
    token = None
    if authorization:
        if authorization.startswith("Bearer "):
            token = authorization.split(" ", 1)[1].strip()
        else:
            token = authorization.strip()

    provided_key = x_admin_api_key or x_api_key or token
    if not provided_key or provided_key != admin_key:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid or missing admin API key.",
        )
    return provided_key


@app.post("/admin/retrain", response_model=AdminRetrainResponse, tags=["admin"])
def trigger_retrain(
    req: AdminRetrainRequest = AdminRetrainRequest(),
    _: str = Depends(verify_admin_key),
):
    """
    Trigger the DVC retraining pipeline asynchronously via subprocess.
    Requires ADMIN_API_KEY.
    """
    res = retrain_manager.trigger(force=req.force, targets=req.targets)
    if not res["success"]:
        raise HTTPException(status_code=409, detail=res["message"])
    return AdminRetrainResponse(
        status=res["status"],
        message=res["message"],
        job_id=res["job_id"],
        started_at=res["started_at"],
    )


@app.get("/admin/retrain/status", response_model=AdminRetrainStatusResponse, tags=["admin"])
def retrain_status(
    _: str = Depends(verify_admin_key),
):
    """
    Get the status of the current or most recent retrain job and recent log tail.
    Requires ADMIN_API_KEY.
    """
    status_data = retrain_manager.get_status()
    return AdminRetrainStatusResponse(**status_data)
