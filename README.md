# Amazon RecSys v2 — Production Multi-Stage Hybrid Recommender Platform

[![CI/CD Pipeline](https://github.com/warutkm/rs/actions/workflows/retrain.yml/badge.svg)](https://github.com/warutkm/rs/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Next.js 14](https://img.shields.io/badge/Next.js-14.2-black.svg)](https://nextjs.org/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-red.svg)](https://qdrant.tech/)
[![LightGBM](https://img.shields.io/badge/LightGBM-LambdaMART-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-9cf.svg)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)

An enterprise-grade, two-stage **Retrieval → Ranking → LLM Explanation** hybrid recommendation platform trained on the **Amazon Reviews 2023** dataset (Video Games, Musical Instruments, and Software categories; 44,301 catalog items and 12,569 customer profiles).

---

## 1. System Architecture

```
                                 ┌─────────────────────────────────┐
                                 │     Next.js 14 Web Interface   │
                                 │  Personalization · Search · Ops │
                                 └────────────────┬────────────────┘
                                                  │ REST / JSON (Async HTTP)
                                 ┌────────────────▼────────────────┐
                                 │      FastAPI Serving Engine     │
                                 │  Tracing · Metrics · Lifespan   │
                                 └────────────────┬────────────────┘
                                                  │
                 ┌────────────────────────────────┴────────────────────────────────┐
                 │                                                                 │
  ┌──────────────▼───────────────┐                                  ┌──────────────▼───────────────┐
  │   Stage 1: Retrieval Layer   │                                  │   LLM Query Understanding    │
  │  (ANN + Multi-Tower Models)  │                                  │   (Gemini 3.5 Flash-Lite)    │
  │                              │                                  │                              │
  │  • Qdrant HNSW ANN (e5-base) │                                  │  • Intent classification     │
  │  • Implicit ALS (CF)         │                                  │  • Category/price extraction │
  │  • Surprise SVD++            │                                  │  • Semantic query rewrite    │
  │  • PyTorch Two-Tower Cosine  │                                  └──────────────┬───────────────┘
  │  • PyTorch Matrix Fact. (MF) │                                                 │
  │  • Apriori Co-occurrence     │                                                 │
  └──────────────┬───────────────┘                                                 │
                 │ Top ~150-200 Candidate Item IDs                                 │
                 ├─────────────────────────────────────────────────────────────────┘
                 │
  ┌──────────────▼────────────────────────────────┐
  │       Stage 2: Ranking Engine                 │
  │       (LightGBM LambdaMART / NDCG@10)         │
  │                                               │
  │  10-Dimensional Real-time Feature Vector:     │
  │  [ALS, SVD++, MF, NCF, Content, Apriori,      │
  │   Price Affinity, Recency, Pop, Helpfulness]  │
  └──────────────┬────────────────────────────────┘
                 │ Top K Ranked Items (e.g. Top 8)
  ┌──────────────▼────────────────────────────────┐
  │       Stage 3: LLM Explanation Layer          │
  │       (Gemini 3.5 Flash-Lite + Redis)         │
  │                                               │
  │  • Instant fallback & async Redis caching     │
  │  • Feature-grounded 1-sentence explanations   │
  └──────────────┬────────────────────────────────┘
                 │
  ┌──────────────┴───────────────┬───────────────────────────────┬───────────────────────────────┐
  │                              │                               │                               │
┌─▼────────────────────────────┐ ┌─▼───────────────────────────┐ ┌─▼───────────────────────────┐ ┌─▼───────────────────────────┐
│     Upstash / Local Redis    │ │      PostgreSQL Database    │ │   Qdrant Vector Database    │ │     MLflow Model Registry   │
│ Response & Explanation Cache │ │ Interaction & Feedback Log  │ │ 44,301 384-d e5 Embeddings  │ │ Experiment DS11-v2 Tracking │
└──────────────────────────────┘ └─────────────────────────────┘ └─────────────────────────────┘ └─────────────────────────────┘
```

---

## 2. Multi-Stage Pipeline & Technology Stack

| Layer | Implementation & Technologies | Key Capabilities |
|---|---|---|
| **Frontend Web App** | Next.js 14 App Router, TypeScript, Tailwind CSS, Lucide Icons | Real-time persona switcher, live recommendation rails, review-satisfaction ranking, hybrid search, observability dashboard |
| **Serving Backend** | FastAPI (Async), Pydantic v2, Starlette Middleware, Uvicorn | Sub-25ms response time, X-Request-ID distributed tracing, JSON `/metrics` latency percentiles |
| **Vector Retrieval (ANN)** | Qdrant (HNSW Index), `intfloat/e5-base-v2` dense embeddings | Sub-5ms cosine similarity search across 44,301 products with payload filtering |
| **Candidate Generators** | Implicit ALS, Surprise SVD++, PyTorch Matrix Factorization, PyTorch Neural CF, Apriori Association Rules | Collaborative, content, and frequent co-purchase candidate retrieval |
| **Two-Tower Neural Retrieval** | PyTorch Dual-Encoder (User Tower & Item Tower) | High-throughput semantic & interaction space dot-product candidate retrieval |
| **Ranking Engine** | LightGBM LambdaMART (`LGBMRanker`, NDCG@10 objective) | Re-ranks candidates using a 10-dimensional multi-model signal vector |
| **LLM Reasoning & NLP** | Google Gemini 3.5 Flash-Lite, TF-IDF + SVM, T5 Summarization | Natural language search query parsing and feature-grounded recommendation rationale |
| **Caching Layer** | Redis (`redis.asyncio`) with in-memory TTL fallback | Response caching (30s TTL) and explanation caching (24h TTL) |
| **Telemetry & Storage** | PostgreSQL (asyncpg / psycopg2) | Logs user interactions (`click`, `view`, `purchase`, `rating`, `cart`) for retraining |
| **Data & Pipeline DAG** | DVC (Data Version Control) | Deterministic, reproducible pipeline from raw ingestion to model evaluation (`dvc repro`) |
| **MLOps & Tracking** | MLflow (`DS11-v2` experiment) | Tracks hyperparameters, validation NDCG, HitRate@K, Precision@K, and model binaries |

---

## 3. Repository Structure

```
.
├── config.py                          # Centralized configuration, constants, and paths
├── requirements.txt                   # Pinned production dependencies
├── docker-compose.yml                 # Multi-service stack (api, postgres, redis, qdrant)
├── dvc.yaml                           # DVC reproducible ML pipeline DAG
├── dvc.lock                           # DVC state lockfile
├── GEMINI.md                          # Project rules & engineering conventions
├── PROJECT_MANIFEST.md                # Compact architectural manifest
├── RECSYS_V2_WORKFLOW_AND_DESIGN.md   # Master architectural specification
│
├── api/                               # FastAPI Serving Application
│   ├── main.py                        # Production async server (/v2/recommend, /v2/similar, etc.)
│   ├── schemas.py                     # Pydantic v2 validation models
│   ├── cache.py                       # Redis asynchronous caching & telemetry
│   ├── db.py                          # PostgreSQL asynchronous event logger
│   ├── logging_config.py              # Structured JSON logging & metrics collector
│   ├── retrain_manager.py             # Subprocess manager for DVC pipeline execution
│   └── Dockerfile                     # Production container image
│
├── pipeline/                          # Orchestration & Vector Sync
│   └── sync_embeddings.py             # Qdrant collection initialization & upsert
│
├── src/                               # ML Pipeline Stages
│   ├── 01_data_ingestion.py           # Raw data stream extraction (Amazon Reviews 2023)
│   ├── 02_preprocessing.py            # Imputation, feature engineering & parquet export
│   ├── 03_sentiment_nlp.py            # VADER / TextBlob & TF-IDF + SVM classifier
│   ├── 03_b_t5_summarization.py       # T5 abstractive review summarizer
│   ├── 04_apriori_recommender.py      # Frequent itemset mining & association rules
│   ├── 05_content_cf_recommender.py   # Content-based & item/user collaborative filtering
│   ├── 06_mf_ncf_pytorch.py           # PyTorch Matrix Factorization & Neural CF
│   ├── 07_semantic_search.py          # e5 dense embedding generation & BM25 indexer
│   ├── 08_hybrid_engine.py            # Baseline hybrid heuristic engine
│   ├── 09_als_svdpp.py                # Implicit ALS & Surprise SVD++ models
│   ├── 10_ab_comparison.ipynb         # Offline metric evaluation & A/B comparison suite
│   ├── 11_mlflow_report.ipynb         # MLflow metric leaderboard generator
│   ├── 12_ranker_features.py          # Ranker feature extraction & negative sampling
│   ├── 12_ranker.py                   # LightGBM LambdaMART ranker training
│   ├── 13_two_tower.py                # PyTorch Two-Tower dual encoder model
│   └── 14_llm_layer.py                # Gemini 3.5 Flash-Lite explanation & query parsing
│
├── tests/                             # Automated Test Suites (Pytest)
│   ├── test_api.py                    # Serving layer integration tests (14 tests)
│   ├── test_sync_embeddings.py        # Qdrant vector index & ANN tests
│   ├── test_ranker.py                 # Ranker feature extraction & LightGBM tests
│   ├── test_two_tower.py              # Two-Tower neural model tests
│   ├── test_llm_layer.py              # LLM explanation & query parsing tests
│   ├── test_admin_retrain.py          # Admin retrain authorization & execution tests
│   └── test_model_artifacts.py        # Model artifact persistence & MLflow verification
│
├── web/                               # Next.js 14 Production Web Application
│   ├── app/                           # App Router pages (/ , /product/[id], /search, /admin)
│   ├── components/                    # UI Components (ProductCard, Navbar, UserSwitcher, etc.)
│   ├── context/                       # React context (UserContext state management)
│   ├── lib/                           # Type-safe API client (RecSysAPI) and demo users
│   └── package.json                   # Web dependencies & Next.js build scripts
│
├── data/                              # [Git-ignored] Clean parquet datasets & ID mappings
├── embeddings/                        # [Git-ignored] Precomputed e5 .npy embeddings
├── models/                            # [Git-ignored] Serialized model artifacts (.pkl, .pth, .npz)
└── mlflow/                            # [Git-ignored] MLflow experiment logs (DS11-v2)
```

---

## 4. API Reference (v2 REST Interface)

All endpoints are strictly namespaced under `/v2/` with OpenAPI documentation available at `http://127.0.0.1:8000/docs`:

### `POST /v2/recommend`
Generates personalized rankings or cold-start recommendations.
```json
// Request
{
  "user_id": "AE3RQLFSVY5DOCCDWJIQRQVCDS4Q",
  "item_id": null,
  "top_k": 8,
  "category_filter": "Video Games",
  "product_type": null,
  "sort_by": "ranker"
}

// Response
{
  "user_id": "AE3RQLFSVY5DOCCDWJIQRQVCDS4Q",
  "item_id": null,
  "cold_start": false,
  "source": "personalized_ranker",
  "model_version": "v2.0",
  "results": [
    {
      "item_id": "B00HM1XPN4",
      "title": "Redragon S101 Gaming Keyboard, M601 Mouse, RGB Backlit",
      "score": 0.9421,
      "source": "personalized_ranker",
      "category": "Video_Games",
      "price": 35.99,
      "average_rating": 4.6,
      "explanation": "High collaborative match with your previous activity in Video Games.",
      "feature_signals": {
        "als_score": 0.882,
        "content_score": 0.915,
        "apriori_lift": 1.450,
        "popularity": 0.960
      }
    }
  ]
}
```

### `GET /v2/similar/{item_id}`
Retrieves nearest neighbors from the Qdrant HNSW vector index alongside the queried product's metadata.
- **Parameters**: `top_k` (default: 10), `category_filter` (optional), `price_ceiling` (optional).

### `GET /v2/search`
Executes hybrid semantic vector search (`e5-base-v2`) + lexical keyword matching (`BM25`) with LLM query understanding.
- **Parameters**: `q` (search query), `top_k`, `category`, `price_max`.

### `POST /v2/events`
Logs interaction events (`click`, `view`, `purchase`, `rating`, `cart`) into PostgreSQL for model retraining.

### `GET /v2/health` & `GET /metrics`
- `/v2/health`: Multi-subsystem connection diagnostics (DB, Redis, Vector DB, Ranker).
- `/metrics`: Structured JSON telemetry containing p50, p95, p99 latencies, cache hit rate, and request volume.

---

## 5. Quickstart & Local Reproduction

### Prerequisites
- Python 3.10+
- Node.js 18+ & npm
- Docker & Docker Compose

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/warutkm/rs.git
cd rs

# Python environment
python -m venv venv
# Windows: venv\Scripts\activate | Linux/macOS: source venv/bin/activate
pip install -r requirements.txt

# Web dependencies
cd web && npm install && cd ..
```

### 2. Launch Infrastructure Services
```bash
docker-compose up -d postgres redis qdrant
```

### 3. Run Pipeline Stages & Sync Embeddings
```bash
# Data preprocessing & feature engineering
python src/01_data_ingestion.py
python src/02_preprocessing.py

# Sync vectors into Qdrant HNSW collection
python pipeline/sync_embeddings.py --recreate

# Train LightGBM LambdaMART ranker
python src/12_ranker_features.py
python src/12_ranker.py
```

### 4. Execute Automated Test Suite
```bash
pytest tests/ -v
```

### 5. Launch Backend & Frontend
```bash
# Terminal 1: FastAPI Backend
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Next.js Frontend
cd web
npm run dev
```
Open **[http://localhost:3000](http://localhost:3000)** in your browser.

---

## 6. Engineering Invariants & Quality Standards

1. **Identifier Invariant**: `item_id == parent_asin` throughout all data frames, feature matrices, Qdrant payloads, PostgreSQL records, and API schemas.
2. **Deterministic Vector Upsert**: Vector IDs are derived using `uuid.uuid5(uuid.NAMESPACE_DNS, item_id)`.
3. **Reproducibility**: Pipeline DAG is versioned in `dvc.yaml` and executed via `dvc repro`.
4. **MLflow Tracking**: All model artifacts log parameters, metrics, and models to MLflow experiment `DS11-v2`.
5. **Observability**: Zero heavy monitoring containers; real-time telemetry is exposed through the structured `/metrics` endpoint.

---

## 7. License
Distributed under the MIT License. See `LICENSE` for more information.
