# Enterprise Hybrid Recommender Platform (Amazon Reviews 2023)

[![CI/CD Pipeline](https://github.com/warutkm/rs/actions/workflows/retrain.yml/badge.svg)](https://github.com/warutkm/rs/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-red.svg)](https://qdrant.tech/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-9cf.svg)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)

A production-grade, two-stage **Retrieval → Ranking** hybrid recommendation engine trained on the **Amazon Reviews 2023** dataset (Video Games, Musical Instruments, and Software categories).

---

## 1. System Architecture Overview

```
                         ┌─────────────────────────┐
                         │        Frontend         │
                         │    Next.js + Tailwind   │
                         └────────────┬────────────┘
                                      │ REST / JSON
                         ┌────────────▼────────────┐
                         │   FastAPI Async Server  │
                         │ Auth · Rate Limit · Log │
                         └────────────┬────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │  Query Rewrite — Gemini Flash-Lite  │
                    │  Structured Filter + Semantic Query │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │           Retrieval Layer           │
                    │  Qdrant HNSW (ANN) · ALS/SVD++/NCF  │
                    │  e5 Dense Embeddings · Apriori Lift │
                    └─────────────────┬───────────────────┘
                                      │ Top ~200 candidates
                    ┌─────────────────▼───────────────────┐
                    │      Ranking Stage — LightGBM       │
                    │             LambdaMART              │
                    │  Multi-model scores, price, recency │
                    └─────────────────┬───────────────────┘
                                      │ Top 10-20 ranked
                    ┌─────────────────▼───────────────────┐
                    │   Explanation — Gemini Flash-Lite   │
                    │   Async feature-grounded summary    │
                    └─────────────────┬───────────────────┘
                                      │
      ┌───────────────────────┬───────┴───────────────┬───────────────────────┐
      │                       │                       │                       │
┌─────▼──────────────┐  ┌─────▼──────────────┐  ┌─────▼──────────────┐  ┌─────▼──────────────┐
│    Upstash Redis   │  │   PostgreSQL DB    │  │ Qdrant Cloud / HNSW│  │  MLflow Registry   │
│ Response/LLM Cache │  │ Users/Events/Logs  │  │ 44k item vectors   │  │ Staging → Prod Gate│
└────────────────────┘  └────────────────────┘  └────────────────────┘  └────────────────────┘
```

---

## 2. Model Evolution & Core Technologies

| Stage | v1 Foundation (Complete) | v2 Enterprise Platform (Current Target) |
|---|---|---|
| **Retrieval / ANN** | In-memory `product_vecs` dict & linear scan | **Qdrant Vector DB** (HNSW index, sub-10ms cosine search, payload filtering) |
| **Ranking** | Hand-weighted heuristic union (`HybridRecommender`) | **Learned-to-Rank LightGBM LambdaMART** (`LGBMRanker`) on multi-model candidate features |
| **Candidate Generators** | Standalone ALS, SVD++, PyTorch MF, PyTorch NCF, Apriori | Multi-tower candidate pool feeds Stage 2 ranker |
| **Semantic Search** | `intfloat/e5-base-v2` dense + BM25 sparse hybrid | LLM query understanding + Qdrant hybrid retrieval |
| **Explanations** | Static summaries (T5) | Dynamic **Gemini 3.5 Flash-Lite** explanations grounded in ranker feature vectors |
| **Data Ingestion & DAG** | Python scripts with DVC stages | **DVC** pipeline DAG + GitHub Actions automated retrain workflow |
| **Backend & Cache** | FastAPI synchronous prototype | Async FastAPI, **Redis** response/explanation cache, **PostgreSQL** events |
| **Observability** | Console print statements | Structured JSON logging + `/metrics` endpoint (p50/p95/p99 latency, cache hit rate) |

---

## 3. Repository Structure

```
.
├── config.py                  # Centralized configuration, paths, hyperparameters
├── requirements.txt           # Pinned production dependencies
├── docker-compose.yml         # Local stack (FastAPI, PostgreSQL, Redis, Qdrant)
├── dvc.yaml                   # DVC reproducible ML pipeline DAG
├── GEMINI.md                  # Project rules & engineering conventions
├── PROJECT_MANIFEST.md        # Compact architectural directory manifest
├── RECSYS_V2_WORKFLOW_AND_DESIGN.md # Comprehensive v2 architectural design doc
│
├── .github/workflows/         # CI/CD & automation workflows
│   └── retrain.yml            # (Phase 2) Scheduled cron dvc repro retraining workflow
│
├── pipeline/                  # v2 Background sync & orchestration tasks
│   └── sync_embeddings.py     # Qdrant collection initialization & vector upsert
│
├── src/                       # Pipeline stage scripts (v1 baseline + v2 ranker)
│   ├── 01_data_ingestion.py   # Raw data stream ingestion & parent_asin alignment
│   ├── 02_preprocessing.py    # Imputation, feature engineering & parquet export
│   ├── 03_sentiment_nlp.py    # VADER / TextBlob & TF-IDF + SVM classifier
│   ├── 03_b_t5_summarization.py # T5 abstractive review summarizer
│   ├── 04_apriori_recommender.py # Frequent itemset mining & association rules
│   ├── 05_content_cf_recommender.py # Content-based & item/user collaborative filtering
│   ├── 06_mf_ncf_pytorch.py   # PyTorch Matrix Factorization & Neural CF models
│   ├── 07_semantic_search.py  # e5 dense embedding generation & BM25 indexer
│   ├── 08_hybrid_engine.py    # v1 hybrid heuristic engine
│   ├── 09_als_svdpp.py        # Implicit ALS & Surprise SVD++ models
│   ├── 10_ab_comparison.ipynb # Offline metric evaluation & A/B evaluation suite
│   └── 11_mlflow_report.ipynb # MLflow metric leaderboard generator
│
├── api/                       # Serving layer
│   ├── main.py                # FastAPI endpoints (/recommend, /similar, /health, /admin/retrain)
│   ├── retrain_manager.py     # (Phase 2) Subprocess manager for DVC pipeline execution
│   ├── schemas.py             # Pydantic request/response validation schemas
│   └── test_api.py            # API integration test suite
│
├── tests/                     # Automated test suites
│   ├── test_sync_embeddings.py # Vector index and invariant unit tests
│   ├── test_admin_retrain.py  # (Phase 2) Admin retrain trigger and auth tests
│   └── test_retrain_workflow.py # (Phase 2) Workflow syntax and schedule tests
│
├── data/                      # [Git-ignored] Parquet / CSV datasets
├── embeddings/                # [Git-ignored] Precomputed .npy embeddings
├── models/                    # [Git-ignored] Serialized model artifacts (.pth, .pkl, .npz)
└── mlflow/                    # [Git-ignored] MLflow experiment logs (DS11-v2)
```

---

## 4. Key Invariants & Engineering Principles

1. **`item_id == parent_asin`**: All catalog joins, model indices, feature lookups, and API endpoints strictly use `parent_asin` as the primary identifier (`item_id`). Raw `asin` is never used for indexing.
2. **Deterministic Point IDs**: Qdrant point IDs are deterministically generated via `uuid.uuid5(uuid.NAMESPACE_DNS, item_id)` ensuring idempotent upserts.
3. **DVC DAG Ownership**: Pipeline execution is owned exclusively by `dvc.yaml`. Retrain triggers run `dvc repro`.
4. **MLflow Tracking**: All model runs are logged to MLflow experiment `DS11-v2` (`mlflow.set_tracking_uri('mlflow/')`).

---

## 5. Quickstart & Reproduction Guide

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- (Optional) NVIDIA GPU for deep learning models / e5 embedding generation

### Step 1: Clone & Environment Setup
```bash
git clone https://github.com/warutkm/rs.git
cd rs

# Create virtual environment
python -m venv venv
# On Linux/macOS: source venv/bin/activate
# On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Launch Supporting Services
Start Qdrant, Redis, and PostgreSQL containers:
```bash
docker-compose up -d qdrant redis postgres
```
Verify Qdrant is running at `http://localhost:6333` (Web UI at `http://localhost:6333/dashboard`).

### Step 3: Run Data & Embedding Pipeline
Execute the pipeline stages to ingest raw data, preprocess, and generate embeddings:
```bash
# Ingest and preprocess data
python src/01_data_ingestion.py
python src/02_preprocessing.py

# Generate text embeddings & item metadata
python src/07_semantic_search.py --mode local
```
*(Or reproduce the full DAG via `dvc repro`)*

### Step 4: Sync Embeddings into Qdrant
Upsert precomputed embeddings and metadata payload into the Qdrant `products` collection:
```bash
python pipeline/sync_embeddings.py --recreate
```
This indexes all 44,301 items with Cosine distance and creates payload indexes for `category`, `price`, `item_id`, and `average_rating`.

### Step 5: Run Automated Tests
```bash
pytest -v
```

### Step 6: Start API Service
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
API Documentation: `http://localhost:8000/docs`

---

## 6. Implementation Progress & Roadmap

- [x] **Phase 0: Scaffold & Config** — Scaffolding, `docker-compose.yml`, `config.py`, and project rules.
- [x] **Phase 1: Ingestion Invariants & Qdrant HNSW Sync** — Validated `item_id = parent_asin`, implemented `pipeline/sync_embeddings.py`, verified ANN search on 44,301 items.
- [x] **Phase 2: Scheduled Retrain Automation** — GitHub Actions cron retrain DAG & `/admin/retrain` endpoint.
- [ ] **Phase 3: Model Artifact Refresh** — Regenerate trained ALS, SVD++, MF, NCF, and Apriori artifacts.
- [ ] **Phase 4: Ranker Feature Store & LightGBM LambdaMART** — Build `data/ranker_train.parquet` and train learned ranker.
- [ ] **Phase 5: Two-Tower Retrieval Model** — Unified user/item embedding tower (optional stretch).
- [ ] **Phase 6: LLM Explanation & Query Understanding** — Gemini 3.5 Flash-Lite query rewriting and cached explanations.
- [ ] **Phase 7: Async FastAPI v2 Platform** — Async endpoints, Redis cache, PostgreSQL event logging, rate limiting.
- [ ] **Phase 8: Next.js Frontend Application** — Modern interactive UI with user switching, search, and personal rails.
- [ ] **Phase 9: CI/CD Pipeline** — GitHub Actions lint, test, smoke-retrain, Docker build.
- [ ] **Phase 10: Observability** — Structured JSON logging & `/metrics` latency and hit-rate monitoring.
- [ ] **Phase 11: Production Deployment** — Tier 1 public deployment (Render / Qdrant Cloud / Neon / Upstash).
- [ ] **Phase 12: Offline Evaluation & Leaderboard** — Re-run A/B evaluation suite with Ranker and refresh MLflow report.
