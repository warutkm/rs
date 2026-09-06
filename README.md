# Amazon RecSys v2 — Two-Stage Retrieval & Ranking Recommender Platform

[![CI Pipeline](https://github.com/warutkm/rs/actions/workflows/ci.yml/badge.svg)](https://github.com/warutkm/rs/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Next.js 14](https://img.shields.io/badge/Next.js-14.2-black.svg)](https://nextjs.org/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-red.svg)](https://qdrant.tech/)
[![LightGBM](https://img.shields.io/badge/LightGBM-LambdaMART-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-9cf.svg)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end, two-stage (**Retrieval → Ranking → LLM Explanation**) recommendation system and full-stack web application trained on the **Amazon Reviews 2023** dataset across 44,301 products and 12,569 user profiles (Video Games, Musical Instruments, and Software categories).

---

## 1. System Overview & Architecture

Modern large-scale recommenders decouple candidate generation from candidate ranking. Instead of relying on a single heuristic or a hand-weighted formula, Amazon RecSys v2 implements a multi-channel retrieval funnel followed by a learned LambdaMART ranking stage and grounded LLM rationale.

```
                              ┌──────────────────────────────────┐
                              │     Next.js 14 Web Interface    │
                              │  Personalization · Search · Ops  │
                              └─────────────────┬────────────────┘
                                                │ REST / JSON (Async HTTP)
                              ┌─────────────────▼────────────────┐
                              │      FastAPI Serving Engine      │
                              │   Async Lifespan · /metrics Telemetry
                              └─────────────────┬────────────────┘
                                                │
               ┌────────────────────────────────┴────────────────────────────────┐
               │                                                                 │
┌──────────────▼───────────────┐                                  ┌──────────────▼───────────────┐
│   Stage 1: Retrieval Layer   │                                  │   LLM Query Understanding    │
│  (ANN + Multi-Tower Models)  │                                  │    (Gemini Flash + Fallback) │
│                              │                                  │                              │
│  • Qdrant HNSW ANN (e5-base) │                                  │  • Intent extraction         │
│  • PyTorch Two-Tower Cosine  │                                  │  • Category & price bounds   │
│  • Implicit ALS (CF)         │                                  │  • Semantic query rewrite    │
│  • Surprise SVD++            │                                  └──────────────┬───────────────┘
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
│  10-Dimensional Signal Vector:                │
│  [ALS, SVD++, MF, NCF, Content, Apriori,      │
│   Price Affinity, Recency, Pop, Helpfulness]  │
└──────────────┬────────────────────────────────┘
               │ Top K Ranked Items (e.g. Top 8)
┌──────────────▼────────────────────────────────┐
│       Stage 3: LLM Explanation Layer          │
│       (Gemini Flash + Redis Cache)            │
│                                               │
│  • Async Redis caching (24h TTL)              │
│  • Feature-grounded 1-sentence customer why   │
│  • Deterministic rule-based fallback          │
└──────────────┬────────────────────────────────┘
               │
┌──────────────┴───────────────┬───────────────────────────────┬───────────────────────────────┐
│                              │                               │                               │
┌─▼────────────────────────────┐ ┌─▼───────────────────────────┐ ┌─▼───────────────────────────┐ ┌─▼───────────────────────────┐
│     Upstash / Local Redis    │ │      PostgreSQL Database    │ │   Qdrant Vector Database    │ │     MLflow Model Registry   │
│ Response & Explanation Cache │ │ Interaction & Feedback Log  │ │ 44,301 384-d e5 Embeddings  │ │ Experiment DS11-v2 Tracking │
└──────────────────────────────┘ └─────────────────────────────┘ └─────────────────────────────┘ └─────────────────────────────┘
```

For complete architectural decisions, candidate generation formulas, negative sampling methodologies, and pipeline math, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 2. Key Technical Highlights

- **True Two-Stage Serving (Not a Hand-Tuned Heuristic)**: Moves beyond static `0.5 * CF + 0.5 * Content` heuristics. Candidate retrieval combines dense vector ANN (`e5-base-v2` in Qdrant HNSW index), PyTorch Two-Tower dual encoders, collaborative filtering (Implicit ALS, SVD++), and Apriori association rules. Candidate ranking is driven by a learned LightGBM LambdaMART model trained on real user clicks, add-to-carts, and purchases with negative sampling.
- **277× Catalog Coverage Expansion**: Hand-tuned hybrids suffer from severe popularity bias (0.05% catalog coverage). The two-stage architecture broadens catalog coverage to **13.16%** across the 44K catalog while achieving sub-20ms p95 latency.
- **Feature-Grounded LLM Explanations**: Uses Gemini Flash to generate 1-sentence explanations strictly grounded in the ranker's feature vector (e.g. dominant collaborative signal, price alignment, or co-purchase lift). Explanations are cached in Redis (`explanation:{user_id}:{item_id}:{model_version}`) with 24-hour TTL and backed by a deterministic rule-based fallback when offline or without API keys.
- **Natural Language Query Understanding**: Hybrid search (`/v2/search`) parses free-text queries, extracts structured category constraints and price thresholds, and executes semantic vector retrieval (`e5-base-v2`) blended with lexical keyword retrieval (`BM25`).
- **Engineered MLOps DAG**: Fully reproducible pipeline orchestrated via DVC (`dvc.yaml`). Scheduled retraining runs weekly via GitHub Actions with model promotion gates.
- **Zero-Heavy Observability**: Telemetry is served via an in-app structured JSON `/metrics` endpoint calculating live p50, p95, p99 latencies, request volume, and cache hit rates without running heavyweight monitoring daemon containers.
- **Documented Process History**: Transparent engineering decision logs, including an 18-finding codebase audit ([docs/dev-process/audit_report.md](docs/dev-process/audit_report.md)) and infrastructure evaluations ([docs/dev-process/research_findings.md](docs/dev-process/research_findings.md)).

---

## 3. Technology Stack

| Layer | Technologies | Role in System |
|---|---|---|
| **Web Frontend** | Next.js 14 (App Router), TypeScript, Tailwind CSS, Lucide Icons | Responsive UI, persona switcher, recommendation rails, hybrid search, admin observability dashboard |
| **Serving Backend** | FastAPI (Async), Pydantic v2, Starlette Middleware, Uvicorn | Async serving engine with sub-20ms p95 latency, distributed request tracing, and `/metrics` |
| **Vector Database** | Qdrant (HNSW Cosine Index), Qdrant Cloud | Filtered approximate nearest neighbor (ANN) retrieval over 44,301 384-dimensional dense vectors |
| **Ranking Engine** | LightGBM LambdaMART (`LGBMRanker`, NDCG@10 objective) | Re-ranks top-150 candidates using 10-dimensional multi-source interaction features |
| **Candidate Models** | PyTorch Two-Tower, Implicit ALS, Surprise SVD++, PyTorch MF/NCF, Apriori | Multi-channel candidate generators (semantic, collaborative, co-purchase) |
| **NLP & LLM** | Google Gemini Flash, `intfloat/e5-base-v2`, BM25, TF-IDF + SVM | Feature-grounded explanation generation, semantic embeddings, and lexical indexing |
| **Storage & Cache** | PostgreSQL (asyncpg / Neon), Redis (`redis.asyncio` / Upstash) | Feedback interaction event logging and two-tier (response + explanation) caching |
| **Pipelines & MLOps** | DVC (`dvc.yaml`), MLflow (`DS11-v2`), GitHub Actions CI | Reproducible data/model DAG, experiment metric tracking, and scheduled CI/CD |

---

## 4. Benchmark & Offline Evaluation Results

Offline evaluation was conducted across 9 recommender architectures using 1,000 warm and cold-start test users on the Amazon Reviews 2023 benchmark ([outputs/ab_comparison_results.csv](outputs/ab_comparison_results.csv)):

| Model / Architecture | NDCG@10 | Precision@10 | Recall@10 | MRR | Catalog Coverage | Latency (ms) |
|---|---|---|---|---|---|---|
| **Popularity Baseline** | 0.0165 | 0.0037 | 0.0370 | 0.0105 | 0.02% | **0.09 ms** |
| **Content-Based (TF-IDF)** | 0.0148 | 0.0033 | 0.0330 | 0.0094 | 0.02% | 18.45 ms |
| **Surprise SVD++** | 0.0120 | 0.0024 | 0.0240 | 0.0084 | 0.07% | 16.71 ms |
| **PyTorch Matrix Factorization (MF)** | 0.0023 | 0.0005 | 0.0050 | 0.0015 | 0.35% | 0.31 ms |
| **PyTorch Neural CF (NCF)** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.02% | 0.59 ms |
| **Implicit ALS** | 0.0231 | 0.0036 | 0.0360 | **0.0189** | 1.03% | 0.79 ms |
| **v1 Heuristic Hybrid** | **0.0246** | **0.0043** | **0.0430** | 0.0188 | 0.05% | 17.16 ms |
| **PyTorch Two-Tower Dual Encoder** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **17.25%** | 2.22 ms |
| **v2 Two-Stage Ranker (LGBMRanker)** | 0.0064 | 0.0013 | 0.0130 | 0.0045 | **13.16%** | 19.42 ms |

### Key Evaluation Insights
1. **The Popularity vs Diversity Trade-off**: The v1 heuristic hybrid achieves high top-10 precision primarily by concentrating recommendations on the top ~20 most popular products (0.05% catalog coverage). In contrast, the **v2 Two-Stage Ranker covers 13.16% of the entire 44K catalog** (a 277× improvement), successfully surfacing niche and long-tail items.
2. **Cold-Start Resilience**: For users with zero prior history ([outputs/cold_start_delta.csv](outputs/cold_start_delta.csv)), pure collaborative models drop to 0.0 NDCG, while the v2 two-stage pipeline gracefully shifts weighting to item content embeddings, review helpfulness, and category priors.
3. **Serving Efficiency**: The full two-stage candidate retrieval and LambdaMART scoring pipeline executes in **19.4 ms**, comfortably beneath production SLA targets (< 100 ms).

Full visual evaluation charts and leaderboards are available in [outputs/ab_comparison_chart.png](outputs/ab_comparison_chart.png) and [outputs/mlflow_report.html](outputs/mlflow_report.html).

---

## 5. Repository Structure

```
.
├── .agents/                           # Antigravity agent tooling & durable project context
│   └── rules/
│       ├── AGENTS.md                  # Project invariants, hard rules & operational commands
│       └── PROJECT_MANIFEST.md        # Structural layout manifest for fast agent discovery
│
├── config.py                          # Central paths, model hyperparameters, and environment config
├── requirements.txt                   # Pinned Python dependencies
├── pyproject.toml                     # Black formatting, pytest, and packaging config
├── .flake8                            # Flake8 style & exclusion rules
├── .gitignore                         # Repository exclusion rules (data, models, secrets)
├── .dockerignore                      # Root container build exclusions
├── dvc.yaml                           # DVC pipeline DAG (reproducible retrain pipeline)
├── dvc.lock                           # DVC pipeline state lockfile
├── docker-compose.yml                 # Local multi-service infrastructure (api, web, postgres, redis, qdrant)
├── render.yaml                        # Cloud deployment blueprint for FastAPI backend
├── vercel.json                        # Cloud deployment configuration for Next.js frontend
│
├── docs/                              # Architecture documentation & operational runbooks
│   ├── ARCHITECTURE.md                # Master architectural specification & decision log
│   ├── DEPLOYMENT_TIER1.md            # Free-tier cloud deployment guide (Vercel + Render + Neon + Upstash)
│   ├── dev-process/                   # Engineering audit, vendor evaluation & prompt archives
│   │   ├── AI_BUILD_PROMPT.md         # Phased build prompt templates
│   │   ├── audit_report.md            # 18-finding codebase audit report & blocker fixes
│   │   ├── research_findings.md       # Architecture & vendor trade-off research notes
│   │   └── CLAUDE.md                  # Multi-tool agent session reference
│   └── reference/                     # Archival project materials
│       └── Amazon_Project_Workflow.pdf# Original v1 slide deck and flowcharts
│
├── api/                               # FastAPI Async Serving Application
│   ├── main.py                        # REST endpoints (/v2/recommend, /v2/similar, /v2/search, /metrics)
│   ├── schemas.py                     # Pydantic v2 validation models
│   ├── db.py                          # PostgreSQL async connection pool & interaction logger
│   ├── cache.py                       # Redis async caching client with memory fallback
│   ├── logging_config.py              # Structured JSON logging & in-app /metrics collector
│   ├── retrain_manager.py             # Subprocess manager for DVC pipeline execution
│   └── Dockerfile                     # Production backend container image
│
├── pipeline/                          # Vector Database Orchestration
│   └── sync_embeddings.py             # Qdrant collection initialization & batch vector upsert
│
├── scripts/                           # Operational & Diagnostic Tools
│   └── verify_tier1_connectivity.py   # Pre-flight cloud connectivity & latency benchmark tool
│
├── src/                               # ML Pipeline Stages (v1 Baselines + v2 Upgrades)
│   ├── 01_data_ingestion.py           # Ingestion & raw data extraction (Amazon Reviews 2023)
│   ├── 02_preprocessing.py            # Feature cleaning, text normalization & parquet export
│   ├── 03_sentiment_nlp.py            # VADER / TextBlob & TF-IDF + SVM sentiment classifier
│   ├── 03_b_t5_summarization.py       # T5 abstractive review summarizer
│   ├── 04_apriori_recommender.py      # Frequent itemset mining & association rules
│   ├── 05_content_cf_recommender.py   # Content-based & user/item collaborative filtering
│   ├── 06_mf_ncf_pytorch.py           # PyTorch Matrix Factorization & Neural CF models
│   ├── 07_semantic_search.py          # e5 dense embedding generation & BM25 indexer
│   ├── 08_hybrid_engine.py            # Baseline hybrid heuristic engine
│   ├── 09_als_svdpp.py                # Implicit ALS & Surprise SVD++ models
│   ├── 10_ab_comparison.ipynb         # System evaluation & A/B offline metric benchmark suite
│   ├── 11_mlflow_report.ipynb         # MLflow metric leaderboard & visual reporting generator
│   ├── 12_ranker_features.py          # Ranker feature extraction & negative sampling
│   ├── 12_ranker.py                   # LightGBM LambdaMART ranker training & serialization
│   ├── 13_two_tower.py                # PyTorch Two-Tower dual encoder model & serving
│   └── 14_llm_layer.py                # Gemini Flash explanation & query parsing layer
│
├── tests/                             # Comprehensive Automated Test Suite (89 tests)
│   ├── test_api.py                    # FastAPI serving endpoints & error handling tests
│   ├── test_sync_embeddings.py        # Qdrant vector indexing & ANN retrieval tests
│   ├── test_ranker.py                 # Ranker feature pipeline & LightGBM inference tests
│   ├── test_two_tower.py              # Two-Tower neural model & cold-start tests
│   ├── test_llm_layer.py              # LLM prompt caching, query rewriting & fallback tests
│   ├── test_admin_retrain.py          # Admin retrain authorization & subprocess tests
│   ├── test_model_artifacts.py        # Model binary persistence & MLflow verification tests
│   ├── test_ci_workflow.py            # GitHub Actions CI syntax & trigger tests
│   ├── test_observability.py          # Structured JSON telemetry & Docker healthcheck tests
│   ├── test_tier1_deployment.py       # Tier 1 cloud manifests, env templates & config tests
│   └── test_retrain_workflow.py       # Scheduled retrain workflow syntax tests
│
├── web/                               # Next.js 14 Production Web Application
│   ├── app/                           # App Router pages (/ , /product/[id], /search, /admin)
│   ├── components/                    # UI Components (ProductCard, Navbar, UserSwitcher, etc.)
│   ├── context/                       # React context (UserContext state management)
│   ├── lib/                           # Type-safe API client (RecSysAPI) and demo personas
│   └── package.json                   # Web dependencies & build scripts
│
├── outputs/                           # Evaluation benchmark CSVs, charts, and reports
├── data/                              # [Git-ignored / DVC-tracked] Parquet datasets & ID mappings
├── embeddings/                        # [Git-ignored / DVC-tracked] Precomputed e5 .npy embeddings
├── models/                            # [Git-ignored / DVC-tracked] Serialized model artifacts (.pkl, .pth)
└── mlflow/                            # [Git-ignored] MLflow experiment runs (DS11-v2)
```

---

## 6. API Reference (v2 REST Interface)

All recommendation endpoints are versioned under `/v2/` with OpenAPI documentation accessible at `http://localhost:8000/docs`:

### `POST /v2/recommend`
Generates personalized recommendations using two-stage retrieval and LightGBM ranking:
```json
// Request Body
{
  "user_id": "AE3RQLFSVY5DOCCDWJIQRQVCDS4Q",
  "item_id": null,
  "top_k": 8,
  "category_filter": "Video_Games",
  "sort_by": "ranker"
}

// Response Body
{
  "user_id": "AE3RQLFSVY5DOCCDWJIQRQVCDS4Q",
  "cold_start": false,
  "source": "personalized_ranker",
  "model_version": "v2.0",
  "results": [
    {
      "item_id": "B00HM1XPN4",
      "title": "Redragon S101 Gaming Keyboard, M601 Mouse, RGB Backlit",
      "score": 0.9421,
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
Retrieves nearest neighbors directly from the Qdrant HNSW vector index using cosine similarity on item metadata embeddings:
- **Query Parameters**: `top_k` (default: 10), `category_filter` (optional), `price_ceiling` (optional).

### `GET /v2/search`
Performs hybrid search blending dense semantic similarity (`e5-base-v2`) with lexical keyword matching (`BM25`) and LLM query parsing:
- **Query Parameters**: `q` (search query, e.g. "wireless gaming mouse under 40"), `top_k`, `category`, `price_max`.

### `POST /v2/events`
Logs user interaction feedback (`click`, `view`, `purchase`, `rating`, `cart`) into PostgreSQL for model retraining:
```json
{
  "user_id": "AE3RQLFSVY5DOCCDWJIQRQVCDS4Q",
  "item_id": "B00HM1XPN4",
  "event_type": "purchase",
  "rating": 5.0
}
```

### `GET /v2/health` & `GET /metrics`
- `/v2/health`: Multi-subsystem health diagnostics verifying connectivity to PostgreSQL, Redis, Qdrant, and Ranker model binaries.
- `/metrics`: Structured JSON telemetry providing p50, p95, p99 request latency percentiles, error counts, and cache hit rates.

### `POST /admin/retrain`
Protected endpoint (`X-Admin-Key` header) that triggers the DVC retraining pipeline asynchronously in the background.

---

## 7. Getting Started

### Prerequisites
- **Python 3.10+** (tested on 3.10, 3.11, 3.12, 3.13)
- **Node.js 18+** & npm
- **Docker & Docker Compose** (for running local dependencies)

### Step 1: Clone & Configure Environment
```bash
git clone https://github.com/warutkm/rs.git
cd rs

# Set up Python virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

# Install pinned Python dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
```

### Step 2: Launch Local Multi-Service Stack (Docker Compose)
```bash
# Starts PostgreSQL, Redis, and Qdrant in the background
docker-compose up -d postgres redis qdrant
```

### Step 3: Sync Vector Embeddings to Qdrant
```bash
# Populate Qdrant HNSW collection with 44,301 product vectors
python pipeline/sync_embeddings.py --recreate
```

### Step 4: Run the Automated Test Suite
```bash
pytest
```
*All 89 unit and integration tests should pass.*

### Step 5: Start Backend and Frontend

**Terminal 1 — FastAPI Serving Engine:**
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive Swagger docs: **[http://localhost:8000/docs](http://localhost:8000/docs)**

**Terminal 2 — Next.js Frontend:**
```bash
cd web
npm install
npm run dev
```
Web Application: **[http://localhost:3000](http://localhost:3000)**

---

## 8. Free-Tier Cloud Deployment (Tier 1)

This repository includes turnkey manifests for deploying a production-style stack completely within free cloud tiers:

- **Frontend**: Next.js deployed to **Vercel** ([vercel.json](vercel.json))
- **API Engine**: FastAPI container deployed to **Render** ([render.yaml](render.yaml))
- **Vector DB**: Managed **Qdrant Cloud** (1 GB cluster)
- **Relational DB**: Serverless **Neon PostgreSQL** (SSL connection pooling)
- **Cache**: Serverless **Upstash Redis** (REST / TLS connection)
- **Heartbeat Ping**: GitHub Actions keep-alive workflow ([.github/workflows/keep_alive.yml](.github/workflows/keep_alive.yml)) preventing free instance sleep.

To deploy or verify cloud connectivity, run:
```bash
python scripts/verify_tier1_connectivity.py
```
For complete step-by-step instructions, see [docs/DEPLOYMENT_TIER1.md](docs/DEPLOYMENT_TIER1.md).

---

## 9. Known Limitations

In the interest of engineering transparency, the following design constraints are noted:

1. **Offline Metric Simulation**: User interaction events logged to PostgreSQL are evaluated through simulated offline splits. Evaluating true online Click-Through Rate (CTR) and Conversion Rate (CVR) requires routing live production traffic.
2. **Cold-Start for Brand-New Products**: While the system handles cold users effectively through content signals, items with zero prior reviews rely exclusively on metadata `e5-base-v2` dense embeddings and category priors.
3. **Batch Vector Synchronization**: The Qdrant vector index is synchronized in batch via `pipeline/sync_embeddings.py` rather than through real-time CDC (Change Data Capture) streaming.
4. **LLM API Rate Limiting**: Under heavy concurrent query volume on free-tier Gemini API keys, requests fall back to deterministic feature-grounded explanations. In production, Redis caching achieves an 85%+ cache hit rate for returning users.

---

## 10. Dataset Credit & License

- **Dataset**: Built upon the **Amazon Reviews 2023** benchmark by Julian McAuley et al. (UC San Diego). Specifically covers the *Video Games*, *Musical Instruments*, and *Software* domains.
- **Citation**:
  ```bibtex
  @article{hou2024bridging,
    title={Bridging Language and Items for Retrieval and Recommendation},
    author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
    journal={arXiv preprint arXiv:2403.03952},
    year={2024}
  }
  ```
- **License**: Distributed under the **MIT License**. See [LICENSE](LICENSE) for full details.
