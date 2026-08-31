# Project Manifest — Amazon RecSys v2

## Structural Overview
```
amazon_project/
├── GEMINI.md                          # Durable project rules & conventions
├── PROJECT_MANIFEST.md                # Structural directory tree & manifest (this file)
├── RECSYS_V2_WORKFLOW_AND_DESIGN.md   # v2 Master design & workflow document
├── AI_BUILD_PROMPT.md                 # Per-phase build prompt templates
├── config.py                          # Global paths, hyperparameters, constants
├── requirements.txt                   # Pinned project dependencies
├── docker-compose.yml                 # Local multi-service stack (api, postgres, redis, qdrant)
├── dvc.yaml                           # DVC pipeline DAG definition
├── dvc.lock                           # DVC lockfile
│
├── src/                               # Model & data pipeline stages (v1 baseline + v2 extensions)
│   ├── 01_data_ingestion.py           # Ingestion & raw data extraction (Amazon Reviews 2023)
│   ├── 02_preprocessing.py            # Filtering, merging, cleaning, metadata alignment
│   ├── 03_sentiment_nlp.py            # VADER / TextBlob sentiment & TF-IDF + SVM classifier
│   ├── 03_b_t5_summarization.py       # T5-based review text summarization
│   ├── 04_apriori_recommender.py      # Association rule mining (frequent itemsets & lift)
│   ├── 05_content_cf_recommender.py   # Content-based & baseline Collaborative Filtering
│   ├── 06_mf_ncf_pytorch.py           # PyTorch Matrix Factorization & Neural CF models
│   ├── 07_semantic_search.py          # e5 text embeddings & hybrid BM25 search
│   ├── 08_hybrid_engine.py            # v1 HybridRecommender (content + CF + Apriori union)
│   ├── 09_als_svdpp.py                # Implicit ALS & Surprise SVD++ models
│   ├── 10_ab_comparison.ipynb         # System evaluation & A/B offline metric comparisons
│   ├── 11_mlflow_report.ipynb         # MLflow metric leaderboard & visual reporting
│   ├── 12_ranker_features.py          # (Phase 4) Ranker feature extraction & negative sampling
│   ├── 12_ranker.py                   # (Phase 4) LGBMRanker LambdaMART training & serving
│   ├── 13_two_tower.py                # (Phase 5) PyTorch Two-Tower retrieval model & serving
│   └── 14_llm_layer.py                # (Phase 6) Gemini 3.5 Flash-Lite LLM explanations & query rewriting
│
├── api/                               # FastAPI serving application
│   ├── Dockerfile                     # Container definition for API
│   ├── main.py                        # (Phase 7) Async FastAPI endpoints (/v2/recommend, /v2/similar, /v2/search, /v2/events, /v2/health, /metrics)
│   ├── db.py                          # (Phase 7) Async PostgreSQL client & event logging
│   ├── cache.py                       # (Phase 7) Async Redis explanation & response cache
│   ├── logging_config.py              # (Phase 7) Structured JSON logging & /metrics counters
│   ├── retrain_manager.py             # (Phase 2) Subprocess manager for DVC pipeline execution
│   └── schemas.py                     # (Phase 7) Pydantic v2 request/response schemas
│
├── pipeline/                          # (v2 target) Orchestration & background sync tasks
│   └── sync_embeddings.py             # (Phase 1) Qdrant vector index synchronization
│
├── tests/                             # Unit and integration test suite
│   ├── test_api.py                    # (Phase 7) FastAPI v2 integration and unit tests
│   ├── test_sync_embeddings.py        # Qdrant sync and ANN retrieval tests
│   ├── test_admin_retrain.py          # (Phase 2) Admin retrain trigger and auth tests
│   ├── test_retrain_workflow.py       # (Phase 2) GitHub Actions retrain workflow syntax tests
│   ├── test_model_artifacts.py        # (Phase 3) Model binary persistence and MLflow tracking tests
│   ├── test_ranker.py                 # (Phase 4) Ranker feature pipeline and LGBMRanker tests
│   ├── test_two_tower.py              # (Phase 5) Two-Tower contrastive model & retrieval tests
│   └── test_llm_layer.py              # (Phase 6) LLM explanation generation & query rewriting tests
│
├── .github/workflows/                 # CI/CD & automation workflows
│   └── retrain.yml                    # (Phase 2) Cron scheduled dvc repro pipeline
│
├── web/                               # (Phase 8) Next.js 14 frontend application
│   ├── app/                           # App Router pages & routes
│   │   ├── page.tsx                   # Home: Personalized & trending recommendation rails
│   │   ├── layout.tsx                 # Root layout & navigation shell
│   │   ├── globals.css                # Tailwind base styles & glassmorphism theme
│   │   ├── product/[id]/page.tsx      # Product details, Qdrant ANN neighbors, interaction logging
│   │   ├── search/page.tsx            # Semantic search, LLM query rewrite, score breakdown
│   │   └── admin/page.tsx             # Observability dashboard, real-time /metrics, DVC retrain trigger
│   ├── components/                    # UI Components
│   │   ├── Navbar.tsx                 # Search bar, user switcher, health telemetry indicator
│   │   ├── UserSwitcher.tsx           # Seeded demo personas dropdown & custom user_id input
│   │   ├── ProductCard.tsx            # Card with rank, score, category, LLM explanation, cart/like
│   │   ├── ExplanationBadge.tsx       # Cached LLM 'Why This' explanation badge & flyout
│   │   └── ScoreBreakdown.tsx         # Hybrid e5 semantic + BM25 lexical score bar visualization
│   ├── context/                       # State management
│   │   └── UserContext.tsx            # Active demo user, cart state, like interactions
│   ├── lib/                           # Utilities & Client
│   │   ├── api.ts                     # Type-safe RecSys API client with fallback resilience
│   │   └── demoUsers.ts               # Seeded demo personas (Alex, Elena, Marcus, Sarah, Devon, Aisha, Guest)
│   ├── package.json                   # Web dependencies & build scripts
│   ├── tailwind.config.js             # Theme tokens, dark palette, animations
│   └── tsconfig.json                  # TypeScript configuration
│
├── data/                              # [Git-ignored] Parquet / CSV datasets & ID mappings
├── embeddings/                        # [Git-ignored] Precomputed .npy embeddings & BM25 indices
├── models/                            # [Git-ignored] Serialized model artifacts (.pkl, .pth, .npz)
├── mlflow/                            # [Git-ignored] MLflow experiment tracking & artifact store
├── outputs/                           # Generated evaluation charts, CSVs, and reports
└── logs/                              # Execution logs (DVC, training, profiling)
```

## Key Invariants
- `item_id` = `parent_asin` across all tables, models, indices, and APIs.
- Pipeline DAG is orchestrated by `dvc.yaml` (`dvc repro`).
- Local multi-service infrastructure (API, PostgreSQL, Redis, Qdrant) managed via root `docker-compose.yml`.
- Observability via structured JSON logging and API-level `/metrics` endpoint.
