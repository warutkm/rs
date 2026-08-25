# DS11 Recommender → Production Platform
### v2 Rework: Workflow, Architecture & Decision Log
Owner: Utkrishta · Base: DS11 hybrid recommender (Amazon Reviews 2023) · Status: planning → build

---

## 0. What this document is

Your v1 project (content-based + CF hybrid, ALS/SVD++/MF/NCF, MLflow, DVC, FastAPI) is complete and
correct. This doc is the single source of truth for **v2**: turning it into something that looks and behaves
like a real production recommender — multi-stage serving, a real backend + frontend, deployed, observable,
and built using a token-efficient AI-assisted workflow instead of one long chat thread.

Everything here is additive to your v1 code, not a rewrite from zero. The phase plan in §11 tells you exactly
which v1 files survive untouched, which get upgraded, and which are new.

## 1. Gap analysis — why rework, file by file

| v1 component | v1 limitation | v2 upgrade | Why it matters at "industry" scale |
|---|---|---|---|
| `product_vecs` dict + brute-force cosine (07) | O(n) scan over all items per query | Vector DB (Qdrant) with HNSW ANN index | Sub-100ms retrieval as catalog grows past what fits in RAM |
| `HybridRecommender` hand-weighted union (08) | Fixed weights (`cf_weight`, `content_weight`) tuned by hand | Learned-to-rank model (LightGBM LambdaMART) over CF score, content score, Apriori signal, price, recency, popularity | Weights that adapt to data instead of a guess; standard "ranking stage" pattern |
| ALS / SVD++ / MF / NCF (05, 06, 09) | Each is a standalone retrieval+score model | Keep all four as **candidate generators**; feed their embeddings into a two-tower retrieval model | This is exactly the two-stage retrieval→ranking pattern used by every large-scale recommender (candidate generation narrows millions→hundreds, ranking narrows hundreds→dozens) |
| Feature computation baked into pickles at train time | Features (recency, popularity) go stale until next full retrain | Features load from parquet into memory at API startup (same pattern as v1's models); Redis is response/explanation cache only | Static features don't need a feature store; a "real-time signals" store is a future addition once there's actual traffic to justify it |
| `dvc repro` run manually | No trigger, no schedule | GitHub Actions cron triggers `dvc repro` directly — DVC stays the one DAG owner | Reproducible *and* schedulable, with no second scheduler process or duplicate DAG definition |
| FastAPI, 4 endpoints, no auth/cache/logging (12) | Fine for a grading demo, not for a live service | Async FastAPI, Redis response cache, request logging to Postgres, rate limiting | This is the difference between "runs on my laptop" and "a service" |
| No frontend | Recommendations only visible via curl/notebook | Next.js app: browsing, search, "why this" explanations | Lets anyone (a recruiter, a professor) actually use it |
| MLflow local file store | Not queryable outside your machine | Same MLflow, plus a model registry stage (staging→production) and a CI job that gates promotion on metrics | Turns "7 runs logged" into an actual release process |
| Static, pre-computed explanations (none) | System has no "why" | LLM-generated explanations from the same feature vector the ranker used, cached | The one genuinely new 2026-era ingredient — see §6 |

Nothing in v1 gets thrown away. ALS, SVD++, MF, NCF, TF-IDF+SVM, Apriori, e5 embeddings, T5 summaries —
all of it becomes an input signal to the v2 ranking stage instead of a final answer on its own.

---

## 2. Target architecture

```
                         ┌─────────────────────────┐
                         │        Frontend         │
                         │   Next.js (Vercel)      │
                         └────────────┬─────────────┘
                                      │ REST/JSON
                         ┌────────────▼─────────────┐
                         │       FastAPI (async)     │
                         │  auth · rate limit · log  │
                         └────────────┬──────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │  Query rewrite — Gemini 3.5 Flash-Lite │
                    │  free-text search only, BEFORE         │
                    │  retrieval: structured filter +        │
                    │  rewritten semantic query (§6)          │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │            Retrieval layer            │
                    │   Qdrant (ANN) · ALS/SVD++/MF/NCF ·   │
                    │   e5 semantic · Apriori                │
                    └─────────────────┬───────────────────┘
                                      │ top-200 candidates
                    ┌─────────────────▼───────────────────┐
                    │      Ranking stage — LightGBM          │
                    │             LambdaMART                 │
                    │  features: CF scores, content score,   │
                    │  Apriori lift, price, recency,         │
                    │  popularity, helpful_votes (loaded      │
                    │  from parquet at startup, §4)           │
                    └─────────────────┬───────────────────┘
                                      │ top-10-20
                    ┌─────────────────▼───────────────────┐
                    │  Explanation gen — Gemini 3.5 Flash-Lite│
                    │  AFTER ranking, async, cached (§6)      │
                    └─────────────────┬───────────────────┘
                                      │
   ┌──────────────────┐      ┌────────▼────────┐      ┌───────────────────┐
   │  Redis cache        │      │  Postgres        │      │  MLflow registry   │
   │  response + explan-  │      │  users, events, │      │  staging→prod       │
   │  ation cache only     │      │  request logs   │      │  gates on metrics    │
   └──────────────────────┘      └────────────────┘      └───────────────────┘

   Offline: DVC (ingest→preprocess→embed→train) — DVC owns the DAG. GitHub Actions cron triggers
   `dvc repro` on a schedule or via /admin/retrain; it's a scheduler only, not a second DAG (§4).
```

---

## 3. Tech stack decisions

| Layer | v1 | v2 choice | Free-tier alternative | Why |
|---|---|---|---|---|
| Retrieval index | brute-force dict scan | **Qdrant** | Qdrant Cloud free 1GB cluster | Purpose-built ANN, easy Python client, self-hostable in the same docker-compose |
| Ranking | hand-weighted union | **LightGBM** (LambdaMART, `lightgbm.LGBMRanker`) | — (runs on CPU, no infra cost) | Standard learn-to-rank library, small enough to train on your existing 6GB-VRAM box (it barely uses the GPU) |
| Response cache | none | **Redis** | Upstash Redis free tier | Sub-ms reads for response cache and LLM explanation cache; item features load from parquet at startup (they're static between retrains — a Redis "feature store" adds sync complexity without real-time traffic to justify it) |
| Relational store | none | **PostgreSQL** | Neon or Supabase free tier | Users, interaction logs, feedback — the data your *next* retrain will use |
| Orchestration | `dvc repro` manual | **DVC** (DAG owner) + **GitHub Actions cron** (scheduler) | — | DVC already owns the DAG and caching; GH Actions cron triggers `dvc repro` on a schedule (weekly or on-demand via `/admin/retrain`). Prefect was adding a scheduler process, a database, and an extra container for a pipeline that runs at most weekly — not justified for a solo project |
| Backend | FastAPI (sync-ish) | **FastAPI** (async endpoints, `httpx`/`asyncpg`/`redis.asyncio`) | — | Same framework, properly async this time |
| Frontend | none | **Next.js + Tailwind** | Vercel free tier | Fastest path to a live, shareable demo |
| LLM (explanations, query understanding) | none | **Gemini 3.5 Flash-Lite** (`gemini-3.5-flash-lite`) | Google AI API pay-as-you-go (~$0.075/$0.30 per 1M in/out tokens); 2x rate limit headroom & sub-150ms latency; Claude Haiku 4.5 is an alternative | Both tasks (one-sentence explanations, query rewriting) are simple structured-output generation — 3.5 Flash-Lite gives ultra-low latency, maximum RPM headroom, and lowest cost. See §6 |
| Model tracking | MLflow (local) | MLflow (same) + registry stage transitions | — | You already have this working; just formalize promotion |
| Data versioning | DVC (3 stages) | DVC (same stages, triggered by GH Actions cron or manual `dvc repro`) | — | No reason to replace something that already works |
| CI/CD | none | **GitHub Actions**: lint → test → smoke-retrain → build image → (on tag) deploy | free for public/student repos | Turns "it works on my machine" into a repeatable release |
| Containers | Docker + docker-compose | Docker + docker-compose (bigger service list) | — | Same tool, more services |
| Deployment | none | See §9 — three tiers, default to Tier 1 | Render + Vercel + Neon + Upstash + Qdrant Cloud, all free tiers | Railway and Fly.io have deprecated permanent free tiers (trial credits only). Render is the only remaining zero-cost backend host. See §9 for cold-start and Qdrant suspension notes |
| Observability | print statements | **Structured JSON logging** + **`/metrics` endpoint** (JSON counters: latency p50/p95/p99, cache hit rate, requests per endpoint) | Grafana Cloud free tier if you later want dashboards | Self-hosting Prometheus + Grafana adds two containers and dashboard maintenance for a single-service demo — not worth it. A `/metrics` JSON endpoint gives the same interview-demo answer ("here's my system's health") with zero infra. Upgrade path: add OpenTelemetry SDK → Grafana Cloud when needed |

---

## 4. Data & feature pipeline v2

Files 01–02 (ingestion, preprocessing) stay almost untouched — the `parent_asin` fix and user-activity filter
were already correct. What's new:

- **Scheduled retrains**: GitHub Actions cron (`.github/workflows/retrain.yml`) runs `dvc repro` on a
  weekly schedule. For on-demand retrains, the `/admin/retrain` endpoint triggers `dvc repro` via subprocess,
  gated behind a simple API key. No separate scheduler process needed — DVC owns the DAG and caching,
  GH Actions owns the trigger.
- **Feature loading at startup**: per-item features (price, rating_mean, helpful_votes, recency, popularity)
  load from `data/clean_merge_df.parquet` into a dict at API startup. These features are static between
  retrains — Redis adds sync complexity without real-time traffic to justify a "feature store." Redis is
  used for the response cache and LLM explanation cache only.
- **Embedding sync task**: after file 07 produces `meta_embeds.npy`, upsert vectors into Qdrant with
  `item_id` as the point ID and metadata payload (category, price, title) for filtered search.

---

## 5. Model layer v2

- **Candidate generation (unchanged math, new role):** ALS, SVD++, MF, NCF, e5-semantic-search, and Apriori
  each produce a candidate list (~50-100 items) for a given user/item. Union + dedupe → ~150-250 candidates.
- **Two-tower retrieval model (new, optional stretch):** a small PyTorch model that learns a unified
  user-embedding and item-embedding from the *existing* MF/NCF embeddings concatenated with e5 vectors,
  trained with a contrastive/in-batch-negative loss. Its output embeddings are what actually get indexed in
  Qdrant, so retrieval improves without discarding any v1 model.
- **Ranking stage (new, this is the important one):** `LGBMRanker` trained on the same train/test split as
  file 06/09/10, with features = [als_score, svdpp_score, mf_score, ncf_score, content_score, apriori_lift,
  price_score, recency, popularity, helpful_votes]. Label = graded relevance from the rating. This directly
  replaces the manual `cf_weight`/`content_weight` dict in `HybridRecommender` with something learned —
  log it to MLflow as an 8th run (`run_name='Ranker'`).
- **Cold-start:** v1's cold-start paths (new user → content-only, new item → embedding-only) carry forward
  conceptually, but they need explicit handling in the *ranker's* serving path: zero-fill CF score features
  for unknown users, fall back to content/popularity features only. This is a new code path in the ranker
  serving logic, not a copy-paste from v1's `HybridRecommender`.

### 5a. Ranker training data construction (the hard part)

The ranking model is easy to train once you have the data. Building that data is the actual work:

1. **Score generation**: load each trained model (ALS, SVD++, MF, NCF, content-based, CF) and run
   `.predict(user, item)` for every (user, item) pair in the training set. This produces a per-interaction
   feature row. The existing training scripts (05, 06, 09) train and save models — they don't output
   per-interaction score columns. A new script (`src/12_ranker_features.py`) is needed for this step.
2. **Negative sampling**: the dataset only contains items users *did* interact with. The ranker needs
   negative examples (items that were candidates but the user didn't engage with). Strategy: for each user
   in training, sample 5–10 random items they didn't interact with, weighted by item popularity (so the
   ranker learns to distinguish "plausible but wrong" from "right," not just "random garbage" from "right").
   Assign these relevance label 0.
3. **Leakage prevention**: the CF models were trained on the full training set. Using their predictions on
   that same set as ranker features inflates the CF score features. Mitigate by: (a) training each CF model
   on fold-out subsets and using out-of-fold predictions as features, or (b) using a temporal holdout where
   the CF models are trained on data before time T and the ranker features are generated on data after T.
   Option (b) is simpler for this project given the existing leave-one-out temporal split in file 06.
4. **Group structure**: `LGBMRanker` requires a `group` parameter — an array of query-group sizes. Each
   user is a "query," and the group size is the number of candidates (positives + negatives) for that user.
   The training DataFrame must be sorted by user_id before training.
5. **Output**: a single parquet file `data/ranker_train.parquet` with columns:
   `[user_id, item_id, als_score, svdpp_score, mf_score, ncf_score, content_score, apriori_lift,
   price_score, recency, popularity, helpful_votes, relevance_label]`.

---

## 6. The LLM layer — what's genuinely new

Two additions, both designed to stay cheap:

1. **"Why this" explanations.** For each top-N ranked item, generate a one-sentence explanation from the
   *same feature vector* the ranker used (e.g. "frequently bought with items you rated highly, and priced
   near your usual range"). **Latency model**: the `/v2/recommend` endpoint returns the ranked list
   immediately. Explanation generation fires asynchronously (background task) and writes results to Redis
   keyed by `(user_id, item_id, model_version)`. On subsequent requests for the same user/top-items, the
   cached explanation is returned inline. First-time requests show recommendations without explanations —
   this is the right tradeoff for a demo (explanations appear on second load, not never).
2. **Query understanding for search.** When a user types a free-text query ("cheap wireless headphones for
   gym"), use a fast model to turn it into a structured filter (price ceiling, category) + a rewritten
   semantic query, then hand that to the existing e5 + Qdrant retrieval. This is query rewriting, not
   generation — small, fast, cacheable by exact query string.

For both, use **prompt caching**: a static system prompt (task instructions + the fixed schema you want
back) as the cached prefix, with only the per-request product/user data as the volatile suffix. Both the
Google AI API and Anthropic API support this pattern.

**Model choice**: **Gemini 3.5 Flash-Lite** (`gemini-3.5-flash-lite`) is the default for both tasks — it is the most cost-effective and latency-optimized option (~$0.075/$0.30 per 1M input/output tokens) with 2x the rate-limit headroom of standard Flash models. Both tasks (query rewriting and 1-sentence explanations) are strictly constrained, structured extraction and generation where Flash-Lite achieves near-instant (~100ms) execution with zero quality loss. Configured via `LLM_MODEL = "gemini-3.5-flash-lite"` in `config.py`.

---

## 7. Backend design (FastAPI v2)

Endpoints (superset of v1's four):

- `POST /v2/recommend` — `{user_id, item_id?, top_k}` → ranked list with `source` tag (personalized / cold-start / trending) and a cached explanation string
- `GET /v2/similar/{item_id}` — Qdrant ANN lookup
- `GET /v2/search?q=...` — LLM query rewrite → hybrid e5+BM25 retrieval (your v1 file 07 logic, unchanged)
- `POST /v2/events` — log a click/purchase/rating to Postgres (this is what makes future retrains better than a one-shot snapshot)
- `GET /v2/health`, `GET /metrics` — health + JSON metric counters (latency p50/p95/p99, cache hit rate, requests per endpoint — no Prometheus, see §3/§9)
- `POST /admin/retrain` — triggers `dvc repro` via subprocess, gated behind a simple API key

Cross-cutting: async DB/cache clients, Redis response cache (short TTL on `/v2/recommend`), request logging
middleware writing to Postgres, `slowapi` for rate limiting, `X-Request-ID` for tracing.

---

## 8. Frontend (Next.js)

Minimum pages to make this feel like a product rather than an API:

- **Home** — trending rail (no login needed) + "for you" rail (once a demo user is picked)
- **Product page** — item detail, "similar items" (Qdrant), and the cached explanation text
- **Search** — free-text box hitting `/v2/search`
- **Simple demo-user switcher** — no real auth needed; a dropdown of a handful of seeded `user_id`s is enough to *show* personalization changing
- **/admin** (optional) — pulls MLflow leaderboard + `/metrics` into a couple of charts; this is your v2 replacement for the "MLflow report" deliverable, live instead of a static HTML export

---

## 9. Deployment — three tiers, pick based on time/budget

| Tier | What | Where | Cost |
|---|---|---|---|
| **Tier 0 — local** | Everything in one `docker-compose.yml`: api, frontend, postgres, redis, qdrant | Your machine | Free |
| **Tier 1 — recommended default** | Backend on Render (Docker deploy), frontend on Vercel, Postgres on Neon, Redis on Upstash, Qdrant Cloud free cluster | Public URL | Free tiers, $0 |
| **Tier 2 — stretch/portfolio-talking-point** | AWS ECS Fargate or GCP Cloud Run for the API, managed Postgres (RDS/Cloud SQL), a real K8s manifest set (even if you only ever run it once), Terraform for the infra | AWS/GCP | Real cost — do this only if you want the "I've deployed on AWS with IaC" line, and stop it right after demoing |

Do Tier 0 first (it's most of the engineering work anyway), get Tier 1 live, and treat Tier 2 as optional
polish once everything else works — don't let cloud-account setup block the actual build.

**Free-tier limitations to know about (Tier 1):**
- **Render**: free web services spin down after 15 minutes of inactivity. Cold start takes 30–60 seconds.
  This is fine for a portfolio demo — just warn viewers. If you want to eliminate it for a live presentation,
  use UptimeRobot (free) to ping the health endpoint every 14 minutes, or upgrade to Render's $7/mo starter.
- **Qdrant Cloud**: free clusters suspend after 1 week of inactivity and are deleted after 4 weeks. Re-seed
  the vectors after resuming. For a live demo, keep the cluster active by running the embedding sync task
  weekly (the same GH Actions cron that runs `dvc repro` can trigger this).
- **Railway / Fly.io**: both have deprecated permanent free tiers as of 2025–2026. Railway offers a 30-day
  $5 trial; Fly.io requires a payment method. Neither is suitable as a default zero-cost backend host.

---

## 10. Token-efficient AI-assisted build workflow

The uploaded reference doc's core ideas are legitimate — prompt caching, keeping a manifest instead of
re-crawling the repo, model tiering, bounding output length. Here's the same set of ideas grounded in what
Claude Code actually supports, so you can use it directly rather than configuring things that don't exist:

- **`GEMINI.md` at the repo root** — durable, load-bearing facts only (build/test commands, the
  `item_id = parent_asin` rule, "always log to MLflow experiment `DS11-v2`"). Keep it under ~200 lines;
  it's read every session, so it should be things the agent can't infer from the code itself.
- **`PROJECT_MANIFEST.md`** — the compressed directory tree (see §12) so an agent understands layout without
  reading every file. Regenerate it when the structure changes; don't let it drift.
- **Git-ignore the heavy stuff** — `data/*.csv`, `data/*.parquet`, `embeddings/*.npy`, `__pycache__/`,
  `mlflow/mlruns/`. This keeps both git *and* agent file searches away from multi-GB binaries.
- **One phase, one session.** Finish a phase from §11, commit, then start the next session with
  a short brief pointing at `PROJECT_MANIFEST.md`, `GEMINI.md`, and the specific phase from this document —
  this doc's phase table doubles as your session briefs. Long-running threads that drift across phases are
  where context rot shows up as repeated mistakes.
- **`/compact` proactively**, not on autopilot — trigger it yourself with a note of what you want to do next,
  rather than relying on automatic compaction mid-task.
- **Subagents for read-heavy exploration** — e.g. "find every remaining place that uses `asin` instead of
  `item_id`" or "summarize how the v1 Apriori module works" — so the intermediate file-reading noise doesn't
  sit in your main session.
- **Model tiering, for real** — Sonnet 5 (the default) for architecture-level work: ranking-model design,
  API contracts, the two-tower loss function. Route boilerplate — Pydantic schemas, CRUD handlers,
  Dockerfiles, presentational Next.js components — to Haiku 4.5, either via `/model haiku` or a dedicated
  subagent configured to use it.
- **Bound the output, not just the input** — when asking for a specific file, ask for "the file only, no
  explanation" rather than a narrated walkthrough; output tokens cost more than input tokens and unbounded
  explanations are the actual waste in most agentic coding sessions.
- **Cache boundary in the LLM-explanation feature itself** (§6) — this is the one place token efficiency is
  a *product* concern, not just a *development* convenience: cache the static instruction/schema prefix,
  keep only the per-item data volatile.

---

## 11. Phase-by-phase execution plan

| Phase | What | v1 files touched | New files |
|---|---|---|---|
| 0 | Repo scaffold, `GEMINI.md`, `PROJECT_MANIFEST.md`, docker-compose skeleton (postgres, redis, qdrant added, empty) | — | `GEMINI.md`, `PROJECT_MANIFEST.md`, `docker-compose.yml` |
| 1 | Confirm 01–02 still run clean; add Qdrant embedding-sync task | 01, 02, 07 | `pipeline/sync_embeddings.py` |
| 2 | GitHub Actions cron workflow for scheduled retrains (`dvc repro`); `/admin/retrain` endpoint stub | — | `.github/workflows/retrain.yml` |
| 3 | Re-run 03–06, 09 unchanged (regenerate trained models — not ranker features yet, just the model artifacts) | 03–06, 09 | — |
| 4 | **Ranker training data + LGBMRanker**: build `data/ranker_train.parquet` (score generation from each trained model + negative sampling + leakage-safe split); train `LGBMRanker`; log as `Ranker` run in MLflow | 08 | `src/12_ranker_features.py`, `src/12_ranker.py` |
| 5 | (optional) two-tower retrieval model — skip unless Phases 1–4 are done and retrieval quality needs improvement | — | `src/13_two_tower.py` |
| 6 | LLM explanations + query rewriting, with prompt caching (Gemini 3.5 Flash-Lite default) | — | `src/14_llm_layer.py` |
| 7 | FastAPI v2 — async, auth, cache, logging, new endpoints | 12 (api/) | `api/main.py` (rewrite), `api/db.py`, `api/cache.py` |
| 8 | Frontend (Next.js) | — | `web/` |
| 9 | CI: GitHub Actions (lint, test, smoke-retrain, build, deploy on tag) | — | `.github/workflows/ci.yml` |
| 10 | Observability: structured JSON logging config + `/metrics` endpoint (JSON counters, no Prometheus/Grafana containers) | — | `api/logging_config.py` |
| 11 | Deploy Tier 0 → Tier 1 | — | deployment configs per host |
| 12 | Re-run A/B notebook (10) with Ranker + two-tower added as new systems; refresh MLflow report (11) | 10, 11 | — |
| 13 | (optional) Tier 2 stubs: Terraform + K8s manifests, deploy once, screenshot/document, tear down | — | `infra/terraform/`, `infra/k8s/` |

---

## 12. Repo structure (v2)

```
amazon_project/
├── GEMINI.md
├── PROJECT_MANIFEST.md
├── docker-compose.yml
├── config.py
├── dvc.yaml
├── pipeline/
│   └── sync_embeddings.py
├── src/                        # 01-11 unchanged from v1, plus:
│   ├── 12_ranker_features.py   # score generation + negative sampling → data/ranker_train.parquet
│   ├── 12_ranker.py            # LGBMRanker training + MLflow logging
│   ├── 13_two_tower.py         # optional
│   └── 14_llm_layer.py
├── api/
│   ├── main.py
│   ├── db.py
│   ├── cache.py
│   ├── schemas.py
│   └── logging_config.py       # structured JSON logging + /metrics counters
├── web/                         # Next.js app
├── infra/                       # optional Tier 2
│   ├── terraform/
│   └── k8s/
├── .github/workflows/
│   ├── ci.yml
│   └── retrain.yml              # cron-triggered dvc repro
├── data/  embeddings/  models/  mlflow/  outputs/    # same as v1, git-ignored
```

---

## 13. Decision log (short form)

- **Two-stage retrieval→ranking over one hybrid function** — matches how every large-scale recommender
  (Google, Redis's own reference guide, recent SIGIR two-tower work) is actually structured; also directly
  fixes the "hand-tuned weights" weakness in v1's `HybridRecommender`.
- **Qdrant over pgvector** — you already have Postgres for relational data; a dedicated vector DB is faster
  to stand up for ANN specifically and has a generous free cloud tier for the deployed version.
- **LightGBM over a deep ranker** — trains in minutes on CPU, which matches your existing hardware
  constraints (same reasoning v1 used for ALS/SVD++ on CPU).
- **DVC + GitHub Actions cron over Prefect/Airflow** — DVC already owns the DAG and caching; adding a
  separate scheduler (Prefect) meant maintaining two DAG definitions or a thin cron wrapper with extra
  infrastructure (scheduler process, database, container). GitHub Actions cron provides the same trigger
  capability with zero operational overhead.
- **Free-tier deployment (Tier 1) as the default target, not AWS** — gets you a real public URL without a
  cloud bill; AWS/GCP (Tier 2) is explicitly optional polish, not a blocker to calling this "done."
- **Gemini 3.5 Flash-Lite (`gemini-3.5-flash-lite`) for the LLM layer by default** — the two LLM tasks (one-sentence explanations, query
  rewriting) are simple structured-output generation; 3.5 Flash-Lite provides 2x the rate-limit headroom, ~100ms latency, and lowest token cost (~$0.075/$0.30 per 1M tokens) without compromising structured accuracy.
- **Structured logging + `/metrics` over Prometheus + Grafana** — self-hosting two observability containers
  for a single-service demo adds operational weight without proportional value. A `/metrics` JSON endpoint
  gives the same interview-demo answer; Grafana Cloud free tier is the upgrade path if dashboards are wanted.

---

## 13a. Known deferred items (from v1 codebase audit)

Fixed already: F1–F7, F9, F10, F13 (the confirmed blockers + the two elevated main-guard warnings).
Everything below was deliberately left alone as low-risk — don't fix these speculatively, but don't let
them get lost either. Revisit at the phase noted, or opportunistically if you're already in that file.

| # | Issue | File | Revisit at |
|---|---|---|---|
| F8 | MLflow logs `epochs: 8`/`10` for MF/NCF while actually training 15 | `06_mf_ncf_pytorch.py` | Before Phase 12 (A/B + MLflow report) — wrong logged params make the leaderboard misleading |
| F22 | Eval uses global `np.random.choice`, not a seeded generator | `05_content_cf_recommender.py` | Before Phase 12 — non-deterministic eval makes system comparisons untrustworthy |
| F17 | Fragile `COL_CATEGORY` fallback for a column name that may shift between merge/preprocess | `05_content_cf_recommender.py` | Before Phase 5 (ranker/two-tower reuse this file's output) |
| F11 | MLflow tracking URI string differs between `config.py` and pipeline scripts | `config.py` + all `src/*.py` | Before Phase 7 (backend + registry work touches tracking config directly) |
| F20 | Vestigial `asin_to_item_idx` getattr fallback, always resolves to `item_map` | `api/main.py` | Naturally — Phase 7 rewrites this file anyway |
| F15 | Hardcoded paths instead of `config.py` constants | most `src/*.py` | Opportunistic, no forcing phase |
| F16 | Dead `register_dvc_stage()` generating a redundant `.sh` file | `07_semantic_search.py` | Opportunistic |
| F18 | `torch` imported eagerly at module top instead of lazily | `config.py` | Opportunistic |
| F19 | Commented-out `apriori_gt_confirmed_pct` metric logging | `04_apriori_recommender.py` | Opportunistic |
| F21 | `colab_download_embeddings()` only prints instructions, doesn't download | `07_semantic_search.py` | Opportunistic |
| F14 | Extra `phase1_data_ingestion` MLflow run beyond the 7 core runs | `01_data_ingestion.py` | Not a bug — just document it, no fix needed |

---

## 14. Delivery checklist

- [x] `GEMINI.md` + `PROJECT_MANIFEST.md` committed
- [ ] GitHub Actions retrain workflow runs `dvc repro` end to end (and `/admin/retrain` works on demand)
- [ ] Qdrant populated, ANN search returns sane neighbors
- [ ] `LGBMRanker` trained, logged to MLflow as `Ranker`
- [ ] LLM explanations cached in Redis, cache-hit rate visible in `/metrics`
- [ ] FastAPI v2 endpoints pass `api/test_api.py` (extend v1's tests)
- [ ] Next.js frontend hits the live API, shows personalization changing per demo user
- [ ] GitHub Actions CI green on push
- [ ] Tier 0 `docker-compose up` starts everything
- [ ] Tier 1 deployed, public URL works
- [ ] `10_ab_comparison.ipynb` re-run with Ranker (+ two-tower if built) added as new systems
- [ ] `11_mlflow_report.ipynb` re-exported with all runs including `Ranker`
