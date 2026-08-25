# v2 Design Doc — Research Findings Report

**Session type**: Research only (no code changes)
**Date**: 2026-08-25

---

## Axis 1: Retrieval / Serving Infra — **KEEP (Qdrant)**

**Scout findings**: Qdrant, ChromaDB, pgvector, and Milvus Lite were evaluated. pgvector avoids a new service but lacks HNSW payload filtering. ChromaDB is developer-friendly but not designed for filtered ANN at serving time. Milvus Lite is overkill. Qdrant's Rust-based engine is purpose-built for filtered ANN retrieval (e.g., "similar items in category X under price Y"), which is exactly the serving pattern §7's `/v2/similar` and `/v2/search` endpoints need. Qdrant Cloud free tier provides 1 GB RAM / 4 GB disk — sufficient for ~1M vectors at 768 dims, far beyond the 44K items in this catalog. One risk: Qdrant Cloud suspends free clusters after 1 week of inactivity and deletes after 4 weeks, so the Tier 1 deployment needs a keep-alive ping or docs acknowledging this.

**Architect decision**: **KEEP**. Qdrant is the right choice for this project. The filtered-ANN capability directly supports the product-search and similar-items endpoints, the free tier is generous for this catalog size, and it self-hosts cleanly in docker-compose for Tier 0. Add a note about the inactivity suspension policy to §9.

---

## Axis 2: Ranking Approach — **KEEP (LightGBM LambdaMART)**

**Scout findings**: LightGBM `LGBMRanker`, XGBoost `XGBRanker`, and CatBoost `CatBoostRanker` were compared. All three are mature, CPU-trainable, and well-suited for small datasets. CatBoost has native categorical handling (useful if category features were raw strings, but they're already encoded). XGBoost has slightly more community documentation for LTR. LightGBM trains fastest and is already the plan's choice. For 44K items and ~126K interactions, any of the three trains in under a minute. scikit-learn has no native LTR support.

**Architect decision**: **KEEP**. LightGBM LambdaMART is correct. No reason to switch — the difference between the three boosting libraries is negligible at this scale, and LightGBM has the cleanest `LGBMRanker` API with native `group` parameter support. The real risk is not the model choice but the training data construction (negative sampling, feature leakage, score generation), which the design doc under-specifies — see §5 amendment below.

---

## Axis 3: LLM Integration Pattern — **REPLACE provider default (Gemini 2.5 Flash over Haiku 4.5)**

**Scout findings**: For the two high-volume LLM tasks (one-sentence explanations, query rewriting), the scout compared Claude Haiku 4.5 ($1.00/$5.00 per 1M input/output tokens) vs Gemini 2.5 Flash ($0.30/$2.50 per 1M input/output tokens). Gemini Flash is 50–70% cheaper at equivalent capability for these simple, structured-output tasks. Both support prompt caching. Local models (Llama 4, Qwen 3) are cheaper only above ~1M tokens/day, which a demo project won't hit. The grounding pattern (LLM receives only the top-N candidates + their ranker feature vectors, not the whole catalog) is correct and unchanged.

**Architect decision**: **REPLACE the default provider**. Change §6 to recommend **Gemini 2.5 Flash** as the default for both LLM tasks (explanations + query rewriting), with Claude Haiku 4.5 as an equally valid alternative. The tasks are simple, structured-output generation — the cheaper model wins. Keep the architecture pattern (prompt caching, feature-vector grounding, Redis caching of results) unchanged. Also clarify the latency model: first request for a (user, item_set) pair returns recommendations immediately and generates explanations asynchronously, filling them into the cache for subsequent requests.

---

## Axis 4: Orchestration — **REPLACE (DVC stages + GitHub Actions cron, drop Prefect)**

**Scout findings**: Prefect, Dagster, GitHub Actions cron, and DVC-as-orchestrator were evaluated for a solo developer. Prefect adds a scheduler, retry logic, and a UI dashboard — but for a single pipeline that runs weekly at most, GitHub Actions cron provides the same trigger capability with zero operational overhead (no running scheduler process, no database, no extra container in docker-compose). DVC already owns the DAG and caching. The design doc's own ambiguity ("wraps `dvc repro` or calls the same Python functions directly") confirms that Prefect's role was never clearly defined — it was hovering between "cron wrapper" and "parallel DAG engine," and this project needs neither.

**Architect decision**: **REPLACE**. Drop Prefect. Use DVC as the DAG owner (it already is) and GitHub Actions cron as the scheduler for automated retrains. For local/manual runs, `dvc repro` is the command. The `/admin/retrain` endpoint triggers `dvc repro` via subprocess. This removes one service, one dependency, and one conceptual ambiguity. The Prefect flow file (`flows/pipeline_flow.py`) is replaced by a GitHub Actions workflow (`.github/workflows/retrain.yml`) that runs `dvc repro` on a cron schedule.

---

## Axis 5: Deployment — **KEEP (Render + Vercel + Neon + Upstash + Qdrant Cloud), ADD cold-start note**

**Scout findings**: Render, Railway, and Fly.io were compared. Railway dropped its permanent free tier (now a 30-day $5 trial). Fly.io also deprecated its free tier for new accounts. Render remains the only one with a permanent free web service tier, though it spins down after 15 minutes of inactivity with a 30–60 second cold start. For a portfolio demo where the user knows to expect a brief initial load, this is acceptable. Qdrant Cloud free tier suspends after 1 week of inactivity.

**Architect decision**: **KEEP** the current Tier 1 stack. Render is the only remaining zero-cost option for persistent backend hosting. Add an explicit note in §9 about the cold-start behavior (30–60s after 15 min idle) and the Qdrant Cloud suspension policy (1 week idle → suspend, 4 weeks → delete). Suggest a simple UptimeRobot ping (free) as a workaround if the user wants to avoid cold starts for live demos.

---

## Axis 6: Observability — **REPLACE (structured logging + FastAPI `/metrics` endpoint, drop self-hosted Prometheus + Grafana)**

**Scout findings**: Self-hosting Prometheus + Grafana in docker-compose for a single-service demo adds two containers, persistent storage configuration, and dashboard JSON maintenance — operational weight that doesn't match the value for a solo project. The modern approach is OpenTelemetry SDK for instrumentation + Grafana Cloud free tier for storage/visualization. However, for this project the simpler path is even better: structured JSON logging (Python `logging` + `json` formatter) gives full debuggability, and a `/metrics` endpoint returning JSON counters (latency p50/p95/p99, cache hit rate, requests per endpoint) gives the interview-demo "show me your system's health" answer without any external infrastructure.

**Architect decision**: **REPLACE**. Drop the self-hosted Prometheus + Grafana containers from docker-compose. Keep the `/metrics` endpoint (it's trivial to implement in FastAPI middleware) and add structured JSON logging. If the user later wants dashboards, Grafana Cloud's free tier (10K metrics, 50GB logs) is the upgrade path — but the containers themselves are not worth maintaining in Tier 0. This removes two services from docker-compose and simplifies the monitoring/ directory to a single logging config file.

---

## Design Doc Amendments Required

Based on the six axis decisions, the following sections of `RECSYS_V2_WORKFLOW_AND_DESIGN.md` need edits:

| Section | Change | Reason |
|---------|--------|--------|
| §3 (Tech stack) | Replace Prefect row → DVC + GitHub Actions cron | Axis 4: drop Prefect |
| §3 (Tech stack) | Update LLM row → Gemini 2.5 Flash as default, Haiku as alternative | Axis 3: cheaper provider |
| §3 (Tech stack) | Update Observability row → structured logging + `/metrics` endpoint | Axis 6: drop Prom+Grafana |
| §4 (Pipeline) | Replace Prefect flow reference → `dvc repro` + GH Actions cron | Axis 4 |
| §5 (Model layer) | Add subsection on ranker training data construction | Prior review finding |
| §6 (LLM layer) | Update model name to Gemini 2.5 Flash; clarify async explanation latency | Axis 3 |
| §9 (Deployment) | Add cold-start and Qdrant suspension notes | Axis 5 |
| §11 (Phase plan) | Phase 2: GH Actions workflow instead of Prefect flow | Axis 4 |
| §11 (Phase plan) | Phase 3→4 gap: add score-generation step | Prior review finding |
| §11 (Phase plan) | Phase 10: `/metrics` endpoint, no Prom+Grafana containers | Axis 6 |
| §12 (Repo structure) | Remove `flows/` directory, update `monitoring/` | Axes 4 + 6 |

---

## Orphan-File Check

This is a research-only session. No code files were created or modified. The orphan-file table is empty by design.

| Proposed File | Caller / Importer | Status |
|---------------|-------------------|--------|
| *(none)* | — | — |

---

## Four Direct Questions (from prior review, confirmed by research)

### 1. Single most likely thing to break: Ranker training data construction
The LGBMRanker needs a training DataFrame with per-model scores as features. The existing scripts (05, 06, 09) train models and save artifacts — they don't output per-interaction score columns. A new score-generation step is needed between Phase 3 and Phase 4. This is now called out in the §5 and §11 amendments.

### 2. Over-engineered: Redis feature store + Prometheus/Grafana
Redis feature store solved a problem this project doesn't have (real-time feature updates from live traffic). Prometheus + Grafana added two containers for a single-service demo. Both are now simplified in the amendments.

### 3. Under-engineered: Ranker training pipeline + Prefect/DVC boundary
The ranker training data construction was a one-line mention that glossed over negative sampling, feature generation, and leakage prevention. The Prefect/DVC ambiguity is resolved by dropping Prefect entirely.

### 4. Phase ordering gap: Phase 3 → Phase 4
Phase 3 ("re-run scripts unchanged") doesn't produce what Phase 4 needs (per-interaction score columns). The amendment adds an explicit score-generation task to Phase 4's scope and new file list.
