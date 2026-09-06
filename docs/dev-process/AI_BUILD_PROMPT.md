# Build Prompts for RecSys v2 Rework

This document provides:
1. **Rule File Seed** (`GEMINI.md` for Antigravity)
2. **Reusable Per-Phase Prompt Template** (for starting fresh sessions in Agent Manager)
3. **Phase 0 Worked Example** (ready to paste to kick off the v2 rework)

---

## 1. Rule File Seed (`GEMINI.md`)

*Place at the repository root (`e:\rs\GEMINI.md`). Antigravity automatically discovers and injects these rules on every turn.*

```markdown
# Amazon RecSys v2 — Antigravity Project Rules & Context

## What this is
DS11 hybrid recommender (v1: content-based + ALS/SVD++/MF/NCF + Apriori, complete) being reworked into a
two-stage retrieval→ranking system with a real backend, frontend, and deployment. Full plan, architecture,
and rationale live in `RECSYS_V2_WORKFLOW_AND_DESIGN.md` — read it before starting any phase, don't ask to
re-explain decisions that are already written down there.

## Hard rules (do not violate)
- `item_id` = `parent_asin`, always. Never join or index on the raw `asin` column.
- Every model training run logs to MLflow experiment `DS11-v2` (`mlflow.set_tracking_uri('mlflow/')`,
  `mlflow.set_experiment('DS11-v2')`) with params + metrics, before you consider a phase done.
- Don't touch `src/01`–`src/11` unless the current phase in the design doc explicitly says to.
- New Python deps go in `requirements.txt` (or `pyproject.toml`, pick one and stay consistent), pinned.
- Data/embeddings/model binaries are git-ignored — never `git add` anything under `data/`, `embeddings/`,
  `models/`, `mlflow/mlruns/`.
- `src/*.py` pipeline scripts must guard execution behind `if __name__ == "__main__":` — the scheduled
  retrain workflow and downstream modules import these directly rather than running them as standalone scripts.
- Long-running commands (`dvc repro`, `docker build`, `docker-compose up`, full training runs) exceed the
  agent's terminal tool timeout — never run them directly. Give the exact command with output redirected
  to a log file (e.g. `dvc repro 2>&1 | tee logs/dvc_repro.log`), wait for confirmation it's done, then
  read the log file yourself. Quick things — single-file syntax checks, one-module unit tests — are fine
  to run directly; this rule is only for the slow ones.
- DVC owns the pipeline DAG (`dvc.yaml`). There is no separate orchestration framework — GitHub Actions
  cron just triggers `dvc repro` on a schedule. Don't introduce a second DAG definition.
- No Prometheus/Grafana containers — observability is a structured JSON `/metrics` endpoint (see design
  doc §3/§9/§11). Don't add monitoring containers to docker-compose.

## Commands
- Run tests: `pytest`
- Run API locally: `uvicorn api.main:app --reload`
- Full local stack: `docker-compose up`
- Retrain pipeline: `dvc repro` (redirect to a log file per the long-running-command rule above)

## Working style
- One phase per Agent session (see `RECSYS_V2_WORKFLOW_AND_DESIGN.md` §11). When a phase is done:
  tests pass, MLflow run logged if applicable, changes committed with a message naming the phase — then
  stop and tell me it's ready for a fresh session.
- For boilerplate (Pydantic schemas, CRUD handlers, Dockerfiles, presentational frontend components), use a
  fast model — don't burn heavy reasoning on things that don't need it.
- When asked for a specific file, give the file. Skip long walkthroughs unless asked.
```

---

## 2. Reusable Per-Phase Prompt Template

*Fill in the bracketed placeholders and paste as the first prompt in a **new session** (via Antigravity Agent Manager) for each phase in §11:*

```markdown
Phase [N] of the v2 recommender rework: [Phase Name from RECSYS_V2_WORKFLOW_AND_DESIGN.md §11]

Context:
- Read RECSYS_V2_WORKFLOW_AND_DESIGN.md, section [N], before doing anything else.
- Read PROJECT_MANIFEST.md for current repo layout — don't crawl the whole tree.
- GEMINI.md has the hard rules; follow them without me repeating them here.

Scope for this session — ONLY this phase:
[1-3 bullet points of what "done" means for this phase, copied/adapted from the design doc's phase table (§11) and delivery checklist (§14)]

Explicitly out of scope this session:
[Name the next phase or two, so the agent stays strictly within scope]

Constraints:
- If you need to explore unfamiliar parts of the codebase, keep file reads targeted — don't fill this session's context with unnecessary bulk dumps.
- Output code files directly; keep prose explanation to a couple of sentences per file, not a walkthrough.
- If something in the design doc conflicts with what you find in the actual code, report the conflict and stop — don't silently guess.

When the scope above is met: run the tests, confirm they pass, commit with message "[Phase Name]: <one-line summary>", update the relevant checklist line in RECSYS_V2_WORKFLOW_AND_DESIGN.md §14, and tell me it's ready to start a fresh conversation for the next phase.
```

---

## 3. Worked Example — Phase 0 (Ready to paste as-is)

*Paste this prompt into a new Agent conversation to execute Phase 0:*

```markdown
Phase 0 of the v2 recommender rework: repo scaffold + agent context files

Context:
- Read RECSYS_V2_WORKFLOW_AND_DESIGN.md sections 0, 11, and 12 before doing anything else — section 12 has the target repo tree.
- This is a fresh scaffold on top of the existing v1 amazon_project/ repo. Don't touch src/01-11.

Scope for this session — ONLY this phase:
- Ensure GEMINI.md exists at repo root with the standard rules seed from AI_BUILD_PROMPT.md §1.
- Generate PROJECT_MANIFEST.md: a compressed structural tree of the actual current repo (under 150 lines, structural only — no large file content dumps).
- Add a root docker-compose.yml with clean skeleton service definitions for: api, postgres, redis, qdrant.
  No prometheus/grafana containers — observability is a JSON `/metrics` endpoint inside the API itself.
- Verify .gitignore covers data/, embeddings/, models/, mlflow/mlruns/, __pycache__/, .env.

Explicitly out of scope this session:
- Phase 1 (Qdrant embedding sync) and any model training/retrieval changes.

Constraints:
- Output files directly; keep prose explanation to a couple of sentences per file.

When the scope above is met: run `docker compose config` to confirm the compose file is valid, commit with message "phase 0: repo scaffold + agent context files", check off Phase 0 items in RECSYS_V2_WORKFLOW_AND_DESIGN.md §14, and tell me it's ready to move to a new conversation for Phase 1.
```
