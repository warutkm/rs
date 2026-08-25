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
  retrain workflow and downstream modules import these directly.
- DVC owns the pipeline DAG (`dvc.yaml`). There is no separate orchestration framework — GitHub Actions
  cron just triggers `dvc repro` on a schedule. Don't introduce a second DAG definition.
- No Prometheus/Grafana containers — observability is a structured JSON `/metrics` endpoint (see design
  doc §3/§9/§11). Don't add monitoring containers to docker-compose.
- Long-running commands (`dvc repro`, `docker build`, `docker-compose up`, full training runs) exceed the
  agent's terminal tool timeout — never run them directly. Give the exact command with output redirected
  to a log file (e.g. `dvc repro 2>&1 | tee logs/dvc_repro.log`), wait for confirmation it's done, then
  read the log file yourself. Quick things — single-file syntax checks, one-module unit tests — are fine
  to run directly; this rule is only for the slow ones.

## Commands
- Run tests: `pytest`
- Run API locally: `uvicorn api.main:app --reload`
- Full local stack: `docker-compose up`
- Retrain pipeline: `dvc repro` (redirect to a log file per the long-running-command rule above)

## Working style
- One phase per session (see `RECSYS_V2_WORKFLOW_AND_DESIGN.md` §11). When a phase is done: tests pass,
  MLflow run logged if applicable, changes committed with a message naming the phase — then stop and tell
  the user it's ready for `/clear`.
- For boilerplate (Pydantic schemas, CRUD handlers, Dockerfiles, presentational frontend components), use a
  fast model — don't burn heavy reasoning on things that don't need it.
- When asked for a specific file, give the file. Skip long walkthroughs unless asked.
- **Next-Phase Prompt Hand-off**: When a phase is successfully completed, committed, and its checklist item updated in `RECSYS_V2_WORKFLOW_AND_DESIGN.md` §14, ALWAYS provide the exact, fully populated prompt for the **next phase** formatted directly from `AI_BUILD_PROMPT.md §2` (with zero placeholder brackets left) so the user can immediately paste it into a new conversation.
