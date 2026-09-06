# Codebase Audit Report — DS11 Recommender System

**Auditor**: Antigravity  
**Date**: 2026-08-24  
**Scope**: All files in `src/`, `api/`, `config.py`, `requirements.txt`, `dvc.yaml`, `Dockerfile`, `docker-compose.yml`

---

## 1. Findings Table

| # | Severity | File | Line(s) / Function | Issue | Suggested Fix |
|---|----------|------|---------------------|-------|---------------|
| F1 | **BLOCKER** | [09_als_svdpp.py](file:///e:/rs/src/09_als_svdpp.py#L89) | L89 | **Circular / undeclared DVC dependency**: Code loads `models/user_item_matrix.npz` (produced by Phase 8), but the DVC `als_svdpp` stage does NOT list it as a dep. Meanwhile, DVC `hybrid` stage lists it as a dep AND produces it. **`dvc repro` from clean state will fail** because `als_svdpp` needs this file before `hybrid` runs, but `hybrid` is the one that creates it. | Either (a) have `08_hybrid_engine.py` or `06_mf_ncf_pytorch.py` produce the matrix in an earlier DVC stage, or (b) restructure so `09_als_svdpp.py` builds its own matrix from `train_df.parquet` instead of loading the Phase 8 artifact. |
| F2 | **BLOCKER** | [09_als_svdpp.py](file:///e:/rs/src/09_als_svdpp.py#L37) | L37, L69-70 | **DVC stage `als_svdpp` has wrong deps**: Code loads `data/train_df.parquet` (L37) and `data/test_df.parquet` (L69), but the DVC stage declares `data/cleaned_cf_dataset.parquet` as its data dep. The code doesn't even read `cleaned_cf_dataset.parquet`. DVC won't auto-re-run this stage if `train_df.parquet` changes. | Update `dvc.yaml` `als_svdpp` deps to `data/train_df.parquet`, `data/test_df.parquet`, and `models/user_item_matrix.npz`. Remove `data/cleaned_cf_dataset.parquet` from deps. |
| F3 | **BLOCKER** | [dvc.yaml](file:///e:/rs/dvc.yaml#L94) | L94 | **Hybrid stage declares its own output as a dependency**: `models/user_item_matrix.npz` is listed as a dep of the `hybrid` stage, but `08_hybrid_engine.py` creates it at runtime (L477). DVC tracks this as an input, meaning it must exist before the stage runs. On a clean checkout, `dvc repro` will fail for this stage. | Remove `models/user_item_matrix.npz` from the `hybrid` deps list and add it to its `outs` list. |
| F4 | **BLOCKER** | [requirements.txt](file:///e:/rs/requirements.txt) | — | **Missing critical dependencies**: The following packages are `import`ed in source but absent from `requirements.txt`: `scikit-surprise` (09), `dill` (04, 08), `requests` (01), `tqdm` (01, 06, 07), `seaborn` (03), `wordcloud` (03), `pyarrow` (03b — needed even though pandas can use it implicitly, `pq.read_schema` requires explicit pyarrow), `matplotlib` (02, 03, 06), `umap-learn` (06, guarded by try/except so not fatal). **Docker build will fail** for any pipeline step using these. | Add to `requirements.txt`: `scikit-surprise`, `dill`, `requests`, `tqdm`, `seaborn`, `wordcloud`, `pyarrow`, `matplotlib`. Optionally `umap-learn`. |
| F5 | **BLOCKER** | [07_semantic_search.py](file:///e:/rs/src/07_semantic_search.py#L110) | L110 | **`colab_upload_parquet()` called but never defined**: In the Colab path, `load_data()` calls `colab_upload_parquet()` (L110), but this function does not exist anywhere in the file or project. Runtime `NameError` crash if triggered. | Define the function (e.g., using `google.colab.files.upload()`) or remove the dead Colab-upload path if the file is always pre-placed. |
| F6 | **BLOCKER** | [docker-compose.yml](file:///e:/rs/api/docker-compose.yml#L22) | L22 | **Healthcheck uses `curl` but `curl` is not installed in the Docker image**: The `python:3.10-slim` base image does not include `curl`, and the Dockerfile doesn't install it. The healthcheck `CMD curl -f http://localhost:8000/health` will always fail, causing Docker to report the container as unhealthy. | Either install `curl` in the Dockerfile (`apt-get install -y curl`), or switch healthcheck to `python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"`. |
| F7 | **BLOCKER** | [Dockerfile](file:///e:/rs/api/Dockerfile#L18-L19) | L18-19 | **Dockerfile copies `requirements.txt` from project root but is missing the deps from F4**: Even if F4 is fixed, the Dockerfile `COPY requirements.txt` → `pip install -r requirements.txt` would fail on `dill` import at startup when deserializing `hybrid_recommender.pkl` (which uses `dill`). | Fix F4 first; dill + pyarrow must be in requirements.txt for the Docker image to work. |
| F8 | **WARNING** | [06_mf_ncf_pytorch.py](file:///e:/rs/src/06_mf_ncf_pytorch.py#L351) | L342→L351, L379→L388 | **MLflow logs wrong epoch count**: MF trains for 15 epochs (L342) but logs `"epochs": 8` (L351). NCF trains for 15 epochs (L379) but logs `"epochs": 10` (L388). **MLflow report will show incorrect hyperparameters.** | Change logged values to 15 for both, or define an `MF_EPOCHS` / `NCF_EPOCHS` constant used in both the training call and the log call. |
| F9 | **WARNING** | [02_preprocessing.py](file:///e:/rs/src/02_preprocessing.py) | entire file | **No `if __name__ == "__main__"` guard**: All code executes at import time. If any other module ever imports from this file (e.g., to reuse `clean_text()`), the entire preprocessing pipeline runs — loading 638 MB CSV, generating plots, and overwriting `clean_merge_df.parquet`. Also prevents use as a library. | Wrap lines 15-245 inside `def main(): ...` and add `if __name__ == "__main__": main()`. |
| F10 | **WARNING** | [03_sentiment_nlp.py](file:///e:/rs/src/03_sentiment_nlp.py) | entire file | **Same issue as F9**: No main guard. All code runs at import. Also not registered as a DVC stage, which means NLP/sentiment model training is invisible to the reproducibility pipeline. SVM model artifacts exist but aren't tracked by DVC. | Add main guard. Optionally add a DVC stage for the sentiment step. |
| F11 | **WARNING** | [config.py](file:///e:/rs/config.py#L73) | L73 | **MLflow tracking URI inconsistency**: `config.py` defines `MLFLOW_TRACKING_URI = "file:./mlflow"` but every script uses the bare string `"mlflow/"` directly. The `file:./mlflow` scheme is correct MLflow URI format; bare `"mlflow/"` is interpreted as a relative directory path, which also works but is **not equivalent** (different URI scheme). `config.setup_mlflow()` is only called in `01_data_ingestion.py`; all other files ignore it. | Either use `config.setup_mlflow()` everywhere, or standardise the string. The `"mlflow/"` form works for file-based tracking, but the inconsistency means `config.py` is a dead abstraction for MLflow. |
| F12 | **WARNING** | [08_hybrid_engine.py](file:///e:/rs/src/08_hybrid_engine.py#L47-L48) | L47-48 | **MLflow `set_tracking_uri` + `set_experiment` at module level**: These run at import time regardless of whether `main()` is called. If the API imports this module (it does via `dill.load` of `HybridRecommender`), MLflow is configured as a side effect. Not directly harmful but breaks separation of concerns. | Move into `main()` or gate behind `if __name__`. |
| F13 | **WARNING** | [09_als_svdpp.py](file:///e:/rs/src/09_als_svdpp.py) | entire file | **No main guard (same as F9/F10)**: All code — training ALS, SVD++, MLflow logging — runs at import time. Also has duplicate `import scipy.sparse as sp` (L8 and L87). | Wrap in `def main(): ...` + `if __name__` guard. Remove duplicate import. |
| F14 | **WARNING** | [01_data_ingestion.py](file:///e:/rs/src/01_data_ingestion.py#L100) | L100 | **Only file that calls `config.setup_mlflow()`**: Uses the centralized function, but then the run it opens (`phase1_data_ingestion`) is **not one of the 7 expected runs** listed in the task context (SVM, content_only, MF, NCF, ALS, SVDpp, Hybrid). It's an extra run. Not a bug, but the context says "7 runs expected" — there are actually **10 unique run names** across the codebase (phase1_data_ingestion, SVM, T5_summary, content_only, cf_item_item, MF, NCF, ALS, SVDpp, Hybrid). | Document the actual set of MLflow runs. The 7 specified in context are present and correctly configured; the extras (phase1, T5_summary, cf_item_item) are bonus. |
| F15 | **WARNING** | [02_preprocessing.py](file:///e:/rs/src/02_preprocessing.py#L20-L21) | L20-21 | **Hardcoded relative paths instead of `config.py` constants**: Uses `"data/merge_df.csv"` and `"data/clean_merge_df.parquet"` as strings. All other files (01, 03b, 04, 05, 06, 07) also use hardcoded paths. `config.py` defines `MERGE_CSV_PATH`, `CLEAN_PARQUET_PATH`, etc. but only file 01 imports from config. | Use `config.*_PATH` constants everywhere, or accept that config.py path constants are a dead abstraction except for file 01 and the API. |
| F16 | **WARNING** | [07_semantic_search.py](file:///e:/rs/src/07_semantic_search.py#L511-L513) | L511-513 | **`register_dvc_stage()` writes a `.sh` file and calls `chmod()`**: On Windows (user's OS), `sh.chmod(0o755)` will silently succeed but produce no actual permission change. The generated `dvc_embed_stage.sh` is dead code since DVC stages are already registered in `dvc.yaml`. | Remove `register_dvc_stage()` entirely — the stage is already in `dvc.yaml`. |
| F17 | **WARNING** | [05_content_cf_recommender.py](file:///e:/rs/src/05_content_cf_recommender.py#L54) | L54 | **`COL_CATEGORY = "main_category_meta"` may mismatch**: The column produced by the merge in 01 would be `main_category_meta` (due to `suffixes=("_rev", "_meta")`), but after preprocessing in 02 it could be just `main_category`. The code has a fallback (L82) but the mismatch is fragile. | Verify the actual column name in `clean_merge_df.parquet` and set `COL_CATEGORY` accurately, or use the fallback pattern consistently. |
| F18 | NICE-TO-HAVE | [config.py](file:///e:/rs/config.py#L2) | L2 | **`import torch` at top of config.py**: Forces PyTorch to load for any script that imports config, even those that don't need GPU detection. Adds ~3 sec startup overhead. | Lazy-import torch inside `DEVICE` assignment or the functions that need it. |
| F19 | NICE-TO-HAVE | [04_apriori_recommender.py](file:///e:/rs/src/04_apriori_recommender.py#L355-L356) | L355-356 | **Commented-out MLflow logging**: `# mlflow.log_metric("apriori_gt_confirmed_pct", gt_pct)` — Apriori validation metric is computed but never logged. Not logged to any run. | Either uncomment and add a dedicated MLflow run for Apriori, or remove the dead comment. |
| F20 | NICE-TO-HAVE | [api/main.py](file:///e:/rs/api/main.py#L306-L307) | L306-307 | **`asin_to_item_idx` phantom attribute**: The code does `getattr(cf_rec, "asin_to_item_idx", getattr(cf_rec, "item_map", {}))`. `asin_to_item_idx` is never defined on `CollaborativeFilteringRecommender` — the fallback to `item_map` always triggers. The variable name and docstring comments referencing this attribute are vestigial from an earlier design. | Clean up: remove the `asin_to_item_idx` branch and reference `item_map` directly. Update docstrings at L12, L100, L209, L287, L301. |
| F21 | NICE-TO-HAVE | [07_semantic_search.py](file:///e:/rs/src/07_semantic_search.py#L88-L96) | L88-96 | **`colab_download_embeddings()` only prints instructions**: Doesn't actually trigger `files.download()` — just prints the Python code the user should paste. Confusing UX. | Actually call `files.download()` inside the function or rename to `print_download_instructions()`. |
| F22 | NICE-TO-HAVE | [05_content_cf_recommender.py](file:///e:/rs/src/05_content_cf_recommender.py#L326) | L326, L351 | **`np.random.choice(train_items)` in eval uses global random state**: Seed selection during evaluation uses the deprecated global `np.random` API rather than a seeded `np.random.default_rng()`. Makes eval results non-deterministic across runs if the global seed is not set beforehand. | Use `rng = np.random.default_rng(RANDOM_STATE)` and `rng.choice()`. |
| F23 | NICE-TO-HAVE | [Dockerfile](file:///e:/rs/api/Dockerfile#L21) | L21 | **Pre-downloads e5-base-v2 model into Docker image**: `RUN python -c "... SentenceTransformer('intfloat/e5-base-v2')"` adds ~400 MB to the image layer. Good for cold-start latency but makes builds slow and images large. | Accept this tradeoff or mount a model cache volume instead. Not a bug. |
| F24 | NICE-TO-HAVE | — | — | **`torchvision` and `torchaudio` in requirements.txt**: Neither is imported anywhere in the codebase. Adds ~500 MB to Docker image for no benefit. | Remove `torchvision` and `torchaudio` from `requirements.txt`. |

---

## 2. Project-Specific Checks Summary

### ✅ Check 1 — No `asin` as join key
No remaining `left_on='asin'` or `on='asin'` anywhere. All merges use `parent_asin` (file 01, L125). All downstream code uses `item_id` consistently. **PASS.**

### ✅ Check 2 — No random sampling on merged dataframe
The two `.sample()` calls found are in evaluation code:
- `09_als_svdpp.py` L218: samples from `df_implicit` for RMSE calculation (legitimate)
- `08_hybrid_engine.py` L342: samples test users for evaluation (legitimate)

User-activity filter (`user_counts >= 5`) is the only filtering mechanism in `01_data_ingestion.py` L150-152. **PASS.**

### ✅ Check 3 — `item_id` used consistently
Files 05, 06, 07, 08, 09 all use `item_id` as the canonical key for embeddings, product_vecs, CF matrices, and index lookups. No mix of `asin` / `parent_asin` / `item_id` across files. **PASS.**

### ⚠️ Check 4 — MLflow runs
All 7 expected runs exist with correct `set_tracking_uri` + `set_experiment` before `start_run`:

| Run Name | File | URI | Experiment | Params/Metrics |
|----------|------|-----|------------|----------------|
| SVM | 03_sentiment_nlp.py | `"mlflow/"` | `"DS11"` | ✅ params + metrics |
| content_only | 05_content_cf_recommender.py | `"mlflow/"` (via MLFLOW_URI) | `"DS11"` (via EXPERIMENT_NAME) | ✅ |
| MF | 06_mf_ncf_pytorch.py | `"mlflow/"` | `"DS11"` | ⚠️ epochs param wrong (F8) |
| NCF | 06_mf_ncf_pytorch.py | `"mlflow/"` | `"DS11"` | ⚠️ epochs param wrong (F8) |
| ALS | 09_als_svdpp.py | `"mlflow/"` | `"DS11"` | ✅ |
| SVDpp | 09_als_svdpp.py | `"mlflow/"` | `"DS11"` | ✅ |
| Hybrid | 08_hybrid_engine.py | `MLFLOW_URI` = `"mlflow/"` | `"DS11"` | ✅ |

**3 bonus runs** also exist: `phase1_data_ingestion`, `T5_summary`, `cf_item_item`.

**PARTIAL PASS** — all 7 runs exist and are logged, but MF/NCF have incorrect epoch params (F8).

### ❌ Check 5 — DVC stages
The DVC pipeline has **8 stages** (ingest, preprocess, summarize, apriori, content_cf, mf_ncf, embed, als_svdpp, hybrid) — more than the 3 listed in the task context (ingest→preprocess→embed). All stages are registered in `dvc.yaml` and `dvc.lock` exists with hashes.

However, **`dvc repro` from a clean state will NOT succeed** due to:
- F1: Circular dependency on `user_item_matrix.npz` (Phase 8 creates it, Phase 9 needs it, but Phase 9 must run before Phase 8 can consume its outputs)
- F2: `als_svdpp` stage declares wrong deps
- F3: `hybrid` stage lists its own output as a dep

**FAIL** — DVC pipeline is not end-to-end reproducible from clean state.

### ✅ Check 6 — Dead code
| Dead Code | Location | Status |
|-----------|----------|--------|
| `config.create_dirs()` | [config.py](file:///e:/rs/config.py#L90) L90 | Only called in `__main__` block; every script has its own `create_dirs()` |
| `config.set_seed()` | [config.py](file:///e:/rs/config.py#L116) L116 | Only called in `__main__` block; never used by pipeline scripts |
| `config.setup_mlflow()` | [config.py](file:///e:/rs/config.py#L107) L107 | Only used by `01_data_ingestion.py`; all other files inline the setup |
| `register_dvc_stage()` | [07_semantic_search.py](file:///e:/rs/src/07_semantic_search.py#L499) L499 | Generates a `.sh` file for a stage already in `dvc.yaml` |
| `colab_download_embeddings()` | [07_semantic_search.py](file:///e:/rs/src/07_semantic_search.py#L88) L88 | Only prints instructions; never actually downloads |
| Config path constants | [config.py](file:///e:/rs/config.py#L34-L63) L34-63 | `MERGE_CSV_PATH`, `CLEAN_PARQUET_PATH`, `CF_DATA_PATH`, etc. — only `01_data_ingestion.py` uses them; all other files hardcode paths |

### ✅ Check 7 — Incomplete implementations
**One found**: `colab_upload_parquet()` is called at L110 of `07_semantic_search.py` but never defined (F5). No bare `pass` stubs or TODO-marked functions found.

### ⚠️ Check 8 — Error handling
- **API**: Good. All model loads in the lifespan handler have try/except. Endpoints raise `HTTPException` appropriately.
- **Pipeline scripts**: File loads in 02, 03, 03b, 04 have **no error handling** — if `clean_merge_df.parquet` doesn't exist, they crash with an unguarded `FileNotFoundError`. Acceptable for batch scripts but risky in a DVC pipeline where stage ordering might be wrong.

### ⚠️ Check 9 — Config/hardcoding
See F15 — every file except 01 hardcodes its paths. Magic numbers (e.g., `MIN_ITEM_FREQ = 10`, `BATCH_SIZE = 64/8`, `HYBRID_EMB_W = 0.55`) are defined as module constants, which is acceptable. No API keys or secrets found.

### ❌ Check 10 — requirements.txt
See F4 for the full list. Also F24 for unused deps.

### ✅ Check 11 — Deliverables exist

| File | Exists | Size |
|------|--------|------|
| `outputs/final_top500_product_summary.csv` | ✅ | 193 KB |
| `outputs/ab_comparison_results.csv` | ✅ | 396 B |
| `outputs/mlflow_report.html` | ✅ | 809 KB |
| `models/hybrid_recommender.pkl` | ✅ | 317 MB |

All deliverables exist and are non-empty. **PASS.**

### ⚠️ Check 12 — Docker
Cannot run `docker-compose up` in this audit session, but static analysis reveals:
- F6: Healthcheck will fail (no `curl` installed)
- F7: Missing pip dependencies will crash on model load
- F4 must be fixed first

**Predicted FAIL** for clean Docker startup.

---

## 3. Verdict

> **This codebase is NOT a safe base for v2 without fixing the BLOCKER items first.** The pipeline code works when run manually in the right order on a machine with all dependencies installed, and the trained artifacts are all present and valid. However, there are **7 BLOCKER-level findings** that prevent the system from being reproducible (`dvc repro` fails), deployable (Docker build/startup fails due to missing deps and a broken healthcheck), and robust (a `NameError` crash in the Colab path). The WARNING items — especially the wrong MLflow epoch params (F8) and the lack of main guards (F9/F10/F13) — add fragility but are not blocking. Once the BLOCKERs are fixed, this is a solid, well-structured codebase with clean separation between pipeline stages, a well-designed hybrid recommender, and comprehensive API tests.

---

## 4. BLOCKER Action List

Hand this back to fix in a separate session:

1. **F4 — Fix `requirements.txt`**: Add `scikit-surprise`, `dill`, `requests`, `tqdm`, `seaborn`, `wordcloud`, `pyarrow`, `matplotlib`. Remove unused `torchvision`, `torchaudio`.

2. **F1 + F2 + F3 — Fix DVC `dvc.yaml` dependency graph**:
   - **`als_svdpp` stage**: Change deps from `data/cleaned_cf_dataset.parquet` to `data/train_df.parquet` + `data/test_df.parquet` + `models/user_item_matrix.npz`.
   - **`hybrid` stage**: Move `models/user_item_matrix.npz` from deps to outs.
   - **Stage ordering**: Ensure `hybrid` runs BEFORE `als_svdpp` (since hybrid produces `user_item_matrix.npz` that ALS needs), OR refactor `09_als_svdpp.py` to build its own matrix.

3. **F5 — Define or remove `colab_upload_parquet()`** in `07_semantic_search.py` L110.

4. **F6 — Fix Docker healthcheck**: Install `curl` in Dockerfile or switch to a Python-based healthcheck.

5. **F7 — Verify Docker build after F4 is fixed**: Rebuild image, confirm `dill` import works at startup for `hybrid_recommender.pkl` deserialization.
