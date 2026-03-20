"""
Phase 8 — Hybrid Engine + Cold-Start
File: src/08_hybrid_engine.py

Workflow actions:
  8.1  HybridRecommender — wraps Apriori + CF + ProductRecommender;
         union of top-20 from each engine, weighted fusion
  8.2  Cold-start: new user  — ProductRecommender + semantic search only
  8.3  Cold-start: new item  — product_text embedding similarity only (< 3 reviews)
  8.4  final_recommendation(item_id, user_id) — auto-detects cold-start;
         returns metadata with source tag per result
  8.5  Log Hybrid run to MLflow — cf_weight, content_weight, apriori_weight,
         Recall@10, NDCG@10, Precision@10
  8.6  Serialize with dill — dill.dump(hybrid_recommender, 'models/hybrid_recommender.pkl')

Depends on (all must exist before running):
  models/apriori_recommender.pkl    <- Phase 4
  models/product_recommender.pkl   <- Phase 5
  models/cf_recommender.pkl        <- Phase 5
  embeddings/product_vecs.npz      <- Phase 7  (keys=['keys','vecs'])
  data/clean_merge_df.parquet      <- Phase 1/2
  data/test_df.parquet             <- Phase 6  (for evaluation metrics)
"""

import os
import json
import sys
import numpy as np
import pandas as pd
import dill          # used for both loading PKLs and saving hybrid_recommender
import mlflow

# =============================================================================
# PATHS
# =============================================================================
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR) if os.path.basename(SRC_DIR) == "src" else SRC_DIR

DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
EMBED_DIR  = os.path.join(BASE_DIR, "embeddings")
MLFLOW_URI = "file:///" + os.path.join(BASE_DIR, "mlflow").replace("\\", "/")

CLEAN_PARQUET      = os.path.join(DATA_DIR,   "clean_merge_df.parquet")
TEST_PARQUET       = os.path.join(DATA_DIR,   "test_df.parquet")
APRIORI_PKL        = os.path.join(MODELS_DIR, "apriori_recommender.pkl")
PRODUCT_PKL        = os.path.join(MODELS_DIR, "product_recommender.pkl")
CF_PKL             = os.path.join(MODELS_DIR, "cf_recommender.pkl")
PRODUCT_VECS_NPZ   = os.path.join(EMBED_DIR,  "product_vecs.npz")
HYBRID_PKL         = os.path.join(MODELS_DIR, "hybrid_recommender.pkl")

# =============================================================================
# Import Phase 4 + Phase 5 classes so pickle can resolve them when loading PKLs
# Both files saved their models with pickle.dump() — the class definition must
# be importable in this process before dill/pickle can deserialise the objects.
# =============================================================================
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Python cannot import files starting with a digit using normal import syntax.
# Use importlib to load Phase 4 + Phase 5 modules so pickle can resolve
# AprioriRecommender, ProductRecommender, CollaborativeFilteringRecommender
# when deserialising the saved PKL files.
import importlib

_mod04 = importlib.import_module("04_apriori_recommender")
AprioriRecommender = _mod04.AprioriRecommender                          # noqa: F841

_mod05 = importlib.import_module("05_content_cf_recommender")
ProductRecommender               = _mod05.ProductRecommender            # noqa: F841
CollaborativeFilteringRecommender = _mod05.CollaborativeFilteringRecommender  # noqa: F841

# =============================================================================
# MLflow — same URI and experiment name used by all phases
# =============================================================================
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("DS11")


# =============================================================================
# LOADERS
# =============================================================================

def _load_pkl(path: str):
    """Use dill for all PKL loading — handles scipy sparse + class references."""
    with open(path, "rb") as f:
        return dill.load(f)


def load_product_vecs() -> dict:
    """
    Load embeddings/product_vecs.npz saved by Phase 7.
    Phase 7 saved with: np.savez(path, keys=id_array, vecs=matrix)
    Returns: {item_id: np.ndarray(768,)} — all values float32.
    """
    data   = np.load(PRODUCT_VECS_NPZ, allow_pickle=True)
    ids    = [str(i) for i in data["keys"].tolist()]
    vecs   = data["vecs"].astype(np.float32)           # (N, 768)
    print(f"[load] product_vecs: {len(ids):,} items  shape={vecs.shape}")
    return {iid: vecs[i] for i, iid in enumerate(ids)}


# =============================================================================
# METRIC HELPERS
# =============================================================================

def _recall(rec: list, rel: set, k: int) -> float:
    return sum(1 for r in rec[:k] if r in rel) / len(rel) if rel else 0.0

def _ndcg(rec: list, rel: set, k: int) -> float:
    dcg  = sum(1.0 / np.log2(i + 2) for i, r in enumerate(rec[:k]) if r in rel)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg if idcg else 0.0

def _precision(rec: list, rel: set, k: int) -> float:
    return sum(1 for r in rec[:k] if r in rel) / k if k else 0.0


# =============================================================================
# 8.1  HybridRecommender
# =============================================================================

class HybridRecommender:
    """
    8.1 — Wraps AprioriRecommender + CollaborativeFilteringRecommender +
    ProductRecommender.  Collects top-20 candidates from each engine, applies
    weighted score fusion, and returns a ranked list with per-result source tags.

    Cold-start paths:
      8.2 — new user  (user_id not in CF known_users) → content + semantic only
      8.3 — new item  (review count < 3)              → embedding similarity only
    """

    def __init__(
        self,
        apriori_rec,
        cf_rec,
        product_rec,
        product_vecs:     dict,          # {item_id: np.ndarray(768,)}
        merge_df:         pd.DataFrame,
        cf_weight:        float = 0.45,
        content_weight:   float = 0.35,
        apriori_weight:   float = 0.20,
        top_k_per_engine: int   = 20,
    ):
        self.apriori_rec    = apriori_rec
        self.cf_rec         = cf_rec
        self.product_rec    = product_rec
        self.product_vecs   = product_vecs
        self.cf_weight      = cf_weight
        self.content_weight = content_weight
        self.apriori_weight = apriori_weight
        self.top_k          = top_k_per_engine

        # Review counts per item_id — drives 8.3 gate
        self._review_counts: dict = (
            merge_df.groupby("item_id")["rating"].count().to_dict()
        )

        # Known users from CF model — drives 8.2 gate
        # getattr guard handles any CF model that doesn't store known_users
        self._known_users: set = set(getattr(cf_rec, "known_users", []))

        # Pre-compute embedding matrix once — avoids rebuild on every API call
        # (product_vecs is {item_id: np.ndarray(768,)})
        self._vec_ids:    list       = list(product_vecs.keys())
        self._vec_matrix: np.ndarray = np.array(
            list(product_vecs.values()), dtype=np.float32
        )  # shape (N, 768)

        # Lightweight item metadata for decorating results
        self._item_meta: dict = self._build_meta(merge_df)

    # ── metadata ─────────────────────────────────────────────────────────────

    @staticmethod
    def _build_meta(df: pd.DataFrame) -> dict:
        cols = ["item_id"] + [
            c for c in ["title_meta", "main_category", "price"] if c in df.columns
        ]
        return (
            df[cols]
            .drop_duplicates("item_id")
            .set_index("item_id")
            .to_dict("index")
        )

    def _title(self, item_id: str) -> str:
        return self._item_meta.get(item_id, {}).get("title_meta", item_id)

    def _review_count(self, item_id: str) -> int:
        return self._review_counts.get(item_id, 0)

    # ── cold-start gates ──────────────────────────────────────────────────────

    def _is_new_user(self, user_id: str) -> bool:
        """8.2 — True when user_id has no CF interaction history."""
        return str(user_id) not in self._known_users

    def _is_new_item(self, item_id: str) -> bool:
        """8.3 — True when item has fewer than 3 reviews."""
        return self._review_count(item_id) < 3

    # ── embedding cosine similarity (used by both cold-start paths) ───────────

    def _semantic_similar(self, item_id: str, top_n: int) -> list:
        """
        Cosine similarity over product_vecs.
        Uses pre-computed _vec_matrix (built once in __init__) — safe for
        repeated API calls without per-request matrix reconstruction.
        Returns list[dict] {item_id, title, score, source}.
        """
        if item_id not in self.product_vecs:
            return []

        q   = self.product_vecs[item_id]          # (768,)
        ids = self._vec_ids                        # pre-computed in __init__
        mat = self._vec_matrix                     # pre-computed in __init__ (N, 768)

        # Normalise for cosine similarity
        qn   = q   / (np.linalg.norm(q)                              + 1e-9)
        mn   = mat / (np.linalg.norm(mat, axis=1, keepdims=True)     + 1e-9)
        sims = mn @ qn                             # (N,)

        top_i = np.argpartition(-sims, min(top_n + 1, len(sims) - 1))[: top_n + 1]
        top_i = top_i[np.argsort(-sims[top_i])]

        out = []
        for i in top_i:
            iid = ids[i]
            if iid == item_id:
                continue
            out.append({
                "item_id": iid,
                "title":   self._title(iid),
                "score":   round(float(sims[i]), 6),
                "source":  "semantic",
            })
            if len(out) == top_n:
                break
        return out

    # ── 8.2 cold-start: new user ──────────────────────────────────────────────

    def _cold_start_new_user(self, item_id: str, top_n: int) -> list:
        """
        8.2 — No CF signal available.
        Route to ProductRecommender + semantic embedding similarity only.
        """
        candidates = []

        # ProductRecommender (content-based scoring)
        for r in self.product_rec.get_recommendations(item_id, top_n=self.top_k):
            candidates.append({
                "item_id": str(r["item_id"]),
                "title":   self._title(str(r["item_id"])),
                "score":   float(r["score"]),
                "source":  "content_cold_start",
            })

        # Semantic embedding similarity
        for r in self._semantic_similar(item_id, top_n=self.top_k):
            candidates.append({**r, "source": "semantic_cold_start"})

        return self._dedup(candidates, top_n, exclude=item_id)

    # ── 8.3 cold-start: new item ──────────────────────────────────────────────

    def _cold_start_new_item(self, item_id: str, top_n: int) -> list:
        """
        8.3 — Fewer than 3 reviews; no reliable rating signal.
        Use product_text embedding similarity only.
        """
        results = self._semantic_similar(item_id, top_n=top_n)
        for r in results:
            r["source"] = "embedding_new_item"
        return results

    # ── warm-path engine calls ────────────────────────────────────────────────

    def _get_apriori(self, item_id: str) -> list:
        """Calls AprioriRecommender; returns list[dict] {item_id, score}."""
        recs = self.apriori_rec.recommend_apriori(item_id, top_k=self.top_k)
        out  = []
        for r in (recs or []):
            if isinstance(r, dict):
                iid   = str(r.get("item_id") or r.get("consequent") or "")
                score = float(r.get("lift", r.get("confidence", 1.0)))
            else:
                iid, score = str(r), 1.0
            if iid:
                out.append({"item_id": iid, "score": score, "source": "apriori"})
        return out

    def _get_cf(self, item_id: str) -> list:
        """
        Calls CollaborativeFilteringRecommender.
        Phase 5 recommend_products_cf() returns list[dict] {item_id, score}.
        """
        recs = self.cf_rec.recommend_products_cf(item_id, top_k=self.top_k)
        return [
            {
                "item_id": str(r["item_id"]),
                "score":   float(r["score"]),
                "source":  "cf",
            }
            for r in (recs or [])
        ]

    def _get_content(self, item_id: str) -> list:
        """
        Calls ProductRecommender.
        Phase 5 get_recommendations() returns list[dict] {item_id, score}.
        """
        recs = self.product_rec.get_recommendations(item_id, top_n=self.top_k)
        return [
            {
                "item_id": str(r["item_id"]),
                "score":   float(r["score"]),
                "source":  "content",
            }
            for r in (recs or [])
        ]

    # ── weighted score fusion ─────────────────────────────────────────────────

    def _fuse(
        self,
        apriori:  list,
        cf:       list,
        content:  list,
        exclude:  str,
        top_n:    int,
    ) -> list:
        """
        Weighted fusion: hybrid_score = w_cf*cf + w_content*content + w_apriori*apriori.
        Items appearing in only one engine keep their single weighted component.
        Dominant source tag added for transparency.
        """
        def _index(lst, weight):
            return {r["item_id"]: r["score"] * weight for r in lst}

        a_scores = _index(apriori, self.apriori_weight)
        c_scores = _index(cf,      self.cf_weight)
        p_scores = _index(content, self.content_weight)

        all_ids = (set(a_scores) | set(c_scores) | set(p_scores)) - {exclude}

        fused = []
        for iid in all_ids:
            hybrid_score = (
                a_scores.get(iid, 0.0)
                + c_scores.get(iid, 0.0)
                + p_scores.get(iid, 0.0)
            )
            by_src = {
                "apriori": a_scores.get(iid, 0.0),
                "cf":      c_scores.get(iid, 0.0),
                "content": p_scores.get(iid, 0.0),
            }
            dominant = max(by_src, key=by_src.get)
            fused.append({
                "item_id": iid,
                "title":   self._title(iid),
                "score":   round(hybrid_score, 6),
                "source":  f"hybrid:{dominant}",
            })

        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[:top_n]

    # ── deduplication helper ──────────────────────────────────────────────────

    @staticmethod
    def _dedup(candidates: list, top_n: int, exclude: str = "") -> list:
        """Keep highest score per item_id; sort descending; return top_n."""
        best = {}
        for c in candidates:
            iid = c["item_id"]
            if iid == exclude:
                continue
            if iid not in best or c["score"] > best[iid]["score"]:
                best[iid] = c
        return sorted(best.values(), key=lambda x: x["score"], reverse=True)[:top_n]

    # =========================================================================
    # 8.4  final_recommendation — main public API
    # =========================================================================

    def final_recommendation(
        self,
        item_id: str,
        user_id: str,
        top_n:   int = 10,
    ) -> list:
        """
        8.4 — Detects cold-start automatically; returns ranked list with
        per-result source tags.

        Returns:
            list[dict]: [{item_id, title, score, source}, ...]
        """
        item_id = str(item_id)
        user_id = str(user_id)

        # 8.2 — new user: no CF history
        if self._is_new_user(user_id):
            print(f"[Hybrid] NEW USER  user_id={user_id!r}")
            return self._cold_start_new_user(item_id, top_n)

        # 8.3 — new item: fewer than 3 reviews
        if self._is_new_item(item_id):
            print(f"[Hybrid] NEW ITEM  item_id={item_id!r}  "
                  f"reviews={self._review_count(item_id)}")
            return self._cold_start_new_item(item_id, top_n)

        # 8.1 — warm path: all three engines
        results = self._fuse(
            self._get_apriori(item_id),
            self._get_cf(item_id),
            self._get_content(item_id),
            exclude=item_id,
            top_n=top_n,
        )
        return results


# =============================================================================
# EVALUATION  (for 8.5 MLflow metrics)
# =============================================================================

def evaluate_hybrid(
    hybrid:   HybridRecommender,
    test_df:  pd.DataFrame,
    k:        int = 10,
    n_users:  int = 500,
) -> dict:
    """
    Evaluates the hybrid recommender on the Phase 6 test split.

    Evaluation protocol:
      - relevant = all items the user interacted with in test_df (ground truth)
      - query    = one item from relevant, excluded from the relevant set so the
                   recommender is not trivially scored on the seed item itself
      - Uses sorted(relevant)[0] for deterministic query selection across runs

    Returns Recall@k, NDCG@k, Precision@k.
    """
    test_df = test_df.copy()
    test_df["item_id"] = test_df["item_id"].astype(str)
    test_df["user_id"] = test_df["user_id"].astype(str)

    users = (
        test_df["user_id"]
        .drop_duplicates()
        .sample(min(n_users, test_df["user_id"].nunique()), random_state=42)
        .tolist()
    )

    R, N, P = [], [], []
    for uid in users:
        u_rows  = test_df[test_df["user_id"] == uid]
        all_items = set(u_rows["item_id"].tolist())

        if not all_items:
            continue

        # Deterministic query: sorted so results are reproducible across runs
        query = sorted(all_items)[0]

        if len(all_items) == 1:
            # Phase 6 temporal split: exactly 1 test item per user (last interaction).
            # Standard single-item leave-one-out: seed with that item,
            # then check if the recommender surfaces it in top-k.
            relevant = all_items          # {query}
        else:
            # Multiple test items: leave-one-out — exclude query from relevant
            relevant = all_items - {query}

        try:
            rec_ids = [
                r["item_id"]
                for r in hybrid.final_recommendation(query, uid, top_n=k)
            ]
        except Exception as e:
            print(f"[WARN] evaluate_hybrid user={uid}: {e}")
            rec_ids = []

        R.append(_recall(rec_ids,    relevant, k))
        N.append(_ndcg(rec_ids,      relevant, k))
        P.append(_precision(rec_ids, relevant, k))

    if not R:
        print("[WARN] evaluate_hybrid: no users evaluated — check test_df split.")
        return {f"recall_at_{k}": 0.0, f"ndcg_at_{k}": 0.0, f"precision_at_{k}": 0.0}

    return {
        f"recall_at_{k}":    round(float(np.mean(R)), 4),
        f"ndcg_at_{k}":      round(float(np.mean(N)), 4),
        f"precision_at_{k}": round(float(np.mean(P)), 4),
    }


# =============================================================================
# 8.5  MLflow logging
# =============================================================================

def log_hybrid_mlflow(hybrid: HybridRecommender, metrics: dict) -> None:
    """
    8.5 — One MLflow run named 'Hybrid' under experiment 'DS11'.
    Params : cf_weight, content_weight, apriori_weight, top_k_per_engine
    Metrics: recall_at_10, ndcg_at_10, precision_at_10
    Artifact: models/hybrid_recommender.pkl
    """
    with mlflow.start_run(run_name="Hybrid"):
        mlflow.log_param("cf_weight",        hybrid.cf_weight)
        mlflow.log_param("content_weight",   hybrid.content_weight)
        mlflow.log_param("apriori_weight",   hybrid.apriori_weight)
        mlflow.log_param("top_k_per_engine", hybrid.top_k)
        for metric_name, metric_val in metrics.items():
            mlflow.log_metric(metric_name, metric_val)
        mlflow.log_artifact(HYBRID_PKL)
    print(f"[MLflow] Hybrid run logged → {metrics}")


# =============================================================================
# 8.6  dill serialisation
# =============================================================================

def save_hybrid(hybrid: HybridRecommender, path: str = HYBRID_PKL) -> None:
    """
    8.6 — Serialize with dill (handles scipy sparse matrices and class
    references that standard pickle cannot always round-trip).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        dill.dump(hybrid, f)
    print(f"[dill]  Saved → {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")


def load_hybrid(path: str = HYBRID_PKL) -> HybridRecommender:
    """Load the serialised HybridRecommender — used by Phase 12 FastAPI."""
    with open(path, "rb") as f:
        return dill.load(f)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Phase 8 — Hybrid Engine + Cold-Start")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/6] Loading clean_merge_df …")
    df = pd.read_parquet(CLEAN_PARQUET)
    df["item_id"] = df["item_id"].astype(str)
    df["user_id"] = df["user_id"].astype(str)
    print(f"      rows={len(df):,}  "
          f"items={df['item_id'].nunique():,}  "
          f"users={df['user_id'].nunique():,}")

    # ── 2. Load product_vecs (Phase 7) ────────────────────────────────────────
    print("\n[2/6] Loading product_vecs.npz …")
    product_vecs = load_product_vecs()
    print(f"      {len(product_vecs):,} item vectors loaded")

    # ── 3. Load component models (Phase 4 + Phase 5) ──────────────────────────
    print("\n[3/6] Loading component recommenders …")
    apriori_rec = _load_pkl(APRIORI_PKL)
    product_rec = _load_pkl(PRODUCT_PKL)
    cf_rec      = _load_pkl(CF_PKL)
    print(f"      AprioriRecommender     ✓")
    print(f"      ProductRecommender     ✓")
    print(f"      CFRecommender          ✓  "
          f"known_users={len(cf_rec.known_users):,}")

    # ── 4. Build HybridRecommender (8.1) ──────────────────────────────────────
    print("\n[4/6] Building HybridRecommender …")
    hybrid = HybridRecommender(
        apriori_rec      = apriori_rec,
        cf_rec           = cf_rec,
        product_rec      = product_rec,
        product_vecs     = product_vecs,
        merge_df         = df,
        cf_weight        = 0.45,
        content_weight   = 0.35,
        apriori_weight   = 0.20,
        top_k_per_engine = 20,
    )
    print(f"      known_users={len(hybrid._known_users):,}  "
          f"tracked_items={len(hybrid._review_counts):,}")

    # ── 5. Smoke tests ────────────────────────────────────────────────────────
    print("\n[5/6] Smoke tests …")

    sample_item = df["item_id"].iloc[100]
    sample_user = df["user_id"].iloc[100]

    # Warm path (8.1)
    print(f"\n  [8.1] Warm path   item={sample_item!r}  user={sample_user!r}")
    for r in hybrid.final_recommendation(sample_item, sample_user, top_n=5):
        print(f"        [{r['source']:<22}] {r['item_id'][:20]:<22} score={r['score']:.4f}  {r['title'][:40]}")

    # Cold-start: new user (8.2)
    print("\n  [8.2] Cold-start NEW USER")
    for r in hybrid.final_recommendation(sample_item, "__NEW_USER__", top_n=5):
        print(f"        [{r['source']:<22}] {r['item_id'][:20]:<22} score={r['score']:.4f}")

    # Cold-start: new item (8.3) — inject a synthetic low-review item
    rare_item = "__RARE_ITEM_TEST__"
    hybrid._review_counts[rare_item] = 1
    ref_vec = next(iter(product_vecs.values()))
    hybrid.product_vecs[rare_item] = (
        ref_vec + np.random.randn(*ref_vec.shape).astype(np.float32) * 0.02
    )
    print(f"\n  [8.3] Cold-start NEW ITEM  (reviews=1)")
    for r in hybrid.final_recommendation(rare_item, sample_user, top_n=5):
        print(f"        [{r['source']:<22}] {r['item_id'][:20]:<22} score={r['score']:.4f}")
    # Remove synthetic item
    hybrid._review_counts.pop(rare_item)
    hybrid.product_vecs.pop(rare_item)

    # ── 6. Evaluate → save → MLflow ───────────────────────────────────────────
    print("\n[6/6] Evaluate → Save (8.6) → MLflow (8.5) …")

    if os.path.exists(TEST_PARQUET):
        print("      Evaluating on test_df.parquet …")
        test_df = pd.read_parquet(TEST_PARQUET)
        metrics = evaluate_hybrid(hybrid, test_df, k=10, n_users=500)
    else:
        print("      [WARN] test_df.parquet not found — run Phase 6 first.")
        print("             Using zero placeholder metrics for MLflow log.")
        metrics = {"recall_at_10": 0.0, "ndcg_at_10": 0.0, "precision_at_10": 0.0}

    print(f"      Metrics: {metrics}")

    save_hybrid(hybrid, HYBRID_PKL)         # 8.6
    log_hybrid_mlflow(hybrid, metrics)      # 8.5

    print(f"\n{'=' * 60}")
    print("✓  Phase 8 complete.")
    print(f"   Model  : {HYBRID_PKL}")
    print(f"   MLflow : {MLFLOW_URI}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()