"""
Phase 8 — Hybrid Engine + Cold-Start
File: src/08_hybrid_engine.py
"""

import os
import json
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import dill
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

CLEAN_PARQUET    = os.path.join(DATA_DIR,   "clean_merge_df.parquet")
TEST_PARQUET     = os.path.join(DATA_DIR,   "test_df.parquet")
TRAIN_PARQUET    = os.path.join(DATA_DIR,   "train_df.parquet")        # FIX: added
APRIORI_PKL      = os.path.join(MODELS_DIR, "apriori_recommender.pkl")
PRODUCT_PKL      = os.path.join(MODELS_DIR, "product_recommender.pkl")
CF_PKL           = os.path.join(MODELS_DIR, "cf_recommender.pkl")
PRODUCT_VECS_NPZ = os.path.join(EMBED_DIR,  "product_vecs.npz")
HYBRID_PKL       = os.path.join(MODELS_DIR, "hybrid_recommender.pkl")
USER_ITEM_NPZ    = os.path.join(MODELS_DIR, "user_item_matrix.npz")    # FIX: added

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import importlib
_mod04 = importlib.import_module("04_apriori_recommender")
AprioriRecommender = _mod04.AprioriRecommender

_mod05 = importlib.import_module("05_content_cf_recommender")
ProductRecommender                = _mod05.ProductRecommender
CollaborativeFilteringRecommender = _mod05.CollaborativeFilteringRecommender

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("DS11")


# =============================================================================
# LOADERS
# =============================================================================

def _load_pkl(path: str):
    with open(path, "rb") as f:
        return dill.load(f)


def load_product_vecs() -> dict:
    data = np.load(PRODUCT_VECS_NPZ, allow_pickle=True)
    ids  = [str(i) for i in data["keys"].tolist()]
    vecs = data["vecs"].astype(np.float32)
    print(f"[load] product_vecs: {len(ids):,} items  shape={vecs.shape}")
    return {iid: vecs[i] for i, iid in enumerate(ids)}


# =============================================================================
# METRIC HELPERS
# =============================================================================

def _recall(rec, rel, k):
    return sum(1 for r in rec[:k] if r in rel) / len(rel) if rel else 0.0

def _ndcg(rec, rel, k):
    dcg  = sum(1.0 / np.log2(i + 2) for i, r in enumerate(rec[:k]) if r in rel)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg if idcg else 0.0

def _precision(rec, rel, k):
    return sum(1 for r in rec[:k] if r in rel) / k if k else 0.0


# =============================================================================
# 8.1  HybridRecommender
# =============================================================================

class HybridRecommender:

    def __init__(
        self,
        apriori_rec,
        cf_rec,
        product_rec,
        product_vecs:      dict,
        merge_df:          pd.DataFrame,
        user_item_matrix,                   # FIX: store explicitly
        cf_weight:         float = 0.45,
        content_weight:    float = 0.35,
        apriori_weight:    float = 0.20,
        top_k_per_engine:  int   = 20,
        new_item_threshold: int  = 1,       # FIX: lowered from 3 to 1
    ):
        self.apriori_rec      = apriori_rec
        self.cf_rec           = cf_rec
        self.product_rec      = product_rec
        self.product_vecs     = product_vecs
        self.cf_weight        = cf_weight
        self.content_weight   = content_weight
        self.apriori_weight   = apriori_weight
        self.top_k            = top_k_per_engine
        self.new_item_threshold = new_item_threshold

        # FIX: store user_item_matrix for Phase 10 ALS
        self.user_item_matrix = user_item_matrix

        # Review counts per item_id
        self._review_counts: dict = (
            merge_df.groupby("item_id")["rating"].count().to_dict()
        )

        # FIX: use cf_rec.user_map (not known_users — confirmed attribute name)
        self._known_users: set = set(cf_rec.user_map.keys())

        # FIX: store user_to_idx and idx_to_item explicitly for Phase 10
        self.user_to_idx = cf_rec.user_map        # user_id -> idx
        self.idx_to_item = cf_rec.idx_to_item     # idx -> item_id

        # Pre-compute embedding matrix
        self._vec_ids:    list       = list(product_vecs.keys())
        self._vec_matrix: np.ndarray = np.array(
            list(product_vecs.values()), dtype=np.float32
        )

        # Item metadata
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
        # FIX: check _known_users which is now populated from cf_rec.user_map
        return str(user_id) not in self._known_users

    def _is_new_item(self, item_id: str) -> bool:
        """8.3 — True when item has fewer than new_item_threshold reviews."""
        # FIX: use configurable threshold (default 1 instead of 3)
        return self._review_count(item_id) < self.new_item_threshold

    # ── get seen items for a user (for filtering already seen) ───────────────

    def _get_seen_items(self, user_id: str) -> set:
        """
        FIX: Returns set of item_ids already seen by user.
        Used to filter out already seen items from recommendations.
        """
        if user_id not in self.user_to_idx:
            return set()
        user_idx = self.user_to_idx[user_id]
        if self.user_item_matrix is None:
            return set()
        row = self.user_item_matrix[user_idx]
        seen_indices = row.indices.tolist()
        return {
            self.idx_to_item[i]
            for i in seen_indices
            if i in self.idx_to_item
        }

    # ── embedding cosine similarity ───────────────────────────────────────────

    def _semantic_similar(self, item_id: str, top_n: int) -> list:
        if item_id not in self.product_vecs:
            return []
        q   = self.product_vecs[item_id]
        ids = self._vec_ids
        mat = self._vec_matrix
        qn  = q   / (np.linalg.norm(q) + 1e-9)
        mn  = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
        sims = mn @ qn
        top_i = np.argpartition(-sims, min(top_n + 1, len(sims) - 1))[:top_n + 1]
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

    # ── cold-start paths ──────────────────────────────────────────────────────

    def _cold_start_new_user(self, item_id: str, top_n: int) -> list:
        candidates = []
        for r in self.product_rec.get_recommendations(item_id, top_n=self.top_k):
            candidates.append({
                "item_id": str(r["item_id"]),
                "title":   self._title(str(r["item_id"])),
                "score":   float(r["score"]),
                "source":  "content_cold_start",
            })
        for r in self._semantic_similar(item_id, top_n=self.top_k):
            candidates.append({**r, "source": "semantic_cold_start"})
        return self._dedup(candidates, top_n, exclude=item_id)

    def _cold_start_new_item(self, item_id: str, top_n: int) -> list:
        results = self._semantic_similar(item_id, top_n=top_n)
        for r in results:
            r["source"] = "embedding_new_item"
        return results

    # ── warm-path engine calls ────────────────────────────────────────────────

    def _get_apriori(self, item_id: str) -> list:
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
        recs = self.cf_rec.recommend_products_cf(item_id, top_k=self.top_k)
        return [
            {"item_id": str(r["item_id"]), "score": float(r["score"]), "source": "cf"}
            for r in (recs or [])
        ]

    def _get_content(self, item_id: str) -> list:
        recs = self.product_rec.get_recommendations(item_id, top_n=self.top_k)
        return [
            {"item_id": str(r["item_id"]), "score": float(r["score"]), "source": "content"}
            for r in (recs or [])
        ]

    # ── weighted score fusion ─────────────────────────────────────────────────

    def _fuse(self, apriori, cf, content, exclude, top_n):
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
    def _dedup(candidates, top_n, exclude=""):
        best = {}
        for c in candidates:
            iid = c["item_id"]
            if iid == exclude:
                continue
            if iid not in best or c["score"] > best[iid]["score"]:
                best[iid] = c
        return sorted(best.values(), key=lambda x: x["score"], reverse=True)[:top_n]

    # =========================================================================
    # 8.4  final_recommendation
    # =========================================================================

    def final_recommendation(
        self,
        item_id: str,
        user_id: str,
        top_n:   int = 10,
    ) -> list:
        item_id = str(item_id)
        user_id = str(user_id)

        # 8.2 — new user
        if self._is_new_user(user_id):
            return self._cold_start_new_user(item_id, top_n)

        # 8.3 — new item
        if self._is_new_item(item_id):
            return self._cold_start_new_item(item_id, top_n)

        # FIX: get seen items to exclude from recommendations
        seen_items = self._get_seen_items(user_id)

        # 8.1 — warm path
        results = self._fuse(
            self._get_apriori(item_id),
            self._get_cf(item_id),
            self._get_content(item_id),
            exclude=item_id,
            top_n=top_n + len(seen_items),   # over-fetch then filter
        )

        # FIX: filter out already seen items
        results = [r for r in results if r["item_id"] not in seen_items]
        return results[:top_n]


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_hybrid(
    hybrid:   HybridRecommender,
    test_df:  pd.DataFrame,
    train_df: pd.DataFrame,         # FIX: added train_df for seed selection
    k:        int = 10,
    n_users:  int = 500,
) -> dict:
    """
    FIX: Correct evaluation protocol.
    - seed  = last item from user's training history (input != ground truth)
    - ground truth = item in test_df (last interaction overall)
    """
    test_df  = test_df.copy()
    train_df = train_df.copy()

    test_df["item_id"]  = test_df["item_id"].astype(str)
    test_df["user_id"]  = test_df["user_id"].astype(str)
    train_df["item_id"] = train_df["item_id"].astype(str)
    train_df["user_id"] = train_df["user_id"].astype(str)

    # Build seed lookup: user_id -> last training item
    seed_lookup = (
        train_df.sort_values("timestamp")
        .groupby("user_id")["item_id"]
        .last()
        .to_dict()
    )

    users = (
        test_df["user_id"]
        .drop_duplicates()
        .sample(min(n_users, test_df["user_id"].nunique()), random_state=42)
        .tolist()
    )

    R, N, P = [], [], []
    for uid in users:
        # Ground truth = test item
        u_rows   = test_df[test_df["user_id"] == uid]
        relevant = set(u_rows["item_id"].tolist())

        if not relevant:
            continue

        # FIX: seed from training history, not test item
        seed = seed_lookup.get(uid)
        if seed is None:
            continue

        try:
            rec_ids = [
                r["item_id"]
                for r in hybrid.final_recommendation(seed, uid, top_n=k)
            ]
        except Exception as e:
            print(f"[WARN] evaluate_hybrid user={uid}: {e}")
            rec_ids = []

        R.append(_recall(rec_ids,    relevant, k))
        N.append(_ndcg(rec_ids,      relevant, k))
        P.append(_precision(rec_ids, relevant, k))

    if not R:
        print("[WARN] evaluate_hybrid: no users evaluated.")
        return {f"recall_at_{k}": 0.0, f"ndcg_at_{k}": 0.0, f"precision_at_{k}": 0.0}

    return {
        f"recall_at_{k}":    round(float(np.mean(R)), 4),
        f"ndcg_at_{k}":      round(float(np.mean(N)), 4),
        f"precision_at_{k}": round(float(np.mean(P)), 4),
    }


# =============================================================================
# 8.5 + 8.6
# =============================================================================

def log_hybrid_mlflow(hybrid, metrics):
    with mlflow.start_run(run_name="Hybrid"):
        mlflow.log_param("cf_weight",         hybrid.cf_weight)
        mlflow.log_param("content_weight",    hybrid.content_weight)
        mlflow.log_param("apriori_weight",    hybrid.apriori_weight)
        mlflow.log_param("top_k_per_engine",  hybrid.top_k)
        mlflow.log_param("new_item_threshold", hybrid.new_item_threshold)
        for name, val in metrics.items():
            mlflow.log_metric(name, val)
        mlflow.log_artifact(HYBRID_PKL)
    print(f"[MLflow] Hybrid run logged → {metrics}")


def save_hybrid(hybrid, path=HYBRID_PKL):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        dill.dump(hybrid, f)
    print(f"[dill] Saved → {path}  ({os.path.getsize(path)/1e6:.1f} MB)")


def load_hybrid(path=HYBRID_PKL):
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
    print("\n[1/6] Loading data …")
    df       = pd.read_parquet(CLEAN_PARQUET)
    train_df = pd.read_parquet(TRAIN_PARQUET)   # FIX: load train_df

    df["item_id"]       = df["item_id"].astype(str)
    df["user_id"]       = df["user_id"].astype(str)
    train_df["item_id"] = train_df["item_id"].astype(str)
    train_df["user_id"] = train_df["user_id"].astype(str)

    print(f"  merge_df : {len(df):,} rows")
    print(f"  train_df : {len(train_df):,} rows")

    # ── 2. Load product_vecs ──────────────────────────────────────────────────
    print("\n[2/6] Loading product_vecs.npz …")
    product_vecs = load_product_vecs()

    # ── 3. Load component models ──────────────────────────────────────────────
    print("\n[3/6] Loading component recommenders …")
    apriori_rec = _load_pkl(APRIORI_PKL)
    product_rec = _load_pkl(PRODUCT_PKL)
    cf_rec      = _load_pkl(CF_PKL)

    # FIX: inject train_df into product_rec for seed selection in Phase 10
    if hasattr(product_rec, "set_user_history"):
        product_rec.set_user_history(train_df)
        print(f"  ProductRecommender user_history injected ✓")

    print(f"  AprioriRecommender  ✓")
    print(f"  ProductRecommender  ✓")
    print(f"  CFRecommender       ✓  known_users={len(cf_rec.user_map):,}")

    # ── FIX: load or build user_item_matrix ───────────────────────────────────
    # ── FIX Option 2: Phase 8 builds and saves user_item_matrix ──────────────────
    print("\n[3b] Building user_item_matrix from train_df …")

    with open(os.path.join(DATA_DIR, "user_map.json"), "r") as f:
        user_map = {k: int(v) for k, v in json.load(f).items()}
    with open(os.path.join(DATA_DIR, "item_map.json"), "r") as f:
        item_map = {k: int(v) for k, v in json.load(f).items()}

    train_aligned = train_df[
        train_df["user_id"].isin(user_map) &
        train_df["item_id"].isin(item_map)
    ].copy()

    train_aligned["user_idx"] = train_aligned["user_id"].map(user_map).astype(int)
    train_aligned["item_idx"] = train_aligned["item_id"].map(item_map).astype(int)
    train_aligned["confidence"] = train_aligned["rating"].astype(np.float32) / 5.0

    rows = train_aligned["item_idx"].values
    cols = train_aligned["user_idx"].values
    data = train_aligned["confidence"].astype(np.float32).values

    n_users = len(user_map)
    n_items = len(item_map)

    item_user_matrix = sp.csr_matrix((data, (rows, cols)), shape=(n_items, n_users))
    user_item_matrix = item_user_matrix.T.tocsr()

    # Save for Phase 9 to load directly
    sp.save_npz(USER_ITEM_NPZ, user_item_matrix)
    print(f"  user_item_matrix: {user_item_matrix.shape}  nnz={user_item_matrix.nnz}")
    print(f"  Saved → {USER_ITEM_NPZ}")

    # ── 4. Build HybridRecommender ────────────────────────────────────────────
    print("\n[4/6] Building HybridRecommender …")
    hybrid = HybridRecommender(
        apriori_rec       = apriori_rec,
        cf_rec            = cf_rec,
        product_rec       = product_rec,
        product_vecs      = product_vecs,
        merge_df          = df,
        user_item_matrix  = user_item_matrix,   # FIX: pass matrix
        cf_weight         = 0.45,
        content_weight    = 0.35,
        apriori_weight    = 0.20,
        top_k_per_engine  = 20,
        new_item_threshold = 1,                 # FIX: lowered threshold
    )
    print(f"  known_users={len(hybrid._known_users):,}  "
          f"tracked_items={len(hybrid._review_counts):,}")

    # ── 5. Smoke tests ────────────────────────────────────────────────────────
    print("\n[5/6] Smoke tests …")
    sample_item = df["item_id"].iloc[100]
    sample_user = df["user_id"].iloc[100]

    print(f"\n  [8.1] Warm path  item={sample_item!r}  user={sample_user!r}")
    for r in hybrid.final_recommendation(sample_item, sample_user, top_n=5):
        print(f"    [{r['source']:<22}] {r['item_id']:<22} score={r['score']:.4f}")

    print("\n  [8.2] Cold-start NEW USER")
    for r in hybrid.final_recommendation(sample_item, "__NEW_USER__", top_n=5):
        print(f"    [{r['source']:<22}] {r['item_id']:<22} score={r['score']:.4f}")

    rare_item = "__RARE_ITEM_TEST__"
    hybrid._review_counts[rare_item] = 0
    ref_vec = next(iter(product_vecs.values()))
    hybrid.product_vecs[rare_item] = (
        ref_vec + np.random.randn(*ref_vec.shape).astype(np.float32) * 0.02
    )
    print(f"\n  [8.3] Cold-start NEW ITEM (reviews=0)")
    for r in hybrid.final_recommendation(rare_item, sample_user, top_n=5):
        print(f"    [{r['source']:<22}] {r['item_id']:<22} score={r['score']:.4f}")
    hybrid._review_counts.pop(rare_item)
    hybrid.product_vecs.pop(rare_item)

    # ── 6. Evaluate → Save → MLflow ───────────────────────────────────────────
    print("\n[6/6] Evaluate → Save → MLflow …")

    if os.path.exists(TEST_PARQUET) and os.path.exists(TRAIN_PARQUET):
        test_df = pd.read_parquet(TEST_PARQUET)
        # FIX: pass train_df to evaluate_hybrid
        metrics = evaluate_hybrid(hybrid, test_df, train_df, k=10, n_users=500)
    else:
        print("  [WARN] test_df or train_df not found — using zero metrics.")
        metrics = {"recall_at_10": 0.0, "ndcg_at_10": 0.0, "precision_at_10": 0.0}

    print(f"  Metrics: {metrics}")

    save_hybrid(hybrid, HYBRID_PKL)
    log_hybrid_mlflow(hybrid, metrics)

    print(f"\n{'=' * 60}")
    print("✓  Phase 8 complete.")
    print(f"   Model  : {HYBRID_PKL}")
    print(f"   MLflow : {MLFLOW_URI}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()