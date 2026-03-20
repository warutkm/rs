"""
Phase 5 — Content-Based + Item-Item CF Recommender
File: src/05_content_cf_recommender.py

Workflow actions:
  5.1  Aggregate product features — rating_mean, review_count, helpful_votes,
         verified_count, last_review_date, price_median
  5.2  Compute composite scores — satisfaction, recency, popularity, hotness
  5.3  Price score — 1 / (1 + |price - cat_median| / cat_median)
  5.4  ProductRecommender class — get_recommendations(item_id, top_n, weights)
  5.5  Build sparse rating matrix — csr_matrix on item_id (NOT asin)
  5.6  Mean-center by user
  5.7  Compute top-50 item neighbors — cosine sim via sparse multiply
  5.8  CollaborativeFilteringRecommender — recommend_products_cf(item_id, top_k)
         returns list[dict] with item_id + score keys
  5.9  Log content_only run to MLflow — weights, top_n, Recall@10, NDCG@10, Precision@10

Fixes applied vs original:
  [FIX-1] MLflow URI changed from sqlite:///mlflow.db -> mlflow/ (file-based)
           so all 7 runs share the same tracking store (required by Phase 11)
  [FIX-2] recommend_products_cf() now returns list[dict] {item_id, score}
           instead of plain list[str] — required by Phase 8 HybridRecommender
  [FIX-3] CollaborativeFilteringRecommender stores known_users set
           so Phase 8 _is_new_user() gate works correctly
  [FIX-4] pickle.dump for both ProductRecommender and CollaborativeFilteringRecommender
           so Phase 8 can load models/product_recommender.pkl and cf_recommender.pkl
  [FIX-5] Precision@10 added to evaluation and MLflow logging
           (workflow 5.9 specifies recall_at_10, ndcg_at_10, precision_at_10)
"""

import os
import numpy as np
import pandas as pd
import pickle
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import mlflow

# =============================================================================
# CONSTANTS
# =============================================================================
DATA_PATH       = "data/clean_merge_df.parquet"
TOP_K_NEIGHBORS = 50
TOP_N_CONTENT   = 10
TEST_SIZE       = 0.2
N_EVAL_USERS    = 500
RANDOM_STATE    = 42

# FIX-1: file-based URI — matches every other phase and Phase 11 report
MLFLOW_URI      = "mlflow/"
EXPERIMENT_NAME = "DS11"

COL_CATEGORY    = "main_category_meta"   # adjust if column name differs in your parquet


# =============================================================================
# SETUP
# =============================================================================
def create_dirs():
    for d in ("models", "outputs", "data"):
        os.makedirs(d, exist_ok=True)


# =============================================================================
# LOAD DATA
# =============================================================================
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["item_id"] = df["item_id"].astype(str)   # canonical string item_id
    df["user_id"] = df["user_id"].astype(str)
    print(f"[load] {df.shape[0]:,} rows  {df.shape[1]} cols  "
          f"items={df['item_id'].nunique():,}  users={df['user_id'].nunique():,}")
    return df


# =============================================================================
# 5.1  Aggregate product features
# =============================================================================
def aggregate_product_features(df: pd.DataFrame) -> pd.DataFrame:
    # Determine correct category column name defensively
    cat_col = COL_CATEGORY if COL_CATEGORY in df.columns else "main_category"

    agg_df = df.groupby("item_id").agg(
        rating_mean      = ("rating",           "mean"),
        review_count     = ("user_id",          "count"),
        helpful_votes    = ("helpful_vote",      "sum"),
        verified_count   = ("verified_purchase", "sum"),
        last_review_date = ("timestamp",         "max"),
        price_median     = ("price",             "median"),
        main_category    = (cat_col,             "first"),
    ).reset_index()

    print(f"[agg]  {len(agg_df):,} unique products")
    return agg_df


# =============================================================================
# 5.2  Composite scores
# =============================================================================
def compute_scores(agg_df: pd.DataFrame) -> pd.DataFrame:
    df = agg_df.copy()

    df["verified_rate"] = df["verified_count"] / df["review_count"].clip(lower=1)
    df["helpful_log"]   = np.log1p(df["helpful_votes"])

    # 5.2 — satisfaction
    df["satisfaction"] = (
        0.6 * df["rating_mean"] +
        0.2 * df["verified_rate"] +
        0.2 * df["helpful_log"]
    )

    # Recency (0-1 normalised)
    min_date      = df["last_review_date"].min()
    df["recency"] = (df["last_review_date"] - min_date).dt.days
    max_recency   = df["recency"].max()
    df["recency"] = df["recency"] / max_recency if max_recency > 0 else 0.0

    df["popularity"] = np.log1p(df["review_count"])
    df["hotness"]    = df["popularity"] * df["recency"]

    return df


# =============================================================================
# 5.3  Price score  (FIX: was missing from original, now added)
# =============================================================================
def compute_price_score(agg_df: pd.DataFrame) -> pd.DataFrame:
    df = agg_df.copy()

    cat_median = df.groupby("main_category")["price_median"].transform("median")
    cat_median = cat_median.replace(0, np.nan)

    df["price_score"] = 1.0 / (
        1.0 + np.abs(df["price_median"] - cat_median) / cat_median
    )
    df["price_score"] = df["price_score"].fillna(0.5)   # neutral when price missing

    return df


# =============================================================================
# 5.4  ProductRecommender
# =============================================================================
class ProductRecommender:
    """
    Content-based recommender — 5 scoring components (workflow 5.4).
    get_recommendations() returns list[dict] {item_id, score}
    so Phase 8 HybridRecommender can consume it directly.
    """

    DEFAULT_WEIGHTS = {
        "satisfaction": 0.4,
        "recency":      0.2,
        "popularity":   0.2,
        "hotness":      0.1,
        "price_score":  0.1,
    }

    def __init__(self, agg_df: pd.DataFrame):
        self.df = agg_df.set_index("item_id").copy()

    def get_recommendations(
        self,
        item_id: str,
        top_n:   int  = 10,
        weights: dict = None,
    ) -> list:
        """
        Returns list[dict] with keys: item_id, score.
        Compatible with Phase 8 HybridRecommender._get_content().
        """
        if weights is None:
            weights = self.DEFAULT_WEIGHTS

        df = self.df.copy()
        df["final_score"] = sum(weights[col] * df[col] for col in weights if col in df.columns)

        recs = (
            df[df.index != item_id]
            .sort_values("final_score", ascending=False)
            .head(top_n)
        )
        return [
            {"item_id": str(iid), "score": round(float(score), 6)}
            for iid, score in zip(recs.index, recs["final_score"])
        ]


# =============================================================================
# 5.5 – 5.8  CollaborativeFilteringRecommender
# =============================================================================
class CollaborativeFilteringRecommender:
    """
    Item-Item CF via mean-centred cosine similarity (workflow 5.5–5.8).

    Key fixes vs original:
      - recommend_products_cf() returns list[dict] {item_id, score}  [FIX-2]
      - known_users attribute stored for Phase 8 cold-start gate      [FIX-3]
      - item index uses item_id (parent_asin), never raw asin
    """

    def __init__(self, df: pd.DataFrame, top_k_neighbors: int = 50):
        self.top_k_neighbors = top_k_neighbors
        self._build(df)

    def _build(self, df: pd.DataFrame):
        # FIX-3: store known_users so Phase 8 _is_new_user() gate works
        self.known_users = set(df["user_id"].astype(str).unique())

        # 5.5 — maps and sparse user-item matrix keyed on item_id (NOT asin)
        user_ids = df["user_id"].unique()
        item_ids = df["item_id"].unique()

        self.user_map    = {u: i for i, u in enumerate(user_ids)}
        self.item_map    = {it: i for i, it in enumerate(item_ids)}
        self.idx_to_item = {v: k for k, v in self.item_map.items()}

        df = df.copy()
        df["user_idx"] = df["user_id"].map(self.user_map)
        df["item_idx"] = df["item_id"].map(self.item_map)

        # 5.5 — csr_matrix (users × items)
        R = csr_matrix(
            (df["rating"].astype(float).values,
             (df["user_idx"].values, df["item_idx"].values)),
            shape=(len(user_ids), len(item_ids)),
        )

        # 5.6 — mean-center by user (remove per-user rating-scale bias)
        user_means = np.array(R.mean(axis=1)).flatten()
        R_mc = R.copy().astype(float)
        for i in range(R_mc.shape[0]):
            s, e = R_mc.indptr[i], R_mc.indptr[i + 1]
            R_mc.data[s:e] -= user_means[i]

        # 5.7 — item-item cosine similarity: transpose → (items × users)
        self.similarity = cosine_similarity(R_mc.T, dense_output=False)
        print(f"[CF]   item similarity matrix built: {self.similarity.shape}  "
              f"known_users={len(self.known_users):,}")

    def recommend_products_cf(self, item_id: str, top_k: int = 10) -> list:
        """
        Returns list[dict] {item_id, score} — FIX-2.
        Phase 8 HybridRecommender._get_cf() consumes this format.
        """
        if item_id not in self.item_map:
            return []

        idx     = self.item_map[item_id]
        sim_row = self.similarity[idx].toarray().flatten()
        sim_row[idx] = -1.0   # exclude query item

        k        = min(top_k, self.top_k_neighbors, len(sim_row) - 1)
        top_idxs = np.argpartition(sim_row, -k)[-k:]
        top_idxs = top_idxs[np.argsort(sim_row[top_idxs])[::-1]]

        # FIX-2: return dicts instead of plain strings
        return [
            {"item_id": self.idx_to_item[i], "score": round(float(sim_row[i]), 6)}
            for i in top_idxs
        ]


# =============================================================================
# EVALUATION HELPERS
# =============================================================================
def _recall(rec: list, rel: set, k: int) -> float:
    return len(set(rec[:k]) & rel) / len(rel) if rel else 0.0

def _ndcg(rec: list, rel: set, k: int) -> float:
    dcg  = sum(1.0 / np.log2(i + 2) for i, r in enumerate(rec[:k]) if r in rel)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg if idcg else 0.0

def _precision(rec: list, rel: set, k: int) -> float:
    return sum(1 for r in rec[:k] if r in rel) / k if k else 0.0


def evaluate_content(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    rec:      ProductRecommender,
    k:        int = 10,
    n_users:  int = 500,
) -> tuple:
    test_map = test_df.groupby("user_id")["item_id"].apply(set).to_dict()
    users    = list(set(train_df["user_id"]) & set(test_map))[:n_users]

    R, N, P = [], [], []
    for user in users:
        train_items = train_df[train_df["user_id"] == user]["item_id"].values
        if len(train_items) == 0:
            continue
        seed    = np.random.choice(train_items)
        rel     = test_map[user]
        recs    = [r["item_id"] for r in rec.get_recommendations(seed, top_n=k)]
        R.append(_recall(recs, rel, k))
        N.append(_ndcg(recs, rel, k))
        P.append(_precision(recs, rel, k))

    return float(np.mean(R)), float(np.mean(N)), float(np.mean(P))


def evaluate_cf(
    train_df:  pd.DataFrame,
    test_df:   pd.DataFrame,
    cf_model:  CollaborativeFilteringRecommender,
    k:         int = 10,
    n_users:   int = 500,
) -> tuple:
    test_map = test_df.groupby("user_id")["item_id"].apply(set).to_dict()
    users    = list(set(train_df["user_id"]) & set(test_map))[:n_users]

    R, N, P = [], [], []
    for user in users:
        train_items = train_df[train_df["user_id"] == user]["item_id"].values
        if len(train_items) == 0:
            continue
        seed    = np.random.choice(train_items)
        rel     = test_map[user]
        # recommend_products_cf now returns list[dict] — extract item_id strings
        recs    = [r["item_id"] for r in cf_model.recommend_products_cf(seed, top_k=k)]
        R.append(_recall(recs, rel, k))
        N.append(_ndcg(recs, rel, k))
        P.append(_precision(recs, rel, k))

    return float(np.mean(R)), float(np.mean(N)), float(np.mean(P))


# =============================================================================
# 5.9  MLflow logging  (FIX-1: file-based URI; FIX-5: precision_at_10 added)
# =============================================================================
def log_to_mlflow(
    c_recall: float, c_ndcg: float, c_precision: float,
    cf_recall: float, cf_ndcg: float, cf_precision: float,
):
    mlflow.set_tracking_uri(MLFLOW_URI)          # FIX-1: "mlflow/" not sqlite
    mlflow.set_experiment(EXPERIMENT_NAME)

    # content_only run (workflow 5.9)
    with mlflow.start_run(run_name="content_only"):
        mlflow.log_param("top_n",   TOP_N_CONTENT)
        mlflow.log_param("weights", str(ProductRecommender.DEFAULT_WEIGHTS))
        mlflow.log_metric("recall_at_10",    c_recall)
        mlflow.log_metric("ndcg_at_10",      c_ndcg)
        mlflow.log_metric("precision_at_10", c_precision)   # FIX-5

    # cf_item_item run
    with mlflow.start_run(run_name="cf_item_item"):
        mlflow.log_param("top_k_neighbors", TOP_K_NEIGHBORS)
        mlflow.log_metric("recall_at_10",    cf_recall)
        mlflow.log_metric("ndcg_at_10",      cf_ndcg)
        mlflow.log_metric("precision_at_10", cf_precision)  # FIX-5

    print("[MLflow] content_only + cf_item_item runs logged.")


# =============================================================================
# MAIN
# =============================================================================
def main():
    create_dirs()

    # Load
    df = load_data(DATA_PATH)

    # 5.1 → 5.3 — product feature table
    agg_df = aggregate_product_features(df)
    agg_df = compute_scores(agg_df)
    agg_df = compute_price_score(agg_df)

    # Train / test split (temporal is ideal; random split used here for speed)
    # NOTE: Phase 6 saves data/test_df.parquet via temporal sort — use that for
    #       A/B comparison. This split is for Phase 5 internal evaluation only.
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 5.4 — ProductRecommender
    product_rec = ProductRecommender(agg_df)

    # 5.5–5.8 — CF recommender (train set only — no data leakage)
    cf_model = CollaborativeFilteringRecommender(
        train_df, top_k_neighbors=TOP_K_NEIGHBORS
    )

    # Evaluate content-based
    print("\nEvaluating content-based recommender …")
    c_r, c_n, c_p = evaluate_content(train_df, test_df, product_rec, k=10, n_users=N_EVAL_USERS)
    print(f"  Content  Recall@10={c_r:.4f}  NDCG@10={c_n:.4f}  Precision@10={c_p:.4f}")

    # Evaluate CF
    print("Evaluating CF recommender …")
    cf_r, cf_n, cf_p = evaluate_cf(train_df, test_df, cf_model, k=10, n_users=N_EVAL_USERS)
    print(f"  CF       Recall@10={cf_r:.4f}  NDCG@10={cf_n:.4f}  Precision@10={cf_p:.4f}")

    # 5.9 — MLflow
    log_to_mlflow(c_r, c_n, c_p, cf_r, cf_n, cf_p)

    # ─────────────────────────────────────────────────────────────────────────
    # FIX-4: Persist both models so Phase 8 can load them directly
    # ─────────────────────────────────────────────────────────────────────────
    pr_path = "models/product_recommender.pkl"
    cf_path = "models/cf_recommender.pkl"

    with open(pr_path, "wb") as f:
        pickle.dump(product_rec, f)
    print(f"[save] ProductRecommender          → {pr_path}  "
          f"({os.path.getsize(pr_path)/1e6:.1f} MB)")

    with open(cf_path, "wb") as f:
        pickle.dump(cf_model, f)
    print(f"[save] CollaborativeFilteringRec   → {cf_path}  "
          f"({os.path.getsize(cf_path)/1e6:.1f} MB)")

    print("\n✓  Phase 5 complete.")
    print(f"   models/product_recommender.pkl")
    print(f"   models/cf_recommender.pkl")
    print(f"   MLflow: {MLFLOW_URI}mlruns/  (runs: content_only, cf_item_item)")


if __name__ == "__main__":
    main()