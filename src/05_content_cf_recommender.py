import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import mlflow

# =========================
# CONSTANTS
# =========================
DATA_PATH        = "data/clean_merge_df.parquet"
TOP_K_NEIGHBORS  = 50
TOP_N_CONTENT    = 10
TEST_SIZE        = 0.2
N_EVAL_USERS     = 500
RANDOM_STATE     = 42
MLFLOW_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME  = "DS11"
COL_CATEGORY = "main_category_meta"

# =========================
# CREATE DIRS
# =========================
def create_dirs():
    os.makedirs("models",  exist_ok=True)
    os.makedirs("outputs", exist_ok=True)


# =========================
# LOAD DATA
# =========================
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"Loaded data: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df

# =========================
# 5.1 AGGREGATE PRODUCT FEATURES
# Per item_id: rating_mean, review_count, helpful_votes,
# verified_count, last_review_date, price_median
# =========================
def aggregate_product_features(df: pd.DataFrame) -> pd.DataFrame:
    agg_df = df.groupby("item_id").agg(
        rating_mean      = ("rating",           "mean"),
        review_count     = ("user_id",          "count"),
        helpful_votes    = ("helpful_vote",      "sum"),
        verified_count   = ("verified_purchase", "sum"),
        last_review_date = ("timestamp",         "max"),
        price_median     = ("price",             "median"),
        main_category    = (COL_CATEGORY,        "first"),  # <-- uses constant
    ).reset_index()
    print(f"Aggregated {len(agg_df):,} unique products")
    return agg_df


# =========================
# 5.2 COMPUTE COMPOSITE SCORES
# satisfaction = 0.6*rating + 0.2*verified_rate + 0.2*log(helpful)
# recency, popularity, hotness
# =========================
def compute_scores(agg_df: pd.DataFrame) -> pd.DataFrame:
    df = agg_df.copy()

    df["verified_rate"] = df["verified_count"] / df["review_count"]
    df["helpful_log"]   = np.log1p(df["helpful_votes"])

    df["satisfaction"] = (
        0.6 * df["rating_mean"] +
        0.2 * df["verified_rate"] +
        0.2 * df["helpful_log"]
    )

    # Recency (normalized 0-1) with safe divide
    min_date      = df["last_review_date"].min()
    df["recency"] = (df["last_review_date"] - min_date).dt.days
    max_recency   = df["recency"].max()
    if max_recency > 0:
        df["recency"] = df["recency"] / max_recency

    df["popularity"] = np.log1p(df["review_count"])
    df["hotness"]    = df["popularity"] * df["recency"]

    return df


# =========================
# 5.3 PRICE SCORE
# price_score = 1 / (1 + abs(price - category_median) / category_median)
# =========================
def compute_price_score(agg_df: pd.DataFrame) -> pd.DataFrame:
    df = agg_df.copy()

    category_median = df.groupby("main_category")["price_median"].transform("median")
    category_median = category_median.replace(0, np.nan)  # avoid division by zero

    df["price_score"] = 1 / (
        1 + np.abs(df["price_median"] - category_median) / category_median
    )
    df["price_score"] = df["price_score"].fillna(0)

    return df


# =========================
# 5.4 PRODUCT RECOMMENDER CLASS
# get_recommendations(item_id, top_n, weights) with 5 scoring components
# =========================
class ProductRecommender:
    """
    Content-based recommender using 5 composite scoring components:
    satisfaction, recency, popularity, hotness, price_score.
    """

    DEFAULT_WEIGHTS = {
        "satisfaction": 0.4,
        "recency":      0.2,
        "popularity":   0.2,
        "hotness":      0.1,
        "price_score":  0.1,
    }

    def __init__(self, agg_df: pd.DataFrame):
        self.df = agg_df.copy()

    def get_recommendations(
        self,
        item_id: str,
        top_n:   int  = 10,
        weights: dict = None,
    ) -> pd.DataFrame:
        if weights is None:
            weights = self.DEFAULT_WEIGHTS

        scores = sum(weights[col] * self.df[col] for col in weights)
        self.df["final_score"] = scores

        recs = (
            self.df[self.df["item_id"] != item_id]
            .sort_values("final_score", ascending=False)
            .head(top_n)
        )
        return recs[["item_id", "final_score"]].reset_index(drop=True)


# =========================
# 5.5 - 5.8 COLLABORATIVE FILTERING RECOMMENDER CLASS
# Encapsulates prep + neighbor compute + recommend_products_cf(item_id, top_k)
# =========================
class CollaborativeFilteringRecommender:
    """
    Item-Item Collaborative Filtering recommender.

    Encapsulates:
      - sparse matrix construction        (5.5)
      - mean-centering by user            (5.6)
      - cosine similarity computation     (5.7)
      - recommend_products_cf(item_id)    (5.8)
    """

    def __init__(self, df: pd.DataFrame, top_k_neighbors: int = 50):
        self.top_k_neighbors = top_k_neighbors
        self._build(df)

    def _build(self, df: pd.DataFrame):
        """Build sparse matrix, mean-center, compute item similarity."""

        # 5.5 - maps and sparse matrix (item_id NOT asin)
        user_ids = df["user_id"].unique()
        item_ids = df["item_id"].unique()

        self.user_map    = {u: i for i, u in enumerate(user_ids)}
        self.item_map    = {item: idx for idx, item in enumerate(item_ids)}
        self.idx_to_item = {v: k for k, v in self.item_map.items()}

        df = df.copy()
        df["user_idx"] = df["user_id"].map(self.user_map)
        df["item_idx"] = df["item_id"].map(self.item_map)

        matrix = csr_matrix(
            (df["rating"].values, (df["user_idx"].values, df["item_idx"].values)),
            shape=(len(user_ids), len(item_ids)),
        )

        # 5.6 - mean-center by user
        user_means = np.array(matrix.mean(axis=1)).flatten()
        matrix_c   = matrix.copy().astype(float)
        for i in range(matrix_c.shape[0]):
            s, e = matrix_c.indptr[i], matrix_c.indptr[i + 1]
            matrix_c.data[s:e] -= user_means[i]

        # 5.7 - item-item cosine similarity (top-50 capped at query time)
        self.similarity = cosine_similarity(matrix_c.T, dense_output=False)
        print(f"CF model built — item similarity matrix: {self.similarity.shape}")

    def recommend_products_cf(self, item_id: str, top_k: int = 10) -> list:
        """Return top_k most similar item_ids to the given item_id."""
        if item_id not in self.item_map:
            return []

        idx     = self.item_map[item_id]
        sim_row = self.similarity[idx].toarray().flatten()
        sim_row[idx] = -1.0  # exclude the query item itself

        k           = min(top_k, self.top_k_neighbors, len(sim_row) - 1)
        top_indices = np.argpartition(sim_row, -k)[-k:]
        top_indices = top_indices[np.argsort(sim_row[top_indices])[::-1]]

        return [self.idx_to_item[i] for i in top_indices]


# =========================
# EVALUATION METRICS
# =========================
def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(recommended[:k]) & relevant) / len(relevant)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    dcg  = sum(1 / np.log2(i + 2) for i, item in enumerate(recommended[:k]) if item in relevant)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


# =========================
# EVALUATION RUNNERS
# =========================
def evaluate_cf(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    cf_model: CollaborativeFilteringRecommender,
    k:        int = 10,
    n_users:  int = 500,
) -> tuple[float, float]:

    test_user_items = test_df.groupby("user_id")["item_id"].apply(set).to_dict()
    users = list(set(train_df["user_id"]) & set(test_user_items.keys()))[:n_users]

    recalls, ndcgs = [], []
    for user in users:
        train_items = train_df[train_df["user_id"] == user]["item_id"].values
        if len(train_items) == 0:
            continue
        seed     = np.random.choice(train_items)
        relevant = test_user_items[user]
        recs     = cf_model.recommend_products_cf(seed, top_k=k)
        recalls.append(recall_at_k(recs, relevant, k))
        ndcgs.append(ndcg_at_k(recs,     relevant, k))

    return float(np.mean(recalls)), float(np.mean(ndcgs))


def evaluate_content(
    train_df:    pd.DataFrame,
    test_df:     pd.DataFrame,
    recommender: ProductRecommender,
    k:           int = 10,
    n_users:     int = 500,
) -> tuple[float, float]:

    test_user_items = test_df.groupby("user_id")["item_id"].apply(set).to_dict()
    users = list(set(train_df["user_id"]) & set(test_user_items.keys()))[:n_users]

    recalls, ndcgs = [], []
    for user in users:
        train_items = train_df[train_df["user_id"] == user]["item_id"].values
        if len(train_items) == 0:
            continue
        seed     = np.random.choice(train_items)
        relevant = test_user_items[user]
        recs     = recommender.get_recommendations(seed, top_n=k)["item_id"].tolist()
        recalls.append(recall_at_k(recs, relevant, k))
        ndcgs.append(ndcg_at_k(recs,     relevant, k))

    return float(np.mean(recalls)), float(np.mean(ndcgs))


# =========================
# 5.9 MLFLOW LOGGING
# =========================
def log_to_mlflow(
    content_recall: float,
    content_ndcg:   float,
    cf_recall:      float,
    cf_ndcg:        float,
):
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # content_only run — log weights dict as specified in 5.9
    with mlflow.start_run(run_name="content_only"):
        mlflow.log_param("top_n",   TOP_N_CONTENT)
        mlflow.log_param("weights", str(ProductRecommender.DEFAULT_WEIGHTS))
        mlflow.log_metric("recall_at_10", content_recall)
        mlflow.log_metric("ndcg_at_10",   content_ndcg)

    # CF run
    with mlflow.start_run(run_name="cf_item_item"):
        mlflow.log_param("top_k_neighbors", TOP_K_NEIGHBORS)
        mlflow.log_metric("recall_at_10", cf_recall)
        mlflow.log_metric("ndcg_at_10",   cf_ndcg)

    print("MLflow runs logged successfully")


# =========================
# MAIN
# =========================
def main():
    create_dirs()

    # Load raw data
    df = load_data(DATA_PATH)

    # Build product feature table (5.1 -> 5.3)
    agg_df = aggregate_product_features(df)
    agg_df = compute_scores(agg_df)
    agg_df = compute_price_score(agg_df)

    # Train / test split on interactions
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 5.4 - content-based recommender
    product_recommender = ProductRecommender(agg_df)

    # 5.5-5.8 - CF recommender (built on train only to avoid data leakage)
    cf_model = CollaborativeFilteringRecommender(
        train_df, top_k_neighbors=TOP_K_NEIGHBORS
    )

    # Evaluate content-based
    print("Evaluating content-based recommender ...")
    content_recall, content_ndcg = evaluate_content(
        train_df, test_df, product_recommender, k=10, n_users=N_EVAL_USERS
    )
    print(f"  Content  Recall@10 : {content_recall:.4f}")
    print(f"  Content  NDCG@10   : {content_ndcg:.4f}")

    # Evaluate CF
    print("Evaluating CF recommender ...")
    cf_recall, cf_ndcg = evaluate_cf(
        train_df, test_df, cf_model, k=10, n_users=N_EVAL_USERS
    )
    print(f"  CF       Recall@10 : {cf_recall:.4f}")
    print(f"  CF       NDCG@10   : {cf_ndcg:.4f}")

    # 5.9 - log both runs to MLflow
    log_to_mlflow(content_recall, content_ndcg, cf_recall, cf_ndcg)

    print("Phase 5 completed successfully!")


if __name__ == "__main__":
    main()