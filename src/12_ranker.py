"""
src/12_ranker.py
================
Phase 4 — LGBMRanker (LambdaMART) Model Training & Serving Layer

Workflow steps:
  1. Load ranker dataset from data/ranker_train.parquet.
  2. Split query groups by user_id into train and validation sets.
  3. Train LightGBM LambdaMART ranker (LGBMRanker) on candidate model scores + metadata features:
       [als_score, svdpp_score, mf_score, ncf_score, content_score,
        apriori_lift, price_score, recency, popularity, helpful_votes]
  4. Evaluate NDCG@10 and MAP@10 on validation groups.
  5. Serialize model to models/lgbm_ranker.txt and models/lgbm_ranker.pkl.
  6. Log run to MLflow experiment 'DS11-v2' with run_name='Ranker'.
  7. Provide cold-start aware serving helper for online candidate ranking.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow

# Setup paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR) if os.path.basename(SRC_DIR) == "src" else SRC_DIR
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config

FEATURE_COLS = [
    "als_score",
    "svdpp_score",
    "mf_score",
    "ncf_score",
    "content_score",
    "apriori_lift",
    "price_score",
    "recency",
    "popularity",
    "helpful_votes",
]


def compute_ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    """
    Computes NDCG@k for a single query group.
    """
    if len(y_true) == 0 or np.sum(y_true) == 0:
        return 0.0

    # Rank items by predicted score descending
    rank_order = np.argsort(-y_pred)
    sorted_true = y_true[rank_order][:k]

    # DCG
    discounts = np.log2(np.arange(len(sorted_true)) + 2.0)
    gains = np.power(2.0, sorted_true) - 1.0
    dcg = np.sum(gains / discounts)

    # Ideal DCG
    ideal_order = np.sort(y_true)[::-1][:k]
    ideal_gains = np.power(2.0, ideal_order) - 1.0
    ideal_discounts = np.log2(np.arange(len(ideal_order)) + 2.0)
    idcg = np.sum(ideal_gains / ideal_discounts)

    return float(dcg / idcg) if idcg > 0 else 0.0


def compute_map_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10, rel_threshold: float = 3.0) -> float:
    """
    Computes Average Precision @ k for a single query group.
    Items with y_true >= rel_threshold (or > 0) are considered relevant.
    """
    binary_rel = (y_true >= rel_threshold).astype(int)
    n_rel = np.sum(binary_rel)
    if n_rel == 0:
        return 0.0

    rank_order = np.argsort(-y_pred)[:k]
    hits = binary_rel[rank_order]

    cum_hits = np.cumsum(hits)
    ranks = np.arange(1, len(hits) + 1)
    precisions = cum_hits / ranks

    ap = np.sum(precisions * hits) / min(n_rel, k)
    return float(ap)


def evaluate_ranking_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    k: int = 10,
    rel_threshold: float = 3.0,
) -> tuple[float, float]:
    """
    Evaluates mean NDCG@k and MAP@k across all query groups.
    """
    ndcgs = []
    maps = []

    start = 0
    for g_size in groups:
        end = start + g_size
        group_true = y_true[start:end]
        group_pred = y_pred[start:end]

        ndcg = compute_ndcg_at_k(group_true, group_pred, k=k)
        map_val = compute_map_at_k(group_true, group_pred, k=k, rel_threshold=rel_threshold)

        ndcgs.append(ndcg)
        maps.append(map_val)

        start = end

    mean_ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0
    mean_map = float(np.mean(maps)) if maps else 0.0
    return mean_ndcg, mean_map


class RankerService:
    """
    Online serving helper for scoring candidate items using the trained LightGBM ranker.
    Includes cold-start fallback (zero-filling CF score features for unknown users).
    """

    def __init__(
        self,
        model=None,
        model_path: str = config.LGBM_RANKER_PATH,
        feature_cols: list = None,
    ):
        self.feature_cols = feature_cols or FEATURE_COLS
        self.model = model

        if self.model is None and os.path.exists(model_path):
            self.booster = lgb.Booster(model_file=model_path)
        elif self.model is not None and hasattr(self.model, "booster_"):
            self.booster = self.model.booster_
        else:
            self.booster = None

    def rank(
        self,
        candidates_df: pd.DataFrame,
        is_cold_start_user: bool = False,
    ) -> pd.DataFrame:
        """
        Ranks candidate items for a user.
        candidates_df must contain item_id and candidate features.
        """
        if candidates_df.empty or self.booster is None:
            return candidates_df

        df = candidates_df.copy()

        # Cold-start handling: zero-fill CF features for unknown users
        if is_cold_start_user:
            cf_features = ["als_score", "svdpp_score", "mf_score", "ncf_score", "apriori_lift"]
            for f in cf_features:
                df[f] = 0.0

        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0.0

        X = df[self.feature_cols].values
        scores = self.booster.predict(X)
        df["ranker_score"] = scores
        return df.sort_values("ranker_score", ascending=False).reset_index(drop=True)


def train_ranker(
    train_parquet_path: str = config.RANKER_TRAIN_PATH,
    model_txt_path: str = config.LGBM_RANKER_PATH,
    model_pkl_path: str = config.LGBM_RANKER_PKL_PATH,
    feature_cols: list = None,
    val_ratio: float = 0.2,
    n_estimators: int = 150,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    random_state: int = config.RANDOM_STATE,
) -> tuple[lgb.LGBMRanker, dict]:
    """
    Trains LGBMRanker LambdaMART model, computes evaluation metrics,
    saves model artifacts, and logs run to MLflow.
    """
    features = feature_cols or FEATURE_COLS
    print(f"\n==========================================")
    print(f"Training LGBMRanker on {train_parquet_path}")
    print(f"Features: {features}")
    print(f"==========================================")

    # 1. Load data
    df = pd.read_parquet(train_parquet_path)
    print(f"Loaded {len(df):,} interaction rows across {df['user_id'].nunique():,} unique users.")

    # 2. Split query groups by user_id
    users = df["user_id"].unique()
    rng = np.random.default_rng(random_state)
    n_val_users = int(len(users) * val_ratio)
    val_users_set = set(rng.choice(users, size=n_val_users, replace=False))

    train_df = df[~df["user_id"].isin(val_users_set)].sort_values("user_id").reset_index(drop=True)
    val_df = df[df["user_id"].isin(val_users_set)].sort_values("user_id").reset_index(drop=True)

    train_groups = train_df.groupby("user_id", sort=False).size().values
    val_groups = val_df.groupby("user_id", sort=False).size().values

    X_train = train_df[features].values
    y_train = train_df["relevance_label"].values

    X_val = val_df[features].values
    y_val = val_df["relevance_label"].values

    print(f"Train set: {len(train_df):,} rows | {len(train_groups):,} users")
    print(f"Val set:   {len(val_df):,} rows | {len(val_groups):,} users")

    # 3. Instantiate LGBMRanker
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[10],
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        random_state=random_state,
        importance_type="gain",
        n_jobs=-1,
    )

    # 4. Train model
    print("\nFitting LGBMRanker (LambdaMART)...")
    ranker.fit(
        X_train,
        y_train,
        group=train_groups,
        eval_set=[(X_val, y_val)],
        eval_group=[val_groups],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20, verbose=False),
            lgb.log_evaluation(period=25),
        ],
    )

    # 5. Evaluate on Validation Set
    val_preds = ranker.predict(X_val)
    val_ndcg, val_map = evaluate_ranking_metrics(y_val, val_preds, val_groups, k=10)

    train_preds = ranker.predict(X_train)
    train_ndcg, train_map = evaluate_ranking_metrics(y_train, train_preds, train_groups, k=10)

    print(f"\n==========================================")
    print(f"Ranker Evaluation Results:")
    print(f"  Val   NDCG@10: {val_ndcg:.4f}")
    print(f"  Val   MAP@10:  {val_map:.4f}")
    print(f"  Train NDCG@10: {train_ndcg:.4f}")
    print(f"  Train MAP@10:  {train_map:.4f}")
    print(f"  Best Iteration: {ranker.best_iteration_}")
    print(f"==========================================")

    # 6. Feature Importances
    importances = ranker.feature_importances_
    fi_dict = dict(zip(features, [float(i) for i in importances]))
    fi_df = pd.DataFrame({
        "feature": features,
        "importance_gain": importances,
    }).sort_values("importance_gain", ascending=False).reset_index(drop=True)

    print("\nFeature Importances (Gain):")
    for row in fi_df.itertuples():
        print(f"  {row.feature:<15} : {row.importance_gain:12.2f}")

    # 7. Save model artifacts
    os.makedirs(os.path.dirname(model_txt_path), exist_ok=True)
    ranker.booster_.save_model(model_txt_path)
    print(f"\nSaved model to {model_txt_path}")

    with open(model_pkl_path, "wb") as f:
        pickle.dump(ranker, f)
    print(f"Saved model to {model_pkl_path}")

    fi_csv_path = os.path.join(config.OUTPUTS_DIR, "ranker_feature_importance.csv")
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    fi_df.to_csv(fi_csv_path, index=False)
    print(f"Saved feature importance table to {fi_csv_path}")

    # 8. Log to MLflow
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    print(f"\nLogging to MLflow experiment '{config.MLFLOW_EXPERIMENT}' as 'Ranker'...")
    with mlflow.start_run(run_name="Ranker"):
        mlflow.log_param("model_type", "LightGBM LambdaMART (LGBMRanker)")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("num_leaves", num_leaves)
        mlflow.log_param("objective", "lambdarank")
        mlflow.log_param("features", json.dumps(features))
        mlflow.log_param("n_train_users", int(len(train_groups)))
        mlflow.log_param("n_val_users", int(len(val_groups)))
        mlflow.log_param("n_train_rows", int(len(train_df)))
        mlflow.log_param("n_val_rows", int(len(val_df)))
        mlflow.log_param("best_iteration", int(ranker.best_iteration_ or n_estimators))

        mlflow.log_metric("ndcg_at_10", val_ndcg)
        mlflow.log_metric("map_at_10", val_map)
        mlflow.log_metric("train_ndcg_at_10", train_ndcg)
        mlflow.log_metric("train_map_at_10", train_map)

        for feat, imp in fi_dict.items():
            mlflow.log_metric(f"importance_{feat}", imp)

        mlflow.log_artifact(model_txt_path)
        mlflow.log_artifact(model_pkl_path)
        mlflow.log_artifact(fi_csv_path)

    print("Ranker MLflow logging complete.")
    
    metrics = {
        "ndcg_at_10": val_ndcg,
        "map_at_10": val_map,
        "train_ndcg_at_10": train_ndcg,
        "train_map_at_10": train_map,
        "best_iteration": ranker.best_iteration_,
        "feature_importances": fi_dict,
    }
    return ranker, metrics


def main():
    config.create_dirs()
    config.set_seed()
    train_ranker()


if __name__ == "__main__":
    main()
