import os
os.environ["MKL_NUM_THREADS"] = "1"

import json
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import mlflow

from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from surprise import Dataset, Reader, SVDpp
from surprise.model_selection import train_test_split
from surprise.accuracy import rmse

# =========================
# CREATE DIRS
# =========================
def create_dirs():
    os.makedirs("models",  exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

create_dirs()

# =========================
# MLFLOW SETUP
# =========================
mlflow.set_tracking_uri("mlflow/")
mlflow.set_experiment("DS11")

# =========================
# LOAD DATA
# FIX: use train_df only — not full dataset (prevents leakage)
# =========================
print("Loading train dataset...")
df = pd.read_parquet("data/train_df.parquet")     # FIX: was cleaned_cf_dataset
df = df.dropna(subset=["user_id", "item_id", "rating"])
print(f"Train interactions: {len(df)}")

# =========================
# LOAD MAPS
# =========================
with open("data/user_map.json", "r") as f:
    user_map = json.load(f)

with open("data/item_map.json", "r") as f:
    item_map = json.load(f)

user_map = {k: int(v) for k, v in user_map.items()}
item_map = {k: int(v) for k, v in item_map.items()}

inv_item_map = {int(v): k for k, v in item_map.items()}

# =========================
# ALIGN DF WITH MAPS
# =========================
df = df[df["user_id"].isin(user_map) & df["item_id"].isin(item_map)]
df["user_idx"] = df["user_id"].map(user_map).astype(int)
df["item_idx"] = df["item_id"].map(item_map).astype(int)

print(f"Aligned interactions: {len(df)}")
print(f"Users: {df['user_idx'].nunique()}  Items: {df['item_idx'].nunique()}")

# =========================
# LOAD TEST + TRAIN SPLIT
# FIX: load train_df for seed lookup, test_df for ground truth
# =========================
test_df  = pd.read_parquet("data/test_df.parquet")
train_df = pd.read_parquet("data/train_df.parquet")

# FIX: seed lookup — last training item per user
seed_lookup = (
    train_df.sort_values("timestamp")
    .groupby("user_id")["item_id"]
    .last()
    .to_dict()
)

# Ground truth — test item per user
test_user_items = test_df.groupby("user_id")["item_id"].apply(list).to_dict()

# =========================
# LOAD MATRIX (built by Phase 8)
# =========================
print("\nLoading user_item_matrix from Phase 8...")
import scipy.sparse as sp

user_item_matrix = sp.load_npz("models/user_item_matrix.npz")
item_user_matrix = user_item_matrix.T.tocsr()

print(f"user_item_matrix: {user_item_matrix.shape}")   # (12569, 44301)
print(f"item_user_matrix: {item_user_matrix.shape}")   # (44301, 12569)

# ALS trains directly on item_user_matrix loaded above

# =========================
# TRAIN ALS
# =========================
print("\nTraining ALS...")

als_model = AlternatingLeastSquares(
    factors       = 64,
    iterations    = 20,
    regularization= 0.1,
    use_gpu       = False
)

# FIX: implicit expects item_user_matrix for fit()
# but user_item_matrix for recommend()
# FIX: pass user_item_matrix to fit() — implicit 0.7.x expects (n_users, n_items)
als_model.fit(user_item_matrix)

print(f"user_factors: {als_model.user_factors.shape}")
print(f"item_factors: {als_model.item_factors.shape}")

# Sanity check
sample_ids, sample_scores = als_model.recommend(
    0, user_item_matrix[0], N=5, filter_already_liked_items=True
)
print("Sample ALS recs:", list(zip(sample_ids.tolist(), sample_scores.tolist())))

# =========================
# ALS EVALUATION
# FIX: use seed from train, ground truth from test
# =========================
def recall_at_k_als(model, k=10):
    recalls = []
    test_users = list(test_user_items.keys())
    sampled    = np.random.choice(
        test_users, size=min(2000, len(test_users)), replace=False
    )

    for user_id in sampled:
        user_idx = user_map.get(user_id)
        if user_idx is None:
            continue

        # FIX: skip users with no training seed
        if user_id not in seed_lookup:
            continue

        try:
            item_indices, _ = model.recommend(
                user_idx,
                user_item_matrix[user_idx],
                N=k,
                filter_already_liked_items=True
            )
        except Exception as e:
            continue

        if len(item_indices) == 0:
            continue

        rec_items  = [inv_item_map.get(int(i)) for i in item_indices]
        rec_items  = [r for r in rec_items if r is not None]

        # FIX: ground truth from test_df, not from same split
        true_items = test_user_items.get(user_id, [])
        if not true_items:
            continue

        hits = len(set(rec_items) & set(true_items))
        recalls.append(hits / len(true_items))

    return float(np.mean(recalls)) if recalls else 0.0


def ndcg_at_k_als(model, k=10):
    ndcgs = []
    test_users = list(test_user_items.keys())
    sampled    = np.random.choice(
        test_users, size=min(2000, len(test_users)), replace=False
    )

    for user_id in sampled:
        user_idx = user_map.get(user_id)
        if user_idx is None:
            continue

        if user_id not in seed_lookup:
            continue

        try:
            item_indices, _ = model.recommend(
                user_idx,
                user_item_matrix[user_idx],
                N=k,
                filter_already_liked_items=True
            )
        except Exception as e:
            continue

        if len(item_indices) == 0:
            continue

        rec_items  = [inv_item_map.get(int(i)) for i in item_indices]
        rec_items  = [r for r in rec_items if r is not None]

        true_items = test_user_items.get(user_id, [])
        if not true_items:
            continue

        dcg  = sum(
            1 / np.log2(i + 2)
            for i, item in enumerate(rec_items)
            if item in true_items
        )
        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(true_items), k)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(ndcgs)) if ndcgs else 0.0


def als_rmse(model, df_imp, n_samples=20000):
    preds, actuals = [], []
    sample_df = df_imp.sample(min(n_samples, len(df_imp)), random_state=42)
    for row in sample_df.itertuples():
        try:
            pred = model.user_factors[row.user_idx].dot(
                model.item_factors[row.item_idx]
            )
            preds.append(pred)
            actuals.append(row.confidence)
        except:
            continue
    if not preds:
        return 0.0
    return float(np.sqrt(np.mean(
        (np.array(preds) - np.array(actuals)) ** 2
    )))


# Build df_implicit from df for rmse calculation
df_implicit = df.copy()
df_implicit["confidence"] = df_implicit["rating"].astype(np.float32) / 5.0

# RUN EVALUATION
print("\nEvaluating ALS...")
als_recall   = recall_at_k_als(als_model)
als_ndcg     = ndcg_at_k_als(als_model)
als_rmse_val = als_rmse(als_model, df_implicit)

print(f"ALS RMSE: {als_rmse_val:.4f}")
print(f"ALS Recall@10: {als_recall:.4f}")
print(f"ALS NDCG@10:   {als_ndcg:.4f}")

# =========================
# LOG ALS TO MLFLOW
# =========================
with mlflow.start_run(run_name="ALS"):
    mlflow.log_param("factors",         64)
    mlflow.log_param("iterations",      20)
    mlflow.log_param("regularization",  0.1)
    mlflow.log_param("confidence_scale","rating/5.0")
    mlflow.log_param("train_data",      "train_df only (leave-one-out split)")

    mlflow.log_metric("rmse",         als_rmse_val)
    mlflow.log_metric("recall_at_10", als_recall)
    mlflow.log_metric("ndcg_at_10",   als_ndcg)

    als_model.save("models/als_model.npz")
    mlflow.log_artifact("models/als_model.npz")
    mlflow.log_artifact("models/user_item_matrix.npz")

print("ALS DONE")

# =========================
# TRAIN SVD++
# FIX: use train_df only — not full df
# =========================
print("\nTraining SVD++...")

reader = Reader(rating_scale=(
    df["rating"].min(), df["rating"].max()
))
surprise_data = Dataset.load_from_df(
    df[["user_id", "item_id", "rating"]], reader
)

# FIX: use full train_df for SVD++ — internal split for validation only
trainset, testset = train_test_split(
    surprise_data, test_size=0.1, random_state=42
)

svdpp_model = SVDpp(n_factors=50, n_epochs=20)
svdpp_model.fit(trainset)

predictions  = svdpp_model.test(testset)
svd_rmse_val = rmse(predictions)

print(f"SVD++ RMSE: {svd_rmse_val:.4f}")

# =========================
# LOG SVD++ TO MLFLOW
# =========================
with mlflow.start_run(run_name="SVDpp"):
    mlflow.log_param("n_factors",  50)
    mlflow.log_param("n_epochs",   20)
    mlflow.log_param("train_data", "train_df only (leave-one-out split)")

    mlflow.log_metric("rmse", svd_rmse_val)

    with open("models/svdpp_model.pkl", "wb") as f:
        pickle.dump(svdpp_model, f)

    mlflow.log_artifact("models/svdpp_model.pkl")

print("SVD++ DONE")
print("\nPhase 9 complete.")
print("Outputs:")
print("  models/als_model.npz")
print("  models/svdpp_model.pkl")
print("  models/user_item_matrix.npz")