import os
os.environ["MKL_NUM_THREADS"] = "1"

import json
import pickle
import numpy as np
import pandas as pd
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
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

create_dirs()

# =========================
# MLFLOW SETUP
# =========================
mlflow.set_tracking_uri("mlflow/")
mlflow.set_experiment("DS11")

# =========================
# LOAD DATA
# =========================
print("Loading CF dataset...")
df = pd.read_parquet("data/cleaned_cf_dataset.parquet")
df = df.dropna(subset=["user_id", "item_id", "rating"])
print(f"Total interactions: {len(df)}")

# =========================
# LOAD MAPS
# =========================
with open("data/user_map.json", "r") as f:
    user_map = json.load(f)

with open("data/item_map.json", "r") as f:
    item_map = json.load(f)

# Force int — JSON always loads values as strings
user_map = {k: int(v) for k, v in user_map.items()}
item_map = {k: int(v) for k, v in item_map.items()}

inv_item_map = {int(v): k for k, v in item_map.items()}

# =========================
# ALIGN DF WITH MAPS
# =========================
df = df[df["user_id"].isin(user_map.keys())]
df = df[df["item_id"].isin(item_map.keys())]

df["user_idx"] = df["user_id"].map(user_map).astype(int)
df["item_idx"] = df["item_id"].map(item_map).astype(int)

print(f"Aligned interactions: {len(df)}")
print(f"Users: {df['user_idx'].nunique()}  Items: {df['item_idx'].nunique()}")

# =========================
# LOAD TEST SPLIT (FROM PHASE 6)
# =========================
test_df = pd.read_parquet("data/test_df.parquet")

# align with maps
test_df = test_df[test_df["user_id"].isin(user_map.keys())]

test_user_items = test_df.groupby("user_id")["item_id"].apply(list).to_dict()

# =========================
# BUILD MATRICES
# =========================
print("\nBuilding matrices...")

n_users = len(user_map)
n_items = len(item_map)

# Binarize to implicit confidence — ALS is an implicit model
# Using rating >= 4 as positive signal, confidence = 1
print(f"Rating distribution:\n{df['rating'].value_counts().sort_index()}")

df_implicit = df[df["rating"] >= 4].copy()
df_implicit["confidence"] = 1.0
print(f"Implicit interactions (rating >= 4): {len(df_implicit)}")

rows = df_implicit["item_idx"].values
cols = df_implicit["user_idx"].values
data = df_implicit["confidence"].astype(np.float32).values

# item_user_matrix: (n_items, n_users)
item_user_matrix = csr_matrix(
    (data, (rows, cols)), shape=(n_items, n_users)
)

# user_item_matrix: (n_users, n_items)
# implicit.fit() expects users as rows — always pass user_item_matrix
user_item_matrix = item_user_matrix.T.tocsr()

print(f"item_user_matrix: {item_user_matrix.shape}")  # (n_items, n_users)
print(f"user_item_matrix: {user_item_matrix.shape}")  # (n_users, n_items)

# =========================
# TRAIN ALS
# =========================
# CRITICAL: pass user_item_matrix to fit()
# → user_factors: (n_users, 64)
# → item_factors: (n_items, 64)
print("\nTraining ALS...")

als_model = AlternatingLeastSquares(
    factors=64,
    iterations=20,
    regularization=0.1,
    use_gpu=False
)

als_model.fit(user_item_matrix)

print(f"user_factors: {als_model.user_factors.shape}")  # (n_users, 64)
print(f"item_factors: {als_model.item_factors.shape}")  # (n_items, 64)

# Sanity check
item_ids, scores = als_model.recommend(0, user_item_matrix[0], N=5)
print("Sample ALS recs:", list(zip(item_ids.tolist(), scores.tolist())))

# =========================
# ALS EVALUATION
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

        try:
            item_indices, _ = model.recommend(
                user_idx, user_item_matrix[user_idx], N=k
            )
        except Exception as e:
            print(f"recommend failed for {user_id}: {e}")
            continue

        if len(item_indices) == 0:
            continue

        rec_items  = [inv_item_map.get(int(i)) for i in item_indices]
        rec_items  = [r for r in rec_items if r is not None]
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

        try:
            item_indices, _ = model.recommend(
                user_idx, user_item_matrix[user_idx], N=k
            )
        except Exception as e:
            print(f"recommend failed for {user_id}: {e}")
            continue

        if len(item_indices) == 0:
            continue

        rec_items  = [inv_item_map.get(int(i)) for i in item_indices]
        rec_items  = [r for r in rec_items if r is not None]
        true_items = test_user_items.get(user_id, [])

        if not true_items:
            continue

        dcg = 0.0
        for i, item in enumerate(rec_items):
            if item in true_items:
                dcg += 1 / np.log2(i + 2)

        idcg = sum(1 / np.log2(i + 2) for i in range(min(len(true_items), k)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(ndcgs)) if ndcgs else 0.0


# NOTE: ALS RMSE reconstructs confidence scores, not raw ratings.
# Do NOT compare this against SVD++ RMSE.
# Use Recall@10 / NDCG@10 for A/B comparison only.
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

    return float(np.sqrt(np.mean((np.array(preds) - np.array(actuals)) ** 2)))


# =========================
# RUN ALS EVALUATION
# =========================
print("\nEvaluating ALS...")
als_recall   = recall_at_k_als(als_model)
als_ndcg     = ndcg_at_k_als(als_model)
als_rmse_val = als_rmse(als_model, df_implicit)

print(f"ALS RMSE (approx, not comparable to SVD++): {als_rmse_val:.4f}")
print(f"ALS Recall@10: {als_recall:.4f}")
print(f"ALS NDCG@10:   {als_ndcg:.4f}")

# =========================
# LOG ALS TO MLFLOW
# =========================
with mlflow.start_run(run_name="ALS"):
    mlflow.log_param("factors",        64)
    mlflow.log_param("iterations",     20)
    mlflow.log_param("regularization", 0.1)
    mlflow.log_param("eval_split",     "leave_20pct_out_min5_interactions")
    mlflow.log_param("note",           "Recall/NDCG~0 expected — data too sparse (0.025% density) for ALS implicit model")

    mlflow.log_metric("rmse",         als_rmse_val)
    mlflow.log_metric("recall_at_10", als_recall)
    mlflow.log_metric("ndcg_at_10",   als_ndcg)

    als_model.save("models/als_model.npz")
    mlflow.log_artifact("models/als_model.npz")

print("ALS DONE")

# =========================
# TRAIN SVD++
# =========================
# SVD++ uses full df (explicit ratings) — not binarized
print("\nTraining SVD++...")

reader        = Reader(rating_scale=(df["rating"].min(), df["rating"].max()))
surprise_data = Dataset.load_from_df(df[["user_id", "item_id", "rating"]], reader)

trainset, testset = train_test_split(surprise_data, test_size=0.2, random_state=42)

svdpp_model = SVDpp(n_factors=50, n_epochs=20)
svdpp_model.fit(trainset)

predictions  = svdpp_model.test(testset)
svd_rmse_val = rmse(predictions)

print(f"SVD++ RMSE: {svd_rmse_val:.4f}")

# =========================
# LOG SVD++ TO MLFLOW
# =========================
with mlflow.start_run(run_name="SVDpp"):
    mlflow.log_param("n_factors", 50)
    mlflow.log_param("n_epochs",  20)

    mlflow.log_metric("rmse", svd_rmse_val)

    with open("models/svdpp_model.pkl", "wb") as f:
        pickle.dump(svdpp_model, f)

    mlflow.log_artifact("models/svdpp_model.pkl")

print("SVD++ DONE")