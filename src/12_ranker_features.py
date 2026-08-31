"""
src/12_ranker_features.py
=========================
Phase 4 — Ranker Training Data Construction

Workflow steps:
  1. Load train/test interactions (item_id = parent_asin) and ID maps.
  2. Perform popularity-weighted negative sampling (5-10 negatives per user with relevance 0).
  3. Load trained candidate models: ALS, SVD++, MF, NCF, Content-based, Apriori.
  4. Generate per-interaction feature rows:
       [als_score, svdpp_score, mf_score, ncf_score, content_score,
        apriori_lift, price_score, recency, popularity, helpful_votes]
  5. Assign relevance labels: rating for positive interactions, 0.0 for negatives.
  6. Structure by query group (sorted by user_id) and export data/ranker_train.parquet.
"""

import os
import sys
import json
import pickle
import importlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Setup paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR) if os.path.basename(SRC_DIR) == "src" else SRC_DIR
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config

# Unpickling hooks for custom classes
import __main__

try:
    _mod04 = importlib.import_module("04_apriori_recommender")
    __main__.AprioriRecommender = _mod04.AprioriRecommender
except Exception:
    pass

try:
    _mod05 = importlib.import_module("05_content_cf_recommender")
    __main__.ProductRecommender = _mod05.ProductRecommender
    __main__.CollaborativeFilteringRecommender = _mod05.CollaborativeFilteringRecommender
except Exception:
    pass

try:
    _mod06 = importlib.import_module("06_mf_ncf_pytorch")
    NCF = _mod06.NCF
    MF = _mod06.MF
except Exception:

    class NCF(nn.Module):
        def __init__(self, n_users, n_items, emb_dim=64):
            super().__init__()
            self.user_emb_gmf = nn.Embedding(n_users, emb_dim)
            self.item_emb_gmf = nn.Embedding(n_items, emb_dim)
            self.user_emb_mlp = nn.Embedding(n_users, emb_dim)
            self.item_emb_mlp = nn.Embedding(n_items, emb_dim)
            self.mlp = nn.Sequential(
                nn.Linear(emb_dim * 2, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 32), nn.ReLU()
            )
            self.final = nn.Linear(emb_dim + 32, 1)

        def forward(self, u, i):
            gmf = self.user_emb_gmf(u) * self.item_emb_gmf(i)
            mlp_out = self.mlp(torch.cat([self.user_emb_mlp(u), self.item_emb_mlp(i)], dim=1))
            return self.final(torch.cat([gmf, mlp_out], dim=1)).squeeze()


def load_candidate_models(user_map: dict, item_map: dict, device: str = "cpu"):
    """
    Loads all trained candidate models from models/ directory.
    Returns dictionary of loaded models and feature lookup tables.
    """
    models = {}
    n_users = len(user_map)
    n_items = len(item_map)

    # 1. ALS model
    if os.path.exists(config.ALS_MODEL_PATH):
        from implicit.als import AlternatingLeastSquares

        als_model = AlternatingLeastSquares(factors=64, iterations=20, regularization=0.1)
        models["als"] = als_model.load(config.ALS_MODEL_PATH)
        print(f"  [loaded] ALS model from {config.ALS_MODEL_PATH}")
    else:
        models["als"] = None
        print(f"  [warn] ALS model not found at {config.ALS_MODEL_PATH}")

    # 2. SVD++ model
    if os.path.exists(config.SVDPP_MODEL_PATH):
        with open(config.SVDPP_MODEL_PATH, "rb") as f:
            models["svdpp"] = pickle.load(f)
        print(f"  [loaded] SVD++ model from {config.SVDPP_MODEL_PATH}")
    else:
        models["svdpp"] = None
        print(f"  [warn] SVD++ model not found at {config.SVDPP_MODEL_PATH}")

    # 3. MF model
    if os.path.exists(config.MF_MODEL_PATH):
        mf_state = torch.load(config.MF_MODEL_PATH, map_location=device)
        models["mf_u_emb"] = mf_state["user_emb.weight"].cpu().numpy()
        models["mf_i_emb"] = mf_state["item_emb.weight"].cpu().numpy()
        models["mf_u_bias"] = mf_state["user_bias.weight"].cpu().numpy().squeeze()
        models["mf_i_bias"] = mf_state["item_bias.weight"].cpu().numpy().squeeze()
        print(f"  [loaded] MF model from {config.MF_MODEL_PATH}")
    else:
        models["mf_u_emb"] = None
        print(f"  [warn] MF model not found at {config.MF_MODEL_PATH}")

    # 4. NCF model
    if os.path.exists(config.NCF_MODEL_PATH):
        ncf_model = NCF(n_users, n_items, emb_dim=64)
        ncf_model.load_state_dict(torch.load(config.NCF_MODEL_PATH, map_location=device))
        ncf_model.to(device)
        ncf_model.eval()
        models["ncf"] = ncf_model
        print(f"  [loaded] NCF model from {config.NCF_MODEL_PATH}")
    else:
        models["ncf"] = None
        print(f"  [warn] NCF model not found at {config.NCF_MODEL_PATH}")

    # 5. Product recommender (content-based features)
    prod_pkl = os.path.join(config.MODELS_DIR, "product_recommender.pkl")
    if os.path.exists(prod_pkl):
        import dill

        with open(prod_pkl, "rb") as f:
            prod_rec = dill.load(f)
        models["prod_df"] = prod_rec.df
        print(f"  [loaded] ProductRecommender features ({len(prod_rec.df):,} items)")
    else:
        models["prod_df"] = None
        print(f"  [warn] ProductRecommender not found at {prod_pkl}")

    # 6. Apriori recommender
    apriori_pkl = os.path.join(config.MODELS_DIR, "apriori_recommender.pkl")
    if os.path.exists(apriori_pkl):
        import dill

        with open(apriori_pkl, "rb") as f:
            apriori_rec = dill.load(f)
        models["apriori_rules"] = apriori_rec.rule_dict
        print(f"  [loaded] AprioriRecommender ({len(apriori_rec.rule_dict):,} antecedent rules)")
    else:
        models["apriori_rules"] = {}
        print(f"  [warn] AprioriRecommender not found at {apriori_pkl}")

    return models


def sample_popularity_negatives(
    pos_df: pd.DataFrame,
    all_item_ids: np.ndarray,
    item_weights: np.ndarray,
    n_negatives: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    For each user in pos_df, samples n_negatives items from all_item_ids
    weighted by item_weights, excluding items the user has interacted with.
    """
    rng = np.random.default_rng(random_state)
    user_pos_items = pos_df.groupby("user_id")["item_id"].apply(set).to_dict()

    neg_rows = []
    # Pre-sample candidate negative pools for efficiency
    for user_id, pos_set in user_pos_items.items():
        # Over-sample to account for collisions with pos_set
        sample_size = n_negatives * 3
        candidates = rng.choice(all_item_ids, size=sample_size, p=item_weights, replace=True)
        chosen = []
        for item in candidates:
            if item not in pos_set and item not in chosen:
                chosen.append(item)
                if len(chosen) == n_negatives:
                    break

        # Fallback if oversampling didn't yield enough unique negatives
        if len(chosen) < n_negatives:
            remaining = rng.choice(all_item_ids, size=n_negatives * 5, replace=True)
            for item in remaining:
                if item not in pos_set and item not in chosen:
                    chosen.append(item)
                    if len(chosen) == n_negatives:
                        break

        for neg_item in chosen:
            neg_rows.append(
                {
                    "user_id": user_id,
                    "item_id": neg_item,
                    "rating": 0.0,
                    "is_positive": 0,
                }
            )

    neg_df = pd.DataFrame(neg_rows)
    return neg_df


def build_ranker_features(
    interactions_df_path: str = (
        config.TRAIN_PARQUET_PATH
        if hasattr(config, "TRAIN_PARQUET_PATH")
        else os.path.join(config.DATA_DIR, "train_df.parquet")
    ),
    output_path: str = config.RANKER_TRAIN_PATH,
    n_negatives: int = 5,
    batch_size: int = 8192,
    device: str = config.DEVICE,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """
    Builds ranker dataset:
      - Reads positive interactions
      - Samples popularity-weighted negatives
      - Computes features across all models
      - Sorts query groups by user_id
      - Exports to Parquet
    """
    print(f"\n==========================================")
    print(f"Generating Ranker Features -> {output_path}")
    print(f"==========================================")

    # 1. Load ID mappings
    with open(config.USER_MAP_PATH, "r") as f:
        user_map = json.load(f)
    with open(config.ITEM_MAP_PATH, "r") as f:
        item_map = json.load(f)

    user_map = {str(k): int(v) for k, v in user_map.items()}
    item_map = {str(k): int(v) for k, v in item_map.items()}

    # 2. Load interactions
    pos_df = pd.read_parquet(interactions_df_path)
    pos_df["user_id"] = pos_df["user_id"].astype(str)
    pos_df["item_id"] = pos_df["item_id"].astype(str)
    pos_df["rating"] = pos_df["rating"].astype(float)
    pos_df["is_positive"] = 1

    print(f"Loaded {len(pos_df):,} positive interactions across {pos_df['user_id'].nunique():,} users.")

    # 3. Prepare item popularity weights for negative sampling
    all_item_ids = np.array(list(item_map.keys()))

    # Calculate item counts from clean parquet or positive interactions
    if os.path.exists(config.CLEAN_PARQUET_PATH):
        clean_df = pd.read_parquet(config.CLEAN_PARQUET_PATH, columns=["item_id"])
        item_counts = clean_df["item_id"].astype(str).value_counts()
    else:
        item_counts = pos_df["item_id"].value_counts()

    counts_arr = np.array([item_counts.get(iid, 1) for iid in all_item_ids], dtype=np.float64)
    # Popularity distribution smoothing: (count + 1)^0.75
    weights = np.power(counts_arr + 1.0, 0.75)
    weights /= weights.sum()

    # 4. Sample negatives
    print(f"Sampling {n_negatives} popularity-weighted negatives per user...")
    neg_df = sample_popularity_negatives(
        pos_df=pos_df,
        all_item_ids=all_item_ids,
        item_weights=weights,
        n_negatives=n_negatives,
        random_state=random_state,
    )
    print(f"Sampled {len(neg_df):,} negative interactions.")

    # 5. Combine positives and negatives
    keep_cols = ["user_id", "item_id", "rating", "is_positive"]
    combined_df = pd.concat([pos_df[keep_cols], neg_df[keep_cols]], ignore_index=True)

    # Graded relevance label: rating for positive (1.0 to 5.0), 0.0 for negative
    combined_df["relevance_label"] = np.where(combined_df["is_positive"] == 1, combined_df["rating"], 0.0).astype(
        np.float32
    )

    # Map integer indices
    combined_df["user_idx"] = combined_df["user_id"].map(user_map).fillna(-1).astype(int)
    combined_df["item_idx"] = combined_df["item_id"].map(item_map).fillna(-1).astype(int)

    n_rows = len(combined_df)
    print(f"Total candidate rows for feature generation: {n_rows:,}")

    # 6. Load candidate models
    models = load_candidate_models(user_map, item_map, device=device)

    # 7. Compute ALS score
    print("Computing ALS scores...")
    als_scores = np.zeros(n_rows, dtype=np.float32)
    if models["als"] is not None:
        u_factors = models["als"].user_factors
        i_factors = models["als"].item_factors
        valid_mask = (combined_df["user_idx"] >= 0) & (combined_df["item_idx"] >= 0)
        valid_u = combined_df.loc[valid_mask, "user_idx"].values
        valid_i = combined_df.loc[valid_mask, "item_idx"].values
        als_scores[valid_mask] = np.sum(u_factors[valid_u] * i_factors[valid_i], axis=1)
    combined_df["als_score"] = als_scores

    # 8. Compute MF score
    print("Computing MF scores...")
    mf_scores = np.zeros(n_rows, dtype=np.float32)
    if models["mf_u_emb"] is not None:
        valid_mask = (combined_df["user_idx"] >= 0) & (combined_df["item_idx"] >= 0)
        valid_u = combined_df.loc[valid_mask, "user_idx"].values
        valid_i = combined_df.loc[valid_mask, "item_idx"].values
        mf_scores[valid_mask] = (
            np.sum(models["mf_u_emb"][valid_u] * models["mf_i_emb"][valid_i], axis=1)
            + models["mf_u_bias"][valid_u]
            + models["mf_i_bias"][valid_i]
        )
    combined_df["mf_score"] = mf_scores

    # 9. Compute NCF score (batched PyTorch)
    print("Computing NCF scores...")
    ncf_scores = np.zeros(n_rows, dtype=np.float32)
    if models["ncf"] is not None:
        valid_mask = (combined_df["user_idx"] >= 0) & (combined_df["item_idx"] >= 0)
        valid_indices = np.where(valid_mask)[0]
        u_vals = combined_df.loc[valid_mask, "user_idx"].values
        i_vals = combined_df.loc[valid_mask, "item_idx"].values

        ncf_model = models["ncf"]
        with torch.no_grad():
            for start in range(0, len(valid_indices), batch_size):
                end = min(start + batch_size, len(valid_indices))
                batch_u = torch.tensor(u_vals[start:end], dtype=torch.long, device=device)
                batch_i = torch.tensor(i_vals[start:end], dtype=torch.long, device=device)
                preds = ncf_model(batch_u, batch_i).cpu().numpy()
                idx_slice = valid_indices[start:end]
                ncf_scores[idx_slice] = preds
    combined_df["ncf_score"] = ncf_scores

    # 10. Compute SVD++ score
    print("Computing SVD++ scores...")
    svdpp_scores = np.zeros(n_rows, dtype=np.float32)
    if models["svdpp"] is not None:
        svdpp = models["svdpp"]
        uids = combined_df["user_id"].values
        iids = combined_df["item_id"].values
        svdpp_list = [svdpp.predict(uid=u, iid=i).est for u, i in zip(uids, iids)]
        svdpp_scores = np.array(svdpp_list, dtype=np.float32)
    combined_df["svdpp_score"] = svdpp_scores

    # 11. Compute Apriori Lift
    print("Computing Apriori lift features...")
    rule_dict = models["apriori_rules"]
    # Pre-build user -> {consequent: max_lift}
    user_history = pos_df.groupby("user_id")["item_id"].apply(list).to_dict()
    user_apriori_lifts = {}
    for uid, items in user_history.items():
        lifts = {}
        for it in items:
            for conseq, lift, conf in rule_dict.get(it, []):
                if conseq not in lifts or lift > lifts[conseq]:
                    lifts[conseq] = float(lift)
        if lifts:
            user_apriori_lifts[uid] = lifts

    uids = combined_df["user_id"].values
    iids = combined_df["item_id"].values
    apriori_lifts = np.array([user_apriori_lifts.get(u, {}).get(i, 0.0) for u, i in zip(uids, iids)], dtype=np.float32)
    combined_df["apriori_lift"] = apriori_lifts

    # 12. Compute Content & Product Metadata features
    print("Computing Content and Product features...")
    prod_df = models["prod_df"]
    if prod_df is not None:
        # Calculate composite content score
        weights = {
            "satisfaction": 0.4,
            "recency": 0.2,
            "popularity": 0.2,
            "hotness": 0.1,
            "price_score": 0.1,
        }
        score_series = sum(weights[col] * prod_df[col] for col in weights if col in prod_df.columns)
        prod_meta = prod_df.copy()
        prod_meta["content_score"] = score_series.astype(np.float32)

        # Select required columns
        lookup_cols = ["content_score", "price_score", "recency", "popularity", "helpful_votes"]
        for c in lookup_cols:
            if c not in prod_meta.columns:
                prod_meta[c] = 0.0

        item_feature_dict = prod_meta[lookup_cols].to_dict(orient="index")

        content_scores = []
        price_scores = []
        recencies = []
        popularities = []
        helpful_votes_list = []

        for iid in iids:
            meta = item_feature_dict.get(iid, None)
            if meta:
                content_scores.append(meta.get("content_score", 0.0))
                price_scores.append(meta.get("price_score", 0.5))
                recencies.append(meta.get("recency", 0.0))
                popularities.append(meta.get("popularity", 0.0))
                helpful_votes_list.append(meta.get("helpful_votes", 0.0))
            else:
                content_scores.append(0.0)
                price_scores.append(0.5)
                recencies.append(0.0)
                popularities.append(0.0)
                helpful_votes_list.append(0.0)

        combined_df["content_score"] = np.array(content_scores, dtype=np.float32)
        combined_df["price_score"] = np.array(price_scores, dtype=np.float32)
        combined_df["recency"] = np.array(recencies, dtype=np.float32)
        combined_df["popularity"] = np.array(popularities, dtype=np.float32)
        combined_df["helpful_votes"] = np.array(helpful_votes_list, dtype=np.float32)
    else:
        combined_df["content_score"] = 0.0
        combined_df["price_score"] = 0.5
        combined_df["recency"] = 0.0
        combined_df["popularity"] = 0.0
        combined_df["helpful_votes"] = 0.0

    # 13. Sort query groups by user_id
    output_columns = [
        "user_id",
        "item_id",
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
        "relevance_label",
    ]

    combined_df = combined_df.sort_values("user_id").reset_index(drop=True)[output_columns]

    # 14. Save parquet
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined_df.to_parquet(output_path, index=False)
    print(
        f"\n[Done] Exported {len(combined_df):,} feature rows for "
        f"{combined_df['user_id'].nunique():,} users to {output_path}"
    )
    print(f"Features: {output_columns[2:-1]}")
    print(f"Relevance distribution:\n{combined_df['relevance_label'].value_counts().sort_index()}")

    return combined_df


def main():
    config.create_dirs()
    config.set_seed()
    build_ranker_features(
        interactions_df_path=os.path.join(config.DATA_DIR, "train_df.parquet"),
        output_path=config.RANKER_TRAIN_PATH,
        n_negatives=5,
        device=config.DEVICE,
    )


if __name__ == "__main__":
    main()
