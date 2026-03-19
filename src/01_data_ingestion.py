"""
01_data_ingestion.py
Phase 1 — Data Ingestion & Merging (WORKING FINAL VERSION)
"""

import os
import sys
import time
import requests
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Allow running from repo root OR from src/
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    CATEGORIES,
    CHUNK_SIZE,
    MERGE_CSV_PATH,
    CLEAN_PARQUET_PATH,
    CF_DATA_PATH,
    setup_mlflow,
)

import mlflow


# =========================
# Utility
# =========================
def create_dirs():
    os.makedirs("data", exist_ok=True)


def download_file(url, path, retries=3):
    for attempt in range(retries):
        try:
            print(f"  Downloading → {os.path.basename(path)} (Attempt {attempt+1})")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with open(path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return

        except Exception as e:
            print(f"  Retry {attempt+1} failed: {e}")
            time.sleep(2)

    raise RuntimeError(f"Failed to download {url}")


# =========================
# Load data
# =========================
def load_category_data(category):
    print(f"\n[1.1/1.2] Loading category: {category}")

    # ✅ CORRECT WORKING URLS
    review_url = f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/{category}.jsonl"
    meta_url   = f"https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_{category}.jsonl"

    review_path = f"data/{category}.jsonl"
    meta_path   = f"data/meta_{category}.jsonl"

    # Download if not exists
    if not os.path.exists(review_path):
        download_file(review_url, review_path)

    if not os.path.exists(meta_path):
        download_file(meta_url, meta_path)

    # Load limited rows
    review_df = pd.read_json(review_path, lines=True, nrows=CHUNK_SIZE)
    meta_df   = pd.read_json(meta_path,   lines=True, nrows=CHUNK_SIZE)

    # Add category tag
    review_df["main_category"] = category
    meta_df["main_category"]   = category

    print(f"  review: {review_df.shape}")
    print(f"  meta  : {meta_df.shape}")

    return review_df, meta_df


# =========================
# Main pipeline
# =========================
def main():
    create_dirs()
    setup_mlflow()

    with mlflow.start_run(run_name="phase1_data_ingestion"):

        review_dfs, meta_dfs = [], []

        # Load all categories
        for cat in tqdm(CATEGORIES, desc="Loading categories"):
            r_df, m_df = load_category_data(cat)
            review_dfs.append(r_df)
            meta_dfs.append(m_df)

        # Concat
        print("\n[1.3] Concatenating...")
        review_df = pd.concat(review_dfs, ignore_index=True)
        meta_df   = pd.concat(meta_dfs,   ignore_index=True)
        del review_dfs, meta_dfs

        print(f"  review_df: {review_df.shape}")
        print(f"  meta_df  : {meta_df.shape}")

        # Merge (CRITICAL FIX)
        print("\n[1.4] Merging on parent_asin...")
        merge_df = review_df.merge(
            meta_df,
            on="parent_asin",
            suffixes=("_rev", "_meta"),
            how="inner",
        )
        del review_df, meta_df

        print(f"  merged_df: {merge_df.shape}")

        # Validation
        required_cols = ["user_id", "parent_asin", "rating", "timestamp"]
        missing = [c for c in required_cols if c not in merge_df.columns]
        assert not missing, f"Missing columns: {missing}"

        # item_id
        merge_df["item_id"] = merge_df["parent_asin"]
        merge_df["user_id"] = merge_df["user_id"].astype(str)
        merge_df["item_id"] = merge_df["item_id"].astype(str)

        # Drop null keys
        print("\n[1.6] Dropping null parent_asin...")
        merge_df = merge_df.dropna(subset=["parent_asin"])

        # User filter
        print("\n[1.7/1.8] User filter...")

        user_counts  = merge_df["user_id"].value_counts()
        active_users = user_counts[user_counts >= 5].index
        merge_df     = merge_df[merge_df["user_id"].isin(active_users)]

        rows  = len(merge_df)
        users = merge_df["user_id"].nunique()
        items = merge_df["item_id"].nunique()

        avg_interactions = rows / users
        density          = rows / (users * items)

        print("\nFINAL STATS:")
        print(f"  Rows   : {rows:,}")
        print(f"  Users  : {users:,}")
        print(f"  Items  : {items:,}")
        print(f"  Avg interactions/user: {avg_interactions:.2f}")
        print(f"  Density: {density:.6f}")


        # =========================
        # Fix price column (CRITICAL)
        # =========================
        if "price" in merge_df.columns:
            print("\n[Fix] Cleaning price column...")
        
            # Extract numeric part from strings like "from 34.00"
            merge_df["price"] = (
                merge_df["price"]
                .astype(str)
                .str.extract(r"(\d+\.?\d*)")[0]   # extract number
            )
        
            # Convert to float
            merge_df["price"] = pd.to_numeric(merge_df["price"], errors="coerce")
        # Save
        print("\n[1.9] Saving...")
        merge_df.to_csv(MERGE_CSV_PATH, index=False)
        merge_df.to_parquet(CLEAN_PARQUET_PATH, index=False)

        merge_df[["user_id", "item_id", "rating", "timestamp"]].to_parquet(
            CF_DATA_PATH, index=False
        )

        # MLflow
        mlflow.log_metrics({
            "rows": rows,
            "users": users,
            "items": items,
            "avg_interactions": avg_interactions,
            "density": density
        })

    print("\n✅ Phase 1 COMPLETED!")


if __name__ == "__main__":
    main()