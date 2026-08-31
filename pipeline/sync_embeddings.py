"""
pipeline/sync_embeddings.py
Phase 1 — Qdrant Vector DB Embedding Sync Pipeline

Upserts precomputed item metadata embeddings from `embeddings/meta_embeds.npy`
and `embeddings/meta_item_ids.json` into Qdrant collection with payload metadata
(item_id, category, price, title, average_rating, rating_number, store) for fast
HNSW Approximate Nearest Neighbor (ANN) search and filtered retrieval.
"""

import os
import sys
import json
import uuid
import argparse
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Allow running from repo root OR from pipeline/
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import (
    META_EMBEDS_PATH,
    META_ITEM_IDS_PATH,
    CLEAN_PARQUET_PATH,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
)

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    PayloadSchemaType,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
)


def item_id_to_point_id(item_id: str) -> str:
    """
    Generate a deterministic UUID v5 point ID from an item_id string (parent_asin).
    Preserves strict 1:1 mapping between item_id and Qdrant point ID.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(item_id)))


def get_qdrant_client(
    url: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    api_key: Optional[str] = None,
) -> QdrantClient:
    """
    Instantiate and return a QdrantClient instance.
    """
    if url:
        return QdrantClient(url=url, api_key=api_key)
    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, api_key=api_key or QDRANT_API_KEY)

    target_host = host or QDRANT_HOST
    target_port = port or QDRANT_PORT
    return QdrantClient(host=target_host, port=target_port, api_key=api_key or QDRANT_API_KEY)


def load_embeddings_and_metadata(
    embeds_path: str = META_EMBEDS_PATH,
    item_ids_path: str = META_ITEM_IDS_PATH,
    parquet_path: str = CLEAN_PARQUET_PATH,
) -> Tuple[np.ndarray, List[str], Dict[str, Dict[str, Any]]]:
    """
    Load precomputed embeddings, item ID list, and metadata lookup table.
    """
    if not os.path.exists(embeds_path):
        raise FileNotFoundError(f"Embeddings file not found at: {embeds_path}")
    if not os.path.exists(item_ids_path):
        raise FileNotFoundError(f"Item IDs file not found at: {item_ids_path}")

    print(f"[1/4] Loading precomputed embeddings from {embeds_path} ...")
    embeds = np.load(embeds_path)
    if embeds.ndim == 1:
        embeds = np.expand_dims(embeds, axis=0)

    print(f"[2/4] Loading item IDs from {item_ids_path} ...")
    with open(item_ids_path, "r", encoding="utf-8") as f:
        item_ids = json.load(f)

    if len(embeds) != len(item_ids):
        raise ValueError(f"Dimension mismatch: {len(embeds)} embedding vectors vs {len(item_ids)} item IDs.")

    print(f"[3/4] Building metadata lookup from {parquet_path} ...")
    meta_lookup: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)

        # Deduplicate per item_id (parent_asin) taking first non-null properties
        grouped = df.groupby("item_id").first().reset_index()
        for _, row in grouped.iterrows():
            iid = str(row["item_id"])
            title = str(row.get("title_meta", "") or row.get("title_rev", "") or "")
            category = str(
                row.get("main_category_meta", "")
                or row.get("main_category_rev", "")
                or row.get("main_category", "")
                or ""
            )

            try:
                price = float(row.get("price", 0.0))
                if np.isnan(price):
                    price = 0.0
            except (TypeError, ValueError):
                price = 0.0

            try:
                avg_rating = float(row.get("average_rating", 0.0))
                if np.isnan(avg_rating):
                    avg_rating = 0.0
            except (TypeError, ValueError):
                avg_rating = 0.0

            try:
                rating_num = float(row.get("rating_number", 0.0))
                if np.isnan(rating_num):
                    rating_num = 0.0
            except (TypeError, ValueError):
                rating_num = 0.0

            store = str(row.get("store", "") or "")

            meta_lookup[iid] = {
                "item_id": iid,
                "title": title,
                "category": category,
                "price": price,
                "average_rating": avg_rating,
                "rating_number": rating_num,
                "store": store,
            }
    else:
        print(f"  [Warning] Parquet file not found at {parquet_path}. Using fallback metadata.")

    return embeds, item_ids, meta_lookup


def init_collection(
    client: QdrantClient,
    collection_name: str = QDRANT_COLLECTION_NAME,
    vector_size: int = 768,
    recreate: bool = False,
) -> None:
    """
    Initialize Qdrant collection and payload indexes if not present.
    """
    exists = client.collection_exists(collection_name)
    if exists and recreate:
        print(f"Recreating collection '{collection_name}' ...")
        client.delete_collection(collection_name)
        exists = False

    if not exists:
        print(f"Creating collection '{collection_name}' (dim={vector_size}, metric=COSINE) ...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

        # Create payload field indexes for efficient filtering
        indexes = [
            ("category", PayloadSchemaType.KEYWORD),
            ("price", PayloadSchemaType.FLOAT),
            ("item_id", PayloadSchemaType.KEYWORD),
            ("average_rating", PayloadSchemaType.FLOAT),
        ]
        for field_name, field_schema in indexes:
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )
            except Exception as e:
                print(f"  Note on index {field_name}: {e}")
    else:
        print(f"Collection '{collection_name}' already exists.")


def sync_embeddings(
    client: Optional[QdrantClient] = None,
    collection_name: str = QDRANT_COLLECTION_NAME,
    batch_size: int = 500,
    recreate: bool = False,
    embeds_path: str = META_EMBEDS_PATH,
    item_ids_path: str = META_ITEM_IDS_PATH,
    parquet_path: str = CLEAN_PARQUET_PATH,
) -> Dict[str, Any]:
    """
    Main orchestration function to sync embeddings into Qdrant.
    """
    if client is None:
        client = get_qdrant_client()

    embeds, item_ids, meta_lookup = load_embeddings_and_metadata(
        embeds_path=embeds_path,
        item_ids_path=item_ids_path,
        parquet_path=parquet_path,
    )

    vector_size = embeds.shape[1]
    total_items = len(item_ids)
    print(f"\n[4/4] Syncing {total_items:,} items into Qdrant collection '{collection_name}' ...")

    init_collection(
        client=client,
        collection_name=collection_name,
        vector_size=vector_size,
        recreate=recreate,
    )

    points_batch: List[PointStruct] = []
    total_upserted = 0

    for i in tqdm(range(total_items), desc="Upserting batches"):
        iid = str(item_ids[i])
        point_id = item_id_to_point_id(iid)
        vector = embeds[i].tolist()

        payload = meta_lookup.get(
            iid,
            {
                "item_id": iid,
                "title": "",
                "category": "",
                "price": 0.0,
                "average_rating": 0.0,
                "rating_number": 0.0,
                "store": "",
            },
        )

        points_batch.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
        )

        if len(points_batch) >= batch_size:
            client.upsert(collection_name=collection_name, points=points_batch)
            total_upserted += len(points_batch)
            points_batch = []

    if points_batch:
        client.upsert(collection_name=collection_name, points=points_batch)
        total_upserted += len(points_batch)

    count_info = client.count(collection_name=collection_name)
    total_points = count_info.count
    print(f"\n✅ Successfully synced {total_upserted:,} items. Total collection points: {total_points:,}")

    return {
        "collection_name": collection_name,
        "upserted_points": total_upserted,
        "total_points": total_points,
        "vector_size": vector_size,
    }


def search_similar_items(
    client: QdrantClient,
    query_vector: Optional[List[float]] = None,
    item_id: Optional[str] = None,
    collection_name: str = QDRANT_COLLECTION_NAME,
    top_k: int = 5,
    category_filter: Optional[str] = None,
    price_ceiling: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Search for similar items using Approximate Nearest Neighbor (ANN) search,
    optionally filtering by category and/or price ceiling.
    """
    if query_vector is None and item_id is None:
        raise ValueError("Either query_vector or item_id must be provided.")

    if query_vector is None and item_id is not None:
        point_id = item_id_to_point_id(item_id)
        records = client.retrieve(
            collection_name=collection_name,
            ids=[point_id],
            with_vectors=True,
        )
        if not records or records[0].vector is None:
            raise ValueError(f"Item ID '{item_id}' (Point ID '{point_id}') not found in collection.")
        query_vector = records[0].vector

    filter_conditions = []
    if category_filter:
        filter_conditions.append(FieldCondition(key="category", match=MatchValue(value=category_filter)))
    if price_ceiling is not None:
        filter_conditions.append(FieldCondition(key="price", range=Range(lte=price_ceiling)))

    query_filter = Filter(must=filter_conditions) if filter_conditions else None

    search_result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
    )

    results = []
    for point in search_result.points:
        results.append(
            {
                "point_id": str(point.id),
                "score": float(point.score),
                "item_id": point.payload.get("item_id") if point.payload else None,
                "title": point.payload.get("title") if point.payload else None,
                "category": point.payload.get("category") if point.payload else None,
                "price": point.payload.get("price") if point.payload else None,
                "average_rating": point.payload.get("average_rating") if point.payload else None,
            }
        )

    return results


def verify_sync(
    client: Optional[QdrantClient] = None,
    collection_name: str = QDRANT_COLLECTION_NAME,
    sample_k: int = 5,
) -> bool:
    """
    Verify that the Qdrant collection is populated and perform test ANN queries.
    """
    if client is None:
        client = get_qdrant_client()

    print(f"\n{'='*60}")
    print(f"  Verifying Qdrant Collection: {collection_name}")
    print(f"{'='*60}")

    if not client.collection_exists(collection_name):
        print(f"❌ Collection '{collection_name}' does not exist.")
        return False

    count = client.count(collection_name).count
    print(f"Total Points in Collection: {count:,}")
    if count == 0:
        print("❌ Collection is empty.")
        return False

    # Perform sample query with a sample point
    sample_res = client.scroll(collection_name=collection_name, limit=1, with_vectors=True)
    points, _ = sample_res
    if not points:
        print("❌ Failed to retrieve a sample point.")
        return False

    sample_point = points[0]
    sample_vector = sample_point.vector
    sample_iid = sample_point.payload.get("item_id")
    sample_title = sample_point.payload.get("title")
    sample_cat = sample_point.payload.get("category")

    print(f"\n[ANN Search Test 1: Unfiltered Nearest Neighbors]")
    print(f"Query Item: [{sample_iid}] {sample_title} (Category: {sample_cat})")

    top_items = search_similar_items(
        client=client,
        query_vector=sample_vector,
        collection_name=collection_name,
        top_k=sample_k,
    )

    for rank, item in enumerate(top_items, 1):
        print(
            f"  #{rank} [Score: {item['score']:.4f}] {item['item_id']} | "
            f"Price: ${item['price']} | Cat: {item['category']} | {item['title'][:60]}"
        )

    # Check that top match has score ~ 1.0 (self-match)
    assert len(top_items) > 0, "Top items query returned empty list."
    assert top_items[0]["score"] >= 0.99, f"Expected top score ~1.0, got {top_items[0]['score']}"

    # Perform filtered query test
    if sample_cat:
        print(f"\n[ANN Search Test 2: Filtered by Category = '{sample_cat}']")
        filtered_items = search_similar_items(
            client=client,
            query_vector=sample_vector,
            collection_name=collection_name,
            top_k=sample_k,
            category_filter=sample_cat,
        )
        for rank, item in enumerate(filtered_items, 1):
            print(
                f"  #{rank} [Score: {item['score']:.4f}] {item['item_id']} | "
                f"Cat: {item['category']} | {item['title'][:60]}"
            )
            assert item["category"] == sample_cat, f"Filter mismatch: expected {sample_cat}, got {item['category']}"

    print(f"\n✅ ANN search verification passed successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Sync item embeddings into Qdrant collection.")
    parser.add_argument("--host", type=str, default=None, help="Qdrant host")
    parser.add_argument("--port", type=int, default=None, help="Qdrant port")
    parser.add_argument("--collection", type=str, default=QDRANT_COLLECTION_NAME, help="Collection name")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for upserts")
    parser.add_argument("--recreate", action="store_true", help="Recreate collection if it exists")
    parser.add_argument("--verify-only", action="store_true", help="Run verification checks only")
    args = parser.parse_args()

    client = get_qdrant_client(host=args.host, port=args.port)

    if args.verify_only:
        success = verify_sync(client=client, collection_name=args.collection)
        sys.exit(0 if success else 1)

    sync_embeddings(
        client=client,
        collection_name=args.collection,
        batch_size=args.batch_size,
        recreate=args.recreate,
    )

    verify_sync(client=client, collection_name=args.collection)


if __name__ == "__main__":
    main()
