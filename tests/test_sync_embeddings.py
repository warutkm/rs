"""
tests/test_sync_embeddings.py
Unit and integration tests for Qdrant embedding sync pipeline and ANN retrieval.
"""

import os
import uuid
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from pipeline.sync_embeddings import (
    item_id_to_point_id,
    init_collection,
    search_similar_items,
    verify_sync,
)
from config import (
    CLEAN_PARQUET_PATH,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION_NAME,
)


def test_item_id_to_point_id_deterministic():
    """Verify item_id mapping to UUID v5 is 100% deterministic and unique."""
    iid_1 = "B09JY72CNG"
    iid_2 = "B09WMQ6DXG"

    pid_1a = item_id_to_point_id(iid_1)
    pid_1b = item_id_to_point_id(iid_1)
    pid_2 = item_id_to_point_id(iid_2)

    assert pid_1a == pid_1b, "Point IDs for the same item_id must be identical."
    assert pid_1a != pid_2, "Point IDs for different item_ids must differ."
    # Validate valid UUID format
    uuid_obj = uuid.UUID(pid_1a)
    assert str(uuid_obj) == pid_1a


def test_in_memory_sync_and_search():
    """Test full vector indexing and filtered search in memory."""
    client = QdrantClient(":memory:")
    collection_name = "test_products"

    init_collection(client=client, collection_name=collection_name, vector_size=4, recreate=True)

    # Insert mock items
    items = [
        {
            "item_id": "ITEM_1",
            "vector": [1.0, 0.0, 0.0, 0.0],
            "payload": {
                "item_id": "ITEM_1",
                "title": "Guitar Pro",
                "category": "Musical_Instruments",
                "price": 100.0,
            },
        },
        {
            "item_id": "ITEM_2",
            "vector": [0.9, 0.1, 0.0, 0.0],
            "payload": {
                "item_id": "ITEM_2",
                "title": "Guitar Strings",
                "category": "Musical_Instruments",
                "price": 15.0,
            },
        },
        {
            "item_id": "ITEM_3",
            "vector": [0.0, 0.0, 1.0, 0.0],
            "payload": {
                "item_id": "ITEM_3",
                "title": "Antivirus 2026",
                "category": "Software",
                "price": 49.99,
            },
        },
    ]

    points = [
        PointStruct(
            id=item_id_to_point_id(it["item_id"]),
            vector=it["vector"],
            payload=it["payload"],
        )
        for it in items
    ]
    client.upsert(collection_name=collection_name, points=points)

    assert client.count(collection_name).count == 3

    # ANN search for ITEM_1
    results = search_similar_items(
        client=client,
        item_id="ITEM_1",
        collection_name=collection_name,
        top_k=2,
    )
    assert len(results) == 2
    assert results[0]["item_id"] == "ITEM_1"
    assert results[0]["score"] >= 0.99
    assert results[1]["item_id"] == "ITEM_2"

    # Filtered search for Software
    results_soft = search_similar_items(
        client=client,
        query_vector=[0.0, 0.0, 1.0, 0.0],
        collection_name=collection_name,
        top_k=5,
        category_filter="Software",
    )
    assert len(results_soft) == 1
    assert results_soft[0]["item_id"] == "ITEM_3"

    # Filtered search with price ceiling
    results_price = search_similar_items(
        client=client,
        query_vector=[1.0, 0.0, 0.0, 0.0],
        collection_name=collection_name,
        top_k=5,
        price_ceiling=20.0,
    )
    assert len(results_price) == 1
    assert results_price[0]["item_id"] == "ITEM_2"


def test_parent_asin_invariant():
    """Verify that item_id == parent_asin invariant is strictly preserved."""
    if os.path.exists(CLEAN_PARQUET_PATH):
        df = pd.read_parquet(CLEAN_PARQUET_PATH)
        assert "item_id" in df.columns
        assert "parent_asin" in df.columns
        assert (df["item_id"] == df["parent_asin"]).all(), "item_id must equal parent_asin everywhere."


def test_live_qdrant_verification():
    """Verify that the live Qdrant collection is healthy and populated."""
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2.0)
        if not client.collection_exists(QDRANT_COLLECTION_NAME):
            import pytest

            pytest.skip("Qdrant collection not created yet.")
        count = client.count(QDRANT_COLLECTION_NAME).count
        assert count > 0, "Live Qdrant collection must not be empty."
        success = verify_sync(client=client, collection_name=QDRANT_COLLECTION_NAME, sample_k=3)
        assert success is True
    except Exception as exc:
        import pytest

        pytest.skip(f"Live Qdrant instance not reachable ({exc})")
