"""
tests/test_ranker.py
====================
Unit and integration tests for Phase 4:
  - Ranker feature dataset generation & schema invariants
  - Popularity-weighted negative sampling logic
  - Ranking metric evaluation helpers (NDCG@k, MAP@k)
  - LightGBM model artifact persistence and inference
  - Cold-start handling in RankerService
  - MLflow 'Ranker' run logging & tracking verification
"""

import os
import pickle
import importlib
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import pytest

import config

_mod_features = importlib.import_module("src.12_ranker_features")
sample_popularity_negatives = _mod_features.sample_popularity_negatives

_mod_ranker = importlib.import_module("src.12_ranker")
FEATURE_COLS = _mod_ranker.FEATURE_COLS
compute_ndcg_at_k = _mod_ranker.compute_ndcg_at_k
compute_map_at_k = _mod_ranker.compute_map_at_k
evaluate_ranking_metrics = _mod_ranker.evaluate_ranking_metrics
RankerService = _mod_ranker.RankerService

pytestmark = pytest.mark.skipif(
    not os.path.exists(config.RANKER_TRAIN_PATH),
    reason="Ranker training artifacts not present (run 'dvc repro' to generate)",
)


def test_ranker_train_parquet_exists_and_schema():
    """Verify data/ranker_train.parquet exists with correct schema and ordering."""
    assert os.path.exists(config.RANKER_TRAIN_PATH), f"Missing {config.RANKER_TRAIN_PATH}"
    df = pd.read_parquet(config.RANKER_TRAIN_PATH)
    assert len(df) > 0, "ranker_train.parquet is empty"

    expected_cols = [
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
    for col in expected_cols:
        assert col in df.columns, f"Missing column {col} in ranker_train.parquet"

    # Query groups must be contiguous by user_id
    user_ids = df["user_id"].values
    user_changes = np.sum(user_ids[:-1] != user_ids[1:])
    unique_users = len(np.unique(user_ids))
    assert user_changes == unique_users - 1, "Query groups are not contiguous by user_id"

    # Check relevance labels are non-negative
    assert (df["relevance_label"] >= 0.0).all(), "Negative relevance label found"
    assert (df["relevance_label"] == 0.0).any(), "No negative samples (relevance=0) found"
    assert (df["relevance_label"] > 0.0).any(), "No positive samples (relevance>0) found"


def test_popularity_negative_sampling():
    """Verify popularity-weighted negative sampling excludes positive items and produces correct counts."""
    pos_data = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u3"],
            "item_id": ["i1", "i2", "i2", "i3"],
            "rating": [5.0, 4.0, 5.0, 3.0],
        }
    )
    all_items = np.array(["i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8"])
    weights = np.ones(len(all_items)) / len(all_items)

    neg_df = sample_popularity_negatives(
        pos_df=pos_data,
        all_item_ids=all_items,
        item_weights=weights,
        n_negatives=3,
        random_state=42,
    )

    assert len(neg_df) == 9  # 3 users * 3 negatives
    assert (neg_df["rating"] == 0.0).all()
    assert (neg_df["is_positive"] == 0).all()

    # Verify no positive collisions
    for _, row in neg_df.iterrows():
        u = row["user_id"]
        pos_items = set(pos_data[pos_data["user_id"] == u]["item_id"])
        assert row["item_id"] not in pos_items, f"Negative item {row['item_id']} collided with user {u} positive items"


def test_ranking_metrics_ndcg_and_map():
    """Verify NDCG@k and MAP@k calculations on known synthetic cases."""
    # Perfect ranking
    y_true_perfect = np.array([5.0, 4.0, 3.0, 0.0, 0.0])
    y_pred_perfect = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    ndcg_perfect = compute_ndcg_at_k(y_true_perfect, y_pred_perfect, k=5)
    map_perfect = compute_map_at_k(y_true_perfect, y_pred_perfect, k=5, rel_threshold=3.0)
    assert np.isclose(ndcg_perfect, 1.0), f"Expected perfect NDCG=1.0, got {ndcg_perfect}"
    assert np.isclose(map_perfect, 1.0), f"Expected perfect MAP=1.0, got {map_perfect}"

    # Reverse ranking
    y_true_rev = np.array([5.0, 0.0, 0.0])
    y_pred_rev = np.array([0.1, 0.8, 0.9])
    ndcg_rev = compute_ndcg_at_k(y_true_rev, y_pred_rev, k=3)
    assert ndcg_rev < 1.0, f"Expected degraded NDCG, got {ndcg_rev}"

    # All zeros
    y_true_zero = np.array([0.0, 0.0, 0.0])
    y_pred_zero = np.array([0.5, 0.4, 0.3])
    assert compute_ndcg_at_k(y_true_zero, y_pred_zero, k=3) == 0.0
    assert compute_map_at_k(y_true_zero, y_pred_zero, k=3) == 0.0

    # Multi-group evaluation helper
    y_true_all = np.array([5.0, 4.0, 0.0, 0.0, 4.0])
    y_pred_all = np.array([0.9, 0.8, 0.1, 0.9, 0.1])
    groups = np.array([3, 2])
    mean_ndcg, mean_map = evaluate_ranking_metrics(y_true_all, y_pred_all, groups, k=5)
    assert 0.0 <= mean_ndcg <= 1.0
    assert 0.0 <= mean_map <= 1.0


def test_lgbm_ranker_artifacts_and_prediction():
    """Verify saved LGBMRanker artifacts can be loaded and produce valid ranking predictions."""
    assert os.path.exists(config.LGBM_RANKER_PATH), f"Missing {config.LGBM_RANKER_PATH}"
    assert os.path.exists(config.LGBM_RANKER_PKL_PATH), f"Missing {config.LGBM_RANKER_PKL_PATH}"

    # Load from booster file
    booster = lgb.Booster(model_file=config.LGBM_RANKER_PATH)
    dummy_input = np.random.randn(5, len(FEATURE_COLS))
    scores = booster.predict(dummy_input)
    assert len(scores) == 5
    assert not np.isnan(scores).any()

    # Load from pickle
    with open(config.LGBM_RANKER_PKL_PATH, "rb") as f:
        pickled_model = pickle.load(f)
    pkl_scores = pickled_model.predict(dummy_input)
    assert np.allclose(scores, pkl_scores, atol=1e-5)


def test_ranker_service_serving_and_cold_start():
    """Verify RankerService candidate ranking with both warm and cold-start user paths."""
    service = RankerService(model_path=config.LGBM_RANKER_PATH)
    assert service.booster is not None

    candidates_df = pd.DataFrame(
        {
            "item_id": ["item_A", "item_B", "item_C"],
            "als_score": [1.2, 0.5, 0.1],
            "svdpp_score": [4.5, 4.0, 3.2],
            "mf_score": [1.0, 0.8, 0.2],
            "ncf_score": [4.8, 3.9, 3.0],
            "content_score": [2.5, 2.8, 1.9],
            "apriori_lift": [10.0, 0.0, 0.0],
            "price_score": [0.8, 0.6, 0.5],
            "recency": [0.9, 0.7, 0.4],
            "popularity": [3.5, 4.0, 2.1],
            "helpful_votes": [25.0, 50.0, 5.0],
        }
    )

    # Warm user ranking
    ranked_warm = service.rank(candidates_df, is_cold_start_user=False)
    assert "ranker_score" in ranked_warm.columns
    assert len(ranked_warm) == 3
    # Sorted descending by ranker_score
    assert (ranked_warm["ranker_score"].diff().dropna() <= 0).all()

    # Cold-start user ranking (CF features zeroed out)
    ranked_cold = service.rank(candidates_df, is_cold_start_user=True)
    assert "ranker_score" in ranked_cold.columns
    assert len(ranked_cold) == 3
    assert (ranked_cold["ranker_score"].diff().dropna() <= 0).all()


def test_mlflow_ranker_run_exists():
    """Verify MLflow experiment 'DS11-v2' has a completed 'Ranker' run with expected metrics."""
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    exp = mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
    assert exp is not None, f"Experiment {config.MLFLOW_EXPERIMENT} not found"

    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    ranker_runs = runs[runs["tags.mlflow.runName"] == "Ranker"]
    assert len(ranker_runs) > 0, "No MLflow run named 'Ranker' found in DS11-v2"

    latest_run = ranker_runs.iloc[0]
    assert latest_run["status"] == "FINISHED"
    assert "metrics.ndcg_at_10" in latest_run and pd.notna(latest_run["metrics.ndcg_at_10"])
    assert "metrics.map_at_10" in latest_run and pd.notna(latest_run["metrics.map_at_10"])
    assert latest_run["metrics.ndcg_at_10"] > 0.8
    assert latest_run["metrics.map_at_10"] > 0.8
