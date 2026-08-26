import os
import sys
import pickle
import json
import numpy as np
import scipy.sparse as sp
import torch
import pytest
import mlflow
import pandas as pd

from config import (
    MODELS_DIR,
    OUTPUTS_DIR,
    DATA_DIR,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT,
    ALS_MODEL_PATH,
    SVDPP_MODEL_PATH,
    MF_MODEL_PATH,
    NCF_MODEL_PATH,
    SVM_MODEL_PATH,
    VECTORIZER_PATH,
    SUMMARY_OUTPUT_PATH,
)


@pytest.fixture(scope="module", autouse=True)
def set_mlflow_env():
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


def test_required_model_binaries_exist():
    """Verify that all Phase 3 required model binaries exist on disk."""
    required_files = [
        ALS_MODEL_PATH,
        SVDPP_MODEL_PATH,
        MF_MODEL_PATH,
        NCF_MODEL_PATH,
        os.path.join(MODELS_DIR, "apriori_recommender.pkl"),
        os.path.join(MODELS_DIR, "product_recommender.pkl"),
        os.path.join(MODELS_DIR, "cf_recommender.pkl"),
        SVM_MODEL_PATH,
        VECTORIZER_PATH,
        SUMMARY_OUTPUT_PATH,
        os.path.join(OUTPUTS_DIR, "apriori_rules.csv"),
        os.path.join(DATA_DIR, "train_df.parquet"),
        os.path.join(DATA_DIR, "test_df.parquet"),
    ]
    for path in required_files:
        assert os.path.exists(path), f"Expected artifact missing: {path}"
        assert os.path.getsize(path) > 0, f"Artifact is empty: {path}"


def test_mlflow_experiment_runs():
    """Verify that all Phase 3 training runs are logged to MLflow experiment DS11-v2."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
    assert exp is not None, f"MLflow experiment {MLFLOW_EXPERIMENT} not found"

    runs_df = mlflow.search_runs(experiment_names=[MLFLOW_EXPERIMENT])
    assert not runs_df.empty, f"No runs found in MLflow experiment {MLFLOW_EXPERIMENT}"

    run_names = set(runs_df["tags.mlflow.runName"].dropna())
    expected_runs = {
        "SVM",
        "T5_summary",
        "Apriori",
        "content_only",
        "cf_item_item",
        "MF",
        "NCF",
        "ALS",
        "SVDpp",
    }
    missing_runs = expected_runs - run_names
    assert not missing_runs, f"Missing MLflow runs in {MLFLOW_EXPERIMENT}: {missing_runs}"

    # Check that all expected runs have FINISHED status
    for _, row in runs_df[runs_df["tags.mlflow.runName"].isin(expected_runs)].iterrows():
        assert row["status"] == "FINISHED", f"Run {row['tags.mlflow.runName']} status is {row['status']}"


def test_load_and_infer_apriori():
    """Verify loading and inferring with AprioriRecommender."""
    path = os.path.join(MODELS_DIR, "apriori_recommender.pkl")
    import dill
    with open(path, "rb") as f:
        model = dill.load(f)
    assert hasattr(model, "recommend_apriori")
    assert len(model.rule_dict) > 0
    sample_key = next(iter(model.rule_dict))
    recs = model.recommend_apriori(sample_key, top_k=3)
    assert isinstance(recs, list)
    if recs:
        assert "item_id" in recs[0]
        assert "score" in recs[0]
        assert recs[0]["source"] == "apriori"


def test_load_and_infer_content_and_cf():
    """Verify loading and inferring with ProductRecommender and CollaborativeFilteringRecommender."""
    import importlib
    import dill
    import __main__
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    mod05 = importlib.import_module("05_content_cf_recommender")
    __main__.ProductRecommender = mod05.ProductRecommender
    __main__.CollaborativeFilteringRecommender = mod05.CollaborativeFilteringRecommender

    pr_path = os.path.join(MODELS_DIR, "product_recommender.pkl")
    cf_path = os.path.join(MODELS_DIR, "cf_recommender.pkl")

    with open(pr_path, "rb") as f:
        product_rec = dill.load(f)
    with open(cf_path, "rb") as f:
        cf_rec = dill.load(f)

    assert hasattr(product_rec, "get_recommendations")
    assert hasattr(cf_rec, "recommend_products_cf")

    sample_item = next(iter(product_rec.df.index))
    p_recs = product_rec.get_recommendations(sample_item, top_n=3)
    assert isinstance(p_recs, list)

    sample_cf_item = next(iter(cf_rec.item_map.keys()))
    cf_recs = cf_rec.recommend_products_cf(sample_cf_item, top_k=3)
    assert isinstance(cf_recs, list)


def test_load_pytorch_models():
    """Verify loading PyTorch MF and NCF models."""
    import importlib
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    mod06 = importlib.import_module("06_mf_ncf_pytorch")
    MF = mod06.MF
    NCF = mod06.NCF
    EMB_DIM = mod06.EMB_DIM

    with open(os.path.join(DATA_DIR, "user_map.json"), "r") as f:
        user_map = json.load(f)
    with open(os.path.join(DATA_DIR, "item_map.json"), "r") as f:
        item_map = json.load(f)

    n_users = len(user_map)
    n_items = len(item_map)

    mf = MF(n_users, n_items, EMB_DIM)
    mf.load_state_dict(torch.load(MF_MODEL_PATH, map_location="cpu", weights_only=True))
    mf.eval()

    ncf = NCF(n_users, n_items, EMB_DIM)
    ncf.load_state_dict(torch.load(NCF_MODEL_PATH, map_location="cpu", weights_only=True))
    ncf.eval()

    u = torch.tensor([0], dtype=torch.long)
    i = torch.tensor([0], dtype=torch.long)
    with torch.no_grad():
        mf_out = mf(u, i)
        ncf_out = ncf(u, i)

    assert mf_out.shape == torch.Size([]) or mf_out.numel() == 1
    assert ncf_out.shape == torch.Size([]) or ncf_out.numel() == 1


def test_load_als_and_svdpp():
    """Verify loading implicit ALS model and Surprise SVD++ model."""
    from implicit.als import AlternatingLeastSquares

    als = AlternatingLeastSquares(factors=64)
    # als_model.npz contains factors
    npz_data = np.load(ALS_MODEL_PATH)
    assert "user_factors" in npz_data or "item_factors" in npz_data

    with open(SVDPP_MODEL_PATH, "rb") as f:
        svdpp = pickle.load(f)
    assert hasattr(svdpp, "predict")
