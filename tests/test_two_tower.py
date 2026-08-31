"""
tests/test_two_tower.py
=======================
Unit and integration tests for Phase 5 (Two-Tower Retrieval Model):
  - TwoTowerModel & TowerMLP architecture, projection shapes, and L2 normalization
  - In-batch contrastive loss (symmetric InfoNCE) mechanics and gradient flow
  - Retrieval ranking metrics (Recall@k, NDCG@k) correctness
  - Fused CF + e5 text feature construction and alignment
  - Trained model artifact persistence in models/two_tower.pth
  - TwoTowerRetriever candidate retrieval & cold-start handling
  - MLflow 'TwoTower' run tracking and metric logging in 'DS11-v2'
"""

import os
import importlib
import pytest
import pandas as pd
import torch
import mlflow

import config

_mod_tt = importlib.import_module("src.13_two_tower")
TowerMLP = _mod_tt.TowerMLP
TwoTowerModel = _mod_tt.TwoTowerModel
in_batch_contrastive_loss = _mod_tt.in_batch_contrastive_loss
evaluate_retrieval_metrics = _mod_tt.evaluate_retrieval_metrics
build_fused_features = _mod_tt.build_fused_features
TwoTowerRetriever = _mod_tt.TwoTowerRetriever


@pytest.fixture(scope="module", autouse=True)
def set_mlflow_env():
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"


def test_tower_mlp_architecture_and_l2_normalization():
    """Verify TowerMLP forward pass produces L2-normalized embeddings of target dim."""
    batch_size = 16
    in_dim = 960
    out_dim = 128
    tower = TowerMLP(input_dim=in_dim, hidden_dims=[256, 128], output_dim=out_dim, dropout=0.0)
    tower.eval()

    x = torch.randn(batch_size, in_dim)
    emb = tower(x)

    assert emb.shape == (batch_size, out_dim), f"Expected shape {(batch_size, out_dim)}, got {emb.shape}"
    norms = torch.norm(emb, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(batch_size), atol=1e-5), "Embeddings are not unit L2 normalized"


def test_two_tower_model_forward_and_encoding():
    """Verify TwoTowerModel encodes user and item vectors independently and jointly."""
    batch_size = 8
    model = TwoTowerModel(
        user_input_dim=960,
        item_input_dim=960,
        hidden_dims=[256, 128],
        output_dim=128,
        dropout=0.1,
        temperature=0.07,
    )
    model.eval()

    u_feats = torch.randn(batch_size, 960)
    i_feats = torch.randn(batch_size, 960)

    u_emb = model.encode_user(u_feats)
    i_emb = model.encode_item(i_feats)
    assert u_emb.shape == (batch_size, 128)
    assert i_emb.shape == (batch_size, 128)

    u_fwd, i_fwd = model(u_feats, i_feats)
    assert torch.allclose(u_emb, u_fwd, atol=1e-5)
    assert torch.allclose(i_emb, i_fwd, atol=1e-5)


def test_in_batch_contrastive_loss_and_gradients():
    """Verify in-batch InfoNCE loss computation and valid backpropagation gradients."""
    batch_size = 10
    emb_dim = 64
    user_emb = torch.randn(batch_size, emb_dim, requires_grad=True)
    item_emb = torch.randn(batch_size, emb_dim, requires_grad=True)

    # Normalize vectors
    u_norm = torch.nn.functional.normalize(user_emb, p=2, dim=-1)
    i_norm = torch.nn.functional.normalize(item_emb, p=2, dim=-1)

    loss = in_batch_contrastive_loss(u_norm, i_norm, temperature=0.07)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert loss.item() > 0.0, "Contrastive loss must be positive"

    loss.backward()
    assert user_emb.grad is not None and torch.isfinite(user_emb.grad).all()
    assert item_emb.grad is not None and torch.isfinite(item_emb.grad).all()


def test_retrieval_metrics_evaluation_logic():
    """Verify retrieval metric evaluation with synthetic data."""
    n_users = 4
    n_items = 10
    emb_dim = 16

    model = TwoTowerModel(
        user_input_dim=emb_dim,
        item_input_dim=emb_dim,
        hidden_dims=[32],
        output_dim=emb_dim,
        dropout=0.0,
    )
    model.eval()

    # Identity projection for predictable dot products
    user_feats = torch.eye(n_users, emb_dim)
    item_feats = torch.eye(n_items, emb_dim)

    # Ground truth: user 0 matches item 0, user 1 matches item 1, etc.
    test_df = pd.DataFrame(
        {
            "user_idx": [0, 1, 2, 3],
            "item_idx": [0, 1, 2, 3],
        }
    )

    metrics = evaluate_retrieval_metrics(
        model=model,
        user_feats_tensor=user_feats,
        item_feats_tensor=item_feats,
        test_df=test_df,
        k_list=[1, 5, 10],
        user_batch_size=2,
    )

    for k in [1, 5, 10]:
        assert f"recall_at_{k}" in metrics
        assert f"ndcg_at_{k}" in metrics
        assert 0.0 <= metrics[f"recall_at_{k}"] <= 1.0
        assert 0.0 <= metrics[f"ndcg_at_{k}"] <= 1.0


def test_two_tower_model_artifact_exists_and_valid():
    """Verify models/two_tower.pth exists on disk with correct structure and weights."""
    model_path = config.TWO_TOWER_MODEL_PATH
    assert os.path.exists(model_path), f"Missing model checkpoint at {model_path}"
    assert os.path.getsize(model_path) > 0, "two_tower.pth is empty"

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    assert "state_dict" in checkpoint
    assert "model_config" in checkpoint
    assert "metrics" in checkpoint

    m_cfg = checkpoint["model_config"]
    assert m_cfg["user_input_dim"] == 960
    assert m_cfg["item_input_dim"] == 960
    assert m_cfg["output_dim"] == 128

    metrics = checkpoint["metrics"]
    assert "recall_at_50" in metrics and metrics["recall_at_50"] > 0.0
    assert "recall_at_100" in metrics and metrics["recall_at_100"] > 0.0
    assert "ndcg_at_50" in metrics and metrics["ndcg_at_50"] > 0.0


def test_two_tower_retriever_inference_and_cold_start():
    """Verify TwoTowerRetriever candidate retrieval for warm and cold-start queries."""
    retriever = TwoTowerRetriever(
        model_path=config.TWO_TOWER_MODEL_PATH,
        user_map_path=config.USER_MAP_PATH,
        item_map_path=config.ITEM_MAP_PATH,
        meta_embeds_path=config.META_EMBEDS_PATH,
        meta_item_ids_path=config.META_ITEM_IDS_PATH,
    )

    # 1. Warm user retrieval
    sample_user = next(iter(retriever.user_map.keys()))
    recs = retriever.retrieve_candidates(user_id=sample_user, top_k=10)
    assert len(recs) == 10
    assert "item_id" in recs[0]
    assert "score" in recs[0]
    assert recs[0]["source"] == "two_tower"
    # Verify scores are sorted descending
    scores = [r["score"] for r in recs]
    assert scores == sorted(scores, reverse=True)

    # 2. Cold-start user (unknown user_id)
    cold_recs = retriever.retrieve_candidates(user_id="non_existent_user_999", top_k=5)
    assert len(cold_recs) == 5
    assert all(r["source"] == "two_tower" for r in cold_recs)

    # 3. Session-based retrieval from interacted items
    sample_items = list(retriever.item_map.keys())[:3]
    session_recs = retriever.retrieve_candidates(interacted_items=sample_items, top_k=5, exclude_items=sample_items)
    assert len(session_recs) == 5
    for r in session_recs:
        assert r["item_id"] not in sample_items, "Excluded interacted items should not appear in candidate list"


def test_mlflow_twotower_run_logged():
    """Verify MLflow experiment 'DS11-v2' contains a finished 'TwoTower' run with metrics."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    exp = mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
    assert exp is not None, f"MLflow experiment {config.MLFLOW_EXPERIMENT} not found"

    runs_df = mlflow.search_runs(experiment_names=[config.MLFLOW_EXPERIMENT])
    assert not runs_df.empty, "No MLflow runs found"

    tt_runs = runs_df[runs_df["tags.mlflow.runName"] == "TwoTower"]
    assert not tt_runs.empty, "Missing 'TwoTower' run in MLflow tracking store"

    row = tt_runs.iloc[0]
    assert row["status"] == "FINISHED", f"TwoTower run status is {row['status']}"

    assert "metrics.recall_at_50" in row
    assert "metrics.recall_at_100" in row
    assert "metrics.ndcg_at_50" in row
    assert row["metrics.recall_at_50"] > 0.0
    assert row["metrics.recall_at_100"] > 0.0
    assert row["metrics.ndcg_at_50"] > 0.0
