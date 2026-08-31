"""
src/13_two_tower.py
===================
Phase 5 — PyTorch Two-Tower Retrieval Model

Learns unified user and item representations by fusing Collaborative Filtering
embeddings (MF/NCF) with e5 text semantic embeddings (meta_embeds.npy), trained
with in-batch negative contrastive loss (symmetric InfoNCE).

Workflow:
  1. Build fused multi-modal features:
     - Item features: [MF item emb (64) + NCF GMF (64) + NCF MLP (64) + e5 meta text (768)] = 960-dim
     - User features: [MF user emb (64) + NCF GMF (64) + NCF MLP (64) + user history text (768)] = 960-dim
  2. Instantiate TwoTowerModel (User Tower MLP & Item Tower MLP with L2-normalized projections).
  3. Train using symmetric In-Batch Negative Contrastive Loss (InfoNCE).
  4. Evaluate retrieval ranking metrics on test set:
     Recall@10, Recall@20, Recall@50, Recall@100, NDCG@10, NDCG@50, NDCG@100.
  5. Serialize trained model artifact to models/two_tower.pth.
  6. Log run to MLflow experiment 'DS11-v2' with run_name='TwoTower'.
  7. Provide cold-start aware TwoTowerRetriever serving layer for ANN / online candidate retrieval.
"""

import os
import sys
import json
import time
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import mlflow

# Setup paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR) if os.path.basename(SRC_DIR) == "src" else SRC_DIR
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config


# ===========================================================================
# 1. Dataset & Feature Preparation
# ===========================================================================


def build_fused_features(
    train_df_path: str = "data/train_df.parquet",
    user_map_path: str = "data/user_map.json",
    item_map_path: str = "data/item_map.json",
    meta_embeds_path: str = "embeddings/meta_embeds.npy",
    meta_item_ids_path: str = "embeddings/meta_item_ids.json",
    mf_model_path: str = "models/mf_model.pth",
    ncf_model_path: str = "models/ncf_model.pth",
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[str, int], List[str]]:
    """
    Constructs aligned user and item fused feature matrices combining CF embeddings
    (MF + NCF) and e5 text embeddings.

    Returns:
        user_features: np.ndarray of shape (n_users, 960)
        item_features: np.ndarray of shape (n_items, 960)
        user_map: dict mapping user_id -> user_idx
        item_map: dict mapping item_id -> item_idx
        meta_items: list of meta item_ids matching meta_embeds
    """
    # 1. Load mappings
    with open(user_map_path, "r", encoding="utf-8") as f:
        user_map: Dict[str, int] = json.load(f)
    with open(item_map_path, "r", encoding="utf-8") as f:
        item_map: Dict[str, int] = json.load(f)
    with open(meta_item_ids_path, "r", encoding="utf-8") as f:
        meta_items: List[str] = json.load(f)

    n_users = len(user_map)
    n_items = len(item_map)

    # 2. Align e5 Item Text Embeddings
    meta_embeds = np.load(meta_embeds_path)  # (N_meta, 768)
    meta_item_to_idx = {asin: i for i, asin in enumerate(meta_items)}

    item_text_mat = np.zeros((n_items, meta_embeds.shape[1]), dtype=np.float32)
    for item_id, item_idx in item_map.items():
        if item_id in meta_item_to_idx:
            item_text_mat[item_idx] = meta_embeds[meta_item_to_idx[item_id]]

    # 3. Construct User Text Profile (mean e5 embedding of interacted training items)
    train_df = pd.read_parquet(train_df_path)
    user_text_mat = np.zeros((n_users, meta_embeds.shape[1]), dtype=np.float32)
    user_counts = np.zeros(n_users, dtype=np.float32)

    u_indices = train_df["user_idx"].values
    i_indices = train_df["item_idx"].values
    for u_idx, i_idx in zip(u_indices, i_indices):
        user_text_mat[u_idx] += item_text_mat[i_idx]
        user_counts[u_idx] += 1.0

    user_text_mat /= np.maximum(user_counts[:, None], 1.0)
    user_norms = np.linalg.norm(user_text_mat, axis=1, keepdims=True)
    user_text_mat = np.divide(user_text_mat, np.maximum(user_norms, 1e-9))

    # 4. Extract CF Embeddings from MF and NCF
    mf_sd = torch.load(mf_model_path, map_location="cpu", weights_only=True)
    ncf_sd = torch.load(ncf_model_path, map_location="cpu", weights_only=True)

    user_cf = np.hstack(
        [
            mf_sd["user_emb.weight"].cpu().numpy(),  # (n_users, 64)
            ncf_sd["user_emb_gmf.weight"].cpu().numpy(),  # (n_users, 64)
            ncf_sd["user_emb_mlp.weight"].cpu().numpy(),  # (n_users, 64)
        ]
    ).astype(np.float32)

    item_cf = np.hstack(
        [
            mf_sd["item_emb.weight"].cpu().numpy(),  # (n_items, 64)
            ncf_sd["item_emb_gmf.weight"].cpu().numpy(),  # (n_items, 64)
            ncf_sd["item_emb_mlp.weight"].cpu().numpy(),  # (n_items, 64)
        ]
    ).astype(np.float32)

    # 5. Concatenate CF + Text features
    user_features = np.hstack([user_cf, user_text_mat]).astype(np.float32)
    item_features = np.hstack([item_cf, item_text_mat]).astype(np.float32)

    return user_features, item_features, user_map, item_map, meta_items


class InteractionDataset(Dataset):
    """
    PyTorch Dataset of positive (user_idx, item_idx) interaction pairs.
    """

    def __init__(self, u_indices: np.ndarray, i_indices: np.ndarray):
        self.u_indices = torch.tensor(u_indices, dtype=torch.long)
        self.i_indices = torch.tensor(i_indices, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.u_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.u_indices[idx], self.i_indices[idx]


# ===========================================================================
# 2. PyTorch Two-Tower Architecture
# ===========================================================================


class TowerMLP(nn.Module):
    """
    Multi-Layer Perceptron Tower with Batch Normalization, ReLU activations,
    Dropout regularization, and L2 projection normalization.
    """

    def __init__(
        self,
        input_dim: int = 960,
        hidden_dims: List[int] = [512, 256],
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        in_d = input_dim
        for h_d in hidden_dims:
            layers.append(nn.Linear(in_d, h_d))
            layers.append(nn.BatchNorm1d(h_d))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_d = h_d
        layers.append(nn.Linear(in_d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return F.normalize(out, p=2, dim=-1)


class TwoTowerModel(nn.Module):
    """
    PyTorch Two-Tower Retrieval Model for Amazon RecSys v2.
    Learns dense representations for Users and Items in a shared metric space.
    """

    def __init__(
        self,
        user_input_dim: int = 960,
        item_input_dim: int = 960,
        hidden_dims: List[int] = [512, 256],
        output_dim: int = 128,
        dropout: float = 0.1,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.user_input_dim = user_input_dim
        self.item_input_dim = item_input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.temperature_val = temperature

        self.user_tower = TowerMLP(
            input_dim=user_input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
        )
        self.item_tower = TowerMLP(
            input_dim=item_input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
        )
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32), requires_grad=False)

    def encode_user(self, user_features: torch.Tensor) -> torch.Tensor:
        """Projects user feature vectors to L2-normalized embedding space."""
        return self.user_tower(user_features)

    def encode_item(self, item_features: torch.Tensor) -> torch.Tensor:
        """Projects item feature vectors to L2-normalized embedding space."""
        return self.item_tower(item_features)

    def forward(self, user_features: torch.Tensor, item_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass encoding both user and item representations."""
        user_emb = self.encode_user(user_features)
        item_emb = self.encode_item(item_features)
        return user_emb, item_emb


# ===========================================================================
# 3. Loss Function & Evaluation Metrics
# ===========================================================================


def in_batch_contrastive_loss(
    user_emb: torch.Tensor, item_emb: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    """
    Computes symmetric InfoNCE contrastive loss with in-batch negatives.

    Args:
        user_emb: (batch_size, emb_dim) L2-normalized user vectors
        item_emb: (batch_size, emb_dim) L2-normalized item vectors
        temperature: Softmax temperature parameter tau

    Returns:
        Scalar contrastive cross-entropy loss
    """
    batch_size = user_emb.shape[0]
    logits = torch.matmul(user_emb, item_emb.T) / temperature  # (B, B)
    labels = torch.arange(batch_size, device=user_emb.device, dtype=torch.long)
    loss_u2i = F.cross_entropy(logits, labels)
    loss_i2u = F.cross_entropy(logits.T, labels)
    return (loss_u2i + loss_i2u) / 2.0


def evaluate_retrieval_metrics(
    model: TwoTowerModel,
    user_feats_tensor: torch.Tensor,
    item_feats_tensor: torch.Tensor,
    test_df: pd.DataFrame,
    k_list: List[int] = [10, 20, 50, 100],
    user_batch_size: int = 2000,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Evaluates candidate retrieval ranking metrics (Recall@k and NDCG@k) across
    all test set users.

    Args:
        model: Trained TwoTowerModel
        user_feats_tensor: Tensor of all user features (n_users, 960)
        item_feats_tensor: Tensor of all item features (n_items, 960)
        test_df: DataFrame containing leave-one-out test user_idx and item_idx
        k_list: List of cutoffs k for Recall@k and NDCG@k
        user_batch_size: User chunk size for memory-efficient matrix multiplication
        device: PyTorch device

    Returns:
        Dictionary of evaluated metrics
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    n_items = item_feats_tensor.shape[0]
    max_k = max(k_list)

    with torch.no_grad():
        # 1. Pre-encode all catalog items in chunks
        all_item_embs_list = []
        chunk_size = 4096
        for i in range(0, n_items, chunk_size):
            chunk_f = item_feats_tensor[i : i + chunk_size].to(device)
            all_item_embs_list.append(model.encode_item(chunk_f))
        all_item_embs = torch.cat(all_item_embs_list, dim=0)  # (n_items, emb_dim)

        # 2. Extract test user indices and ground truth target item indices
        test_u_indices = test_df["user_idx"].values
        test_i_targets = test_df["item_idx"].values
        n_test = len(test_df)

        test_u_tensor = torch.tensor(test_u_indices, dtype=torch.long, device=device)
        test_u_feats = user_feats_tensor[test_u_tensor].to(device)
        test_u_embs = model.encode_user(test_u_feats)  # (n_test, emb_dim)

        # 3. Accumulate hits and DCG scores for each k
        recall_hits = {k: 0.0 for k in k_list}
        ndcg_sums = {k: 0.0 for k in k_list}

        for u_start in range(0, n_test, user_batch_size):
            u_end = min(u_start + user_batch_size, n_test)
            u_chunk_embs = test_u_embs[u_start:u_end]  # (B_chunk, emb_dim)

            # Cosine similarity matrix (B_chunk, n_items)
            sim_scores = torch.matmul(u_chunk_embs, all_item_embs.T)
            top_indices = torch.topk(sim_scores, max_k, dim=1).indices.cpu().numpy()

            targets = test_i_targets[u_start:u_end]
            for row_idx, target in enumerate(targets):
                preds = top_indices[row_idx]
                target_match = np.where(preds == target)[0]

                if len(target_match) > 0:
                    rank = target_match[0] + 1  # 1-based rank
                    for k in k_list:
                        if rank <= k:
                            recall_hits[k] += 1.0
                            ndcg_sums[k] += 1.0 / np.log2(rank + 1.0)

        results = {}
        for k in k_list:
            results[f"recall_at_{k}"] = float(recall_hits[k] / n_test)
            results[f"ndcg_at_{k}"] = float(ndcg_sums[k] / n_test)

    return results


# ===========================================================================
# 4. Training Pipeline & MLflow Logging
# ===========================================================================


def train_two_tower(
    epochs: int = 10,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_dims: List[int] = [512, 256],
    output_dim: int = 128,
    dropout: float = 0.1,
    temperature: float = 0.07,
    model_output_path: str = "models/two_tower.pth",
    device_str: Optional[str] = None,
) -> Tuple[TwoTowerModel, Dict[str, Any]]:
    """
    Trains the PyTorch Two-Tower model on fused CF + text representations,
    evaluates retrieval metrics, serializes checkpoint to model_output_path,
    and logs run to MLflow experiment 'DS11-v2' with run_name='TwoTower'.
    """
    if device_str is None:
        device = torch.device(config.DEVICE)
    else:
        device = torch.device(device_str)

    print(f"\n==========================================")
    print(f"Two-Tower Model Training & Retrieval Evaluation")
    print(f"Device: {device} | Epochs: {epochs} | Batch size: {batch_size}")
    print(f"==========================================")

    # 1. Build & Load Fused Features
    print("[13.1] Building fused CF + e5 text features...")
    user_feats, item_feats, user_map, item_map, meta_items = build_fused_features()
    user_feats_tensor = torch.tensor(user_feats, dtype=torch.float32)
    item_feats_tensor = torch.tensor(item_feats, dtype=torch.float32)

    n_users, user_input_dim = user_feats.shape
    n_items, item_input_dim = item_feats.shape
    print(f"  User feature matrix: {user_feats.shape} (users: {n_users:,}, dim: {user_input_dim})")
    print(f"  Item feature matrix: {item_feats.shape} (items: {n_items:,}, dim: {item_input_dim})")

    # 2. Build DataLoader
    print("[13.2] Loading interaction dataset...")
    train_df = pd.read_parquet("data/train_df.parquet")
    test_df = pd.read_parquet("data/test_df.parquet")

    train_dataset = InteractionDataset(
        u_indices=train_df["user_idx"].values,
        i_indices=train_df["item_idx"].values,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    print(f"  Train interactions: {len(train_df):,} | Test users: {len(test_df):,}")

    # 3. Model & Optimizer
    print("[13.3] Initializing TwoTowerModel...")
    model = TwoTowerModel(
        user_input_dim=user_input_dim,
        item_input_dim=item_input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        dropout=dropout,
        temperature=temperature,
    ).to(device)

    user_feats_device = user_feats_tensor.to(device)
    item_feats_device = item_feats_tensor.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 4. Training Loop
    print("\n[13.4] Starting training loop...")
    history_loss: List[float] = []
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for u_batch, i_batch in train_loader:
            u_batch = u_batch.to(device)
            i_batch = i_batch.to(device)

            u_f = user_feats_device[u_batch]
            i_f = item_feats_device[i_batch]

            u_emb, i_emb = model(u_f, i_f)
            loss = in_batch_contrastive_loss(u_emb, i_emb, temperature=model.temperature_val)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        history_loss.append(avg_loss)
        print(
            f"  Epoch {epoch:2d}/{epochs:2d} | "
            f"In-Batch Contrastive Loss: {avg_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

    train_time = time.time() - t_start
    print(f"Training completed in {train_time:.2f}s.")

    # 5. Evaluate Retrieval Metrics
    print("\n[13.5] Evaluating Candidate Retrieval Metrics on Test Set...")
    metrics = evaluate_retrieval_metrics(
        model=model,
        user_feats_tensor=user_feats_tensor,
        item_feats_tensor=item_feats_tensor,
        test_df=test_df,
        k_list=[10, 20, 50, 100],
        user_batch_size=2000,
        device=device,
    )
    metrics["train_loss"] = float(history_loss[-1])
    metrics["training_time_sec"] = float(train_time)

    print(f"==========================================")
    print(f"Two-Tower Retrieval Results (All {len(test_df):,} Catalog Test Items):")
    print(f"  Recall@10:  {metrics['recall_at_10']:.4f}  |  NDCG@10:  {metrics['ndcg_at_10']:.4f}")
    print(f"  Recall@20:  {metrics['recall_at_20']:.4f}  |  NDCG@20:  {metrics['ndcg_at_20']:.4f}")
    print(f"  Recall@50:  {metrics['recall_at_50']:.4f}  |  NDCG@50:  {metrics['ndcg_at_50']:.4f}")
    print(f"  Recall@100: {metrics['recall_at_100']:.4f}  |  NDCG@100: {metrics['ndcg_at_100']:.4f}")
    print(f"==========================================")

    # 6. Save Model Artifact
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "model_config": {
            "user_input_dim": user_input_dim,
            "item_input_dim": item_input_dim,
            "hidden_dims": hidden_dims,
            "output_dim": output_dim,
            "dropout": dropout,
            "temperature": temperature,
        },
        "metrics": metrics,
        "n_users": n_users,
        "n_items": n_items,
    }
    torch.save(checkpoint, model_output_path)
    print(f"\nSaved Two-Tower checkpoint to {model_output_path}")

    # 7. Log to MLflow
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    print(f"Logging to MLflow experiment '{config.MLFLOW_EXPERIMENT}' as 'TwoTower'...")
    with mlflow.start_run(run_name="TwoTower"):
        mlflow.log_params(
            {
                "model_type": "TwoTowerRetrieval",
                "user_input_dim": user_input_dim,
                "item_input_dim": item_input_dim,
                "hidden_dims": json.dumps(hidden_dims),
                "output_dim": output_dim,
                "dropout": dropout,
                "temperature": temperature,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "n_users": n_users,
                "n_items": n_items,
            }
        )
        for metric_name, val in metrics.items():
            mlflow.log_metric(metric_name, val)

        mlflow.log_artifact(model_output_path)

    print("TwoTower MLflow logging complete.")
    return model, metrics


# ===========================================================================
# 5. Serving & Inference Service
# ===========================================================================


class TwoTowerRetriever:
    """
    Serving and candidate generation service using the trained TwoTowerModel.
    Supports warm and cold-start user/item retrieval.
    """

    def __init__(
        self,
        model_path: str = "models/two_tower.pth",
        user_map_path: str = "data/user_map.json",
        item_map_path: str = "data/item_map.json",
        meta_embeds_path: str = "embeddings/meta_embeds.npy",
        meta_item_ids_path: str = "embeddings/meta_item_ids.json",
        clean_parquet_path: str = "data/clean_merge_df.parquet",
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # 1. Load checkpoint and initialize model
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        m_cfg = checkpoint["model_config"]

        self.model = TwoTowerModel(
            user_input_dim=m_cfg["user_input_dim"],
            item_input_dim=m_cfg["item_input_dim"],
            hidden_dims=m_cfg["hidden_dims"],
            output_dim=m_cfg["output_dim"],
            dropout=m_cfg.get("dropout", 0.1),
            temperature=m_cfg["temperature"],
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # 2. Load mappings
        with open(user_map_path, "r", encoding="utf-8") as f:
            self.user_map: Dict[str, int] = json.load(f)
        with open(item_map_path, "r", encoding="utf-8") as f:
            self.item_map: Dict[str, int] = json.load(f)
        self.idx_to_item: Dict[int, str] = {v: k for k, v in self.item_map.items()}

        with open(meta_item_ids_path, "r", encoding="utf-8") as f:
            self.meta_items: List[str] = json.load(f)
        self.meta_item_to_idx: Dict[str, int] = {asin: i for i, asin in enumerate(self.meta_items)}
        self.meta_embeds = np.load(meta_embeds_path)

        # 3. Load fused user & item features
        user_feats, item_feats, _, _, _ = build_fused_features(
            user_map_path=user_map_path,
            item_map_path=item_map_path,
            meta_embeds_path=meta_embeds_path,
            meta_item_ids_path=meta_item_ids_path,
        )
        self.item_feats_tensor = torch.tensor(item_feats, dtype=torch.float32)
        self.user_feats_tensor = torch.tensor(user_feats, dtype=torch.float32)
        self.mean_user_feat = torch.tensor(np.mean(user_feats, axis=0), dtype=torch.float32)

        # 4. Precompute catalog item embeddings
        with torch.no_grad():
            self.catalog_item_embs = []
            chunk_size = 4096
            for i in range(0, len(item_feats), chunk_size):
                chunk_f = self.item_feats_tensor[i : i + chunk_size].to(self.device)
                self.catalog_item_embs.append(self.model.encode_item(chunk_f).cpu())
            self.catalog_item_embs = torch.cat(self.catalog_item_embs, dim=0)  # (n_items, emb_dim)

    def encode_user_id(self, user_id: str) -> torch.Tensor:
        """
        Encodes a single user_id into a normalized 128-dim dense embedding.
        Falls back to mean user representation for cold-start users.
        """
        with torch.no_grad():
            if user_id in self.user_map:
                u_idx = self.user_map[user_id]
                u_feat = self.user_feats_tensor[u_idx : u_idx + 1].to(self.device)
            else:
                u_feat = self.mean_user_feat.unsqueeze(0).to(self.device)
            return self.model.encode_user(u_feat).squeeze(0).cpu()

    def encode_user_from_history(self, interacted_item_ids: List[str]) -> torch.Tensor:
        """
        Encodes an arbitrary user session from a list of interacted item_ids (cold-start user profile).
        """
        with torch.no_grad():
            # 1. User text profile from interacted items
            valid_vecs = []
            for item_id in interacted_item_ids:
                if item_id in self.meta_item_to_idx:
                    valid_vecs.append(self.meta_embeds[self.meta_item_to_idx[item_id]])

            if valid_vecs:
                text_prof = np.mean(valid_vecs, axis=0)
                norm = np.linalg.norm(text_prof)
                text_prof = text_prof / max(norm, 1e-9)
            else:
                text_prof = np.zeros(768, dtype=np.float32)

            # 2. Cold user CF features (zeros)
            cf_feats = np.zeros(192, dtype=np.float32)
            user_feat = np.hstack([cf_feats, text_prof]).astype(np.float32)
            u_feat_tensor = torch.tensor(user_feat, dtype=torch.float32).unsqueeze(0).to(self.device)
            return self.model.encode_user(u_feat_tensor).squeeze(0).cpu()

    def retrieve_candidates(
        self,
        user_id: Optional[str] = None,
        interacted_items: Optional[List[str]] = None,
        top_k: int = 50,
        exclude_items: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-k candidate items using Two-Tower cosine similarity against catalog.

        Returns:
            List of dicts: [{'item_id': str, 'score': float, 'source': 'two_tower'}]
        """
        if user_id and user_id in self.user_map:
            u_emb = self.encode_user_id(user_id)
        elif interacted_items:
            u_emb = self.encode_user_from_history(interacted_items)
        elif user_id:
            u_emb = self.encode_user_id(user_id)
        else:
            u_emb = self.encode_user_id("unknown_user")

        with torch.no_grad():
            # Dot product with all catalog item embeddings
            scores = torch.matmul(self.catalog_item_embs, u_emb).numpy()

        exclude_set = set(exclude_items or [])
        sorted_indices = np.argsort(-scores)

        results = []
        for idx in sorted_indices:
            item_id = self.idx_to_item[int(idx)]
            if item_id in exclude_set:
                continue
            results.append(
                {
                    "item_id": item_id,
                    "score": float(scores[idx]),
                    "source": "two_tower",
                }
            )
            if len(results) >= top_k:
                break

        return results


# ===========================================================================
# 6. Main Script Execution
# ===========================================================================


def main():
    config.create_dirs()
    config.set_seed()
    train_two_tower(
        epochs=10,
        batch_size=1024,
        lr=1e-3,
        weight_decay=1e-4,
        hidden_dims=[512, 256],
        output_dim=128,
        dropout=0.1,
        temperature=0.07,
        model_output_path=config.TWO_TOWER_MODEL_PATH,
    )


if __name__ == "__main__":
    main()
