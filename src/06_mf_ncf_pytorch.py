import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import mlflow

# =========================
# CREATE DIRS (USER PREF)
# =========================
def create_dirs():
    for d in ["data", "models", "outputs", "mlflow"]:
        os.makedirs(d, exist_ok=True)

create_dirs()

# =========================
# CONFIG
# =========================
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4096
EMB_DIM    = 64
LR         = 1e-3


# =========================
# DATASET (6.4)
# =========================
class RatingDataset(Dataset):
    def __init__(self, df):
        self.users   = df["user_idx"].values
        self.items   = df["item_idx"].values
        self.ratings = df["rating"].values.astype(np.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.users[idx]),
            torch.tensor(self.items[idx]),
            torch.tensor(self.ratings[idx])
        )


# =========================
# MF MODEL (6.5)
# =========================
class MF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim):
        super().__init__()
        self.user_emb  = nn.Embedding(n_users, emb_dim)
        self.item_emb  = nn.Embedding(n_items, emb_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, u, i):
        dot = (self.user_emb(u) * self.item_emb(i)).sum(dim=1)
        return dot + self.user_bias(u).squeeze() + self.item_bias(i).squeeze()


# =========================
# NCF MODEL (6.6)
# =========================
class NCF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim):
        super().__init__()

        self.user_emb_gmf = nn.Embedding(n_users, emb_dim)
        self.item_emb_gmf = nn.Embedding(n_items, emb_dim)

        self.user_emb_mlp = nn.Embedding(n_users, emb_dim)
        self.item_emb_mlp = nn.Embedding(n_items, emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.final = nn.Linear(emb_dim + 32, 1)

    def forward(self, u, i):
        gmf     = self.user_emb_gmf(u) * self.item_emb_gmf(i)
        mlp_out = self.mlp(torch.cat([self.user_emb_mlp(u), self.item_emb_mlp(i)], dim=1))
        return self.final(torch.cat([gmf, mlp_out], dim=1)).squeeze()


# =========================
# TRAIN (6.7)
# =========================
def train_model(model, train_loader, epochs):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn   = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for u, i, r in train_loader:
            u, i, r = u.to(DEVICE), i.to(DEVICE), r.to(DEVICE)

            pred = model(u, i)
            loss = loss_fn(pred, r)

            optimizer.zero_grad()
            loss.backward()

            # stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            total_loss += loss.item()

        rmse = np.sqrt(total_loss / len(train_loader))
        print(f"  Epoch {epoch+1}/{epochs}  RMSE: {rmse:.4f}")

    return model


# =========================
# EVALUATION (6.8)
# =========================
def evaluate_rmse(model, df):
    model.eval()
    preds, targets = [], []

    loader = DataLoader(
        RatingDataset(df),
        batch_size=4096,
        pin_memory=True
    )

    with torch.no_grad():
        for u, i, r in loader:
            preds.extend(model(u.to(DEVICE), i.to(DEVICE)).cpu().numpy())
            targets.extend(r.numpy())

    return float(np.sqrt(np.mean((np.array(preds) - np.array(targets)) ** 2)))


def evaluate_ranking(model, df, num_items, k=10):
    model.eval()
    recall_list, ndcg_list = [], []

    for user, group in tqdm(df.groupby("user_idx"), desc="  Ranking eval"):
        true_item = group["item_idx"].values[0]

        negatives = np.random.choice(num_items, 500, replace=False)
        negatives = negatives[negatives != true_item]

        items = np.append(negatives, true_item)

        with torch.no_grad():
            scores = model(
                torch.tensor([user] * len(items)).to(DEVICE),
                torch.tensor(items).to(DEVICE)
            ).cpu().numpy()

        ranked = items[np.argsort(-scores)]

        if true_item in ranked[:k]:
            rank = int(np.where(ranked == true_item)[0][0])
            recall_list.append(1)
            ndcg_list.append(1 / np.log2(rank + 2))
        else:
            recall_list.append(0)
            ndcg_list.append(0)

    return float(np.mean(recall_list)), float(np.mean(ndcg_list))


# =========================
# UMAP + METRICS (6.12 BONUS)
# =========================
def plot_umap_and_metrics(mf_model, ncf_model, metrics):
    try:
        import umap
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("  UMAP skipped → install: pip install umap-learn")
        return

    print("\n[6.12] Generating UMAP + metric chart...")

    mf_embs  = mf_model.item_emb.weight.detach().cpu().numpy()
    ncf_embs = ncf_model.item_emb_gmf.weight.detach().cpu().numpy()

    idx = np.random.choice(len(mf_embs), min(10000, len(mf_embs)), replace=False)
    reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=1)

    mf_2d  = reducer.fit_transform(mf_embs[idx])
    ncf_2d = reducer.fit_transform(ncf_embs[idx])

    fig = plt.figure(figsize=(18, 6))
    gs  = gridspec.GridSpec(1, 3, figure=fig)

    ax0 = fig.add_subplot(gs[0])
    ax0.scatter(mf_2d[:, 0], mf_2d[:, 1], s=2, alpha=0.4)
    ax0.set_title("MF Embeddings")

    ax1 = fig.add_subplot(gs[1])
    ax1.scatter(ncf_2d[:, 0], ncf_2d[:, 1], s=2, alpha=0.4)
    ax1.set_title("NCF Embeddings")

    ax2 = fig.add_subplot(gs[2])
    names = ["RMSE", "Recall@10", "NDCG@10"]
    mf_vals  = [metrics["MF"]["rmse"], metrics["MF"]["recall@10"], metrics["MF"]["ndcg@10"]]
    ncf_vals = [metrics["NCF"]["rmse"], metrics["NCF"]["recall@10"], metrics["NCF"]["ndcg@10"]]

    x = np.arange(len(names))
    w = 0.35

    ax2.bar(x - w/2, mf_vals,  w, label="MF")
    ax2.bar(x + w/2, ncf_vals, w, label="NCF")

    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.legend()

    plt.savefig("outputs/phase6_umap_metrics.png")
    plt.close()

    print("  Saved → outputs/phase6_umap_metrics.png")


# =========================
# MAIN
# =========================
def main():
    print("[6.1] Loading data...")
    df = pd.read_parquet("data/clean_merge_df.parquet")
    df = df[["user_id", "item_id", "rating", "timestamp"]].copy()

    # Save CF dataset for Phase 9
    df.to_parquet("data/cleaned_cf_dataset.parquet")

    print("[6.2] Encoding IDs...")
    user_map = {u: i for i, u in enumerate(df["user_id"].unique())}
    item_map = {it: j for j, it in enumerate(df["item_id"].unique())}

    df["user_idx"] = df["user_id"].map(user_map)
    df["item_idx"] = df["item_id"].map(item_map)

    json.dump(user_map, open("data/user_map.json", "w"))
    json.dump(item_map, open("data/item_map.json", "w"))

    print("[6.3] Train/Test split...")
    df = df.sort_values("timestamp")
    test_idx = df.groupby("user_idx").tail(1).index

    test_df  = df.loc[test_idx]
    train_df = df.drop(test_idx)

    test_df.to_parquet("data/test_df.parquet")

    print("[6.4] DataLoader...")
    train_loader = DataLoader(
        RatingDataset(train_df),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True
    )

    n_users = len(user_map)
    n_items = len(item_map)

    mlflow.set_tracking_uri("mlflow/")
    mlflow.set_experiment("DS11")

    # ================= MF =================
    print("\nTraining MF...")
    mf_model = MF(n_users, n_items, EMB_DIM)

    with mlflow.start_run(run_name="MF"):
        mf_model = train_model(mf_model, train_loader, 8)

        mf_rmse = evaluate_rmse(mf_model, test_df)
        mf_recall, mf_ndcg = evaluate_ranking(mf_model, test_df, n_items)

        mlflow.log_params({"emb_dim": EMB_DIM, "epochs": 8, "lr": LR})
        mlflow.log_metrics({"rmse": mf_rmse, "recall_at_10": mf_recall, "ndcg_at_10": mf_ndcg})

        torch.save(mf_model.state_dict(), "models/mf_model.pth")
        mlflow.log_artifact("models/mf_model.pth")

    # ================= NCF =================
    print("\nTraining NCF...")
    ncf_model = NCF(n_users, n_items, EMB_DIM)

    with mlflow.start_run(run_name="NCF"):
        ncf_model = train_model(ncf_model, train_loader, 10)

        ncf_rmse = evaluate_rmse(ncf_model, test_df)
        ncf_recall, ncf_ndcg = evaluate_ranking(ncf_model, test_df, n_items)

        mlflow.log_params({"emb_dim": EMB_DIM, "epochs": 10, "lr": LR})
        mlflow.log_metrics({"rmse": ncf_rmse, "recall_at_10": ncf_recall, "ndcg_at_10": ncf_ndcg})

        torch.save(ncf_model.state_dict(), "models/ncf_model.pth")
        mlflow.log_artifact("models/ncf_model.pth")

    # ================= SAVE METRICS =================
    print("\nSaving metrics...")
    metrics = {
        "MF": {
            "rmse": float(mf_rmse),
            "recall@10": float(mf_recall),
            "ndcg@10": float(mf_ndcg)
        },
        "NCF": {
            "rmse": float(ncf_rmse),
            "recall@10": float(ncf_recall),
            "ndcg@10": float(ncf_ndcg)
        }
    }

    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    plot_umap_and_metrics(mf_model, ncf_model, metrics)

    print("\n✅ Phase 6 completed successfully 🚀")


if __name__ == "__main__":
    main()