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
# CREATE DIRS
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

print(f"Using device: {DEVICE}")

# =========================
# DATASET
# =========================
class RatingDataset(Dataset):
    def __init__(self, df):
        self.users   = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.items   = torch.tensor(df["item_idx"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values.astype(np.float32), dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]


# =========================
# MF MODEL
# =========================
class MF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim):
        super().__init__()
        self.user_emb  = nn.Embedding(n_users, emb_dim)
        self.item_emb  = nn.Embedding(n_items, emb_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)

        nn.init.normal_(self.user_emb.weight,  std=0.01)
        nn.init.normal_(self.item_emb.weight,  std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, u, i):
        dot = (self.user_emb(u) * self.item_emb(i)).sum(dim=1)
        return dot + self.user_bias(u).squeeze() + self.item_bias(i).squeeze()


# =========================
# NCF MODEL
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
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.final = nn.Linear(emb_dim + 32, 1)

        nn.init.normal_(self.user_emb_gmf.weight, std=0.01)
        nn.init.normal_(self.item_emb_gmf.weight, std=0.01)
        nn.init.normal_(self.user_emb_mlp.weight, std=0.01)
        nn.init.normal_(self.item_emb_mlp.weight, std=0.01)

    def forward(self, u, i):
        gmf     = self.user_emb_gmf(u) * self.item_emb_gmf(i)
        mlp_out = self.mlp(
            torch.cat([self.user_emb_mlp(u), self.item_emb_mlp(i)], dim=1)
        )
        return self.final(torch.cat([gmf, mlp_out], dim=1)).squeeze()


# =========================
# TRAIN
# =========================
def train_model(model, train_loader, epochs, weight_decay=1e-5):
    model.to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.5
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for u, i, r in train_loader:
            u, i, r = u.to(DEVICE), i.to(DEVICE), r.to(DEVICE)
            pred = model(u, i)
            loss = nn.MSELoss()(pred, r)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        epoch_rmse = np.sqrt(total_loss / len(train_loader))
        print(f"  Epoch {epoch+1}/{epochs}  RMSE: {epoch_rmse:.4f}  LR: {scheduler.get_last_lr()[0]:.6f}")

    return model


# =========================
# EVALUATE RMSE
# =========================
def evaluate_rmse(model, df):
    model.eval()
    preds, targets = [], []

    loader = DataLoader(
        RatingDataset(df),
        batch_size=4096,
        pin_memory=(DEVICE.type == "cuda")
    )

    with torch.no_grad():
        for u, i, r in loader:
            out = model(u.to(DEVICE), i.to(DEVICE)).cpu().numpy()
            preds.extend(out)
            targets.extend(r.numpy())

    return float(np.sqrt(np.mean((np.array(preds) - np.array(targets)) ** 2)))


# =========================
# EVALUATE RANKING
# =========================
def evaluate_ranking(model, test_df, n_items, k=10, seed_lookup=None):
    model.eval()
    recall_list, ndcg_list = [], []

    for user_idx, group in tqdm(
        test_df.groupby("user_idx"), desc="  Ranking eval", leave=False
    ):
        true_items = group["item_idx"].values

        if seed_lookup and user_idx not in seed_lookup:
            continue

        negatives = np.setdiff1d(
            np.random.choice(n_items, 600, replace=False),
            true_items
        )[:500]

        candidates = np.concatenate([true_items, negatives])

        with torch.no_grad():
            u_tensor = torch.tensor(
                [user_idx] * len(candidates), dtype=torch.long
            ).to(DEVICE)
            i_tensor = torch.tensor(
                candidates, dtype=torch.long
            ).to(DEVICE)
            scores = model(u_tensor, i_tensor).cpu().numpy()

        ranked_indices = np.argsort(-scores)
        ranked_items   = candidates[ranked_indices]

        true_set = set(true_items.tolist())
        top_k    = set(ranked_items[:k].tolist())
        hits     = len(top_k & true_set)

        recall_list.append(1 if hits > 0 else 0)

        ndcg = 0.0
        for rank, item in enumerate(ranked_items[:k]):
            if item in true_set:
                ndcg = 1 / np.log2(rank + 2)
                break
        ndcg_list.append(ndcg)

    return float(np.mean(recall_list)), float(np.mean(ndcg_list))


# =========================
# UMAP + METRICS PLOT (BONUS)
# FIX: accepts active_run_id so the plot can be logged inside the NCF run
# =========================
def plot_umap_and_metrics(mf_model, ncf_model, metrics):
    try:
        import umap
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("  UMAP skipped — install: pip install umap-learn matplotlib")
        return

    print("\nGenerating UMAP + metric chart...")

    mf_embs  = mf_model.item_emb.weight.detach().cpu().numpy()
    ncf_embs = ncf_model.item_emb_gmf.weight.detach().cpu().numpy()

    n   = min(10000, len(mf_embs))
    idx = np.random.choice(len(mf_embs), n, replace=False)

    reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
    mf_2d   = reducer.fit_transform(mf_embs[idx])
    ncf_2d  = reducer.fit_transform(ncf_embs[idx])

    fig = plt.figure(figsize=(18, 6))
    gs  = gridspec.GridSpec(1, 3, figure=fig)

    ax0 = fig.add_subplot(gs[0])
    ax0.scatter(mf_2d[:, 0],  mf_2d[:, 1],  s=2, alpha=0.4)
    ax0.set_title("MF item embeddings (UMAP)")
    ax0.set_xlabel("UMAP-1"); ax0.set_ylabel("UMAP-2")

    ax1 = fig.add_subplot(gs[1])
    ax1.scatter(ncf_2d[:, 0], ncf_2d[:, 1], s=2, alpha=0.4, color="orange")
    ax1.set_title("NCF item embeddings (UMAP)")
    ax1.set_xlabel("UMAP-1"); ax1.set_ylabel("UMAP-2")

    ax2 = fig.add_subplot(gs[2])
    names    = ["RMSE", "Recall@10", "NDCG@10"]
    mf_vals  = [metrics["MF"]["rmse"],  metrics["MF"]["recall@10"],  metrics["MF"]["ndcg@10"]]
    ncf_vals = [metrics["NCF"]["rmse"], metrics["NCF"]["recall@10"], metrics["NCF"]["ndcg@10"]]

    x = np.arange(len(names))
    w = 0.35
    ax2.bar(x - w/2, mf_vals,  w, label="MF",  color="steelblue")
    ax2.bar(x + w/2, ncf_vals, w, label="NCF", color="orange")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_title("MF vs NCF metrics")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("outputs/phase6_umap_metrics.png", dpi=150)
    plt.close()
    print("  Saved → outputs/phase6_umap_metrics.png")


# =========================
# MAIN
# =========================
def main():
    # ---- Load data ----
    print("[6.1] Loading data...")
    df = pd.read_parquet("data/clean_merge_df.parquet")
    df = df[["user_id", "item_id", "rating", "timestamp"]].copy()
    df = df.dropna(subset=["user_id", "item_id", "rating"])
    df["rating"] = df["rating"].astype(np.float32)
    print(f"  Total interactions: {len(df)}")

    # ---- Encode IDs ----
    print("[6.2] Encoding IDs...")
    unique_users = df["user_id"].unique()
    unique_items = df["item_id"].unique()

    user_map = {u: int(i) for i, u in enumerate(unique_users)}
    item_map = {it: int(j) for j, it in enumerate(unique_items)}

    df["user_idx"] = df["user_id"].map(user_map).astype(int)
    df["item_idx"] = df["item_id"].map(item_map).astype(int)

    with open("data/user_map.json", "w") as f:
        json.dump(user_map, f)
    with open("data/item_map.json", "w") as f:
        json.dump(item_map, f)

    print(f"  Users: {len(user_map)}  Items: {len(item_map)}")

    # ---- Train/Test split ----
    print("[6.3] Train/Test split (leave-1-out by timestamp)...")
    df = df.sort_values(["user_idx", "timestamp"])
    test_idx  = df.groupby("user_idx").tail(1).index
    test_df   = df.loc[test_idx].reset_index(drop=True)
    train_df  = df.drop(test_idx).reset_index(drop=True)

    train_df.to_parquet("data/train_df.parquet", index=False)
    test_df.to_parquet("data/test_df.parquet",   index=False)
    train_df[["user_id", "item_id", "rating", "timestamp"]].to_parquet(
        "data/cleaned_cf_dataset.parquet", index=False
    )

    print(f"  Train: {len(train_df)}  Test: {len(test_df)}")
    print(f"  Test users: {test_df['user_id'].nunique()}")

    # ---- DataLoader ----
    print("[6.4] Building DataLoader...")
    train_loader = DataLoader(
        RatingDataset(train_df),
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=(DEVICE.type == "cuda"),
        num_workers=0
    )

    n_users = len(user_map)
    n_items = len(item_map)

    seed_lookup = (
        train_df.sort_values(["user_idx", "timestamp"])
        .groupby("user_idx")["item_idx"]
        .last()
        .to_dict()
    )

    mlflow.set_tracking_uri("mlflow/")
    mlflow.set_experiment("DS11")

    # ================= MF =================
    print("\n[6.5] Training MF...")
    mf_model = MF(n_users, n_items, EMB_DIM)

    with mlflow.start_run(run_name="MF"):
        mf_model  = train_model(mf_model, train_loader, epochs=15)
        mf_rmse   = evaluate_rmse(mf_model, test_df)
        mf_recall, mf_ndcg = evaluate_ranking(
            mf_model, test_df, n_items, seed_lookup=seed_lookup
        )

        print(f"  MF  RMSE={mf_rmse:.4f}  Recall@10={mf_recall:.4f}  NDCG@10={mf_ndcg:.4f}")

        mlflow.log_params({
            "emb_dim": EMB_DIM, "epochs": 8,
            "lr": LR, "batch_size": BATCH_SIZE
        })
        mlflow.log_metrics({
            "rmse": mf_rmse,
            "recall_at_10": mf_recall,
            "ndcg_at_10": mf_ndcg
        })

        torch.save(mf_model.state_dict(), "models/mf_model.pth")
        mlflow.log_artifact("models/mf_model.pth")

    # ================= NCF =================
    print("\n[6.6] Training NCF...")
    ncf_model = NCF(n_users, n_items, EMB_DIM)

    # ---- Save metrics dict before entering NCF run ----
    # (needed by plot_umap_and_metrics which runs inside the NCF run)
    metrics = {
        "MF":  {
            "rmse": float(mf_rmse),
            "recall@10": float(mf_recall),
            "ndcg@10": float(mf_ndcg)
        },
        "NCF": {}   # filled after NCF training below
    }

    with mlflow.start_run(run_name="NCF"):
        ncf_model  = train_model(ncf_model, train_loader, epochs=15)
        ncf_rmse   = evaluate_rmse(ncf_model, test_df)
        ncf_recall, ncf_ndcg = evaluate_ranking(
            ncf_model, test_df, n_items, seed_lookup=seed_lookup
        )

        print(f"  NCF RMSE={ncf_rmse:.4f}  Recall@10={ncf_recall:.4f}  NDCG@10={ncf_ndcg:.4f}")

        mlflow.log_params({
            "emb_dim": EMB_DIM, "epochs": 10,
            "lr": LR, "batch_size": BATCH_SIZE
        })
        mlflow.log_metrics({
            "rmse": ncf_rmse,
            "recall_at_10": ncf_recall,
            "ndcg_at_10": ncf_ndcg
        })

        torch.save(ncf_model.state_dict(), "models/ncf_model.pth")
        mlflow.log_artifact("models/ncf_model.pth")

        # FIX: generate UMAP plot INSIDE the NCF run so log_artifact works
        metrics["NCF"] = {
            "rmse": float(ncf_rmse),
            "recall@10": float(ncf_recall),
            "ndcg@10": float(ncf_ndcg)
        }
        plot_umap_and_metrics(mf_model, ncf_model, metrics)
        # FIX: log the UMAP plot to MLflow — was missing before
        mlflow.log_artifact("outputs/phase6_umap_metrics.png")

    # ================= SAVE METRICS JSON =================
    print("\nSaving metrics...")
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("  Saved → models/metrics.json")

    print("\nPhase 6 complete.")


if __name__ == "__main__":
    main()