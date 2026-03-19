import os
import torch

# =========================
# BASE DIR
# =========================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# =========================
# DATASET CONFIG
# =========================
CATEGORIES = [
    "Video_Games",
    "Musical_Instruments",
    "Software"
]

CHUNK_SIZE = 100_000

# =========================
# PATHS
# =========================
DATA_DIR        = os.path.join(BASE_DIR, "data")
EMBEDDINGS_DIR  = os.path.join(BASE_DIR, "embeddings")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR     = os.path.join(BASE_DIR, "outputs")
MLFLOW_DIR      = os.path.join(BASE_DIR, "mlflow")
SRC_DIR         = os.path.join(BASE_DIR, "src")
API_DIR         = os.path.join(BASE_DIR, "api")

# -------------------------
# DATA FILES
# -------------------------
MERGE_CSV_PATH      = os.path.join(DATA_DIR, "merge_df.csv")
CLEAN_PARQUET_PATH  = os.path.join(DATA_DIR, "clean_merge_df.parquet")

CF_DATA_PATH        = os.path.join(DATA_DIR, "cleaned_cf_dataset.parquet")
USER_MAP_PATH       = os.path.join(DATA_DIR, "user_map.json")
ITEM_MAP_PATH       = os.path.join(DATA_DIR, "item_map.json")

# -------------------------
# EMBEDDINGS
# -------------------------
REVIEW_EMBEDS_PATH  = os.path.join(EMBEDDINGS_DIR, "review_embeds.npy")
META_EMBEDS_PATH    = os.path.join(EMBEDDINGS_DIR, "meta_embeds.npy")

# -------------------------
# MODELS
# -------------------------
MF_MODEL_PATH       = os.path.join(MODELS_DIR, "mf_model.pth")
NCF_MODEL_PATH      = os.path.join(MODELS_DIR, "ncf_model.pth")
ALS_MODEL_PATH      = os.path.join(MODELS_DIR, "als_model.npz")
SVDPP_MODEL_PATH    = os.path.join(MODELS_DIR, "svdpp_model.pkl")
HYBRID_MODEL_PATH   = os.path.join(MODELS_DIR, "hybrid_recommender.pkl")

SVM_MODEL_PATH      = os.path.join(MODELS_DIR, "svm_model.pkl")
VECTORIZER_PATH     = os.path.join(MODELS_DIR, "svm_vectorizer.pkl")

# -------------------------
# OUTPUTS
# -------------------------
SUMMARY_OUTPUT_PATH = os.path.join(OUTPUTS_DIR, "final_top500_product_summary.csv")
AB_RESULTS_PATH     = os.path.join(OUTPUTS_DIR, "ab_comparison_results.csv")

# =========================
# DEVICE
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# MLflow CONFIG
# =========================
MLFLOW_TRACKING_URI = "file:./mlflow"
MLFLOW_EXPERIMENT   = "DS11"

# =========================
# HUGGINGFACE CONFIG
# =========================
HF_DATASET_NAME = "McAuley-Lab/Amazon-Reviews-2023"

# =========================
# REPRODUCIBILITY
# =========================
RANDOM_STATE = 42


# =========================
# CREATE DIRS (STANDARDIZED)
# =========================
def create_dirs():
    dirs = [
        DATA_DIR,
        EMBEDDINGS_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
        MLFLOW_DIR,
        SRC_DIR,
        API_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# =========================
# MLflow SETUP
# =========================
def setup_mlflow():
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)


# =========================
# SEED CONTROL (VERY IMPORTANT)
# =========================
def set_seed():
    import random
    import numpy as np
    import torch

    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)


# =========================
# OPTIONAL ENTRY POINT
# =========================
if __name__ == "__main__":
    create_dirs()
    setup_mlflow()
    set_seed()
    print("✅ Config initialized successfully.")