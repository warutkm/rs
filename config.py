import os
import torch

# =========================
# BASE DIR & ENV LOADING
# =========================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

_env_file = os.path.join(BASE_DIR, ".env")
if os.path.exists(_env_file):
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_file)
    except ImportError:
        with open(_env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k not in os.environ:
                        os.environ[k] = v

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
META_ITEM_IDS_PATH  = os.path.join(EMBEDDINGS_DIR, "meta_item_ids.json")

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
TWO_TOWER_MODEL_PATH= os.path.join(MODELS_DIR, "two_tower.pth")

RANKER_TRAIN_PATH   = os.path.join(DATA_DIR, "ranker_train.parquet")
RANKER_TEST_PATH    = os.path.join(DATA_DIR, "ranker_test.parquet")
LGBM_RANKER_PATH    = os.path.join(MODELS_DIR, "lgbm_ranker.txt")
LGBM_RANKER_PKL_PATH= os.path.join(MODELS_DIR, "lgbm_ranker.pkl")

RANKER_FEATURES     = [
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
]

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
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
MLFLOW_TRACKING_URI = "file:./mlflow"
MLFLOW_EXPERIMENT   = "DS11-v2"

# =========================
# QDRANT CONFIG (v2)
# =========================
QDRANT_HOST            = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT            = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_URL             = os.getenv("QDRANT_URL", None)
QDRANT_API_KEY         = os.getenv("QDRANT_API_KEY", None)
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "products")

# =========================
# HUGGINGFACE CONFIG
# =========================
HF_DATASET_NAME = "McAuley-Lab/Amazon-Reviews-2023"

# =========================
# REPRODUCIBILITY
# =========================
RANDOM_STATE = 42

# =========================
# LLM CONFIG (v2)
# =========================
LLM_MODEL             = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
GEMINI_API_KEY        = os.getenv("GEMINI_API_KEY", "")
MODEL_VERSION         = os.getenv("MODEL_VERSION", "v2.0")

# =========================
# REDIS CONFIG (v2)
# =========================
REDIS_HOST            = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT            = int(os.getenv("REDIS_PORT", "6379"))
REDIS_URL             = os.getenv("REDIS_URL", None)
REDIS_PASSWORD        = os.getenv("REDIS_PASSWORD", None)
EXPLANATION_CACHE_TTL = int(os.getenv("EXPLANATION_CACHE_TTL", "86400"))  # 24 hours

PIPELINE_DIR          = os.path.join(BASE_DIR, "pipeline")


# =========================
# ADMIN & RETRAIN CONFIG (v2)
# =========================
ADMIN_API_KEY    = os.getenv("ADMIN_API_KEY", "ds11_admin_secret_key_v2")
LOGS_DIR         = os.path.join(BASE_DIR, "logs")
RETRAIN_LOG_PATH = os.path.join(LOGS_DIR, "dvc_repro.log")

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
        PIPELINE_DIR,
        LOGS_DIR,
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