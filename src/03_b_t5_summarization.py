"""
03b_t5_summarization.py — Phase 3 (Part B): T5 Summarization + Export

Covers:
  3.6 T5 summarization (BONUS)
  3.7 Export summary_df

Run AFTER 03a_sentiment_tfidf_svm.py

Changes vs previous version:
  - All imports moved to top (torch + transformers no longer split across file)
  - device auto-detected via torch.cuda.is_available() instead of hardcoded -1
  - grouped.get_group(asin) guarded with try/except KeyError
  - gc.collect() runs immediately after del summariser (before MLflow I/O)
  - Progress print uses len(top500) instead of hardcoded 500
"""

import os
import gc
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import pyarrow.parquet as pq
import torch
import mlflow
from transformers import pipeline

# =============================================================================
# CREATE DIRS
# =============================================================================
def create_dirs():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("mlflow",  exist_ok=True)

create_dirs()

# =============================================================================
# LOAD DATA — minimal columns only to keep RAM footprint small
# =============================================================================
DATA_PATH = "data/clean_merge_df.parquet"

available_cols = pq.read_schema(DATA_PATH).names

REQUIRED_COLS = ["rating", "text_clean"]
OPTIONAL_COLS = ["item_id", "parent_asin"]

item_col = next((c for c in OPTIONAL_COLS if c in available_cols), None)
if item_col is None:
    raise ValueError(
        "Neither 'item_id' nor 'parent_asin' found in parquet. "
        "Ensure 01_data_ingestion.py ran correctly."
    )

load_cols = REQUIRED_COLS + [item_col]
df = pd.read_parquet(DATA_PATH, columns=load_cols)
print(f"Loaded {len(df):,} rows — columns: {load_cols}")

# =============================================================================
# SENTIMENT — recomputed from rating (no dependency on 03a)
#   0 = negative (rating <= 2)
#   1 = neutral  (rating == 3)
#   2 = positive (rating >= 4)
#   avg_sentiment in the CSV is the mean of these codes (0–2 continuous scale)
# =============================================================================
def rating_to_sentiment_code(r: float) -> int:
    if r <= 2:
        return 0
    elif r == 3:
        return 1
    return 2

df["sentiment_code"] = df["rating"].apply(rating_to_sentiment_code)

# =============================================================================
# TOP-500 PRODUCTS BY REVIEW COUNT
# =============================================================================
top500 = df[item_col].value_counts().head(500).index.tolist()

df_top500 = df[df[item_col].isin(top500)].copy()
del df
gc.collect()

print(f"Top-500 subset: {df_top500.shape}")

# Pre-group once — avoids repeated O(n) boolean indexing inside the loop
grouped = df_top500.groupby(item_col)

# =============================================================================
# LOAD T5 MODEL
#   device auto-detected: GPU if available, CPU otherwise
#   keeping CPU as the safe default for constrained environments
# =============================================================================
T5_MODEL            = "t5-small"
MAX_INPUT_CHARS     = 1_500   # ~350 tokens — within T5-small's 512 token limit
MAX_OUT_LEN         = 120
MIN_OUT_LEN         = 30
REVIEWS_PER_PRODUCT = 50

device = 0 if torch.cuda.is_available() else -1
device_label = f"cuda:{device}" if device >= 0 else "cpu"
print(f"\nLoading {T5_MODEL} on {device_label} …")

summariser = pipeline(
    "summarization",
    model=T5_MODEL,
    tokenizer=T5_MODEL,
    device=device,
)
print("Model ready.")

# =============================================================================
# SUMMARIZATION LOOP
# =============================================================================
records   = []
n_skipped = 0
n_total   = len(top500)

for i, asin in enumerate(top500, 1):

    # Guard against KeyError — possible if groupby dropped an asin
    try:
        product_df = grouped.get_group(asin)
    except KeyError:
        n_skipped += 1
        continue

    # Aggregate stats (cheap — done before the expensive T5 call)
    avg_rating = round(product_df["rating"].mean(), 3)
    avg_sent   = round(product_df["sentiment_code"].mean(), 3)
    n_reviews  = len(product_df)

    # Build input text
    texts = (
        product_df["text_clean"]
        .dropna()
        .astype(str)
        .head(REVIEWS_PER_PRODUCT)
        .tolist()
    )
    combined = " ".join(texts)[:MAX_INPUT_CHARS]

    if len(combined.split()) < 20:
        n_skipped += 1
        continue

    # Summarize
    try:
        with torch.no_grad():
            result = summariser(
                combined,
                max_length=MAX_OUT_LEN,
                min_length=MIN_OUT_LEN,
                do_sample=False,
                truncation=True,
            )
        summary = result[0]["summary_text"]
    except Exception as e:
        summary = f"[error: {e}]"

    records.append({
        "asin":          asin,
        "summary":       summary,
        "avg_sentiment": avg_sent,
        "n_reviews":     n_reviews,
        "avg_rating":    avg_rating,
    })

    if i % 50 == 0:
        print(f"  [{i}/{n_total}] products summarized …")

# =============================================================================
# BUILD + EXPORT SUMMARY CSV
#   outputs/final_top500_product_summary.csv
#   workflow-spec columns: asin, summary, avg_sentiment, n_reviews
#   extra kept:            avg_rating
# =============================================================================
summary_df = pd.DataFrame(records)
print(f"\nDone: {len(summary_df)} summarized | {n_skipped} skipped")

OUTPUT_PATH = "outputs/final_top500_product_summary.csv"
summary_df[
    ["asin", "summary", "avg_sentiment", "n_reviews", "avg_rating"]
].to_csv(OUTPUT_PATH, index=False)
print(f"Saved → {OUTPUT_PATH}")

# Free T5 model before MLflow artifact upload I/O
del summariser, df_top500, grouped
gc.collect()

# =============================================================================
# MLFLOW — T5_summary run
# =============================================================================
if not summary_df.empty:
    avg_len = summary_df["summary"].str.split().str.len().mean()
else:
    avg_len = 0.0
    
mlflow.set_tracking_uri("mlflow/")
mlflow.set_experiment("DS11")

with mlflow.start_run(run_name="T5_summary"):
    mlflow.log_param("model",                T5_MODEL)
    mlflow.log_param("device",               device_label)
    mlflow.log_param("max_input_chars",      MAX_INPUT_CHARS)
    mlflow.log_param("max_output_length",    MAX_OUT_LEN)
    mlflow.log_param("min_output_length",    MIN_OUT_LEN)
    mlflow.log_param("reviews_per_product",  REVIEWS_PER_PRODUCT)
    mlflow.log_param("sentiment_source",     "rating_recomputed")

    mlflow.log_metric("n_summarized",       len(summary_df))
    mlflow.log_metric("n_skipped",          n_skipped)
    mlflow.log_metric("avg_summary_length", round(avg_len, 2))

    mlflow.log_artifact(OUTPUT_PATH)

print(f"MLflow T5_summary run logged — "
      f"n_summarized={len(summary_df)}, avg_summary_length={avg_len:.1f} words")

# =============================================================================
# DONE
# =============================================================================
print("\n✓ Phase 3B complete.")
print(f"  {OUTPUT_PATH}")