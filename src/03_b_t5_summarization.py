"""
03b_t5_summarization.py — Phase 3 (Part B): T5 Summarization + Export

Covers:
  3.6 T5 summarization (BONUS)
  3.7 Export summary_df

Run AFTER 03a_sentiment_tfidf_svm.py
"""

import os
import gc
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import pyarrow.parquet as pq
import torch
import mlflow
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def create_dirs():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("mlflow", exist_ok=True)


def rating_to_sentiment_code(r: float) -> int:
    if r <= 2:
        return 0
    elif r == 3:
        return 1
    return 2


def main():
    create_dirs()

    # =============================================================================
    # LOAD DATA — minimal columns only to keep RAM footprint small
    # =============================================================================
    data_path = "data/clean_merge_df.parquet"

    available_cols = pq.read_schema(data_path).names

    required_cols = ["rating", "text_clean"]
    optional_cols = ["item_id", "parent_asin"]

    item_col = next((c for c in optional_cols if c in available_cols), None)
    if item_col is None:
        raise ValueError(
            "Neither 'item_id' nor 'parent_asin' found in parquet. "
            "Ensure 01_data_ingestion.py ran correctly."
        )

    load_cols = required_cols + [item_col]
    df = pd.read_parquet(data_path, columns=load_cols)
    print(f"Loaded {len(df):,} rows — columns: {load_cols}")

    # =============================================================================
    # SENTIMENT — recomputed from rating
    # =============================================================================
    df["sentiment_code"] = df["rating"].apply(rating_to_sentiment_code)

    # =============================================================================
    # TOP-500 PRODUCTS BY REVIEW COUNT
    # =============================================================================
    top500 = df[item_col].value_counts().head(500).index.tolist()

    df_top500 = df[df[item_col].isin(top500)].copy()
    del df
    gc.collect()

    print(f"Top-500 subset: {df_top500.shape}")

    grouped = df_top500.groupby(item_col)

    # =============================================================================
    # LOAD T5 MODEL
    # =============================================================================
    t5_model = "t5-small"
    max_input_chars = 1_500   # ~350 tokens — within T5-small's 512 token limit
    max_out_len = 120
    min_out_len = 30
    reviews_per_product = 50

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading {t5_model} on {device_str} …")

    tokenizer = AutoTokenizer.from_pretrained(t5_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(t5_model).to(device_str)
    model.eval()
    print("Model ready.")

    # =============================================================================
    # SUMMARIZATION LOOP
    # =============================================================================
    records = []
    n_skipped = 0
    n_total = len(top500)

    for i, asin in enumerate(top500, 1):
        try:
            product_df = grouped.get_group(asin)
        except KeyError:
            n_skipped += 1
            continue

        avg_rating = round(product_df["rating"].mean(), 3)
        avg_sent = round(product_df["sentiment_code"].mean(), 3)
        n_reviews = len(product_df)

        texts = (
            product_df["text_clean"]
            .dropna()
            .astype(str)
            .head(reviews_per_product)
            .tolist()
        )
        combined = " ".join(texts)[:max_input_chars]

        if len(combined.split()) < 20:
            n_skipped += 1
            continue

        try:
            input_text = "summarize: " + combined
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device_str)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_out_len,
                    min_length=min_out_len,
                    do_sample=False,
                )
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            summary = f"[error: {e}]"

        records.append({
            "asin": asin,
            "summary": summary,
            "avg_sentiment": avg_sent,
            "n_reviews": n_reviews,
            "avg_rating": avg_rating,
        })

        if i % 50 == 0:
            print(f"  [{i}/{n_total}] products summarized …")

    # =============================================================================
    # BUILD + EXPORT SUMMARY CSV
    # =============================================================================
    summary_df = pd.DataFrame(records)
    print(f"\nDone: {len(summary_df)} summarized | {n_skipped} skipped")

    output_path = "outputs/final_top500_product_summary.csv"
    summary_df[
        ["asin", "summary", "avg_sentiment", "n_reviews", "avg_rating"]
    ].to_csv(output_path, index=False)
    print(f"Saved → {output_path}")

    del model, tokenizer, df_top500, grouped
    gc.collect()

    # =============================================================================
    # MLFLOW — T5_summary run
    # =============================================================================
    if not summary_df.empty:
        avg_len = summary_df["summary"].str.split().str.len().mean()
    else:
        avg_len = 0.0

    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri("mlflow/")
    mlflow.set_experiment("DS11-v2")

    with mlflow.start_run(run_name="T5_summary"):
        mlflow.log_param("model", t5_model)
        mlflow.log_param("device", device_str)
        mlflow.log_param("max_input_chars", max_input_chars)
        mlflow.log_param("max_output_length", max_out_len)
        mlflow.log_param("min_output_length", min_out_len)
        mlflow.log_param("reviews_per_product", reviews_per_product)
        mlflow.log_param("sentiment_source", "rating_recomputed")

        mlflow.log_metric("n_summarized", len(summary_df))
        mlflow.log_metric("n_skipped", n_skipped)
        mlflow.log_metric("avg_summary_length", round(avg_len, 2))

        mlflow.log_artifact(output_path)

    print(f"MLflow T5_summary run logged — "
          f"n_summarized={len(summary_df)}, avg_summary_length={avg_len:.1f} words")

    print("\n✓ Phase 3B complete.")
    print(f"  {output_path}")


if __name__ == "__main__":
    main()