import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CREATE DIRS
# =========================
def create_dirs():
    dirs = ["data", "outputs", "models", "embeddings"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

create_dirs()

# =========================
# PATHS
# =========================
INPUT_PATH  = "data/merge_df.csv"
OUTPUT_PATH = "data/clean_merge_df.parquet"

# =========================
# LOAD DATA
# =========================
print("Loading data...")
df = pd.read_csv(INPUT_PATH)

original_shape = df.shape
original_columns = set(df.columns)

print(f"Initial shape: {original_shape}")

# =========================
# 2.1 NULL AUDIT
# =========================
print("\n--- 2.1 NULL AUDIT ---")

na_before    = df.isnull().sum()
null_percent = (na_before / len(df)) * 100

null_report = pd.DataFrame({
    "column":       df.columns,
    "null_count":   na_before.values,
    "null_percent": null_percent.values
}).sort_values("null_percent", ascending=False)

print(null_report.to_string(index=False))
null_report.to_csv("outputs/null_report_before.csv", index=False)

# Plot null %
cols_with_nulls = null_report[null_report["null_percent"] > 0]
if not cols_with_nulls.empty:
    fig, ax = plt.subplots(figsize=(12, max(4, len(cols_with_nulls) * 0.4)))
    ax.barh(cols_with_nulls["column"], cols_with_nulls["null_percent"])
    ax.axvline(40, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Null %")
    ax.set_title("Null % per Column (before cleaning)")
    plt.tight_layout()
    plt.savefig("outputs/null_percent_plot.png", dpi=120)
    plt.close()

# =========================
# 2.2 DROP HIGH-NULL COLUMNS (>40%)
# =========================
print("\n--- 2.2 DROP HIGH-NULL COLUMNS ---")

cols_to_drop = null_percent[null_percent > 40].index.tolist()
df.drop(columns=cols_to_drop, inplace=True)

print(f"Dropped columns: {cols_to_drop}")
print(f"Shape after drop: {df.shape}")

# =========================
# 2.3 IMPUTATION
# =========================
print("\n--- 2.3 IMPUTATION ---")

if "rating_number" in df.columns:
    df["rating_number"] = pd.to_numeric(df["rating_number"], errors="coerce")
    df["rating_number"] = df["rating_number"].fillna(df["rating_number"].mean())

if "store" in df.columns:
    df["store"] = df["store"].fillna("Unknown")

if "price" in df.columns:
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["price"] = df["price"].fillna(df["price"].median())

# Fill object columns only
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].fillna("")

print("Imputation done.")

# =========================
# 2.4 TYPE CASTING
# =========================
print("\n--- 2.4 TYPE CASTING ---")

if "main_category" in df.columns:
    df["main_category"] = df["main_category"].astype("category")

if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

if "item_id" in df.columns:
    df["item_id"] = df["item_id"].astype(str)

print("Type casting done.")

# =========================
# 2.5 TIME-OF-DAY BINNING
# =========================
print("\n--- 2.5 TIME-OF-DAY BINNING ---")

if "timestamp" in df.columns:
    hour = df["timestamp"].dt.hour
    df["hour"] = hour
    df["time_of_day"] = pd.Categorical(
        np.select(
            [hour < 6, hour < 12, hour < 18],
            ["night", "morning", "midday"],
            default="night"   # 18-23 → night
        ),
        categories=["night", "morning", "midday"],
        ordered=False
    )

# =========================
# 2.6 DATE FEATURES
# =========================
print("\n--- 2.6 DATE FEATURES ---")

if "timestamp" in df.columns:
    df["year"]      = df["timestamp"].dt.year
    df["month"]     = df["timestamp"].dt.month
    df["day"]       = df["timestamp"].dt.day
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

# =========================
# 2.7 TEXT CLEANING
# =========================
print("\n--- 2.7 TEXT CLEANING ---")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

if "text" in df.columns:
    df["text_clean"] = df["text"].apply(clean_text)

# =========================
# 2.8 PRODUCT TEXT (CRITICAL)
# =========================
print("\n--- 2.8 PRODUCT TEXT ---")

def safe_join_features(x):
    if isinstance(x, list):
        return " ".join(map(str, x))
    return str(x)

def safe_dict_to_text(x):
    if isinstance(x, dict):
        return " ".join([f"{k} {v}" for k, v in x.items()])
    return str(x)

def build_product_text(row):
    return " ".join([
        str(row.get("title_meta", "")),
        str(row.get("description", "")),
        safe_join_features(row.get("features", "")),
        safe_dict_to_text(row.get("details", "")),
        str(row.get("categories", ""))
    ])

df["product_text"] = df.apply(build_product_text, axis=1)

# =========================
# 2.9 FULL REVIEW TEXT
# =========================
print("\n--- 2.9 FULL REVIEW TEXT ---")

def build_review_text(row):
    return f"{row.get('title_rev', '')} {row.get('text', '')}".strip()

df["full_review_text"] = df.apply(build_review_text, axis=1)

# =========================
# 2.10 FINAL REPORT
# =========================
print("\n--- 2.10 FINAL REPORT ---")

na_after = df.isnull().sum()

final_columns = set(df.columns)
columns_added   = list(final_columns - original_columns)
columns_removed = list(original_columns - final_columns)

pd.DataFrame({
    "column": df.columns,
    "na_before": na_before.reindex(df.columns).values,
    "na_after": na_after.values
}).to_csv("outputs/null_comparison.csv", index=False)

print("\n========== PREPROCESSING REPORT ==========")
print(f"Original shape    : {original_shape}")
print(f"Final shape       : {df.shape}")
print(f"Columns dropped   : {columns_removed}")
print(f"Columns added     : {columns_added}")
print(f"Total columns now : {len(df.columns)}")
print("===========================================")

# =========================
# SANITY CHECK
# =========================
print("\n--- SANITY CHECK ---")
if "user_id" in df.columns:
    print("Unique users :", df["user_id"].nunique())

if "item_id" in df.columns:
    print("Unique items :", df["item_id"].nunique())

# =========================
# SAVE
# =========================
print("\nSaving file...")
df.to_parquet(OUTPUT_PATH, index=False)

print(f"Saved → {OUTPUT_PATH}")

# =========================
# 2.11 DVC COMMAND
# =========================
print("\n--- DVC COMMAND ---")
print(
    "dvc run -n preprocess "
    "-d src/02_preprocessing.py "
    "-d data/merge_df.csv "
    "-o data/clean_merge_df.parquet"
)