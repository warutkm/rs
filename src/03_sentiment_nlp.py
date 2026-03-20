"""
03a_sentiment_tfidf_svm.py — Phase 3 (Part A): Sentiment Labels + TF-IDF + LinearSVC

Covers:
  3.1 Sentiment labels
  3.2 TF-IDF vectorization
  3.3 LinearSVC training
  3.4 MLflow logging
  3.5 Model persistence
  3.8 Visualizations

Run BEFORE 03b_t5_summarization.py

Changes vs previous version:
  - Removed np.random.seed(42) — redundant; sklearn uses its own random_state
  - Scatter plot now uses np.random.default_rng(42) for reproducibility
  - Confusion matrix tick labels derived from sorted(y_test.unique())
    so labels are always correct regardless of which classes are present
  - class_weight + random_state added to MLflow params
  - Word cloud wrapped in fig/ax for consistency with other plots
  - collocations=False added to WordCloud to avoid bigram noise
"""

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

import mlflow

# =============================================================================
# CREATE DIRS
# =============================================================================
def create_dirs():
    os.makedirs("models",  exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("mlflow",  exist_ok=True)

create_dirs()

# =============================================================================
# LOAD DATA
# =============================================================================
DATA_PATH = "data/clean_merge_df.parquet"
df = pd.read_parquet(DATA_PATH)
print(f"Loaded data shape: {df.shape}")

# =============================================================================
# SAFETY: HANDLE MISSING TEXT
# =============================================================================
df["text_clean"] = df["text_clean"].fillna("")

# =============================================================================
# 3.1 SENTIMENT LABELS
#     rating <= 2  -> negative
#     rating == 3  -> neutral
#     rating >= 4  -> positive
# =============================================================================
def create_sentiment(rating: float) -> str:
    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    return "positive"

df["sentiment"] = df["rating"].apply(create_sentiment)

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())
print("\nSentiment % distribution:")
print((df["sentiment"].value_counts(normalize=True) * 100).round(2))

# =============================================================================
# 3.2 TF-IDF VECTORIZATION
#     max_features=15000, ngram_range=(1, 2) on text_clean
# =============================================================================
vectorizer = TfidfVectorizer(
    max_features=15_000,
    ngram_range=(1, 2),
)

X = vectorizer.fit_transform(df["text_clean"])
y = df["sentiment"]

print(f"\nTF-IDF shape: {X.shape}")

# Extract only what visualizations need, then free the full DataFrame
sentiment_series  = df["sentiment"].copy()
rating_series     = df["rating"].copy()
category_series   = df["main_category"].copy() if "main_category" in df.columns else None
text_clean_series = df["text_clean"].copy()
del df

# =============================================================================
# 3.3 TRAIN LinearSVC — 80/20 stratified split
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify = y if y.nunique() > 1 else None,
)
del X  # free full matrix after split
del y

model = LinearSVC(max_iter=3000, class_weight="balanced")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
del X_train, X_test

# =============================================================================
# METRICS
# =============================================================================
report      = classification_report(y_test, y_pred)
cm          = confusion_matrix(y_test, y_pred)
accuracy    = accuracy_score(y_test, y_pred)
f1_macro    = f1_score(y_test, y_pred, average="macro")
f1_weighted = f1_score(y_test, y_pred, average="weighted")

print("\n=== Classification Report ===")
print(report)
print("\n=== Confusion Matrix ===")
print(cm)

with open("outputs/classification_report.txt", "w") as fh:
    fh.write(report)

# Confusion matrix heatmap
# Tick labels derived from sorted unique classes — matches sklearn's ordering exactly
# and is safe if a class is absent from the test split
present_labels = sorted(y_test.unique())
label_abbrev   = {"negative": "neg", "neutral": "neu", "positive": "pos"}
tick_labels    = [label_abbrev.get(l, l) for l in present_labels]

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=tick_labels,
    yticklabels=tick_labels,
    ax=ax,
)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150)
plt.close()

# =============================================================================
# 3.4 MLFLOW
#     experiment: DS11  |  run_name: SVM
# =============================================================================
mlflow.set_tracking_uri("mlflow/")
mlflow.set_experiment("DS11")

with mlflow.start_run(run_name="SVM"):
    mlflow.log_param("max_features",  15_000)
    mlflow.log_param("ngram_range",   "(1,2)")
    mlflow.log_param("test_size",     0.2)
    mlflow.log_param("random_state",  42)
    mlflow.log_param("model",         "LinearSVC")
    mlflow.log_param("class_weight",  "balanced")

    mlflow.log_metric("accuracy",    accuracy)
    mlflow.log_metric("f1_macro",    f1_macro)
    mlflow.log_metric("f1_weighted", f1_weighted)

    mlflow.log_artifact("outputs/classification_report.txt")
    mlflow.log_artifact("outputs/confusion_matrix.png")

print(f"\nMLflow logged — accuracy={accuracy:.4f}, f1_macro={f1_macro:.4f}")

# =============================================================================
# 3.5 PERSIST MODELS
#     models/svm_vectorizer.pkl
#     models/svm_model.pkl
# =============================================================================
with open("models/svm_vectorizer.pkl", "wb") as fh:
    pickle.dump(vectorizer, fh)

with open("models/svm_model.pkl", "wb") as fh:
    pickle.dump(model, fh)

del vectorizer, model
print("Models saved → models/")

# =============================================================================
# 3.8 VISUALIZATIONS
# =============================================================================

# --- (a) Sentiment Distribution bar chart ------------------------------------
sent_counts = sentiment_series.value_counts()
colors = {"positive": "#4CAF50", "neutral": "#FFC107", "negative": "#F44336"}

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(
    sent_counts.index,
    sent_counts.values,
    color=[colors.get(s, "#90A4AE") for s in sent_counts.index],
    edgecolor="white",
)
for bar, val in zip(bars, sent_counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + sent_counts.max() * 0.01,
        f"{val:,}",
        ha="center", va="bottom", fontsize=9,
    )
ax.set_title("Sentiment Distribution")
ax.set_xlabel("Sentiment")
ax.set_ylabel("Count")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/sentiment_bar.png", dpi=150)
plt.close()

# --- (b) Rating vs Sentiment scatter -----------------------------------------
sent_code = sentiment_series.map({"negative": 0, "neutral": 1, "positive": 2})
rng       = np.random.default_rng(42)   # reproducible, independent of global state
n_sample  = min(5_000, len(sentiment_series))
idx       = rng.choice(len(sentiment_series), size=n_sample, replace=False)

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(
    rating_series.iloc[idx],
    sent_code.iloc[idx],
    alpha=0.3, s=10,
    c=sent_code.iloc[idx], cmap="RdYlGn",
)
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["negative", "neutral", "positive"])
ax.set_xlabel("Star Rating")
ax.set_title(f"Rating vs Sentiment ({n_sample:,} sample)")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/rating_vs_sentiment.png", dpi=150)
plt.close()

del rating_series, sentiment_series, sent_code

# --- (c) Word Cloud — Software has richest vocabulary ------------------------
if category_series is not None:
    wc_text   = text_clean_series[category_series == "Software"]
    wc_source = "Software category"
else:
    wc_text   = text_clean_series
    wc_source = "all reviews"

text = " ".join(wc_text.astype(str).values[:50_000])

wc = WordCloud(
    width=800, height=400,
    background_color="white",
    colormap="viridis",
    collocations=False,   # avoids bigram noise in the cloud
).generate(text)

fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wc, interpolation="bilinear")
ax.axis("off")
ax.set_title(f"Word Cloud — {wc_source}", fontsize=12)
plt.tight_layout()
plt.savefig("outputs/wordcloud.png", dpi=150)
plt.close()

del text_clean_series, wc_text, text

# =============================================================================
# DONE
# =============================================================================
print("\n✓ Phase 3A complete. Run 03b_t5_summarization.py next.")
print("  outputs/classification_report.txt")
print("  outputs/confusion_matrix.png")
print("  outputs/sentiment_bar.png")
print("  outputs/rating_vs_sentiment.png")
print("  outputs/wordcloud.png")
print("  models/svm_vectorizer.pkl")
print("  models/svm_model.pkl")