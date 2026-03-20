"""
Usage
─────
# Auto-detect environment:
    python 07_semantic_search.py

# Force a specific mode:
    python 07_semantic_search.py --mode colab
    python 07_semantic_search.py --mode local

# After Colab finishes:
    Download embeddings/review_embeds.npy
              embeddings/meta_embeds.npy
              embeddings/meta_item_ids.json
    Place them in  <local_project>/embeddings/
    Then run:  python 07_semantic_search.py --mode local
"""

# ═══════════════════════════════════════════════════════════════
# 0.  IMPORTS
# ═══════════════════════════════════════════════════════════════
import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# 1.  ENVIRONMENT DETECTION  (§1.6)
# ═══════════════════════════════════════════════════════════════
def detect_env() -> str:
    """Returns 'colab' or 'local'."""
    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        pass
    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ:
        return "colab"
    return "local"

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--mode", choices=["colab", "local"], default=None)
_parser.add_argument("--parquet", type=str, default=None)
_args, _ = _parser.parse_known_args()

ENV = _args.mode or detect_env()
print(f"[ENV] Running in '{ENV}' mode  (§1.6)")

# ═══════════════════════════════════════════════════════════════
# 2.  CONFIG
# ═══════════════════════════════════════════════════════════════
DATA_PATH  = Path("data/clean_merge_df.parquet")
EMBED_DIR  = Path("embeddings")
MODEL_NAME = "intfloat/e5-base-v2"
USE_FP16   = True

# §1.6 batch sizes
BATCH_SIZE = 64 if ENV == "colab" else 8   # Colab ~20 min | Local ~3-4 hrs

# Retrieval
HYBRID_EMB_W    = 0.55
HYBRID_BM25_W   = 0.45
HYBRID_THRESH   = 0.20
HYBRID_MIN_HITS = 20

# product_vecs fusion
MANY_REV_W = (0.7, 0.3)   # (review_mean, meta)  when >= 3 reviews
FEW_REV_W  = (0.3, 0.7)   # swapped               when <  3 reviews

# User profiles
TOP_N_USERS = 50

# ═══════════════════════════════════════════════════════════════
# 3.  DIRECTORY SETUP
# ═══════════════════════════════════════════════════════════════
EMBED_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════
# 4.  COLAB HELPERS
# ═══════════════════════════════════════════════════════════════
def colab_download_embeddings():
    print("\n[COLAB] Encoding complete. Run this in a new notebook cell to download:")
    print("from google.colab import files")
    for fname in ["review_embeds.npy", "meta_embeds.npy", "meta_item_ids.json"]:
        fp = EMBED_DIR / fname
        if fp.exists():
            print(f'files.download("embeddings/{fname}")')
        else:
            print(f"# ⚠  embeddings/{fname} not found — skipping")

# ═══════════════════════════════════════════════════════════════
# 5.  DATA LOADING
# ═══════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    global DATA_PATH

    # Priority 1: --parquet CLI flag (use this in Colab script mode)
    if _args.parquet:
        DATA_PATH = Path(_args.parquet)

    # Priority 2: interactive upload (notebook cell only, not !python)
    elif ENV == "colab" and not DATA_PATH.exists():
        DATA_PATH = colab_upload_parquet()

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found.\n"
            "  Colab : upload the file in a notebook cell first, then run:\n"
            "          !python 07_semantic_search.py --mode colab --parquet clean_merge_df.parquet\n"
            "  Local : place clean_merge_df.parquet in the project root."
        )

    print(f"[DATA] Loading {DATA_PATH} …")
    df = pd.read_parquet(DATA_PATH)
    print(f"       Shape: {df.shape}")
    return df

# ═══════════════════════════════════════════════════════════════
# 6.  MODEL LOADING
# ═══════════════════════════════════════════════════════════════
def load_model():
    from sentence_transformers import SentenceTransformer
    print(f"[MODEL] {MODEL_NAME}  fp16={USE_FP16}  batch={BATCH_SIZE}")
    m = SentenceTransformer(MODEL_NAME)
    if USE_FP16:
        m = m.half()
    return m

# ═══════════════════════════════════════════════════════════════
# HELPER: batch encode
# ═══════════════════════════════════════════════════════════════
def batch_encode(model, texts: list, desc: str) -> np.ndarray:
    chunks = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=desc, unit="batch"):
        emb = model.encode(texts[i : i + BATCH_SIZE], normalize_embeddings=True)
        chunks.append(emb.astype(np.float32))
    return np.vstack(chunks)

# ═══════════════════════════════════════════════════════════════
# 7.1  BUILD product_text  [Fix]
# ═══════════════════════════════════════════════════════════════
def build_product_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    product_text = title_meta + description + ' '.join(features)
                   + str(details) + str(categories)
    Uses corrected/new fields per workflow Fix tag.
    """
    print("[7.1] Building product_text …")

    def _build(row):
        title   = str(row.get("title_meta",  "") or "")
        desc    = str(row.get("description", "") or "")
        feats   = row.get("features", [])
        feats   = " ".join(feats) if isinstance(feats, list) else str(feats or "")
        details = str(row.get("details",     "") or "")
        cats    = str(row.get("categories",  "") or "")
        return f"{title} {desc} {feats} {details} {cats}".strip()

    df = df.copy()
    df["product_text"] = df.apply(_build, axis=1)

    # Safe access for optional review text columns
    t_rev  = df["title_rev"].fillna("") if "title_rev" in df.columns else ""
    t_body = df["text"].fillna("")      if "text"      in df.columns else ""
    df["full_review_text"] = (t_rev + " " + t_body).str.strip()

    print(f"       product_text[0]: {df['product_text'].iloc[0][:100]} …")
    return df

# ═══════════════════════════════════════════════════════════════
# 7.2  ENCODE REVIEWS → review_embeds.npy  [Core]
# ═══════════════════════════════════════════════════════════════
def encode_reviews(model, df: pd.DataFrame) -> np.ndarray:
    """
    §1.6  Colab: batch=64, ~20 min | Local: batch=8, ~3-4 hrs
    normalize=True; float32.
    Local caches the result — re-run skips encoding.
    """
    out = EMBED_DIR / "review_embeds.npy"
    if ENV == "local" and out.exists():
        print(f"[7.2] Cache hit — loading {out}")
        return np.load(out)

    print(f"[7.2] Encoding {len(df):,} reviews  batch={BATCH_SIZE} …")
    embeds = batch_encode(model, df["full_review_text"].fillna("").tolist(), "Reviews")
    np.save(out, embeds)
    print(f"       Saved {out}  shape={embeds.shape}")
    return embeds

# ═══════════════════════════════════════════════════════════════
# 7.3  ENCODE META per item_id → meta_embeds.npy  [Core]
# ═══════════════════════════════════════════════════════════════
def encode_meta(model, df: pd.DataFrame):
    """Group by item_id (parent_asin) → first product_text; encode; save."""
    out_emb = EMBED_DIR / "meta_embeds.npy"
    out_ids = EMBED_DIR / "meta_item_ids.json"

    meta_df = df.groupby("item_id").first().reset_index()

    if ENV == "local" and out_emb.exists() and out_ids.exists():
        print(f"[7.3] Cache hit — loading {out_emb}")
        return meta_df, np.load(out_emb)

    print(f"[7.3] Encoding {len(meta_df):,} items  batch={BATCH_SIZE} …")
    meta_embs = batch_encode(model, meta_df["product_text"].fillna("").tolist(), "Meta")
    np.save(out_emb, meta_embs)
    meta_df["item_id"].to_json(out_ids, orient="records")
    print(f"       Saved {out_emb}  shape={meta_embs.shape}")
    print(f"       Saved {out_ids}")
    return meta_df, meta_embs

# ═══════════════════════════════════════════════════════════════
# COLAB-ONLY ENCODE ENTRY POINT
# ═══════════════════════════════════════════════════════════════
def colab_encode_only():
    """
    Colab one-time encode (§1.6):
      7.1 product_text → 7.2 review_embeds → 7.3 meta_embeds → download
    """
    assert ENV == "colab", "Must run in Google Colab"
    df    = load_data()
    model = load_model()
    df    = build_product_text(df)
    encode_reviews(model, df)
    encode_meta(model, df)
    colab_download_embeddings()
    print("\n✅  Colab encode done.")
    print("    Place downloaded .npy files in local embeddings/ then run locally.")

# ═══════════════════════════════════════════════════════════════
# 7.4  BUILD product_vecs DICT  [Core]
# ═══════════════════════════════════════════════════════════════
def build_product_vecs(df: pd.DataFrame,
                       review_embeds: np.ndarray,
                       meta_df: pd.DataFrame,
                       meta_embeds: np.ndarray) -> dict:
    """
    >= 3 reviews : 0.7 * rev_mean + 0.3 * meta
    <  3 reviews : 0.3 * rev_mean + 0.7 * meta   (swapped)
    L2-normalize; key = item_id
    """
    print("[7.4] Building product_vecs …")
    meta_id_to_emb = {iid: meta_embeds[i]
                      for i, iid in enumerate(meta_df["item_id"].tolist())}

    df = df.copy()
    df["_idx"] = np.arange(len(df))

    product_vecs: dict = {}
    for item_id, grp in tqdm(df.groupby("item_id"), desc="product_vecs"):
        rev_embs = review_embeds[grp["_idx"].values]
        rev_mean = rev_embs.mean(axis=0)
        meta_emb = meta_id_to_emb.get(item_id)
        n_rev    = len(grp)

        if meta_emb is None:
            vec = rev_mean
        else:
            rw, mw = MANY_REV_W if n_rev >= 3 else FEW_REV_W
            vec = rw * rev_mean + mw * meta_emb

        norm = np.linalg.norm(vec)
        product_vecs[item_id] = (vec / norm if norm > 0 else vec).astype(np.float32)

    out = EMBED_DIR / "product_vecs.npz"
    np.savez(out,
             keys=np.array(list(product_vecs.keys())),
             vecs=np.array(list(product_vecs.values())))
    print(f"       Saved {out}  n_items={len(product_vecs):,}")
    return product_vecs

# ═══════════════════════════════════════════════════════════════
# 7.5  BUILD BM25 INDEX  [Extra]
# ═══════════════════════════════════════════════════════════════
def build_bm25_index(df: pd.DataFrame):
    """Tokenize bm25_text per item_id; BM25Okapi fit."""
    print("[7.5] Building BM25 index …")
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("       ⚠  rank_bm25 not installed — pip install rank_bm25")
        return None, []

    meta_df  = df.groupby("item_id").first().reset_index()
    item_ids = meta_df["item_id"].tolist()
    corpus   = [t.lower().split() for t in meta_df["product_text"].fillna("").tolist()]
    bm25     = BM25Okapi(corpus)

    with open(EMBED_DIR / "bm25_corpus.json", "w") as f:
        json.dump({"item_ids": item_ids, "corpus": corpus}, f)

    print(f"       BM25 ready  n_docs={len(item_ids):,}")
    return bm25, item_ids

# ═══════════════════════════════════════════════════════════════
# 7.6  HYBRID RETRIEVAL  [Extra]
#   score = 0.55*emb_norm + 0.45*bm25_norm
#   threshold=0.20 ; min 20 results
# ═══════════════════════════════════════════════════════════════
def hybrid_retrieve(query: str,
                    model,
                    product_vecs: dict,
                    bm25_model,
                    bm25_item_ids: list,
                    top_k: int = 50) -> pd.DataFrame:

    def _minmax(arr):
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-9)

    # Embedding similarity
    q_emb      = model.encode([query], normalize_embeddings=True)[0]
    item_ids   = list(product_vecs.keys())
    vecs       = np.array(list(product_vecs.values()))
    emb_norm   = _minmax((vecs @ q_emb).astype(np.float64))

    # BM25
    bm25_raw   = np.array(bm25_model.get_scores(query.lower().split()))
    bm25_norm  = _minmax(bm25_raw)
    bm25_map   = dict(zip(bm25_item_ids, bm25_norm))
    bm25_align = np.array([bm25_map.get(iid, 0.0) for iid in item_ids])

    # Hybrid
    hybrid = HYBRID_EMB_W * emb_norm + HYBRID_BM25_W * bm25_align

    results = (
        pd.DataFrame({
            "item_id":      item_ids,
            "emb_score":    emb_norm,
            "bm25_score":   bm25_align,
            "hybrid_score": hybrid,
        })
        .sort_values("hybrid_score", ascending=False)
        .reset_index(drop=True)
    )

    above = results[results["hybrid_score"] >= HYBRID_THRESH]
    if len(above) >= HYBRID_MIN_HITS:
        return above.head(top_k).reset_index(drop=True)
    return results.head(max(HYBRID_MIN_HITS, top_k)).reset_index(drop=True)

# ═══════════════════════════════════════════════════════════════
# 7.7  SCORE-BAND CLUSTERING  [Extra]
#   4 bands by similarity; centroid cosine ordering within each band
# ═══════════════════════════════════════════════════════════════
def score_band_cluster(results: pd.DataFrame,
                       product_vecs: dict,
                       n_bands: int = 4) -> pd.DataFrame:
    if results.empty:
        return results

    lo, hi = results["hybrid_score"].min(), results["hybrid_score"].max()
    edges  = np.linspace(lo, hi, n_bands + 1)
    frames = []

    for b in range(n_bands - 1, -1, -1):          # highest band first
        mask = (results["hybrid_score"] >= edges[b]) & \
               (results["hybrid_score"] <= edges[b + 1])
        band = results[mask].copy()
        if band.empty:
            continue

        vecs = [product_vecs[iid] for iid in band["item_id"] if iid in product_vecs]
        if not vecs:
            frames.append(band)
            continue

        centroid = np.mean(vecs, axis=0)
        c_norm   = np.linalg.norm(centroid)
        centroid = centroid / c_norm if c_norm > 0 else centroid

        band["centroid_cos"] = [
            float(product_vecs[iid] @ centroid) if iid in product_vecs else 0.0
            for iid in band["item_id"]
        ]
        frames.append(band.sort_values("centroid_cos", ascending=False))

    return pd.concat(frames, ignore_index=True) if frames else results

# ═══════════════════════════════════════════════════════════════
# 7.8  RERANKING MODES  [Extra]
#   quality / trending / personalized / relevance
# ═══════════════════════════════════════════════════════════════
def rerank(results: pd.DataFrame,
           df: pd.DataFrame,
           mode: str = "relevance",
           user_id: str = None,
           user_profile_vecs: dict = None,
           product_vecs: dict = None) -> pd.DataFrame:
    """
    mode='relevance'    → sort by hybrid_score only
    mode='quality'      → 0.60*hybrid + 0.40*normalised_avg_rating
    mode='trending'     → 0.60*hybrid + 0.40*log_normalised_review_count
    mode='personalized' → 0.50*hybrid + 0.50*cosine_to_user_profile
    """
    results = results.copy()

    if mode == "relevance":
        return results.sort_values("hybrid_score", ascending=False).reset_index(drop=True)

    agg = (
        df.groupby("item_id")["rating"]
        .agg(avg_rating="mean", review_count="count")
        .reset_index()
    )
    results = results.merge(agg, on="item_id", how="left")
    results["avg_rating"]   = results["avg_rating"].fillna(3.0)
    results["review_count"] = results["review_count"].fillna(1)

    if mode == "quality":
        results["mode_score"] = (results["avg_rating"] - 1.0) / 4.0
        alpha = (0.60, 0.40)

    elif mode == "trending":
        max_cnt = results["review_count"].max()
        results["mode_score"] = (
            np.log1p(results["review_count"]) / (np.log1p(max_cnt) + 1e-9)
        )
        alpha = (0.60, 0.40)

    elif mode == "personalized":
        if None in (user_id, user_profile_vecs, product_vecs):
            print("       ⚠  personalized needs user_id + user_profile_vecs + product_vecs")
            return results.sort_values("hybrid_score", ascending=False).reset_index(drop=True)
        u_vec = user_profile_vecs.get(user_id)
        if u_vec is None:
            print(f"       ⚠  No embedding for user {user_id} — fallback to relevance")
            return results.sort_values("hybrid_score", ascending=False).reset_index(drop=True)
        p_raw = np.array([
            float(product_vecs[iid] @ u_vec) if iid in product_vecs else 0.0
            for iid in results["item_id"]
        ])
        lo, hi = p_raw.min(), p_raw.max()
        results["mode_score"] = (p_raw - lo) / (hi - lo + 1e-9)
        alpha = (0.50, 0.50)

    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    results["final_score"] = alpha[0] * results["hybrid_score"] + alpha[1] * results["mode_score"]
    return results.sort_values("final_score", ascending=False).reset_index(drop=True)

# ═══════════════════════════════════════════════════════════════
# 7.9  USER PROFILE EMBEDDINGS  [Extra]
#   Top-50 users; rating-weighted review text → top_user_embeddings.npy
# ═══════════════════════════════════════════════════════════════
def build_user_profile_embeddings(model, df: pd.DataFrame) -> dict:
    print(f"[7.9] User profile embeddings  top={TOP_N_USERS} users …")

    top_users = (
        df.groupby("user_id")["rating"]
        .count()
        .nlargest(TOP_N_USERS)
        .index.tolist()
    )
    user_df = df[df["user_id"].isin(top_users)].copy()

    def _weighted_text(grp):
        weights = grp["rating"].fillna(3.0).clip(1, 5).astype(int)
        t_rev   = grp["title_rev"].fillna("") if "title_rev" in grp.columns else pd.Series([""] * len(grp), index=grp.index)
        t_body  = grp["text"].fillna("")       if "text"      in grp.columns else pd.Series([""] * len(grp), index=grp.index)
        texts   = (t_rev + " " + t_body).str.strip()
        parts   = []
        for txt, w in zip(texts, weights):
            parts.extend([txt] * w)          # repeat by rating weight
        return " ".join(parts)[:3000]

    user_texts = (
        user_df.groupby("user_id")
        .apply(_weighted_text)
        .reset_index()
        .rename(columns={0: "profile_text"})
    )

    user_ids = user_texts["user_id"].tolist()
    embeds   = batch_encode(model, user_texts["profile_text"].tolist(), "User profiles")

    out_emb = EMBED_DIR / "top_user_embeddings.npy"
    out_ids = EMBED_DIR / "top_user_ids.json"
    np.save(out_emb, embeds)
    with open(out_ids, "w") as f:
        json.dump(user_ids, f)

    print(f"       Saved {out_emb}  shape={embeds.shape}")
    print(f"       Saved {out_ids}")
    return {uid: embeds[i].astype(np.float32) for i, uid in enumerate(user_ids)}

# ═══════════════════════════════════════════════════════════════
# 7.10  DVC EMBED STAGE  [NEW]
#   Must run locally (§1.6 — DVC not available in Colab)
# ═══════════════════════════════════════════════════════════════
def register_dvc_stage():
    cmd = (
        "dvc run -n embed \\\n"
        "    -d src/07_semantic_search.py \\\n"
        "    -d clean_merge_df.parquet \\\n"
        "    -o embeddings/ \\\n"
        "    python src/07_semantic_search.py"
    )
    print("\n[7.10] DVC embed stage  (local only — §1.6):")
    print("─" * 62)
    print(cmd)
    print("─" * 62)
    sh = Path("dvc_embed_stage.sh")
    sh.write_text(f"#!/usr/bin/env bash\n{cmd}\n")
    sh.chmod(0o755)
    print(f"       Shell helper → {sh}")

# ═══════════════════════════════════════════════════════════════
# LOCAL PIPELINE  (7.4 → 7.10)
# ═══════════════════════════════════════════════════════════════
def local_pipeline(df, review_embeds, meta_df, meta_embeds, model):
    product_vecs      = build_product_vecs(df, review_embeds, meta_df, meta_embeds)
    bm25_model, b_ids = build_bm25_index(df)
    user_profile_vecs = build_user_profile_embeddings(model, df)
    register_dvc_stage()

    if bm25_model is None:
        print("\n⚠  Retrieval demo skipped — install rank_bm25 first.")
        return

    QUERY = "wireless bluetooth headphones noise cancelling"
    print(f"\n{'─'*62}")
    print(f"[DEMO] Query: '{QUERY}'")
    print(f"{'─'*62}")

    hits      = hybrid_retrieve(QUERY, model, product_vecs, bm25_model, b_ids)
    print(f"[7.6]  Hybrid hits : {len(hits)}")

    clustered = score_band_cluster(hits, product_vecs)
    print(f"[7.7]  After band clustering : {len(clustered)}")

    for mode in ("relevance", "quality", "trending"):
        rr = rerank(clustered, df, mode=mode)
        print(f"\n[7.8]  mode={mode}")
        print(rr[["item_id", "hybrid_score"]].head(5).to_string(index=False))

    if user_profile_vecs:
        uid = next(iter(user_profile_vecs))
        rr_p = rerank(clustered, df, mode="personalized",
                      user_id=uid,
                      user_profile_vecs=user_profile_vecs,
                      product_vecs=product_vecs)
        print(f"\n[7.8]  mode=personalized  user={uid}")
        print(rr_p[["item_id", "hybrid_score"]].head(5).to_string(index=False))

    print(f"\n{'═'*62}")
    print("✅  Phase 7 complete — artefacts in ./embeddings/")
    print(f"{'═'*62}")

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    print(f"\n{'═'*62}")
    print(" PHASE 7 — Semantic Search + Reranking")
    print(f" Environment : {ENV.upper()}  (§1.6)")
    print(f"{'═'*62}\n")

    if ENV == "colab":
        # ── COLAB: encode only (7.1 → 7.3) then download ─────
        print("ℹ  Colab mode → encode reviews + meta, download .npy files.")
        print("   Then run locally to complete 7.4 – 7.10.\n")
        colab_encode_only()
        return

    # ── LOCAL: full pipeline ───────────────────────────────────
    df    = load_data()
    model = load_model()
    df    = build_product_text(df)                        # 7.1
    rev_e = encode_reviews(model, df)                    # 7.2
    meta_df, meta_e = encode_meta(model, df)             # 7.3
    local_pipeline(df, rev_e, meta_df, meta_e, model)   # 7.4 – 7.10


if __name__ == "__main__":
    main()