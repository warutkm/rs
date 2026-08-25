#!/usr/bin/env bash
dvc run -n embed \
    -d src/07_semantic_search.py \
    -d clean_merge_df.parquet \
    -o embeddings/ \
    python src/07_semantic_search.py
