"""
04_apriori_recommender.py
=========================
Phase 4 — Apriori Market-Basket Analysis (BONUS)

Workflow steps
--------------
4.1  Build baskets        groupby user_id -> list of item_id; keep size 3-12;
                          pre-filter rare items (< MIN_ITEM_FREQ baskets)
                          to avoid memory explosion on wide matrices
4.2  One-hot encode       mlxtend TransactionEncoder -> boolean DataFrame
4.3  Run Apriori          min_support=0.0005; low_memory=True;
                          association_rules(metric='lift', min_threshold=1.0)
4.4  Keep 1->1 rules      antecedent size=1 AND consequent size=1
4.5  Validate             Check bought_together ground truth if available;
                          gracefully skips if column is null/absent
4.6  AprioriRecommender   bidirectional lookup; scored by lift * confidence;
                          source='apriori' tag for hybrid engine

Outputs
-------
outputs/apriori_rules.csv
models/apriori_recommender.pkl
"""

import ast
import os
from collections import defaultdict

import dill
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# ── Tuning knob ─────────────────────────────────────────────────────────────
# Item must appear in at least this many baskets to be kept before encoding.
# Prevents the OOM error caused by 26K+ columns in the one-hot matrix.
# Lower to 5 if you get 0 rules; raise to 20 to run faster.
MIN_ITEM_FREQ = 10


# ---------------------------------------------------------------------------
# 0. Directory setup
# ---------------------------------------------------------------------------

def create_dirs():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_data():
    df = pd.read_parquet("data/clean_merge_df.parquet")
    print(f"Loaded data: {df.shape[0]:,} rows | "
          f"{df['user_id'].nunique():,} users | "
          f"{df['item_id'].nunique():,} unique items")
    return df


# ---------------------------------------------------------------------------
# 4.1 Build baskets
# ---------------------------------------------------------------------------

def build_baskets(df):
    """
    Group item_id by user_id; keep baskets of size 3-12.
    Pre-filter items that appear in fewer than MIN_ITEM_FREQ baskets
    to keep the one-hot matrix tractable for mlxtend's dense Apriori.
    Without this filter, 26K+ columns causes a 56 GiB allocation error.
    """
    baskets_series = df.groupby("user_id")["item_id"].apply(lambda x: list(set(x)))
    baskets_series = baskets_series[
        baskets_series.apply(lambda x: 3 <= len(x) <= 12)
    ]
    print(f"\n[4.1] Baskets after size filter (3-12): {len(baskets_series):,}")

    # Count how many baskets each item appears in
    all_items = [item for basket in baskets_series for item in basket]
    item_counts = pd.Series(all_items).value_counts()
    frequent_items = set(item_counts[item_counts >= MIN_ITEM_FREQ].index)

    print(f"      Items before freq filter : {len(item_counts):,}")
    print(f"      Items after  freq filter  (>= {MIN_ITEM_FREQ} baskets): {len(frequent_items):,}")

    # Rebuild baskets keeping only frequent items; re-apply size filter
    baskets = [
        [i for i in basket if i in frequent_items]
        for basket in baskets_series
    ]
    baskets = [b for b in baskets if 3 <= len(b) <= 12]

    print(f"      Baskets after item filter: {len(baskets):,}")
    print(pd.Series([len(b) for b in baskets]).describe().to_string())

    return baskets


# ---------------------------------------------------------------------------
# 4.2 One-hot encode
# ---------------------------------------------------------------------------

def one_hot_encode(baskets):
    te = TransactionEncoder()
    te_array = te.fit(baskets).transform(baskets)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_).astype(bool)

    print(f"\n[4.2] Encoded shape: {df_encoded.shape} "
          f"({df_encoded.shape[1]:,} items x {df_encoded.shape[0]:,} baskets)")

    return df_encoded


# ---------------------------------------------------------------------------
# 4.3 Apriori + association rules
# ---------------------------------------------------------------------------

def generate_rules(df_encoded):
    print(f"\n[4.3] Running Apriori on "
          f"{df_encoded.shape[1]:,} items x {df_encoded.shape[0]:,} baskets ...")

    if df_encoded.empty or df_encoded.shape[1] == 0:
        print("      Empty encoded DataFrame — nothing to mine.")
        return pd.DataFrame()

    frequent_itemsets = apriori(
        df_encoded,
        min_support=0.0005,
        use_colnames=True,
        low_memory=True,    # processes candidates in chunks — less RAM, slower
    )

    if frequent_itemsets.empty:
        print("      No frequent itemsets found. "
              "Try lowering min_support or MIN_ITEM_FREQ.")
        return pd.DataFrame()

    print(f"      Frequent itemsets: {len(frequent_itemsets):,}")

    rules = association_rules(
        frequent_itemsets,
        metric="lift",
        min_threshold=1.0,
    )

    print(f"      Rules generated: {len(rules):,}")
    return rules


# ---------------------------------------------------------------------------
# 4.4 Filter to 1->1 rules
# ---------------------------------------------------------------------------

def filter_one_to_one_rules(rules):
    if rules.empty:
        return rules

    rules = rules[
        (rules["antecedents"].apply(len) == 1) &
        (rules["consequents"].apply(len) == 1)
    ].copy()

    rules["antecedent"] = rules["antecedents"].apply(lambda x: next(iter(x)))
    rules["consequent"] = rules["consequents"].apply(lambda x: next(iter(x)))

    rules = rules[["antecedent", "consequent", "support", "confidence", "lift"]]
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)

    print(f"\n[4.4] 1->1 rules: {len(rules):,}")
    return rules


# ---------------------------------------------------------------------------
# 4.5 Validate against bought_together
# ---------------------------------------------------------------------------

def validate_rules(rules, df):
    """
    For each rule A->B check if B appears in bought_together for A.
    Gracefully handles three cases:
      1. Column absent from DataFrame (was 100% null and dropped at ingestion)
      2. Column present but entirely null/empty
      3. Column present with real data (normal path)

    Note: bought_together is 100% null for Video_Games, Musical_Instruments,
    and Software in Amazon Reviews 2023. Amazon does not populate cross-sell
    bundles consistently for these categories. Validation is skipped and
    confirmed_pct=0.0 is returned. Document this in the A/B notebook.
    """
    if rules.empty:
        print("\n[4.5] No rules to validate.")
        return 0.0

    if "bought_together" not in df.columns:
        print("\n[4.5] 'bought_together' column not found in DataFrame.")
        print("      Was 100% null in source data and dropped at ingestion.")
        print("      Ground-truth validation skipped. confirmed_pct=0.0")
        return 0.0

    meta_map = (
        df.drop_duplicates("item_id")
          .set_index("item_id")["bought_together"]
          .to_dict()
    )

    non_null = sum(
        1 for v in meta_map.values()
        if isinstance(v, list) and len(v) > 0
    )

    if non_null == 0:
        print("\n[4.5] 'bought_together' is 100% null for these 3 categories.")
        print("      (Video_Games + Musical_Instruments + Software)")
        print("      Ground-truth validation not possible with this dataset.")
        print("      All rules have lift > 1.0 — primary quality signal.")
        print("      confirmed_pct=0.0 — document this in A/B notebook.")
        return 0.0

    # Normal validation path
    confirmed = 0
    for _, row in rules.iterrows():
        A, B = row["antecedent"], row["consequent"]
        bt = meta_map.get(A, None)

        if isinstance(bt, str):
            try:
                bt = ast.literal_eval(bt)
            except (ValueError, SyntaxError):
                bt = None

        if isinstance(bt, list) and B in bt:
            confirmed += 1

    total = len(rules)
    pct = (confirmed / total * 100) if total > 0 else 0.0
    print(f"\n[4.5] Ground truth match: {confirmed:,}/{total:,} ({pct:.2f}%)")
    return pct


# ---------------------------------------------------------------------------
# 4.6 AprioriRecommender
# ---------------------------------------------------------------------------

class AprioriRecommender:
    """
    Lookup-based recommender built from 1->1 Apriori association rules.

    - Bidirectional : indexes both A->B and B->A directions
    - Deduplication : keeps best score per recommended item
    - Scoring       : lift * confidence
    - source tag    : 'apriori' used by HybridRecommender (Phase 8)
    """

    def __init__(self, rules_df, meta_df):
        self.rule_dict = defaultdict(list)

        for _, row in rules_df.iterrows():
            A, B = row["antecedent"], row["consequent"]
            lift, conf = row["lift"], row["confidence"]
            self.rule_dict[A].append((B, lift, conf))   # forward  A->B
            self.rule_dict[B].append((A, lift, conf))   # reverse  B->A

        self.meta = (
            meta_df.drop_duplicates("item_id")
                   .set_index("item_id")["title_meta"]
                   .to_dict()
        )

        print(f"\n[4.6] AprioriRecommender built.")
        print(f"      Items with at least one rule: {len(self.rule_dict):,}")

    def recommend_apriori(self, item_id, top_k=10):
        """
        Return top-K recommendations for item_id.
        Checks both rule directions, deduplicates by best score,
        ranks by lift * confidence descending.
        """
        if item_id not in self.rule_dict:
            return []

        # Deduplicate: keep best score per recommended item
        seen = {}
        for rec_id, lift, conf in self.rule_dict[item_id]:
            if rec_id == item_id:
                continue
            score = lift * conf
            if rec_id not in seen or score > seen[rec_id]:
                seen[rec_id] = score

        ranked = sorted(seen.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            {
                "item_id": rec_id,
                "score": round(score, 4),
                "title": self.meta.get(rec_id, "Unknown"),
                "source": "apriori",    # consumed by HybridRecommender Phase 8
            }
            for rec_id, score in ranked
        ]

    def verify_recommendations(self, item_id, top_k=5):
        """Pretty-print top-K recommendations with all fields for inspection."""
        seed_title = self.meta.get(item_id, item_id)
        recs = self.recommend_apriori(item_id, top_k)

        print(f"\n{'='*60}")
        print(f"Seed : {seed_title}")
        print(f"ID   : {item_id}")
        print(f"{'='*60}")

        if not recs:
            print("  No rules found for this item.")
            return

        for i, r in enumerate(recs, 1):
            print(f"  {i}. {r['title']}")
            print(f"     score={r['score']} | source={r['source']} | id={r['item_id']}")

        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    create_dirs()

    df = load_data()

    # 4.1
    baskets = build_baskets(df)

    # 4.2
    df_encoded = one_hot_encode(baskets)

    # 4.3
    rules = generate_rules(df_encoded)

    # 4.4
    rules = filter_one_to_one_rules(rules)

    if rules.empty:
        print("No rules generated. Lower MIN_ITEM_FREQ or min_support and retry.")
        return

    rules.to_csv("outputs/apriori_rules.csv", index=False)
    print(f"\nRules saved to: outputs/apriori_rules.csv")

    # 4.5
    gt_pct = validate_rules(rules, df)
    # Uncomment when MLflow is wired up in this file (Phase 8 integration):
    # mlflow.log_metric("apriori_gt_confirmed_pct", gt_pct)

    # 4.6
    model = AprioriRecommender(rules_df=rules, meta_df=df)

    # Sanity check on most-connected item (deterministic across runs)
    sample_item = max(model.rule_dict, key=lambda k: len(model.rule_dict[k]))
    model.verify_recommendations(sample_item, top_k=5)

    with open("models/apriori_recommender.pkl", "wb") as f:
        dill.dump(model, f)

    print("Model saved to: models/apriori_recommender.pkl")
    print("\nPhase 4 complete.")


if __name__ == "__main__":
    main()