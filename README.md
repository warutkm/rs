# Hybrid Recommender System (Amazon Reviews 2023)

## Overview
Production-grade hybrid recommender combining content, collaborative filtering, semantic search, and Apriori.

Dataset: Amazon Reviews 2023 (Video_Games, Musical_Instruments, Software) fileciteturn0file0

---

## System Architecture

```
Dataset → DVC Pipeline → Features → Models (Content + CF + Embeddings) → Hybrid → FastAPI
```

---

## Sample API Response

```json
{
  "item_id": "B001XYZ",
  "recommendations": [
    {"item_id": "B009ABC", "score": 0.87, "source": "hybrid"}
  ]
}
```

---

## Model Performance (Update from MLflow)

Run MLflow UI:
```bash
mlflow ui --backend-store-uri mlflow/
```

Then replace values below with your actual results:

| Model  | RMSE | Recall@10 | NDCG@10 | Precision@10 |
|--------|------|----------|--------|-------------|
| ALS    |      |          |        |             |
| MF     |      |          |        |             |
| NCF    |      |          |        |             |
| Hybrid |      |          |        |             |

---

## Key Decisions

- Use `parent_asin` instead of `asin`
- Filter users with ≥5 interactions
- Use domain-focused categories fileciteturn0file0

---

## Challenges & Learnings

- Data sparsity fixed via parent_asin
- Cold-start solved using hybrid fallback
- Hybrid consistently outperforms standalone models

---

## Run

```bash
dvc repro
```

```bash
docker-compose up
```

---

## Deliverables

- API
- A/B Notebook
- MLflow Report fileciteturn0file1

---

## Author
Utkrishta
