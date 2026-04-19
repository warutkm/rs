# Hybrid Recommender System (Amazon Reviews 2023)

## Overview
This project implements a **production-grade hybrid recommender system** that combines:
- Content-based filtering (TF-IDF, embeddings)
- Collaborative filtering (ALS, Matrix Factorization, SVD++)
- Semantic search (e5 embeddings + BM25)
- Market basket analysis (Apriori)

The system is designed to handle **cold-start scenarios**, scale efficiently, and be deployable via **FastAPI + Docker**, with full **MLOps integration (MLflow + DVC)**.

Dataset: Amazon Reviews 2023 (Video_Games, Musical_Instruments, Software) 

---

## Architecture

### Core Components

1. **Data Pipeline (DVC Controlled)**
   - Ingestion → Preprocessing → Embedding
   - Fully reproducible using `dvc repro`

2. **Feature Engineering**
   - Text: TF-IDF + cleaned review text
   - Metadata: title, description, features, details
   - Behavioral: user-item interactions

3. **Models**

#### Content-Based
- TF-IDF + LinearSVC (sentiment)
- Product scoring (rating, helpful votes, recency, price)

#### Collaborative Filtering
- Matrix Factorization (PyTorch)
- Neural Collaborative Filtering (GMF + MLP)
- ALS (Implicit)
- SVD++ (Surprise)

#### Semantic Search
- e5-base-v2 embeddings
- Hybrid retrieval: Embedding + BM25
- Reranking: relevance, quality, trending, personalized

#### Hybrid Engine
- Combines:
  - Content-based
  - Collaborative filtering
  - Apriori rules
- Weighted fusion strategy

---

## Key Design Decisions

### 1. parent_asin as Item ID
Using `parent_asin` instead of `asin` avoids duplicate product variants and ensures dense CF matrices.

### 2. User Activity Filtering
Only users with ≥5 interactions are retained to improve CF learning quality.

### 3. Domain-Focused Dataset
Using 3 related categories improves:
- CF overlap
- NLP richness
- Cross-sell patterns 

---

## Project Structure

```
rs/
├── config.py
├── dvc.yaml
├── data/
├── embeddings/
├── models/
├── outputs/
├── src/
│   ├── 01_data_ingestion.py
│   ├── 02_preprocessing.py
│   ├── 03_sentiment_nlp.py
│   ├── 04_apriori_recommender.py
│   ├── 05_content_cf_recommender.py
│   ├── 06_mf_ncf_pytorch.py
│   ├── 07_semantic_search.py
│   ├── 08_hybrid_engine.py
│   ├── 09_als_svdpp.py
│   ├── 10_ab_comparison.ipynb
│   └── 11_mlflow_report.ipynb
├── api/
│   ├── main.py
│   ├── schemas.py
│   ├── Dockerfile
│   └── docker-compose.yml
```

---

## Installation

```bash
git clone https://github.com/warutkm/rs.git
cd rs

conda create -n rs_env python=3.10
conda activate rs_env

pip install -r requirements.txt
```

---

## Running the Pipeline

```bash
dvc repro
```

Stages:
- ingest
- preprocess
- embed

---

## Model Training

Run scripts sequentially:

```bash
python src/03_sentiment_nlp.py
python src/05_content_cf_recommender.py
python src/06_mf_ncf_pytorch.py
python src/09_als_svdpp.py
python src/08_hybrid_engine.py
```

---

## MLflow Tracking

```bash
mlflow ui --backend-store-uri mlflow/
```

Tracks:
- SVM
- Content-only
- MF
- NCF
- ALS
- SVD++
- Hybrid

---

## API Usage

### Start Service
```bash
docker-compose up --build
```

### Endpoints

#### Health Check
```
GET /health
```

#### Recommend
```
POST /recommend
{
  "item_id": "B001...",
  "user_id": "U123",
  "top_k": 10
}
```

#### Similar Items
```
GET /similar/{item_id}
```

#### Search
```
GET /search?q=wireless+headphones
```

---

## Evaluation

Metrics:
- RMSE
- Recall@10
- NDCG@10
- Precision@10

A/B comparison includes:
- Content-only
- ALS
- MF/NCF
- Hybrid

---

## Cold Start Handling

- **New User** → Content-based + semantic search
- **New Item** → Embedding similarity

---

## MLOps

- **DVC** → Data versioning & reproducibility
- **MLflow** → Experiment tracking
- **Docker** → Deployment
- **FastAPI** → Serving layer

---

## Deliverables

- Hybrid Recommendation API
- A/B Comparison Notebook
- MLflow Experiment Report 

---

## Future Improvements

- Online learning for real-time updates
- Reinforcement learning-based ranking
- Vector database integration (FAISS / Pinecone)

---

## Author
Utkrishta
