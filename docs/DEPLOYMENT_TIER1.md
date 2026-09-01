# Complete Tier 1 Free Cloud Deployment Master Guide — Amazon RecSys v2

This document is the **exact, click-by-click, account-by-account production runbook** for deploying the complete Amazon RecSys v2 system (FastAPI backend + Next.js frontend + Neon PostgreSQL + Upstash Redis + Qdrant Cloud + Gemini LLM) to 100% free cloud tiers ($0/month).

---

## 1. Architecture & Free-Tier Services Matrix

```
                        ┌──────────────────────────────────────────────────┐
                        │             Next.js 14 Web Frontend              │
                        │           Hosted on Vercel (Free Tier)           │
                        │     Public URL: https://amazon-recsys.vercel.app │
                        └────────────────────────┬─────────────────────────┘
                                                 │ HTTPS / JSON API
                        ┌────────────────────────▼─────────────────────────┐
                        │              FastAPI Backend Service             │
                        │          Hosted on Render (Free Web Service)     │
                        │   Public URL: https://amazon-recsys.onrender.com │
                        └──────────┬─────────────┬─────────────┬───────────┘
                                   │             │             │
        ┌──────────────────────────▼──┐   ┌──────▼─────┐   ┌───▼──────────────────────┐
        │     Qdrant Cloud (ANN)      │   │  Upstash   │   │  Neon Serverless PG      │
        │     1GB Managed Cluster     │   │   Redis    │   │  0.5GB Postgres + SSL    │
        │   768-dim e5 Item Vectors   │   │  (rediss)  │   │  Interaction Event Log   │
        └─────────────────────────────┘   └────────────┘   └──────────────────────────┘
```

| Service | Provider | What to Sign Up For | Free Tier Quota | Role in RecSys |
| :--- | :--- | :--- | :--- | :--- |
| **Relational DB** | [Neon](https://neon.tech/) | Free Serverless Postgres | 0.5 GB storage, auto-suspend compute | User interaction logs & click tracking (`events` table) |
| **Cache Store** | [Upstash](https://upstash.com/) | Serverless Redis (TLS) | 10,000 commands/day, 256 MB | Response cache & LLM "Why This" explanation cache |
| **Vector DB** | [Qdrant Cloud](https://cloud.qdrant.io/) | Free 1GB Managed Cluster | 1 GB RAM (~1M vectors, 768-dim) | HNSW Approximate Nearest Neighbors item similarity |
| **LLM Layer** | [Google AI Studio](https://aistudio.google.com/) | Gemini API Key | Free rate limits / Pay-as-you-go | Gemini 3.5 Flash-Lite query rewriting & explanations |
| **Backend Host** | [Render](https://render.com/) | Free Docker Web Service | 512 MB RAM, 0.1 CPU | Async FastAPI serving LambdaMART ranker & API |
| **Frontend Host** | [Vercel](https://vercel.com/) | Free Hobby Account | Unlimited builds & edge hosting | Next.js 14 App Router browsing & demo user switcher |

---

## 2. Master Key & Connection String Mapping Table

Keep this table handy — it defines **where to get each secret** and **where to paste it**:

| Environment Variable | Where to Get It | Where to Paste in Render | Where to Paste in Vercel | Local `.env` |
| :--- | :--- | :--- | :--- | :--- |
| `DATABASE_URL` | Neon Dashboard -> Project -> Connection details | `DATABASE_URL` | *Not needed* | `DATABASE_URL` |
| `REDIS_URL` | Upstash Dashboard -> Database -> Details -> `rediss://...` | `REDIS_URL` | *Not needed* | `REDIS_URL` |
| `QDRANT_URL` | Qdrant Cloud -> Cluster Details -> Endpoint | `QDRANT_URL` | *Not needed* | `QDRANT_URL` |
| `QDRANT_API_KEY` | Qdrant Cloud -> Cluster Details -> API Keys -> Create | `QDRANT_API_KEY` | *Not needed* | `QDRANT_API_KEY` |
| `GEMINI_API_KEY` | Google AI Studio -> Get API Key | `GEMINI_API_KEY` | *Not needed* | `GEMINI_API_KEY` |
| `LLM_MODEL` | Set explicitly to `gemini-3.5-flash-lite` | `LLM_MODEL` | *Not needed* | `LLM_MODEL` |
| `ADMIN_API_KEY` | Any random secret (e.g. `ds11_admin_secret_key_v2`) | `ADMIN_API_KEY` | *Not needed* | `ADMIN_API_KEY` |
| `NEXT_PUBLIC_API_URL` | Your Render Public URL (e.g. `https://amazon-recsys.onrender.com`) | *Not needed* | `NEXT_PUBLIC_API_URL` | `NEXT_PUBLIC_API_URL` |

---

## 3. Step-by-Step Account Setup & Credential Harvesting

### Step 3.1: Neon Serverless PostgreSQL (Database)
1. Go to **[https://neon.tech/](https://neon.tech/)** and sign up (GitHub login recommended).
2. Click **Create Project**:
   - **Project name**: `amazon-recsys-v2`
   - **Postgres version**: `16` (default)
   - **Region**: Choose closest to your location (e.g. `US East (Ohio)` or `US East (N. Virginia)`).
   - Click **Create Project**.
3. In the **Dashboard**:
   - Find the **Connection Details** box on the right.
   - Select **Connection string** (or `Pooled connection`).
   - Copy the full URI string. It will look like:
     ```text
     postgresql://neondb_owner:npg_AbCdEf123456@ep-cool-snow-a5xyz123.us-east-2.aws.neon.tech/neondb?sslmode=require
     ```
4. Save this as `DATABASE_URL`.

---

### Step 3.2: Upstash Serverless Redis (Cache)
1. Go to **[https://upstash.com/](https://upstash.com/)** and sign up (GitHub login recommended).
2. Click **Create Database**:
   - **Name**: `recsys-cache`
   - **Type**: `Regional`
   - **Region**: Select the same region as Neon/Render (e.g., `us-east-1` / Virginia).
   - **TLS (SSL)**: Enabled (checked).
   - **Eviction**: Enabled (e.g., `volatile-lru`).
   - Click **Create**.
3. In your Database dashboard, scroll down to the **Connect** section:
   - Click the **`redis-cli`** or **`ioredis / Python`** tab.
   - Look for the URL starting with **`rediss://`** (note the double 's' for TLS encryption):
     ```text
     rediss://default:AbCdEf1234567890@us1-cool-panda-12345.upstash.io:6379
     ```
4. Save this as `REDIS_URL`.

---

### Step 3.3: Qdrant Cloud Vector Database (ANN Retrieval)
1. Go to **[https://cloud.qdrant.io/](https://cloud.qdrant.io/)** and sign up.
2. Click **Create Cluster** (or **New Cluster**):
   - **Cluster name**: `amazon-recsys-cluster`
   - **Cluster type / Plan**: **Free 1GB** (1 node).
   - **Cloud provider / Region**: AWS or GCP (e.g., `us-east-1` or `us-east4-0.gcp`).
   - Click **Create**.
3. Once the cluster finishes initializing (~1-2 minutes):
   - Under **Cluster Details**, copy the **Cluster URL / Endpoint**:
     ```text
     https://abcd1234-ef56-7890-gh12-ijkl34567890.us-east4-0.gcp.cloud.qdrant.io:6333
     ```
     *(Save this as `QDRANT_URL`)*.
4. Under **Cluster Details** -> **Data Access Control** / **API Keys**:
   - Click **Create API Key**.
   - Set Name: `recsys-key` -> Access: `Manage (Read & Write)`.
   - Copy the generated API key string:
     ```text
     qdrant_api_key_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
     ```
     *(Save this as `QDRANT_API_KEY`)*.

---

### Step 3.4: Google Gemini API (LLM Layer)
1. Go to **[https://aistudio.google.com/](https://aistudio.google.com/)** and sign in with your Google account.
2. Click **Get API key** in the left sidebar.
3. Click **Create API key** -> Select or create a project -> Copy the generated key:
   ```text
   AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz123456
   ```
4. Save this as `GEMINI_API_KEY`.

---

## 4. Seeding Vector Embeddings to Qdrant Cloud

Before deploying the backend, populate your Qdrant Cloud cluster with precomputed item embeddings:

1. Open your terminal in the `e:\rs` workspace.
2. Run the embedding synchronization pipeline targeting your Qdrant Cloud endpoint:
   ```bash
   python pipeline/sync_embeddings.py --url "https://your-cluster-id.us-east4-0.gcp.cloud.qdrant.io:6333" --api-key "your_qdrant_api_key"
   ```
3. The script will:
   - Create the `products` collection with 768-dimensional cosine distance vector parameters.
   - Batch-upsert embeddings with metadata payloads (title, price, category, rating).
   - Build HNSW payload payload indices for fast filtered vector retrieval.

---

## 5. Local Pre-Flight Verification

Verify that all remote credentials work simultaneously:

1. Fill in your `.env` file or pass keys via flags:
   ```bash
   python scripts/verify_tier1_connectivity.py --db-url "postgresql://neondb_owner:password@ep-xyz.us-east-2.aws.neon.tech/neondb?sslmode=require" --redis-url "rediss://default:password@xyz.upstash.io:6379" --qdrant-url "https://your-cluster-id.cloud.qdrant.io:6333" --qdrant-api-key "your_qdrant_api_key" --gemini-key "your_gemini_key"
   ```
2. When all 4 checks display `[PASS]` in green, you are ready to deploy to Render and Vercel.

---

## 6. Backend Deployment on Render

### Step 6.1: Push Repository to GitHub
Ensure your latest code, Dockerfile, and `render.yaml` are pushed to your GitHub repo:
```bash
git push origin main
```

### Step 6.2: Create Web Service on Render
1. Go to **[https://render.com/](https://render.com/)** and log in.
2. Click the **New +** button in top navigation -> Select **Web Service**.
3. Select **Build and deploy from a Git repository** -> Connect your GitHub repo.
4. Fill in the service configuration:
   - **Name**: `amazon-recsys-api`
   - **Region**: `Oregon (US West)` or `Ohio (US East)`
   - **Branch**: `main`
   - **Runtime**: **`Docker`**
   - **Dockerfile Path**: `api/Dockerfile`
   - **Docker Context**: `.`
   - **Instance Type**: **`Free`** ($0 / month)
5. Scroll down to **Environment Variables** -> Click **Add Environment Variable** for each:

   | Key | Value |
   | :--- | :--- |
   | `PYTHONUNBUFFERED` | `1` |
   | `PORT` | `8000` |
   | `DATABASE_URL` | *Your Neon PostgreSQL connection string with `?sslmode=require`* |
   | `REDIS_URL` | *Your Upstash Redis connection string (`rediss://...`)* |
   | `QDRANT_URL` | *Your Qdrant Cloud cluster endpoint* |
   | `QDRANT_API_KEY` | *Your Qdrant Cloud API key* |
   | `GEMINI_API_KEY` | *Your Google Gemini API key* |
   | `LLM_MODEL` | `gemini-3.5-flash-lite` |
   | `MODEL_VERSION` | `v2.0` |
   | `ADMIN_API_KEY` | `ds11_admin_secret_key_v2` |
   | `EXPLANATION_CACHE_TTL`| `86400` |

6. Click **Create Web Service**.
7. Render will build the Docker container and start FastAPI. Once the build finishes, your public backend URL will appear at the top (e.g., `https://amazon-recsys-api.onrender.com`).

---

## 7. Frontend Deployment on Vercel

### Step 7.1: Import Project on Vercel
1. Go to **[https://vercel.com/](https://vercel.com/)** and log in.
2. Click **Add New...** -> Select **Project**.
3. Select your GitHub repository (`amazon_project` / `rs`).

### Step 7.2: Configure Root Directory & Environment
1. In the **Configure Project** screen:
   - **Project Name**: `amazon-recsys-web`
   - **Framework Preset**: `Next.js`
   - **Root Directory**: Click **Edit** -> Select the **`web`** folder -> Click **Continue**.
   - **Build Command**: `npm run build` (auto-detected)
   - **Output Directory**: `.next` (auto-detected)
   - **Install Command**: `npm ci` or `npm install`
2. Expand the **Environment Variables** section:
   - **Key**: `NEXT_PUBLIC_API_URL`
   - **Value**: Your Render backend URL (e.g. `https://amazon-recsys-api.onrender.com`) — *do NOT add a trailing slash*.
   - Click **Add**.
3. Click **Deploy**.

Vercel will build the Next.js frontend and provide your production public URL (e.g. `https://amazon-recsys-web.vercel.app`).

---

## 8. Free-Tier Operational Mitigations

### 8.1 Prevent Render Backend from Sleeping (Cold Start Mitigation)
Render free services sleep after 15 minutes of inactivity (taking 30–60s to wake up on the next request).
To keep it warm:
1. Go to **[https://uptimerobot.com/](https://uptimerobot.com/)** and create a free account.
2. Click **Add New Monitor**:
   - **Monitor Type**: `HTTP(s)`
   - **Friendly Name**: `Amazon RecSys Backend Keep-Alive`
   - **URL (or IP)**: `https://amazon-recsys-api.onrender.com/v2/health`
   - **Monitoring Interval**: `14 minutes`
   - Click **Create Monitor**.
3. UptimeRobot will ping the `/v2/health` endpoint every 14 minutes, keeping the Render free container warm and responsive with zero cold start for demos.

### 8.2 Prevent Qdrant Cloud Inactivity Suspension
Qdrant Cloud free clusters automatically suspend after 7 days without queries.
- The repository includes `.github/workflows/scheduled_retrain.yml` which pings and syncs embeddings weekly, ensuring the cluster stays active continuously.

---

## 9. Live Verification & Testing

Once both services are deployed, run the connectivity verifier against your live Render domain:

```bash
python scripts/verify_tier1_connectivity.py --api-url https://amazon-recsys-api.onrender.com
```

Test the live endpoints via curl:
```bash
# 1. Health Diagnostics
curl -s https://amazon-recsys-api.onrender.com/v2/health

# 2. In-App Metrics Percentiles & Telemetry
curl -s https://amazon-recsys-api.onrender.com/metrics

# 3. Personalized Recommendation Query
curl -X POST https://amazon-recsys-api.onrender.com/v2/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "Alex_Gamer", "top_k": 5}'

# 4. Similar Items Vector Search
curl -s https://amazon-recsys-api.onrender.com/v2/similar/B00002N7S8
```

Open your Vercel frontend URL in any browser:
- Test the demo personas in the navigation bar (Alex, Elena, Marcus, Sarah, Devon, Aisha, Guest).
- Verify that recommendations dynamically re-rank with custom explanations and satisfaction scores.
- Test semantic search in the search bar and check the hybrid score breakdowns.
