/**
 * API Client for Amazon RecSys v2 Backend Service.
 * Provides type-safe endpoints for recommendations, search, similarity,
 * interaction telemetry, health diagnostics, and metrics.
 */

export interface RecommendedItem {
  item_id: string;
  title: string;
  score: number;
  source: string;
  category?: string | null;
  price?: number | null;
  average_rating?: number | null;
  explanation?: string | null;
  feature_signals?: Record<string, number> | null;
}

export interface RecommendResponse {
  user_id: string;
  item_id?: string | null;
  cold_start: boolean;
  source: string;
  results: RecommendedItem[];
  model_version: string;
}

export interface SimilarResponse {
  item_id: string;
  results: RecommendedItem[];
}

export interface SearchResult {
  item_id: string;
  title?: string | null;
  category?: string | null;
  price?: number | null;
  average_rating?: number | null;
  hybrid_score: number;
  emb_score: number;
  bm25_score: number;
}

export interface SearchResponse {
  query: string;
  rewritten_query?: string | null;
  category_filter?: string | null;
  price_max?: number | null;
  intent?: string | null;
  results: SearchResult[];
}

export interface EventCreateRequest {
  user_id: string;
  item_id: string;
  event_type: 'click' | 'view' | 'purchase' | 'rating' | 'cart';
  rating?: number;
  metadata?: Record<string, any>;
}

export interface EventResponse {
  status: string;
  event_id?: string | number | null;
  message?: string | null;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  ranker_loaded: boolean;
  vector_db_connected: boolean;
  redis_connected: boolean;
  db_connected: boolean;
  n_items?: number | null;
  version: string;
}

export interface MetricsResponse {
  total_requests: number;
  requests_per_endpoint: Record<string, number>;
  requests_per_status: Record<string, number>;
  latency_p50_ms: number;
  latency_p95_ms: number;
  latency_p99_ms: number;
  cache_hits: number;
  cache_misses: number;
  cache_hit_rate: number;
  status: string;
}

export interface AdminRetrainResponse {
  status: string;
  message: string;
  job_id?: string | null;
  started_at?: string | null;
}

export interface AdminRetrainStatusResponse {
  status: string;
  job_id?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  return_code?: number | null;
  log_tail?: string | null;
}

const API_BASE = typeof window !== 'undefined'
  ? (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000')
  : (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000');

async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
      ...options?.headers,
    },
  });

  if (!res.ok) {
    const errorText = await res.text();
    let errorDetail = errorText;
    try {
      const parsed = JSON.parse(errorText);
      errorDetail = parsed.detail || parsed.message || errorText;
    } catch {}
    throw new Error(`API Error [${res.status}]: ${errorDetail}`);
  }

  return res.json() as Promise<T>;
}

// Fallback seed catalog for offline / disconnected UI demonstrations
const MOCK_ITEMS: RecommendedItem[] = [
  {
    item_id: "B08N5WRWNW",
    title: "Sony WH-1000XM5 Wireless Industry Leading Noise Canceling Headphones",
    category: "Electronics",
    price: 348.00,
    average_rating: 4.8,
    score: 0.965,
    source: "personalized_ranker",
    explanation: "Frequently bought with audio gear and matches your preference for premium wireless audio.",
    feature_signals: {
      "als_cf_score": 0.912,
      "neural_cf_score": 0.885,
      "content_similarity": 0.945,
      "apriori_lift": 1.720,
      "popularity_weight": 0.980,
      "price_affinity": 0.820
    }
  },
  {
    item_id: "B07W65PB6K",
    title: "Logitech G PRO X Superlight Wireless Gaming Mouse - Ultra-Lightweight",
    category: "Video Games",
    price: 119.99,
    average_rating: 4.7,
    score: 0.942,
    source: "personalized_ranker",
    explanation: "Top candidate from Neural Collaborative Filtering based on your gaming accessories history.",
    feature_signals: {
      "als_cf_score": 0.880,
      "neural_cf_score": 0.940,
      "content_similarity": 0.890,
      "apriori_lift": 1.450,
      "popularity_weight": 0.950,
      "price_affinity": 0.870
    }
  },
  {
    item_id: "B091J3NYVF",
    title: "Keychron Q1 QMK Custom Mechanical Keyboard - Hot-Swappable RGB",
    category: "Video Games",
    price: 169.00,
    average_rating: 4.9,
    score: 0.918,
    source: "personalized_ranker",
    explanation: "High Apriori lift score with recently viewed mechanical switches and desk mats.",
    feature_signals: {
      "als_cf_score": 0.860,
      "neural_cf_score": 0.895,
      "content_similarity": 0.920,
      "apriori_lift": 2.150,
      "popularity_weight": 0.890,
      "price_affinity": 0.840
    }
  },
  {
    item_id: "B0B8K7F81P",
    title: "Anker Prime 20,000mAh Power Bank (200W Output with Smart Digital Display)",
    category: "Cell Phones & Accessories",
    price: 89.99,
    average_rating: 4.6,
    score: 0.884,
    source: "personalized_ranker",
    explanation: "Matches your affinity for fast-charging multi-device power delivery gear.",
    feature_signals: {
      "als_cf_score": 0.820,
      "neural_cf_score": 0.840,
      "content_similarity": 0.870,
      "apriori_lift": 1.250,
      "popularity_weight": 0.910,
      "price_affinity": 0.930
    }
  },
  {
    item_id: "B09B8W5FW7",
    title: "Elgato Stream Deck MK.2 - 15 Customizable LCD Keys for Content Creators",
    category: "Video Games",
    price: 149.99,
    average_rating: 4.8,
    score: 0.865,
    source: "personalized_ranker",
    explanation: "Recommended based on two-tower embedding similarity to high-activity streamer profiles.",
    feature_signals: {
      "als_cf_score": 0.790,
      "neural_cf_score": 0.850,
      "content_similarity": 0.880,
      "apriori_lift": 1.350,
      "popularity_weight": 0.870,
      "price_affinity": 0.810
    }
  }
];

export const RecSysAPI = {
  /**
   * Request personalized recommendations for a user.
   */
  async getRecommendations(params: {
    userId: string;
    itemId?: string;
    topK?: number;
    categoryFilter?: string;
  }): Promise<RecommendResponse> {
    try {
      return await fetchJSON<RecommendResponse>(`${API_BASE}/v2/recommend`, {
        method: 'POST',
        body: JSON.stringify({
          user_id: params.userId,
          item_id: params.itemId || null,
          top_k: params.topK || 12,
          category_filter: params.categoryFilter || null,
        }),
      });
    } catch (err) {
      console.warn('Backend unavailable, using fallback recommendations:', err);
      return {
        user_id: params.userId,
        item_id: params.itemId || null,
        cold_start: params.userId === 'guest_cold_start',
        source: params.userId === 'guest_cold_start' ? 'popular_baseline' : 'personalized_ranker',
        results: MOCK_ITEMS.slice(0, params.topK || 6),
        model_version: 'v2.0-fallback',
      };
    }
  },

  /**
   * Retrieve Qdrant HNSW ANN similar items for a product.
   */
  async getSimilarItems(itemId: string): Promise<SimilarResponse> {
    try {
      return await fetchJSON<SimilarResponse>(`${API_BASE}/v2/similar/${encodeURIComponent(itemId)}`);
    } catch (err) {
      console.warn(`Backend /v2/similar/${itemId} failed, returning mock neighbors:`, err);
      return {
        item_id: itemId,
        results: MOCK_ITEMS.filter((i) => i.item_id !== itemId).slice(0, 4),
      };
    }
  },

  /**
   * Execute free-text semantic + BM25 search with LLM query rewrite.
   */
  async search(query: string): Promise<SearchResponse> {
    try {
      return await fetchJSON<SearchResponse>(`${API_BASE}/v2/search?q=${encodeURIComponent(query)}`);
    } catch (err) {
      console.warn('Search backend unavailable, using mock search response:', err);
      const filtered = MOCK_ITEMS.filter((i) =>
        i.title.toLowerCase().includes(query.toLowerCase()) ||
        (i.category && i.category.toLowerCase().includes(query.toLowerCase()))
      );
      const results: SearchResult[] = (filtered.length ? filtered : MOCK_ITEMS).map((item, idx) => ({
        item_id: item.item_id,
        title: item.title,
        category: item.category,
        price: item.price,
        average_rating: item.average_rating,
        hybrid_score: Math.max(0.4, 0.95 - idx * 0.08),
        emb_score: Math.max(0.35, 0.92 - idx * 0.07),
        bm25_score: Math.max(0.3, 0.88 - idx * 0.09),
      }));

      return {
        query,
        rewritten_query: `high quality ${query} with high ratings`,
        category_filter: query.toLowerCase().includes('game') ? 'Video Games' : 'Electronics',
        price_max: 250,
        intent: 'product_discovery',
        results,
      };
    }
  },

  /**
   * Log user interaction or feedback event to PostgreSQL.
   */
  async logEvent(event: EventCreateRequest): Promise<EventResponse> {
    try {
      return await fetchJSON<EventResponse>(`${API_BASE}/v2/events`, {
        method: 'POST',
        body: JSON.stringify(event),
      });
    } catch (err) {
      console.warn('Failed to log event to backend:', err);
      return { status: 'mock_logged', event_id: `local_${Date.now()}` };
    }
  },

  /**
   * Get multi-service health status.
   */
  async getHealth(): Promise<HealthResponse> {
    try {
      return await fetchJSON<HealthResponse>(`${API_BASE}/v2/health`);
    } catch (err) {
      return {
        status: 'offline_or_starting',
        model_loaded: false,
        ranker_loaded: false,
        vector_db_connected: false,
        redis_connected: false,
        db_connected: false,
        n_items: 0,
        version: '2.0.0',
      };
    }
  },

  /**
   * Get real-time JSON telemetry metrics.
   */
  async getMetrics(): Promise<MetricsResponse> {
    try {
      return await fetchJSON<MetricsResponse>(`${API_BASE}/metrics`);
    } catch (err) {
      return {
        total_requests: 142,
        requests_per_endpoint: {
          '/v2/recommend': 86,
          '/v2/search': 32,
          '/v2/similar/{item_id}': 14,
          '/v2/events': 10,
        },
        requests_per_status: { '200': 140, '404': 2 },
        latency_p50_ms: 18.4,
        latency_p95_ms: 46.2,
        latency_p99_ms: 78.5,
        cache_hits: 54,
        cache_misses: 32,
        cache_hit_rate: 0.628,
        status: 'demo_telemetry',
      };
    }
  },

  /**
   * Trigger DVC retrain pipeline.
   */
  async triggerRetrain(force: boolean = false, targets?: string[]): Promise<AdminRetrainResponse> {
    return await fetchJSON<AdminRetrainResponse>(`${API_BASE}/admin/retrain`, {
      method: 'POST',
      body: JSON.stringify({ force, targets }),
    });
  },

  /**
   * Check status of a retrain job.
   */
  async getRetrainStatus(jobId: string): Promise<AdminRetrainStatusResponse> {
    return await fetchJSON<AdminRetrainStatusResponse>(`${API_BASE}/admin/retrain/status/${jobId}`);
  },
};
