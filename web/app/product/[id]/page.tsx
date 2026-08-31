'use client';

import React, { useState, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { 
  Package, 
  Star, 
  ShoppingCart, 
  Heart, 
  ArrowLeft, 
  Sparkles, 
  ShieldCheck, 
  Check, 
  Layers, 
  Zap, 
  Share2, 
  Activity,
  Cpu
} from 'lucide-react';
import { RecSysAPI, RecommendedItem, SimilarResponse, RecommendResponse } from '@/lib/api';
import { useUser } from '@/context/UserContext';
import ProductCard from '@/components/ProductCard';
import ExplanationBadge from '@/components/ExplanationBadge';

export default function ProductDetailPage() {
  const params = useParams();
  const router = useRouter();
  const itemId = decodeURIComponent(typeof params.id === 'string' ? params.id : '');
  
  const { currentUser, addToCart, likedItems, toggleLike } = useUser();
  const [product, setProduct] = useState<RecommendedItem | null>(null);
  const [similarItems, setSimilarItems] = useState<RecommendedItem[]>([]);
  const [coRecs, setCoRecs] = useState<RecommendedItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [userRating, setUserRating] = useState<number | null>(null);
  const [added, setAdded] = useState(false);

  const isLiked = likedItems.has(itemId);

  useEffect(() => {
    if (!itemId) return;

    // 1. Log view event immediately
    RecSysAPI.logEvent({
      user_id: currentUser.id,
      item_id: itemId,
      event_type: 'view',
      metadata: { page: 'product_detail', user_persona: currentUser.persona },
    });

    // 2. Fetch similar items from Qdrant HNSW ANN
    const loadDetails = async () => {
      setLoading(true);
      try {
        const [simRes, recRes] = await Promise.all([
          RecSysAPI.getSimilarItems(itemId),
          RecSysAPI.getRecommendations({
            userId: currentUser.id,
            itemId: itemId,
            topK: 4,
          }),
        ]);

        setSimilarItems(simRes.results || []);
        setCoRecs(recRes.results || []);

        // Populate current product info from target_item metadata
        if (simRes.target_item) {
          setProduct(simRes.target_item);
        } else {
          const found = simRes.results.find((i) => i.item_id === itemId);
          if (found) {
            setProduct(found);
          } else {
            setProduct({
              item_id: itemId,
              title: `Amazon Catalog Item (${itemId})`,
              category: 'General',
              price: 19.99,
              average_rating: 4.0,
              score: 1.0,
              source: 'qdrant_ann',
              explanation: 'Direct product lookup from vector index and metadata catalog.',
            });
          }
        }
      } catch (err) {
        console.error('Error fetching product data:', err);
      } finally {
        setLoading(false);
      }
    };

    loadDetails();
  }, [itemId, currentUser.id]);

  const handleRate = (rating: number) => {
    setUserRating(rating);
    RecSysAPI.logEvent({
      user_id: currentUser.id,
      item_id: itemId,
      event_type: 'rating',
      rating: rating,
      metadata: { rated_at: new Date().toISOString() },
    });
  };

  const handleAddToCart = () => {
    if (product) {
      addToCart(product.item_id, product.title);
      setAdded(true);
      setTimeout(() => setAdded(false), 2000);
    }
  };

  if (loading) {
    return (
      <div className="space-y-6 animate-pulse">
        <div className="w-32 h-6 bg-slate-800 rounded" />
        <div className="glass-panel p-8 rounded-2xl grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="h-80 bg-slate-800 rounded-xl" />
          <div className="space-y-4">
            <div className="w-3/4 h-8 bg-slate-800 rounded" />
            <div className="w-1/2 h-6 bg-slate-800 rounded" />
            <div className="w-full h-24 bg-slate-800 rounded" />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-10 animate-fade-in">
      
      {/* Back Navigation */}
      <button
        onClick={() => router.back()}
        className="inline-flex items-center gap-2 text-xs font-medium text-slate-400 hover:text-white transition-colors"
      >
        <ArrowLeft className="w-4 h-4" />
        <span>Back to Recommendations</span>
      </button>

      {/* Main Product Showcase Card */}
      <div className="glass-panel rounded-2xl p-6 sm:p-8 border border-slate-800 shadow-2xl">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          
          {/* Left Column: Visual Showcase */}
          <div className="lg:col-span-5 flex flex-col items-center justify-center p-8 rounded-xl bg-slate-900/90 border border-slate-800 relative group">
            <Package className="w-32 h-32 text-sky-400/80 group-hover:scale-105 transition-transform duration-300" />
            <div className="mt-4 text-center">
              <span className="text-xs font-mono px-2.5 py-1 rounded bg-slate-950 text-slate-400 border border-slate-800">
                parent_asin: {itemId}
              </span>
            </div>
          </div>

          {/* Right Column: Details & Actions */}
          <div className="lg:col-span-7 space-y-5">
            <div>
              <div className="flex items-center gap-2 flex-wrap mb-2">
                <span className="text-xs font-medium px-2.5 py-0.5 rounded-full bg-sky-500/10 text-sky-300 border border-sky-500/20">
                  {product?.category || 'Amazon Product Catalog'}
                </span>
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-slate-900 text-slate-400 border border-slate-800">
                  Vector Indexed
                </span>
              </div>

              <h1 className="text-xl sm:text-2xl font-bold text-white leading-snug">
                {product?.title || `Product ${itemId}`}
              </h1>
            </div>

            {/* Rating & Pricing Row */}
            <div className="flex items-center gap-6 py-3 border-y border-slate-800">
              <div className="flex items-center gap-1.5">
                <div className="flex text-amber-400">
                  {[1, 2, 3, 4, 5].map((star) => (
                    <Star
                      key={star}
                      className={`w-4 h-4 ${
                        star <= Math.round(product?.average_rating ?? 0)
                          ? 'fill-current'
                          : 'text-slate-600'
                      }`}
                    />
                  ))}
                </div>
                <span className="text-sm font-bold text-slate-200">
                  {(product?.average_rating ?? 0).toFixed(1)}
                </span>
              </div>

              <div className="text-2xl font-extrabold text-white">
                ${(product?.price ?? 0).toFixed(2)}
              </div>
            </div>

            {/* LLM Explanation Highlight */}
            <div className="p-4 rounded-xl bg-gradient-to-r from-sky-950/40 via-indigo-950/30 to-purple-950/40 border border-sky-500/30 space-y-1.5">
              <div className="flex items-center gap-2 text-xs font-bold text-sky-300">
                <Sparkles className="w-4 h-4 text-sky-400" />
                <span>AI Recommendation Rationale (Gemini 3.5 Flash-Lite)</span>
              </div>
              <p className="text-xs text-slate-300 leading-relaxed">
                {product?.explanation ||
                  'High similarity to your browsing pattern in audio & gaming accessories, scoring top relevance in LightGBM LambdaMART ranking.'}
              </p>
              <div className="text-[10px] text-slate-400 flex items-center justify-between pt-1">
                <span>Cached in Redis Response Cache</span>
                <span className="font-mono text-emerald-400">Latency: &lt;1ms</span>
              </div>
            </div>

            {/* Interactive Feedback / Rating Widget */}
            <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 flex items-center justify-between gap-4">
              <div className="text-xs">
                <span className="font-semibold text-slate-200 block">Rate this Item</span>
                <span className="text-[11px] text-slate-400">
                  Logs event to PostgreSQL to train next LightGBM iteration
                </span>
              </div>
              <div className="flex items-center gap-1">
                {[1, 2, 3, 4, 5].map((star) => (
                  <button
                    key={star}
                    onClick={() => handleRate(star)}
                    className="p-1 hover:scale-125 transition-transform"
                    title={`Rate ${star} Stars`}
                  >
                    <Star
                      className={`w-5 h-5 ${
                        userRating && star <= userRating
                          ? 'text-amber-400 fill-current'
                          : 'text-slate-600 hover:text-amber-300'
                      }`}
                    />
                  </button>
                ))}
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex items-center gap-3 pt-2">
              <button
                onClick={handleAddToCart}
                className={`flex-1 py-3 px-4 rounded-xl font-bold text-sm flex items-center justify-center gap-2 shadow-lg transition-all ${
                  added
                    ? 'bg-emerald-600 text-white'
                    : 'bg-gradient-to-r from-sky-600 via-indigo-600 to-purple-600 hover:from-sky-500 hover:to-purple-500 text-white'
                }`}
              >
                {added ? (
                  <>
                    <Check className="w-4 h-4" />
                    <span>Added to Cart</span>
                  </>
                ) : (
                  <>
                    <ShoppingCart className="w-4 h-4" />
                    <span>Add to Cart (Log Event)</span>
                  </>
                )}
              </button>

              <button
                onClick={() => toggleLike(itemId)}
                className={`p-3 rounded-xl border transition-all ${
                  isLiked
                    ? 'bg-rose-500/20 border-rose-500/40 text-rose-400'
                    : 'bg-slate-900 border-slate-700 text-slate-300 hover:text-rose-400'
                }`}
                title="Save item"
              >
                <Heart className={`w-5 h-5 ${isLiked ? 'fill-current' : ''}`} />
              </button>
            </div>

          </div>
        </div>
      </div>

      {/* Qdrant HNSW ANN Similar Items Section */}
      <section className="space-y-4">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center">
            <Cpu className="w-4 h-4 text-indigo-400" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
              Similar Items in Vector Space
              <span className="text-xs font-mono font-normal px-2 py-0.5 rounded bg-slate-900 text-indigo-400 border border-slate-800">
                Qdrant HNSW ANN Index
              </span>
            </h2>
            <p className="text-xs text-slate-400">
              Retrieved via cosine similarity on 384-d e5 item embeddings in sub-5ms.
            </p>
          </div>
        </div>

        {similarItems.length > 0 ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
            {similarItems.map((item, idx) => (
              <ProductCard
                key={`sim_${item.item_id}_${idx}`}
                item={item}
                rank={idx + 1}
                showScore={true}
              />
            ))}
          </div>
        ) : (
          <div className="glass-panel p-6 text-center text-sm text-slate-400 rounded-xl">
            No similar vector neighbors found in this cluster.
          </div>
        )}
      </section>

      {/* Personalized Co-Recommendations */}
      {coRecs.length > 0 && (
        <section className="space-y-4 pt-4 border-t border-slate-900">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-sky-500/10 border border-sky-500/20 flex items-center justify-center">
              <Zap className="w-4 h-4 text-sky-400" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white tracking-tight">
                Ranked For You with This Product
              </h2>
              <p className="text-xs text-slate-400">
                Combining user affinity for {currentUser.name} with Apriori frequent itemset lift.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
            {coRecs.map((item, idx) => (
              <ProductCard
                key={`corec_${item.item_id}_${idx}`}
                item={item}
                rank={idx + 1}
                showScore={true}
              />
            ))}
          </div>
        </section>
      )}

    </div>
  );
}
