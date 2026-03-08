"""
main.py — FastAPI service for 20 Newsgroups Semantic Search.

ENDPOINTS (required by task):
  POST   /query             — semantic search with cluster-partitioned cache
  GET    /cache/stats       — cache performance statistics
  DELETE /cache             — flush cache and reset stats

ENDPOINTS (additional, for analysis and debugging):
  GET    /health            — service health check
  GET    /clusters/summary  — per-cluster document counts and quality metrics
  GET    /clusters/{id}/docs — top documents in a specific cluster
  GET    /clusters/boundary — most semantically ambiguous documents
  POST   /cache/threshold   — update similarity threshold τ at runtime
  POST   /cache/analyze     — sweep τ values and show precision/recall tradeoff
  GET    /cache/inspect     — inspect all cache entries (debug)
  GET    /results/k_plot    — serve the K-selection elbow plot

STATE MANAGEMENT: FastAPI lifespan context manager
  All heavy state (HNSW index, UMAP model, FCM model, embedding model, cache)
  is loaded ONCE at startup and shared across all requests.
  On shutdown, the cache is persisted to disk for warm restarts.
  
  Using a lifespan context manager (not @app.on_event) is the correct pattern
  for FastAPI >= 0.93 — it avoids deprecation warnings and handles cleanup
  even on exceptions.
"""

import time
import os
import numpy as np
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse

from app.db import (
    init_db, get_doc_count, get_docs_batch, get_doc_by_id,
    get_cluster_summary, get_cluster_docs, get_boundary_documents,
    get_newsgroup_counts
)
from app.embedder import embed_text
from app.vector_store import load_index, search, get_index_size
from app.clusterer import load_models, get_query_membership, get_dominant_clusters, get_k
from app.cache import get_cache
from app.models import (
    QueryRequest, QueryResponse, DocumentResult,
    CacheStatsResponse, ThresholdRequest, ClusterSummaryItem,
    BoundaryDocument, HealthResponse
)


# ── Application state (loaded once at startup) ────────────────────────────────

class AppState:
    models_loaded: bool = False
    index_loaded: bool = False


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all heavy state once at startup; persist cache on shutdown."""
    print("[Startup] Initializing service...")

    init_db()

    state.index_loaded = load_index()
    if not state.index_loaded:
        print("[Startup] WARNING: Vector index not found. Run 02_build_index.py first.")

    state.models_loaded = load_models()
    if not state.models_loaded:
        print("[Startup] WARNING: Cluster models not found. Run 03_build_clusters.py first.")

    # Load cache from disk (warm restart — cache survives container restarts)
    cache = get_cache()
    cache.load()

    print(f"[Startup] Ready. Index: {get_index_size()} vectors, "
          f"Cache: {cache.stats()['total_entries']} entries, "
          f"K={get_k()} clusters")

    yield  # ← service runs here

    # Shutdown: persist cache to disk
    print("[Shutdown] Persisting cache...")
    cache.persist()
    print("[Shutdown] Done.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="20 Newsgroups Semantic Search",
    version="1.0.0",
    description="""
Semantic search over ~16,000 newsgroup posts with fuzzy clustering
and a cluster-partitioned semantic cache.

**Architecture highlights:**
- Embeddings: all-MiniLM-L6-v2 (384-d) with chunk mean-pooling
- Vector store: usearch HNSW graph (O(log N) ANN search, pre-built wheels for all platforms)
- Clustering: Fuzzy C-Means from scratch on UMAP-50 reduced embeddings (K by PC elbow)
- Cache: Cluster-partitioned semantic cache with LRU+TTL eviction (NO Redis, built from scratch)
- Similarity: cosine (= dot product for L2-normalised vectors)
    """,
    lifespan=lifespan,
)


# ══════════════════════════════════════════════════════════════════════════════
# POST /query — Core search endpoint (REQUIRED)
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Semantic search with cluster-partitioned caching.

    Flow:
    1. Embed the query (384-d float32 vector)
    2. Get soft cluster membership (K-dim probability vector via UMAP + FCM)
    3. Check semantic cache (search only dominant cluster partitions → O(N/K))
    4. On HIT: return cached result immediately (no vector search needed)
    5. On MISS: search HNSW index → fetch documents → store in cache → return
    """
    t_start = time.time()

    # Step 1: Embed query
    query_vec = embed_text(req.query)

    # Step 2: Get cluster membership
    if state.models_loaded:
        membership = get_query_membership(query_vec)
        dominant_clusters = get_dominant_clusters(membership)
        dominant_cluster = dominant_clusters[0]
    else:
        membership = np.array([1.0])
        dominant_cluster = 0
        dominant_clusters = [0]

    # Step 3: Cache lookup
    cache = get_cache()
    hit = cache.lookup(query_vec, membership)

    if hit is not None:
        # CACHE HIT — return immediately
        search_time_ms = (time.time() - t_start) * 1000
        return QueryResponse(
            query=req.query,
            cache_hit=True,
            matched_query=hit.query,
            similarity_score=round(float(np.dot(query_vec, hit.embedding)), 4),
            result=hit.result,
            dominant_cluster=dominant_cluster,
            query_membership=membership.tolist(),
            search_time_ms=round(search_time_ms, 2),
        )

    # CACHE MISS — run vector search
    if not state.index_loaded:
        raise HTTPException(503, "Vector index not loaded. Run build scripts first.")

    # Get doc_ids in dominant clusters for filtered search
    filter_doc_ids = None
    if state.models_loaded:
        from app.db import get_cluster_docs as gcd
        cluster_docs = []
        for cid in dominant_clusters:
            rows = gcd(cid, limit=5000)
            cluster_docs.extend([r["id"] for r in rows])
        filter_doc_ids = cluster_docs if cluster_docs else None

    search_results = search(query_vec, k=req.top_k, filter_ids=filter_doc_ids)

    if not search_results:
        # Fallback: unfiltered search
        search_results = search(query_vec, k=req.top_k, filter_ids=None)

    # Fetch document details from SQLite
    doc_ids = [r[0] for r in search_results]
    sim_map = {r[0]: r[1] for r in search_results}
    docs = get_docs_batch(doc_ids) if doc_ids else []

    # Build result list
    result_docs = []
    for doc in docs:
        doc_id = doc["id"]
        # Get this doc's membership vector from DB
        from app.db import get_membership
        doc_membership = get_membership(doc_id, get_k())
        doc_cluster = int(np.argmax(doc_membership)) if doc_membership is not None else 0
        doc_memb_list = doc_membership.tolist() if doc_membership is not None else []

        result_docs.append(DocumentResult(
            doc_id=doc_id,
            newsgroup=doc["newsgroup"],
            subject=doc.get("subject", ""),
            text_preview=doc["raw_text"][:300] + "..." if len(doc["raw_text"]) > 300 else doc["raw_text"],
            similarity_score=round(sim_map.get(doc_id, 0.0), 4),
            dominant_cluster=doc_cluster,
            cluster_membership=doc_memb_list,
        ))

    # Sort by similarity
    result_docs.sort(key=lambda x: -x.similarity_score)

    # Step 5: Store in cache
    cache.store(
        query=req.query,
        query_embedding=query_vec,
        result=result_docs,
        membership=membership,
    )

    search_time_ms = (time.time() - t_start) * 1000
    return QueryResponse(
        query=req.query,
        cache_hit=False,
        matched_query=None,
        similarity_score=None,
        result=result_docs,
        dominant_cluster=dominant_cluster,
        query_membership=membership.tolist(),
        search_time_ms=round(search_time_ms, 2),
    )


# ══════════════════════════════════════════════════════════════════════════════
# GET /cache/stats — Cache statistics (REQUIRED)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/cache/stats")
def cache_stats():
    """
    Return current cache performance statistics.
    
    Key metrics:
    - hit_rate: fraction of queries served from cache (higher = more efficient)
    - partition_sizes: entries per cluster partition (shows distribution)
    - similarity_threshold: current τ value
    """
    return get_cache().stats()


# ══════════════════════════════════════════════════════════════════════════════
# DELETE /cache — Flush cache (REQUIRED)
# ══════════════════════════════════════════════════════════════════════════════

@app.delete("/cache")
def flush_cache():
    """Flush all cache entries and reset hit/miss counters."""
    get_cache().flush()
    return {"status": "flushed", "message": "Cache cleared and stats reset."}


# ══════════════════════════════════════════════════════════════════════════════
# Additional endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse)
def health():
    """Service health check — confirms all components are loaded."""
    return HealthResponse(
        status="ok" if (state.index_loaded and state.models_loaded) else "degraded",
        index_size=get_index_size(),
        cache_entries=get_cache().stats()["total_entries"],
        models_loaded=state.models_loaded,
        k_clusters=get_k(),
    )


@app.get("/clusters/summary")
def clusters_summary():
    """
    Per-cluster summary: document counts, average FCM certainty, boundary counts.
    
    - avg_certainty: mean max_membership across cluster docs. Near 1.0 = crisp cluster.
    - boundary_count: docs with max_membership < 0.4 — genuinely cross-topic posts.
    """
    rows = get_cluster_summary()
    return {
        "clusters": rows,
        "total_clusters": len(rows),
        "note": "boundary_count = docs with max_membership < 0.4 (semantically ambiguous)"
    }


@app.get("/clusters/{cluster_id}/docs")
def cluster_docs(cluster_id: int, limit: int = 10):
    """Top documents in cluster `cluster_id`, ordered by membership certainty."""
    docs = get_cluster_docs(cluster_id, limit=limit)
    if not docs:
        raise HTTPException(404, f"No documents found for cluster {cluster_id}")
    return {
        "cluster_id": cluster_id,
        "count": len(docs),
        "documents": [
            {
                "doc_id": d["id"],
                "newsgroup": d["newsgroup"],
                "subject": d.get("subject", ""),
                "preview": d["raw_text"][:200],
                "membership": d["max_membership"],
            }
            for d in docs
        ]
    }


@app.get("/clusters/boundary")
def boundary_docs(limit: int = 20):
    """
    Return documents with max cluster membership < 0.4.
    These are the most semantically ambiguous posts — cross-topic content.
    
    Example: a post about gun legislation in a political newsgroup
    might have membership [0.35_politics, 0.32_firearms, 0.20_law, ...].
    These are the MOST INTERESTING documents — they sit between clusters.
    """
    docs = get_boundary_documents(limit=limit)
    result = []
    for d in docs:
        memb = np.frombuffer(d["membership_blob"], dtype=np.float32)
        top3 = sorted(enumerate(memb.tolist()), key=lambda x: -x[1])[:3]
        result.append({
            "doc_id": d["id"],
            "newsgroup": d["newsgroup"],
            "subject": d.get("subject", ""),
            "preview": d["raw_text"][:250],
            "max_membership": round(d["max_membership"], 3),
            "dominant_cluster": d["dominant_cluster"],
            "top_3_memberships": [{"cluster": c, "membership": round(m, 3)}
                                   for c, m in top3],
        })
    return {
        "count": len(result),
        "threshold": 0.4,
        "note": "Documents with max_membership < 0.4 belong to multiple clusters equally",
        "documents": result,
    }


@app.post("/cache/threshold")
def update_threshold(req: ThresholdRequest):
    """
    Update similarity threshold τ at runtime — no restart needed.
    
    This endpoint lets you explore how τ affects system behaviour:
    - τ=0.99: near-exact match only → very low hit rate
    - τ=0.92: paraphrase-level → good precision/recall balance (default)
    - τ=0.85: topic-level → higher hit rate, may reduce precision
    
    The interesting question is what each value reveals about your cache structure.
    """
    get_cache().set_threshold(req.threshold)
    return {
        "status": "updated",
        "new_threshold": req.threshold,
        "interpretation": {
            ">=0.95": "near-exact paraphrase only",
            "0.90-0.95": "strong paraphrase (recommended)",
            "0.85-0.90": "topic-level similarity",
            "<0.85": "loose match (may reduce answer precision)",
        }
    }


@app.post("/cache/analyze")
def analyze_threshold():
    """
    Sweep τ from 0.75 to 0.99 on current cache entries.
    Shows how hit rate and precision change with τ.
    
    This reveals the semantic structure of your cached queries:
    - If hit rate drops sharply at τ=0.93 → cache has strong paraphrase clusters
    - If hit rate is flat → queries are semantically diverse (little redundancy)
    
    Run several queries first to populate the cache, then call this endpoint.
    """
    results = get_cache().analyze_threshold()
    if not results:
        return {"message": "Cache is empty. Run some queries first.", "results": []}
    return {
        "current_threshold": get_cache().similarity_threshold,
        "sweep": results,
        "recommendation": "Choose the τ where hit_rate begins to drop — this is the precision knee."
    }


@app.get("/cache/inspect")
def cache_inspect():
    """Inspect all cache entries — shows query, cluster, access count."""
    entries = get_cache().get_all_entries()
    return {
        "total_entries": len(entries),
        "entries": entries
    }


@app.get("/corpus/stats")
def corpus_stats():
    """Corpus statistics — doc count, newsgroup distribution."""
    return {
        "total_documents": get_doc_count(),
        "newsgroup_counts": get_newsgroup_counts(),
        "index_size": get_index_size(),
    }


@app.get("/results/k_plot")
def k_plot():
    """Serve the K-selection elbow plot image."""
    path = "results/k_selection.png"
    if not os.path.exists(path):
        raise HTTPException(404, "K-selection plot not found. Run 03_build_clusters.py first.")
    return FileResponse(path, media_type="image/png")
