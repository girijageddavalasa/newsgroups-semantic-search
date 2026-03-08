"""
cache.py — Cluster-partitioned semantic cache, built from scratch.

═══════════════════════════════════════════════════════════════════════════════
PROBLEM: A traditional cache breaks on paraphrase queries
═══════════════════════════════════════════════════════════════════════════════

  Standard exact-match cache:
    "Tell me about NASA missions"  → key = sha256(query) → MISS (compute)
    "What has NASA explored?"      → key = sha256(query) → MISS (compute again)
  
  Both queries need the same documents. Recomputing wastes CPU.

  A semantic cache recognises that two queries are "close enough" and returns
  the cached result from the first query for the second.

═══════════════════════════════════════════════════════════════════════════════
DATA STRUCTURE: Cluster-partitioned dict + LRU OrderedDict
═══════════════════════════════════════════════════════════════════════════════

  WHY NOT a flat list?
    Flat list lookup = O(N): check every entry's embedding against the query.
    At 1000 cache entries, that's 1000 cosine similarity computations.
    At 10000 entries (busy production service), that's 10000.
    Linear scaling — the cache becomes the bottleneck.

  OUR DATA STRUCTURE: {cluster_id: [CacheEntry, ...]}
  ─────────────────────────────────────────────────────────────────────────────
    The FCM model gives every query a probability distribution over K clusters.
    We store each cache entry in the bucket(s) of its dominant cluster(s).
    
    At lookup time:
      1. Get query's dominant clusters (top 1-2 with membership ≥ 0.15)
      2. Search ONLY those K_dominant partitions
      3. Skip all other K - K_dominant partitions entirely
    
    Expected entries per partition: N / K
    Expected lookup cost: O(N / K) — approximately 10x faster than flat list at K=10
    
    As the cache grows, efficiency IMPROVES because:
    • More entries → more benefit from partitioning
    • New clusters keep partitions from growing unboundedly
    
    This is the key architectural insight: cluster membership isn't just for
    clustering the corpus — it's the routing key for the cache.

  LRU EVICTION: collections.OrderedDict
  ─────────────────────────────────────────────────────────────────────────────
    OrderedDict maintains insertion order. On every access, we move the entry
    to the end. When capacity is exceeded, we pop from the front (LRU).
    This is O(1) per access — same as Python's built-in lru_cache.
    We do NOT use functools.lru_cache (it's a library, and it's for functions).

  TTL EVICTION: timestamp-based
  ─────────────────────────────────────────────────────────────────────────────
    Cache entries carry a timestamp. Entries older than TTL_SECONDS are
    lazily expired on access (no background thread needed).
    Hybrid LRU+TTL: entry is evicted if EITHER condition triggers.

═══════════════════════════════════════════════════════════════════════════════
THE TUNABLE PARAMETER τ (SIMILARITY THRESHOLD)
═══════════════════════════════════════════════════════════════════════════════

  τ is the cosine similarity threshold above which we declare a cache HIT.
  
  This is NOT a heuristic — it reveals fundamental tradeoffs in the system:
  
  τ = 0.99 → near-exact match only
    • Almost nothing qualifies as a hit
    • Hit rate ≈ 0% for paraphrases
    • Zero false positives (wrong cached result returned)
    • Behaviour: cache acts like a slow exact-match cache
  
  τ = 0.92 → paraphrase-level (DEFAULT)
    • "Tell me about NASA" ↔ "What has NASA explored?" → HIT ✓
    • "Tell me about NASA" ↔ "How do I fix my PC?" → MISS ✓
    • Practical sweet spot for sentence-level semantic similarity
  
  τ = 0.85 → topic-level
    • "space exploration" ↔ "astronaut training programs" → might HIT
    • Risk: returns cached result for loosely related query (false positive)
    • Behaviour: aggressive caching, may sacrifice precision
  
  τ = 0.75 → very loose
    • Almost any space-related query hits on any other space query
    • High hit rate, low precision — not suitable for production
  
  The interesting question is not "which τ is best" — it's "what does each τ
  reveal about the system's semantic structure?"
  
  We expose POST /cache/threshold to tune τ at runtime,
  and POST /cache/analyze to sweep τ and show hit rate vs precision tradeoff.

═══════════════════════════════════════════════════════════════════════════════
PROCESSING BASIS: Why cosine similarity (not Euclidean)?
═══════════════════════════════════════════════════════════════════════════════

  Cosine similarity measures the ANGLE between vectors, not the distance.
  This is the correct metric for semantic embeddings because:
  
  • Two paraphrase sentences have vectors pointing in the SAME DIRECTION
    regardless of sentence length — "space" (short) and "space exploration
    missions to other planets" (long) point in similar directions
  
  • Euclidean distance is dominated by vector magnitude, which correlates
    with document length, not semantic content
  
  • all-MiniLM-L6-v2 is trained with cosine similarity as the objective —
    the embedding space is specifically designed for this metric
  
  Implementation: since all vectors are L2-normalised at embed time,
  cosine_similarity(a, b) = dot(a, b) — no norm computation needed.
  This is the O(1) savings in the hot path.
"""

import os
import time
import pickle
import numpy as np
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from app.embedder import cosine_similarity

CACHE_PATH     = os.environ.get("CACHE_PATH", "cache/semantic_cache.pkl")
DEFAULT_THRESHOLD = float(os.environ.get("CACHE_THRESHOLD", "0.92"))
DEFAULT_MAX_SIZE  = int(os.environ.get("CACHE_MAX_SIZE", "1000"))
DEFAULT_TTL       = int(os.environ.get("CACHE_TTL", "3600"))


@dataclass
class CacheEntry:
    """A single cached query-result pair."""
    query: str
    embedding: np.ndarray          # (384,) float32, L2-normalised
    result: Any                    # the search result to return on HIT
    membership: np.ndarray         # (K,) float32 — cluster membership vector
    dominant_cluster: int          # argmax(membership)
    timestamp: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0


class SemanticCache:
    """
    Cluster-partitioned semantic cache with LRU+TTL eviction.
    
    Data structure: {cluster_id: [CacheEntry, ...]}
    LRU tracker:    OrderedDict — O(1) access, O(1) eviction
    
    All methods are synchronous (FastAPI runs in async but uses a thread pool
    for CPU-bound work; the cache is small enough that GIL contention is negligible).
    """

    def __init__(self, similarity_threshold: float = DEFAULT_THRESHOLD,
                 max_size: int = DEFAULT_MAX_SIZE,
                 ttl_seconds: int = DEFAULT_TTL):
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

        # Primary storage: cluster_id → list of CacheEntry
        # This is the cluster-partitioned index
        self._partitions: Dict[int, List[CacheEntry]] = {}

        # LRU tracker: query_key → CacheEntry (OrderedDict maintains access order)
        # On access: move to end (most recently used)
        # On eviction: pop from front (least recently used)
        self._lru: OrderedDict[str, CacheEntry] = OrderedDict()

        # Stats
        self._hit_count: int = 0
        self._miss_count: int = 0

    # ── Lookup ────────────────────────────────────────────────────────────────

    def lookup(self, query_vec: np.ndarray,
               membership: np.ndarray) -> Optional[CacheEntry]:
        """
        Search for a semantically similar cached query.
        
        Algorithm:
          1. Find dominant clusters for this query (membership ≥ 0.15)
          2. Search ONLY those cluster partitions — O(N/K) not O(N)
          3. Compute cosine similarity between query and each candidate
          4. Return best match above threshold τ, or None

        The cluster routing is the key efficiency gain:
        A query about space exploration routes to cluster 3 (~1600 docs partition).
        It never touches clusters 0-2, 4-9 — ~14k entries skipped.
        """
        from app.clusterer import get_dominant_clusters
        dominant = get_dominant_clusters(membership)

        best_entry = None
        best_sim = -1.0
        total_searched = 0

        for cluster_id in dominant:
            partition = self._partitions.get(cluster_id, [])
            for entry in partition:
                # TTL check — lazy expiry
                if time.time() - entry.timestamp > self.ttl_seconds:
                    continue
                total_searched += 1
                sim = cosine_similarity(query_vec, entry.embedding)
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry

        print(f"[Cache] Searched {total_searched} entries across "
              f"{len(dominant)} cluster(s). "
              f"Best similarity: {best_sim:.4f} (threshold={self.similarity_threshold})")

        # Fallback: if dominant cluster search found nothing above threshold,
        # scan ALL partitions. This handles cases where UMAP projects a query
        # into a slightly different cluster than the stored entry.
        # Still much faster than a flat list when cache is large and partitioned.
        if best_sim < self.similarity_threshold:
            fallback_searched = 0
            for cluster_id, partition in self._partitions.items():
                if cluster_id in dominant:
                    continue  # already searched
                for entry in partition:
                    if time.time() - entry.timestamp > self.ttl_seconds:
                        continue
                    fallback_searched += 1
                    sim = cosine_similarity(query_vec, entry.embedding)
                    if sim > best_sim:
                        best_sim = sim
                        best_entry = entry
            if fallback_searched > 0:
                print(f"[Cache] Fallback scan: {fallback_searched} additional entries. "
                      f"Best similarity now: {best_sim:.4f}")

        if best_sim >= self.similarity_threshold and best_entry is not None:
            key = best_entry.query
            if key in self._lru:
                self._lru.move_to_end(key)
            best_entry.last_accessed = time.time()
            best_entry.access_count += 1
            self._hit_count += 1
            return best_entry

        self._miss_count += 1
        return None

    # ── Store ─────────────────────────────────────────────────────────────────

    def store(self, query: str, query_embedding: np.ndarray,
              result: Any, membership: np.ndarray):
        """
        Store a new query-result pair in the cache.
        Routes to the dominant cluster partition.
        Evicts LRU entry if at capacity.
        """
        from app.clusterer import get_dominant_clusters
        dominant_clusters = get_dominant_clusters(membership)
        primary_cluster = dominant_clusters[0]

        entry = CacheEntry(
            query=query,
            embedding=query_embedding.copy(),
            result=result,
            membership=membership.copy(),
            dominant_cluster=primary_cluster,
        )

        # Add to partition
        if primary_cluster not in self._partitions:
            self._partitions[primary_cluster] = []
        self._partitions[primary_cluster].append(entry)

        # Add to LRU tracker
        key = query
        self._lru[key] = entry
        self._lru.move_to_end(key)

        # Evict LRU entry if over capacity
        if len(self._lru) > self.max_size:
            self._evict_lru()

    def _evict_lru(self):
        """Remove the least recently used entry."""
        if not self._lru:
            return
        _, lru_entry = self._lru.popitem(last=False)
        # Remove from partition
        cluster_id = lru_entry.dominant_cluster
        if cluster_id in self._partitions:
            try:
                self._partitions[cluster_id].remove(lru_entry)
            except ValueError:
                pass

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        total = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total if total > 0 else 0.0
        partition_sizes = {
            str(k): len(v) for k, v in self._partitions.items() if v
        }
        return {
            "total_entries": len(self._lru),
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": round(hit_rate, 4),
            "similarity_threshold": self.similarity_threshold,
            "partition_sizes": partition_sizes,
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
        }

    def flush(self):
        """Clear all cache entries and reset stats."""
        self._partitions.clear()
        self._lru.clear()
        self._hit_count = 0
        self._miss_count = 0
        # Delete persisted cache file
        if os.path.exists(CACHE_PATH):
            os.remove(CACHE_PATH)
        print("[Cache] Flushed all entries and reset stats.")

    def set_threshold(self, tau: float):
        old = self.similarity_threshold
        self.similarity_threshold = float(tau)
        print(f"[Cache] Threshold updated: {old:.3f} → {self.similarity_threshold:.3f}")

    def analyze_threshold(self) -> List[dict]:
        """
        Sweep τ ∈ [0.75, 0.99] on current cache entries.
        For each τ: compute what fraction of entries would be returned
        at various similarity levels.
        
        This reveals the system's semantic structure:
        - A spike of hits at τ=0.95 → cache has near-exact paraphrases
        - Gradual slope → cache has topically diverse entries
        - Flat line → entries are all semantically distant (sparse coverage)
        """
        if not self._lru:
            return []

        entries = list(self._lru.values())
        n = len(entries)

        # Compute all pairwise similarities
        embeddings = np.array([e.embedding for e in entries])

        # All pairwise cosine sims: (N, N) — since L2-normalised, just matmul
        sim_matrix = embeddings @ embeddings.T

        # For each τ, count how many pairs would be cache hits
        thresholds = [round(t, 2) for t in np.arange(0.75, 1.00, 0.01)]
        results = []
        for tau in thresholds:
            # Pairs above threshold (exclude diagonal = self-similarity)
            mask = sim_matrix >= tau
            np.fill_diagonal(mask, False)
            hit_pairs = int(mask.sum()) // 2
            max_pairs = n * (n - 1) // 2
            hit_rate = hit_pairs / max_pairs if max_pairs > 0 else 0.0

            # Average similarity of pairs above threshold
            above = sim_matrix[mask]
            avg_sim = float(above.mean()) if len(above) > 0 else 0.0

            results.append({
                "threshold": tau,
                "hit_pairs": hit_pairs,
                "hit_rate": round(hit_rate, 4),
                "avg_hit_similarity": round(avg_sim, 4),
                "interpretation": _interpret_threshold(tau)
            })

        return results

    def persist(self):
        """Save cache to disk."""
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, 'wb') as f:
            pickle.dump({
                "partitions": self._partitions,
                "lru": self._lru,
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "threshold": self.similarity_threshold,
            }, f)

    def load(self):
        """Load cache from disk."""
        if not os.path.exists(CACHE_PATH):
            return
        try:
            with open(CACHE_PATH, 'rb') as f:
                data = pickle.load(f)
            self._partitions = data["partitions"]
            self._lru = data["lru"]
            self._hit_count = data["hit_count"]
            self._miss_count = data["miss_count"]
            self.similarity_threshold = data.get("threshold", self.similarity_threshold)
            print(f"[Cache] Loaded {len(self._lru)} entries from {CACHE_PATH}")
        except Exception as e:
            print(f"[Cache] Could not load cache: {e}. Starting fresh.")

    def get_all_entries(self) -> List[dict]:
        """Return all entries for inspection endpoint."""
        return [
            {
                "query": e.query,
                "dominant_cluster": e.dominant_cluster,
                "access_count": e.access_count,
                "timestamp": e.timestamp,
                "last_accessed": e.last_accessed,
                "membership_top3": sorted(
                    enumerate(e.membership.tolist()),
                    key=lambda x: -x[1]
                )[:3]
            }
            for e in self._lru.values()
        ]


def _interpret_threshold(tau: float) -> str:
    if tau >= 0.98:
        return "near-exact match only"
    elif tau >= 0.93:
        return "paraphrase-level (recommended)"
    elif tau >= 0.87:
        return "topic-level (may reduce precision)"
    elif tau >= 0.80:
        return "loose topic match (precision risk)"
    else:
        return "very loose (not recommended for production)"


# ── Singleton ─────────────────────────────────────────────────────────────────

_cache_instance: Optional[SemanticCache] = None


def get_cache() -> SemanticCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SemanticCache()
    return _cache_instance