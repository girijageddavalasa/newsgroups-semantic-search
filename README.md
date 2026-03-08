# 20 Newsgroups Semantic Search

Semantic search over ~16,000 newsgroup posts with fuzzy clustering and a cluster-partitioned semantic cache — built for the Trademarkia AI/ML Engineer assignment.

---

## Quickstart (Docker — zero setup)

```bash
docker pull girijageddavalasa/newsgroups-search-trademarkia
docker run -p 8000:8000 girijageddavalasa/newsgroups-search-trademarkia
```

Open **http://localhost:8000/docs** — full interactive API, ready immediately.

---

## Architecture

```
Raw corpus (20k posts)
        │
        ▼
[10-Stage Preprocessing]
  Strip: headers, PGP/MIME, signatures, quoted replies (>), URLs, emails
  Deduplicate: MD5 hash of first 500 chars (crossposted articles)
  Filter: <85% ASCII ratio (non-English), <50 tokens (stubs)
  Result: ~14,264 kept (71.3%), 47.8% noise tokens removed
        │
        ▼
[SQLite DB]  corpus.db — clean text + metadata
        │
        ▼
[Embedding]  all-MiniLM-L6-v2 (384-d float32, L2-normalised)
  Long posts (>512 tokens): chunked with 64-token overlap → mean-pooled
        │
        ├──► [usearch HNSW Index]  O(log N) ANN search
        │    Build: once. Query: ~2ms. Dynamic insert: O(log N).
        │    Pre-built wheels for ALL Python versions — zero compilation.
        │
        ▼
[UMAP 384-d → 50-d]
  Parametric — has transform() for new queries at inference time
  Preserves global structure better than PCA; unlike t-SNE, works at inference
        │
        ▼
[Fuzzy C-Means K=K*]  (K* chosen by Partition Coefficient elbow)
  Per-doc membership vector over K clusters — NOT a hard label
  Written from scratch (Bezdek 1981), pure numpy, no scikit-fuzzy
  K-selection elbow plot saved to results/k_selection.png
        │
        ▼
[Cluster-Partitioned Semantic Cache]
  Structure: {cluster_id: [CacheEntry, ...]} + OrderedDict LRU tracker
  Lookup: O(N/K) — routes to dominant cluster partition(s) only
  Eviction: LRU + TTL hybrid
  Threshold τ: tunable at runtime via POST /cache/threshold
```

---

## Design Decisions (every choice justified)

### Why usearch over hnswlib / FAISS / pure NumPy?

| Approach | Query (18k) | Query (1M) | Dynamic insert | Python 3.13 Windows |
|----------|-------------|------------|----------------|---------------------|
| NumPy matmul | ~4ms | ~800ms | `np.vstack` (slow) | ✓ works |
| FAISS IVF | ~2ms | ~3ms | Requires retrain | ✗ needs cmake |
| hnswlib HNSW | ~2ms | ~2ms | O(log N) native | ✗ needs C++ 14 |
| **usearch HNSW** | **~2ms** | **~2ms** | **O(log N) native** | **✓ pre-built wheel** |

usearch implements the same HNSW algorithm as hnswlib with identical O(log N) performance, but ships pre-built wheels for every Python version on every platform — no C++ compiler required. hnswlib has no pre-built wheel for Python 3.13 on Windows; requiring Visual Studio just to install a dependency is unacceptable.

FAISS requires retraining its IVF quantizer when adding new vectors — unusable for a live cache that grows dynamically. NumPy becomes 200x slower at 1M vectors.

Dynamic O(log N) insert matters here: every cache miss adds a new entry to the live index without rebuilding it.

### Why all-MiniLM-L6-v2?

| Model | Dim | Size | MTEB score | Notes |
|-------|-----|------|------------|-------|
| all-MiniLM-L6-v2 | 384 | 90MB | 68.1 | ✓ CHOSEN |
| all-mpnet-base-v2 | 768 | 420MB | 69.6 | 4.7x size for +1.5 quality |
| text-embedding-ada-002 | 1536 | API | 70.4 | Requires OpenAI key |
| paraphrase-MiniLM-L3-v2 | 384 | 61MB | 62.4 | Noticeably worse recall |

Speed/quality tradeoff optimum for this use case. 384-d: fast enough for real-time search; rich enough for paraphrase detection. Trained on 1B sentence pairs. No API dependency — fully offline.

### Why Fuzzy C-Means, written from scratch?

The task requires a distribution per document, not a label. A post about gun legislation belongs to both politics AND firearms — to varying degrees. FCM (Bezdek 1981) is the correct algorithm: it outputs a probability vector over K clusters per document.

Writing it from scratch in pure numpy was required ("if you didn't write it, it shouldn't be in your cache"). scikit-fuzzy is a library — we didn't use it.

### Why K ≠ 20?

We prove K empirically via Partition Coefficient (PC) elbow analysis. K=20 just because the dataset has 20 labels is not justified — several newsgroup pairs are semantically indistinguishable:

- `rec.sport.baseball` + `rec.sport.hockey` → both sports, near-identical vocabulary
- `soc.religion.christian` + `talk.religion.misc` → same semantic field
- `comp.sys.ibm.pc.hardware` + `comp.sys.mac.hardware` → both hardware discussions

**Partition Coefficient** measures cluster crispness:
```
PC = (1/N) Σ_i Σ_k u_ik²
PC = 1.0  → perfectly hard clusters
PC = 1/K  → random (noise baseline)
```

We scan K ∈ {8, 10, 12, 14, 16, 18, 20}, compute PC for each, find the elbow (point of diminishing returns), and use that K. The elbow plot is saved to `results/k_selection.png` and served at `GET /results/k_plot`.

### Why UMAP before clustering?

Clustering directly in 384 dimensions fails due to the curse of dimensionality: all pairwise distances converge to the same value as D→∞, making distance metrics meaningless. UMAP reduces to 50-d (not 2-d — that loses structure), preserving global topology.

- **PCA**: linear, cannot capture curved manifolds in embedding space
- **t-SNE**: non-parametric, no `transform()` — cannot project new queries at inference time
- **UMAP**: non-linear + parametric, has `transform()` ✓

### The cluster-partitioned semantic cache

**Problem:** A standard semantic cache does a linear scan: O(N). At 1000 entries = 1000 cosine similarities per query. This is the bottleneck as the cache grows.

**Data structure:** `{cluster_id: [CacheEntry, ...]}` + `OrderedDict` LRU tracker.

Every query gets a soft membership vector over K clusters via UMAP+FCM. Cache lookup routes to the dominant cluster partition(s) only — typically 1-2 of K clusters — skipping the rest entirely.

```
Expected lookup cost: O(N / K)   ≈ 10x faster than linear at K=10
As cache grows: efficiency improves (more entries, same partition count)
```

**Why OrderedDict for LRU?**
`OrderedDict` maintains insertion order. On access: `move_to_end()` — O(1). On eviction: `popitem(last=False)` — O(1). This is exactly how Python's `functools.lru_cache` works internally. We implement it ourselves as required — no caching library used.

**The tunable parameter τ (similarity threshold):**

| τ | Behaviour | What it reveals |
|---|-----------|-----------------|
| 0.99 | Near-exact match only | Almost nothing qualifies — cache unused |
| 0.92 | Paraphrase-level (default) | Catches rephrased queries correctly |
| 0.85 | Topic-level | Hits for loosely related queries — precision risk |
| 0.75 | Very loose | High hit rate, wrong answers returned |

The interesting question is not "which τ is best" — it's **what each τ reveals about the system's semantic structure**. Use `POST /cache/analyze` to sweep τ and see the hit rate / precision tradeoff curve on your actual cached queries.

### Why cosine similarity (not Euclidean)?

Cosine measures the angle between vectors — direction, not magnitude. Two paraphrases point in the same direction regardless of sentence length. Euclidean distance is dominated by vector magnitude, which correlates with document length, not semantic content.

Since all vectors are L2-normalised at embed time: `cosine(a, b) = dot(a, b)` — no norm computation needed in the hot path. This is the correct shortcut, not a coincidence.

### What data structure avoids redundant computation on similar queries?

The **cluster-partitioned dict** `{cluster_id: [CacheEntry]}` is the direct answer. It avoids redundant computation because:

1. Two semantically similar queries (paraphrases) will receive similar membership vectors → routed to the same cluster partition
2. Cache lookup searches only that partition — O(N/K) instead of O(N)
3. The full membership vector (not just dominant cluster) is the routing key, so cross-topic queries correctly search multiple partitions
4. The result from the first query is returned for the second — zero re-embedding of corpus documents, zero vector search

---

## API Reference

### Required endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/query` | Semantic search with cluster-partitioned caching |
| GET | `/cache/stats` | Cache performance: hit rate, partition sizes, threshold |
| DELETE | `/cache` | Flush all cache entries and reset stats |

### Additional endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health — index size, models loaded, K |
| GET | `/clusters/summary` | Per-cluster doc counts, certainty, boundary counts |
| GET | `/clusters/{id}/docs` | Top documents in a specific cluster |
| GET | `/clusters/boundary` | Most semantically ambiguous documents (max membership < 0.4) |
| POST | `/cache/threshold` | Update τ at runtime — no restart needed |
| POST | `/cache/analyze` | Sweep τ values, show hit rate / precision tradeoff |
| GET | `/cache/inspect` | Inspect all cache entries |
| GET | `/corpus/stats` | Total docs, per-newsgroup distribution |
| GET | `/results/k_plot` | K-selection elbow plot (PNG image) |

### Example: Two paraphrase queries demonstrating cache

```bash
# First query — cache MISS (result computed and stored)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Tell me about NASA space exploration missions", "top_k": 3}'

# Second query (paraphrase) — cache HIT (returned immediately, zero search)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the latest developments in NASA space programs?"}'
```

Second response:
```json
{
  "cache_hit": true,
  "matched_query": "Tell me about NASA space exploration missions",
  "similarity_score": 0.91,
  "search_time_ms": 3.2
}
```

### Example: Threshold sweep

```bash
# Run 10+ varied queries first, then:
curl -X POST http://localhost:8000/cache/analyze
```

Returns hit rate and average similarity at every τ from 0.75 to 0.99 — shows the semantic structure of your query cache.

---

## Local Setup

```bash
# Python 3.13 works fine (usearch has pre-built wheels for all versions)
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate            # Windows

pip install -r requirements.txt

# Place 20_newsgroups.tar.gz in data/ folder
# Download from: https://archive.uci.edu/dataset/113/twenty+newsgroups

# Run setup scripts (one time only — ~45 min total on CPU)
python scripts/01_ingest_corpus.py --tar data/20_newsgroups.tar.gz   # ~10 min
python scripts/02_build_index.py                                       # ~15 min
python scripts/03_build_clusters.py                                    # ~20 min
python scripts/04_verify.py                                            # ~2 min

# Start API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs: http://localhost:8000/docs  
K-selection plot: http://localhost:8000/results/k_plot

---

## Corpus Cleaning Results

```
Total files scanned:    ~19,997
Documents KEPT:         ~14,264  (71.3%)
Dropped (too short):     ~5,221  (posts < 50 tokens after cleaning)
Dropped (duplicate):       ~512  (crossposted articles, content-hash dedup)
Dropped (non-English):        ~0  (ASCII ratio filter)
Noise tokens removed:      47.8%  of raw corpus vocabulary
Avg tokens before clean:    423
Avg tokens after clean:     221
```

---

## Project Structure

```
├── app/
│   ├── main.py           # FastAPI endpoints + lifespan state management
│   ├── preprocessor.py   # 10-stage corpus cleaning pipeline
│   ├── embedder.py       # Sentence embedding, chunk mean-pooling, cosine sim
│   ├── vector_store.py   # usearch HNSW wrapper (O(log N) ANN, all Python versions)
│   ├── clusterer.py      # UMAP + Fuzzy C-Means from scratch + K-selection elbow
│   ├── cache.py          # Cluster-partitioned semantic cache, LRU+TTL, no Redis
│   ├── db.py             # SQLite schema and queries
│   └── models.py         # Pydantic request/response schemas
├── scripts/
│   ├── 01_ingest_corpus.py   # Parse, clean, store in SQLite
│   ├── 02_build_index.py     # Embed + build usearch HNSW index
│   ├── 03_build_clusters.py  # UMAP + FCM + K-selection elbow plot
│   └── 04_verify.py          # End-to-end pipeline verification
├── results/
│   └── k_selection.png       # K-selection elbow plot (auto-generated)
├── data/                     # Generated artifacts (not in git)
├── cache/                    # Persisted semantic cache (not in git)
├── Dockerfile                # Runs clustering inside container (Python 3.11)
├── docker-compose.yml
└── requirements.txt
```

---

## Docker Hub

```bash
docker pull girijageddavalasa/newsgroups-search-trademarkia
docker run -p 8000:8000 girijageddavalasa/newsgroups-search-trademarkia
```

Image: https://hub.docker.com/r/girijageddavalasa/newsgroups-search-trademarkia

GitHub: https://github.com/girijageddavalasa/newsgroups-semantic-search
