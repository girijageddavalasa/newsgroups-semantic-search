"""
vector_store.py — usearch HNSW vector index.

WHY usearch INSTEAD OF hnswlib?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Both implement the same HNSW algorithm with identical O(log N) performance.
The difference is pre-built wheel availability:

  Library    Python 3.13 Windows wheel?   Notes
  ─────────────────────────────────────────────────────
  hnswlib    NO — requires C++ compile    Fails on Windows without Visual Studio
  usearch    YES — pre-built for all      Same algorithm, zero compilation needed ✓

usearch is developed by Unum-Cloud, open source (Apache 2.0), used in production
at scale. API is slightly different from hnswlib but identical performance.

HNSW ALGORITHM (same as hnswlib):
  Hierarchical Navigable Small World graph.
  Build: O(N log N). Query: O(log N). Dynamic insert: O(log N).
  Recall > 0.95 at ef=200 for text embeddings.
  Scales to 1M+ vectors with no code change — unlike pure NumPy (O(N·D) per query).

PARAMETERS:
  connectivity=16:   M parameter — links per node (16 = standard for <1M vectors)
  ef_construction=200: build-time exploration
  metric='cos':      cosine distance (vectors are L2-normalised)
"""

import os
import numpy as np
from typing import List, Tuple, Optional
from usearch.index import Index

INDEX_PATH   = os.environ.get("HNSW_PATH", "data/hnsw_index.bin")
DOC_IDS_PATH = "data/doc_ids.npy"
EMBEDDING_DIM = 384

_index: Optional[Index] = None
_doc_ids: Optional[np.ndarray] = None


def _make_index() -> Index:
    return Index(
        ndim=EMBEDDING_DIM,
        metric='cos',
        connectivity=16,
        expansion_add=200,
        expansion_search=100,
    )


def build_index(embeddings: np.ndarray, doc_ids: List[int]):
    global _index, _doc_ids

    n = len(embeddings)
    print(f"[VectorStore] Building HNSW index: {n} vectors")

    idx = _make_index()
    labels = np.array(doc_ids, dtype=np.int64)
    idx.add(labels, embeddings.astype(np.float32))

    _index = idx
    _doc_ids = labels

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    idx.save(INDEX_PATH)
    np.save(DOC_IDS_PATH, _doc_ids)
    print(f"[VectorStore] Saved to {INDEX_PATH} ({n} vectors)")


def load_index() -> bool:
    global _index, _doc_ids

    if not os.path.exists(INDEX_PATH):
        return False

    idx = _make_index()
    idx.load(INDEX_PATH)
    _index = idx

    if os.path.exists(DOC_IDS_PATH):
        _doc_ids = np.load(DOC_IDS_PATH)

    print(f"[VectorStore] Loaded HNSW index: {len(idx)} vectors")
    return True


def search(query_vec: np.ndarray, k: int = 10,
           filter_ids: Optional[List[int]] = None) -> List[Tuple[int, float]]:
    if _index is None or len(_index) == 0:
        return []

    matches = _index.search(query_vec.astype(np.float32), k * 3)

    results = []
    filter_set = set(filter_ids) if filter_ids else None
    for label, distance in zip(matches.keys, matches.distances):
        doc_id = int(label)
        if filter_set and doc_id not in filter_set:
            continue
        # usearch cosine distance = 1 - cosine_similarity
        sim = float(1.0 - distance)
        results.append((doc_id, sim))
        if len(results) >= k:
            break

    return sorted(results, key=lambda x: x[1], reverse=True)


def get_index_size() -> int:
    return len(_index) if _index is not None else 0


def add_vector(doc_id: int, vec: np.ndarray):
    if _index is not None:
        _index.add(
            np.array([doc_id], dtype=np.int64),
            vec.reshape(1, -1).astype(np.float32)
        )
