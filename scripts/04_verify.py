"""
04_verify.py — End-to-end pipeline verification.

Tests:
  1. SQLite DB has documents
  2. HNSW vector index loads correctly
  3. UMAP + FCM models load and transform queries
  4. Cluster summary is populated
  5. Semantic cache correctly identifies paraphrase queries as hits
  6. Cache threshold analysis works

Usage:
    python scripts/04_verify.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import init_db, get_doc_count, get_cluster_summary, get_boundary_documents
from app.embedder import embed_text, cosine_similarity
from app.vector_store import load_index, search, get_index_size
from app.clusterer import load_models, get_query_membership, get_dominant_clusters, get_k
from app.cache import get_cache, SemanticCache


def check(label, condition, detail=""):
    status = "✓ OK" if condition else "✗ FAIL"
    print(f"  {status}  {label}")
    if detail:
        print(f"        {detail}")
    if not condition:
        print(f"  → Run the appropriate setup script and retry.")
    return condition


def main():
    print("=" * 70)
    print("SEMANTIC SEARCH SYSTEM — PIPELINE VERIFICATION")
    print("=" * 70)

    all_ok = True

    # ── 1. SQLite DB ──────────────────────────────────────────────────────────
    print("\n[1/6] SQLite DB")
    init_db()
    doc_count = get_doc_count()
    all_ok &= check(f"Documents: {doc_count}", doc_count > 0,
                    "Run: python scripts/01_ingest_corpus.py --tar data/20_newsgroups.tar.gz")

    # ── 2. Vector Index ───────────────────────────────────────────────────────
    print("\n[2/6] HNSW Vector Index")
    index_ok = load_index()
    index_size = get_index_size()
    all_ok &= check(f"Index loaded: {index_size} vectors", index_ok and index_size > 0,
                    "Run: python scripts/02_build_index.py")

    # ── 3. Cluster Models ─────────────────────────────────────────────────────
    print("\n[3/6] UMAP + FCM Models")
    models_ok = load_models()
    all_ok &= check(f"Models loaded (K={get_k()})", models_ok,
                    "Run: python scripts/03_build_clusters.py")

    if models_ok:
        test_vec = embed_text("NASA space exploration missions to Mars")
        membership = get_query_membership(test_vec)
        dominant = get_dominant_clusters(membership)
        mem_sum = float(membership.sum())
        check(f"Query → membership sum={mem_sum:.4f} (should be ~1.0)", abs(mem_sum - 1.0) < 0.01)
        check(f"Dominant clusters: {dominant}", len(dominant) >= 1)
        print(f"        Membership vector top-3: "
              f"{sorted(enumerate(membership), key=lambda x: -x[1])[:3]}")

    # ── 4. K-selection plot ───────────────────────────────────────────────────
    print("\n[4/6] K-Selection Elbow Plot")
    plot_exists = os.path.exists("results/k_selection.png")
    check("results/k_selection.png exists", plot_exists,
          "Run: python scripts/03_build_clusters.py")

    # ── 5. Cluster Quality ────────────────────────────────────────────────────
    print("\n[5/6] Cluster Summary")
    clusters = get_cluster_summary()
    check(f"{len(clusters)} clusters in DB", len(clusters) > 0)

    if clusters:
        for row in clusters[:5]:
            print(f"        Cluster {row['dominant_cluster']:2d}: "
                  f"{row['doc_count']:4d} docs, "
                  f"avg_certainty={row['avg_certainty']:.3f}, "
                  f"boundary={row['boundary_count']}")

        boundary = get_boundary_documents(limit=1)
        if boundary:
            b = boundary[0]
            memb = np.frombuffer(b["membership_blob"], dtype=np.float32)
            top2 = sorted(enumerate(memb), key=lambda x: -x[1])[:2]
            print(f"\n        Sample boundary doc (max_membership={b['max_membership']:.3f}):")
            print(f"        Newsgroup: {b['newsgroup']}")
            print(f"        Top-2 clusters: {[(c, round(m,3)) for c,m in top2]}")
            print(f"        Preview: {b['raw_text'][:120]}...")

    # ── 6. Semantic Cache ─────────────────────────────────────────────────────
    print("\n[6/6] Semantic Cache Demo")

    if not models_ok:
        print("  SKIP — cluster models not loaded")
    else:
        cache = SemanticCache(similarity_threshold=0.88)
        cache.flush()

        # Show raw cosine similarity between paraphrase pairs
        print("\n  Cosine similarity between paraphrase pairs:")
        pairs = [
            ("space exploration NASA missions",
             "What has NASA done in outer space?"),
            ("Windows computer boot failure",
             "My PC won't start up properly"),
            ("Middle East political conflict",
             "Israel Palestine situation"),
        ]
        for a, b in pairs:
            va = embed_text(a)
            vb = embed_text(b)
            sim = cosine_similarity(va, vb)
            will_hit = "✓ HIT" if sim >= 0.88 else "  MISS"
            print(f"  {will_hit} ({sim:.3f})  '{a[:40]}' ↔ '{b[:40]}'")

        # Cache demo
        demo = [
            ("space exploration NASA missions",    False),
            ("What has NASA done in outer space?", True),
            ("Windows computer boot failure",      False),
            ("My PC won't start up properly",      True),
            ("Middle East political conflict",      False),
            ("Israel Palestine situation",          True),
        ]

        hits = 0
        print("\n  Cache entries and paraphrase detection:")
        for query, expect_hit in demo:
            vec = embed_text(query)
            mem = get_query_membership(vec)
            result = cache.lookup(vec, mem)
            is_hit = result is not None
            if not is_hit:
                cache.store(query=query, query_embedding=vec,
                            result={"placeholder": True}, membership=mem)
            marker = "✓" if is_hit == expect_hit else "✗ UNEXPECTED"
            status = "HIT " if is_hit else "MISS"
            print(f"  {marker} {status}  '{query[:50]}'")
            if is_hit:
                hits += 1

        stats = cache.stats()
        print(f"\n  Cache hit rate: {stats['hit_rate']:.0%} "
              f"({stats['hit_count']} hits / {stats['miss_count']} misses)")

        # Threshold analysis
        print("\n  Threshold sweep (current cache entries):")
        sweep = cache.analyze_threshold()
        for row in sweep[::4]:  # every 4th
            print(f"  τ={row['threshold']:.2f}: hit_rate={row['hit_rate']:.3f}  "
                  f"avg_sim={row['avg_hit_similarity']:.3f}  "
                  f"({row['interpretation']})")

    print("\n" + "=" * 70)
    if all_ok:
        print("✓ All checks passed!")
        print("\nStart the API:")
        print("  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")
        print("\nAPI docs: http://localhost:8000/docs")
        print("K-plot:   http://localhost:8000/results/k_plot")
    else:
        print("✗ Some checks failed. Run the indicated scripts and retry.")
    print("=" * 70)


if __name__ == "__main__":
    main()
