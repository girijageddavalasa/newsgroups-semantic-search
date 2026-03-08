"""
03_build_clusters.py — UMAP dimensionality reduction + Fuzzy C-Means clustering.

Usage:
    python scripts/03_build_clusters.py           # Full run with K-selection scan
    python scripts/03_build_clusters.py --k 10    # Skip scan, use K=10 directly
    python scripts/03_build_clusters.py --skip-umap  # Reuse existing UMAP model

Expected output:
    data/umap_model.pkl       — fitted UMAP model (joblib format)
    data/umap_reduced.npy     — (N, 50) reduced embeddings
    data/fcm_model.pkl        — fitted FuzzyCMeans model
    data/cluster_analysis.json — cluster summary statistics
    results/k_selection.png   — K-selection elbow plot
    
Time: ~10-20 minutes total (UMAP: ~5-10min, FCM scan: ~10min, final FCM: ~2min)
"""

import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import init_db, get_all_doc_ids, store_membership, get_doc_count
from app.clusterer import fit_umap, fit_fcm, select_k, load_models, FuzzyCMeans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=None,
                        help="Force specific K (skips K-selection scan)")
    parser.add_argument("--skip-umap", action="store_true",
                        help="Reuse existing data/umap_reduced.npy")
    args = parser.parse_args()

    init_db()

    # Load embeddings
    emb_path = "data/embeddings.npy"
    if not os.path.exists(emb_path):
        print("ERROR: data/embeddings.npy not found. Run 02_build_index.py first.")
        sys.exit(1)

    print("[Cluster] Loading embeddings...")
    embeddings = np.load(emb_path)
    print(f"[Cluster] Loaded {embeddings.shape[0]} embeddings, dim={embeddings.shape[1]}")

    # ── Step 1: UMAP ──────────────────────────────────────────────────────────
    reduced_path = "data/umap_reduced.npy"

    if args.skip_umap and os.path.exists(reduced_path):
        print("[Cluster] Loading existing UMAP reduced coords (--skip-umap)")
        X_reduced = np.load(reduced_path).astype(np.float32)
        print(f"[Cluster] Loaded UMAP reduced: {X_reduced.shape}")
        
        # Still need to fit UMAP model for inference if not already saved
        if not os.path.exists("data/umap_model.pkl"):
            print("[Cluster] UMAP model not found — fitting fresh model...")
            X_reduced = fit_umap(embeddings)
    else:
        t_umap = time.time()
        X_reduced = fit_umap(embeddings)
        print(f"[Cluster] UMAP done in {time.time()-t_umap:.1f}s. Shape: {X_reduced.shape}")

    # ── Step 2: K Selection ───────────────────────────────────────────────────
    if args.k is not None:
        best_k = args.k
        print(f"\n[Cluster] Using K={best_k} (provided via --k argument)")
    else:
        print("\n[Cluster] Running K-selection analysis...")
        print("WHY: We prove K empirically via Partition Coefficient elbow.")
        print("     K=20 just because the dataset has 20 labels is NOT justified.")
        best_k, pc_scores = select_k(X_reduced, k_range=[8, 10, 12, 14, 16, 18, 20])
        print(f"\n[Cluster] Selected K={best_k} based on PC elbow analysis.")
        print(f"  Plot saved to: results/k_selection.png")

    # ── Step 3: Final FCM ─────────────────────────────────────────────────────
    print(f"\n[Cluster] Fitting final FCM with K={best_k}...")
    t_fcm = time.time()
    fcm = fit_fcm(X_reduced, k=best_k)
    print(f"[Cluster] FCM done in {time.time()-t_fcm:.1f}s")
    print(f"[Cluster] Membership matrix: ({embeddings.shape[0]}, {best_k})")

    # ── Step 4: Store memberships in SQLite ───────────────────────────────────
    print("\n[Cluster] Storing membership vectors in SQLite...")
    doc_ids = get_all_doc_ids()
    membership_matrix = fcm.membership_   # (N, K)

    for i, (doc_id, membership) in enumerate(zip(doc_ids, membership_matrix)):
        store_membership(doc_id, membership)
        if i % 2000 == 0:
            print(f"  {i}/{len(doc_ids)} stored...")

    print(f"[Cluster] All {len(doc_ids)} memberships stored.")

    # ── Step 5: Cluster analysis ──────────────────────────────────────────────
    print("\n[Cluster] Generating cluster analysis...")
    _generate_analysis(fcm, membership_matrix, doc_ids, best_k)

    print("\n[Cluster] ✓ All done!")
    print(f"  data/umap_model.pkl")
    print(f"  data/fcm_model.pkl")
    print(f"  results/k_selection.png")
    print(f"  data/cluster_analysis.json")
    print("\nNext: python scripts/04_verify.py")


def _generate_analysis(fcm, membership_matrix, doc_ids, K):
    """Print cluster summary and save to JSON."""
    from app.db import get_doc_by_id

    os.makedirs("results", exist_ok=True)

    # Load newsgroup labels for each doc
    from app.db import get_conn
    conn = get_conn()
    rows = conn.execute("SELECT id, newsgroup FROM documents ORDER BY id").fetchall()
    conn.close()
    ng_map = {r[0]: r[1] for r in rows}

    print("\n" + "="*70)
    print("CLUSTER ANALYSIS SUMMARY")
    print("="*70)

    analysis = {"clusters": [], "boundary_fraction": 0.0, "K": K}
    boundary_count = 0

    for k in range(K):
        # Docs whose dominant cluster is k
        cluster_memberships = membership_matrix[:, k]
        dominant_mask = np.argmax(membership_matrix, axis=1) == k
        cluster_doc_ids = [doc_ids[i] for i in range(len(doc_ids)) if dominant_mask[i]]
        cluster_sims = membership_matrix[dominant_mask, k]

        if len(cluster_doc_ids) == 0:
            continue

        avg_certainty = float(cluster_sims.mean())
        bound_in_cluster = int((cluster_sims < 0.4).sum())

        # Top newsgroups in this cluster
        ng_counts = {}
        for did in cluster_doc_ids:
            ng = ng_map.get(did, "unknown")
            ng_counts[ng] = ng_counts.get(ng, 0) + 1
        top_ngs = sorted(ng_counts.items(), key=lambda x: -x[1])[:3]

        # Sample doc
        sample = get_doc_by_id(cluster_doc_ids[0]) if cluster_doc_ids else None
        preview = sample["raw_text"][:120] + "..." if sample else ""

        print(f"\nCluster {k:2d} | {len(cluster_doc_ids):4d} docs | "
              f"avg certainty={avg_certainty:.3f} | boundary={bound_in_cluster}")
        print(f"  Top newsgroups: {top_ngs[:3]}")
        print(f"  Sample: {preview}")

        boundary_count += bound_in_cluster
        analysis["clusters"].append({
            "id": k,
            "doc_count": len(cluster_doc_ids),
            "avg_certainty": round(avg_certainty, 3),
            "boundary_count": bound_in_cluster,
            "top_newsgroups": [{"newsgroup": ng, "count": c} for ng, c in top_ngs],
        })

    total_docs = len(doc_ids)
    boundary_frac = boundary_count / total_docs if total_docs > 0 else 0
    analysis["boundary_fraction"] = round(boundary_frac, 3)
    analysis["total_boundary_docs"] = boundary_count

    print(f"\n[Cluster] Boundary documents (max membership < 0.4): "
          f"{boundary_count} ({100*boundary_frac:.1f}%)")
    print("These are the most semantically ambiguous posts — cross-topic content.")

    with open("data/cluster_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    print("[Cluster] Analysis saved to data/cluster_analysis.json")


if __name__ == "__main__":
    main()
