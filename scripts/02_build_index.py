"""
02_build_index.py — Embed all documents and build HNSW vector index.

Usage:
    python scripts/02_build_index.py

Expected output:
    data/embeddings.npy    — (N, 384) float32 embedding matrix
    data/hnsw_index.bin    — hnswlib HNSW index
    data/doc_ids.npy       — row index → SQLite doc_id mapping

Time: ~8-15 minutes on CPU (downloads model on first run, ~90MB).
"""

import os
import sys
import time
import sqlite3
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import init_db, get_all_doc_ids
from app.embedder import embed_batch
from app.vector_store import build_index

BATCH_SIZE = 64


def main():
    init_db()

    # Load all documents from SQLite
    doc_ids = get_all_doc_ids()
    n = len(doc_ids)
    print(f"[Build] Embedding {n} documents in batches of {BATCH_SIZE}...")

    if n == 0:
        print("ERROR: No documents in DB. Run 01_ingest_corpus.py first.")
        sys.exit(1)

    # Fetch all texts in order
    from app.db import get_conn
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, raw_text FROM documents ORDER BY id"
    ).fetchall()
    conn.close()

    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]

    t_start = time.time()
    embeddings = embed_batch(texts, show_progress=True)
    t_embed = time.time() - t_start
    print(f"\n[Build] Embedding done in {t_embed:.1f}s")
    print(f"[Build] Embeddings shape: {embeddings.shape}")

    # Save raw embeddings
    np.save("data/embeddings.npy", embeddings)
    np.save("data/doc_ids.npy", np.array(ids, dtype=np.int32))
    print("[Build] Saved data/embeddings.npy and data/doc_ids.npy")

    # Store embeddings in SQLite for reference
    print("[Build] Storing embeddings in SQLite...")
    from app.db import get_conn
    conn = get_conn()
    for i, (doc_id, vec) in enumerate(zip(ids, embeddings)):
        blob = vec.astype(np.float32).tobytes()
        conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings "
            "(doc_id INTEGER PRIMARY KEY, embedding_blob BLOB)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO embeddings (doc_id, embedding_blob) VALUES (?, ?)",
            (doc_id, blob)
        )
        if i % 2000 == 0:
            conn.commit()
            print(f"  {i}/{n} stored...")
    conn.commit()
    conn.close()

    # Build HNSW index
    print("\n[Build] Building HNSW index...")
    t_index = time.time()
    build_index(embeddings, ids)
    print(f"[Build] HNSW index built in {time.time()-t_index:.1f}s")

    total = time.time() - t_start
    print(f"\n[Build] Complete in {total:.1f}s")
    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Saved: data/embeddings.npy, data/hnsw_index.bin")
    print("\nNext: python scripts/03_build_clusters.py")


if __name__ == "__main__":
    main()
