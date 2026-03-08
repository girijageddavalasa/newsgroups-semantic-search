"""
db.py — SQLite persistence layer.

WHY SQLITE OVER POSTGRES/MONGODB?
  This is a read-heavy, single-process service with a static corpus.
  SQLite handles 10k+ reads/sec with WAL mode, needs zero infrastructure,
  and ships as a Python built-in. Postgres adds ops complexity for zero gain here.

SCHEMA:
  documents          — cleaned text + metadata per post
  cluster_memberships — (N, K) fuzzy membership matrix, one row per doc
"""

import os
import sqlite3
import numpy as np
from typing import List, Dict, Optional, Tuple

DB_PATH = os.environ.get("DB_PATH", "data/corpus.db")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # concurrent reads without locks
    conn.execute("PRAGMA synchronous=NORMAL") # safe + fast
    return conn


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            newsgroup   TEXT    NOT NULL,
            subject     TEXT,
            raw_text    TEXT    NOT NULL,
            token_count INTEGER NOT NULL,
            content_hash TEXT UNIQUE  -- for deduplication
        );

        CREATE TABLE IF NOT EXISTS cluster_memberships (
            doc_id          INTEGER PRIMARY KEY REFERENCES documents(id),
            membership_blob BLOB    NOT NULL,  -- float32 array, K values
            dominant_cluster INTEGER NOT NULL,
            max_membership   REAL    NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_docs_newsgroup ON documents(newsgroup);
        CREATE INDEX IF NOT EXISTS idx_cluster_dominant ON cluster_memberships(dominant_cluster);
    """)
    conn.commit()
    conn.close()
    print(f"[DB] Initialized at {DB_PATH}")


def insert_document(newsgroup: str, subject: str, raw_text: str,
                    token_count: int, content_hash: str) -> Optional[int]:
    conn = get_conn()
    try:
        cur = conn.execute(
            "INSERT INTO documents (newsgroup, subject, raw_text, token_count, content_hash) "
            "VALUES (?, ?, ?, ?, ?)",
            (newsgroup, subject, raw_text, token_count, content_hash)
        )
        conn.commit()
        return cur.lastrowid
    except sqlite3.IntegrityError:
        return None  # duplicate
    finally:
        conn.close()


def get_doc_count() -> int:
    conn = get_conn()
    count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    conn.close()
    return count


def get_all_doc_ids() -> List[int]:
    conn = get_conn()
    rows = conn.execute("SELECT id FROM documents ORDER BY id").fetchall()
    conn.close()
    return [r[0] for r in rows]


def get_docs_batch(doc_ids: List[int]) -> List[Dict]:
    conn = get_conn()
    placeholders = ",".join("?" * len(doc_ids))
    rows = conn.execute(
        f"SELECT id, newsgroup, subject, raw_text FROM documents WHERE id IN ({placeholders})",
        doc_ids
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_doc_by_id(doc_id: int) -> Optional[Dict]:
    conn = get_conn()
    row = conn.execute(
        "SELECT id, newsgroup, subject, raw_text FROM documents WHERE id=?", (doc_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def store_membership(doc_id: int, membership: np.ndarray):
    blob = membership.astype(np.float32).tobytes()
    dominant = int(np.argmax(membership))
    max_mem = float(np.max(membership))
    conn = get_conn()
    conn.execute(
        "INSERT OR REPLACE INTO cluster_memberships "
        "(doc_id, membership_blob, dominant_cluster, max_membership) VALUES (?,?,?,?)",
        (doc_id, blob, dominant, max_mem)
    )
    conn.commit()
    conn.close()


def get_membership(doc_id: int, K: int) -> Optional[np.ndarray]:
    conn = get_conn()
    row = conn.execute(
        "SELECT membership_blob FROM cluster_memberships WHERE doc_id=?", (doc_id,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return np.frombuffer(row[0], dtype=np.float32)


def get_cluster_summary() -> List[Dict]:
    conn = get_conn()
    rows = conn.execute("""
        SELECT
            cm.dominant_cluster,
            COUNT(*) as doc_count,
            AVG(cm.max_membership) as avg_certainty,
            SUM(CASE WHEN cm.max_membership < 0.4 THEN 1 ELSE 0 END) as boundary_count
        FROM cluster_memberships cm
        GROUP BY cm.dominant_cluster
        ORDER BY cm.dominant_cluster
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_cluster_docs(cluster_id: int, limit: int = 10) -> List[Dict]:
    conn = get_conn()
    rows = conn.execute("""
        SELECT d.id, d.newsgroup, d.subject, d.raw_text,
               cm.max_membership, cm.dominant_cluster
        FROM documents d
        JOIN cluster_memberships cm ON d.id = cm.doc_id
        WHERE cm.dominant_cluster = ?
        ORDER BY cm.max_membership DESC
        LIMIT ?
    """, (cluster_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_boundary_documents(limit: int = 20) -> List[Dict]:
    """Documents with max membership < 0.4 — genuinely cross-topic."""
    conn = get_conn()
    rows = conn.execute("""
        SELECT d.id, d.newsgroup, d.subject, d.raw_text,
               cm.max_membership, cm.dominant_cluster, cm.membership_blob
        FROM documents d
        JOIN cluster_memberships cm ON d.id = cm.doc_id
        WHERE cm.max_membership < 0.4
        ORDER BY cm.max_membership ASC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_newsgroup_counts() -> Dict[str, int]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT newsgroup, COUNT(*) as cnt FROM documents GROUP BY newsgroup ORDER BY newsgroup"
    ).fetchall()
    conn.close()
    return {r["newsgroup"]: r["cnt"] for r in rows}
