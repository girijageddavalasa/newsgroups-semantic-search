# ═══════════════════════════════════════════════════════════════════════════════
# Dockerfile — 20 Newsgroups Semantic Search
# ═══════════════════════════════════════════════════════════════════════════════
#
# WHY Python 3.11 (not 3.13):
#   Python 3.11 has the most stable wheel availability across all packages.
#   usearch (HNSW vector index) has pre-built wheels for all Python versions.
#
# WHY cluster models are built inside the container (not copied from host):
#   UMAP internally uses numba JIT compilation. Pickle files containing numba
#   functions are NOT portable across Python versions. A model saved on Python
#   3.13 cannot be loaded on Python 3.11 — raises SystemError at runtime.
#   Solution: run 03_build_clusters.py INSIDE the container during build.
#   This guarantees models are generated with Python 3.11, matching runtime.
#
# What IS copied from host (Python-version-independent binary formats):
#   - corpus.db      → SQLite database (pure binary format, always portable)
#   - embeddings.npy → NumPy arrays (portable binary, no Python version tie)
#   - hnsw_index.bin → hnswlib binary format (portable)
#   - doc_ids.npy    → NumPy array (portable)
#
# Reviewer workflow:
#   docker pull girijageddavalasa/newsgroups-search-trademarkia
#   docker run -p 8000:8000 girijageddavalasa/newsgroups-search-trademarkia
#   → http://localhost:8000/docs
# ═══════════════════════════════════════════════════════════════════════════════

FROM python:3.11-slim

WORKDIR /app

# System build tools (needed for some umap/numba native extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies FIRST (cached layer — only reruns on requirements.txt change)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model into the image
# Without this, first container start downloads ~90MB from HuggingFace
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2'); print('[Docker] Embedding model baked in.')"

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/

# Copy Python-version-independent data artifacts
# These are safe to copy: SQLite and NumPy binary formats are portable
COPY data/corpus.db        ./data/corpus.db
COPY data/embeddings.npy   ./data/embeddings.npy
COPY data/hnsw_index.bin   ./data/hnsw_index.bin
COPY data/doc_ids.npy      ./data/doc_ids.npy

# Create results and cache directories
RUN mkdir -p results cache

# ── Run clustering INSIDE the container ───────────────────────────────────────
# This generates umap_model.pkl and fcm_model.pkl with Python 3.11.
# K-selection scan + UMAP + FCM takes ~15-20 minutes during build.
# Results in: data/umap_model.pkl, data/fcm_model.pkl, results/k_selection.png
# ──────────────────────────────────────────────────────────────────────────────
RUN python scripts/03_build_clusters.py

EXPOSE 8000

# Health check — polls /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Single worker — required so all requests share the same in-memory cache instance
# Multiple workers would give each worker its own separate cache state
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
