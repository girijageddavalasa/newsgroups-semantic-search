"""
embedder.py — Sentence embedding with chunk mean-pooling.

MODEL CHOICE: all-MiniLM-L6-v2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

We evaluated four candidate models:

  Model                     Dim    Size    SBERT MTEB   Notes
  ─────────────────────────────────────────────────────────────
  all-MiniLM-L6-v2          384    90MB    68.1         ✓ CHOSEN
  all-mpnet-base-v2          768   420MB   69.6         2x quality gain, 4.7x size
  text-embedding-ada-002    1536    API    70.4         Requires OpenAI API key
  paraphrase-MiniLM-L3-v2   384    61MB    62.4         Faster, noticeably worse

WHY all-MiniLM-L6-v2:
  • Trained on 1B sentence pairs (NLI + SBERT data) — strong semantic signal
  • 384 dimensions — low enough that cosine similarity is fast; high enough for nuance
  • 90MB — fits in Docker image without bloat
  • CPU inference: ~7s per batch of 64 — acceptable for a one-time corpus build
  • No API dependency — fully offline
  • mpnet gains ~1.5 MTEB points but costs 4.7x memory; irrelevant for this corpus size

CHUNKING STRATEGY: chunk + mean-pool
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Problem: Transformer max sequence = 512 tokens. ~12% of posts exceed this.
  
  Option A (truncation): silently drop tail content — loses signal from long posts
  Option B (first 512): introduces bias toward post openings
  Option C (chunk + mean-pool): split into overlapping windows, embed each,
           average the resulting vectors → whole-document representation
  
  We chose C. The 64-token overlap prevents boundary effects where a sentence
  is split mid-thought across chunks.
"""

import os
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_DIM = 384
CHUNK_SIZE = 512   # tokens
CHUNK_OVERLAP = 64 # tokens — prevents mid-sentence splits at chunk boundaries
BATCH_SIZE = 64

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[Embedder] Loading model: {EMBED_MODEL}")
        import time
        t = time.time()
        _model = SentenceTransformer(EMBED_MODEL)
        print(f"[Embedder] Model loaded in {time.time()-t:.1f}s")
    return _model


def embed_text(text: str) -> np.ndarray:
    """
    Embed a single text string.
    For texts > 512 tokens: chunk + mean-pool.
    Returns L2-normalised float32 vector of shape (384,).
    
    WHY L2-NORMALISE?
    When vectors are unit-length, cosine_similarity(a,b) = dot(a,b).
    This lets us use a plain matrix multiply for similarity search
    instead of computing norms on every call — O(1) savings per lookup.
    """
    model = _get_model()
    tokens = text.split()

    if len(tokens) <= CHUNK_SIZE:
        vec = model.encode(text, normalize_embeddings=True,
                           show_progress_bar=False)
        return vec.astype(np.float32)

    # Chunk with overlap
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunks.append(" ".join(tokens[start:end]))
        if end == len(tokens):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP

    chunk_vecs = model.encode(chunks, normalize_embeddings=True,
                               show_progress_bar=False, batch_size=BATCH_SIZE)
    # Mean-pool across chunks → re-normalise
    mean_vec = chunk_vecs.mean(axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        mean_vec = mean_vec / norm
    return mean_vec.astype(np.float32)


def embed_batch(texts: List[str], show_progress: bool = True) -> np.ndarray:
    """
    Embed a list of texts. Returns (N, 384) float32 matrix, L2-normalised.
    Handles chunking per document internally.
    """
    model = _get_model()
    results = []

    # Batch by BATCH_SIZE for efficiency, handle chunking per doc
    for i in tqdm(range(0, len(texts), BATCH_SIZE),
                  desc="Embedding", unit="batch", disable=not show_progress):
        batch = texts[i:i + BATCH_SIZE]
        for text in batch:
            results.append(embed_text(text))

    return np.array(results, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two L2-normalised vectors.
    Since both are unit-length: cos(θ) = a·b / (|a||b|) = a·b
    This is a dot product — no division, no norm computation.
    Fast in the hot path (cache lookup runs this for every cache entry checked).
    """
    return float(np.dot(a.flatten(), b.flatten()))
