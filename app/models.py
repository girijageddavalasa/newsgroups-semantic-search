"""models.py — Pydantic request/response schemas for FastAPI."""

from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")


class DocumentResult(BaseModel):
    doc_id: int
    newsgroup: str
    subject: Optional[str]
    text_preview: str
    similarity_score: float
    dominant_cluster: int
    cluster_membership: List[float]


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str]       # original cached query (on HIT)
    similarity_score: Optional[float]  # cache similarity score (on HIT)
    result: List[DocumentResult]
    dominant_cluster: int
    query_membership: List[float]
    search_time_ms: float


class CacheStatsResponse(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float
    similarity_threshold: float
    partition_sizes: Dict[str, int]
    max_size: int
    ttl_seconds: int


class ThresholdRequest(BaseModel):
    threshold: float = Field(..., ge=0.0, le=1.0,
                              description="New similarity threshold τ ∈ [0, 1]")


class ClusterSummaryItem(BaseModel):
    cluster_id: int
    doc_count: int
    avg_certainty: float
    boundary_count: int


class BoundaryDocument(BaseModel):
    doc_id: int
    newsgroup: str
    subject: Optional[str]
    text_preview: str
    max_membership: float
    dominant_cluster: int
    top_memberships: List[dict]


class HealthResponse(BaseModel):
    status: str
    index_size: int
    cache_entries: int
    models_loaded: bool
    k_clusters: int
