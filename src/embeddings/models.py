"""Embedding model definitions for SmartRAG."""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    """Request model for embedding generation."""
    
    texts: List[str] = Field(..., description="List of texts to embed")
    normalize: bool = Field(default=True, description="Whether to normalize embeddings")
    batch_size: Optional[int] = Field(default=None, description="Batch size for processing")
    model_name: Optional[str] = Field(default=None, description="Override model name")


class EmbeddingResponse(BaseModel):
    """Response model for embedding generation."""
    
    embeddings: List[List[float]] = Field(..., description="Generated embeddings")
    model_name: str = Field(..., description="Model used for embedding")
    dimension: int = Field(..., description="Embedding dimension")
    total_tokens: int = Field(..., description="Total tokens processed")
    processing_time: float = Field(..., description="Processing time in seconds")
    normalized: bool = Field(..., description="Whether embeddings are normalized")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class EmbeddingMetadata(BaseModel):
    """Metadata for stored embeddings."""
    
    embedding_id: str = Field(..., description="Unique embedding identifier")
    text: str = Field(..., description="Original text")
    text_hash: str = Field(..., description="Hash of the original text")
    model_name: str = Field(..., description="Model used for embedding")
    dimension: int = Field(..., description="Embedding dimension")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    source_document: Optional[str] = Field(default=None, description="Source document identifier")
    chunk_index: Optional[int] = Field(default=None, description="Chunk index in document")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    custom_metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata")


class CachedEmbedding(BaseModel):
    """Cached embedding with metadata."""
    
    text_hash: str = Field(..., description="Hash of the text")
    embedding: List[float] = Field(..., description="Cached embedding")
    model_name: str = Field(..., description="Model used")
    dimension: int = Field(..., description="Embedding dimension")
    created_at: datetime = Field(..., description="Cache creation time")
    access_count: int = Field(default=1, description="Number of cache hits")
    last_accessed: datetime = Field(default_factory=datetime.utcnow, description="Last access time")


class BatchEmbeddingRequest(BaseModel):
    """Request model for batch embedding generation."""
    
    texts: List[str] = Field(..., description="List of texts to embed")
    metadata: List[Dict[str, Any]] = Field(default_factory=list, description="Metadata for each text")
    normalize: bool = Field(default=True, description="Whether to normalize embeddings")
    batch_size: Optional[int] = Field(default=None, description="Batch size for processing")
    model_name: Optional[str] = Field(default=None, description="Override model name")
    use_cache: bool = Field(default=True, description="Whether to use cache")


class BatchEmbeddingResponse(BaseModel):
    """Response model for batch embedding generation."""
    
    embeddings: List[List[float]] = Field(..., description="Generated embeddings")
    metadata: List[EmbeddingMetadata] = Field(..., description="Embedding metadata")
    model_name: str = Field(..., description="Model used for embedding")
    dimension: int = Field(..., description="Embedding dimension")
    total_texts: int = Field(..., description="Total number of texts processed")
    cache_hits: int = Field(default=0, description="Number of cache hits")
    cache_misses: int = Field(default=0, description="Number of cache misses")
    processing_time: float = Field(..., description="Total processing time in seconds")
    batch_stats: Dict[str, Any] = Field(default_factory=dict, description="Batch processing statistics")


class EmbeddingModelInfo(BaseModel):
    """Information about an embedding model."""
    
    model_name: str = Field(..., description="Model name")
    dimension: int = Field(..., description="Embedding dimension")
    max_sequence_length: int = Field(..., description="Maximum sequence length")
    model_type: str = Field(..., description="Model type (e.g., sentence-transformers)")
    language_support: List[str] = Field(default_factory=list, description="Supported languages")
    description: Optional[str] = Field(default=None, description="Model description")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Model parameters")
    loaded: bool = Field(default=False, description="Whether model is loaded")
    load_time: Optional[float] = Field(default=None, description="Model load time in seconds")
    memory_usage: Optional[int] = Field(default=None, description="Memory usage in bytes")


class SimilarityResult(BaseModel):
    """Result of similarity computation."""
    
    query_embedding: List[float] = Field(..., description="Query embedding")
    candidate_embeddings: List[List[float]] = Field(..., description="Candidate embeddings")
    similarities: List[float] = Field(..., description="Similarity scores")
    method: str = Field(..., description="Similarity method used")
    normalized: bool = Field(..., description="Whether embeddings were normalized")
    computation_time: float = Field(..., description="Computation time in seconds")


class EmbeddingValidationRequest(BaseModel):
    """Request model for embedding validation."""
    
    embeddings: List[List[float]] = Field(..., description="Embeddings to validate")
    expected_dimension: Optional[int] = Field(default=None, description="Expected dimension")
    model_name: Optional[str] = Field(default=None, description="Expected model name")
    check_normalization: bool = Field(default=True, description="Check if normalized")


class EmbeddingValidationResponse(BaseModel):
    """Response model for embedding validation."""
    
    valid: bool = Field(..., description="Whether embeddings are valid")
    dimension: int = Field(..., description="Actual dimension")
    normalized: bool = Field(..., description="Whether embeddings are normalized")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="Embedding statistics")


class EmbeddingSearchRequest(BaseModel):
    """Request model for embedding-based search."""
    
    query_embedding: Optional[List[float]] = Field(default=None, description="Query embedding")
    query_text: Optional[str] = Field(default=None, description="Query text to embed")
    top_k: int = Field(default=10, description="Number of results to return")
    similarity_threshold: Optional[float] = Field(default=None, description="Minimum similarity threshold")
    filter_metadata: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")
    include_scores: bool = Field(default=True, description="Include similarity scores")
    include_metadata: bool = Field(default=True, description="Include embedding metadata")


class EmbeddingSearchResult(BaseModel):
    """Result of embedding-based search."""
    
    embedding_id: str = Field(..., description="Embedding identifier")
    similarity_score: float = Field(..., description="Similarity score")
    text: str = Field(..., description="Original text")
    metadata: Optional[EmbeddingMetadata] = Field(default=None, description="Embedding metadata")
    rank: int = Field(..., description="Result rank")


class EmbeddingSearchResponse(BaseModel):
    """Response model for embedding-based search."""
    
    results: List[EmbeddingSearchResult] = Field(..., description="Search results")
    query_embedding: Optional[List[float]] = Field(default=None, description="Query embedding used")
    total_results: int = Field(..., description="Total number of results")
    search_time: float = Field(..., description="Search time in seconds")
    model_name: str = Field(..., description="Model used for embedding")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Search parameters")


class EmbeddingStats(BaseModel):
    """Statistics about embeddings."""
    
    total_embeddings: int = Field(..., description="Total number of embeddings")
    unique_texts: int = Field(..., description="Number of unique texts")
    models_used: List[str] = Field(..., description="Models used for embeddings")
    average_dimension: float = Field(..., description="Average embedding dimension")
    storage_size: int = Field(..., description="Storage size in bytes")
    cache_hit_rate: float = Field(..., description="Cache hit rate")
    last_updated: datetime = Field(..., description="Last update timestamp")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")