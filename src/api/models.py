"""
Pydantic models for API requests and responses.

This module defines the data models used for API communication,
including validation, serialization, and documentation.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator, HttpUrl
import uuid


class DocumentType(str, Enum):
    """Supported document types."""
    TEXT = "text"
    PDF = "pdf"
    HTML = "html"
    MARKDOWN = "markdown"
    JSON = "json"


class SearchType(str, Enum):
    """Search types."""
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"


class ResponseStatus(str, Enum):
    """Response status values."""
    SUCCESS = "success"
    ERROR = "error"
    PARTIAL = "partial"


# Base models

class BaseRequest(BaseModel):
    """Base request model."""
    request_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    
    class Config:
        use_enum_values = True


class BaseResponse(BaseModel):
    """Base response model."""
    status: ResponseStatus = ResponseStatus.SUCCESS
    message: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        use_enum_values = True


# Document models

class DocumentMetadata(BaseModel):
    """Document metadata."""
    title: Optional[str] = None
    author: Optional[str] = None
    source: Optional[str] = None
    language: Optional[str] = "en"
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list)
    custom_fields: Dict[str, Any] = Field(default_factory=dict)


class DocumentRequest(BaseRequest):
    """Request to process a document."""
    content: str = Field(..., description="Document content", min_length=1)
    document_type: DocumentType = Field(DocumentType.TEXT, description="Document type")
    document_id: Optional[str] = Field(None, description="Optional document ID")
    metadata: Optional[DocumentMetadata] = Field(None, description="Document metadata")
    chunk_size: Optional[int] = Field(None, description="Custom chunk size", ge=100, le=8000)
    chunk_overlap: Optional[int] = Field(None, description="Custom chunk overlap", ge=0, le=500)
    generate_embeddings: bool = Field(True, description="Generate embeddings for chunks")
    store_in_vector_db: bool = Field(True, description="Store chunks in vector database")
    
    @validator('document_id')
    def validate_document_id(cls, v):
        if v is not None and len(v.strip()) == 0:
            raise ValueError('Document ID cannot be empty')
        return v


class ChunkData(BaseModel):
    """Document chunk data."""
    chunk_id: str
    content: str
    start_char: int
    end_char: int
    chunk_index: int
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentResponse(BaseResponse):
    """Response from document processing."""
    document_id: str
    chunks_count: int
    total_chars: int
    processing_time: float
    chunks: Optional[List[ChunkData]] = None
    embeddings_generated: bool = False
    stored_in_vector_db: bool = False


# Search models

class SearchFilters(BaseModel):
    """Search filters."""
    document_ids: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    date_range: Optional[Dict[str, datetime]] = None
    metadata_filters: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('date_range')
    def validate_date_range(cls, v):
        if v is not None:
            if 'start' in v and 'end' in v and v['start'] > v['end']:
                raise ValueError('Start date must be before end date')
        return v


class SearchRequest(BaseRequest):
    """Search request."""
    query: str = Field(..., description="Search query", min_length=1)
    search_type: SearchType = Field(SearchType.HYBRID, description="Search type")
    num_results: int = Field(10, description="Number of results", ge=1, le=100)
    enable_mmr: bool = Field(True, description="Enable MMR diversification")
    mmr_lambda: float = Field(0.5, description="MMR lambda parameter", ge=0.0, le=1.0)
    expand_query: bool = Field(True, description="Enable query expansion")
    filters: Optional[SearchFilters] = Field(None, description="Search filters")
    include_embeddings: bool = Field(False, description="Include embeddings in response")
    include_chunks: bool = Field(True, description="Include chunk content")


class SearchResultItem(BaseModel):
    """Individual search result."""
    document_id: str
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    mmr_score: Optional[float] = None
    rank: int
    embedding: Optional[List[float]] = None


class SearchResponse(BaseResponse):
    """Search response."""
    query: str
    results: List[SearchResultItem]
    total_results: int
    search_time: float
    search_type: SearchType
    applied_filters: Optional[SearchFilters] = None


# RAG models

class RAGOptions(BaseModel):
    """RAG generation options."""
    model: Optional[str] = None
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=8000)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    stop_sequences: Optional[List[str]] = None
    stream: bool = Field(False, description="Enable streaming response")
    context_window: int = Field(4000, description="Context window size", ge=1000, le=8000)
    citation_style: str = Field("numbered", description="Citation style")


class RAGRequest(BaseRequest):
    """RAG request."""
    query: str = Field(..., description="User query", min_length=1)
    search_options: Optional[SearchRequest] = Field(None, description="Search options")
    generation_options: Optional[RAGOptions] = Field(None, description="Generation options")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    include_sources: bool = Field(True, description="Include source citations")
    include_search_results: bool = Field(False, description="Include raw search results")


class CitationInfo(BaseModel):
    """Citation information."""
    citation_id: str
    document_id: str
    chunk_id: str
    title: Optional[str] = None
    author: Optional[str] = None
    source: Optional[str] = None
    url: Optional[HttpUrl] = None
    excerpt: str
    relevance_score: float


class RAGResponse(BaseResponse):
    """RAG response."""
    query: str
    answer: str
    citations: List[CitationInfo] = Field(default_factory=list)
    generation_time: float
    search_time: float
    total_time: float
    model_used: str
    tokens_used: Optional[Dict[str, int]] = None
    conversation_id: Optional[str] = None
    search_results: Optional[List[SearchResultItem]] = None


# Streaming models

class StreamingChunk(BaseModel):
    """Streaming response chunk."""
    chunk_type: str = Field(..., description="Type of chunk (content, citation, metadata)")
    content: Optional[str] = None
    citation: Optional[CitationInfo] = None
    metadata: Optional[Dict[str, Any]] = None
    is_final: bool = False


# Health and status models

class HealthStatus(str, Enum):
    """Health status values."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


class ServiceHealth(BaseModel):
    """Individual service health."""
    name: str
    status: HealthStatus
    response_time: Optional[float] = None
    error: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseResponse):
    """Health check response."""
    overall_status: HealthStatus
    services: List[ServiceHealth]
    uptime: float
    version: str


# Error models

class ErrorDetail(BaseModel):
    """Error detail information."""
    code: str
    message: str
    field: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseResponse):
    """Error response."""
    status: ResponseStatus = ResponseStatus.ERROR
    error_code: str
    error_message: str
    details: List[ErrorDetail] = Field(default_factory=list)
    
    class Config:
        schema_extra = {
            "example": {
                "status": "error",
                "error_code": "VALIDATION_ERROR",
                "error_message": "Request validation failed",
                "details": [
                    {
                        "code": "VALUE_ERROR",
                        "message": "Query cannot be empty",
                        "field": "query"
                    }
                ],
                "request_id": "12345",
                "timestamp": "2024-01-01T00:00:00Z"
            }
        }


# Metrics and analytics models

class UsageMetrics(BaseModel):
    """Usage metrics."""
    requests_count: int
    avg_response_time: float
    error_rate: float
    cache_hit_rate: float
    active_users: int
    period_start: datetime
    period_end: datetime


class PerformanceMetrics(BaseModel):
    """Performance metrics."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    database_connections: int
    queue_size: int


class MetricsResponse(BaseResponse):
    """Metrics response."""
    usage: UsageMetrics
    performance: PerformanceMetrics
    services: List[ServiceHealth]


# Batch processing models

class BatchDocumentRequest(BaseRequest):
    """Batch document processing request."""
    documents: List[DocumentRequest] = Field(..., min_items=1, max_items=100)
    parallel_processing: bool = Field(True, description="Enable parallel processing")
    
    @validator('documents')
    def validate_documents(cls, v):
        if len(v) > 100:
            raise ValueError('Maximum 100 documents per batch')
        return v


class BatchDocumentResponse(BaseResponse):
    """Batch document processing response."""
    total_documents: int
    successful_documents: int
    failed_documents: int
    results: List[DocumentResponse]
    processing_time: float


# Configuration models

class SystemConfiguration(BaseModel):
    """System configuration."""
    max_chunk_size: int
    default_chunk_overlap: int
    embedding_model: str
    vector_db_config: Dict[str, Any]
    cache_config: Dict[str, Any]
    rate_limits: Dict[str, int]
    
    class Config:
        schema_extra = {
            "example": {
                "max_chunk_size": 1000,
                "default_chunk_overlap": 200,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "vector_db_config": {
                    "collection_name": "smartrag",
                    "metric": "cosine"
                },
                "cache_config": {
                    "strategy": "lru",
                    "max_size": 1000,
                    "ttl": 3600
                },
                "rate_limits": {
                    "requests_per_minute": 100,
                    "documents_per_hour": 1000
                }
            }
        }