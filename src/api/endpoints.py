"""
REST API endpoints for SmartRAG.

This module provides FastAPI endpoints for document processing, search,
RAG functionality, and system management.
"""

import asyncio
import logging
import time
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from ..core.config import get_settings
from ..core.document_processor import DocumentProcessor
from ..core.exceptions import SmartRAGError, ValidationError
from ..embeddings.generator import EmbeddingGenerator
from ..retrieval.vector_store import VectorStore
from ..retrieval.search import HybridSearcher, create_hybrid_searcher
from ..llm.client import vLLMClient, ChatMessage
from ..llm.streaming import StreamingHandler
from ..cache.strategies import CacheManager
from .models import (
    DocumentRequest, DocumentResponse, ChunkData,
    SearchRequest, SearchResponse, SearchResultItem,
    RAGRequest, RAGResponse, CitationInfo, StreamingChunk,
    HealthResponse, ServiceHealth, HealthStatus,
    BatchDocumentRequest, BatchDocumentResponse,
    MetricsResponse, UsageMetrics, PerformanceMetrics,
    SystemConfiguration, ErrorResponse
)

logger = logging.getLogger(__name__)

# Global dependencies (will be injected)
document_processor: Optional[DocumentProcessor] = None
embedding_generator: Optional[EmbeddingGenerator] = None
vector_store: Optional[VectorStore] = None
hybrid_searcher: Optional[HybridSearcher] = None
llm_client: Optional[vLLMClient] = None
cache_manager: Optional[CacheManager] = None

# Performance tracking
request_times: List[float] = []
error_counts: Dict[str, int] = {}
start_time = time.time()


# Dependency injection functions
async def get_document_processor() -> DocumentProcessor:
    """Get document processor dependency."""
    if document_processor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document processor not available"
        )
    return document_processor


async def get_embedding_generator() -> EmbeddingGenerator:
    """Get embedding generator dependency."""
    if embedding_generator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding generator not available"
        )
    return embedding_generator


async def get_vector_store() -> VectorStore:
    """Get vector store dependency."""
    if vector_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store not available"
        )
    return vector_store


async def get_hybrid_searcher() -> HybridSearcher:
    """Get hybrid searcher dependency."""
    if hybrid_searcher is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Hybrid searcher not available"
        )
    return hybrid_searcher


async def get_llm_client() -> vLLMClient:
    """Get LLM client dependency."""
    if llm_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM client not available"
        )
    return llm_client


async def get_cache_manager() -> CacheManager:
    """Get cache manager dependency."""
    if cache_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cache manager not available"
        )
    return cache_manager


# Router setup
router = APIRouter()


# Document processing endpoints

@router.post(
    "/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Process a document",
    description="Process a document and optionally generate embeddings and store in vector database"
)
async def process_document(
    request: DocumentRequest,
    background_tasks: BackgroundTasks,
    doc_processor: DocumentProcessor = Depends(get_document_processor),
    emb_generator: EmbeddingGenerator = Depends(get_embedding_generator),
    vs: VectorStore = Depends(get_vector_store)
) -> DocumentResponse:
    """Process a single document."""
    start_time = time.time()
    
    try:
        logger.info(f"Processing document: {request.document_id or 'unnamed'}")
        
        # Process document into chunks
        chunks = await doc_processor.process_document(
            content=request.content,
            document_type=request.document_type.value,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            metadata=request.metadata.dict() if request.metadata else {}
        )
        
        # Generate document ID if not provided
        document_id = request.document_id or f"doc_{int(time.time())}"
        
        # Prepare chunk data
        chunk_data_list = []
        embeddings_generated = False
        stored_in_vector_db = False
        
        if request.generate_embeddings:
            # Generate embeddings
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = await emb_generator.generate_batch(chunk_texts)
            embeddings_generated = True
            
            # Create chunk data with embeddings
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_data = ChunkData(
                    chunk_id=f"{document_id}_chunk_{i}",
                    content=chunk.content,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    chunk_index=i,
                    embedding=embedding.tolist() if request.store_in_vector_db else None,
                    metadata={
                        **chunk.metadata,
                        "document_id": document_id,
                        "document_type": request.document_type.value
                    }
                )
                chunk_data_list.append(chunk_data)
            
            # Store in vector database if requested
            if request.store_in_vector_db:
                await vs.add_documents(
                    ids=[chunk.chunk_id for chunk in chunk_data_list],
                    documents=[chunk.content for chunk in chunk_data_list],
                    embeddings=embeddings,
                    metadatas=[chunk.metadata for chunk in chunk_data_list]
                )
                stored_in_vector_db = True
        else:
            # Create chunk data without embeddings
            for i, chunk in enumerate(chunks):
                chunk_data = ChunkData(
                    chunk_id=f"{document_id}_chunk_{i}",
                    content=chunk.content,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    chunk_index=i,
                    metadata={
                        **chunk.metadata,
                        "document_id": document_id,
                        "document_type": request.document_type.value
                    }
                )
                chunk_data_list.append(chunk_data)
        
        processing_time = time.time() - start_time
        
        # Cache document metadata for future reference
        if cache_manager:
            document_metadata = {
                "document_id": document_id,
                "chunks_count": len(chunk_data_list),
                "total_chars": len(request.content),
                "document_type": request.document_type.value,
                "processed_at": datetime.utcnow().isoformat(),
                "embeddings_generated": embeddings_generated,
                "stored_in_vector_db": stored_in_vector_db
            }
            background_tasks.add_task(
                cache_manager.set,
                f"document_metadata:{document_id}",
                document_metadata,
                ttl=3600
            )
        
        logger.info(f"Document processed successfully: {document_id} ({processing_time:.2f}s)")
        
        return DocumentResponse(
            document_id=document_id,
            chunks_count=len(chunk_data_list),
            total_chars=len(request.content),
            processing_time=processing_time,
            chunks=chunk_data_list,
            embeddings_generated=embeddings_generated,
            stored_in_vector_db=stored_in_vector_db,
            request_id=request.request_id
        )
        
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        error_counts["document_processing"] = error_counts.get("document_processing", 0) + 1
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document processing failed: {str(e)}"
        )


@router.post(
    "/documents/batch",
    response_model=BatchDocumentResponse,
    summary="Process multiple documents",
    description="Process multiple documents in batch with optional parallel processing"
)
async def process_documents_batch(
    request: BatchDocumentRequest,
    doc_processor: DocumentProcessor = Depends(get_document_processor)
) -> BatchDocumentResponse:
    """Process multiple documents in batch."""
    start_time = time.time()
    
    try:
        logger.info(f"Processing {len(request.documents)} documents in batch")
        
        results = []
        successful_count = 0
        failed_count = 0
        
        if request.parallel_processing:
            # Process documents in parallel
            tasks = []
            for doc_request in request.documents:
                task = asyncio.create_task(
                    process_document(doc_request, BackgroundTasks(), doc_processor, 
                                   embedding_generator, vector_store)
                )
                tasks.append(task)
            
            # Wait for all tasks to complete
            results_with_errors = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results_with_errors:
                if isinstance(result, Exception):
                    failed_count += 1
                    # Create error response
                    error_response = DocumentResponse(
                        document_id="error",
                        chunks_count=0,
                        total_chars=0,
                        processing_time=0.0,
                        status="error",
                        message=str(result)
                    )
                    results.append(error_response)
                else:
                    successful_count += 1
                    results.append(result)
        else:
            # Process documents sequentially
            for doc_request in request.documents:
                try:
                    result = await process_document(
                        doc_request, BackgroundTasks(), doc_processor,
                        embedding_generator, vector_store
                    )
                    results.append(result)
                    successful_count += 1
                except Exception as e:
                    failed_count += 1
                    error_response = DocumentResponse(
                        document_id="error",
                        chunks_count=0,
                        total_chars=0,
                        processing_time=0.0,
                        status="error",
                        message=str(e)
                    )
                    results.append(error_response)
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"Batch processing complete: {successful_count} successful, "
            f"{failed_count} failed ({processing_time:.2f}s)"
        )
        
        return BatchDocumentResponse(
            total_documents=len(request.documents),
            successful_documents=successful_count,
            failed_documents=failed_count,
            results=results,
            processing_time=processing_time,
            request_id=request.request_id
        )
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch processing failed: {str(e)}"
        )


# Search endpoints

@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search documents",
    description="Search for documents using hybrid search (vector + BM25)"
)
async def search_documents(
    request: SearchRequest,
    searcher: HybridSearcher = Depends(get_hybrid_searcher),
    cache_mgr: CacheManager = Depends(get_cache_manager)
) -> SearchResponse:
    """Search for documents."""
    start_time = time.time()
    
    try:
        logger.info(f"Searching: '{request.query}' (type: {request.search_type})")
        
        # Check cache first
        cache_key = f"search:{hash(request.query + str(request.dict()))}"
        cached_result = await cache_mgr.get(cache_key)
        
        if cached_result:
            logger.debug("Returning cached search result")
            return SearchResponse(**cached_result)
        
        # Perform search
        search_results = await searcher.search(
            query=request.query,
            num_results=request.num_results,
            enable_mmr=request.enable_mmr,
            expand_query=request.expand_query,
            filters=request.filters.dict() if request.filters else None
        )
        
        # Convert to API response format
        result_items = []
        for result in search_results:
            item = SearchResultItem(
                document_id=result.document_id,
                chunk_id=result.document_id,  # Assuming chunk_id same as document_id for now
                content=result.content,
                score=result.score,
                metadata=result.metadata,
                vector_score=result.vector_score,
                bm25_score=result.bm25_score,
                mmr_score=result.mmr_score,
                rank=result.rank,
                embedding=None  # Don't include embeddings by default
            )
            result_items.append(item)
        
        search_time = time.time() - start_time
        
        response = SearchResponse(
            query=request.query,
            results=result_items,
            total_results=len(result_items),
            search_time=search_time,
            search_type=request.search_type,
            applied_filters=request.filters,
            request_id=request.request_id
        )
        
        # Cache result
        await cache_mgr.set(cache_key, response.dict(), ttl=300)
        
        logger.info(f"Search completed: {len(result_items)} results ({search_time:.2f}s)")
        
        return response
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        error_counts["search"] = error_counts.get("search", 0) + 1
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


# RAG endpoints

@router.post(
    "/rag/query",
    response_model=RAGResponse,
    summary="RAG query",
    description="Perform retrieval-augmented generation"
)
async def rag_query(
    request: RAGRequest,
    searcher: HybridSearcher = Depends(get_hybrid_searcher),
    llm: vLLMClient = Depends(get_llm_client),
    cache_mgr: CacheManager = Depends(get_cache_manager)
) -> RAGResponse:
    """Perform RAG query."""
    start_time = time.time()
    
    try:
        logger.info(f"RAG query: '{request.query}'")
        
        # Check cache first
        cache_key = f"rag:{hash(request.query + str(request.dict()))}"
        cached_result = await cache_mgr.get(cache_key)
        
        if cached_result:
            logger.debug("Returning cached RAG result")
            return RAGResponse(**cached_result)
        
        # Step 1: Search for relevant documents
        search_start = time.time()
        
        search_options = request.search_options or SearchRequest(
            query=request.query,
            num_results=5,
            search_type="hybrid"
        )
        
        search_results = await searcher.search(
            query=search_options.query,
            num_results=search_options.num_results,
            enable_mmr=search_options.enable_mmr,
            expand_query=search_options.expand_query,
            filters=search_options.filters.dict() if search_options.filters else None
        )
        
        search_time = time.time() - search_start
        
        # Step 2: Prepare context from search results
        context_chunks = []
        citations = []
        
        for i, result in enumerate(search_results):
            context_chunks.append(f"[{i+1}] {result.content}")
            
            citation = CitationInfo(
                citation_id=str(i+1),
                document_id=result.document_id,
                chunk_id=result.document_id,
                excerpt=result.content[:200] + "..." if len(result.content) > 200 else result.content,
                relevance_score=result.score,
                **result.metadata
            )
            citations.append(citation)
        
        context = "\n\n".join(context_chunks)
        
        # Step 3: Generate response using LLM
        gen_start = time.time()
        gen_options = request.generation_options or {}
        
        # Create system message with context
        system_message = ChatMessage(
            role="system",
            content=f"""You are a helpful AI assistant. Use the following context to answer the user's question.
            
Context:
{context}

Instructions:
- Answer based on the provided context
- If you use information from the context, cite it using [number] format
- If the context doesn't contain enough information, say so
- Be accurate and concise"""
        )
        
        user_message = ChatMessage(role="user", content=request.query)
        
        llm_response = await llm.generate(
            messages=[system_message, user_message],
            **gen_options.dict() if hasattr(gen_options, 'dict') else gen_options
        )
        
        generation_time = time.time() - gen_start
        total_time = time.time() - start_time
        
        response = RAGResponse(
            query=request.query,
            answer=llm_response.content,
            citations=citations if request.include_sources else [],
            generation_time=generation_time,
            search_time=search_time,
            total_time=total_time,
            model_used=llm_response.model,
            tokens_used=llm_response.usage,
            conversation_id=request.conversation_id,
            search_results=[SearchResultItem(
                document_id=r.document_id,
                chunk_id=r.document_id,
                content=r.content,
                score=r.score,
                metadata=r.metadata,
                rank=r.rank
            ) for r in search_results] if request.include_search_results else None,
            request_id=request.request_id
        )
        
        # Cache result
        await cache_mgr.set(cache_key, response.dict(), ttl=600)
        
        logger.info(f"RAG query completed ({total_time:.2f}s)")
        
        return response
        
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        error_counts["rag"] = error_counts.get("rag", 0) + 1
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG query failed: {str(e)}"
        )


@router.post(
    "/rag/stream",
    summary="Streaming RAG query",
    description="Perform RAG query with streaming response"
)
async def rag_query_stream(
    request: RAGRequest,
    searcher: HybridSearcher = Depends(get_hybrid_searcher),
    llm: vLLMClient = Depends(get_llm_client)
):
    """Perform streaming RAG query."""
    
    async def generate_stream():
        try:
            # Search phase
            yield f"data: {StreamingChunk(chunk_type='metadata', metadata={'phase': 'search'}).json()}\n\n"
            
            search_results = await searcher.search(
                query=request.query,
                num_results=5
            )
            
            # Send citations
            for i, result in enumerate(search_results):
                citation = CitationInfo(
                    citation_id=str(i+1),
                    document_id=result.document_id,
                    chunk_id=result.document_id,
                    excerpt=result.content[:200],
                    relevance_score=result.score
                )
                yield f"data: {StreamingChunk(chunk_type='citation', citation=citation).json()}\n\n"
            
            # Generation phase
            yield f"data: {StreamingChunk(chunk_type='metadata', metadata={'phase': 'generation'}).json()}\n\n"
            
            # Prepare context and messages
            context = "\n\n".join([f"[{i+1}] {r.content}" for i, r in enumerate(search_results)])
            
            system_message = ChatMessage(
                role="system",
                content=f"Use the following context to answer: {context}"
            )
            user_message = ChatMessage(role="user", content=request.query)
            
            # Stream generation
            async for chunk in llm.stream_generate([system_message, user_message]):
                stream_chunk = StreamingChunk(chunk_type="content", content=chunk)
                yield f"data: {stream_chunk.json()}\n\n"
            
            # Final chunk
            yield f"data: {StreamingChunk(chunk_type='metadata', metadata={'phase': 'complete'}, is_final=True).json()}\n\n"
            
        except Exception as e:
            error_chunk = StreamingChunk(
                chunk_type="metadata",
                metadata={"error": str(e)},
                is_final=True
            )
            yield f"data: {error_chunk.json()}\n\n"
    
    return EventSourceResponse(generate_stream())


# Health and system endpoints

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health of all system components"
)
async def health_check() -> HealthResponse:
    """Perform health check."""
    services = []
    overall_status = HealthStatus.HEALTHY
    
    # Check document processor
    try:
        if document_processor:
            services.append(ServiceHealth(
                name="document_processor",
                status=HealthStatus.HEALTHY,
                response_time=0.001
            ))
        else:
            services.append(ServiceHealth(
                name="document_processor",
                status=HealthStatus.UNHEALTHY,
                error="Not initialized"
            ))
            overall_status = HealthStatus.UNHEALTHY
    except Exception as e:
        services.append(ServiceHealth(
            name="document_processor",
            status=HealthStatus.UNHEALTHY,
            error=str(e)
        ))
        overall_status = HealthStatus.UNHEALTHY
    
    # Check embedding generator
    try:
        if embedding_generator:
            test_start = time.time()
            await embedding_generator.generate_single("test")
            response_time = time.time() - test_start
            
            services.append(ServiceHealth(
                name="embedding_generator",
                status=HealthStatus.HEALTHY,
                response_time=response_time
            ))
        else:
            services.append(ServiceHealth(
                name="embedding_generator",
                status=HealthStatus.UNHEALTHY,
                error="Not initialized"
            ))
            overall_status = HealthStatus.UNHEALTHY
    except Exception as e:
        services.append(ServiceHealth(
            name="embedding_generator",
            status=HealthStatus.UNHEALTHY,
            error=str(e)
        ))
        overall_status = HealthStatus.DEGRADED
    
    # Check vector store
    try:
        if vector_store:
            test_start = time.time()
            await vector_store.search(query_embedding=[0.1] * 384, num_results=1)
            response_time = time.time() - test_start
            
            services.append(ServiceHealth(
                name="vector_store",
                status=HealthStatus.HEALTHY,
                response_time=response_time
            ))
        else:
            services.append(ServiceHealth(
                name="vector_store",
                status=HealthStatus.UNHEALTHY,
                error="Not initialized"
            ))
            overall_status = HealthStatus.UNHEALTHY
    except Exception as e:
        services.append(ServiceHealth(
            name="vector_store",
            status=HealthStatus.UNHEALTHY,
            error=str(e)
        ))
        overall_status = HealthStatus.DEGRADED
    
    # Check LLM client
    try:
        if llm_client:
            is_healthy = await llm_client.health_check()
            if is_healthy:
                services.append(ServiceHealth(
                    name="llm_client",
                    status=HealthStatus.HEALTHY
                ))
            else:
                services.append(ServiceHealth(
                    name="llm_client",
                    status=HealthStatus.UNHEALTHY,
                    error="Health check failed"
                ))
                overall_status = HealthStatus.DEGRADED
        else:
            services.append(ServiceHealth(
                name="llm_client",
                status=HealthStatus.UNHEALTHY,
                error="Not initialized"
            ))
            overall_status = HealthStatus.UNHEALTHY
    except Exception as e:
        services.append(ServiceHealth(
            name="llm_client",
            status=HealthStatus.UNHEALTHY,
            error=str(e)
        ))
        overall_status = HealthStatus.DEGRADED
    
    uptime = time.time() - start_time
    
    return HealthResponse(
        overall_status=overall_status,
        services=services,
        uptime=uptime,
        version="1.0.0"
    )


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="System metrics",
    description="Get system performance and usage metrics"
)
async def get_metrics() -> MetricsResponse:
    """Get system metrics."""
    try:
        # Calculate usage metrics
        total_requests = len(request_times)
        avg_response_time = sum(request_times) / len(request_times) if request_times else 0
        total_errors = sum(error_counts.values())
        error_rate = total_errors / total_requests if total_requests > 0 else 0
        
        # Get cache stats
        cache_hit_rate = 0.0
        if cache_manager:
            cache_stats = cache_manager.get_stats()
            cache_hit_rate = cache_stats.get("hit_rate", 0.0)
        
        usage = UsageMetrics(
            requests_count=total_requests,
            avg_response_time=avg_response_time,
            error_rate=error_rate,
            cache_hit_rate=cache_hit_rate,
            active_users=1,  # Simplified
            period_start=datetime.fromtimestamp(start_time),
            period_end=datetime.utcnow()
        )
        
        # Get performance metrics (simplified)
        import psutil
        performance = PerformanceMetrics(
            cpu_usage=psutil.cpu_percent(),
            memory_usage=psutil.virtual_memory().percent,
            disk_usage=psutil.disk_usage('/').percent,
            network_io={"bytes_sent": 0, "bytes_recv": 0},  # Simplified
            database_connections=0,  # Simplified
            queue_size=0  # Simplified
        )
        
        # Get service health
        health_response = await health_check()
        
        return MetricsResponse(
            usage=usage,
            performance=performance,
            services=health_response.services
        )
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}"
        )


# Configuration endpoints

@router.get(
    "/config",
    response_model=SystemConfiguration,
    summary="Get system configuration",
    description="Get current system configuration"
)
async def get_configuration() -> SystemConfiguration:
    """Get system configuration."""
    settings = get_settings()
    
    return SystemConfiguration(
        max_chunk_size=settings.max_chunk_size,
        default_chunk_overlap=settings.chunk_overlap,
        embedding_model=settings.embedding_model,
        vector_db_config={
            "collection_name": settings.qdrant_collection_name,
            "metric": "cosine"
        },
        cache_config={
            "strategy": "lru",
            "max_size": 1000,
            "ttl": 3600
        },
        rate_limits={
            "requests_per_minute": settings.rate_limit_per_minute,
            "documents_per_hour": 1000
        }
    )


# Utility functions for dependency injection
def set_dependencies(
    doc_proc: DocumentProcessor,
    emb_gen: EmbeddingGenerator,
    vs: VectorStore,
    searcher: HybridSearcher,
    llm: vLLMClient,
    cache_mgr: CacheManager
):
    """Set global dependencies."""
    global document_processor, embedding_generator, vector_store, hybrid_searcher, llm_client, cache_manager
    
    document_processor = doc_proc
    embedding_generator = emb_gen
    vector_store = vs
    hybrid_searcher = searcher
    llm_client = llm
    cache_manager = cache_mgr
    
    logger.info("Dependencies set successfully")