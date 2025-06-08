"""
FastAPI application main module.

This module initializes and configures the FastAPI application,
sets up dependencies, and provides the main application factory.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from ..core.config import get_settings
from ..core.document_processor import DocumentProcessor
from ..embeddings.generator import EmbeddingGenerator
from ..retrieval.vector_store import VectorStore
from ..retrieval.search import create_hybrid_searcher
from ..llm.client import vLLMClient
from ..cache.redis_client import RedisClient
from ..cache.strategies import create_lru_cache
from .endpoints import router, set_dependencies
from .middleware import setup_middleware

logger = logging.getLogger(__name__)

# Global state
app_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting SmartRAG API application")
    
    try:
        # Initialize dependencies
        await initialize_dependencies()
        logger.info("Dependencies initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    finally:
        # Cleanup
        await cleanup_dependencies()
        logger.info("Application shutdown complete")


async def initialize_dependencies():
    """Initialize all application dependencies."""
    settings = get_settings()
    
    try:
        # Initialize document processor
        logger.info("Initializing document processor...")
        document_processor = DocumentProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        app_state["document_processor"] = document_processor
        
        # Initialize embedding generator
        logger.info("Initializing embedding generator...")
        embedding_generator = EmbeddingGenerator(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size
        )
        await embedding_generator.initialize()
        app_state["embedding_generator"] = embedding_generator
        
        # Initialize vector store
        logger.info("Initializing vector store...")
        vector_store = VectorStore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            collection_name=settings.qdrant_collection_name,
            vector_size=embedding_generator.vector_size,
            distance=settings.qdrant_distance_metric
        )
        await vector_store.initialize()
        app_state["vector_store"] = vector_store
        
        # Initialize hybrid searcher
        logger.info("Initializing hybrid searcher...")
        hybrid_searcher = await create_hybrid_searcher(
            embedding_generator=embedding_generator,
            vector_weight=settings.search_vector_weight,
            bm25_weight=settings.search_bm25_weight,
            mmr_lambda=settings.search_mmr_lambda
        )
        app_state["hybrid_searcher"] = hybrid_searcher
        
        # Initialize LLM client
        logger.info("Initializing LLM client...")
        llm_client = vLLMClient(
            base_url=settings.vllm_base_url,
            api_key=settings.vllm_api_key,
            model=settings.vllm_model,
            timeout=settings.vllm_timeout,
            max_retries=settings.vllm_max_retries
        )
        app_state["llm_client"] = llm_client
        
        # Initialize cache manager
        logger.info("Initializing cache manager...")
        redis_client = None
        if settings.redis_enabled:
            redis_client = RedisClient(
                serializer=settings.redis_serializer,
                key_prefix=settings.redis_key_prefix
            )
        
        cache_manager = await create_lru_cache(
            max_size=settings.cache_max_size,
            redis_client=redis_client
        )
        app_state["cache_manager"] = cache_manager
        
        # Set dependencies in endpoints
        set_dependencies(
            doc_proc=document_processor,
            emb_gen=embedding_generator,
            vs=vector_store,
            searcher=hybrid_searcher,
            llm=llm_client,
            cache_mgr=cache_manager
        )
        
        logger.info("All dependencies initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize dependencies: {e}")
        await cleanup_dependencies()
        raise


async def cleanup_dependencies():
    """Cleanup application dependencies."""
    logger.info("Cleaning up dependencies...")
    
    # Cleanup in reverse order
    cleanup_tasks = []
    
    if "cache_manager" in app_state:
        cleanup_tasks.append(app_state["cache_manager"].close())
    
    if "llm_client" in app_state:
        cleanup_tasks.append(app_state["llm_client"].close())
    
    if "vector_store" in app_state:
        cleanup_tasks.append(app_state["vector_store"].close())
    
    if "embedding_generator" in app_state:
        cleanup_tasks.append(app_state["embedding_generator"].close())
    
    # Execute cleanup tasks
    if cleanup_tasks:
        try:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    app_state.clear()
    logger.info("Cleanup complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    # Create FastAPI app
    app = FastAPI(
        title="SmartRAG API",
        description="Advanced Retrieval-Augmented Generation API with hybrid search and caching",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url="/redoc" if settings.enable_docs else None,
        openapi_url="/openapi.json" if settings.enable_docs else None
    )
    
    # Setup middleware
    setup_middleware(app)
    
    # Include routers
    app.include_router(
        router,
        prefix="/api/v1",
        tags=["SmartRAG"]
    )
    
    # Custom OpenAPI schema
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title="SmartRAG API",
            version="1.0.0",
            description="""
            ## SmartRAG API
            
            Advanced Retrieval-Augmented Generation API with the following features:
            
            ### Core Features
            - **Document Processing**: Intelligent chunking and embedding generation
            - **Hybrid Search**: Vector similarity + BM25 ranking with MMR diversification
            - **RAG Generation**: Context-aware response generation with citations
            - **Streaming Support**: Real-time response streaming
            - **Caching**: Multi-layer caching with Redis support
            
            ### Search Capabilities
            - Vector similarity search using embeddings
            - BM25 keyword-based search
            - Hybrid search combining both approaches
            - Query expansion and result reranking
            - Maximal Marginal Relevance (MMR) for diversity
            
            ### Performance Features
            - Async/await throughout for high concurrency
            - Connection pooling and retry logic
            - Rate limiting and authentication
            - Health monitoring and metrics
            - Batch processing support
            
            ### API Features
            - RESTful design with OpenAPI documentation
            - Comprehensive error handling
            - Request/response validation
            - Structured logging and monitoring
            - CORS and security headers
            """,
            routes=app.routes,
        )
        
        # Add custom tags
        openapi_schema["tags"] = [
            {
                "name": "Documents",
                "description": "Document processing and management"
            },
            {
                "name": "Search",
                "description": "Search and retrieval operations"
            },
            {
                "name": "RAG",
                "description": "Retrieval-Augmented Generation"
            },
            {
                "name": "System",
                "description": "Health checks and system information"
            }
        ]
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi
    
    # Add root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with API information."""
        return {
            "name": "SmartRAG API",
            "version": "1.0.0",
            "description": "Advanced Retrieval-Augmented Generation API",
            "docs_url": "/docs" if settings.enable_docs else None,
            "health_url": "/api/v1/health",
            "features": [
                "Document Processing",
                "Hybrid Search",
                "RAG Generation", 
                "Streaming Responses",
                "Caching",
                "Rate Limiting",
                "Authentication"
            ]
        }
    
    logger.info("FastAPI application created successfully")
    return app


# Create the app instance
app = create_app()


# Health check endpoint (outside of API versioning for load balancers)
@app.get("/health", tags=["Health"])
async def simple_health_check():
    """Simple health check for load balancers."""
    return {"status": "healthy", "timestamp": "2024-01-01T00:00:00Z"}


# Add startup event for logging
@app.on_event("startup")
async def log_startup():
    """Log startup information."""
    settings = get_settings()
    logger.info(f"SmartRAG API starting up...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Docs enabled: {settings.enable_docs}")
    logger.info(f"Authentication enabled: {settings.enable_auth}")
    logger.info(f"Rate limiting enabled: {settings.enable_rate_limiting}")


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if settings.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the application
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.api_workers,
        log_level="debug" if settings.debug else "info",
        access_log=True
    )