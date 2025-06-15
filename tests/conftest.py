"""Test configuration and fixtures."""

import asyncio
import os
import tempfile
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from src.api.main import app
from src.core.config import settings
from src.core.document_processor import ChunkingConfig, DocumentPipeline
from src.embeddings.generator import EmbeddingGenerator
from src.retrieval.vector_store import FAISSVectorStore


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_client() -> Generator[TestClient, None, None]:
    """Create test client for FastAPI app."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def sample_text() -> str:
    """Sample text for testing."""
    return """
# Introduction

This is a sample document for testing the SmartRAG system.

## Chapter 1: Getting Started

Welcome to the world of Retrieval-Augmented Generation (RAG). This technology combines 
the power of large language models with the ability to retrieve relevant information 
from a knowledge base.

### What is RAG?

RAG stands for Retrieval-Augmented Generation. It is a framework that enhances the 
capabilities of language models by allowing them to access and utilize external 
knowledge sources during the generation process.

## Chapter 2: Implementation

In this chapter, we will explore how to implement a RAG system using modern tools 
and techniques.

### Components

A typical RAG system consists of:

1. **Document Processing**: Converting raw documents into structured chunks
2. **Embedding Generation**: Creating vector representations of text
3. **Vector Storage**: Storing embeddings for efficient retrieval
4. **Query Processing**: Understanding user queries and retrieving relevant content
5. **Response Generation**: Using LLMs to generate responses based on retrieved context

### Code Example

```python
def process_document(text: str) -> List[Chunk]:
    processor = DocumentProcessor()
    chunks = processor.process_text(text)
    return chunks
```

> Note: This is just a simple example to demonstrate the concept.

## Conclusion

RAG systems represent a significant advancement in AI applications, providing more 
accurate and contextually relevant responses by leveraging external knowledge sources.
"""


@pytest.fixture
def chunking_config() -> ChunkingConfig:
    """Test chunking configuration."""
    return ChunkingConfig(
        base_tokens=256,
        min_tokens=64,
        max_tokens=512,
        overlap_ratio=0.1,
    )


@pytest.fixture
def document_pipeline(chunking_config: ChunkingConfig) -> DocumentPipeline:
    """Test document pipeline."""
    return DocumentPipeline(chunking_config)


@pytest.fixture
def embedding_generator() -> EmbeddingGenerator:
    """Test embedding generator."""
    return EmbeddingGenerator(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        batch_size=4,
    )


@pytest.fixture
def vector_store(temp_dir: str) -> FAISSVectorStore:
    """Test vector store."""
    return FAISSVectorStore(
        dimension=384,  # MiniLM-L6-v2 dimension
        index_type="Flat",
        metric="cosine",
        index_path=os.path.join(temp_dir, "test_index"),
    )


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    from src.core.document_processor import Chunk
    
    return [
        Chunk(
            id="chunk_1",
            text="This is the first chunk about RAG systems.",
            doc_id="doc_1",
            section_path=["Introduction"],
            token_count=10,
        ),
        Chunk(
            id="chunk_2", 
            text="This chunk explains vector embeddings and similarity search.",
            doc_id="doc_1",
            section_path=["Chapter 1", "What is RAG?"],
            token_count=12,
        ),
        Chunk(
            id="chunk_3",
            text="Implementation details for building RAG systems with Python.",
            doc_id="doc_1", 
            section_path=["Chapter 2", "Implementation"],
            token_count=11,
        ),
    ]


@pytest_asyncio.fixture
async def async_test_setup():
    """Async setup for tests that need it."""
    # Setup any async resources here
    yield
    # Cleanup async resources here


# Mock settings for testing
@pytest.fixture(autouse=True)
def override_settings(temp_dir: str):
    """Override settings for testing."""
    original_values = {}
    
    # Override with test values
    test_overrides = {
        "faiss_index_path": os.path.join(temp_dir, "test_faiss"),
        "redis_url": "redis://localhost:6379/15",  # Use test DB
        "llm_api_url": "http://localhost:8001/v1",
        "cache_ttl": 60,  # Shorter TTL for tests
        "debug": True,
    }
    
    for key, value in test_overrides.items():
        if hasattr(settings, key):
            original_values[key] = getattr(settings, key)
            setattr(settings, key, value)
    
    yield
    
    # Restore original values
    for key, value in original_values.items():
        setattr(settings, key, value)